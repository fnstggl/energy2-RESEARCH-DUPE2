"""Tests for the self-serve importer (aurelius/selfserve): mapping,
coercion, class detection, and the never-crash contract.

The importer's promise is that ANY tabular file produces either a confident
mapping or a legible diagnosis — never a stack trace, and never a replay on
data that cannot support one.
"""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from aurelius.selfserve.extract import extract
from aurelius.selfserve.fields import (
    coerce_duration,
    coerce_timestamp,
    infer_duration_unit,
    infer_timestamp_unit,
    parse_gres_gpus,
    status_to_failure,
)
from aurelius.selfserve.sniff import TableReadError, build_mapping_plan, read_table
from aurelius.selfserve.validate import TIER_INSUFFICIENT, validate


def _write_csv(path: Path, columns: list[str], rows: list[list]) -> Path:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        w.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# Coercion units
# ---------------------------------------------------------------------------

class TestCoercions:
    def test_timestamp_unit_inference(self):
        assert infer_timestamp_unit(["2026-03-01T00:00:00Z"])[0] == "iso8601"
        assert infer_timestamp_unit(["1772000000"])[0] == "epoch_s"
        assert infer_timestamp_unit(["1772000000123"])[0] == "epoch_ms"
        assert infer_timestamp_unit(["1772000000123456"])[0] == "epoch_us"
        assert infer_timestamp_unit(["12.5", "800.1"])[0] == "relative_s"

    def test_timestamp_coercion_equivalence(self):
        iso = coerce_timestamp("2026-03-01T00:00:00Z", "iso8601")
        ms = coerce_timestamp("1772323200000", "epoch_ms")
        s = coerce_timestamp("1772323200", "epoch_s")
        assert iso == pytest.approx(ms) == pytest.approx(s)

    def test_naive_datetime_assumed_utc(self):
        naive = coerce_timestamp("2026-03-01 00:00:00", "iso8601")
        aware = coerce_timestamp("2026-03-01T00:00:00+00:00", "iso8601")
        assert naive == pytest.approx(aware)

    def test_duration_units(self):
        assert coerce_duration("1500", "ms") == pytest.approx(1.5)
        assert coerce_duration("2", "hours") == pytest.approx(7200)
        assert coerce_duration("90", "seconds") == pytest.approx(90)
        assert coerce_duration("1-02:30:00", "hms") == pytest.approx(
            86400 + 2 * 3600 + 30 * 60)
        assert coerce_duration("00:05:30", "hms") == pytest.approx(330)

    def test_duration_unit_inference(self):
        assert infer_duration_unit("latency_ms", ["120", "80"])[0] == "ms"
        assert infer_duration_unit("runtime_hours", ["2.5"])[0] == "hours"
        assert infer_duration_unit("Elapsed",
                                   ["1-00:00:00", "02:00:00"])[0] == "hms"
        assert infer_duration_unit("duration", ["120"])[0] == "seconds"

    def test_gres_parsing(self):
        assert parse_gres_gpus("billing=64,cpu=64,gres/gpu=8,mem=512G") == \
            (8, None)
        assert parse_gres_gpus("gres/gpu:h100=4") == (4, "h100")
        assert parse_gres_gpus("cpu=8,mem=32G") == (None, None)

    def test_status_table(self):
        assert status_to_failure("FAILED") is True
        assert status_to_failure("CANCELLED by 1042") is True
        assert status_to_failure("COMPLETED") is False
        assert status_to_failure("503") is True
        assert status_to_failure("200") is False
        assert status_to_failure("REQUEUED") is None   # unknown → reported


# ---------------------------------------------------------------------------
# Mapping + class detection
# ---------------------------------------------------------------------------

class TestMapping:
    def test_vendor_flavored_serving_columns(self, tmp_path):
        p = _write_csv(
            tmp_path / "gw.csv",
            ["ts", "model_name", "input_tok", "gen_tok", "http_status",
             "session", "region"],
            [[1772000000000 + i * 500, "m1", 100, 50, 200, "s1", "us"]
             for i in range(50)],
        )
        plan = build_mapping_plan(read_table(p))
        assert plan.workload_class == "serving"
        got = {m.canonical: m.source for m in plan.mappings if m.source}
        assert got["timestamp"] == "ts"
        assert got["prompt_tokens"] == "input_tok"
        assert got["output_tokens"] == "gen_tok"
        assert got["status"] == "http_status"
        assert got["session_id"] == "session"
        ts = plan.get("timestamp")
        assert ts.transform == "epoch_ms"
        assert "region" in plan.unmapped_source_columns

    def test_slurm_sacct_columns(self, tmp_path):
        p = _write_csv(
            tmp_path / "sacct.csv",
            ["JobID", "User", "Submit", "Start", "End", "Elapsed", "State",
             "AllocTRES"],
            [[41000 + i, "u1", "2026-03-01T00:00:00", "2026-03-01T01:00:00",
              "2026-03-01T03:00:00", "02:00:00", "COMPLETED",
              "cpu=32,gres/gpu:a100=4,mem=256G"] for i in range(50)],
        )
        plan = build_mapping_plan(read_table(p))
        assert plan.workload_class == "gpu_jobs"
        got = {m.canonical: m.source for m in plan.mappings if m.source}
        assert got["job_id"] == "JobID"
        assert got["submit_time"] == "Submit"
        assert got["gres"] == "AllocTRES"
        assert got["status"] == "State"
        ext = extract(read_table(p), plan)
        assert ext.rows_usable == 50
        rec = ext.records[0]
        assert rec["gpu_count"] == 4
        assert rec["gpu_type"] == "a100"
        assert rec["duration"] == pytest.approx(7200)
        assert rec["queue_wait"] == pytest.approx(3600)

    def test_user_override_wins(self, tmp_path):
        p = _write_csv(
            tmp_path / "odd.csv",
            ["when", "in_size", "out_size"],
            [[1772000000 + i, 10, 20] for i in range(30)],
        )
        plan = build_mapping_plan(
            read_table(p), workload_class="serving",
            overrides={"timestamp": "when", "prompt_tokens": "in_size",
                       "output_tokens": "out_size"},
        )
        got = {m.canonical: (m.source, m.matched_by)
               for m in plan.mappings if m.source}
        assert got["timestamp"] == ("when", "user")
        assert got["prompt_tokens"] == ("in_size", "user")

    def test_jsonl_input(self, tmp_path):
        p = tmp_path / "log.jsonl"
        with p.open("w") as fh:
            for i in range(40):
                fh.write('{"timestamp": %d, "prompt_tokens": 5, '
                         '"output_tokens": 7}\n' % (1772000000 + i))
        plan = build_mapping_plan(read_table(p))
        assert plan.workload_class == "serving"

    def test_gzipped_input(self, tmp_path):
        p = tmp_path / "log.csv.gz"
        with gzip.open(p, "wt", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "prompt_tokens", "output_tokens"])
            for i in range(30):
                w.writerow([1772000000 + i, 3, 4])
        table = read_table(p)
        assert table.columns == ["timestamp", "prompt_tokens", "output_tokens"]
        assert sum(1 for _ in table.iter_rows()) == 30


# ---------------------------------------------------------------------------
# Never-crash contract
# ---------------------------------------------------------------------------

class TestNeverCrash:
    def test_missing_file(self):
        with pytest.raises(TableReadError):
            read_table("/nonexistent/never.csv")

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        with pytest.raises(TableReadError):
            read_table(p)

    def test_unrelated_columns_refuse_replay(self, tmp_path):
        p = _write_csv(
            tmp_path / "weird.csv",
            ["alpha", "beta", "gamma"],
            [[1, 2, 3] for _ in range(200)],
        )
        table = read_table(p)
        plan = build_mapping_plan(table)
        ext = extract(table, plan)
        readiness = validate(plan, ext)
        assert readiness.tier == TIER_INSUFFICIENT
        assert readiness.refusal_reasons

    def test_partially_corrupt_rows_are_ledgered(self, tmp_path):
        rows = [[1772000000 + i, 10, 20] for i in range(200)]
        rows[50] = ["not-a-time", 10, 20]
        rows[60] = [1772000060, "x", 20]
        p = _write_csv(
            tmp_path / "messy.csv",
            ["timestamp", "prompt_tokens", "output_tokens"], rows)
        table = read_table(p)
        plan = build_mapping_plan(table)
        ext = extract(table, plan)
        assert ext.rows_read == 200
        assert ext.rows_usable == 198
        assert ext.errors.counts.get("row_dropped_timestamp_unparseable") == 1
        assert ext.errors.counts.get("prompt_tokens_unparseable") == 1
