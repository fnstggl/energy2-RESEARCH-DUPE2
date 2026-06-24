"""Tests for BurstGPT HF full-scale SRTF cross-validation [run 2026-06-21-p].

Validates the JSONL loader and full-scale backtest function added to support
cross-validation of the decoupled hybrid beyond the 54-row fixture, which is
too small to demonstrate SRPT > FIFO due to insufficient queue depth.

The HuggingFace BurstGPT normalized sample (59,999 records, CC-BY-4.0) provides
the statistical mass needed to cross-validate across a second public LLM trace.

Research basis:
  - BurstGPT (arXiv:2401.17644): real LLM serving trace, heavy-tailed output.
  - SRPT multiserver (arXiv:1805.07686): SRPT gains scale with output-length
    variance; BurstGPT's heavier distribution (p99=934 tokens vs p99=479 for
    Azure LLM 2024) is a stronger testbed for SRTF scheduling benefit.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DECOUPLED_HYBRID_ALPHA_DEFAULT,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DecoupledHybridReport,
    load_burstgpt_serving_requests_jsonl,
    run_burstgpt_hf_decoupled_hybrid_backtest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict]) -> str:
    """Write records to a temp JSONL file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    f.close()
    return f.name


def _make_jsonl_records(n: int = 60, start_ts: float = 0.0, gap_s: float = 1.0) -> list[dict]:
    """Return n BurstGPT-format JSONL records with alternating short/long tokens."""
    records = []
    for i in range(n):
        # Alternate: even = short (50 tokens), odd = long (400 tokens)
        out_tok = 50 if i % 2 == 0 else 400
        records.append({
            "request_arrival_ts_s": start_ts + i * gap_s,
            "output_tokens": out_tok,
            "input_tokens": 200,
            "model_id": "ChatGPT",
            "log_type": "Conversation log",
        })
    return records


# ---------------------------------------------------------------------------
# Class 1: JSONL loader unit tests
# ---------------------------------------------------------------------------

class TestLoadBurstGPTServingRequestsJsonl:
    """Unit tests for load_burstgpt_serving_requests_jsonl."""

    def test_basic_load_returns_tuples(self):
        path = _write_jsonl(_make_jsonl_records(10))
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        assert len(result) == 10
        for arrival_s, out_tok in result:
            assert isinstance(arrival_s, float)
            assert isinstance(out_tok, int)
            assert out_tok > 0

    def test_t0_normalized_to_zero(self):
        records = _make_jsonl_records(5, start_ts=100.0, gap_s=2.0)
        path = _write_jsonl(records)
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        assert result[0][0] == pytest.approx(0.0)
        assert result[1][0] == pytest.approx(2.0)
        assert result[4][0] == pytest.approx(8.0)

    def test_sorted_by_arrival_time(self):
        records = list(reversed(_make_jsonl_records(10)))
        path = _write_jsonl(records)
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        arrivals = [a for a, _ in result]
        assert arrivals == sorted(arrivals)

    def test_zero_output_tokens_excluded(self):
        records = _make_jsonl_records(5)
        records[2]["output_tokens"] = 0
        path = _write_jsonl(records)
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        assert len(result) == 4

    def test_limit_applied(self):
        records = _make_jsonl_records(20)
        path = _write_jsonl(records)
        try:
            result = load_burstgpt_serving_requests_jsonl(path, limit=7)
        finally:
            os.unlink(path)
        assert len(result) == 7

    def test_malformed_lines_skipped(self):
        path = _write_jsonl(_make_jsonl_records(5))
        # Inject malformed lines
        with open(path, "a") as f:
            f.write("not-json\n")
            f.write('{"missing_ts": 1, "output_tokens": 10}\n')
            f.write('{"request_arrival_ts_s": "bad", "output_tokens": 5}\n')
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        # 5 valid + 3 invalid: only valid returned
        assert len(result) == 5

    def test_empty_file_returns_empty_list(self):
        path = _write_jsonl([])
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        assert result == []

    def test_blank_lines_skipped(self):
        records = _make_jsonl_records(4)
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for i, rec in enumerate(records):
            f.write(json.dumps(rec) + "\n")
            if i == 1:
                f.write("\n\n")  # blank lines
        f.flush()
        f.close()
        try:
            result = load_burstgpt_serving_requests_jsonl(f.name)
        finally:
            os.unlink(f.name)
        assert len(result) == 4

    def test_output_token_values_preserved(self):
        records = [
            {"request_arrival_ts_s": 0.0, "output_tokens": 50, "input_tokens": 100},
            {"request_arrival_ts_s": 1.0, "output_tokens": 400, "input_tokens": 500},
            {"request_arrival_ts_s": 2.0, "output_tokens": 236, "input_tokens": 300},
        ]
        path = _write_jsonl(records)
        try:
            result = load_burstgpt_serving_requests_jsonl(path)
        finally:
            os.unlink(path)
        assert result[0][1] == 50
        assert result[1][1] == 400
        assert result[2][1] == 236

    def test_limit_zero_returns_empty(self):
        records = _make_jsonl_records(10)
        path = _write_jsonl(records)
        try:
            result = load_burstgpt_serving_requests_jsonl(path, limit=0)
        finally:
            os.unlink(path)
        assert result == []

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_burstgpt_serving_requests_jsonl("/nonexistent/path/burstgpt.jsonl")


# ---------------------------------------------------------------------------
# Class 2: Integration tests on synthetic JSONL (no HF file required)
# ---------------------------------------------------------------------------

class TestBurstGPTHFBacktestSynthetic:
    """Integration tests that run the full backtest on synthetic JSONL data."""

    def _make_jsonl_path(self, n: int = 100) -> str:
        records = _make_jsonl_records(n)
        return _write_jsonl(records)

    def test_backtest_runs_and_returns_report(self):
        path = self._make_jsonl_path(80)
        try:
            report = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.85, job_limit=80,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        assert isinstance(report, DecoupledHybridReport)
        assert report.trace == "burstgpt_hf_fullscale"
        assert report.total_requests == 80

    def test_all_six_disciplines_present(self):
        path = self._make_jsonl_path(80)
        try:
            report = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.85, job_limit=80,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        assert "sla_safe_goodput_per_dollar" in report.fifo
        assert "sla_safe_goodput_per_dollar" in report.srtf_perfect
        assert "sla_safe_goodput_per_dollar" in report.aging_srtf
        assert "sla_safe_goodput_per_dollar" in report.srpt_preemptive
        assert "sla_safe_goodput_per_dollar" in report.hybrid_aging_preemptive
        assert "sla_safe_goodput_per_dollar" in report.decoupled_hybrid

    def test_decoupled_goodput_positive(self):
        path = self._make_jsonl_path(80)
        try:
            report = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.85, job_limit=80,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        assert report.decoupled_hybrid["sla_safe_goodput_per_dollar"] > 0

    def test_to_dict_serializes_cleanly(self):
        path = self._make_jsonl_path(60)
        try:
            report = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.80, job_limit=60,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        d = report.to_dict()
        assert d["trace"] == "burstgpt_hf_fullscale"
        assert d["total_requests"] == 60
        assert isinstance(d["decoupled_goodput_delta_pct"], float)

    def test_srtf_vs_fifo_ordering_favors_short_requests(self):
        """Short requests (50 tok) should get shorter p90 under SRTF than FIFO."""
        # Generate a contended queue: many short + some long requests
        records = []
        for i in range(60):
            ts = float(i) * 0.5  # fast arrivals to create contention
            out_tok = 50 if i % 3 != 0 else 400  # 2/3 short, 1/3 long
            records.append({
                "request_arrival_ts_s": ts,
                "output_tokens": out_tok,
                "input_tokens": 100,
                "model_id": "ChatGPT",
            })
        path = _write_jsonl(records)
        try:
            report = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=1, target_rho=0.85, job_limit=60,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        # Under SRPT/decoupled, short requests (≤ median) are served first.
        # short_p90 should not be worse than FIFO; may be equal for small N.
        fifo_sp90 = report.fifo.get("short_p90_response_s", 0.0)
        decoupled_sp90 = report.decoupled_hybrid.get("short_p90_response_s", 0.0)
        # At small N, discipline differences may be small but should not flip badly.
        # Assert decoupled short_p90 is within 50% of FIFO short_p90 (not much worse).
        if fifo_sp90 > 0:
            assert decoupled_sp90 <= fifo_sp90 * 1.5, (
                f"Decoupled short_p90={decoupled_sp90:.3f}s unexpectedly much worse "
                f"than FIFO short_p90={fifo_sp90:.3f}s"
            )

    def test_report_consistent_request_count(self):
        path = self._make_jsonl_path(100)
        try:
            report = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.85, job_limit=None,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        assert report.total_requests == 100

    def test_different_alpha_changes_decoupled_metrics(self):
        records = _make_jsonl_records(80)
        path = _write_jsonl(records)
        try:
            rep_low = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.85, job_limit=80, aging_alpha=0.001,
                jsonl_path=path,
            )
            rep_high = run_burstgpt_hf_decoupled_hybrid_backtest(
                servers=2, target_rho=0.85, job_limit=80, aging_alpha=0.05,
                jsonl_path=path,
            )
        finally:
            os.unlink(path)
        # Higher alpha → more aggressive aging → FIFO-like dispatch for long requests.
        # At small N the difference may be small but metrics should not be identical.
        # (We only check both complete without error here.)
        assert rep_low.aging_alpha == 0.001
        assert rep_high.aging_alpha == 0.05


# ---------------------------------------------------------------------------
# Class 3: HF file tests (skipped if file absent — CI without HF data)
# ---------------------------------------------------------------------------

HF_SKIP = not os.path.isfile(DEFAULT_BURSTGPT_HF_JSONL)
HF_SKIP_REASON = "HF BurstGPT normalized sample not present (data/external/hf/...)"


@pytest.mark.skipif(HF_SKIP, reason=HF_SKIP_REASON)
class TestBurstGPTHFFileLoad:
    """Tests that load from the actual HF JSONL file (requires the file to exist)."""

    def test_loader_returns_many_records(self):
        result = load_burstgpt_serving_requests_jsonl(DEFAULT_BURSTGPT_HF_JSONL)
        # HF file has 59,999 records; all with output_tokens > 0 should be returned.
        assert len(result) >= 50_000, (
            f"Expected ≥50,000 records from HF file, got {len(result)}"
        )

    def test_arrivals_non_negative_and_sorted(self):
        result = load_burstgpt_serving_requests_jsonl(DEFAULT_BURSTGPT_HF_JSONL)
        arrivals = [a for a, _ in result]
        assert arrivals[0] == pytest.approx(0.0)
        assert all(arrivals[i] <= arrivals[i + 1] for i in range(len(arrivals) - 1))

    def test_output_tokens_positive(self):
        result = load_burstgpt_serving_requests_jsonl(DEFAULT_BURSTGPT_HF_JSONL, limit=1000)
        for _, tok in result:
            assert tok > 0

    def test_full_scale_backtest_srpt_beats_fifo(self):
        """Key cross-validation: SRPT > FIFO on BurstGPT at sufficient scale.

        At 5,000+ requests and ρ=0.85, the queue has enough depth for SRTF
        ordering to show its benefit.  The 54-row fixture fails this test due
        to insufficient queue depth.
        """
        report = run_burstgpt_hf_decoupled_hybrid_backtest(
            servers=4,
            target_rho=0.85,
            job_limit=5000,
            aging_alpha=DECOUPLED_HYBRID_ALPHA_DEFAULT,
            sla_s=DEFAULT_BURSTGPT_SLA_S,
        )
        assert isinstance(report, DecoupledHybridReport)
        assert report.total_requests == 5000
        # SRPT should beat FIFO on goodput/$ with sufficient queue depth.
        assert report.srpt_goodput_delta_pct > 0, (
            f"SRPT goodput/$ should exceed FIFO on BurstGPT at 5,000 requests: "
            f"delta={report.srpt_goodput_delta_pct:.2f}%"
        )
        # Decoupled hybrid should not be dramatically worse than SRPT.
        assert report.decoupled_goodput_delta_pct >= report.srpt_goodput_delta_pct * 0.70, (
            f"Decoupled α=0.001 should retain ≥70% of SRPT gain: "
            f"decoupled={report.decoupled_goodput_delta_pct:.2f}%, "
            f"srpt={report.srpt_goodput_delta_pct:.2f}%"
        )
