"""Tests for v2026 incremental FULL_TRACE_EXACT streaming ingestion.

Proves the exact aggregators are mathematically equivalent to conventional full
processing, that streaming a local parquet zip == loading it whole, that
resume/checkpoint works, that disk stays bounded, that labels are correct, and
that no raw trace data is committed. The network/full-trace path is exercised only
when V2026_RAW_DIR / V2026_PROCESSED_DIR is set (opt-in).
"""

from __future__ import annotations

import os
import statistics
import zipfile

import pytest

from aurelius.environment.data_tier import (
    FULL_TRACE_EXACT,
    SUBSET_TRACE,
)
from aurelius.environment.ingestion.v2026_stream import (
    ExactCounter,
    ExactHistogram,
    ExactStats,
    stream_local_zip,
)

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# --- aggregator exactness (no network, no parquet) -------------------------

def test_exact_stats_chunked_equals_whole():
    xs = [float(i % 37) + 0.5 for i in range(1000)]
    whole = ExactStats()
    whole.update(xs)
    # process in 7 arbitrary chunks + merge → must match whole exactly
    chunks = [ExactStats() for _ in range(7)]
    for i, x in enumerate(xs):
        chunks[i % 7].update([x])
    merged = ExactStats()
    for c in chunks:
        merged.merge(c)
    assert merged.to_dict()["n"] == whole.to_dict()["n"] == 1000
    assert abs(merged.to_dict()["mean"] - statistics.mean(xs)) < 1e-9
    assert abs(merged.to_dict()["variance"] - statistics.pvariance(xs)) < 1e-6
    assert merged.to_dict()["min"] == min(xs) and merged.to_dict()["max"] == max(xs)
    assert whole.to_dict()["label"] == FULL_TRACE_EXACT


def test_exact_counter_and_histogram_merge_and_serialize():
    c = ExactCounter()
    c.update(["HP", "LP", "HP", "Other"])
    assert c.to_dict()["counts"] == {"HP": 2, "LP": 1, "Other": 1}
    assert ExactCounter.from_state(c.state()).to_dict() == c.to_dict()
    h = ExactHistogram(0.0, 10.0, 10)
    h.update([1.0, 1.0, 9.5, 100.0, -1.0])
    assert h.above == 1 and h.below == 1
    assert ExactHistogram.from_state(h.state()).to_dict() == h.to_dict()
    assert h.to_dict()["label"] == "FULL_TRACE_APPROX"


# --- local-zip streaming == conventional full load (exactness) -------------

def _make_zip(tmp_path, n_parts=3, rows=500):
    """Build a local zip of STORED parquet partitions (network_hourly schema)."""
    zpath = os.path.join(tmp_path, "net.zip")
    all_rx = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for d in range(n_parts):
            rx = [float((i * 7 + d) % 50) * 0.1 for i in range(rows)]
            tx = [v * 0.8 for v in rx]
            all_rx += rx
            t = pa.table({"server_id": [f"s{i}" for i in range(rows)],
                          "rx_gibps_avg": rx, "tx_gibps_avg": tx})
            pf = os.path.join(tmp_path, f"part{d}.parquet")
            pq.write_table(t, pf)
            zf.write(pf, f"net/day=1/hour=0{d}/part-000.parquet")
            os.remove(pf)
    return zpath, all_rx


def _net_aggs():
    return {"rx": ExactStats()}


def _net_fold(aggs, t):
    aggs["rx"].update(t.column("rx_gibps_avg").to_pylist())


def test_streaming_equals_conventional(tmp_path):
    zpath, all_rx = _make_zip(tmp_path)
    work = os.path.join(tmp_path, "work")
    man = os.path.join(tmp_path, "man.json")
    res = stream_local_zip(zpath, _net_aggs, _net_fold, work_dir=work, manifest_path=man)
    d = res.artifacts["rx"]
    # streamed exact stats == conventional stats over all rows
    assert d["n"] == len(all_rx)
    assert abs(d["mean"] - statistics.mean(all_rx)) < 1e-9
    assert abs(d["variance"] - statistics.pvariance(all_rx)) < 1e-6
    assert res.label == FULL_TRACE_EXACT          # all partitions → exact
    assert not os.listdir(work)                    # bounded disk: cleaned up


def test_resume_and_checkpoint(tmp_path):
    zpath, all_rx = _make_zip(tmp_path, n_parts=4)
    work = os.path.join(tmp_path, "work")
    man = os.path.join(tmp_path, "man.json")
    r1 = stream_local_zip(zpath, _net_aggs, _net_fold, work_dir=work,
                          manifest_path=man, max_partitions=2)
    assert r1.n_partitions_done == 2 and r1.label == SUBSET_TRACE   # partial → honest label
    assert os.path.exists(man)                                      # checkpoint persisted
    n_after_2 = r1.artifacts["rx"]["n"]
    # resume: skip the 2 done, process the rest, merge exactly
    r2 = stream_local_zip(zpath, _net_aggs, _net_fold, work_dir=work, manifest_path=man)
    assert r2.n_partitions_done == 4 and r2.label == FULL_TRACE_EXACT
    assert r2.artifacts["rx"]["n"] == len(all_rx) > n_after_2
    assert abs(r2.artifacts["rx"]["mean"] - statistics.mean(all_rx)) < 1e-9


# --- no raw data committed -------------------------------------------------

def test_no_raw_trace_data_committed():
    """The raw v2026/Mooncake/Azure trace files this PR ingests must not be tracked."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dirs = ("data/external/alibaba_gpu_v2026/raw",
                "data/external/mooncake/raw",
                "data/external/azure_llm_2024/raw")
    tracked = subprocess.run(["git", "ls-files", *raw_dirs], cwd=repo,
                             capture_output=True, text=True).stdout.splitlines()
    bad = [f for f in tracked if not f.endswith(".gitkeep")]
    assert bad == [], f"raw trace data must not be committed: {bad}"


# --- optional full-trace test (network) ------------------------------------

@pytest.mark.skipif(not os.environ.get("V2026_PROCESSED_DIR"),
                    reason="set V2026_PROCESSED_DIR to run the full-trace check")
def test_full_trace_artifact_present():
    import json
    p = os.path.join(os.environ["V2026_PROCESSED_DIR"], "network_hourly_calibration.json")
    d = json.load(open(p))
    assert d["complete"] and d["label"] == FULL_TRACE_EXACT
