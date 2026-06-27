"""Azure trace coverage — proves the ingestion loads the *intended* files.

Background: PR-#95-era ingestion only wired the 2023 one-hour Splitwise files
(`AzureLLMInferenceTrace_{conv,code}.csv`, ~1 h). The audit
(`research/AZURE_TRACE_COVERAGE_AUDIT.md`) found the public source also publishes the
**2024 one-week** trace (`..._1week.csv`, 168 h, ~44 M requests). These tests lock in the
corrected behaviour:

- `ingest_azure` prefers the one-week trace, falls back to one-hour, then the committed
  sample — never a silent downgrade, with the tier reported honestly.
- the bounded-memory hourly binner produces correct exact per-bin arrival counts (proven
  on synthetic rows so it is fast + deterministic), and — when the real one-week trace is
  present locally — yields the full 168 hourly periods that make hourly forecasting valid.

The multi-GB real CSVs are gitignored, so the full-trace assertions are guarded on file
existence and skip cleanly in CI (where only the committed sample is present).
"""

from __future__ import annotations

import os

import pytest

from aurelius.environment.data_tier import FULL_TRACE, SAMPLE_FIXTURE
from aurelius.environment.ingestion import azure
from aurelius.environment.ingestion.azure import (
    FULL_1WEEK_CODE,
    FULL_1WEEK_CONV,
    _active_files,
    _bin_rows,
    _parse_ts,
    _primary_file,
    hourly_arrival_frames,
    ingest_azure,
    to_serving_raw,
)

_HAS_1WEEK = os.path.exists(FULL_1WEEK_CONV)


# --- timestamp parsing (both dataset eras) ---------------------------------

def test_parse_ts_handles_2024_tzaware_and_2023_forms():
    # 2024 one-week form: tz-aware, sub-microsecond fractional seconds
    t2024 = _parse_ts("2024-05-12 00:00:00.001163+00:00")
    # 2023 one-hour form: naive, 7 fractional digits
    t2023 = _parse_ts("2023-11-16 18:15:46.6805900")
    assert t2024 is not None and t2023 is not None
    # only deltas are used downstream — a clean one-second gap parses as exactly 1.0s
    assert _parse_ts("2024-05-12 00:00:01+00:00") - _parse_ts("2024-05-12 00:00:00+00:00") == 1.0
    assert _parse_ts("") is None and _parse_ts("not-a-date") is None


# --- bounded-memory hourly binner (synthetic — fast + exact) ----------------

def _synthetic_rows():
    """3 hours of arrivals with KNOWN per-hour counts: bin0=5, bin1=3, bin2=2.
    Timestamps in seconds-from-epoch-ish floats; tokens vary so stats are non-trivial."""
    rows = []
    base = 1_000_000.0
    for i in range(5):                 # hour 0
        rows.append((base + i * 60.0, 100 + i, 10 + i))
    for i in range(3):                 # hour 1
        rows.append((base + 3600.0 + i * 120.0, 200 + i, 20 + i))
    for i in range(2):                 # hour 2
        rows.append((base + 7200.0 + i * 300.0, 300 + i, 30 + i))
    return rows


def test_bin_rows_exact_counts_and_structure():
    frames = _bin_rows(_synthetic_rows(), bin_seconds=3600.0, sample_stride=1, cycle_len=24)
    assert [f["index"] for f in frames] == [0, 1, 2]
    assert [f["n_requests"] for f in frames] == [5, 3, 2]          # EXACT, every row counted
    assert [f["hour_of_day"] for f in frames] == [0, 1, 2]
    # arrival_rate = exact count / bin_seconds
    assert abs(frames[0]["arrival_rate_per_s"] - 5 / 3600.0) < 1e-12
    # token stats come through (stride=1 → all sampled)
    assert frames[0]["output_token_mean"] == sum(10 + i for i in range(5)) / 5
    assert frames[0]["n_sampled"] == 5


def test_bin_rows_hour_of_day_wraps_past_24h():
    # one arrival in hour 0 and one in hour 25 → hour_of_day wraps 25 % 24 == 1
    rows = [(0.0, 10, 5), (25 * 3600.0 + 1.0, 10, 5)]
    frames = _bin_rows(rows, bin_seconds=3600.0, sample_stride=1, cycle_len=24)
    assert [f["index"] for f in frames] == [0, 25]
    assert [f["hour_of_day"] for f in frames] == [0, 1]


def test_bin_rows_stride_samples_but_counts_all():
    # 100 rows in one hour, stride 10 → exact count 100, but only ~10 sampled for stats
    rows = [(float(i), 50, 7) for i in range(100)]
    frames = _bin_rows(rows, bin_seconds=3600.0, sample_stride=10, cycle_len=24)
    assert frames[0]["n_requests"] == 100
    assert frames[0]["n_sampled"] == 10                            # 0,10,20,...,90


# --- file-preference logic (pure path checks, fast) -------------------------

def test_active_and_primary_file_preference_order():
    files, version, tier = _active_files()
    path, pversion, ptier = _primary_file()
    if _HAS_1WEEK:
        # one-week present → preferred, FULL_TRACE, conv listed first; code wired too
        assert "1week" in version and tier == FULL_TRACE
        assert files and files[0] == FULL_1WEEK_CONV
        if os.path.exists(FULL_1WEEK_CODE):
            assert FULL_1WEEK_CODE in files                        # both services recognised
        assert path == FULL_1WEEK_CONV and ptier == FULL_TRACE and "conv" in pversion
    else:
        # no one-week locally: either the 2023 one-hour files (FULL_TRACE) or the sample
        assert tier in (FULL_TRACE, SAMPLE_FIXTURE)
        assert ptier in (FULL_TRACE, SAMPLE_FIXTURE)


# --- ingest_azure list API --------------------------------------------------

def test_ingest_azure_limit_and_relative_time():
    reqs, status = ingest_azure(limit=50)
    assert len(reqs) <= 50 and reqs                                 # sample fixture guarantees >0
    assert reqs[0][0] == 0.0                                        # arrivals relative to first
    assert all(len(r) == 3 for r in reqs)                          # (arrival, ctx, gen)
    assert all(b >= a for (a, _, _), (b, _2, _3) in zip(reqs, reqs[1:]))  # sorted by arrival
    assert to_serving_raw(reqs)[0] == (0.0, reqs[0][2])            # (arrival, output_tokens)
    assert status.source == "azure_llm"


@pytest.mark.skipif(not _HAS_1WEEK, reason="one-week trace not present (gitignored)")
def test_ingest_azure_reports_full_trace_and_caps_unlimited():
    reqs, status = ingest_azure(limit=10)
    assert status.tier == FULL_TRACE and "1week" in status.trace_version
    # an unlimited one-week call must NOT materialise ~44 M rows — it caps + says so
    capped, st = ingest_azure()
    assert len(capped) == azure.DEFAULT_FULL_CAP
    assert "capped" in st.blocked_reason and "hourly_arrival_frames" in st.blocked_reason


# --- the headline coverage claim: 168 clean hourly periods ------------------

@pytest.mark.skipif(not _HAS_1WEEK, reason="one-week trace not present (gitignored)")
def test_one_week_trace_yields_168_hourly_periods():
    """The audit's central claim — proven against the REAL trace when present. Streams the
    full 27.3 M-row conv week once (bounded memory)."""
    r = hourly_arrival_frames(bin_seconds=3600.0)
    assert r is not None and r["tier"] == FULL_TRACE and r["service"] == "conv"
    assert r["n_bins"] == 168                                       # 7 days x 24 h, no gaps
    assert r["total_requests"] == 27_303_999                       # every row counted, exact
    hours = [f["hour_of_day"] for f in r["frames"]]
    assert set(hours) == set(range(24))                            # full diurnal coverage
    rates = [f["arrival_rate_per_s"] for f in r["frames"]]
    assert min(rates) > 0 and max(rates) > min(rates) * 1.5        # a real (non-flat) diurnal signal
