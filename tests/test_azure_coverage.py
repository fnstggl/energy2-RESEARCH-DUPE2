"""Azure trace coverage + hourly forecasting — proves PR #96 forecasts on the full week.

Background: the forecasting/MPC stack originally trained on the 2023 **one-hour** Splitwise
trace (per-minute periods). The audit (``research/AZURE_TRACE_COVERAGE_AUDIT.md``) found the
public source also ships the 2024 **one-week** trace (168 h, ~44 M requests), which was
unwired. These tests lock in the corrected behaviour:

- ``ingest_azure`` prefers the one-week trace, falls back to one-hour, then the committed
  sample — never a silent downgrade.
- ``azure_period_frames`` bins the full week in bounded memory: EXACT per-bin arrival counts
  (proven on synthetic rows, fast) + a proportional stride sample, and — when the real
  one-week trace is present locally — the full 168 hourly periods.
- the forecaster's seasonal features generalise to the **hourly** cycle (cycle_len=24,
  auto-detected), and a learnable diurnal signal still beats naive.
- the controller's bounded ``sim_seconds`` decision window cuts cost without changing the
  decision contract.

The multi-GB real CSVs are gitignored, so the full-trace assertions skip cleanly in CI.
"""

from __future__ import annotations

import math
import os

import pytest

from aurelius.environment.data_tier import FULL_TRACE, SAMPLE_FIXTURE
from aurelius.environment.forecasting import ForecastingModel, PeriodFrame, build_frames, fit_target
from aurelius.environment.ingestion import azure
from aurelius.environment.ingestion.azure import (
    FULL_1WEEK_CODE,
    FULL_1WEEK_CONV,
    _active_files,
    _bin_stream,
    _parse_ts,
    _primary_file,
    azure_period_frames,
    ingest_azure,
    to_serving_raw,
)

_HAS_1WEEK = os.path.exists(FULL_1WEEK_CONV)


# --- timestamp parsing (both dataset eras) ---------------------------------

def test_parse_ts_handles_2024_tzaware_and_2023_forms():
    assert _parse_ts("2024-05-12 00:00:00.001163+00:00") is not None     # tz-aware, sub-µs
    assert _parse_ts("2023-11-16 18:15:46.6805900") is not None          # naive, 7 frac digits
    # only deltas are used downstream — a clean one-second gap parses as exactly 1.0s
    assert _parse_ts("2024-05-12 00:00:01+00:00") - _parse_ts("2024-05-12 00:00:00+00:00") == 1.0
    assert _parse_ts("") is None and _parse_ts("not-a-date") is None


# --- bounded-memory binner (synthetic — fast + exact) -----------------------

def _synthetic_rows():
    """3 hours of (ts, ctx, gen) with KNOWN per-hour counts: bin0=5, bin1=3, bin2=2."""
    rows, base = [], 1_000_000.0
    for i in range(5):
        rows.append((base + i * 60.0, 1000 + i, 10 + i))            # hour 0 (ctx=1000+, gen=10+)
    for i in range(3):
        rows.append((base + 3600.0 + i * 120.0, 2000 + i, 20 + i))  # hour 1
    for i in range(2):
        rows.append((base + 7200.0 + i * 300.0, 3000 + i, 30 + i))  # hour 2
    return rows


def test_bin_stream_exact_counts_and_proportional_sample():
    per, exact = _bin_stream(_synthetic_rows(), bin_seconds=3600.0, sample_stride=1)
    assert exact == {0: 5, 1: 3, 2: 2}                              # EXACT, every row counted
    assert {b: len(v) for b, v in per.items()} == {0: 5, 1: 3, 2: 2}   # stride 1 → all kept
    # records are (arrival_s_from_start, out_tok=gen, in_tok=ctx)
    assert per[0][0] == (0.0, 10, 1000)


def test_bin_stream_stride_keeps_proportional_subsample_but_counts_all():
    rows = [(float(i), 50, 7) for i in range(100)]                  # 100 rows in bin 0
    per, exact = _bin_stream(rows, bin_seconds=3600.0, sample_stride=10)
    assert exact == {0: 100}                                        # exact count unaffected by stride
    assert len(per[0]) == 10                                        # 0,10,...,90


# --- file-preference logic (pure path checks, fast) -------------------------

def test_active_and_primary_file_preference_order():
    files, version, tier = _active_files()
    path, pversion, ptier = _primary_file()
    if _HAS_1WEEK:
        assert "1week" in version and tier == FULL_TRACE and files[0] == FULL_1WEEK_CONV
        if os.path.exists(FULL_1WEEK_CODE):
            assert FULL_1WEEK_CODE in files                         # both services recognised
        assert path == FULL_1WEEK_CONV and ptier == FULL_TRACE and "conv" in pversion
    else:
        assert tier in (FULL_TRACE, SAMPLE_FIXTURE) and ptier in (FULL_TRACE, SAMPLE_FIXTURE)


def test_ingest_azure_limit_and_relative_time():
    reqs, status = ingest_azure(limit=50)
    assert 0 < len(reqs) <= 50 and reqs[0][0] == 0.0               # capped, relative to first
    assert all(len(r) == 3 for r in reqs)
    assert all(b >= a for (a, *_), (b, *_2) in zip(reqs, reqs[1:]))   # sorted by arrival
    assert to_serving_raw(reqs)[0] == (0.0, reqs[0][2]) and status.source == "azure_llm"


# --- hourly forecasting: cycle_len generalisation ---------------------------

def test_forecaster_detects_hourly_cycle_and_beats_naive_on_diurnal():
    # 7 days of an hourly diurnal arrival series (period 24) + mild trend → learnable
    series = [20 + 9 * math.sin(2 * math.pi * (h % 24) / 24) + 0.3 * (h % 5) for h in range(168)]
    frames = [PeriodFrame(index=h, cycle_pos=h % 24, arrival_rate=v, n_requests=int(v),
                          output_token_mean=v, output_token_p95=v, input_token_mean=v,
                          interarrival_cv=0.0, electricity_price=0.05)
              for h, v in enumerate(series)]
    fm = ForecastingModel().fit(frames, train_frac=0.6)
    assert fm.cycle_len == 24                                       # auto-detected hourly cycle
    assert fm.forecasters["arrival_rate"].cycle_len == 24
    f = fit_target(frames, "arrival_rate", train_frac=0.6, cycle_len=24)
    assert f.holdout_metric <= f.naive_metric + 1e-9               # honesty guarantee
    p = fm.predict(frames[:120], horizon=2).at("arrival_rate", 0)
    assert p.p10 <= p.p50 <= p.p90 <= p.p99                        # calibrated band, ordered


def test_per_minute_cycle_still_60_backward_compatible():
    # a full hour of per-minute periods (cycle_pos 0..59) → cycle_len detected as 60, as before
    per = {p: [(p * 60 + i, 100 + i, 50 + i) for i in range(6)] for p in range(70)}
    fm = ForecastingModel().fit(build_frames(per, period_seconds=60.0, cycle_len=60), train_frac=0.6)
    assert fm.cycle_len == 60                                       # unchanged for per-minute


# --- controller bounded decision window -------------------------------------

def test_controller_sim_seconds_bounds_decision_cost():
    from aurelius.environment.controller import _synth_jobs
    full = _synth_jobs(2.0, 100, 500, 0.5, window_seconds=3600,
                       best_effort_fraction=0.3, kv_service_factor=1.0)
    bounded = _synth_jobs(2.0, 100, 500, 0.5, window_seconds=240,
                          best_effort_fraction=0.3, kv_service_factor=1.0)
    assert len(full) == 7200 and len(bounded) == 480              # n = rate * window
    assert full[0].cls == bounded[0].cls                          # same profile, fewer jobs


# --- the headline coverage claim: 168 clean hourly periods ------------------

@pytest.mark.skipif(not _HAS_1WEEK, reason="one-week trace not present (gitignored)")
def test_one_week_yields_168_hourly_periods_bounded_memory():
    """Audit's central claim, proven on the REAL trace. Streams the 27.3 M-row conv week."""
    pf = azure_period_frames(bin_seconds=3600.0, sample_stride=24)
    assert pf is not None and pf["tier"] == FULL_TRACE and pf["service"] == "conv"
    assert pf["n_bins"] == 168                                     # 7 days x 24 h, gap-free
    assert pf["total_requests"] == 27_303_999                     # exact, every row
    per = pf["per_period"]
    assert set(per) == set(range(168))                            # every hour has a sample
    # proportional sample → bounded but non-trivial; diurnal shape preserved
    loads = [len(per[b]) for b in range(168)]
    assert max(loads) > min(loads) * 1.3 and sum(loads) < pf["total_requests"]
    assert azure.DEFAULT_FULL_CAP == 200_000                      # list API stays bounded
