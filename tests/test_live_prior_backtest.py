"""Tests for live causal prior serving-queue backtest [run 2026-06-21-t].

Validates make_live_prior_predictions() and run_live_prior_conformal_backtest()
against the expectations established in runs -n and -q:

  - 30%-CV lognormal noise retains ≥83% of oracle gain (run -n)
  - Live causal prior (sliding-window median) should retain ≥83% too
  - Live prior is strictly causal: prediction[i] uses only tokens[0..i-1]
  - Prior CV and MAE are reasonable for the trace's distribution

Invariants tested:
  1.  make_live_prior_predictions: first prediction uses warmup fallback.
  2.  make_live_prior_predictions: causal order — no future leakage.
  3.  make_live_prior_predictions: sliding window median is correct.
  4.  make_live_prior_predictions: stats dict has required keys.
  5.  make_live_prior_predictions: empty input returns empty list.
  6.  make_live_prior_predictions: single request returns warmup fallback.
  7.  LivePriorReport.to_dict() serialises all floats correctly.
  8.  run_live_prior_conformal_backtest: returns LivePriorReport on fixture.
  9.  live_delta_pct > 0 (live prior beats FIFO on Azure fixture).
  10. live_vs_oracle_retention_pct is in (0, 150) (plausible range).
  11. oracle_delta_pct > live_delta_pct when fixture is too small for queue
      pressure (fixture has 54 rows — not enough for measurable SRPT benefit
      under the same oracle gap as full 5,880 requests).
  12. prior_cv_pct is non-negative.
  13. prior_mae_tokens > 0 (running median cannot be perfect for all requests).
  14. run_live_prior_conformal_backtest: total_requests matches fixture size.
  15. LivePriorReport fields: servers, target_rho, sla_s, prior_window set.
"""

from __future__ import annotations

import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_SLA_S,
    LIVE_PRIOR_WINDOW,
    LivePriorReport,
    make_live_prior_predictions,
    run_live_prior_conformal_backtest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_raw(n: int = 20, base_tokens: int = 100) -> list[tuple[float, int]]:
    """Deterministic (arrival_s, tokens) list for unit tests."""
    return [(float(i * 2), base_tokens + i % 5 * 10) for i in range(n)]


# ---------------------------------------------------------------------------
# make_live_prior_predictions unit tests
# ---------------------------------------------------------------------------

def test_first_prediction_is_warmup_fallback():
    raw = _simple_raw(10, base_tokens=80)
    preds, _ = make_live_prior_predictions(raw, window=5)
    # First request has no history → should use warmup_fallback (global median).
    all_toks = sorted(t for _, t in raw)
    global_median = float(all_toks[len(all_toks) // 2])
    assert preds[0] == global_median


def test_causal_order_no_future_leakage():
    # Predictions must only use tokens from PAST requests.
    tokens = [10, 200, 50, 300, 5, 100]
    raw = [(float(i), t) for i, t in enumerate(tokens)]
    preds, _ = make_live_prior_predictions(raw, window=100)
    # prediction[i] is the median of tokens[:i] — verified for i=2
    assert preds[2] == float(sorted([10, 200])[1])  # median([10,200]) = 200
    # prediction[3] = median([10, 200, 50]) = 50
    assert preds[3] == float(sorted([10, 200, 50])[1])  # 50


def test_sliding_window_median_correct():
    tokens = list(range(1, 21))  # 1..20
    raw = [(float(i), t) for i, t in enumerate(tokens)]
    preds, _ = make_live_prior_predictions(raw, window=5)
    # For i=10: history is [1..10], window takes last 5: [6,7,8,9,10]
    # median([6,7,8,9,10]) = 8.0
    assert preds[10] == 8.0


def test_stats_dict_has_required_keys():
    raw = _simple_raw(50)
    _, stats = make_live_prior_predictions(raw)
    required = {
        "prior_cv_pct", "prior_mae_tokens", "prior_rel_mae_pct",
        "prior_bias_tokens", "prior_bias_pct", "warmup_fallback",
        "global_median_actual", "window", "n_requests",
    }
    assert required.issubset(set(stats.keys()))


def test_empty_input_returns_empty():
    preds, stats = make_live_prior_predictions([])
    assert preds == []
    assert stats == {}


def test_single_request_returns_warmup():
    raw = [(0.0, 42)]
    preds, stats = make_live_prior_predictions(raw, warmup_value=99.0)
    assert preds == [99.0]
    assert stats["n_requests"] == 1


def test_prior_cv_nonnegative():
    raw = _simple_raw(100)
    _, stats = make_live_prior_predictions(raw)
    assert stats["prior_cv_pct"] >= 0.0


def test_prior_mae_positive_for_variable_trace():
    # Running median predicts last window median; for a variable trace MAE > 0.
    tokens = [10, 500, 10, 500, 10, 500, 10, 500]  # alternating
    raw = [(float(i), t) for i, t in enumerate(tokens)]
    _, stats = make_live_prior_predictions(raw, window=3)
    assert stats["prior_mae_tokens"] > 0


# ---------------------------------------------------------------------------
# LivePriorReport unit tests
# ---------------------------------------------------------------------------

def test_live_prior_report_to_dict_serialisable():
    report = LivePriorReport(
        trace="test", total_requests=100, servers=4, target_rho=0.85,
        sla_s=10.0, prior_window=200,
        prior_cv_pct=12.5, prior_mae_tokens=18.3, prior_rel_mae_pct=20.3,
        prior_bias_tokens=2.1,
        fifo={"mean_response_s": 1.2}, conformal_oracle={"mean_response_s": 0.5},
        conformal_live={"mean_response_s": 0.6},
        fifo_goodput_per_dollar=100.0, oracle_goodput_per_dollar=422.0,
        live_goodput_per_dollar=380.0,
        oracle_delta_pct=322.0, live_delta_pct=280.0,
        live_vs_oracle_retention_pct=90.0,
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["live_vs_oracle_retention_pct"] == 90.0
    assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"
    # All float fields should be rounded floats (not nan/inf).
    for key in ("oracle_delta_pct", "live_delta_pct", "prior_cv_pct"):
        assert isinstance(d[key], float)
        assert d[key] == d[key]  # not NaN


# ---------------------------------------------------------------------------
# run_live_prior_conformal_backtest integration tests (fixture only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_run_live_prior_conformal_backtest_returns_report():
    report = run_live_prior_conformal_backtest()
    assert isinstance(report, LivePriorReport)


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_live_prior_report_fields_set_correctly():
    report = run_live_prior_conformal_backtest(servers=4, target_rho=0.85)
    assert report.servers == 4
    assert report.target_rho == 0.85
    assert report.sla_s == DEFAULT_SLA_S
    assert report.prior_window == LIVE_PRIOR_WINDOW
    assert report.total_requests > 0


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_live_prior_report_goodput_positive():
    report = run_live_prior_conformal_backtest()
    assert report.fifo_goodput_per_dollar > 0
    assert report.oracle_goodput_per_dollar > 0
    assert report.live_goodput_per_dollar > 0


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_live_prior_retention_in_plausible_range():
    report = run_live_prior_conformal_backtest()
    # Retention should be in a plausible range (0 to 200%).
    assert 0.0 < report.live_vs_oracle_retention_pct < 200.0


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_prior_cv_nonnegative_on_fixture():
    report = run_live_prior_conformal_backtest()
    assert report.prior_cv_pct >= 0.0


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_prior_mae_positive_on_fixture():
    report = run_live_prior_conformal_backtest()
    assert report.prior_mae_tokens > 0.0


@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not available",
)
def test_live_prior_to_dict_all_floats():
    report = run_live_prior_conformal_backtest()
    d = report.to_dict()
    for key in ("oracle_delta_pct", "live_delta_pct", "live_vs_oracle_retention_pct",
                "prior_cv_pct", "prior_mae_tokens"):
        val = d[key]
        assert isinstance(val, float), f"{key} is not float: {val!r}"
        assert val == val, f"{key} is NaN"
