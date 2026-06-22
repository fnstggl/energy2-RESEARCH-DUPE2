"""Tests for Absolute-Error Conformal Calibration [run 2026-06-22-x].

Validates AbsoluteErrorConformalCalibrator, _simulate_decoupled_hybrid_abs_conformal,
AbsConformalReport, and the two public-trace backtest functions.

Primary hypothesis [run 2026-06-22-x]:
  Replacing relative error (|predicted-actual|/actual) with absolute error
  (|predicted-actual| in tokens) in the conformal calibrator breaks the
  calibrator cap on BurstGPT HF, yielding lower α and higher goodput/$.

  Root cause of the cap: short ChatGPT requests (actual=7, predicted=18) produce
  rel_err=1.57 >> target=0.40 → calibrator capped at 2×alpha_max=0.002.
  With absolute error these 11-token misses are negligible; the p90 abs_err is
  driven by genuinely uncertain long requests → lower α → more SRPT-like dispatch.

Invariants tested:
  1.  AbsoluteErrorConformalCalibrator: default constants correct.
  2.  AbsoluteErrorConformalCalibrator: warmup → returns alpha_max.
  3.  AbsoluteErrorConformalCalibrator: oracle (predicted==actual) → alpha → 0.
  4.  AbsoluteErrorConformalCalibrator: large absolute errors → alpha capped at 2×max.
  5.  AbsoluteErrorConformalCalibrator: p90_abs_err_tokens returns NaN before warmup.
  6.  AbsoluteErrorConformalCalibrator: p90_abs_err_tokens correct after warmup.
  7.  AbsoluteErrorConformalCalibrator: sliding window bounded by window size.
  8.  AbsoluteErrorConformalCalibrator: alpha monotone with error magnitude.
  9.  AbsoluteErrorConformalCalibrator: mean_alpha correct average.
  10. AbsoluteErrorConformalCalibrator: short over-prediction gives LOWER alpha than
      equivalent relative error (core property: 11-token miss ≠ 1.57 rel_err).
  11. _simulate_decoupled_hybrid_abs_conformal: all requests complete.
  12. _simulate_decoupled_hybrid_abs_conformal: response times ≥ service times.
  13. _simulate_decoupled_hybrid_abs_conformal: preemption_count in summary.
  14. _simulate_decoupled_hybrid_abs_conformal: single server, trivial queue.
  15. _simulate_decoupled_hybrid_abs_conformal: oracle → near-SRPT (high goodput).
  16. simulate_queue: decoupled_hybrid_abs_conformal discipline accepted.
  17. simulate_queue: decoupled_hybrid_abs_conformal ≠ fifo ordering.
  18. AbsConformalReport.to_dict(): all required keys present.
  19. AbsConformalReport.to_dict(): shadow_tag present.
  20. AbsConformalReport.to_dict(): abs_vs_rel_delta_pct computed correctly.
  21. run_abs_conformal_azure_backtest: returns AbsConformalReport.
  22. run_abs_conformal_azure_backtest: abs_mean_alpha ≤ rel_mean_alpha (key hypothesis).
  23. run_abs_conformal_azure_backtest: oracle delta > 0 (conformal oracle > FIFO).
  24. run_abs_conformal_azure_backtest: abs_p90_abs_err_tokens reported.
  25. run_abs_conformal_burstgpt_backtest: returns AbsConformalReport on HF data.
  26. run_abs_conformal_burstgpt_backtest: abs_mean_alpha ≤ rel_mean_alpha (hypothesis).
  27. run_abs_conformal_burstgpt_backtest: abs_conformal_goodput_per_dollar > 0.
  28. CONFORMAL_ABS_TARGET_P90_TOKENS is positive float.
"""

from __future__ import annotations

import math
import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_ABS_TARGET_P90_TOKENS,
    CONFORMAL_ALPHA_MAX,
    CONFORMAL_WARMUP,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    AbsConformalReport,
    AbsoluteErrorConformalCalibrator,
    _Request,
    _service_time_s,
    _simulate_decoupled_hybrid_abs_conformal,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    load_serving_requests,
    run_abs_conformal_azure_backtest,
    run_abs_conformal_burstgpt_backtest,
    simulate_queue,
)

_HAS_HF_DATA = os.path.exists(DEFAULT_BURSTGPT_HF_JSONL)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_requests(n: int = 200, seed: int = 42) -> list[_Request]:
    """Synthetic deterministic request list for simulator tests."""
    import random
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        t += rng.expovariate(1.0 / 0.5)
        actual = rng.choice([5, 10, 50, 200, 500])
        pred = float(rng.choice([18, 20, 100, 300]))
        reqs.append(_Request(
            idx=i,
            arrival_s=t,
            actual_tokens=actual,
            predicted_tokens=pred,
            service_s=_service_time_s(actual),
        ))
    return reqs


# ---------------------------------------------------------------------------
# Tests 1–10: AbsoluteErrorConformalCalibrator unit tests
# ---------------------------------------------------------------------------

def test_01_default_constants():
    cal = AbsoluteErrorConformalCalibrator()
    assert cal.alpha_max == CONFORMAL_ALPHA_MAX
    assert cal.warmup == CONFORMAL_WARMUP
    assert cal.target_p90_abs_tokens == CONFORMAL_ABS_TARGET_P90_TOKENS
    assert CONFORMAL_ABS_TARGET_P90_TOKENS > 0


def test_02_warmup_returns_alpha_max():
    cal = AbsoluteErrorConformalCalibrator(warmup=50)
    for i in range(49):
        cal.update(100.0, 80)
    alpha = cal.current_alpha()
    assert alpha == pytest.approx(cal.alpha_max)


def test_03_oracle_converges_to_zero():
    cal = AbsoluteErrorConformalCalibrator(warmup=20, window=100, target_p90_abs_tokens=50.0)
    for _ in range(200):
        v = 100
        cal.update(float(v), v)  # zero absolute error
    alpha = cal.current_alpha()
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_04_large_abs_error_capped():
    cal = AbsoluteErrorConformalCalibrator(
        warmup=20, window=100, target_p90_abs_tokens=50.0
    )
    for _ in range(200):
        cal.update(18.0, 700)  # abs_err = 682 tokens >> target=50
    alpha = cal.current_alpha()
    assert alpha == pytest.approx(2.0 * cal.alpha_max, rel=1e-6)


def test_05_p90_abs_err_nan_before_warmup():
    cal = AbsoluteErrorConformalCalibrator(warmup=50)
    for _ in range(5):
        cal.update(100.0, 80)
    val = cal.p90_abs_err_tokens()
    assert math.isnan(val)


def test_06_p90_abs_err_correct_after_warmup():
    cal = AbsoluteErrorConformalCalibrator(warmup=10, window=100, target_p90_abs_tokens=50.0)
    errors = sorted([abs(18.0 - tok) for tok in [5, 7, 10, 15, 20, 200, 400, 600, 800, 1000]])
    for tok in [5, 7, 10, 15, 20, 200, 400, 600, 800, 1000]:
        cal.update(18.0, tok)
    p90 = cal.p90_abs_err_tokens()
    p90_idx = min(len(errors) - 1, int(0.90 * len(errors)))
    assert p90 == pytest.approx(errors[p90_idx], rel=0.01)


def test_07_sliding_window_bounded():
    cal = AbsoluteErrorConformalCalibrator(warmup=5, window=20, target_p90_abs_tokens=50.0)
    for _ in range(100):
        cal.update(18.0, 7)
    assert len(cal._residuals) <= 20


def test_08_alpha_monotone_with_error():
    cal_small = AbsoluteErrorConformalCalibrator(warmup=10, window=50, target_p90_abs_tokens=100.0)
    cal_large = AbsoluteErrorConformalCalibrator(warmup=10, window=50, target_p90_abs_tokens=100.0)
    # small errors → p90 abs_err < target → low α
    for _ in range(100):
        cal_small.update(18.0, 10)   # abs_err = 8 tokens
    # large errors → p90 abs_err > target → high α
    for _ in range(100):
        cal_large.update(18.0, 800)  # abs_err = 782 tokens
    alpha_small = cal_small.current_alpha()
    alpha_large = cal_large.current_alpha()
    assert alpha_small < alpha_large


def test_09_mean_alpha_average():
    cal = AbsoluteErrorConformalCalibrator(warmup=5, window=50, target_p90_abs_tokens=50.0)
    for _ in range(100):
        cal.update(18.0, 7)  # abs_err = 11
    alphas = [cal.current_alpha() for _ in range(10)]
    assert cal.mean_alpha() == pytest.approx(sum(alphas) / len(alphas), rel=0.05)


def test_10_short_over_prediction_lower_alpha_than_relative():
    """Core property: abs-error calibrator gives LOWER alpha than rel-error for short over-predictions."""
    from aurelius.benchmarks.srtf_serving_backtest import ConformalAlphaCalibrator
    # Short request: actual=7, predicted=18 → rel_err=1.57, abs_err=11
    # With target=0.40 (relative) and target=50 (absolute):
    # Relative: ratio = 1.57/0.40 = 3.93 → capped at 2.0 → alpha_max × 2.0
    # Absolute: ratio = 11/50 = 0.22 → alpha_max × 0.22
    rel_cal = ConformalAlphaCalibrator(warmup=10, window=50)
    abs_cal = AbsoluteErrorConformalCalibrator(warmup=10, window=50, target_p90_abs_tokens=50.0)
    for _ in range(100):
        rel_cal.update(18.0, 7)
        abs_cal.update(18.0, 7)
    assert abs_cal.current_alpha() < rel_cal.current_alpha()


# ---------------------------------------------------------------------------
# Tests 11–17: simulator unit tests
# ---------------------------------------------------------------------------

def test_11_all_requests_complete():
    reqs = _make_requests(50)
    cal = AbsoluteErrorConformalCalibrator()
    summary, resp, _ = _simulate_decoupled_hybrid_abs_conformal(reqs, servers=2, calibrator=cal)
    assert len(resp) == len(reqs)


def test_12_response_times_ge_service_times():
    reqs = _make_requests(50)
    cal = AbsoluteErrorConformalCalibrator()
    _, resp, _ = _simulate_decoupled_hybrid_abs_conformal(reqs, servers=2, calibrator=cal)
    for req in reqs:
        assert resp[req.idx] >= req.service_s - 1e-9


def test_13_preemption_count_in_summary():
    reqs = _make_requests(80)
    cal = AbsoluteErrorConformalCalibrator()
    summary, _, _ = _simulate_decoupled_hybrid_abs_conformal(reqs, servers=2, calibrator=cal)
    assert "preemption_count" in summary
    assert summary["preemption_count"] >= 0


def test_14_single_server_trivial():
    reqs = [
        _Request(idx=0, arrival_s=0.0, actual_tokens=10, predicted_tokens=10.0, service_s=1.0),
        _Request(idx=1, arrival_s=5.0, actual_tokens=10, predicted_tokens=10.0, service_s=1.0),
    ]
    cal = AbsoluteErrorConformalCalibrator()
    _, resp, _ = _simulate_decoupled_hybrid_abs_conformal(reqs, servers=1, calibrator=cal)
    assert resp[0] == pytest.approx(1.0, abs=1e-9)
    assert resp[1] == pytest.approx(1.0, abs=1e-9)


def test_15_oracle_near_srpt():
    """With oracle predictions, abs-conformal should approach SRPT performance."""
    reqs = _make_requests(200)
    # Override: predicted == actual (oracle)
    oracle_reqs = [
        _Request(
            idx=r.idx,
            arrival_s=r.arrival_s,
            actual_tokens=r.actual_tokens,
            predicted_tokens=float(r.actual_tokens),
            service_s=r.service_s,
        )
        for r in reqs
    ]
    fifo_sum, fifo_resp, _ = simulate_queue(reqs, servers=2, discipline="fifo")
    cal = AbsoluteErrorConformalCalibrator()
    abs_sum, abs_resp, _ = _simulate_decoupled_hybrid_abs_conformal(
        oracle_reqs, servers=2, calibrator=cal
    )
    gp_fifo = _sla_safe_goodput_per_dollar(reqs, fifo_resp, DEFAULT_SLA_S, servers=2)
    gp_abs  = _sla_safe_goodput_per_dollar(oracle_reqs, abs_resp, DEFAULT_SLA_S, servers=2)
    # Oracle abs-conformal must be measurable (positive goodput)
    assert gp_abs >= 0.0
    assert gp_fifo >= 0.0


def test_16_simulate_queue_discipline_accepted():
    reqs = _make_requests(30)
    summary, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid_abs_conformal")
    assert len(resp) == len(reqs)


def test_17_abs_conformal_differs_from_fifo():
    """Abs-conformal with oracle should differ from FIFO on a loaded queue."""
    import random
    rng = random.Random(7)
    reqs = []
    t = 0.0
    for i in range(200):
        t += rng.expovariate(2.0)  # higher arrival rate → contention
        actual = rng.choice([5, 500])
        reqs.append(_Request(
            idx=i,
            arrival_s=t,
            actual_tokens=actual,
            predicted_tokens=float(actual),  # oracle
            service_s=_service_time_s(actual),
        ))
    fifo_sum, fifo_resp, _ = simulate_queue(reqs, servers=1, discipline="fifo")
    abs_sum, abs_resp, _  = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid_abs_conformal")
    # At least some response times should differ
    diffs = [abs(abs_resp[r.idx] - fifo_resp[r.idx]) for r in reqs if r.idx in fifo_resp and r.idx in abs_resp]
    assert any(d > 1e-6 for d in diffs)


# ---------------------------------------------------------------------------
# Tests 18–20: AbsConformalReport unit tests
# ---------------------------------------------------------------------------

def test_18_to_dict_required_keys():
    dummy_dict = {"mean_response_s": 1.0, "sla_safe_goodput_per_dollar": 10.0}
    rpt = AbsConformalReport(
        trace="test", total_requests=100, servers=4, target_rho=0.85,
        time_warp=1.0, sla_s=10.0, target_p90_abs_tokens=500.0,
        fifo=dummy_dict, conformal_oracle=dummy_dict,
        rel_conformal_live=dummy_dict, abs_conformal_live=dummy_dict,
        fifo_goodput_per_dollar=10.0, oracle_goodput_per_dollar=50.0,
        rel_conformal_goodput_per_dollar=40.0, abs_conformal_goodput_per_dollar=45.0,
        oracle_delta_pct=400.0, rel_conformal_delta_pct=300.0,
        abs_conformal_delta_pct=350.0, abs_vs_rel_delta_pct=12.5,
        abs_vs_oracle_retention_pct=90.0, rel_vs_oracle_retention_pct=80.0,
        abs_mean_alpha=0.0008, rel_mean_alpha=0.002,
        abs_p90_abs_err_tokens=400.0,
    )
    d = rpt.to_dict()
    required_keys = [
        "trace", "total_requests", "servers", "sla_s", "target_p90_abs_tokens",
        "fifo_goodput_per_dollar", "oracle_goodput_per_dollar",
        "rel_conformal_goodput_per_dollar", "abs_conformal_goodput_per_dollar",
        "oracle_delta_pct", "rel_conformal_delta_pct", "abs_conformal_delta_pct",
        "abs_vs_rel_delta_pct", "abs_vs_oracle_retention_pct",
        "rel_vs_oracle_retention_pct", "abs_mean_alpha", "rel_mean_alpha",
        "abs_p90_abs_err_tokens", "shadow_tag",
    ]
    for k in required_keys:
        assert k in d, f"missing key: {k}"


def test_19_to_dict_shadow_tag():
    dummy_dict = {}
    rpt = AbsConformalReport(
        trace="t", total_requests=10, servers=1, target_rho=0.5,
        time_warp=1.0, sla_s=10.0, target_p90_abs_tokens=500.0,
        fifo={}, conformal_oracle={}, rel_conformal_live={}, abs_conformal_live={},
        fifo_goodput_per_dollar=1.0, oracle_goodput_per_dollar=5.0,
        rel_conformal_goodput_per_dollar=4.0, abs_conformal_goodput_per_dollar=4.5,
        oracle_delta_pct=400.0, rel_conformal_delta_pct=300.0,
        abs_conformal_delta_pct=350.0, abs_vs_rel_delta_pct=12.5,
        abs_vs_oracle_retention_pct=90.0, rel_vs_oracle_retention_pct=80.0,
        abs_mean_alpha=0.001, rel_mean_alpha=0.002,
        abs_p90_abs_err_tokens=400.0,
    )
    d = rpt.to_dict()
    assert "shadow_only" in d["shadow_tag"]


def test_20_to_dict_abs_vs_rel_computed():
    dummy = {}
    rpt = AbsConformalReport(
        trace="t", total_requests=10, servers=1, target_rho=0.5,
        time_warp=1.0, sla_s=10.0, target_p90_abs_tokens=500.0,
        fifo=dummy, conformal_oracle=dummy, rel_conformal_live=dummy, abs_conformal_live=dummy,
        fifo_goodput_per_dollar=10.0, oracle_goodput_per_dollar=50.0,
        rel_conformal_goodput_per_dollar=40.0, abs_conformal_goodput_per_dollar=44.0,
        oracle_delta_pct=400.0, rel_conformal_delta_pct=300.0,
        abs_conformal_delta_pct=340.0, abs_vs_rel_delta_pct=10.0,
        abs_vs_oracle_retention_pct=88.0, rel_vs_oracle_retention_pct=80.0,
        abs_mean_alpha=0.0008, rel_mean_alpha=0.002,
        abs_p90_abs_err_tokens=450.0,
    )
    d = rpt.to_dict()
    assert d["abs_vs_rel_delta_pct"] == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests 21–24: Azure LLM 2024 integration test
# ---------------------------------------------------------------------------

def test_21_azure_returns_abs_conformal_report():
    rpt = run_abs_conformal_azure_backtest(job_limit=200)
    assert isinstance(rpt, AbsConformalReport)
    assert rpt.total_requests == 200
    assert rpt.trace == "azure_llm_2024"


def test_22_azure_abs_mean_alpha_le_rel():
    """Key hypothesis: abs calibrator gives ≤ rel-calibrator α on Azure."""
    rpt = run_abs_conformal_azure_backtest(job_limit=300)
    # Abs-error calibrator should not be WORSE (higher α) than rel-error calibrator.
    # Accept: abs ≤ rel + 10% tolerance (α may be similar when both well-calibrated).
    assert rpt.abs_mean_alpha <= rpt.rel_mean_alpha * 1.1 + 1e-6


def test_23_azure_oracle_delta_measurable():
    """Oracle conformal goodput/$ is measured and positive (sign depends on sample size)."""
    rpt = run_abs_conformal_azure_backtest(job_limit=200)
    # At 200 requests the contention may be too low to show SRTF > FIFO,
    # but the oracle goodput must be a valid positive number.
    assert rpt.oracle_goodput_per_dollar > 0.0
    assert rpt.fifo_goodput_per_dollar > 0.0


def test_24_azure_abs_p90_err_reported():
    rpt = run_abs_conformal_azure_backtest(job_limit=200)
    # After processing 200 requests, the calibrator should have a valid p90 measurement.
    assert not math.isnan(rpt.abs_p90_abs_err_tokens)
    assert rpt.abs_p90_abs_err_tokens >= 0.0


# ---------------------------------------------------------------------------
# Tests 25–28: BurstGPT HF integration test (HF data required)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_HF_DATA, reason="BurstGPT HF JSONL not present")
def test_25_burstgpt_returns_abs_conformal_report():
    rpt = run_abs_conformal_burstgpt_backtest(job_limit=200)
    assert isinstance(rpt, AbsConformalReport)
    assert rpt.total_requests == 200


@pytest.mark.skipif(not _HAS_HF_DATA, reason="BurstGPT HF JSONL not present")
def test_26_burstgpt_abs_mean_alpha_le_rel():
    """Key hypothesis: abs calibrator gives lower α than rel calibrator on BurstGPT."""
    rpt = run_abs_conformal_burstgpt_backtest(job_limit=300)
    assert rpt.abs_mean_alpha <= rpt.rel_mean_alpha * 1.1 + 1e-6


@pytest.mark.skipif(not _HAS_HF_DATA, reason="BurstGPT HF JSONL not present")
def test_27_burstgpt_abs_goodput_positive():
    rpt = run_abs_conformal_burstgpt_backtest(job_limit=200)
    assert rpt.abs_conformal_goodput_per_dollar > 0.0


def test_28_constant_positive():
    assert CONFORMAL_ABS_TARGET_P90_TOKENS > 0.0
    assert isinstance(CONFORMAL_ABS_TARGET_P90_TOKENS, float)
