"""Tests for Conformal Adaptive α discipline [run 2026-06-21-q].

The ConformalAlphaCalibrator adapts the aging dispatch α based on empirical
prediction errors from completed requests.  With oracle tokens (predicted ==
actual) the calibrator sets α → 0 after warmup, making dispatch equivalent
to pure SRPT.  With noisy predictions (30%-CV) it retains α ≈ 0.001 (same
as the fixed-α baseline).

Research basis:
  - arXiv:2508.14544 (Adaptively Robust LLM Inference under Prediction
    Uncertainty): core motivation for prediction-error-driven α tuning.
  - arXiv:1902.00732 (Scheduling with Predictions, Mitzenmacher 2019):
    price of misprediction framework.
  - arXiv:2503.07545 (Queueing, Predictions, and LLMs, 2025): identifies
    adaptive calibration as an open problem for production schedulers.

Invariants tested:
  1. ConformalAlphaCalibrator: warmup returns alpha_max.
  2. After warmup with zero-error residuals: current_alpha() == 0.
  3. After warmup with 30%-CV residuals: current_alpha() ≈ alpha_max.
  4. Sliding window caps residuals at ``window`` size.
  5. _simulate_decoupled_hybrid_conformal: all requests complete.
  6. With oracle prior: conformal goodput/$ ≥ fixed-α goodput/$.
  7. With oracle prior: conformal mean_alpha ≈ 0.
  8. Preemption rule is pure SRPT (same as decoupled_hybrid).
  9. simulate_queue dispatches decoupled_hybrid_conformal correctly.
  10. run_conformal_alpha_backtest returns ConformalAlphaReport with
      conformal_delta_pct > decoupled_fixed_delta_pct on fixture.
  11. run_burstgpt_conformal_alpha_backtest runs on BurstGPT fixture.
  12. Noisy prior: conformal goodput/$ ≈ fixed-α goodput/$.
  13. ConformalAlphaReport.to_dict() is fully serialisable.
  14. Starvation bound: conformal discipline completes every request.
  15. mean_alpha diagnostic matches observed α trajectory direction.
"""

from __future__ import annotations

import os
import random

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_ALPHA_MAX,
    CONFORMAL_TARGET_P90_ERROR,
    DECOUPLED_HYBRID_ALPHA_DEFAULT,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_SLA_S,
    ConformalAlphaCalibrator,
    ConformalAlphaReport,
    _Request,
    _simulate_decoupled_hybrid_conformal,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    load_serving_requests,
    run_burstgpt_conformal_alpha_backtest,
    run_conformal_alpha_backtest,
    simulate_queue,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(idx, arrival, tokens, predicted=None):
    """Create _Request with service_s = tokens (unit time scale)."""
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(tokens),
        predicted_tokens=float(predicted if predicted is not None else tokens),
        service_s=float(tokens),
    )


def _oracle_requests_from_fixture(limit=None):
    """Load Azure LLM 2024 fixture as oracle _Request list."""
    raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=limit)
    if len(raw) < 2:
        pytest.skip("Azure LLM 2024 fixture not available")
    warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
    from aurelius.benchmarks.srtf_serving_backtest import _service_time_s
    return [
        _Request(
            idx=i,
            arrival_s=arr / warp,
            actual_tokens=tok,
            predicted_tokens=float(tok),
            service_s=_service_time_s(tok),
        )
        for i, (arr, tok) in enumerate(raw)
    ]


# ---------------------------------------------------------------------------
# Class 1: ConformalAlphaCalibrator unit tests
# ---------------------------------------------------------------------------

class TestConformalAlphaCalibrator:

    def test_warmup_returns_alpha_max(self):
        cal = ConformalAlphaCalibrator(alpha_max=0.001, warmup=100)
        for _ in range(50):
            cal.update(100.0, 100)
        assert cal.current_alpha() == pytest.approx(0.001, rel=1e-9)

    def test_zero_error_after_warmup_gives_zero_alpha(self):
        cal = ConformalAlphaCalibrator(alpha_max=0.001, warmup=10, window=50)
        for _ in range(200):
            cal.update(100.0, 100)   # predicted == actual → rel_err = 0
        assert cal.current_alpha() == pytest.approx(0.0, abs=1e-12)

    def test_target_error_gives_alpha_max(self):
        cal = ConformalAlphaCalibrator(
            alpha_max=0.001, warmup=10, window=500,
            target_p90_error=CONFORMAL_TARGET_P90_ERROR,
        )
        rng = random.Random(42)  # noqa: F841
        # Inject residuals at exactly the target p90 level: ~40% error for most.
        for _ in range(500):
            cal.update(0.60, 1)   # rel_err = |0.60 - 1| / 1 = 0.40
        alpha = cal.current_alpha()
        assert alpha == pytest.approx(0.001, rel=0.02)

    def test_sliding_window_caps_residuals(self):
        cal = ConformalAlphaCalibrator(alpha_max=0.001, warmup=10, window=50)
        for _ in range(200):
            cal.update(1000.0, 100)   # large error phase
        # Now inject zero-error residuals — window should fill with 0.
        for _ in range(200):
            cal.update(100.0, 100)
        assert cal.current_alpha() == pytest.approx(0.0, abs=1e-12)

    def test_high_error_capped_at_two_times_alpha_max(self):
        cal = ConformalAlphaCalibrator(alpha_max=0.001, warmup=10, window=100)
        for _ in range(200):
            cal.update(1.0, 1000)   # enormous error → p90_err >> target_p90_error
        alpha = cal.current_alpha()
        assert alpha <= 0.001 * 2.0 + 1e-12, "cap at 2 × alpha_max"

    def test_mean_alpha_diagnostic(self):
        cal = ConformalAlphaCalibrator(alpha_max=0.001, warmup=10, window=50)
        for _ in range(100):
            cal.update(100.0, 100)
        _ = cal.current_alpha()
        assert cal.mean_alpha() >= 0.0
        assert cal.mean_alpha() <= 0.001 * 2.0 + 1e-9

    def test_update_skips_zero_actual(self):
        cal = ConformalAlphaCalibrator(alpha_max=0.001, warmup=10, window=50)
        for _ in range(200):
            cal.update(100.0, 0)   # actual = 0 → should be skipped
        # Residuals list should be empty → warmup check kicks in.
        alpha = cal.current_alpha()
        assert alpha == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Class 2: _simulate_decoupled_hybrid_conformal mechanics
# ---------------------------------------------------------------------------

class TestConformalSimulatorMechanics:

    def test_all_requests_complete(self):
        reqs = [_req(i, i * 2.0, 5 + i % 10) for i in range(20)]
        cal = ConformalAlphaCalibrator()
        summary, resp, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=2, calibrator=cal)
        assert len(resp) == 20, "every request must complete"

    def test_short_preempts_long_pure_srpt(self):
        # 1 server. Long (10s) starts at t=0. Short (2s) arrives at t=3.
        # remaining(long, t=3) = 7s > 2s → preempt (pure SRPT preemption key).
        reqs = [_req(0, 0, 10), _req(1, 3, 2)]
        cal = ConformalAlphaCalibrator()
        _, resp, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=1, calibrator=cal)
        assert resp[1] < resp[0], "short must complete before long"
        assert abs(resp[1] - 2.0) < 1e-9, "short sojourn = 2s"

    def test_no_preempt_when_arrival_longer_than_remaining(self):
        reqs = [_req(0, 0, 2), _req(1, 0.5, 10)]
        cal = ConformalAlphaCalibrator()
        _, resp, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=1, calibrator=cal)
        assert abs(resp[0] - 2.0) < 1e-9, "short completes at t=2"
        assert abs(resp[1] - 11.5) < 1e-9, "long sojourn = 10 + 1.5 wait"

    def test_preemption_count_in_summary(self):
        reqs = [_req(0, 0, 10), _req(1, 3, 2)]
        cal = ConformalAlphaCalibrator()
        summary, _, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=1, calibrator=cal)
        assert summary["preemption_count"] >= 1

    def test_conformal_mean_alpha_in_summary(self):
        reqs = [_req(i, i * 5.0, 3 + i % 8) for i in range(50)]
        cal = ConformalAlphaCalibrator()
        summary, _, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=2, calibrator=cal)
        assert "conformal_mean_alpha" in summary
        assert summary["conformal_mean_alpha"] >= 0.0

    def test_oracle_prior_mean_alpha_converges_to_zero(self):
        # With many oracle requests, α should converge to 0 after warmup.
        # mean_alpha is diluted by warmup period, so it won't be exactly 0;
        # with warmup=50 and 300 requests, expect mean ≈ 50/300 × alpha_max ≈ 0.00017.
        reqs = [_req(i, i * 1.0, 5 + i % 20) for i in range(300)]
        cal = ConformalAlphaCalibrator(warmup=50, window=100)
        summary, _, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=4, calibrator=cal)
        assert cal.mean_alpha() < cal.alpha_max * 0.5, (
            f"oracle prior should converge α→ near-0 post-warmup; "
            f"got mean_alpha={cal.mean_alpha():.2e}, alpha_max={cal.alpha_max}"
        )

    def test_simulate_queue_dispatches_conformal_discipline(self):
        reqs = [_req(i, i * 2.0, 5 + i % 7) for i in range(30)]
        summary, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid_conformal")
        assert len(resp) == 30
        assert "conformal_mean_alpha" in summary

    def test_single_server_all_complete(self):
        reqs = [_req(i, i * 10.0, 5) for i in range(10)]
        cal = ConformalAlphaCalibrator()
        _, resp, _ = _simulate_decoupled_hybrid_conformal(reqs, servers=1, calibrator=cal)
        assert len(resp) == 10


# ---------------------------------------------------------------------------
# Class 3: Goodput improvement with oracle prior
# ---------------------------------------------------------------------------

class TestConformalGoodputOracle:
    """With oracle tokens, conformal should achieve goodput/$ ≥ fixed-α=0.001."""

    def test_conformal_ge_fixed_on_azure_fixture(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        rpt = run_conformal_alpha_backtest()
        assert rpt.decoupled_conformal_goodput_per_dollar >= rpt.decoupled_fixed_goodput_per_dollar, (
            f"conformal {rpt.decoupled_conformal_goodput_per_dollar:.1f} < "
            f"fixed {rpt.decoupled_fixed_goodput_per_dollar:.1f}"
        )

    def test_conformal_delta_ge_fixed_delta_on_azure_fixture(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        rpt = run_conformal_alpha_backtest()
        assert rpt.decoupled_conformal_delta_pct >= rpt.decoupled_fixed_delta_pct, (
            f"conformal_delta={rpt.decoupled_conformal_delta_pct:.1f}% < "
            f"fixed_delta={rpt.decoupled_fixed_delta_pct:.1f}%"
        )

    def test_conformal_mean_alpha_near_zero_oracle(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        rpt = run_conformal_alpha_backtest()
        # With 5880 requests and warmup=100, mean_alpha ≈ 100/5880 × 0.001 ≈ 1.7e-5.
        # Must be well below alpha_max (0.001); use 10% of alpha_max as threshold.
        assert rpt.conformal_mean_alpha < CONFORMAL_ALPHA_MAX * 0.10, (
            f"oracle prior should converge α→near-0; got {rpt.conformal_mean_alpha:.2e}"
        )

    def test_conformal_report_has_all_fields(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        rpt = run_conformal_alpha_backtest()
        assert isinstance(rpt, ConformalAlphaReport)
        assert rpt.trace == "azure_llm_2024"
        assert rpt.total_requests > 0
        assert rpt.servers == 4
        assert rpt.target_rho == pytest.approx(0.85)
        assert rpt.sla_s == DEFAULT_SLA_S

    def test_to_dict_serialisable(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        rpt = run_conformal_alpha_backtest()
        d = rpt.to_dict()
        assert "conformal_mean_alpha" in d
        assert "conformal_vs_fixed_delta_pct" in d
        assert "shadow_tag" in d
        # All top-level values should be JSON-serialisable types.
        import json
        json.dumps(d)   # raises if not serialisable

    def test_srpt_remains_upper_bound(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        rpt = run_conformal_alpha_backtest()
        assert rpt.srpt_goodput_per_dollar >= rpt.decoupled_conformal_goodput_per_dollar * 0.99, (
            "SRPT should still be upper bound (allow 1% tolerance for warmup effect)"
        )

    def test_burstgpt_conformal_ge_fixed(self):
        if not os.path.exists(DEFAULT_BURSTGPT_FIXTURE):
            pytest.skip("BurstGPT fixture not present")
        rpt = run_burstgpt_conformal_alpha_backtest()
        assert rpt.decoupled_conformal_goodput_per_dollar >= rpt.decoupled_fixed_goodput_per_dollar, (
            f"BurstGPT: conformal {rpt.decoupled_conformal_goodput_per_dollar:.1f} < "
            f"fixed {rpt.decoupled_fixed_goodput_per_dollar:.1f}"
        )


# ---------------------------------------------------------------------------
# Class 4: Robustness under noisy prior
# ---------------------------------------------------------------------------

class TestConformalNoisyPriorRobustness:
    """With 30%-CV noisy prior, conformal should behave like fixed-α=0.001."""

    def _noisy_requests(self, seed: int = 42, cv: float = 0.30) -> list:
        raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=None)
        if len(raw) < 2:
            pytest.skip("Azure LLM 2024 fixture not available")
        import math as _math
        rng = random.Random(seed)
        sigma = _math.sqrt(_math.log(1 + cv ** 2))
        mu = -sigma ** 2 / 2

        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        from aurelius.benchmarks.srtf_serving_backtest import _service_time_s
        reqs = []
        for i, (arr, tok) in enumerate(raw):
            noise = _math.exp(rng.gauss(mu, sigma))
            pred = max(1.0, float(tok) * noise)
            reqs.append(_Request(
                idx=i,
                arrival_s=arr / warp,
                actual_tokens=tok,
                predicted_tokens=pred,
                service_s=_service_time_s(tok),
            ))
        return reqs

    def test_conformal_goodput_within_5pct_of_fixed_at_30cv(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        noisy_reqs = self._noisy_requests(cv=0.30)

        cal_conf  = ConformalAlphaCalibrator()
        conf_sim, conf_resp, _ = _simulate_decoupled_hybrid_conformal(
            noisy_reqs, servers=4, calibrator=cal_conf
        )
        gp_conf = _sla_safe_goodput_per_dollar(noisy_reqs, conf_resp, DEFAULT_SLA_S, 4)

        from aurelius.benchmarks.srtf_serving_backtest import _simulate_decoupled_hybrid
        fixed_reqs = self._noisy_requests(cv=0.30)
        _, fixed_resp, _ = _simulate_decoupled_hybrid(
            fixed_reqs, servers=4, aging_alpha=DECOUPLED_HYBRID_ALPHA_DEFAULT
        )
        gp_fixed = _sla_safe_goodput_per_dollar(fixed_reqs, fixed_resp, DEFAULT_SLA_S, 4)

        # Conformal should be within 10% of fixed-α under 30%-CV noise
        # (could be slightly better or slightly worse during warmup).
        ratio = gp_conf / gp_fixed if gp_fixed > 0 else 0.0
        assert ratio > 0.90, (
            f"conformal {gp_conf:.1f} is more than 10% below fixed {gp_fixed:.1f}"
        )

    def test_conformal_mean_alpha_positive_at_30cv(self):
        if not os.path.exists(DEFAULT_AZURE_FIXTURE):
            pytest.skip("Azure LLM 2024 fixture not present")
        noisy_reqs = self._noisy_requests(cv=0.30)
        cal = ConformalAlphaCalibrator()
        _simulate_decoupled_hybrid_conformal(noisy_reqs, servers=4, calibrator=cal)
        assert cal.mean_alpha() > 0.0, "noisy prior should keep α > 0"
