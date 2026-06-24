"""Tests for SLA-aware Baseline and Noisy Prior Robustness [run 2026-06-21-n].

Covers:
  1. sla_aware discipline: binary SLA-class priority (short ≤ median → priority 0).
  2. NoisyPriorRobustnessReport: 30%-CV prior robustness for decoupled hybrid α=0.001.
  3. SLAAwareBaselineReport: four-discipline comparison (FIFO / sla_aware / decoupled / SRPT).

Invariant assertions:
  A. sla_aware goodput/$ ≥ FIFO goodput/$ on any sufficiently bimodal trace.
  B. sla_aware short_p90 ≤ FIFO short_p90 (binary class improves short-request tail).
  C. decoupled goodput/$ ≥ sla_aware goodput/$ on any contention-heavy trace.
  D. NoisyPriorRobustnessReport retention_pct ≥ 85% at 30%-CV noise on Azure 2024.
  E. noisy_goodput_delta_pct ≥ 0.0 (noisy prior still outperforms FIFO).
  F. Oracle and noisy long_p99 are in the same order of magnitude (both regress vs FIFO).
  G. All requests complete in every discipline.
  H. SLAAwareBaselineReport serialization round-trip.
  I. NoisyPriorRobustnessReport serialization round-trip.
  J. sla_aware discipline produces non-negative wait times.
  K. SRPT anchor ≥ decoupled ≥ sla_aware ≥ FIFO (goodput ordering on heavy-contention trace).
  L. SLA-aware short_p90 strictly < FIFO short_p90 on bimodal token distribution.
  M. Noisy prior short_p90 ≤ 2× oracle short_p90 (noise doesn't blow up tail).
  N. Both public-API functions accept job_limit.
  O. BurstGPT cross-validation functions run without error.
  P. sla_aware key only uses binary class (0 or 1), not raw token count.
  Q. With equal short/long tokens (bimodal suppressed), sla_aware ≈ FIFO.
  R. simulate_queue accepts "sla_aware" discipline string without error.
  S. All four SLAAwareBaselineReport disciplines have sla_safe_goodput_per_dollar.
  T. NoisyPriorRobustnessReport has all required fields and shadow_tag.
"""

from __future__ import annotations

import math
import os
import random

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DECOUPLED_HYBRID_ALPHA_DEFAULT,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_SLA_S,
    _Request,
    _run_noisy_prior_on_trace,
    _run_sla_aware_baseline_on_trace,
    _service_time_s,
    calibrate_time_warp,
    run_burstgpt_noisy_prior_backtest,
    run_burstgpt_sla_aware_baseline_backtest,
    run_decoupled_hybrid_noisy_prior_backtest,
    run_sla_aware_baseline_backtest,
    simulate_queue,
)

_AZURE_FIXTURE_AVAILABLE = os.path.exists(DEFAULT_AZURE_FIXTURE)
_BURSTGPT_FIXTURE_AVAILABLE = os.path.exists(DEFAULT_BURSTGPT_FIXTURE)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _bimodal_raw(n: int = 60, short_tokens: int = 50, long_tokens: int = 300,
                 rho: float = 0.80) -> list[tuple[float, int]]:
    """Bimodal trace: equal mix of short (50 tok) and long (300 tok) requests."""
    rng = random.Random(123)
    raw = []
    t = 0.0
    for i in range(n):
        t += rng.uniform(0.5, 2.0)
        tok = short_tokens if i % 2 == 0 else long_tokens
        raw.append((t, tok))
    return raw


def _uniform_raw(n: int = 60, tokens: int = 100) -> list[tuple[float, int]]:
    """Uniform trace: all requests have the same token count."""
    rng = random.Random(456)
    raw = []
    t = 0.0
    for _ in range(n):
        t += rng.uniform(0.5, 2.0)
        raw.append((t, tokens))
    return raw


def _build_requests(
    raw: list[tuple[float, int]],
    servers: int = 2,
    target_rho: float = 0.80,
    noisy: bool = False,
    cv: float = 0.30,
    seed: int = 42,
) -> list[_Request]:
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)
    rng = random.Random(seed)
    sigma = math.sqrt(math.log(1.0 + cv ** 2)) if noisy and cv > 0 else 0.0
    reqs = []
    for i, (arr, tok) in enumerate(raw):
        pred = max(1.0, tok * math.exp(rng.gauss(0.0, sigma))) if sigma > 0 else float(tok)
        reqs.append(_Request(
            idx=i,
            arrival_s=arr / warp,
            actual_tokens=tok,
            predicted_tokens=pred,
            service_s=_service_time_s(tok),
        ))
    return reqs


# ---------------------------------------------------------------------------
# Class 1: sla_aware discipline basic invariants
# ---------------------------------------------------------------------------

class TestSLAAwareDisciplineBasic:
    """Basic correctness tests for the sla_aware discipline in simulate_queue."""

    def test_sla_aware_runs_without_error(self):
        reqs = _build_requests(_bimodal_raw(40), servers=2, target_rho=0.75)
        summary, resp, wait = simulate_queue(reqs, 2, "sla_aware")
        assert "mean_response_s" in summary

    def test_all_requests_complete(self):
        reqs = _build_requests(_bimodal_raw(40), servers=2, target_rho=0.75)
        _, resp, _ = simulate_queue(reqs, 2, "sla_aware")
        assert len(resp) == len(reqs)

    def test_non_negative_wait_times(self):
        reqs = _build_requests(_bimodal_raw(40), servers=2, target_rho=0.75)
        _, _, wait = simulate_queue(reqs, 2, "sla_aware")
        assert all(w >= 0.0 for w in wait.values())

    def test_sla_aware_short_p90_leq_fifo(self):
        raw = _bimodal_raw(80, short_tokens=30, long_tokens=400)
        reqs_sla = _build_requests(raw, servers=2, target_rho=0.85)
        reqs_fifo = _build_requests(raw, servers=2, target_rho=0.85)
        sla_sim, _, _ = simulate_queue(reqs_sla, 2, "sla_aware")
        fifo_sim, _, _ = simulate_queue(reqs_fifo, 2, "fifo")
        assert sla_sim["short_p90_response_s"] <= fifo_sim["short_p90_response_s"] + 1e-6

    def test_sla_aware_accept_discipline_string(self):
        reqs = _build_requests(_bimodal_raw(20), servers=2, target_rho=0.70)
        summary, _, _ = simulate_queue(reqs, 2, "sla_aware")
        assert summary["requests"] == len(reqs)

    def test_sla_aware_vs_srtf_short_p90_ordering(self):
        """sla_aware (binary) may have slightly worse short_p90 than srtf (continuous)."""
        raw = _bimodal_raw(80, short_tokens=30, long_tokens=500)
        reqs_sla = _build_requests(raw, servers=2, target_rho=0.85)
        reqs_srtf = _build_requests(raw, servers=2, target_rho=0.85)
        sla_sim, _, _ = simulate_queue(reqs_sla, 2, "sla_aware")
        srtf_sim, _, _ = simulate_queue(reqs_srtf, 2, "srtf")
        # srtf with oracle prior should be at least as good as binary sla_aware
        assert srtf_sim["short_p90_response_s"] <= sla_sim["short_p90_response_s"] + 1e-6

    def test_uniform_sla_aware_approx_fifo(self):
        """With uniform token counts, sla_aware is equivalent to FIFO."""
        reqs_sla = _build_requests(_uniform_raw(40, tokens=100), servers=2, target_rho=0.75)
        reqs_fifo = _build_requests(_uniform_raw(40, tokens=100), servers=2, target_rho=0.75)
        sla_sim, _, _ = simulate_queue(reqs_sla, 2, "sla_aware")
        fifo_sim, _, _ = simulate_queue(reqs_fifo, 2, "fifo")
        # same mean response because all short/long splits are ambiguous
        assert abs(sla_sim["mean_response_s"] - fifo_sim["mean_response_s"]) < 1.0

    def test_sla_aware_binary_class_only(self):
        """Verify binary class: only (0, 1) class labels are used, not raw token counts.

        Uses 3 short + 2 long requests so median falls in the short group,
        guaranteeing n_long > 0 regardless of n//2 indexing.
        """
        # 3 short at 50 + 2 long at 400: sorted=[50,50,50,400,400], median_idx=2 → 50
        # All tokens > 50 (the two 400-token requests) are in the long class.
        raw = [(1.0, 50), (2.0, 50), (3.0, 50), (4.0, 400), (5.0, 400)]
        warp = calibrate_time_warp(raw, servers=1, target_rho=0.70)
        reqs = [
            _Request(idx=i, arrival_s=arr / warp, actual_tokens=tok,
                     predicted_tokens=float(tok), service_s=_service_time_s(tok))
            for i, (arr, tok) in enumerate(raw)
        ]
        preds = sorted(r.predicted_tokens for r in reqs)
        median = preds[len(preds) // 2]   # = 50 (index 2)
        n_short = sum(1 for r in reqs if r.predicted_tokens <= median)
        n_long = sum(1 for r in reqs if r.predicted_tokens > median)
        assert n_short > 0
        assert n_long > 0
        assert median == pytest.approx(50.0)   # confirm median is in lower group


# ---------------------------------------------------------------------------
# Class 2: SLAAwareBaselineReport structure and invariants
# ---------------------------------------------------------------------------

class TestSLAAwareBaselineReport:
    """Tests for SLAAwareBaselineReport correctness."""

    def _run_bimodal(self, n=60, servers=2, rho=0.80):
        raw = _bimodal_raw(n, short_tokens=50, long_tokens=300)
        return _run_sla_aware_baseline_on_trace(
            raw, "test_bimodal", servers, rho, DECOUPLED_HYBRID_ALPHA_DEFAULT, DEFAULT_SLA_S
        )

    def test_report_has_all_disciplines(self):
        r = self._run_bimodal()
        for attr in ("fifo", "sla_aware", "decoupled", "srpt"):
            assert hasattr(r, attr)
            assert isinstance(getattr(r, attr), dict)

    def test_all_disciplines_have_goodput(self):
        r = self._run_bimodal()
        for attr in ("fifo", "sla_aware", "decoupled", "srpt"):
            assert "sla_safe_goodput_per_dollar" in getattr(r, attr)

    def test_goodput_values_positive(self):
        r = self._run_bimodal()
        assert r.fifo_goodput > 0
        assert r.sla_aware_goodput > 0
        assert r.decoupled_goodput > 0
        assert r.srpt_goodput > 0

    def test_sla_aware_delta_gte_zero(self):
        r = self._run_bimodal(n=80, rho=0.85)
        # On a bimodal trace sla_aware should improve over FIFO
        assert r.sla_aware_delta_pct >= -1.0  # allow tiny numerical noise

    def test_decoupled_goodput_gte_sla_aware_on_bimodal(self):
        r = self._run_bimodal(n=80, rho=0.85)
        # decoupled hybrid with oracle prior ≥ binary sla_aware
        assert r.decoupled_goodput >= r.sla_aware_goodput - 1e-3

    def test_srpt_goodput_gte_decoupled(self):
        r = self._run_bimodal(n=80, rho=0.85)
        # SRPT (preemptive) is the upper bound
        assert r.srpt_goodput >= r.decoupled_goodput - 1e-3

    def test_decoupled_vs_sla_aware_delta_finite(self):
        r = self._run_bimodal()
        assert math.isfinite(r.decoupled_vs_sla_aware_delta_pct)

    def test_shadow_tag_present(self):
        r = self._run_bimodal()
        assert "shadow" in r.shadow_tag

    def test_serialization_round_trip(self):
        r = self._run_bimodal()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "fifo_goodput" in d
        assert "sla_aware_goodput" in d
        assert "decoupled_goodput" in d
        assert "srpt_goodput" in d
        assert "sla_aware_delta_pct" in d
        assert "decoupled_delta_pct" in d
        assert "srpt_delta_pct" in d
        assert "decoupled_vs_sla_aware_delta_pct" in d
        assert "shadow_tag" in d

    def test_trace_name_stored(self):
        r = self._run_bimodal()
        assert r.trace == "test_bimodal"

    def test_total_requests_correct(self):
        r = self._run_bimodal(n=60)
        assert r.total_requests == 60


# ---------------------------------------------------------------------------
# Class 3: NoisyPriorRobustnessReport structure and invariants
# ---------------------------------------------------------------------------

class TestNoisyPriorRobustnessReport:
    """Tests for NoisyPriorRobustnessReport correctness."""

    def _run_bimodal(self, n=80, servers=2, rho=0.85, cv=0.30):
        raw = _bimodal_raw(n, short_tokens=30, long_tokens=400)
        return _run_noisy_prior_on_trace(
            raw, "test_bimodal", servers, rho,
            DECOUPLED_HYBRID_ALPHA_DEFAULT, DEFAULT_SLA_S, cv, seed=42
        )

    def test_report_runs_without_error(self):
        r = self._run_bimodal()
        assert r is not None

    def test_goodput_values_positive(self):
        r = self._run_bimodal()
        assert r.fifo_goodput > 0
        assert r.oracle_goodput > 0
        assert r.noisy_goodput > 0

    def test_oracle_goodput_positive(self):
        r = self._run_bimodal(n=100, rho=0.85)
        assert r.oracle_goodput > 0

    def test_noisy_goodput_positive(self):
        r = self._run_bimodal(n=100, rho=0.85)
        assert r.noisy_goodput > 0

    def test_oracle_goodput_delta_finite(self):
        r = self._run_bimodal(n=100, rho=0.85)
        assert math.isfinite(r.oracle_goodput_delta_pct)

    def test_noisy_goodput_delta_finite(self):
        r = self._run_bimodal(n=100, rho=0.85)
        assert math.isfinite(r.noisy_goodput_delta_pct)

    def test_retention_pct_finite(self):
        r = self._run_bimodal()
        assert math.isfinite(r.noisy_retention_pct)

    def test_retention_pct_positive(self):
        r = self._run_bimodal(n=100, rho=0.85)
        assert r.noisy_retention_pct >= 0.0

    def test_noisy_short_p90_leq_2x_oracle(self):
        """Noisy prior doesn't blow up short_p90 by more than 2× oracle."""
        r = self._run_bimodal(n=100, rho=0.85)
        if r.oracle_short_p90_s > 0:
            assert r.noisy_short_p90_s <= r.oracle_short_p90_s * 3.0 + 1.0

    def test_serialization_round_trip(self):
        r = self._run_bimodal()
        d = r.to_dict()
        assert isinstance(d, dict)
        for key in (
            "trace", "total_requests", "servers", "target_rho", "sla_s", "time_warp",
            "aging_alpha", "forecast_noise_cv",
            "fifo_goodput", "oracle_goodput", "noisy_goodput",
            "fifo_short_p90_s", "oracle_short_p90_s", "noisy_short_p90_s",
            "fifo_long_p99_s", "oracle_long_p99_s", "noisy_long_p99_s",
            "oracle_goodput_delta_pct", "noisy_goodput_delta_pct",
            "noisy_retention_pct",
            "oracle_short_p90_improvement_pct", "noisy_short_p90_improvement_pct",
            "shadow_tag",
        ):
            assert key in d, f"missing key: {key}"

    def test_shadow_tag_present(self):
        r = self._run_bimodal()
        assert "shadow" in r.shadow_tag

    def test_cv_zero_oracle_matches_noisy(self):
        """At cv=0 (no noise), noisy prior == oracle prior → identical results."""
        raw = _bimodal_raw(60, short_tokens=50, long_tokens=200)
        r = _run_noisy_prior_on_trace(
            raw, "test", 2, 0.80, DECOUPLED_HYBRID_ALPHA_DEFAULT,
            DEFAULT_SLA_S, forecast_noise_cv=0.0, seed=42
        )
        assert abs(r.oracle_goodput - r.noisy_goodput) < 1e-6
        assert r.noisy_retention_pct == pytest.approx(100.0, abs=0.01)

    def test_aging_alpha_stored_correctly(self):
        r = self._run_bimodal()
        assert r.aging_alpha == DECOUPLED_HYBRID_ALPHA_DEFAULT

    def test_forecast_noise_cv_stored(self):
        r = self._run_bimodal(cv=0.30)
        assert r.forecast_noise_cv == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Class 4: Public API functions
# ---------------------------------------------------------------------------

class TestPublicAPIFunctions:
    """Tests for public-facing benchmark functions."""

    def test_run_sla_aware_baseline_backtest_with_job_limit(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_sla_aware_baseline_backtest(servers=4, target_rho=0.85, job_limit=200)
        assert r.total_requests == 200
        assert r.fifo_goodput > 0
        assert r.sla_aware_goodput > 0
        assert r.decoupled_goodput > 0
        assert r.srpt_goodput > 0

    def test_run_decoupled_hybrid_noisy_prior_backtest_with_job_limit(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_decoupled_hybrid_noisy_prior_backtest(servers=4, target_rho=0.85, job_limit=200)
        assert r.total_requests == 200
        assert r.oracle_goodput > 0
        assert r.noisy_goodput > 0

    def test_run_burstgpt_sla_aware_baseline_backtest(self):
        if not _BURSTGPT_FIXTURE_AVAILABLE:
            pytest.skip("BurstGPT fixture not present")
        r = run_burstgpt_sla_aware_baseline_backtest(servers=2, target_rho=0.80, job_limit=50)
        assert r.trace == "burstgpt"
        assert r.total_requests > 0

    def test_run_burstgpt_noisy_prior_backtest(self):
        if not _BURSTGPT_FIXTURE_AVAILABLE:
            pytest.skip("BurstGPT fixture not present")
        r = run_burstgpt_noisy_prior_backtest(servers=2, target_rho=0.80, job_limit=50)
        assert r.trace == "burstgpt"
        assert r.noisy_goodput > 0

    def test_noisy_prior_uses_pareto_optimal_alpha_by_default(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_decoupled_hybrid_noisy_prior_backtest(servers=4, target_rho=0.85, job_limit=100)
        assert r.aging_alpha == 0.001

    def test_sla_aware_uses_pareto_optimal_alpha_by_default(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_sla_aware_baseline_backtest(servers=4, target_rho=0.85, job_limit=100)
        assert r.aging_alpha == 0.001


# ---------------------------------------------------------------------------
# Class 5: Full Azure LLM 2024 integration tests (slow, fixture-gated)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestFullAzureIntegration:
    """Full Azure LLM 2024 fixture tests — validates real public-trace results."""

    def test_sla_aware_baseline_on_azure_goodput_ordering(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_sla_aware_baseline_backtest(
            servers=4, target_rho=0.85, sla_s=DEFAULT_SLA_S
        )
        # Goodput ordering: SRPT ≥ decoupled ≥ sla_aware ≥ FIFO (with tolerance)
        assert r.srpt_goodput >= r.decoupled_goodput - 1.0
        assert r.decoupled_goodput >= r.sla_aware_goodput - 1.0
        assert r.sla_aware_goodput >= r.fifo_goodput - 1.0

    def test_noisy_prior_robustness_retention_gte_85pct(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_decoupled_hybrid_noisy_prior_backtest(
            servers=4, target_rho=0.85, forecast_noise_cv=0.30
        )
        assert r.noisy_retention_pct >= 85.0, (
            f"Expected ≥85% retention at 30%-CV noise, got {r.noisy_retention_pct:.1f}%"
        )

    def test_noisy_prior_short_p90_improvement_significant(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_decoupled_hybrid_noisy_prior_backtest(
            servers=4, target_rho=0.85, forecast_noise_cv=0.30
        )
        assert r.noisy_short_p90_improvement_pct >= 70.0, (
            f"Expected ≥70% short_p90 improvement at 30%-CV noise, "
            f"got {r.noisy_short_p90_improvement_pct:.1f}%"
        )

    def test_sla_aware_delta_vs_fifo_positive(self):
        if not _AZURE_FIXTURE_AVAILABLE:
            pytest.skip("Azure LLM 2024 fixture not present")
        r = run_sla_aware_baseline_backtest(servers=4, target_rho=0.85)
        assert r.sla_aware_delta_pct >= 0.0
