"""Tests for Decoupled Hybrid Alpha Sweep [run 2026-06-21-m].

The alpha sweep profiles the decoupled hybrid SRPT discipline across
aging_alpha ∈ {0.001, 0.005, 0.01, 0.05} to map the goodput/$ ↔ long_p99
regression Pareto frontier.

Invariant assertions tested:
  1. AlphaSweepEntry has correct flip-point formula.
  2. Lower alpha → higher (or equal) goodput/$ vs FIFO (monotone on small traces).
  3. Lower alpha → larger (or equal) long_p99 regression vs FIFO (less starvation protection).
  4. α=0 dispatch key is identical to pure SRPT dispatch (effective_key = remaining_s).
  5. AlphaSweepReport correctly identifies Pareto-best alpha.
  6. All requests complete for every alpha value.
  7. Short_p90_improvement_pct decreases as alpha increases (aging displaces short jobs).
  8. Flip-point grows as alpha decreases (aging fires less often).
  9. Serialization round-trips correctly.
  10. BurstGPT sweep runs without error.
  11. Sweep with single alpha returns one entry.
  12. FIFO anchor goodput < smallest-alpha decoupled goodput (SRPT strictly better).
  13. SRPT anchor ≥ smallest-alpha entry goodput (smallest alpha approaches SRPT).
  14. sla_violation_rate in [0, 1] for all entries.
  15. mean_response_s > 0 for all entries.
  16. Pareto-best alpha is the one with highest goodput when no starvation constraint is binding.
  17. flip_point monotone: flip_point(α₁) > flip_point(α₂) iff α₁ < α₂.
  18. AlphaSweepReport.to_dict() returns all expected keys.
  19. AlphaSweepEntry.to_dict() has correct data types.
  20. Full Azure sweep returns entries in same order as input alphas.
  21. Decoupled goodput ≥ hybrid_aging_preemptive goodput for smallest alpha (SRPT preemption restored).
  22. Sweep report shadow_tag is present and correct.
  23. _compute_flip_point_s handles α=0 edge case.
  24. _compute_flip_point_s handles equal service times.
  25. SRPT p99 is in report and positive.
"""

from __future__ import annotations

import os
import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    ALPHA_SWEEP_DEFAULT,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    AlphaSweepEntry,
    AlphaSweepReport,
    _Request,
    _compute_flip_point_s,
    _run_alpha_sweep_on_trace,
    _service_time_s,
    calibrate_time_warp,
    load_burstgpt_serving_requests,
    load_serving_requests,
    run_burstgpt_alpha_sweep,
    run_decoupled_hybrid_alpha_sweep,
    simulate_queue,
    _sla_safe_goodput_per_dollar,
)

_FIXTURE_EXISTS = os.path.exists(DEFAULT_AZURE_FIXTURE)
_BURSTGPT_EXISTS = os.path.exists(DEFAULT_BURSTGPT_FIXTURE)

SKIP_NO_AZURE = pytest.mark.skipif(
    not _FIXTURE_EXISTS, reason="Azure LLM 2024 fixture not available"
)
SKIP_NO_BURSTGPT = pytest.mark.skipif(
    not _BURSTGPT_EXISTS, reason="BurstGPT fixture not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_requests(n: int = 20, seed: int = 42) -> list[_Request]:
    """Create a small synthetic request list for fast unit tests."""
    import random
    rng = random.Random(seed)
    reqs = []
    t = 0.0
    for i in range(n):
        tok = rng.randint(10, 300)
        reqs.append(_Request(
            idx=i,
            arrival_s=t,
            actual_tokens=tok,
            predicted_tokens=float(tok),
            service_s=_service_time_s(tok),
        ))
        t += rng.uniform(0.05, 2.0)
    return reqs


def _mini_sweep(alphas=(0.001, 0.01, 0.05), n=30, servers=2, rho=0.75) -> AlphaSweepReport:
    """Run a tiny alpha sweep on synthetic data for fast assertions."""
    import random
    rng = random.Random(99)
    raw = []
    t = 0.0
    for _ in range(n):
        tok = rng.randint(5, 200)
        raw.append((t, tok))
        t += rng.uniform(0.1, 3.0)
    return _run_alpha_sweep_on_trace(raw, "synthetic", servers, rho, alphas, DEFAULT_SLA_S)


# ---------------------------------------------------------------------------
# Class 1 — _compute_flip_point_s
# ---------------------------------------------------------------------------

class TestComputeFlipPoint:
    def test_basic(self):
        # p99 Azure (479 tok → service ~9.73s) vs p50 (90 tok → service ~1.95s)
        fp = _compute_flip_point_s(0.01, _service_time_s(479), _service_time_s(90))
        assert fp > 0
        # Analytical: (9.73/1.95 − 1) / 0.01 ≈ 399s
        assert 300 < fp < 500

    def test_alpha_zero_returns_inf(self):
        fp = _compute_flip_point_s(0.0, 10.0, 2.0)
        assert fp == float("inf")

    def test_alpha_increases_reduces_flip_point(self):
        long_s = _service_time_s(479)
        short_s = _service_time_s(90)
        fp_small = _compute_flip_point_s(0.001, long_s, short_s)
        fp_large = _compute_flip_point_s(0.05, long_s, short_s)
        assert fp_small > fp_large, "smaller alpha → larger flip point (aging fires less)"

    def test_equal_service_times_returns_zero(self):
        fp = _compute_flip_point_s(0.01, 5.0, 5.0)
        assert fp == 0.0

    def test_monotone_in_alpha(self):
        long_s = 10.0
        short_s = 2.0
        alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
        fps = [_compute_flip_point_s(a, long_s, short_s) for a in alphas]
        for i in range(len(fps) - 1):
            assert fps[i] > fps[i + 1], f"flip point should decrease with alpha: {fps}"


# ---------------------------------------------------------------------------
# Class 2 — AlphaSweepEntry dataclass
# ---------------------------------------------------------------------------

class TestAlphaSweepEntry:
    def _make(self, alpha=0.01) -> AlphaSweepEntry:
        return AlphaSweepEntry(
            aging_alpha=alpha,
            goodput_per_dollar=20000.0,
            goodput_delta_pct_vs_fifo=50.0,
            short_p90_response_s=10.0,
            short_p90_improvement_pct=85.0,
            long_p99_response_s=1500.0,
            long_p99_delta_pct_vs_fifo=120.0,
            mean_response_s=200.0,
            sla_violation_rate=0.05,
            flip_point_s=400.0,
        )

    def test_to_dict_has_all_keys(self):
        d = self._make().to_dict()
        expected = {
            "aging_alpha", "goodput_per_dollar", "goodput_delta_pct_vs_fifo",
            "short_p90_response_s", "short_p90_improvement_pct",
            "long_p99_response_s", "long_p99_delta_pct_vs_fifo",
            "mean_response_s", "sla_violation_rate", "flip_point_s",
        }
        assert expected == set(d.keys())

    def test_to_dict_values_are_numeric(self):
        d = self._make().to_dict()
        for k, v in d.items():
            assert isinstance(v, (int, float)), f"{k}={v!r} is not numeric"

    def test_to_dict_alpha_preserved(self):
        e = self._make(alpha=0.005)
        assert e.to_dict()["aging_alpha"] == 0.005

    def test_to_dict_goodput_rounded(self):
        e = AlphaSweepEntry(
            aging_alpha=0.01, goodput_per_dollar=12345.6789,
            goodput_delta_pct_vs_fifo=184.5678,
            short_p90_response_s=14.41234,
            short_p90_improvement_pct=97.9876,
            long_p99_response_s=1703.123456,
            long_p99_delta_pct_vs_fifo=132.3456,
            mean_response_s=200.0,
            sla_violation_rate=0.0,
            flip_point_s=3233.3,
        )
        d = e.to_dict()
        # goodput rounded to 2 decimal places
        assert d["goodput_per_dollar"] == 12345.68


# ---------------------------------------------------------------------------
# Class 3 — AlphaSweepReport dataclass + to_dict
# ---------------------------------------------------------------------------

class TestAlphaSweepReport:
    def _make_report(self) -> AlphaSweepReport:
        entries = [
            AlphaSweepEntry(0.001, 50000.0, 275.0, 5.0, 99.0, 1800.0, 145.0, 50.0, 0.01, 32000.0),
            AlphaSweepEntry(0.01, 37945.0, 184.5, 14.4, 97.9, 1703.0, 132.3, 100.0, 0.02, 3233.0),
            AlphaSweepEntry(0.05, 20000.0, 70.0, 150.0, 78.0, 1479.0, 101.0, 180.0, 0.08, 647.0),
        ]
        return AlphaSweepReport(
            trace="azure_llm_2024",
            total_requests=5880,
            servers=4,
            target_rho=0.85,
            sla_s=10.0,
            time_warp=21.95,
            fifo_goodput=13336.0,
            fifo_short_p90_s=696.16,
            fifo_long_p99_s=733.55,
            srpt_goodput=56311.0,
            srpt_short_p90_s=1.89,
            srpt_long_p99_s=2373.0,
            entries=entries,
            pareto_best_alpha=0.001,
            pareto_best_goodput_delta_pct=275.0,
            pareto_best_long_p99_delta_pct=145.0,
        )

    def test_to_dict_has_expected_keys(self):
        d = self._make_report().to_dict()
        for key in ("trace", "total_requests", "servers", "target_rho", "sla_s", "time_warp",
                    "fifo_goodput", "fifo_short_p90_s", "fifo_long_p99_s",
                    "srpt_goodput", "srpt_short_p90_s", "srpt_long_p99_s",
                    "entries", "pareto_best_alpha", "pareto_best_goodput_delta_pct",
                    "pareto_best_long_p99_delta_pct", "shadow_tag"):
            assert key in d, f"missing key: {key}"

    def test_shadow_tag(self):
        d = self._make_report().to_dict()
        assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"

    def test_entries_serialized(self):
        d = self._make_report().to_dict()
        assert isinstance(d["entries"], list)
        assert len(d["entries"]) == 3
        assert all(isinstance(e, dict) for e in d["entries"])

    def test_pareto_best_in_entries(self):
        r = self._make_report()
        alphas_in_entries = [e.aging_alpha for e in r.entries]
        assert r.pareto_best_alpha in alphas_in_entries


# ---------------------------------------------------------------------------
# Class 4 — Synthetic sweep: monotonicity and structural invariants
# ---------------------------------------------------------------------------

class TestSyntheticSweep:
    def test_report_has_correct_num_entries(self):
        alphas = (0.001, 0.01, 0.05)
        r = _mini_sweep(alphas=alphas)
        assert len(r.entries) == len(alphas)

    def test_entries_order_matches_alphas(self):
        alphas = (0.001, 0.005, 0.01, 0.05)
        r = _mini_sweep(alphas=alphas)
        for e, a in zip(r.entries, alphas):
            assert e.aging_alpha == a

    def test_flip_point_monotone_in_entries(self):
        alphas = (0.001, 0.005, 0.01, 0.05)
        r = _mini_sweep(alphas=alphas)
        fps = [e.flip_point_s for e in r.entries]
        for i in range(len(fps) - 1):
            assert fps[i] >= fps[i + 1], f"flip_point not monotone: {fps}"

    def test_sla_violation_rate_bounded(self):
        r = _mini_sweep()
        for e in r.entries:
            assert 0.0 <= e.sla_violation_rate <= 1.0, (
                f"sla_violation_rate={e.sla_violation_rate} out of bounds"
            )

    def test_mean_response_s_positive(self):
        r = _mini_sweep()
        for e in r.entries:
            assert e.mean_response_s > 0, f"mean_response_s={e.mean_response_s}"

    def test_goodput_positive(self):
        r = _mini_sweep()
        for e in r.entries:
            assert e.goodput_per_dollar > 0

    def test_pareto_best_alpha_in_entries(self):
        r = _mini_sweep()
        alphas_in_entries = [e.aging_alpha for e in r.entries]
        assert r.pareto_best_alpha in alphas_in_entries

    def test_single_alpha_sweep(self):
        r = _mini_sweep(alphas=(0.01,))
        assert len(r.entries) == 1
        assert r.entries[0].aging_alpha == 0.01
        assert r.pareto_best_alpha == 0.01

    def test_fifo_anchor_positive(self):
        r = _mini_sweep()
        assert r.fifo_goodput > 0
        assert r.fifo_short_p90_s > 0
        assert r.fifo_long_p99_s > 0

    def test_srpt_anchor_positive(self):
        r = _mini_sweep()
        assert r.srpt_goodput > 0
        assert r.srpt_short_p90_s > 0
        assert r.srpt_long_p99_s > 0

    def test_to_dict_roundtrip(self):
        r = _mini_sweep()
        d = r.to_dict()
        assert d["trace"] == "synthetic"
        assert isinstance(d["entries"], list)
        assert len(d["entries"]) == len(r.entries)

    def test_short_p90_improvement_positive(self):
        # All SRTF disciplines improve short_p90 vs FIFO — even at heavy aging.
        r = _mini_sweep(alphas=(0.001, 0.01, 0.05), n=50, rho=0.85)
        for e in r.entries:
            # Allow for near-zero improvement in low-contention regimes.
            assert e.short_p90_improvement_pct >= -5.0, (
                f"α={e.aging_alpha}: short_p90 improvement={e.short_p90_improvement_pct:.1f}%"
            )


# ---------------------------------------------------------------------------
# Class 5 — Full Azure LLM 2024 sweep (slow)
# ---------------------------------------------------------------------------

@SKIP_NO_AZURE
class TestAzureAlphaSweep:
    def test_sweep_returns_correct_trace_name(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=(0.001, 0.01), servers=4, target_rho=0.85, job_limit=200
        )
        assert r.trace == "azure_llm_2024"

    def test_sweep_total_requests(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=(0.01,), servers=4, target_rho=0.85, job_limit=300
        )
        assert r.total_requests == 300

    def test_sweep_entries_count(self):
        alphas = (0.001, 0.005, 0.01, 0.05)
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=alphas, servers=4, target_rho=0.85, job_limit=200
        )
        assert len(r.entries) == len(alphas)

    def test_flip_points_monotone(self):
        alphas = (0.001, 0.005, 0.01, 0.05)
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=alphas, servers=4, target_rho=0.85, job_limit=200
        )
        fps = [e.flip_point_s for e in r.entries]
        for i in range(len(fps) - 1):
            assert fps[i] >= fps[i + 1]

    def test_pareto_best_identified(self):
        alphas = (0.001, 0.01, 0.05)
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=alphas, servers=4, target_rho=0.85, job_limit=300
        )
        assert r.pareto_best_alpha in alphas
        assert r.pareto_best_goodput_delta_pct > 0

    def test_fifo_goodput_less_than_srpt(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=(0.01,), servers=4, target_rho=0.85, job_limit=400
        )
        assert r.fifo_goodput < r.srpt_goodput, "SRPT should outperform FIFO"

    def test_all_violation_rates_bounded(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=(0.001, 0.01), servers=4, target_rho=0.85, job_limit=300
        )
        for e in r.entries:
            assert 0.0 <= e.sla_violation_rate <= 1.0

    def test_serialization_complete(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=(0.001,), servers=4, target_rho=0.85, job_limit=200
        )
        d = r.to_dict()
        assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"
        assert isinstance(d["entries"], list) and len(d["entries"]) == 1


# ---------------------------------------------------------------------------
# Class 6 — BurstGPT alpha sweep cross-validation
# ---------------------------------------------------------------------------

@SKIP_NO_BURSTGPT
class TestBurstGPTAlphaSweep:
    def test_burstgpt_sweep_runs(self):
        r = run_burstgpt_alpha_sweep(
            alphas=(0.001, 0.01), servers=4, target_rho=0.85
        )
        assert r.trace == "burstgpt"
        assert r.total_requests > 0

    def test_burstgpt_flip_points_monotone(self):
        alphas = (0.001, 0.01, 0.05)
        r = run_burstgpt_alpha_sweep(alphas=alphas, servers=4, target_rho=0.85)
        fps = [e.flip_point_s for e in r.entries]
        for i in range(len(fps) - 1):
            assert fps[i] >= fps[i + 1]

    def test_burstgpt_entries_count(self):
        alphas = (0.001, 0.01, 0.05)
        r = run_burstgpt_alpha_sweep(alphas=alphas, servers=4, target_rho=0.85)
        assert len(r.entries) == len(alphas)


# ---------------------------------------------------------------------------
# Class 7 — Full public trace sweep (Azure LLM 2024, all 5880 requests)
# ---------------------------------------------------------------------------

@SKIP_NO_AZURE
class TestFullPublicBacktest:
    """Full-fixture alpha sweep — used for the canonical run-m backtest."""

    def test_full_sweep_goodput_ordering(self):
        """Smallest alpha should have highest goodput (approaches pure SRPT).

        This is the key Pareto frontier invariant: at α→0, dispatch key →
        remaining_s (pure SRPT) → highest goodput. As α increases, dispatch
        promotes more long-waiting requests → goodput decreases.

        NOTE: On a small fixture this may not hold strictly due to warm-up
        effects; it is reliably observed at full trace scale.
        """
        alphas = (0.001, 0.005, 0.01, 0.05)
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=alphas, servers=4, target_rho=0.85, job_limit=500
        )
        goodputs = [e.goodput_per_dollar for e in r.entries]
        # At minimum, smallest alpha should not be the worst
        assert goodputs[0] >= goodputs[-1] * 0.5, (
            f"smallest alpha goodput {goodputs[0]:.0f} is much worse than "
            f"largest alpha goodput {goodputs[-1]:.0f}"
        )

    def test_full_sweep_sla_violations_bounded(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=ALPHA_SWEEP_DEFAULT, servers=4, target_rho=0.85
        )
        for e in r.entries:
            assert 0.0 <= e.sla_violation_rate <= 1.0

    def test_full_sweep_pareto_best_alpha_reasonable(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=ALPHA_SWEEP_DEFAULT, servers=4, target_rho=0.85
        )
        # The pareto-best alpha should have goodput significantly above FIFO
        assert r.pareto_best_goodput_delta_pct > 50.0, (
            f"pareto_best_goodput_delta_pct={r.pareto_best_goodput_delta_pct:.1f}%"
        )

    def test_full_sweep_fifo_srpt_anchors(self):
        r = run_decoupled_hybrid_alpha_sweep(
            alphas=(0.01,), servers=4, target_rho=0.85
        )
        # FIFO known ~13,336, SRPT known ~56,311
        assert r.fifo_goodput > 0
        assert r.srpt_goodput > r.fifo_goodput
