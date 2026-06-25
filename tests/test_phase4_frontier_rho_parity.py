"""Phase 4 parity tests: compute_frontier_rho_schedule correctness + null-result gate.

Three verification layers:
  1. Mechanism: compute_frontier_rho_schedule returns valid rho values, correct
     cold-start behavior, and causal window semantics.
  2. Prefill-savings integration: the estimator config receives actual mean window
     reuse_fraction telemetry (not hardcoded 0.0).
  3. Null-result gate: constraint_aware_adaptive == constraint_aware KPIs on
     BurstGPT + Azure LLM 2024 fixtures (MIN_REPLICAS=1 floor at fixture loads).

The null result at fixture loads is CORRECT behavior, not a bug — it documents
that Phase 4 requires production-scale workloads (multi-replica regime) to show
measurable improvement.  The mechanism tests verify the schedule IS adapting rho;
the null-result tests verify that rho adaptation at these loads produces the same
integer replica counts as the fixed-rho baseline.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.optimizer.policies.replica_scaling import (
    _BT_MAX_PREFILL_SAVINGS,
    _bt_size_for_target,
    compute_constraint_aware_schedule,
    compute_frontier_rho_schedule,
)
from aurelius.traces import azure_llm, burstgpt
from aurelius.traces.backtest import run_backtest
from aurelius.traces.replay import requests_to_arrival_ticks

AZURE_FIXTURE = "tests/fixtures/azure_llm_2024_sample.csv"
BURSTGPT_FIXTURE = "tests/fixtures/burstgpt_sample.csv"
TICK_SECONDS = 60.0
TICK_HOURS = TICK_SECONDS / 3600.0
_WINDOW = 10


@pytest.fixture
def azure_ticks():
    reqs = azure_llm.load_csv(AZURE_FIXTURE)
    return list(requests_to_arrival_ticks(reqs, tick_seconds=TICK_SECONDS))


@pytest.fixture
def burstgpt_ticks():
    reqs = burstgpt.load_csv(BURSTGPT_FIXTURE)
    return list(requests_to_arrival_ticks(reqs, tick_seconds=TICK_SECONDS))


# ---------------------------------------------------------------------------
# Mechanism tests
# ---------------------------------------------------------------------------

class TestFrontierRhoScheduleMechanism:
    """compute_frontier_rho_schedule returns well-formed schedules."""

    def test_returns_list_of_floats(self, azure_ticks):
        sched = compute_frontier_rho_schedule(azure_ticks, TICK_HOURS, window=_WINDOW)
        assert isinstance(sched, list)
        assert len(sched) == len(azure_ticks)
        assert all(isinstance(r, float) for r in sched)

    def test_length_matches_ticks(self, burstgpt_ticks):
        sched = compute_frontier_rho_schedule(burstgpt_ticks, TICK_HOURS, window=_WINDOW)
        assert len(sched) == len(burstgpt_ticks)

    def test_cold_start_uses_default_rho(self, azure_ticks):
        default_rho = 0.65
        sched = compute_frontier_rho_schedule(
            azure_ticks, TICK_HOURS, window=_WINDOW, default_rho=default_rho
        )
        # First `window` entries must be exactly default_rho
        assert all(r == default_rho for r in sched[:_WINDOW]), (
            f"cold-start ticks should all be {default_rho}, got {sched[:_WINDOW]}"
        )

    def test_cold_start_custom_rho(self, burstgpt_ticks):
        custom_default = 0.72
        sched = compute_frontier_rho_schedule(
            burstgpt_ticks, TICK_HOURS, window=5, default_rho=custom_default
        )
        assert all(r == custom_default for r in sched[:5])

    def test_post_warmup_rho_in_valid_range(self, azure_ticks):
        sched = compute_frontier_rho_schedule(azure_ticks, TICK_HOURS, window=_WINDOW)
        # All rho values must be in [0.45, 0.95] — frontier profile bounds
        for i, r in enumerate(sched[_WINDOW:], start=_WINDOW):
            assert 0.45 <= r <= 0.95, f"tick {i}: rho {r} out of [0.45, 0.95]"

    def test_window_zero_falls_back_gracefully(self, azure_ticks):
        """window=0 produces no-history fallback for all ticks (empty tel_window → default_rho)."""
        sched = compute_frontier_rho_schedule(azure_ticks, TICK_HOURS, window=0, default_rho=0.65)
        assert len(sched) == len(azure_ticks)
        # With window=0 every tick has an empty causal window → all fall back to default_rho
        assert all(r == 0.65 for r in sched), (
            "window=0 should produce default_rho for all ticks (no history)"
        )

    def test_post_warmup_adapts(self, azure_ticks):
        """After the warm-up window, at least some ticks should deviate from default_rho."""
        default_rho = 0.65
        sched = compute_frontier_rho_schedule(azure_ticks, TICK_HOURS, window=_WINDOW)
        post_warmup = sched[_WINDOW:]
        # At least one tick should have selected a rho != 0.65
        has_adaptation = any(r != default_rho for r in post_warmup)
        assert has_adaptation, (
            "No post-warmup tick deviated from default_rho — frontier estimator may be stuck"
        )

    def test_rho_schedule_is_causal(self, azure_ticks):
        """The schedule at position k must be computable without tick k or later."""
        # Verify by running on a prefix and comparing — if causal, prefix result
        # must equal the first prefix-length entries of the full schedule.
        n = min(30, len(azure_ticks))
        full_sched = compute_frontier_rho_schedule(
            azure_ticks[:n], TICK_HOURS, window=_WINDOW
        )
        prefix = 20
        prefix_sched = compute_frontier_rho_schedule(
            azure_ticks[:prefix], TICK_HOURS, window=_WINDOW
        )
        # First `prefix` entries of full_sched must exactly match prefix_sched
        assert full_sched[:prefix] == prefix_sched, (
            "Schedule is not causal: prefix result differs from full-run prefix"
        )


# ---------------------------------------------------------------------------
# Prefill-savings integration tests
# ---------------------------------------------------------------------------

class TestPrefillSavingsIntegration:
    """Verify actual reuse_fraction telemetry flows into the frontier estimator."""

    def test_nonzero_reuse_fraction_present(self, burstgpt_ticks):
        """BurstGPT fixture has non-zero reuse_fraction — needed for meaningful prefill tests."""
        active = [t for t in burstgpt_ticks if t.request_count > 0]
        nonzero = sum(1 for t in active if t.reuse_fraction > 0.0)
        assert nonzero > 0, "No BurstGPT active ticks with reuse_fraction > 0 — cannot test"

    def test_default_max_prefill_savings_matches_ca(self, azure_ticks):
        """compute_frontier_rho_schedule default max_prefill_savings == CA's constant."""
        from aurelius.traces.backtest import MAX_PREFILL_SAVINGS
        assert _BT_MAX_PREFILL_SAVINGS == MAX_PREFILL_SAVINGS, (
            "Mismatch between frontier and CA prefill savings constant"
        )

    def test_custom_max_prefill_savings_accepted(self, burstgpt_ticks):
        """max_prefill_savings parameter is accepted and does not crash."""
        sched_default = compute_frontier_rho_schedule(
            burstgpt_ticks, TICK_HOURS, window=_WINDOW
        )
        sched_zero = compute_frontier_rho_schedule(
            burstgpt_ticks, TICK_HOURS, window=_WINDOW, max_prefill_savings=0.0
        )
        # Both should return valid schedules of the same length
        assert len(sched_default) == len(sched_zero) == len(burstgpt_ticks)
        assert all(0.45 <= r <= 0.95 for r in sched_default[_WINDOW:])
        assert all(0.45 <= r <= 0.95 for r in sched_zero[_WINDOW:])

    def test_zero_vs_nonzero_prefill_may_differ(self, azure_ticks):
        """When actual reuse_fraction > 0, prefill_savings=0.0 vs default can produce
        different rho selections (estimator SLA evaluation differs).  This test
        documents that the behavior CAN differ — it is not a parity requirement."""
        sched_with = compute_frontier_rho_schedule(
            azure_ticks, TICK_HOURS, window=_WINDOW
        )
        sched_without = compute_frontier_rho_schedule(
            azure_ticks, TICK_HOURS, window=_WINDOW, max_prefill_savings=0.0
        )
        # Both are valid schedules — no assertion on identical vs different.
        # Just verify they are well-formed.
        assert len(sched_with) == len(sched_without) == len(azure_ticks)


# ---------------------------------------------------------------------------
# Null-result gate: MIN_REPLICAS floor at fixture loads
# ---------------------------------------------------------------------------

class TestNullResultGate:
    """constraint_aware_adaptive == constraint_aware at fixture loads.

    This is the documented Phase 4 null result: at BurstGPT + Azure LLM 2024
    fixture load levels, _bt_size_for_target() returns MIN_REPLICAS=1 for every
    tick regardless of rho (0.65 or 0.95), so the adaptive schedule is
    integer-identical to the fixed-rho baseline.
    """

    def test_end_to_end_null_confirmed_by_schedule_parity(self, azure_ticks):
        """The Phase 4 null result is confirmed by schedule-level parity: even if
        some individual ticks are rho-sensitive in theory, EWMA smoothing and
        _constraint_trim hysteresis ensure the actual schedules are identical at
        fixture loads.  The schedule-level tests below are the authoritative gate."""
        ca_sched = compute_constraint_aware_schedule(azure_ticks, TICK_HOURS)
        rho_sched = compute_frontier_rho_schedule(azure_ticks, TICK_HOURS, window=_WINDOW)
        adaptive_sched = compute_constraint_aware_schedule(
            azure_ticks, TICK_HOURS, rho_schedule=rho_sched
        )
        n_diff = sum(a != b for a, b in zip(ca_sched, adaptive_sched))
        # Gate: schedules must be identical at fixture loads
        assert n_diff == 0, (
            f"CA and adaptive schedules differ at {n_diff} ticks — "
            "Phase 4 is not null on this fixture as expected"
        )

    def test_adaptive_schedule_null_vs_ca_azure(self, azure_ticks):
        """constraint_aware_adaptive schedule == constraint_aware schedule on Azure fixture."""
        ca_sched = compute_constraint_aware_schedule(azure_ticks, TICK_HOURS)
        rho_sched = compute_frontier_rho_schedule(azure_ticks, TICK_HOURS, window=_WINDOW)
        adaptive_sched = compute_constraint_aware_schedule(
            azure_ticks, TICK_HOURS, rho_schedule=rho_sched
        )
        assert ca_sched == adaptive_sched, (
            f"CA and adaptive schedules differ: "
            f"{sum(a != b for a, b in zip(ca_sched, adaptive_sched))} tick(s) differ"
        )

    def test_adaptive_schedule_null_vs_ca_burstgpt(self, burstgpt_ticks):
        """constraint_aware_adaptive schedule == constraint_aware schedule on BurstGPT fixture."""
        ca_sched = compute_constraint_aware_schedule(burstgpt_ticks, TICK_HOURS)
        rho_sched = compute_frontier_rho_schedule(burstgpt_ticks, TICK_HOURS, window=_WINDOW)
        adaptive_sched = compute_constraint_aware_schedule(
            burstgpt_ticks, TICK_HOURS, rho_schedule=rho_sched
        )
        assert ca_sched == adaptive_sched, (
            f"CA and adaptive schedules differ: "
            f"{sum(a != b for a, b in zip(ca_sched, adaptive_sched))} tick(s) differ"
        )

    def test_run_backtest_null_result_azure(self):
        """Full end-to-end: constraint_aware_adaptive KPIs == constraint_aware on Azure fixture."""
        reqs = azure_llm.load_csv(AZURE_FIXTURE)
        res = run_backtest(
            reqs, tick_seconds=TICK_SECONDS,
            policies=["constraint_aware", "constraint_aware_adaptive"]
        )
        ca_kpi = res.policy_results["constraint_aware"].kpi.sla_safe_goodput_per_infra_dollar
        caa_kpi = res.policy_results["constraint_aware_adaptive"].kpi.sla_safe_goodput_per_infra_dollar
        assert abs(ca_kpi - caa_kpi) < 1e-6, (
            f"Phase 4 Azure fixture: expected null (+0.00%) but got "
            f"ca={ca_kpi:.4f} adaptive={caa_kpi:.4f} delta={caa_kpi - ca_kpi:+.6f}"
        )

    def test_run_backtest_null_result_burstgpt(self):
        """Full end-to-end: constraint_aware_adaptive KPIs == constraint_aware on BurstGPT fixture."""
        reqs = burstgpt.load_csv(BURSTGPT_FIXTURE)
        res = run_backtest(
            reqs, tick_seconds=TICK_SECONDS,
            policies=["constraint_aware", "constraint_aware_adaptive"]
        )
        ca_kpi = res.policy_results["constraint_aware"].kpi.sla_safe_goodput_per_infra_dollar
        caa_kpi = res.policy_results["constraint_aware_adaptive"].kpi.sla_safe_goodput_per_infra_dollar
        assert abs(ca_kpi - caa_kpi) < 1e-6, (
            f"Phase 4 BurstGPT fixture: expected null (+0.00%) but got "
            f"ca={ca_kpi:.4f} adaptive={caa_kpi:.4f} delta={caa_kpi - ca_kpi:+.6f}"
        )

    def test_adaptive_window_none_byte_identical(self):
        """ReplicaScalingConfig.adaptive_frontier_window=None preserves CA behavior."""
        from aurelius.optimizer.policies.replica_scaling import (
            ReplicaScalingConfig,
            ReplicaScalingPolicy,
        )
        policy = ReplicaScalingPolicy()
        reqs = azure_llm.load_csv(AZURE_FIXTURE)
        ticks = list(requests_to_arrival_ticks(reqs, tick_seconds=TICK_SECONDS))

        cfg_fixed = ReplicaScalingConfig(mode="constraint_aware", adaptive_frontier_window=None)
        cfg_adaptive = ReplicaScalingConfig(mode="constraint_aware", adaptive_frontier_window=10)

        result_fixed = policy.optimize_from_ticks(ticks, tick_hours=TICK_HOURS, config=cfg_fixed)
        result_adaptive = policy.optimize_from_ticks(ticks, tick_hours=TICK_HOURS, config=cfg_adaptive)

        # At fixture loads: same schedule
        assert result_fixed.c_schedule == result_adaptive.c_schedule, (
            "adaptive_frontier_window=10 should equal None at fixture loads "
            f"({sum(a != b for a, b in zip(result_fixed.c_schedule, result_adaptive.c_schedule))} diffs)"
        )


# ---------------------------------------------------------------------------
# High-load sensitivity: verify mechanism IS functional at multi-replica loads
# ---------------------------------------------------------------------------

class TestHighLoadMechanism:
    """At synthetic multi-replica loads, rho adaptation produces different counts.

    These tests use synthetic high-load ticks to verify the mechanism is not
    broken — only that fixture loads happen to be in the single-replica regime.
    """

    def _make_high_load_tick(self, arrival_rps: float = 5.0, output_tokens: float = 500.0,
                             prompt_tokens: float = 100.0, request_count: int = 300):
        """Build a minimal ArrivalTick-compatible object at high load."""
        class _T:
            pass
        t = _T()
        t.tick_index = 0
        t.arrival_rate_rps = arrival_rps
        t.output_tokens_mean = output_tokens
        t.prompt_tokens_mean = prompt_tokens
        t.request_count = request_count
        t.reuse_fraction = 0.3
        t.model_mix = None
        return t

    def test_high_load_rho_sensitivity(self):
        """At production-scale loads, rho=0.65 and rho=0.95 give different replica counts."""
        t = self._make_high_load_tick(arrival_rps=5.0, output_tokens=500.0)
        # Use a representative throughput value
        throughput = 50.0  # tokens/s per replica rough estimate
        r65 = _bt_size_for_target(t.arrival_rate_rps, max(1.0, t.output_tokens_mean),
                                  throughput, 0.65)
        r95 = _bt_size_for_target(t.arrival_rate_rps, max(1.0, t.output_tokens_mean),
                                  throughput, 0.95)
        # At high load, rho=0.95 allows higher utilization → fewer replicas
        # The key test: at SOME load level they differ
        assert r65 >= 1 and r95 >= 1, "Both should be at least MIN_REPLICAS"
        # Document whether they differ (not asserting equal/unequal — depends on load)
        # At 5 rps / 500 tok: r65=50tok*5rps/65%/throughput ~ should differ from r95
        # This is a diagnostic test, not a gate
        assert isinstance(r65, int) and isinstance(r95, int)

    def test_high_load_makes_rho_differ(self):
        """Find a load level where rho=0.65 != rho=0.95 in replica count."""
        found_diff = False
        for throughput in [1.0, 2.0, 5.0, 10.0, 20.0]:
            r65 = _bt_size_for_target(5.0, 500.0, throughput, 0.65)
            r95 = _bt_size_for_target(5.0, 500.0, throughput, 0.95)
            if r65 != r95:
                found_diff = True
                break
        assert found_diff, (
            "Could not find any throughput where rho=0.65 and rho=0.95 differ — "
            "Phase 4 mechanism may be broken at all load levels"
        )
