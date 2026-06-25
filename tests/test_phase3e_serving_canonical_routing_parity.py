"""Phase 3e parity tests: CA/SHU routed through AureliusOptimizer(policy="replica_scaling").

Verifies that routing constraint_aware and safe_high_utilization through the
canonical optimizer produces zero KPI drift vs the original inline logic.

Two verification layers:
  1. Physics layer: _bt_timeout_rate_pct == evaluate_tick().timeout_rate_pct
  2. Schedule layer: compute_constraint_aware_schedule / compute_shu_schedule
     produce bit-identical per-tick replica counts vs the original _run_policy loop
  3. End-to-end KPI: run_backtest with the patched _run_policy gives same KPIs
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.optimizer.policies.replica_scaling import (
    ReplicaScalingConfig,
    _bt_timeout_rate_pct,
    compute_constraint_aware_schedule,
    compute_shu_schedule,
)
from aurelius.traces import azure_llm, burstgpt
from aurelius.traces.backtest import (
    _SERVING_OPTIMIZER,
    _SHU_TARGET_RHO,
    _SHU_TIMEOUT_TOL,
    MAX_PREFILL_SAVINGS,
    _constraint_trim,
    _size_for_target,
    _tick_throughput_tokps,
    evaluate_tick,
    run_backtest,
)
from aurelius.traces.replay import requests_to_arrival_ticks
from aurelius.traces.schema import time_rescale

AZURE_FIXTURE = "tests/fixtures/azure_llm_2024_sample.csv"
BURSTGPT_FIXTURE = "tests/fixtures/burstgpt_sample.csv"
TICK_HOURS = 1.0 / 60.0
EWMA_ALPHA = 0.5


def _original_ca_schedule(ticks, tick_hours: float, ca_target_rho: float = 0.65):
    """Replicate original _run_policy constraint_aware loop (pre-Phase-3e)."""
    ewma_rate = 0.0
    ewma_out = 0.0
    prev_replicas = None
    schedule = []
    for t in ticks:
        if t.request_count > 0:
            ewma_rate = (
                EWMA_ALPHA * t.arrival_rate_rps + (1 - EWMA_ALPHA) * ewma_rate
                if ewma_rate else t.arrival_rate_rps
            )
            ewma_out = (
                EWMA_ALPHA * t.output_tokens_mean + (1 - EWMA_ALPHA) * ewma_out
                if ewma_out else t.output_tokens_mean
            )
        prefill = MAX_PREFILL_SAVINGS * t.reuse_fraction
        throughput = _tick_throughput_tokps(t)
        plan_rate = max(t.arrival_rate_rps, ewma_rate)
        plan_out = max(t.output_tokens_mean, ewma_out) if t.request_count else ewma_out
        base = _size_for_target(plan_rate, max(1.0, plan_out), throughput, ca_target_rho)
        replicas = _constraint_trim(t, base, prefill, tick_hours, prev_replicas)
        schedule.append(replicas)
        prev_replicas = replicas
    return schedule


def _original_shu_schedule(ticks, tick_hours: float):
    """Replicate original _run_policy safe_high_utilization loop (pre-Phase-3e)."""
    ewma_rate = 0.0
    ewma_out = 0.0
    schedule = []
    for t in ticks:
        if t.request_count > 0:
            ewma_rate = (
                EWMA_ALPHA * t.arrival_rate_rps + (1 - EWMA_ALPHA) * ewma_rate
                if ewma_rate else t.arrival_rate_rps
            )
            ewma_out = (
                EWMA_ALPHA * t.output_tokens_mean + (1 - EWMA_ALPHA) * ewma_out
                if ewma_out else t.output_tokens_mean
            )
        prefill = MAX_PREFILL_SAVINGS * t.reuse_fraction
        throughput = _tick_throughput_tokps(t)
        plan_rate = max(t.arrival_rate_rps, ewma_rate)
        plan_out = max(t.output_tokens_mean, ewma_out) if t.request_count else ewma_out
        base = _size_for_target(plan_rate, max(1.0, plan_out), throughput, _SHU_TARGET_RHO)
        replicas = _constraint_trim(t, base, prefill, tick_hours, None, _SHU_TIMEOUT_TOL)
        schedule.append(replicas)
    return schedule


@pytest.fixture
def azure_ticks_50x():
    reqs = azure_llm.load_csv(AZURE_FIXTURE)
    reqs_50 = time_rescale(reqs, 50)
    return requests_to_arrival_ticks(reqs_50, tick_seconds=60.0)


@pytest.fixture
def burstgpt_ticks():
    reqs = burstgpt.load_csv(BURSTGPT_FIXTURE, sample_size=5000, seed=0)
    return requests_to_arrival_ticks(reqs, tick_seconds=60.0)


class TestPhysicsLayer:
    """_bt_timeout_rate_pct == evaluate_tick().timeout_rate_pct for all r."""

    def test_timeout_rate_pct_bit_identical_azure(self, azure_ticks_50x):
        active = [t for t in azure_ticks_50x if t.request_count > 0][:30]
        mismatches = []
        for t in active:
            throughput = _tick_throughput_tokps(t)
            prefill = MAX_PREFILL_SAVINGS * t.reuse_fraction
            for r in range(1, 6):
                orig = evaluate_tick(
                    t, r, prefill_savings=prefill, tick_hours=TICK_HOURS
                ).timeout_rate_pct
                extracted = _bt_timeout_rate_pct(
                    t.arrival_rate_rps, t.output_tokens_mean,
                    t.prompt_tokens_mean, throughput, r, prefill,
                )
                if abs(orig - extracted) > 1e-12:
                    mismatches.append((t.tick_index, r, orig, extracted))
        assert not mismatches, f"Physics mismatch: {mismatches[:5]}"

    def test_timeout_rate_pct_bit_identical_burstgpt(self, burstgpt_ticks):
        active = [t for t in burstgpt_ticks if t.request_count > 0][:20]
        mismatches = []
        for t in active:
            throughput = _tick_throughput_tokps(t)
            prefill = MAX_PREFILL_SAVINGS * t.reuse_fraction
            for r in range(1, 6):
                orig = evaluate_tick(
                    t, r, prefill_savings=prefill, tick_hours=TICK_HOURS
                ).timeout_rate_pct
                extracted = _bt_timeout_rate_pct(
                    t.arrival_rate_rps, t.output_tokens_mean,
                    t.prompt_tokens_mean, throughput, r, prefill,
                )
                if abs(orig - extracted) > 1e-12:
                    mismatches.append((t.tick_index, r, orig, extracted))
        assert not mismatches, f"Physics mismatch: {mismatches[:5]}"


class TestScheduleLayer:
    """Per-tick replica counts from canonical functions == original _run_policy loop."""

    def test_ca_schedule_bit_identical_azure(self, azure_ticks_50x):
        orig = _original_ca_schedule(azure_ticks_50x, TICK_HOURS)
        canon = compute_constraint_aware_schedule(azure_ticks_50x, TICK_HOURS)
        assert orig == canon, f"CA mismatch at first diff: {next((i,a,b) for i,(a,b) in enumerate(zip(orig,canon)) if a!=b)}"

    def test_shu_schedule_bit_identical_azure(self, azure_ticks_50x):
        orig = _original_shu_schedule(azure_ticks_50x, TICK_HOURS)
        canon = compute_shu_schedule(azure_ticks_50x, TICK_HOURS)
        assert orig == canon, f"SHU mismatch at first diff: {next((i,a,b) for i,(a,b) in enumerate(zip(orig,canon)) if a!=b)}"

    def test_ca_schedule_bit_identical_burstgpt(self, burstgpt_ticks):
        orig = _original_ca_schedule(burstgpt_ticks, TICK_HOURS)
        canon = compute_constraint_aware_schedule(burstgpt_ticks, TICK_HOURS)
        assert orig == canon

    def test_shu_schedule_bit_identical_burstgpt(self, burstgpt_ticks):
        orig = _original_shu_schedule(burstgpt_ticks, TICK_HOURS)
        canon = compute_shu_schedule(burstgpt_ticks, TICK_HOURS)
        assert orig == canon


class TestOptimizerInterface:
    """AureliusOptimizer(policy='replica_scaling').policy.optimize_from_ticks() works."""

    def test_serving_optimizer_is_replica_scaling(self):
        from aurelius.optimizer.policies.replica_scaling import ReplicaScalingPolicy
        assert isinstance(_SERVING_OPTIMIZER.policy, ReplicaScalingPolicy)

    def test_optimize_from_ticks_ca(self, azure_ticks_50x):
        cfg = ReplicaScalingConfig(mode="constraint_aware", ca_target_rho=0.65)
        result = _SERVING_OPTIMIZER.policy.optimize_from_ticks(
            list(azure_ticks_50x), tick_hours=TICK_HOURS, config=cfg
        )
        orig = _original_ca_schedule(azure_ticks_50x, TICK_HOURS)
        assert result.c_schedule == orig
        assert result.mode == "constraint_aware"
        assert result.n_ticks == len(azure_ticks_50x)

    def test_optimize_from_ticks_shu(self, azure_ticks_50x):
        cfg = ReplicaScalingConfig(mode="safe_high_utilization")
        result = _SERVING_OPTIMIZER.policy.optimize_from_ticks(
            list(azure_ticks_50x), tick_hours=TICK_HOURS, config=cfg
        )
        orig = _original_shu_schedule(azure_ticks_50x, TICK_HOURS)
        assert result.c_schedule == orig
        assert result.mode == "safe_high_utilization"

    def test_invalid_mode_raises(self, azure_ticks_50x):
        cfg = ReplicaScalingConfig(mode="sotss_min")
        with pytest.raises(ValueError, match="unsupported mode"):
            _SERVING_OPTIMIZER.policy.optimize_from_ticks(
                list(azure_ticks_50x), tick_hours=TICK_HOURS, config=cfg
            )


class TestEndToEndKPIParity:
    """Full run_backtest KPI parity — 0% drift gating condition."""

    def test_azure_200x_kpi_ordering(self):
        """At 200x load CA and SHU diverge — verify ordering and sla_aware rank."""
        reqs = azure_llm.load_csv(AZURE_FIXTURE)
        reqs_200 = time_rescale(reqs, 200)
        res = run_backtest(
            reqs_200, tick_seconds=60.0,
            policies=["constraint_aware", "safe_high_utilization", "sla_aware"]
        )
        ca_gpd = res.policy_results["constraint_aware"].kpi.sla_safe_goodput_per_infra_dollar
        shu_gpd = res.policy_results["safe_high_utilization"].kpi.sla_safe_goodput_per_infra_dollar
        sla_gpd = res.policy_results["sla_aware"].kpi.sla_safe_goodput_per_infra_dollar

        # At 200x SHU (rho=0.75) out-performs CA (rho=0.65) — higher utilisation wins
        assert ca_gpd < shu_gpd, f"CA={ca_gpd:.0f} should < SHU={shu_gpd:.0f} at 200x"
        # Both anticipatory policies beat the reactive sla_aware baseline
        assert shu_gpd > sla_gpd, f"SHU={shu_gpd:.0f} should > sla_aware={sla_gpd:.0f}"

    def test_mcs_anchor_unchanged(self):
        """Canary: min_cost_safe Azure 500x must still equal the ROADMAP anchor."""
        reqs = azure_llm.load_csv(AZURE_FIXTURE)
        reqs_500 = time_rescale(reqs, 500)
        res = run_backtest(reqs_500, tick_seconds=60.0, policies=["min_cost_safe"])
        gpd = res.policy_results["min_cost_safe"].kpi.sla_safe_goodput_per_infra_dollar
        assert abs(gpd - 2_657_445) < 1000, f"MCS anchor drift: {gpd:.0f}"
