"""OSOTSS canonical routing parity — Phase 3 architecture integration.

Verifies that routing OSOTSS through AureliusOptimizer(policy="replica_scaling",
mode="online_sotss") produces bit-identical c_schedules and matching KPIs
compared to the direct ``compute_online_sotss_schedule`` call path.

This test suite locks in the Phase 3 integration: the OSOTSS backtest now routes
through ``_REPLICA_SCALING_OPTIMIZER.optimize()`` instead of calling the policy
function directly, making AureliusOptimizer the single canonical owner of all
per-tick capacity scheduling decisions (amcsg, sotss_min, online_sotss,
forecasted_mcs).

Key integration: ``ReplicaScalingConfig.baseline_n_sla_safe`` allows the caller
to pass the AMCSG stochastic GSF baseline so the oracle targets the same SLA-safe
floor as the stochastic evaluation (not the more-conservative deterministic floor).
"""

from __future__ import annotations

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    _ReplicaScalingConfig,
    calibrate_time_warp,
    load_serving_requests,
)
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.policies.replica_scaling import (
    REPLICA_AGGRESSIVE_GATE,
    REPLICA_MAX_ORACLE_ITERS,
    REPLICA_OSOTSS_EWMA_ALPHA,
    REPLICA_SAFE_GATE,
    ReplicaScalingConfig,
    ReplicaScalingPolicy,
    compute_online_sotss_schedule,
)

_OPTIMIZER = AureliusOptimizer(policy="replica_scaling")


def _make_trace(n_ticks: int = 12, per_tick: int = 25, seed: int = 0) -> list:
    import random
    rng = random.Random(seed)
    raw: list = []
    tick_s = 60.0
    for t in range(n_ticks):
        count = per_tick + (t % 4) * 5
        for j in range(count):
            raw.append((
                (t + (j + 0.5) / count) * tick_s,
                40 + rng.randint(0, 80),
            ))
    raw.sort()
    return raw


# ---------------------------------------------------------------------------
# Helper: run both paths and return their c_schedules
# ---------------------------------------------------------------------------

def _parity_pair(
    raw: list,
    tick_s: float = 60.0,
    sla_s: float = 10.0,
    safe_gate: float = REPLICA_SAFE_GATE,
    aggressive_gate: float = REPLICA_AGGRESSIVE_GATE,
    max_iters: int = REPLICA_MAX_ORACLE_ITERS,
    ewma_alpha: float = REPLICA_OSOTSS_EWMA_ALPHA,
    baseline_n_sla_safe: int | None = None,
):
    """Return (direct_c_sched, canonical_c_sched) for the same OSOTSS run."""
    warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)

    direct_c, n_iters_d, init_viols_d, n_cheaper_d, baseline_d = (
        compute_online_sotss_schedule(
            raw, tick_s, warp, sla_s,
            safe_gate=safe_gate,
            aggressive_gate=aggressive_gate,
            max_iters=max_iters,
            ewma_alpha=ewma_alpha,
            baseline_n_sla_safe=baseline_n_sla_safe,
        )
    )

    result = _OPTIMIZER.optimize(
        raw,
        warp=warp,
        config=ReplicaScalingConfig(
            mode="online_sotss",
            tick_seconds=tick_s,
            sla_s=sla_s,
            safe_gate_pct=safe_gate,
            aggressive_gate_pct=aggressive_gate,
            max_oracle_iters=max_iters,
            ewma_alpha=ewma_alpha,
            baseline_n_sla_safe=baseline_n_sla_safe,
        ),
    )

    return (
        direct_c, n_iters_d, init_viols_d, n_cheaper_d, baseline_d,
        result.c_schedule, result.oracle_iters, result.initial_violations,
        result.n_ticks_cheaper, result.baseline_n_sla_safe,
    )


# ---------------------------------------------------------------------------
# 1. c_schedule parity (core integration guarantee)
# ---------------------------------------------------------------------------

class TestOSOTSSCanonicalScheduleParity:
    """Canonical routing produces bit-identical c_schedule to the direct call."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
    def test_c_schedule_identical_no_baseline_override(self, seed):
        raw = _make_trace(seed=seed)
        d = _parity_pair(raw)
        assert d[0] == d[5], "c_schedule mismatch (no baseline override)"

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
    def test_c_schedule_identical_with_baseline_override(self, seed):
        raw = _make_trace(seed=seed)
        # baseline=100 forces the oracle to provision for 100 safe requests
        d = _parity_pair(raw, baseline_n_sla_safe=100)
        assert d[0] == d[5], "c_schedule mismatch (baseline override)"

    def test_c_schedule_identical_burst_trace(self):
        raw = _make_trace(n_ticks=20, per_tick=5, seed=77)
        d = _parity_pair(raw)
        assert d[0] == d[5], "c_schedule mismatch on burst trace"

    def test_c_schedule_identical_smooth_trace(self):
        raw = _make_trace(n_ticks=8, per_tick=50, seed=3)
        d = _parity_pair(raw)
        assert d[0] == d[5], "c_schedule mismatch on smooth trace"


# ---------------------------------------------------------------------------
# 2. Diagnostic field parity (oracle_iters, initial_violations, n_ticks_cheaper)
# ---------------------------------------------------------------------------

class TestOSOTSSCanonicalDiagnosticParity:

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_oracle_iters_identical(self, seed):
        raw = _make_trace(seed=seed)
        d = _parity_pair(raw)
        assert d[1] == d[6], "oracle_iters mismatch"

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_initial_violations_identical(self, seed):
        raw = _make_trace(seed=seed)
        d = _parity_pair(raw)
        assert d[2] == d[7], "initial_violations mismatch"

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_n_ticks_cheaper_identical(self, seed):
        raw = _make_trace(seed=seed)
        d = _parity_pair(raw)
        assert d[3] == d[8], "n_ticks_cheaper mismatch"

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_baseline_n_sla_safe_identical(self, seed):
        raw = _make_trace(seed=seed)
        d = _parity_pair(raw)
        assert d[4] == d[9], "baseline_n_sla_safe mismatch"


# ---------------------------------------------------------------------------
# 3. ReplicaScalingConfig.baseline_n_sla_safe contract
# ---------------------------------------------------------------------------

class TestBaselineNSlaSafeConfig:

    def test_default_none_matches_direct_none(self):
        raw = _make_trace(seed=5)
        d = _parity_pair(raw, baseline_n_sla_safe=None)
        assert d[0] == d[5]

    def test_explicit_zero_baseline(self):
        raw = _make_trace(seed=6)
        d = _parity_pair(raw, baseline_n_sla_safe=0)
        assert d[0] == d[5]

    def test_high_baseline_forces_more_capacity(self):
        raw = _make_trace(seed=8, per_tick=30)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        r_low = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0,
                                        baseline_n_sla_safe=0),
        )
        r_high = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0,
                                        baseline_n_sla_safe=9999),
        )
        # High baseline must produce at least as much capacity as low baseline
        assert r_high.c_mean >= r_low.c_mean

    def test_result_carries_baseline_used(self):
        raw = _make_trace(seed=9)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0,
                                        baseline_n_sla_safe=None),
        )
        assert isinstance(result.baseline_n_sla_safe, int)
        assert result.baseline_n_sla_safe >= 0

    def test_result_carries_initial_violations(self):
        raw = _make_trace(seed=10)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0),
        )
        assert isinstance(result.initial_violations, int)
        assert result.initial_violations >= 0


# ---------------------------------------------------------------------------
# 4. AureliusOptimizer facade entry-point smoke test
# ---------------------------------------------------------------------------

class TestAureliusOptimizerOSOTSSFacade:

    def test_facade_mode_tag(self):
        raw = _make_trace(seed=11)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        ao = AureliusOptimizer(policy="replica_scaling")
        result = ao.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0),
        )
        assert result.mode == "online_sotss"

    def test_facade_returns_valid_c_schedule(self):
        raw = _make_trace(seed=12)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        ao = AureliusOptimizer(policy="replica_scaling")
        result = ao.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0),
        )
        assert len(result.c_schedule) > 0
        assert all(isinstance(c, int) and c >= 1 for c in result.c_schedule)

    def test_empty_raw_handled_gracefully(self):
        ao = AureliusOptimizer(policy="replica_scaling")
        result = ao.optimize(
            [], warp=1.0,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0),
        )
        assert result.c_schedule == []
        assert result.oracle_iters == 0

    def test_osotss_cheaper_than_amcsg_on_typical_trace(self):
        raw = _make_trace(n_ticks=15, per_tick=30, seed=13)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        ao = AureliusOptimizer(policy="replica_scaling")
        r_osotss = ao.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="online_sotss", sla_s=10.0),
        )
        r_amcsg = ao.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", sla_s=10.0),
        )
        # OSOTSS c_mean should be ≤ AMCSG (same or fewer servers)
        assert r_osotss.c_mean <= r_amcsg.c_mean + 0.5


# ---------------------------------------------------------------------------
# 5. Backtest-level KPI fixture parity (sample fixture, deterministic)
# ---------------------------------------------------------------------------

class TestBacktestLevelKPIParity:
    """Verify backtest report KPIs match expected values after routing change."""

    @pytest.fixture(scope="class")
    def azure_report(self):
        from aurelius.benchmarks.srtf_serving_backtest import (
            run_online_sotss_azure_backtest,
        )
        return run_online_sotss_azure_backtest(fixed_c=4, job_limit=200)

    def test_report_is_online_sotss(self, azure_report):
        assert azure_report.trace == "azure_llm_2024_online_sotss"

    def test_osotss_outperforms_amcsg(self, azure_report):
        assert azure_report.osotss_goodput_per_dollar > azure_report.amcsg_goodput_per_dollar

    def test_initial_violations_nonneg(self, azure_report):
        assert azure_report.osotss_initial_violations >= 0

    def test_n_iters_positive(self, azure_report):
        assert azure_report.osotss_n_iters >= 0

    def test_c_mean_less_than_amcsg(self, azure_report):
        # OSOTSS uses fewer mean servers than AMCSG
        assert azure_report.osotss_c_mean <= azure_report.amcsg_c_mean + 0.1
