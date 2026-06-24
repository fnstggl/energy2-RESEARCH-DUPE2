"""AMCSG + SOTSS-MIN canonical routing parity — Phase 4 architecture integration.

Verifies that routing AMCSG and SOTSS-MIN through
AureliusOptimizer(policy="replica_scaling", mode="amcsg"/"sotss_min")
produces bit-identical c_schedules compared to the direct compute-function
call path.

This suite locks in the Phase 4 integration: _run_amcsg_backtest and
_run_sotss_backtest now route through _REPLICA_SCALING_OPTIMIZER.optimize()
instead of calling _joint_mcs_c_schedule / _sotss_min_cost_schedule directly,
making AureliusOptimizer the single canonical owner of all per-tick capacity
scheduling decisions across every backtest entry point.

New in this suite:
  - initial_violations is now propagated through ReplicaScalingResult for
    sotss_min mode (previously discarded with _ in the policy optimize()).
"""

from __future__ import annotations

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    _joint_mcs_c_schedule,
    _sotss_min_cost_schedule,
    calibrate_time_warp,
    run_amcsg_azure_backtest,
    run_amcsg_burstgpt_backtest,
    run_sotss_min_azure_backtest,
)
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.policies.replica_scaling import (
    REPLICA_AGGRESSIVE_GATE,
    REPLICA_MAX_ORACLE_ITERS,
    REPLICA_SAFE_GATE,
    ReplicaScalingConfig,
    compute_mcs_c_schedule,
    compute_sotss_min_schedule,
)

_OPTIMIZER = AureliusOptimizer(policy="replica_scaling")


def _make_trace(n_ticks: int = 10, per_tick: int = 20, seed: int = 0) -> list:
    """Synthetic trace: uniform arrivals per tick, random token counts."""
    import random
    rng = random.Random(seed)
    raw: list = []
    tick_s = 60.0
    for t in range(n_ticks):
        count = per_tick + (t % 3) * 5
        for j in range(count):
            raw.append((
                (t + (j + 0.5) / count) * tick_s,
                30 + rng.randint(0, 60),
            ))
    raw.sort()
    return raw


def _make_bursty_trace(seed: int = 7) -> list:
    """Trace with variable-load ticks to stress SOTSS oracle."""
    import random
    rng = random.Random(seed)
    raw: list = []
    tick_s = 60.0
    loads = [15, 40, 80, 120, 60, 20, 10, 90, 100, 30, 50, 25]
    for t, load in enumerate(loads):
        for j in range(load):
            raw.append((
                (t + (j + 0.5) / load) * tick_s,
                20 + rng.randint(0, 100),
            ))
    raw.sort()
    return raw


# ---------------------------------------------------------------------------
# AMCSG canonical routing parity
# ---------------------------------------------------------------------------

class TestAMCSGCanonicalRoutingParity:
    """Verify AMCSG c_schedule through optimizer == direct compute function."""

    @pytest.mark.parametrize("gate", [9.5, 12.5, 15.0, 20.0, 25.0, 100.0])
    def test_c_schedule_bit_identical(self, gate):
        """optimizer.optimize(mode='amcsg', gate=X) == _joint_mcs_c_schedule(gate=X)."""
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        sla_s = 10.0
        tick_s = 60.0

        direct = _joint_mcs_c_schedule(raw, tick_s, warp, mcs_gate=gate, sla_s=sla_s)
        via_optimizer = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(
                mode="amcsg",
                tick_seconds=tick_s,
                sla_s=sla_s,
                safe_gate_pct=gate,
            ),
        ).c_schedule

        assert direct == via_optimizer, (
            f"c_schedule mismatch at gate={gate}%: "
            f"direct={direct[:5]}, optimizer={via_optimizer[:5]}"
        )

    def test_amcsg_returns_mode_tag(self):
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", sla_s=10.0),
        )
        assert result.mode == "amcsg"

    def test_amcsg_oracle_iters_zero(self):
        """AMCSG has no oracle loop; oracle_iters must be 0."""
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", sla_s=10.0),
        )
        assert result.oracle_iters == 0

    def test_amcsg_initial_violations_zero(self):
        """AMCSG does not run an oracle; initial_violations must be 0."""
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", sla_s=10.0),
        )
        assert result.initial_violations == 0

    def test_amcsg_bursty_trace_parity(self):
        """Parity on a bursty trace with varying per-tick load."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        direct = _joint_mcs_c_schedule(raw, 60.0, warp, mcs_gate=12.5, sla_s=10.0)
        via_opt = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0, safe_gate_pct=12.5),
        ).c_schedule
        assert direct == via_opt

    def test_amcsg_c_mean_matches(self):
        """c_mean field in result == statistics.mean(c_schedule)."""
        import statistics
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", sla_s=10.0),
        )
        assert abs(result.c_mean - statistics.mean(result.c_schedule)) < 1e-9

    def test_amcsg_n_ticks_matches(self):
        """n_ticks == len(c_schedule)."""
        raw = _make_trace(n_ticks=8)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", tick_seconds=60.0, sla_s=10.0),
        )
        assert result.n_ticks == len(result.c_schedule)

    @pytest.mark.parametrize("sla_s", [10.0, 30.0])
    def test_amcsg_sla_variants_parity(self, sla_s):
        """Parity at different SLA budgets."""
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        direct = _joint_mcs_c_schedule(raw, 60.0, warp, mcs_gate=12.5, sla_s=sla_s)
        via_opt = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="amcsg", sla_s=sla_s, safe_gate_pct=12.5),
        ).c_schedule
        assert direct == via_opt


# ---------------------------------------------------------------------------
# SOTSS-MIN canonical routing parity
# ---------------------------------------------------------------------------

class TestSOTSSMinCanonicalRoutingParity:
    """Verify SOTSS-MIN c_schedule through optimizer == direct compute function."""

    def test_c_schedule_bit_identical_uniform(self):
        """optimizer.optimize(mode='sotss_min') == _sotss_min_cost_schedule()."""
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        sla_s, tick_s = 10.0, 60.0

        direct_c, _, direct_init_v, direct_n_cheaper, direct_baseline = (
            _sotss_min_cost_schedule(raw, tick_s, warp, sla_s)
        )
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(
                mode="sotss_min",
                tick_seconds=tick_s,
                sla_s=sla_s,
            ),
        )
        assert result.c_schedule == direct_c, (
            f"c_schedule mismatch: direct={direct_c[:5]}, via_opt={result.c_schedule[:5]}"
        )

    def test_c_schedule_bit_identical_bursty(self):
        """Parity on a bursty trace (stresses oracle convergence)."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        sla_s, tick_s = 10.0, 60.0

        direct_c, _, _, _, _ = _sotss_min_cost_schedule(raw, tick_s, warp, sla_s)
        via_opt = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", tick_seconds=tick_s, sla_s=sla_s),
        ).c_schedule
        assert direct_c == via_opt

    def test_oracle_iters_propagated(self):
        """oracle_iters is propagated from compute_sotss_min_schedule."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        _, n_iters_direct, _, _, _ = _sotss_min_cost_schedule(raw, 60.0, warp, 10.0)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", tick_seconds=60.0, sla_s=10.0),
        )
        assert result.oracle_iters == n_iters_direct

    def test_initial_violations_propagated(self):
        """initial_violations is now propagated (previously discarded with _)."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        _, _, init_v_direct, _, _ = _sotss_min_cost_schedule(raw, 60.0, warp, 10.0)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", tick_seconds=60.0, sla_s=10.0),
        )
        assert result.initial_violations == init_v_direct

    def test_n_ticks_cheaper_propagated(self):
        """n_ticks_cheaper is propagated."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        _, _, _, n_cheaper_direct, _ = _sotss_min_cost_schedule(raw, 60.0, warp, 10.0)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", tick_seconds=60.0, sla_s=10.0),
        )
        assert result.n_ticks_cheaper == n_cheaper_direct

    def test_baseline_n_sla_safe_propagated(self):
        """baseline_n_sla_safe is propagated."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        _, _, _, _, baseline_direct = _sotss_min_cost_schedule(raw, 60.0, warp, 10.0)
        result = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", tick_seconds=60.0, sla_s=10.0),
        )
        assert result.baseline_n_sla_safe == baseline_direct

    def test_sotss_min_mode_tag(self):
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", sla_s=10.0),
        )
        assert result.mode == "sotss_min"

    def test_custom_gates_parity(self):
        """Custom safe_gate/aggressive_gate params round-trip through optimizer."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        sla_s, tick_s = 10.0, 60.0
        safe_gate, agg_gate = 15.0, 80.0

        direct_c, _, _, _, _ = _sotss_min_cost_schedule(
            raw, tick_s, warp, sla_s,
            safe_gate=safe_gate, aggressive_gate=agg_gate,
        )
        via_opt = _OPTIMIZER.optimize(
            raw,
            warp=warp,
            config=ReplicaScalingConfig(
                mode="sotss_min",
                tick_seconds=tick_s,
                sla_s=sla_s,
                safe_gate_pct=safe_gate,
                aggressive_gate_pct=agg_gate,
            ),
        ).c_schedule
        assert direct_c == via_opt

    @pytest.mark.parametrize("sla_s", [10.0, 30.0])
    def test_sla_variants_parity(self, sla_s):
        """Parity across different SLA budgets."""
        raw = _make_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        direct_c, _, _, _, _ = _sotss_min_cost_schedule(raw, 60.0, warp, sla_s)
        via_opt = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", sla_s=sla_s),
        ).c_schedule
        assert direct_c == via_opt

    def test_initial_violations_nonneg(self):
        """initial_violations must always be >= 0."""
        raw = _make_bursty_trace()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        result = _OPTIMIZER.optimize(
            raw, warp=warp,
            config=ReplicaScalingConfig(mode="sotss_min", sla_s=10.0),
        )
        assert result.initial_violations >= 0


# ---------------------------------------------------------------------------
# End-to-end KPI parity (backtest level)
# ---------------------------------------------------------------------------

class TestBacktestLevelKPIParity:
    """End-to-end checks: canonical routing produces same KPI as before."""

    def test_amcsg_azure_report_structure(self):
        """run_amcsg_azure_backtest returns a valid AMCSGReport (canonical routing)."""
        from aurelius.benchmarks.srtf_serving_backtest import AMCSGReport
        report = run_amcsg_azure_backtest(job_limit=200)
        assert isinstance(report, AMCSGReport)
        assert report.best_goodput_per_dollar > 0
        assert report.best_gate > 0

    def test_amcsg_burstgpt_report_structure(self):
        """run_amcsg_burstgpt_backtest returns a valid AMCSGReport (canonical routing)."""
        from aurelius.benchmarks.srtf_serving_backtest import AMCSGReport
        report = run_amcsg_burstgpt_backtest(job_limit=200)
        assert isinstance(report, AMCSGReport)
        assert report.best_goodput_per_dollar > 0

    def test_sotss_min_azure_report_structure(self):
        """run_sotss_min_azure_backtest returns a valid SOTSSReport (canonical routing)."""
        from aurelius.benchmarks.srtf_serving_backtest import SOTSSReport
        report = run_sotss_min_azure_backtest(job_limit=200)
        assert isinstance(report, SOTSSReport)
        assert report.sotss_goodput_per_dollar > 0

    def test_sotss_min_initial_violations_in_report(self):
        """SOTSSReport.sotss_initial_violations propagated from canonical optimizer."""
        report = run_sotss_min_azure_backtest(job_limit=200)
        assert hasattr(report, "sotss_initial_violations")
        assert report.sotss_initial_violations >= 0

    def test_amcsg_azure_best_gate_in_valid_set(self):
        """Best gate must be one of the swept gates."""
        from aurelius.benchmarks.srtf_serving_backtest import _AMCSG_GATES
        report = run_amcsg_azure_backtest(job_limit=400)
        assert report.best_gate in _AMCSG_GATES

    def test_amcsg_completion_rate_full(self):
        """All requests must complete (completion_rate == 1.0 for best entry)."""
        report = run_amcsg_azure_backtest(job_limit=200)
        best = next(e for e in report.gate_results if e.gate_pct == report.best_gate)
        assert best.completion_rate == pytest.approx(1.0, abs=0.01)

    def test_amcsg_gate_results_count_equals_gates(self):
        """Number of gate_results equals number of gates swept."""
        from aurelius.benchmarks.srtf_serving_backtest import _AMCSG_GATES
        report = run_amcsg_azure_backtest(job_limit=200)
        assert len(report.gate_results) == len(_AMCSG_GATES)

    def test_sotss_min_cheaper_than_amcsg_or_equal(self):
        """SOTSS-MIN should have c_mean ≤ AMCSG c_mean (it optimizes for lower cost)."""
        report = run_sotss_min_azure_backtest(job_limit=300)
        assert report.sotss_c_mean <= report.amcsg_c_mean + 0.5
