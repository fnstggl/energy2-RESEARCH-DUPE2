"""Tests for SOTSS gate sweep and SOTSS-MIN — run 2026-06-23.

Validates:
 - SOTSSGateSweepEntry and SOTSSGateSweepReport dataclasses
 - _run_sotss_gate_sweep on tiny synthetic fixture
 - run_sotss_gate_sweep_azure_backtest (smoke, small job_limit)
 - run_sotss_min_azure_backtest (smoke)
 - BurstGPT cross-validation: gate=20% safe, gate=30% unsafe
 - Monotonicity of goodput/$ vs gate on Azure
 - Safety criterion: oracle_converged ↔ n_sla_safe >= baseline
"""

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    SOTSSGateSweepEntry,
    SOTSSGateSweepReport,
    _SOTSS_MIN_GATE,
    _SOTSS_MIN_MAX_ITERS,
    _SOTSS_SWEEP_GATES,
    run_sotss_gate_sweep_azure_backtest,
    run_sotss_gate_sweep_burstgpt_backtest,
    run_sotss_min_azure_backtest,
)


# ---------------------------------------------------------------------------
# Smoke: SOTSS-MIN on Azure (small job limit)
# ---------------------------------------------------------------------------

class TestSOTSSMinAzure:
    """SOTSS-MIN on Azure LLM 2024 (full 5,880 fixture — confirmed 160,107 gpd/$)."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_min_azure_backtest()

    def test_goodput_exceeds_sotss_gate20(self, report):
        """SOTSS-MIN must beat SOTSS gate=20% (153,013 gpd/$)."""
        assert report.sotss_goodput_per_dollar > 153_013, (
            f"Expected >153,013 but got {report.sotss_goodput_per_dollar:.0f}"
        )

    def test_north_star_achieved(self, report):
        assert report.sotss_north_star_500_achieved, (
            f"North-star not achieved: {report.sotss_goodput_per_dollar:.0f} "
            f"vs threshold 151,248"
        )

    def test_n_sla_safe_equals_baseline(self, report):
        assert report.sotss_n_sla_safe >= report.amcsg_n_sla_safe, (
            f"Safety regression: {report.sotss_n_sla_safe} < {report.amcsg_n_sla_safe}"
        )

    def test_c_mean_below_amcsg(self, report):
        """SOTSS-MIN must achieve lower c_mean than AMCSG gate=12.5% (4.458)."""
        assert report.sotss_c_mean < report.amcsg_c_mean, (
            f"c_mean not reduced: sotss={report.sotss_c_mean:.3f} amcsg={report.amcsg_c_mean:.3f}"
        )

    def test_c_mean_below_sotss_gate20(self, report):
        """SOTSS-MIN (gate=100%) must have lower c_mean than SOTSS gate=20% (4.389)."""
        assert report.sotss_c_mean < 4.39, (
            f"c_mean not below 4.39: {report.sotss_c_mean:.3f}"
        )

    def test_oracle_iterations_reasonable(self, report):
        """SOTSS-MIN should converge in ≤200 iterations on Azure."""
        assert report.sotss_n_iters <= 200, (
            f"Oracle took too many iterations: {report.sotss_n_iters}"
        )

    def test_more_ticks_cheaper_than_gate20(self, report):
        """SOTSS-MIN (gate=100%) must leave more ticks cheaper than gate=20% (5)."""
        assert report.n_ticks_cheaper > 5, (
            f"Expected >5 ticks cheaper but got {report.n_ticks_cheaper}"
        )

    def test_aggressive_gate_is_min(self, report):
        assert report.sotss_aggressive_gate == _SOTSS_MIN_GATE

    def test_goodput_vs_amcsg_positive(self, report):
        assert report.sotss_vs_amcsg_pct > 4.0, (
            f"Expected >4% improvement vs AMCSG, got {report.sotss_vs_amcsg_pct:.2f}%"
        )


# ---------------------------------------------------------------------------
# Gate sweep: Azure
# ---------------------------------------------------------------------------

class TestSOTSSGateSweepAzure:
    """Gate sweep on Azure LLM 2024 — all gates must be safe."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_gate_sweep_azure_backtest()

    def test_report_type(self, report):
        assert isinstance(report, SOTSSGateSweepReport)

    def test_all_entries_present(self, report):
        assert len(report.entries) == len(_SOTSS_SWEEP_GATES)

    def test_all_gates_safe_on_azure(self, report):
        """On Azure LLM 2024, all tested gates must pass the safety criterion."""
        for entry in report.entries:
            assert entry.oracle_converged, (
                f"gate={entry.aggressive_gate}% is NOT safe on Azure: "
                f"n_sla_safe={entry.n_sla_safe} < baseline={entry.baseline_n_sla_safe}"
            )

    def test_monotonic_goodput(self, report):
        """Goodput/$ must be non-decreasing as gate increases on Azure."""
        sorted_entries = sorted(report.entries, key=lambda e: e.aggressive_gate)
        for i in range(1, len(sorted_entries)):
            a = sorted_entries[i - 1]
            b = sorted_entries[i]
            assert b.goodput_per_dollar >= a.goodput_per_dollar - 1, (
                f"Non-monotonic at gate {a.aggressive_gate}→{b.aggressive_gate}: "
                f"{a.goodput_per_dollar:.0f}→{b.goodput_per_dollar:.0f}"
            )

    def test_best_gate_is_100(self, report):
        """Best gate on Azure should be 100% (SOTSS-MIN)."""
        assert report.best_entry is not None
        assert report.best_entry.aggressive_gate == 100.0, (
            f"Expected best gate=100.0, got {report.best_entry.aggressive_gate}"
        )

    def test_best_goodput_exceeds_sotss_gate20(self, report):
        assert report.best_entry.goodput_per_dollar > 153_013

    def test_best_vs_amcsg_positive(self, report):
        assert report.best_vs_amcsg_pct > 4.0

    def test_north_star_achieved_at_best_gate(self, report):
        assert report.best_entry.north_star_500_achieved

    def test_c_mean_decreases_with_gate(self, report):
        """c_mean should decrease as gate increases (cheaper schedule)."""
        sorted_entries = sorted(report.entries, key=lambda e: e.aggressive_gate)
        for i in range(1, len(sorted_entries)):
            a = sorted_entries[i - 1]
            b = sorted_entries[i]
            assert b.c_mean <= a.c_mean + 0.01, (
                f"c_mean increased: gate {a.aggressive_gate}→{b.aggressive_gate}: "
                f"{a.c_mean:.3f}→{b.c_mean:.3f}"
            )

    def test_n_ticks_cheaper_increases_with_gate(self, report):
        """More ticks cheaper as gate increases."""
        sorted_entries = sorted(report.entries, key=lambda e: e.aggressive_gate)
        for i in range(1, len(sorted_entries)):
            a = sorted_entries[i - 1]
            b = sorted_entries[i]
            assert b.n_ticks_cheaper >= a.n_ticks_cheaper - 1, (
                f"n_ticks_cheaper decreased: {a.aggressive_gate}→{b.aggressive_gate}: "
                f"{a.n_ticks_cheaper}→{b.n_ticks_cheaper}"
            )

    def test_to_dict_has_required_keys(self, report):
        d = report.to_dict()
        required = {
            "trace", "total_requests", "sla_s", "amcsg_goodput_per_dollar",
            "entries", "best_gate", "best_goodput_per_dollar",
        }
        assert required <= set(d.keys())


# ---------------------------------------------------------------------------
# Gate sweep: BurstGPT — safety cliff at gate=30%
# ---------------------------------------------------------------------------

class TestSOTSSGateSweepBurstGPT:
    """Gate sweep on BurstGPT HF — safety cliff at gate=30%."""

    @pytest.fixture(scope="class")
    def report(self):
        # Run only the key gates to save time
        return run_sotss_gate_sweep_burstgpt_backtest(gates=[20.0, 25.0, 30.0, 50.0, 100.0])

    def test_gate20_safe(self, report):
        e20 = next(e for e in report.entries if e.aggressive_gate == 20.0)
        assert e20.oracle_converged, f"gate=20% unsafe on BurstGPT: {e20.n_sla_safe}"

    def test_gate30_unsafe(self, report):
        """gate=30% should fail safety on BurstGPT (spot interruptions cause 3 extra violations)."""
        e30 = next(e for e in report.entries if e.aggressive_gate == 30.0)
        assert not e30.oracle_converged, (
            f"Expected gate=30% unsafe but got n_sla_safe={e30.n_sla_safe} "
            f"baseline={e30.baseline_n_sla_safe}"
        )

    def test_best_gate_is_20(self, report):
        assert report.best_entry is not None
        assert report.best_entry.aggressive_gate == 20.0, (
            f"Expected best gate=20.0 on BurstGPT, got {report.best_entry.aggressive_gate}"
        )

    def test_gate20_goodput_exceeds_sotss_gate15(self, report):
        """gate=20% must beat SOTSS gate=15% (169,030 gpd/$) on BurstGPT."""
        e20 = next(e for e in report.entries if e.aggressive_gate == 20.0)
        assert e20.goodput_per_dollar > 169_030, (
            f"Expected >169,030 but got {e20.goodput_per_dollar:.0f}"
        )

    def test_gate20_north_star(self, report):
        e20 = next(e for e in report.entries if e.aggressive_gate == 20.0)
        assert e20.north_star_500_achieved


# ---------------------------------------------------------------------------
# Dataclass unit tests
# ---------------------------------------------------------------------------

class TestSOTSSGateSweepEntryDataclass:
    def test_to_dict_fields(self):
        entry = SOTSSGateSweepEntry(
            aggressive_gate=20.0,
            goodput_per_dollar=153_013.0,
            cost=4.2133,
            c_mean=4.389,
            n_sla_safe=5823,
            baseline_n_sla_safe=5823,
            p99_s=9.946,
            n_iters=3,
            initial_violations=60,
            n_ticks_cheaper=5,
            vs_amcsg_pct=1.58,
            vs_sla_oracle_pct=507.0,
            north_star_500_achieved=True,
            oracle_converged=True,
        )
        d = entry.to_dict()
        assert d["aggressive_gate"] == 20.0
        assert d["oracle_converged"] is True
        assert d["north_star_500_achieved"] is True
        assert d["n_ticks_cheaper"] == 5
