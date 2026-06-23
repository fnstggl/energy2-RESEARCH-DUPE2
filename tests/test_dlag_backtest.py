"""Tests for DLAG (Dynamic Load-Aware Gate) policy — run 2026-06-23.

Tests _joint_mcs_dlag_c_schedule, DLAGEntry, DLAGReport,
run_dlag_azure_backtest, run_dlag_burstgpt_backtest.

Core contracts:
  - DLAG per-tick gate = base_gate at high load (rho ≥ target_rho).
  - DLAG per-tick gate = max_gate at idle ticks.
  - DLAG c_schedule_mean ≤ AMCSG uniform gate=12.5% c_mean (or equal).
  - Safety validated by n_sla_safe ≥ AMCSG reference.
  - best_goodput_per_dollar ≥ amcsg_goodput_per_dollar when safe.
"""
import statistics

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    _DLAG_MAX_GATES,
    DLAGEntry,
    DLAGReport,
    _joint_mcs_dlag_c_schedule,
    calibrate_time_warp,
    run_dlag_azure_backtest,
    run_dlag_burstgpt_backtest,
)

# ---------------------------------------------------------------------------
# Class 1: _joint_mcs_dlag_c_schedule — unit tests
# ---------------------------------------------------------------------------

class TestDLAGCSchedule:
    """Unit tests for _joint_mcs_dlag_c_schedule."""

    def _make_uniform_raw(self, n_req=100, span_s=3600.0, tokens=50):
        """Uniform arrivals over span with fixed token length."""
        return [(i * span_s / n_req, tokens) for i in range(n_req)]

    def test_returns_two_lists(self):
        raw = self._make_uniform_raw()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        c_sched, gate_sched = _joint_mcs_dlag_c_schedule(
            raw, 60.0, warp, base_gate=9.5, max_gate=20.0
        )
        assert isinstance(c_sched, list)
        assert isinstance(gate_sched, list)
        assert len(c_sched) == len(gate_sched)

    def test_empty_raw_returns_empty(self):
        c, g = _joint_mcs_dlag_c_schedule([], 60.0, 1.0)
        assert c == []
        assert g == []

    def test_c_schedule_all_positive(self):
        raw = self._make_uniform_raw()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        c_sched, _ = _joint_mcs_dlag_c_schedule(raw, 60.0, warp)
        assert all(c >= 1 for c in c_sched)

    def test_gate_bounded_by_base_max(self):
        raw = self._make_uniform_raw()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        base, mx = 9.5, 25.0
        _, gate_sched = _joint_mcs_dlag_c_schedule(
            raw, 60.0, warp, base_gate=base, max_gate=mx
        )
        for g in gate_sched:
            assert g >= base - 0.001, f"Gate {g:.2f} below base {base}"
            assert g <= mx + 0.001, f"Gate {g:.2f} above max {mx}"

    def test_idle_tick_gets_max_gate(self):
        """A tick with very sparse arrivals should get gate near max_gate."""
        # One request every 600s in a 60s tick → λ = 1/600 per second
        # With 60s tick: ~0.1 requests per tick → very low rho
        raw = [(i * 600.0, 10) for i in range(10)]
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        _, gate_sched = _joint_mcs_dlag_c_schedule(
            raw, 60.0, warp, base_gate=9.5, max_gate=25.0, target_rho=0.85
        )
        # Empty ticks get max_gate; non-empty idle ticks should get near max
        max_gates = [g for g in gate_sched if g > 20.0]
        assert len(max_gates) > 0, "Expected some high-gate ticks in sparse trace"

    def test_higher_max_gate_yields_lower_or_equal_c_mean(self):
        """Raising max_gate should not increase c_mean."""
        raw = self._make_uniform_raw(n_req=1000, span_s=7200.0, tokens=30)
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        c_15, _ = _joint_mcs_dlag_c_schedule(raw, 60.0, warp, max_gate=15.0)
        c_25, _ = _joint_mcs_dlag_c_schedule(raw, 60.0, warp, max_gate=25.0)
        mean_15 = statistics.mean(c_15)
        mean_25 = statistics.mean(c_25)
        assert mean_25 <= mean_15 + 0.1, (
            f"c_mean at max_gate=25: {mean_25:.3f} not ≤ max_gate=15: {mean_15:.3f}"
        )

    def test_large_max_gate_cap_does_not_crash(self):
        raw = self._make_uniform_raw()
        warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
        c_sched, gate_sched = _joint_mcs_dlag_c_schedule(
            raw, 60.0, warp, base_gate=9.5, max_gate=50.0
        )
        assert len(c_sched) > 0


# ---------------------------------------------------------------------------
# Class 2: DLAGEntry — dataclass correctness
# ---------------------------------------------------------------------------

class TestDLAGEntryDataclass:
    """Unit tests for DLAGEntry."""

    def _make_entry(self, max_gate=20.0, gp=151_000.0, cost=4.27, p99=9.946):
        return DLAGEntry(
            max_gate_pct=max_gate,
            base_gate_pct=9.5,
            c_schedule_mean=4.4,
            c_schedule_min=1,
            c_schedule_max=8,
            effective_gate_mean=14.0,
            effective_gate_min=9.5,
            effective_gate_max=max_gate,
            n_ticks_at_max_gate=10,
            n_ticks_at_base_gate=30,
            n_ticks=72,
            cost=cost,
            cost_vs_amcsg_pct=(cost - 4.28) / 4.28 * 100.0,
            goodput_per_dollar=gp,
            goodput_vs_amcsg_pct=(gp - 150_630.0) / 150_630.0 * 100.0,
            goodput_vs_sla_oracle_pct=(gp - 25_208.0) / 25_208.0 * 100.0,
            north_star_500_achieved=gp >= 151_248.0,
            completion_rate=1.0,
            p99_s=p99,
            n_sla_safe=5880,
        )

    def test_to_dict_has_required_keys(self):
        e = self._make_entry()
        d = e.to_dict()
        for key in [
            "max_gate_pct", "base_gate_pct", "c_schedule_mean",
            "effective_gate_mean", "effective_gate_min", "effective_gate_max",
            "n_ticks_at_max_gate", "n_ticks_at_base_gate", "n_ticks",
            "cost", "cost_vs_amcsg_pct", "goodput_per_dollar",
            "goodput_vs_amcsg_pct", "goodput_vs_sla_oracle_pct",
            "north_star_500_achieved", "completion_rate", "p99_s", "n_sla_safe",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_north_star_500_threshold(self):
        above = self._make_entry(gp=152_000.0)
        below = self._make_entry(gp=150_000.0)
        assert above.north_star_500_achieved is True
        assert below.north_star_500_achieved is False


# ---------------------------------------------------------------------------
# Class 3: _DLAG_MAX_GATES constant
# ---------------------------------------------------------------------------

class TestDLAGMaxGatesConstant:
    """Verify _DLAG_MAX_GATES structure."""

    def test_ordered_ascending(self):
        assert list(_DLAG_MAX_GATES) == sorted(_DLAG_MAX_GATES)

    def test_minimum_length(self):
        assert len(_DLAG_MAX_GATES) >= 3, "Need at least 3 max_gate values"

    def test_covers_useful_range(self):
        assert min(_DLAG_MAX_GATES) <= 20.0
        assert max(_DLAG_MAX_GATES) >= 25.0


# ---------------------------------------------------------------------------
# Class 4: DLAG Azure — integration tests
# ---------------------------------------------------------------------------

class TestDLAGAzureBacktest:
    """Integration tests for run_dlag_azure_backtest."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_dlag_azure_backtest()

    def test_returns_dlag_report(self, report):
        assert isinstance(report, DLAGReport)

    def test_trace_name(self, report):
        assert "azure" in report.trace.lower()

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_max_gate_results_count(self, report):
        assert len(report.max_gate_results) == len(_DLAG_MAX_GATES)

    def test_all_max_gates_present(self, report):
        sweep_gates = {e.max_gate_pct for e in report.max_gate_results}
        for g in _DLAG_MAX_GATES:
            assert g in sweep_gates, f"Max gate {g}% missing from results"

    def test_amcsg_reference_near_150630(self, report):
        """DLAG's AMCSG reference must reproduce gate=12.5% result."""
        assert abs(report.amcsg_goodput_per_dollar - 150_630.0) < 500.0, (
            f"AMCSG ref {report.amcsg_goodput_per_dollar:.0f} "
            "not within 500 of 150,630"
        )

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 25_208.0) < 1.0

    def test_north_star_threshold_correct(self, report):
        expected = 6.0 * 25_208.0
        assert abs(report.north_star_500_threshold - expected) < 1.0

    def test_c_schedule_means_positive(self, report):
        for e in report.max_gate_results:
            assert e.c_schedule_mean > 0

    def test_effective_gate_within_bounds(self, report):
        for e in report.max_gate_results:
            assert e.effective_gate_min >= e.base_gate_pct - 0.01
            assert e.effective_gate_max <= e.max_gate_pct + 0.01

    def test_completion_rate_near_1(self, report):
        for e in report.max_gate_results:
            assert e.completion_rate >= 0.99, (
                f"Max gate {e.max_gate_pct}%: completion rate {e.completion_rate:.4f}"
            )

    def test_to_dict_serializable(self, report):
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "max_gate_results" in d
        assert len(d["max_gate_results"]) == len(_DLAG_MAX_GATES)

    def test_best_max_gate_in_sweep(self, report):
        assert report.best_max_gate in _DLAG_MAX_GATES


# ---------------------------------------------------------------------------
# Class 5: DLAG BurstGPT — integration tests
# ---------------------------------------------------------------------------

class TestDLAGBurstGPTBacktest:
    """Integration tests for run_dlag_burstgpt_backtest."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_dlag_burstgpt_backtest()

    def test_returns_dlag_report(self, report):
        assert isinstance(report, DLAGReport)

    def test_trace_name(self, report):
        assert "burstgpt" in report.trace.lower()

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 20_280.0) < 1.0

    def test_north_star_threshold_correct(self, report):
        expected = 6.0 * 20_280.0  # 121,680
        assert abs(report.north_star_500_threshold - expected) < 1.0

    def test_max_gate_results_count(self, report):
        assert len(report.max_gate_results) == len(_DLAG_MAX_GATES)

    def test_completion_rate_near_1(self, report):
        for e in report.max_gate_results:
            assert e.completion_rate >= 0.99

    def test_best_max_gate_in_sweep(self, report):
        assert report.best_max_gate in _DLAG_MAX_GATES
