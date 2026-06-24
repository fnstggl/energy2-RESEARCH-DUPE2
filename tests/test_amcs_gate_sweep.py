"""Tests for AMCSG (Adaptive MCS Gate Sweep) policy — run 2026-06-27.

Tests _run_amcsg_backtest, AMCSGEntry, AMCSGReport,
run_amcsg_azure_backtest, run_amcsg_burstgpt_backtest.

Core contract:
  - gate=9.5% entry must reproduce GSF(0.95) baseline (run 2026-06-26).
  - Higher gates may reduce c_schedule_mean (Erlang-C conservatism effect).
  - best_goodput_per_dollar ≥ baseline_goodput_per_dollar.
  - best_north_star_500_achieved documents whether +500% was reached.
"""
import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    _AMCSG_GATES,
    GPU_HOUR_USD,
    AMCSGEntry,
    AMCSGReport,
    run_amcsg_azure_backtest,
    run_amcsg_burstgpt_backtest,
)

# ---------------------------------------------------------------------------
# Class 1: AMCSGEntry — dataclass correctness
# ---------------------------------------------------------------------------

class TestAMCSGEntryDataclass:
    """Unit tests for the AMCSGEntry dataclass."""

    def _make_entry(self, gate_pct=9.5, gp=149_235.0, cost=4.32,
                    c_mean=4.5, n_ticks=72, p99=9.946, n_sla_safe=5880):
        return AMCSGEntry(
            gate_pct=gate_pct,
            c_schedule_mean=c_mean,
            c_schedule_min=1,
            c_schedule_max=8,
            n_ticks=n_ticks,
            cost=cost,
            cost_vs_baseline_pct=0.0,
            goodput_per_dollar=gp,
            goodput_vs_baseline_pct=0.0,
            goodput_vs_sla_oracle_pct=(gp - 25_208.0) / 25_208.0 * 100.0,
            north_star_300_achieved=gp >= 100_832.0,
            north_star_500_achieved=gp >= 151_248.0,
            completion_rate=1.0,
            p99_s=p99,
            n_sla_safe=n_sla_safe,
        )

    def test_to_dict_has_required_keys(self):
        e = self._make_entry()
        d = e.to_dict()
        for key in [
            "gate_pct", "c_schedule_mean", "c_schedule_min", "c_schedule_max",
            "n_ticks", "cost", "cost_vs_baseline_pct", "goodput_per_dollar",
            "goodput_vs_baseline_pct", "goodput_vs_sla_oracle_pct",
            "north_star_300_achieved", "north_star_500_achieved",
            "completion_rate", "p99_s", "n_sla_safe",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_north_star_300_threshold(self):
        e_above = self._make_entry(gp=149_235.0)
        e_below = self._make_entry(gp=80_000.0)
        assert e_above.north_star_300_achieved is True
        assert e_below.north_star_300_achieved is False

    def test_north_star_500_threshold(self):
        e_above = self._make_entry(gp=152_000.0)
        e_below = self._make_entry(gp=149_235.0)
        assert e_above.north_star_500_achieved is True
        assert e_below.north_star_500_achieved is False

    def test_goodput_vs_oracle_formula(self):
        oracle = 25_208.0
        gp = 149_235.0
        e = self._make_entry(gp=gp)
        expected_pct = (gp - oracle) / oracle * 100.0
        assert abs(e.goodput_vs_sla_oracle_pct - expected_pct) < 0.01


# ---------------------------------------------------------------------------
# Class 2: AMCSGReport — structure and invariants
# ---------------------------------------------------------------------------

class TestAMCSGReportStructure:
    """Tests for AMCSGReport dataclass structure."""

    def _make_entry(self, gate_pct, gp):
        return AMCSGEntry(
            gate_pct=gate_pct,
            c_schedule_mean=4.5 - (gate_pct - 9.5) * 0.05,
            c_schedule_min=1,
            c_schedule_max=8,
            n_ticks=72,
            cost=4.32 - (gate_pct - 9.5) * 0.02,
            cost_vs_baseline_pct=-(gate_pct - 9.5) * 0.5,
            goodput_per_dollar=gp,
            goodput_vs_baseline_pct=(gp - 149_235.0) / 149_235.0 * 100.0,
            goodput_vs_sla_oracle_pct=(gp - 25_208.0) / 25_208.0 * 100.0,
            north_star_300_achieved=gp >= 100_832.0,
            north_star_500_achieved=gp >= 151_248.0,
            completion_rate=1.0,
            p99_s=9.946,
            n_sla_safe=5880,
        )

    def _make_report(self):
        entries = [
            self._make_entry(9.5, 149_235.0),
            self._make_entry(11.0, 151_500.0),
            self._make_entry(12.5, 153_000.0),
        ]
        return AMCSGReport(
            trace="test_trace",
            total_requests=5880,
            fixed_c=4,
            target_rho=0.85,
            sla_s=10.0,
            tick_seconds=60.0,
            rng_seed=42,
            spot_price_usd_hr=0.80,
            demand_price_usd_hr=GPU_HOUR_USD,
            p_interrupt_hourly=0.10,
            zfhc_threshold=8,
            sla_oracle_goodput_per_dollar=25_208.0,
            north_star_300_threshold=100_832.0,
            north_star_500_threshold=151_248.0,
            baseline_gate=9.5,
            baseline_goodput_per_dollar=149_235.0,
            baseline_cost=4.32,
            baseline_c_schedule_mean=4.5,
            gate_results=entries,
            best_gate=12.5,
            best_goodput_per_dollar=153_000.0,
            best_vs_baseline_pct=2.52,
            best_vs_sla_oracle_pct=507.0,
            best_north_star_500_achieved=True,
            max_safe_gate=12.5,
            erlang_c_margin_pct=3.0,
        )

    def test_to_dict_has_required_keys(self):
        report = self._make_report()
        d = report.to_dict()
        for key in [
            "trace", "total_requests", "fixed_c", "target_rho", "sla_s",
            "baseline_gate", "baseline_goodput_per_dollar", "baseline_cost",
            "baseline_c_schedule_mean", "gate_results", "best_gate",
            "best_goodput_per_dollar", "best_vs_baseline_pct",
            "best_vs_sla_oracle_pct", "best_north_star_500_achieved",
            "max_safe_gate", "erlang_c_margin_pct",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_gate_results_are_serialized(self):
        report = self._make_report()
        d = report.to_dict()
        assert isinstance(d["gate_results"], list)
        assert len(d["gate_results"]) == 3

    def test_erlang_c_margin_is_max_safe_minus_baseline(self):
        report = self._make_report()
        assert abs(report.erlang_c_margin_pct - (report.max_safe_gate - 9.5)) < 0.001


# ---------------------------------------------------------------------------
# Class 3: Constants
# ---------------------------------------------------------------------------

class TestAMCSGConstants:
    """Verify _AMCSG_GATES constant."""

    def test_gates_contains_baseline(self):
        assert 9.5 in _AMCSG_GATES, "Baseline gate 9.5% must be in sweep"

    def test_gates_are_ordered(self):
        assert list(_AMCSG_GATES) == sorted(_AMCSG_GATES), "Gates must be sorted ascending"

    def test_gates_min_max(self):
        assert min(_AMCSG_GATES) <= 9.5
        assert max(_AMCSG_GATES) >= 15.0

    def test_gates_length(self):
        assert len(_AMCSG_GATES) >= 4, "Need at least 4 gate values for sweep"


# ---------------------------------------------------------------------------
# Class 4: Azure backtest — integration tests
# ---------------------------------------------------------------------------

class TestRunAmcsgAzureBacktest:
    """Integration tests for run_amcsg_azure_backtest."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_amcsg_azure_backtest()

    def test_returns_amcsg_report(self, report):
        assert isinstance(report, AMCSGReport)

    def test_trace_name(self, report):
        assert "azure" in report.trace.lower()

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_gate_results_count(self, report):
        assert len(report.gate_results) == len(_AMCSG_GATES)

    def test_all_gates_present(self, report):
        sweep_gates = {e.gate_pct for e in report.gate_results}
        for g in _AMCSG_GATES:
            assert g in sweep_gates, f"Gate {g}% missing from results"

    def test_baseline_reproduces_gsf_result(self, report):
        """gate=9.5% must reproduce GSF(0.95) baseline from run 2026-06-26."""
        assert abs(report.baseline_goodput_per_dollar - 149_235.0) < 500.0, (
            f"Baseline {report.baseline_goodput_per_dollar:.0f} "
            "not within 500 of GSF reference 149,235"
        )

    def test_baseline_cost_within_tolerance(self, report):
        assert abs(report.baseline_cost - 4.32) < 0.05, (
            f"Baseline cost {report.baseline_cost:.4f} not near $4.32"
        )

    def test_baseline_c_schedule_mean(self, report):
        assert abs(report.baseline_c_schedule_mean - 4.5) < 0.5, (
            f"Baseline c_mean={report.baseline_c_schedule_mean} not near 4.5"
        )

    def test_baseline_completion_rate(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.completion_rate == pytest.approx(1.0, abs=0.001)

    def test_north_star_300_achieved_at_baseline(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.north_star_300_achieved is True

    def test_best_goodput_per_dollar_gte_baseline(self, report):
        assert report.best_goodput_per_dollar >= report.baseline_goodput_per_dollar

    def test_best_gate_is_in_sweep(self, report):
        assert report.best_gate in _AMCSG_GATES

    def test_max_safe_gate_gte_baseline(self, report):
        assert report.max_safe_gate >= 9.5

    def test_gate_results_monotone_cost_nondecreasing_at_high_gate(self, report):
        """Higher gate should yield c_schedule_mean <= lower gate (or same)."""
        sorted_entries = sorted(report.gate_results, key=lambda e: e.gate_pct)
        for i in range(len(sorted_entries) - 1):
            # c_mean should be non-increasing as gate increases
            assert sorted_entries[i + 1].c_schedule_mean <= (
                sorted_entries[i].c_schedule_mean + 0.5
            ), (
                f"c_mean not non-increasing at gate {sorted_entries[i+1].gate_pct}%"
            )

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 25_208.0) < 1.0

    def test_north_star_500_threshold_correct(self, report):
        expected = 6.0 * 25_208.0
        assert abs(report.north_star_500_threshold - expected) < 1.0


# ---------------------------------------------------------------------------
# Class 5: BurstGPT backtest — integration tests
# ---------------------------------------------------------------------------

class TestRunAmcsgBurstgptBacktest:
    """Integration tests for run_amcsg_burstgpt_backtest."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_amcsg_burstgpt_backtest()

    def test_returns_amcsg_report(self, report):
        assert isinstance(report, AMCSGReport)

    def test_trace_name(self, report):
        assert "burstgpt" in report.trace.lower()

    def test_gate_results_count(self, report):
        assert len(report.gate_results) == len(_AMCSG_GATES)

    def test_baseline_reproduces_gsf_result(self, report):
        """gate=9.5% must reproduce GSF(0.95) BurstGPT result from run 2026-06-26."""
        assert abs(report.baseline_goodput_per_dollar - 167_767.0) < 500.0, (
            f"Baseline {report.baseline_goodput_per_dollar:.0f} "
            "not within 500 of GSF reference 167,767"
        )

    def test_baseline_completion_rate(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.completion_rate == pytest.approx(1.0, abs=0.001)

    def test_north_star_300_achieved_at_baseline(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.north_star_300_achieved is True

    def test_best_goodput_per_dollar_gte_baseline(self, report):
        assert report.best_goodput_per_dollar >= report.baseline_goodput_per_dollar

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 20_280.0) < 1.0

    def test_north_star_500_threshold_correct(self, report):
        expected = 6.0 * 20_280.0  # 121,680
        assert abs(report.north_star_500_threshold - expected) < 1.0

    def test_erlang_c_margin_nonneg(self, report):
        assert report.erlang_c_margin_pct >= 0.0
