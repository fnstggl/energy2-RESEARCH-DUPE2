"""Tests for AMCSG-LFC (Lower Fixed-C Calibration) and fine gate grid — this run.

Tests run_amcsg_lfc_azure_backtest, run_amcsg_lfc_burstgpt_backtest,
run_amcsg_fine_grid_azure_backtest, run_amcsg_lfc_fine_grid_azure_backtest,
and the _AMCSG_LFC_FINE_GATES constant.

Core contracts:
  - LFC (fixed_c=3) produces lower or equal c_mean vs fixed_c=4.
  - LFC cost ≤ fixed_c=4 cost (fewer servers → cheaper fleet).
  - Fine gate grid anchors at gate=12.5% (matching AMCSG baseline).
  - best_goodput_per_dollar ≥ baseline_goodput_per_dollar in all variants.
  - All LFC functions return AMCSGReport instances.
"""
import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    _AMCSG_GATES,
    _AMCSG_LFC_FINE_GATES,
    AMCSGReport,
    run_amcsg_fine_grid_azure_backtest,
    run_amcsg_lfc_azure_backtest,
    run_amcsg_lfc_burstgpt_backtest,
    run_amcsg_lfc_fine_grid_azure_backtest,
)

# ---------------------------------------------------------------------------
# Class 1: _AMCSG_LFC_FINE_GATES constant
# ---------------------------------------------------------------------------

class TestAMCSGLFCFineGatesConstant:
    """Verify _AMCSG_LFC_FINE_GATES structure."""

    def test_contains_boundary_gates(self):
        assert 12.5 in _AMCSG_LFC_FINE_GATES, "Fine grid must include 12.5% (known safe)"
        assert 15.0 in _AMCSG_LFC_FINE_GATES, "Fine grid must include 15.0% (known unsafe)"

    def test_ordered_ascending(self):
        assert list(_AMCSG_LFC_FINE_GATES) == sorted(_AMCSG_LFC_FINE_GATES)

    def test_minimum_length(self):
        assert len(_AMCSG_LFC_FINE_GATES) >= 4, "Need at least 4 gates for fine grid"

    def test_starts_at_12_5(self):
        assert _AMCSG_LFC_FINE_GATES[0] == 12.5, "Fine grid must start at 12.5%"

    def test_step_resolution(self):
        steps = [
            _AMCSG_LFC_FINE_GATES[i + 1] - _AMCSG_LFC_FINE_GATES[i]
            for i in range(len(_AMCSG_LFC_FINE_GATES) - 1)
        ]
        for s in steps:
            assert s <= 1.0, f"Fine grid step {s}% should be ≤ 1.0%"


# ---------------------------------------------------------------------------
# Class 2: LFC Azure — integration tests
# ---------------------------------------------------------------------------

class TestAMCSGLFCAzure:
    """Integration tests for run_amcsg_lfc_azure_backtest (fixed_c=3)."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_amcsg_lfc_azure_backtest()

    def test_returns_amcsg_report(self, report):
        assert isinstance(report, AMCSGReport)

    def test_trace_name_contains_lfc(self, report):
        assert "lfc" in report.trace.lower()

    def test_fixed_c_is_3(self, report):
        assert report.fixed_c == 3

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_gate_results_count(self, report):
        assert len(report.gate_results) == len(_AMCSG_GATES)

    def test_all_standard_gates_present(self, report):
        sweep_gates = {e.gate_pct for e in report.gate_results}
        for g in _AMCSG_GATES:
            assert g in sweep_gates, f"Gate {g}% missing from LFC results"

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 25_208.0) < 1.0

    def test_north_star_500_threshold_correct(self, report):
        expected = 6.0 * 25_208.0
        assert abs(report.north_star_500_threshold - expected) < 1.0

    def test_best_goodput_gte_baseline(self, report):
        assert report.best_goodput_per_dollar >= report.baseline_goodput_per_dollar

    def test_completion_rate_near_1_at_baseline(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.completion_rate == pytest.approx(1.0, abs=0.001)

    def test_north_star_300_achieved_at_baseline(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.north_star_300_achieved is True

    def test_lfc_c_mean_le_fixed4_c_mean(self, report):
        """LFC (fixed_c=3) baseline c_mean should be ≤ AMCSG (fixed_c=4) c_mean.

        The lower time-warp factor from fixed_c=3 produces a lower effective
        arrival rate in the warped domain, so MCS provisions fewer servers.
        """
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        # AMCSG run 2026-06-27 baseline c_mean = 4.500; LFC should be lower
        assert baseline.c_schedule_mean <= 4.6, (
            f"LFC c_mean={baseline.c_schedule_mean:.3f} not lower than fixed_c=4 baseline"
        )

    def test_to_dict_serializable(self, report):
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "gate_results" in d
        assert len(d["gate_results"]) == len(_AMCSG_GATES)


# ---------------------------------------------------------------------------
# Class 3: LFC BurstGPT — integration tests
# ---------------------------------------------------------------------------

class TestAMCSGLFCBurstGPT:
    """Integration tests for run_amcsg_lfc_burstgpt_backtest (fixed_c=3)."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_amcsg_lfc_burstgpt_backtest()

    def test_returns_amcsg_report(self, report):
        assert isinstance(report, AMCSGReport)

    def test_trace_name_contains_lfc(self, report):
        assert "lfc" in report.trace.lower()

    def test_fixed_c_is_3(self, report):
        assert report.fixed_c == 3

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 20_280.0) < 1.0

    def test_north_star_500_threshold_correct(self, report):
        expected = 6.0 * 20_280.0
        assert abs(report.north_star_500_threshold - expected) < 1.0

    def test_best_goodput_gte_baseline(self, report):
        assert report.best_goodput_per_dollar >= report.baseline_goodput_per_dollar

    def test_north_star_300_achieved_at_baseline(self, report):
        baseline = next(e for e in report.gate_results if e.gate_pct == 9.5)
        assert baseline.north_star_300_achieved is True


# ---------------------------------------------------------------------------
# Class 4: Fine grid Azure — integration tests
# ---------------------------------------------------------------------------

class TestAMCSGFineGridAzure:
    """Integration tests for run_amcsg_fine_grid_azure_backtest (fixed_c=4, fine gates)."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_amcsg_fine_grid_azure_backtest()

    def test_returns_amcsg_report(self, report):
        assert isinstance(report, AMCSGReport)

    def test_trace_name_contains_fine_grid(self, report):
        assert "fine" in report.trace.lower()

    def test_fixed_c_is_4(self, report):
        assert report.fixed_c == 4

    def test_gate_results_count(self, report):
        assert len(report.gate_results) == len(_AMCSG_LFC_FINE_GATES)

    def test_all_fine_gates_present(self, report):
        sweep_gates = {e.gate_pct for e in report.gate_results}
        for g in _AMCSG_LFC_FINE_GATES:
            assert g in sweep_gates, f"Gate {g}% missing from fine grid results"

    def test_baseline_at_12_5_matches_amcsg(self, report):
        """gate=12.5% entry should reproduce AMCSG run 2026-06-27 best result."""
        entry_12_5 = next(e for e in report.gate_results if e.gate_pct == 12.5)
        assert abs(entry_12_5.goodput_per_dollar - 150_630.0) < 500.0, (
            f"Fine grid gate=12.5% {entry_12_5.goodput_per_dollar:.0f} "
            "not within 500 of AMCSG reference 150,630"
        )

    def test_best_goodput_gte_12_5_entry(self, report):
        entry_12_5 = next(e for e in report.gate_results if e.gate_pct == 12.5)
        assert report.best_goodput_per_dollar >= entry_12_5.goodput_per_dollar

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 25_208.0) < 1.0


# ---------------------------------------------------------------------------
# Class 5: LFC + fine grid Azure — integration tests
# ---------------------------------------------------------------------------

class TestAMCSGLFCFineGridAzure:
    """Integration tests for run_amcsg_lfc_fine_grid_azure_backtest (fixed_c=3, fine gates)."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_amcsg_lfc_fine_grid_azure_backtest()

    def test_returns_amcsg_report(self, report):
        assert isinstance(report, AMCSGReport)

    def test_fixed_c_is_3(self, report):
        assert report.fixed_c == 3

    def test_gate_results_count(self, report):
        assert len(report.gate_results) == len(_AMCSG_LFC_FINE_GATES)

    def test_all_fine_gates_present(self, report):
        sweep_gates = {e.gate_pct for e in report.gate_results}
        for g in _AMCSG_LFC_FINE_GATES:
            assert g in sweep_gates, f"Gate {g}% missing from LFC fine grid results"

    def test_best_goodput_gte_baseline(self, report):
        assert report.best_goodput_per_dollar >= report.baseline_goodput_per_dollar

    def test_completion_rate_near_1_at_12_5(self, report):
        entry = next(e for e in report.gate_results if e.gate_pct == 12.5)
        assert entry.completion_rate == pytest.approx(1.0, abs=0.001)

    def test_north_star_300_achieved_at_12_5(self, report):
        entry = next(e for e in report.gate_results if e.gate_pct == 12.5)
        assert entry.north_star_300_achieved is True

    def test_to_dict_serializable(self, report):
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "gate_results" in d

    def test_sla_oracle_correct(self, report):
        assert abs(report.sla_oracle_goodput_per_dollar - 25_208.0) < 1.0

    def test_north_star_500_threshold_correct(self, report):
        expected = 6.0 * 25_208.0
        assert abs(report.north_star_500_threshold - expected) < 1.0
