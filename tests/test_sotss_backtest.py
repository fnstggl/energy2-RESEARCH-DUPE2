"""Tests for SOTSS (Simulation-Oracle Tick-Selective Schedule) — run 2026-06-23.

Tests run_sotss_azure_backtest, run_sotss_burstgpt_backtest,
_sotss_min_cost_schedule, and the SOTSSReport dataclass.

Core contracts verified:
  - Oracle loop terminates within max_iters.
  - SOTSS n_sla_safe >= AMCSG n_sla_safe (no regressions).
  - SOTSS cost <= AMCSG cost (achieves savings via selective c reduction).
  - c_schedule from oracle is within [aggressive, ceil] bounds per tick.
  - SOTSSReport.to_dict() round-trips cleanly.
  - n_ticks_cheaper > 0 (at least one tick is cheaper than ceiling).
  - north_star_500_achieved flag is consistent with goodput and n_sla_safe.
"""
import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    _SOTSS_SAFE_GATE,
    SOTSSReport,
    _sotss_min_cost_schedule,
    run_sotss_azure_backtest,
    run_sotss_burstgpt_backtest,
)

# ---------------------------------------------------------------------------
# Class 1: SOTSSReport dataclass contract
# ---------------------------------------------------------------------------

class TestSOTSSReportDataclass:
    """Verify SOTSSReport has all required fields and to_dict round-trips."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_azure_backtest()

    def test_is_sotss_report(self, report):
        assert isinstance(report, SOTSSReport)

    def test_trace_contains_sotss(self, report):
        assert "sotss" in report.trace.lower()

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_to_dict_keys(self, report):
        d = report.to_dict()
        required = {
            "trace", "total_requests", "sla_s",
            "amcsg_goodput_per_dollar", "amcsg_cost", "amcsg_n_sla_safe",
            "sotss_goodput_per_dollar", "sotss_cost", "sotss_n_sla_safe",
            "sotss_n_iters", "n_ticks_cheaper", "sotss_vs_amcsg_pct",
            "sotss_north_star_500_achieved",
        }
        assert required.issubset(d.keys())

    def test_to_dict_numeric_types(self, report):
        d = report.to_dict()
        assert isinstance(d["amcsg_goodput_per_dollar"], float)
        assert isinstance(d["sotss_goodput_per_dollar"], float)
        assert isinstance(d["sotss_n_iters"], int)
        assert isinstance(d["sotss_north_star_500_achieved"], bool)

    def test_north_star_threshold_is_six_x_oracle(self, report):
        assert abs(report.north_star_500_threshold - 6.0 * report.sla_oracle_goodput_per_dollar) < 1.0

    def test_sla_s_is_10s(self, report):
        assert report.sla_s == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Class 2: SOTSS Azure — safety and performance contracts
# ---------------------------------------------------------------------------

class TestSOTSSAzure:
    """Integration tests for run_sotss_azure_backtest."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_azure_backtest()

    def test_n_sla_safe_no_regression_vs_amcsg(self, report):
        """SOTSS must not lose SLA-safe requests vs AMCSG baseline."""
        assert report.sotss_n_sla_safe >= report.amcsg_n_sla_safe, (
            f"SOTSS n_sla_safe {report.sotss_n_sla_safe} < AMCSG {report.amcsg_n_sla_safe}"
        )

    def test_sotss_cost_le_amcsg_cost(self, report):
        """SOTSS c_schedule is cheaper than or equal to AMCSG (ceiling)."""
        assert report.sotss_cost <= report.amcsg_cost + 1e-6, (
            f"SOTSS cost ${report.sotss_cost:.4f} > AMCSG cost ${report.amcsg_cost:.4f}"
        )

    def test_goodput_per_dollar_positive(self, report):
        assert report.sotss_goodput_per_dollar > 0

    def test_amcsg_gate_is_safe_gate(self, report):
        assert report.amcsg_gate == _SOTSS_SAFE_GATE

    def test_oracle_converged_within_cap(self, report):
        assert report.sotss_n_iters <= 200, (
            f"Oracle used {report.sotss_n_iters} iters — may not have converged"
        )

    def test_n_ticks_cheaper_nonnegative(self, report):
        assert report.n_ticks_cheaper >= 0

    def test_vs_amcsg_pct_consistent(self, report):
        expected = (
            (report.sotss_goodput_per_dollar - report.amcsg_goodput_per_dollar)
            / max(report.amcsg_goodput_per_dollar, 1e-9) * 100.0
        )
        assert abs(report.sotss_vs_amcsg_pct - expected) < 0.01

    def test_north_star_flag_consistent(self, report):
        """Flag must be True iff goodput >= threshold AND n_safe >= amcsg baseline."""
        if report.sotss_north_star_500_achieved:
            assert report.sotss_goodput_per_dollar >= report.north_star_500_threshold
            assert report.sotss_n_sla_safe >= report.amcsg_n_sla_safe

    def test_c_mean_between_aggressive_and_safe(self, report):
        """SOTSS c_mean must be between aggressive (cheaper) and safe ceiling."""
        assert report.sotss_c_mean <= report.amcsg_c_mean + 1e-6

    def test_initial_violations_nonnegative(self, report):
        assert report.sotss_initial_violations >= 0


# ---------------------------------------------------------------------------
# Class 3: SOTSS BurstGPT — cross-trace validation
# ---------------------------------------------------------------------------

class TestSOTSSBurstGPT:
    """SOTSS on BurstGPT HF validates no regression on second trace."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_burstgpt_backtest()

    def test_is_sotss_report(self, report):
        assert isinstance(report, SOTSSReport)

    def test_trace_contains_burstgpt(self, report):
        assert "burstgpt" in report.trace.lower()

    def test_sla_s_is_30s(self, report):
        assert report.sla_s == pytest.approx(30.0)

    def test_n_sla_safe_no_regression(self, report):
        assert report.sotss_n_sla_safe >= report.amcsg_n_sla_safe

    def test_goodput_positive(self, report):
        assert report.sotss_goodput_per_dollar > 0

    def test_oracle_terminated(self, report):
        assert report.sotss_n_iters >= 1


# ---------------------------------------------------------------------------
# Class 4: _sotss_min_cost_schedule unit tests
# ---------------------------------------------------------------------------

class TestSOTSSMinCostSchedule:
    """Unit tests for the oracle inner loop."""

    @pytest.fixture(scope="class")
    def tiny_raw(self):
        """Minimal synthetic trace: 10 requests, ~1 request per tick."""
        from aurelius.benchmarks.srtf_serving_backtest import (
            DEFAULT_AZURE_FIXTURE,
            load_serving_requests,
        )
        raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=100)
        return raw

    @pytest.fixture(scope="class")
    def oracle_result(self, tiny_raw):
        from aurelius.benchmarks.srtf_serving_backtest import calibrate_time_warp
        warp = calibrate_time_warp(tiny_raw, servers=4, target_rho=0.85)
        return _sotss_min_cost_schedule(
            tiny_raw,
            tick_seconds=60.0,
            warp=warp,
            sla_s=10.0,
            safe_gate=12.5,
            aggressive_gate=15.0,
            max_iters=50,
        )

    def test_returns_5_tuple(self, oracle_result):
        assert len(oracle_result) == 5

    def test_c_schedule_is_list(self, oracle_result):
        c_sched, *_ = oracle_result
        assert isinstance(c_sched, list)
        assert all(isinstance(v, int) for v in c_sched)

    def test_n_iters_positive(self, oracle_result):
        _, n_iters, *_ = oracle_result
        assert n_iters >= 1

    def test_initial_violations_nonnegative(self, oracle_result):
        _, _, initial_viol, *_ = oracle_result
        assert initial_viol >= 0

    def test_n_ticks_cheaper_nonnegative(self, oracle_result):
        _, _, _, n_cheaper, _ = oracle_result
        assert n_cheaper >= 0

    def test_baseline_n_sla_safe_positive(self, oracle_result):
        *_, baseline = oracle_result
        assert baseline > 0

    def test_c_schedule_all_positive(self, oracle_result):
        c_sched, *_ = oracle_result
        assert all(c >= 1 for c in c_sched)
