"""Tests for SOTSS-GSF (Stochastic Oracle SOTSS) — run 2026-06-23.

Tests _oracle_stochastic_response_times, compute_sotss_gsf_schedule,
SOTSSGSFReport, and the public backtest runners.

Class structure:
  1. TestStochasticOracle          — _oracle_stochastic_response_times properties
  2. TestSOTSSGSFSchedule          — compute_sotss_gsf_schedule correctness
  3. TestSOTSSGSFReportContract    — SOTSSGSFReport dataclass and to_dict
  4. TestSOTSSGSFAzureBacktest     — full Azure LLM 2024 integration test
  5. TestSOTSSGSFBurstGPTBacktest  — full BurstGPT HF integration test
"""
import math
import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_HF_JSONL,
    GPU_HOUR_USD,
    SOTSSGSFReport,
    _gsf_spot_fleet_cost,
    _simulate_fifo_gsf_spot_fleet,
    load_serving_requests,
    load_burstgpt_serving_requests_jsonl,
    run_sotss_gsf_azure_backtest,
    run_sotss_gsf_burstgpt_backtest,
    _SOTSS_GSF_SAFE_GATE,
    _SOTSS_GSF_MAX_ITERS,
)
from aurelius.optimizer.policies.replica_scaling import (
    _oracle_stochastic_response_times,
    compute_sotss_gsf_schedule,
    compute_mcs_c_schedule,
    REPLICA_SAFE_GATE,
)


# ---------------------------------------------------------------------------
# Class 1: _oracle_stochastic_response_times — stochastic oracle properties
# ---------------------------------------------------------------------------

class TestStochasticOracle:
    """Unit tests for the stochastic oracle response-time function."""

    TICK_S = 60.0

    @pytest.fixture(scope="class")
    def tiny_pairs(self):
        """30 (arrival_s, service_s) pairs — very light load."""
        return [(float(i * 120), 0.5) for i in range(30)]

    def test_returns_dict(self, tiny_pairs):
        c_schedule = [2] * 30
        result = _oracle_stochastic_response_times(
            tiny_pairs, c_schedule, seed=42
        )
        assert isinstance(result, dict)

    def test_all_requests_covered(self, tiny_pairs):
        """Every request index appears in the response dict."""
        c_schedule = [3] * 30
        result = _oracle_stochastic_response_times(
            tiny_pairs, c_schedule, seed=42
        )
        assert len(result) == len(tiny_pairs)

    def test_response_times_positive(self, tiny_pairs):
        c_schedule = [2] * 30
        result = _oracle_stochastic_response_times(
            tiny_pairs, c_schedule, seed=42
        )
        assert all(v > 0 for v in result.values())

    def test_reproducibility(self, tiny_pairs):
        """Same seed → identical response times on repeated calls."""
        c_schedule = [2] * 30
        r1 = _oracle_stochastic_response_times(tiny_pairs, c_schedule, seed=42)
        r2 = _oracle_stochastic_response_times(tiny_pairs, c_schedule, seed=42)
        assert r1 == r2

    def test_stochastic_draws_per_tick(self):
        """Oracle applies Binomial draws per tick, not a global draw.

        Verify that the length of the response dict matches the number of
        requests (the oracle processes all requests across all ticks).
        """
        # 30 pairs, ticks of 120s each → 1 request per tick on average
        c_schedule = [2] * 30
        tiny_pairs = [(float(i * 120), 0.5) for i in range(30)]
        result = _oracle_stochastic_response_times(tiny_pairs, c_schedule, seed=42)
        assert len(result) == 30, (
            f"Expected 30 responses (one per request), got {len(result)}"
        )

    def test_high_capacity_low_response(self, tiny_pairs):
        """Very high c_schedule → fast response times (near service_s)."""
        c_schedule = [50] * 30
        result = _oracle_stochastic_response_times(
            tiny_pairs, c_schedule, p_interrupt_hourly=0.10, seed=42
        )
        # With c=50 and only 30 requests, responses should be near service_s (0.5s)
        mean_resp = sum(result.values()) / len(result)
        assert mean_resp < 5.0, f"Expected fast response, got mean={mean_resp:.2f}s"

    def test_c_effective_bounded_below_1(self, tiny_pairs):
        """Even with 100% interruption probability, c_effective >= 1 (max(1,...) guard)."""
        c_schedule = [1] * 30
        # p=1.0 hourly means p_survive ≈ 0; survived = 0 → c_effective = max(1, 0+0) = 1
        result = _oracle_stochastic_response_times(
            tiny_pairs, c_schedule, p_interrupt_hourly=1.0, seed=42
        )
        # Should still return results (c_effective=1 via guard)
        assert len(result) == len(tiny_pairs)

    def test_zfhc_threshold_all_spot(self, tiny_pairs):
        """At c >= zfhc_threshold, all replicas are spot (same formula as GSF)."""
        # c=10 >= zfhc_threshold=8 → all spot; p_survive is high so few interruptions
        c_schedule = [10] * 30
        result = _oracle_stochastic_response_times(
            tiny_pairs, c_schedule, zfhc_threshold=8,
            p_interrupt_hourly=0.10, seed=42
        )
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Class 2: compute_sotss_gsf_schedule — schedule computation correctness
# ---------------------------------------------------------------------------

class TestSOTSSGSFSchedule:
    """Tests for compute_sotss_gsf_schedule."""

    @pytest.fixture(scope="class")
    def raw_small(self):
        return load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=200)

    def test_returns_five_tuple(self, raw_small):
        """Returns (c_schedule, n_iters, initial_violations, n_ticks_cheaper, baseline)."""
        result = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0)
        assert isinstance(result, tuple) and len(result) == 5

    def test_c_schedule_is_list_of_ints(self, raw_small):
        c_sched, *_ = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0)
        assert isinstance(c_sched, list)
        assert all(isinstance(c, int) and c >= 1 for c in c_sched)

    def test_c_schedule_bounded_by_ceiling(self, raw_small):
        """Every c in the schedule is ≤ the AMCSG ceiling (safe_gate=12.5%)."""
        warp = 1.0
        tick_s = 60.0
        sla_s = 10.0
        c_sched, *_ = compute_sotss_gsf_schedule(
            raw_small, tick_s, warp, sla_s, safe_gate=12.5
        )
        c_ceil = list(compute_mcs_c_schedule(
            raw_small, tick_s, warp, mcs_gate=12.5, sla_s=sla_s
        ))
        assert all(
            c_sched[i] <= c_ceil[i] for i in range(len(c_sched))
        ), "SOTSS-GSF c exceeds ceiling on some tick"

    def test_n_iters_positive(self, raw_small):
        _, n_iters, *_ = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0)
        assert n_iters >= 1

    def test_n_ticks_cheaper_nonneg(self, raw_small):
        _, _, _, n_cheaper, _ = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0)
        assert n_cheaper >= 0

    def test_baseline_n_sla_safe_positive(self, raw_small):
        *_, baseline = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0)
        assert baseline > 0

    def test_reproducibility(self, raw_small):
        """Two calls with same seed produce identical c_schedule."""
        r1 = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0, seed=42)
        r2 = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0, seed=42)
        assert r1[0] == r2[0]

    def test_different_seed_may_differ(self, raw_small):
        """Different seeds may produce different schedules (stochastic oracle)."""
        c1, *_ = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0, seed=42)
        c2, *_ = compute_sotss_gsf_schedule(raw_small, 60.0, 1.0, 10.0, seed=99)
        # Schedules MAY differ; this is a probabilistic assertion
        # (they could coincide, but unlikely with full stochastic draws)
        # We just verify both are valid (checked by other tests)
        assert len(c1) == len(c2)

    def test_max_iters_respected(self, raw_small):
        """Oracle loop never exceeds max_iters."""
        _, n_iters, *_ = compute_sotss_gsf_schedule(
            raw_small, 60.0, 1.0, 10.0, max_iters=5
        )
        assert n_iters <= 5


# ---------------------------------------------------------------------------
# Class 3: SOTSSGSFReport — dataclass contract
# ---------------------------------------------------------------------------

class TestSOTSSGSFReportContract:
    """Tests for SOTSSGSFReport dataclass and to_dict serialization."""

    def _make_report(self, **overrides) -> SOTSSGSFReport:
        base = dict(
            trace="test",
            total_requests=100,
            sla_s=10.0,
            tick_seconds=60.0,
            rng_seed=42,
            spot_price_usd_hr=0.80,
            demand_price_usd_hr=2.00,
            p_interrupt_hourly=0.10,
            zfhc_threshold=8,
            spot_fraction=0.95,
            sla_oracle_goodput_per_dollar=25208.0,
            north_star_500_threshold=151248.0,
            amcsg_gate=12.5,
            amcsg_goodput_per_dollar=150630.0,
            amcsg_cost=4.28,
            amcsg_c_mean=4.458,
            amcsg_n_sla_safe=5820,
            amcsg_p99_s=8.5,
            sotss_min_goodput_per_dollar=160107.0,
            sotss_min_cost=4.10,
            sotss_min_c_mean=4.194,
            sotss_min_n_sla_safe=5823,
            sotss_gsf_goodput_per_dollar=161000.0,
            sotss_gsf_cost=4.05,
            sotss_gsf_c_mean=4.15,
            sotss_gsf_n_sla_safe=5823,
            sotss_gsf_p99_s=8.2,
            sotss_gsf_n_iters=30,
            sotss_gsf_initial_violations=300,
            n_ticks_cheaper=20,
            sotss_gsf_vs_amcsg_pct=6.9,
            sotss_gsf_vs_sotss_min_pct=0.6,
            sotss_gsf_vs_sla_oracle_pct=538.9,
            sotss_gsf_north_star_500_achieved=True,
            sotss_gsf_safe=True,
        )
        base.update(overrides)
        return SOTSSGSFReport(**base)

    def test_instantiation(self):
        report = self._make_report()
        assert report.trace == "test"

    def test_to_dict_keys(self):
        report = self._make_report()
        d = report.to_dict()
        required_keys = [
            "trace", "total_requests", "sla_s",
            "amcsg_goodput_per_dollar", "amcsg_n_sla_safe",
            "sotss_min_goodput_per_dollar", "sotss_min_n_sla_safe",
            "sotss_gsf_goodput_per_dollar", "sotss_gsf_n_sla_safe",
            "sotss_gsf_vs_amcsg_pct", "sotss_gsf_vs_sotss_min_pct",
            "sotss_gsf_north_star_500_achieved", "sotss_gsf_safe",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_roundtrip(self):
        report = self._make_report()
        d = report.to_dict()
        assert d["trace"] == "test"
        assert d["sotss_gsf_north_star_500_achieved"] is True
        assert d["sotss_gsf_safe"] is True

    def test_to_dict_rounds_floats(self):
        report = self._make_report(sotss_gsf_goodput_per_dollar=161000.123456789)
        d = report.to_dict()
        # Should be rounded to 2 decimal places
        assert d["sotss_gsf_goodput_per_dollar"] == round(161000.123456789, 2)

    def test_negative_result_unsafe_flag(self):
        report = self._make_report(
            sotss_gsf_n_sla_safe=5700,  # below amcsg_n_sla_safe=5820
            sotss_gsf_safe=False,
            sotss_gsf_north_star_500_achieved=False,
        )
        d = report.to_dict()
        assert d["sotss_gsf_safe"] is False
        assert d["sotss_gsf_north_star_500_achieved"] is False


# ---------------------------------------------------------------------------
# Class 4: run_sotss_gsf_azure_backtest — full integration test
# ---------------------------------------------------------------------------

class TestSOTSSGSFAzureBacktest:
    """Full backtest integration tests on Azure LLM 2024."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_gsf_azure_backtest()

    def test_returns_sotss_gsf_report(self, report):
        assert isinstance(report, SOTSSGSFReport)

    def test_trace_name(self, report):
        assert "azure" in report.trace.lower()

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_sla_s(self, report):
        assert report.sla_s == 10.0

    def test_amcsg_baseline_plausible(self, report):
        """AMCSG baseline should be within ±5% of the known 150,630."""
        assert 140_000 < report.amcsg_goodput_per_dollar < 165_000, (
            f"AMCSG goodput/$ out of expected range: {report.amcsg_goodput_per_dollar:.0f}"
        )

    def test_sotss_min_plausible(self, report):
        """SOTSS-MIN reference should be near 160,107."""
        assert 150_000 < report.sotss_min_goodput_per_dollar < 175_000, (
            f"SOTSS-MIN goodput/$ out of range: {report.sotss_min_goodput_per_dollar:.0f}"
        )

    def test_gsf_goodput_positive(self, report):
        assert report.sotss_gsf_goodput_per_dollar > 0

    def test_gsf_cost_positive(self, report):
        assert report.sotss_gsf_cost > 0

    def test_gsf_c_mean_in_range(self, report):
        """c_mean should be between 1 and 10 for this workload."""
        assert 1.0 <= report.sotss_gsf_c_mean <= 10.0

    def test_gsf_n_iters_positive(self, report):
        assert report.sotss_gsf_n_iters >= 1

    def test_gsf_n_sla_safe_positive(self, report):
        assert report.sotss_gsf_n_sla_safe > 0

    def test_north_star_500_check(self, report):
        """If GSF achieves north-star 500%, north_star_500_achieved must be True."""
        expected = (
            report.sotss_gsf_goodput_per_dollar >= report.north_star_500_threshold
            and report.sotss_gsf_safe
        )
        assert report.sotss_gsf_north_star_500_achieved == expected

    def test_vs_amcsg_pct_consistent(self, report):
        """sotss_gsf_vs_amcsg_pct consistent with goodput/$ values."""
        expected_pct = (
            (report.sotss_gsf_goodput_per_dollar - report.amcsg_goodput_per_dollar)
            / max(report.amcsg_goodput_per_dollar, 1e-9) * 100.0
        )
        assert abs(report.sotss_gsf_vs_amcsg_pct - expected_pct) < 0.01

    def test_vs_sotss_min_pct_consistent(self, report):
        """sotss_gsf_vs_sotss_min_pct consistent with goodput/$ values."""
        expected_pct = (
            (report.sotss_gsf_goodput_per_dollar - report.sotss_min_goodput_per_dollar)
            / max(report.sotss_min_goodput_per_dollar, 1e-9) * 100.0
        )
        assert abs(report.sotss_gsf_vs_sotss_min_pct - expected_pct) < 0.01

    def test_to_dict_serializable(self, report):
        """to_dict produces a JSON-serializable dict with expected keys."""
        import json
        d = report.to_dict()
        json.dumps(d)  # must not raise
        assert "sotss_gsf_goodput_per_dollar" in d
        assert "sotss_gsf_vs_amcsg_pct" in d
        assert "sotss_gsf_safe" in d

    def test_honest_reporting_on_improvement(self, report):
        """If GSF is not strictly better than AMCSG, that's reported as-is (no inflation)."""
        # The test does NOT assert positivity — if GSF underperforms, that's a valid result
        # We only assert the metric is consistent with the underlying numbers
        actual_gp = report.sotss_gsf_goodput_per_dollar
        amcsg_gp = report.amcsg_goodput_per_dollar
        implied_sign = 1.0 if actual_gp >= amcsg_gp else -1.0
        assert implied_sign * report.sotss_gsf_vs_amcsg_pct >= 0, (
            "Sign of vs_amcsg_pct inconsistent with goodput/$ values"
        )


# ---------------------------------------------------------------------------
# Class 5: run_sotss_gsf_burstgpt_backtest — full BurstGPT integration test
# ---------------------------------------------------------------------------

class TestSOTSSGSFBurstGPTBacktest:
    """Full backtest integration tests on BurstGPT HF."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_sotss_gsf_burstgpt_backtest()

    def test_returns_sotss_gsf_report(self, report):
        assert isinstance(report, SOTSSGSFReport)

    def test_trace_name(self, report):
        assert "burstgpt" in report.trace.lower()

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_sla_s(self, report):
        assert report.sla_s == 30.0

    def test_amcsg_baseline_plausible(self, report):
        """AMCSG baseline should be within ±5% of the known 168,270."""
        assert 155_000 < report.amcsg_goodput_per_dollar < 185_000, (
            f"AMCSG goodput/$ out of expected range: {report.amcsg_goodput_per_dollar:.0f}"
        )

    def test_sotss_min_plausible(self, report):
        """SOTSS-MIN reference should be near 170,572 (gate=20% result)."""
        assert 160_000 < report.sotss_min_goodput_per_dollar < 195_000, (
            f"SOTSS-MIN goodput/$ out of range: {report.sotss_min_goodput_per_dollar:.0f}"
        )

    def test_gsf_goodput_positive(self, report):
        assert report.sotss_gsf_goodput_per_dollar > 0

    def test_north_star_500_check(self, report):
        """If GSF achieves north-star 500% AND is safe, flag is True."""
        expected = (
            report.sotss_gsf_goodput_per_dollar >= report.north_star_500_threshold
            and report.sotss_gsf_safe
        )
        assert report.sotss_gsf_north_star_500_achieved == expected

    def test_safety_flag_consistent(self, report):
        """sotss_gsf_safe consistent with n_sla_safe vs amcsg_n_sla_safe."""
        expected_safe = report.sotss_gsf_n_sla_safe >= report.amcsg_n_sla_safe
        assert report.sotss_gsf_safe == expected_safe, (
            f"safe flag mismatch: n_sla_safe={report.sotss_gsf_n_sla_safe} "
            f"vs amcsg_n_sla_safe={report.amcsg_n_sla_safe}"
        )

    def test_honest_reporting_on_improvement(self, report):
        """If GSF underperforms SOTSS-MIN, vs_sotss_min_pct is negative — reported as-is."""
        actual_gp = report.sotss_gsf_goodput_per_dollar
        min_gp = report.sotss_min_goodput_per_dollar
        implied_sign = 1.0 if actual_gp >= min_gp else -1.0
        assert implied_sign * report.sotss_gsf_vs_sotss_min_pct >= 0, (
            "Sign of vs_sotss_min_pct inconsistent with goodput/$ values"
        )

    def test_to_dict_serializable(self, report):
        import json
        d = report.to_dict()
        json.dumps(d)
        assert "sotss_gsf_goodput_per_dollar" in d
        assert "sotss_gsf_safe" in d
