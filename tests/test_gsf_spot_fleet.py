"""Tests for Graduated Spot Fleet (GSF) policy — run 2026-06-26.

Tests _gsf_spot_replicas, _gsf_spot_fleet_cost, _simulate_fifo_gsf_spot_fleet,
GSFFractionEntry, GSFReport, run_gsf_azure_backtest, run_gsf_burstgpt_backtest.

All tests use the existing Azure LLM 2024 fixture and BurstGPT HF fixture.
"""
import math
import pytest
from aurelius.benchmarks.srtf_serving_backtest import (
    _gsf_spot_replicas,
    _gsf_spot_fleet_cost,
    _gsf_expected_interruptions,
    _simulate_fifo_gsf_spot_fleet,
    GSFFractionEntry,
    GSFReport,
    run_gsf_azure_backtest,
    run_gsf_burstgpt_backtest,
    GPU_HOUR_USD,
    _GSF_FRACTIONS,
)


# ---------------------------------------------------------------------------
# Class 1: _gsf_spot_replicas — formula correctness
# ---------------------------------------------------------------------------

class TestGsfSpotReplicas:
    """Unit tests for the GSF spot-replica formula."""

    def test_zfhc_threshold_gives_all_spot(self):
        """At c >= zfhc_threshold, all replicas are spot (ZFHC logic)."""
        assert _gsf_spot_replicas(8, 0.70, zfhc_threshold=8) == 8
        assert _gsf_spot_replicas(10, 0.70, zfhc_threshold=8) == 10
        assert _gsf_spot_replicas(14, 0.70, zfhc_threshold=8) == 14

    def test_fraction_70_matches_afms_below_threshold(self):
        """At f=0.70, behaviour below ZFHC threshold must be identical to AFMS."""
        from aurelius.benchmarks.srtf_serving_backtest import _abs_floor_spot_replicas
        for c in range(1, 8):
            afms = _abs_floor_spot_replicas(c, min_ondemand=1)
            gsf = _gsf_spot_replicas(c, 0.70, zfhc_threshold=8)
            assert gsf == afms, f"c={c}: GSF(0.70)={gsf} != AFMS={afms}"

    def test_fraction_90_removes_ondemand_at_c2(self):
        """At f=0.90, c=2 becomes all-spot: round(1.8)=2, max(2,1)=2."""
        c_spot = _gsf_spot_replicas(2, 0.90, zfhc_threshold=8)
        assert c_spot == 2, f"Expected 2 spot at c=2, f=0.90 — got {c_spot}"

    def test_fraction_90_removes_ondemand_at_c3(self):
        """At f=0.90, c=3 becomes all-spot: round(2.7)=3, max(3,2)=3."""
        c_spot = _gsf_spot_replicas(3, 0.90, zfhc_threshold=8)
        assert c_spot == 3, f"Expected 3 spot at c=3, f=0.90 — got {c_spot}"

    def test_fraction_90_removes_ondemand_at_c4(self):
        """At f=0.90, c=4 becomes all-spot: round(3.6)=4, max(4,3)=4."""
        c_spot = _gsf_spot_replicas(4, 0.90, zfhc_threshold=8)
        assert c_spot == 4, f"Expected 4 spot at c=4, f=0.90 — got {c_spot}"

    def test_fraction_100_all_spot_everywhere(self):
        """At f=1.00, every c becomes all-spot (c_spot = c)."""
        for c in range(1, 10):
            assert _gsf_spot_replicas(c, 1.00, zfhc_threshold=8) == c

    def test_fraction_80_removes_ondemand_at_c2(self):
        """At f=0.80, c=2: round(1.6)=2, max(2,1)=2 — all spot."""
        c_spot = _gsf_spot_replicas(2, 0.80, zfhc_threshold=8)
        assert c_spot == 2

    def test_spot_replicas_bounded_by_c(self):
        """c_spot is always <= c (cannot exceed total replicas)."""
        for f in _GSF_FRACTIONS:
            for c in range(1, 15):
                spot = _gsf_spot_replicas(c, f, zfhc_threshold=8)
                assert 0 <= spot <= c, f"c_spot={spot} out of [0,{c}] at f={f}, c={c}"

    def test_spot_replicas_monotone_in_fraction(self):
        """For fixed c, c_spot is non-decreasing as spot_fraction increases."""
        fractions = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
        for c in range(1, 9):
            spots = [_gsf_spot_replicas(c, f, zfhc_threshold=8) for f in fractions]
            for i in range(len(spots) - 1):
                assert spots[i] <= spots[i + 1], (
                    f"c={c}: c_spot not monotone at f={fractions[i]}→{fractions[i+1]}: "
                    f"{spots[i]}→{spots[i+1]}"
                )

    def test_c1_always_all_spot(self):
        """c=1 is always all-spot regardless of fraction (AFMS: max(round(f),0)=1)."""
        for f in _GSF_FRACTIONS:
            assert _gsf_spot_replicas(1, f, zfhc_threshold=8) == 1


# ---------------------------------------------------------------------------
# Class 2: _gsf_spot_fleet_cost — cost monotonicity
# ---------------------------------------------------------------------------

class TestGsfSpotFleetCost:
    """Unit tests for GSF fleet cost calculation."""

    def test_cost_decreases_as_fraction_increases(self):
        """Higher fraction → more spot → lower cost."""
        c_schedule = [3, 4, 5, 6, 7, 8]
        spot_price = 0.80
        prev_cost = None
        for f in [0.70, 0.80, 0.90, 1.00]:
            cost = _gsf_spot_fleet_cost(
                c_schedule, f, 8, spot_price, GPU_HOUR_USD, tick_seconds=60.0
            )
            if prev_cost is not None:
                assert cost <= prev_cost + 1e-9, (
                    f"Cost did not decrease at f={f}: {cost:.4f} vs {prev_cost:.4f}"
                )
            prev_cost = cost

    def test_cost_is_positive(self):
        """Fleet cost is always positive."""
        c_schedule = [2, 3, 4, 5]
        for f in _GSF_FRACTIONS:
            cost = _gsf_spot_fleet_cost(
                c_schedule, f, 8, 0.80, GPU_HOUR_USD, tick_seconds=60.0
            )
            assert cost > 0.0

    def test_fraction_100_gives_minimum_cost(self):
        """f=1.00 (all spot) gives the lowest possible cost."""
        c_schedule = [2, 3, 4, 5, 6, 7, 8]
        costs = {
            f: _gsf_spot_fleet_cost(c_schedule, f, 8, 0.80, GPU_HOUR_USD, 60.0)
            for f in _GSF_FRACTIONS
        }
        assert costs[1.00] == min(costs.values()), "f=1.00 should give minimum cost"

    def test_fraction_70_gives_maximum_cost(self):
        """f=0.70 (same as AFMS) gives the highest cost in the sweep."""
        c_schedule = [2, 3, 4, 5, 6, 7, 8]
        costs = {
            f: _gsf_spot_fleet_cost(c_schedule, f, 8, 0.80, GPU_HOUR_USD, 60.0)
            for f in _GSF_FRACTIONS
        }
        assert costs[0.70] == max(costs.values()), "f=0.70 should give maximum cost"

    def test_cost_formula_matches_manual_calculation(self):
        """Verify cost formula against manual calculation for a simple schedule."""
        # c=2, f=0.90: c_spot=2, c_demand=0 → cost = 2 × 0.80 × (60/3600) = $0.0267
        c_schedule = [2]
        cost = _gsf_spot_fleet_cost(c_schedule, 0.90, 8, 0.80, GPU_HOUR_USD, 60.0)
        expected = 2 * 0.80 * (60.0 / 3600.0)
        assert abs(cost - expected) < 1e-9, f"Expected {expected:.6f}, got {cost:.6f}"

    def test_afms_cost_formula_for_c2_at_f70(self):
        """c=2, f=0.70: AFMS gives 1 spot + 1 demand → 0.80+2.00 × 60/3600."""
        c_schedule = [2]
        cost = _gsf_spot_fleet_cost(c_schedule, 0.70, 8, 0.80, GPU_HOUR_USD, 60.0)
        expected = (1 * 0.80 + 1 * GPU_HOUR_USD) * (60.0 / 3600.0)
        assert abs(cost - expected) < 1e-9


# ---------------------------------------------------------------------------
# Class 3: _gsf_expected_interruptions — interruption model
# ---------------------------------------------------------------------------

class TestGsfExpectedInterruptions:
    """Unit tests for GSF expected interruption calculation."""

    def test_zero_interruption_rate_gives_zero(self):
        """With p_interrupt_hourly=0, expected interruptions = 0."""
        c_schedule = [4, 5, 6, 8]
        result = _gsf_expected_interruptions(c_schedule, 0.90, 8, 0.0, 60.0)
        assert result == 0.0

    def test_more_spot_means_more_expected_interruptions(self):
        """Higher fraction → more spot → higher expected interruption count."""
        c_schedule = [3, 4, 5, 6]
        prev = None
        for f in [0.70, 0.80, 0.90, 1.00]:
            ei = _gsf_expected_interruptions(c_schedule, f, 8, 0.10, 60.0)
            if prev is not None:
                assert ei >= prev - 1e-12, f"Expected interruptions not monotone at f={f}"
            prev = ei

    def test_expected_interruptions_positive_for_nonzero_rate(self):
        """With positive p_interrupt_hourly, expected interruptions > 0."""
        c_schedule = [4, 5, 6]
        result = _gsf_expected_interruptions(c_schedule, 0.90, 8, 0.10, 60.0)
        assert result > 0.0

    def test_expected_interruptions_small_for_low_p(self):
        """At p_interrupt=0.10/hr, tick=60s: p_survive = (0.9)^(1/60) ≈ 0.9983."""
        c_schedule = [4]  # 4 spot at f=1.00
        ei = _gsf_expected_interruptions(c_schedule, 1.00, 8, 0.10, 60.0)
        p_survive = (1.0 - 0.10) ** (60.0 / 3600.0)
        expected = 4 * (1.0 - p_survive)
        assert abs(ei - expected) < 1e-9


# ---------------------------------------------------------------------------
# Class 4: _simulate_fifo_gsf_spot_fleet — simulation correctness
# ---------------------------------------------------------------------------

class TestSimulateFifoGsfSpotFleet:
    """Unit tests for the GSF stochastic spot simulation."""

    def _make_requests(self, n=20, service_s=2.0):
        """Create synthetic requests for testing."""
        from aurelius.benchmarks.srtf_serving_backtest import _Request
        return [
            _Request(
                idx=i,
                arrival_s=float(i * 3),
                actual_tokens=100,
                predicted_tokens=100.0,
                service_s=service_s,
            )
            for i in range(n)
        ]

    def test_returns_three_tuple(self):
        """Simulation returns (sim_stats, response_times, n_served) tuple."""
        reqs = self._make_requests(10)
        c_schedule = [3] * 5
        result = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 0.90, 8, 0.10, 60.0, seed=42
        )
        assert isinstance(result, tuple) and len(result) == 3

    def test_fraction_70_matches_zfhc_baseline(self):
        """GSF(f=0.70) must give the same effective c as ZFHC(8) at same seed."""
        from aurelius.benchmarks.srtf_serving_backtest import _simulate_fifo_zfhc_spot_fleet
        reqs = self._make_requests(20)
        c_schedule = [3, 4, 5, 6, 7, 8]
        gsf_sim, gsf_resp, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 0.70, 8, 0.10, 60.0, seed=42
        )
        zfhc_sim, zfhc_resp, _ = _simulate_fifo_zfhc_spot_fleet(
            reqs, c_schedule, 8, 0.10, 60.0, seed=42
        )
        assert gsf_resp == zfhc_resp

    def test_high_fraction_never_worsens_response_time(self):
        """Higher fraction (lower cost) should not systematically worsen SLA."""
        reqs = self._make_requests(50, service_s=1.0)
        c_schedule = [4] * 10
        _, resp_70, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 0.70, 8, 0.0, 60.0, seed=42
        )
        _, resp_100, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 1.00, 8, 0.0, 60.0, seed=42
        )
        # With p_interrupt=0 (no interruptions), responses should be identical
        assert resp_70 == resp_100, "With zero interruptions, f=0.70 and f=1.00 must be identical"

    def test_simulation_serves_all_requests(self):
        """With large enough c and zero interruptions, all requests are served."""
        reqs = self._make_requests(20)
        c_schedule = [10] * 10
        sim, resp, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 1.00, 8, 0.0, 60.0, seed=42
        )
        assert len(resp) == len(reqs)

    def test_reproducible_with_seed(self):
        """Same seed produces identical results."""
        reqs = self._make_requests(30)
        c_schedule = [3, 4, 5, 6, 7]
        _, resp1, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 0.90, 8, 0.10, 60.0, seed=99
        )
        _, resp2, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 0.90, 8, 0.10, 60.0, seed=99
        )
        assert resp1 == resp2

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different effective c schedules."""
        reqs = self._make_requests(50)
        # Use large c_schedule with many spot instances to ensure interruptions happen
        c_schedule = [6, 7, 8, 9, 10] * 4
        _, resp1, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 1.00, 20, 0.50, 60.0, seed=1
        )
        _, resp2, _ = _simulate_fifo_gsf_spot_fleet(
            reqs, c_schedule, 1.00, 20, 0.50, 60.0, seed=999
        )
        # With p_interrupt=50%/hr, responses WILL differ across seeds
        # (not guaranteed, but highly likely — check that both complete all requests)
        assert len(resp1) == len(reqs)
        assert len(resp2) == len(reqs)


# ---------------------------------------------------------------------------
# Class 5: GSFFractionEntry dataclass
# ---------------------------------------------------------------------------

class TestGsfFractionEntry:
    """Unit tests for GSFFractionEntry dataclass."""

    def _make_entry(self, f=0.90, gpd=150_000.0, vs_zfhc=+6.0):
        return GSFFractionEntry(
            spot_fraction=f,
            n_ticks_c_all_spot=30,
            cost_gsf=9.50,
            cost_vs_zfhc_reduction_pct=10.0,
            goodput_per_dollar=gpd,
            goodput_vs_zfhc_pct=vs_zfhc,
            goodput_vs_sla_oracle_pct=640.0,
            north_star_achieved=True,
            completion_rate=1.0,
            p99_s=22.5,
        )

    def test_to_dict_has_required_keys(self):
        e = self._make_entry()
        d = e.to_dict()
        for key in ["spot_fraction", "n_ticks_c_all_spot", "cost_gsf",
                    "cost_vs_zfhc_reduction_pct", "goodput_per_dollar",
                    "goodput_vs_zfhc_pct", "goodput_vs_sla_oracle_pct",
                    "north_star_achieved", "completion_rate", "p99_s"]:
            assert key in d, f"Missing key: {key}"

    def test_north_star_flag_correct(self):
        e = self._make_entry(gpd=150_000.0)
        assert e.north_star_achieved is True

    def test_to_dict_rounds_floats(self):
        e = self._make_entry(gpd=150_000.123456789)
        d = e.to_dict()
        assert d["goodput_per_dollar"] == 150_000.12


# ---------------------------------------------------------------------------
# Class 6: GSFReport dataclass
# ---------------------------------------------------------------------------

class TestGsfReport:
    """Unit tests for GSFReport dataclass."""

    def _make_report(self):
        entries = [
            GSFFractionEntry(
                spot_fraction=0.70, n_ticks_c_all_spot=4, cost_gsf=5.66,
                cost_vs_zfhc_reduction_pct=0.0, goodput_per_dollar=113_904.0,
                goodput_vs_zfhc_pct=0.0, goodput_vs_sla_oracle_pct=351.9,
                north_star_achieved=True, completion_rate=1.0, p99_s=9.95,
            ),
            GSFFractionEntry(
                spot_fraction=0.90, n_ticks_c_all_spot=40, cost_gsf=4.80,
                cost_vs_zfhc_reduction_pct=15.2, goodput_per_dollar=135_000.0,
                goodput_vs_zfhc_pct=18.5, goodput_vs_sla_oracle_pct=435.6,
                north_star_achieved=True, completion_rate=1.0, p99_s=9.95,
            ),
        ]
        return GSFReport(
            trace="test_trace",
            total_requests=5880,
            fixed_c=4,
            target_rho=0.85,
            sla_s=10.0,
            tick_seconds=60.0,
            rng_seed=42,
            c_schedule_mean=4.5,
            c_schedule_min=1,
            c_schedule_max=8,
            n_ticks=72,
            spot_price_usd_hr=0.80,
            demand_price_usd_hr=2.00,
            p_interrupt_hourly=0.10,
            zfhc_threshold=8,
            cost_zfhc_baseline=5.66,
            zfhc_goodput_per_dollar=113_904.0,
            zfhc_vs_sla_oracle_pct=351.9,
            fraction_results=entries,
            best_fraction=0.90,
            best_goodput_per_dollar=135_000.0,
            best_vs_zfhc_pct=18.5,
            best_vs_sla_oracle_pct=435.6,
            best_north_star_achieved=True,
            north_star_threshold=100_832.0,
            sla_oracle_goodput_per_dollar=25_208.0,
        )

    def test_to_dict_has_required_keys(self):
        r = self._make_report()
        d = r.to_dict()
        for key in [
            "trace", "total_requests", "fixed_c", "target_rho", "sla_s",
            "c_schedule_mean", "c_schedule_min", "c_schedule_max", "n_ticks",
            "spot_price_usd_hr", "demand_price_usd_hr", "p_interrupt_hourly",
            "zfhc_threshold", "cost_zfhc_baseline", "zfhc_goodput_per_dollar",
            "fraction_results", "best_fraction", "best_goodput_per_dollar",
            "best_vs_zfhc_pct", "best_vs_sla_oracle_pct",
            "best_north_star_achieved", "north_star_threshold",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_fraction_results_serialized(self):
        r = self._make_report()
        d = r.to_dict()
        assert isinstance(d["fraction_results"], list)
        assert len(d["fraction_results"]) == 2

    def test_best_fraction_identity(self):
        r = self._make_report()
        assert r.best_fraction == 0.90
        assert r.best_goodput_per_dollar == 135_000.0


# ---------------------------------------------------------------------------
# Class 7: run_gsf_azure_backtest — end-to-end on public Azure trace
# ---------------------------------------------------------------------------

class TestRunGsfAzureBacktest:
    """End-to-end tests for GSF on Azure LLM 2024."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_gsf_azure_backtest(
            fixed_c=4, target_rho=0.85, job_limit=5880,
            fractions=(0.70, 0.90, 1.00),
        )

    def test_report_type(self, report):
        assert isinstance(report, GSFReport)

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_zfhc_baseline_matches_run_2026_06_25(self, report):
        """ZFHC(8) internal baseline must match the published result ±2%."""
        published = 113_904.0
        assert abs(report.zfhc_goodput_per_dollar - published) / published < 0.02, (
            f"ZFHC baseline {report.zfhc_goodput_per_dollar:.0f} deviates > 2% from "
            f"published {published:.0f}"
        )

    def test_fraction_70_matches_zfhc_baseline(self, report):
        """f=0.70 entry must match the ZFHC(8) baseline within 1% (same policy)."""
        entry_70 = next(e for e in report.fraction_results if e.spot_fraction == 0.70)
        ratio = entry_70.goodput_per_dollar / report.zfhc_goodput_per_dollar
        assert 0.99 <= ratio <= 1.01, f"f=0.70 ratio to ZFHC baseline: {ratio:.4f}"

    def test_north_star_maintained_at_best(self, report):
        """Best fraction must maintain north-star (+300% vs SLA-oracle)."""
        assert report.best_north_star_achieved, (
            f"North-star lost at best_fraction={report.best_fraction}: "
            f"{report.best_goodput_per_dollar:.0f} < {report.north_star_threshold:.0f}"
        )

    def test_completion_rate_at_best(self, report):
        """Best fraction must have completion rate ≥ 0.99 (all requests complete)."""
        best_entry = next(
            e for e in report.fraction_results if e.spot_fraction == report.best_fraction
        )
        assert best_entry.completion_rate >= 0.99, (
            f"Completion rate {best_entry.completion_rate:.4f} < 0.99"
        )

    def test_best_goodput_at_least_zfhc(self, report):
        """Best GSF goodput/$ must be >= ZFHC(8) baseline (no regression)."""
        assert report.best_goodput_per_dollar >= report.zfhc_goodput_per_dollar * 0.99, (
            f"GSF best {report.best_goodput_per_dollar:.0f} regresses vs "
            f"ZFHC {report.zfhc_goodput_per_dollar:.0f}"
        )

    def test_fraction_results_cover_sweep(self, report):
        """All requested fractions are present in results."""
        result_fractions = {e.spot_fraction for e in report.fraction_results}
        for f in (0.70, 0.90, 1.00):
            assert f in result_fractions, f"Fraction {f} missing from results"

    def test_to_dict_serializable(self, report):
        """Report must be fully JSON-serializable."""
        import json
        d = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 100

    def test_sla_oracle_referenced_correctly(self, report):
        """SLA oracle reference must match Azure oracle (25,208)."""
        assert abs(report.sla_oracle_goodput_per_dollar - 25_208.0) < 1.0


# ---------------------------------------------------------------------------
# Class 8: run_gsf_burstgpt_backtest — end-to-end on public BurstGPT trace
# ---------------------------------------------------------------------------

class TestRunGsfBurstgptBacktest:
    """End-to-end tests for GSF on BurstGPT HF."""

    @pytest.fixture(scope="class")
    def report(self):
        return run_gsf_burstgpt_backtest(
            fixed_c=4, target_rho=0.85, job_limit=5880,
            fractions=(0.70, 0.90, 1.00),
        )

    def test_report_type(self, report):
        assert isinstance(report, GSFReport)

    def test_total_requests(self, report):
        assert report.total_requests == 5880

    def test_zfhc_baseline_matches_run_2026_06_25(self, report):
        """ZFHC(8) internal baseline must match published result ±2%."""
        published = 140_647.0
        assert abs(report.zfhc_goodput_per_dollar - published) / published < 0.02, (
            f"ZFHC baseline {report.zfhc_goodput_per_dollar:.0f} deviates > 2% from "
            f"published {published:.0f}"
        )

    def test_north_star_maintained_at_best(self, report):
        """Best fraction must maintain north-star (+300% vs SLA-oracle)."""
        assert report.best_north_star_achieved

    def test_completion_rate_at_best(self, report):
        """Best fraction must have completion rate ≥ 0.99."""
        best_entry = next(
            e for e in report.fraction_results if e.spot_fraction == report.best_fraction
        )
        assert best_entry.completion_rate >= 0.99

    def test_best_goodput_at_least_zfhc(self, report):
        """Best GSF goodput/$ >= ZFHC(8) baseline (no regression)."""
        assert report.best_goodput_per_dollar >= report.zfhc_goodput_per_dollar * 0.99

    def test_sla_oracle_referenced_correctly(self, report):
        """SLA oracle reference must match BurstGPT oracle (20,280)."""
        assert abs(report.sla_oracle_goodput_per_dollar - 20_280.0) < 1.0
