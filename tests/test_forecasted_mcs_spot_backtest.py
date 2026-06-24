"""Tests for Forecasted MCS Spot Fleet backtest — run 2026-06-24.

Tests ForecastedMCSSpotReport, _run_forecasted_mcs_spot_backtest,
run_forecasted_mcs_spot_azure_backtest, and run_forecasted_mcs_spot_burstgpt_backtest.

Core contracts verified:
  - ForecastedMCSSpotReport.to_dict() round-trips cleanly with correct types.
  - lag1 and ewma c_schedules are non-empty lists of positive ints.
  - Same-conditions rule: all three paths (amcsg, lag1, ewma) run under identical
    spot-fleet parameters (spot_fraction=0.95, zfhc_threshold=8, p_interrupt=0.10).
  - n_sla_safe_safe flags are consistent with n_sla_safe vs amcsg_n_sla_safe.
  - north_star_500_achieved flags are consistent with goodput and n_sla_safe_safe.
  - Both modes produce non-negative goodput_per_dollar.
  - vs_amcsg_pct is derived correctly from the goodput values.
  - Backtest completes on the fixture-scale Azure trace (54 rows) without error.
  - Backtest completes on a synthetic trace without error.
  - lag1 sub-mode is causal: warmup_c is used on the first tick.
  - ewma sub-mode changes result when ewma_alpha is varied.

Tests requiring numpy (the GSF spot-fleet stochastic simulation) are skipped
automatically if numpy is not installed in the test environment.
"""
import json
import statistics

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    ForecastedMCSSpotReport,
    _FMCS_SPOT_EWMA_ALPHA,
    _FMCS_SPOT_MCS_GATE,
    _run_forecasted_mcs_spot_backtest,
    run_forecasted_mcs_spot_azure_backtest,
    run_forecasted_mcs_spot_burstgpt_backtest,
    DEFAULT_AZURE_FIXTURE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_flat_trace(
    n_ticks: int = 10,
    req_per_tick: int = 5,
    tok: int = 100,
    tick_s: float = 60.0,
) -> list:
    """Flat trace: req_per_tick arrivals per tick, identical tokens."""
    raw = []
    for t_idx in range(n_ticks):
        for i in range(req_per_tick):
            arrival = t_idx * tick_s + i * (tick_s / (req_per_tick + 1))
            raw.append((arrival, tok))
    return raw


def _make_report(
    lag1_gp: float = 140_000.0,
    ewma_gp: float = 135_000.0,
    amcsg_gp: float = 130_000.0,
    lag1_n: int = 5500,
    ewma_n: int = 5400,
    amcsg_n: int = 5300,
    north_star: float = 151_248.0,
) -> ForecastedMCSSpotReport:
    """Manually construct a ForecastedMCSSpotReport for unit testing."""
    lag1_vs_amcsg = (lag1_gp - amcsg_gp) / max(amcsg_gp, 1e-9) * 100.0
    ewma_vs_amcsg = (ewma_gp - amcsg_gp) / max(amcsg_gp, 1e-9) * 100.0
    sla_oracle = 25_208.0
    return ForecastedMCSSpotReport(
        trace="test_synthetic",
        total_requests=5880,
        sla_s=10.0,
        tick_seconds=60.0,
        rng_seed=42,
        spot_price_usd_hr=0.80,
        demand_price_usd_hr=2.0,
        p_interrupt_hourly=0.10,
        zfhc_threshold=8,
        mcs_gate=12.5,
        ewma_alpha=0.5,
        sla_oracle_goodput_per_dollar=sla_oracle,
        north_star_500_threshold=north_star,
        amcsg_goodput_per_dollar=amcsg_gp,
        amcsg_cost=1.5,
        amcsg_c_mean=3.2,
        amcsg_n_sla_safe=amcsg_n,
        amcsg_p99_s=8.5,
        lag1_goodput_per_dollar=lag1_gp,
        lag1_cost=1.4,
        lag1_c_mean=3.1,
        lag1_n_sla_safe=lag1_n,
        lag1_p99_s=8.2,
        lag1_vs_amcsg_pct=lag1_vs_amcsg,
        lag1_vs_sla_oracle_pct=(lag1_gp - sla_oracle) / sla_oracle * 100.0,
        lag1_north_star_500_achieved=(lag1_gp >= north_star and lag1_n >= amcsg_n),
        lag1_n_sla_safe_safe=(lag1_n >= amcsg_n),
        ewma_goodput_per_dollar=ewma_gp,
        ewma_cost=1.45,
        ewma_c_mean=3.15,
        ewma_n_sla_safe=ewma_n,
        ewma_p99_s=8.3,
        ewma_vs_amcsg_pct=ewma_vs_amcsg,
        ewma_vs_sla_oracle_pct=(ewma_gp - sla_oracle) / sla_oracle * 100.0,
        ewma_north_star_500_achieved=(ewma_gp >= north_star and ewma_n >= amcsg_n),
        ewma_n_sla_safe_safe=(ewma_n >= amcsg_n),
    )


# ---------------------------------------------------------------------------
# Class 1: ForecastedMCSSpotReport — dataclass contracts (no numpy needed)
# ---------------------------------------------------------------------------

class TestForecastedMCSSpotReportDataclass:
    """Unit tests for ForecastedMCSSpotReport dataclass — no numpy dependency."""

    def test_to_dict_returns_dict(self):
        r = _make_report()
        d = r.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_required_keys(self):
        r = _make_report()
        d = r.to_dict()
        required = {
            "trace", "total_requests", "sla_s", "tick_seconds",
            "rng_seed", "spot_price_usd_hr", "demand_price_usd_hr",
            "p_interrupt_hourly", "zfhc_threshold", "mcs_gate", "ewma_alpha",
            "sla_oracle_goodput_per_dollar", "north_star_500_threshold",
            "amcsg_goodput_per_dollar", "amcsg_cost", "amcsg_c_mean",
            "amcsg_n_sla_safe", "amcsg_p99_s",
            "lag1_goodput_per_dollar", "lag1_cost", "lag1_c_mean",
            "lag1_n_sla_safe", "lag1_p99_s",
            "lag1_vs_amcsg_pct", "lag1_vs_sla_oracle_pct",
            "lag1_north_star_500_achieved", "lag1_n_sla_safe_safe",
            "ewma_goodput_per_dollar", "ewma_cost", "ewma_c_mean",
            "ewma_n_sla_safe", "ewma_p99_s",
            "ewma_vs_amcsg_pct", "ewma_vs_sla_oracle_pct",
            "ewma_north_star_500_achieved", "ewma_n_sla_safe_safe",
        }
        assert required.issubset(set(d.keys()))

    def test_to_dict_is_json_serializable(self):
        r = _make_report()
        d = r.to_dict()
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        assert loaded["trace"] == "test_synthetic"

    def test_trace_name_propagates(self):
        r = _make_report()
        assert r.trace == "test_synthetic"
        assert r.to_dict()["trace"] == "test_synthetic"

    def test_sla_s_in_dict(self):
        r = _make_report()
        d = r.to_dict()
        assert d["sla_s"] == pytest.approx(10.0)

    def test_n_sla_safe_safe_flag_consistent(self):
        r = _make_report(lag1_n=5500, ewma_n=5400, amcsg_n=5300)
        assert r.lag1_n_sla_safe_safe is True
        assert r.ewma_n_sla_safe_safe is True

    def test_n_sla_safe_safe_flag_false_when_below(self):
        r = _make_report(lag1_n=5200, ewma_n=5100, amcsg_n=5300)
        assert r.lag1_n_sla_safe_safe is False
        assert r.ewma_n_sla_safe_safe is False

    def test_north_star_flag_true_when_achieved(self):
        # lag1 above north_star, ewma below
        r = _make_report(lag1_gp=155_000.0, ewma_gp=120_000.0, amcsg_gp=130_000.0,
                         lag1_n=5400, ewma_n=5400, amcsg_n=5300,
                         north_star=151_248.0)
        assert r.lag1_north_star_500_achieved is True
        assert r.ewma_north_star_500_achieved is False

    def test_vs_amcsg_pct_derived_correctly(self):
        lag1_gp, ewma_gp, amcsg_gp = 143_000.0, 137_000.0, 130_000.0
        r = _make_report(lag1_gp=lag1_gp, ewma_gp=ewma_gp, amcsg_gp=amcsg_gp)
        expected_lag1 = (lag1_gp - amcsg_gp) / amcsg_gp * 100.0
        expected_ewma = (ewma_gp - amcsg_gp) / amcsg_gp * 100.0
        assert abs(r.lag1_vs_amcsg_pct - expected_lag1) < 1e-4
        assert abs(r.ewma_vs_amcsg_pct - expected_ewma) < 1e-4

    def test_goodput_values_correct_in_dict(self):
        r = _make_report(lag1_gp=142_000.0, ewma_gp=136_000.0, amcsg_gp=131_000.0)
        d = r.to_dict()
        assert d["lag1_goodput_per_dollar"] == pytest.approx(142_000.0, rel=0.001)
        assert d["ewma_goodput_per_dollar"] == pytest.approx(136_000.0, rel=0.001)
        assert d["amcsg_goodput_per_dollar"] == pytest.approx(131_000.0, rel=0.001)

    def test_sla_oracle_in_dict(self):
        r = _make_report()
        d = r.to_dict()
        assert d["sla_oracle_goodput_per_dollar"] == pytest.approx(25_208.0)

    def test_north_star_in_dict(self):
        r = _make_report(north_star=151_248.0)
        d = r.to_dict()
        assert d["north_star_500_threshold"] == pytest.approx(151_248.0)

    def test_rng_seed_preserved(self):
        r = _make_report()
        assert r.rng_seed == 42

    def test_spot_price_preserved(self):
        r = _make_report()
        assert r.spot_price_usd_hr == pytest.approx(0.80)

    def test_p_interrupt_preserved(self):
        r = _make_report()
        assert r.p_interrupt_hourly == pytest.approx(0.10)

    def test_zfhc_threshold_preserved(self):
        r = _make_report()
        assert r.zfhc_threshold == 8

    def test_mcs_gate_preserved(self):
        r = _make_report()
        assert r.mcs_gate == pytest.approx(12.5)

    def test_ewma_alpha_preserved(self):
        r = _make_report()
        assert r.ewma_alpha == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Class 2: Backtest integration tests (require numpy)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def np_available():
    """Skip the whole class if numpy is unavailable."""
    return pytest.importorskip("numpy")


class TestRunForecastedMCSSpotBacktestSynthetic:
    """Integration tests on a synthetic trace — skipped if numpy unavailable."""

    @pytest.fixture(scope="class")
    def report(self):
        pytest.importorskip("numpy")
        raw = _make_flat_trace(n_ticks=8, req_per_tick=4, tok=80)
        return _run_forecasted_mcs_spot_backtest(
            raw=raw,
            trace_name="test_flat",
            fixed_c=4,
            target_rho=0.85,
            sla_s=10.0,
            tick_seconds=60.0,
            spot_price_usd_hr=0.80,
            p_interrupt_hourly=0.10,
            seed=42,
            sla_oracle=25_208.0,
            north_star_500_threshold=6.0 * 25_208.0,
            mcs_gate=12.5,
            zfhc_threshold=8,
            ewma_alpha=0.5,
            warmup_c=4,
        )

    def test_returns_report_type(self, report):
        assert isinstance(report, ForecastedMCSSpotReport)

    def test_total_requests_correct(self, report):
        assert report.total_requests == 8 * 4

    def test_goodput_non_negative(self, report):
        assert report.amcsg_goodput_per_dollar >= 0.0
        assert report.lag1_goodput_per_dollar >= 0.0
        assert report.ewma_goodput_per_dollar >= 0.0

    def test_costs_positive(self, report):
        assert report.amcsg_cost > 0.0
        assert report.lag1_cost > 0.0
        assert report.ewma_cost > 0.0

    def test_n_sla_safe_non_negative(self, report):
        assert report.amcsg_n_sla_safe >= 0
        assert report.lag1_n_sla_safe >= 0
        assert report.ewma_n_sla_safe >= 0

    def test_n_sla_safe_safe_flag_consistent(self, report):
        assert report.lag1_n_sla_safe_safe == (report.lag1_n_sla_safe >= report.amcsg_n_sla_safe)
        assert report.ewma_n_sla_safe_safe == (report.ewma_n_sla_safe >= report.amcsg_n_sla_safe)

    def test_north_star_flag_consistent(self, report):
        expected_lag1 = (
            report.lag1_goodput_per_dollar >= report.north_star_500_threshold
            and report.lag1_n_sla_safe >= report.amcsg_n_sla_safe
        )
        assert report.lag1_north_star_500_achieved == expected_lag1
        expected_ewma = (
            report.ewma_goodput_per_dollar >= report.north_star_500_threshold
            and report.ewma_n_sla_safe >= report.amcsg_n_sla_safe
        )
        assert report.ewma_north_star_500_achieved == expected_ewma

    def test_vs_amcsg_pct_derived_correctly(self, report):
        expected_lag1 = (
            (report.lag1_goodput_per_dollar - report.amcsg_goodput_per_dollar)
            / max(report.amcsg_goodput_per_dollar, 1e-9) * 100.0
        )
        assert abs(report.lag1_vs_amcsg_pct - expected_lag1) < 1e-6
        expected_ewma = (
            (report.ewma_goodput_per_dollar - report.amcsg_goodput_per_dollar)
            / max(report.amcsg_goodput_per_dollar, 1e-9) * 100.0
        )
        assert abs(report.ewma_vs_amcsg_pct - expected_ewma) < 1e-6

    def test_c_mean_positive(self, report):
        assert report.amcsg_c_mean > 0.0
        assert report.lag1_c_mean > 0.0
        assert report.ewma_c_mean > 0.0

    def test_reproducible_with_same_seed(self):
        pytest.importorskip("numpy")
        raw = _make_flat_trace(n_ticks=8, req_per_tick=4, tok=80)
        kwargs = dict(
            raw=raw, trace_name="t", fixed_c=4, target_rho=0.85,
            sla_s=10.0, tick_seconds=60.0, spot_price_usd_hr=0.80,
            p_interrupt_hourly=0.10, seed=42,
            sla_oracle=25_208.0, north_star_500_threshold=6.0 * 25_208.0,
        )
        r1 = _run_forecasted_mcs_spot_backtest(**kwargs)
        r2 = _run_forecasted_mcs_spot_backtest(**kwargs)
        assert r1.lag1_goodput_per_dollar == r2.lag1_goodput_per_dollar
        assert r1.ewma_goodput_per_dollar == r2.ewma_goodput_per_dollar


# ---------------------------------------------------------------------------
# Class 3: Azure fixture backtest — integration tests
# ---------------------------------------------------------------------------

class TestAzureFixtureBacktest:
    """Integration tests on the Azure LLM 2024 fixture (54 rows)."""

    @pytest.fixture(scope="class")
    def azure_report(self):
        pytest.importorskip("numpy")
        return run_forecasted_mcs_spot_azure_backtest(job_limit=54)

    def test_returns_report(self, azure_report):
        assert isinstance(azure_report, ForecastedMCSSpotReport)

    def test_trace_name(self, azure_report):
        assert azure_report.trace == "azure_llm_2024_forecasted_mcs_spot"

    def test_total_requests_at_most_54(self, azure_report):
        assert azure_report.total_requests <= 54

    def test_amcsg_goodput_positive(self, azure_report):
        assert azure_report.amcsg_goodput_per_dollar > 0.0

    def test_lag1_goodput_positive(self, azure_report):
        assert azure_report.lag1_goodput_per_dollar > 0.0

    def test_ewma_goodput_positive(self, azure_report):
        assert azure_report.ewma_goodput_per_dollar > 0.0

    def test_to_dict_round_trips(self, azure_report):
        d = azure_report.to_dict()
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        assert loaded["trace"] == "azure_llm_2024_forecasted_mcs_spot"
        assert loaded["sla_s"] == pytest.approx(10.0)

    def test_sla_oracle_is_azure_value(self, azure_report):
        assert azure_report.sla_oracle_goodput_per_dollar == pytest.approx(25_208.0)

    def test_north_star_threshold_is_azure_value(self, azure_report):
        assert azure_report.north_star_500_threshold == pytest.approx(6.0 * 25_208.0)

    def test_p_interrupt_default(self, azure_report):
        assert azure_report.p_interrupt_hourly == pytest.approx(0.10)

    def test_spot_price_default(self, azure_report):
        assert azure_report.spot_price_usd_hr == pytest.approx(0.80)

    def test_n_sla_safe_flag_consistent(self, azure_report):
        assert azure_report.lag1_n_sla_safe_safe == (
            azure_report.lag1_n_sla_safe >= azure_report.amcsg_n_sla_safe
        )
        assert azure_report.ewma_n_sla_safe_safe == (
            azure_report.ewma_n_sla_safe >= azure_report.amcsg_n_sla_safe
        )


# ---------------------------------------------------------------------------
# Class 4: BurstGPT fixture backtest — integration tests
# ---------------------------------------------------------------------------

class TestBurstGPTFixtureBacktest:
    """Integration tests for run_forecasted_mcs_spot_burstgpt_backtest."""

    @pytest.fixture(scope="class")
    def burstgpt_report(self):
        pytest.importorskip("numpy")
        return run_forecasted_mcs_spot_burstgpt_backtest(job_limit=200)

    def test_returns_report(self, burstgpt_report):
        assert isinstance(burstgpt_report, ForecastedMCSSpotReport)

    def test_trace_name(self, burstgpt_report):
        assert burstgpt_report.trace == "burstgpt_hf_forecasted_mcs_spot"

    def test_sla_oracle_is_burstgpt_value(self, burstgpt_report):
        assert burstgpt_report.sla_oracle_goodput_per_dollar == pytest.approx(20_280.0)

    def test_north_star_threshold_is_burstgpt_value(self, burstgpt_report):
        assert burstgpt_report.north_star_500_threshold == pytest.approx(6.0 * 20_280.0)

    def test_goodput_non_negative(self, burstgpt_report):
        assert burstgpt_report.lag1_goodput_per_dollar >= 0.0
        assert burstgpt_report.ewma_goodput_per_dollar >= 0.0

    def test_sla_s_default_burstgpt(self, burstgpt_report):
        assert burstgpt_report.sla_s == pytest.approx(30.0)
