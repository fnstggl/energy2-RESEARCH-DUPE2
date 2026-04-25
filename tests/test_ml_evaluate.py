"""Tests for aurelius/ml/evaluate.py — Phase 2 ML evaluation.

Coverage:
  Unit tests  — each metric function in isolation + edge cases
  Integration — evaluate_price_forecaster / evaluate_carbon_forecaster with
                a real fitted forecaster on a temporal holdout split
  Save/Load   — PriceQuantileForecaster and CarbonQuantileForecaster
                joblib serialisation roundtrip
  Promotion   — should_promote() gate logic
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aurelius.ml.evaluate import (
    ForecastEvalMetrics,
    compute_calibration_error,
    compute_downside_risk,
    compute_mae,
    compute_mape,
    compute_p50_bias,
    compute_p90_coverage,
    compute_rmse,
    evaluate_carbon_forecaster,
    evaluate_price_forecaster,
    should_promote,
)
from aurelius.models import CarbonIntensity, EnergyPrice

UTC = timezone.utc
T0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(
    n_hours: int = 48,
    region: str = "us-west",
    base: float = 50.0,
    start: datetime | None = None,
) -> list[EnergyPrice]:
    start = start or T0
    return [
        EnergyPrice(
            timestamp=start + timedelta(hours=h),
            region=region,
            # Mild diurnal pattern: cheaper at night, expensive midday
            price_per_mwh=base + 10 * math.sin(2 * math.pi * (h % 24) / 24),
        )
        for h in range(n_hours)
    ]


def _make_carbon(
    n_hours: int = 48,
    region: str = "us-west",
    base: float = 350.0,
    start: datetime | None = None,
) -> list[CarbonIntensity]:
    start = start or T0
    return [
        CarbonIntensity(
            timestamp=start + timedelta(hours=h),
            region=region,
            gco2_per_kwh=base + 20 * math.sin(2 * math.pi * (h % 24) / 24),
        )
        for h in range(n_hours)
    ]


def _make_metrics(**kwargs) -> ForecastEvalMetrics:
    defaults = dict(
        n_samples=100,
        mape=5.0,
        rmse=3.0,
        mae=2.5,
        p50_bias=0.1,
        p90_coverage=0.91,
        calibration_error=0.01,
        downside_risk=0.5,
        region="us-west",
    )
    defaults.update(kwargs)
    return ForecastEvalMetrics(**defaults)


# ============================================================================
# UNIT TESTS — individual metric functions
# ============================================================================


class TestComputeMAPE:
    def test_perfect_forecast_is_zero(self):
        actuals = [100.0, 200.0, 300.0]
        preds = [100.0, 200.0, 300.0]
        assert compute_mape(actuals, preds) == pytest.approx(0.0)

    def test_known_value(self):
        # 1 point: actual=100, pred=90  → |error|=10, relative=0.10, MAPE=10%
        assert compute_mape([100.0], [90.0]) == pytest.approx(10.0)

    def test_two_points_mean(self):
        # point1: 10% error, point2: 5% error → MAPE = 7.5%
        assert compute_mape([100.0, 200.0], [90.0, 190.0]) == pytest.approx(7.5)

    def test_near_zero_actuals_excluded(self):
        # Second actual is near-zero → only the first point contributes
        result = compute_mape([100.0, 0.0], [90.0, 50.0])
        assert result == pytest.approx(10.0)

    def test_all_near_zero_actuals_returns_nan(self):
        result = compute_mape([0.0, 0.0], [1.0, 2.0])
        assert math.isnan(result)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_mape([1.0, 2.0], [1.0])

    def test_empty_returns_nan(self):
        assert math.isnan(compute_mape([], []))


class TestComputeRMSE:
    def test_perfect_forecast(self):
        assert compute_rmse([10.0, 20.0], [10.0, 20.0]) == pytest.approx(0.0)

    def test_known_value(self):
        # errors: [3, 4] → MSE = (9+16)/2 = 12.5 → RMSE = 3.536
        result = compute_rmse([3.0, 4.0], [0.0, 0.0])
        assert result == pytest.approx(math.sqrt(12.5))

    def test_empty_returns_nan(self):
        assert math.isnan(compute_rmse([], []))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_rmse([1.0], [1.0, 2.0])


class TestComputeMAE:
    def test_perfect_forecast(self):
        assert compute_mae([5.0, 10.0], [5.0, 10.0]) == pytest.approx(0.0)

    def test_known_value(self):
        assert compute_mae([10.0, 20.0], [8.0, 25.0]) == pytest.approx(3.5)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_mae([], []))


class TestComputeP50Bias:
    def test_unbiased(self):
        assert compute_p50_bias([10.0, 20.0], [10.0, 20.0]) == pytest.approx(0.0)

    def test_positive_bias(self):
        # actual > pred → positive bias (under-forecast)
        assert compute_p50_bias([110.0], [100.0]) == pytest.approx(10.0)

    def test_negative_bias(self):
        # actual < pred → negative bias (over-forecast)
        assert compute_p50_bias([90.0], [100.0]) == pytest.approx(-10.0)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_p50_bias([], []))


class TestComputeP90Coverage:
    def test_all_covered(self):
        # All actuals ≤ p90 predictions
        assert compute_p90_coverage([80.0, 90.0], [90.0, 100.0]) == pytest.approx(1.0)

    def test_none_covered(self):
        assert compute_p90_coverage([110.0, 120.0], [100.0, 110.0]) == pytest.approx(0.0)

    def test_half_covered(self):
        assert compute_p90_coverage([100.0, 200.0], [150.0, 150.0]) == pytest.approx(0.5)

    def test_exactly_at_p90_is_covered(self):
        assert compute_p90_coverage([100.0], [100.0]) == pytest.approx(1.0)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_p90_coverage([], []))


class TestComputeCalibrationError:
    def test_perfect_calibration(self):
        # 90 out of 100 covered → calibration_error = 0
        # 90 actuals are below 900, 10 are above 900
        actuals = [1.0] * 90 + [1000.0] * 10
        preds = [900.0] * 100
        # actuals ≤ 900: 90 points → coverage = 0.90 → error = 0
        assert compute_calibration_error(actuals, preds) == pytest.approx(0.0)

    def test_over_conservative(self):
        # All covered → coverage = 1.0 → error = 0.10
        actuals = [1.0, 2.0, 3.0]
        preds = [100.0, 100.0, 100.0]
        assert compute_calibration_error(actuals, preds) == pytest.approx(0.10)

    def test_under_conservative(self):
        # None covered → coverage = 0.0 → error = 0.90
        actuals = [200.0]
        preds = [100.0]
        assert compute_calibration_error(actuals, preds) == pytest.approx(0.90)


class TestComputeDownsideRisk:
    def test_all_covered_zero_risk(self):
        assert compute_downside_risk([80.0, 90.0], [100.0, 100.0]) == pytest.approx(0.0)

    def test_known_exceedance(self):
        # actual=120, p90=100 → exceedance=20; actual=80 → 0; mean=10
        assert compute_downside_risk([120.0, 80.0], [100.0, 100.0]) == pytest.approx(10.0)

    def test_exactly_at_p90_is_zero(self):
        assert compute_downside_risk([100.0], [100.0]) == pytest.approx(0.0)

    def test_empty_returns_nan(self):
        assert math.isnan(compute_downside_risk([], []))


# ============================================================================
# UNIT TESTS — ForecastEvalMetrics
# ============================================================================


class TestForecastEvalMetrics:
    def test_to_dict_roundtrip(self):
        m = _make_metrics()
        d = m.to_dict()
        assert d["n_samples"] == m.n_samples
        assert d["mape"] == m.mape
        assert d["region"] == m.region

    def test_is_valid_normal(self):
        assert _make_metrics().is_valid() is True

    def test_is_valid_zero_samples(self):
        assert _make_metrics(n_samples=0).is_valid() is False

    def test_is_valid_nan_mape(self):
        assert _make_metrics(mape=float("nan")).is_valid() is False

    def test_is_valid_inf_mape(self):
        assert _make_metrics(mape=float("inf")).is_valid() is False


# ============================================================================
# INTEGRATION — evaluate_price_forecaster
# ============================================================================


class TestEvaluatePriceForecaster:
    """Integration tests that fit a real PriceQuantileForecaster on synthetic data."""

    @pytest.fixture(scope="class")
    def train_and_holdout_prices(self):
        train = _make_prices(n_hours=168, region="us-west", start=T0)
        holdout_start = T0 + timedelta(hours=168)
        holdout = _make_prices(n_hours=48, region="us-west", start=holdout_start)
        return train, holdout

    @pytest.fixture(scope="class")
    def fitted_price_forecaster(self, train_and_holdout_prices):
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train, _ = train_and_holdout_prices
        config = PriceModelConfig(seed=42, n_estimators=20)
        forecaster = PriceQuantileForecaster(config)
        forecaster.fit(train)
        return forecaster

    def test_returns_metrics_for_each_region(
        self, fitted_price_forecaster, train_and_holdout_prices
    ):
        _, holdout = train_and_holdout_prices
        metrics = evaluate_price_forecaster(fitted_price_forecaster, holdout)
        assert "us-west" in metrics
        assert "_aggregate" in metrics

    def test_n_samples_matches_holdout_size(
        self, fitted_price_forecaster, train_and_holdout_prices
    ):
        _, holdout = train_and_holdout_prices
        metrics = evaluate_price_forecaster(fitted_price_forecaster, holdout)
        assert metrics["us-west"].n_samples == len(holdout)
        assert metrics["_aggregate"].n_samples == len(holdout)

    def test_metrics_are_non_negative(
        self, fitted_price_forecaster, train_and_holdout_prices
    ):
        _, holdout = train_and_holdout_prices
        metrics = evaluate_price_forecaster(fitted_price_forecaster, holdout)
        m = metrics["_aggregate"]
        assert m.mape >= 0
        assert m.rmse >= 0
        assert m.mae >= 0
        assert 0.0 <= m.p90_coverage <= 1.0
        assert m.calibration_error >= 0
        assert m.downside_risk >= 0

    def test_all_metrics_are_valid(
        self, fitted_price_forecaster, train_and_holdout_prices
    ):
        _, holdout = train_and_holdout_prices
        metrics = evaluate_price_forecaster(fitted_price_forecaster, holdout)
        assert metrics["us-west"].is_valid()
        assert metrics["_aggregate"].is_valid()

    def test_empty_holdout_returns_empty(self, fitted_price_forecaster):
        result = evaluate_price_forecaster(fitted_price_forecaster, [])
        assert result == {}

    def test_two_regions_aggregate(self):
        """Aggregate should combine both regions."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train = (
            _make_prices(168, "us-west", start=T0)
            + _make_prices(168, "us-east", base=60.0, start=T0)
        )
        holdout_start = T0 + timedelta(hours=168)
        holdout = (
            _make_prices(24, "us-west", start=holdout_start)
            + _make_prices(24, "us-east", base=60.0, start=holdout_start)
        )
        config = PriceModelConfig(seed=42, n_estimators=20)
        forecaster = PriceQuantileForecaster(config)
        forecaster.fit(train)
        metrics = evaluate_price_forecaster(forecaster, holdout)
        assert "us-west" in metrics
        assert "us-east" in metrics
        assert metrics["_aggregate"].n_samples == 48

    def test_with_recent_prices_for_lag_features(
        self, fitted_price_forecaster, train_and_holdout_prices
    ):
        """Evaluation with recent_prices provided should still complete."""
        train, holdout = train_and_holdout_prices
        recent = train[-48:]  # Last 48 hours of training as "recent"
        metrics = evaluate_price_forecaster(fitted_price_forecaster, holdout, recent)
        assert "_aggregate" in metrics
        assert metrics["_aggregate"].is_valid()


# ============================================================================
# INTEGRATION — evaluate_carbon_forecaster
# ============================================================================


class TestEvaluateCarbonForecaster:
    @pytest.fixture(scope="class")
    def train_and_holdout_carbon(self):
        train = _make_carbon(n_hours=168, region="us-west", start=T0)
        holdout_start = T0 + timedelta(hours=168)
        holdout = _make_carbon(n_hours=48, region="us-west", start=holdout_start)
        return train, holdout

    @pytest.fixture(scope="class")
    def fitted_carbon_forecaster(self, train_and_holdout_carbon):
        from aurelius.forecasting.carbon_model import (
            CarbonQuantileForecaster,
            CarbonModelConfig,
        )
        train, _ = train_and_holdout_carbon
        config = CarbonModelConfig(seed=42, n_estimators=20)
        forecaster = CarbonQuantileForecaster(config)
        forecaster.fit(train)
        return forecaster

    def test_returns_metrics(self, fitted_carbon_forecaster, train_and_holdout_carbon):
        _, holdout = train_and_holdout_carbon
        metrics = evaluate_carbon_forecaster(fitted_carbon_forecaster, holdout)
        assert "us-west" in metrics
        assert "_aggregate" in metrics

    def test_n_samples(self, fitted_carbon_forecaster, train_and_holdout_carbon):
        _, holdout = train_and_holdout_carbon
        metrics = evaluate_carbon_forecaster(fitted_carbon_forecaster, holdout)
        assert metrics["us-west"].n_samples == len(holdout)

    def test_p90_coverage_in_range(
        self, fitted_carbon_forecaster, train_and_holdout_carbon
    ):
        _, holdout = train_and_holdout_carbon
        metrics = evaluate_carbon_forecaster(fitted_carbon_forecaster, holdout)
        cov = metrics["_aggregate"].p90_coverage
        assert 0.0 <= cov <= 1.0

    def test_empty_holdout_returns_empty(self, fitted_carbon_forecaster):
        assert evaluate_carbon_forecaster(fitted_carbon_forecaster, []) == {}


# ============================================================================
# INTEGRATION — save() / load() roundtrip
# ============================================================================


class TestPriceForecasterSaveLoad:
    def test_save_creates_file(self, tmp_path):
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train = _make_prices(168)
        f = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=10))
        f.fit(train)
        out = tmp_path / "price_model.pkl"
        f.save(out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_load_roundtrip_is_fitted(self, tmp_path):
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train = _make_prices(168)
        original = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=10))
        original.fit(train)
        path = tmp_path / "price_model.pkl"
        original.save(path)
        loaded = PriceQuantileForecaster.load(path)
        assert loaded.is_fitted

    def test_load_roundtrip_predictions_match(self, tmp_path):
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train = _make_prices(168)
        original = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=10))
        original.fit(train)
        path = tmp_path / "price_model.pkl"
        original.save(path)
        loaded = PriceQuantileForecaster.load(path)

        ts = [T0 + timedelta(hours=200 + h) for h in range(5)]
        preds_orig = original.predict("us-west", ts)
        preds_loaded = loaded.predict("us-west", ts)
        for a, b in zip(preds_orig, preds_loaded):
            assert a.p50 == pytest.approx(b.p50, abs=1e-6)
            assert a.p90 == pytest.approx(b.p90, abs=1e-6)

    def test_load_missing_key_raises_valueerror(self, tmp_path):
        import joblib
        path = tmp_path / "bad.pkl"
        joblib.dump({"config": None}, path)  # Missing required keys
        with pytest.raises(ValueError, match="missing keys"):
            from aurelius.forecasting.price_model import PriceQuantileForecaster
            PriceQuantileForecaster.load(path)

    def test_load_nonexistent_file_raises(self, tmp_path):
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        with pytest.raises(Exception):  # FileNotFoundError or similar
            PriceQuantileForecaster.load(tmp_path / "nope.pkl")

    def test_save_load_preserves_metadata(self, tmp_path):
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train = _make_prices(168)
        original = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=10))
        original.fit(train)
        path = tmp_path / "price_model.pkl"
        original.save(path)
        loaded = PriceQuantileForecaster.load(path)
        assert loaded.metadata is not None
        assert loaded.metadata.training_samples == len(train)
        assert loaded.metadata.regions == original.metadata.regions

    def test_save_load_unfitted_model(self, tmp_path):
        """An unfitted model can be saved and loaded; is_fitted remains False."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        f = PriceQuantileForecaster()
        path = tmp_path / "unfitted.pkl"
        f.save(path)
        loaded = PriceQuantileForecaster.load(path)
        assert not loaded.is_fitted


class TestCarbonForecasterSaveLoad:
    def test_save_creates_file(self, tmp_path):
        from aurelius.forecasting.carbon_model import CarbonQuantileForecaster, CarbonModelConfig
        train = _make_carbon(168)
        f = CarbonQuantileForecaster(CarbonModelConfig(seed=42, n_estimators=10))
        f.fit(train)
        path = tmp_path / "carbon_model.pkl"
        f.save(path)
        assert path.exists()

    def test_load_roundtrip_predictions_match(self, tmp_path):
        from aurelius.forecasting.carbon_model import CarbonQuantileForecaster, CarbonModelConfig
        train = _make_carbon(168)
        original = CarbonQuantileForecaster(CarbonModelConfig(seed=42, n_estimators=10))
        original.fit(train)
        path = tmp_path / "carbon_model.pkl"
        original.save(path)
        loaded = CarbonQuantileForecaster.load(path)

        ts = [T0 + timedelta(hours=200 + h) for h in range(5)]
        preds_orig = original.predict("us-west", ts)
        preds_loaded = loaded.predict("us-west", ts)
        for a, b in zip(preds_orig, preds_loaded):
            assert a.p50 == pytest.approx(b.p50, abs=1e-6)
            assert a.p90 == pytest.approx(b.p90, abs=1e-6)

    def test_load_missing_key_raises_valueerror(self, tmp_path):
        import joblib
        path = tmp_path / "bad.pkl"
        joblib.dump({"config": None}, path)
        with pytest.raises(ValueError, match="missing keys"):
            from aurelius.forecasting.carbon_model import CarbonQuantileForecaster
            CarbonQuantileForecaster.load(path)


# ============================================================================
# UNIT TESTS — should_promote()
# ============================================================================


class TestShouldPromote:
    def test_promotes_when_mape_improves_sufficiently(self):
        current = _make_metrics(mape=10.0, calibration_error=0.02, downside_risk=1.0)
        candidate = _make_metrics(mape=8.0, calibration_error=0.02, downside_risk=1.0)
        assert should_promote(current, candidate, min_mape_improvement_pct=1.0) is True

    def test_denies_when_mape_improvement_too_small(self):
        current = _make_metrics(mape=10.0)
        candidate = _make_metrics(mape=9.5)  # only 0.5 pp improvement
        assert should_promote(current, candidate, min_mape_improvement_pct=1.0) is False

    def test_denies_when_mape_worsens(self):
        current = _make_metrics(mape=5.0)
        candidate = _make_metrics(mape=6.0)
        assert should_promote(current, candidate) is False

    def test_denies_when_calibration_worsens_too_much(self):
        current = _make_metrics(mape=10.0, calibration_error=0.01)
        candidate = _make_metrics(mape=8.0, calibration_error=0.08)  # +0.07 > 0.05
        assert should_promote(current, candidate, max_calibration_degradation=0.05) is False

    def test_allows_small_calibration_degradation(self):
        current = _make_metrics(mape=10.0, calibration_error=0.01, downside_risk=1.0)
        candidate = _make_metrics(mape=8.0, calibration_error=0.04, downside_risk=1.0)
        assert should_promote(current, candidate, max_calibration_degradation=0.05) is True

    def test_denies_when_downside_risk_worsens_too_much(self):
        current = _make_metrics(mape=10.0, calibration_error=0.01, downside_risk=1.0)
        candidate = _make_metrics(mape=8.0, calibration_error=0.01, downside_risk=1.2)
        assert (
            should_promote(
                current, candidate,
                max_downside_risk_increase_pct=10.0,
            )
            is False
        )

    def test_allows_downside_risk_within_tolerance(self):
        current = _make_metrics(mape=10.0, calibration_error=0.01, downside_risk=1.0)
        candidate = _make_metrics(mape=8.0, calibration_error=0.01, downside_risk=1.05)
        assert (
            should_promote(
                current, candidate,
                max_downside_risk_increase_pct=10.0,
            )
            is True
        )

    def test_skip_downside_check_when_current_risk_is_zero(self):
        current = _make_metrics(mape=10.0, calibration_error=0.01, downside_risk=0.0)
        candidate = _make_metrics(mape=8.0, calibration_error=0.01, downside_risk=5.0)
        # No relative increase check when current = 0
        assert should_promote(current, candidate) is True

    def test_denies_when_current_metrics_invalid(self):
        current = _make_metrics(mape=float("nan"))
        candidate = _make_metrics(mape=5.0)
        assert should_promote(current, candidate) is False

    def test_denies_when_candidate_metrics_invalid(self):
        current = _make_metrics(mape=10.0)
        candidate = _make_metrics(mape=float("nan"))
        assert should_promote(current, candidate) is False

    def test_denies_when_both_invalid(self):
        assert should_promote(
            _make_metrics(n_samples=0),
            _make_metrics(n_samples=0),
        ) is False

    def test_deterministic(self):
        current = _make_metrics(mape=10.0)
        candidate = _make_metrics(mape=8.0)
        r1 = should_promote(current, candidate)
        r2 = should_promote(current, candidate)
        assert r1 == r2


# ============================================================================
# ADVERSARIAL / EDGE CASE TESTS
# ============================================================================


class TestAdversarialEdgeCases:
    def test_mape_excludes_near_zero_actuals_not_negative(self):
        # _MAPE_FLOOR excludes |actual| < 1e-6 (division-by-zero guard).
        # Negative prices like -5.0 have abs(-5.0)=5.0 ≥ 1e-6, so they ARE
        # included in MAPE — both points contribute.
        actuals = [-5.0, 100.0]
        preds = [0.0, 90.0]
        # point1: |(-5 - 0) / (-5)| = 1.00 → 100%
        # point2: |(100 - 90) / 100| = 0.10 → 10%
        # MAPE = (100 + 10) / 2 = 55%
        result = compute_mape(actuals, preds)
        assert result == pytest.approx(55.0)

    def test_mape_truly_near_zero_actual_is_excluded(self):
        # An actual value < 1e-6 (e.g., floating-point rounding artifacts) IS excluded
        actuals = [0.0, 100.0]
        preds = [50.0, 90.0]
        # Only second point qualifies: 10/100 = 10%
        result = compute_mape(actuals, preds)
        assert result == pytest.approx(10.0)

    def test_p90_coverage_boundary_is_inclusive(self):
        # Exactly equal to p90 should be counted as covered
        assert compute_p90_coverage([100.0], [100.0]) == pytest.approx(1.0)
        assert compute_p90_coverage([100.001], [100.0]) == pytest.approx(0.0)

    def test_single_sample_metrics(self):
        metrics = _compute_region_metrics_helper([50.0], [45.0], [55.0], "us-west")
        assert metrics.n_samples == 1
        assert metrics.is_valid()
        assert metrics.p90_coverage == pytest.approx(1.0)  # 50 <= 55

    def test_evaluate_with_unfitted_forecaster_uses_baseline(self):
        """Unfitted forecaster falls back to baseline — metrics should still compute."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster
        unfitted = PriceQuantileForecaster()
        holdout = _make_prices(24, region="us-west")
        metrics = evaluate_price_forecaster(unfitted, holdout)
        assert "_aggregate" in metrics
        # Baseline forecaster always returns same p50; MAPE will be nonzero
        # but should be a finite number
        assert not math.isnan(metrics["_aggregate"].mape)

    def test_unseen_region_uses_baseline_gracefully(self):
        """A holdout region not in training data falls back to baseline, not crash."""
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        train = _make_prices(168, "us-west", start=T0)
        f = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=10))
        f.fit(train)
        # Holdout from a region never seen during training
        holdout_start = T0 + timedelta(hours=168)
        holdout = _make_prices(24, "eu-north", base=40.0, start=holdout_start)
        metrics = evaluate_price_forecaster(f, holdout)
        # Should return metrics for eu-north (via baseline fallback) and aggregate
        assert "eu-north" in metrics
        assert "_aggregate" in metrics
        assert metrics["eu-north"].n_samples == 24
        assert metrics["eu-north"].is_valid()

    def test_evaluate_multi_region_aggregate_sums_samples(self):
        from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
        n_west = 24
        n_east = 36
        train = (
            _make_prices(168, "us-west", start=T0)
            + _make_prices(168, "us-east", base=60.0, start=T0)
        )
        holdout_start = T0 + timedelta(hours=168)
        holdout = (
            _make_prices(n_west, "us-west", start=holdout_start)
            + _make_prices(n_east, "us-east", base=60.0, start=holdout_start)
        )
        f = PriceQuantileForecaster(PriceModelConfig(seed=42, n_estimators=10))
        f.fit(train)
        metrics = evaluate_price_forecaster(f, holdout)
        assert metrics["_aggregate"].n_samples == n_west + n_east


# ---------------------------------------------------------------------------
# Local helper used by adversarial tests (avoids importing private function)
# ---------------------------------------------------------------------------

def _compute_region_metrics_helper(
    actuals, p50_preds, p90_preds, region
) -> ForecastEvalMetrics:
    return ForecastEvalMetrics(
        n_samples=len(actuals),
        mape=compute_mape(actuals, p50_preds),
        rmse=compute_rmse(actuals, p50_preds),
        mae=compute_mae(actuals, p50_preds),
        p50_bias=compute_p50_bias(actuals, p50_preds),
        p90_coverage=compute_p90_coverage(actuals, p90_preds),
        calibration_error=compute_calibration_error(actuals, p90_preds),
        downside_risk=compute_downside_risk(actuals, p90_preds),
        region=region,
    )
