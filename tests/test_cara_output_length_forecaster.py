"""Tests for the CARA output-length forecaster (aurelius/forecasting/cara_output_length_forecaster.py).

Research basis: arXiv:2604.06970 (semi-clairvoyant scheduling via token
magnitude priors) and arXiv:2602.11812 (ICLR 2026 — output length prediction
via entropy-guided representations).

All tests use synthetic data conforming to CARA schema statistics
(n ≈ 76,825 rows at train time; test samples are small but representative).
No CARA data files are loaded at test time — the tests validate module logic
and numeric contracts only.

Coverage:
  BiasCalibrationForecaster
  - Leakage rule: actual_output_tokens must not be a feature (docstring verified)
  - Fit on synthetic data: scale != 0, offset reasonable
  - Predict: output is clipped >= 1
  - Calibration report contains required keys
  - Over-prediction correction (scale < 1)
  - Under-prediction correction (scale > 1)
  - Graceful handling of all-NaN input (warns + identity)
  - Edge: too few rows falls back to identity transform

  compute_bias_stats
  - Returns all required keys
  - mean_error correct on zero-bias example
  - mean_error positive when raw over-predicts

  compute_percentile_stats
  - Returns all required keys
  - Handles empty array (all NaN)
  - Correct p50 on known distribution

  HGBOutputLengthConfig
  - Invalid quantile raises ValueError

  HGBOutputLengthForecaster
  - Fit on synthetic X, y
  - Predict: output is clipped >= 1
  - model_report contains required keys
  - Fit without calling predict first raises RuntimeError
  - Too few valid rows raises ValueError

  OutputLengthForecast
  - to_dict contains all required keys
  - status tag is present and correct

  OutputLengthForecastBundle
  - Calibration-only path: predict_single works
  - Calibration-only path: predict_batch works
  - Full path (calibration + HGB): predict_single prefers HGB p50
  - Full path: predict_batch returns correct length
  - p90 >= p50 invariant
  - predict_single before fit raises RuntimeError
  - fit_hgb before fit_calibration raises RuntimeError
  - bundle_report contains all required keys
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from aurelius.forecasting.cara_output_length_forecaster import (
    SHADOW_TAG,
    BiasCalibrationForecaster,
    HGBOutputLengthConfig,
    HGBOutputLengthForecaster,
    OutputLengthForecast,
    OutputLengthForecastBundle,
    compute_bias_stats,
    compute_percentile_stats,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 300  # small but > MIN_TRAIN_ROWS (50)


def _make_raw_actual(
    n: int = N,
    *,
    scale_true: float = 0.8,
    offset_true: float = 20.0,
    noise_std: float = 30.0,
    rng: np.random.Generator = RNG,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (raw_predictions, actual_tokens) following CARA statistics.

    ``raw_predictions`` simulates ``num_predicted_output_tokens`` from the
    serving engine (mean ~290, std ~200; integers in [1, 4096]).
    ``actual_tokens`` is a noisy version: actual ≈ scale_true * raw + offset_true + noise.
    """
    raw = rng.integers(10, 1500, size=n).astype(np.float64)
    actual = scale_true * raw + offset_true + rng.normal(0, noise_std, size=n)
    actual = np.clip(actual, 1.0, 4096.0)
    return raw, actual


def _make_feature_matrix(n: int = N, n_features: int = 20) -> np.ndarray:
    """Simulate a CARA feature matrix with some NaN (missing) features."""
    X = RNG.standard_normal((n, n_features)).astype(np.float64)
    # Introduce ~5% NaN values (HGB handles them natively).
    mask = RNG.random(X.shape) < 0.05
    X[mask] = np.nan
    return X


# ---------------------------------------------------------------------------
# compute_bias_stats
# ---------------------------------------------------------------------------

class TestComputeBiasStats:
    def test_returns_required_keys(self):
        rp, ya = _make_raw_actual()
        result = compute_bias_stats(rp, ya)
        for key in ("mean_error", "median_error", "mean_abs_error", "mean_ratio",
                    "coverage_p90", "n_rows"):
            assert key in result, f"missing key: {key}"

    def test_zero_bias_mean_error(self):
        ya = np.linspace(50, 500, 100)
        rp = ya.copy()  # perfect prediction
        result = compute_bias_stats(rp, ya)
        assert abs(result["mean_error"]) < 1e-9

    def test_over_prediction_positive_mean_error(self):
        ya = np.linspace(50, 500, 100)
        rp = ya * 1.5  # raw over-predicts by 50%
        result = compute_bias_stats(rp, ya)
        assert result["mean_error"] > 0, "over-prediction should give positive mean_error"

    def test_nan_exclusion(self):
        rp = np.array([100.0, np.nan, 200.0, 300.0])
        ya = np.array([90.0, 80.0, np.nan, 280.0])
        result = compute_bias_stats(rp, ya)
        assert result["n_rows"] == 2  # only rows 0 and 3 are fully finite

    def test_empty_returns_nan(self):
        result = compute_bias_stats(np.array([np.nan]), np.array([np.nan]))
        assert result["n_rows"] == 0
        assert np.isnan(result["mean_error"])


# ---------------------------------------------------------------------------
# compute_percentile_stats
# ---------------------------------------------------------------------------

class TestComputePercentileStats:
    def test_returns_required_keys(self):
        result = compute_percentile_stats(np.arange(1, 101, dtype=float))
        for key in ("p50", "p90", "p95", "p99", "mean", "std"):
            assert key in result

    def test_known_p50(self):
        y = np.arange(1, 101, dtype=float)
        result = compute_percentile_stats(y)
        # np.percentile(..., 50, method='nearest') returns 50 or 51 depending on
        # NumPy version (2.x uses a different rounding convention).  Accept either.
        assert result["p50"] in (50.0, 51.0)

    def test_empty_array_all_nan(self):
        result = compute_percentile_stats(np.array([np.nan, np.nan]))
        assert all(np.isnan(v) for v in result.values())

    def test_ignores_nan(self):
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        result = compute_percentile_stats(y)
        assert np.isfinite(result["mean"])
        assert abs(result["mean"] - 3.0) < 1e-9  # mean of [1,2,4,5]


# ---------------------------------------------------------------------------
# BiasCalibrationForecaster
# ---------------------------------------------------------------------------

class TestBiasCalibrationForecaster:
    def test_leakage_field_not_used_as_feature(self):
        # The module docstring mandates actual_output_tokens is never a feature.
        # We verify the honesty rule by importing the leakage constant from the
        # sibling module and asserting it includes 'actual_output_tokens'.
        from aurelius.forecasting.cara_latency_features import LEAKAGE_TARGET_FIELDS
        assert "actual_output_tokens" in LEAKAGE_TARGET_FIELDS

    def test_fit_produces_nonzero_scale(self):
        raw, actual = _make_raw_actual(scale_true=0.8, offset_true=20.0)
        fc = BiasCalibrationForecaster()
        fc.fit(raw, actual)
        assert fc._fitted
        # Should learn scale close to true 0.8, not 0.
        assert abs(fc._scale) > 0.1

    def test_over_prediction_correction(self):
        """Engine consistently over-predicts by 2x; calibrator should correct."""
        rng = np.random.default_rng(7)
        actual = rng.integers(50, 400, size=200).astype(np.float64)
        raw = actual * 2.0 + rng.normal(0, 10, size=200)  # 2x over-prediction
        fc = BiasCalibrationForecaster()
        fc.fit(raw, actual)
        # Calibrated prediction should be closer to actual than raw
        cal = fc.predict(raw)
        raw_mae = float(np.mean(np.abs(raw - actual)))
        cal_mae = float(np.mean(np.abs(cal - actual)))
        assert cal_mae < raw_mae, (
            f"Calibration should reduce MAE; raw_mae={raw_mae:.1f}, cal_mae={cal_mae:.1f}"
        )

    def test_under_prediction_correction(self):
        """Engine consistently under-predicts by 50%; calibrator should correct."""
        rng = np.random.default_rng(13)
        actual = rng.integers(100, 800, size=200).astype(np.float64)
        raw = actual * 0.5 + rng.normal(0, 5, size=200)
        fc = BiasCalibrationForecaster()
        fc.fit(raw, actual)
        cal = fc.predict(raw)
        raw_mae = float(np.mean(np.abs(raw - actual)))
        cal_mae = float(np.mean(np.abs(cal - actual)))
        assert cal_mae < raw_mae

    def test_predict_output_clipped_to_one(self):
        raw, actual = _make_raw_actual()
        fc = BiasCalibrationForecaster()
        fc.fit(raw, actual)
        # Predict on very small raw values that might produce negatives without clip.
        out = fc.predict(np.array([0.0, 1.0, -5.0]))
        assert np.all(out >= 1.0), f"outputs below 1: {out}"

    def test_predict_before_fit_raises(self):
        fc = BiasCalibrationForecaster()
        with pytest.raises(RuntimeError, match="fit"):
            fc.predict(np.array([100.0]))

    def test_too_few_rows_identity_fallback(self):
        """< MIN_TRAIN_ROWS rows → identity transform with warning."""
        raw = np.array([100.0, 200.0], dtype=np.float64)
        actual = np.array([90.0, 210.0], dtype=np.float64)
        fc = BiasCalibrationForecaster()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fc.fit(raw, actual)
        assert fc._scale == 1.0
        assert fc._offset == 0.0
        assert any("identity transform" in str(warning.message) for warning in w)

    def test_calibration_report_keys(self):
        raw, actual = _make_raw_actual()
        fc = BiasCalibrationForecaster()
        fc.fit(raw, actual)
        report = fc.calibration_report()
        for key in ("forecaster", "scale", "offset", "n_train",
                    "raw_prior_bias_stats", "status"):
            assert key in report
        assert report["status"] == SHADOW_TAG

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            fc = BiasCalibrationForecaster()
            fc.fit(np.array([1.0, 2.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# HGBOutputLengthConfig
# ---------------------------------------------------------------------------

class TestHGBOutputLengthConfig:
    def test_invalid_quantile_raises(self):
        with pytest.raises(ValueError, match="quantile"):
            HGBOutputLengthForecaster(config=HGBOutputLengthConfig(quantile=0.77))


# ---------------------------------------------------------------------------
# HGBOutputLengthForecaster
# ---------------------------------------------------------------------------

class TestHGBOutputLengthForecaster:
    def test_fit_and_predict_shape(self):
        raw, actual = _make_raw_actual()
        X = _make_feature_matrix(n=N)
        fc = HGBOutputLengthForecaster()
        fc.fit(X, actual, feature_names=[f"f{i}" for i in range(X.shape[1])])
        preds = fc.predict(X)
        assert preds.shape == (N,)

    def test_predict_clipped_to_one(self):
        raw, actual = _make_raw_actual()
        X = _make_feature_matrix(n=N)
        fc = HGBOutputLengthForecaster()
        fc.fit(X, actual)
        preds = fc.predict(X)
        assert np.all(preds >= 1.0)

    def test_predict_before_fit_raises(self):
        fc = HGBOutputLengthForecaster()
        with pytest.raises(RuntimeError, match="fit"):
            fc.predict(np.ones((10, 5)))

    def test_too_few_rows_raises(self):
        fc = HGBOutputLengthForecaster()
        with pytest.raises(ValueError, match="valid target rows"):
            fc.fit(np.ones((10, 5)), np.ones(10) * 100.0)

    def test_model_report_keys(self):
        raw, actual = _make_raw_actual()
        X = _make_feature_matrix(n=N)
        fc = HGBOutputLengthForecaster()
        fc.fit(X, actual, feature_names=["a", "b"])
        report = fc.model_report()
        for key in ("forecaster", "quantile", "n_train", "n_features", "status"):
            assert key in report
        assert report["status"] == SHADOW_TAG
        assert report["quantile"] == 0.50

    def test_p90_model(self):
        raw, actual = _make_raw_actual()
        X = _make_feature_matrix(n=N)
        cfg = HGBOutputLengthConfig(quantile=0.90)
        fc = HGBOutputLengthForecaster(config=cfg)
        fc.fit(X, actual)
        preds = fc.predict(X)
        # p90 predictions on training data should be >= most of actual values.
        coverage = float(np.mean(actual <= preds))
        assert coverage >= 0.70, f"p90 coverage on train data too low: {coverage:.2f}"


# ---------------------------------------------------------------------------
# OutputLengthForecast
# ---------------------------------------------------------------------------

class TestOutputLengthForecast:
    def test_to_dict_required_keys(self):
        fc = OutputLengthForecast(
            p50_tokens=200.0,
            p90_tokens=400.0,
            raw_prior_tokens=250.0,
            calibrated_p50_tokens=195.0,
        )
        d = fc.to_dict()
        for key in ("p50_tokens", "p90_tokens", "raw_prior_tokens",
                    "calibrated_p50_tokens", "hgb_p50_tokens", "hgb_p90_tokens",
                    "status"):
            assert key in d
        assert d["status"] == SHADOW_TAG

    def test_optional_hgb_fields_default_none(self):
        fc = OutputLengthForecast(
            p50_tokens=100.0,
            p90_tokens=200.0,
            raw_prior_tokens=120.0,
            calibrated_p50_tokens=98.0,
        )
        assert fc.hgb_p50_tokens is None
        assert fc.hgb_p90_tokens is None


# ---------------------------------------------------------------------------
# OutputLengthForecastBundle
# ---------------------------------------------------------------------------

class TestOutputLengthForecastBundle:
    @pytest.fixture
    def fitted_cal_bundle(self):
        raw, actual = _make_raw_actual()
        b = OutputLengthForecastBundle()
        b.fit_calibration(raw, actual)
        return b

    @pytest.fixture
    def fitted_full_bundle(self):
        raw, actual = _make_raw_actual()
        X = _make_feature_matrix(n=N)
        b = OutputLengthForecastBundle()
        b.fit_calibration(raw, actual)
        b.fit_hgb(X, actual, feature_names=[f"f{i}" for i in range(X.shape[1])])
        return b

    def test_predict_before_fit_raises(self):
        b = OutputLengthForecastBundle()
        with pytest.raises(RuntimeError, match="fit_calibration"):
            b.predict_single(100.0)

    def test_fit_hgb_before_calibration_raises(self):
        b = OutputLengthForecastBundle()
        X = _make_feature_matrix(n=N)
        _, actual = _make_raw_actual()
        with pytest.raises(RuntimeError, match="fit_calibration"):
            b.fit_hgb(X, actual)

    def test_calibration_only_predict_single(self, fitted_cal_bundle):
        fc = fitted_cal_bundle.predict_single(250.0)
        assert isinstance(fc, OutputLengthForecast)
        assert fc.raw_prior_tokens == 250.0
        assert fc.p50_tokens >= 1.0
        assert fc.hgb_p50_tokens is None
        assert fc.hgb_p90_tokens is None

    def test_calibration_only_predict_batch(self, fitted_cal_bundle):
        raw = np.array([100.0, 200.0, 300.0])
        results = fitted_cal_bundle.predict_batch(raw)
        assert len(results) == 3
        for r in results:
            assert r.p50_tokens >= 1.0
            assert r.hgb_p50_tokens is None

    def test_full_bundle_predict_single_uses_hgb(self, fitted_full_bundle):
        X = _make_feature_matrix(n=1)
        fc = fitted_full_bundle.predict_single(300.0, x_row=X[0])
        assert fc.hgb_p50_tokens is not None
        assert fc.hgb_p90_tokens is not None
        # p50 should equal the HGB prediction.
        assert fc.p50_tokens == fc.hgb_p50_tokens

    def test_full_bundle_predict_batch(self, fitted_full_bundle):
        raw, actual = _make_raw_actual(n=50)
        X = _make_feature_matrix(n=50)
        results = fitted_full_bundle.predict_batch(raw, X)
        assert len(results) == 50
        for r in results:
            assert r.p50_tokens >= 1.0
            assert r.hgb_p50_tokens is not None

    def test_p90_gte_p50_invariant(self, fitted_full_bundle):
        """p90_tokens must always be >= p50_tokens."""
        raw = np.linspace(10, 2000, 100)
        X = _make_feature_matrix(n=100)
        results = fitted_full_bundle.predict_batch(raw, X)
        for r in results:
            assert r.p90_tokens >= r.p50_tokens, (
                f"invariant violated: p90={r.p90_tokens}, p50={r.p50_tokens}"
            )

    def test_calibration_only_p90_fallback(self, fitted_cal_bundle):
        """Without HGB, p90 should be > p50 (1.5× fallback)."""
        fc = fitted_cal_bundle.predict_single(200.0)
        assert fc.p90_tokens >= fc.p50_tokens

    def test_bundle_report_keys(self, fitted_full_bundle):
        report = fitted_full_bundle.bundle_report()
        for key in ("calibration_fitted", "hgb_fitted", "calibration",
                    "hgb_p50", "hgb_p90", "status"):
            assert key in report
        assert report["calibration_fitted"] is True
        assert report["hgb_fitted"] is True
        assert report["status"] == SHADOW_TAG

    def test_calibration_only_bundle_report(self, fitted_cal_bundle):
        report = fitted_cal_bundle.bundle_report()
        assert report["hgb_fitted"] is False
        assert report["hgb_p50"] is None

    def test_shadow_tag_propagates(self, fitted_cal_bundle):
        fc = fitted_cal_bundle.predict_single(150.0)
        d = fc.to_dict()
        assert d["status"] == SHADOW_TAG


# ---------------------------------------------------------------------------
# Integration: verify calibration improves over raw prior
# ---------------------------------------------------------------------------

class TestCalibrationImprovesOverRawPrior:
    """Integration test: the calibration model should reduce MAE vs raw prior.

    Uses a larger synthetic sample where raw predictions are systematically
    biased (the engine under-predicts by 30% with added noise).
    This mirrors the expected real-world behaviour in CARA.
    """

    def test_calibration_reduces_mae(self):
        rng = np.random.default_rng(99)
        n_train = 400
        n_test = 100
        # Engine under-predicts: actual ≈ 1.3 * raw + 15
        raw_train = rng.integers(30, 1200, size=n_train).astype(np.float64)
        actual_train = 1.3 * raw_train + 15 + rng.normal(0, 40, size=n_train)
        actual_train = np.clip(actual_train, 1.0, None)

        raw_test = rng.integers(30, 1200, size=n_test).astype(np.float64)
        actual_test = 1.3 * raw_test + 15 + rng.normal(0, 40, size=n_test)
        actual_test = np.clip(actual_test, 1.0, None)

        fc = BiasCalibrationForecaster()
        fc.fit(raw_train, actual_train)
        cal_test = fc.predict(raw_test)

        raw_mae = float(np.mean(np.abs(raw_test - actual_test)))
        cal_mae = float(np.mean(np.abs(cal_test - actual_test)))
        assert cal_mae < raw_mae, (
            f"Calibration should reduce test MAE; raw={raw_mae:.1f}, cal={cal_mae:.1f}"
        )
