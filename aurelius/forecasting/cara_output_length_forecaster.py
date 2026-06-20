"""CARA output-length forecaster v1 — calibrated actual-output-token prediction.

Builds a calibrated predictor of ``actual_output_tokens`` using only
predict-time features from the CARA dataset. This enables semi-clairvoyant
scheduling: the predicted p50 length serves as a magnitude prior for request
ordering, enabling priority to short requests that are likely to complete fast
(reducing tail latency and improving SLA-safe goodput/$).

Research basis
--------------
- "Scheduling the Unschedulable" (arXiv:2604.06970, SOSP 2026): token
  magnitude priors increase P90 short-request performance by 32% vs FIFO;
  removing priors causes 5.8× p95 regression.
- "Predicting LLM Output Length via Entropy-Guided Representations"
  (arXiv:2602.11812, ICLR 2026): external schedulers can achieve strong
  output-length prediction from request features, reducing MAE by 29%.

Design rules (binding)
-----------------------
- ``actual_output_tokens`` is leakage at prediction time.  It MUST NOT be a
  feature in production models; it is used ONLY as the training label.
- ``num_predicted_output_tokens`` IS allowed at prediction time (it is the
  model's own estimate, emitted at request-arrival time by the serving engine).
- The ``BiasCalibrationForecaster`` wraps ``num_predicted_output_tokens`` with
  a learned scale + offset correction — no additional features required; it is
  a pure calibration/debiasing of the model's self-estimate.
- The ``HGBOutputLengthForecaster`` uses all predict-time features from
  ``cara_latency_features.PREDICT_TIME_NUMERIC_FEATURES`` + categoricals.
- Both forecasters use sklearn; they do NOT call any controller, scheduler,
  or optimizer.
- Shadow-mode only: NO output of this module is wired into any controller
  default. Callers must explicitly opt in.
- Not production-ready: directional / simulator evidence only.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import HuberRegressor
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cara_output_length_forecaster requires scikit-learn; "
        "install via `pip install scikit-learn`"
    ) from e

# Target field (label only — never a feature in predicted_only mode).
TARGET_FIELD = "actual_output_tokens"

# Primary predict-time signal: model's self-estimate.
PRIOR_FIELD = "num_predicted_output_tokens"

# Minimum training rows before the model is considered reliable.
MIN_TRAIN_ROWS = 50

# Quantiles supported by HGBOutputLengthForecaster.
SUPPORTED_QUANTILES = (0.50, 0.90, 0.95)

# Shadow-mode tag emitted in every result dict.
SHADOW_TAG = "shadow_only_not_production_ready"


# ---------------------------------------------------------------------------
# Calibration analysis helpers (pure-function, no model state)
# ---------------------------------------------------------------------------


def compute_bias_stats(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
) -> dict:
    """Characterise systematic bias in ``y_pred`` versus ``y_actual``.

    Returns a dict with:
    - ``mean_error``: mean(pred - actual)  (positive = over-prediction)
    - ``median_error``: median(pred - actual)
    - ``mean_abs_error``: MAE
    - ``mean_ratio``: mean(pred / actual)  (>1 = over-prediction on average)
    - ``coverage_p90``: fraction of rows where actual <= pred * 1.2 + 50
      (loose upper-bound coverage check)
    - ``n_rows``: number of rows included
    """
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    y_actual = np.asarray(y_actual, dtype=np.float64).ravel()
    assert len(y_pred) == len(y_actual), "length mismatch"
    mask = np.isfinite(y_pred) & np.isfinite(y_actual) & (y_actual > 0)
    yp = y_pred[mask]
    ya = y_actual[mask]
    n = int(mask.sum())
    if n == 0:
        return {
            "mean_error": float("nan"),
            "median_error": float("nan"),
            "mean_abs_error": float("nan"),
            "mean_ratio": float("nan"),
            "coverage_p90": float("nan"),
            "n_rows": 0,
        }
    err = yp - ya
    return {
        "mean_error": float(np.mean(err)),
        "median_error": float(np.median(err)),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "mean_ratio": float(np.mean(yp / ya)),
        "coverage_p90": float(np.mean(ya <= yp * 1.2 + 50)),
        "n_rows": n,
    }


def compute_percentile_stats(y: np.ndarray) -> dict:
    """Return p50/p90/p95/p99/mean/std of a 1-D array (ignores NaN)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if y.size == 0:
        return {k: float("nan") for k in ("p50", "p90", "p95", "p99", "mean", "std")}
    return {
        "p50": float(np.percentile(y, 50, method="nearest")),
        "p90": float(np.percentile(y, 90, method="nearest")),
        "p95": float(np.percentile(y, 95, method="nearest")),
        "p99": float(np.percentile(y, 99, method="nearest")),
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
    }


# ---------------------------------------------------------------------------
# Model 1: BiasCalibrationForecaster
# ---------------------------------------------------------------------------


@dataclass
class BiasCalibrationForecaster:
    """Calibrates the model's self-predicted output length.

    ``num_predicted_output_tokens`` is a valuable but biased signal — the
    serving engine emits it at request time before generation starts. This
    forecaster learns the systematic scale (multiplicative) and offset
    (additive) corrections from training data, reducing MAE and producing
    better-calibrated p50 estimates for scheduling priors.

    The learned transform is: ``calibrated = scale * raw + offset``
    where ``(scale, offset)`` are fitted by Huber regression (robust to
    outliers from very long sequences).
    """

    _scale: float = field(default=1.0, init=False, repr=False)
    _offset: float = field(default=0.0, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)
    _bias_stats_train: dict = field(default_factory=dict, init=False, repr=False)
    _n_train: int = field(default=0, init=False, repr=False)

    def fit(
        self,
        raw_predictions: np.ndarray,
        actual_tokens: np.ndarray,
    ) -> "BiasCalibrationForecaster":
        """Fit scale + offset from (raw_prediction, actual_output_tokens) pairs.

        Both arrays must be 1-D; NaN / non-positive rows are excluded.
        Requires at least MIN_TRAIN_ROWS valid rows.
        """
        rp = np.asarray(raw_predictions, dtype=np.float64).ravel()
        ya = np.asarray(actual_tokens, dtype=np.float64).ravel()
        assert len(rp) == len(ya), "length mismatch between raw_predictions and actual_tokens"

        mask = np.isfinite(rp) & np.isfinite(ya) & (ya > 0) & (rp >= 0)
        n = int(mask.sum())
        if n < MIN_TRAIN_ROWS:
            warnings.warn(
                f"BiasCalibrationForecaster: only {n} valid training rows "
                f"(< {MIN_TRAIN_ROWS}); keeping identity transform.",
                UserWarning,
                stacklevel=2,
            )
            self._scale = 1.0
            self._offset = 0.0
            self._fitted = True
            self._n_train = n
            self._bias_stats_train = compute_bias_stats(rp[mask], ya[mask])
            return self

        X_train = rp[mask].reshape(-1, 1)
        y_train = ya[mask]

        reg = HuberRegressor(epsilon=1.35, max_iter=200)
        reg.fit(X_train, y_train)
        self._scale = float(reg.coef_[0])
        self._offset = float(reg.intercept_)
        self._fitted = True
        self._n_train = n

        # Raw-prior bias before calibration (for audit reporting).
        self._bias_stats_train = compute_bias_stats(rp[mask], ya[mask])
        return self

    def predict(self, raw_predictions: np.ndarray) -> np.ndarray:
        """Apply the learned scale + offset.  Output is clipped to [1, ∞)."""
        if not self._fitted:
            raise RuntimeError("BiasCalibrationForecaster.fit() must be called before predict()")
        rp = np.asarray(raw_predictions, dtype=np.float64).ravel()
        calibrated = self._scale * rp + self._offset
        return np.clip(calibrated, 1.0, None)

    def calibration_report(self) -> dict:
        """Return a serialisable dict describing the calibration."""
        return {
            "forecaster": "BiasCalibrationForecaster",
            "scale": self._scale,
            "offset": self._offset,
            "n_train": self._n_train,
            "raw_prior_bias_stats": self._bias_stats_train,
            "status": SHADOW_TAG,
        }


# ---------------------------------------------------------------------------
# Model 2: HGBOutputLengthForecaster
# ---------------------------------------------------------------------------


@dataclass
class HGBOutputLengthConfig:
    """Configuration for the HGB output length forecaster."""
    quantile: float = 0.50
    max_iter: int = 200
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    learning_rate: float = 0.1
    random_state: int = 42


@dataclass
class HGBOutputLengthForecaster:
    """HGB quantile regression for output token length.

    Predicts ``actual_output_tokens`` at a given quantile (p50/p90/p95).
    Uses all predict-time features from the CARA feature matrix:
    ``num_predicted_output_tokens``, ``num_prompt_tokens``, queue state,
    KV-cache pressure, instance type.

    Inputs:
        X: 2-D numpy array built by the caller (e.g. from
           ``cara_latency_features.build_feature_matrix``).  No column
           corresponding to ``actual_output_tokens`` may appear.
        y: 1-D array of ``actual_output_tokens`` labels.

    Outputs:
        predict(X): 1-D array of predicted output token counts (clipped ≥ 1).
    """

    config: HGBOutputLengthConfig = field(default_factory=HGBOutputLengthConfig)
    _model: Optional[HistGradientBoostingRegressor] = field(
        default=None, init=False, repr=False
    )
    _fitted: bool = field(default=False, init=False, repr=False)
    _n_train: int = field(default=0, init=False, repr=False)
    _feature_names: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if self.config.quantile not in SUPPORTED_QUANTILES:
            raise ValueError(
                f"quantile {self.config.quantile} not in {SUPPORTED_QUANTILES}"
            )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "HGBOutputLengthForecaster":
        """Fit the HGB quantile model.

        Args:
            X: (n_samples, n_features) array.
            y: (n_samples,) array of actual_output_tokens.
            feature_names: optional list of column names for audit reporting.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        assert X.shape[0] == len(y), "X and y must have the same number of rows"

        # Drop NaN-target rows (NaN features are handled by HGB natively).
        mask = np.isfinite(y) & (y > 0)
        n = int(mask.sum())
        if n < MIN_TRAIN_ROWS:
            raise ValueError(
                f"HGBOutputLengthForecaster: only {n} valid target rows (< {MIN_TRAIN_ROWS})"
            )

        self._model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.config.quantile,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
        )
        self._model.fit(X[mask], y[mask])
        self._fitted = True
        self._n_train = n
        if feature_names is not None:
            self._feature_names = list(feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted output token counts at the configured quantile."""
        if not self._fitted or self._model is None:
            raise RuntimeError("HGBOutputLengthForecaster.fit() must be called before predict()")
        X = np.asarray(X, dtype=np.float64)
        out = self._model.predict(X)
        return np.clip(np.asarray(out, dtype=np.float64), 1.0, None)

    def model_report(self) -> dict:
        return {
            "forecaster": "HGBOutputLengthForecaster",
            "quantile": self.config.quantile,
            "n_train": self._n_train,
            "n_features": len(self._feature_names),
            "status": SHADOW_TAG,
        }


# ---------------------------------------------------------------------------
# Ensemble: OutputLengthForecastBundle
# ---------------------------------------------------------------------------


@dataclass
class OutputLengthForecast:
    """Per-request output length forecast at multiple quantiles.

    Fields:
        p50_tokens: Calibrated p50 estimate. Use as scheduling prior (magnitude).
        p90_tokens: Calibrated p90 estimate. Use as SLA buffer.
        raw_prior_tokens: Original ``num_predicted_output_tokens`` from engine.
        calibrated_p50_tokens: BiasCalibrationForecaster output (always valid).
        hgb_p50_tokens: HGB p50 (None when HGB not fitted).
        hgb_p90_tokens: HGB p90 (None when HGB not fitted).
    """

    p50_tokens: float
    p90_tokens: float
    raw_prior_tokens: float
    calibrated_p50_tokens: float
    hgb_p50_tokens: Optional[float] = None
    hgb_p90_tokens: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "p50_tokens": self.p50_tokens,
            "p90_tokens": self.p90_tokens,
            "raw_prior_tokens": self.raw_prior_tokens,
            "calibrated_p50_tokens": self.calibrated_p50_tokens,
            "hgb_p50_tokens": self.hgb_p50_tokens,
            "hgb_p90_tokens": self.hgb_p90_tokens,
            "status": SHADOW_TAG,
        }


@dataclass
class OutputLengthForecastBundle:
    """Combines BiasCalibrationForecaster + optional HGB quantile models.

    Designed for shadow-mode evaluation: all outputs are tagged
    ``shadow_only_not_production_ready``. Wire into production only after
    pilot telemetry validation (docs/RESULTS.md §8).

    Usage::

        bundle = OutputLengthForecastBundle()
        bundle.fit_calibration(raw_predictions, actual_tokens)
        # Optional: fit HGB on the CARA feature matrix.
        bundle.fit_hgb(X_train, y_train, feature_names=feature_names)

        forecasts = bundle.predict(raw_predictions, X)  # X may be None
    """

    _calibration: BiasCalibrationForecaster = field(
        default_factory=BiasCalibrationForecaster, init=False, repr=False
    )
    _hgb_p50: Optional[HGBOutputLengthForecaster] = field(
        default=None, init=False, repr=False
    )
    _hgb_p90: Optional[HGBOutputLengthForecaster] = field(
        default=None, init=False, repr=False
    )
    _calibration_fitted: bool = field(default=False, init=False, repr=False)

    def fit_calibration(
        self,
        raw_predictions: np.ndarray,
        actual_tokens: np.ndarray,
    ) -> "OutputLengthForecastBundle":
        """Fit the bias calibration model."""
        self._calibration.fit(raw_predictions, actual_tokens)
        self._calibration_fitted = True
        return self

    def fit_hgb(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "OutputLengthForecastBundle":
        """Fit HGB p50 and p90 models on the CARA feature matrix.

        Both quantiles are fitted; each model gets a fresh HGB instance.
        Requires calibration to already be fitted.
        """
        if not self._calibration_fitted:
            raise RuntimeError("fit_calibration() must be called before fit_hgb()")
        self._hgb_p50 = HGBOutputLengthForecaster(
            config=HGBOutputLengthConfig(quantile=0.50)
        )
        self._hgb_p50.fit(X, y, feature_names=feature_names)
        self._hgb_p90 = HGBOutputLengthForecaster(
            config=HGBOutputLengthConfig(quantile=0.90)
        )
        self._hgb_p90.fit(X, y, feature_names=feature_names)
        return self

    def predict_single(
        self,
        raw_prior: float,
        x_row: Optional[np.ndarray] = None,
    ) -> OutputLengthForecast:
        """Forecast for a single request.

        Args:
            raw_prior: ``num_predicted_output_tokens`` from the serving engine.
            x_row: 1-D feature vector (same column order as fit_hgb X).
                   Required if HGB models are fitted; ignored otherwise.
        """
        if not self._calibration_fitted:
            raise RuntimeError("fit_calibration() must be called first")

        rp = np.array([raw_prior], dtype=np.float64)
        cal_p50 = float(self._calibration.predict(rp)[0])

        hgb_p50 = None
        hgb_p90 = None
        if self._hgb_p50 is not None and x_row is not None:
            X2d = np.asarray(x_row, dtype=np.float64).reshape(1, -1)
            hgb_p50 = float(self._hgb_p50.predict(X2d)[0])
            if self._hgb_p90 is not None:
                hgb_p90 = float(self._hgb_p90.predict(X2d)[0])

        # Best p50: prefer HGB when available (it uses more features).
        p50 = hgb_p50 if hgb_p50 is not None else cal_p50
        # Best p90: prefer HGB; fall back to calibrated p50 * 1.5.
        p90 = hgb_p90 if hgb_p90 is not None else cal_p50 * 1.5

        return OutputLengthForecast(
            p50_tokens=p50,
            p90_tokens=max(p90, p50),
            raw_prior_tokens=float(raw_prior),
            calibrated_p50_tokens=cal_p50,
            hgb_p50_tokens=hgb_p50,
            hgb_p90_tokens=hgb_p90,
        )

    def predict_batch(
        self,
        raw_predictions: np.ndarray,
        X: Optional[np.ndarray] = None,
    ) -> list[OutputLengthForecast]:
        """Forecast for a batch of requests.

        Returns a list of OutputLengthForecast in the same order.
        """
        if not self._calibration_fitted:
            raise RuntimeError("fit_calibration() must be called first")

        rp = np.asarray(raw_predictions, dtype=np.float64).ravel()
        cal_p50_arr = self._calibration.predict(rp)

        hgb_p50_arr: Optional[np.ndarray] = None
        hgb_p90_arr: Optional[np.ndarray] = None
        if self._hgb_p50 is not None and X is not None:
            hgb_p50_arr = self._hgb_p50.predict(X)
            if self._hgb_p90 is not None:
                hgb_p90_arr = self._hgb_p90.predict(X)

        results = []
        for i, raw in enumerate(rp):
            cal_p50 = float(cal_p50_arr[i])
            h50 = float(hgb_p50_arr[i]) if hgb_p50_arr is not None else None
            h90 = float(hgb_p90_arr[i]) if hgb_p90_arr is not None else None
            p50 = h50 if h50 is not None else cal_p50
            p90 = h90 if h90 is not None else cal_p50 * 1.5
            results.append(OutputLengthForecast(
                p50_tokens=p50,
                p90_tokens=max(p90, p50),
                raw_prior_tokens=float(raw),
                calibrated_p50_tokens=cal_p50,
                hgb_p50_tokens=h50,
                hgb_p90_tokens=h90,
            ))
        return results

    def bundle_report(self) -> dict:
        """Return a serialisable audit report."""
        return {
            "calibration_fitted": self._calibration_fitted,
            "hgb_fitted": self._hgb_p50 is not None,
            "calibration": (
                self._calibration.calibration_report()
                if self._calibration_fitted else None
            ),
            "hgb_p50": (
                self._hgb_p50.model_report() if self._hgb_p50 is not None else None
            ),
            "hgb_p90": (
                self._hgb_p90.model_report() if self._hgb_p90 is not None else None
            ),
            "status": SHADOW_TAG,
        }
