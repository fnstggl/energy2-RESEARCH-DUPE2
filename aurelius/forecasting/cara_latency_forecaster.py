"""CARA latency forecaster v1 — TTFT + E2E quantile regression.

Three families of predictors are implemented:

1. ``Baseline*`` — strongest-realistic baselines documented in the
   mission spec PHASE 1. These are the bars the ML model must beat
   before it can be tagged ``candidate_for_shadow_integration``.
2. ``HistGradientBoostingQuantileForecaster`` — sklearn HGB with
   ``loss="quantile"`` for p50 / p95 / p99 regression. Default model
   for v1.
3. ``RandomForestMedianForecaster`` — sklearn RandomForest as a
   robustness candidate.

Every predictor exposes ``fit(X_train, y_train, **kwargs)`` and
``predict(X_holdout) -> np.ndarray``.

No model in this module is wired into any controller or scheduler. The
forecaster is research-class until shadow-integration gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from sklearn.ensemble import (
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
except ImportError as e:  # pragma: no cover - tests assert sklearn present.
    raise ImportError(
        "CARA latency forecaster requires scikit-learn; install via "
        "`pip install scikit-learn`"
    ) from e


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def _percentile(arr: np.ndarray, q: float) -> float:
    """Nearest-rank percentile (deterministic, no interpolation)."""
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q, method="nearest"))


@dataclass
class GlobalConstantP95Baseline:
    """Predicts the training-set p95 of the target for every row."""
    quantile: float = 95.0

    def __post_init__(self):
        self._value: float = float("nan")

    def fit(self, X, y) -> "GlobalConstantP95Baseline":
        self._value = _percentile(np.asarray(y, dtype=np.float64), self.quantile)
        return self

    def predict(self, X) -> np.ndarray:
        n = X.shape[0]
        return np.full(n, self._value, dtype=np.float64)


@dataclass
class GroupConstantQuantileBaseline:
    """Per-group quantile predictor.

    ``group_keys_train`` and ``group_keys_predict`` are 1-D arrays
    (usually strings) identifying the bucket each row belongs to. Unseen
    buckets at prediction time fall back to the global quantile.
    """
    quantile: float = 95.0

    def __post_init__(self):
        self._per_group: dict = {}
        self._fallback: float = float("nan")

    def fit(self, X, y, *, group_keys_train) -> "GroupConstantQuantileBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        groups = np.asarray(group_keys_train, dtype=object)
        self._fallback = _percentile(y_arr, self.quantile)
        self._per_group = {}
        for g in np.unique(groups):
            mask = groups == g
            if mask.sum() >= 1:
                self._per_group[g] = _percentile(y_arr[mask], self.quantile)
        return self

    def predict(self, X, *, group_keys_predict) -> np.ndarray:
        groups = np.asarray(group_keys_predict, dtype=object)
        out = np.full(len(groups), self._fallback, dtype=np.float64)
        for i, g in enumerate(groups):
            if g in self._per_group:
                out[i] = self._per_group[g]
        return out


@dataclass
class SimpleRulePlacementScoreBaseline:
    """Placement-score baseline: instance-type p95 + queue-depth penalty.

    The score is a *predicted latency*, not a routing decision. The
    routing backtest ranks candidates by ascending score.
    """
    quantile: float = 95.0
    queue_penalty_per_depth: float = 0.05  # +50 ms per queued request

    def __post_init__(self):
        self._instance_p95: dict = {}
        self._fallback: float = float("nan")

    def fit(self, X, y, *, instance_types_train,
            queue_depths_train) -> "SimpleRulePlacementScoreBaseline":
        y_arr = np.asarray(y, dtype=np.float64)
        it = np.asarray(instance_types_train, dtype=object)
        self._fallback = _percentile(y_arr, self.quantile)
        for g in np.unique(it):
            mask = it == g
            if mask.sum() >= 1:
                self._instance_p95[g] = _percentile(y_arr[mask], self.quantile)
        return self

    def predict(self, X, *, instance_types_predict,
                queue_depths_predict) -> np.ndarray:
        it = np.asarray(instance_types_predict, dtype=object)
        qd = np.asarray(queue_depths_predict, dtype=np.float64)
        out = np.empty(len(it), dtype=np.float64)
        for i in range(len(it)):
            base = self._instance_p95.get(it[i], self._fallback)
            out[i] = base + self.queue_penalty_per_depth * max(0.0, qd[i])
        return out


# ---------------------------------------------------------------------------
# Gradient boosting + Random Forest ML forecasters
# ---------------------------------------------------------------------------


@dataclass
class HistGradientBoostingQuantileForecaster:
    """sklearn HistGradientBoostingRegressor with quantile loss.

    Quantile must be in (0, 1) (e.g. 0.5 for p50, 0.95 for p95).
    """
    quantile: float = 0.95
    max_iter: int = 300
    max_depth: int = 8
    learning_rate: float = 0.06
    min_samples_leaf: int = 50
    seed: int = 1773889
    monotonic_cst: Optional[tuple] = None  # passed through if provided

    def __post_init__(self):
        self._model: Optional[HistGradientBoostingRegressor] = None

    def fit(self, X, y) -> "HistGradientBoostingQuantileForecaster":
        if not (0.0 < self.quantile < 1.0):
            raise ValueError(
                f"quantile must be in (0,1), got {self.quantile!r}"
            )
        kw = {}
        if self.monotonic_cst is not None:
            kw["monotonic_cst"] = list(self.monotonic_cst)
        self._model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=self.quantile,
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            **kw,
        )
        self._model.fit(X, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not fitted")
        return self._model.predict(X)


@dataclass
class RandomForestMedianForecaster:
    """Robustness candidate — random forest mean predictor.

    Random forest doesn't natively do quantile regression in sklearn;
    we use it as a median-style robustness check vs HGB.
    """
    n_estimators: int = 150
    max_depth: Optional[int] = 18
    min_samples_leaf: int = 20
    n_jobs: int = -1
    seed: int = 1773889

    def __post_init__(self):
        self._model: Optional[RandomForestRegressor] = None

    def fit(self, X, y) -> "RandomForestMedianForecaster":
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        self._model.fit(X, np.asarray(y, dtype=np.float64))
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not fitted")
        return self._model.predict(X)


# ---------------------------------------------------------------------------
# Safety calibration — conservative multiplier to reduce underprediction.
# ---------------------------------------------------------------------------


@dataclass
class ConservativeMultiplierCalibration:
    """Multiplies a base predictor's output by ``multiplier`` to reduce
    severe-underprediction rate. ``multiplier > 1`` favours safety."""
    multiplier: float = 1.0
    base: object = None

    def fit(self, X, y) -> "ConservativeMultiplierCalibration":
        if self.base is not None:
            self.base.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        if self.base is None:
            raise RuntimeError("calibration base not set")
        return self.base.predict(X) * float(self.multiplier)


# ---------------------------------------------------------------------------
# Out-of-distribution fallback to baseline
# ---------------------------------------------------------------------------


@dataclass
class FallbackToBaseline:
    """If the model's predicted value falls below ``floor_fraction`` of
    the baseline prediction, fall back to the baseline (safety floor)."""
    ml: object = None
    baseline_predictions: Optional[np.ndarray] = None
    floor_fraction: float = 0.5

    def predict_with_floor(self, X) -> np.ndarray:
        if self.ml is None or self.baseline_predictions is None:
            raise RuntimeError("fallback wrapper not configured")
        ml_pred = self.ml.predict(X)
        bp = np.asarray(self.baseline_predictions, dtype=np.float64)
        if bp.shape[0] != ml_pred.shape[0]:
            raise ValueError(
                f"baseline_predictions length {bp.shape[0]} != "
                f"ml predictions length {ml_pred.shape[0]}"
            )
        floor = self.floor_fraction * bp
        return np.where(ml_pred < floor, bp, ml_pred)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Standard regression metrics + tail-safety metrics."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return {"count": 0}
    abs_err = np.abs(yp - yt)
    under = yp < yt
    sev_under = yp < (yt / 2.0)  # actual > 2x predicted
    # Calibration coverage: fraction of actuals <= predicted (if predicted
    # is a p95/p99 estimate, this should be ≥ 0.95 / 0.99).
    coverage = float((yt <= yp).mean())
    return {
        "count": int(yt.size),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean((yp - yt) ** 2))),
        "median_abs_err": float(np.median(abs_err)),
        "p90_abs_err": _percentile(abs_err, 90),
        "p95_abs_err": _percentile(abs_err, 95),
        "p99_abs_err": _percentile(abs_err, 99),
        "underprediction_rate": float(under.mean()),
        "severe_underprediction_rate": float(sev_under.mean()),
        "predicted_p99_within_2x": float(
            ((yp / np.maximum(yt, 1e-9)) <= 2.0).mean()
        ),
        "calibration_coverage": coverage,
    }


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Pinball / quantile loss for a quantile predictor.

    Lower is better. ``quantile`` in (0, 1)."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if yt.size == 0:
        return float("nan")
    diff = yt - yp
    loss = np.maximum(quantile * diff, (quantile - 1.0) * diff)
    return float(loss.mean())


def quantile_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, quantile: float,
) -> dict:
    """Quantile-specific metrics. Calibration error = |coverage - target|."""
    base = regression_metrics(y_true, y_pred)
    base["quantile"] = quantile
    base["pinball_loss"] = pinball_loss(y_true, y_pred, quantile)
    target = quantile if quantile < 1.0 else quantile / 100.0
    if "calibration_coverage" in base:
        base["calibration_error"] = float(
            abs(base["calibration_coverage"] - target)
        )
    return base


def subgroup_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, group_keys: np.ndarray,
    *, min_count: int = 100,
) -> dict:
    """Per-subgroup regression metrics with INSUFFICIENT_SAMPLE flag."""
    out: dict = {"subgroups": {}, "insufficient_sample_groups": []}
    groups = np.asarray(group_keys, dtype=object)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        m = regression_metrics(yt[mask], yp[mask])
        if m.get("count", 0) < min_count:
            m["flags"] = ["INSUFFICIENT_SAMPLE"]
            out["insufficient_sample_groups"].append(str(g))
        out["subgroups"][str(g)] = m
    return out


# ---------------------------------------------------------------------------
# Incremental alpha gate
# ---------------------------------------------------------------------------


def incremental_alpha_pct(
    baseline_metric: float, candidate_metric: float, *, lower_is_better: bool = True,
) -> float:
    """Return % improvement of candidate vs baseline."""
    if baseline_metric is None or candidate_metric is None:
        return 0.0
    b = float(baseline_metric)
    c = float(candidate_metric)
    if b == 0 or np.isnan(b) or np.isnan(c):
        return 0.0
    if lower_is_better:
        return 100.0 * (b - c) / b
    return 100.0 * (c - b) / b


GATE_THRESHOLDS = {
    "diagnostic_only": 0.0,
    "promising_needs_validation": 2.0,
    "candidate_for_shadow_integration": 5.0,
    "strong_candidate": 10.0,
}


def classify_gate_status(
    alpha_pct: float, *,
    tail_underpred_rate: float, baseline_tail_underpred_rate: float,
    safety_regression_count: int,
) -> str:
    """Map (alpha %, safety regressions) to a gate label."""
    if alpha_pct < GATE_THRESHOLDS["promising_needs_validation"]:
        return "diagnostic_only"
    if safety_regression_count > 0:
        return "diagnostic_only"
    if tail_underpred_rate > baseline_tail_underpred_rate:
        return "diagnostic_only"
    if alpha_pct < GATE_THRESHOLDS["candidate_for_shadow_integration"]:
        return "promising_needs_validation"
    if alpha_pct < GATE_THRESHOLDS["strong_candidate"]:
        return "candidate_for_shadow_integration"
    return "strong_candidate"
