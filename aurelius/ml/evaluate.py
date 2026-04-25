"""ML forecasting evaluation for Aurelius — Phase 2.

Computes evaluation metrics for quantile forecasters on holdout data:
- MAPE  (Mean Absolute Percentage Error)
- RMSE  (Root Mean Squared Error)
- MAE   (Mean Absolute Error)
- p50 bias (mean signed error: realized − p50)
- p90 empirical coverage (fraction of realized ≤ p90; target = 0.90)
- calibration error (|empirical_coverage − 0.90|)
- downside risk (mean exceedance above p90)

Model-promotion rule:
  A candidate model replaces the active model only when it improves the
  primary metric (MAPE) by the minimum threshold AND does not materially
  worsen calibration error or downside risk.

IMPORTANT: This module does NOT enforce the train/holdout split.
The caller is responsible for ensuring the forecaster was NOT trained
on holdout data (see BacktestEngine / TemporalSplitter for split logic).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from typing import Optional, Any

from aurelius.models import EnergyPrice, CarbonIntensity

logger = logging.getLogger(__name__)

# Floor for MAPE denominator — prevents division by near-zero actuals
# (e.g. negative-price electricity hours are excluded from MAPE)
_MAPE_FLOOR = 1e-6

# Target quantile for calibration (must match model's p90 output)
_TARGET_QUANTILE = 0.90


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastEvalMetrics:
    """Evaluation metrics for a quantile forecaster on holdout data.

    All metrics are computed from realised holdout values only.
    No forecast values from the training window enter these computations.

    Attributes:
        n_samples:         Number of holdout points evaluated.
        mape:              Mean Absolute Percentage Error (%). Lower is better.
        rmse:              Root Mean Squared Error. Lower is better.
        mae:               Mean Absolute Error. Lower is better.
        p50_bias:          Mean(realized − p50). Positive = systematic under-forecast.
        p90_coverage:      Fraction of realized values ≤ p90. Target: 0.90.
        calibration_error: |p90_coverage − 0.90|. Lower is better; 0 = perfect.
        downside_risk:     Mean of max(0, realized − p90). Lower is better.
        region:            Region tag (empty string = aggregated across regions).
    """
    n_samples: int
    mape: float
    rmse: float
    mae: float
    p50_bias: float
    p90_coverage: float
    calibration_error: float
    downside_risk: float
    region: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def is_valid(self) -> bool:
        """True when metrics are numerically defined and have sufficient samples."""
        return (
            self.n_samples >= 1
            and not math.isnan(self.mape)
            and not math.isinf(self.mape)
        )


# ---------------------------------------------------------------------------
# Pure metric functions (no side-effects, no I/O)
# ---------------------------------------------------------------------------

def compute_mape(actuals: list[float], p50_preds: list[float]) -> float:
    """Mean Absolute Percentage Error (%).

    Points where |actual| < _MAPE_FLOOR are excluded to avoid division by
    near-zero values (e.g. negative-price hours).

    Returns NaN when no valid points remain.
    """
    if len(actuals) != len(p50_preds):
        raise ValueError(f"Length mismatch: {len(actuals)} vs {len(p50_preds)}")
    errors = []
    for a, p in zip(actuals, p50_preds):
        if abs(a) >= _MAPE_FLOOR:
            errors.append(abs((a - p) / a))
    if not errors:
        return float("nan")
    return sum(errors) / len(errors) * 100.0


def compute_rmse(actuals: list[float], p50_preds: list[float]) -> float:
    """Root Mean Squared Error."""
    if len(actuals) != len(p50_preds):
        raise ValueError(f"Length mismatch: {len(actuals)} vs {len(p50_preds)}")
    n = len(actuals)
    if n == 0:
        return float("nan")
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actuals, p50_preds)) / n)


def compute_mae(actuals: list[float], p50_preds: list[float]) -> float:
    """Mean Absolute Error."""
    if len(actuals) != len(p50_preds):
        raise ValueError(f"Length mismatch: {len(actuals)} vs {len(p50_preds)}")
    n = len(actuals)
    if n == 0:
        return float("nan")
    return sum(abs(a - p) for a, p in zip(actuals, p50_preds)) / n


def compute_p50_bias(actuals: list[float], p50_preds: list[float]) -> float:
    """Signed mean error (realized − p50).

    Positive = model consistently under-forecasts (actual > prediction).
    Negative = model consistently over-forecasts.
    """
    if len(actuals) != len(p50_preds):
        raise ValueError(f"Length mismatch: {len(actuals)} vs {len(p50_preds)}")
    n = len(actuals)
    if n == 0:
        return float("nan")
    return sum(a - p for a, p in zip(actuals, p50_preds)) / n


def compute_p90_coverage(actuals: list[float], p90_preds: list[float]) -> float:
    """Empirical coverage rate: fraction of realized values ≤ p90.

    Target = 0.90.  Values > 0.90 mean the model is over-conservative.
    Values < 0.90 mean the model underestimates tail risk.
    """
    if len(actuals) != len(p90_preds):
        raise ValueError(f"Length mismatch: {len(actuals)} vs {len(p90_preds)}")
    n = len(actuals)
    if n == 0:
        return float("nan")
    covered = sum(1 for a, p in zip(actuals, p90_preds) if a <= p)
    return covered / n


def compute_calibration_error(
    actuals: list[float],
    p90_preds: list[float],
    target: float = _TARGET_QUANTILE,
) -> float:
    """Absolute deviation of empirical coverage from target coverage.

    0 = perfect calibration.  > 0 = either over- or under-conservative.
    """
    coverage = compute_p90_coverage(actuals, p90_preds)
    if math.isnan(coverage):
        return float("nan")
    return abs(coverage - target)


def compute_downside_risk(actuals: list[float], p90_preds: list[float]) -> float:
    """Mean exceedance above p90 = mean(max(0, realized − p90)).

    Zero when all realized values are ≤ p90.
    Large values indicate the model is systematically underestimating tail risk.
    """
    if len(actuals) != len(p90_preds):
        raise ValueError(f"Length mismatch: {len(actuals)} vs {len(p90_preds)}")
    n = len(actuals)
    if n == 0:
        return float("nan")
    return sum(max(0.0, a - p) for a, p in zip(actuals, p90_preds)) / n


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_region_metrics(
    actuals: list[float],
    p50_preds: list[float],
    p90_preds: list[float],
    region: str = "",
) -> ForecastEvalMetrics:
    n = len(actuals)
    return ForecastEvalMetrics(
        n_samples=n,
        mape=compute_mape(actuals, p50_preds),
        rmse=compute_rmse(actuals, p50_preds),
        mae=compute_mae(actuals, p50_preds),
        p50_bias=compute_p50_bias(actuals, p50_preds),
        p90_coverage=compute_p90_coverage(actuals, p90_preds),
        calibration_error=compute_calibration_error(actuals, p90_preds),
        downside_risk=compute_downside_risk(actuals, p90_preds),
        region=region,
    )


# ---------------------------------------------------------------------------
# High-level evaluators
# ---------------------------------------------------------------------------

def evaluate_price_forecaster(
    forecaster: Any,  # PriceQuantileForecaster — using Any to avoid circular import
    holdout_prices: list[EnergyPrice],
    recent_prices: Optional[list[EnergyPrice]] = None,
) -> dict[str, ForecastEvalMetrics]:
    """Evaluate a fitted PriceQuantileForecaster on holdout data.

    Generates p50/p90 predictions for each holdout timestamp, then computes
    evaluation metrics per region and as an aggregate across all regions.

    The forecaster MUST be fitted exclusively on data strictly before the
    holdout window. This function does not enforce that constraint — the
    caller is responsible (use TemporalSplitter / BacktestEngine).

    Args:
        forecaster:      Fitted PriceQuantileForecaster instance.
        holdout_prices:  EnergyPrice objects from the holdout window (actuals).
        recent_prices:   EnergyPrice objects available at decision time (from the
                         end of the training window). Used for lag features.

    Returns:
        Dict mapping region → ForecastEvalMetrics.
        Always includes an "_aggregate" key when at least one region evaluates.
        Returns empty dict when holdout_prices is empty.
    """
    if not holdout_prices:
        logger.warning("evaluate_price_forecaster: holdout is empty")
        return {}

    by_region: dict[str, list[EnergyPrice]] = {}
    for p in holdout_prices:
        by_region.setdefault(p.region, []).append(p)

    all_actuals: list[float] = []
    all_p50: list[float] = []
    all_p90: list[float] = []
    result: dict[str, ForecastEvalMetrics] = {}

    for region in sorted(by_region):
        region_prices = sorted(by_region[region], key=lambda p: p.timestamp)
        timestamps = [p.timestamp for p in region_prices]
        actuals = [p.price_per_mwh for p in region_prices]

        try:
            forecasts = forecaster.predict(region, timestamps, recent_prices)
        except Exception as exc:
            logger.warning(f"Price prediction failed for region {region}: {exc}")
            continue

        if len(forecasts) != len(actuals):
            logger.warning(
                f"Forecast/actual length mismatch for {region}: "
                f"{len(forecasts)} forecasts vs {len(actuals)} actuals"
            )
            continue

        p50_preds = [f.p50 for f in forecasts]
        p90_preds = [f.p90 for f in forecasts]

        result[region] = _compute_region_metrics(actuals, p50_preds, p90_preds, region)
        all_actuals.extend(actuals)
        all_p50.extend(p50_preds)
        all_p90.extend(p90_preds)

    if all_actuals:
        result["_aggregate"] = _compute_region_metrics(
            all_actuals, all_p50, all_p90, "_aggregate"
        )

    return result


def evaluate_carbon_forecaster(
    forecaster: Any,  # CarbonQuantileForecaster — using Any to avoid circular import
    holdout_carbon: list[CarbonIntensity],
    recent_carbon: Optional[list[CarbonIntensity]] = None,
) -> dict[str, ForecastEvalMetrics]:
    """Evaluate a fitted CarbonQuantileForecaster on holdout data.

    Same contract as evaluate_price_forecaster but for carbon intensity.
    Carbon actuals are gCO2/kWh; forecaster.predict() returns CarbonQuantileForecast.

    Args:
        forecaster:     Fitted CarbonQuantileForecaster instance.
        holdout_carbon: CarbonIntensity objects from the holdout window.
        recent_carbon:  Recent carbon data at prediction time (for lag features).

    Returns:
        Dict mapping region → ForecastEvalMetrics.
        Always includes "_aggregate" when at least one region evaluates.
    """
    if not holdout_carbon:
        logger.warning("evaluate_carbon_forecaster: holdout is empty")
        return {}

    by_region: dict[str, list[CarbonIntensity]] = {}
    for c in holdout_carbon:
        by_region.setdefault(c.region, []).append(c)

    all_actuals: list[float] = []
    all_p50: list[float] = []
    all_p90: list[float] = []
    result: dict[str, ForecastEvalMetrics] = {}

    for region in sorted(by_region):
        region_carbon = sorted(by_region[region], key=lambda c: c.timestamp)
        timestamps = [c.timestamp for c in region_carbon]
        actuals = [c.gco2_per_kwh for c in region_carbon]

        try:
            forecasts = forecaster.predict(region, timestamps, recent_carbon)
        except Exception as exc:
            logger.warning(f"Carbon prediction failed for region {region}: {exc}")
            continue

        if len(forecasts) != len(actuals):
            logger.warning(
                f"Forecast/actual length mismatch for {region}: "
                f"{len(forecasts)} vs {len(actuals)}"
            )
            continue

        p50_preds = [f.p50 for f in forecasts]
        p90_preds = [f.p90 for f in forecasts]

        result[region] = _compute_region_metrics(actuals, p50_preds, p90_preds, region)
        all_actuals.extend(actuals)
        all_p50.extend(p50_preds)
        all_p90.extend(p90_preds)

    if all_actuals:
        result["_aggregate"] = _compute_region_metrics(
            all_actuals, all_p50, all_p90, "_aggregate"
        )

    return result


# ---------------------------------------------------------------------------
# Model promotion gate
# ---------------------------------------------------------------------------

def should_promote(
    current_metrics: ForecastEvalMetrics,
    candidate_metrics: ForecastEvalMetrics,
    min_mape_improvement_pct: float = 1.0,
    max_calibration_degradation: float = 0.05,
    max_downside_risk_increase_pct: float = 10.0,
) -> bool:
    """Return True if the candidate model should replace the active model.

    Promotion requires ALL of the following to hold simultaneously:

    1. PRIMARY METRIC: candidate MAPE is lower (better) by at least
       min_mape_improvement_pct percentage points.

    2. CALIBRATION SAFETY: candidate calibration_error does not increase by
       more than max_calibration_degradation (absolute).

    3. DOWNSIDE SAFETY: candidate downside_risk does not increase by more
       than max_downside_risk_increase_pct (relative %). Checked only when
       current downside_risk > 0.

    If either metrics object is invalid (NaN / no samples), promotion is denied.

    Args:
        current_metrics:              Metrics of the currently active model.
        candidate_metrics:            Metrics of the challenger model.
        min_mape_improvement_pct:     Minimum MAPE reduction required (pp).
        max_calibration_degradation:  Max allowable calibration error increase.
        max_downside_risk_increase_pct: Max allowable downside risk increase (%).

    Returns:
        True  → promote the candidate.
        False → keep the current model.
    """
    if not current_metrics.is_valid() or not candidate_metrics.is_valid():
        logger.warning("should_promote: one or both metrics objects are invalid; denying promotion")
        return False

    # 1. MAPE must improve
    mape_delta = current_metrics.mape - candidate_metrics.mape
    if mape_delta < min_mape_improvement_pct:
        logger.info(
            "Promotion denied: MAPE improvement %.3f pp < threshold %.3f pp",
            mape_delta,
            min_mape_improvement_pct,
        )
        return False

    # 2. Calibration must not materially worsen
    cal_delta = candidate_metrics.calibration_error - current_metrics.calibration_error
    if cal_delta > max_calibration_degradation:
        logger.info(
            "Promotion denied: calibration error worsened by %.4f (max %.4f)",
            cal_delta,
            max_calibration_degradation,
        )
        return False

    # 3. Downside risk must not materially worsen
    if current_metrics.downside_risk > 0:
        risk_increase_pct = (
            (candidate_metrics.downside_risk - current_metrics.downside_risk)
            / current_metrics.downside_risk * 100.0
        )
        if risk_increase_pct > max_downside_risk_increase_pct:
            logger.info(
                "Promotion denied: downside risk increased by %.2f %% (max %.2f %%)",
                risk_increase_pct,
                max_downside_risk_increase_pct,
            )
            return False

    logger.info(
        "Promotion approved: MAPE %.3f %% → %.3f %% (improvement %.3f pp)",
        current_metrics.mape,
        candidate_metrics.mape,
        mape_delta,
    )
    return True
