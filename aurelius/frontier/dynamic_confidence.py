"""Dynamic Serving Frontier — confidence update logic (v1).

Given a :class:`DynamicFrontierCalibrationRecord`, produce a new
confidence value with a categorical reason code. The update is bounded,
asymmetric (safety wrongness costs more than accuracy), and deterministic
— no ML, no future leakage.

Hard rules:

- ``false_safe`` penalises confidence more than other errors.
- Confidence does **NOT** rise when safety was wrong (``false_safe`` or
  unknown).
- All updates are clamped to ``[min_confidence, max_confidence]``.
- Per-step movement is clamped to ``max_update_per_step``.
- Reason codes are categorical so downstream summaries can audit them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .dynamic_evaluation import DynamicFrontierCalibrationRecord


@dataclass
class ConfidenceUpdateConfig:
    """Pre-registered confidence-update settings.

    Defaults are pre-registered. They are not tuned per trace. The
    rolling-replay loop may tweak ``conservative_margin``-style knobs
    elsewhere; this module only updates the per-decision confidence.
    """

    false_safe_penalty: float = 0.15
    false_unsafe_penalty: float = 0.05
    conservative_miss_penalty: float = 0.03
    accurate_safe_reward: float = 0.02
    # Large prediction-error penalty (per metric, additive).
    large_goodput_error_penalty: float = 0.02
    large_timeout_error_penalty: float = 0.02
    large_queue_error_penalty: float = 0.02
    # Magnitude thresholds (relative for goodput; absolute for timeout
    # in pct units and for queue p99 in ms).
    large_goodput_error_threshold_pct: float = 0.20
    large_timeout_error_threshold_pct: float = 5.0
    large_queue_error_threshold_ms: float = 500.0
    # Bounding rules.
    max_update_per_step: float = 0.20
    min_confidence: float = 0.0
    max_confidence: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence
                <= self.max_confidence <= 1.0):
            raise ValueError(
                f"min/max confidence must be in [0,1] with min <= max; "
                f"got min={self.min_confidence}, max={self.max_confidence}")
        for name in ("false_safe_penalty", "false_unsafe_penalty",
                     "conservative_miss_penalty", "accurate_safe_reward",
                     "large_goodput_error_penalty",
                     "large_timeout_error_penalty",
                     "large_queue_error_penalty",
                     "max_update_per_step"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0; "
                                 f"got {getattr(self, name)}")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def update_confidence(
    record: DynamicFrontierCalibrationRecord,
    config: Optional[ConfidenceUpdateConfig] = None,
) -> Tuple[float, str]:
    """Return the updated confidence and a categorical reason code.

    The reason code lists the dominant factors, comma-joined. The
    confidence is clamped to ``[min_confidence, max_confidence]`` and to
    ``confidence_before ± max_update_per_step``.
    """
    cfg = config or ConfidenceUpdateConfig()
    cur = _clamp(record.confidence_before, cfg.min_confidence,
                 cfg.max_confidence)
    delta = 0.0
    reasons: list[str] = []

    # 1 — safety verdict drives most of the update.
    if record.false_safe:
        delta -= cfg.false_safe_penalty
        reasons.append("false_safe")
    if record.false_unsafe:
        delta -= cfg.false_unsafe_penalty
        reasons.append("false_unsafe")
    if record.conservative_miss:
        delta -= cfg.conservative_miss_penalty
        reasons.append("conservative_miss")

    # 2 — prediction-error penalties (per metric).
    pred = record.prediction
    out = record.outcome
    if (record.prediction_error_goodput is not None
            and pred.predicted_goodput_per_dollar is not None
            and pred.predicted_goodput_per_dollar != 0):
        rel = abs(record.prediction_error_goodput
                  / pred.predicted_goodput_per_dollar)
        if rel >= cfg.large_goodput_error_threshold_pct:
            delta -= cfg.large_goodput_error_penalty
            reasons.append("large_goodput_error")
    if (record.prediction_error_timeout is not None
            and abs(record.prediction_error_timeout)
            >= cfg.large_timeout_error_threshold_pct):
        delta -= cfg.large_timeout_error_penalty
        reasons.append("large_timeout_error")
    if (record.prediction_error_queue_p99 is not None
            and abs(record.prediction_error_queue_p99)
            >= cfg.large_queue_error_threshold_ms):
        delta -= cfg.large_queue_error_penalty
        reasons.append("large_queue_error")

    # 3 — accurate-safe reward only when safety was correct AND there
    # was no false-safe / conservative-miss flag, AND the realized
    # outcome was actually safe (we do not reward unobservable safety).
    if (record.safety_correct is True
            and not record.false_safe
            and not record.false_unsafe
            and not record.conservative_miss
            and out.was_safe is True):
        delta += cfg.accurate_safe_reward
        reasons.append("accurate_safe")

    # 4 — clamp to max_update_per_step and to [min, max].
    delta = _clamp(delta, -cfg.max_update_per_step, cfg.max_update_per_step)
    new_conf = _clamp(cur + delta, cfg.min_confidence, cfg.max_confidence)

    # 5 — invariant: confidence MUST NOT rise when safety was wrong.
    if record.false_safe or record.safety_correct is False:
        if new_conf > cur:
            new_conf = cur
            reasons.append("blocked_rise_unsafe")

    if not reasons:
        reasons.append("no_change")
    return new_conf, ",".join(reasons)


def apply_confidence_update(
    record: DynamicFrontierCalibrationRecord,
    config: Optional[ConfidenceUpdateConfig] = None,
) -> DynamicFrontierCalibrationRecord:
    """Return a new record with the confidence_after / reason filled."""
    new_conf, reason = update_confidence(record, config=config)
    # Dataclass is frozen → construct a new instance.
    return DynamicFrontierCalibrationRecord(
        prediction=record.prediction,
        outcome=record.outcome,
        prediction_error_goodput=record.prediction_error_goodput,
        prediction_error_timeout=record.prediction_error_timeout,
        prediction_error_queue_p99=record.prediction_error_queue_p99,
        risk_calibration_error=record.risk_calibration_error,
        safety_correct=record.safety_correct,
        false_safe=record.false_safe,
        false_unsafe=record.false_unsafe,
        conservative_miss=record.conservative_miss,
        oracle_alpha_available=record.oracle_alpha_available,
        oracle_alpha_captured=record.oracle_alpha_captured,
        oracle_alpha_capture_pct=record.oracle_alpha_capture_pct,
        confidence_before=record.confidence_before,
        confidence_after=new_conf,
        confidence_update_reason=reason,
    )
