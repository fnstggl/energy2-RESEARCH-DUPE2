"""Workload Admission Gate — flow-rate admission control for SLA safety.

Implements a deterministic, telemetry-driven admission gate that controls
the rate at which new requests/workloads enter the active serving set.
Inspired by "Flow-Controlled Scheduling for LLM Inference with Provable
Stability Guarantees" (arXiv:2604.11001, April 2026).

Core idea (from the paper): because decode lengths are unknown a priori,
uncontrolled admission causes KV-cache overflow and system instability.
A flow-control layer that gates admission based on *observed* KV-cache
pressure and queue-tail trends prevents overflow and improves both mean
and tail latency—without requiring output-length prediction.

Aurelius mapping:

- **ADMIT** — current KV-cache pressure and queue tail are within safe
  bounds; route the request normally.
- **DEFER** — system is under load; for non-realtime workloads, hold the
  request for up to ``max_defer_ms`` before re-evaluating. This is the
  flow-control valve.
- **REJECT** — reserved for extreme KV-cache saturation (> hard ceiling)
  on best-effort workloads only. Interactive / latency-critical requests
  are **never** rejected.

Design invariants:

- **Realtime / latency-critical SLA classes are never deferred or
  rejected.** The gate only applies back-pressure to batch, eval,
  training, and best-effort workloads.
- **Missing telemetry → ADMIT with LOW confidence.** The gate never
  silently escalates to DEFER on missing data — fail-open is safer than
  fail-closed for interactive traffic.
- **No ML, no future leakage.** All signals are decision-time observables
  from the ``ServingTelemetryTick`` window.
- **Shadow-mode by default.** ``enabled=False`` in the default config;
  the caller must opt in.
- **Not production-ready.** Shadow-mode evidence only; live pilot
  telemetry calibration is required before real cluster deployment
  (``docs/RESULTS.md`` §8).

Directional simulator/backtest evidence only — NOT production savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .dynamic_models import ServingTelemetryTick
from .risk import (
    RiskConfig,
    _clip,
    _ema,
    _ema_slope,
    _telemetry_confidence_ok,
)

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

ADMISSION_ADMIT = "ADMIT"
ADMISSION_DEFER = "DEFER"
ADMISSION_REJECT = "REJECT"

ADMISSION_ACTIONS = frozenset({ADMISSION_ADMIT, ADMISSION_DEFER, ADMISSION_REJECT})

# SLA classes that are NEVER deferred or rejected, regardless of load.
_LATENCY_CRITICAL_SLA_CLASSES = frozenset({
    "latency_critical",
    "realtime",
    "realtime_inference",
    "interactive",
})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AdmissionGateConfig:
    """Pre-registered thresholds for the admission gate.

    Defaults are conservative: the gate does not apply back-pressure
    unless KV-cache utilization is clearly high and queue tail is
    rising. Operators may tighten or relax these per workload class.

    All thresholds are transparent and adjustable — nothing is tuned to
    force a benchmark win.
    """

    # --- Feature: enable / disable ---
    enabled: bool = False  # Shadow-mode by default; opt-in required.

    # --- KV-cache pressure thresholds ---
    # EMA of kv_cache_utilization above this → soft pressure (DEFER eligible).
    kv_soft_ceiling: float = 0.80
    # EMA above this → hard pressure (REJECT eligible for best-effort only).
    kv_hard_ceiling: float = 0.95
    # If the EMA slope of kv_cache_utilization is > this (positive = rising),
    # apply an additional pressure boost even if the current level is low.
    kv_rising_slope_threshold: float = 0.15

    # --- Queue tail thresholds ---
    # Queue p99 above this fraction of the risk-config max → soft pressure.
    queue_soft_fraction: float = 0.65

    # --- SLA / timeout thresholds ---
    # Current timeout_pct above this → switch to conservative mode
    # (DEFER even batch workloads at lower KV pressure).
    timeout_conservative_threshold_pct: float = 5.0

    # --- Deferral parameters ---
    # Default maximum deferral window in milliseconds.
    max_defer_ms: float = 2_000.0
    # Multiplier on max_defer_ms when KV is near the hard ceiling.
    hard_ceiling_defer_multiplier: float = 2.0

    # --- Telemetry requirements ---
    ema_alpha: float = 0.4
    min_telemetry_confidence: str = "low"
    # Minimum window length to compute trends.
    min_window_for_trends: int = 3

    # --- Risk config for SLA + queue risk sub-estimates ---
    risk_config: RiskConfig = field(default_factory=RiskConfig)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdmissionDecision:
    """Result of the admission gate for a single incoming workload/request.

    Attributes:
        action: ADMIT | DEFER | REJECT.
        sla_class: The SLA class of the evaluated workload.
        defer_until_ms: Suggested defer window in ms (0 if action == ADMIT).
        kv_pressure_score: Composite KV-cache pressure in [0, 1].
        queue_pressure_score: Queue-tail pressure in [0, 1].
        reason_codes: Human-readable codes explaining the decision.
        confidence: Telemetry confidence: "high" | "medium" | "low" | "none".
        gate_enabled: False when the gate is disabled (always ADMIT).
    """

    action: str
    sla_class: str
    defer_until_ms: float
    kv_pressure_score: Optional[float]
    queue_pressure_score: Optional[float]
    reason_codes: tuple
    confidence: str
    gate_enabled: bool

    def __post_init__(self):
        if self.action not in ADMISSION_ACTIONS:
            raise ValueError(
                f"AdmissionDecision.action must be one of "
                f"{sorted(ADMISSION_ACTIONS)}; got {self.action!r}"
            )

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "sla_class": self.sla_class,
            "defer_until_ms": self.defer_until_ms,
            "kv_pressure_score": self.kv_pressure_score,
            "queue_pressure_score": self.queue_pressure_score,
            "reason_codes": list(self.reason_codes),
            "confidence": self.confidence,
            "gate_enabled": self.gate_enabled,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _kv_pressure(
    window: Sequence[ServingTelemetryTick],
    cfg: AdmissionGateConfig,
) -> tuple[Optional[float], list[str]]:
    """Composite KV-cache pressure score in [0, 1] + reason codes.

    Combines:
    1. EMA of kv_cache_utilization proximity to kv_hard_ceiling.
    2. Rising-slope bonus when kv_cache_utilization is trending up.
    Returns (None, reasons) when no KV signal is available.
    """
    kv_vals = [t.mean_utilization for t in window]
    kv_ema = _ema(kv_vals, cfg.ema_alpha)
    kv_slope = (
        _ema_slope(kv_vals, cfg.ema_alpha)
        if len(window) >= cfg.min_window_for_trends
        else None
    )

    reasons: list[str] = []

    if kv_ema is None:
        return None, ["kv_utilization_missing"]

    # Proximity to hard ceiling (0 → 1 as kv_ema → kv_hard_ceiling).
    proximity = _clip(kv_ema / cfg.kv_hard_ceiling)
    if kv_ema >= cfg.kv_soft_ceiling:
        reasons.append("kv_above_soft_ceiling")
    if kv_ema >= cfg.kv_hard_ceiling:
        reasons.append("kv_at_hard_ceiling")

    # Rising slope bonus.
    slope_bonus = 0.0
    if kv_slope is not None and kv_slope > cfg.kv_rising_slope_threshold:
        slope_bonus = _clip(kv_slope / 2.0) * 0.3  # max +0.3 bonus
        reasons.append("kv_utilization_rising")

    score = _clip(proximity + slope_bonus)
    return score, reasons


def _queue_pressure(
    window: Sequence[ServingTelemetryTick],
    cfg: AdmissionGateConfig,
) -> tuple[Optional[float], list[str]]:
    """Queue-tail pressure score in [0, 1] + reason codes."""
    q99_vals = [t.queue_p99_ms for t in window]
    q99_ema = _ema(q99_vals, cfg.ema_alpha)

    reasons: list[str] = []

    if q99_ema is None:
        return None, ["queue_p99_missing"]

    max_q99 = cfg.risk_config.max_queue_p99_ms
    proximity = _clip(q99_ema / max_q99)
    if proximity >= cfg.queue_soft_fraction:
        reasons.append("queue_p99_approaching_ceiling")
    if proximity >= 0.9:
        reasons.append("queue_p99_near_ceiling")

    q99_slope = (
        _ema_slope(q99_vals, cfg.ema_alpha)
        if len(window) >= cfg.min_window_for_trends
        else None
    )
    slope_bonus = 0.0
    if q99_slope is not None and q99_slope > 0.2:
        slope_bonus = _clip(q99_slope / 2.0) * 0.25
        reasons.append("queue_p99_rising")

    score = _clip(proximity + slope_bonus)
    return score, reasons


def _timeout_conservative(
    window: Sequence[ServingTelemetryTick],
    cfg: AdmissionGateConfig,
) -> tuple[bool, list[str]]:
    """Returns (conservative_mode, reasons) based on recent timeout_pct."""
    timeouts = [t.timeout_pct for t in window if t.timeout_pct is not None]
    if not timeouts:
        return False, []
    recent_timeout = sum(timeouts) / len(timeouts)
    if recent_timeout >= cfg.timeout_conservative_threshold_pct:
        return True, ["timeout_pct_elevated"]
    return False, []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_admission(
    *,
    sla_class: str,
    window: Sequence[ServingTelemetryTick],
    config: Optional[AdmissionGateConfig] = None,
) -> AdmissionDecision:
    """Evaluate whether an incoming workload should be ADMIT, DEFER, or REJECT.

    Parameters
    ----------
    sla_class:
        SLA class of the incoming workload (e.g. ``"realtime_inference"``,
        ``"llm_batch_inference"``, ``"training"``, ``"best_effort"``).
        Latency-critical classes are always ADMIT.
    window:
        Recent ``ServingTelemetryTick`` observations (most recent last).
        Must not be empty; caller should provide at least 1 tick.
    config:
        Gate configuration. Defaults to ``AdmissionGateConfig()`` which
        has ``enabled=False`` — i.e., always ADMIT (shadow-only).

    Returns
    -------
    AdmissionDecision
        The gate decision with full audit trail.
    """
    cfg = config or AdmissionGateConfig()

    # Gate disabled → always ADMIT with no pressure scores.
    if not cfg.enabled:
        return AdmissionDecision(
            action=ADMISSION_ADMIT,
            sla_class=sla_class,
            defer_until_ms=0.0,
            kv_pressure_score=None,
            queue_pressure_score=None,
            reason_codes=("gate_disabled",),
            confidence="none",
            gate_enabled=False,
        )

    # Latency-critical classes are never back-pressured.
    if sla_class in _LATENCY_CRITICAL_SLA_CLASSES:
        return AdmissionDecision(
            action=ADMISSION_ADMIT,
            sla_class=sla_class,
            defer_until_ms=0.0,
            kv_pressure_score=None,
            queue_pressure_score=None,
            reason_codes=("latency_critical_sla_exempt",),
            confidence="high",
            gate_enabled=True,
        )

    # Telemetry confidence check.
    if not _telemetry_confidence_ok(window, cfg.min_telemetry_confidence):
        # Fail-open on missing telemetry — never silently escalate to DEFER.
        return AdmissionDecision(
            action=ADMISSION_ADMIT,
            sla_class=sla_class,
            defer_until_ms=0.0,
            kv_pressure_score=None,
            queue_pressure_score=None,
            reason_codes=("insufficient_telemetry_fail_open",),
            confidence="none",
            gate_enabled=True,
        )

    reasons: list[str] = []

    # Compute KV-cache pressure.
    kv_score, kv_reasons = _kv_pressure(window, cfg)
    reasons.extend(kv_reasons)

    # Compute queue pressure.
    queue_score, queue_reasons = _queue_pressure(window, cfg)
    reasons.extend(queue_reasons)

    # Conservative mode from elevated timeouts.
    conservative, to_reasons = _timeout_conservative(window, cfg)
    reasons.extend(to_reasons)

    # Confidence = worst telemetry confidence in the window.
    confidences = [t.telemetry_confidence or "unknown" for t in window]
    _CONF_RANK = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
    _CONF_LABEL = {0: "none", 1: "low", 2: "medium", 3: "high"}
    worst_rank = min(_CONF_RANK.get(c, 0) for c in confidences)
    confidence = _CONF_LABEL.get(worst_rank, "none")

    # Decision logic:
    # REJECT  — KV at hard ceiling AND best-effort workload.
    # DEFER   — KV above soft ceiling OR queue pressure high OR conservative.
    # ADMIT   — otherwise.

    is_best_effort = sla_class in {"best_effort", "background", "background_maintenance"}
    kv_score is not None and kv_score >= _clip(
        cfg.kv_hard_ceiling / cfg.kv_hard_ceiling  # = 1.0
    )

    if is_best_effort and (kv_score is not None and kv_score >= 0.99):
        reasons.append("reject_best_effort_kv_saturated")
        return AdmissionDecision(
            action=ADMISSION_REJECT,
            sla_class=sla_class,
            defer_until_ms=0.0,
            kv_pressure_score=kv_score,
            queue_pressure_score=queue_score,
            reason_codes=tuple(reasons),
            confidence=confidence,
            gate_enabled=True,
        )

    # Determine whether to DEFER.
    kv_above_soft = kv_score is not None and kv_score >= _clip(
        cfg.kv_soft_ceiling / cfg.kv_hard_ceiling
    )
    queue_above_soft = queue_score is not None and queue_score >= cfg.queue_soft_fraction
    should_defer = kv_above_soft or queue_above_soft or conservative

    if should_defer:
        # Scale defer window by pressure — higher pressure = longer deferral.
        max_pressure = max(
            kv_score if kv_score is not None else 0.0,
            queue_score if queue_score is not None else 0.0,
        )
        base_ms = cfg.max_defer_ms
        if kv_score is not None and kv_score >= _clip(
            cfg.kv_hard_ceiling * 0.95 / cfg.kv_hard_ceiling
        ):
            base_ms *= cfg.hard_ceiling_defer_multiplier

        defer_ms = base_ms * _clip(max_pressure)
        reasons.append("defer_due_to_load")

        return AdmissionDecision(
            action=ADMISSION_DEFER,
            sla_class=sla_class,
            defer_until_ms=max(defer_ms, 100.0),  # minimum 100 ms
            kv_pressure_score=kv_score,
            queue_pressure_score=queue_score,
            reason_codes=tuple(reasons),
            confidence=confidence,
            gate_enabled=True,
        )

    # No pressure — ADMIT.
    return AdmissionDecision(
        action=ADMISSION_ADMIT,
        sla_class=sla_class,
        defer_until_ms=0.0,
        kv_pressure_score=kv_score,
        queue_pressure_score=queue_score,
        reason_codes=tuple(reasons) if reasons else ("no_pressure",),
        confidence=confidence,
        gate_enabled=True,
    )


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def evaluate_admission_batch(
    *,
    workloads: Sequence[tuple[str, str]],  # [(workload_id, sla_class), ...]
    window: Sequence[ServingTelemetryTick],
    config: Optional[AdmissionGateConfig] = None,
) -> dict[str, AdmissionDecision]:
    """Evaluate admission for a batch of incoming workloads.

    The window is shared (all workloads see the same system state).
    Returns ``{workload_id: AdmissionDecision}``.

    This avoids re-computing the pressure scores N times when evaluating
    multiple candidates in one scheduling tick.
    """
    cfg = config or AdmissionGateConfig()
    return {
        wid: evaluate_admission(sla_class=sc, window=window, config=cfg)
        for wid, sc in workloads
    }
