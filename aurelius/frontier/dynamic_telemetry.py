"""Telemetry-window builders for the Dynamic Safe Frontier Estimator.

This module accepts three kinds of inputs and normalizes them into a
sequence of :class:`ServingTelemetryTick`:

1. **Azure 2024 aggregated arrival ticks** (the ``ArrivalTick`` dataclass
   produced by ``aurelius/traces/replay.py``) — used by the dynamic
   benchmark.
2. **Existing serving / backtest tick summaries** (per-tick dicts with
   ``goodput_per_dollar`` / ``queue_p99_ms`` / etc.) — used when the
   caller already has aggregated metrics.
3. **Fixture telemetry dicts** — used by tests and pilot bring-up.

Missing fields stay ``None`` (the contract from
``docs/PILOT_TELEMETRY_CONTRACT.md`` §1). The validator returns a
structured result that names every missing required field; it never
silently zero-fills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .dynamic_models import ServingTelemetryTick

# Minimum fields required to estimate the dynamic frontier. ``rps`` +
# at least one queue percentile + an active-replica count is the
# pre-registered minimum.
DEFAULT_REQUIRED_FIELDS = ("observed_rps", "queue_p99_ms", "active_replicas")


@dataclass(frozen=True)
class TelemetryWindowValidation:
    """Result of validating a telemetry window.

    ``ok`` is True iff the window has at least ``min_ticks`` ticks AND
    every required field is present on at least ``min_field_coverage``
    fraction of ticks.
    """

    ok: bool
    reason: str
    n_ticks: int
    min_ticks: int
    required_fields: tuple
    missing_field_coverage: dict  # field -> ticks missing it (count)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "n_ticks": self.n_ticks,
                "min_ticks": self.min_ticks,
                "required_fields": list(self.required_fields),
                "missing_field_coverage": dict(self.missing_field_coverage)}


def _coerce_optional_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN/inf
        return None
    return out


def _coerce_optional_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def telemetry_tick_from_arrival_tick(
    arrival_tick,
    *,
    active_replicas: Optional[int] = None,
    queue_p50_ms: Optional[float] = None,
    queue_p95_ms: Optional[float] = None,
    queue_p99_ms: Optional[float] = None,
    latency_p99_ms: Optional[float] = None,
    timeout_pct: Optional[float] = None,
    sla_violation_pct: Optional[float] = None,
    mean_utilization: Optional[float] = None,
    scale_events_delta: Optional[int] = None,
    churn_delta: Optional[float] = None,
    gpu_hours_delta: Optional[float] = None,
    telemetry_confidence: str = "medium",
    source: str = "arrival_tick",
) -> ServingTelemetryTick:
    """Convert one ``aurelius/traces/replay.ArrivalTick`` to a serving
    telemetry tick. Caller may supply optional observed signals (queue
    percentiles, timeouts, etc.) — anything not supplied stays ``None``.
    """
    duration = max(1e-9, getattr(arrival_tick, "duration_s", 60.0))
    tokens_total = (getattr(arrival_tick, "total_prompt_tokens", 0)
                    + getattr(arrival_tick, "total_output_tokens", 0))
    prompt_tokens_per_s = (getattr(arrival_tick, "total_prompt_tokens", 0)
                           / duration) if duration else None
    output_tokens_per_s = (getattr(arrival_tick, "total_output_tokens", 0)
                           / duration) if duration else None
    # If gpu_hours_delta wasn't supplied but active_replicas is, infer
    # gpu_hours_delta = replicas * (duration / 3600).
    if gpu_hours_delta is None and active_replicas is not None:
        gpu_hours_delta = float(active_replicas) * (duration / 3600.0)
    return ServingTelemetryTick(
        timestamp_s=float(getattr(arrival_tick, "start_s", 0.0)),
        observed_rps=_coerce_optional_float(
            getattr(arrival_tick, "arrival_rate_rps", None)),
        prompt_tokens_per_s=prompt_tokens_per_s,
        output_tokens_per_s=output_tokens_per_s,
        total_tokens_per_s=(tokens_total / duration) if duration else None,
        active_replicas=active_replicas,
        gpu_hours_delta=gpu_hours_delta,
        mean_utilization=mean_utilization,
        queue_p50_ms=queue_p50_ms,
        queue_p95_ms=queue_p95_ms,
        queue_p99_ms=queue_p99_ms,
        latency_p99_ms=latency_p99_ms,
        timeout_pct=timeout_pct,
        sla_violation_pct=sla_violation_pct,
        scale_events_delta=scale_events_delta,
        churn_delta=churn_delta,
        telemetry_confidence=telemetry_confidence,
        source=source,
    )


def telemetry_tick_from_inference_service_state(
    state,
    *,
    timestamp_s: Optional[float] = None,
    tick_duration_s: Optional[float] = None,
    mean_utilization: Optional[float] = None,
    gpu_hours_delta: Optional[float] = None,
    scale_events_delta: Optional[int] = None,
    churn_delta: Optional[float] = None,
    telemetry_confidence: Optional[str] = None,
    source: str = "inference_service_state",
) -> ServingTelemetryTick:
    """Bridge: build a :class:`ServingTelemetryTick` from a real
    :class:`aurelius.state.models.InferenceServiceState` observation.

    Only carries fields the connector actually emitted — everything else
    stays ``None`` (`docs/PILOT_TELEMETRY_CONTRACT.md` §1). In
    particular:

    - ``timeout_pct`` is derived from ``error_rate_pct`` ONLY when both
      are present, because none of vLLM / Triton / Ray Serve metrics
      surface a per-tick timeout share. Pilot deployments that want a
      true timeout share should compute it upstream (latency_p99 vs SLA
      threshold) and pass it via ``InferenceServiceState.error_rate_pct``
      OR set ``timeout_pct`` directly on the resulting tick.
    - ``mean_utilization`` is NOT in :class:`InferenceServiceState` — it
      comes from DCGM (`gpu.util_pct`) or a controller-level signal. The
      caller passes it in.
    - ``gpu_hours_delta`` is inferred from
      ``replicas * tick_duration_s / 3600`` when both are present.
    - ``scale_events_delta`` / ``churn_delta`` are NOT in
      :class:`InferenceServiceState`; they come from K8s deployment
      generation / autoscaling history and are passed in by the caller.

    The result is fully compatible with the dynamic frontier estimator's
    :func:`validate_dynamic_window` (default required fields:
    ``observed_rps``, ``queue_p99_ms``, ``active_replicas``).
    """
    if timestamp_s is None:
        ts = getattr(state, "timestamp", None)
        timestamp_s = (ts.timestamp() if ts is not None else 0.0)

    # Queue depth in vLLM is *requests waiting*, not a millisecond
    # percentile. The frontier estimator expects queue wait in ms — the
    # connector layer already converts engine queue percentiles to ms
    # (see InferenceServiceState.queue_time_p95_ms / p99). Pass them
    # through when present.
    queue_p99 = getattr(state, "queue_time_p99_ms", None)
    if queue_p99 is None:
        # Fall back to e2e p99 latency — overestimates queue wait but
        # documented as a fallback in the audit.
        queue_p99 = getattr(state, "p99_latency_ms", None)
    queue_p95 = getattr(state, "queue_time_p95_ms", None)
    if queue_p95 is None:
        queue_p95 = getattr(state, "p95_latency_ms", None)
    queue_p50 = getattr(state, "queue_time_p50_ms", None)

    # Active replicas: only Ray Serve currently emits this; otherwise the
    # caller must supply it from K8s pod-count or HPA telemetry.
    active_replicas = getattr(state, "replicas", None)

    rps_running = getattr(state, "requests_running", None)
    rps_waiting = getattr(state, "requests_waiting", None)
    tokens_per_s = getattr(state, "tokens_per_s", None)

    # error_rate_pct → timeout_pct (best-effort): when the connector
    # exposes an error rate AND the workload has a hard SLA, the
    # error_rate is the closest real-time analog. Otherwise stay None
    # — pilots can fill it explicitly.
    err_rate = getattr(state, "error_rate_pct", None)
    timeout_pct = err_rate  # honest equivalence — never invent

    latency_p50 = getattr(state, "p50_latency_ms", None)
    latency_p95 = getattr(state, "p95_latency_ms", None)
    latency_p99 = getattr(state, "p99_latency_ms", None)

    # gpu_hours_delta: replicas * duration / 3600 — only when both known.
    if (gpu_hours_delta is None and active_replicas is not None
            and tick_duration_s is not None and tick_duration_s > 0):
        gpu_hours_delta = float(active_replicas) * (
            float(tick_duration_s) / 3600.0)

    # Confidence: honour the connector's provenance if present.
    if telemetry_confidence is None:
        prov = getattr(state, "provenance", None)
        if prov is not None and getattr(prov, "confidence", None) in (
                "low", "medium", "high", "unknown"):
            telemetry_confidence = prov.confidence
        else:
            telemetry_confidence = "medium"

    return ServingTelemetryTick(
        timestamp_s=float(timestamp_s),
        observed_rps=_coerce_optional_float(rps_running),
        prompt_tokens_per_s=None,
        output_tokens_per_s=None,
        total_tokens_per_s=_coerce_optional_float(tokens_per_s),
        active_replicas=_coerce_optional_int(active_replicas),
        gpu_hours_delta=_coerce_optional_float(gpu_hours_delta),
        mean_utilization=_coerce_optional_float(mean_utilization),
        queue_p50_ms=_coerce_optional_float(queue_p50),
        queue_p95_ms=_coerce_optional_float(queue_p95),
        queue_p99_ms=_coerce_optional_float(queue_p99),
        latency_p50_ms=_coerce_optional_float(latency_p50),
        latency_p95_ms=_coerce_optional_float(latency_p95),
        latency_p99_ms=_coerce_optional_float(latency_p99),
        timeout_pct=_coerce_optional_float(timeout_pct),
        sla_violation_pct=None,
        scale_events_delta=_coerce_optional_int(scale_events_delta),
        churn_delta=_coerce_optional_float(churn_delta),
        telemetry_confidence=telemetry_confidence,
        source=source,
    )


def build_serving_telemetry_window(
    ticks: Iterable,
    *,
    source: str = "unknown",
    telemetry_confidence: str = "medium",
) -> list[ServingTelemetryTick]:
    """Normalize ``ticks`` (ArrivalTick, dict, or ServingTelemetryTick) into
    a list of :class:`ServingTelemetryTick`. Order is preserved.
    """
    out: list[ServingTelemetryTick] = []
    for t in ticks:
        if isinstance(t, ServingTelemetryTick):
            out.append(t)
            continue
        if isinstance(t, dict):
            d = dict(t)
            d.setdefault("telemetry_confidence", telemetry_confidence)
            d.setdefault("source", source)
            # Drop any keys not in the dataclass to stay strict.
            allowed = set(ServingTelemetryTick.__dataclass_fields__)
            payload = {k: d.get(k) for k in allowed}
            payload["timestamp_s"] = float(payload.get("timestamp_s") or 0.0)
            for k in ("observed_rps", "prompt_tokens_per_s",
                      "output_tokens_per_s", "total_tokens_per_s",
                      "gpu_hours_delta", "mean_utilization",
                      "queue_p50_ms", "queue_p95_ms", "queue_p99_ms",
                      "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
                      "timeout_pct", "sla_violation_pct", "churn_delta"):
                payload[k] = _coerce_optional_float(payload.get(k))
            for k in ("active_replicas", "scale_events_delta"):
                payload[k] = _coerce_optional_int(payload.get(k))
            payload.setdefault("telemetry_confidence", telemetry_confidence)
            payload.setdefault("source", source)
            out.append(ServingTelemetryTick(**payload))
            continue
        # Assume an aurelius/traces/replay.ArrivalTick-like object
        out.append(telemetry_tick_from_arrival_tick(
            t, telemetry_confidence=telemetry_confidence, source=source))
    return out


def validate_dynamic_window(
    window: Sequence[ServingTelemetryTick],
    *,
    min_ticks: int = 8,
    required_fields: Sequence[str] = DEFAULT_REQUIRED_FIELDS,
    min_field_coverage: float = 0.5,
) -> TelemetryWindowValidation:
    """Validate that the window has enough ticks and field coverage."""
    n = len(window) if window is not None else 0
    if n < min_ticks:
        return TelemetryWindowValidation(
            ok=False,
            reason=f"n_ticks={n} below minimum {min_ticks}",
            n_ticks=n, min_ticks=min_ticks,
            required_fields=tuple(required_fields),
            missing_field_coverage={})
    missing_cov: dict = {}
    for f in required_fields:
        missing = sum(1 for t in window
                      if getattr(t, f, None) is None)
        if missing:
            missing_cov[f] = missing
        if (n - missing) / max(1, n) < min_field_coverage:
            return TelemetryWindowValidation(
                ok=False,
                reason=(f"field {f!r} present on only "
                        f"{n - missing}/{n} ticks "
                        f"({(n - missing) / n:.2%} < required "
                        f"{min_field_coverage:.0%})"),
                n_ticks=n, min_ticks=min_ticks,
                required_fields=tuple(required_fields),
                missing_field_coverage=missing_cov)
    return TelemetryWindowValidation(
        ok=True, reason="ok", n_ticks=n, min_ticks=min_ticks,
        required_fields=tuple(required_fields),
        missing_field_coverage=missing_cov)
