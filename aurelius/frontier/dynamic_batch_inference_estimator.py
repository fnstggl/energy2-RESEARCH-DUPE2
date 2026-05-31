"""Dynamic Batch Inference Frontier v1 — telemetry-driven estimator + controller.

A **sibling** of:
- the static Batch Inference Frontier (``batch_inference_*.py``),
- the Dynamic Safe Frontier Estimator for serving rho
  (``dynamic_estimator.py`` / ``dynamic_controller.py``).

Hard invariants (asserted by tests):

- This module does **NOT** import any serving rho controller module
  (``controller.py``, ``dynamic_controller.py``, ``estimator.py``,
  ``dynamic_estimator.py``). The dynamic batch estimator computes its
  predictions over batch-shape arrival ticks directly.
- No future leakage. The estimator may only read the recent window the
  caller passes in.
- Missing telemetry stays ``None`` — never zero-filled.
- ``DynamicBatchInferenceDecision.executable_in_real_cluster`` is
  ``False`` at construction; constructing with ``True`` raises.
- Recommendation-only at v1. Real cluster execution requires both
  ``allow_real_execution=True`` AND a non-stub executor (the shim ships
  only a no-op default).
- No ML in v1. Predictions come from deterministic EWMA next-tick
  projections + the unchanged serving physics in
  ``aurelius/traces/backtest.py``.
- Synthetic deadlines are scenario knobs the caller passes in. The
  dynamic batch estimator never reads a real deadline from a serving
  trace itself.

Multi-axis candidate space (multi-dimensional — not a scalar rho):

    (target_rho, batch_window_seconds, batch_concurrency,
     deferral_window_seconds, deadline_slack_seconds)

The new degree of freedom — ``deferral_window_seconds`` — is the lever
that justifies a batch-specific dynamic estimator over the existing
serving dynamic estimator: at peak ticks the batch frontier can defer a
fraction of arrivals into a subsequent tick (within the deadline-slack
budget) to keep peak rho safer than a serving estimator could.

This module is **research / audit code**; integration into the
constraint-aware scheduler is gated by the
``docs/BATCH_INFERENCE_FRONTIER_INCREMENTAL_ALPHA_AUDIT.md`` 2% gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence

from ..traces.replay import ArrivalTick
from .batch_inference_models import (
    BatchInferenceFrontierCandidate,
    BatchInferenceFrontierPoint,
    BatchInferenceFrontierSchemaError,
    BatchInferenceSafetyStatus,
    BatchInferenceWorkloadProfile,
    EXECUTION_MODE_SHADOW,
    _BATCH_EXECUTION_MODES,
)
from .batch_inference_safety import (
    BatchInferenceSafetyConfig,
    classify_batch_point_safety,
)


# ---------------------------------------------------------------------------
# Decision enum specific to the DYNAMIC batch controller.
# ---------------------------------------------------------------------------

DYNAMIC_BATCH_ACTIONS = frozenset({
    "RECOMMEND_BATCH_FRONTIER",
    "KEEP_CURRENT_BATCH_POLICY",
    "LOWER_BATCH_PRESSURE",
    "DEFER_BURST",            # peak-shift specific to batch
    "ISOLATE_FROM_INTERACTIVE",
    "INSUFFICIENT_TELEMETRY",
})


# ---------------------------------------------------------------------------
# Dynamic models.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatchArrivalTelemetryTick:
    """One observed tick of batch-shape arrival telemetry.

    Every numeric field is ``Optional``. Missing telemetry stays ``None``.
    """

    timestamp_s: float
    arrival_rate_rps: Optional[float] = None
    prompt_tokens_mean: Optional[float] = None
    output_tokens_mean: Optional[float] = None
    total_output_tokens: Optional[int] = None
    request_count: Optional[int] = None
    active_replicas: Optional[int] = None
    observed_rho: Optional[float] = None
    queue_p99_ms: Optional[float] = None
    timeout_pct: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    deadline_miss_pct: Optional[float] = None
    deferred_arrivals_pending: Optional[int] = None
    telemetry_confidence: str = "unknown"
    source: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DynamicBatchFrontierEstimate:
    """Per-window frontier estimate produced by the dynamic batch estimator.

    Carries the candidate sweep at the predicted next-tick load, plus
    summary descriptors (window length, current candidate, fallback
    reason on empty window).
    """

    workload_id: str
    window_ticks: int
    current_candidate: Optional[BatchInferenceFrontierCandidate]
    recommended_candidate: Optional[BatchInferenceFrontierCandidate]
    recommended_point: Optional[BatchInferenceFrontierPoint]
    candidate_points: tuple
    risk_at_current: Optional[float]
    fallback_reason: Optional[str]
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "workload_id": self.workload_id,
            "window_ticks": self.window_ticks,
            "current_candidate": (self.current_candidate.to_dict()
                                  if self.current_candidate is not None
                                  else None),
            "recommended_candidate": (
                self.recommended_candidate.to_dict()
                if self.recommended_candidate is not None else None),
            "recommended_point": (self.recommended_point.to_dict()
                                  if self.recommended_point is not None
                                  else None),
            "candidate_points": [p.to_dict() for p in self.candidate_points],
            "risk_at_current": self.risk_at_current,
            "fallback_reason": self.fallback_reason,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class DynamicBatchInferenceDecision:
    """Recommendation-only dynamic batch-frontier decision.

    Hard invariants:
    - ``executable_in_real_cluster`` is ``False`` at construction;
      constructing with ``True`` raises.
    - ``action`` is one of ``DYNAMIC_BATCH_ACTIONS``.
    - ``execution_mode`` is one of the batch-execution modes.
    """

    workload_id: str
    current_candidate: Optional[BatchInferenceFrontierCandidate]
    recommended_candidate: Optional[BatchInferenceFrontierCandidate]
    recommended_point: Optional[BatchInferenceFrontierPoint]
    action: str
    reason: str
    expected_goodput_per_dollar_delta: Optional[float] = None
    expected_deadline_miss_delta_pct: Optional[float] = None
    risk_at_current: Optional[float] = None
    confidence: str = "unknown"
    execution_mode: str = EXECUTION_MODE_SHADOW
    executable_in_simulator: bool = True
    executable_in_real_cluster: bool = False
    safety_vetoes: tuple = ()
    hysteresis_applied: bool = False
    source: str = "dynamic_batch_inference_frontier_v1"
    notes: tuple = ()

    def __post_init__(self):
        if self.action not in DYNAMIC_BATCH_ACTIONS:
            raise BatchInferenceFrontierSchemaError(
                f"unknown dynamic batch action {self.action!r}; "
                f"expected one of {sorted(DYNAMIC_BATCH_ACTIONS)}")
        if self.execution_mode not in _BATCH_EXECUTION_MODES:
            raise BatchInferenceFrontierSchemaError(
                f"unknown execution_mode {self.execution_mode!r}")
        if self.executable_in_real_cluster:
            raise BatchInferenceFrontierSchemaError(
                "dynamic-batch-frontier decisions are recommendation-only "
                "at construction; real execution requires the caller to "
                "pass allow_real_execution=True to "
                "execute_dynamic_batch_inference_decision (no-op stub by "
                "default)")

    @property
    def is_actionable(self) -> bool:
        return self.action in (
            "RECOMMEND_BATCH_FRONTIER",
            "LOWER_BATCH_PRESSURE",
            "DEFER_BURST",
            "ISOLATE_FROM_INTERACTIVE",
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["current_candidate"] = (self.current_candidate.to_dict()
                                  if self.current_candidate is not None
                                  else None)
        d["recommended_candidate"] = (self.recommended_candidate.to_dict()
                                      if self.recommended_candidate is not None
                                      else None)
        d["recommended_point"] = (self.recommended_point.to_dict()
                                  if self.recommended_point is not None
                                  else None)
        d["safety_vetoes"] = list(self.safety_vetoes)
        d["notes"] = list(self.notes)
        return d


# ---------------------------------------------------------------------------
# Estimator configuration.
# ---------------------------------------------------------------------------

@dataclass
class DynamicBatchEstimatorConfig:
    """Settings for the dynamic batch frontier estimator.

    All values are pre-registered defaults — never tuned per trace.
    """

    min_window_ticks: int = 8
    ewma_alpha: float = 0.4
    # Sweep grid:
    candidate_rhos: tuple = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    candidate_deadline_slack_seconds: tuple = (60.0, 300.0, 900.0, 3600.0)
    # Deferral window — the new degree of freedom vs the serving dynamic
    # estimator. 0 = no deferral (parity with serving). Strictly positive
    # values let the estimator project peak-shifting into the future
    # within the deadline budget.
    candidate_deferral_seconds: tuple = (0.0, 60.0, 300.0)
    candidate_batch_concurrency: tuple = (1, 2)
    # Risk threshold above which the current candidate is "UNSAFE NOW".
    unsafe_risk_threshold: float = 0.75
    # Used to compute risk_at_current from observed timeout %.
    risk_timeout_floor_pct: float = 1.0
    risk_timeout_ceiling_pct: float = 10.0
    # Used to compute risk from queue_p99.
    risk_queue_p99_floor_ms: float = 200.0
    risk_queue_p99_ceiling_ms: float = 2000.0


# ---------------------------------------------------------------------------
# Controller configuration.
# ---------------------------------------------------------------------------

@dataclass
class DynamicBatchControllerConfig:
    """Settings for choose_dynamic_batch_decision."""

    rho_deadband: float = 0.05
    slack_deadband_seconds: float = 60.0
    deferral_deadband_seconds: float = 30.0
    kpi_deadband_pct: float = 0.02
    lower_pressure_risk_threshold: float = 0.75
    # If the recommendation's deferral > 0, emit DEFER_BURST (the
    # peak-shift action) rather than RECOMMEND_BATCH_FRONTIER. This is
    # what makes the dynamic batch decision distinguishable from the
    # dynamic serving decision in observability.
    expose_defer_burst_action: bool = True
    churn_suppresses_raise: bool = True


# ---------------------------------------------------------------------------
# Estimator implementation.
# ---------------------------------------------------------------------------

def _bt():  # pragma: no cover - import indirection
    from aurelius.traces import backtest as bt
    return bt


def _ewma_window(values: Sequence[float], alpha: float
                 ) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    s = present[0]
    for v in present[1:]:
        s = alpha * v + (1.0 - alpha) * s
    return s


def _risk_at_current(window: Sequence[BatchArrivalTelemetryTick],
                     cfg: DynamicBatchEstimatorConfig) -> Optional[float]:
    """Bounded risk score in [0, 1] from observed timeout + queue."""
    timeouts = [t.timeout_pct for t in window]
    queues = [t.queue_p99_ms for t in window]
    to_ewma = _ewma_window(timeouts, cfg.ewma_alpha)
    q_ewma = _ewma_window(queues, cfg.ewma_alpha)
    if to_ewma is None and q_ewma is None:
        return None
    parts: list[float] = []
    if to_ewma is not None:
        span = max(1e-9, cfg.risk_timeout_ceiling_pct
                   - cfg.risk_timeout_floor_pct)
        r = (to_ewma - cfg.risk_timeout_floor_pct) / span
        parts.append(max(0.0, min(1.0, r)))
    if q_ewma is not None:
        span = max(1e-9, cfg.risk_queue_p99_ceiling_ms
                   - cfg.risk_queue_p99_floor_ms)
        r = (q_ewma - cfg.risk_queue_p99_floor_ms) / span
        parts.append(max(0.0, min(1.0, r)))
    return sum(parts) / len(parts)


def _project_next_tick(window: Sequence[BatchArrivalTelemetryTick],
                       ewma_alpha: float) -> dict:
    """Project the next-tick arrival shape from the recent window."""
    rate = _ewma_window([t.arrival_rate_rps for t in window], ewma_alpha)
    pmean = _ewma_window([t.prompt_tokens_mean for t in window], ewma_alpha)
    omean = _ewma_window([t.output_tokens_mean for t in window], ewma_alpha)
    rc = _ewma_window([float(t.request_count or 0) for t in window],
                      ewma_alpha)
    return {
        "arrival_rate_rps": rate or 0.0,
        "prompt_tokens_mean": pmean or 0.0,
        "output_tokens_mean": omean or 0.0,
        "request_count": int(rc or 0),
    }


def _build_synthetic_arrival_tick(proj: dict, *,
                                  defer_fraction: float,
                                  tick_seconds: float) -> ArrivalTick:
    """Construct a one-tick ArrivalTick from the projected next-tick load,
    with ``defer_fraction`` of work shifted out (modelled as a lower
    effective request count + tokens for THIS tick; the deferred mass is
    accounted for elsewhere in the dynamic replay)."""
    effective_rate = max(0.0, proj["arrival_rate_rps"]
                         * (1.0 - max(0.0, min(1.0, defer_fraction))))
    effective_count = max(0, int(round(proj["request_count"]
                                        * (1.0 - defer_fraction))))
    total_prompt = int(round(effective_count * proj["prompt_tokens_mean"]))
    total_output = int(round(effective_count * proj["output_tokens_mean"]))
    return ArrivalTick(
        tick_index=0,
        start_s=0.0,
        end_s=tick_seconds,
        duration_s=tick_seconds,
        request_count=effective_count,
        arrival_rate_rps=effective_rate,
        prompt_tokens_mean=proj["prompt_tokens_mean"],
        output_tokens_mean=proj["output_tokens_mean"],
        total_prompt_tokens=total_prompt,
        total_output_tokens=total_output,
        failures=0,
        distinct_cache_keys=0,
        reuse_fraction=0.0,
        model_mix={"azure-llm": effective_count} if effective_count
                  else {},
        log_type_mix={"unknown": effective_count} if effective_count
                     else {},
    )


def _defer_fraction_for(deferral_seconds: float, projected_rps: float,
                        target_rho: float, output_mean: float,
                        throughput_tokps: float) -> float:
    """Translate a per-window deferral budget into a fractional shed.

    If projected rho > target rho, defer enough fraction to bring it back
    down to target rho. If projected rho <= target rho, no deferral.
    Caller-tunable via ``deferral_seconds`` — when 0, fraction = 0.
    """
    if deferral_seconds <= 0:
        return 0.0
    bt = _bt()
    mu_full = bt._mu_full(max(1.0, output_mean), throughput_tokps)
    if mu_full <= 0 or projected_rps <= 0:
        return 0.0
    proj_rho = projected_rps / (1 * mu_full)
    if proj_rho <= target_rho:
        return 0.0
    # Defer down to target rho.
    max_fraction = max(0.0, (proj_rho - target_rho) / proj_rho)
    # Bound by the deferral budget: we cannot defer more than what a
    # 1-tick window can absorb, modeled here as 1.0 (full tick worth).
    return min(max_fraction, 1.0)


def _evaluate_candidate(window: Sequence[BatchArrivalTelemetryTick],
                        cand: BatchInferenceFrontierCandidate,
                        *, cfg: DynamicBatchEstimatorConfig,
                        tick_seconds: float) -> dict:
    """Predict next-tick KPI + deadline-miss for one candidate."""
    bt = _bt()
    proj = _project_next_tick(window, cfg.ewma_alpha)
    if proj["request_count"] <= 0:
        return {
            "predicted_goodput_per_dollar": 0.0,
            "predicted_sla_safe_goodput": 0.0,
            "predicted_deadline_miss_rate_pct": 0.0,
            "predicted_timeout_rate_pct": 0.0,
            "predicted_queue_p95_ms": 0.0,
            "predicted_queue_p99_ms": 0.0,
            "predicted_latency_p95_ms": 0.0,
            "predicted_latency_p99_ms": 0.0,
            "predicted_gpu_hours": 0.0,
            "predicted_mean_utilization": float(cand.target_rho or 0.0),
            "predicted_cost_per_sla_compliant_token": None,
        }
    R = cand.target_rho or 0.65
    deferral_s = (cand.deferral_window_seconds
                  if cand.deferral_window_seconds is not None else 0.0)
    # Choose the actual fraction to defer based on the projected rho.
    synth = _build_synthetic_arrival_tick(
        proj, defer_fraction=0.0, tick_seconds=tick_seconds)
    throughput = bt._tick_throughput_tokps(synth)
    defer_frac = _defer_fraction_for(
        deferral_s, proj["arrival_rate_rps"], R,
        proj["output_tokens_mean"], throughput)
    tick = _build_synthetic_arrival_tick(
        proj, defer_fraction=defer_frac, tick_seconds=tick_seconds)
    tick_hours = tick_seconds / 3600.0
    replicas = bt._size_for_target(
        tick.arrival_rate_rps, max(1.0, tick.output_tokens_mean),
        bt._tick_throughput_tokps(tick), R)
    ev = bt.evaluate_tick(tick, replicas, prefill_savings=0.0,
                          tick_hours=tick_hours)
    # Deadline-miss: predicted p99 latency vs sla_ms + deadline slack.
    slack_ms = (1000.0 * cand.deadline_slack_seconds
                if cand.deadline_slack_seconds is not None else None)
    if slack_ms is None or tick.request_count <= 0:
        deadline_miss_pct = None
    else:
        budget = ev.sla_ms + slack_ms
        deadline_miss_pct = 100.0 if ev.latency_p99_ms > budget else 0.0
    # goodput/$ -> use the same shape as the existing batch estimator
    # (sla_compliant_tokens / cost). For per-tick prediction we
    # approximate goodput as output_tokens × (1 - timeout/100).
    sla_safe_tokens = (tick.total_output_tokens
                       * max(0.0, 1.0 - ev.timeout_rate_pct / 100.0))
    gpu_hours = sum(ev.gpu_hours_by_type.values()) if ev.gpu_hours_by_type else 0.0
    # public-list cost model — same as the existing batch estimator.
    GPU_HOURLY_USD = 2.04
    gpu_cost = gpu_hours * GPU_HOURLY_USD
    total_cost = gpu_cost + ev.energy_cost
    goodput_per_dollar = (sla_safe_tokens / total_cost
                          if total_cost > 0 else 0.0)
    return {
        "predicted_goodput_per_dollar": float(goodput_per_dollar),
        "predicted_sla_safe_goodput": float(sla_safe_tokens),
        "predicted_deadline_miss_rate_pct": deadline_miss_pct,
        "predicted_timeout_rate_pct": float(ev.timeout_rate_pct),
        "predicted_queue_p95_ms": float(ev.queue_wait_p95_ms),
        "predicted_queue_p99_ms": float(ev.queue_wait_p99_ms),
        "predicted_latency_p95_ms": float(ev.latency_p95_ms),
        "predicted_latency_p99_ms": float(ev.latency_p99_ms),
        "predicted_gpu_hours": float(gpu_hours),
        "predicted_mean_utilization": float(ev.rho),
        "predicted_cost_per_sla_compliant_token": (
            float(total_cost / sla_safe_tokens)
            if sla_safe_tokens > 0 else None),
    }


def estimate_dynamic_batch_frontier(
    profile: BatchInferenceWorkloadProfile,
    window: Sequence[BatchArrivalTelemetryTick],
    *,
    current_candidate: Optional[BatchInferenceFrontierCandidate] = None,
    candidates: Optional[Iterable[BatchInferenceFrontierCandidate]] = None,
    estimator_config: Optional[DynamicBatchEstimatorConfig] = None,
    safety_config: Optional[BatchInferenceSafetyConfig] = None,
    tick_seconds: float = 60.0,
) -> DynamicBatchFrontierEstimate:
    """Estimate the dynamic batch frontier for one workload over the
    recent ``window``.

    Pure / deterministic / stdlib-only. Per-tick fallback when the
    window is too short or empty: ``recommended_candidate = None`` and
    a populated ``fallback_reason``.
    """
    cfg = estimator_config or DynamicBatchEstimatorConfig()
    safety = safety_config or BatchInferenceSafetyConfig()
    win = list(window or [])

    # Insufficient window
    if len(win) < cfg.min_window_ticks:
        return DynamicBatchFrontierEstimate(
            workload_id=profile.workload_id,
            window_ticks=len(win),
            current_candidate=current_candidate,
            recommended_candidate=None,
            recommended_point=None,
            candidate_points=tuple(),
            risk_at_current=None,
            fallback_reason=(
                f"window_too_short_{len(win)}<{cfg.min_window_ticks}"),
            notes=("insufficient_window",),
        )

    # Build candidate grid if not provided.
    cand_list: list[BatchInferenceFrontierCandidate] = list(candidates or [])
    if not cand_list:
        for R in cfg.candidate_rhos:
            for slk in cfg.candidate_deadline_slack_seconds:
                for defer in cfg.candidate_deferral_seconds:
                    for C in cfg.candidate_batch_concurrency:
                        cand_list.append(BatchInferenceFrontierCandidate(
                            target_rho=R, deadline_slack_seconds=slk,
                            deferral_window_seconds=defer,
                            batch_concurrency=C,
                            source_policy=(f"rho{R}_slack{slk}s_"
                                           f"defer{defer}s_conc{C}")))

    risk_now = _risk_at_current(win, cfg)

    points: list[BatchInferenceFrontierPoint] = []
    for c in cand_list:
        if c.target_rho is None:
            points.append(BatchInferenceFrontierPoint(
                candidate=c,
                safety_status=BatchInferenceSafetyStatus.INSUFFICIENT_TELEMETRY,
                safety_vetoes=("candidate_target_rho_missing",),
                notes=("candidate target_rho is None",)))
            continue
        metrics = _evaluate_candidate(
            win, c, cfg=cfg, tick_seconds=tick_seconds)
        provisional = BatchInferenceFrontierPoint(
            candidate=c, safety_status=BatchInferenceSafetyStatus.SAFE,
            **metrics)
        status, vetoes = classify_batch_point_safety(
            provisional, safety, profile=profile,
            telemetry_confidence=profile.telemetry_confidence)
        points.append(BatchInferenceFrontierPoint(
            candidate=c, safety_status=status, safety_vetoes=vetoes,
            notes=("dynamic_v1",), **metrics))

    safe = [p for p in points if p.is_safe]
    if not safe:
        return DynamicBatchFrontierEstimate(
            workload_id=profile.workload_id,
            window_ticks=len(win),
            current_candidate=current_candidate,
            recommended_candidate=None,
            recommended_point=None,
            candidate_points=tuple(points),
            risk_at_current=risk_now,
            fallback_reason="no_safe_candidate_in_window",
            notes=("no_safe_point",))

    best = max(safe, key=lambda p: (p.predicted_goodput_per_dollar or 0.0))
    return DynamicBatchFrontierEstimate(
        workload_id=profile.workload_id,
        window_ticks=len(win),
        current_candidate=current_candidate,
        recommended_candidate=best.candidate,
        recommended_point=best,
        candidate_points=tuple(points),
        risk_at_current=risk_now,
        fallback_reason=None,
        notes=())


# ---------------------------------------------------------------------------
# Controller — turns an estimate into a DynamicBatchInferenceDecision.
# ---------------------------------------------------------------------------

def _within_deadband(a: BatchInferenceFrontierCandidate,
                     b: BatchInferenceFrontierCandidate,
                     cfg: DynamicBatchControllerConfig) -> bool:
    if a is None or b is None:
        return False
    if (a.target_rho is not None and b.target_rho is not None
            and abs(a.target_rho - b.target_rho) > cfg.rho_deadband):
        return False
    if (a.deadline_slack_seconds is not None
            and b.deadline_slack_seconds is not None
            and abs(a.deadline_slack_seconds - b.deadline_slack_seconds)
            > cfg.slack_deadband_seconds):
        return False
    if (a.deferral_window_seconds is not None
            and b.deferral_window_seconds is not None
            and abs(a.deferral_window_seconds - b.deferral_window_seconds)
            > cfg.deferral_deadband_seconds):
        return False
    return True


def choose_dynamic_batch_decision(
    estimate: DynamicBatchFrontierEstimate,
    *,
    current_candidate: Optional[BatchInferenceFrontierCandidate] = None,
    config: Optional[DynamicBatchControllerConfig] = None,
    previous_action: Optional[str] = None,
    confidence: str = "medium",
) -> DynamicBatchInferenceDecision:
    """Pick a recommendation-only :class:`DynamicBatchInferenceDecision`."""
    cfg = config or DynamicBatchControllerConfig()

    # 1. Fallback (insufficient window / no safe).
    if estimate.fallback_reason:
        return DynamicBatchInferenceDecision(
            workload_id=estimate.workload_id,
            current_candidate=current_candidate or estimate.current_candidate,
            recommended_candidate=None,
            recommended_point=None,
            action="INSUFFICIENT_TELEMETRY",
            reason=estimate.fallback_reason,
            risk_at_current=estimate.risk_at_current,
            confidence=confidence,
            notes=tuple(estimate.notes))

    cur = current_candidate or estimate.current_candidate
    best = estimate.recommended_point
    best_cand = estimate.recommended_candidate

    # 2. Risk-at-current → LOWER_BATCH_PRESSURE.
    if (estimate.risk_at_current is not None
            and estimate.risk_at_current >= cfg.lower_pressure_risk_threshold):
        return DynamicBatchInferenceDecision(
            workload_id=estimate.workload_id,
            current_candidate=cur,
            recommended_candidate=best_cand,
            recommended_point=best,
            action="LOWER_BATCH_PRESSURE",
            reason=(f"risk_at_current={estimate.risk_at_current:.3f} "
                    f">= {cfg.lower_pressure_risk_threshold}"),
            expected_goodput_per_dollar_delta=None,
            risk_at_current=estimate.risk_at_current,
            confidence=confidence,
            safety_vetoes=tuple())

    # 3. Deadband collapse → KEEP_CURRENT_BATCH_POLICY.
    cur_point = None
    if cur is not None:
        for p in estimate.candidate_points:
            cc = p.candidate
            if (cc.target_rho == cur.target_rho
                    and cc.deadline_slack_seconds
                    == cur.deadline_slack_seconds
                    and cc.deferral_window_seconds
                    == cur.deferral_window_seconds):
                cur_point = p
                break

    if (cur is not None and cur_point is not None and best_cand is not None
            and _within_deadband(best_cand, cur, cfg)):
        cur_kpi = cur_point.predicted_goodput_per_dollar or 0.0
        best_kpi = (best.predicted_goodput_per_dollar or 0.0
                    if best is not None else 0.0)
        if cur_kpi <= 0 or abs(best_kpi - cur_kpi) / cur_kpi <= cfg.kpi_deadband_pct:
            return DynamicBatchInferenceDecision(
                workload_id=estimate.workload_id,
                current_candidate=cur,
                recommended_candidate=cur,
                recommended_point=cur_point,
                action="KEEP_CURRENT_BATCH_POLICY",
                reason="deadband_collapsed_to_keep",
                expected_goodput_per_dollar_delta=best_kpi - cur_kpi,
                risk_at_current=estimate.risk_at_current,
                confidence=confidence,
                hysteresis_applied=True,
                notes=("deadband_collapsed_to_keep",))

    # 4. DEFER_BURST exposure — when the recommended deferral_window > 0
    # and exceeds the current deferral by at least the deadband.
    rec_defer = (best_cand.deferral_window_seconds or 0.0
                 if best_cand is not None else 0.0)
    cur_defer = (cur.deferral_window_seconds or 0.0
                 if cur is not None else 0.0)
    if (cfg.expose_defer_burst_action and rec_defer > 0.0
            and rec_defer - cur_defer > cfg.deferral_deadband_seconds):
        delta = None
        if cur_point is not None and best is not None:
            delta = ((best.predicted_goodput_per_dollar or 0.0)
                     - (cur_point.predicted_goodput_per_dollar or 0.0))
        return DynamicBatchInferenceDecision(
            workload_id=estimate.workload_id,
            current_candidate=cur,
            recommended_candidate=best_cand,
            recommended_point=best,
            action="DEFER_BURST",
            reason=(f"recommend deferral_window={rec_defer:.0f}s "
                    f"(current {cur_defer:.0f}s)"),
            expected_goodput_per_dollar_delta=delta,
            risk_at_current=estimate.risk_at_current,
            confidence=confidence)

    # 5. Standard RECOMMEND_BATCH_FRONTIER.
    delta = None
    if cur_point is not None and best is not None:
        delta = ((best.predicted_goodput_per_dollar or 0.0)
                 - (cur_point.predicted_goodput_per_dollar or 0.0))
    return DynamicBatchInferenceDecision(
        workload_id=estimate.workload_id,
        current_candidate=cur,
        recommended_candidate=best_cand,
        recommended_point=best,
        action="RECOMMEND_BATCH_FRONTIER",
        reason="highest_predicted_safe_goodput_per_dollar",
        expected_goodput_per_dollar_delta=delta,
        risk_at_current=estimate.risk_at_current,
        confidence=confidence)


def execute_dynamic_batch_inference_decision(
    decision: DynamicBatchInferenceDecision,
    *,
    allow_real_execution: bool = False,
    executor=None,
):
    """Stub real-execution shim. Shadow-only by default."""
    if not allow_real_execution:
        return {"mode": "shadow", "executed": False,
                "reason": "real_execution_disabled_by_default"}
    if executor is None:
        return {"mode": "real_disabled", "executed": False,
                "reason": "no_real_executor_supplied"}
    return executor(decision)  # pragma: no cover


# ---------------------------------------------------------------------------
# Telemetry adapter: ArrivalTick + per-tick eval result -> BatchArrivalTelemetryTick.
# ---------------------------------------------------------------------------

def telemetry_tick_from_arrival_tick(
    tick: ArrivalTick,
    *,
    timeout_pct: Optional[float] = None,
    queue_p99_ms: Optional[float] = None,
    latency_p99_ms: Optional[float] = None,
    observed_rho: Optional[float] = None,
    active_replicas: Optional[int] = None,
    telemetry_confidence: str = "medium",
    source: str = "azure_2024_replay",
) -> BatchArrivalTelemetryTick:
    """Build a :class:`BatchArrivalTelemetryTick` from an :class:`ArrivalTick`."""
    return BatchArrivalTelemetryTick(
        timestamp_s=tick.start_s,
        arrival_rate_rps=tick.arrival_rate_rps,
        prompt_tokens_mean=tick.prompt_tokens_mean,
        output_tokens_mean=tick.output_tokens_mean,
        total_output_tokens=tick.total_output_tokens,
        request_count=tick.request_count,
        active_replicas=active_replicas,
        observed_rho=observed_rho,
        queue_p99_ms=queue_p99_ms,
        timeout_pct=timeout_pct,
        latency_p99_ms=latency_p99_ms,
        deadline_miss_pct=None,
        deferred_arrivals_pending=None,
        telemetry_confidence=telemetry_confidence,
        source=source,
    )
