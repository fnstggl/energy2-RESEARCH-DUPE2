"""Dynamic Serving Frontier — rolling calibration + multi-pass replay.

This is the closed-loop calibration harness for the Dynamic Safe
Frontier Estimator (v1). It replays a trace, lets the estimator emit
one prediction per decision window, evaluates the realized outcome at
the next window, updates confidence, and aggregates calibration metrics.

Hard rules (asserted by tests):

- **No future leakage.** The estimator never reads telemetry beyond its
  current rolling window. The oracle/baseline series is computed up
  front but is **never** visible to the estimator at decision time —
  only the calibration loop consumes it.
- **No production-savings claims.** Simulator / shadow-mode evidence
  only.
- **Multi-pass stopping** is bounded. The harness will not loop
  indefinitely just to chase the oracle target.
- **Allowed updates only.** Between passes, the harness may adjust the
  conservative margin, deadband, hysteresis multiplier, and the unsafe
  risk threshold within bounded ranges. It may **not** disable safety
  vetoes, change oracle labels, or hide unsafe points.
- **Honest reporting.** If oracle-alpha capture does not reach the
  target, the harness reports why (insufficient signal, safety floor
  hit, or simply diminishing returns).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence

from .dynamic_confidence import (
    ConfidenceUpdateConfig,
    apply_confidence_update,
)
from .dynamic_controller import DynamicControllerConfig, choose_dynamic_rho
from .dynamic_estimator import DynamicEstimatorConfig, estimate_dynamic_frontier
from .dynamic_evaluation import (
    DynamicFrontierCalibrationRecord,
    DynamicFrontierObservedOutcome,
    DynamicFrontierPrediction,
    OracleSeriesPoint,
    compute_calibration_record,
    compute_frontier_calibration_summary,
)
from .dynamic_models import (
    DynamicFrontierEstimate,
    ServingTelemetryTick,
)
from .models import WorkloadFrontierProfile
from .risk import RiskConfig
from .safety import SafetyConfig

# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

@dataclass
class CalibrationReplayConfig:
    """Settings for one calibration-replay pass.

    Pre-registered defaults — none tuned per trace. Multi-pass tweaks
    are clamped by :class:`MultiPassCalibrationConfig`.
    """

    # Rolling-window sizes in *ticks*.
    window_ticks: int = 60
    # Decision interval (in ticks). Set to 1 ⇒ decision every tick.
    decision_interval_ticks: int = 1
    # Pre-registered candidate rhos.
    candidate_rhos: tuple = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
    # Bootstrap rho for the first decision (before history exists).
    bootstrap_rho: float = 0.65
    # Pre-registered safety thresholds (also used for ``was_safe``).
    safety_timeout_pct: float = 10.0
    safety_queue_p99_ms: float = 2000.0
    # Bootstrap confidence (per-workload, before any updates).
    initial_confidence: float = 0.5
    # Source label.
    source: str = "dynamic_frontier_estimator_v1"

    def __post_init__(self) -> None:
        if self.window_ticks <= 0:
            raise ValueError(f"window_ticks must be > 0; "
                             f"got {self.window_ticks}")
        if self.decision_interval_ticks <= 0:
            raise ValueError(
                f"decision_interval_ticks must be > 0; "
                f"got {self.decision_interval_ticks}")
        if not (0.0 <= self.initial_confidence <= 1.0):
            raise ValueError(
                f"initial_confidence must be in [0,1]; "
                f"got {self.initial_confidence}")


@dataclass
class MultiPassCalibrationConfig:
    """Settings for a bounded multi-pass calibration run.

    Tunable parameters are clamped within bounded ranges so the harness
    can never weaken safety to chase the oracle. The target is honest:
    we report whether we reached it, but never force it.
    """

    passes: int = 3
    target_oracle_alpha_capture: float = 0.95
    min_safety_rate: float = 0.99
    max_false_safe_rate: float = 0.01
    # Bounded parameter ranges (start, min, max).
    conservative_margin_enabled_init: bool = True
    unsafe_risk_threshold_init: float = 0.75
    unsafe_risk_threshold_min: float = 0.55
    unsafe_risk_threshold_max: float = 0.85
    deadband_rho_init: float = 0.05
    deadband_rho_min: float = 0.02
    deadband_rho_max: float = 0.10
    hysteresis_multiplier_init: float = 2.0
    hysteresis_multiplier_min: float = 1.0
    hysteresis_multiplier_max: float = 3.0

    def __post_init__(self) -> None:
        if self.passes <= 0:
            raise ValueError(f"passes must be > 0; got {self.passes}")
        if not (0.0 <= self.target_oracle_alpha_capture <= 1.0):
            raise ValueError(
                f"target_oracle_alpha_capture must be in [0,1]; "
                f"got {self.target_oracle_alpha_capture}")
        if not (0.0 <= self.max_false_safe_rate <= 1.0):
            raise ValueError(
                f"max_false_safe_rate must be in [0,1]; "
                f"got {self.max_false_safe_rate}")
        if not (0.0 <= self.min_safety_rate <= 1.0):
            raise ValueError(
                f"min_safety_rate must be in [0,1]; "
                f"got {self.min_safety_rate}")


# ---------------------------------------------------------------------------
# Replay outcome containers.
# ---------------------------------------------------------------------------

@dataclass
class CalibrationPassResult:
    """One pass of the calibration replay.

    ``records`` is the per-decision ledger (predictions + outcomes +
    calibration metrics + confidence updates). ``summary`` is the
    aggregated metrics dict.
    """

    pass_index: int
    records: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    estimator_config_snapshot: dict = field(default_factory=dict)
    controller_config_snapshot: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


@dataclass
class CalibrationReplayResult:
    """Full multi-pass calibration replay output."""

    passes: list = field(default_factory=list)
    stopped_reason: str = ""
    target_oracle_alpha_capture: float = 0.95
    reached_target: bool = False
    safety_floor_held: bool = True
    initial_oracle_alpha_capture: Optional[float] = None
    final_oracle_alpha_capture: Optional[float] = None
    overfit_risk_notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passes": [
                {
                    "pass_index": p.pass_index,
                    "summary": p.summary,
                    "estimator_config_snapshot": p.estimator_config_snapshot,
                    "controller_config_snapshot": p.controller_config_snapshot,
                    "n_records": len(p.records),
                    "notes": list(p.notes),
                }
                for p in self.passes
            ],
            "stopped_reason": self.stopped_reason,
            "target_oracle_alpha_capture": self.target_oracle_alpha_capture,
            "reached_target": self.reached_target,
            "safety_floor_held": self.safety_floor_held,
            "initial_oracle_alpha_capture": self.initial_oracle_alpha_capture,
            "final_oracle_alpha_capture": self.final_oracle_alpha_capture,
            "overfit_risk_notes": list(self.overfit_risk_notes),
        }


# ---------------------------------------------------------------------------
# Replay driver.
# ---------------------------------------------------------------------------

# The replay needs three callables from the caller:
#
#   tick_evaluator(target_rho, tick_index) -> dict — applies ``target_rho``
#       to one tick and returns realized metrics. The dict MUST carry
#       at least: rho, timeout_pct, queue_p99_ms, latency_p99_ms,
#       gpu_hours, goodput_per_dollar.
#
#   telemetry_builder(arrival_tick, eval_result) -> ServingTelemetryTick
#       — converts the realized eval into the telemetry tick the
#       estimator will see at the *next* decision step.
#
#   oracle_series_builder(...) -> Sequence[OracleSeriesPoint] — computes
#       the analysis-only oracle and baseline goodput/$ for each window.
#       MUST be passed in pre-computed; the replay loop never peeks
#       ahead.

EvalFn = Callable[[float, int], dict]
TelemetryFn = Callable[[object, dict], ServingTelemetryTick]


def run_dynamic_frontier_calibration_replay(
    *,
    workload_profile: WorkloadFrontierProfile,
    ticks: Sequence,
    eval_fn: EvalFn,
    telemetry_fn: TelemetryFn,
    oracle_series: Optional[Sequence[OracleSeriesPoint]] = None,
    config: Optional[CalibrationReplayConfig] = None,
    estimator_cfg: Optional[DynamicEstimatorConfig] = None,
    controller_cfg: Optional[DynamicControllerConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
    confidence_cfg: Optional[ConfidenceUpdateConfig] = None,
    initial_confidence: Optional[float] = None,
    pass_index: int = 0,
) -> CalibrationPassResult:
    """One offline streaming replay pass.

    ``ticks`` is the trace (any sequence whose elements ``eval_fn`` and
    ``telemetry_fn`` understand). The estimator sees only the rolling
    window of *past* realized telemetry ticks — never the future. The
    oracle/baseline series is consumed by the calibration record builder
    AFTER the prediction is made.
    """
    cfg = config or CalibrationReplayConfig()
    est_cfg = estimator_cfg or DynamicEstimatorConfig()
    ctrl_cfg = controller_cfg or DynamicControllerConfig()
    safety = safety_cfg or SafetyConfig(
        max_timeout_pct=cfg.safety_timeout_pct,
        max_queue_p99_ms=cfg.safety_queue_p99_ms)
    rcfg = risk_cfg or RiskConfig(
        max_timeout_pct=cfg.safety_timeout_pct,
        max_queue_p99_ms=cfg.safety_queue_p99_ms,
        min_telemetry_confidence="low")
    conf_cfg = confidence_cfg or ConfidenceUpdateConfig()

    history: list[ServingTelemetryTick] = []
    current_rho = cfg.bootstrap_rho
    prev_action: Optional[str] = None
    cur_confidence = (initial_confidence if initial_confidence is not None
                      else cfg.initial_confidence)

    predictions: list[DynamicFrontierPrediction] = []
    outcomes: list[DynamicFrontierObservedOutcome] = []
    records: list[DynamicFrontierCalibrationRecord] = []
    rho_history: list[float] = []
    action_history: list[str] = []
    decision_estimates: list[DynamicFrontierEstimate] = []

    for i, tick in enumerate(ticks):
        ts = float(getattr(tick, "start_s", i) or i)
        # 1 — Decide rho for this tick using only past history (no leakage).
        prediction: Optional[DynamicFrontierPrediction] = None
        if (len(history) >= est_cfg.min_window_ticks
                and (i % cfg.decision_interval_ticks == 0)):
            window = history[-cfg.window_ticks:]
            est = estimate_dynamic_frontier(
                workload_profile=workload_profile,
                telemetry_window=window,
                current_rho=current_rho,
                candidate_rhos=cfg.candidate_rhos,
                config=est_cfg,
                safety_config=safety,
                risk_config=rcfg)
            dec = choose_dynamic_rho(est, current_rho=current_rho,
                                     config=ctrl_cfg,
                                     previous_action=prev_action)
            decision_estimates.append(est)
            # Pull the candidate-at-recommended for predicted metrics.
            rec_cand = None
            if est.recommended_rho is not None:
                rec_cand = next(
                    (c for c in est.candidate_points
                     if abs(c.rho_target - est.recommended_rho) < 1e-9),
                    None)
            pred_goodput = (rec_cand.predicted_goodput_per_dollar
                            if rec_cand is not None else None)
            pred_timeout = (rec_cand.predicted_timeout_pct
                            if rec_cand is not None else None)
            pred_queue = (rec_cand.predicted_queue_p99_ms
                          if rec_cand is not None else None)
            pred_latency = (rec_cand.predicted_latency_p99_ms
                            if rec_cand is not None else None)
            pred_gpuh = (rec_cand.predicted_gpu_hours
                         if rec_cand is not None else None)
            sla_risk = (rec_cand.predicted_sla_risk_probability
                        if rec_cand is not None else None)
            queue_risk = (rec_cand.predicted_queue_blowup_probability
                          if rec_cand is not None else None)
            cur_cand = next(
                (c for c in est.candidate_points
                 if current_rho is not None
                 and abs(c.rho_target - current_rho) < 0.05),
                None)
            pred_goodput_delta = None
            if (pred_goodput is not None and cur_cand is not None
                    and cur_cand.predicted_goodput_per_dollar is not None):
                pred_goodput_delta = (pred_goodput
                                      - cur_cand.predicted_goodput_per_dollar)
            prediction = DynamicFrontierPrediction(
                timestamp_s=ts,
                workload_id=workload_profile.workload_id,
                current_rho=current_rho,
                recommended_rho=dec.recommended_rho,
                action=dec.action,
                predicted_goodput_per_dollar=pred_goodput,
                predicted_goodput_delta=pred_goodput_delta,
                predicted_timeout_pct=pred_timeout,
                predicted_queue_p99_ms=pred_queue,
                predicted_latency_p99_ms=pred_latency,
                predicted_sla_risk_probability=sla_risk,
                predicted_queue_blowup_probability=queue_risk,
                predicted_gpu_hours=pred_gpuh,
                confidence=cur_confidence,
                risk_reason_codes=tuple(rec_cand.risk_reason_codes
                                        if rec_cand is not None else ()),
                source=cfg.source)
            new_rho = (dec.recommended_rho
                       if dec.recommended_rho is not None else current_rho)
            current_rho = new_rho
            prev_action = dec.action

        # 2 — Apply current rho to the tick (realized observation).
        ev = eval_fn(current_rho, i)
        rho_history.append(current_rho)
        action_history.append(prediction.action if prediction else "BOOTSTRAP")

        # 3 — Build the observed outcome for this decision window.
        was_safe = None
        if (ev.get("timeout_pct") is not None
                or ev.get("queue_p99_ms") is not None):
            tpct = ev.get("timeout_pct")
            q99 = ev.get("queue_p99_ms")
            if ((tpct is not None and tpct > cfg.safety_timeout_pct)
                    or (q99 is not None and q99 > cfg.safety_queue_p99_ms)):
                was_safe = False
            else:
                was_safe = True
        outcome = DynamicFrontierObservedOutcome(
            timestamp_s=ts,
            workload_id=workload_profile.workload_id,
            applied_rho=ev.get("rho"),
            observed_goodput_per_dollar=ev.get("goodput_per_dollar"),
            observed_timeout_pct=ev.get("timeout_pct"),
            observed_queue_p99_ms=ev.get("queue_p99_ms"),
            observed_latency_p99_ms=ev.get("latency_p99_ms"),
            observed_sla_violation_pct=ev.get("sla_violation_pct"),
            observed_gpu_hours=ev.get("gpu_hours"),
            observed_churn=ev.get("churn"),
            was_safe=was_safe,
            source=cfg.source)

        # 4 — If we emitted a prediction this step, build a calibration
        # record and update confidence.
        if prediction is not None:
            oracle = (oracle_series[i] if oracle_series is not None
                      and i < len(oracle_series) else None)
            rec = compute_calibration_record(
                prediction, outcome, oracle=oracle,
                safety_timeout_pct=cfg.safety_timeout_pct,
                safety_queue_p99_ms=cfg.safety_queue_p99_ms)
            rec = apply_confidence_update(rec, config=conf_cfg)
            records.append(rec)
            predictions.append(prediction)
            outcomes.append(outcome)
            cur_confidence = rec.confidence_after

        # 5 — Feed the telemetry tick into history for the NEXT decision.
        history.append(telemetry_fn(tick, ev))

    summary = compute_frontier_calibration_summary(records)
    return CalibrationPassResult(
        pass_index=pass_index,
        records=records,
        summary=summary,
        estimator_config_snapshot={
            "min_window_ticks": est_cfg.min_window_ticks,
            "unsafe_risk_threshold": est_cfg.unsafe_risk_threshold,
            "conservative_margin_enabled":
                est_cfg.conservative_margin_enabled,
            "local_delta": est_cfg.local_delta,
            "n_local_candidates": est_cfg.n_local_candidates,
        },
        controller_config_snapshot={
            "deadband_rho": ctrl_cfg.deadband_rho,
            "deadband_kpi_pct": ctrl_cfg.deadband_kpi_pct,
            "lower_rho_risk_threshold": ctrl_cfg.lower_rho_risk_threshold,
            "hysteresis_multiplier": ctrl_cfg.hysteresis_multiplier,
            "churn_suppresses_raise": ctrl_cfg.churn_suppresses_raise,
        },
    )


# ---------------------------------------------------------------------------
# Multi-pass driver.
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _propose_safe_update(
    *,
    pass_summary: dict,
    estimator_cfg: DynamicEstimatorConfig,
    controller_cfg: DynamicControllerConfig,
    multi_cfg: MultiPassCalibrationConfig,
) -> tuple[DynamicEstimatorConfig, DynamicControllerConfig, list[str]]:
    """Propose a single bounded parameter update for the next pass.

    Rules:

    - Never weaken safety. If ``false_safe_rate`` >
      ``max_false_safe_rate``, tighten the unsafe risk threshold and
      bump the deadband — do NOT loosen anything.
    - If the run is safe but conservative (capture below target AND
      ``false_unsafe_rate`` or ``conservative_miss_rate`` is high), allow
      a small *safe* relaxation: raise the unsafe risk threshold within
      its bounded range. Never above ``unsafe_risk_threshold_max``.
    - Always emit a note explaining what changed and why.
    """
    notes: list[str] = []
    new_est = replace(estimator_cfg)
    new_ctrl = replace(controller_cfg)

    fs_rate = pass_summary.get("false_safe_rate") or 0.0
    cm_rate = pass_summary.get("conservative_miss_rate") or 0.0
    fu_rate = pass_summary.get("false_unsafe_rate") or 0.0
    capture = pass_summary.get("oracle_alpha_capture_pct_overall")
    capture_mean = pass_summary.get("oracle_alpha_capture_pct_mean")

    # Safety floor first.
    if fs_rate > multi_cfg.max_false_safe_rate:
        # Tighten: lower unsafe_risk_threshold and widen deadband.
        new_est.unsafe_risk_threshold = _clamp(
            estimator_cfg.unsafe_risk_threshold - 0.05,
            multi_cfg.unsafe_risk_threshold_min,
            multi_cfg.unsafe_risk_threshold_max)
        new_ctrl.deadband_rho = _clamp(
            controller_cfg.deadband_rho + 0.01,
            multi_cfg.deadband_rho_min,
            multi_cfg.deadband_rho_max)
        notes.append(
            f"tighten:unsafe_risk_threshold->{new_est.unsafe_risk_threshold} "
            f"(false_safe_rate {fs_rate:.4f} > "
            f"{multi_cfg.max_false_safe_rate})")
        return new_est, new_ctrl, notes

    # Safe + conservative — try a *small* bounded relaxation.
    cap = capture if capture is not None else capture_mean
    if (cap is not None and cap < multi_cfg.target_oracle_alpha_capture
            and (fu_rate > 0.0 or cm_rate > 0.0)):
        new_est.unsafe_risk_threshold = _clamp(
            estimator_cfg.unsafe_risk_threshold + 0.02,
            multi_cfg.unsafe_risk_threshold_min,
            multi_cfg.unsafe_risk_threshold_max)
        new_ctrl.deadband_rho = _clamp(
            controller_cfg.deadband_rho - 0.005,
            multi_cfg.deadband_rho_min,
            multi_cfg.deadband_rho_max)
        new_ctrl.hysteresis_multiplier = _clamp(
            controller_cfg.hysteresis_multiplier - 0.1,
            multi_cfg.hysteresis_multiplier_min,
            multi_cfg.hysteresis_multiplier_max)
        notes.append(
            f"relax_within_bounds:unsafe_risk_threshold->"
            f"{new_est.unsafe_risk_threshold:.3f}, deadband_rho->"
            f"{new_ctrl.deadband_rho:.4f}, hyst->"
            f"{new_ctrl.hysteresis_multiplier:.2f} (capture={cap:.4f} < "
            f"{multi_cfg.target_oracle_alpha_capture})")
        return new_est, new_ctrl, notes

    notes.append("no_change_proposed")
    return new_est, new_ctrl, notes


def run_multi_pass_calibration(
    *,
    workload_profile: WorkloadFrontierProfile,
    ticks: Sequence,
    eval_fn: EvalFn,
    telemetry_fn: TelemetryFn,
    oracle_series: Optional[Sequence[OracleSeriesPoint]] = None,
    config: Optional[CalibrationReplayConfig] = None,
    multi_pass_config: Optional[MultiPassCalibrationConfig] = None,
    estimator_cfg: Optional[DynamicEstimatorConfig] = None,
    controller_cfg: Optional[DynamicControllerConfig] = None,
    safety_cfg: Optional[SafetyConfig] = None,
    risk_cfg: Optional[RiskConfig] = None,
    confidence_cfg: Optional[ConfidenceUpdateConfig] = None,
) -> CalibrationReplayResult:
    """Run up to ``passes`` calibration-replay passes with bounded
    parameter updates between them.

    Stopping criteria (any one triggers stop):

    1. ``passes`` reached.
    2. ``oracle_alpha_capture_pct_overall`` >= target AND safety floor held.
    3. Safety floor broken in the latest pass and the proposed update
       would not improve it (defensive stop — do NOT loop unsafe).
    """
    multi_cfg = multi_pass_config or MultiPassCalibrationConfig()
    est_cfg = estimator_cfg or DynamicEstimatorConfig(
        unsafe_risk_threshold=multi_cfg.unsafe_risk_threshold_init,
        conservative_margin_enabled=multi_cfg.conservative_margin_enabled_init)
    ctrl_cfg = controller_cfg or DynamicControllerConfig(
        deadband_rho=multi_cfg.deadband_rho_init,
        hysteresis_multiplier=multi_cfg.hysteresis_multiplier_init)

    result = CalibrationReplayResult(
        target_oracle_alpha_capture=multi_cfg.target_oracle_alpha_capture)
    for pi in range(multi_cfg.passes):
        pr = run_dynamic_frontier_calibration_replay(
            workload_profile=workload_profile,
            ticks=ticks,
            eval_fn=eval_fn,
            telemetry_fn=telemetry_fn,
            oracle_series=oracle_series,
            config=config,
            estimator_cfg=est_cfg,
            controller_cfg=ctrl_cfg,
            safety_cfg=safety_cfg,
            risk_cfg=risk_cfg,
            confidence_cfg=confidence_cfg,
            pass_index=pi)
        result.passes.append(pr)
        cap = pr.summary.get("oracle_alpha_capture_pct_overall")
        if pi == 0:
            result.initial_oracle_alpha_capture = cap
        result.final_oracle_alpha_capture = cap

        fs_rate = pr.summary.get("false_safe_rate") or 0.0
        if fs_rate > multi_cfg.max_false_safe_rate:
            result.safety_floor_held = False

        # Stop conditions.
        if (cap is not None
                and cap >= multi_cfg.target_oracle_alpha_capture
                and fs_rate <= multi_cfg.max_false_safe_rate):
            result.reached_target = True
            result.stopped_reason = (
                f"target_reached:capture={cap:.4f} >= "
                f"{multi_cfg.target_oracle_alpha_capture}")
            break

        # Last pass — no proposal needed.
        if pi == multi_cfg.passes - 1:
            result.stopped_reason = (
                f"passes_exhausted:{multi_cfg.passes}, final_capture="
                f"{cap}")
            break

        # Propose bounded update for next pass.
        est_cfg, ctrl_cfg, notes = _propose_safe_update(
            pass_summary=pr.summary,
            estimator_cfg=est_cfg,
            controller_cfg=ctrl_cfg,
            multi_cfg=multi_cfg)
        pr.notes.extend(notes)

        # Defensive stop: if the proposal made no change AND we are
        # below target, no point in looping.
        if "no_change_proposed" in notes:
            result.stopped_reason = (
                "no_useful_update_proposed_below_target:final_capture="
                f"{cap}")
            break

    # Overfit-risk notes — flag whenever the test-window is the only
    # window seen across all passes.
    result.overfit_risk_notes.append(
        "calibration_window_is_replay_window:any tuning that helps here "
        "may not generalize; pilot telemetry remains required before any "
        "production claim."
    )
    if (result.reached_target
            and result.initial_oracle_alpha_capture is not None
            and result.final_oracle_alpha_capture is not None
            and (result.final_oracle_alpha_capture
                 - result.initial_oracle_alpha_capture) > 0.20):
        result.overfit_risk_notes.append(
            "large_capture_lift_in_calibration:"
            f"{result.initial_oracle_alpha_capture:.4f} -> "
            f"{result.final_oracle_alpha_capture:.4f}; treat as in-sample "
            "fit, not out-of-sample evidence."
        )
    return result
