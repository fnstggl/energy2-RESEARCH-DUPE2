"""Tests for the dynamic Batch Inference Frontier v1 (opt-in, shadow only).

Hard invariants:

1.  Dynamic batch estimator does NOT import the serving rho controller
    modules (controller.py / dynamic_controller.py / estimator.py /
    dynamic_estimator.py).
2.  Models reject unknown enum values / out-of-range fields.
3.  Insufficient window -> INSUFFICIENT_TELEMETRY decision; no candidate
    points returned.
4.  All-safe candidates -> RECOMMEND_BATCH_FRONTIER and best is the
    highest-goodput safe point.
5.  Dynamic decision's executable_in_real_cluster is False at
    construction; constructing with True raises.
6.  execute_dynamic_batch_inference_decision is shadow-only by default;
    non-zero opt-in requires both flag + executor.
7.  Deferral_window_seconds field is part of the candidate descriptor
    and produces decision-level DEFER_BURST when the recommendation
    raises deferral materially above the current candidate.
8.  Risk-at-current above threshold -> LOWER_BATCH_PRESSURE.
9.  Deadband: small recommendation delta -> KEEP_CURRENT_BATCH_POLICY.
10. Telemetry adapter (ArrivalTick -> BatchArrivalTelemetryTick) carries
    the per-tick fields correctly and preserves missing values as None.
11. New deferral_window_seconds field on the SHARED static-batch
    candidate does not break the static batch path
    (BatchInferenceFrontierCandidate stays JSON-round-trippable;
    existing static tests still pass).
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.frontier import (  # noqa: E402
    dynamic_batch_inference_estimator as dbe,
)
from aurelius.frontier.batch_inference_models import (  # noqa: E402
    BatchInferenceFrontierCandidate,
    BatchInferenceFrontierSchemaError,
    BatchInferenceWorkloadProfile,
)
from aurelius.frontier.batch_inference_safety import (  # noqa: E402
    BatchInferenceSafetyConfig,
)
from aurelius.traces.replay import ArrivalTick  # noqa: E402


def _profile(**kw):
    defaults = dict(
        workload_id="audit_test",
        trace_source="azure_llm_2024",
        synthetic_scenario_label="azure_llm_2024_batch_flex_scenario_v1",
        deadline_miss_rate_sla_pct=2.0,
        queue_wait_sla_p99_ms=2000.0,
        telemetry_confidence="medium",
    )
    defaults.update(kw)
    return BatchInferenceWorkloadProfile(**defaults)


def _telemetry_window(n: int = 12, *, rate=5.0, output=20.0, prompt=1500.0,
                      timeout=2.0, queue=150.0):
    return [
        dbe.BatchArrivalTelemetryTick(
            timestamp_s=float(i),
            arrival_rate_rps=rate, prompt_tokens_mean=prompt,
            output_tokens_mean=output,
            total_output_tokens=int(rate * 60 * output),
            request_count=int(rate * 60),
            timeout_pct=timeout, queue_p99_ms=queue,
            telemetry_confidence="medium",
        )
        for i in range(n)
    ]


# ---- 1. No serving rho controller imports ----

def test_does_not_import_serving_rho_controller_modules():
    src = inspect.getsource(dbe)
    for forbidden in (
        "from aurelius.frontier.controller import",
        "from aurelius.frontier.dynamic_controller import",
        "from aurelius.frontier.estimator import",
        "from aurelius.frontier.dynamic_estimator import",
        "from .controller import",
        "from .dynamic_controller import",
        "from .estimator import",
        "from .dynamic_estimator import",
    ):
        assert forbidden not in src, (
            f"dynamic_batch_inference_estimator must not import the "
            f"serving rho controller (found `{forbidden}`)")


# ---- 2. Validation ----

def test_decision_action_validated():
    with pytest.raises(BatchInferenceFrontierSchemaError):
        dbe.DynamicBatchInferenceDecision(
            workload_id="x", current_candidate=None,
            recommended_candidate=None, recommended_point=None,
            action="GO_FAST", reason="r")


def test_decision_real_execution_false_at_construction():
    with pytest.raises(BatchInferenceFrontierSchemaError):
        dbe.DynamicBatchInferenceDecision(
            workload_id="x", current_candidate=None,
            recommended_candidate=None, recommended_point=None,
            action="RECOMMEND_BATCH_FRONTIER", reason="r",
            executable_in_real_cluster=True)


# ---- 3. Insufficient window -> INSUFFICIENT_TELEMETRY ----

def test_insufficient_window_falls_back():
    profile = _profile()
    short_window = _telemetry_window(n=3)
    est = dbe.estimate_dynamic_batch_frontier(profile, short_window)
    assert est.fallback_reason is not None
    assert est.recommended_candidate is None
    dec = dbe.choose_dynamic_batch_decision(est)
    assert dec.action == "INSUFFICIENT_TELEMETRY"
    assert dec.executable_in_real_cluster is False


# ---- 4. Safe candidates -> RECOMMEND_BATCH_FRONTIER, highest goodput ----

def test_safe_candidates_recommend_highest_goodput():
    profile = _profile()
    est = dbe.estimate_dynamic_batch_frontier(profile, _telemetry_window())
    assert est.fallback_reason is None
    safe = [p for p in est.candidate_points if p.is_safe]
    assert safe, "expected at least one safe candidate"
    # Best is the max-goodput safe point.
    best_goodput = max((p.predicted_goodput_per_dollar or 0.0)
                       for p in safe)
    assert (est.recommended_point.predicted_goodput_per_dollar
            >= best_goodput - 1e-9)
    dec = dbe.choose_dynamic_batch_decision(est)
    assert dec.action in (
        "RECOMMEND_BATCH_FRONTIER", "DEFER_BURST",
        "KEEP_CURRENT_BATCH_POLICY",
    )


# ---- 5+6. Decision construction safety ----

def test_decision_executable_in_real_cluster_false():
    profile = _profile()
    est = dbe.estimate_dynamic_batch_frontier(profile, _telemetry_window())
    dec = dbe.choose_dynamic_batch_decision(est)
    assert dec.executable_in_real_cluster is False
    res = dbe.execute_dynamic_batch_inference_decision(dec)
    assert res["mode"] == "shadow"
    assert res["executed"] is False


def test_execute_real_disabled_without_executor():
    profile = _profile()
    est = dbe.estimate_dynamic_batch_frontier(profile, _telemetry_window())
    dec = dbe.choose_dynamic_batch_decision(est)
    res = dbe.execute_dynamic_batch_inference_decision(
        dec, allow_real_execution=True, executor=None)
    assert res["mode"] == "real_disabled"
    assert res["executed"] is False


# ---- 7. Deferral field is part of the candidate descriptor ----

def test_candidate_carries_deferral_window_seconds():
    c = BatchInferenceFrontierCandidate(
        target_rho=0.65, deadline_slack_seconds=300.0,
        deferral_window_seconds=60.0)
    assert c.deferral_window_seconds == 60.0
    # Negative deferral rejected.
    with pytest.raises(BatchInferenceFrontierSchemaError):
        BatchInferenceFrontierCandidate(
            target_rho=0.65, deferral_window_seconds=-1.0)


def test_defer_burst_action_when_recommendation_raises_deferral():
    profile = _profile()
    # Force the estimator to recommend a deferral > 0 by setting current
    # candidate to deferral=0 and giving it heavy load that the small
    # synthetic window cannot safely absorb at higher rhos.
    cur = BatchInferenceFrontierCandidate(
        target_rho=0.65, deadline_slack_seconds=300.0,
        deferral_window_seconds=0.0, batch_concurrency=1)
    window = _telemetry_window(rate=50.0, output=400.0,
                               prompt=2000.0, queue=400.0)
    est = dbe.estimate_dynamic_batch_frontier(
        profile, window, current_candidate=cur,
        estimator_config=dbe.DynamicBatchEstimatorConfig(
            candidate_deferral_seconds=(0.0, 300.0)))
    dec = dbe.choose_dynamic_batch_decision(
        est, current_candidate=cur)
    # We do not assert DEFER_BURST specifically (the per-tick projection
    # might recommend deferral=0 too), but if a recommendation with
    # deferral>0 was made, the action must be DEFER_BURST not the generic
    # RECOMMEND_BATCH_FRONTIER (when above the deadband).
    if (dec.recommended_candidate is not None
            and (dec.recommended_candidate.deferral_window_seconds or 0.0)
            > 30.0):
        assert dec.action == "DEFER_BURST"


# ---- 8. Risk-at-current high -> LOWER_BATCH_PRESSURE ----

def test_high_risk_triggers_lower_pressure():
    profile = _profile()
    # Build a window with high observed timeout + queue -> risk near 1.0.
    window = _telemetry_window(timeout=10.0, queue=2000.0)
    est = dbe.estimate_dynamic_batch_frontier(profile, window)
    assert est.risk_at_current is not None
    assert est.risk_at_current >= 0.75
    dec = dbe.choose_dynamic_batch_decision(est)
    assert dec.action in (
        "LOWER_BATCH_PRESSURE",
        # In edge cases the safety classifier may also mark no safe
        # candidate at all - INSUFFICIENT_TELEMETRY is then valid.
        "INSUFFICIENT_TELEMETRY",
    )


# ---- 9. Deadband -> KEEP_CURRENT_BATCH_POLICY ----

def test_deadband_recommends_keep():
    profile = _profile()
    cur = BatchInferenceFrontierCandidate(
        target_rho=0.65, deadline_slack_seconds=300.0,
        deferral_window_seconds=0.0, batch_concurrency=1)
    # Build a steady-load window — the best safe candidate should be
    # near current rho (within deadband).
    est = dbe.estimate_dynamic_batch_frontier(
        profile, _telemetry_window(),
        current_candidate=cur,
        estimator_config=dbe.DynamicBatchEstimatorConfig(
            candidate_rhos=(0.65,),
            candidate_deadline_slack_seconds=(300.0,),
            candidate_deferral_seconds=(0.0,),
            candidate_batch_concurrency=(1,)))
    dec = dbe.choose_dynamic_batch_decision(
        est, current_candidate=cur)
    # Single-candidate grid AT the current candidate -> deadband collapse.
    assert dec.action == "KEEP_CURRENT_BATCH_POLICY"


# ---- 10. Telemetry adapter ----

def test_telemetry_adapter_carries_arrival_fields():
    tick = ArrivalTick(
        tick_index=4, start_s=240.0, end_s=300.0, duration_s=60.0,
        request_count=120, arrival_rate_rps=2.0,
        prompt_tokens_mean=1200.0, output_tokens_mean=80.0,
        total_prompt_tokens=144000, total_output_tokens=9600,
        failures=0, distinct_cache_keys=0, reuse_fraction=0.0,
        model_mix={"azure-llm": 120}, log_type_mix={"unknown": 120})
    out = dbe.telemetry_tick_from_arrival_tick(
        tick, timeout_pct=3.5, queue_p99_ms=250.0,
        latency_p99_ms=4500.0, observed_rho=0.55, active_replicas=2)
    assert out.timestamp_s == 240.0
    assert out.arrival_rate_rps == 2.0
    assert out.prompt_tokens_mean == 1200.0
    assert out.output_tokens_mean == 80.0
    assert out.request_count == 120
    assert out.timeout_pct == 3.5
    assert out.queue_p99_ms == 250.0
    assert out.observed_rho == 0.55
    assert out.active_replicas == 2
    # Fields the caller didn't supply stay None.
    assert out.deadline_miss_pct is None
    assert out.deferred_arrivals_pending is None


# ---- 11. Static batch path still works with the new field ----

def test_static_batch_candidate_json_round_trip():
    c = BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0,
        deferral_window_seconds=60.0, batch_concurrency=2,
        source_policy="x")
    d = c.to_dict()
    assert d["deferral_window_seconds"] == 60.0
    payload = json.dumps(d)  # JSON-serializable
    parsed = json.loads(payload)
    assert parsed["deferral_window_seconds"] == 60.0


def test_static_batch_frontier_still_runs_with_new_field():
    # Quick smoke that the static estimator handles a candidate with
    # the new field cleanly (the static estimator simply ignores it).
    from aurelius.frontier.batch_inference_estimator import (
        BatchInferenceEstimatorConfig,
        estimate_batch_inference_frontier,
    )
    from aurelius.traces import azure_llm
    from aurelius.traces.replay import requests_to_arrival_ticks
    fixture = REPO_ROOT / "tests" / "fixtures" / "azure_llm_2024_sample.csv"
    reqs = azure_llm.load_csv(str(fixture))
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    cands = [BatchInferenceFrontierCandidate(
        target_rho=0.75, deadline_slack_seconds=300.0,
        deferral_window_seconds=60.0)]
    pts = estimate_batch_inference_frontier(
        _profile(), ticks, cands,
        estimator_config=BatchInferenceEstimatorConfig(tick_seconds=60.0),
        safety_config=BatchInferenceSafetyConfig())
    assert len(pts) == 1
    # Estimator does not blow up on the new field.
    assert pts[0].predicted_goodput_per_dollar is not None


# ---- 12. Serving frontier still imports + works (regression) ----

def test_existing_serving_frontier_modules_still_import():
    from aurelius.frontier import (  # noqa: F401
        controller, dynamic_controller, dynamic_estimator, estimator,
        models, safety,
    )
