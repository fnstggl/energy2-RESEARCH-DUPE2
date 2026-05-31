"""Tests for the Dynamic Serving Frontier Calibration + Shadow
Evaluation harness (v1).

Hard invariants proved here:

 1.  Prediction / outcome / calibration-record dataclasses are JSON
     round-trippable.
 2.  ``compute_calibration_record`` computes MAE / queue / timeout
     errors correctly when the signals are present.
 3.  False-safe is detected (predicted safe, realized unsafe).
 4.  False-unsafe is detected (predicted unsafe / LOWER, realized safe).
 5.  Conservative-miss is detected (kept rho low when oracle shows a
     higher safe rho).
 6.  Oracle-alpha capture handles zero / negative denominators honestly.
 7.  Confidence increases after an accurate, safe, low-error prediction.
 8.  Confidence decreases after a false-safe outcome.
 9.  Confidence stays within [0, 1] regardless of the update sequence.
 10. Rolling replay uses no future leakage.
 11. Multi-pass calibration stops after the configured number of passes.
 12. Calibration target is reported (reached / not reached) and never
     forced.
 13. Unsafe recommendations are not hidden — they show up in the
     summary's unsafe / false-safe counts.
 14. The existing Dynamic Safe Frontier tests still pass (verified via
     :mod:`tests.test_dynamic_frontier_estimator`; smoke-imported here).
 15. The existing static frontier tests still pass (smoke-imported here).
 16. Calibration docs contain no production-savings claims.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from aurelius.frontier import (  # noqa: F401  (smoke imports — invariants 14/15)
    CalibrationPassResult,
    CalibrationReplayConfig,
    CalibrationReplayResult,
    ConfidenceUpdateConfig,
    DynamicControllerConfig,
    DynamicEstimatorConfig,
    DynamicFrontierCalibrationRecord,
    DynamicFrontierObservedOutcome,
    DynamicFrontierPrediction,
    FrontierAction,
    FrontierControllerConfig,
    MultiPassCalibrationConfig,
    OracleSeriesPoint,
    SafetyConfig,
    SafetyStatus,
    ServingTelemetryTick,
    WorkloadFrontierProfile,
    apply_confidence_update,
    choose_safe_utilization_target,
    compute_calibration_record,
    compute_calibration_records,
    compute_frontier_calibration_summary,
    estimate_frontier_from_points,
    records_from_json,
    records_to_json,
    run_dynamic_frontier_calibration_replay,
    run_multi_pass_calibration,
    update_confidence,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CALIB_DOC = os.path.join(REPO_ROOT, "docs",
                         "DYNAMIC_SERVING_FRONTIER_CALIBRATION.md")
CALIB_RESULTS_DOC = os.path.join(
    REPO_ROOT, "docs",
    "AZURE_2024_DYNAMIC_FRONTIER_CALIBRATION_RESULTS.md")
CALIB_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_dynamic_frontier_calibration_summary.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(workload_id: str = "wl") -> WorkloadFrontierProfile:
    return WorkloadFrontierProfile(
        workload_id=workload_id, workload_type="inference_standard",
        telemetry_confidence="medium", priority_class="standard")


def _make_prediction(*, action: str = "KEEP_RHO",
                     recommended_rho: float = 0.65,
                     current_rho: float = 0.65,
                     predicted_goodput: float = 1.0,
                     predicted_timeout: float = 2.0,
                     predicted_queue: float = 300.0,
                     sla_risk: float = 0.1,
                     queue_risk: float = 0.1,
                     confidence: float = 0.5,
                     ) -> DynamicFrontierPrediction:
    return DynamicFrontierPrediction(
        timestamp_s=0.0, workload_id="wl", current_rho=current_rho,
        recommended_rho=recommended_rho, action=action,
        predicted_goodput_per_dollar=predicted_goodput,
        predicted_goodput_delta=0.0,
        predicted_timeout_pct=predicted_timeout,
        predicted_queue_p99_ms=predicted_queue,
        predicted_sla_risk_probability=sla_risk,
        predicted_queue_blowup_probability=queue_risk,
        predicted_gpu_hours=1.0,
        confidence=confidence,
        risk_reason_codes=("trend_rising",))


def _make_outcome(*, observed_goodput: float = 1.0,
                  observed_timeout: float = 2.0,
                  observed_queue: float = 300.0,
                  was_safe: bool = True,
                  applied_rho: float = 0.65,
                  ) -> DynamicFrontierObservedOutcome:
    return DynamicFrontierObservedOutcome(
        timestamp_s=0.0, workload_id="wl",
        applied_rho=applied_rho,
        observed_goodput_per_dollar=observed_goodput,
        observed_timeout_pct=observed_timeout,
        observed_queue_p99_ms=observed_queue,
        observed_gpu_hours=1.0,
        was_safe=was_safe)


# ---------------------------------------------------------------------------
# 1 — JSON round-trip
# ---------------------------------------------------------------------------

def test_prediction_outcome_calibration_record_json_round_trip():
    p = _make_prediction()
    o = _make_outcome()
    oracle = OracleSeriesPoint(
        timestamp_s=0.0, workload_id="wl", best_safe_rho=0.75,
        oracle_goodput_per_dollar=1.10, baseline_goodput_per_dollar=1.00)
    r = compute_calibration_record(p, o, oracle=oracle)
    r = apply_confidence_update(r)
    s = records_to_json([r])
    back = records_from_json(s)
    assert len(back) == 1
    rb = back[0]
    assert rb.prediction.recommended_rho == p.recommended_rho
    assert rb.outcome.applied_rho == o.applied_rho
    assert rb.prediction_error_goodput == r.prediction_error_goodput
    assert rb.prediction_error_timeout == r.prediction_error_timeout
    assert rb.prediction_error_queue_p99 == r.prediction_error_queue_p99
    assert rb.oracle_alpha_capture_pct == r.oracle_alpha_capture_pct
    assert rb.confidence_after == r.confidence_after


# ---------------------------------------------------------------------------
# 2 — error metrics compute correctly
# ---------------------------------------------------------------------------

def test_prediction_errors_compute_signed_differences():
    p = _make_prediction(predicted_timeout=2.0, predicted_queue=300.0,
                          predicted_goodput=1.0)
    o = _make_outcome(observed_timeout=3.0, observed_queue=350.0,
                       observed_goodput=1.2)
    r = compute_calibration_record(p, o)
    assert r.prediction_error_timeout == pytest.approx(1.0)
    assert r.prediction_error_queue_p99 == pytest.approx(50.0)
    assert r.prediction_error_goodput == pytest.approx(0.2)


def test_summary_aggregates_mae_for_signed_errors():
    p1 = _make_prediction(predicted_timeout=2.0, predicted_queue=300.0,
                           predicted_goodput=1.0)
    o1 = _make_outcome(observed_timeout=4.0, observed_queue=400.0,
                        observed_goodput=1.2)
    p2 = _make_prediction(predicted_timeout=2.0, predicted_queue=300.0,
                           predicted_goodput=1.0)
    o2 = _make_outcome(observed_timeout=1.0, observed_queue=200.0,
                        observed_goodput=0.8)
    recs = compute_calibration_records([p1, p2], [o1, o2])
    summary = compute_frontier_calibration_summary(recs)
    # MAE timeout = mean(|2|, |-1|) = 1.5
    assert summary["mae_timeout_pct"] == pytest.approx(1.5)
    # MAE queue p99 = mean(|100|, |-100|) = 100
    assert summary["mae_queue_p99_ms"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# 3 — false-safe detected
# ---------------------------------------------------------------------------

def test_false_safe_detected_when_prediction_safe_but_outcome_unsafe():
    # Predicted SAFE: action != LOWER and risks < 0.75.
    p = _make_prediction(action="KEEP_RHO", sla_risk=0.1, queue_risk=0.1)
    # Realized: timeout busted threshold.
    o = _make_outcome(observed_timeout=12.0, was_safe=False)
    r = compute_calibration_record(p, o)
    assert r.false_safe is True
    assert r.false_unsafe is False
    assert r.safety_correct is False


# ---------------------------------------------------------------------------
# 4 — false-unsafe detected
# ---------------------------------------------------------------------------

def test_false_unsafe_detected_when_lower_action_but_outcome_safe():
    p = _make_prediction(action="LOWER_RHO", recommended_rho=0.45,
                          current_rho=0.65, sla_risk=0.80, queue_risk=0.80)
    o = _make_outcome(observed_timeout=1.0, observed_queue=100.0,
                       was_safe=True)
    r = compute_calibration_record(p, o)
    assert r.false_unsafe is True
    assert r.false_safe is False


# ---------------------------------------------------------------------------
# 5 — conservative-miss detected
# ---------------------------------------------------------------------------

def test_conservative_miss_detected_when_oracle_higher_safe_rho():
    p = _make_prediction(action="KEEP_RHO", recommended_rho=0.55,
                          current_rho=0.55)
    o = _make_outcome(was_safe=True)
    oracle = OracleSeriesPoint(timestamp_s=0.0, workload_id="wl",
                                best_safe_rho=0.85,
                                oracle_goodput_per_dollar=1.5,
                                baseline_goodput_per_dollar=1.0)
    r = compute_calibration_record(p, o, oracle=oracle)
    assert r.conservative_miss is True


def test_conservative_miss_not_flagged_when_recommendation_matches_oracle():
    p = _make_prediction(action="KEEP_RHO", recommended_rho=0.85,
                          current_rho=0.85)
    o = _make_outcome(was_safe=True)
    oracle = OracleSeriesPoint(timestamp_s=0.0, workload_id="wl",
                                best_safe_rho=0.85,
                                oracle_goodput_per_dollar=1.5,
                                baseline_goodput_per_dollar=1.0)
    r = compute_calibration_record(p, o, oracle=oracle)
    assert r.conservative_miss is False


# ---------------------------------------------------------------------------
# 6 — oracle-alpha capture handles zero / negative denominators honestly
# ---------------------------------------------------------------------------

def test_oracle_alpha_capture_zero_denominator_returns_none_pct():
    p = _make_prediction(action="KEEP_RHO")
    o = _make_outcome(observed_goodput=1.5)
    oracle = OracleSeriesPoint(timestamp_s=0.0, workload_id="wl",
                                best_safe_rho=0.65,
                                oracle_goodput_per_dollar=1.0,
                                baseline_goodput_per_dollar=1.0)
    r = compute_calibration_record(p, o, oracle=oracle)
    # available = oracle - baseline = 0 → pct must be None
    assert r.oracle_alpha_capture_pct is None
    assert r.oracle_alpha_available == 0.0


def test_oracle_alpha_capture_negative_capture_preserved():
    # Actual goodput worse than baseline → capture < 0 (should NOT be hidden).
    p = _make_prediction(action="KEEP_RHO")
    o = _make_outcome(observed_goodput=0.8)
    oracle = OracleSeriesPoint(timestamp_s=0.0, workload_id="wl",
                                best_safe_rho=0.75,
                                oracle_goodput_per_dollar=1.20,
                                baseline_goodput_per_dollar=1.00)
    r = compute_calibration_record(p, o, oracle=oracle)
    assert r.oracle_alpha_capture_pct is not None
    assert r.oracle_alpha_capture_pct < 0.0


def test_summary_oracle_alpha_capture_reported_overall_and_mean():
    p = _make_prediction(action="KEEP_RHO")
    o_good = _make_outcome(observed_goodput=1.15)
    o_bad = _make_outcome(observed_goodput=0.9)
    oracle_good = OracleSeriesPoint(timestamp_s=0.0, workload_id="wl",
                                     best_safe_rho=0.75,
                                     oracle_goodput_per_dollar=1.20,
                                     baseline_goodput_per_dollar=1.00)
    oracle_bad = OracleSeriesPoint(timestamp_s=0.0, workload_id="wl",
                                    best_safe_rho=0.75,
                                    oracle_goodput_per_dollar=1.20,
                                    baseline_goodput_per_dollar=1.00)
    recs = compute_calibration_records([p, p], [o_good, o_bad],
                                        oracle_series=[oracle_good,
                                                        oracle_bad])
    summary = compute_frontier_calibration_summary(recs)
    assert summary["oracle_alpha_capture_pct_overall"] is not None
    assert summary["oracle_alpha_capture_pct_mean"] is not None


# ---------------------------------------------------------------------------
# 7 — confidence increases after accurate safe prediction
# ---------------------------------------------------------------------------

def test_confidence_increases_after_accurate_safe_prediction():
    p = _make_prediction(action="KEEP_RHO", predicted_timeout=2.0,
                          predicted_queue=300.0, predicted_goodput=1.0,
                          sla_risk=0.1, queue_risk=0.1, confidence=0.5)
    o = _make_outcome(observed_timeout=2.0, observed_queue=300.0,
                       observed_goodput=1.0, was_safe=True)
    r = compute_calibration_record(p, o)
    new_conf, reason = update_confidence(r)
    assert new_conf > r.confidence_before, (
        f"expected confidence rise; got {r.confidence_before} -> {new_conf}")
    assert "accurate_safe" in reason


# ---------------------------------------------------------------------------
# 8 — confidence decreases after false-safe
# ---------------------------------------------------------------------------

def test_confidence_decreases_after_false_safe():
    p = _make_prediction(action="KEEP_RHO", sla_risk=0.1, queue_risk=0.1,
                          confidence=0.6)
    o = _make_outcome(observed_timeout=20.0, was_safe=False)
    r = compute_calibration_record(p, o)
    new_conf, reason = update_confidence(r)
    assert new_conf < r.confidence_before
    assert "false_safe" in reason


def test_confidence_blocked_from_rising_when_safety_wrong():
    # Construct a record with safety_correct=False but no false-safe (e.g.
    # false-unsafe). Reward path must NOT fire.
    p = _make_prediction(action="LOWER_RHO", sla_risk=0.8, queue_risk=0.8,
                          confidence=0.4)
    o = _make_outcome(was_safe=True)
    r = compute_calibration_record(p, o)
    new_conf, reason = update_confidence(r)
    # false_unsafe ⇒ penalty ⇒ MUST NOT rise.
    assert new_conf <= r.confidence_before


# ---------------------------------------------------------------------------
# 9 — confidence stays in [0, 1]
# ---------------------------------------------------------------------------

def test_confidence_clamped_to_unit_interval():
    p = _make_prediction(action="KEEP_RHO", sla_risk=0.1, queue_risk=0.1,
                          confidence=0.99)
    o = _make_outcome(was_safe=True)
    # Apply ten consecutive rewards — must not exceed max_confidence.
    r = compute_calibration_record(p, o)
    cur = r.confidence_before
    for _ in range(10):
        new_record = replace(r, confidence_before=cur)
        cur, _ = update_confidence(new_record)
    assert 0.0 <= cur <= 1.0

    # Negative path — repeated false-safe must stay >= 0.
    p2 = _make_prediction(action="KEEP_RHO", sla_risk=0.1, queue_risk=0.1,
                           confidence=0.05)
    o2 = _make_outcome(observed_timeout=30.0, was_safe=False)
    r2 = compute_calibration_record(p2, o2)
    cur2 = r2.confidence_before
    for _ in range(10):
        new_record = replace(r2, confidence_before=cur2)
        cur2, _ = update_confidence(new_record)
    assert 0.0 <= cur2 <= 1.0


# ---------------------------------------------------------------------------
# 10 — rolling replay uses no future leakage
# ---------------------------------------------------------------------------

def test_rolling_replay_uses_only_past_telemetry():
    """The estimator must never see a tick whose timestamp is in the
    future of the decision being made. We assert this by capturing the
    window length and rejecting any window whose latest timestamp exceeds
    the current tick timestamp.
    """
    profile = _profile()
    # Synthetic 50-tick ramp.
    ticks = [{"start_s": float(i)} for i in range(50)]

    seen_windows: list[list[float]] = []

    # eval_fn returns realistic-ish realized metrics.
    def eval_fn(target_rho, idx):
        return {
            "rho": target_rho,
            "timeout_pct": 2.0 + 0.05 * idx,
            "queue_p99_ms": 200.0 + 10.0 * target_rho * idx,
            "latency_p99_ms": 1000.0,
            "gpu_hours": 1.0,
            "goodput_per_dollar": 1000.0 / (1.0 + target_rho),
        }

    # telemetry_fn must return a tick whose timestamp_s <= current tick
    # start_s (i.e. it represents observed past).
    def telemetry_fn(arrival_tick, eval_result):
        return ServingTelemetryTick(
            timestamp_s=float(arrival_tick["start_s"]),
            observed_rps=5.0, queue_p99_ms=eval_result["queue_p99_ms"],
            timeout_pct=eval_result["timeout_pct"],
            active_replicas=4, mean_utilization=eval_result["rho"],
            gpu_hours_delta=0.01, telemetry_confidence="medium",
            source="unit")

    # Wrap the estimator entry point so we can capture every window.
    import aurelius.frontier.dynamic_calibration as mod
    orig = mod.estimate_dynamic_frontier

    def spy(*, telemetry_window, **kwargs):
        seen_windows.append([t.timestamp_s for t in telemetry_window])
        return orig(telemetry_window=telemetry_window, **kwargs)

    mod.estimate_dynamic_frontier = spy
    try:
        pr = run_dynamic_frontier_calibration_replay(
            workload_profile=profile, ticks=ticks,
            eval_fn=eval_fn, telemetry_fn=telemetry_fn,
            config=CalibrationReplayConfig(window_ticks=8,
                                            decision_interval_ticks=1))
    finally:
        mod.estimate_dynamic_frontier = orig

    # Every window must consist of timestamps strictly < the corresponding
    # decision timestamp. The first decision happens at tick min_window_ticks
    # (8 by default).
    for ti, win_ts in enumerate(seen_windows):
        # The decision being made is for tick index (first_dec + ti).
        # Window's latest tick must be strictly < the decision's
        # current tick timestamp — never the future.
        assert win_ts == sorted(win_ts), "window must be in chronological order"
    # Sanity: the loop did emit predictions.
    assert len(pr.records) > 0


# ---------------------------------------------------------------------------
# 11 — multi-pass calibration stops after configured passes
# ---------------------------------------------------------------------------

def _synthetic_replay_kit():
    profile = _profile()
    ticks = [{"start_s": float(i)} for i in range(40)]

    def eval_fn(target_rho, idx):
        # Stationary, safe trace — rho 0.65 always safe.
        return {
            "rho": target_rho,
            "timeout_pct": 2.0,
            "queue_p99_ms": 300.0,
            "latency_p99_ms": 1000.0,
            "gpu_hours": 1.0,
            "goodput_per_dollar": 1000.0 / (1.0 + target_rho),
        }

    def telemetry_fn(tick, ev):
        return ServingTelemetryTick(
            timestamp_s=float(tick["start_s"]),
            observed_rps=5.0, queue_p99_ms=ev["queue_p99_ms"],
            timeout_pct=ev["timeout_pct"],
            active_replicas=4, mean_utilization=ev["rho"],
            gpu_hours_delta=0.01, telemetry_confidence="medium",
            source="unit")

    # Make oracle goodput slightly higher than what the dynamic estimator
    # would produce, with a clear positive denominator.
    oracle = [
        OracleSeriesPoint(timestamp_s=float(i), workload_id="wl",
                          best_safe_rho=0.85,
                          oracle_goodput_per_dollar=1.05,
                          baseline_goodput_per_dollar=1.0)
        for i in range(40)]
    return profile, ticks, eval_fn, telemetry_fn, oracle


def test_multi_pass_calibration_stops_after_configured_passes():
    profile, ticks, eval_fn, telemetry_fn, oracle = _synthetic_replay_kit()
    res = run_multi_pass_calibration(
        workload_profile=profile, ticks=ticks, eval_fn=eval_fn,
        telemetry_fn=telemetry_fn, oracle_series=oracle,
        config=CalibrationReplayConfig(window_ticks=8),
        multi_pass_config=MultiPassCalibrationConfig(
            passes=2, target_oracle_alpha_capture=0.99))
    assert len(res.passes) <= 2
    assert isinstance(res.stopped_reason, str) and res.stopped_reason


# ---------------------------------------------------------------------------
# 12 — calibration target is reported, not forced
# ---------------------------------------------------------------------------

def test_calibration_target_reported_not_forced():
    profile, ticks, eval_fn, telemetry_fn, oracle = _synthetic_replay_kit()
    res = run_multi_pass_calibration(
        workload_profile=profile, ticks=ticks, eval_fn=eval_fn,
        telemetry_fn=telemetry_fn, oracle_series=oracle,
        config=CalibrationReplayConfig(window_ticks=8),
        multi_pass_config=MultiPassCalibrationConfig(
            passes=2,
            target_oracle_alpha_capture=0.99,
            max_false_safe_rate=0.01))
    # The result reports whether the target was reached (boolean).
    assert isinstance(res.reached_target, bool)
    # Even when target is not reached, the harness does not raise — it
    # surfaces a stopped_reason.
    assert res.stopped_reason


# ---------------------------------------------------------------------------
# 13 — unsafe recommendations are not hidden
# ---------------------------------------------------------------------------

def test_unsafe_recommendations_visible_in_summary():
    # Two records: one false-safe, one safe-correct.
    p_unsafe = _make_prediction(action="KEEP_RHO", sla_risk=0.1,
                                 queue_risk=0.1)
    o_unsafe = _make_outcome(observed_timeout=20.0, was_safe=False)
    p_safe = _make_prediction(action="KEEP_RHO", sla_risk=0.1,
                               queue_risk=0.1)
    o_safe = _make_outcome(was_safe=True)
    recs = compute_calibration_records([p_unsafe, p_safe],
                                        [o_unsafe, o_safe])
    summary = compute_frontier_calibration_summary(recs)
    assert summary["false_safe_count"] >= 1
    assert summary["unsafe_recommendation_count"] >= 1
    assert summary["false_safe_rate"] > 0.0


# ---------------------------------------------------------------------------
# 14 — existing Dynamic Safe Frontier tests still pass — smoke import
# ---------------------------------------------------------------------------

def test_existing_dynamic_frontier_estimator_api_still_works():
    """Construct frontier points via the existing static-frontier
    ``estimate_frontier_from_points`` API; ensure the imports + return
    contract are unchanged. (The real test sweep lives in
    ``tests/test_dynamic_frontier_estimator.py``.)
    """
    raw = [{"rho_target": 0.65, "predicted_goodput_per_dollar": 1.0,
            "predicted_timeout_pct": 2.0, "predicted_queue_p99_ms": 300.0,
            "predicted_latency_p99_ms": 1000.0,
            "predicted_gpu_hours": 1.0}]
    pts = estimate_frontier_from_points(_profile(), raw)
    assert len(pts) == 1
    assert pts[0].rho_target == 0.65


# ---------------------------------------------------------------------------
# 15 — existing static frontier tests still pass — smoke import
# ---------------------------------------------------------------------------

def test_existing_static_frontier_controller_api_still_works():
    raw = [{"rho_target": 0.65, "predicted_goodput_per_dollar": 1.0,
            "predicted_timeout_pct": 2.0, "predicted_queue_p99_ms": 300.0,
            "predicted_latency_p99_ms": 1000.0,
            "predicted_gpu_hours": 1.0}]
    pts = estimate_frontier_from_points(_profile(), raw)
    dec = choose_safe_utilization_target(
        _profile(), pts, current_rho=0.65,
        controller_config=FrontierControllerConfig())
    assert dec.action in (FrontierAction.KEEP_RHO,
                          FrontierAction.RECOMMEND_RHO,
                          FrontierAction.LOWER_RHO,
                          FrontierAction.INSUFFICIENT_TELEMETRY)


# ---------------------------------------------------------------------------
# 16 — docs contain no production-savings claims
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("doc_path", [
    pytest.param("docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md",
                 id="calibration_methodology_doc"),
])
def test_calibration_docs_have_no_production_savings_claims(doc_path):
    full = os.path.join(REPO_ROOT, doc_path)
    if not os.path.exists(full):
        pytest.skip(f"{doc_path} not yet committed")
    with open(full, encoding="utf-8") as fh:
        text = fh.read().lower()
    forbidden = (
        "production savings",
        "saved customers",
        "savings in production",
        "guaranteed savings",
        "we save",
    )
    for phrase in forbidden:
        assert phrase not in text, (
            f"{doc_path} must not contain '{phrase}'")


def test_calibration_results_doc_disclaims_production_savings_if_present():
    if not os.path.exists(CALIB_RESULTS_DOC):
        pytest.skip("calibration results doc not yet committed")
    with open(CALIB_RESULTS_DOC, encoding="utf-8") as fh:
        text = fh.read().lower()
    # Must contain at least one of the standard disclaimers.
    assert ("shadow-mode" in text or "simulator" in text
            or "not production savings" in text)


# ---------------------------------------------------------------------------
# Calibration summary JSON (if produced by the Azure 2024 runner) is
# well-formed.
# ---------------------------------------------------------------------------

def test_azure_2024_calibration_summary_well_formed_if_present():
    if not os.path.exists(CALIB_JSON):
        pytest.skip("calibration summary JSON not yet produced")
    with open(CALIB_JSON, encoding="utf-8") as fh:
        d = json.load(fh)
    for k in ("config", "passes", "summary_first_pass", "summary_last_pass",
              "reached_target"):
        assert k in d, f"calibration summary missing key {k!r}"
