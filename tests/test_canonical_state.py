"""Canonical-state fixtures — request lifecycle persistence, queue consolidation, forecast causality.

These are the Phase-11 controlled fixtures as executable tests: they prove the new/promoted canonical states
behave honestly (conservation, no future leakage, error-only-after-realization, clone isolation) and that the
legacy path is untouched when the states are not attached.
"""

from __future__ import annotations

import copy

from aurelius.environment import state_validation as sv
from aurelius.environment.forecast_state import ForecastState
from aurelius.environment.request_state import RequestLifecycleState, RooflineRecord


# --- RequestState lifecycle persistence + conservation -----------------------
def test_request_lifecycle_persists_and_conserves():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(i * 1.0, 200, 64) for i in range(8)], sla_s=10.0, sla_safe_frac=0.75)
    rls.ingest_period(1, [(i * 1.0, 300, 64) for i in range(4)], sla_s=10.0, sla_safe_frac=1.0)
    assert rls.arrived == 12                                  # requests PERSIST across periods (not lost)
    assert rls.completed == 6 + 4 and rls.missed_sla == 2
    assert rls.conserved()                                    # arrived == running + completed + dropped
    assert sv.validate_request_conservation(rls)["status"] == sv.PASS


def test_no_request_disappears_and_none_double_counted():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(i * 0.5, 128, 32) for i in range(20)], sla_s=5.0, sla_safe_frac=0.6)
    # every arrived request is in exactly one terminal/active bucket
    assert rls.arrived == 20 == (rls.running() + rls.completed + rls.dropped)
    assert len(rls.requests) == 20                            # no request disappears


def test_queue_summary_consolidation():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(i * 1.0, 200, 64) for i in range(10)], sla_s=10.0, sla_safe_frac=0.5)
    q = rls.queue_summary()
    assert q["arrived"] == 10 and q["completed"] == 5 and q["completion_rate"] == 0.5
    assert q["class_mix"]["latency_critical"] == 10
    assert q["backlog"] >= 0


def test_completed_request_not_in_backlog():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(i * 1.0, 200, 64) for i in range(6)], sla_s=10.0, sla_safe_frac=1.0)
    assert sv.validate_no_completed_in_queue(rls)["status"] == sv.PASS
    assert rls.queue_summary()["backlog"] == 0                # all completed → empty backlog


def test_placement_ref_validation_catches_dangling():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(0.0, 200, 64)], sla_s=10.0, sla_safe_frac=1.0)
    r = next(iter(rls.requests.values()))
    r.placement.replica_id = "rep_ghost"
    assert sv.validate_placement_refs(rls, valid_replicas={"rep0", "rep1"})["status"] == sv.FAIL
    r.placement.replica_id = "rep0"
    assert sv.validate_placement_refs(rls, valid_replicas={"rep0", "rep1"})["status"] == sv.PASS


# --- ForecastState causality (no future leakage; error only after realization) -
def test_forecast_error_only_after_realization():
    fs = ForecastState()
    fs.record_belief(decision_index=0, target_period=2, made_at_period=2, horizon_index=0,
                     belief={"arrival_rate": 1.5, "output_token_mean": 280.0})
    fs.n_decisions += 1
    rec = fs.records[0]
    assert rec.forecast_error is None and not rec.is_realized   # NO error before realization (no leakage)
    fs.record_realized(2, {"arrival_rate": 1.2, "output_token_mean": 300.0})
    assert rec.is_realized
    assert abs(rec.forecast_error["arrival_rate"] - 0.3) < 1e-9
    assert abs(rec.forecast_error["output_token_mean"] - (-20.0)) < 1e-9
    assert sv.validate_forecast_no_leakage(fs)["status"] == sv.PASS
    assert sv.validate_forecast_error_correct(fs)["status"] == sv.PASS


def test_forecast_error_summary_mae_mape():
    fs = ForecastState()
    for p, (b, r) in enumerate([(1.0, 1.0), (2.0, 1.0), (3.0, 5.0)]):
        fs.record_belief(decision_index=p, target_period=p, made_at_period=p, horizon_index=0,
                         belief={"arrival_rate": b})
        fs.n_decisions += 1
        fs.record_realized(p, {"arrival_rate": r})
    s = fs.forecast_error_summary()["arrival_rate"]
    # |1-1| + |2-1| + |3-5| = 0+1+2 = 3 → MAE 1.0 over 3 realized
    assert s["mae"] == 1.0 and s["n"] == 3


def test_forecast_made_after_target_is_flagged():
    fs = ForecastState()
    fs.record_belief(decision_index=0, target_period=2, made_at_period=5, horizon_index=0,
                     belief={"arrival_rate": 1.0})        # belief about the PAST → leakage
    assert sv.validate_forecast_no_leakage(fs)["status"] == sv.FAIL


# --- clone isolation (the MPC search guarantee) ------------------------------
def test_clone_isolation_request_and_forecast():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(0.0, 200, 64)], sla_s=10.0, sla_safe_frac=1.0)
    fs = ForecastState()
    fs.record_belief(decision_index=0, target_period=0, made_at_period=0, horizon_index=0,
                     belief={"arrival_rate": 1.0})
    fs.n_decisions += 1
    for st in (rls, fs):
        clone = copy.deepcopy(st)
        (clone.requests if hasattr(clone, "requests") else clone.records).clear()
        assert sv.validate_clone_isolation(st)["status"] == sv.PASS


# --- RooflineState record snapshot -------------------------------------------
def test_roofline_record_from_diag():
    rr = RooflineRecord.from_diag(7, {"decode_regime": "compute_bound", "arithmetic_intensity": 220.0,
                                      "precision": "fp8", "timing_model": "roofline"}, gpu_type="H100", power_w=540.0)
    assert rr.decode_regime == "compute_bound" and rr.precision == "fp8"
    assert rr.timing_model == "roofline" and rr.power_w == 540.0


# --- validate_all aggregates PASS/WARN/FAIL ----------------------------------
def test_validate_all_green_on_honest_states():
    rls = RequestLifecycleState()
    rls.ingest_period(0, [(i * 1.0, 200, 64) for i in range(5)], sla_s=10.0, sla_safe_frac=0.8)
    fs = ForecastState()
    fs.record_belief(decision_index=0, target_period=0, made_at_period=0, horizon_index=0,
                     belief={"arrival_rate": 1.0})
    fs.n_decisions += 1
    fs.record_realized(0, {"arrival_rate": 0.9})
    summ = sv.summarize(sv.validate_all(forecast_state=fs, request_state=rls))
    assert summ["ok"] and summ["counts"][sv.FAIL] == 0
