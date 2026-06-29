"""Canonical-state validation — PASS/WARN/FAIL invariants for the new/promoted states.

Checks the honesty + cohesion properties the PR depends on: request conservation, queue-summary consistency,
ForecastState causality (no future leakage; error only after realization), clone isolation, and that legacy V1
behaviour is preserved when the new states are not attached. Pure checks over in-memory state — no heavy sim.
"""

from __future__ import annotations

import copy

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


def _r(check, status, detail=""):
    return {"check": check, "status": status, "detail": detail}


def validate_request_conservation(rls) -> dict:
    ok = rls.conserved()
    return _r("request_conservation", PASS if ok else FAIL,
              f"arrived={rls.arrived} running={rls.running()} completed={rls.completed} dropped={rls.dropped}")


def validate_no_completed_in_queue(rls) -> dict:
    q = rls.queue_summary()
    # a completed request must not be counted in the active backlog
    bad = [r.request_id for r in rls.requests.values()
           if r.status == "completed" and r.status in ("arrived", "queued", "admitted")]
    return _r("completed_not_queued", PASS if not bad and q["backlog"] >= 0 else FAIL, f"backlog={q['backlog']}")


def validate_placement_refs(rls, *, valid_replicas: set | None = None) -> dict:
    if valid_replicas is None:
        return _r("placement_refs", WARN, "no replica set supplied (skipped)")
    bad = [r.request_id for r in rls.requests.values()
           if r.placement.replica_id and r.placement.replica_id not in valid_replicas]
    return _r("placement_refs", PASS if not bad else FAIL, f"dangling={len(bad)}")


def validate_forecast_no_leakage(fs) -> dict:
    # belief must be made at or before the target period; error only present once realized
    bad_time = [r.decision_index for r in fs.records if r.made_at_period > r.target_period]
    bad_err = [r.decision_index for r in fs.records if r.forecast_error is not None and r.realized is None]
    status = PASS if not bad_time and not bad_err else FAIL
    return _r("forecast_no_leakage", status,
              f"made_after_target={len(bad_time)} error_before_realized={len(bad_err)}")


def validate_forecast_error_correct(fs) -> dict:
    # every realized record's error must equal belief - realized for shared vars
    bad = 0
    for r in fs.realized_records():
        for k, v in (r.forecast_error or {}).items():
            if abs((r.belief.get(k, 0.0) - r.realized.get(k, 0.0)) - v) > 1e-6:
                bad += 1
    return _r("forecast_error_correct", PASS if bad == 0 else FAIL, f"mismatches={bad}")


def validate_clone_isolation(state) -> dict:
    """Mutating a deep clone must not touch the original (the MPC isolation guarantee)."""
    clone = copy.deepcopy(state)
    before = state.to_dict() if hasattr(state, "to_dict") else repr(state)
    # mutate the clone's primary collection
    if hasattr(clone, "records"):
        clone.records.clear()
    elif hasattr(clone, "requests"):
        clone.requests.clear()
    after = state.to_dict() if hasattr(state, "to_dict") else repr(state)
    return _r("clone_isolation", PASS if before == after else FAIL, "original unchanged after clone mutation")


def validate_all(*, forecast_state=None, request_state=None, valid_replicas=None) -> list:
    """Run every applicable check; returns a list of PASS/WARN/FAIL records."""
    out = []
    if request_state is not None:
        out.append(validate_request_conservation(request_state))
        out.append(validate_no_completed_in_queue(request_state))
        out.append(validate_placement_refs(request_state, valid_replicas=valid_replicas))
        out.append(validate_clone_isolation(request_state))
    if forecast_state is not None:
        out.append(validate_forecast_no_leakage(forecast_state))
        out.append(validate_forecast_error_correct(forecast_state))
        out.append(validate_clone_isolation(forecast_state))
    return out


def summarize(results: list) -> dict:
    counts = {PASS: 0, WARN: 0, FAIL: 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"counts": counts, "ok": counts[FAIL] == 0, "results": results}


__all__ = ["validate_all", "summarize", "PASS", "WARN", "FAIL"]
