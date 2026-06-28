"""Planning/eval fidelity tests (MPC search-architecture PR).

Lock the mechanism: when planning parity is enabled, the planning rollout runs the same phase + hybrid-cost
+ roofline model the evaluator uses, so the planner SEES precision's COST benefit (the PR #111 miss) — and
the default (parity off) reproduces the prior behaviour exactly. These are integration tests on the real
Azure+Mooncake inputs; they skip cleanly when that data is unavailable.
"""

from __future__ import annotations

import pytest

from aurelius.environment.actions import ActionBundle
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import build_mpc_inputs, make_world_state, train_forecasters


def _inputs():
    inp = build_mpc_inputs(hourly_stride=96, sim_seconds=180.0, use_world_state=True,
                           control_dt_seconds=60.0)
    if inp is None:
        pytest.skip("no Azure serving data available")
    return inp


def _bundle(precision):
    return ActionBundle(precision_policy=precision, capacity_policy="backlog_aware",
                        ordering_policy="abs_conformal", admission_policy="off", batching_policy="balanced")


def test_planning_parity_makes_simulate_period_see_fp8_cost_benefit():
    """The core mechanism, unit-level: a planning-style kv_state (unique prefix + hybrid cost mode) makes an
    fp8 bundle bill no more than bf16 (memory-bandwidth-bound decode → fewer bytes → cheaper) — i.e. the
    planning path now exposes the cost channel the evaluator uses."""
    from aurelius.environment.world_simulator import simulate_period
    inp = _inputs()
    fleet, cm, per, wsp = inp["fleet_state"], inp["cost_model"], inp["per"], inp["common"].get("world_state_params")
    p = next(iter(per))
    recs = [(r[0], int(r[1]), int(r[2]) if len(r) > 2 else int(r[1])) for r in per[p][:64]]
    if not recs:
        pytest.skip("empty period")
    kv = {"hash_seq": [(f"u{i}",) for i in range(len(recs))], "routing": "kv_aware",
          "capacity_blocks": 256, "cost_mode": "hybrid_capacity_work"}
    common = dict(sla_s=10.0, tick_seconds=10.0, cost_model=cm, fleet_state=fleet, period_hours=180.0 / 3600.0)
    out = {}
    for prec in ("bf16", "fp8"):
        b = _bundle(prec)
        fc = {"arrival_rate": len(recs) / 180.0, "arrival_p90": 1.3 * len(recs) / 180.0, "mean_service_s": 1.0}
        out[prec] = simulate_period(make_world_state(wsp), b, recs, fc, replay_kwargs=b.replay_kwargs(),
                                    kv_state=dict(kv), **common)
    assert out["fp8"].operator_cost <= out["bf16"].operator_cost + 1e-9    # the cost channel is visible


def test_planning_off_picks_bf16_planning_on_picks_fp8_on_memory_bound_azure():
    """End-to-end: the SAME decision flips bf16→fp8 when planning parity is enabled (the PR #111 miss)."""
    inp = _inputs()
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    n = len(frames)
    fm, _ = train_forecasters(frames, n - 20)
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    sel = {}
    for mode in (None, "hybrid_capacity_work"):
        ws = make_world_state(common.get("world_state_params"))
        ctrl = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        ctrl.horizon_steps = 1
        ctrl.planning_kv_cost_mode = mode
        ctrl.planning_prompt_tokens = 857 if mode else None
        sel[mode] = ctrl.decide(frames[: n - 1]).bundle.precision_policy
    assert sel[None] == "bf16"                       # latency-only planning is blind to fp8's cost benefit
    assert sel["hybrid_capacity_work"] == "fp8"      # parity planning SEES the cost benefit → selects fp8


def test_planning_parity_off_by_default():
    """The default controller has planning parity OFF (the dt=60 rerun shows enabling it regresses gp/$ on
    this workload via an SLA-representation gap — so it stays opt-in until that is closed)."""
    inp = _inputs()
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    fm, _ = train_forecasters(frames, len(frames) - 20)
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ctrl = build_controller(fm, fleet, cm, cfg, common, world_state=make_world_state(common.get("world_state_params")))
    assert ctrl.planning_kv_cost_mode is None and ctrl.planning_prompt_tokens is None
