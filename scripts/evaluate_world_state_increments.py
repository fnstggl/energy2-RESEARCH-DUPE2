#!/usr/bin/env python3
"""Incremental (per-action) held-out evaluation of the STATEFUL actions on the persistent world.

Companion to ``evaluate_action_increments.py`` (which isolates capacity_multiplier + batching).
This one isolates the three world-state actions connected on top of the persistent canonical world
(prewarm / placement / migration). It trains the controller ONCE, then evaluates it on the SAME
held-out periods under a ladder of *frozen* search spaces:

  core            prewarm := off, placement := topology_blind, migration := off  (only the PR-#100
                  connected set varies on the world path). The control.
  +prewarm        placement + migration pinned to no-op; prewarm free.
  +placement      prewarm + migration pinned; placement free.
  +migration      prewarm + placement pinned; migration free.
  full            nothing pinned — all nine connected surfaces vary.

Each rung freezes surfaces EXPLICITLY (``CandidateBundleGenerator(frozen=...)``) with a recorded
reason, so nothing is silently excluded. The marginal contribution of a stateful action is
``rung − core`` on gp/$ AND on the SLA-violation rate, reported together so a gp/$ gain bought by
worse SLA (or by warm-hold cost) is visible. We also report the chosen-value frequency and the new
stateful ledgers (cold-start events, warm-hold GPU-hours, migration cost, mean topology factor).

Usage:
  python -m scripts.evaluate_world_state_increments --json
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.candidate_search import CandidateBundleGenerator
from aurelius.environment.controller import ModelPredictiveEconomicController, run_period_episode
from aurelius.environment.training import build_mpc_inputs, make_world_state, train_mpc_policy

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")

_NOOP = {"prewarm_policy": "off", "placement_policy": "topology_blind", "migration_policy": "off"}
_RUNGS = [
    ("core", dict(_NOOP), "control: all three stateful knobs pinned to no-op (PR-#100 set varies)"),
    ("+prewarm", {"placement_policy": "topology_blind", "migration_policy": "off"},
     "isolate prewarm: placement + migration held at no-op"),
    ("+placement", {"prewarm_policy": "off", "migration_policy": "off"},
     "isolate placement: prewarm + migration held at no-op"),
    ("+migration", {"prewarm_policy": "off", "placement_policy": "topology_blind"},
     "isolate migration: prewarm + placement held at no-op"),
    ("full", {}, "shipped planner: every connected surface free"),
]


def _episode(fm, trained, inp, frozen, reason):
    common = inp["common"]
    wsp = common.get("world_state_params")
    gen = CandidateBundleGenerator(frozen=dict(frozen),
                                   frozen_reasons={k: reason for k in frozen})
    ws = make_world_state(wsp)
    ctrl = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=inp["fleet_state"], cost_model=inp["cost_model"],
        horizon=trained["controller_config"]["horizon"],
        risk_weight=trained["controller_config"]["risk_weight"],
        confidence_min=trained["controller_config"]["confidence_min"],
        sla_s=common["sla_s"], period_seconds=common["period_seconds"],
        tick_seconds=common["tick_seconds"], kv_service_factor=common.get("kv_service_factor", 1.0),
        kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
        cost_scenario=common.get("cost_scenario", "owned"), sim_seconds=common.get("sim_seconds"),
        candidate_generator=gen, search_budget=256, world_state=ws)
    e0, e1 = trained["splits"]["eval"]
    rep = run_period_episode(
        "mpc", lambda h: ctrl.decide(h).to_dict(), inp["per"], inp["frames"],
        list(range(e0, e1)), fleet_state=inp["fleet_state"], cost_model=inp["cost_model"],
        world_state=ws, **common)
    return rep, sorted(gen.surfaces())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-stride", type=int, default=24)
    ap.add_argument("--sim-seconds", type=float, default=240.0)
    ap.add_argument("--out-dir", default=_OUT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.hourly_stride, sim_seconds=args.sim_seconds,
                           use_world_state=True)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    trained, fm = train_mpc_policy(inp["frames"], inp["per"], fleet_state=inp["fleet_state"],
                                   cost_model=inp["cost_model"], common=inp["common"])
    rungs = {}
    for label, frozen, reason in _RUNGS:
        rep, free = _episode(fm, trained, inp, frozen, reason)
        rungs[label] = {"free_surfaces": free, "frozen": frozen,
                        "goodput_per_dollar": round(rep.goodput_per_dollar, 2),
                        "sla_violation_rate": round(rep.sla_violation_rate, 5),
                        "queue_delay_p95": round(rep.queue_delay_p95, 3),
                        "gpu_hours": round(rep.gpu_hours, 3),
                        "prewarm_mix": rep.prewarm_mix, "placement_mix": rep.placement_mix,
                        "migration_mix": rep.migration_mix,
                        "cold_start_events": rep.cold_start_events,
                        "warm_hold_gpu_hours": round(rep.warm_hold_gpu_hours, 3),
                        "migration_cost": round(rep.migration_cost, 3),
                        "mean_topology_factor": round(rep.mean_topology_factor, 5)}

    base = rungs["core"]
    marginals = {}
    for label in ("+prewarm", "+placement", "+migration", "full"):
        r = rungs[label]
        gpd_pct = (100.0 * (r["goodput_per_dollar"] - base["goodput_per_dollar"])
                   / base["goodput_per_dollar"]) if base["goodput_per_dollar"] else 0.0
        marginals[label] = {
            "gp_per_dollar_pct_vs_core": round(gpd_pct, 3),
            "sla_violation_rate_delta": round(r["sla_violation_rate"] - base["sla_violation_rate"], 5),
            "pareto_sla_not_worse": r["sla_violation_rate"] <= base["sla_violation_rate"] + 1e-9,
        }

    out = {"eval_periods": trained["splits"]["eval"], "controller_config": trained["controller_config"],
           "coverage": inp.get("coverage"), "world_state_params": inp["common"].get("world_state_params"),
           "rungs": rungs, "marginals_vs_core": marginals,
           "note": "SIMULATED (directional simulator evidence on a TRACE_DERIVED_SAMPLE cluster), "
                   "not production telemetry"}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "world_state_increment_report.json"), "w") as f:
        json.dump(out, f, indent=2)

    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"world-state incremental eval over periods {trained['splits']['eval']}")
    for label, r in rungs.items():
        print(f"  {label:12} gp/$={r['goodput_per_dollar']:>11.1f}  sla={r['sla_violation_rate']:.4f}  "
              f"q_p95={r['queue_delay_p95']:.2f}s  cold={r['cold_start_events']}  "
              f"warm_hold={r['warm_hold_gpu_hours']:.1f}h  mig=${r['migration_cost']:.1f}")
    print("  marginal vs core:")
    for label, m in marginals.items():
        print(f"    {label:12} {m['gp_per_dollar_pct_vs_core']:+.2f}% gp/$  "
              f"ΔSLA={m['sla_violation_rate_delta']:+.4f}  pareto_ok={m['pareto_sla_not_worse']}")


if __name__ == "__main__":
    main()
