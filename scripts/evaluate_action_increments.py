#!/usr/bin/env python3
"""Incremental (per-action) held-out evaluation of the NEWLY-CONNECTED action surfaces.

The full backtest (``scripts/evaluate_mpc_controller.py``) lets the planner vary every
connected surface at once. That answers "is the whole bundle better?" but NOT "what did each
newly-connected action add?" — the question the next-batch task requires us to answer per
action, honestly, with a fair baseline.

This script trains the controller ONCE (same forecasters + tuned config as the full backtest)
then evaluates it on the SAME held-out periods under a ladder of *frozen* search spaces:

  pr99_core       capacity_multiplier := 1.0  AND  batching := conservative   (both new knobs
                  pinned to their no-op → only the PR-#99 connected set varies). The control.
  +capacity_mult  batching pinned conservative; capacity_multiplier free to move.
  +batching       capacity_multiplier pinned 1.0; batching free to move.
  full            nothing pinned — all six connected surfaces vary (the shipped planner).

Each rung freezes surfaces EXPLICITLY via ``CandidateBundleGenerator(frozen=...)`` with a
recorded reason, so nothing is silently excluded. The marginal contribution of an action is
``rung − pr99_core`` on gp/$ AND on the SLA-violation rate — reported together so a gp/$ gain
bought by more SLA misses is visible, never hidden. We also print the chosen-value frequency
for each new knob (how often the planner actually used each level on the real week).

Usage:
  python -m scripts.evaluate_action_increments --json
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.candidate_search import CandidateBundleGenerator
from aurelius.environment.controller import ModelPredictiveEconomicController, run_period_episode
from aurelius.environment.training import build_mpc_inputs, train_mpc_policy

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")

# Each rung: (label, frozen-surface -> pinned no-op value, human reason).
_RUNGS = [
    ("pr99_core", {"capacity_multiplier": 1.0, "batching_policy": "conservative"},
     "control: both next-batch knobs pinned to their no-op (only the PR-#99 set varies)"),
    ("+capacity_mult", {"batching_policy": "conservative"},
     "isolate capacity_multiplier: batching held at its no-op"),
    ("+batching", {"capacity_multiplier": 1.0},
     "isolate batching: capacity_multiplier held at its no-op"),
    ("full", {}, "shipped planner: every connected surface free"),
]


def _episode(fm, trained, inp, frozen, reason):
    common = inp["common"]
    gen = CandidateBundleGenerator(
        frozen=dict(frozen),
        frozen_reasons={k: reason for k in frozen})
    ctrl = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=inp["fleet_state"], cost_model=inp["cost_model"],
        horizon=trained["controller_config"]["horizon"],
        risk_weight=trained["controller_config"]["risk_weight"],
        confidence_min=trained["controller_config"]["confidence_min"],
        sla_s=common["sla_s"], period_seconds=common["period_seconds"],
        tick_seconds=common["tick_seconds"], kv_service_factor=common.get("kv_service_factor", 1.0),
        kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
        cost_scenario=common.get("cost_scenario", "owned"), sim_seconds=common.get("sim_seconds"),
        candidate_generator=gen, search_budget=256)
    e0, e1 = trained["splits"]["eval"]
    rep = run_period_episode(
        "mpc", lambda h: ctrl.decide(h).to_dict(), inp["per"], inp["frames"],
        list(range(e0, e1)), fleet_state=inp["fleet_state"], cost_model=inp["cost_model"],
        **common)
    return rep, sorted(gen.surfaces())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-stride", type=int, default=24)
    ap.add_argument("--sim-seconds", type=float, default=240.0)
    ap.add_argument("--out-dir", default=_OUT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.hourly_stride, sim_seconds=args.sim_seconds)
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
                        "routing_mix": rep.routing_mix,
                        "capacity_multiplier_mix": {str(k): v for k, v in rep.capacity_multiplier_mix.items()},
                        "batching_mix": rep.batching_mix}

    base = rungs["pr99_core"]
    marginals = {}
    for label in ("+capacity_mult", "+batching", "full"):
        r = rungs[label]
        gpd_pct = (100.0 * (r["goodput_per_dollar"] - base["goodput_per_dollar"])
                   / base["goodput_per_dollar"]) if base["goodput_per_dollar"] else 0.0
        marginals[label] = {
            "gp_per_dollar_pct_vs_pr99_core": round(gpd_pct, 3),
            "sla_violation_rate_delta": round(r["sla_violation_rate"] - base["sla_violation_rate"], 5),
            "pareto_sla_not_worse": r["sla_violation_rate"] <= base["sla_violation_rate"] + 1e-9,
        }

    out = {"eval_periods": trained["splits"]["eval"], "controller_config": trained["controller_config"],
           "coverage": inp.get("coverage"), "rungs": rungs, "marginals_vs_pr99_core": marginals,
           "note": "SIMULATED (directional simulator evidence), not production telemetry"}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "action_increment_report.json"), "w") as f:
        json.dump(out, f, indent=2)

    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"incremental held-out eval over periods {trained['splits']['eval']}")
    for label, r in rungs.items():
        print(f"  {label:16} gp/$={r['goodput_per_dollar']:>11.1f}  sla_viol={r['sla_violation_rate']:.4f}  "
              f"q_p95={r['queue_delay_p95']:.2f}s  free={r['free_surfaces']}")
    print("  marginal vs pr99_core:")
    for label, m in marginals.items():
        print(f"    {label:14} {m['gp_per_dollar_pct_vs_pr99_core']:+.2f}% gp/$  "
              f"ΔSLA={m['sla_violation_rate_delta']:+.4f}  pareto_ok={m['pareto_sla_not_worse']}")


if __name__ == "__main__":
    main()
