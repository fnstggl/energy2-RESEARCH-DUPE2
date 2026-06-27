#!/usr/bin/env python3
"""Horizon ablation for the receding-horizon MPC (research/MPC_HORIZON_ANALYSIS.md).

Trains forecasters once, then runs the world-state MPC on the held-out periods at a grid of horizons
H (in SIM STEPS) — committing only the first action each control interval — and reports gp/$, SLA,
GPU-hours, the chosen-action mixes, the per-decision runtime, the total world-steps simulated, and
the Pareto gate vs a fair baseline. H=1 is the single-period controller (backward-compat anchor).

Usage: python -m scripts.sweep_mpc_horizon --stride 96 --horizons 1,2,4
"""

from __future__ import annotations

import argparse
import json
import os
import time

from aurelius.environment.controller import ModelPredictiveEconomicController, run_period_episode
from aurelius.environment.training import (
    DEFAULT_BASELINES,
    build_mpc_inputs,
    claim_gate,
    make_world_state,
    train_forecasters,
)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--sim-seconds", type=float, default=180.0)
    ap.add_argument("--horizons", default="1,2,4")
    ap.add_argument("--risk-weight", type=float, default=0.3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=args.sim_seconds, use_world_state=True)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    t1 = max(4, int(len(frames) * 0.5))
    t2 = min(max(t1 + 2, int(len(frames) * 0.75)), len(frames) - 1)
    splits = {"eval": [t2, len(frames)]}
    fm, _ = train_forecasters(frames, t1)

    fair = run_period_episode(
        "fair", (lambda a: (lambda h: dict(a)))(DEFAULT_BASELINES["aurelius_canonical_kv_routing"]),
        inp["per"], frames, list(range(*splits["eval"])), fleet_state=fleet, cost_model=cm,
        world_state=make_world_state(common.get("world_state_params")), **common)

    rows = []
    for H in [int(x) for x in args.horizons.split(",")]:
        ws = make_world_state(common.get("world_state_params"))
        ctrl = ModelPredictiveEconomicController(
            forecasters=fm, fleet_state=fleet, cost_model=cm, risk_weight=args.risk_weight,
            confidence_min=0.1, sla_s=common["sla_s"], period_seconds=common["period_seconds"],
            tick_seconds=common["tick_seconds"], kv_service_factor=common.get("kv_service_factor", 1.0),
            kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
            cost_scenario=common.get("cost_scenario", "owned"), sim_seconds=common.get("sim_seconds"),
            world_state=ws, horizon_steps=H)
        t0 = time.monotonic()
        rep = run_period_episode("mpc", lambda h: ctrl.decide(h).to_dict(), inp["per"], frames,
                                 list(range(*splits["eval"])), fleet_state=fleet, cost_model=cm,
                                 world_state=ws, **common)
        wall = time.monotonic() - t0
        gate = claim_gate({"mpc_controller": rep, "fair": fair})
        diag = ctrl.last_decision_diag
        rows.append({"horizon_steps": H, "lookahead_minutes": diag.get("lookahead_minutes"),
                     "goodput_per_dollar": round(rep.goodput_per_dollar, 1),
                     "sla_violation_rate": round(rep.sla_violation_rate, 4),
                     "queue_delay_p95": round(rep.queue_delay_p95, 2), "gpu_hours": round(rep.gpu_hours, 2),
                     "capacity_multiplier_mix": {str(k): v for k, v in rep.capacity_multiplier_mix.items()},
                     "prewarm_mix": rep.prewarm_mix, "placement_mix": rep.placement_mix,
                     "migration_mix": rep.migration_mix, "wall_seconds_total": round(wall, 1),
                     "runtime_s_per_decision": round(wall / max(1, splits["eval"][1] - splits["eval"][0]), 3),
                     "world_steps_last_decision": diag.get("world_steps_simulated"),
                     "beats_fair": gate["beats_fair_baseline"],
                     "pareto_sla_not_worse": gate["pareto_sla_not_worse"],
                     "headline_allowed": gate["headline_claim_allowed"]})

    out = {"stride": args.stride, "eval_periods": splits["eval"],
           "fair_gp_per_dollar": round(fair.goodput_per_dollar, 1), "fair_sla": round(fair.sla_violation_rate, 4),
           "rows": rows}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "mpc_horizon_ablation.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"fair gp/$={out['fair_gp_per_dollar']} sla={out['fair_sla']}")
    print(f"{'H':>2} {'look(min)':>9} {'gp/$':>10} {'sla':>7} {'gpu_h':>7} {'rt/dec':>7}  cap_mix | gate")
    for r in rows:
        cmix = ",".join(f"{k}x{v}" for k, v in sorted(r["capacity_multiplier_mix"].items()))
        print(f"{r['horizon_steps']:>2} {str(r['lookahead_minutes']):>9} {r['goodput_per_dollar']:>10.0f} "
              f"{r['sla_violation_rate']:>7.4f} {r['gpu_hours']:>7.1f} {r['runtime_s_per_decision']:>7.3f}  "
              f"{cmix} | {r['beats_fair']}/{r['pareto_sla_not_worse']}/{r['headline_allowed']}")


if __name__ == "__main__":
    main()
