#!/usr/bin/env python3
"""Phase-5 objective/horizon sweep on the CALIBRATED persistent world.

The regression fix is a cost-model calibration (warm-hold idle-timeout), NOT an objective change.
This sweep proves that: it runs the calibrated world-state MPC at a grid of FIXED
(risk_weight, horizon) configs on the held-out periods and reports, for each, the capacity_multiplier
mix + prewarm/placement/migration mix + gp/$ + SLA + GPU-hours + the Pareto gate vs the fair baseline.

Key questions:
- Does the controller still pick capacity_multiplier=1.5 every period after the fix? (Expect: no.)
- Does a calibrated, NON-zero risk_weight still respect SLA? (We do NOT just set risk_weight=0.)

Usage: python -m scripts.sweep_world_state_objective --json [--stride 48]
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.controller import ModelPredictiveEconomicController, run_period_episode
from aurelius.environment.training import (
    DEFAULT_BASELINES,
    build_mpc_inputs,
    claim_gate,
    make_world_state,
    train_forecasters,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")


def _episode(fm, inp, splits, risk_weight, horizon):
    common, fleet, cm = inp["common"], inp["fleet_state"], inp["cost_model"]
    ws = make_world_state(common.get("world_state_params"))
    ctrl = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=fleet, cost_model=cm, horizon=horizon, risk_weight=risk_weight,
        confidence_min=0.1, sla_s=common["sla_s"], period_seconds=common["period_seconds"],
        tick_seconds=common["tick_seconds"], kv_service_factor=common.get("kv_service_factor", 1.0),
        kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
        cost_scenario=common.get("cost_scenario", "owned"), sim_seconds=common.get("sim_seconds"),
        world_state=ws)
    e0, e1 = splits["eval"]
    return run_period_episode("mpc", lambda h: ctrl.decide(h).to_dict(), inp["per"], inp["frames"],
                              list(range(e0, e1)), fleet_state=fleet, cost_model=cm,
                              world_state=ws, **common)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--sim-seconds", type=float, default=180.0)
    ap.add_argument("--risk-weights", default="0.0,0.25,0.5,1.0")
    ap.add_argument("--horizons", default="1,2")
    ap.add_argument("--out-dir", default=_OUT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=args.sim_seconds, use_world_state=True)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    frames = inp["frames"]
    t1 = max(4, int(len(frames) * 0.5))
    t2 = max(t1 + 2, int(len(frames) * 0.75))
    t2 = min(t2, len(frames) - 1)
    splits = {"train_cut": t1, "val": [t1, t2], "eval": [t2, len(frames)]}
    fm, _ = train_forecasters(frames, t1)

    # a fair static baseline for the gate (no stateful actions; capacity 1.0x)
    base_ws = make_world_state(inp["common"].get("world_state_params"))
    fair = run_period_episode(
        "aurelius_canonical_kv_routing",
        (lambda a: (lambda h: dict(a)))(DEFAULT_BASELINES["aurelius_canonical_kv_routing"]),
        inp["per"], frames, list(range(*splits["eval"])), fleet_state=inp["fleet_state"],
        cost_model=inp["cost_model"], world_state=base_ws, **inp["common"])

    rows = []
    for rw in [float(x) for x in args.risk_weights.split(",")]:
        for hz in [int(x) for x in args.horizons.split(",")]:
            rep = _episode(fm, inp, splits, rw, hz)
            gate = claim_gate({"mpc_controller": rep, "aurelius_canonical_kv_routing": fair})
            rows.append({"risk_weight": rw, "horizon": hz,
                         "goodput_per_dollar": round(rep.goodput_per_dollar, 1),
                         "sla_violation_rate": round(rep.sla_violation_rate, 4),
                         "gpu_hours": round(rep.gpu_hours, 2),
                         "capacity_multiplier_mix": {str(k): v for k, v in rep.capacity_multiplier_mix.items()},
                         "prewarm_mix": rep.prewarm_mix, "placement_mix": rep.placement_mix,
                         "migration_mix": rep.migration_mix,
                         "beats_fair": gate["beats_fair_baseline"],
                         "pareto_sla_not_worse": gate["pareto_sla_not_worse"],
                         "headline_allowed": gate["headline_claim_allowed"]})

    out = {"stride": args.stride, "eval_periods": splits["eval"],
           "fair_baseline_gp_per_dollar": round(fair.goodput_per_dollar, 1),
           "fair_sla": round(fair.sla_violation_rate, 4), "rows": rows}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "world_state_objective_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"fair baseline gp/$={out['fair_baseline_gp_per_dollar']} sla={out['fair_sla']}")
    print(f"{'rw':>4} {'hz':>3} {'gp/$':>10} {'sla':>7} {'gpu_h':>7}  cap_mix | gate(beats/pareto/headline)")
    for r in rows:
        cm = ",".join(f"{k}x{v}" for k, v in sorted(r["capacity_multiplier_mix"].items()))
        print(f"{r['risk_weight']:>4} {r['horizon']:>3} {r['goodput_per_dollar']:>10.0f} "
              f"{r['sla_violation_rate']:>7.4f} {r['gpu_hours']:>7.1f}  {cm} | "
              f"{r['beats_fair']}/{r['pareto_sla_not_worse']}/{r['headline_allowed']}")


if __name__ == "__main__":
    main()
