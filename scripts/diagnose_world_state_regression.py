#!/usr/bin/env python3
"""Phase-1 diagnostic: per-period evidence for the persistent-world-state MPC regression.

Trains the world-state MPC exactly as the eval does, then walks the held-out periods logging, for
each one: the chosen ActionBundle, the controller's PREDICTED risk-adjusted score for that bundle,
the predicted score of the capacity_multiplier={0.75,1.0,1.5} alternatives (everything else fixed)
so we can SEE why 1.5x wins, and the REALIZED per-period metrics (gp/$, GPU-hours, cold-starts,
warm-hold, queue p95, SLA). Writes a JSON + a table. Pure diagnosis — changes no model.

Usage: python -m scripts.diagnose_world_state_regression --json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from aurelius.benchmarks.srtf_serving_backtest import _service_time_s
from aurelius.environment.actions import ActionBundle
from aurelius.environment.controller import ModelPredictiveEconomicController
from aurelius.environment.training import build_mpc_inputs, make_world_state, train_mpc_policy
from aurelius.environment.world_simulator import simulate_period

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")


def _score_capacity_options(ctrl, ws, bundle, ar, tm, tp, cv, win, be):
    """Re-score the chosen bundle at each capacity_multiplier under the CURRENT world state, the
    same way the controller does (point + p90-risk synthetic load), returning the risk-adjusted
    score + its components. Shows WHY a given capacity level wins."""
    from aurelius.environment.controller import _synth_jobs
    by_routing = ctrl.kv_service_factor_by_routing or {}
    factor = by_routing.get(bundle.routing_policy, ctrl.kv_service_factor)
    fc = {"arrival_rate": ar.mean, "arrival_p90": ar.p90,
          "mean_service_s": max(_service_time_s(int(tm.mean)), 1e-3)}
    out = {}
    for mult in (0.75, 1.0, 1.5):
        b = bundle.with_overrides(capacity_multiplier=mult)
        pj = _synth_jobs(ar.mean, tm.mean, tp.value, cv.mean, window_seconds=win,
                         best_effort_fraction=be, kv_service_factor=1.0)
        rj = _synth_jobs(ar.p90, tm.p90, tp.p99, cv.p90, window_seconds=win,
                         best_effort_fraction=be, kv_service_factor=1.0)
        common = dict(sla_s=ctrl.sla_s, tick_seconds=ctrl.tick_seconds, base_service_factor=factor,
                      replay_kwargs=b.replay_kwargs(), cost_model=ctrl.cost_model,
                      fleet_state=ctrl.fleet_state, best_effort_fraction=be,
                      period_hours=max(win, 1.0) / 3600.0)
        pe = simulate_period(ws, b, [(j.arrival_s, j.actual_tokens, j.actual_tokens) for j in pj],
                             fc, mutate=False, **common)
        re = simulate_period(ws, b, [(j.arrival_s, j.actual_tokens, j.actual_tokens) for j in rj],
                             fc, mutate=False, **common)
        exp_gpd, risk_viol = pe.goodput_per_dollar, re.sla_violation_rate
        score = exp_gpd - ctrl.risk_weight * risk_viol * exp_gpd
        out[str(mult)] = {"score": round(score, 1), "exp_gpd": round(exp_gpd, 1),
                     "risk_viol": round(risk_viol, 4), "point_sla": round(pe.sla_violation_rate, 4),
                     "point_cold": pe.cold_start_events, "risk_cold": re.cold_start_events}
    return out


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
    common, fleet, cm = inp["common"], inp["fleet_state"], inp["cost_model"]
    trained, fm = train_mpc_policy(inp["frames"], inp["per"], fleet_state=fleet, cost_model=cm,
                                   common=common)
    cfg = trained["controller_config"]
    ws = make_world_state(common.get("world_state_params"))
    ctrl = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=fleet, cost_model=cm, horizon=cfg["horizon"],
        risk_weight=cfg["risk_weight"], confidence_min=cfg["confidence_min"], sla_s=common["sla_s"],
        period_seconds=common["period_seconds"], tick_seconds=common["tick_seconds"],
        kv_service_factor=common.get("kv_service_factor", 1.0),
        kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
        cost_scenario=common.get("cost_scenario", "owned"), sim_seconds=common.get("sim_seconds"),
        world_state=ws)
    e0, e1 = trained["splits"]["eval"]
    win = common.get("sim_seconds") or common["period_seconds"]
    be = fleet.best_effort_fraction
    period_hours = max(common["period_seconds"], 1.0) / 3600.0

    rows = []
    for p in range(e0, e1):
        d = ctrl.decide(inp["frames"][:p])
        b = d.bundle if d.bundle is not None else ActionBundle()
        fb = fm.predict(inp["frames"][:p], horizon=cfg["horizon"])
        ar, tm, tp, cv = (fb.at(t, 0) for t in ("arrival_rate", "output_token_mean",
                                                "output_token_p95", "interarrival_cv"))
        cap_scores = _score_capacity_options(ctrl, ws, b, ar, tm, tp, cv, win, be)
        # realized: apply the chosen bundle to the REAL period load + advance the state.
        recs = sorted(inp["per"].get(p, []), key=lambda r: r[0])
        realized = None
        if recs:
            factor = (ctrl.kv_service_factor_by_routing or {}).get(b.routing_policy, ctrl.kv_service_factor)
            t0 = recs[0][0]
            prev = inp["per"].get(p - 1, recs)
            fc = {"arrival_rate": len(prev) / max(common["period_seconds"], 1e-9),
                  "arrival_p90": 1.3 * len(prev) / max(common["period_seconds"], 1e-9),
                  "mean_service_s": (statistics.mean(_service_time_s(int(r[1])) for r in prev) if prev else 1.0)}
            oc = simulate_period(ws, b, [(r[0] - t0, int(r[1]), r[2] if len(r) > 2 else r[1]) for r in recs],
                                 fc, sla_s=common["sla_s"], tick_seconds=common["tick_seconds"],
                                 base_service_factor=factor, replay_kwargs=b.replay_kwargs(),
                                 cost_model=cm, fleet_state=fleet, best_effort_fraction=be,
                                 period_hours=period_hours, mutate=True)
            realized = {"gp_per_dollar": round(oc.goodput_per_dollar, 1),
                        "sla_viol": round(oc.sla_violation_rate, 4), "gpu_hours": round(oc.kpi.gpu_hours, 3),
                        "cold_start_events": oc.cold_start_events, "warm_hold_h": round(oc.wasted_prewarm_hours, 2),
                        "queue_p95": round(oc.queue_delay_p95, 2), "warm_capacity": oc.warm_capacity,
                        "peak_c": oc.kpi.c_max, "migration_cost": oc.migration_cost,
                        "topology_factor": oc.topology_factor}
        rows.append({"period": p, "chosen": {"capacity_multiplier": b.capacity_multiplier,
                     "batching_policy": b.batching_policy, "routing_policy": b.routing_policy,
                     "prewarm_policy": b.prewarm_policy, "placement_policy": b.placement_policy,
                     "migration_policy": b.migration_policy},
                     "predicted_score": round(d.score, 1), "capacity_option_scores": cap_scores,
                     "realized": realized})

    out = {"controller_config": cfg, "risk_weight": cfg["risk_weight"], "eval_periods": [e0, e1],
           "rows": rows}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "world_state_regression_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"config {cfg}; per-period (chosen cap_mult | predicted scores 0.75/1.0/1.5 | realized gp/$ sla cold)")
    for r in rows:
        cs = r["capacity_option_scores"]
        rz = r["realized"] or {}
        print(f"  p{r['period']:>3} cap={r['chosen']['capacity_multiplier']} "
              f"prewarm={r['chosen']['prewarm_policy'][:4]} place={r['chosen']['placement_policy'][:7]} | "
              f"score .75={cs['0.75']['score']:>9} 1.0={cs['1.0']['score']:>9} 1.5={cs['1.5']['score']:>9} "
              f"| risk_viol .75={cs['0.75']['risk_viol']:.2f} 1.0={cs['1.0']['risk_viol']:.2f} 1.5={cs['1.5']['risk_viol']:.2f} "
              f"| real gp/$={rz.get('gp_per_dollar',0):>9} sla={rz.get('sla_viol',0):.3f} cold={rz.get('cold_start_events',0)}")


if __name__ == "__main__":
    main()
