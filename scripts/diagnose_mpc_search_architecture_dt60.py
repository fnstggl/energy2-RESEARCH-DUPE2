#!/usr/bin/env python3
"""dt=60 rerun for the MPC search-architecture + planning/eval fidelity fix (Phase 11).

PR #111 found the live MPC under-selected FP8 because the planning rollout was blind to the phase/cost/
roofline economics the evaluator uses. This compares, on the same bounded 6-hour Azure window + Mooncake
prefix stream + hybrid cost mode + Pareto gate:

  A  live MPC, planning OFF   (reproduces PR #111: planning sees only latency → picks bf16)
  B  live MPC, planning ON    (this PR: planning runs the same phase+cost+roofline model → should pick fp8)
  C  static fp8
  D  static fp8 + spec        (the PR #111 best static stack)

Headline question: does B recover most of D's value while staying Pareto-safe? Reported, never forced —
if not, the residual gap (search regret / cost-mode / physical) is named.

Usage: python -m scripts.diagnose_mpc_search_architecture_dt60 --eval-periods 120 --mpc-periods 40
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.controller import run_period_episode
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import (
    build_mpc_inputs,
    claim_gate,
    make_world_state,
    train_forecasters,
)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_BASE = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off",
         "batching_policy": "balanced", "routing_policy": "kv_aware",
         "precision_policy": "bf16", "spec_decode_policy": "off", "clock_policy": "base"}


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _static(action):
    d = dict(action)
    return lambda h: dict(d)


def _row(rep):
    return {"gp_per_dollar": round(rep.goodput_per_dollar, 1), "sla": round(rep.sla_violation_rate, 4),
            "ttft_p95": round(rep.mean_ttft_p95, 4), "queue_p99": round(rep.queue_delay_p99, 3),
            "gpu_hours": round(rep.gpu_hours, 2), "realized_gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "energy_j": round(rep.total_energy_j, 0), "precision_mix": rep.precision_mix,
            "spec_mix": rep.spec_decode_mix, "clock_mix": rep.clock_mix, "batching_mix": rep.batching_mix,
            "routing_mix": rep.routing_mix, "regime_mix": rep.decode_regime_mix}


def _median_prompt(per, ev):
    ins = []
    for p in ev:
        for r in per.get(p, []):
            ins.append(int(r[2]) if len(r) > 2 else int(r[1]))
    ins.sort()
    return ins[len(ins) // 2] if ins else 512


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-periods", type=int, default=120)
    ap.add_argument("--mpc-periods", type=int, default=40)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--capacity-blocks", type=int, default=256)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=180.0, use_world_state=True,
                           control_dt_seconds=60.0)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames, per = (inp["common"], inp["fleet_state"], inp["cost_model"],
                                      inp["frames"], inp["per"])
    n = len(frames)
    ev = list(range(max(8, n - args.eval_periods), n))
    mpc_ev = list(range(max(8, n - args.mpc_periods), n))
    pool = _mooncake_pool(args.mooncake_limit)
    if not pool:
        raise SystemExit("no Mooncake prefix pool available")
    kv = dict(kv_state_pool=pool, kv_capacity_blocks=args.capacity_blocks, kv_cost_mode="hybrid_capacity_work")
    med_prompt = _median_prompt(per, ev)

    rows = {}
    # static reference stacks
    for name, extra in [("A_pre_roofline", {}), ("C_static_fp8", {"precision_policy": "fp8"}),
                        ("D_static_fp8_spec", {"precision_policy": "fp8", "batching_policy": "aggressive",
                                               "spec_decode_policy": "medium"})]:
        ws = make_world_state(common.get("world_state_params"))
        rep = run_period_episode(name, _static({**_BASE, **extra}), per, frames, ev, fleet_state=fleet,
                                 cost_model=cm, world_state=ws, **common, **kv)
        rows[name] = {"_rep": rep, **_row(rep)}

    # live MPC arms: planning OFF (A) vs planning ON (B)
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    train_cut = max(8, n - args.mpc_periods - 8)
    fm, _ = train_forecasters(frames, train_cut)
    for name, plan_mode in [("MPC_A_planning_off", None), ("MPC_B_planning_on", "hybrid_capacity_work")]:
        ws_plan = make_world_state(common.get("world_state_params"))
        ctrl = build_controller(fm, fleet, cm, cfg, common, world_state=ws_plan)
        ctrl.horizon_steps = 1
        ctrl.planning_kv_cost_mode = plan_mode
        ctrl.planning_prompt_tokens = med_prompt if plan_mode else None
        plans = []

        def _dec(h, _c=ctrl, _p=plans):
            d = _c.decide(h)
            sp = (_c.last_decision_diag or {}).get("search_plan")
            if sp:
                _p.append(sp)
            return d.to_dict()

        ws_eval = make_world_state(common.get("world_state_params"))
        rep = run_period_episode(name, _dec, per, frames, mpc_ev, fleet_state=fleet, cost_model=cm,
                                 world_state=ws_eval, **common, **kv)
        strat = {}
        ev_counts = []
        for sp in plans:
            strat[sp.get("strategy")] = strat.get(sp.get("strategy"), 0) + 1
            ev_counts.append(sp.get("candidates_evaluated", 0))
        rows[name] = {"_rep": rep, **_row(rep),
                      "search_strategy_mix": strat,
                      "mean_candidates_evaluated": round(sum(ev_counts) / len(ev_counts), 1) if ev_counts else 0,
                      "raw_candidate_count": plans[-1].get("raw_candidate_count") if plans else None,
                      "planning_prompt_tokens": ctrl.planning_prompt_tokens}

    # gates + deltas (fair baseline A; strong baseline D)
    fair = rows["A_pre_roofline"]["_rep"]
    d_gpd = rows["D_static_fp8_spec"]["gp_per_dollar"]
    a_gpd = rows["MPC_A_planning_off"]["gp_per_dollar"]
    out_rows = {}
    for name, r in rows.items():
        g = claim_gate({"mpc_controller": r["_rep"], "fair": fair})
        out_rows[name] = {k: v for k, v in r.items() if k != "_rep"}
        out_rows[name]["vs_pre_roofline_pct"] = g["candidate_vs_baseline_pct"]
        out_rows[name]["pareto_sla_not_worse"] = g["pareto_sla_not_worse"]
        out_rows[name]["headline_allowed"] = g["headline_claim_allowed"]
        out_rows[name]["pct_of_static_fp8_spec"] = round(100.0 * r["gp_per_dollar"] / d_gpd, 1) if d_gpd else None
    b_gpd = rows["MPC_B_planning_on"]["gp_per_dollar"]
    summary = {"median_prompt_tokens": med_prompt, "static_fp8_spec_gpd": d_gpd,
               "mpc_planning_off_gpd": a_gpd, "mpc_planning_on_gpd": b_gpd,
               "B_minus_A_pct": round(100.0 * (b_gpd - a_gpd) / a_gpd, 2) if a_gpd else None,
               "B_recovers_pct_of_static": round(100.0 * b_gpd / d_gpd, 1) if d_gpd else None,
               "B_precision_selected": rows["MPC_B_planning_on"]["precision_mix"],
               "A_precision_selected": rows["MPC_A_planning_off"]["precision_mix"]}

    out = {"control_dt_seconds": 60.0, "eval_periods": len(ev), "mpc_periods": len(mpc_ev),
           "mooncake_pool_size": len(pool), "cost_mode": "hybrid_capacity_work",
           "rows": out_rows, "summary": summary,
           "claim_safety": {"headline": "only if Pareto gate passes vs fair baseline (A)",
                            "planning_parity": "synthetic unique-prefix kv_state + hybrid cost (no invented reuse)",
                            "magnitudes": "simulator_inferred"}}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "mpc_search_architecture_dt60.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"dt=60 eval={len(ev)} mpc={len(mpc_ev)} pool={len(pool)} median_prompt={med_prompt}")
    print(f"{'arm':>22} {'gp/$':>9} {'sla':>7} {'%ofD':>6} {'precision':>22} {'pareto/head':>12}")
    for name in ("A_pre_roofline", "C_static_fp8", "D_static_fp8_spec", "MPC_A_planning_off", "MPC_B_planning_on"):
        r = out_rows[name]
        print(f"{name:>22} {r['gp_per_dollar']:>9.0f} {r['sla']:>7.4f} {str(r['pct_of_static_fp8_spec']):>6} "
              f"{str(r['precision_mix']):>22} {str(r['pareto_sla_not_worse'])[:1]}/{str(r['headline_allowed'])[:1]:>10}")
    print(f"\nB (planning ON) vs A (planning OFF): {summary['B_minus_A_pct']}%  |  "
          f"B recovers {summary['B_recovers_pct_of_static']}% of static fp8+spec")
    print(f"A precision={summary['A_precision_selected']}  ->  B precision={summary['B_precision_selected']}")


if __name__ == "__main__":
    main()
