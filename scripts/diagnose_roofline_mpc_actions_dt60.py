#!/usr/bin/env python3
"""dt=60 diagnostic for the roofline-economic MPC actions (Phase 11).

Two halves, both on the deterministic 6-hour Azure window with the Mooncake-derived prefix stream, the
same hybrid cost mode, and the same Pareto gate:

1. **Static action ladder (A→F).** Force each roofline action on top of the previous, so each row isolates
   the CAUSAL effect of adding precision / roofline batching / speculative decoding / clock / co-location.
   A is the pre-roofline bundle (the fair baseline). A headline for any stack requires the Pareto gate to
   pass vs A (beats gp/$ AND SLA not worse).

2. **Live MPC selection.** The adaptive planner (beam + regret audit) chooses the roofline bundle per
   period. We report what it selected, whether it beat the BEST static stack (the strong baseline — so the
   adaptive controller cannot win by beating a strawman), and the per-decision regret.

Every effect flows through the roofline physics (TTFT / GPU-seconds / energy / SLA / cost) — no bonuses.
co-location is forced ON in stack F only to SHOW it hurts without a background-work trace (it is frozen
off in the live planner). int4 is never stacked (quality-unsafe without a quality model).

Usage: python -m scripts.diagnose_roofline_mpc_actions_dt60 --eval-periods 120 --mpc-periods 60
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
# cumulative roofline ladder (each row adds ONE action on top of the previous).
_STACKS = [
    ("A_pre_roofline", {}),
    ("B_precision_fp8", {"precision_policy": "fp8"}),
    ("C_plus_roofline_batching", {"precision_policy": "fp8", "batching_policy": "aggressive"}),
    ("D_plus_spec_decode", {"precision_policy": "fp8", "batching_policy": "aggressive",
                            "spec_decode_policy": "medium"}),
    ("E_plus_clock_low", {"precision_policy": "fp8", "batching_policy": "aggressive",
                          "spec_decode_policy": "medium", "clock_policy": "low"}),
    ("F_plus_colocation", {"precision_policy": "fp8", "batching_policy": "aggressive",
                           "spec_decode_policy": "medium", "clock_policy": "low",
                           "colocation_policy": "conservative"}),
]


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _static(action):
    return (lambda a: (lambda h: dict(a)))(action)


def _row(rep):
    return {"goodput_per_dollar": round(rep.goodput_per_dollar, 1),
            "sla_violation_rate": round(rep.sla_violation_rate, 4),
            "mean_ttft_p95": round(rep.mean_ttft_p95, 4),
            "queue_delay_p95": round(rep.queue_delay_p95, 3), "queue_delay_p99": round(rep.queue_delay_p99, 3),
            "gpu_hours": round(rep.gpu_hours, 2), "realized_gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "total_energy_j": round(rep.total_energy_j, 0), "mean_power_w": round(rep.mean_power_w, 1),
            "decode_regime_mix": rep.decode_regime_mix,
            "mean_decode_arithmetic_intensity": rep.mean_decode_arithmetic_intensity,
            "mean_ridge_point": rep.mean_ridge_point, "quality_sla_risk_mean": rep.quality_sla_risk_mean,
            "precision_mix": rep.precision_mix, "spec_decode_mix": rep.spec_decode_mix,
            "clock_mix": rep.clock_mix, "colocation_mix": rep.colocation_mix,
            "batching_mix": rep.batching_mix}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-periods", type=int, default=120)
    ap.add_argument("--mpc-periods", type=int, default=60)
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
    pool = _mooncake_pool(args.mooncake_limit)
    if not pool:
        raise SystemExit("no Mooncake prefix pool available")
    kv = dict(kv_state_pool=pool, kv_capacity_blocks=args.capacity_blocks, kv_cost_mode="hybrid_capacity_work")

    # --- 1) static roofline ladder A→F -------------------------------------
    rows = {}
    for name, extra in _STACKS:
        ws = make_world_state(common.get("world_state_params"))
        rep = run_period_episode(name, _static({**_BASE, **extra}), per, frames, ev, fleet_state=fleet,
                                 cost_model=cm, world_state=ws, **common, **kv)
        rows[name] = {"_rep": rep, **_row(rep)}
    fair = rows["A_pre_roofline"]["_rep"]
    ladder_gates = {}
    for name in [s[0] for s in _STACKS[1:]]:
        g = claim_gate({"mpc_controller": rows[name]["_rep"], "fair": fair})
        ladder_gates[name] = {"beats_fair": g["beats_fair_baseline"],
                              "pareto_sla_not_worse": g["pareto_sla_not_worse"],
                              "headline_allowed": g["headline_claim_allowed"],
                              "delta_pct": g["candidate_vs_baseline_pct"]}
    # the strongest STATIC stack that is Pareto-safe vs A (the honest strong baseline for the MPC arm)
    safe = [(nm, rows[nm]["goodput_per_dollar"]) for nm, gg in ladder_gates.items()
            if gg["headline_allowed"]]
    best_static = max(safe, key=lambda t: t[1])[0] if safe else "A_pre_roofline"

    # --- 2) live MPC selection (adaptive planner + regret) -----------------
    mpc_ev = list(range(max(8, n - args.mpc_periods), n))
    train_cut = max(8, n - args.mpc_periods - 8)
    fm, _ = train_forecasters(frames, train_cut)
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ws = make_world_state(common.get("world_state_params"))
    ctrl = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
    ctrl.horizon_steps = 1                                  # bounded: 1-step rollout for the diagnostic
    last_plan = {}

    def _decider(h):
        d = ctrl.decide(h)
        nonlocal last_plan
        sp = (ctrl.last_decision_diag or {}).get("search_plan")
        if sp:
            last_plan = sp
        return d.to_dict()

    ws_eval = make_world_state(common.get("world_state_params"))
    mpc_rep = run_period_episode("mpc_roofline", _decider, per, frames, mpc_ev, fleet_state=fleet,
                                 cost_model=cm, world_state=ws_eval, **common, **kv)
    # the MPC arm is judged against the BEST STATIC stack (strong baseline) AND against A.
    mpc_vs_best = claim_gate({"mpc_controller": mpc_rep, "fair": rows[best_static]["_rep"]})
    mpc_vs_a = claim_gate({"mpc_controller": mpc_rep, "fair": fair})

    for r in rows.values():
        r.pop("_rep", None)
    out = {"control_dt_seconds": 60.0, "eval_periods": len(ev), "mpc_periods": len(mpc_ev),
           "mooncake_pool_size": len(pool), "cost_mode": "hybrid_capacity_work",
           "ladder": rows, "ladder_gates": ladder_gates, "best_static_stack": best_static,
           "mpc_arm": {**_row(mpc_rep),
                       "vs_best_static": {"baseline": best_static,
                                          "beats": mpc_vs_best["beats_fair_baseline"],
                                          "pareto_sla_not_worse": mpc_vs_best["pareto_sla_not_worse"],
                                          "headline_allowed": mpc_vs_best["headline_claim_allowed"],
                                          "delta_pct": mpc_vs_best["candidate_vs_baseline_pct"]},
                       "vs_pre_roofline": {"beats": mpc_vs_a["beats_fair_baseline"],
                                           "pareto_sla_not_worse": mpc_vs_a["pareto_sla_not_worse"],
                                           "delta_pct": mpc_vs_a["candidate_vs_baseline_pct"]},
                       "last_search_plan": last_plan},
           "claim_safety": {"ladder": "trace_derived_workload + simulator_inferred_mechanism_magnitudes",
                            "headline": "only if Pareto gate passes vs the strong (best-static) baseline",
                            "int4": "never stacked (quality-unsafe without a quality model)",
                            "colocation": "stack F only, to SHOW it hurts with no background-work trace"}}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "roofline_mpc_actions_dt60.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"dt=60, eval={len(ev)} periods, pool={len(pool)}, cost=hybrid")
    print(f"{'stack':>26} {'gp/$':>9} {'sla':>7} {'ttft95':>7} {'realGPUs':>9} {'energy':>9} {'gate(beat/pareto/head)':>22}")
    for name, _e in _STACKS:
        r = rows[name]
        g = ladder_gates.get(name)
        gate = f"{g['beats_fair']}/{g['pareto_sla_not_worse']}/{g['headline_allowed']}" if g else "(fair baseline)"
        print(f"{name:>26} {r['goodput_per_dollar']:>9.0f} {r['sla_violation_rate']:>7.4f} "
              f"{r['mean_ttft_p95']:>7.3f} {r['realized_gpu_seconds']:>9.0f} {r['total_energy_j']:>9.0f} {gate:>22}")
    m = out["mpc_arm"]
    print(f"\nMPC selection (best_static={best_static}): gp/$={m['goodput_per_dollar']:.0f} "
          f"sla={m['sla_violation_rate']:.4f} precision={m['precision_mix']} spec={m['spec_decode_mix']} "
          f"clock={m['clock_mix']}")
    print(f"  vs best_static: beats={m['vs_best_static']['beats']} "
          f"pareto={m['vs_best_static']['pareto_sla_not_worse']} headline={m['vs_best_static']['headline_allowed']} "
          f"({m['vs_best_static']['delta_pct']}%)")
    if last_plan:
        print(f"  last search_plan: strategy={last_plan.get('strategy')} raw={last_plan.get('raw_candidate_count')} "
              f"eval={last_plan.get('candidates_evaluated')} regret={last_plan.get('estimated_regret')}")


if __name__ == "__main__":
    main()
