#!/usr/bin/env python3
"""dt=60 scenario-forecaster + oracle diagnostic (small PR).

PR #112 showed the live MPC's remaining error is the planner's predictive WORKLOAD model: planning against a
single synthetic median under-represents SLA pressure, so the planner under-values speculative decoding. This
compares four planners on the same bounded 6-hour Azure window + Mooncake prefixes + hybrid cost + Pareto
gate, holding planning/eval cost-parity ON throughout (so only the WORKLOAD model varies):

  Current   MPC plans against the single median workload (PR #112)
  Scenario  MPC plans across a small trace-derived ensemble incl. SLA-pressure futures (this PR)
  Oracle    MPC plans against the EXACT realized future workload (a labelled diagnostic — uses future info)
  Static    the best static stack (fp8 + spec) — the optimum target

and reports the REGRET DECOMPOSITION  Static → Oracle → Scenario → Current:
  Static − Oracle    = the gap perfect forecasting CANNOT close (search / physics / causal-MPC structure)
  Oracle − Scenario  = the residual forecast-fidelity gap (imperfect ensemble)
  Scenario − Current = the improvement the ensemble buys over the median

Usage: python -m scripts.diagnose_mpc_scenario_forecaster_dt60 --mpc-periods 16
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.controller import run_period_episode
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import (
    build_mpc_inputs,
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


def _median_prompt(per, ev):
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in ev for r in per.get(p, []))
    return ins[len(ins) // 2] if ins else 512


def _row(rep):
    return {"gp_per_dollar": round(rep.goodput_per_dollar, 1), "sla": round(rep.sla_violation_rate, 4),
            "ttft_p95": round(rep.mean_ttft_p95, 4), "realized_gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "precision_mix": rep.precision_mix, "spec_mix": rep.spec_decode_mix, "clock_mix": rep.clock_mix}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-periods", type=int, default=120)
    ap.add_argument("--mpc-periods", type=int, default=16)
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

    # static optimum (fp8 + spec)
    ws = make_world_state(common.get("world_state_params"))
    d_rep = run_period_episode("static_fp8_spec",
                               (lambda a: (lambda h: dict(a)))({**_BASE, "precision_policy": "fp8",
                                "batching_policy": "aggressive", "spec_decode_policy": "medium"}),
                               per, frames, ev, fleet_state=fleet, cost_model=cm,
                               world_state=ws, **common, **kv)
    rows = {"Static_fp8_spec": _row(d_rep)}

    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    fm, _ = train_forecasters(frames, max(8, n - args.mpc_periods - 8))

    def _run(name, *, scenarios, oracle):
        ws_plan = make_world_state(common.get("world_state_params"))
        ctrl = build_controller(fm, fleet, cm, cfg, common, world_state=ws_plan)
        ctrl.horizon_steps = 1
        ctrl.planning_kv_cost_mode = "hybrid_capacity_work"     # cost-parity ON for all arms (PR #112)
        ctrl.planning_prompt_tokens = med_prompt
        ctrl.planning_scenarios = scenarios
        cursor = {"i": 0}

        def _dec(h):
            if oracle:
                p = mpc_ev[min(cursor["i"], len(mpc_ev) - 1)]
                cursor["i"] += 1
                recs = sorted(per.get(p, []), key=lambda r: r[0])
                ctrl.planning_oracle_records = [(r[0], int(r[1]), int(r[2]) if len(r) > 2 else int(r[1]))
                                                for r in recs] or None
            return ctrl.decide(h).to_dict()

        ws_eval = make_world_state(common.get("world_state_params"))
        rep = run_period_episode(name, _dec, per, frames, mpc_ev, fleet_state=fleet, cost_model=cm,
                                 world_state=ws_eval, **common, **kv)
        rows[name] = _row(rep)
        return rep.goodput_per_dollar

    cur = _run("Current_median", scenarios=False, oracle=False)
    sce = _run("Scenario_ensemble", scenarios=True, oracle=False)
    ora = _run("Oracle_exact_future", scenarios=False, oracle=True)
    static = d_rep.goodput_per_dollar

    scen_gain = sce - cur
    decomp = {
        "static_optimum": round(static, 1), "oracle_mpc": round(ora, 1),
        "scenario_mpc": round(sce, 1), "current_mpc": round(cur, 1),
        # workload-model improvement (median → SLA-pressure ensemble) — the lever this PR adds:
        "scenario_minus_current": round(scen_gain, 1),
        # residual a PERFECT forecast would add beyond the ensemble (ensemble → exact future):
        "oracle_minus_scenario": round(ora - sce, 1),
        # per-period adaptation value vs the best FIXED bundle (adaptive arms can EXCEED the static optimum):
        "oracle_minus_static": round(ora - static, 1), "scenario_minus_static": round(sce - static, 1),
        "adaptive_exceeds_static": bool(sce > static or ora > static),
        # what fraction of the ensemble's gain a perfect forecast would add on top (small ⇒ forecasting nearly closed):
        "forecast_residual_frac_of_scenario_gain": (round((ora - sce) / scen_gain, 3)
                                                    if abs(scen_gain) > 1e-6 else None),
    }
    out = {"control_dt_seconds": 60.0, "eval_periods": len(ev), "mpc_periods": len(mpc_ev),
           "median_prompt_tokens": med_prompt, "cost_mode": "hybrid_capacity_work", "rows": rows,
           "regret_decomposition": decomp,
           "claim_safety": {"oracle": "uses FUTURE info (labelled diagnostic) — not a deployable policy",
                            "scenarios": "trace-derived statistical extrapolation; risk-averse prior weights",
                            "magnitudes": "simulator_inferred on a bounded window"}}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "mpc_scenario_forecaster_dt60.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"dt=60 eval={len(ev)} mpc={len(mpc_ev)} median_prompt={med_prompt}")
    print(f"{'arm':>20} {'gp/$':>9} {'sla':>7} {'precision':>22} {'spec':>26}")
    for name in ("Static_fp8_spec", "Oracle_exact_future", "Scenario_ensemble", "Current_median"):
        r = rows[name]
        print(f"{name:>20} {r['gp_per_dollar']:>9.0f} {r['sla']:>7.4f} {str(r['precision_mix']):>22} {str(r['spec_mix']):>26}")
    print("\nREGRET DECOMPOSITION (gp/$):  Current → Scenario → Oracle  (Static = best FIXED bundle)")
    print(f"  Current {decomp['current_mpc']:.0f} → Scenario {decomp['scenario_mpc']:.0f} "
          f"→ Oracle {decomp['oracle_mpc']:.0f}   |   Static {decomp['static_optimum']:.0f}")
    print(f"  workload-model gain (Scenario−Current) = {decomp['scenario_minus_current']:.0f}")
    print(f"  forecast residual (Oracle−Scenario) = {decomp['oracle_minus_scenario']:.0f} "
          f"({decomp['forecast_residual_frac_of_scenario_gain']}× of the gain)")
    print(f"  per-period adaptation vs static (Scenario−Static) = {decomp['scenario_minus_static']:.0f} "
          f"| adaptive_exceeds_static={decomp['adaptive_exceeds_static']}")


if __name__ == "__main__":
    main()
