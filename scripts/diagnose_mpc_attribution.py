#!/usr/bin/env python3
"""OFFLINE MPC attribution + validation + roadmap (diagnostic only; never runs online).

Converts "where is planner value coming from?" from intuition into evidence, using the OFFLINE tier of the
Decision Diagnostics Engine. Three measurements on the bounded Azure window:

1. **Validation** — current full MPC (all action surfaces live) vs SLA-aware / FIFO / greedy baselines.
2. **Forecast attribution (leave-one-out)** — start from the ORACLE planner (every forecast = the realised
   future) and degrade ONE forecast variable back to its model value; the gp/$ drop is that variable's
   planner-value. Only variables the planner CONSUMES are attributed; the rest are reported ABSENT.
3. **Regret decomposition + roadmap** — Current→Scenario→Oracle (the workload-model gap is forecast quality
   by construction); world-model fidelity is flagged UNMEASURABLE in pure simulation; the roadmap is ranked
   directly from the attribution.

No controller / forecaster / simulator changes. Usage: python -m scripts.diagnose_mpc_attribution --mpc-periods 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
from aurelius.environment.decision_diagnostics import (
    ABSENT_FORECASTS,
    CONSUMED_FORECASTS,
    LeaveOneOutAttributor,
    generate_roadmap,
    regret_decomposition,
)
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import build_mpc_inputs, make_world_state, train_forecasters

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_DEGRADABLE = ("arrival_rate", "output_length", "prompt_length", "interarrival_cv")  # electricity ~ const


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _degrade(records, var, med_out, med_prompt, mean_count):
    """Degrade ONE forecast dimension of the oracle records back to the model's (constant) forecast."""
    if var is None:
        return records
    if var == "output_length":
        return [(r[0], med_out, r[2]) for r in records]
    if var == "prompt_length":
        return [(r[0], r[1], med_prompt) for r in records]
    if var == "arrival_rate":
        if len(records) >= mean_count:
            return records[:mean_count]
        pad = [(records[-1][0] if records else 0.0, med_out, med_prompt)] * (mean_count - len(records))
        return records + pad
    if var == "interarrival_cv":                       # regularise timing to uniform inter-arrivals
        n = len(records)
        if n < 2:
            return records
        span = records[-1][0] - records[0][0]
        return [(records[0][0] + i * span / (n - 1), r[1], r[2]) for i, r in enumerate(records)]
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-periods", type=int, default=120)
    ap.add_argument("--mpc-periods", type=int, default=8)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
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
    kv = dict(kv_state_pool=pool, kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    outs = sorted(int(r[1]) for p in ev for r in per.get(p, []))
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in ev for r in per.get(p, []))
    med_out = outs[len(outs) // 2] if outs else 64
    med_prompt = ins[len(ins) // 2] if ins else 512
    mean_count = round(statistics.mean(len(per.get(p, [])) for p in mpc_ev)) or 1

    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    fm, _ = train_forecasters(frames, max(8, n - args.mpc_periods - 8))

    # --- Phase 1: validation — current full MPC vs baselines -----------------
    def _ep(name, decide_fn, periods, **kw):
        ws = make_world_state(common.get("world_state_params"))
        return run_period_episode(name, decide_fn, per, frames, periods, fleet_state=fleet,
                                  cost_model=cm, world_state=ws, **common, **kv, **kw).goodput_per_dollar

    def _mpc_decider(oracle_var=None):
        ws = make_world_state(common.get("world_state_params"))
        ctrl = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        ctrl.horizon_steps = 1
        ctrl.planning_kv_cost_mode = "hybrid_capacity_work"
        ctrl.planning_prompt_tokens = med_prompt
        cur = {"i": 0}

        def _dec(h):
            p = mpc_ev[min(cur["i"], len(mpc_ev) - 1)]
            cur["i"] += 1
            recs = sorted(per.get(p, []), key=lambda r: r[0])
            base = [(r[0], int(r[1]), int(r[2]) if len(r) > 2 else int(r[1])) for r in recs]
            ctrl.planning_oracle_records = _degrade(base, oracle_var, med_out, med_prompt, mean_count) or None
            return ctrl.decide(h).to_dict()
        return _dec, ws

    def fifo(h):
        return {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off"}

    def greedy(h):
        return {"capacity": "backlog_aware", "ordering": "fifo", "admission": "off",
                "routing_policy": "kv_aware", "batching_policy": "aggressive"}

    def sla_aware(h):
        return dict(SLA_AWARE_FALLBACK)
    val = {}
    for nm, fn in [("fifo", fifo), ("greedy", greedy), ("sla_aware", sla_aware)]:
        val[nm] = round(_ep(nm, fn, ev), 1)
    dec_full, ws_full = _mpc_decider(None)
    val["current_mpc_full"] = round(_ep("current_mpc", dec_full, mpc_ev), 1)
    best_base = max(("fifo", "greedy", "sla_aware"), key=lambda k: val[k])
    headline = {"current_mpc": val["current_mpc_full"], "best_baseline": best_base,
                "best_baseline_gpd": val[best_base],
                "delta_pct": round(100.0 * (val["current_mpc_full"] - val[best_base]) / val[best_base], 2)
                if val[best_base] else None}

    # --- Phase 2: forecast attribution (leave-one-out from the oracle) -------
    def evaluate(var):
        dec, _ws = _mpc_decider(var)
        return _ep(f"oracle_deg_{var}", dec, mpc_ev)

    cache = {}

    def _ev_cached(var):
        if var not in cache:
            cache[var] = evaluate(var)
        return cache[var]

    attribution = LeaveOneOutAttributor().attribute(_DEGRADABLE, _ev_cached)
    # electricity_price: in the objective but ~constant over the window → ~0 planning contribution (documented)
    attribution["contributions_pct"].setdefault("electricity_price", 0.0)

    # --- Phase 4: regret decomposition + roadmap (reuse #113 Current/Scenario/Oracle if present) ----
    scen_art = os.path.join(_OUT, "mpc_scenario_forecaster_dt60.json")
    if os.path.exists(scen_art):
        sd = json.load(open(scen_art))["regret_decomposition"]
        cur_gpd, scen_gpd, ora_gpd = sd["current_mpc"], sd["scenario_mpc"], sd["oracle_mpc"]
    else:
        cur_gpd = scen_gpd = ora_gpd = attribution["oracle_gp_per_dollar"]
    regret = regret_decomposition(current_gpd=cur_gpd, scenario_gpd=scen_gpd, oracle_gpd=ora_gpd,
                                  search_regret_frac=0.0, objective_gap_frac=0.0)
    roadmap = generate_roadmap(attribution, regret)

    out = {"eval_periods": len(ev), "mpc_periods": len(mpc_ev), "median_output": med_out,
           "median_prompt": med_prompt, "validation": val, "headline": headline,
           "forecast_attribution": attribution, "consumed_forecasts": list(CONSUMED_FORECASTS),
           "absent_forecasts": ABSENT_FORECASTS, "regret_decomposition": regret, "roadmap": roadmap}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "mpc_attribution.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"VALIDATION: current MPC {headline['current_mpc']:.0f} vs best baseline "
          f"{best_base} {headline['best_baseline_gpd']:.0f}  ({headline['delta_pct']}%)")
    print("FORECAST ATTRIBUTION (leave-one-out, % of forecast planner-value):")
    for v, p in sorted(attribution["contributions_pct"].items(), key=lambda t: -t[1]):
        tag = "ABSENT" if v in ABSENT_FORECASTS else ""
        print(f"  {v:>18}: {p:>5}%  {tag}")
    print(f"REGRET: forecast {regret['forecast_quality_pct']}% | search {regret['search_pct']}% | "
          f"world-model {regret['world_model_fidelity_pct']}")
    print("ROADMAP:", " > ".join(r["improvement"] for r in roadmap[:4]))


if __name__ == "__main__":
    main()
