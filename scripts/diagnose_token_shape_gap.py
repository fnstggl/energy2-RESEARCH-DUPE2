#!/usr/bin/env python3
"""Track C — forecast gap-closure diagnostic: does the token-shape forecaster close the PR #114 gap?

Two measurements on the bounded Azure window, all OFFLINE (diagnostic only — no controller/simulator change):

1. **Four-arm gap-to-oracle.** Plan the SAME real eval workload under four planning workloads and compare
   SLA-safe gp/$ + SLA:
     current   — single synthetic median (PR #112, planning_scenarios off)
     scenario  — parametric 6-scenario ensemble (PR #113 build_scenarios)        [arm A]
     tokenshape— recent-empirical token-shape ensemble (this PR)                  [arm B]
     oracle    — the EXACT realised future                                        [arm C]
   Report each arm's gp/$, SLA, gap to oracle, and the % of the (current→oracle) gap closed.

2. **Leave-one-out attribution BEFORE vs AFTER.** From the ORACLE, degrade one forecast variable back to a
   model forecast and measure the gp/$ drop. BEFORE degrades output/prompt to the GLOBAL median (the PR #114
   measurement); AFTER degrades them to the token-shape forecaster's RECENT-window prediction. arrival/cv
   degrade is held IDENTICAL across before/after by design (they are not this forecaster's target) so the
   output/prompt attribution shift is unconfounded. Success = the forecaster SHRINKS the output_length and
   prompt_length attribution. Not forced — if token shape is stationary, recent≈global and nothing moves
   (reported as such).

Usage: python -m scripts.diagnose_token_shape_gap --mpc-periods 6 --fit-window 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from aurelius.environment.controller import run_period_episode
from aurelius.environment.decision_diagnostics import LeaveOneOutAttributor
from aurelius.environment.token_shape_forecaster import TokenShapeForecaster
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import build_mpc_inputs, make_world_state, train_forecasters

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_DEGRADABLE = ("arrival_rate", "output_length", "prompt_length", "interarrival_cv")


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _degrade(records, var, tgt_out, tgt_prompt, mean_count):
    """Degrade ONE forecast dimension of the oracle records back to a (constant) model forecast."""
    if var is None:
        return records
    if var == "output_length":
        return [(r[0], tgt_out, r[2]) for r in records]
    if var == "prompt_length":
        return [(r[0], r[1], tgt_prompt) for r in records]
    if var == "arrival_rate":
        if len(records) >= mean_count:
            return records[:mean_count]
        pad = [(records[-1][0] if records else 0.0, tgt_out, tgt_prompt)] * (mean_count - len(records))
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
    ap.add_argument("--mpc-periods", type=int, default=6)
    ap.add_argument("--fit-window", type=int, default=8)     # recent periods the token-shape forecaster sees
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--ewma-half-life", type=float, default=0.0)
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

    def _recent(p):
        return [q for q in range(max(8, p - args.fit_window), p) if per.get(q)]

    def _tsf(p):
        return TokenShapeForecaster.fit(per, _recent(p), ewma_half_life=args.ewma_half_life,
                                        period_seconds=common["period_seconds"])

    def _ep(name, decide_fn):
        ws = make_world_state(common.get("world_state_params"))
        return run_period_episode(name, decide_fn, per, frames, mpc_ev, fleet_state=fleet,
                                  cost_model=cm, world_state=ws, **common, **kv)

    def _ctrl():
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        return c

    # --- arm deciders ---------------------------------------------------------
    def _arm(kind):
        c = _ctrl()
        if kind == "scenario":
            c.planning_scenarios = True
        elif kind == "tokenshape":
            c.planning_scenarios = True
        cur = {"i": 0}

        def _dec(h):
            p = mpc_ev[min(cur["i"], len(mpc_ev) - 1)]
            cur["i"] += 1
            recs = sorted(per.get(p, []), key=lambda r: r[0])
            base = [(r[0], int(r[1]), int(r[2]) if len(r) > 2 else int(r[1])) for r in recs]
            if kind == "oracle":
                c.planning_oracle_records = base or None
            elif kind == "tokenshape":
                c.scenario_builder = _tsf(p)
            return c.decide(h).to_dict()
        return _dec

    arms = {}
    for kind in ("current", "scenario", "tokenshape", "oracle"):
        arms[kind] = round(_ep(kind, _arm(kind)).goodput_per_dollar, 1)
    oracle_gpd = arms["oracle"]
    cur_gpd = arms["current"]

    def _closed(arm_gpd):
        gap_cur = oracle_gpd - cur_gpd
        return round(100.0 * (arm_gpd - cur_gpd) / gap_cur, 1) if gap_cur else None
    gap = {k: {"gp_per_dollar": arms[k], "gap_to_oracle": round(oracle_gpd - arms[k], 1),
               "pct_of_current_oracle_gap_closed": _closed(arms[k])} for k in arms}

    # --- attribution before (global median) vs after (token-shape recent prediction) -----------
    def _attrib(mode):
        def _deg_decider(var):
            c = _ctrl()
            cur = {"i": 0}

            def _dec(h):
                p = mpc_ev[min(cur["i"], len(mpc_ev) - 1)]
                cur["i"] += 1
                recs = sorted(per.get(p, []), key=lambda r: r[0])
                base = [(r[0], int(r[1]), int(r[2]) if len(r) > 2 else int(r[1])) for r in recs]
                t_out, t_prompt = med_out, med_prompt
                if mode == "after" and var in ("output_length", "prompt_length"):
                    q = _tsf(p).q
                    t_out, t_prompt = round(q.out_p50), round(q.prompt_p50)
                c.planning_oracle_records = _degrade(base, var, t_out, t_prompt, mean_count) or None
                return c.decide(h).to_dict()
            return _dec

        cache = {}

        def _evc(var):
            if var not in cache:
                cache[var] = _ep(f"{mode}_{var}", _deg_decider(var)).goodput_per_dollar
            return cache[var]
        a = LeaveOneOutAttributor().attribute(_DEGRADABLE, _evc)
        a["contributions_pct"].setdefault("electricity_price", 0.0)
        return a

    before, after = _attrib("before"), _attrib("after")

    out = {"mpc_periods": len(mpc_ev), "fit_window": args.fit_window, "ewma_half_life": args.ewma_half_life,
           "median_output": med_out, "median_prompt": med_prompt, "arms": gap,
           "attribution_before": before, "attribution_after": after,
           "shift": {v: round(after["contributions_pct"].get(v, 0.0) - before["contributions_pct"].get(v, 0.0), 1)
                     for v in ("output_length", "prompt_length", "interarrival_cv", "arrival_rate")},
           "raw_drop_before": before.get("raw_drops", {}), "raw_drop_after": after.get("raw_drops", {})}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "token_shape_gap.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("ARMS (SLA-safe gp/$, gap to oracle, % of current→oracle gap closed):")
    for k in ("current", "scenario", "tokenshape", "oracle"):
        g = gap[k]
        print(f"  {k:>11}: {g['gp_per_dollar']:>9.0f}  gap={g['gap_to_oracle']:>8.0f}  "
              f"closed={g['pct_of_current_oracle_gap_closed']}%")
    print("ATTRIBUTION output_length / prompt_length  (before → after):")
    for v in ("output_length", "prompt_length", "interarrival_cv", "arrival_rate"):
        print(f"  {v:>16}: {before['contributions_pct'].get(v, 0):>5}% → "
              f"{after['contributions_pct'].get(v, 0):>5}%  (Δ {out['shift'][v]:+})")
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
