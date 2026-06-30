#!/usr/bin/env python3
"""Batch-1 action-knob ablation ladder (Phase 7) — what each new knob contributes on the frozen Benchmark v1.

Runs the frozen request cap from Phase 0 (default 120 = uncapped-equivalent) on one market+window through the
SAME reward path as the headline benchmark, with the hierarchical_search planner under eight ablation masks:

  1 baseline            (no new knobs)            5 kv + gpu
  2 + kv-cache precision                          6 kv + pd
  3 + heterogeneous gpu assignment                7 gpu + pd
  4 + prefill/decode disaggregation               8 all three

plus the comparison arms: production_scheduler, sla_aware, the prior Aurelius benchmark default
(aurelius_mpc_current_default), and oracle_diagnostic. For each arm it reports gp/$, the absolute + percent
gp/$ delta vs the baseline ablation arm AND vs production_scheduler, the SLA violation rate, the Pareto
pass/fail (gp/$ up AND SLA not worse), runtime, candidates generated/evaluated, and which new knobs the
planner actually SELECTED (the kv / pd / assignment mix). gpu_assignment is NOT_APPLICABLE on the production
fleet (single dominant GPU type), so its ablation arms are reported as such — never a fake gain.

Usage: python -m scripts.run_batch1_ablation [--cap 120] [--market pjm] [--max-decisions 3] [--win-len 4]
"""

from __future__ import annotations

import argparse
import json
import os
import time

from scripts.run_ladder_benchmark import build_market, select_windows

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "results")
_ARTIFACT = os.path.join(_OUT, "batch1_action_knob_ablation.json")

KV = "kv_cache_precision_policy"
PD = "prefill_decode_policy"
GPU = "gpu_assignment_policy"

ABLATION_ARMS = [
    ("1_baseline", frozenset()),
    ("2_kv_only", frozenset({KV})),
    ("3_gpu_assignment_only", frozenset({GPU})),
    ("4_pd_only", frozenset({PD})),
    ("5_kv_plus_gpu", frozenset({KV, GPU})),
    ("6_kv_plus_pd", frozenset({KV, PD})),
    ("7_gpu_plus_pd", frozenset({GPU, PD})),
    ("8_all_three", frozenset({KV, PD, GPU})),
]


def _hierarchical_decider(ctx, win, mask, *, med_prompt):
    from aurelius.environment.controller import DEFAULT_BENCHMARK_PLANNER_MODE
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm, fm = ctx["common"], ctx["fleet"], ctx["cm"], ctx["fm"]
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ws = make_world_state(common.get("world_state_params"))
    c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
    c.horizon_steps = 1
    c.planning_kv_cost_mode = "hybrid_capacity_work"
    c.planning_prompt_tokens = med_prompt
    c.electricity_price_aware = True
    c.planner_mode = DEFAULT_BENCHMARK_PLANNER_MODE
    c.planner_budget = 100
    c.allowed_new_knobs = mask                       # the ablation mask
    return (lambda h: c.decide(h).to_dict()), c


def _baseline_decider(arm, ctx, win, *, med_prompt):
    from aurelius.environment.production_baselines import baseline_decider
    if arm in ("production_scheduler", "sla_aware"):
        return baseline_decider(arm), None
    if arm == "aurelius_mpc_current_default":
        from aurelius.environment.training import _controller as build_controller
        from aurelius.environment.training import make_world_state
        common, fleet, cm, fm = ctx["common"], ctx["fleet"], ctx["cm"], ctx["fm"]
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15},
                             common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        c.electricity_price_aware = True
        c.physics_guided = True
        return (lambda h: c.decide(h).to_dict()), c
    if arm == "oracle_diagnostic":
        from aurelius.environment.controller import DEFAULT_BENCHMARK_PLANNER_MODE
        from aurelius.environment.training import _controller as build_controller
        from aurelius.environment.training import make_world_state
        common, fleet, cm, fm = ctx["common"], ctx["fleet"], ctx["cm"], ctx["fm"]
        per = ctx["per"]
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15},
                             common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        c.electricity_price_aware = True
        c.planner_mode = DEFAULT_BENCHMARK_PLANNER_MODE
        c.planner_budget = 100

        def _oracle(h):
            c.planning_oracle_records = per.get(len(h), [])
            return c.decide(h).to_dict()
        return _oracle, c
    raise ValueError(arm)


def _run(decider_ctrl, arm, ctx, win):
    from aurelius.environment.controller import run_period_episode
    from aurelius.environment.training import make_world_state
    common, fleet, cm, frames, per = ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"], ctx["per"]
    prices = ctx["prices"]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    decide_fn, ctrl = decider_ctrl
    ws2 = make_world_state(common.get("world_state_params"))
    t0 = time.monotonic()
    rep = run_period_episode(arm, decide_fn, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=prices, **common, **kv)
    secs = round(time.monotonic() - t0, 2)
    out = {"gp_per_dollar": round(rep.goodput_per_dollar, 2),
           "sla_violation_rate": round(rep.sla_violation_rate, 6),
           "kv_cache_precision_mix": dict(rep.kv_cache_precision_mix),
           "prefill_decode_mix": dict(rep.prefill_decode_mix),
           "precision_mix": dict(rep.precision_mix), "runtime_s": secs}
    if ctrl is not None and getattr(ctrl, "last_decision_diag", None):
        d = ctrl.last_decision_diag
        out["candidates_generated"] = d.get("candidate_bundles_generated") or d.get("theoretical_bundles")
        out["candidates_evaluated"] = d.get("candidate_bundles_evaluated")
    return out


def _pct(c, b):
    return round(100.0 * (c - b) / b, 3) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=120)        # the frozen Benchmark v1 cap (Phase 0)
    ap.add_argument("--market", default="pjm")
    ap.add_argument("--win-len", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=3)
    ap.add_argument("--mooncake-limit", type=int, default=12000)
    args = ap.parse_args()

    ctx = build_market(args.market, req_cap=args.cap, mooncake_limit=args.mooncake_limit)
    wins = select_windows(ctx["prices"], ctx["n"], win_len=args.win_len, quick=True)
    wname, win = next(iter(wins.items()))
    win = win[:args.max_decisions]
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in ctx["per"].get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512

    rows = {}
    for arm, mask in ABLATION_ARMS:
        rows[arm] = _run(_hierarchical_decider(ctx, win, mask, med_prompt=med_prompt), arm, ctx, win)
        rows[arm]["mask"] = sorted(mask)
        print(f"  {arm:<24} gp/$={rows[arm]['gp_per_dollar']} sla={rows[arm]['sla_violation_rate']} "
              f"kv={rows[arm]['kv_cache_precision_mix']} pd={rows[arm]['prefill_decode_mix']}", flush=True)
    for arm in ("production_scheduler", "sla_aware", "aurelius_mpc_current_default", "oracle_diagnostic"):
        rows[arm] = _run(_baseline_decider(arm, ctx, win, med_prompt=med_prompt), arm, ctx, win)
        print(f"  {arm:<24} gp/$={rows[arm]['gp_per_dollar']} sla={rows[arm]['sla_violation_rate']}", flush=True)

    base = rows["1_baseline"]["gp_per_dollar"]
    prod = rows["production_scheduler"]["gp_per_dollar"]
    sla = rows["sla_aware"]["gp_per_dollar"]
    base_sla = rows["1_baseline"]["sla_violation_rate"]
    summary = {}
    for arm, _m in ABLATION_ARMS:
        r = rows[arm]
        summary[arm] = {
            "gp_per_dollar": r["gp_per_dollar"],
            "vs_baseline_abs": round(r["gp_per_dollar"] - base, 2), "vs_baseline_pct": _pct(r["gp_per_dollar"], base),
            "vs_production_abs": round(r["gp_per_dollar"] - prod, 2), "vs_production_pct": _pct(r["gp_per_dollar"], prod),
            "vs_sla_aware_pct": _pct(r["gp_per_dollar"], sla),
            "sla_violation_rate": r["sla_violation_rate"],
            "pareto_vs_baseline": (r["gp_per_dollar"] >= base and r["sla_violation_rate"] <= base_sla + 1e-9),
            "pareto_vs_production": (r["gp_per_dollar"] >= prod
                                     and r["sla_violation_rate"] <= rows["production_scheduler"]["sla_violation_rate"] + 1e-9),
            "kv_selected": {k: v for k, v in r["kv_cache_precision_mix"].items() if k != "inherit_weight_precision"},
            "pd_selected": {k: v for k, v in r["prefill_decode_mix"].items() if k != "shared"},
            "runtime_s": r["runtime_s"]}
    state = {"config": {"cap": args.cap, "market": args.market, "window": wname,
                        "max_decisions": args.max_decisions, "med_prompt": med_prompt},
             "rows": rows, "summary": summary,
             "comparators": {"production_scheduler": prod, "sla_aware": sla,
                             "aurelius_mpc_current_default": rows["aurelius_mpc_current_default"]["gp_per_dollar"],
                             "oracle_diagnostic": rows["oracle_diagnostic"]["gp_per_dollar"]}}
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(state, f, indent=2)
    print(f"DONE → {_ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
