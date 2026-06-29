#!/usr/bin/env python3
"""Search-containment diagnostic — does the all-knobs result get worse because of SEARCH, not capability?

Holds the window / dt / cost / cap / baseline FIXED and varies ONLY the candidate space the controller searches:
  * clock_only       — the 3 clock bundles the PR #121 bounded "all-knobs" arm actually used (--search clock)
  * grid_multi_knob  — a small EXHAUSTIVE grid over clock × precision × capacity × batching (tractable)
  * adaptive         — the real adaptive beam+local planner over the full connected space (the +82.1% search)

For each: raw candidate count, evaluated count, chosen bundle, gp/$, SLA. Then search regret =
best-known gp/$ (over the union of all enumerated bundles) − chosen gp/$, and the gp/$ / SLA deltas vs the
SLA-aware baseline. Diagnostic only — no controller/simulator/gate change. Deterministic (seed-0).

Usage: python -m scripts.diagnose_search_containment --market pjm --window expensive --decisions 2
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "search_containment_diagnostic.json")


def _grid_candidates():
    from aurelius.environment.actions import ActionBundle
    clocks = ("base", "low", "high")
    precs = ("bf16", "fp8")
    caps = (1.0, 1.5)
    batches = ("conservative", "aggressive")
    out = []
    for cl, pr, ca, ba in itertools.product(clocks, precs, caps, batches):
        out.append(ActionBundle(clock_policy=cl, precision_policy=pr, capacity_multiplier=ca, batching_policy=ba))
    return out


def _clock_candidates():
    from aurelius.environment.actions import ActionBundle
    return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]


def run(market, window, decisions, req_cap):
    from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    from scripts.run_checkpointed_electricity_backtest import build_market, select_windows

    ctx = build_market(market, req_cap=req_cap, mooncake_limit=20000)
    common, fleet, cm = ctx["common"], ctx["fleet"], ctx["cm"]
    frames, per, prices, fm = ctx["frames"], ctx["per"], ctx["prices"], ctx["fm"]
    wins = select_windows(prices, ctx["n"], win_len=6, quick=False)
    win = wins.get(window, next(iter(wins.values())))[:decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512

    def episode(mode):
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        evals = [0]
        if mode == "clock_only":
            c.candidates = _clock_candidates()
        elif mode == "grid_multi_knob":
            c.candidates = _grid_candidates()
        elif mode == "adaptive":
            c.use_adaptive_search = True
            c.optimize_simulated = False
        raw = (len(c.candidates) if getattr(c, "candidates", None) is not None else None)
        clocks, precs, caps, batches = {}, {}, {}, {}

        def _dec(h):
            d = c.decide(h)
            dd = d.to_dict()
            b = d.bundle
            if b is not None:
                clocks[b.clock_policy] = clocks.get(b.clock_policy, 0) + 1
                precs[getattr(b, "precision_policy", "bf16")] = precs.get(getattr(b, "precision_policy", "bf16"), 0) + 1
                caps[getattr(b, "capacity_multiplier", 1.0)] = caps.get(getattr(b, "capacity_multiplier", 1.0), 0) + 1
                batches[getattr(b, "batching_policy", "conservative")] = batches.get(getattr(b, "batching_policy", "conservative"), 0) + 1
            ld = c.last_decision_diag or {}
            evals[0] = max(evals[0], ld.get("candidate_bundles_evaluated", 0) or 0)
            return dd
        ws2 = make_world_state(common.get("world_state_params"))
        rep = run_period_episode(mode, _dec, per, frames, win, fleet_state=fleet, cost_model=cm,
                                 world_state=ws2, electricity_prices=prices, **common, **kv)
        return {"mode": mode, "raw_candidates": raw, "evaluated": evals[0],
                "gp_per_dollar": round(rep.goodput_per_dollar, 1),
                "sla_violation_rate": round(rep.sla_violation_rate, 5),
                "clock_mix": clocks, "precision_mix": precs, "capacity_mix": {str(k): v for k, v in caps.items()},
                "batching_mix": batches}

    # baseline (SLA-aware) for the gp/$ delta
    ws3 = make_world_state(common.get("world_state_params"))
    base_rep = run_period_episode("sla_aware", lambda h: dict(SLA_AWARE_FALLBACK), per, frames, win,
                                  fleet_state=fleet, cost_model=cm, world_state=ws3, electricity_prices=prices,
                                  **common, **kv)
    baseline = {"mode": "sla_aware_baseline", "gp_per_dollar": round(base_rep.goodput_per_dollar, 1),
                "sla_violation_rate": round(base_rep.sla_violation_rate, 5)}
    arms = [episode(m) for m in ("clock_only", "grid_multi_knob", "adaptive")]
    best_gp = max(a["gp_per_dollar"] for a in arms)
    for a in arms:
        a["search_regret_gp$"] = round(best_gp - a["gp_per_dollar"], 1)
        b = baseline["gp_per_dollar"]
        a["vs_baseline_abs"] = round(a["gp_per_dollar"] - b, 1)
        a["vs_baseline_rel_pct"] = round(100.0 * (a["gp_per_dollar"] - b) / b, 3) if b else None
        a["sla_delta_vs_baseline"] = round(a["sla_violation_rate"] - baseline["sla_violation_rate"], 5)
    return {"market": market, "window": window, "decisions": decisions, "req_cap": req_cap,
            "periods": [int(win[0]), int(win[-1])], "baseline": baseline, "arms": arms,
            "best_known_gp$": best_gp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="pjm")
    ap.add_argument("--window", default="expensive")
    ap.add_argument("--decisions", type=int, default=2)
    ap.add_argument("--req-cap", type=int, default=80)
    args = ap.parse_args()
    res = run(args.market, args.window, args.decisions, args.req_cap)
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"market={res['market']} window={res['window']} periods={res['periods']} "
          f"baseline gp/$={res['baseline']['gp_per_dollar']} sla={res['baseline']['sla_violation_rate']}")
    for a in res["arms"]:
        print(f"  {a['mode']:16} raw={a['raw_candidates']} eval={a['evaluated']:>3}  "
              f"gp/$={a['gp_per_dollar']:>10} ({a['vs_baseline_rel_pct']:+}% vs base)  "
              f"sla_Δ={a['sla_delta_vs_baseline']:+}  regret={a['search_regret_gp$']}  "
              f"clock={a['clock_mix']} prec={a['precision_mix']} cap={a['capacity_mix']}")
    print(f"best_known gp/$={res['best_known_gp$']} → {_ARTIFACT}")


if __name__ == "__main__":
    main()
