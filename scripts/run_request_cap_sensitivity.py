#!/usr/bin/env python3
"""Benchmark v1 request-cap sensitivity sweep (Batch-1 Phase 0).

Runs the SAME ladder harness (`run_ladder_benchmark`) at a sequence of per-period request caps
(56 / 80 / 120 / 200 / uncapped) on one market+window so the benchmark-freeze decision is grounded in
measured runtime, completion, and gp/$ curves — NOT a guess. Reuses the tested cell harness verbatim
(isolated subprocess per cell, hard timeout, identical reward path); the only swept variable is
`req_cap`. The 'uncapped' rung uses a very high cap (the real per-period Azure volume is below it, so
no request is dropped — the served-count column shows the true volume).

For each (cap, arm) it records: status (COMPLETED/TIMEOUT/FAILED), runtime, gp/$, SLA violation rate,
and the requests actually served in the window. The summary adds the headline (Aurelius vs
production_scheduler), the Pareto clause (gp/$ up AND SLA not worse), and the secondary bar vs sla_aware.

Usage: python -m scripts.run_request_cap_sensitivity [--caps 56,80,120,200,100000] [--market pjm]
       [--win-len 4] [--max-decisions 3] [--arms ...] [--cell-timeout-seconds 240]
"""

from __future__ import annotations

import argparse
import json
import os
import time

from scripts.run_ladder_benchmark import (
    _CTX,
    build_market,
    run_cell,
    select_windows,
)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "research", "results")
_ARTIFACT = os.path.join(_OUT, "request_cap_sensitivity.json")

_DEFAULT_ARMS = ("sla_aware", "production_scheduler", "aurelius_mpc_current_default",
                 "aurelius_mpc_hierarchical_search", "oracle_diagnostic")
_HEADLINE = "production_scheduler"


def _pct(c, b):
    return round(100.0 * (c - b) / b, 2) if b else None


def _served(ctx, win):
    """Requests actually served in the window (post-cap) — the true per-period volume column."""
    per = ctx["per"]
    return sum(len(per.get(p, [])) for p in win)


def run_cap(market, cap, *, arms, win_len, max_decisions, timeout, mooncake_limit):
    _CTX[market] = build_market(market, req_cap=cap, mooncake_limit=mooncake_limit)
    wins = select_windows(_CTX[market]["prices"], _CTX[market]["n"], win_len=win_len, quick=True)
    wname, win = next(iter(wins.items()))
    win = win[:max_decisions]
    served = _served(_CTX[market], win)
    rows = {}
    t0 = time.monotonic()
    for arm in arms:
        status, result, secs = run_cell(market, win, arm, max_decisions=max_decisions, timeout=timeout)
        rows[arm] = {"status": status, "seconds": secs,
                     "gp_per_dollar": (result or {}).get("gp_per_dollar"),
                     "sla_violation_rate": (result or {}).get("sla_violation_rate")}
        tag = result.get("gp_per_dollar") if result else None
        print(f"  cap={cap:>6} {arm:<34} {status:<9} gp/$={tag} ({secs}s)", flush=True)
    _CTX.pop(market, None)
    return {"cap": cap, "window": wname, "periods": [int(win[0]), int(win[-1])],
            "requests_served_window": served, "n_decisions": len(win),
            "wall_seconds": round(time.monotonic() - t0, 1), "arms": rows}


def summarize(caps_rows):
    out = {}
    for r in caps_rows:
        arms = r["arms"]
        prod = arms.get(_HEADLINE, {}).get("gp_per_dollar")
        sla = arms.get("sla_aware", {}).get("gp_per_dollar")

        def comp(opt):
            g = arms.get(opt, {}).get("gp_per_dollar")
            if g is None:
                return None
            e = {"gp_per_dollar": g}
            if prod is not None:
                e["vs_production_scheduler"] = {
                    "abs": round(g - prod, 2), "pct": _pct(g, prod),
                    "sla_not_worse": (arms[opt].get("sla_violation_rate") or 0.0)
                    <= (arms[_HEADLINE].get("sla_violation_rate") or 0.0) + 1e-9}
            if sla is not None:
                e["vs_sla_aware"] = {"abs": round(g - sla, 2), "pct": _pct(g, sla)}
            return e
        n_done = sum(1 for a in arms.values() if a["status"] == "COMPLETED")
        n_to = sum(1 for a in arms.values() if a["status"] == "TIMEOUT")
        out[str(r["cap"])] = {
            "requests_served_window": r["requests_served_window"], "window": r["window"],
            "wall_seconds": r["wall_seconds"], "cells_completed": n_done, "cells_timed_out": n_to,
            "baseline_gp_per_dollar": prod, "sla_aware_gp_per_dollar": sla,
            "aurelius_hierarchical": comp("aurelius_mpc_hierarchical_search"),
            "aurelius_current_default": comp("aurelius_mpc_current_default"),
            "oracle_gp_per_dollar": arms.get("oracle_diagnostic", {}).get("gp_per_dollar"),
            "sla_violation_rate": {a: v.get("sla_violation_rate") for a, v in arms.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="56,80,120,200,100000")
    ap.add_argument("--market", default="pjm")
    ap.add_argument("--arms", default=",".join(_DEFAULT_ARMS))
    ap.add_argument("--win-len", type=int, default=4)
    ap.add_argument("--max-decisions", type=int, default=3)
    ap.add_argument("--cell-timeout-seconds", type=int, default=240)
    ap.add_argument("--mooncake-limit", type=int, default=12000)
    args = ap.parse_args()

    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rows = []
    for cap in caps:
        print(f"[cap {cap}] build {args.market} …", flush=True)
        rows.append(run_cap(args.market, cap, arms=arms, win_len=args.win_len,
                            max_decisions=args.max_decisions, timeout=args.cell_timeout_seconds,
                            mooncake_limit=args.mooncake_limit))
    state = {"config": {"market": args.market, "arms": arms, "caps": caps,
                        "win_len": args.win_len, "max_decisions": args.max_decisions},
             "rows": rows, "summary": summarize(rows)}
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(state, f, indent=2)
    print(f"DONE → {_ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
