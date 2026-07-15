#!/usr/bin/env python3
"""Uncapped search-strategy ablation — is SEARCH where the uncapped headline value lives?

Runs the EXACT uncapped Benchmark V1 harness from `run_request_cap_sweep.py` (same
`build_market` uncapped build, same `select_windows(win_len=6)` expensive window truncated
to 3 decisions, same `run_period_episode` persistent-world reward path, same real diurnal
prices, same 300 s isolated-subprocess cell timeout) and varies ONLY the search strategy
available to the planner. The forecaster, simulator, objective, cost model, constraint
gates, baseline, windows, and load are identical across arms by construction.

Arms (weak search → headline search), all planner arms sharing ONE controller config and
differing only in the candidate set / search mode (mirrors `_make_decider` in
`run_ladder_benchmark.py` and the V1 ablation arms in
`run_physics_guided_planner_backtest.py`):

  1 production_scheduler              the benchmark baseline anchor (decide_fn, no MPC code)
  2 clock_only                        single-surface search: GPU clock bundles only
  3 fixed_24_grid                     fixed 24-bundle grid (clock x precision x capacity x batching)
  4 physics_guided_candidates         regime-specific generated set, argmax (containment only)
  5 aurelius_mpc_current_default      bounded beam over the generated set
  6 aurelius_mpc_hierarchical_search  the PR #124 headline planner (budget 100)
  7 exhaustive_default4               exhaustive safe grid: is enumeration even tractable uncapped?

Sanity contract: arms 1 and 6 must reproduce the published uncapped cells in
`request_cap_sweep.json` (production ~130.5k/130.2k/130.0k gp/$; hierarchical ~1.042M/
1.065M/1.112M gp/$) — if they do not, the run is invalid and must not be quoted.

Usage: python -m scripts.run_uncapped_search_ablation [--markets pjm,ercot,caiso]
       [--cell-timeout-seconds 300] [--resume] [--req-cap N] [--max-decisions 3]
(--req-cap / --max-decisions exist ONLY for smoke tests; the artifact records them.)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time

from scripts.run_ladder_benchmark import _row, build_market, select_windows

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "uncapped_search_ablation.json")
_CTX: dict = {}

ARMS = ("production_scheduler", "clock_only", "fixed_24_grid", "physics_guided_candidates",
        "aurelius_mpc_current_default", "aurelius_mpc_hierarchical_search", "exhaustive_default4")
_MOONCAKE = 12000                                        # same as PR #124 / request_cap_sweep


def _make_decider(arm, ctx, *, med_prompt):
    """One controller config for every planner arm; ONLY the search strategy differs."""
    if arm == "production_scheduler":
        from aurelius.environment.production_baselines import baseline_decider
        return baseline_decider(arm), None

    from aurelius.environment.physics_guided_planner import BoundedBeamPlanner
    from aurelius.environment.search_regret_auditor import (
        clock_only_candidates,
        exhaustive_default4_candidates,
        grid24_candidates,
    )
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state

    common, fleet, cm, fm = ctx["common"], ctx["fleet"], ctx["cm"], ctx["fm"]
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ws = make_world_state(common.get("world_state_params"))
    c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
    c.horizon_steps = 1
    c.planning_kv_cost_mode = "hybrid_capacity_work"
    c.planning_prompt_tokens = med_prompt
    c.electricity_price_aware = True                     # identical across all planner arms

    if arm == "clock_only":
        c.candidates = clock_only_candidates()
    elif arm == "fixed_24_grid":
        c.candidates = grid24_candidates()
    elif arm == "exhaustive_default4":
        c.candidates = exhaustive_default4_candidates()  # SAFE (bf16/fp8) exhaustive grid
    elif arm == "physics_guided_candidates":
        c.physics_guided = True
        c.physics_planner_obj = BoundedBeamPlanner(beam=False, widen=False)
    elif arm == "aurelius_mpc_current_default":
        c.physics_guided = True
        c.use_adaptive_search = False
    elif arm == "aurelius_mpc_hierarchical_search":
        from aurelius.environment.controller import DEFAULT_BENCHMARK_PLANNER_MODE
        c.planner_mode = DEFAULT_BENCHMARK_PLANNER_MODE
        c.planner_budget = 100
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return (lambda h: c.decide(h).to_dict()), c


def evaluate_cell(ctx, win, arm, *, max_decisions):
    """Identical episode path to run_ladder_benchmark.evaluate_cell; only _make_decider differs."""
    from aurelius.environment.controller import run_period_episode
    from aurelius.environment.training import make_world_state
    common, fleet, cm, frames, per = ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"], ctx["per"]
    prices = ctx["prices"]
    win = win[:max_decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512

    decide_fn, ctrl = _make_decider(arm, ctx, med_prompt=med_prompt)
    ws2 = make_world_state(common.get("world_state_params"))
    rep = run_period_episode(arm, decide_fn, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=prices, **common, **kv)
    out = {"arm": arm, "periods": [int(win[0]), int(win[-1])], "n_decisions": len(win),
           **_row(rep, prices, win)}
    if ctrl is not None and getattr(ctrl, "last_decision_diag", None):
        d = ctrl.last_decision_diag
        out["search"] = {"method": d.get("search_method"),
                         "candidate_bundles_evaluated": d.get("candidate_bundles_evaluated"),
                         "theoretical_bundles": d.get("theoretical_bundles"),
                         "runtime_s": d.get("runtime_s"), "timed_out": d.get("timed_out")}
    return out


def _cell_worker(market, win, arm, max_decisions, q):
    try:
        q.put(("COMPLETED", evaluate_cell(_CTX[market], win, arm, max_decisions=max_decisions)))
    except Exception as e:                                  # noqa: BLE001
        import traceback
        q.put(("FAILED", {"error": repr(e), "trace": traceback.format_exc()[-1000:]}))


def run_cell(market, win, arm, *, max_decisions, timeout):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_cell_worker, args=(market, win, arm, max_decisions, q))
    t0 = time.monotonic()
    p.start()
    try:
        status, result = q.get(timeout=timeout)
    except queue.Empty:
        status, result = "TIMEOUT", None
    if p.is_alive():
        p.terminate()
    p.join(timeout=3)
    return status, result, round(time.monotonic() - t0, 1)


def _save(state):
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(state, f, indent=2)


def _pct(c, b):
    return round(100.0 * (c - b) / b, 2) if b else None


def summarize(state):
    """Per market: gp/$ per arm, delta vs production anchor, and share of the headline delta."""
    by_market: dict = {}
    for key, cell in state["cells"].items():
        market, arm = key.split("|")
        by_market.setdefault(market, {})[arm] = cell
    out = {}
    for market, arms in by_market.items():
        def gp(a):
            r = arms.get(a, {}).get("result") or {}
            return r.get("gp_per_dollar")
        def sla(a):
            r = arms.get(a, {}).get("result") or {}
            return r.get("sla_violation_rate")
        base, head = gp("production_scheduler"), gp("aurelius_mpc_hierarchical_search")
        row = {"status": {a: arms.get(a, {}).get("status") for a in ARMS},
               "gp_per_dollar": {a: gp(a) for a in ARMS},
               "sla_violation_rate": {a: sla(a) for a in ARMS},
               "pct_vs_production": {a: _pct(gp(a), base) for a in ARMS if a != "production_scheduler"}}
        if base and head:
            row["headline_delta_recovered_pct"] = {
                a: (round(100.0 * (gp(a) - base) / (head - base), 2) if gp(a) is not None else None)
                for a in ARMS if a != "production_scheduler"}
        out[market] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--cell-timeout-seconds", type=int, default=300)
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--max-decisions", type=int, default=3)
    ap.add_argument("--req-cap", type=int, default=None)     # smoke tests only
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    arms = [a for a in args.arms.split(",") if a in ARMS]

    state = {"cells": {}}
    if args.resume and os.path.exists(_ARTIFACT):
        try:
            state = json.load(open(_ARTIFACT))
        except (json.JSONDecodeError, OSError):
            pass
    state["config"] = {
        "arms": arms, "max_decisions": args.max_decisions, "cell_timeout_s": args.cell_timeout_seconds,
        "req_cap": args.req_cap or "uncapped", "win_len": args.win_len, "mooncake_limit": _MOONCAKE,
        "harness": "PR124_benchmark_v1 (run_period_episode, persistent world, real diurnal prices); "
                   "identical to request_cap_sweep uncapped cells; only the search strategy varies per arm",
        "single_variable": "forecaster, simulator, objective, cost model, constraint gates, baseline, "
                           "windows, and load identical across arms; candidate set / search mode is the "
                           "only difference between planner arms"}
    _save(state)

    for market in markets:
        print(f"[build] {market} (req_cap={args.req_cap or 'uncapped'}) …", flush=True)
        t0 = time.monotonic()
        _CTX[market] = build_market(market, req_cap=args.req_cap, mooncake_limit=_MOONCAKE)
        wins = select_windows(_CTX[market]["prices"], _CTX[market]["n"], win_len=args.win_len, quick=False)
        win = wins.get("expensive", next(iter(wins.values())))[:args.max_decisions]
        counts = {p: len(_CTX[market]["per"].get(p, [])) for p in win}
        print(f"  built in {round(time.monotonic()-t0,1)}s; expensive win {win} counts {counts} "
              f"(sum {sum(counts.values())})", flush=True)
        state.setdefault("windows", {})[market] = {"win": [int(p) for p in win], "counts": counts}
        for arm in arms:
            key = f"{market}|{arm}"
            prev = state["cells"].get(key, {}).get("status")
            prev_to = state["cells"].get(key, {}).get("seconds") or 0
            if prev == "TIMEOUT" and args.cell_timeout_seconds > prev_to:
                prev = None                              # retry timed-out cell at a larger budget
            if prev in ("COMPLETED", "TIMEOUT"):
                print(f"  = {key} (resume: {prev})", flush=True)
                continue
            status, result, secs = run_cell(market, win, arm,
                                            max_decisions=args.max_decisions,
                                            timeout=args.cell_timeout_seconds)
            state["cells"][key] = {"status": status, "result": result, "seconds": secs}
            if status == "COMPLETED":
                print(f"  ✓ {key}  gp/$={result.get('gp_per_dollar')} "
                      f"sla={result.get('sla_violation_rate')} ({secs}s)", flush=True)
            else:
                err = (result or {}).get("error")
                print(f"  x {key}  {status} ({secs}s) {err or ''}", flush=True)
            _save(state)
        _CTX.pop(market, None)

    state["summary"] = summarize(state)
    _save(state)
    print(f"DONE → {_ARTIFACT}", flush=True)
    print(json.dumps(state["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
