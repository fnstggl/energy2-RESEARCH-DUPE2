#!/usr/bin/env python3
"""Benchmark V1 request-cap sweep — can the frozen V1 benchmark be UNCAPPED, or what is the highest stable cap?

`cap` here means the **request cap**: the max number of replayed requests **per benchmark period** (NOT a
runtime / candidate / GPU / capacity cap). Diagnostic ONLY: no simulator/reward/cost/gate/baseline/planner/
action change, no tuning. Runs the EXACT PR #124 Benchmark V1 harness (`run_period_episode`, persistent world,
real diurnal prices, 3 decisions, the pjm/ercot/caiso expensive windows). The chosen cap must be the **highest
stable, reproducible** request cap all arms complete — NOT the one with the best Aurelius headline.

Priority order (per the task):
  1. Try UNCAPPED first (req_cap = None) on every required arm + market.
  2. If uncapped completes all arms without timeout → recommend UNCAPPED; stop.
  3. If uncapped fails/times out → probe the boundary to find the highest stable request cap, testing only as
     many caps as needed.

Arms (the V1 headline set): `sla_aware`, `production_scheduler`, `aurelius_mpc_hierarchical_search`.

Efficiency: only `per` (the per-period request list) depends on req_cap; frames/forecasters/fleet/cost/prices/
Mooncake pool are cap-independent → build ONCE per market (uncapped) and re-slice `per` per cap (no rebuild).
Each (cap, market, arm) cell runs in an isolated subprocess with a HARD timeout (default 300 s, the PR #124
cell timeout); a slow cell is killed → TIMEOUT, partial preserved, run continues. Monotonicity: a baseline that
completes UNCAPPED completes at every lower cap (strictly less work) → it is not re-run (marked
STABLE_BY_MONOTONICITY); only the binding (planner) arm is boundary-probed. The artifact is written after every
cell.

Usage: python -m scripts.run_request_cap_sweep   [--cell-timeout-seconds 300] [--markets pjm,ercot,caiso]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time

from scripts.run_ladder_benchmark import build_market, evaluate_cell, select_windows

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "request_cap_sweep.json")
_CTX: dict = {}                                          # market → FULL (uncapped) build, fork-inherited by cells

ARMS = ("sla_aware", "production_scheduler", "aurelius_mpc_hierarchical_search")
_SLOW_ARM = "aurelius_mpc_hierarchical_search"           # the Aurelius planner arm
_MAX_DECISIONS = 3                                       # same as PR #124 Benchmark V1
_MOONCAKE = 12000                                        # same as PR #124


def _capped_per(full_per, cap):
    return {p: (rs if cap is None else rs[:cap]) for p, rs in full_per.items()}


def _cell_worker(market, cap, arm, q):
    try:
        full = _CTX[market]
        capped = {**full, "per": _capped_per(full["per"], cap)}
        res = evaluate_cell(capped, full["win"], arm, max_decisions=_MAX_DECISIONS)
        q.put(("COMPLETED", res))
    except Exception as e:                                  # noqa: BLE001 — record, never crash the run
        import traceback
        q.put(("FAILED", {"error": repr(e), "trace": traceback.format_exc()[-1000:]}))


def run_cell(market, cap, arm, *, timeout):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_cell_worker, args=(market, cap, arm, q))
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
    """Per (market, cap): the V1 headline deltas (abs + pct) vs production_scheduler and sla_aware + Pareto."""
    summ = {}
    for key, cell in state["cells"].items():
        market, cap, arm = key.split("|")
        summ.setdefault(f"{market}|{cap}", {})[arm] = cell
    out = {}
    for mk, arms in summ.items():
        def gp(a):
            r = arms.get(a, {}).get("result") or {}
            return r.get("gp_per_dollar")
        def sla(a):
            r = arms.get(a, {}).get("result") or {}
            return r.get("sla_violation_rate")
        h, ps, sa = gp(_SLOW_ARM), gp("production_scheduler"), gp("sla_aware")
        row = {"arms_status": {a: arms.get(a, {}).get("status") for a in ARMS},
               "gp_per_dollar": {a: gp(a) for a in ARMS},
               "sla_violation_rate": {a: sla(a) for a in ARMS},
               "actual_requests": (arms.get("sla_aware", {}).get("actual_requests")),
               "cap_binding": (arms.get("sla_aware", {}).get("cap_binding"))}
        if h is not None and ps is not None:
            row["aurelius_vs_production_scheduler"] = {
                "abs": round(h - ps, 1), "pct": _pct(h, ps),
                "pareto_pass": bool(h > ps and (sla(_SLOW_ARM) or 0) <= (sla("production_scheduler") or 0) + 1e-9)}
        if h is not None and sa is not None:
            row["aurelius_vs_sla_aware"] = {
                "abs": round(h - sa, 1), "pct": _pct(h, sa),
                "pareto_pass": bool(h > sa and (sla(_SLOW_ARM) or 0) <= (sla("sla_aware") or 0) + 1e-9)}
        out[mk] = row
    return out


def _actual_and_binding(full_counts, win, cap):
    actual = sum(min(full_counts[p], cap if cap is not None else full_counts[p]) for p in win)
    binding = cap is not None and any(full_counts[p] > cap for p in win)
    return actual, binding


def _record(state, key, status, result, secs, actual, binding, counters, *, note=None):
    cell = {"status": status, "result": result, "seconds": secs, "actual_requests": actual, "cap_binding": binding}
    if note:
        cell["note"] = note
    if status == "COMPLETED":
        counters["done"] += 1
        sd = result.get("search") or {}
        cell["candidates_evaluated"] = sd.get("candidate_bundles_evaluated")
        cell["planner_runtime_s"] = sd.get("runtime_s")
        print(f"  ✓ {key}  gp/$={result.get('gp_per_dollar')} sla={result.get('sla_violation_rate')} "
              f"req={actual} ({secs}s)", flush=True)
    elif status in ("TIMEOUT", "TIMEOUT_INFERRED"):
        counters["to"] += 1
        print(f"  x {key}  {status} ({secs}s, req={actual})", flush=True)
    elif status == "STABLE_BY_MONOTONICITY":
        print(f"  = {key}  STABLE_BY_MONOTONICITY (completed uncapped → completes at cap {note})", flush=True)
    else:
        counters["fa"] += 1
        print(f"  ! {key}  FAILED: {result.get('error') if result else '?'}", flush=True)
    state["cells"][key] = cell
    _save(state)


# descending search (uncapped → down). The binding arm is the SLOWEST-to-SIMULATE arm; pjm showed it is the
# `sla_aware` baseline (conservative/no-admission/no-batching → it replays every request serially), NOT the
# planner (which completed uncapped). So the probe gates on `sla_aware` first and only runs the costlier arms at
# a cap where sla_aware completes. Anchors: uncapped + cap 56 (the committed V1 cap) are always measured for the
# sensitivity curve; the descending probe finds the highest stable cap in between.
_DESC_PROBE = (200000, 150000, 100000, 50000, 10000, 1000)


def _completed(state, key):
    c = state["cells"].get(key)
    return c["status"] if c else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--cell-timeout-seconds", type=int, default=300)   # the PR #124 V1 cell timeout
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    timeout = args.cell_timeout_seconds

    state = {"cells": {}, "highest_stable_cap": {}}
    if args.resume and os.path.exists(_ARTIFACT):
        try:
            state = json.load(open(_ARTIFACT))
            state.setdefault("highest_stable_cap", {})
        except (json.JSONDecodeError, OSError):
            pass
    state["config"] = {"arms": list(ARMS), "max_decisions": _MAX_DECISIONS, "cell_timeout_s": timeout,
                       "harness": "PR124_benchmark_v1 (run_period_episode, persistent world, real diurnal prices)",
                       "method": "uncapped first; descending probe gated on the binding (slowest-to-simulate) arm"}
    _save(state)
    counters = {"done": 0, "to": 0, "fa": 0}

    def cell(market, cap, arm, full_counts, win):
        """Run one (or reuse a resumed) cell; returns its status."""
        cap_label = "uncapped" if cap is None else str(cap)
        key = f"{market}|{cap_label}|{arm}"
        prev = _completed(state, key)
        if prev in ("COMPLETED", "TIMEOUT"):
            return prev                                 # resume: already measured
        actual, binding = _actual_and_binding(full_counts, win, cap)
        status, result, secs = run_cell(market, cap, arm, timeout=timeout)
        _record(state, key, status, result, secs, actual, binding, counters)
        return status

    for market in markets:
        print(f"[build] {market} (uncapped, once) …", flush=True)
        _CTX[market] = build_market(market, req_cap=None, mooncake_limit=_MOONCAKE)
        wins = select_windows(_CTX[market]["prices"], _CTX[market]["n"], win_len=args.win_len, quick=False)
        win = wins.get("expensive", next(iter(wins.values())))[:_MAX_DECISIONS]
        _CTX[market]["win"] = win
        full_counts = {p: len(_CTX[market]["per"].get(p, [])) for p in win}
        print(f"  uncapped per-period counts {win} = {full_counts} (sum {sum(full_counts.values())})", flush=True)

        # anchors: uncapped (the "can we uncap" answer) + cap 56 (the committed V1) — all arms, for sensitivity.
        unc = {a: cell(market, None, a, full_counts, win) for a in ARMS}
        for a in ARMS:
            cell(market, 56, a, full_counts, win)
        if all(unc[a] == "COMPLETED" for a in ARMS):
            state["highest_stable_cap"][market] = "uncapped"
            print(f"  → {market}: UNCAPPED stable (all arms completed)", flush=True)
            _save(state)
            _CTX.pop(market, None)
            continue

        # descending probe (uncapped is unstable): gate on sla_aware; the first (highest) cap where ALL arms
        # complete is the highest stable cap.
        highest_stable = None
        for cap in _DESC_PROBE:
            if cell(market, cap, "sla_aware", full_counts, win) != "COMPLETED":
                continue                                # binding arm still times out at this cap → go lower
            ok = all(cell(market, cap, a, full_counts, win) == "COMPLETED"
                     for a in ("production_scheduler", _SLOW_ARM))
            if ok:
                highest_stable = cap
                break                                   # highest stable cap found (descending)
        if highest_stable is None:
            highest_stable = 56 if all(_completed(state, f"{market}|56|{a}") == "COMPLETED" for a in ARMS) else None
        state["highest_stable_cap"][market] = highest_stable
        print(f"  → {market}: highest stable request cap = {highest_stable}", flush=True)
        _save(state)
        _CTX.pop(market, None)

    per_market = state["highest_stable_cap"]
    if per_market and all(v == "uncapped" for v in per_market.values()):
        state["recommended_benchmark_cap"] = "uncapped"
    else:
        numeric = [v for v in per_market.values() if isinstance(v, int)]
        state["recommended_benchmark_cap"] = min(numeric) if numeric else None
    state["summary"] = summarize(state)
    _save(state)
    print(f"DONE: {counters['done']} completed, {counters['to']} timeout, {counters['fa']} failed | "
          f"highest_stable={per_market} | recommended={state.get('recommended_benchmark_cap')} → {_ARTIFACT}",
          flush=True)


if __name__ == "__main__":
    main()
