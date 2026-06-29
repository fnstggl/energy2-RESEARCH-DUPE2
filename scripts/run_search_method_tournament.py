#!/usr/bin/env python3
"""MPC Search-Method Tournament — bounded, checkpointed, never jams; measures (does not assume a winner).

Compares MPC search methods on the SAME world model / workload / action space / baselines / seed, at equal
EVALUATION budgets (the portable currency) as well as wall-clock. Windows: controlled synthetic bottleneck
fixtures (memory-bound decode, compute-bound prefill, SLA-tight, queue-bound — fast + EXHAUSTIVE-able, so the
budget curves / ablations / true-optimum regret run here) and real electricity-market planning decisions
(pjm/ercot/caiso expensive — the PR #121 containment window + real-world validation). CELL = one window; all
methods × budgets share that window's memoized rollout cache (a bundle is scored once, reused by every
method). Per cell: fork-subprocess + hard timeout + checkpoint-after-every-cell + resume. A timeout marks
TIMEOUT and never blocks the run.

No simulator / reward / Pareto-gate / cost-model / baseline / action-semantics change; no tuning. Every KPI
is reported absolute + percent. Default planner is NOT changed by this script — it only measures.
Usage: python -m scripts.run_search_method_tournament --quick   (then --full --resume)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import time

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "search_method_tournament.json")
_ASSETS: dict = {}

# the main roster + the budget ladder (the shared per-window cache makes higher budgets nearly free).
MAIN_METHODS = ("clock_only", "fixed_grid", "expanded_grid", "physics_guided_grid", "random_grid",
                "beam_search", "progressive_widening", "hierarchical_search", "coordinate_descent",
                "cross_entropy", "random_restart", "simulated_annealing", "hybrid", "exhaustive_small")
ABLATION_METHODS = ("random_grid", "physics_guided_grid", "beam_no_anchors", "beam_search",
                    "beam_physics_seed", "progressive_widening")
# the budget ladder. Capped at 100 for the bounded run — beyond that the adaptive methods explore the large
# CONNECTED space (placement/migration/routing/…) and the per-window rollout union balloons; 250/500/1000 are
# left for a follow-up at higher per-window timeout (reported, never hidden). The shared per-window cache
# makes the ladder near-free incrementally.
BUDGETS = (10, 25, 50, 100)
ABLATION_BUDGETS = (25, 100)


def _assets():
    """Standard fleet / cost-model / world params (built once; inherited by fork workers)."""
    if "ready" not in _ASSETS:
        from scripts.run_checkpointed_electricity_backtest import build_market
        ctx = build_market("pjm", req_cap=80, mooncake_limit=6000)
        _ASSETS.update(fleet=ctx["fleet"], cm=ctx["cm"],
                       world_params=ctx["common"].get("world_state_params"), ready=True)
    return _ASSETS


def evaluate_window(spec):
    """Run the full method × budget matrix for one window over a shared cache. `spec` describes the window."""
    from aurelius.environment.physics_guided_candidates import SAFE_BASELINE_BUNDLE
    from aurelius.environment.planner.planner_tournament import (
        SYNTHETIC_FIXTURES,
        market_window_scorer,
        run_tournament_window,
        synthetic_fixture_scorer,
    )
    kind = spec["kind"]
    methods = spec.get("methods", MAIN_METHODS)
    budgets = spec.get("budgets", BUDGETS)
    if kind == "synthetic":
        fx = next(f for f in SYNTHETIC_FIXTURES if f.name == spec["fixture"])
        a = _assets()
        scorer, state = synthetic_fixture_scorer(fx, a["fleet"], a["cm"], a["world_params"])
        exhaustive = True
        meta = {"fixture": fx.name, "expected_regime": fx.expected_regime, "sla_s": fx.sla_s}
    else:
        scorer, state, meta = market_window_scorer(spec["market"], spec["window"], req_cap=spec.get("req_cap", 80))
        exhaustive = False                                   # the connected space is too large to enumerate
    res = run_tournament_window(scorer, state, methods=methods, budgets=budgets,
                                baseline_bundle=SAFE_BASELINE_BUNDLE, exhaustive_for_regret=exhaustive)
    res["meta"] = {**meta, "kind": kind, "exhaustive_regret": exhaustive}
    return res


def _worker(spec, q):
    try:
        q.put(("COMPLETED", evaluate_window(spec)))
    except Exception as e:                                    # noqa: BLE001 — record, never crash the run
        import traceback
        q.put(("FAILED", {"error": repr(e), "trace": traceback.format_exc()[-1200:]}))


def run_cell(spec, *, timeout):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(spec, q))
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


def _load(force):
    if not force and os.path.exists(_ARTIFACT):
        try:
            return json.load(open(_ARTIFACT))
        except (json.JSONDecodeError, OSError):
            pass
    return {"windows": {}, "ablation": {}, "config": {}}


def _save(state):
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(state, f, indent=2)


def summarize(state):
    """Aggregate across windows (avg/median/var/worst/best Pareto-safe gp/$), per-window winner, timeout rate,
    and the Pareto frontiers (evaluations vs gp/$, runtime vs gp/$, evaluations vs regret)."""
    from aurelius.environment.planner.planner_tournament import aggregate, pareto_frontier
    windows = {k: v["result"] for k, v in state["windows"].items()
               if v.get("status") == "COMPLETED" and isinstance(v.get("result"), dict)}
    if not windows:
        return {}
    max_b = max(BUDGETS)
    agg = aggregate(windows, methods=MAIN_METHODS, budgets=BUDGETS)
    # Pareto frontiers at the largest budget: each method = one point (mean across windows).
    import statistics
    pts_eval, pts_runtime, pts_regret = [], [], []
    for m in MAIN_METHODS:
        gps, evals, walls, regrets = [], [], [], []
        for wr in windows.values():
            c = wr["cells"].get(f"{m}|{max_b}")
            if not c:
                continue
            gps.append(c["gp_per_dollar"])
            evals.append(c["candidates_evaluated"])
            walls.append(c["wall_clock_s"])
            rg = wr["regret"]["per_method"].get(m, {})
            if rg.get("regret_abs") is not None:
                regrets.append(rg["regret_abs"])
        if gps:
            g = statistics.mean(gps)
            pts_eval.append({"label": m, "cost": statistics.mean(evals), "reward": g})
            pts_runtime.append({"label": m, "cost": statistics.mean(walls), "reward": g})
            if regrets:
                pts_regret.append({"label": m, "cost": statistics.mean(evals), "reward": -statistics.mean(regrets)})
    statuses = [v.get("status") for v in state["windows"].values()]
    return {"aggregate": agg,
            "pareto_efficient_evals_vs_gp": pareto_frontier(pts_eval),
            "pareto_efficient_runtime_vs_gp": pareto_frontier(pts_runtime),
            "pareto_efficient_evals_vs_regret": pareto_frontier(pts_regret),
            "timeout_rate": round(sum(s == "TIMEOUT" for s in statuses) / max(1, len(statuses)), 3),
            "n_windows_completed": len(windows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--cell-timeout-seconds", type=int, default=420)
    ap.add_argument("--req-cap", type=int, default=80)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    args = ap.parse_args()

    from aurelius.environment.planner.planner_tournament import SYNTHETIC_FIXTURES
    synth = [{"id": f"synthetic|{f.name}", "kind": "synthetic", "fixture": f.name} for f in SYNTHETIC_FIXTURES]
    markets = [m for m in args.markets.split(",") if m]
    if args.quick:
        synth = synth[:2]
        markets = markets[:1]
    market_specs = [{"id": f"market|{m}|expensive", "kind": "market", "market": m, "window": "expensive",
                     "req_cap": args.req_cap} for m in markets]
    specs = synth + market_specs

    state = _load(args.force) if (args.resume or not args.force) else {"windows": {}, "ablation": {}, "config": {}}
    state.setdefault("windows", {})
    state.setdefault("ablation", {})
    state["config"] = {"methods": list(MAIN_METHODS), "budgets": list(BUDGETS),
                       "cell_timeout_s": args.cell_timeout_seconds, "quick": args.quick,
                       "deterministic": "seed=0 fixed world; shared per-window rollout cache",
                       "note": "diagnostic only — default planner not changed; measures, does not assume"}
    _save(state)
    print("[assets] building standard fleet/cost/world …", flush=True)
    _assets()
    for spec in specs:
        key = spec["id"]
        if not args.force and state["windows"].get(key, {}).get("status") in ("COMPLETED", "TIMEOUT"):
            print(f"  skip {key} ({state['windows'][key]['status']})", flush=True)
            continue
        status, result, secs = run_cell(spec, timeout=args.cell_timeout_seconds)
        state["windows"][key] = {"status": status, "runtime_s": secs, "result": result}
        _save(state)
        nm = (result.get("rollouts_total") if isinstance(result, dict) else None)
        print(f"  {status:<10} {key}  {secs}s  rollouts={nm}", flush=True)

    if not args.no_ablation:
        for spec in synth:
            key = f"ablation|{spec['fixture']}"
            if not args.force and state["ablation"].get(key, {}).get("status") in ("COMPLETED", "TIMEOUT"):
                continue
            astatus, aresult, asecs = run_cell({**spec, "methods": ABLATION_METHODS, "budgets": ABLATION_BUDGETS},
                                               timeout=args.cell_timeout_seconds)
            state["ablation"][key] = {"status": astatus, "runtime_s": asecs, "result": aresult}
            _save(state)
            print(f"  ablation {astatus:<10} {key}  {asecs}s", flush=True)

    state["summary"] = summarize(state)
    _save(state)
    from collections import Counter
    counts = Counter(v["status"] for v in state["windows"].values())
    print(f"DONE: {dict(counts)} → {_ARTIFACT}", flush=True)
    if state.get("summary", {}).get("aggregate"):
        print("RANKED (avg Pareto-safe gp/$):", state["summary"]["aggregate"]["ranked_by_avg_pareto_safe_gp"][:5])


if __name__ == "__main__":
    main()
