#!/usr/bin/env python3
"""Physics-guided planner backtest — bounded, checkpointed, never jams, reports every KPI absolute + percent.

Compares the new physics-guided bounded-beam planner against the PR #121 clock-only artifact, the fixed
24-grid, and the exhaustive default-4 ground truth (the search-regret reference), on the SAME windows the
PR #121 diagnostic used. Reuses the #117/#118 checkpointing infra (fork-subprocess + hard timeout +
checkpoint-after-every-cell + resume). CELL = (market, window, arm).

Arms:
  1 strongest_sla_aware_baseline   SLA_AWARE_FALLBACK (the fair baseline / Pareto reference)
  2 clock_only                     the PR #121 bounded artifact (3 clock bundles) — the thing we delete
  3 fixed_24_grid                  the diagnostic 24-grid (clock×{bf16,fp8}×{1.0,1.5}×{cons,aggr})
  4 physics_guided_candidates      generated set, argmax (containment only — no beam)
  5 physics_guided_beam            bounded beam over the generated set (+ coupling polish), no widening
  6 physics_guided_widening        beam + progressive widening (the full planner)
  7 oracle_forecast                physics beam + planning against the EXACT future (forecast-regret ceiling)
  8 exhaustive_default4            the 81-bundle exhaustive ground truth (search-regret reference; heavier)

Search regret is measured against arm 8 (exhaustive) on the same window: regret(arm) = gp/$(exhaustive) −
gp/$(arm), absolute and %. No tuning; the Pareto gate is unchanged; no headline unless a cell COMPLETED and
the gate passes. Deterministic (seed-0 fixed world; no RNG). Per-cell status COMPLETED/TIMEOUT/FAILED.
Usage: python -m scripts.run_physics_guided_planner_backtest --quick   (then --full --resume)
"""

from __future__ import annotations

import argparse
import bisect
import json
import multiprocessing as mp
import os
import queue
import statistics
import time

from scripts.run_checkpointed_electricity_backtest import build_market, select_windows

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "physics_guided_planner_backtest.json")
_CTX: dict = {}

ARMS = ("strongest_sla_aware_baseline", "clock_only", "fixed_24_grid", "physics_guided_candidates",
        "physics_guided_beam", "physics_guided_widening", "oracle_forecast", "exhaustive_default4",
        "exhaustive_int4_diagnostic")
_PHYSICS = {"physics_guided_candidates", "physics_guided_beam", "physics_guided_widening", "oracle_forecast"}
# the SAFE exhaustive (bf16/fp8) is the headline ground truth for search regret; the int4 arm is the
# quality-risked ceiling, reported separately and labelled unsafe (never a headline).
_EXHAUSTIVE_REF = "exhaustive_default4"
_INT4_DIAGNOSTIC = "exhaustive_int4_diagnostic"


def _planner_for(arm):
    from aurelius.environment.physics_guided_planner import BoundedBeamPlanner
    if arm == "physics_guided_candidates":
        return BoundedBeamPlanner(beam=False, widen=False)
    if arm in ("physics_guided_beam", "oracle_forecast"):
        return BoundedBeamPlanner(beam=True, widen=False)
    return BoundedBeamPlanner(beam=True, widen=True)        # physics_guided_widening


def _price_percentile_fn(prices):
    """Deterministic percentile of a period's price within the market's price distribution (soft prior)."""
    vals = sorted(prices.values()) if prices else [0.0]

    def pct(p):
        v = prices.get(p)
        if v is None or len(vals) < 2:
            return None
        return bisect.bisect_left(vals, v) / max(1, len(vals) - 1)
    return pct


def evaluate_cell(ctx, win, arm, *, max_decisions):
    from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
    from aurelius.environment.search_regret_auditor import (
        clock_only_candidates,
        exhaustive_default4_candidates,
        grid24_candidates,
    )
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm = ctx["common"], ctx["fleet"], ctx["cm"]
    frames, per, prices, fm = ctx["frames"], ctx["per"], ctx["prices"], ctx["fm"]
    win = win[:max_decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    pct = _price_percentile_fn(prices)
    diags: list = []

    if arm == "strongest_sla_aware_baseline":
        def _dec(h):
            return dict(SLA_AWARE_FALLBACK)
    else:
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        if arm == "clock_only":
            c.candidates = clock_only_candidates()
        elif arm == "fixed_24_grid":
            c.candidates = grid24_candidates()
        elif arm == "exhaustive_default4":
            c.candidates = exhaustive_default4_candidates()                       # SAFE (bf16/fp8) ground truth
        elif arm == "exhaustive_int4_diagnostic":
            c.candidates = exhaustive_default4_candidates(allow_quality_risk=True)  # +int4 ceiling (unsafe)
        else:                                                  # physics arms
            c.physics_guided = True
            c.physics_planner_obj = _planner_for(arm)

        def _dec(h):
            if arm == "oracle_forecast":
                c.planning_oracle_records = per.get(len(h))
            c.current_price_percentile = pct(len(h))           # soft prior (exact percentile, no leakage)
            d = c.decide(h)
            ld = c.last_decision_diag or {}
            sp = ld.get("search_plan") or {}
            diags.append({
                "generated": sp.get("generated_candidates", ld.get("theoretical_bundles")),
                "evaluated": sp.get("evaluated_candidates", ld.get("candidate_bundles_evaluated")),
                "margin": sp.get("decision_margin"), "widening_rounds": sp.get("widening_rounds"),
                "known_strong_contained": sp.get("known_strong_contained"),
                "prev_best_contained": sp.get("prev_best_contained"),
                "anchors_included": sp.get("anchors_included")})
            return d.to_dict()

    ws2 = make_world_state(common.get("world_state_params"))
    rep = run_period_episode(arm, _dec, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=prices, **common, **kv)

    def _mean(key):
        vals = [d[key] for d in diags if d.get(key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    return {"arm": arm, "periods": [int(win[0]), int(win[-1])], "n_decisions": len(win),
            "gp_per_dollar": round(rep.goodput_per_dollar, 1),
            "sla_violation_rate": round(rep.sla_violation_rate, 5),
            "gpu_hours": round(rep.gpu_hours, 4), "gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "energy_kwh": round(rep.total_energy_j / 3.6e6, 4),
            "operator_cost": round(rep.total_operator_cost, 5),
            "energy_cost": round(rep.energy_cost, 5),
            "queue_p95": round(getattr(rep, "queue_delay_p95", 0.0), 4),
            "clock_mix": rep.clock_mix, "precision_mix": rep.precision_mix, "batching_mix": rep.batching_mix,
            "capacity_multiplier_mix": {str(k): v for k, v in rep.capacity_multiplier_mix.items()},
            "mean_power_w": rep.mean_power_w,
            "candidates_generated_mean": _mean("generated"), "candidates_evaluated_mean": _mean("evaluated"),
            "decision_margin_mean": _mean("margin"), "widening_rounds_mean": _mean("widening_rounds"),
            "known_strong_contained": all(d.get("known_strong_contained") for d in diags) if (diags and arm in _PHYSICS) else None,
            "anchors_first": diags[0].get("anchors_included") if (diags and arm in _PHYSICS) else None}


def _cell_worker(market, win, arm, max_decisions, q):
    try:
        q.put(("COMPLETED", evaluate_cell(_CTX[market], win, arm, max_decisions=max_decisions)))
    except Exception as e:                                      # noqa: BLE001 — record, never crash the run
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


def _load():
    if os.path.exists(_ARTIFACT):
        try:
            return json.load(open(_ARTIFACT))
        except (json.JSONDecodeError, OSError):
            pass
    return {"cells": {}, "config": {}}


def _save(state):
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(state, f, indent=2)


def _delta(base, new, lower_better=False):
    if base is None or new is None:
        return None
    ab = round(new - base, 4)
    rel = round(100.0 * (new - base) / base, 3) if abs(base) > 1e-12 else None
    return {"baseline": base, "new": new, "abs": ab, "rel_pct": rel,
            "better": "lower" if lower_better else "higher",
            "approx_unchanged": rel is not None and abs(rel) < 0.1}


def summarize(state):
    """Per (market, window): every arm vs the SLA-aware baseline (all KPIs absolute + percent), the Pareto
    gate, and search regret vs the exhaustive ground truth (absolute + percent)."""
    from types import SimpleNamespace

    from aurelius.environment.training import claim_gate
    by_mw: dict = {}
    for key, cell in state["cells"].items():
        if cell.get("status") != "COMPLETED":
            continue
        market, window, arm = key.split("|")
        by_mw.setdefault((market, window), {})[arm] = cell["result"]
    summ = {}
    for (market, window), arms in by_mw.items():
        base = arms.get("strongest_sla_aware_baseline")
        exh = arms.get(_EXHAUSTIVE_REF)
        exh_gp = exh["gp_per_dollar"] if exh else None
        s = {"arms": {}}
        for arm, r in arms.items():
            entry = {"gp_per_dollar": r["gp_per_dollar"], "sla_violation_rate": r["sla_violation_rate"],
                     "candidates_generated_mean": r.get("candidates_generated_mean"),
                     "candidates_evaluated_mean": r.get("candidates_evaluated_mean"),
                     "decision_margin_mean": r.get("decision_margin_mean"),
                     "widening_rounds_mean": r.get("widening_rounds_mean"),
                     "known_strong_contained": r.get("known_strong_contained")}
            if base:
                entry["vs_baseline"] = {
                    "gp_per_dollar": _delta(base["gp_per_dollar"], r["gp_per_dollar"]),
                    "sla_violation_rate": _delta(base["sla_violation_rate"], r["sla_violation_rate"], lower_better=True),
                    "gpu_hours": _delta(base["gpu_hours"], r["gpu_hours"], lower_better=True),
                    "energy_kwh": _delta(base["energy_kwh"], r["energy_kwh"], lower_better=True),
                    "operator_cost": _delta(base["operator_cost"], r["operator_cost"], lower_better=True)}
                g = claim_gate({"mpc_controller": SimpleNamespace(goodput_per_dollar=r["gp_per_dollar"],
                                                                  sla_violation_rate=r["sla_violation_rate"]),
                                "sla_aware": SimpleNamespace(goodput_per_dollar=base["gp_per_dollar"],
                                                             sla_violation_rate=base["sla_violation_rate"])})
                entry["headline_safe"] = bool(g["beats_fair_baseline"] and g["pareto_sla_not_worse"])
            if exh_gp is not None:
                entry["search_regret_vs_exhaustive"] = _delta(r["gp_per_dollar"], exh_gp)  # exhaustive − arm
            s["arms"][arm] = entry
        summ[f"{market}|{window}"] = s
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm")
    ap.add_argument("--windows", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-decisions", type=int, default=2)
    ap.add_argument("--max-requests-per-period", type=int, default=80)
    ap.add_argument("--cell-timeout-seconds", type=int, default=240)
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.markets, args.max_decisions = "pjm", min(args.max_decisions, 2)
        args.windows = args.windows or "expensive"

    markets = [m for m in args.markets.split(",") if m]
    arms = [a for a in args.arms.split(",") if a in ARMS]
    state = _load() if (args.resume and not args.force) else {"cells": {}, "config": {}}
    state["config"] = {"max_decisions": args.max_decisions, "req_cap": args.max_requests_per_period,
                       "cell_timeout_s": args.cell_timeout_seconds, "win_len": args.win_len,
                       "deterministic": "seed=0 fixed world-state; no RNG",
                       "search_regret_reference": _EXHAUSTIVE_REF}
    _save(state)
    for market in markets:
        print(f"[build] {market} …", flush=True)
        _CTX[market] = build_market(market, req_cap=args.max_requests_per_period, mooncake_limit=args.mooncake_limit)
        wins = select_windows(_CTX[market]["prices"], _CTX[market]["n"], win_len=args.win_len, quick=args.quick)
        if args.windows:
            wins = {k: v for k, v in wins.items() if k in args.windows.split(",")}
        for wname, win in wins.items():
            for arm in arms:
                key = f"{market}|{wname}|{arm}"
                if not args.force and state["cells"].get(key, {}).get("status") in ("COMPLETED", "TIMEOUT"):
                    print(f"  skip {key} ({state['cells'][key]['status']})", flush=True)
                    continue
                status, result, secs = run_cell(market, win, arm, max_decisions=args.max_decisions,
                                                timeout=args.cell_timeout_seconds)
                state["cells"][key] = {"status": status, "runtime_s": secs, "result": result}
                _save(state)
                gp = result.get("gp_per_dollar") if isinstance(result, dict) else None
                ev = result.get("candidates_evaluated_mean") if isinstance(result, dict) else None
                print(f"  {status:<11} {key}  {secs}s  gp$={gp} eval={ev}", flush=True)
    state["summary"] = summarize(state)
    _save(state)
    from collections import Counter
    counts = Counter(c["status"] for c in state["cells"].values())
    print(f"DONE: {dict(counts)} → {_ARTIFACT}", flush=True)
    print("SUMMARY:", json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
