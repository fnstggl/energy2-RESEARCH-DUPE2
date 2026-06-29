#!/usr/bin/env python3
"""Checkpointed all-knobs backtest — tractable, resumable, never jams (the PR #118 gap).

Reuses the PR #117/#118 checkpointing infrastructure (build_market / select_windows + fork-subprocess +
hard-timeout + checkpoint-after-every-cell + resume). CELL = (market, window, arm). Each cell also captures the
NEW canonical states: a `ForecastState` (planner belief vs realized → forecast error) and a
`RequestLifecycleState` (per-request conservation + queue summary), so the backtest reports state diagnostics,
not just gp/$.

Arms:
  1 strongest_sla_aware_baseline      fixed SLA-aware policy (real price applied to cost)
  2 deployable_all_knobs              MPC, all knobs, constant fleet price, legacy timing (the deployable today)
  3 all_knobs_roofline_timing         arm 2 + PR #119 roofline-resolved timing (AURELIUS_TIMING_MODEL=roofline)
  4 all_knobs_real_price              arm 2 + real electricity price
  5 all_knobs_n2                      arm 4 + N2 (electricity_price_aware)
  6 all_knobs_scenario_forecast       arm 5 + scenario ensemble planning
  7 oracle_forecast_all_knobs         arm 5 + planning against the EXACT future (oracle diagnostic)
  8 v2_reference_all_knobs            SKIPPED_TOO_HEAVY (no tractable V2 serving path on this branch)

Per-cell status: COMPLETED / TIMEOUT / FAILED / SKIPPED_TOO_HEAVY. Runtime caps: --max-decisions,
--max-requests-per-period, --cell-timeout-seconds, --search {clock,adaptive}. clock-focused is the tractable
default (PR #118 showed full adaptive is heavy at hourly cadence → adaptive cells are timeout-protected and may
be SKIPPED_TOO_HEAVY). No tuning; the Pareto gate is unchanged; no headline unless a cell COMPLETED and passes.
Usage: python -m scripts.run_checkpointed_all_knobs_backtest --quick   (then --full --resume)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import time

from scripts.run_checkpointed_electricity_backtest import build_market, select_windows

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "checkpointed_all_knobs_backtest.json")
_CTX: dict = {}

ARMS = ("strongest_sla_aware_baseline", "deployable_all_knobs", "all_knobs_roofline_timing",
        "all_knobs_real_price", "all_knobs_n2", "all_knobs_scenario_forecast",
        "oracle_forecast_all_knobs", "v2_reference_all_knobs")
_REAL_PRICE_ARMS = {"all_knobs_real_price", "all_knobs_n2", "all_knobs_scenario_forecast", "oracle_forecast_all_knobs"}
_N2_ARMS = {"all_knobs_n2", "all_knobs_scenario_forecast", "oracle_forecast_all_knobs"}
_SCENARIO_ARMS = {"all_knobs_scenario_forecast"}
_ORACLE_ARMS = {"oracle_forecast_all_knobs"}
_ROOFLINE_ARMS = {"all_knobs_roofline_timing"}
_SKIP_HEAVY = {"v2_reference_all_knobs": "no tractable V2 serving path on this branch (V2 is audit-doc only)"}


def _clock_candidates():
    from aurelius.environment.actions import ActionBundle
    return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]


def evaluate_cell(ctx, win, arm, *, max_decisions, search):
    from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
    from aurelius.environment.forecast_state import ForecastState
    from aurelius.environment.request_state import RequestLifecycleState
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm = ctx["common"], ctx["fleet"], ctx["cm"]
    frames, per, prices, fm = ctx["frames"], ctx["per"], ctx["prices"], ctx["fm"]
    win = win[:max_decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    elec_prices = prices if arm in _REAL_PRICE_ARMS else None
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    fs = ForecastState()

    if arm == "strongest_sla_aware_baseline":
        def _dec(h):
            return dict(SLA_AWARE_FALLBACK)
    else:
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        c.electricity_price_aware = arm in _N2_ARMS
        c.planning_scenarios = arm in _SCENARIO_ARMS
        c.forecast_state = fs                                   # capture planner belief
        if search == "clock":
            c.candidates = _clock_candidates()
        else:
            c.use_adaptive_search = True
            c.optimize_simulated = False

        def _dec(h):
            if arm in _ORACLE_ARMS:
                c.planning_oracle_records = per.get(len(h))
            return c.decide(h).to_dict()

    ws2 = make_world_state(common.get("world_state_params"))
    rep = run_period_episode(arm, _dec, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=elec_prices, forecast_state=fs, **common, **kv)

    # canonical RequestState: promote each period's requests with the arm's realised SLA-safe fraction
    rls = RequestLifecycleState()
    safe_frac = max(0.0, 1.0 - rep.sla_violation_rate)
    for p in win:
        rls.ingest_period(p, per.get(p, []), sla_s=common.get("sla_s", 10.0), sla_safe_frac=safe_frac)

    return {"arm": arm, "periods": [int(win[0]), int(win[-1])], "n_decisions": len(win),
            "gp_per_dollar": round(rep.goodput_per_dollar, 1),
            "sla_violation_rate": round(rep.sla_violation_rate, 5),
            "gpu_hours": round(rep.gpu_hours, 4), "gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "energy_kwh": round(rep.total_energy_j / 3.6e6, 4), "operator_cost": round(rep.total_operator_cost, 5),
            "queue_p95": round(getattr(rep, "queue_delay_p95_mean", 0.0), 4),
            "clock_mix": rep.clock_mix, "precision_mix": rep.precision_mix, "mean_power_w": rep.mean_power_w,
            "avg_price_paid": round(statistics.mean(prices.get(p, 0.0) for p in win) if win else 0.0, 5),
            "forecast_error": fs.forecast_error_summary(),
            "request_state": {"arrived": rls.arrived, "completed": rls.completed, "missed_sla": rls.missed_sla,
                              "conserved": rls.conserved(), "queue_summary": rls.queue_summary()}}


def _cell_worker(market, win, arm, max_decisions, search, q):
    try:
        if arm in _ROOFLINE_ARMS:
            os.environ["AURELIUS_TIMING_MODEL"] = "roofline"      # PR #119 timing (this subprocess only)
        q.put(("COMPLETED", evaluate_cell(_CTX[market], win, arm, max_decisions=max_decisions, search=search)))
    except Exception as e:                                        # noqa: BLE001 — record, never crash the run
        import traceback
        q.put(("FAILED", {"error": repr(e), "trace": traceback.format_exc()[-800:]}))


def run_cell(market, win, arm, *, max_decisions, search, timeout):
    if arm in _SKIP_HEAVY:
        return "SKIPPED_TOO_HEAVY", {"reason": _SKIP_HEAVY[arm]}, 0.0
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_cell_worker, args=(market, win, arm, max_decisions, search, q))
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


def summarize(state):
    """Per (market, window): deployable all-knobs gp/$ vs the SLA-aware baseline + Pareto gate, with each KPI's
    absolute + relative delta (the reporting standard)."""
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
        dep = arms.get("deployable_all_knobs")
        s = {}
        if base and dep:
            def delta(metric, lower_better=False):
                b, n = base.get(metric), dep.get(metric)
                if b is None or n is None:
                    return None
                ab = round(n - b, 4)
                rel = round(100.0 * (n - b) / b, 3) if abs(b) > 1e-12 else None
                return {"baseline": b, "new": n, "abs": ab, "rel_pct": rel,
                        "better": "lower" if lower_better else "higher",
                        "approx_unchanged": rel is not None and abs(rel) < 0.1}
            s["gp_per_dollar"] = delta("gp_per_dollar")
            s["sla_violation_rate"] = delta("sla_violation_rate", lower_better=True)
            s["gpu_hours"] = delta("gpu_hours", lower_better=True)
            s["energy_kwh"] = delta("energy_kwh", lower_better=True)
            s["operator_cost"] = delta("operator_cost", lower_better=True)
            g = claim_gate({"mpc_controller": SimpleNamespace(goodput_per_dollar=dep["gp_per_dollar"],
                                                              sla_violation_rate=dep["sla_violation_rate"]),
                            "sla_aware": SimpleNamespace(goodput_per_dollar=base["gp_per_dollar"],
                                                         sla_violation_rate=base["sla_violation_rate"])})
            s["pareto_beats_baseline"] = bool(g["beats_fair_baseline"])
            s["pareto_sla_not_worse"] = bool(g["pareto_sla_not_worse"])
            s["headline_safe"] = bool(g["beats_fair_baseline"] and g["pareto_sla_not_worse"])
        # oracle gap (forecast regret) + forecast error of the deployable arm
        oracle, n2 = arms.get("oracle_forecast_all_knobs"), arms.get("all_knobs_n2")
        if n2 and oracle:
            s["oracle_gap_gp$"] = round(oracle["gp_per_dollar"] - n2["gp_per_dollar"], 1)
        if dep:
            s["deployable_forecast_error"] = dep.get("forecast_error")
            s["deployable_request_conserved"] = dep.get("request_state", {}).get("conserved")
        summ[f"{market}|{window}"] = s
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--windows", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--search", default="clock", choices=["clock", "adaptive"])
    ap.add_argument("--max-decisions", type=int, default=3)
    ap.add_argument("--max-requests-per-period", type=int, default=80)
    ap.add_argument("--cell-timeout-seconds", type=int, default=180)
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.markets, args.max_decisions, args.win_len = "pjm", min(args.max_decisions, 2), min(args.win_len, 4)
        args.arms = "strongest_sla_aware_baseline,deployable_all_knobs,all_knobs_real_price,all_knobs_n2,oracle_forecast_all_knobs"

    markets = [m for m in args.markets.split(",") if m]
    arms = [a for a in args.arms.split(",") if a in ARMS]
    state = _load() if (args.resume and not args.force) else {"cells": {}, "config": {}}
    state["config"] = {"max_decisions": args.max_decisions, "req_cap": args.max_requests_per_period,
                       "cell_timeout_s": args.cell_timeout_seconds, "win_len": args.win_len,
                       "search": args.search, "quick": args.quick,
                       "deterministic": "seed=0 fixed world-state; no RNG"}
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
                if not args.force and state["cells"].get(key, {}).get("status") in ("COMPLETED", "TIMEOUT", "SKIPPED_TOO_HEAVY"):
                    print(f"  skip {key} ({state['cells'][key]['status']})", flush=True)
                    continue
                status, result, secs = run_cell(market, win, arm, max_decisions=args.max_decisions,
                                                search=args.search, timeout=args.cell_timeout_seconds)
                state["cells"][key] = {"status": status, "runtime_s": secs, "result": result,
                                       "price_stats": {"min": round(min(_CTX[market]["prices"].get(p, 0) for p in win), 5),
                                                       "max": round(max(_CTX[market]["prices"].get(p, 0) for p in win), 5)}}
                _save(state)
                gp = result.get("gp_per_dollar") if isinstance(result, dict) else None
                print(f"  {status:<17} {key}  {secs}s  gp$={gp}", flush=True)
    state["summary"] = summarize(state)
    _save(state)
    from collections import Counter
    counts = Counter(c["status"] for c in state["cells"].values())
    print(f"DONE: {dict(counts)} → {_ARTIFACT}", flush=True)
    print("SUMMARY:", json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
