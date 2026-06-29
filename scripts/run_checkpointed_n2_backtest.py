#!/usr/bin/env python3
"""Checkpointed N2 backtest — SLA-slack power arbitrage + all-knobs + forecast-mode arms.

Extends the PR #117 checkpointed runner (reuses its `build_market` / `select_windows` and the same
fork-subprocess + hard-timeout + checkpoint-after-every-cell + resume machinery — so it cannot jam). Each
CELL = (market, window, arm). The all-knobs arms use the real adaptive search (beam-pruned connected space),
NOT the exhaustive 314928-bundle space that timed out in #117; a per-cell timeout still protects every cell.

Arms (isolation: N2 = ONLINE serving clock/DVFS only; deferrable = time-shifting only; never conflated):
  1 strongest_sla_aware_baseline        fixed SLA-aware policy (real price applied to cost)
  2 current_main_all_knobs_flat_price    adaptive all-knobs, constant fleet price, not N2
  3 all_knobs_real_price_no_n2           adaptive all-knobs, real price, electricity_price_aware OFF
  4 all_knobs_real_price_n2_dvfs         arm 3 + electricity_price_aware ON  (N2 power arbitrage)
  5 all_knobs_real_price_deferrable_only arm 3 + price-aware deferrable shifting (separate ledger)
  6 all_knobs_real_price_n2_plus_deferrable  arm 4 + deferrable
  7 oracle_forecast_all_knobs            arm 4 but planning against the EXACT future (oracle diagnostic)
  8 scenario_forecast_all_knobs          arm 4 but planning across the scenario ensemble

N2 value = arm4 − arm3 (gp/$); deferrable serving Δ = arm5 − arm3 (≈0, separate ledger); oracle gap =
arm7 − arm4; scenario value = arm8 − arm4. No tuning; the Pareto gate is unchanged; no headline unless a cell
COMPLETED and the gate passed. Usage: python -m scripts.run_checkpointed_n2_backtest --quick  (then --full --resume)
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
_ARTIFACT = os.path.join(_OUT, "checkpointed_n2_backtest.json")
_CTX: dict = {}

ARMS = ("strongest_sla_aware_baseline", "current_main_all_knobs_flat_price", "all_knobs_real_price_no_n2",
        "all_knobs_real_price_n2_dvfs", "all_knobs_real_price_deferrable_only",
        "all_knobs_real_price_n2_plus_deferrable", "oracle_forecast_all_knobs", "scenario_forecast_all_knobs")
_N2_ARMS = {"all_knobs_real_price_n2_dvfs", "all_knobs_real_price_n2_plus_deferrable",
            "oracle_forecast_all_knobs", "scenario_forecast_all_knobs"}        # electricity_price_aware (N2)
_DEFERRABLE_ARMS = {"all_knobs_real_price_deferrable_only", "all_knobs_real_price_n2_plus_deferrable"}
_FLAT_ARMS = {"current_main_all_knobs_flat_price"}
_ORACLE_ARMS = {"oracle_forecast_all_knobs"}
_SCENARIO_ARMS = {"scenario_forecast_all_knobs"}
_MPC_ARMS = set(ARMS) - {"strongest_sla_aware_baseline"}


def _clock_candidates():
    from aurelius.environment.actions import ActionBundle
    return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]


def evaluate_cell(ctx, win, arm, *, max_decisions, search):
    """Run one arm over one window → metrics + N2 ledger (+ deferrable ledger). Bounded by max_decisions."""
    from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm = ctx["common"], ctx["fleet"], ctx["cm"]
    frames, per, prices, fm = ctx["frames"], ctx["per"], ctx["prices"], ctx["fm"]
    win = win[:max_decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    elec_prices = None if arm in _FLAT_ARMS else prices
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    n2_log: list = []

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
        if search == "clock":
            c.candidates = _clock_candidates()                 # clock-focused (fast N2 isolation)
        else:
            c.use_adaptive_search = True                       # real all-knobs adaptive search (beam-pruned)
            c.optimize_simulated = False

        def _dec(h):
            if arm in _ORACLE_ARMS:
                c.planning_oracle_records = per.get(len(h))     # the EXACT future for this period (oracle)
            d = c.decide(h)
            el = d.forecast.get("electricity", {}) if isinstance(d.forecast, dict) else {}
            n2_log.append({"clock": el.get("selected_clock"), "slack_ms": el.get("sla_slack_ms"),
                           "price": el.get("forecast_price_per_kwh")})
            return d.to_dict()

    ws2 = make_world_state(common.get("world_state_params"))
    rep = run_period_episode(arm, _dec, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=elec_prices, **common, **kv)
    downs = [r for r in n2_log if r.get("clock") == "low"]
    slacks = [r["slack_ms"] for r in n2_log if r.get("slack_ms") is not None]
    out = {"arm": arm, "periods": [int(win[0]), int(win[-1])], "n_decisions": len(win),
           "gp_per_dollar": round(rep.goodput_per_dollar, 1),
           "sla_violation_rate": round(rep.sla_violation_rate, 5),
           "gpu_hours": round(rep.gpu_hours, 3), "energy_kwh": round(rep.total_energy_j / 3.6e6, 4),
           "operator_cost": round(rep.total_operator_cost, 5), "clock_mix": rep.clock_mix,
           "precision_mix": rep.precision_mix, "spec_decode_mix": rep.spec_decode_mix,
           "mean_power_w": rep.mean_power_w,
           "avg_price_paid": round(statistics.mean(prices.get(p, 0.0) for p in win) if win else 0.0, 5),
           # N2 online-serving ledger (never time-shifts serving)
           "n2": {"downclock_decisions": len(downs), "n_decisions": len(n2_log),
                  "downclock_fraction": round(len(downs) / max(1, len(n2_log)), 3),
                  "mean_sla_slack_ms": round(statistics.mean(slacks), 1) if slacks else None,
                  "serving_time_shifted": False}}

    if arm in _DEFERRABLE_ARMS:
        from aurelius.environment.deferrable import generate_deferrable_pool, run_deferrable_episode
        idx = list(range(len(win)))
        busy = sorted(ctx["full_counts"].get(p, 0) for p in win)
        cut = busy[int(0.75 * (len(busy) - 1))] if busy else 0
        spare = {i: (0.0 if ctx["full_counts"].get(p, 0) >= cut and cut > 0 else 1e7) for i, p in enumerate(win)}
        wprices = {i: prices.get(p, 0.06) for i, p in enumerate(win)}
        pa = run_deferrable_episode(generate_deferrable_pool(8, horizon_periods=len(win)), periods=idx,
                                    prices=wprices, spare_by_period=spare, policy="price_aware")
        asap = run_deferrable_episode(generate_deferrable_pool(8, horizon_periods=len(win)), periods=idx,
                                      prices=wprices, spare_by_period=spare, policy="asap")
        out["deferrable"] = {"shifting_saving": round(asap["electricity_cost"] - pa["electricity_cost"], 5),
                             "asap_cost": asap["electricity_cost"], "deadlines_respected": pa["missed"] == 0}
    return out


def _cell_worker(market, win, arm, max_decisions, search, q):
    try:
        q.put(("COMPLETED", evaluate_cell(_CTX[market], win, arm, max_decisions=max_decisions, search=search)))
    except Exception as e:                                   # noqa: BLE001 — record, never crash the run
        import traceback
        q.put(("FAILED", {"error": repr(e), "trace": traceback.format_exc()[-800:]}))


def run_cell(market, win, arm, *, max_decisions, search, timeout):
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


def _load_checkpoint():
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
    """Per (market, window): N2 value, deferrable serving Δ, oracle gap, scenario value, Pareto gate."""
    from types import SimpleNamespace

    from aurelius.environment.controller import SLA_AWARE_FALLBACK  # noqa: F401  (parity import)
    from aurelius.environment.training import claim_gate
    by_mw: dict = {}
    for key, cell in state["cells"].items():
        if cell.get("status") != "COMPLETED":
            continue
        market, window, arm = key.split("|")
        by_mw.setdefault((market, window), {})[arm] = cell["result"]
    summ = {}
    for (market, window), arms in by_mw.items():
        def gp(a):
            return arms[a]["gp_per_dollar"] if a in arms else None
        base = gp("strongest_sla_aware_baseline")
        no_n2, n2 = gp("all_knobs_real_price_no_n2"), gp("all_knobs_real_price_n2_dvfs")
        defr, both = gp("all_knobs_real_price_deferrable_only"), gp("all_knobs_real_price_n2_plus_deferrable")
        oracle, scen = gp("oracle_forecast_all_knobs"), gp("scenario_forecast_all_knobs")
        s = {"baseline": base, "all_knobs_real_price_no_n2": no_n2}
        if no_n2 and n2 is not None:
            s["n2_value_gp$"] = round(n2 - no_n2, 1)
        if no_n2 and defr is not None:
            s["deferrable_serving_gp$_delta"] = round(defr - no_n2, 1)
        if no_n2 and both is not None and n2 is not None and defr is not None:
            s["n2_plus_deferrable_gp$"] = round(both - no_n2, 1)
            s["interaction_gp$"] = round(both - n2 - defr + no_n2, 1)
        if n2 and oracle is not None:
            s["oracle_gap_gp$"] = round(oracle - n2, 1)        # value still missing vs the exact future
        if n2 and scen is not None:
            s["scenario_value_gp$"] = round(scen - n2, 1)
        # Pareto gate: best all-knobs N2 arm vs the SLA-aware baseline
        best_arm = "all_knobs_real_price_n2_plus_deferrable" if both is not None else "all_knobs_real_price_n2_dvfs"
        if best_arm in arms and "strongest_sla_aware_baseline" in arms:
            g = claim_gate({"mpc_controller": SimpleNamespace(goodput_per_dollar=arms[best_arm]["gp_per_dollar"],
                                                              sla_violation_rate=arms[best_arm]["sla_violation_rate"]),
                            "sla_aware": SimpleNamespace(goodput_per_dollar=arms["strongest_sla_aware_baseline"]["gp_per_dollar"],
                                                         sla_violation_rate=arms["strongest_sla_aware_baseline"]["sla_violation_rate"])})
            s["pareto_beats_baseline"] = bool(g["beats_fair_baseline"])
            s["pareto_sla_not_worse"] = bool(g["pareto_sla_not_worse"])
            s["headline_safe"] = bool(g["beats_fair_baseline"] and g["pareto_sla_not_worse"])
        n2l = arms.get("all_knobs_real_price_n2_dvfs", {}).get("n2")
        if n2l:
            s["n2_downclock_fraction"] = n2l["downclock_fraction"]
            s["n2_mean_sla_slack_ms"] = n2l["mean_sla_slack_ms"]
        d = arms.get("all_knobs_real_price_n2_plus_deferrable", {}).get("deferrable") or \
            arms.get("all_knobs_real_price_deferrable_only", {}).get("deferrable")
        if d:
            s["deferrable_shifting_saving_$"] = d["shifting_saving"]
            s["deferrable_deadlines_respected"] = d["deadlines_respected"]
        summ[f"{market}|{window}"] = s
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--windows", default="")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--search", default="adaptive", choices=["adaptive", "clock"])
    ap.add_argument("--max-decisions", type=int, default=3)
    ap.add_argument("--max-requests-per-period", type=int, default=80)
    ap.add_argument("--cell-timeout-seconds", type=int, default=300)
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.markets, args.max_decisions, args.win_len, args.search = "pjm", min(args.max_decisions, 2), min(args.win_len, 4), "clock"
        args.arms = "strongest_sla_aware_baseline,all_knobs_real_price_no_n2,all_knobs_real_price_n2_dvfs"

    markets = [m for m in args.markets.split(",") if m]
    arms = [a for a in args.arms.split(",") if a in ARMS]
    state = _load_checkpoint() if (args.resume and not args.force) else {"cells": {}, "config": {}}
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
                if not args.force and state["cells"].get(key, {}).get("status") in ("COMPLETED", "TIMEOUT"):
                    print(f"  skip {key} ({state['cells'][key]['status']})", flush=True)
                    continue
                status, result, secs = run_cell(market, win, arm, max_decisions=args.max_decisions,
                                                search=args.search, timeout=args.cell_timeout_seconds)
                state["cells"][key] = {"status": status, "runtime_s": secs, "result": result,
                                       "price_stats": {"min": round(min(_CTX[market]["prices"].get(p, 0) for p in win), 5),
                                                       "max": round(max(_CTX[market]["prices"].get(p, 0) for p in win), 5)}}
                _save(state)
                gp = result.get("gp_per_dollar") if isinstance(result, dict) else None
                print(f"  {status:<9} {key}  {secs}s  gp$={gp}", flush=True)
    state["summary"] = summarize(state)
    _save(state)
    done = sum(1 for c in state["cells"].values() if c["status"] == "COMPLETED")
    to = sum(1 for c in state["cells"].values() if c["status"] == "TIMEOUT")
    fa = sum(1 for c in state["cells"].values() if c["status"] == "FAILED")
    print(f"DONE: {done} completed, {to} timeout, {fa} failed → {_ARTIFACT}", flush=True)
    print("SUMMARY:", json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
