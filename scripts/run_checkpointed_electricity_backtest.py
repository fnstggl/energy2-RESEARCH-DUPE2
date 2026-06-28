#!/usr/bin/env python3
"""Checkpointed historical electricity backtest — cannot jam (PR #116 lesson).

Each CELL = (market, window, arm) runs in an ISOLATED subprocess with a HARD timeout; a slow cell is killed
and recorded TIMEOUT, never blocking the run. Inputs are built ONCE per market (parent) and shared with cells
via fork copy-on-write. The artifact is written after EVERY cell; --resume skips completed cells.

Tractability: the electricity-isolation arms search a CLOCK-FOCUSED candidate set (clock ∈ {base,low,high},
non-electric knobs at default) so each MPC decision scores 3 bundles — fast, and it isolates the electricity
effect. Only `all_knobs_current_aurelius` uses the full action space (one slow, timeout-protected cell).

Arms (same serving workload, same Pareto gate, hourly cadence so the diurnal price varies):
  1 baseline_sla_aware            fixed SLA-aware policy (real price applied to cost)
  2 current_main_mpc_flat_price   MPC, constant fleet price, not price-aware
  3 current_main_mpc_real_price   MPC, real price, not price-aware
  4 real_price_dvfs_only          MPC, real price, PRICE-AWARE clock
  5 real_price_deferrable_only    arm 3 + price-aware deferrable shifting (separate ledger)
  6 real_price_dvfs_plus_deferrable  arm 4 + price-aware deferrable shifting
  7 all_knobs_current_aurelius    arm 6 but full action space (total Aurelius)

No tuning; no Pareto-gate weakening; no headline unless a cell COMPLETED and the gate passed.
Usage: python -m scripts.run_checkpointed_electricity_backtest --quick   (then --full --resume)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import statistics
import time

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "checkpointed_electricity_backtest.json")
_CTX: dict = {}                                          # market → built inputs (fork-inherited by cells)

ARMS = ("baseline_sla_aware", "current_main_mpc_flat_price", "current_main_mpc_real_price",
        "real_price_dvfs_only", "real_price_deferrable_only", "real_price_dvfs_plus_deferrable",
        "all_knobs_current_aurelius")
_DEFERRABLE_ARMS = {"real_price_deferrable_only", "real_price_dvfs_plus_deferrable", "all_knobs_current_aurelius"}
_PRICE_AWARE_ARMS = {"real_price_dvfs_only", "real_price_dvfs_plus_deferrable", "all_knobs_current_aurelius"}


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def build_market(market, *, req_cap, mooncake_limit):
    """Heavy per-market build (parent only): inputs + forecasters + real price path. Cached in _CTX."""
    from aurelius.environment.training import build_mpc_inputs, train_forecasters
    inp = build_mpc_inputs(hourly_stride=1, sim_seconds=60.0, use_world_state=True,
                           control_dt_seconds=3600.0, electricity_market=market)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    per = inp["per"]
    per = {p: sorted(per.get(p, []), key=lambda r: r[0])[:req_cap] for p in per}
    frames = inp["frames"]
    n = len(frames)
    fm, _ = train_forecasters(frames, max(8, n - 32))
    pool = _mooncake_pool(mooncake_limit)
    return {"market": market, "common": inp["common"], "fleet": inp["fleet_state"], "cm": inp["cost_model"],
            "frames": frames, "per": per, "n": n, "fm": fm, "pool": pool,
            "prices": inp["electricity"]["prices"], "provenance": inp["electricity"]["provenance"],
            "full_counts": {p: len(inp["per"].get(p, [])) for p in inp["per"]}}


def select_windows(prices, n, *, win_len, quick):
    """Pick representative windows by the price character of their hours: cheap / volatile / expensive."""
    starts = list(range(max(8, n - 96), n - win_len + 1))   # recent span (keeps train<eval)
    cand = []
    for s in starts:
        w = list(range(s, s + win_len))
        pv = [prices.get(p, 0.0) for p in w]
        cand.append((w, statistics.mean(pv), statistics.pstdev(pv) if len(pv) > 1 else 0.0))
    if not cand:
        return {}
    cheap = min(cand, key=lambda c: c[1])[0]
    expensive = max(cand, key=lambda c: c[1])[0]
    volatile = max(cand, key=lambda c: c[2])[0]
    if quick:
        return {"expensive": expensive}                     # quick: the window where electricity matters most
    out = {"cheap": cheap, "volatile": volatile, "expensive": expensive}
    long_len = min(24, n - 8)
    if long_len >= win_len:
        out["long24"] = list(range(n - long_len, n))
    return out


def _clock_candidates():
    from aurelius.environment.actions import ActionBundle
    return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]


def _row(rep, prices, win):
    return {"gp_per_dollar": round(rep.goodput_per_dollar, 1), "sla_violation_rate": round(rep.sla_violation_rate, 5),
            "gpu_hours": round(rep.gpu_hours, 3), "gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "energy_kwh": round(rep.total_energy_j / 3.6e6, 4), "operator_cost": round(rep.total_operator_cost, 5),
            "clock_mix": rep.clock_mix, "precision_mix": rep.precision_mix,
            "avg_price_paid": round(statistics.mean(prices.get(p, 0.0) for p in win) if win else 0.0, 5)}


def evaluate_cell(ctx, win, arm, *, max_decisions):
    """Run one arm over one window → metrics (+ deferrable ledger). Bounded by max_decisions."""
    from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm, frames, per = ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"], ctx["per"]
    prices, fm = ctx["prices"], ctx["fm"]
    win = win[:max_decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    elec_prices = None if arm == "current_main_mpc_flat_price" else prices
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512

    if arm == "baseline_sla_aware":
        def _dec(h):
            return dict(SLA_AWARE_FALLBACK)
    else:
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        c.electricity_price_aware = arm in _PRICE_AWARE_ARMS
        if arm != "all_knobs_current_aurelius":
            c.candidates = _clock_candidates()              # clock-focused isolation (fast); full space for arm 7

        def _dec(h):
            return c.decide(h).to_dict()

    ws2 = make_world_state(common.get("world_state_params"))
    rep = run_period_episode(arm, _dec, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=elec_prices, **common, **kv)
    out = {"arm": arm, "periods": [int(win[0]), int(win[-1])], "n_decisions": len(win), **_row(rep, prices, win)}

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
        out["deferrable"] = {"price_aware": pa, "asap_cost": asap["electricity_cost"],
                             "shifting_saving": round(asap["electricity_cost"] - pa["electricity_cost"], 5),
                             "deadlines_respected": pa["missed"] == 0}
    return out


def _cell_worker(market, win, arm, max_decisions, q):
    try:
        q.put(("COMPLETED", evaluate_cell(_CTX[market], win, arm, max_decisions=max_decisions)))
    except Exception as e:                                  # noqa: BLE001 — record, never crash the run
        import traceback
        q.put(("FAILED", {"error": repr(e), "trace": traceback.format_exc()[-800:]}))


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
    """Per (market, window): lifts, interaction, Pareto gate — from COMPLETED cells only."""
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
        def gp(a):
            return arms[a]["gp_per_dollar"] if a in arms else None
        base, flat, real = gp("baseline_sla_aware"), gp("current_main_mpc_flat_price"), gp("current_main_mpc_real_price")
        dvfs, defr, both = gp("real_price_dvfs_only"), gp("real_price_deferrable_only"), gp("real_price_dvfs_plus_deferrable")
        s = {"baseline": base, "real_no_actions": real}
        if real and dvfs is not None:
            s["dvfs_lift_gp$"] = round(dvfs - real, 1)
        if real and defr is not None:
            s["deferrable_serving_gp$_delta"] = round(defr - real, 1)
        if real and both is not None and dvfs is not None and defr is not None:
            s["combined_lift_gp$"] = round(both - real, 1)
            s["interaction_gp$"] = round(both - dvfs - defr + real, 1)
        if base and both is not None:
            s["all_elec_vs_baseline_pct"] = round(100.0 * (both - base) / base, 2)
        # Pareto gate: best electricity arm vs baseline (real prices)
        best = "real_price_dvfs_plus_deferrable" if both is not None else "real_price_dvfs_only"
        if best in arms and "baseline_sla_aware" in arms:
            g = claim_gate({"mpc_controller": SimpleNamespace(goodput_per_dollar=arms[best]["gp_per_dollar"],
                                                              sla_violation_rate=arms[best]["sla_violation_rate"]),
                            "sla_aware": SimpleNamespace(goodput_per_dollar=arms["baseline_sla_aware"]["gp_per_dollar"],
                                                         sla_violation_rate=arms["baseline_sla_aware"]["sla_violation_rate"])})
            s["pareto_beats_baseline"] = bool(g["beats_fair_baseline"])
            s["pareto_sla_not_worse"] = bool(g["pareto_sla_not_worse"])
            s["headline_safe"] = bool(g["beats_fair_baseline"] and g["pareto_sla_not_worse"])
        # deferrable energy-shifting saving (separate ledger)
        d = arms.get("real_price_dvfs_plus_deferrable", {}).get("deferrable") or arms.get("real_price_deferrable_only", {}).get("deferrable")
        if d:
            s["deferrable_shifting_saving_$"] = d["shifting_saving"]
            s["deferrable_deadlines_respected"] = d["deadlines_respected"]
        summ[f"{market}|{window}"] = s
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--windows", default="")                # subset of cheap,volatile,expensive,long24
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-decisions", type=int, default=4)
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
        args.markets, args.max_decisions, args.win_len = "pjm", min(args.max_decisions, 3), min(args.win_len, 4)
        args.arms = "baseline_sla_aware,current_main_mpc_real_price,real_price_dvfs_only"

    markets = [m for m in args.markets.split(",") if m]
    arms = [a for a in args.arms.split(",") if a in ARMS]
    state = _load_checkpoint() if (args.resume and not args.force) else {"cells": {}, "config": {}}
    state["config"] = {"max_decisions": args.max_decisions, "req_cap": args.max_requests_per_period,
                       "cell_timeout_s": args.cell_timeout_seconds, "win_len": args.win_len,
                       "quick": args.quick, "deterministic": "seed=0 fixed world-state; no RNG"}
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
                state["cells"][key] = {"status": status, "runtime_s": secs, "result": result,
                                       "price_stats": {"min": round(min(_CTX[market]["prices"].get(p, 0) for p in win), 5),
                                                       "max": round(max(_CTX[market]["prices"].get(p, 0) for p in win), 5)}}
                _save(state)                                # checkpoint after EVERY cell
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
