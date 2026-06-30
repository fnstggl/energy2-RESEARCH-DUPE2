#!/usr/bin/env python3
"""Checkpointed production-scheduler benchmark ladder — the headline comparison (Phase D/E).

Each CELL = (market, window, arm) runs in an ISOLATED subprocess with a HARD timeout; a slow cell is killed
and recorded TIMEOUT, never blocking the run (the PR #116 / #122 / #123 harness). Inputs are built ONCE per
market (parent) and shared with cells via fork copy-on-write. The artifact is written after EVERY cell;
--resume skips completed cells.

The 8 ladder arms — ALL scored through the SAME unchanged reward path (`run_period_episode` → persistent world
simulator), the SAME real electricity prices applied to cost, the SAME workload / SLA / cadence:

  1 fifo                              naive reactive autoscale + FCFS (the weak rung)
  2 vllm_only                         vLLM default: continuous batching + FIFO + reactive autoscale
  3 topology_aware                    rack-local placement, no SLA scheduler
  4 sla_aware                         SRPT-conformal + backlog autoscale (the hardest HONEST bar)
  5 production_scheduler              the canonical realistic GPU-fleet scheduler (HEADLINE bar) — a reactive
                                      heuristic with the serving-stack levers (continuous batching, KV routing,
                                      rack placement, class admission, warm pool), NO economic/oracle arbitrage
  6 aurelius_mpc_current_default      Aurelius MPC, current deployable default = physics-guided bounded beam
  7 aurelius_mpc_hierarchical_search  Aurelius MPC, the PR #123 tournament winner (selectable planner mode)
  8 oracle_diagnostic                 NON-deployable upper bound: strongest search + EXACT future workload

Hard separation (the user's clarification): production_scheduler is a benchmark baseline only — a `decide_fn`
in the evaluation layer (`aurelius/environment/production_baselines.py`). It is NOT a planner mode, shares NO
MPC-search / economic / oracle / hierarchical code, and never chooses an Aurelius action. Arms 6/7/8 are the
Aurelius MPC controller path (`planner_mode` / `physics_guided` / `planning_oracle_records`). They are separate
arms by construction.

Headline = arm 7 (or 6) vs arm 5, with a secondary hard bar vs arm 4. Every gp/$ comparison reports BOTH the
absolute and the percent delta. No headline unless the Pareto gate passes (gp/$ up AND SLA not worse). int4 /
quality-risked levers are excluded from arms 6/7/8 (`allow_quality_risk=False`) so the headline cannot lean on
an unsupported quality assumption. No tuning to the benchmark; the oracle is diagnostic only.

Usage: python -m scripts.run_ladder_benchmark --quick     (then --full --resume)
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
_ARTIFACT = os.path.join(_OUT, "ladder_benchmark.json")
_CTX: dict = {}                                          # market → built inputs (fork-inherited by cells)

# the 8 ladder arms, in ladder order (weak → honest bar → headline bar → optimizer → diagnostic ceiling).
ARMS = ("fifo", "vllm_only", "topology_aware", "sla_aware", "production_scheduler",
        "aurelius_mpc_current_default", "aurelius_mpc_hierarchical_search", "oracle_diagnostic")
_BASELINE_ARMS = {"fifo", "vllm_only", "topology_aware", "sla_aware", "production_scheduler"}
_MPC_ARMS = {"aurelius_mpc_current_default", "aurelius_mpc_hierarchical_search", "oracle_diagnostic"}
_HEADLINE = "production_scheduler"          # future gp/$ claims default to this (NOT fifo, NOT oracle)
_WEAK = {"fifo"}                            # the only rung the gate treats as weak


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
            "prices": inp["electricity"]["prices"], "provenance": inp["electricity"]["provenance"]}


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
    return {"cheap": cheap, "volatile": volatile, "expensive": expensive}


def _row(rep, prices, win):
    """The per-arm KPI row — all the headline + Pareto + attribution KPIs from one EpisodeReport."""
    return {
        "gp_per_dollar": round(rep.goodput_per_dollar, 2),
        "sla_violation_rate": round(rep.sla_violation_rate, 5),
        "sla_safe_goodput": round(rep.sla_safe_goodput, 3),
        "operator_cost": round(rep.total_operator_cost, 5),
        "gpu_hours": round(rep.gpu_hours, 3),
        "energy_kwh": round(rep.total_energy_j / 3.6e6, 4),
        "queue_delay_p95": round(rep.queue_delay_p95, 4),
        "queue_delay_p99": round(rep.queue_delay_p99, 4),
        "cold_start_events": rep.cold_start_events,
        "warm_hold_gpu_hours": round(rep.warm_hold_gpu_hours, 4),
        "migration_cost": round(rep.migration_cost, 5),
        "mean_kv_prefix_hit_rate": round(rep.mean_kv_prefix_hit_rate, 4),
        "mean_topology_factor": round(rep.mean_topology_factor, 4),
        "quality_sla_risk_mean": round(rep.quality_sla_risk_mean, 5),
        # action mixes — what the arm actually DID (attribution + the no-economic-arbitrage check)
        "routing_mix": rep.routing_mix, "batching_mix": rep.batching_mix,
        "placement_mix": rep.placement_mix, "migration_mix": rep.migration_mix,
        "prewarm_mix": rep.prewarm_mix, "capacity_multiplier_mix": rep.capacity_multiplier_mix,
        "precision_mix": rep.precision_mix, "clock_mix": rep.clock_mix, "spec_decode_mix": rep.spec_decode_mix,
        "avg_price_paid": round(statistics.mean(prices.get(p, 0.0) for p in win) if win else 0.0, 5),
    }


def _make_decider(arm, ctx, win, *, max_decisions, med_prompt):
    """Build the arm's `decide_fn(history)`. Baselines come from production_baselines (NO MPC import); the
    MPC arms build the controller and select the search branch via planner_mode / physics_guided / oracle."""
    from aurelius.environment.production_baselines import baseline_decider
    if arm in _BASELINE_ARMS:
        # the evaluation-layer baseline: a pure decide_fn. NEVER the MPC controller path.
        return baseline_decider(arm), None

    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm, fm = ctx["common"], ctx["fleet"], ctx["cm"], ctx["fm"]
    per = ctx["per"]
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ws = make_world_state(common.get("world_state_params"))
    c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
    c.horizon_steps = 1
    c.planning_kv_cost_mode = "hybrid_capacity_work"
    c.planning_prompt_tokens = med_prompt
    c.electricity_price_aware = True            # MPC arms use the (causal) real-price path — Aurelius's signal
    # int4 / quality-risked levers are excluded from every MPC arm by the planner DEFAULTS: the physics-guided
    # generator (`physics_guided_candidates`, allow_quality_risk=False) and the planner package
    # (`_planner_package_decide`, hardcoded allow_quality_risk=False) both gate int4 off → headline-safe.

    if arm == "aurelius_mpc_current_default":
        c.physics_guided = True                 # the PR #122 deployable default (bounded-beam over core grid)
        c.use_adaptive_search = False           # (redundant: physics_guided is checked first) — clarifies intent
        return (lambda h: c.decide(h).to_dict()), c
    if arm == "aurelius_mpc_hierarchical_search":
        from aurelius.environment.controller import DEFAULT_BENCHMARK_PLANNER_MODE
        c.planner_mode = DEFAULT_BENCHMARK_PLANNER_MODE   # the PR #124 gate-promoted DEFAULT benchmark planner
        c.planner_budget = 100
        return (lambda h: c.decide(h).to_dict()), c
    if arm == "oracle_diagnostic":
        # NON-deployable ceiling: the strongest search planning against the EXACT future of the served period.
        c.planner_mode = "hierarchical_search"
        c.planner_budget = 100

        def _oracle_decide(h):
            p = len(h)                          # run_period_episode calls decide_fn(frames[:p]) → p = len(h)
            c.planning_oracle_records = per.get(p, [])   # the EXACT future workload of the period being served
            return c.decide(h).to_dict()
        return _oracle_decide, c
    raise ValueError(f"unknown arm {arm!r}")


def evaluate_cell(ctx, win, arm, *, max_decisions):
    """Run one arm over one window → KPI row. Bounded by max_decisions. Same world / prices for every arm."""
    from aurelius.environment.controller import run_period_episode
    from aurelius.environment.training import make_world_state
    common, fleet, cm, frames, per = ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"], ctx["per"]
    prices = ctx["prices"]
    win = win[:max_decisions]
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512

    decide_fn, ctrl = _make_decider(arm, ctx, win, max_decisions=max_decisions, med_prompt=med_prompt)
    # every arm is CHARGED the same real diurnal electricity prices on the energy it uses (fair denominator);
    # only the MPC arms additionally OPTIMIZE against the (causal) price path (electricity_price_aware).
    ws2 = make_world_state(common.get("world_state_params"))
    rep = run_period_episode(arm, decide_fn, per, frames, win, fleet_state=fleet, cost_model=cm,
                             world_state=ws2, electricity_prices=prices, **common, **kv)
    out = {"arm": arm, "periods": [int(win[0]), int(win[-1])], "n_decisions": len(win), **_row(rep, prices, win)}
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
    except Exception as e:                                  # noqa: BLE001 — record, never crash the run
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


def _pct(cand, base):
    return round(100.0 * (cand - base) / base, 3) if base else None


def summarize(state):
    """Per (market, window): the headline (arm 7/6 vs production_scheduler), the secondary bar (vs sla_aware),
    abs + pct deltas, the Pareto clause, and the oracle gap. From COMPLETED cells only."""
    by_mw: dict = {}
    for key, cell in state["cells"].items():
        if cell.get("status") != "COMPLETED":
            continue
        market, window, arm = key.split("|")
        by_mw.setdefault((market, window), {})[arm] = cell["result"]
    summ = {}
    for (market, window), arms in by_mw.items():
        def kpi(a, k="gp_per_dollar"):
            return arms[a][k] if a in arms else None
        s = {"arms_present": sorted(arms)}
        s["gp_per_dollar"] = {a: kpi(a) for a in ARMS if a in arms}
        s["sla_violation_rate"] = {a: kpi(a, "sla_violation_rate") for a in ARMS if a in arms}
        prod = kpi(_HEADLINE)
        sla = kpi("sla_aware")
        for opt in ("aurelius_mpc_hierarchical_search", "aurelius_mpc_current_default"):
            g = kpi(opt)
            if g is None:
                continue
            entry = {}
            if prod is not None:
                entry["vs_production_scheduler"] = {
                    "abs_delta": round(g - prod, 2), "pct_delta": _pct(g, prod),
                    "sla_not_worse": (kpi(opt, "sla_violation_rate") <= kpi(_HEADLINE, "sla_violation_rate") + 1e-9)
                                     if opt in arms and _HEADLINE in arms else None}
            if sla is not None:
                entry["vs_sla_aware"] = {
                    "abs_delta": round(g - sla, 2), "pct_delta": _pct(g, sla),
                    "sla_not_worse": (kpi(opt, "sla_violation_rate") <= kpi("sla_aware", "sla_violation_rate") + 1e-9)
                                     if opt in arms and "sla_aware" in arms else None}
            orc = kpi("oracle_diagnostic")
            if orc is not None:
                entry["oracle_gap"] = {"abs": round(orc - g, 2), "pct_of_oracle": _pct(g, orc)}
            s[opt] = entry
        summ[f"{market}|{window}"] = s
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="pjm,ercot,caiso")
    ap.add_argument("--windows", default="")                # subset of cheap,volatile,expensive
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--max-decisions", type=int, default=4)
    ap.add_argument("--max-requests-per-period", type=int, default=64)
    ap.add_argument("--cell-timeout-seconds", type=int, default=300)
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--mooncake-limit", type=int, default=12000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args()

    markets = [m.strip() for m in args.markets.split(",") if m.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if args.quick:
        markets = markets[:1]
    state = _load_checkpoint() if (args.resume and not args.force) else {"cells": {}, "config": {}}
    state["config"] = {"markets": markets, "arms": arms, "max_decisions": args.max_decisions,
                       "req_cap": args.max_requests_per_period, "cell_timeout_s": args.cell_timeout_seconds,
                       "win_len": args.win_len, "quick": bool(args.quick)}
    _save(state)

    done = to = fa = 0
    for market in markets:
        print(f"[build] {market} …", flush=True)
        _CTX[market] = build_market(market, req_cap=args.max_requests_per_period,
                                    mooncake_limit=args.mooncake_limit)
        wins = select_windows(_CTX[market]["prices"], _CTX[market]["n"], win_len=args.win_len, quick=args.quick)
        if args.windows:
            want = {w.strip() for w in args.windows.split(",")}
            wins = {k: v for k, v in wins.items() if k in want}
        for wname, win in wins.items():
            for arm in arms:
                key = f"{market}|{wname}|{arm}"
                if key in state["cells"] and state["cells"][key].get("status") == "COMPLETED" and not args.force:
                    done += 1
                    continue
                status, result, secs = run_cell(market, win, arm, max_decisions=args.max_decisions,
                                                timeout=args.cell_timeout_seconds)
                state["cells"][key] = {"status": status, "result": result, "seconds": secs}
                _save(state)
                if status == "COMPLETED":
                    done += 1
                    g = result.get("gp_per_dollar")
                    print(f"  ✓ {key}  gp/$={g}  sla={result.get('sla_violation_rate')}  ({secs}s)", flush=True)
                elif status == "TIMEOUT":
                    to += 1
                    print(f"  x {key}  TIMEOUT ({secs}s)", flush=True)
                else:
                    fa += 1
                    print(f"  ! {key}  FAILED: {result.get('error') if result else '?'} ({secs}s)", flush=True)
        _CTX.pop(market, None)                              # free the market build before the next one

    state["summary"] = summarize(state)
    _save(state)
    print(f"DONE: {done} completed, {to} timeout, {fa} failed → {_ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
