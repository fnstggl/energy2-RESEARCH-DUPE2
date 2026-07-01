#!/usr/bin/env python3
"""Is hierarchical_search's ~75-eval pick actually the PEAK of the deployable action space, or is there a much
better bundle we never tried? Diagnostic ONLY — no simulator/reward/gate/cost/baseline/planner/action change,
no tuning. It scores bundles through the SAME single-decision rollout the PR #123 tournament used
(`market_window_scorer`, pjm·expensive, req_cap 80), searching FAR harder than 75 evals:

  1. baselines in this harness: sla_aware, production_scheduler, hierarchical_search@100 (the ~75-eval pick).
  2. a large UNIFORM-RANDOM sample of the deployable, headline-safe space (12 connected knobs, int4 EXCLUDED →
     209,952 bundles), scored in parallel — the empirical max lower-bounds the true peak.
  3. an adaptive HIGH-BUDGET search (cross_entropy @2000) — tests whether a smart search beyond 75 evals climbs.

Reports: the best gp/$ (and SLA) found by ANY of the above vs hierarchical@100, the gap (abs + %), how the peak
bundle differs, coverage (distinct evals / 209,952), and the peak vs production_scheduler. If the union-max ≈
hierarchical@100 despite ~100× more evals, that is real evidence 75 evals is near the peak; if it is much
higher, we found headroom (and which search found it). SIMULATED magnitudes; single decision, one window.

Usage: python -m scripts.diagnose_action_space_peak [--random N] [--workers K] [--ce-budget B] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import random
import time

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "action_space_peak.json")
_MARKET, _WINDOW, _REQ_CAP = "pjm", "expensive", 80

# the deployable, headline-safe option set (== ACTION_SPECS connected surfaces, int4 EXCLUDED). Product = 209,952.
DEPLOY_OPTIONS = {
    "capacity_policy": ["reactive_lag1", "backlog_aware", "forecasted_mcs"],
    "ordering_policy": ["fifo", "abs_conformal"],
    "admission_policy": ["off", "class_aware"],
    "routing_policy": ["round_robin", "shortest_queue", "kv_aware"],
    "capacity_multiplier": [1.0, 0.75, 1.5],
    "batching_policy": ["conservative", "balanced", "aggressive"],
    "prewarm_policy": ["off", "conservative", "aggressive"],
    "placement_policy": ["topology_blind", "rack_local", "network_aware"],
    "migration_policy": ["off", "conservative", "aggressive"],
    "precision_policy": ["bf16", "fp8"],                       # int4 excluded (headline-safe)
    "spec_decode_policy": ["off", "shallow", "medium", "aggressive"],
    "clock_policy": ["base", "low", "high"],
}
_SPACE_SIZE = 1
for _v in DEPLOY_OPTIONS.values():
    _SPACE_SIZE *= len(_v)                                     # 209,952

_SCORER = None                                                # built in parent, fork-inherited by workers
_STATE = None


def _build():
    global _SCORER, _STATE
    from aurelius.environment.planner.planner_tournament import market_window_scorer
    _SCORER, _STATE, meta = market_window_scorer(_MARKET, _WINDOW, req_cap=_REQ_CAP)
    return meta


def _random_bundle(rng):
    from aurelius.environment.actions import ActionBundle
    return ActionBundle(**{k: rng.choice(v) for k, v in DEPLOY_OPTIONS.items()})


def _sample_worker(seed, n, q):
    """Score n uniform-random deployable bundles; return the best by REWARD (how Aurelius selects) and by raw
    gp/$ (the pure ceiling, SLA aside)."""
    try:
        rng = random.Random(seed)
        best_r = (-1e18, None, None, None)                    # (reward, gp, sla, surfaces)
        best_g = (-1e18, None, None)                          # (gp, sla, surfaces)
        for _ in range(n):
            b = _random_bundle(rng)
            r = _SCORER.score(b)
            gp, sla = _SCORER.gp_sla(b)
            if r > best_r[0]:
                best_r = (r, gp, sla, b.non_default_surfaces())
            if gp > best_g[0]:
                best_g = (gp, sla, b.non_default_surfaces())
        q.put(("OK", {"n": n, "best_by_reward": best_r, "best_by_gp": best_g}))
    except Exception as e:                                     # noqa: BLE001
        import traceback
        q.put(("ERR", {"error": repr(e), "trace": traceback.format_exc()[-800:]}))


def _parallel_random(total, workers):
    ctx = mp.get_context("fork")
    per = max(1, total // workers)
    procs, q = [], ctx.Queue()
    for w in range(workers):
        p = ctx.Process(target=_sample_worker, args=(1000 + w, per, q))
        p.start()
        procs.append(p)
    results = []
    for _ in procs:
        try:
            results.append(q.get(timeout=3600))
        except queue.Empty:
            break
    for p in procs:
        p.join(timeout=3)
    ok = [r for s, r in results if s == "OK"]
    n_scored = sum(r["n"] for r in ok)
    br = max((r["best_by_reward"] for r in ok), key=lambda t: t[0], default=(-1e18, None, None, None))
    bg = max((r["best_by_gp"] for r in ok), key=lambda t: t[0], default=(-1e18, None, None))
    return n_scored, br, bg


def _score_bundle(b):
    gp, sla = _SCORER.gp_sla(b)
    return {"gp_per_dollar": round(gp, 1), "sla_violation_rate": round(sla, 4),
            "surfaces": b.non_default_surfaces()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ce-budget", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from aurelius.environment.physics_guided_candidates import SAFE_BASELINE_BUNDLE
    from aurelius.environment.planner.candidate_generators import named_anchor_keys
    from aurelius.environment.planner.search_methods import run_method
    from aurelius.environment.production_baselines import ProductionScheduler
    from scripts.diagnose_headline_reconciliation import _bundle_from_action
    from scripts.run_checkpointed_electricity_backtest import build_market, select_windows

    print(f"[build] {_MARKET}·{_WINDOW} scorer (req_cap {_REQ_CAP}) …", flush=True)
    meta = _build()
    period = meta["period"]

    # baselines IN THIS HARNESS
    ctx = build_market(_MARKET, req_cap=_REQ_CAP, mooncake_limit=6000)
    wins = select_windows(ctx["prices"], ctx["n"], win_len=6, quick=False)
    assert wins.get(_WINDOW, [period])[0] == period
    ps_bundle = _bundle_from_action(ProductionScheduler().decide(ctx["frames"][:period]))
    prod = _score_bundle(ps_bundle)
    sla_aware = _score_bundle(SAFE_BASELINE_BUNDLE)
    nk = named_anchor_keys(None)
    hres = run_method("hierarchical_search", _SCORER.score, budget=100, state=_STATE, named_keys=nk,
                      allow_quality_risk=False)
    hier = {**_score_bundle(hres.best_bundle), "evaluated": hres.candidates_evaluated}
    print(f"  sla_aware={sla_aware['gp_per_dollar']}  production_scheduler={prod['gp_per_dollar']}  "
          f"hierarchical@100={hier['gp_per_dollar']} (evals {hier['evaluated']})", flush=True)

    total = 60 if args.smoke else args.random
    workers = 1 if args.smoke else args.workers
    print(f"[random] scoring {total} uniform-random deployable bundles on {workers} worker(s) "
          f"(space={_SPACE_SIZE}) …", flush=True)
    t0 = time.monotonic()
    n_rand, br, bg = _parallel_random(total, workers)
    print(f"  random best-by-reward gp/$={round(br[1], 1) if br[1] else None} (sla {br[2]}); "
          f"best-by-gp gp/$={round(bg[0], 1)} (sla {bg[1]}); scored {n_rand} in {round(time.monotonic()-t0)}s",
          flush=True)

    ce = None
    if not args.smoke and args.ce_budget:
        print(f"[adaptive] cross_entropy @budget {args.ce_budget} …", flush=True)
        cres = run_method("cross_entropy", _SCORER.score, budget=args.ce_budget, state=_STATE, named_keys=nk,
                          allow_quality_risk=False)
        ce = {**_score_bundle(cres.best_bundle), "evaluated": cres.candidates_evaluated}
        print(f"  cross_entropy best gp/$={ce['gp_per_dollar']} (evals {ce['evaluated']})", flush=True)

    # union peak = best gp/$ among all Aurelius-selectable (reward-max) picks + the raw-gp ceiling.
    candidates = {"hierarchical@100": hier, "random_best_by_reward":
                  {"gp_per_dollar": round(br[1], 1) if br[1] else None, "sla_violation_rate": br[2],
                   "surfaces": br[3]}}
    if ce:
        candidates["cross_entropy"] = ce
    peak_name = max((k for k in candidates if candidates[k]["gp_per_dollar"] is not None),
                    key=lambda k: candidates[k]["gp_per_dollar"])
    peak = candidates[peak_name]
    distinct = (hier["evaluated"] + n_rand + (ce["evaluated"] if ce else 0))
    out = {
        "harness": "single_decision tournament (market_window_scorer, pjm·expensive, req_cap 80)",
        "deployable_space_size": _SPACE_SIZE, "int4_excluded": True, "period": period,
        "baselines": {"sla_aware": sla_aware, "production_scheduler": prod},
        "hierarchical_at_100": hier,
        "search_harder": {"random_scored": n_rand,
                          "random_best_by_reward_gp": round(br[1], 1) if br[1] else None,
                          "random_best_by_reward_sla": br[2], "random_best_by_reward_surfaces": br[3],
                          "random_best_by_raw_gp": round(bg[0], 1), "random_best_by_raw_gp_sla": bg[1],
                          "random_best_by_raw_gp_surfaces": bg[2],
                          "cross_entropy": ce},
        "peak": {"which": peak_name, **peak,
                 "distinct_bundles_evaluated_total": distinct,
                 "coverage_pct_of_deployable_space": round(100.0 * distinct / _SPACE_SIZE, 4)},
        "peak_vs_hierarchical_at_100": {
            "abs": round(peak["gp_per_dollar"] - hier["gp_per_dollar"], 1),
            "pct": round(100.0 * (peak["gp_per_dollar"] - hier["gp_per_dollar"]) / hier["gp_per_dollar"], 2)},
        "peak_vs_production_scheduler_pct": round(
            100.0 * (peak["gp_per_dollar"] - prod["gp_per_dollar"]) / prod["gp_per_dollar"], 2)
            if prod["gp_per_dollar"] else None,
        "hierarchical_vs_production_scheduler_pct": round(
            100.0 * (hier["gp_per_dollar"] - prod["gp_per_dollar"]) / prod["gp_per_dollar"], 2)
            if prod["gp_per_dollar"] else None,
        "note": "Diagnostic only; SIMULATED; single decision; a large-sample LOWER BOUND on the true peak, not "
                "an exhaustive proof of global optimality.",
    }
    if not args.smoke:
        os.makedirs(_OUT, exist_ok=True)
        with open(_ARTIFACT, "w") as f:
            json.dump(out, f, indent=2)
    print("=== PEAK ===", flush=True)
    print(f"  peak ({peak_name}) gp/$={peak['gp_per_dollar']}  vs hierarchical@100 "
          f"{out['peak_vs_hierarchical_at_100']['pct']:+}%  | vs production_scheduler "
          f"{out['peak_vs_production_scheduler_pct']:+}% (hierarchical was "
          f"{out['hierarchical_vs_production_scheduler_pct']:+}%)", flush=True)
    print(f"  coverage: {distinct} distinct / {_SPACE_SIZE} = "
          f"{out['peak']['coverage_pct_of_deployable_space']}%", flush=True)
    if not args.smoke:
        print(f"→ {_ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
