"""Tournament orchestrator + WindowScorer — compare search methods on the SAME world, at equal budgets.

A `WindowScorer` exposes one memoized `score(bundle) -> reward` plus the bundle's gp/$ and SLA, built from
the SAME causal rollout the controller uses (`simulate_period` / `ModelPredictiveEconomicController._rollout_world`)
— so every method is scored by the identical world model. Two flavours:

  * `market_window_scorer` — a real electricity-market planning decision (reuses the controller's own
    `_rollout_world`); the faithful real-world validation (reproduces the PR #121 containment window).
  * `synthetic_fixture_scorer` — a controlled single-period workload (`_synth_jobs` → `simulate_period`) on
    the standard fleet/cost/world; fast and EXHAUSTIVE-able, so the budget curves, ablations and TRUE-optimum
    regret run here.

`run_tournament_window` runs each method at each evaluation budget over a SHARED cache (a bundle scored once
is reused by every method — the rollout is deterministic), measuring world rollouts, candidates
generated/evaluated, node expansions, CPU time, wall clock and peak memory, then gp/$-per-evaluation and
gp/$-per-rollout. The reward / cost / Pareto gate / baselines are byte-identical; this layer only decides
which bundles get probed. Deterministic (seed-0).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from ..physics_guided_candidates import PlannerRegimeState
from .candidate_generators import classify_regimes, named_anchor_keys
from .search_methods import run_method
from .search_regret import compute_window_regret

_RISK_WEIGHT = 0.5          # matches the controller's default risk_weight (reward = gp − risk·sla·gp)


def _key(b) -> tuple:
    return tuple(sorted(b.to_dict().items()))


# --- synthetic bottleneck fixtures (controlled regimes; the fast, exhaustive-able science) ------------
@dataclass
class Fixture:
    name: str
    arrival_rate: float
    tok_mean: float
    tok_p95: float
    cv: float
    prompt_tokens: int
    sla_s: float = 10.0
    window_seconds: float = 6.0           # bounded job count (~arrival·6 ≈ market req-cap scale) for tractable rollouts
    price: float = 0.05
    expected_regime: str = "mixed"


# market-scale fixtures (≈40–96 jobs, market-scale output lengths) so each rollout is cheap; the token SHAPE
# (decode-vs-prefill ratio) sets the regime, not the absolute size.
SYNTHETIC_FIXTURES = (
    # decode-heavy (output ≫ prompt) → memory-bandwidth-bound; comfortable SLA → slack to downclock / fp8 / batch
    Fixture("memory_bound_decode", arrival_rate=8.0, tok_mean=160, tok_p95=480, cv=0.6,
            prompt_tokens=64, sla_s=20.0, expected_regime="memory_bound"),
    # prefill-heavy (prompt ≫ output) → compute-bound; tight SLA
    Fixture("compute_bound_prefill", arrival_rate=5.0, tok_mean=24, tok_p95=80, cv=1.4,
            prompt_tokens=1024, sla_s=8.0, expected_regime="compute_bound"),
    # regular high-rate latency-critical load → SLA tight
    Fixture("sla_tight", arrival_rate=10.0, tok_mean=96, tok_p95=220, cv=0.35,
            prompt_tokens=256, sla_s=6.0, expected_regime="SLA_tight"),
    # bursty heavy load → queue/capacity pressure
    Fixture("queue_bound", arrival_rate=12.0, tok_mean=128, tok_p95=360, cv=1.8,
            prompt_tokens=300, sla_s=10.0, expected_regime="queue_bound"),
)


def synthetic_fixture_scorer(fx: Fixture, fleet, cost_model, world_params):
    """A memoized `score`/`gp_sla` over a synthetic single-period workload, scored by `simulate_period` on a
    fresh clone (so precision/clock/batching/spec reach reward through the roofline physics, like the live
    path). Returns `(WindowScorer, PlannerRegimeState)`. Standard fleet/cost/world; only the workload varies."""
    from ..controller import _synth_jobs
    from ..training import make_world_state
    from ..world_simulator import clone_world_state_for_candidate, simulate_period
    from .candidate_generators import (
        core_grid,  # noqa: F401  (kept for callers that introspect the space)
    )
    be = getattr(fleet, "best_effort_fraction", 0.05)
    jobs_obj = _synth_jobs(fx.arrival_rate, fx.tok_mean, fx.tok_p95, fx.cv, window_seconds=fx.window_seconds,
                           best_effort_fraction=be, kv_service_factor=1.0)
    jobs = [(j.arrival_s, j.actual_tokens, fx.prompt_tokens) for j in jobs_obj]
    fcast = {"arrival_rate": fx.arrival_rate, "arrival_p90": 1.3 * fx.arrival_rate,
             "mean_service_s": max(0.05, fx.tok_mean * 0.002)}
    base_ws = make_world_state(world_params)
    common0 = dict(sla_s=fx.sla_s, tick_seconds=fx.window_seconds, base_service_factor=1.0,
                   cost_model=cost_model, fleet_state=fleet, cost_scenario="owned", best_effort_fraction=be,
                   period_hours=max(fx.window_seconds, 1.0) / 3600.0, dt_seconds=fx.window_seconds)
    cache: dict = {}

    def _eval(ab):
        clone = clone_world_state_for_candidate(base_ws)
        out = simulate_period(clone, ab, jobs, fcast, replay_kwargs=ab.replay_kwargs(), mutate=False,
                              energy_price_per_kwh=fx.price, **common0)
        gp = float(out.goodput_per_dollar)
        sla = float(out.sla_violation_rate)
        reward = gp - _RISK_WEIGHT * sla * gp
        return reward, gp, sla

    scorer = WindowScorer(_eval, cache)
    # classify the regime from cheap signals (decode regime via roofline of the representative workload)
    decode_regime = _decode_regime(fx.tok_mean, fx.prompt_tokens, fleet)
    state = PlannerRegimeState(decode_regime=decode_regime,
                               sla_slack=(0.3 if fx.sla_s >= 15 else (-0.1 if fx.sla_s <= 7 else 0.05)),
                               capacity_pressure=min(1.0, max(0.0, (fx.arrival_rate / 12.0) - 1.0 + fx.cv / 3)),
                               queue_pressure=min(1.0, fx.cv / 2.0), price_percentile=0.5,
                               output_token_mean=fx.tok_mean, confidence=0.85)
    return scorer, state


def market_window_scorer(market: str, window: str, *, req_cap: int = 80, decision_index: int = 0,
                         price_percentile: float | None = None):
    """A memoized `score`/`gp_sla` for ONE real electricity-market planning decision, reusing the controller's
    own single-decision rollout (`_rollout_world`). The faithful real-world validation. Returns
    `(WindowScorer, PlannerRegimeState, meta)`."""
    from scripts.run_checkpointed_electricity_backtest import build_market, select_windows

    from ..forecast_trajectory import build_trajectory
    from ..simulation_clock import SimulationClock
    from ..training import _controller as build_controller
    from ..training import make_world_state
    ctx = build_market(market, req_cap=req_cap, mooncake_limit=6000)
    wins = select_windows(ctx["prices"], ctx["n"], win_len=6, quick=False)
    win = wins.get(window, next(iter(wins.values())))
    period = win[decision_index]
    common, fleet, cm, frames, per, prices, fm = (ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"],
                                                  ctx["per"], ctx["prices"], ctx["fm"])
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for r in per.get(period, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    cfg = {"horizon": 4, "risk_weight": _RISK_WEIGHT, "confidence_min": 0.15}
    ws = make_world_state(common.get("world_state_params"))
    c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
    c.horizon_steps = 1
    c.planning_kv_cost_mode = "hybrid_capacity_work"
    c.planning_prompt_tokens = med_prompt
    history = frames[:period]
    fb = c.forecasters.predict(history, horizon=c.horizon)
    ar, tm, tp, cv, pr = (fb.at(t, 0) for t in ("arrival_rate", "output_token_mean", "output_token_p95",
                                                "interarrival_cv", "electricity_price"))
    be = c.fleet_state.best_effort_fraction
    H = 1
    clock = SimulationClock(dt_seconds=c.period_seconds)
    traj = build_trajectory(c.forecasters, history, clock, H, mode=c.uncertainty_mode)
    by_routing = c.kv_service_factor_by_routing or {}
    cache: dict = {}

    def _eval(ab):
        factor = by_routing.get(ab.routing_policy, c.kv_service_factor)
        cumulative, steps = c._rollout_world(ab, traj, be=be, factor=factor, horizon_steps=H)
        gp = steps[0]["gp_per_dollar"] if steps else 0.0
        sla = steps[0]["risk_viol"] if steps else 0.0
        return cumulative, gp, sla

    if price_percentile is None:
        vals = sorted(prices.values())
        import bisect
        price_percentile = bisect.bisect_left(vals, prices.get(period, 0.0)) / max(1, len(vals) - 1)
    conf = max(0.0, 1.0 - ((ar.p90 - ar.p10) / max(1e-9, ar.mean) if ar else 1.0))
    state = PlannerRegimeState(
        decode_regime=c._decode_regime_hint(tm), sla_slack=None,
        capacity_pressure=max(0.0, min(1.0, (ar.p90 / ar.mean) - 1.0)) if ar and ar.mean else 0.0,
        price_percentile=price_percentile, output_token_mean=(tm.mean if tm else None), confidence=conf)
    scorer = WindowScorer(_eval, cache)
    meta = {"market": market, "window": window, "period": int(period),
            "median_prompt": med_prompt, "price_percentile": round(price_percentile, 4)}
    return scorer, state, meta


def _decode_regime(tok_mean, prompt_tokens, fleet) -> str | None:
    try:
        from ..roofline import ServingConfig, Workload, roofline_regime
        gpu = (max(fleet.gpu_type_mix, key=fleet.gpu_type_mix.get)
               if getattr(fleet, "gpu_type_mix", None) else "A100")
        wl = Workload(prompt_tokens=int(prompt_tokens), decode_tokens=max(1, int(tok_mean)),
                      context_len=int(prompt_tokens) + 320)
        return roofline_regime("decode", ServingConfig(gpu=gpu, batch_size=16), wl)["roofline_regime"]
    except Exception:
        return None


@dataclass
class WindowScorer:
    """Memoized scorer: `score(b)` returns reward; `gp_sla(b)` returns (gp/$, SLA). `rollouts` = cache misses."""
    eval_fn: object                               # bundle -> (reward, gp, sla)
    cache: dict = field(default_factory=dict)     # bundle key -> (reward, gp, sla)
    rollouts: int = 0

    def score(self, bundle) -> float:
        k = _key(bundle)
        if k not in self.cache:
            self.cache[k] = self.eval_fn(bundle)
            self.rollouts += 1
        return self.cache[k][0]

    def gp_sla(self, bundle) -> tuple:
        k = _key(bundle)
        if k not in self.cache:
            self.cache[k] = self.eval_fn(bundle)
            self.rollouts += 1
        return self.cache[k][1], self.cache[k][2]

    def reward_cache(self) -> dict:
        return {k: v[0] for k, v in self.cache.items()}


# --- per-window tournament ---------------------------------------------------------------------------
def run_tournament_window(scorer: WindowScorer, state: PlannerRegimeState, *, methods, budgets,
                          baseline_bundle, exhaustive_for_regret: bool = False, seed: int = 0,
                          allow_quality_risk: bool = False, prev_best=None) -> dict:
    """Run every method at every budget on one window. Returns a dict with per-(method,budget) metrics,
    the regret table (true-exhaustive when `exhaustive_for_regret`), the baseline, and the active regimes."""
    named = named_anchor_keys(prev_best)
    regimes = sorted(classify_regimes(state))
    # baseline gp/$ + SLA (the fair comparison point) — scored once through the same world.
    base_gp, base_sla = scorer.gp_sla(baseline_bundle)
    cells: dict = {}
    last_results: dict = {}     # method -> MethodResult at the LARGEST budget (for regret + anchor contract)
    for budget in budgets:
        for m in methods:
            r0_rollouts = scorer.rollouts
            cpu0, wall0 = time.process_time(), time.monotonic()
            res = run_method(m, scorer.score, budget=budget, state=state, named_keys=named, seed=seed,
                             allow_quality_risk=allow_quality_risk,
                             exhaustive_surfaces=None)
            wall = time.monotonic() - wall0
            cpu = time.process_time() - cpu0
            new_rollouts = scorer.rollouts - r0_rollouts
            gp, sla = (scorer.gp_sla(res.best_bundle) if res.best_bundle is not None else (0.0, 1.0))
            evals = max(1, res.candidates_evaluated)
            cells[f"{m}|{budget}"] = {
                "method": m, "budget": budget, "gp_per_dollar": round(gp, 2),
                "sla_violation_rate": round(sla, 5), "best_reward": round(res.best_reward, 4),
                "selected_bundle": res.best_bundle.non_default_surfaces() if res.best_bundle else {},
                "candidates_generated": res.candidates_generated,
                "candidates_evaluated": res.candidates_evaluated, "world_rollouts": new_rollouts,
                "node_expansions": res.node_expansions, "total_score_calls": res.total_score_calls,
                "cpu_time_s": round(cpu, 5), "wall_clock_s": round(wall, 5),
                # working set = the (bundle-key → reward) entries the method holds; peak memory ∝ this. A
                # deterministic, hardware-independent memory proxy (tracemalloc would 2–5× the rollout cost).
                "working_set_bundles": res.candidates_evaluated,
                "peak_mem_kb_est": round(res.candidates_evaluated * 0.6, 1),
                "gp_per_evaluation": round(gp / evals, 3), "gp_per_rollout": round(gp / max(1, new_rollouts), 3),
                "anchors_evaluated": res.anchors_evaluated,
                "vs_baseline_gp_abs": round(gp - base_gp, 2),
                "vs_baseline_gp_pct": round(100.0 * (gp - base_gp) / base_gp, 3) if abs(base_gp) > 1e-9 else None,
                "sla_not_worse": bool(sla <= base_sla + 1e-9),
                "headline_safe": bool(gp > base_gp and sla <= base_sla + 1e-9),
                "extra": res.extra}
            last_results[m] = res
    true_opt = None
    if exhaustive_for_regret:
        rc = scorer.reward_cache()
        if rc:
            bk = max(rc, key=rc.get)
            true_opt = (rc[bk], bk)        # the union cache IS the exhaustive optimum when a method enumerated it
    regret = compute_window_regret(last_results, scorer.reward_cache(), true_optimum=true_opt, prev_best=prev_best)
    return {"regimes": regimes, "baseline_gp": round(base_gp, 2), "baseline_sla": round(base_sla, 5),
            "cells": cells, "regret": regret, "rollouts_total": scorer.rollouts}


# --- aggregation + Pareto frontier -------------------------------------------------------------------
def aggregate(window_results: dict, *, methods, budgets) -> dict:
    """Across windows: per-method average / median / variance / worst / best of Pareto-safe gp/$ (at the
    largest budget), the per-window winner, and the timeout count. The recommended planner maximises the
    AVERAGE Pareto-safe gp/$, not a single best number."""
    max_b = max(budgets)
    by_method: dict = {m: [] for m in methods}
    safe_by_method: dict = {m: [] for m in methods}
    per_window_winner: dict = {}
    for wname, wr in window_results.items():
        best_m, best_gp = None, -1e18
        for m in methods:
            cell = wr["cells"].get(f"{m}|{max_b}")
            if not cell:
                continue
            by_method[m].append(cell["gp_per_dollar"])
            if cell["headline_safe"]:
                safe_by_method[m].append(cell["gp_per_dollar"])
            if cell["gp_per_dollar"] > best_gp:
                best_m, best_gp = m, cell["gp_per_dollar"]
        per_window_winner[wname] = best_m
    summary: dict = {}
    for m in methods:
        vals = by_method[m]
        safe = safe_by_method[m]
        summary[m] = {
            "avg_gp": round(statistics.mean(vals), 2) if vals else None,
            "median_gp": round(statistics.median(vals), 2) if vals else None,
            "variance_gp": round(statistics.pvariance(vals), 2) if len(vals) > 1 else 0.0,
            "worst_gp": round(min(vals), 2) if vals else None,
            "best_gp": round(max(vals), 2) if vals else None,
            "avg_pareto_safe_gp": round(statistics.mean(safe), 2) if safe else None,
            "pareto_safe_fraction": round(len(safe) / len(vals), 3) if vals else None,
            "n_windows": len(vals)}
    ranked = sorted([m for m in methods if summary[m]["avg_pareto_safe_gp"] is not None],
                    key=lambda m: -summary[m]["avg_pareto_safe_gp"])
    return {"per_method": summary, "per_window_winner": per_window_winner,
            "ranked_by_avg_pareto_safe_gp": ranked,
            "worst_case_method": min((m for m in methods if summary[m]["avg_gp"] is not None),
                                     key=lambda m: summary[m]["worst_gp"], default=None)}


def pareto_frontier(points: list) -> list:
    """Return the indices of the Pareto-efficient points minimising `cost` and maximising `reward`.
    `points` = [{"label", "cost", "reward"}]. A point is efficient if nothing has ≤cost AND ≥reward."""
    eff = []
    for i, p in enumerate(points):
        dominated = any((q["cost"] <= p["cost"] and q["reward"] >= p["reward"] and
                         (q["cost"] < p["cost"] or q["reward"] > p["reward"])) for j, q in enumerate(points) if j != i)
        if not dominated:
            eff.append(p["label"])
    return eff


__all__ = ["Fixture", "SYNTHETIC_FIXTURES", "WindowScorer", "synthetic_fixture_scorer",
           "market_window_scorer", "run_tournament_window", "aggregate", "pareto_frontier"]
