"""Focused tests for the MPC search-method tournament (planner package).

Synthetic score functions (no world build) prove the contracts: required anchors are always included, the
ACTION_SUBSET_CONTAINMENT diagnostic winner is always contained, the fixed grid is the full
precision×batching×capacity×clock product, the physics generator changes by regime, beam search beats
coordinate descent on a coupled optimum the latter cannot reach, progressive widening expands on a small
margin and stops on a large one, the evaluation budget can never run away (anti-jam), the subprocess timeout
returns TIMEOUT, search regret is computed correctly, exhaustive_small finds the true optimum, and replay is
deterministic. Fast."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time

import pytest

from aurelius.environment.actions import ActionBundle
from aurelius.environment.physics_guided_candidates import PlannerRegimeState
from aurelius.environment.planner.candidate_generators import (
    CORE_GRID_SURFACES,
    DIAGNOSTIC_WINNER,
    classify_regimes,
    core_grid,
    enforce_anchors,
    named_anchor_keys,
    physics_guided_candidates,
    required_anchors,
)
from aurelius.environment.planner.search_methods import (
    ALL_METHODS,
    BudgetedScorer,
    run_method,
)
from aurelius.environment.planner.search_regret import compute_window_regret


def _key(b):
    return tuple(sorted(b.to_dict().items()))


def _state(**kw):
    base = dict(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.85)
    base.update(kw)
    return PlannerRegimeState(**base)


# --- anchors + containment ---------------------------------------------------------------------------
def test_required_anchors_always_included():
    anchors = required_anchors(include_core_grid=True)
    keys = {_key(b) for b in anchors}
    for nk in named_anchor_keys():
        assert nk in keys
    assert _key(DIAGNOSTIC_WINNER) in keys
    assert _key(ActionBundle()) in keys                       # neutral


def test_diagnostic_winner_always_contained_in_methods():
    nk = named_anchor_keys()
    st = _state()
    def score(b):
        return 1.0 + (0.5 if b.precision_policy == "fp8" else 0)
    for m in ALL_METHODS:
        if m == "clock_only":
            continue                                          # the deliberate degenerate artifact
        r = run_method(m, score, budget=50, state=st, named_keys=nk, seed=0)
        assert _key(DIAGNOSTIC_WINNER) in r.evaluated_keys, f"{m} did not evaluate the diagnostic winner"
        assert r.anchors_evaluated, f"{m} missed a named anchor"


def test_enforce_anchors_restores_missing():
    merged, missing = enforce_anchors([ActionBundle(clock_policy="low")], include_core_grid=False)
    assert not missing or _key(DIAGNOSTIC_WINNER) in {_key(b) for b in merged}
    assert _key(DIAGNOSTIC_WINNER) in {_key(b) for b in merged}


# --- fixed grid is the full core product -------------------------------------------------------------
def test_fixed_grid_is_full_core_product():
    grid = core_grid()
    expected = 1
    for opts in CORE_GRID_SURFACES.values():
        expected *= len(opts)
    assert len(grid) == expected == 54
    assert _key(ActionBundle(precision_policy="fp8", batching_policy="aggressive",
                             capacity_multiplier=1.5, clock_policy="high")) in {_key(b) for b in grid}


# --- physics generator changes by regime -------------------------------------------------------------
def test_physics_generator_changes_by_regime():
    # the SET always contains the core grid (the invariant), but the regime prior changes the ORDER — which
    # is what matters under a budget cap (regime-preferred candidates are evaluated first).
    mem = physics_guided_candidates(_state(decode_regime="memory_bandwidth_bound"))
    comp = physics_guided_candidates(_state(decode_regime="compute_bound"))
    assert [_key(b) for b in mem] != [_key(b) for b in comp]
    assert {_key(b) for b in mem} == {_key(b) for b in comp}   # same set (core grid) — invariant holds


def test_classify_regimes_labels():
    assert "memory_bound" in classify_regimes(_state(decode_regime="memory_bandwidth_bound"))
    assert "compute_bound" in classify_regimes(_state(decode_regime="compute_bound"))
    assert "queue_bound" in classify_regimes(_state(queue_pressure=0.8))
    assert "power_expensive" in classify_regimes(_state(price_percentile=0.9))


# --- beam beats coordinate descent on a coupled optimum that is NOT an anchor ------------------------
def test_beam_finds_coupled_optimum_coordinate_misses():
    nk = named_anchor_keys()
    st = _state()
    # the coupled win is {balanced batching + 0.75 capacity} — NOT an anchor; neither lever alone improves.
    # non-coupled moves score LOW (50) so the balanced partial survives beam pruning (a true coupling test).
    def score(b):
        bal = b.batching_policy == "balanced"
        cap = b.capacity_multiplier == 0.75
        if bal and cap:
            return 200.0
        if b.precision_policy == "fp8" and b.batching_policy == "aggressive" and b.clock_policy == "high":
            return 130.0                                       # the diagnostic-winner anchor (below the coupled win)
        if not b.non_default_surfaces():
            return 100.0                                       # neutral
        if bal or cap:
            return 99.0                                        # each lever alone — worse than neutral
        return 50.0
    beam = run_method("beam_search", score, budget=80, state=st, named_keys=nk, seed=0)
    coord = run_method("coordinate_descent", score, budget=80, state=st, named_keys=nk, seed=0)
    assert beam.best_reward == 200.0                          # beam couples balanced+0.75
    assert coord.best_reward < 200.0                          # coordinate cannot reach the coupling
    assert beam.best_reward > coord.best_reward


# --- progressive widening expands on small margin, stops on large ------------------------------------
def test_progressive_widening_expands_small_margin():
    nk = named_anchor_keys()
    st = _state(confidence=0.9, sla_slack=0.3)
    r = run_method("progressive_widening", lambda b: 100.0 + 1e-4 * len(b.non_default_surfaces()),
                   budget=200, state=st, named_keys=nk, seed=0)
    assert r.extra.get("widening_rounds", 0) >= 1


def test_progressive_widening_stops_large_margin():
    nk = named_anchor_keys()
    st = _state(confidence=0.9, sla_slack=0.3)
    def clear(b):                                            # a UNIQUE clear winner (no score ties)
        return 1000.0 if (b.precision_policy == "fp8" and b.batching_policy == "aggressive"
                          and b.clock_policy == "high" and b.capacity_multiplier == 1.0) else 50.0
    r = run_method("progressive_widening", clear, budget=200, state=st, named_keys=nk, seed=0)
    assert r.extra.get("widening_rounds", 0) == 0


# --- evaluation budget can never run away (anti-jam at the method level) ------------------------------
def test_budget_never_exceeded():
    nk = named_anchor_keys()
    st = _state()
    calls = {"n": 0}
    def score(b):
        calls["n"] += 1
        return float(hash(_key(b)) % 1000)
    for m in ALL_METHODS:
        calls["n"] = 0
        r = run_method(m, score, budget=20, state=st, named_keys=nk, seed=0)
        assert r.candidates_evaluated <= 20, f"{m} evaluated {r.candidates_evaluated} > 20"
        assert calls["n"] <= 20 + 5, f"{m} called score_fn {calls['n']} times (runaway)"  # +cache hits only


def _sleeper(_spec, q):
    time.sleep(30)
    q.put(("COMPLETED", {}))


def test_timeout_does_not_jam():
    """The fork-subprocess + Queue.get(timeout) pattern the runner uses returns TIMEOUT, never hangs."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_sleeper, args=(None, q))
    p.start()
    t0 = time.monotonic()
    try:
        status = q.get(timeout=1.0)[0]
    except queue.Empty:
        status = "TIMEOUT"
    if p.is_alive():
        p.terminate()
    p.join(timeout=3)
    assert status == "TIMEOUT"
    assert time.monotonic() - t0 < 5.0                       # did not jam


# --- search regret computed correctly ----------------------------------------------------------------
def test_search_regret_computed_correctly():
    from types import SimpleNamespace
    nk_b = ActionBundle(precision_policy="fp8")              # the "best" bundle
    cache = {_key(ActionBundle()): 100.0, _key(nk_b): 150.0, _key(ActionBundle(clock_policy="high")): 120.0}
    results = {
        "m_good": SimpleNamespace(best_reward=150.0, best_bundle=nk_b, candidates_generated=3,
                                  candidates_evaluated=3, anchors_evaluated=True, evaluated_keys=set(cache)),
        "m_bad": SimpleNamespace(best_reward=120.0, best_bundle=ActionBundle(clock_policy="high"),
                                 candidates_generated=1, candidates_evaluated=1, anchors_evaluated=False,
                                 evaluated_keys={_key(ActionBundle(clock_policy="high"))}),
    }
    rep = compute_window_regret(results, cache, true_optimum=(150.0, _key(nk_b)))
    assert rep["reference_kind"] == "TRUE_EXHAUSTIVE"
    assert rep["per_method"]["m_good"]["regret_abs"] == 0.0
    assert rep["per_method"]["m_bad"]["regret_abs"] == 30.0
    assert rep["per_method"]["m_bad"]["regret_pct"] == 20.0
    # best-known reference when no exhaustive
    rep2 = compute_window_regret(results, cache, true_optimum=None)
    assert rep2["reference_kind"] == "NOT_TRUE_EXHAUSTIVE"
    assert rep2["per_method"]["m_good"]["regret_abs"] == 0.0


# --- exhaustive_small finds the true optimum on a tiny fixture ----------------------------------------
def test_exhaustive_small_finds_true_optimum():
    nk = named_anchor_keys()
    st = _state()
    target = ActionBundle(precision_policy="fp8", batching_policy="balanced", capacity_multiplier=0.75,
                          clock_policy="low")
    def score(b):
        return 999.0 if _key(b) == _key(target) else 1.0
    r = run_method("exhaustive_small", score, budget=200, state=st, named_keys=nk, seed=0)
    assert _key(r.best_bundle) == _key(target)
    assert r.extra.get("true_exhaustive") is True


# --- deterministic replay ----------------------------------------------------------------------------
def test_deterministic_replay():
    nk = named_anchor_keys()
    def score(b):
        return 100.0 + (10 if b.precision_policy == "fp8" else 0) + (8 if b.batching_policy == "aggressive" else 0)
    for m in ("beam_search", "cross_entropy", "simulated_annealing", "hybrid", "hierarchical_search"):
        a = run_method(m, score, budget=60, state=_state(), named_keys=nk, seed=0)
        b = run_method(m, score, budget=60, state=_state(), named_keys=nk, seed=0)
        assert _key(a.best_bundle) == _key(b.best_bundle), f"{m} not deterministic"
        assert a.candidates_evaluated == b.candidates_evaluated


def test_budgeted_scorer_cache_hits_are_free():
    bs = BudgetedScorer(lambda b: 1.0, budget=2)
    bs.score(ActionBundle())
    bs.score(ActionBundle())          # cache hit — no budget cost
    bs.score(ActionBundle(clock_policy="low"))
    assert len(bs.evaluated) == 2
    with pytest.raises(Exception):
        bs.score(ActionBundle(clock_policy="high"))   # 3rd distinct > budget 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
