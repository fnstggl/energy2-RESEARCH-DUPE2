"""`hierarchical_search` as a selectable Aurelius planner mode (Phase B) — the contract tests.

These pin: the controller exposes the package planner modes (default OFF → behaviour-preserving); the
hierarchical search is anchor-contained, bounded (runtime-capped), and reports the search counts the gate
reads; and the planner package is the Aurelius optimiser path that imports NOTHING from the production
baseline (the reverse separation to `test_production_scheduler_baseline`).
"""

from __future__ import annotations

import ast
import os

from aurelius.environment.physics_guided_candidates import PlannerRegimeState
from aurelius.environment.planner.candidate_generators import named_anchor_keys
from aurelius.environment.planner.search_methods import run_method

_PLANNER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "aurelius", "environment", "planner")


def _state():
    return PlannerRegimeState(decode_regime="memory_bound", capacity_pressure=0.5, price_percentile=0.8,
                              output_token_mean=160, confidence=0.8)


def _score(b):
    # deterministic, world-free reward with a gradient over the connected surfaces so the search has signal.
    return (float(getattr(b, "capacity_multiplier", 1.0))
            + (0.2 if b.routing_policy == "kv_aware" else 0.0)
            + (0.1 if b.placement_policy == "rack_local" else 0.0))


# ---- controller wiring: modes exposed, default OFF -------------------------------------------------
def test_controller_exposes_package_planner_modes_default_off():
    from aurelius.environment.controller import (
        PACKAGE_PLANNER_MODES,
        ModelPredictiveEconomicController,
    )
    assert "hierarchical_search" in PACKAGE_PLANNER_MODES
    assert "hierarchical_search_with_progressive_widening" in PACKAGE_PLANNER_MODES
    # the dataclass default leaves planner_mode OFF → existing benchmarks are behaviour-preserving.
    c = ModelPredictiveEconomicController.__dataclass_fields__["planner_mode"]
    assert c.default is None


def test_mode_to_method_maps_hierarchical():
    from aurelius.environment.controller import _MODE_TO_METHOD
    assert _MODE_TO_METHOD["hierarchical_search"] == "hierarchical_search"
    assert _MODE_TO_METHOD["hierarchical_search_with_progressive_widening"] == "hierarchical_search"
    assert _MODE_TO_METHOD["exhaustive_small_diagnostic"] == "exhaustive_small"


def test_default_benchmark_planner_is_hierarchical_and_wirable():
    """The gate PASSED → hierarchical_search is the DEFAULT benchmark planner (declared constant), and the
    benchmark controller builder accepts it (the flip is wired, not merely opt-in)."""
    import inspect

    from aurelius.environment.controller import (
        DEFAULT_BENCHMARK_PLANNER_MODE,
        PACKAGE_PLANNER_MODES,
    )
    from aurelius.environment.training import _controller
    assert DEFAULT_BENCHMARK_PLANNER_MODE == "hierarchical_search"
    assert DEFAULT_BENCHMARK_PLANNER_MODE in PACKAGE_PLANNER_MODES
    # `_controller` exposes a planner_mode hook (default None = behaviour-preserving for existing callers).
    sig = inspect.signature(_controller)
    assert "planner_mode" in sig.parameters
    assert sig.parameters["planner_mode"].default is None


# ---- the search itself: anchor-contained, bounded, reports counts ----------------------------------
def test_hierarchical_contains_anchors():
    r = run_method("hierarchical_search", _score, budget=100, state=_state(), named_keys=named_anchor_keys(None))
    assert r.anchors_evaluated is True                       # the named anchors are always in the evaluated set


def test_hierarchical_runtime_bounded_by_budget():
    for budget in (10, 25, 100):
        r = run_method("hierarchical_search", _score, budget=budget, state=_state(),
                       named_keys=named_anchor_keys(None))
        assert r.candidates_evaluated <= budget             # never exceeds the DISTINCT-evaluation budget
        assert r.candidates_evaluated >= 1


def test_hierarchical_reports_counts_for_the_gate():
    r = run_method("hierarchical_search", _score, budget=100, state=_state(), named_keys=named_anchor_keys(None))
    d = r.to_dict()
    for k in ("method", "candidates_generated", "candidates_evaluated", "total_score_calls",
              "anchors_evaluated", "best_reward"):
        assert k in d
    assert r.method == "hierarchical_search"
    assert isinstance(r.evaluated_keys, set) and r.evaluated_keys   # the set the regret auditor checks


def test_hierarchical_reaches_connected_surfaces():
    """The PR #123 "reach" property: hierarchical evaluates bundles that vary the CONNECTED surfaces
    (placement / capacity-policy / admission / routing) — not just the core grid. This is why it beats a
    core-grid search."""
    r = run_method("hierarchical_search", _score, budget=100, state=_state(), named_keys=named_anchor_keys(None))
    connected_fields = {"placement_policy", "capacity_policy", "admission_policy", "routing_policy"}
    # at least one evaluated bundle sets a connected surface away from its core-grid no-op default.
    reached = False
    for key in r.evaluated_keys:                              # each key is a sorted tuple of (field, value)
        kd = dict(key)
        if (kd.get("placement_policy", "topology_blind") != "topology_blind"
                or kd.get("capacity_policy", "reactive_lag1") != "reactive_lag1"
                or kd.get("admission_policy", "off") != "off"
                or kd.get("routing_policy", "round_robin") != "round_robin"):
            reached = True
            break
    assert reached, f"hierarchical did not reach any connected surface in {connected_fields}"


def test_hierarchical_deterministic():
    a = run_method("hierarchical_search", _score, budget=100, state=_state(), named_keys=named_anchor_keys(None))
    b = run_method("hierarchical_search", _score, budget=100, state=_state(), named_keys=named_anchor_keys(None))
    assert a.best_reward == b.best_reward
    assert a.best_bundle.to_dict() == b.best_bundle.to_dict()


# ---- separation: the planner package never imports the production baseline --------------------------
def test_planner_package_does_not_import_production_baselines():
    """The reverse of the production-baseline separation: the Aurelius optimiser (planner package) must not
    reach into the evaluation-layer baseline. They are independent arms."""
    leaks = []
    for fn in os.listdir(_PLANNER_DIR):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(_PLANNER_DIR, fn)).read())
        for node in ast.walk(tree):
            mods = ([n.name for n in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            leaks += [f"{fn}:{m}" for m in mods if "production_baselines" in (m or "")]
    assert not leaks, f"planner package leaked a production_baselines import: {leaks}"
