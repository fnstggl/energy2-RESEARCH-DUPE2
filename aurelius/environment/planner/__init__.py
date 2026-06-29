"""Reusable MPC planner/search package — the search-method tournament layer.

A *diagnostic* layer (no simulator / reward / Pareto-gate / cost-model / baseline / action-semantics change)
that compares MPC search methods against each other on the SAME world model, workload, action space,
baselines, seed, and runtime/evaluation budgets — to identify the best planner architecture for maximizing
SLA-safe goodput/$. See `candidate_generators.py`, `search_methods.py`, `search_regret.py`,
`planner_tournament.py` and `research/MPC_SEARCH_METHOD_TOURNAMENT.md`.
"""

from .candidate_generators import (
    CORE_GRID_SURFACES,
    DIAGNOSTIC_WINNER,
    REGIMES,
    classify_regimes,
    core_grid,
    expanded_grid,
    physics_guided_candidates,
    random_candidates,
    required_anchors,
)

__all__ = [
    "REGIMES", "DIAGNOSTIC_WINNER", "CORE_GRID_SURFACES", "classify_regimes", "core_grid",
    "expanded_grid", "physics_guided_candidates", "random_candidates", "required_anchors",
]
