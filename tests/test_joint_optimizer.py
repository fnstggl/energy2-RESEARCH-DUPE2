"""Tests for the joint-optimization combination search (the path to compounding).

Verifies the joint loop composes the deployable serving levers (capacity /
ordering / admission) on one trace, measures the interaction honestly, prices on
the on-demand denominator, and is deterministic + reproducible.
"""

from __future__ import annotations

from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.joint import (
    JointResult,
    combination_search,
    peak_shave_admission,
)


def _trace(n=1200):
    out = []
    for i in range(n):
        tok = 100 + (i % 6) * 50 + (250 if 300 < i < 500 else 0)
        out.append((float(i) * 1.5, tok))
    return out


def test_peak_shave_admission_preserves_requests_and_smooths():
    raw = _trace()
    shaped = peak_shave_admission(raw, 60.0, 1.0, threshold=1.2)
    # no request dropped (flow control defers, never drops)
    assert len(shaped) == len(raw)
    # same multiset of token counts
    assert sorted(t for _, t in shaped) == sorted(t for _, t in raw)
    # arrivals remain sorted/non-negative
    arrs = [a for a, _ in shaped]
    assert arrs == sorted(arrs) and arrs[0] >= 0


def test_combination_search_runs_full_lattice_on_demand():
    raw = _trace()
    res = combination_search(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0, seed=42, trace_id="unit")
    assert isinstance(res, JointResult)
    # 2x2x2 lattice = 8 cells, all priced on-demand
    assert len(res.cells) == 8
    assert res.denominator == "on_demand"
    labels = {c.label for c in res.cells}
    assert {"base", "C", "O", "A", "C+O", "C+A", "O+A", "C+O+A"} == labels
    # interaction verdict is one of the three honest categories
    assert res.interaction in {"compounding", "substitutive", "neutral"}
    # best overall is at least as good as the base
    assert res.best_overall_gpd >= res.base_gpd


def test_combination_search_deterministic():
    raw = _trace()
    a = combination_search(raw, tick_seconds=60.0, warp=1.0, sla_s=10.0,
                           seed=1, trace_id="unit")
    b = combination_search(raw, tick_seconds=60.0, warp=1.0, sla_s=10.0,
                           seed=1, trace_id="unit")
    assert a.trace_hash == b.trace_hash
    assert [round(c.goodput_per_dollar, 6) for c in a.cells] == \
           [round(c.goodput_per_dollar, 6) for c in b.cells]


def test_optimizer_exposes_optimize_joint():
    raw = _trace()
    res = AureliusOptimizer().optimize_joint(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0, trace_id="unit")
    assert isinstance(res, JointResult)
    # the compounding flag is consistent with the measured margin
    if res.compounding:
        assert res.best_overall_gpd > res.best_single_gpd
