"""Tests for the unified replay engine (Phase 1b-A) — the closed joint loop.

Verifies the engine is a genuine single discrete-event loop on one evolving state
(capacity reacts to the live backlog ordering+admission shape), is deterministic,
prices on the on-demand denominator, respects workload-class semantics, and —
the headline — produces compounding ONLY when the data carries class structure.
"""

from __future__ import annotations

import pytest

from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.unified_replay import (
    CLASS_BEST_EFFORT,
    CLASS_LATENCY,
    Job,
    UnifiedLatticeResult,
    run_unified_combination,
    run_unified_replay,
)


def _burst_jobs(n=1500, be_fraction=0.0):
    """Synthetic trace with a mid-run arrival burst; optional best-effort tier."""
    n_be = int(be_fraction * n)
    be_stride = max(1, n // n_be) if n_be > 0 else 0
    out = []
    t = 0.0
    for i in range(n):
        # dense burst in the middle forces a real queue
        t += 0.25 if 500 < i < 900 else 1.2
        tok = 120 + (i % 5) * 40
        cls = CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY
        out.append(Job(idx=i, arrival_s=t, actual_tokens=tok, predicted_tokens=float(tok),
                       service_s=0.4 + tok * 0.01, cls=cls))
    return out


def test_run_unified_replay_deterministic_and_priced_on_demand():
    jobs = _burst_jobs()
    a = run_unified_replay(jobs, tick_seconds=30.0, sla_s=10.0, capacity="backlog_aware")
    b = run_unified_replay(jobs, tick_seconds=30.0, sla_s=10.0, capacity="backlog_aware")
    assert a.to_dict() == b.to_dict()
    # on-demand denominator: cost == sum(c[t]) * tick_hr * $2
    assert a.cost_usd == pytest.approx(sum(a.c_trace) * 30.0 / 3600.0 * 2.0, rel=1e-9)
    assert a.goodput_per_dollar > 0


def test_closed_loop_capacity_reacts_to_backlog():
    """The defining property: backlog_aware capacity provisions MORE under a burst
    than the reactive baseline does — i.e. capacity genuinely reacts to the live
    queue state inside the loop (not a precomputed schedule)."""
    jobs = _burst_jobs()
    reactive = run_unified_replay(jobs, tick_seconds=30.0, sla_s=10.0, capacity="reactive_lag1")
    backlog = run_unified_replay(jobs, tick_seconds=30.0, sla_s=10.0, capacity="backlog_aware")
    # the backlog-aware controller adapts c to the observed queue → peak c differs
    assert backlog.c_max >= reactive.c_max
    assert backlog.c_trace != reactive.c_trace


def test_latency_critical_is_never_deferred():
    jobs = _burst_jobs()  # all latency-critical
    k = run_unified_replay(jobs, tick_seconds=30.0, sla_s=10.0, admission="class_aware")
    assert k.n_deferred == 0  # admission must never gate latency-critical load


def test_combination_lattice_shape_and_objective_ranking():
    jobs = _burst_jobs()
    res = run_unified_combination(jobs, tick_seconds=30.0, sla_s=10.0, trace_id="unit")
    assert isinstance(res, UnifiedLatticeResult)
    assert len(res.cells) == 8
    assert res.denominator == "on_demand"
    labels = {c.label for c in res.cells}
    assert {"base", "C", "O", "A", "C+O", "C+A", "O+A", "C+O+A"} == labels
    assert res.interaction in {"compounding", "substitutive", "neutral"}
    # best_overall is ranked by the ObjectiveLayer → at least as good as base
    assert res.best_overall_gpd >= res.base_gpd


def test_optimizer_exposes_closed_loop_method():
    jobs = _burst_jobs()
    res = AureliusOptimizer().optimize_joint_closed_loop(
        jobs, tick_seconds=30.0, sla_s=10.0, trace_id="unit")
    assert isinstance(res, UnifiedLatticeResult)


def test_multi_class_unlocks_admission_vs_single_class():
    """Headline mechanism on a synthetic trace: admission is inert on a single-class
    workload (nothing legal to defer) but active on a multi-class one — the data
    structure, not the optimizer, gates the lever."""
    single = run_unified_combination(_burst_jobs(be_fraction=0.0),
                                     tick_seconds=30.0, sla_s=10.0, trace_id="single")
    multi = run_unified_combination(_burst_jobs(be_fraction=0.4),
                                    tick_seconds=30.0, sla_s=10.0, trace_id="multi")
    # single-class: no best-effort exists, so admission defers nothing
    assert single.n_best_effort == 0
    assert all(c.n_deferred == 0 for c in single.cells)
    # multi-class: best-effort exists and admission-on cells defer some of it
    assert multi.n_best_effort > 0
    assert any(c.n_deferred > 0 for c in multi.cells if c.admission != "off")
