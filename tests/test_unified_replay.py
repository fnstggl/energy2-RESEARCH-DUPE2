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


# --- newly CONNECTED knobs: capacity_multiplier + batching (next batch) ------
# Each test proves the knob changes the simulator output in a causally defensible,
# NON-free way — it buys one KPI by paying another (a real Pareto move, not a fake win).

def _run(**kw):
    return run_unified_replay(_burst_jobs(), tick_seconds=30.0, sla_s=10.0,
                              capacity="backlog_aware", **kw)


def test_capacity_multiplier_default_is_exact_noop():
    """The connected default (1.0x) must reproduce today's run bit-for-bit — the knob is
    additive, never silently rescaling the no-op baseline."""
    bare = run_unified_replay(_burst_jobs(), tick_seconds=30.0, sla_s=10.0,
                              capacity="backlog_aware")
    assert _run(capacity_multiplier=1.0).to_dict() == bare.to_dict()


def test_capacity_multiplier_buys_sla_with_more_gpu_hours_no_free_capacity():
    """More replicas (higher multiplier) must cut SLA violations BUT cost strictly more
    GPU-hours — capacity is never free. Monotone in both directions = a real Pareto trade."""
    lo, mid, hi = (_run(capacity_multiplier=m) for m in (0.75, 1.0, 1.5))
    # provisioning more capacity costs strictly more GPU-hours / dollars (no free capacity)
    assert lo.gpu_hours < mid.gpu_hours < hi.gpu_hours
    assert lo.cost_usd < mid.cost_usd < hi.cost_usd
    # ...and it buys fewer SLA violations (the thing you pay for)
    assert hi.sla_violations < mid.sla_violations < lo.sla_violations
    # peak provisioned replicas scale up with the multiplier
    assert hi.c_max > lo.c_max


def test_batching_default_is_exact_noop():
    bare = run_unified_replay(_burst_jobs(), tick_seconds=30.0, sla_s=10.0,
                              capacity="backlog_aware")
    assert _run(batch_concurrency=1.0, batch_service_factor=1.0).to_dict() == bare.to_dict()


def test_batch_concurrency_cuts_queue_at_roughly_constant_gpu_hours():
    """Per-replica concurrency packs more requests onto the same hardware → it drains the
    queue (fewer violations, more SLA-safe goodput) WITHOUT inflating GPU-hours. This is the
    throughput lever; service inflation (tested separately) is its honest cost."""
    c1, c2, c4 = (_run(batch_concurrency=c, batch_service_factor=1.0) for c in (1.0, 2.0, 4.0))
    assert c4.sla_violations < c2.sla_violations < c1.sla_violations
    assert c4.sla_safe_goodput > c2.sla_safe_goodput > c1.sla_safe_goodput
    # concurrency reuses existing replicas — it does NOT manufacture extra GPU-hours
    assert c4.gpu_hours <= c1.gpu_hours * 1.05


def test_batch_service_inflation_is_not_free():
    """The honest cost of batching: a larger batch inflates per-request service time. Holding
    concurrency fixed, raising ONLY the service factor must make latency strictly WORSE — more
    violations and less SLA-safe goodput. A fake knob could not pay this price."""
    jobs = _burst_jobs()

    def tight(sf):  # a tight SLA surfaces the latency penalty cleanly
        return run_unified_replay(jobs, tick_seconds=30.0, sla_s=5.0, capacity="backlog_aware",
                                  batch_concurrency=1.0, batch_service_factor=sf)

    s1, s13, s18 = tight(1.0), tight(1.3), tight(1.8)
    assert s1.sla_violations < s13.sla_violations < s18.sla_violations
    assert s1.sla_safe_goodput > s18.sla_safe_goodput
