"""Tests for the canonical production-like dataset assembler + signal matrix.

Verifies the assembler is deterministic, preserves the real spine, labels the
synthetic overlay honestly, and that the signal-matrix audit is internally
consistent and never quietly overstates fidelity.
"""

from __future__ import annotations

from aurelius.datasets import (
    CANONICAL_SIGNAL_MATRIX,
    augment_with_best_effort,
    coverage_by_lever,
    coverage_by_tier,
    realizable_today,
    simulator_or_absent,
    to_jobs,
)
from aurelius.datasets.signal_matrix import (
    TIER_ABSENT,
    TIER_MEASURED,
    TIER_PROXY,
    TIER_SIMULATOR,
    TIER_SYNTHETIC,
)
from aurelius.optimizer.unified_replay import CLASS_BEST_EFFORT, CLASS_LATENCY


def _raw(n=400):
    return [(float(i) * 1.5, 100 + (i % 7) * 30) for i in range(n)]


def test_to_jobs_warps_arrivals_and_sets_class():
    raw = _raw()
    jobs = to_jobs(raw, warp=2.0, cls=CLASS_LATENCY)
    assert len(jobs) == len(raw)
    assert all(j.cls == CLASS_LATENCY for j in jobs)
    # arrival divided by warp into sim time
    assert jobs[10].arrival_s == raw[10][0] / 2.0


def test_augment_is_deterministic_and_preserves_spine():
    raw = _raw()
    a, man_a = augment_with_best_effort(raw, warp=2.0, fraction=0.4)
    b, _ = augment_with_best_effort(raw, warp=2.0, fraction=0.4)

    def _key(jobs):
        return [(j.idx, j.arrival_s, j.actual_tokens, j.cls) for j in jobs]

    assert _key(a) == _key(b)
    # spine preserved exactly as latency-critical
    spine = [j for j in a if j.cls == CLASS_LATENCY]
    assert len(spine) == len(raw)
    # overlay labeled best-effort and counted in the manifest
    overlay = [j for j in a if j.cls == CLASS_BEST_EFFORT]
    assert man_a.n_best_effort == len(overlay)
    assert man_a.overlay_tier == "SYNTHETIC"
    assert round(len(overlay) / len(raw), 1) == 0.4


def test_overlay_tokens_resampled_from_real_spine_distribution():
    raw = _raw()
    jobs, _ = augment_with_best_effort(raw, warp=1.0, fraction=0.5, token_multiplier=1.0)
    spine_tokens = {tok for _, tok in raw}
    overlay = [j for j in jobs if j.cls == CLASS_BEST_EFFORT]
    # every overlay token (mult=1.0) is a real spine token value (no invented tokens)
    assert all(j.actual_tokens in spine_tokens for j in overlay)


def test_signal_matrix_tiers_are_valid_and_audited():
    valid = {TIER_MEASURED, TIER_PROXY, TIER_SYNTHETIC, TIER_SIMULATOR, TIER_ABSENT}
    assert all(s.tier in valid for s in CANONICAL_SIGNAL_MATRIX)
    cov = coverage_by_tier()
    assert sum(cov.values()) == len(CANONICAL_SIGNAL_MATRIX)
    # the audit must be honest: at least one signal is simulator-only or absent
    assert cov[TIER_SIMULATOR] + cov[TIER_ABSENT] >= 1
    # and the admission-unlocking workload_class lever is realizable today
    levers = coverage_by_lever()
    assert "admission" in levers


def test_realizable_vs_ceiling_partition():
    realizable = set(s.name for s in realizable_today())
    ceiling = set(s.name for s in simulator_or_absent())
    # no signal is both realizable-today and on the hard ceiling
    assert realizable.isdisjoint(ceiling)
    # the spine signals are realizable; fabric congestion is on the ceiling
    assert "arrival_time" in realizable
    assert "fabric_congestion" in ceiling
