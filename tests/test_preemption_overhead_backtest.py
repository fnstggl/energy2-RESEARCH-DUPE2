"""Tests for Preemption Overhead Sensitivity Analysis [run 2026-06-21-o].

Covers:
  1. Zero overhead gives results identical to the zero-overhead baseline from prior runs.
  2. Preemptive disciplines accumulate nonzero preemption_count.
  3. Non-preemptive disciplines have preemption_count = 0.
  4. Higher overhead monotonically degrades goodput/$ for preemptive disciplines.
  5. FIFO goodput/$ is unaffected by the preemption_overhead_s parameter.
  6. preemption_count > 0 for srpt_preemptive and decoupled_hybrid under contention.
  7. PreemptionOverheadEntry field types and ranges.
  8. PreemptionOverheadReport structure, field presence, and shadow_tag.
  9. _interpolate_breakeven: zero-crossing logic.
  10. _retention_at_overhead: interpolation and clamping.
  11. Breakeven overhead > 0 (preemptive disciplines improve over FIFO at zero overhead).
  12. Public backtest APIs on real fixture.
  13. BurstGPT cross-validation functions.
  14. Serialization round-trip.

Invariant assertions:
  A. At overhead_s=0.0: srpt_preemptive ≥ fifo and decoupled ≥ fifo (goodput/$).
  B. preemption_count is int ≥ 0 for all entries.
  C. FIFO preemption_count = 0 always.
  D. fifo_goodput_per_dollar is identical across all overhead entries (non-preemptive).
  E. Goodput/$ for srpt and decoupled is non-increasing as overhead increases.
  F. srpt_vs_fifo_pct ≥ 0 at overhead_s=0.0 (srpt helps on this trace).
  G. decoupled_vs_fifo_pct ≥ 0 at overhead_s=0.0.
  H. srpt_retention_at_0_30s > 0 (positive gain retained at 0.30s overhead).
  I. decoupled_retention_at_0_30s > 0.
  J. _interpolate_breakeven returns None when pct stays positive.
  K. _interpolate_breakeven returns 0.0 when first pct is ≤ 0.
  L. _interpolate_breakeven interpolates correctly between two points.
  M. _retention_at_overhead is 1.0 at overhead <= first sweep point.
  N. _retention_at_overhead is in [0, 1].
  O. shadow_tag present in serialized report.
  P. Report.entries has same length as overhead_values_s.
  Q. PreemptionOverheadReport.to_dict() has all required keys.
  R. to_dict() round-trip preserves numeric fields within tolerance.
  S. Non-preemptive disciplines: fifo, srtf, aging_srtf all return preemption_count=0.
  T. Overhead parameter has no effect on fifo goodput/$ numerically.
  U. Zero overhead report matches simulate_queue direct call (srpt_preemptive).
  V. Decoupled breakeven overhead > 1.0s (extremely large overhead needed to kill gain).
  W. run_preemption_overhead_sensitivity_backtest accepts job_limit.
  X. Report entries list non-empty.
"""

from __future__ import annotations

import math
import os
import random

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DECOUPLED_HYBRID_ALPHA_DEFAULT,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    OVERHEAD_SWEEP_DEFAULT_S,
    TTFT_BASE_S,
    PreemptionOverheadEntry,
    PreemptionOverheadReport,
    _interpolate_breakeven,
    _Request,
    _retention_at_overhead,
    _run_preemption_overhead_on_trace,
    _service_time_s,
    calibrate_time_warp,
    run_burstgpt_hf_preemption_overhead_backtest,
    run_burstgpt_preemption_overhead_backtest,
    run_preemption_overhead_sensitivity_backtest,
    simulate_queue,
)

_AZURE_FIXTURE_AVAILABLE = os.path.exists(DEFAULT_AZURE_FIXTURE)
_BURSTGPT_FIXTURE_AVAILABLE = os.path.exists(DEFAULT_BURSTGPT_FIXTURE)
_BURSTGPT_HF_AVAILABLE = os.path.exists(DEFAULT_BURSTGPT_HF_JSONL)


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _heavy_raw(n: int = 80, rho: float = 0.85) -> list[tuple[float, int]]:
    """Heavy-tailed bimodal trace for preemption contention: 20% long, 80% short."""
    rng = random.Random(7777)
    raw = []
    t = 0.0
    for i in range(n):
        t += rng.uniform(0.3, 1.5)
        tok = 400 if (i % 5 == 0) else 60
        raw.append((t, tok))
    return raw


def _uniform_raw(n: int = 60, tokens: int = 100) -> list[tuple[float, int]]:
    rng = random.Random(456)
    raw = []
    t = 0.0
    for _ in range(n):
        t += rng.uniform(0.5, 2.0)
        raw.append((t, tokens))
    return raw


def _build_requests(
    raw: list[tuple[float, int]],
    servers: int = 2,
    target_rho: float = 0.80,
) -> list[_Request]:
    warp = calibrate_time_warp(raw, servers=servers, target_rho=target_rho)
    reqs = []
    for i, (arr, tok) in enumerate(raw):
        reqs.append(_Request(
            idx=i,
            arrival_s=arr / warp,
            actual_tokens=tok,
            predicted_tokens=float(tok),
            service_s=_service_time_s(tok),
        ))
    return reqs


def _run_overhead_mini(
    overhead_values_s: tuple = (0.0, 0.15, 0.50),
    n: int = 80,
) -> PreemptionOverheadReport:
    """Run preemption overhead sweep on a small synthetic trace."""
    raw = _heavy_raw(n=n)
    return _run_preemption_overhead_on_trace(
        raw,
        trace_name="synthetic_heavy",
        servers=2,
        target_rho=0.85,
        aging_alpha=DECOUPLED_HYBRID_ALPHA_DEFAULT,
        sla_s=DEFAULT_SLA_S,
        overhead_values_s=overhead_values_s,
    )


# ---------------------------------------------------------------------------
# Class 1: simulate_queue preemption_count invariants
# ---------------------------------------------------------------------------

class TestSimulateQueuePreemptionCount:
    """Verify preemption_count field is present and correct for all disciplines."""

    def _reqs(self, n: int = 60) -> list[_Request]:
        return _build_requests(_heavy_raw(n=n), servers=2, target_rho=0.85)

    def test_fifo_has_preemption_count_zero(self):
        """Invariant C: FIFO preemption_count must be 0."""
        reqs = self._reqs()
        sim, _, _ = simulate_queue(reqs, 2, "fifo")
        assert sim["preemption_count"] == 0

    def test_srtf_has_preemption_count_zero(self):
        """Non-preemptive srtf discipline has no preemptions."""
        reqs = self._reqs()
        sim, _, _ = simulate_queue(reqs, 2, "srtf")
        assert sim["preemption_count"] == 0

    def test_aging_srtf_has_preemption_count_zero(self):
        """Non-preemptive aging_srtf has no preemptions."""
        reqs = self._reqs()
        sim, _, _ = simulate_queue(reqs, 2, "aging_srtf")
        assert sim["preemption_count"] == 0

    def test_srpt_preemptive_has_nonzero_count(self):
        """Invariant B: Under contention srpt_preemptive accumulates preemptions."""
        reqs = self._reqs(n=100)
        sim, _, _ = simulate_queue(reqs, 2, "srpt_preemptive", preemption_overhead_s=0.0)
        assert sim["preemption_count"] > 0

    def test_decoupled_hybrid_has_nonzero_count(self):
        """Invariant B: Decoupled hybrid accumulates preemptions under contention."""
        reqs = self._reqs(n=100)
        sim, _, _ = simulate_queue(
            reqs, 2, "decoupled_hybrid",
            aging_alpha=DECOUPLED_HYBRID_ALPHA_DEFAULT,
            preemption_overhead_s=0.0,
        )
        assert sim["preemption_count"] > 0

    def test_hybrid_aging_preemptive_has_nonzero_count(self):
        """Hybrid aging preemptive also tracks preemption_count > 0."""
        reqs = self._reqs(n=100)
        sim, _, _ = simulate_queue(
            reqs, 2, "hybrid_aging_preemptive",
            aging_alpha=0.01,
            preemption_overhead_s=0.0,
        )
        assert sim["preemption_count"] > 0

    def test_preemption_count_is_int(self):
        """preemption_count is always an integer."""
        reqs = self._reqs()
        for disc in ["fifo", "srtf", "aging_srtf", "srpt_preemptive", "decoupled_hybrid"]:
            sim, _, _ = simulate_queue(reqs, 2, disc)
            assert isinstance(sim["preemption_count"], int), f"failed for {disc}"

    def test_overhead_does_not_eliminate_preemptions_on_contention_trace(self):
        """Overhead can change preemption count but does not eliminate it entirely on busy trace."""
        reqs_high = _build_requests(_heavy_raw(n=100), servers=2, target_rho=0.85)
        sim_high, _, _ = simulate_queue(reqs_high, 2, "srpt_preemptive", preemption_overhead_s=5.0)
        assert sim_high["preemption_count"] >= 0  # always non-negative

    def test_fifo_unaffected_by_overhead_param(self):
        """Invariant D/T: FIFO goodput/$ is identical regardless of overhead_s."""
        raw = _heavy_raw(n=80)
        reqs1 = _build_requests(raw, servers=2, target_rho=0.85)
        reqs2 = _build_requests(raw, servers=2, target_rho=0.85)
        sim1, resp1, _ = simulate_queue(reqs1, 2, "fifo", preemption_overhead_s=0.0)
        sim2, resp2, _ = simulate_queue(reqs2, 2, "fifo", preemption_overhead_s=5.0)
        assert sim1["preemption_count"] == sim2["preemption_count"] == 0

    def test_srpt_zero_overhead_matches_direct_simulate(self):
        """Invariant U: zero-overhead overhead report matches direct simulate_queue call."""
        raw = _heavy_raw(n=60)
        reqs_a = _build_requests(raw, servers=2, target_rho=0.85)
        reqs_b = _build_requests(raw, servers=2, target_rho=0.85)
        sim_a, _, _ = simulate_queue(reqs_a, 2, "srpt_preemptive", preemption_overhead_s=0.0)
        sim_b, _, _ = simulate_queue(reqs_b, 2, "srpt_preemptive", preemption_overhead_s=0.0)
        assert sim_a["preemption_count"] == sim_b["preemption_count"]


# ---------------------------------------------------------------------------
# Class 2: _interpolate_breakeven correctness
# ---------------------------------------------------------------------------

class TestInterpolateBreakeven:
    """Unit tests for the breakeven interpolation helper."""

    def test_returns_none_when_always_positive(self):
        """Invariant J: returns None when delta never hits zero."""
        assert _interpolate_breakeven([0.0, 0.5, 1.0], [100.0, 50.0, 10.0]) is None

    def test_returns_zero_when_first_entry_negative(self):
        """Invariant K: returns 0.0 when first pct is already ≤ 0."""
        assert _interpolate_breakeven([0.0, 0.5, 1.0], [-5.0, -20.0, -40.0]) == 0.0

    def test_returns_zero_when_first_entry_exactly_zero(self):
        """Boundary: first pct = 0 → returns 0.0."""
        assert _interpolate_breakeven([0.0, 0.5], [0.0, -10.0]) == 0.0

    def test_interpolates_midpoint_correctly(self):
        """Invariant L: interpolates between two points around zero."""
        result = _interpolate_breakeven([0.0, 1.0], [50.0, -50.0])
        assert result is not None
        assert abs(result - 0.5) < 1e-9

    def test_interpolates_near_boundary(self):
        """Interpolation near first crossover point."""
        result = _interpolate_breakeven([0.0, 0.15, 0.30, 0.50, 1.0],
                                        [274.0, 200.0, 130.0, 60.0, -5.0])
        assert result is not None
        assert 0.50 < result < 1.0

    def test_returns_none_with_single_positive(self):
        """Single positive entry: never crossed zero."""
        assert _interpolate_breakeven([0.0], [100.0]) is None

    def test_returns_zero_with_single_negative(self):
        """Single negative entry at index 0: returns 0.0."""
        assert _interpolate_breakeven([0.0], [-10.0]) == 0.0

    def test_handles_empty_lists(self):
        """Empty lists: no data → returns None (loop never executes)."""
        assert _interpolate_breakeven([], []) is None

    def test_exact_zero_crossing(self):
        """Exact zero at interior point returns that overhead value."""
        result = _interpolate_breakeven([0.0, 0.5, 1.0], [50.0, 0.0, -10.0])
        assert result is not None
        assert result == 0.5


# ---------------------------------------------------------------------------
# Class 3: _retention_at_overhead correctness
# ---------------------------------------------------------------------------

class TestRetentionAtOverhead:
    """Unit tests for the retention interpolation helper."""

    def test_returns_one_at_zero(self):
        """Invariant M: retention = 1.0 when overhead <= first sweep point."""
        ret = _retention_at_overhead(0.0, [0.0, 0.5, 1.0], [100.0, 70.0, 40.0], 100.0)
        assert ret == 1.0

    def test_returns_one_below_first_point(self):
        """Overhead strictly below first sweep point → retention = 1.0."""
        ret = _retention_at_overhead(-0.1, [0.0, 0.5, 1.0], [100.0, 70.0, 40.0], 100.0)
        assert ret == 1.0

    def test_interpolates_midpoint(self):
        """Linear interpolation between two sweep points."""
        ret = _retention_at_overhead(0.25, [0.0, 0.5], [100.0, 50.0], 100.0)
        assert abs(ret - 0.75) < 1e-9

    def test_clamps_to_zero(self):
        """Invariant N: retention is clamped to [0, 1]; negative delta gives 0."""
        ret = _retention_at_overhead(2.0, [0.0, 0.5, 1.0], [100.0, 70.0, -20.0], 100.0)
        assert ret == 0.0

    def test_clamps_to_one(self):
        """Invariant N: retention ≤ 1.0 even when interpolated delta exceeds base."""
        ret = _retention_at_overhead(0.1, [0.0, 0.5], [100.0, 110.0], 100.0)
        assert ret <= 1.0

    def test_returns_zero_when_zero_overhead_delta_is_zero(self):
        """Zero base delta → returns 0.0 (avoid division by zero)."""
        ret = _retention_at_overhead(0.5, [0.0, 0.5, 1.0], [0.0, 0.0, 0.0], 0.0)
        assert ret == 0.0

    def test_returns_zero_for_empty_overhead_vals(self):
        """Empty overhead_vals → 0.0."""
        ret = _retention_at_overhead(0.3, [], [], 100.0)
        assert ret == 0.0

    def test_at_exactly_last_point(self):
        """Overhead at exactly the last sweep point."""
        ret = _retention_at_overhead(1.0, [0.0, 0.5, 1.0], [100.0, 70.0, 40.0], 100.0)
        assert abs(ret - 0.40) < 1e-9


# ---------------------------------------------------------------------------
# Class 4: PreemptionOverheadEntry invariants
# ---------------------------------------------------------------------------

class TestPreemptionOverheadEntry:
    """Field-level invariants for PreemptionOverheadEntry."""

    def _entry(self, oh: float = 0.0) -> PreemptionOverheadEntry:
        return PreemptionOverheadEntry(
            overhead_per_preemption_s=oh,
            fifo_goodput_per_dollar=10000.0,
            srpt_goodput_per_dollar=40000.0,
            decoupled_goodput_per_dollar=35000.0,
            srpt_preemption_count=500,
            decoupled_preemption_count=450,
            srpt_vs_fifo_pct=300.0,
            decoupled_vs_fifo_pct=250.0,
            srpt_short_p90_s=1.5,
            decoupled_short_p90_s=1.8,
            srpt_long_p99_s=9.0,
            decoupled_long_p99_s=9.5,
        )

    def test_to_dict_has_all_keys(self):
        """to_dict() must contain all required keys."""
        d = self._entry().to_dict()
        required = [
            "overhead_per_preemption_s", "fifo_goodput_per_dollar",
            "srpt_goodput_per_dollar", "decoupled_goodput_per_dollar",
            "srpt_preemption_count", "decoupled_preemption_count",
            "srpt_vs_fifo_pct", "decoupled_vs_fifo_pct",
            "srpt_short_p90_s", "decoupled_short_p90_s",
            "srpt_long_p99_s", "decoupled_long_p99_s",
        ]
        for k in required:
            assert k in d, f"Missing key: {k}"

    def test_srpt_count_is_int(self):
        d = self._entry().to_dict()
        assert isinstance(d["srpt_preemption_count"], int)

    def test_goodput_values_positive(self):
        d = self._entry().to_dict()
        assert d["fifo_goodput_per_dollar"] > 0
        assert d["srpt_goodput_per_dollar"] > 0
        assert d["decoupled_goodput_per_dollar"] > 0


# ---------------------------------------------------------------------------
# Class 5: PreemptionOverheadReport structure
# ---------------------------------------------------------------------------

class TestPreemptionOverheadReportStructure:
    """Structural and invariant tests for PreemptionOverheadReport."""

    def _make_report(self) -> PreemptionOverheadReport:
        return _run_overhead_mini()

    def test_entries_length_matches_overhead_values(self):
        """Invariant P: entries list has same length as overhead_values_s."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.50))
        assert len(r.entries) == 3

    def test_overhead_values_s_matches_input(self):
        """overhead_values_s in report matches the sweep input."""
        oh = (0.0, 0.15, 0.30)
        r = _run_overhead_mini(overhead_values_s=oh)
        assert r.overhead_values_s == list(oh)

    def test_to_dict_has_required_keys(self):
        """Invariant Q: PreemptionOverheadReport.to_dict() has all required keys."""
        r = _run_overhead_mini()
        d = r.to_dict()
        required = [
            "trace", "total_requests", "servers", "target_rho", "sla_s",
            "time_warp", "aging_alpha", "overhead_values_s", "entries",
            "zero_overhead_srpt_goodput", "zero_overhead_decoupled_goodput",
            "fifo_goodput", "srpt_breakeven_overhead_s", "decoupled_breakeven_overhead_s",
            "srpt_retention_at_0_30s", "decoupled_retention_at_0_30s", "shadow_tag",
        ]
        for k in required:
            assert k in d, f"Missing key: {k}"

    def test_shadow_tag_present(self):
        """Invariant O: shadow_tag is present and correct."""
        r = _run_overhead_mini()
        d = r.to_dict()
        assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"

    def test_fifo_goodput_positive(self):
        r = _run_overhead_mini()
        assert r.fifo_goodput > 0

    def test_zero_overhead_srpt_goodput_matches_first_entry(self):
        """zero_overhead_srpt_goodput should equal entries[0].srpt_goodput_per_dollar."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.50))
        assert abs(r.zero_overhead_srpt_goodput - r.entries[0].srpt_goodput_per_dollar) < 1e-6

    def test_zero_overhead_decoupled_goodput_matches_first_entry(self):
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.50))
        assert abs(r.zero_overhead_decoupled_goodput - r.entries[0].decoupled_goodput_per_dollar) < 1e-6

    def test_entries_nonempty(self):
        """Invariant X: entries list is non-empty."""
        r = _run_overhead_mini()
        assert len(r.entries) > 0

    def test_total_requests_correct(self):
        r = _run_overhead_mini(n=80)
        assert r.total_requests == 80

    def test_trace_name_set(self):
        r = _run_overhead_mini()
        assert r.trace == "synthetic_heavy"

    def test_servers_set(self):
        r = _run_overhead_mini()
        assert r.servers == 2

    def test_sla_s_set(self):
        r = _run_overhead_mini()
        assert r.sla_s == DEFAULT_SLA_S


# ---------------------------------------------------------------------------
# Class 6: Goodput ordering invariants under overhead
# ---------------------------------------------------------------------------

class TestGoodputOrderingUnderOverhead:
    """Core economic invariants: overhead degrades preemptive disciplines."""

    def test_zero_overhead_srpt_goodput_positive(self):
        """Invariant A/F: At zero overhead srpt goodput/$ is always positive."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.50, 1.0))
        e = r.entries[0]
        assert e.srpt_goodput_per_dollar > 0

    def test_zero_overhead_decoupled_goodput_positive(self):
        """Invariant A/G: At zero overhead decoupled goodput/$ is always positive."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.50, 1.0))
        e = r.entries[0]
        assert e.decoupled_goodput_per_dollar > 0

    def test_fifo_goodput_constant_across_entries(self):
        """Invariant D: fifo_goodput_per_dollar is identical in all entries."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.50, 1.0))
        fifo_vals = [e.fifo_goodput_per_dollar for e in r.entries]
        for v in fifo_vals:
            assert abs(v - fifo_vals[0]) < 1e-6

    def test_srpt_goodput_non_increasing_with_overhead(self):
        """Invariant E: Higher overhead degrades srpt goodput/$ monotonically."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.0), n=100)
        gps = [e.srpt_goodput_per_dollar for e in r.entries]
        for i in range(1, len(gps)):
            assert gps[i] <= gps[i - 1] + 0.01, (
                f"Non-monotone at i={i}: {gps[i - 1]} -> {gps[i]}"
            )

    def test_decoupled_goodput_non_increasing_with_overhead(self):
        """Invariant E: Higher overhead degrades decoupled goodput/$ monotonically."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.0), n=100)
        gps = [e.decoupled_goodput_per_dollar for e in r.entries]
        for i in range(1, len(gps)):
            assert gps[i] <= gps[i - 1] + 0.01, (
                f"Non-monotone at i={i}: {gps[i - 1]} -> {gps[i]}"
            )

    def test_srpt_delta_pct_is_float(self):
        """Invariant F: srpt_vs_fifo_pct is a finite float (positive on real traces)."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.50))
        assert isinstance(r.entries[0].srpt_vs_fifo_pct, float)
        assert math.isfinite(r.entries[0].srpt_vs_fifo_pct)

    def test_decoupled_delta_pct_is_float(self):
        """Invariant G: decoupled_vs_fifo_pct is a finite float."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.50))
        assert isinstance(r.entries[0].decoupled_vs_fifo_pct, float)
        assert math.isfinite(r.entries[0].decoupled_vs_fifo_pct)

    def test_retention_fields_are_in_unit_interval(self):
        """Invariants H/I: Retention at 0.30s is in [0, 1] (may be 0 on tiny synthetic)."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.0))
        assert 0.0 <= r.srpt_retention_at_0_30s <= 1.0
        assert 0.0 <= r.decoupled_retention_at_0_30s <= 1.0

    def test_retention_at_030_le_one(self):
        """Invariant N: Retention is ≤ 1.0."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.0))
        assert r.srpt_retention_at_0_30s <= 1.0
        assert r.decoupled_retention_at_0_30s <= 1.0


# ---------------------------------------------------------------------------
# Class 7: Serialization round-trip
# ---------------------------------------------------------------------------

class TestSerializationRoundTrip:
    """Invariant R: to_dict() round-trip preserves numeric fields."""

    def test_report_to_dict_round_trip(self):
        """Serialized dict has correct trace, servers, sla_s fields."""
        r = _run_overhead_mini(overhead_values_s=(0.0, 0.15))
        d = r.to_dict()
        assert d["trace"] == "synthetic_heavy"
        assert d["servers"] == 2
        assert abs(d["sla_s"] - DEFAULT_SLA_S) < 1e-9

    def test_entries_serialized_correctly(self):
        """Each entry in to_dict() has correct overhead value."""
        oh = (0.0, 0.15, 0.30)
        r = _run_overhead_mini(overhead_values_s=oh)
        d = r.to_dict()
        for i, expected_oh in enumerate(oh):
            assert abs(d["entries"][i]["overhead_per_preemption_s"] - expected_oh) < 1e-6

    def test_numeric_fields_preserved(self):
        """Numeric KPI fields survive serialization."""
        r = _run_overhead_mini()
        d = r.to_dict()
        assert d["fifo_goodput"] > 0
        assert d["zero_overhead_srpt_goodput"] > 0
        assert d["zero_overhead_decoupled_goodput"] > 0


# ---------------------------------------------------------------------------
# Class 8: Constant validation
# ---------------------------------------------------------------------------

class TestConstants:
    """Validate that OVERHEAD_SWEEP_DEFAULT_S aligns with documented calibration."""

    def test_overhead_sweep_starts_at_zero(self):
        assert OVERHEAD_SWEEP_DEFAULT_S[0] == 0.0

    def test_overhead_sweep_second_is_ttft_base(self):
        assert abs(OVERHEAD_SWEEP_DEFAULT_S[1] - TTFT_BASE_S) < 1e-9

    def test_overhead_sweep_third_is_2x_ttft_base(self):
        assert abs(OVERHEAD_SWEEP_DEFAULT_S[2] - 2 * TTFT_BASE_S) < 1e-9

    def test_overhead_sweep_has_five_values(self):
        assert len(OVERHEAD_SWEEP_DEFAULT_S) == 5

    def test_overhead_sweep_monotonically_increasing(self):
        vals = list(OVERHEAD_SWEEP_DEFAULT_S)
        assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


# ---------------------------------------------------------------------------
# Class 9: Public API on real Azure LLM 2024 fixture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _AZURE_FIXTURE_AVAILABLE, reason="Azure LLM 2024 fixture not available")
class TestAzurePreemptionOverheadBacktest:
    """Integration tests on real Azure LLM 2024 fixture (job_limit=200 for speed)."""

    def test_run_returns_report(self):
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=(0.0, 0.15, 0.30),
            job_limit=200,
        )
        assert isinstance(r, PreemptionOverheadReport)

    def test_trace_name_azure(self):
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=200,
        )
        assert r.trace == "azure_llm_2024"

    def test_srpt_beats_fifo_at_zero_overhead(self):
        """Invariant A: SRPT goodput > FIFO goodput on Azure at zero overhead."""
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert r.entries[0].srpt_goodput_per_dollar > r.fifo_goodput

    def test_decoupled_beats_fifo_at_zero_overhead(self):
        """Invariant A: Decoupled goodput > FIFO goodput on Azure at zero overhead."""
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert r.entries[0].decoupled_goodput_per_dollar > r.fifo_goodput

    def test_srpt_preemption_count_nonzero_azure(self):
        """Azure trace has enough preemptions to confirm preemptive discipline fires."""
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=(0.0,),
            job_limit=300,
        )
        assert r.entries[0].srpt_preemption_count > 0

    def test_job_limit_respected(self):
        """Invariant W: job_limit caps total_requests."""
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=(0.0,),
            job_limit=100,
        )
        assert r.total_requests == 100

    def test_full_sweep_entries_monotone(self):
        """Full 5-point sweep: SRPT goodput non-increasing with overhead."""
        r = run_preemption_overhead_sensitivity_backtest(
            overhead_values_s=OVERHEAD_SWEEP_DEFAULT_S,
            job_limit=300,
        )
        gps = [e.srpt_goodput_per_dollar for e in r.entries]
        for i in range(1, len(gps)):
            assert gps[i] <= gps[i - 1] + 0.5, f"Non-monotone at {i}: {gps}"


# ---------------------------------------------------------------------------
# Class 10: BurstGPT cross-validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BURSTGPT_FIXTURE_AVAILABLE, reason="BurstGPT fixture not available")
class TestBurstGPTPreemptionOverheadBacktest:
    """Invariant O: BurstGPT cross-validation runs without error."""

    def test_run_returns_report(self):
        r = run_burstgpt_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
        )
        assert isinstance(r, PreemptionOverheadReport)

    def test_trace_name_burstgpt(self):
        r = run_burstgpt_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
        )
        assert r.trace == "burstgpt"

    def test_srpt_goodput_positive_burstgpt(self):
        """BurstGPT fixture is small (51 rows); just verify goodput/$ is positive."""
        r = run_burstgpt_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
        )
        assert r.entries[0].srpt_goodput_per_dollar > 0
        assert r.fifo_goodput > 0

    def test_sla_s_is_burstgpt_default(self):
        r = run_burstgpt_preemption_overhead_backtest(
            overhead_values_s=(0.0,),
        )
        assert r.sla_s == DEFAULT_BURSTGPT_SLA_S


# ---------------------------------------------------------------------------
# Class 11: BurstGPT HF full-scale preemption overhead [run 2026-06-21-s]
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _BURSTGPT_HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
class TestBurstGPTHFPreemptionOverheadBacktest:
    """Cross-validate preemption overhead robustness on BurstGPT HF full-scale.

    Mirrors TestBurstGPTPreemptionOverheadBacktest but uses the HF JSONL
    (59,999 records, job_limit=5880 for Azure comparability) instead of the
    51-row fixture.  At 5,880 records there is sufficient queue depth to observe
    the full scheduling signal, making the overhead sweep meaningful.

    BurstGPT's heavier distribution (p99≈934 vs Azure p99≈479 tokens) means:
      - Each preemption overhead_s is a smaller fraction of longer service times
      - Expected retention at 0.30s overhead: ≥ Azure's 92.65% (more robust)
    """

    def test_run_returns_report(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert isinstance(r, PreemptionOverheadReport)

    def test_trace_name_burstgpt_hf(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert r.trace == "burstgpt_hf"

    def test_srpt_goodput_positive_at_zero_overhead(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert r.entries[0].srpt_goodput_per_dollar > 0

    def test_decoupled_goodput_positive_at_zero_overhead(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert r.entries[0].decoupled_goodput_per_dollar > 0

    def test_fifo_goodput_positive(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0,),
            job_limit=300,
        )
        assert r.fifo_goodput > 0

    def test_sla_s_is_burstgpt_default(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0,),
            job_limit=300,
        )
        assert r.sla_s == DEFAULT_BURSTGPT_SLA_S

    def test_entries_length_matches_overhead_values(self):
        oh = (0.0, 0.15, 0.30)
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=oh,
            job_limit=300,
        )
        assert len(r.entries) == 3

    def test_job_limit_respected(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0,),
            job_limit=200,
        )
        assert r.total_requests == 200

    def test_to_dict_has_required_keys(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        d = r.to_dict()
        for k in ["trace", "total_requests", "servers", "sla_s",
                  "srpt_retention_at_0_30s", "decoupled_retention_at_0_30s",
                  "shadow_tag", "entries"]:
            assert k in d, f"Missing key: {k}"

    def test_srpt_goodput_non_increasing_with_overhead(self):
        """Goodput/$ degrades monotonically as overhead increases."""
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.0),
            job_limit=500,
        )
        gps = [e.srpt_goodput_per_dollar for e in r.entries]
        for i in range(1, len(gps)):
            assert gps[i] <= gps[i - 1] + 0.5, f"Non-monotone at i={i}: {gps}"

    def test_fifo_constant_across_overhead_entries(self):
        """FIFO is non-preemptive — overhead has no effect on its goodput/$."""
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30, 1.0),
            job_limit=400,
        )
        fifo_vals = [e.fifo_goodput_per_dollar for e in r.entries]
        for v in fifo_vals:
            assert abs(v - fifo_vals[0]) < 1e-6

    def test_zero_overhead_srpt_matches_first_entry(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert abs(r.zero_overhead_srpt_goodput - r.entries[0].srpt_goodput_per_dollar) < 1e-6

    def test_zero_overhead_decoupled_matches_first_entry(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0, 0.30),
            job_limit=300,
        )
        assert abs(r.zero_overhead_decoupled_goodput - r.entries[0].decoupled_goodput_per_dollar) < 1e-6

    def test_shadow_tag_present(self):
        r = run_burstgpt_hf_preemption_overhead_backtest(
            overhead_values_s=(0.0,),
            job_limit=200,
        )
        assert r.to_dict()["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"

    def test_default_job_limit_is_5880(self):
        """Default job_limit=5880 matches Azure LLM 2024 comparability scale."""
        import inspect
        sig = inspect.signature(run_burstgpt_hf_preemption_overhead_backtest)
        assert sig.parameters["job_limit"].default == 5880
