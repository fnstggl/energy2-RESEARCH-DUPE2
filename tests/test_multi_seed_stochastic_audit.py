"""Tests for multi-seed stochastic audit — Five-Failure Rule benchmark realism.

Run 2026-06-24.  Validates the multi-seed audit infrastructure without
running full-scale benchmarks (uses 2-seed subsets for speed).

Research goal: characterise whether the BurstGPT n_sla_safe gap
(OSOTSS 5849 vs AMCSG 5864) is structural or a seed artifact.
"""

from __future__ import annotations

import pytest

from aurelius.benchmarks.multi_seed_stochastic_audit import (
    AUDIT_SEEDS,
    CANONICAL_AMCSG_AZURE_N_SLA_SAFE,
    CANONICAL_AMCSG_BURSTGPT_N_SLA_SAFE,
    CANONICAL_OSOTSS_AZURE_N_SLA_SAFE,
    CANONICAL_OSOTSS_BURSTGPT_N_SLA_SAFE,
    MultiSeedAuditReport,
    SeedResult,
    TraceAuditSummary,
    _summarize_trace,
    run_multi_seed_azure_audit,
    run_multi_seed_burstgpt_audit,
)


# ---------------------------------------------------------------------------
# Unit tests — _summarize_trace (pure logic, no I/O)
# ---------------------------------------------------------------------------

def _make_seed_result(seed: int, amcsg: int, osotss: int, trace: str = "test") -> SeedResult:
    gap = osotss - amcsg
    pct = 100.0 * (osotss - amcsg) / amcsg if amcsg > 0 else 0.0
    return SeedResult(
        trace=trace,
        seed=seed,
        amcsg_n_sla_safe=amcsg,
        osotss_n_sla_safe=osotss,
        amcsg_goodput_per_dollar=float(amcsg) * 10.0,
        osotss_goodput_per_dollar=float(osotss) * 10.0,
        gap_n_sla_safe=gap,
        osotss_vs_amcsg_pct=pct,
    )


def test_summarize_structural_gap():
    """When OSOTSS always < AMCSG, gap is structural."""
    results = [
        _make_seed_result(42, amcsg=5864, osotss=5849),
        _make_seed_result(123, amcsg=5862, osotss=5845),
        _make_seed_result(456, amcsg=5860, osotss=5847),
    ]
    summary = _summarize_trace("burstgpt_hf", results)
    assert summary.goodput_gap_is_structural is True
    assert summary.goodput_gap_is_noise is False
    assert summary.seeds_osotss_wins == 0
    assert summary.seeds_osotss_loses == 3


def test_summarize_noise_gap():
    """When OSOTSS wins on at least one seed, gap is noise."""
    results = [
        _make_seed_result(42, amcsg=5864, osotss=5849),    # lose
        _make_seed_result(123, amcsg=5862, osotss=5865),   # win
        _make_seed_result(456, amcsg=5860, osotss=5858),   # lose
    ]
    summary = _summarize_trace("burstgpt_hf", results)
    assert summary.goodput_gap_is_structural is False
    assert summary.goodput_gap_is_noise is True
    assert summary.seeds_osotss_wins == 1
    assert summary.seeds_osotss_loses == 2


def test_summarize_gap_statistics():
    """Gap statistics (mean, std, min, max) are computed correctly."""
    results = [
        _make_seed_result(42, amcsg=5864, osotss=5849),    # gap=-15
        _make_seed_result(123, amcsg=5862, osotss=5859),   # gap=-3
        _make_seed_result(456, amcsg=5860, osotss=5850),   # gap=-10
    ]
    summary = _summarize_trace("test", results)
    assert summary.gap_min == -15
    assert summary.gap_max == -3
    assert abs(summary.gap_mean - (-28 / 3)) < 0.01
    assert summary.gap_std > 0.0


def test_summarize_single_seed():
    """Single-seed summary has zero std (no division by zero)."""
    results = [_make_seed_result(42, amcsg=5864, osotss=5849)]
    summary = _summarize_trace("test", results)
    assert summary.amcsg_n_sla_safe_std == 0.0
    assert summary.osotss_n_sla_safe_std == 0.0
    assert summary.gap_std == 0.0
    assert summary.gap_min == -15
    assert summary.gap_max == -15


def test_summarize_exact_tie():
    """OSOTSS matching AMCSG exactly counts as a win."""
    results = [_make_seed_result(42, amcsg=5864, osotss=5864)]
    summary = _summarize_trace("test", results)
    assert summary.seeds_osotss_wins == 1
    assert summary.seeds_osotss_loses == 0
    assert summary.goodput_gap_is_noise is True
    assert summary.goodput_gap_is_structural is False


def test_seed_result_to_dict():
    """SeedResult.to_dict() serialises all fields."""
    r = _make_seed_result(42, amcsg=5864, osotss=5849)
    d = r.to_dict()
    assert d["seed"] == 42
    assert d["amcsg_n_sla_safe"] == 5864
    assert d["osotss_n_sla_safe"] == 5849
    assert d["gap_n_sla_safe"] == -15


def test_trace_audit_summary_to_dict():
    """TraceAuditSummary.to_dict() includes per-seed list."""
    results = [_make_seed_result(42, amcsg=5864, osotss=5849)]
    summary = _summarize_trace("test", results)
    d = summary.to_dict()
    assert "per_seed" in d
    assert len(d["per_seed"]) == 1
    assert d["per_seed"][0]["seed"] == 42


# ---------------------------------------------------------------------------
# Constants smoke-tests
# ---------------------------------------------------------------------------

def test_canonical_constants_match_roadmap():
    """Canonical single-seed values match ROADMAP.md entries."""
    assert CANONICAL_AMCSG_BURSTGPT_N_SLA_SAFE == 5864
    assert CANONICAL_OSOTSS_BURSTGPT_N_SLA_SAFE == 5849
    assert CANONICAL_AMCSG_AZURE_N_SLA_SAFE == 5823
    assert CANONICAL_OSOTSS_AZURE_N_SLA_SAFE == 5823
    # Canonical seed=42 gap on BurstGPT
    canonical_gap = CANONICAL_OSOTSS_BURSTGPT_N_SLA_SAFE - CANONICAL_AMCSG_BURSTGPT_N_SLA_SAFE
    assert canonical_gap == -15


def test_audit_seeds_contains_canonical():
    """The canonical seed (42) is always in the audit seed list."""
    assert 42 in AUDIT_SEEDS


def test_audit_seeds_has_at_least_two():
    """Multi-seed validation requires at least 2 seeds for std computation."""
    assert len(AUDIT_SEEDS) >= 2


# ---------------------------------------------------------------------------
# Integration smoke-tests — 2-seed fast subset (no slow oracle runs)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_multi_seed_azure_audit_canonical_seed():
    """Canonical seed=42 Azure result matches ROADMAP.md values within ±5."""
    summary = run_multi_seed_azure_audit(seeds=[42])
    canonical = summary.per_seed[0]
    # AMCSG at seed=42: 5823 (from ROADMAP)
    assert abs(canonical.amcsg_n_sla_safe - CANONICAL_AMCSG_AZURE_N_SLA_SAFE) <= 5, (
        f"AMCSG Azure n_sla_safe={canonical.amcsg_n_sla_safe} "
        f"deviates from canonical {CANONICAL_AMCSG_AZURE_N_SLA_SAFE}"
    )
    # OSOTSS at seed=42: 5823 (from ROADMAP — matches AMCSG on Azure)
    assert abs(canonical.osotss_n_sla_safe - CANONICAL_OSOTSS_AZURE_N_SLA_SAFE) <= 5, (
        f"OSOTSS Azure n_sla_safe={canonical.osotss_n_sla_safe} "
        f"deviates from canonical {CANONICAL_OSOTSS_AZURE_N_SLA_SAFE}"
    )
    # OSOTSS should be better than AMCSG on goodput/$
    assert canonical.osotss_goodput_per_dollar > canonical.amcsg_goodput_per_dollar, (
        "OSOTSS should have higher goodput/$ than AMCSG on Azure"
    )


@pytest.mark.slow
def test_multi_seed_burstgpt_audit_canonical_seed():
    """Canonical seed=42 BurstGPT result matches ROADMAP.md values within ±5."""
    summary = run_multi_seed_burstgpt_audit(seeds=[42])
    canonical = summary.per_seed[0]
    # AMCSG at seed=42: 5864 (from ROADMAP)
    assert abs(canonical.amcsg_n_sla_safe - CANONICAL_AMCSG_BURSTGPT_N_SLA_SAFE) <= 5, (
        f"AMCSG BurstGPT n_sla_safe={canonical.amcsg_n_sla_safe} "
        f"deviates from canonical {CANONICAL_AMCSG_BURSTGPT_N_SLA_SAFE}"
    )
    # OSOTSS at seed=42: 5849 (from ROADMAP)
    assert abs(canonical.osotss_n_sla_safe - CANONICAL_OSOTSS_BURSTGPT_N_SLA_SAFE) <= 5, (
        f"OSOTSS BurstGPT n_sla_safe={canonical.osotss_n_sla_safe} "
        f"deviates from canonical {CANONICAL_OSOTSS_BURSTGPT_N_SLA_SAFE}"
    )
    # OSOTSS should have higher goodput/$ than AMCSG on BurstGPT
    assert canonical.osotss_goodput_per_dollar > canonical.amcsg_goodput_per_dollar, (
        "OSOTSS should have higher goodput/$ than AMCSG on BurstGPT"
    )


@pytest.mark.slow
def test_multi_seed_azure_summary_fields_complete():
    """TraceAuditSummary for Azure has all required fields populated."""
    summary = run_multi_seed_azure_audit(seeds=[42, 123])
    assert len(summary.per_seed) == 2
    assert summary.amcsg_n_sla_safe_std >= 0.0
    assert summary.osotss_n_sla_safe_std >= 0.0
    assert summary.gap_std >= 0.0
    assert summary.seeds_osotss_wins + summary.seeds_osotss_loses == 2
    assert isinstance(summary.goodput_gap_is_structural, bool)
    assert isinstance(summary.goodput_gap_is_noise, bool)
    # to_dict should not raise
    d = summary.to_dict()
    assert d["trace"] == "azure_llm_2024"
    assert len(d["per_seed"]) == 2
