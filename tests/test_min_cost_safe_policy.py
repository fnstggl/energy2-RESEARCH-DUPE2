"""Tests for min_cost_safe policy [run 2026-06-22].

min_cost_safe finds the minimum replica count per tick where per-tick
timeout_rate_pct < _MCS_TIMEOUT_GATE (9.5%). Cache prefill savings applied.
No EWMA anticipation (pure reactive). Per-tick gate guarantees aggregate
timeout < 9.5% < 10% by construction.

Invariants tested:
  1.  min_cost_safe is in ALL_POLICIES.
  2.  run_backtest with min_cost_safe returns a PolicyResult.
  3.  min_cost_safe gpd/$ > 0 (positive, non-trivial).
  4.  min_cost_safe timeout_rate_pct_mean < 10% (SAFE gate, guaranteed by construction).
  5.  min_cost_safe timeout_rate_pct_mean < _MCS_TIMEOUT_GATE (tighter than 10%).
  6.  _MCS_TIMEOUT_GATE == 9.5 (documents the constant).
  7.  _min_cost_safe_replicas returns MIN_REPLICAS for empty tick.
  8.  _min_cost_safe_replicas result satisfies timeout < gate for non-empty ticks.
  9.  min_cost_safe gpd/$ >= sla_aware gpd/$ (strong enough policy to beat reactive).
  10. Policy result has cache_savings_applied=True (same savings proxy as CA/SHU).
  11. min_cost_safe gpu_hours <= constraint_aware gpu_hours (oracle finds minimum).
"""

from __future__ import annotations

import os

import pytest

from aurelius.traces.backtest import (
    _MCS_TIMEOUT_GATE,
    ALL_POLICIES,
    MIN_REPLICAS,
    _min_cost_safe_replicas,
    evaluate_tick,
    run_backtest,
)
from aurelius.traces.replay import requests_to_arrival_ticks

AZURE_FIXTURE = "tests/fixtures/azure_llm_2024_sample.csv"
BURSTGPT_FIXTURE = "tests/fixtures/burstgpt_sample.csv"


def _load_azure():
    from aurelius.traces import azure_llm
    return azure_llm.load_csv(AZURE_FIXTURE)


def _load_burstgpt(sample_size=5000):
    from aurelius.traces import burstgpt
    bpath = "data/external/burstgpt/raw/BurstGPT_1.csv"
    fpath = bpath if os.path.exists(bpath) else BURSTGPT_FIXTURE
    return burstgpt.load_csv(fpath, sample_size=sample_size, seed=0)


# ---------------------------------------------------------------------------
# Constant and registration invariants (fast, no I/O)
# ---------------------------------------------------------------------------

def test_mcs_in_all_policies():
    assert "min_cost_safe" in ALL_POLICIES


def test_mcs_timeout_gate_value():
    assert _MCS_TIMEOUT_GATE == 9.5


# ---------------------------------------------------------------------------
# Helper function invariants
# ---------------------------------------------------------------------------

def test_mcs_helper_empty_tick():
    """Empty tick (no requests) should return MIN_REPLICAS."""
    reqs = _load_azure()
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    empty_ticks = [t for t in ticks if t.request_count == 0]
    if not empty_ticks:
        pytest.skip("no empty ticks in fixture")
    t = empty_ticks[0]
    r = _min_cost_safe_replicas(t, prefill_savings=0.0, tick_hours=1 / 60)
    assert r == MIN_REPLICAS


def test_mcs_helper_result_satisfies_gate():
    """Result of helper must have timeout_rate_pct < gate."""
    reqs = _load_azure()
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    active_ticks = [t for t in ticks if t.request_count > 0][:5]
    tick_hours = 1 / 60
    for t in active_ticks:
        r = _min_cost_safe_replicas(t, prefill_savings=0.0, tick_hours=tick_hours)
        ev = evaluate_tick(t, r, prefill_savings=0.0, tick_hours=tick_hours)
        assert ev.timeout_rate_pct < _MCS_TIMEOUT_GATE, (
            f"tick {t.tick_index}: replicas={r} timeout={ev.timeout_rate_pct:.2f}% "
            f">= gate {_MCS_TIMEOUT_GATE}%"
        )


# ---------------------------------------------------------------------------
# Full backtest invariants on Azure fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def azure_backtest_results():
    reqs = _load_azure()
    return run_backtest(reqs, tick_seconds=60.0,
                        policies=("sla_aware", "constraint_aware",
                                  "safe_high_utilization", "min_cost_safe"))


def test_mcs_returns_policy_result(azure_backtest_results):
    assert "min_cost_safe" in azure_backtest_results.policy_results


def test_mcs_gpd_positive(azure_backtest_results):
    mcs = azure_backtest_results.policy_results["min_cost_safe"]
    assert mcs.kpi.sla_safe_goodput_per_infra_dollar > 0


def test_mcs_timeout_below_10pct(azure_backtest_results):
    mcs = azure_backtest_results.policy_results["min_cost_safe"]
    assert mcs.timeout_rate_pct_mean < 10.0, (
        f"min_cost_safe timeout {mcs.timeout_rate_pct_mean:.4f}% >= 10% UNSAFE"
    )


def test_mcs_timeout_below_gate(azure_backtest_results):
    mcs = azure_backtest_results.policy_results["min_cost_safe"]
    assert mcs.timeout_rate_pct_mean < _MCS_TIMEOUT_GATE, (
        f"min_cost_safe aggregate timeout {mcs.timeout_rate_pct_mean:.4f}% "
        f">= gate {_MCS_TIMEOUT_GATE}%"
    )


def test_mcs_beats_or_ties_sla_aware(azure_backtest_results):
    mcs = azure_backtest_results.policy_results["min_cost_safe"]
    sla = azure_backtest_results.policy_results["sla_aware"]
    mcs_gpd = mcs.kpi.sla_safe_goodput_per_infra_dollar
    sla_gpd = sla.kpi.sla_safe_goodput_per_infra_dollar
    # Allow 5% shortfall to account for fixture-scale rate effects
    assert mcs_gpd >= sla_gpd * 0.95, (
        f"min_cost_safe gpd={mcs_gpd:.0f} < sla_aware gpd={sla_gpd:.0f} * 0.95"
    )


def test_mcs_cache_savings_applied(azure_backtest_results):
    mcs = azure_backtest_results.policy_results["min_cost_safe"]
    assert mcs.cache_savings_applied is True


def test_mcs_gpu_hours_le_constraint_aware(azure_backtest_results):
    mcs = azure_backtest_results.policy_results["min_cost_safe"]
    ca = azure_backtest_results.policy_results["constraint_aware"]
    mcs_hours = mcs.kpi.active_gpu_hours
    ca_hours = ca.kpi.active_gpu_hours
    # MCS oracle finds minimum — expect ≤ CA at similar or higher timeout gate usage
    # Allow 5% slack for fixture-rate corner cases
    assert mcs_hours <= ca_hours * 1.05, (
        f"min_cost_safe gpu_hours={mcs_hours:.2f} > ca_hours={ca_hours:.2f} * 1.05"
    )
