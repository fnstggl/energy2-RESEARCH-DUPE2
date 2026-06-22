"""Tests for safe_high_utilization policy [run 2026-06-22].

safe_high_utilization uses EWMA-anticipatory sizing (same as constraint_aware)
at rho=0.75 (vs 0.65 for CA), strict 0% per-tick timeout tolerance, and no
hysteresis. Validated by run_azure_2024_safe_utilization_frontier.py which shows
anticipatory@0.75 is SAFE at 9.465% aggregate timeout on the full Azure 2024 trace.

Invariants tested:
  1.  safe_high_utilization is in ALL_POLICIES.
  2.  run_backtest with safe_high_utilization returns a PolicyResult.
  3.  safe_high_utilization gpd/$ > 0 (positive, non-trivial).
  4.  safe_high_utilization timeout_rate_pct_mean < 10% (SAFE gate).
  5.  safe_high_utilization gpd/$ >= constraint_aware gpd/$ (ALPHA_WIN or TIE on this trace).
  6.  _SHU_TARGET_RHO > constraint_aware's 0.65 and < utilization_aware's 0.85.
  7.  _SHU_TIMEOUT_TOL == 0.0 (strict same as constraint_aware — relaxed tol unsafe at rho=0.75).
  8.  _constraint_trim with timeout_tol=0.0 is backward-compatible (same as original).
  9.  _constraint_trim with timeout_tol > 0.0 accepts fewer replicas when safe.
  10. safe_high_utilization uses fewer or equal gpu_hours than constraint_aware (cheaper).
  11. safe_high_utilization sla_compliant_goodput is positive (real serving numerator).
  12. Policy result has cache_savings_applied=True (same savings proxy as CA).
  13. At BurstGPT scale 300: gpd/$ improvement at least TIE vs constraint_aware.
"""

from __future__ import annotations

import math
import os

import pytest

from aurelius.traces.backtest import (
    ALL_POLICIES,
    _SHU_TARGET_RHO,
    _SHU_TIMEOUT_TOL,
    _constraint_trim,
    evaluate_tick,
    run_backtest,
)
from aurelius.traces.replay import requests_to_arrival_ticks
from aurelius.traces.schema import time_rescale

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
# Unit tests (constants and _constraint_trim)
# ---------------------------------------------------------------------------

def test_shu_in_all_policies():
    """Invariant 1: safe_high_utilization is registered."""
    assert "safe_high_utilization" in ALL_POLICIES


def test_shu_target_rho_between_ca_and_util():
    """Invariant 6: rho=0.75 is between CA(0.65) and utilization_aware(0.85)."""
    assert _SHU_TARGET_RHO > 0.65
    assert _SHU_TARGET_RHO < 0.85


def test_shu_timeout_tol_strict():
    """Invariant 7: strict 0% tolerance required for safety at rho=0.75."""
    assert _SHU_TIMEOUT_TOL == 0.0


def test_constraint_trim_backward_compat():
    """Invariant 8: timeout_tol=0.0 default reproduces original behavior."""
    reqs = _load_azure()
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    active = [t for t in ticks if t.request_count > 0]
    if not active:
        pytest.skip("no active ticks in fixture")
    tick = active[0]
    tick_hours = 60.0 / 3600.0
    # Both calls should produce identical results
    r_default = _constraint_trim(tick, 5, 0.0, tick_hours, None, timeout_tol=0.0)
    r_explicit = _constraint_trim(tick, 5, 0.0, tick_hours, None)
    assert r_default == r_explicit


def test_constraint_trim_relaxed_tol():
    """Invariant 9: relaxed tolerance accepts fewer replicas when safe."""
    reqs = _load_azure()
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=60.0)
    active = [t for t in ticks if t.request_count > 0]
    if not active:
        pytest.skip("no active ticks in fixture")
    tick_hours = 60.0 / 3600.0
    results = []
    for tick in active[:20]:
        r_strict = _constraint_trim(tick, 10, 0.0, tick_hours, None, timeout_tol=0.0)
        r_relaxed = _constraint_trim(tick, 10, 0.0, tick_hours, None, timeout_tol=5.0)
        results.append((r_strict, r_relaxed))
    # Relaxed tolerance should never use MORE replicas than strict
    assert all(r_relaxed <= r_strict for r_strict, r_relaxed in results)


# ---------------------------------------------------------------------------
# Integration tests (run_backtest on fixture)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def azure_result():
    reqs = _load_azure()
    return run_backtest(reqs, tick_seconds=60.0,
                        policies=("sla_aware", "constraint_aware", "safe_high_utilization"))


@pytest.fixture(scope="module")
def burstgpt_scale300_result():
    reqs = _load_burstgpt()
    scaled = time_rescale(reqs, 300)
    return run_backtest(scaled, tick_seconds=60.0,
                        policies=("sla_aware", "constraint_aware", "safe_high_utilization"))


def test_shu_policy_result_present(azure_result):
    """Invariant 2: run_backtest produces a PolicyResult for safe_high_utilization."""
    assert "safe_high_utilization" in azure_result.policy_results


def test_shu_gpd_positive(azure_result):
    """Invariant 3: safe_high_utilization gpd/$ is positive."""
    shu = azure_result.policy_results["safe_high_utilization"]
    gpd = shu.kpi.sla_safe_goodput_per_infra_dollar
    assert gpd is not None and gpd > 0.0


def test_shu_timeout_safe(azure_result):
    """Invariant 4: safe_high_utilization aggregate timeout < 10% gate."""
    shu = azure_result.policy_results["safe_high_utilization"]
    assert shu.timeout_rate_pct_mean < 10.0


def test_shu_beats_or_ties_ca(azure_result):
    """Invariant 5: safe_high_utilization gpd/$ >= constraint_aware (main claim)."""
    shu = azure_result.policy_results["safe_high_utilization"]
    ca = azure_result.policy_results["constraint_aware"]
    shu_gpd = shu.kpi.sla_safe_goodput_per_infra_dollar or 0.0
    ca_gpd = ca.kpi.sla_safe_goodput_per_infra_dollar or 0.0
    # Allow 0.1% tolerance for floating-point noise
    assert shu_gpd >= ca_gpd * 0.999, (
        f"safe_high_utilization ({shu_gpd:.2f}) should be >= constraint_aware ({ca_gpd:.2f})")


def test_shu_lower_or_equal_gpu_hours(azure_result):
    """Invariant 10: safe_high_utilization uses fewer or equal GPU hours (cheaper)."""
    shu = azure_result.policy_results["safe_high_utilization"]
    ca = azure_result.policy_results["constraint_aware"]
    assert shu.kpi.active_gpu_hours <= ca.kpi.active_gpu_hours * 1.001


def test_shu_positive_sla_compliant_goodput(azure_result):
    """Invariant 11: sla_compliant_goodput is positive (real tokens served)."""
    shu = azure_result.policy_results["safe_high_utilization"]
    assert shu.kpi.sla_compliant_goodput > 0


def test_shu_cache_savings_applied(azure_result):
    """Invariant 12: cache_savings_applied=True (same as constraint_aware)."""
    shu = azure_result.policy_results["safe_high_utilization"]
    assert shu.cache_savings_applied is True


def test_shu_beats_ca_at_scale300(burstgpt_scale300_result):
    """Invariant 13: at scale 300 SHU gpd/$ >= constraint_aware (TIE or ALPHA_WIN)."""
    shu = burstgpt_scale300_result.policy_results["safe_high_utilization"]
    ca = burstgpt_scale300_result.policy_results["constraint_aware"]
    shu_gpd = shu.kpi.sla_safe_goodput_per_infra_dollar or 0.0
    ca_gpd = ca.kpi.sla_safe_goodput_per_infra_dollar or 0.0
    assert shu_gpd >= ca_gpd * 0.999, (
        f"scale-300 SHU ({shu_gpd:.2f}) should be >= CA ({ca_gpd:.2f})")
    timeout_ok = shu.timeout_rate_pct_mean < 10.0
    assert timeout_ok, f"SHU timeout {shu.timeout_rate_pct_mean:.2f}% exceeds 10% gate"
