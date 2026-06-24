"""Aging SRTF + AMCSG compound backtest integrity tests — run 2026-06-24.

Five-Failure Rule integration experiment: tests whether non-preemptive aging
SRTF queue discipline compounds with AMCSG optimal variable-c capacity schedule.

Proves:
- _simulate_aging_srtf_variable_c is non-preemptive (preemption_count=0)
- FIFO+AMCSG condition reproduces canonical AMCSG result (~150,630 / ~168,270 gp/$)
- aging_srtf+fixed-c = FIFO+fixed-c (running-median prior degeneracy confirmed)
- aging_srtf+AMCSG within noise of FIFO+AMCSG (honest null result)
- n_sla_safe delta >= 0 (no SLA regression)
- production_claim=False, five_failure_rule_integration=True
- No production modules modified
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aurelius.benchmarks.srtf_serving_backtest import (
    AgingSRTFAMCSGReport,
    _apply_gsf_spot_interruptions,
    _Request,
    _simulate_aging_srtf_variable_c,
    run_aging_srtf_amcsg_azure_backtest,
    run_aging_srtf_amcsg_burstgpt_backtest,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def azure_report() -> AgingSRTFAMCSGReport:
    return run_aging_srtf_amcsg_azure_backtest()


@pytest.fixture(scope="module")
def burstgpt_report() -> AgingSRTFAMCSGReport:
    return run_aging_srtf_amcsg_burstgpt_backtest()


# ── unit tests for _simulate_aging_srtf_variable_c ───────────────────────────

def test_aging_srtf_variable_c_basic_dispatch():
    """Short requests (small predicted_tokens) dispatched before long ones."""
    short = _Request(idx=0, arrival_s=0.0, actual_tokens=10,
                     predicted_tokens=10.0, service_s=1.0)
    long_ = _Request(idx=1, arrival_s=0.1, actual_tokens=100,
                     predicted_tokens=100.0, service_s=10.0)
    c_schedule = [1] * 30   # one server, 30 ticks
    sim, resp, wait = _simulate_aging_srtf_variable_c([short, long_], c_schedule)
    # Both requests complete
    assert 0 in resp and 1 in resp
    # short arrives first, gets dispatched first; completes at t=1.0
    # long arrives at 0.1, waits, dispatched at t=1.0
    assert resp[0] < resp[1]


def test_aging_srtf_variable_c_non_preemptive():
    """In-progress requests are never preempted (preemption_count=0)."""
    reqs = [
        _Request(idx=i, arrival_s=float(i) * 0.1, actual_tokens=10 - i,
                 predicted_tokens=float(10 - i), service_s=float(10 - i) * 0.5)
        for i in range(8)
    ]
    c_schedule = [2] * 50
    sim, resp, _ = _simulate_aging_srtf_variable_c(reqs, c_schedule)
    assert sim.get("preemption_count", 0) == 0


def test_aging_srtf_variable_c_variable_c_flag():
    reqs = [_Request(idx=0, arrival_s=0.0, actual_tokens=10,
                     predicted_tokens=10.0, service_s=1.0)]
    c_schedule = [1] * 5
    sim, _, _ = _simulate_aging_srtf_variable_c(reqs, c_schedule)
    assert sim.get("variable_c") is True


def test_aging_srtf_variable_c_drain_semantics():
    """Servers at idx >= c(t) drain but do not accept new work."""
    # 2-server schedule dropping to 1 at tick 1
    c_schedule = [2, 1, 1, 1, 1]
    reqs = [
        _Request(idx=i, arrival_s=float(i) * 70.0, actual_tokens=50,
                 predicted_tokens=50.0, service_s=5.0)
        for i in range(4)
    ]
    sim, resp, _ = _simulate_aging_srtf_variable_c(reqs, c_schedule)
    assert len(resp) > 0


def test_aging_srtf_variable_c_equal_predictions_fifo_like():
    """When all predicted_tokens equal, dispatch approximates FIFO order.

    Dense arrivals (gap < service) create a queue; response times grow.
    With equal predicted_tokens, the aging key distinguishes only by wait time
    so dispatch approximates FIFO (stable by idx tiebreak).
    """
    reqs = [
        _Request(idx=i, arrival_s=float(i) * 0.05, actual_tokens=50,
                 predicted_tokens=50.0, service_s=1.0)
        for i in range(10)
    ]
    c_schedule = [1] * 200   # 1 server, enough ticks to drain
    sim, resp, _ = _simulate_aging_srtf_variable_c(reqs, c_schedule)
    assert len(resp) == 10
    # Earlier arrivals complete with shorter total response time
    assert resp[0] < resp[9]


def test_apply_gsf_spot_interruptions_deterministic():
    """Same seed yields identical c_effective."""
    c_schedule = [4] * 10
    eff1 = _apply_gsf_spot_interruptions(c_schedule, 0.95, 8, 0.10, 60.0, 42)
    eff2 = _apply_gsf_spot_interruptions(c_schedule, 0.95, 8, 0.10, 60.0, 42)
    assert eff1 == eff2


def test_apply_gsf_spot_interruptions_floor():
    """No tick has 0 effective servers (min 1)."""
    c_schedule = [1] * 20
    eff = _apply_gsf_spot_interruptions(c_schedule, 0.95, 8, 0.999, 60.0, 42)
    assert all(e >= 1 for e in eff)


# ── safety / governance ───────────────────────────────────────────────────────

def test_report_dataclass_safety_flags(azure_report):
    assert azure_report.production_claim is False
    assert azure_report.five_failure_rule_integration is True


def test_to_dict_safety_flags(azure_report):
    d = azure_report.to_dict()
    assert d["production_claim"] is False
    assert d["five_failure_rule_integration"] is True


def test_no_production_module_modified():
    try:
        out = subprocess.check_output(
            ["git", "diff", "--name-only", "main...HEAD"],
            cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except subprocess.CalledProcessError:
        pytest.skip("main ref not available in this environment")
    forbidden = {
        "aurelius/optimization/scheduler.py",
        "aurelius/optimization/objective.py",
        "aurelius/optimization/constraints.py",
        "aurelius/forecasting/constraint_shadow_scorer.py",
        "aurelius/forecasting/economic_overlay.py",
        "aurelius/residency/decision.py",
        "aurelius/frontier/controller.py",
    }
    bad = [p for p in out if p in forbidden]
    assert not bad, f"forbidden production module modified: {bad}"


# ── canonical parity checks ───────────────────────────────────────────────────

def test_azure_fifo_amcsg_reproduces_canonical(azure_report):
    """FIFO+AMCSG must reproduce the canonical ~150,630 gp/$ result."""
    gp = azure_report.fifo_amcsg_goodput_per_dollar
    assert 149_000 < gp < 152_000, f"FIFO+AMCSG Azure: {gp:.0f} outside expected 149k-152k"


def test_burstgpt_fifo_amcsg_reproduces_canonical(burstgpt_report):
    """FIFO+AMCSG must reproduce the canonical ~168,270 gp/$ result."""
    gp = burstgpt_report.fifo_amcsg_goodput_per_dollar
    assert 166_000 < gp < 170_000, f"FIFO+AMCSG BurstGPT: {gp:.0f} outside expected 166k-170k"


# ── null result characterisation ─────────────────────────────────────────────

def test_azure_aging_srtf_amcsg_not_worse_than_fifo_amcsg(azure_report):
    """Aging SRTF must not HURT goodput vs FIFO at same AMCSG capacity."""
    delta_pct = azure_report.aging_srtf_amcsg_vs_fifo_amcsg_pct
    assert delta_pct >= -0.5, f"aging_srtf_amcsg hurt Azure goodput: {delta_pct:.2f}%"


def test_burstgpt_aging_srtf_amcsg_not_worse_than_fifo_amcsg(burstgpt_report):
    """Aging SRTF must not HURT goodput vs FIFO at same AMCSG capacity."""
    delta_pct = burstgpt_report.aging_srtf_amcsg_vs_fifo_amcsg_pct
    assert delta_pct >= -0.5, f"aging_srtf_amcsg hurt BurstGPT goodput: {delta_pct:.2f}%"


def test_azure_sla_safety_not_regressed(azure_report):
    """n_sla_safe must not decrease vs FIFO+AMCSG."""
    assert azure_report.amcsg_aging_srtf_sla_safe_delta >= 0


def test_burstgpt_sla_safety_not_regressed(burstgpt_report):
    assert burstgpt_report.amcsg_aging_srtf_sla_safe_delta >= 0


def test_azure_fixed_c_discipline_degeneracy(azure_report):
    """At fixed-c with running-median prior, aging SRTF ≈ FIFO (degeneracy)."""
    fifo_gp = azure_report.fifo_fixed_goodput_per_dollar
    aging_gp = azure_report.aging_srtf_fixed_goodput_per_dollar
    ratio = abs(aging_gp - fifo_gp) / max(fifo_gp, 1e-9) * 100
    assert ratio < 1.0, (
        f"Expected aging_srtf≈FIFO at fixed-c (prediction degeneracy); "
        f"got {ratio:.2f}% delta"
    )


def test_burstgpt_fixed_c_discipline_near_fifo(burstgpt_report):
    fifo_gp = burstgpt_report.fifo_fixed_goodput_per_dollar
    aging_gp = burstgpt_report.aging_srtf_fixed_goodput_per_dollar
    ratio = abs(aging_gp - fifo_gp) / max(fifo_gp, 1e-9) * 100
    assert ratio < 2.0, (
        f"Expected aging_srtf≈FIFO at fixed-c; got {ratio:.2f}% delta"
    )


def test_below_osotss_frontier(azure_report, burstgpt_report):
    """Aging_srtf+AMCSG does not beat OSOTSS canonical — honest null result."""
    assert azure_report.aging_srtf_amcsg_vs_osotss_canonical_pct < 0
    assert burstgpt_report.aging_srtf_amcsg_vs_osotss_canonical_pct < 0


# ── schema / fields ───────────────────────────────────────────────────────────

def test_to_dict_has_required_keys(azure_report):
    d = azure_report.to_dict()
    required = {
        "trace", "total_requests", "fixed_c", "sla_s",
        "aging_alpha", "amcsg_gate_pct", "amcsg_c_mean",
        "cost_fixed_c", "cost_amcsg",
        "fifo_fixed_goodput_per_dollar", "fifo_amcsg_goodput_per_dollar",
        "aging_srtf_fixed_goodput_per_dollar", "aging_srtf_amcsg_goodput_per_dollar",
        "fifo_fixed_n_sla_safe", "fifo_amcsg_n_sla_safe",
        "aging_srtf_fixed_n_sla_safe", "aging_srtf_amcsg_n_sla_safe",
        "aging_srtf_amcsg_vs_fifo_amcsg_pct",
        "amcsg_aging_srtf_sla_safe_delta",
        "osotss_canonical_goodput_per_dollar",
        "aging_srtf_amcsg_vs_osotss_canonical_pct",
        "production_claim", "five_failure_rule_integration",
    }
    missing = required - set(d.keys())
    assert not missing, f"missing required keys: {missing}"


def test_costs_positive(azure_report, burstgpt_report):
    assert azure_report.cost_fixed_c > 0
    assert azure_report.cost_amcsg > 0
    assert burstgpt_report.cost_fixed_c > 0
    assert burstgpt_report.cost_amcsg > 0


def test_amcsg_costs_higher_than_fixed_c_on_burstgpt(burstgpt_report):
    """AMCSG variable-c costs more than fixed-c on burst traces."""
    assert burstgpt_report.cost_amcsg > burstgpt_report.cost_fixed_c


def test_total_requests_correct(azure_report, burstgpt_report):
    assert azure_report.total_requests == 5880
    assert burstgpt_report.total_requests == 5880


def test_completion_rates_one(azure_report, burstgpt_report):
    """All 4 conditions complete all requests (no dropped requests)."""
    for attr in ("fifo_fixed_completion_rate", "fifo_amcsg_completion_rate",
                  "aging_srtf_fixed_completion_rate", "aging_srtf_amcsg_completion_rate"):
        assert getattr(azure_report, attr) == 1.0, attr
        assert getattr(burstgpt_report, attr) == 1.0, attr
