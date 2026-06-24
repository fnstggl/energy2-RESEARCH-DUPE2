"""Stochastic safety margin backtest — OSOTSS interrupt_safety_margin sweep.

Sweeps ``interrupt_safety_margin`` ∈ {0, 10, 15, 20, 25, 30} on both canonical
public traces:
  * Azure LLM 2024  (5880 requests, SLA=10s)
  * BurstGPT HF     (5880 requests, SLA=30s)

Research motivation (GAP_ANALYSIS.md, 2026-06-24):
  On BurstGPT, OSOTSS (margin=0) misses AMCSG by 15 requests (n_sla_safe=5849
  vs 5864).  Root cause is the stochastic/deterministic simulation mismatch: the
  oracle convergence check runs deterministic FIFO (no interruptions) while the
  final GSF evaluation uses Binomial interruptions (p=10%/hr, p_survive≈0.9982
  per tick). Expected interruption-induced SLA misses ≈ 5880 × (1−0.9982^98) ≈
  10–20 requests per run.  Adding ``interrupt_safety_margin`` to the oracle
  convergence target forces the oracle to over-provision enough to absorb the
  expected stochastic interruption buffer.

Same-conditions checklist (constitution §same-conditions):
  * Same trace, same SLA, same cost denominator, same GPU-hour accounting
  * Same physics, same capacity model, same pricing model
  * Same decision-time information (causal EWMA, past observations only)
  * Same evaluation method (GSF spot-fleet simulation, seed=42)
  * Baseline: OSOTSS margin=0 (current OSOTSS default)
  * Candidate: OSOTSS margin ∈ {10, 15, 20, 25, 30}
  * Second baseline: AMCSG gate=12.5% (strongest fair deployable)

Primary KPI: SLA-safe goodput/$ (tokens within SLA per dollar spent).
Frontier improvement requires: candidate > baseline on BOTH traces on
goodput/$ AND n_sla_safe >= AMCSG baseline on BOTH traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .srtf_serving_backtest import (
    run_online_sotss_azure_backtest,
    run_online_sotss_burstgpt_backtest,
)

# Canonical sweep values
MARGIN_SWEEP = [0, 10, 15, 20, 25, 30]

# AMCSG reference targets (canonical from ROADMAP.md)
AMCSG_AZURE_GOODPUT_PER_DOLLAR = 150_630.0
AMCSG_AZURE_N_SLA_SAFE = 5823
AMCSG_BURSTGPT_GOODPUT_PER_DOLLAR = 168_270.0
AMCSG_BURSTGPT_N_SLA_SAFE = 5864

# OSOTSS margin=0 reference (current best deployable from ROADMAP.md)
OSOTSS_BASELINE_AZURE_GOODPUT = 159_578.0
OSOTSS_BASELINE_AZURE_N_SLA_SAFE = 5823
OSOTSS_BASELINE_BURSTGPT_GOODPUT = 178_109.0
OSOTSS_BASELINE_BURSTGPT_N_SLA_SAFE = 5849


@dataclass
class MarginSweepEntry:
    """Single margin value result on one trace."""

    trace: str
    interrupt_safety_margin: int
    goodput_per_dollar: float
    n_sla_safe: int
    cost: float
    c_mean: float
    p99_s: float
    n_iters: int
    amcsg_goodput_per_dollar: float
    amcsg_n_sla_safe: int
    vs_amcsg_pct: float
    vs_osotss_baseline_pct: float
    sla_safe_vs_amcsg: int  # n_sla_safe - amcsg_n_sla_safe


@dataclass
class StochasticSafetyMarginReport:
    """Stochastic safety margin sweep report over both public traces.

    Sweep of ``interrupt_safety_margin`` ∈ {0, 10, 15, 20, 25, 30} on
    Azure LLM 2024 and BurstGPT HF.  Primary question: does any margin value
    close BurstGPT's 15-request SLA gap without hurting Azure?

    Fields:
        azure_sweep:       Per-margin results for Azure LLM 2024.
        burstgpt_sweep:    Per-margin results for BurstGPT HF.
        best_azure_margin: Margin giving highest Azure goodput/$ while keeping
                           n_sla_safe >= AMCSG.
        best_burstgpt_margin: Margin giving highest BurstGPT goodput/$ while
                              keeping n_sla_safe >= AMCSG (= 5864).
        is_frontier_improvement: True if any single margin achieves both traces
                                  simultaneously above the margin=0 baseline on
                                  goodput/$ AND n_sla_safe >= AMCSG on both.
        best_joint_margin:  The margin (if any) satisfying the frontier criterion.
        verdict:            "FRONTIER_IMPROVEMENT", "NEGATIVE_RESULT", or
                            "PARTIAL_RESULT" (improves one trace but not both).
    """

    azure_sweep: list[MarginSweepEntry]
    burstgpt_sweep: list[MarginSweepEntry]
    best_azure_margin: int
    best_burstgpt_margin: int
    is_frontier_improvement: bool
    best_joint_margin: int | None
    verdict: str
    raw: dict = field(default_factory=dict)


def _sweep_one_trace(
    trace: str,
    margins: list[int],
    amcsg_goodput: float,
    amcsg_n_safe: int,
    baseline_goodput: float,
    baseline_n_safe: int,
    runner,
    **kwargs,
) -> list[MarginSweepEntry]:
    """Run runner for each margin and collect MarginSweepEntry objects."""
    entries = []
    for m in margins:
        report = runner(interrupt_safety_margin=m, **kwargs)
        vs_amcsg = (report.osotss_goodput_per_dollar - amcsg_goodput) / max(amcsg_goodput, 1e-9) * 100.0
        vs_baseline = (
            (report.osotss_goodput_per_dollar - baseline_goodput) / max(baseline_goodput, 1e-9) * 100.0
        )
        entries.append(
            MarginSweepEntry(
                trace=trace,
                interrupt_safety_margin=m,
                goodput_per_dollar=report.osotss_goodput_per_dollar,
                n_sla_safe=report.osotss_n_sla_safe,
                cost=report.osotss_cost,
                c_mean=report.osotss_c_mean,
                p99_s=report.osotss_p99_s,
                n_iters=report.osotss_n_iters,
                amcsg_goodput_per_dollar=report.amcsg_goodput_per_dollar,
                amcsg_n_sla_safe=report.amcsg_n_sla_safe,
                vs_amcsg_pct=vs_amcsg,
                vs_osotss_baseline_pct=vs_baseline,
                sla_safe_vs_amcsg=report.osotss_n_sla_safe - report.amcsg_n_sla_safe,
            )
        )
    return entries


def run_stochastic_safety_margin_azure_backtest(
    margins: list[int] | None = None,
    **kwargs,
) -> list[MarginSweepEntry]:
    """Sweep interrupt_safety_margin on Azure LLM 2024.

    Args:
        margins:  List of margin values to sweep (default {0,10,15,20,25,30}).
        **kwargs: Extra kwargs forwarded to run_online_sotss_azure_backtest.

    Returns:
        List of MarginSweepEntry, one per margin value.
    """
    if margins is None:
        margins = MARGIN_SWEEP
    return _sweep_one_trace(
        trace="azure_llm_2024",
        margins=margins,
        amcsg_goodput=AMCSG_AZURE_GOODPUT_PER_DOLLAR,
        amcsg_n_safe=AMCSG_AZURE_N_SLA_SAFE,
        baseline_goodput=OSOTSS_BASELINE_AZURE_GOODPUT,
        baseline_n_safe=OSOTSS_BASELINE_AZURE_N_SLA_SAFE,
        runner=run_online_sotss_azure_backtest,
        **kwargs,
    )


def run_stochastic_safety_margin_burstgpt_backtest(
    margins: list[int] | None = None,
    **kwargs,
) -> list[MarginSweepEntry]:
    """Sweep interrupt_safety_margin on BurstGPT HF.

    Args:
        margins:  List of margin values to sweep (default {0,10,15,20,25,30}).
        **kwargs: Extra kwargs forwarded to run_online_sotss_burstgpt_backtest.

    Returns:
        List of MarginSweepEntry, one per margin value.
    """
    if margins is None:
        margins = MARGIN_SWEEP
    return _sweep_one_trace(
        trace="burstgpt_hf",
        margins=margins,
        amcsg_goodput=AMCSG_BURSTGPT_GOODPUT_PER_DOLLAR,
        amcsg_n_safe=AMCSG_BURSTGPT_N_SLA_SAFE,
        baseline_goodput=OSOTSS_BASELINE_BURSTGPT_GOODPUT,
        baseline_n_safe=OSOTSS_BASELINE_BURSTGPT_N_SLA_SAFE,
        runner=run_online_sotss_burstgpt_backtest,
        **kwargs,
    )


def run_stochastic_safety_margin_full_backtest(
    margins: list[int] | None = None,
    azure_kwargs: dict | None = None,
    burstgpt_kwargs: dict | None = None,
) -> StochasticSafetyMarginReport:
    """Full stochastic safety margin sweep on both public traces.

    Runs OSOTSS with each value in ``margins`` on Azure LLM 2024 and BurstGPT
    HF under identical conditions.  Identifies whether any margin value closes
    BurstGPT's 15-request SLA gap while maintaining or improving Azure metrics.

    Frontier improvement criterion (requires both traces simultaneously):
      * goodput/$ >= OSOTSS margin=0 baseline on both traces
      * n_sla_safe >= AMCSG n_sla_safe on both traces
      * For BurstGPT: n_sla_safe >= 5864 (current gap is 5849 → need +15)

    Args:
        margins:         Margin values to sweep. Default {0, 10, 15, 20, 25, 30}.
        azure_kwargs:    Extra kwargs forwarded to Azure runner.
        burstgpt_kwargs: Extra kwargs forwarded to BurstGPT runner.

    Returns:
        StochasticSafetyMarginReport with per-margin per-trace metrics and
        overall frontier verdict.
    """
    if margins is None:
        margins = MARGIN_SWEEP
    az_kw = azure_kwargs or {}
    bg_kw = burstgpt_kwargs or {}

    azure_sweep = run_stochastic_safety_margin_azure_backtest(margins=margins, **az_kw)
    burstgpt_sweep = run_stochastic_safety_margin_burstgpt_backtest(margins=margins, **bg_kw)

    # Best margin per trace: highest goodput/$ while n_sla_safe >= AMCSG target
    def _best_margin(sweep: list[MarginSweepEntry], amcsg_n: int) -> int:
        eligible = [e for e in sweep if e.n_sla_safe >= amcsg_n]
        if not eligible:
            # Fallback: best n_sla_safe
            return max(sweep, key=lambda e: e.n_sla_safe).interrupt_safety_margin
        return max(eligible, key=lambda e: e.goodput_per_dollar).interrupt_safety_margin

    best_az = _best_margin(azure_sweep, AMCSG_AZURE_N_SLA_SAFE)
    best_bg = _best_margin(burstgpt_sweep, AMCSG_BURSTGPT_N_SLA_SAFE)

    # Frontier: find a single margin that beats margin=0 baseline on both traces
    best_joint: int | None = None
    best_joint_score = -1.0
    for m in margins:
        az = next((e for e in azure_sweep if e.interrupt_safety_margin == m), None)
        bg = next((e for e in burstgpt_sweep if e.interrupt_safety_margin == m), None)
        if az is None or bg is None:
            continue
        az_ok = (
            az.goodput_per_dollar >= OSOTSS_BASELINE_AZURE_GOODPUT
            and az.n_sla_safe >= AMCSG_AZURE_N_SLA_SAFE
        )
        bg_ok = (
            bg.goodput_per_dollar >= OSOTSS_BASELINE_BURSTGPT_GOODPUT
            and bg.n_sla_safe >= AMCSG_BURSTGPT_N_SLA_SAFE
        )
        if az_ok and bg_ok:
            joint_score = az.goodput_per_dollar + bg.goodput_per_dollar
            if joint_score > best_joint_score:
                best_joint = m
                best_joint_score = joint_score

    # Partial: any margin closes BurstGPT gap even if Azure regresses
    bg_gap_closed = any(
        e.n_sla_safe >= AMCSG_BURSTGPT_N_SLA_SAFE for e in burstgpt_sweep
    )

    if best_joint is not None:
        verdict = "FRONTIER_IMPROVEMENT"
    elif bg_gap_closed:
        verdict = "PARTIAL_RESULT"
    else:
        verdict = "NEGATIVE_RESULT"

    return StochasticSafetyMarginReport(
        azure_sweep=azure_sweep,
        burstgpt_sweep=burstgpt_sweep,
        best_azure_margin=best_az,
        best_burstgpt_margin=best_bg,
        is_frontier_improvement=best_joint is not None,
        best_joint_margin=best_joint,
        verdict=verdict,
        raw={
            "azure_sweep": [vars(e) for e in azure_sweep],
            "burstgpt_sweep": [vars(e) for e in burstgpt_sweep],
        },
    )
