"""Oracle Soft-SLA Continuation (OSSC) backtest — borderline_margin_s sweep.

Sweeps ``borderline_margin_s`` ∈ {0.0, 0.5, 1.0, 2.0, 3.0, 5.0} on both
canonical public traces:
  * Azure LLM 2024  (5880 requests, SLA=10s)
  * BurstGPT HF     (5880 requests, SLA=30s)

Research motivation (GAP_ANALYSIS.md, 2026-06-24):
  On BurstGPT, OSOTSS (margin=0) misses AMCSG by 15 requests (n_sla_safe=5849
  vs 5864).  Root cause: the oracle terminates at violators=[] (all deterministic
  FIFO violations eliminated) but 15 borderline-tick requests fail under stochastic
  spot interruptions that reduce effective capacity.

  OSSC (Oracle Soft-SLA Continuation) adds a post-convergence phase: after
  primary convergence (violators=[]), identify requests whose deterministic
  response time is within ``borderline_margin_s`` seconds of the SLA limit and
  add capacity to their ticks.  These ticks are most vulnerable to stochastic
  spot interruptions.

  Root-cause diagnosis (from SSM run 2026-06-24): the ``interrupt_safety_margin``
  approach failed because the secondary violators=[] condition fires before the
  primary convergence check can be evaluated — the oracle already converged.
  OSSC avoids this by operating as a separate post-convergence phase.

Same-conditions checklist (constitution §same-conditions):
  * Same trace, same SLA, same cost denominator, same GPU-hour accounting
  * Same physics, same capacity model, same pricing model
  * Same decision-time information (causal EWMA, past observations only)
  * Same evaluation method (GSF spot-fleet simulation, seed=42)
  * Baseline: OSOTSS borderline_margin_s=0 (current OSOTSS default)
  * Candidate: OSOTSS borderline_margin_s ∈ {0.5, 1.0, 2.0, 3.0, 5.0}
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

# Canonical sweep values — 0.0 is the no-OSSC baseline
BORDERLINE_MARGIN_SWEEP: list[float] = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

# AMCSG reference targets (canonical from ROADMAP.md)
AMCSG_AZURE_GOODPUT_PER_DOLLAR = 150_630.0
AMCSG_AZURE_N_SLA_SAFE = 5823
AMCSG_BURSTGPT_GOODPUT_PER_DOLLAR = 168_270.0
AMCSG_BURSTGPT_N_SLA_SAFE = 5864

# OSOTSS borderline_margin_s=0 reference (current best deployable from ROADMAP.md)
OSOTSS_BASELINE_AZURE_GOODPUT = 159_578.0
OSOTSS_BASELINE_AZURE_N_SLA_SAFE = 5823
OSOTSS_BASELINE_BURSTGPT_GOODPUT = 178_109.0
OSOTSS_BASELINE_BURSTGPT_N_SLA_SAFE = 5849


@dataclass
class BorderlineSweepEntry:
    """Single borderline_margin_s value result on one trace."""

    trace: str
    borderline_margin_s: float
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
class BorderlineOSOTSSReport:
    """OSSC borderline_margin_s sweep report over both public traces.

    Sweep of ``borderline_margin_s`` ∈ {0.0, 0.5, 1.0, 2.0, 3.0, 5.0} on
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

    azure_sweep: list[BorderlineSweepEntry]
    burstgpt_sweep: list[BorderlineSweepEntry]
    best_azure_margin: float
    best_burstgpt_margin: float
    is_frontier_improvement: bool
    best_joint_margin: float | None
    verdict: str
    raw: dict = field(default_factory=dict)


def _sweep_one_trace(
    trace: str,
    margins: list[float],
    amcsg_goodput: float,
    amcsg_n_safe: int,
    baseline_goodput: float,
    baseline_n_safe: int,
    runner,
    **kwargs,
) -> list[BorderlineSweepEntry]:
    """Run runner for each borderline_margin_s and collect BorderlineSweepEntry objects."""
    entries = []
    for m in margins:
        report = runner(borderline_margin_s=m, **kwargs)
        vs_amcsg = (
            (report.osotss_goodput_per_dollar - amcsg_goodput)
            / max(amcsg_goodput, 1e-9) * 100.0
        )
        vs_baseline = (
            (report.osotss_goodput_per_dollar - baseline_goodput)
            / max(baseline_goodput, 1e-9) * 100.0
        )
        entries.append(
            BorderlineSweepEntry(
                trace=trace,
                borderline_margin_s=m,
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


def run_borderline_osotss_azure_backtest(
    margins: list[float] | None = None,
    **kwargs,
) -> list[BorderlineSweepEntry]:
    """Sweep borderline_margin_s on Azure LLM 2024.

    Args:
        margins:  List of borderline_margin_s values to sweep.
                  Default {0.0, 0.5, 1.0, 2.0, 3.0, 5.0}.
        **kwargs: Extra kwargs forwarded to run_online_sotss_azure_backtest.

    Returns:
        List of BorderlineSweepEntry, one per margin value.
    """
    if margins is None:
        margins = BORDERLINE_MARGIN_SWEEP
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


def run_borderline_osotss_burstgpt_backtest(
    margins: list[float] | None = None,
    **kwargs,
) -> list[BorderlineSweepEntry]:
    """Sweep borderline_margin_s on BurstGPT HF.

    Args:
        margins:  List of borderline_margin_s values to sweep.
                  Default {0.0, 0.5, 1.0, 2.0, 3.0, 5.0}.
        **kwargs: Extra kwargs forwarded to run_online_sotss_burstgpt_backtest.

    Returns:
        List of BorderlineSweepEntry, one per margin value.
    """
    if margins is None:
        margins = BORDERLINE_MARGIN_SWEEP
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


def run_borderline_osotss_full_backtest(
    margins: list[float] | None = None,
    azure_kwargs: dict | None = None,
    burstgpt_kwargs: dict | None = None,
) -> BorderlineOSOTSSReport:
    """Full OSSC borderline_margin_s sweep on both public traces.

    Runs OSOTSS with each value in ``margins`` on Azure LLM 2024 and BurstGPT
    HF under identical conditions.  Identifies whether any margin value closes
    BurstGPT's 15-request SLA gap while maintaining or improving Azure metrics.

    Frontier improvement criterion (requires both traces simultaneously):
      * goodput/$ >= OSOTSS margin=0 baseline on both traces
      * n_sla_safe >= AMCSG n_sla_safe on both traces
      * For BurstGPT: n_sla_safe >= 5864 (current gap is 5849 → need +15)

    Args:
        margins:         Margin values to sweep. Default {0.0,0.5,1.0,2.0,3.0,5.0}.
        azure_kwargs:    Extra kwargs forwarded to Azure runner.
        burstgpt_kwargs: Extra kwargs forwarded to BurstGPT runner.

    Returns:
        BorderlineOSOTSSReport with per-margin per-trace metrics and
        overall frontier verdict.
    """
    if margins is None:
        margins = BORDERLINE_MARGIN_SWEEP
    az_kw = azure_kwargs or {}
    bg_kw = burstgpt_kwargs or {}

    azure_sweep = run_borderline_osotss_azure_backtest(margins=margins, **az_kw)
    burstgpt_sweep = run_borderline_osotss_burstgpt_backtest(margins=margins, **bg_kw)

    # Best margin per trace: highest goodput/$ while n_sla_safe >= AMCSG target
    def _best_margin(sweep: list[BorderlineSweepEntry], amcsg_n: int) -> float:
        eligible = [e for e in sweep if e.n_sla_safe >= amcsg_n]
        if not eligible:
            return max(sweep, key=lambda e: e.n_sla_safe).borderline_margin_s
        return max(eligible, key=lambda e: e.goodput_per_dollar).borderline_margin_s

    best_az = _best_margin(azure_sweep, AMCSG_AZURE_N_SLA_SAFE)
    best_bg = _best_margin(burstgpt_sweep, AMCSG_BURSTGPT_N_SLA_SAFE)

    # Frontier: find a single margin that beats margin=0 baseline on both traces
    best_joint: float | None = None
    best_joint_score = -1.0
    for m in margins:
        az = next((e for e in azure_sweep if e.borderline_margin_s == m), None)
        bg = next((e for e in burstgpt_sweep if e.borderline_margin_s == m), None)
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

    return BorderlineOSOTSSReport(
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
