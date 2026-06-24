"""Adaptive EWMA backtest — Online SOTSS fixed vs adaptive alpha.

Compares OSOTSS with fixed EWMA alpha=0.1 against OSOTSS with adaptive EWMA
(burst-sensitive alpha boost) on the two canonical public traces:
  * Azure LLM 2024  (5880 requests, SLA=10s)
  * BurstGPT HF     (5880 requests, SLA=30s)

Research motivation (GAP_ANALYSIS.md, 2026-06-23):
  On BurstGPT the fixed-alpha OSOTSS misses AMCSG by 15 requests (0.26%).
  Root cause: EWMA alpha=0.1 is too slow to track burst patterns; oracle fixes
  the wrong ticks because EWMA underestimates service time after a quiet period.
  Adaptive EWMA boosts alpha temporarily when actual load spikes above
  burst_threshold × current EWMA estimate, allowing faster convergence on bursts.

Same-conditions checklist (constitution §same-conditions):
  * Same trace, same SLA, same cost denominator, same GPU-hour accounting
  * Same physics, same capacity model, same pricing model
  * Same decision-time information (both use only past observations — causal)
  * Same evaluation method (GSF spot-fleet simulation, seed=42)
  * Baseline: OSOTSS fixed alpha=0.1 (current best non-oracle deployable)
  * Candidate: OSOTSS adaptive alpha (burst_threshold=1.5, burst_alpha=0.5,
    burst_cooldown_ticks=2)

Primary KPI: SLA-safe goodput/$ (tokens within SLA per dollar).
Frontier improvement requires: candidate > fixed on goodput/$ AND
n_sla_safe >= fixed_n_sla_safe on BOTH traces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .srtf_serving_backtest import (
    run_online_sotss_azure_backtest,
    run_online_sotss_burstgpt_backtest,
)

# Canonical adaptive EWMA hyperparameters
BURST_THRESHOLD = 1.5
BURST_ALPHA = 0.5
BURST_COOLDOWN_TICKS = 2
FIXED_EWMA_ALPHA = 0.1


@dataclass
class AdaptiveEWMAReport:
    """Comparison report: OSOTSS fixed vs adaptive EWMA.

    All metrics collected under identical conditions on both public traces.
    """

    # Azure LLM 2024
    azure_fixed_goodput_per_dollar: float
    azure_fixed_n_sla_safe: int
    azure_adaptive_goodput_per_dollar: float
    azure_adaptive_n_sla_safe: int
    azure_improvement_pct: float

    # BurstGPT HF
    burstgpt_fixed_goodput_per_dollar: float
    burstgpt_fixed_n_sla_safe: int
    burstgpt_adaptive_goodput_per_dollar: float
    burstgpt_adaptive_n_sla_safe: int
    burstgpt_improvement_pct: float

    # Hyperparameters used
    ewma_alpha: float
    burst_threshold: float
    burst_alpha: float
    burst_cooldown_ticks: int

    # Verdict
    is_frontier_improvement: bool

    raw: dict = field(default_factory=dict)


def run_adaptive_ewma_azure_backtest(
    ewma_alpha: float = FIXED_EWMA_ALPHA,
    burst_threshold: float = BURST_THRESHOLD,
    burst_alpha: float = BURST_ALPHA,
    burst_cooldown_ticks: int = BURST_COOLDOWN_TICKS,
    **kwargs,
) -> "AdaptiveEWMAReport":
    """Azure-only adaptive EWMA comparison (fixed vs adaptive)."""
    fixed = run_online_sotss_azure_backtest(
        ewma_alpha=ewma_alpha,
        ewma_mode="fixed",
        **kwargs,
    )
    adaptive = run_online_sotss_azure_backtest(
        ewma_alpha=ewma_alpha,
        ewma_mode="adaptive",
        burst_threshold=burst_threshold,
        burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks,
        **kwargs,
    )
    improvement = (
        (adaptive.osotss_goodput_per_dollar - fixed.osotss_goodput_per_dollar)
        / max(fixed.osotss_goodput_per_dollar, 1e-9)
        * 100.0
    )
    return AdaptiveEWMAReport(
        azure_fixed_goodput_per_dollar=fixed.osotss_goodput_per_dollar,
        azure_fixed_n_sla_safe=fixed.osotss_n_sla_safe,
        azure_adaptive_goodput_per_dollar=adaptive.osotss_goodput_per_dollar,
        azure_adaptive_n_sla_safe=adaptive.osotss_n_sla_safe,
        azure_improvement_pct=improvement,
        burstgpt_fixed_goodput_per_dollar=0.0,
        burstgpt_fixed_n_sla_safe=0,
        burstgpt_adaptive_goodput_per_dollar=0.0,
        burstgpt_adaptive_n_sla_safe=0,
        burstgpt_improvement_pct=0.0,
        ewma_alpha=ewma_alpha,
        burst_threshold=burst_threshold,
        burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks,
        is_frontier_improvement=(
            adaptive.osotss_goodput_per_dollar >= fixed.osotss_goodput_per_dollar
            and adaptive.osotss_n_sla_safe >= fixed.osotss_n_sla_safe
        ),
        raw={"azure_fixed": fixed, "azure_adaptive": adaptive},
    )


def run_adaptive_ewma_burstgpt_backtest(
    ewma_alpha: float = FIXED_EWMA_ALPHA,
    burst_threshold: float = BURST_THRESHOLD,
    burst_alpha: float = BURST_ALPHA,
    burst_cooldown_ticks: int = BURST_COOLDOWN_TICKS,
    **kwargs,
) -> "AdaptiveEWMAReport":
    """BurstGPT-only adaptive EWMA comparison (fixed vs adaptive)."""
    fixed = run_online_sotss_burstgpt_backtest(
        ewma_alpha=ewma_alpha,
        ewma_mode="fixed",
        **kwargs,
    )
    adaptive = run_online_sotss_burstgpt_backtest(
        ewma_alpha=ewma_alpha,
        ewma_mode="adaptive",
        burst_threshold=burst_threshold,
        burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks,
        **kwargs,
    )
    improvement = (
        (adaptive.osotss_goodput_per_dollar - fixed.osotss_goodput_per_dollar)
        / max(fixed.osotss_goodput_per_dollar, 1e-9)
        * 100.0
    )
    return AdaptiveEWMAReport(
        azure_fixed_goodput_per_dollar=0.0,
        azure_fixed_n_sla_safe=0,
        azure_adaptive_goodput_per_dollar=0.0,
        azure_adaptive_n_sla_safe=0,
        azure_improvement_pct=0.0,
        burstgpt_fixed_goodput_per_dollar=fixed.osotss_goodput_per_dollar,
        burstgpt_fixed_n_sla_safe=fixed.osotss_n_sla_safe,
        burstgpt_adaptive_goodput_per_dollar=adaptive.osotss_goodput_per_dollar,
        burstgpt_adaptive_n_sla_safe=adaptive.osotss_n_sla_safe,
        burstgpt_improvement_pct=improvement,
        ewma_alpha=ewma_alpha,
        burst_threshold=burst_threshold,
        burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks,
        is_frontier_improvement=(
            adaptive.osotss_goodput_per_dollar >= fixed.osotss_goodput_per_dollar
            and adaptive.osotss_n_sla_safe >= fixed.osotss_n_sla_safe
        ),
        raw={"burstgpt_fixed": fixed, "burstgpt_adaptive": adaptive},
    )


def run_adaptive_ewma_full_backtest(
    ewma_alpha: float = FIXED_EWMA_ALPHA,
    burst_threshold: float = BURST_THRESHOLD,
    burst_alpha: float = BURST_ALPHA,
    burst_cooldown_ticks: int = BURST_COOLDOWN_TICKS,
    azure_kwargs: dict | None = None,
    burstgpt_kwargs: dict | None = None,
) -> AdaptiveEWMAReport:
    """Full two-trace adaptive EWMA comparison (fixed vs adaptive) on both public traces.

    Runs OSOTSS fixed and adaptive on Azure LLM 2024 and BurstGPT HF under
    identical conditions. Frontier improvement requires candidate >= fixed on
    goodput/$ AND n_sla_safe on BOTH traces.

    Args:
        ewma_alpha:           Fixed-mode EWMA alpha (base rate, default 0.1).
        burst_threshold:      Load ratio above which alpha is boosted (default 1.5).
        burst_alpha:          Elevated alpha during burst cooldown (default 0.5).
        burst_cooldown_ticks: Ticks boosted alpha persists (default 2).
        azure_kwargs:         Extra kwargs forwarded to the Azure runner.
        burstgpt_kwargs:      Extra kwargs forwarded to the BurstGPT runner.

    Returns:
        AdaptiveEWMAReport with per-trace metrics and overall frontier verdict.
    """
    az_kw = azure_kwargs or {}
    bg_kw = burstgpt_kwargs or {}

    az_fixed = run_online_sotss_azure_backtest(
        ewma_alpha=ewma_alpha, ewma_mode="fixed", **az_kw,
    )
    az_adaptive = run_online_sotss_azure_backtest(
        ewma_alpha=ewma_alpha, ewma_mode="adaptive",
        burst_threshold=burst_threshold, burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks, **az_kw,
    )
    bg_fixed = run_online_sotss_burstgpt_backtest(
        ewma_alpha=ewma_alpha, ewma_mode="fixed", **bg_kw,
    )
    bg_adaptive = run_online_sotss_burstgpt_backtest(
        ewma_alpha=ewma_alpha, ewma_mode="adaptive",
        burst_threshold=burst_threshold, burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks, **bg_kw,
    )

    az_imp = (
        (az_adaptive.osotss_goodput_per_dollar - az_fixed.osotss_goodput_per_dollar)
        / max(az_fixed.osotss_goodput_per_dollar, 1e-9)
        * 100.0
    )
    bg_imp = (
        (bg_adaptive.osotss_goodput_per_dollar - bg_fixed.osotss_goodput_per_dollar)
        / max(bg_fixed.osotss_goodput_per_dollar, 1e-9)
        * 100.0
    )

    is_frontier = (
        az_adaptive.osotss_goodput_per_dollar >= az_fixed.osotss_goodput_per_dollar
        and az_adaptive.osotss_n_sla_safe >= az_fixed.osotss_n_sla_safe
        and bg_adaptive.osotss_goodput_per_dollar >= bg_fixed.osotss_goodput_per_dollar
        and bg_adaptive.osotss_n_sla_safe >= bg_fixed.osotss_n_sla_safe
    )

    return AdaptiveEWMAReport(
        azure_fixed_goodput_per_dollar=az_fixed.osotss_goodput_per_dollar,
        azure_fixed_n_sla_safe=az_fixed.osotss_n_sla_safe,
        azure_adaptive_goodput_per_dollar=az_adaptive.osotss_goodput_per_dollar,
        azure_adaptive_n_sla_safe=az_adaptive.osotss_n_sla_safe,
        azure_improvement_pct=az_imp,
        burstgpt_fixed_goodput_per_dollar=bg_fixed.osotss_goodput_per_dollar,
        burstgpt_fixed_n_sla_safe=bg_fixed.osotss_n_sla_safe,
        burstgpt_adaptive_goodput_per_dollar=bg_adaptive.osotss_goodput_per_dollar,
        burstgpt_adaptive_n_sla_safe=bg_adaptive.osotss_n_sla_safe,
        burstgpt_improvement_pct=bg_imp,
        ewma_alpha=ewma_alpha,
        burst_threshold=burst_threshold,
        burst_alpha=burst_alpha,
        burst_cooldown_ticks=burst_cooldown_ticks,
        is_frontier_improvement=is_frontier,
        raw={
            "azure_fixed": az_fixed,
            "azure_adaptive": az_adaptive,
            "burstgpt_fixed": bg_fixed,
            "burstgpt_adaptive": bg_adaptive,
        },
    )
