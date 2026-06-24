"""Multi-seed stochastic audit — Five-Failure Rule benchmark realism check.

Run 2026-06-24.  Five-Failure Rule active (5/5): stop adding new modules;
focus on integration, replay validation, benchmark realism, bottleneck
diagnosis, architecture simplification.

RESEARCH QUESTION (GAP_ANALYSIS Q10-11):
    All OSOTSS/AMCSG results use seed=42.  The BurstGPT n_sla_safe gap
    (OSOTSS 5849 vs AMCSG 5864, 15 requests; best-OSSC 5861 vs AMCSG 5864,
    3 requests) may be a seed artifact rather than a structural limitation.

    Specifically: AMCSG's fixed higher-c schedule absorbs stochastic spot
    interruptions globally.  For a given seed, the 15 or 3 remaining failures
    arise from Binomial(c_spot, 0.9982) capacity shortfalls on specific ticks.
    With a different seed, different ticks fail — and AMCSG's higher-c may
    absorb those differently from OSOTSS's leaner c-schedule.

    If n_sla_safe(OSOTSS) >= n_sla_safe(AMCSG) on some seeds, the gap is
    not structural.  If the gap is consistently negative across all seeds, it
    is structural (the leaner OSOTSS c-schedule is genuinely more exposed).

METHODOLOGY:
    For seeds ∈ {42, 123, 456, 789, 1337}:
        - Run run_online_sotss_burstgpt_backtest(seed=seed)  → both AMCSG and
          OSOTSS n_sla_safe / goodput/$ in one call (same-conditions).
        - Run run_online_sotss_azure_backtest(seed=seed) similarly.
    Report: mean, std, min, max of the gap (OSOTSS − AMCSG) across seeds.

Same-conditions checklist:
    * Same trace (Azure LLM 2024, BurstGPT HF — both)
    * Same SLA (10s / 30s)
    * Same cost denominator, GPU-hour accounting, physics
    * Same capacity model, pricing model, decision-time information (causal EWMA)
    * Same evaluation method (GSF spot-fleet simulation)
    * Only the RNG seed varies
    * No new optimizer logic — purely varying the evaluation seed

No new production decision, no new optimizer.  This is benchmark realism only.
Classification target: Benchmark Realism Audit (not a frontier improvement claim).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import List

from .srtf_serving_backtest import (
    run_online_sotss_azure_backtest,
    run_online_sotss_burstgpt_backtest,
)

# Canonical multi-seed set — 42 is the existing canonical seed
AUDIT_SEEDS: List[int] = [42, 123, 456, 789, 1337]

# Canonical references from ROADMAP.md (seed=42 results)
CANONICAL_AMCSG_BURSTGPT_N_SLA_SAFE = 5864
CANONICAL_OSOTSS_BURSTGPT_N_SLA_SAFE = 5849
CANONICAL_AMCSG_AZURE_N_SLA_SAFE = 5823
CANONICAL_OSOTSS_AZURE_N_SLA_SAFE = 5823


@dataclass
class SeedResult:
    """Single-seed AMCSG vs OSOTSS result on one trace."""

    trace: str
    seed: int
    amcsg_n_sla_safe: int
    osotss_n_sla_safe: int
    amcsg_goodput_per_dollar: float
    osotss_goodput_per_dollar: float
    gap_n_sla_safe: int           # osotss − amcsg (negative = osotss loses)
    osotss_vs_amcsg_pct: float    # goodput/$ gain of OSOTSS over AMCSG

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "seed": self.seed,
            "amcsg_n_sla_safe": self.amcsg_n_sla_safe,
            "osotss_n_sla_safe": self.osotss_n_sla_safe,
            "amcsg_goodput_per_dollar": round(self.amcsg_goodput_per_dollar, 2),
            "osotss_goodput_per_dollar": round(self.osotss_goodput_per_dollar, 2),
            "gap_n_sla_safe": self.gap_n_sla_safe,
            "osotss_vs_amcsg_pct": round(self.osotss_vs_amcsg_pct, 4),
        }


@dataclass
class TraceAuditSummary:
    """Multi-seed summary for one trace."""

    trace: str
    seeds: List[int]
    per_seed: List[SeedResult]
    amcsg_n_sla_safe_mean: float
    amcsg_n_sla_safe_std: float
    amcsg_n_sla_safe_min: int
    amcsg_n_sla_safe_max: int
    osotss_n_sla_safe_mean: float
    osotss_n_sla_safe_std: float
    osotss_n_sla_safe_min: int
    osotss_n_sla_safe_max: int
    gap_mean: float               # mean(osotss − amcsg) — negative = structurally worse
    gap_std: float
    gap_min: int
    gap_max: int
    seeds_osotss_wins: int        # count(seeds where osotss_n_sla_safe >= amcsg_n_sla_safe)
    seeds_osotss_loses: int       # count(seeds where osotss_n_sla_safe < amcsg_n_sla_safe)
    goodput_gap_is_structural: bool   # True if gap < 0 for every seed
    goodput_gap_is_noise: bool        # True if gap >= 0 for at least one seed

    def to_dict(self) -> dict:
        return {
            "trace": self.trace,
            "seeds": self.seeds,
            "per_seed": [r.to_dict() for r in self.per_seed],
            "amcsg_n_sla_safe_mean": round(self.amcsg_n_sla_safe_mean, 2),
            "amcsg_n_sla_safe_std": round(self.amcsg_n_sla_safe_std, 3),
            "amcsg_n_sla_safe_min": self.amcsg_n_sla_safe_min,
            "amcsg_n_sla_safe_max": self.amcsg_n_sla_safe_max,
            "osotss_n_sla_safe_mean": round(self.osotss_n_sla_safe_mean, 2),
            "osotss_n_sla_safe_std": round(self.osotss_n_sla_safe_std, 3),
            "osotss_n_sla_safe_min": self.osotss_n_sla_safe_min,
            "osotss_n_sla_safe_max": self.osotss_n_sla_safe_max,
            "gap_mean": round(self.gap_mean, 2),
            "gap_std": round(self.gap_std, 3),
            "gap_min": self.gap_min,
            "gap_max": self.gap_max,
            "seeds_osotss_wins": self.seeds_osotss_wins,
            "seeds_osotss_loses": self.seeds_osotss_loses,
            "goodput_gap_is_structural": self.goodput_gap_is_structural,
            "goodput_gap_is_noise": self.goodput_gap_is_noise,
        }


@dataclass
class MultiSeedAuditReport:
    """Complete multi-seed stochastic audit report.

    Tests whether the BurstGPT 15-request n_sla_safe gap (OSOTSS vs AMCSG)
    is structural (consistent across seeds) or a seed artifact.
    """

    audit_seeds: List[int]
    azure_summary: TraceAuditSummary
    burstgpt_summary: TraceAuditSummary
    # Cross-trace finding
    burstgpt_gap_is_structural: bool
    azure_gap_is_structural: bool
    conclusion: str  # human-readable finding

    def to_dict(self) -> dict:
        return {
            "audit_seeds": self.audit_seeds,
            "azure_summary": self.azure_summary.to_dict(),
            "burstgpt_summary": self.burstgpt_summary.to_dict(),
            "burstgpt_gap_is_structural": self.burstgpt_gap_is_structural,
            "azure_gap_is_structural": self.azure_gap_is_structural,
            "conclusion": self.conclusion,
        }


def _summarize_trace(trace: str, per_seed: List[SeedResult]) -> TraceAuditSummary:
    amcsg_vals = [r.amcsg_n_sla_safe for r in per_seed]
    osotss_vals = [r.osotss_n_sla_safe for r in per_seed]
    gaps = [r.gap_n_sla_safe for r in per_seed]
    n = len(per_seed)

    amcsg_std = statistics.stdev(amcsg_vals) if n > 1 else 0.0
    osotss_std = statistics.stdev(osotss_vals) if n > 1 else 0.0
    gap_std = statistics.stdev(gaps) if n > 1 else 0.0

    wins = sum(1 for g in gaps if g >= 0)
    loses = sum(1 for g in gaps if g < 0)

    return TraceAuditSummary(
        trace=trace,
        seeds=[r.seed for r in per_seed],
        per_seed=per_seed,
        amcsg_n_sla_safe_mean=statistics.mean(amcsg_vals),
        amcsg_n_sla_safe_std=amcsg_std,
        amcsg_n_sla_safe_min=min(amcsg_vals),
        amcsg_n_sla_safe_max=max(amcsg_vals),
        osotss_n_sla_safe_mean=statistics.mean(osotss_vals),
        osotss_n_sla_safe_std=osotss_std,
        osotss_n_sla_safe_min=min(osotss_vals),
        osotss_n_sla_safe_max=max(osotss_vals),
        gap_mean=statistics.mean(gaps),
        gap_std=gap_std,
        gap_min=min(gaps),
        gap_max=max(gaps),
        seeds_osotss_wins=wins,
        seeds_osotss_loses=loses,
        goodput_gap_is_structural=(wins == 0),
        goodput_gap_is_noise=(wins > 0),
    )


def run_multi_seed_burstgpt_audit(
    seeds: List[int] = AUDIT_SEEDS,
    fixed_c: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
) -> TraceAuditSummary:
    """Multi-seed AMCSG vs OSOTSS audit on BurstGPT HF.

    For each seed in ``seeds``, runs ``run_online_sotss_burstgpt_backtest``
    which returns AMCSG (gate=12.5%) and OSOTSS (gate=100%→min-stable)
    results for that seed under identical conditions.

    Args:
        seeds:       List of RNG seeds to sweep.
        fixed_c:     Replica count for time-warp calibration.
        target_rho:  Target utilisation.
        job_limit:   Request cap (5880 = canonical).

    Returns:
        TraceAuditSummary with per-seed results and gap statistics.
    """
    per_seed: List[SeedResult] = []
    for seed in seeds:
        report = run_online_sotss_burstgpt_backtest(
            fixed_c=fixed_c,
            target_rho=target_rho,
            job_limit=job_limit,
            seed=seed,
        )
        gap = report.osotss_n_sla_safe - report.amcsg_n_sla_safe
        pct = report.osotss_vs_amcsg_pct
        per_seed.append(SeedResult(
            trace="burstgpt_hf",
            seed=seed,
            amcsg_n_sla_safe=report.amcsg_n_sla_safe,
            osotss_n_sla_safe=report.osotss_n_sla_safe,
            amcsg_goodput_per_dollar=report.amcsg_goodput_per_dollar,
            osotss_goodput_per_dollar=report.osotss_goodput_per_dollar,
            gap_n_sla_safe=gap,
            osotss_vs_amcsg_pct=pct,
        ))
    return _summarize_trace("burstgpt_hf", per_seed)


def run_multi_seed_azure_audit(
    seeds: List[int] = AUDIT_SEEDS,
    fixed_c: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
) -> TraceAuditSummary:
    """Multi-seed AMCSG vs OSOTSS audit on Azure LLM 2024.

    For each seed in ``seeds``, runs ``run_online_sotss_azure_backtest``
    which returns AMCSG (gate=12.5%) and OSOTSS results for that seed.

    Args:
        seeds:       List of RNG seeds to sweep.
        fixed_c:     Replica count for time-warp calibration.
        target_rho:  Target utilisation.
        job_limit:   Request cap (5880 = canonical).

    Returns:
        TraceAuditSummary with per-seed results and gap statistics.
    """
    per_seed: List[SeedResult] = []
    for seed in seeds:
        report = run_online_sotss_azure_backtest(
            fixed_c=fixed_c,
            target_rho=target_rho,
            job_limit=job_limit,
            seed=seed,
        )
        gap = report.osotss_n_sla_safe - report.amcsg_n_sla_safe
        pct = report.osotss_vs_amcsg_pct
        per_seed.append(SeedResult(
            trace="azure_llm_2024",
            seed=seed,
            amcsg_n_sla_safe=report.amcsg_n_sla_safe,
            osotss_n_sla_safe=report.osotss_n_sla_safe,
            amcsg_goodput_per_dollar=report.amcsg_goodput_per_dollar,
            osotss_goodput_per_dollar=report.osotss_goodput_per_dollar,
            gap_n_sla_safe=gap,
            osotss_vs_amcsg_pct=pct,
        ))
    return _summarize_trace("azure_llm_2024", per_seed)


def run_multi_seed_audit(
    seeds: List[int] = AUDIT_SEEDS,
    fixed_c: int = 4,
    target_rho: float = 0.85,
    job_limit: int = 5880,
) -> MultiSeedAuditReport:
    """Full multi-seed stochastic audit on Azure LLM 2024 + BurstGPT HF.

    Same-conditions: both traces use the same seed list, same physics,
    same provisioning model.  Only the trace (arrival pattern, output lengths,
    SLA budget) differs across the two.

    Args:
        seeds:       List of RNG seeds to sweep (default: 5 canonical seeds).
        fixed_c:     Replica count for time-warp calibration.
        target_rho:  Target utilisation.
        job_limit:   Request cap (5880 = canonical).

    Returns:
        MultiSeedAuditReport with per-trace summaries and cross-trace finding.
    """
    azure_summary = run_multi_seed_azure_audit(
        seeds=seeds, fixed_c=fixed_c, target_rho=target_rho, job_limit=job_limit,
    )
    burstgpt_summary = run_multi_seed_burstgpt_audit(
        seeds=seeds, fixed_c=fixed_c, target_rho=target_rho, job_limit=job_limit,
    )

    bpg_structural = burstgpt_summary.goodput_gap_is_structural
    az_structural = azure_summary.goodput_gap_is_structural

    if bpg_structural and az_structural:
        conclusion = (
            "STRUCTURAL GAP: OSOTSS n_sla_safe < AMCSG n_sla_safe on EVERY seed "
            "on BOTH traces. The gap is structural (leaner c-schedule is genuinely "
            "more exposed to spot interruptions). Further oracle-convergence approaches "
            "are unlikely to close it without a stochastic oracle."
        )
    elif bpg_structural and not az_structural:
        conclusion = (
            "MIXED: BurstGPT gap is structural (OSOTSS loses on all seeds); "
            "Azure gap is NOT structural (OSOTSS ties or wins on some seeds). "
            "Azure n_sla_safe parity is achievable; BurstGPT 15-request gap is hard."
        )
    elif not bpg_structural:
        conclusion = (
            "SEED ARTIFACT (partial or full): OSOTSS matches or exceeds AMCSG "
            "n_sla_safe on at least one seed on BurstGPT. The single-seed (42) "
            "gap is not a universal structural limitation. Multi-seed evaluation "
            "is warranted before accepting the gap as fundamental."
        )
    else:
        conclusion = "UNDETERMINED — mixed findings across traces."

    return MultiSeedAuditReport(
        audit_seeds=seeds,
        azure_summary=azure_summary,
        burstgpt_summary=burstgpt_summary,
        burstgpt_gap_is_structural=bpg_structural,
        azure_gap_is_structural=az_structural,
        conclusion=conclusion,
    )
