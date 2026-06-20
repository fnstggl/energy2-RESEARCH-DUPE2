"""SRTF scheduling backtest — measures impact of predicted_output_tokens sort key.

Compares two scheduler variants on the canonical 1000-job CAISO/PJM/ERCOT trace:

  ``baseline``
      Standard greedy sort: (−priority, deadline).  No length prior.

  ``srtf``
      SRTF-enhanced greedy sort: (−priority, sla_class_rank, length_prior, deadline).
      LLM-serving jobs (realtime_inference + llm_batch_inference) receive a
      ``predicted_output_tokens`` prior proportional to ``runtime_hours``; other
      workload types receive no prior (fall back to deadline tiebreak).

Research basis
--------------
- "Scheduling the Unschedulable" (arXiv:2604.06970, SOSP 2026):
  SRTF + token magnitude priors increase P90 short-request performance by 32%
  vs FIFO and prevent 5.8× p95 regression when priors are removed.
- "TRAIL: Embedding-Based Scheduling for LLMs" (arXiv:2410.01035):
  SPRPT (Shortest Predicted Remaining Processing Time) via intermediate-layer
  embeddings achieves near-SRTF performance without clairvoyant length access.
- "Robust Length Prediction" (arXiv:2604.07931):
  ELIS iterative SRTF with encoder-based length predictor shows strong
  latency improvement on multi-tenant LLM serving clusters.

Design rules
------------
- ``actual_output_tokens`` is NEVER used as a feature (leakage guard).
- The SRTF prior is set to ``runtime_hours * SRTF_TOKENS_PER_HOUR``  as a
  directional proxy when real CARA output-length forecasts are unavailable.
- Only the sort KEY changes — the objective function and constraint checks are
  identical in both variants, ensuring a clean A/B comparison.
- Shadow-mode only: NOT a production-savings claim (docs/RESULTS.md §8).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from ..models import Job, OptimizationConfig
from .canonical_backtests import build_canonical_jobs, load_canonical_price_data

# ---------------------------------------------------------------------------
# Proxy constant
# ---------------------------------------------------------------------------

# Token-equivalent units per GPU-hour of compute.  Used as SRTF prior for LLM
# workloads when real output-length forecasts are unavailable.  500 k tokens /
# GPU-hour is a rough production-serving order-of-magnitude for 7B–13B models.
SRTF_TOKENS_PER_HOUR: float = 500_000.0

# Workload types that carry meaningful token-count priors in an LLM serving
# context.  Other types (training, fine_tuning, etc.) do not have "output
# tokens" — they receive no SRTF prior and fall back to deadline tiebreak.
SRTF_ELIGIBLE_WORKLOAD_TYPES: frozenset = frozenset({
    "realtime_inference",
    "llm_batch_inference",
})


# ---------------------------------------------------------------------------
# Job augmentation
# ---------------------------------------------------------------------------

def augment_jobs_with_srtf_priors(
    jobs: list[Job],
    tokens_per_hour: float = SRTF_TOKENS_PER_HOUR,
    eligible_types: frozenset = SRTF_ELIGIBLE_WORKLOAD_TYPES,
) -> list[Job]:
    """Return jobs with ``predicted_output_tokens`` set for SRTF-eligible types.

    Only ``realtime_inference`` and ``llm_batch_inference`` workloads receive
    a token prior (proportional to ``runtime_hours``).  All other workload
    types keep ``predicted_output_tokens = None``, preserving the original
    deadline-based tiebreak for training / data-processing jobs.

    The input jobs are NOT mutated; new ``Job`` objects are returned.

    Args:
        jobs: Original job list (no prior set).
        tokens_per_hour: Scaling constant converting runtime to token units.
        eligible_types: Workload types that receive an SRTF length prior.

    Returns:
        New list of ``Job`` objects with SRTF priors set where applicable.
    """
    augmented = []
    for j in jobs:
        if j.workload_type in eligible_types:
            augmented.append(
                dataclasses.replace(
                    j,
                    predicted_output_tokens=j.runtime_hours * tokens_per_hour,
                )
            )
        else:
            augmented.append(j)
    return augmented


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class SRTFBacktestReport:
    """Benchmark results comparing baseline vs SRTF scheduler on canonical trace.

    Positive ``goodput_per_dollar_delta`` means SRTF improves SLA-safe goodput/$.
    """

    # Workload population
    total_jobs: int
    srtf_eligible_jobs: int
    srtf_eligible_pct: float

    # Goodput / infra dollar (all jobs)
    baseline_goodput_per_dollar: float
    srtf_goodput_per_dollar: float
    goodput_per_dollar_delta: float       # srtf - baseline
    goodput_per_dollar_delta_pct: float   # % change

    # Realized energy cost
    baseline_realized_cost_usd: float
    srtf_realized_cost_usd: float
    realized_cost_delta_usd: float

    # SLA compliance
    baseline_deadline_misses: int
    srtf_deadline_misses: int

    # SRTF ordering quality
    srtf_eligible_pct_scheduled_short_first: float  # fraction of lc/batch jobs sorted short-first

    # Meta
    tokens_per_hour_proxy: float = SRTF_TOKENS_PER_HOUR
    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        return {
            "total_jobs": self.total_jobs,
            "srtf_eligible_jobs": self.srtf_eligible_jobs,
            "srtf_eligible_pct": round(self.srtf_eligible_pct, 1),
            "baseline_goodput_per_dollar": round(self.baseline_goodput_per_dollar, 6),
            "srtf_goodput_per_dollar": round(self.srtf_goodput_per_dollar, 6),
            "goodput_per_dollar_delta": round(self.goodput_per_dollar_delta, 6),
            "goodput_per_dollar_delta_pct": round(self.goodput_per_dollar_delta_pct, 4),
            "baseline_realized_cost_usd": round(self.baseline_realized_cost_usd, 2),
            "srtf_realized_cost_usd": round(self.srtf_realized_cost_usd, 2),
            "realized_cost_delta_usd": round(self.realized_cost_delta_usd, 2),
            "baseline_deadline_misses": self.baseline_deadline_misses,
            "srtf_deadline_misses": self.srtf_deadline_misses,
            "srtf_eligible_pct_scheduled_short_first": round(
                self.srtf_eligible_pct_scheduled_short_first, 3
            ),
            "tokens_per_hour_proxy": self.tokens_per_hour_proxy,
            "shadow_tag": self.shadow_tag,
        }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _compute_schedule_metrics(
    schedule,
    jobs: list[Job],
    rt_prices: dict,
    gpu_hour_usd: float = 2.0,
    migration_network_usd: float = 0.5,
) -> tuple[float, float, int]:
    """Return (sla_safe_goodput_per_dollar, realized_cost, deadline_misses)."""
    from datetime import timedelta

    from ..backtesting.evaluator import evaluate_schedule

    realized = evaluate_schedule(
        schedule, jobs, rt_prices,
        {r: {} for r in rt_prices}, warn_on_missing=False,
    )
    energy_cost = realized.total_energy_cost_usd

    job_by_id = {j.job_id: j for j in jobs}
    baseline_sched_by_id = {}  # no baseline region needed for simple cost
    gpu_infra = 0.0
    migrations = 0
    deadline_misses = 0
    goodput = 0.0

    for d in schedule:
        job = job_by_id.get(d.job_id)
        if job is None:
            continue
        gpu_infra += gpu_hour_usd * max(0, job.gpu_count) * job.runtime_hours
        completion = d.end_time
        if completion > job.deadline:
            deadline_misses += 1
        else:
            unit = max(0.0, job.gpu_count * job.runtime_hours)
            goodput += unit if unit > 0 else job.runtime_hours

    network_cost = migrations * migration_network_usd
    total_infra = energy_cost + gpu_infra + network_cost
    gp_per_dollar = goodput / total_infra if total_infra > 0 else 0.0
    return gp_per_dollar, total_infra, deadline_misses


def _pct_eligible_short_first(
    schedule,
    jobs: list[Job],
    eligible_types: frozenset = SRTF_ELIGIBLE_WORKLOAD_TYPES,
) -> float:
    """Fraction of SRTF-eligible jobs that were processed in short-runtime order.

    Checks that among consecutive pairs of eligible jobs in the SORTED order
    used by the scheduler, shorter-runtime ones precede longer ones.
    """
    eligible = [
        j for j in jobs if j.workload_type in eligible_types
    ]
    if len(eligible) < 2:
        return 1.0
    sched_by_id = {d.job_id: d for d in schedule}
    # Sort eligible jobs by their scheduled start time (proxy for processing order).
    eligible_sorted = sorted(
        eligible,
        key=lambda j: sched_by_id[j.job_id].start_time if j.job_id in sched_by_id else j.deadline,
    )
    ordered = 0
    pairs = 0
    for i in range(len(eligible_sorted) - 1):
        j1 = eligible_sorted[i]
        j2 = eligible_sorted[i + 1]
        pairs += 1
        if j1.runtime_hours <= j2.runtime_hours:
            ordered += 1
    return ordered / pairs if pairs > 0 else 1.0


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_srtf_backtest(
    seed: int = 20260201,
    job_count: int = 1000,
    method: str = "greedy",
    gpu_hour_usd: float = 2.0,
    migration_network_usd: float = 0.5,
    tokens_per_hour: float = SRTF_TOKENS_PER_HOUR,
) -> SRTFBacktestReport:
    """Run SRTF scheduling benchmark on the canonical 1000-job CAISO/PJM/ERCOT trace.

    Builds the canonical job trace, loads real energy price data, then runs
    two scheduler variants:

    1. **Baseline**: No length prior — sort by (−priority, deadline).
    2. **SRTF**: With ``predicted_output_tokens`` from runtime_hours proxy —
       sort by (−priority, sla_class_rank, length_prior, deadline).

    Args:
        seed: Canonical job trace seed (matches CANONICAL_SEED = 20260201).
        job_count: Number of jobs to generate.
        method: Scheduler solve method.
        gpu_hour_usd: GPU compute cost for goodput/$ KPI.
        migration_network_usd: Migration network cost per event.
        tokens_per_hour: Token proxy constant (runtime_hours × this = prior).

    Returns:
        ``SRTFBacktestReport`` with per-variant metrics and delta.

    Raises:
        FileNotFoundError: If canonical price CSV files are absent.
    """
    from ..optimization.scheduler import JobScheduler

    # 1. Build canonical jobs.
    raw_jobs = build_canonical_jobs(seed=seed, count=job_count)
    baseline_jobs = raw_jobs  # no predicted_output_tokens set

    # 2. SRTF variant: augment eligible jobs with length priors.
    srtf_jobs = augment_jobs_with_srtf_priors(
        raw_jobs,
        tokens_per_hour=tokens_per_hour,
        eligible_types=SRTF_ELIGIBLE_WORKLOAD_TYPES,
    )

    # 3. Load canonical price data.
    da, rt = load_canonical_price_data()
    carbon = {r: {} for r in da}

    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)

    # 4. Baseline schedule (no SRTF priors).
    baseline_scheduler = JobScheduler(cfg)
    baseline_result = baseline_scheduler.solve(baseline_jobs, da, carbon, method=method)
    baseline_schedule = baseline_result.schedule

    # 5. SRTF schedule (with predicted_output_tokens set for eligible jobs).
    srtf_scheduler = JobScheduler(cfg)
    srtf_result = srtf_scheduler.solve(srtf_jobs, da, carbon, method=method)
    srtf_schedule = srtf_result.schedule

    # 6. Compute KPIs.
    baseline_gpd, baseline_cost, baseline_misses = _compute_schedule_metrics(
        baseline_schedule, baseline_jobs, rt, gpu_hour_usd, migration_network_usd,
    )
    srtf_gpd, srtf_cost, srtf_misses = _compute_schedule_metrics(
        srtf_schedule, srtf_jobs, rt, gpu_hour_usd, migration_network_usd,
    )

    # 7. SRTF ordering quality.
    pct_short_first = _pct_eligible_short_first(srtf_schedule, srtf_jobs)

    eligible_count = sum(
        1 for j in raw_jobs if j.workload_type in SRTF_ELIGIBLE_WORKLOAD_TYPES
    )

    delta = srtf_gpd - baseline_gpd
    delta_pct = (delta / baseline_gpd * 100.0) if baseline_gpd > 0 else 0.0

    return SRTFBacktestReport(
        total_jobs=len(raw_jobs),
        srtf_eligible_jobs=eligible_count,
        srtf_eligible_pct=eligible_count / len(raw_jobs) * 100.0,
        baseline_goodput_per_dollar=baseline_gpd,
        srtf_goodput_per_dollar=srtf_gpd,
        goodput_per_dollar_delta=delta,
        goodput_per_dollar_delta_pct=delta_pct,
        baseline_realized_cost_usd=baseline_cost,
        srtf_realized_cost_usd=srtf_cost,
        realized_cost_delta_usd=srtf_cost - baseline_cost,
        baseline_deadline_misses=baseline_misses,
        srtf_deadline_misses=srtf_misses,
        srtf_eligible_pct_scheduled_short_first=pct_short_first,
        tokens_per_hour_proxy=tokens_per_hour,
    )
