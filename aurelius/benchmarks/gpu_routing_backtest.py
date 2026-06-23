"""GPU-routing benchmark: region_gpu_types extension to the canonical backtest.

Measures the impact of enabling GpuPlacementScorer on the canonical 1000-job
CAISO/PJM/ERCOT trace. Augments jobs with WORKLOAD_DEFAULT_SLA_CLASS so that
realtime_inference workloads receive ``latency_critical`` SLA scoring and the
GPU placement penalty fires correctly.

Key synthetic metadata
-----------------------
- ``CANONICAL_REGION_GPU_TYPES``: us-west→a100, us-east→h100, us-south→t4
  Based on CARA fleet composition (major CA AI clusters = A100; Northeast H100
  capacity = PJM zone; Texas lower-tier = T4/K80 fleet in ERCOT zone).
- Synthetic ``TTFTShadowPrior``: CARA-calibrated p50 medians per GPU type:
  H100≈0.12 s, A100≈0.28 s, T4≈0.95 s (8× spread; CARA paper cites 9×
  TTFT p99 spread across instance types; arXiv:2604.07472).

Benchmark variants
------------------
Both variants run on the SAME canonical 1000-job trace (same seed, window,
price data). The only difference is whether ``GpuPlacementScorer`` is active:

  ``baseline``
      ``JobScheduler(cfg)`` — no GPU routing (existing behavior).

  ``gpu_routing``
      ``JobScheduler(cfg, gpu_placement_scorer=scorer, region_gpu_types=...)``.
      ``latency_critical`` jobs receive a TTFT-based penalty that makes H100
      placements (us-east) rank higher than A100 (us-west) or T4 (us-south).

Primary KPIs
------------
- ``pct_latency_critical_on_best_gpu``: fraction of ``latency_critical`` jobs
  routed to H100 (us-east).  Higher is better.
- ``mean_gpu_penalty_latency_critical``: mean TTFT latency penalty score
  across all ``latency_critical`` jobs.  Lower is better.
- ``realized_energy_cost_delta_usd``: change in realized energy cost from GPU
  routing vs baseline (positive = GPU routing cost more energy; negative =
  less).  Expected ≈ 0 for small lc fraction; the goodput gain is TTFT, not $.
- ``routing_improvement_pp``: pp gain in H100-routing rate for lc jobs.

This is a simulator / directional-only result.  Not a production-savings claim
(docs/RESULTS.md §8).  Shadow-mode only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..models import WORKLOAD_DEFAULT_SLA_CLASS, Job, OptimizationConfig, ScheduleDecision
from .canonical_backtests import build_canonical_jobs, load_canonical_price_data

# ---------------------------------------------------------------------------
# Synthetic GPU TTFT prior constants — CARA-calibrated
# ---------------------------------------------------------------------------

# CARA dataset (asdwb/cara_latency_prediction, 76 825 rows, 5 GPU types):
# Empirically observed p50 TTFT values for 7B/13B-class models on three
# GPU generations.  These drive the GpuPlacementScorer penalty ranking.
SYNTHETIC_GPU_TTFT_P50_S: dict[str, float] = {
    "h100": 0.12,  # H100 SXM5 — fastest; ~120 ms median for 7B/13B
    "a100": 0.28,  # A100 80 GB — middle; ~280 ms
    "t4":   0.95,  # T4 — slowest; ~950 ms (~8× slower than H100)
}

# Rows per GPU type in the synthetic prior.  Must be >= GpuPlacementConfig.
# min_subgroup_rows (default 50) so the scorer trusts the estimate.
_SYNTHETIC_ROWS_PER_GPU: int = 200

# Synthetic row model size token for the prior (model-7b_h100 etc.)
_SYNTHETIC_MODEL_PREFIX: str = "model-7b"

# ---------------------------------------------------------------------------
# Region → GPU type mapping (CARA fleet composition)
# ---------------------------------------------------------------------------

# Canonical regions defined in canonical_backtests.CANONICAL_REGIONS.
CANONICAL_REGION_GPU_TYPES: dict[str, str] = {
    "us-west":  "a100",  # CAISO zone — major CA AI clusters (A100 fleet)
    "us-east":  "h100",  # PJM zone   — Northeast H100 capacity
    "us-south": "t4",    # ERCOT zone — Texas lower-tier GPU capacity (T4 fleet)
}

# The GPU type that GpuPlacementScorer will rank #1 (lowest TTFT).
BEST_GPU_TYPE: str = "h100"
BEST_GPU_REGION: str = "us-east"  # Must match CANONICAL_REGION_GPU_TYPES above.


# ---------------------------------------------------------------------------
# Synthetic prior builder
# ---------------------------------------------------------------------------

def build_synthetic_prior(
    seed: int = 42,
    rows_per_gpu: int = _SYNTHETIC_ROWS_PER_GPU,
    ttft_values: Optional[dict] = None,
) -> "TTFTShadowPrior":  # type: ignore[name-defined]
    """Build a ``TTFTShadowPrior`` calibrated to CARA median TTFT values.

    Generates ``rows_per_gpu`` synthetic rows per GPU type (H100/A100/T4)
    with values drawn from a narrow Gaussian around each GPU type's known
    median TTFT.  The resulting prior will have:

    - ``by_gpu["h100"] ≈ 0.12 s``
    - ``by_gpu["a100"] ≈ 0.28 s``
    - ``by_gpu["t4"]   ≈ 0.95 s``
    - ``by_gpu_counts[g] = rows_per_gpu`` for each GPU type

    Args:
        seed: RNG seed for reproducibility.
        rows_per_gpu: Rows generated per GPU type.  Must be ≥ 50 (default
            ``GpuPlacementConfig.min_subgroup_rows``) or the scorer will
            return ``insufficient_sample`` instead of a real penalty.
        ttft_values: Override median TTFT values.  Defaults to
            ``SYNTHETIC_GPU_TTFT_P50_S``.

    Returns:
        A fitted ``TTFTShadowPrior`` instance.
    """
    from ..forecasting.ttft_shadow_prior import TTFTShadowPrior

    rng = random.Random(seed)
    medians = ttft_values or SYNTHETIC_GPU_TTFT_P50_S
    rows = []
    for gpu_type, median_s in medians.items():
        for _ in range(rows_per_gpu):
            # 10 % coefficient of variation — realistic within a GPU type.
            noise = rng.gauss(0.0, median_s * 0.10)
            actual_ttft = max(0.01, median_s + noise)
            rows.append({
                "instance_type": f"{_SYNTHETIC_MODEL_PREFIX}_{gpu_type}",
                "actual_ttft_s": actual_ttft,
                "num_prompt_tokens": rng.randint(50, 800),
            })

    prior = TTFTShadowPrior()
    prior.fit_from_rows(rows)
    return prior


# ---------------------------------------------------------------------------
# Job augmentation
# ---------------------------------------------------------------------------

def augment_jobs_with_sla_class(jobs: list[Job]) -> list[Job]:
    """Return jobs with ``sla_class`` set from ``WORKLOAD_DEFAULT_SLA_CLASS``.

    The canonical job builder does not set ``sla_class`` — it defaults to
    ``"best_effort"``.  This function copies each job, setting ``sla_class``
    from the workload-type default so that ``realtime_inference`` jobs receive
    ``"latency_critical"`` SLA scoring (the only class that activates
    ``GpuPlacementScorer`` penalties).

    The original jobs are NOT mutated.
    """
    import dataclasses

    augmented = []
    for j in jobs:
        sla_class = WORKLOAD_DEFAULT_SLA_CLASS.get(j.workload_type, "best_effort")
        augmented.append(dataclasses.replace(j, sla_class=sla_class))
    return augmented


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class GpuRoutingReport:
    """Benchmark results comparing baseline vs GPU-routing scheduler.

    Fields starting with ``baseline_`` and ``gpu_routing_`` contain per-policy
    numbers.  Fields ending in ``_delta`` are ``gpu_routing - baseline``.
    """

    # Population stats
    total_jobs: int
    latency_critical_jobs: int
    latency_critical_pct: float

    # Routing quality for latency_critical jobs
    baseline_pct_on_best_gpu: float        # fraction of lc jobs on H100 in baseline
    gpu_routing_pct_on_best_gpu: float     # fraction of lc jobs on H100 with routing
    routing_improvement_pp: float          # percentage-point gain (>0 = better)

    # TTFT penalty scores (mean over latency_critical jobs)
    baseline_mean_gpu_penalty: float
    gpu_routing_mean_gpu_penalty: float
    penalty_reduction: float               # baseline - gpu_routing (>0 = better)

    # Energy cost
    baseline_realized_energy_cost_usd: float
    gpu_routing_realized_energy_cost_usd: float
    realized_energy_cost_delta_usd: float  # gpu_routing - baseline

    # SLA-safe goodput / infra dollar (all jobs)
    baseline_goodput_per_dollar: float
    gpu_routing_goodput_per_dollar: float
    goodput_per_dollar_delta: float        # gpu_routing - baseline

    # SLA-safe goodput / infra dollar (latency_critical subset only)
    baseline_lc_goodput_per_dollar: float
    gpu_routing_lc_goodput_per_dollar: float
    lc_goodput_per_dollar_delta: float

    # Meta
    best_gpu_type: str = BEST_GPU_TYPE
    best_gpu_region: str = BEST_GPU_REGION
    region_gpu_types: dict = field(default_factory=dict)
    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        return {
            "total_jobs": self.total_jobs,
            "latency_critical_jobs": self.latency_critical_jobs,
            "latency_critical_pct": round(self.latency_critical_pct, 1),
            "baseline_pct_on_best_gpu": round(self.baseline_pct_on_best_gpu, 3),
            "gpu_routing_pct_on_best_gpu": round(self.gpu_routing_pct_on_best_gpu, 3),
            "routing_improvement_pp": round(self.routing_improvement_pp, 3),
            "baseline_mean_gpu_penalty": round(self.baseline_mean_gpu_penalty, 4),
            "gpu_routing_mean_gpu_penalty": round(self.gpu_routing_mean_gpu_penalty, 4),
            "penalty_reduction": round(self.penalty_reduction, 4),
            "baseline_realized_energy_cost_usd": round(
                self.baseline_realized_energy_cost_usd, 2
            ),
            "gpu_routing_realized_energy_cost_usd": round(
                self.gpu_routing_realized_energy_cost_usd, 2
            ),
            "realized_energy_cost_delta_usd": round(self.realized_energy_cost_delta_usd, 2),
            "baseline_goodput_per_dollar": round(self.baseline_goodput_per_dollar, 6),
            "gpu_routing_goodput_per_dollar": round(self.gpu_routing_goodput_per_dollar, 6),
            "goodput_per_dollar_delta": round(self.goodput_per_dollar_delta, 6),
            "baseline_lc_goodput_per_dollar": round(self.baseline_lc_goodput_per_dollar, 6),
            "gpu_routing_lc_goodput_per_dollar": round(
                self.gpu_routing_lc_goodput_per_dollar, 6
            ),
            "lc_goodput_per_dollar_delta": round(self.lc_goodput_per_dollar_delta, 6),
            "best_gpu_type": self.best_gpu_type,
            "best_gpu_region": self.best_gpu_region,
            "region_gpu_types": self.region_gpu_types,
            "shadow_tag": self.shadow_tag,
        }


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def _compute_goodput_per_dollar(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
    rt_prices: dict,
    gpu_hour_usd: float,
    migration_network_usd: float,
    baseline_regions: dict,
    job_filter: Optional[set] = None,
) -> tuple[float, float, float]:
    """Return (sla_safe_goodput, total_infra_usd, goodput_per_dollar).

    Args:
        job_filter: If provided, restrict to job IDs in this set.
    """
    from datetime import timedelta

    from ..backtesting.evaluator import evaluate_schedule

    if job_filter is not None:
        filtered_schedule = [d for d in schedule if d.job_id in job_filter]
        filtered_jobs = [j for j in jobs if j.job_id in job_filter]
    else:
        filtered_schedule = schedule
        filtered_jobs = jobs

    if not filtered_schedule:
        return 0.0, 0.0, 0.0

    job_by_id = {j.job_id: j for j in filtered_jobs}
    realized = evaluate_schedule(
        filtered_schedule, filtered_jobs, rt_prices,
        {r: {} for r in rt_prices}, warn_on_missing=False,
    )
    energy = realized.total_energy_cost_usd

    gpu_infra = 0.0
    migrations = 0
    deadline_misses = 0
    goodput = 0.0
    for d in filtered_schedule:
        job = job_by_id.get(d.job_id)
        if job is None:
            continue
        gpu_infra += gpu_hour_usd * max(0, job.gpu_count) * job.runtime_hours
        moved = d.region != baseline_regions.get(d.job_id, d.region)
        if moved:
            migrations += 1
        completion = d.end_time
        if moved and job.migration_cost_hours:
            completion += timedelta(hours=job.migration_cost_hours)
        if completion > job.deadline:
            deadline_misses += 1
        else:
            unit = max(0.0, job.gpu_count * job.runtime_hours)
            goodput += unit if unit > 0 else job.runtime_hours

    network_cost = migrations * migration_network_usd
    total_infra = energy + gpu_infra + network_cost
    gp_per_dollar = goodput / total_infra if total_infra > 0 else 0.0
    return goodput, total_infra, gp_per_dollar


def _compute_ttft_penalty(
    schedule: list[ScheduleDecision],
    lc_jobs: list[Job],
    region_gpu_types: dict,
    scorer,
) -> tuple[float, float]:
    """Return (mean_penalty, pct_on_best_gpu) for latency_critical jobs."""
    if not lc_jobs:
        return 0.0, 0.0

    sched_by_id = {d.job_id: d for d in schedule}
    penalties = []
    on_best = 0
    for job in lc_jobs:
        d = sched_by_id.get(job.job_id)
        if d is None:
            continue
        gtype = region_gpu_types.get(d.region)
        if gtype is None:
            penalties.append(0.0)
            continue
        if gtype == BEST_GPU_TYPE:
            on_best += 1

        # Score against the full peer set (all 3 GPU types).
        ps = scorer.score(
            gpu_type=gtype,
            model_size=None,
            prompt_tokens=None,
            sla_class=job.sla_class,
            peer_ttft_p50s={g: t for g, t in SYNTHETIC_GPU_TTFT_P50_S.items()},
        )
        penalties.append(ps.latency_penalty)

    mean_penalty = sum(penalties) / len(penalties) if penalties else 0.0
    pct_on_best = on_best / len(lc_jobs) if lc_jobs else 0.0
    return mean_penalty, pct_on_best


def run_gpu_routing_backtest(
    seed: int = 20260201,
    job_count: int = 1000,
    method: str = "greedy",
    region_gpu_types: Optional[dict] = None,
    gpu_hour_usd: float = 2.0,
    migration_network_usd: float = 0.5,
    prior_seed: int = 42,
    prior_rows_per_gpu: int = _SYNTHETIC_ROWS_PER_GPU,
) -> GpuRoutingReport:
    """Run the GPU routing benchmark on the canonical 1000-job trace.

    Loads the canonical price data, builds the canonical job trace, augments
    jobs with ``sla_class`` derived from workload type, then runs two
    scheduler variants — baseline (no GPU routing) and GPU-routing (scorer
    enabled) — and compares routing quality for ``latency_critical`` jobs.

    Args:
        seed: Canonical job trace seed (matches ``CANONICAL_SEED = 20260201``).
        job_count: Number of jobs to generate.
        method: Scheduler solve method (``"greedy"`` for the canonical run).
        region_gpu_types: GPU type per region.  Defaults to
            ``CANONICAL_REGION_GPU_TYPES``.
        gpu_hour_usd: GPU compute cost used in goodput/$ KPI.
        migration_network_usd: Migration network cost per event.
        prior_seed: RNG seed for the synthetic prior.
        prior_rows_per_gpu: Rows per GPU type in the synthetic prior.

    Returns:
        ``GpuRoutingReport`` with per-variant metrics and routing delta.

    Raises:
        FileNotFoundError: If the canonical price CSV files are absent.
    """
    from ..forecasting.gpu_placement_scorer import GpuPlacementConfig, GpuPlacementScorer

    # Phase 3: route through the canonical AureliusOptimizer (energy policy);
    # GPU placement is passed through as a scheduler kwarg, behavior unchanged.
    from ..optimizer import AureliusOptimizer

    rgt = region_gpu_types or CANONICAL_REGION_GPU_TYPES

    # 1. Build canonical jobs and augment with sla_class.
    raw_jobs = build_canonical_jobs(seed=seed, count=job_count)
    jobs = augment_jobs_with_sla_class(raw_jobs)
    lc_jobs = [j for j in jobs if j.sla_class == "latency_critical"]

    # 2. Load canonical price data.
    da, rt = load_canonical_price_data()
    carbon = {r: {} for r in da}

    # 3. Build synthetic TTFT prior and GPU placement scorer.
    prior = build_synthetic_prior(seed=prior_seed, rows_per_gpu=prior_rows_per_gpu)
    scorer = GpuPlacementScorer(
        prior=prior,
        config=GpuPlacementConfig(
            enabled=True,
            min_subgroup_rows=50,
            penalty_floor=0.05,
            penalty_ceil=0.50,
        ),
    )

    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)

    # 4. Baseline: no GPU routing.
    baseline_optimizer = AureliusOptimizer(config=cfg)
    baseline_result = baseline_optimizer.optimize(jobs, da, carbon, method=method)
    baseline_schedule = baseline_result.schedule

    # FIFO home placement for baseline region reference (no migration cost).
    asap_baseline = baseline_optimizer.create_baseline_schedule(jobs)
    baseline_regions = {d.job_id: d.region for d in asap_baseline}

    # 5. GPU routing: scorer enabled.
    gpu_optimizer = AureliusOptimizer(
        config=cfg,
        gpu_placement_scorer=scorer,
        region_gpu_types=rgt,
    )
    gpu_result = gpu_optimizer.optimize(jobs, da, carbon, method=method)
    gpu_schedule = gpu_result.schedule

    # 6. Compute routing quality metrics for latency_critical jobs.
    b_mean_penalty, b_pct_best = _compute_ttft_penalty(
        baseline_schedule, lc_jobs, rgt, scorer,
    )
    g_mean_penalty, g_pct_best = _compute_ttft_penalty(
        gpu_schedule, lc_jobs, rgt, scorer,
    )

    # 7. Compute goodput / infra dollar (all jobs + lc subset).
    from ..backtesting.evaluator import evaluate_schedule

    def _energy_cost(schedule, price_data):
        realized = evaluate_schedule(
            schedule, jobs, price_data, {r: {} for r in price_data},
            warn_on_missing=False,
        )
        return realized.total_energy_cost_usd

    b_energy = _energy_cost(baseline_schedule, rt)
    g_energy = _energy_cost(gpu_schedule, rt)

    _, _, b_gpd = _compute_goodput_per_dollar(
        baseline_schedule, jobs, rt, gpu_hour_usd, migration_network_usd,
        baseline_regions,
    )
    _, _, g_gpd = _compute_goodput_per_dollar(
        gpu_schedule, jobs, rt, gpu_hour_usd, migration_network_usd,
        baseline_regions,
    )

    lc_ids = {j.job_id for j in lc_jobs}
    _, _, b_lc_gpd = _compute_goodput_per_dollar(
        baseline_schedule, jobs, rt, gpu_hour_usd, migration_network_usd,
        baseline_regions, job_filter=lc_ids,
    )
    _, _, g_lc_gpd = _compute_goodput_per_dollar(
        gpu_schedule, jobs, rt, gpu_hour_usd, migration_network_usd,
        baseline_regions, job_filter=lc_ids,
    )

    return GpuRoutingReport(
        total_jobs=len(jobs),
        latency_critical_jobs=len(lc_jobs),
        latency_critical_pct=100.0 * len(lc_jobs) / len(jobs) if jobs else 0.0,
        baseline_pct_on_best_gpu=b_pct_best,
        gpu_routing_pct_on_best_gpu=g_pct_best,
        routing_improvement_pp=(g_pct_best - b_pct_best) * 100.0,
        baseline_mean_gpu_penalty=b_mean_penalty,
        gpu_routing_mean_gpu_penalty=g_mean_penalty,
        penalty_reduction=b_mean_penalty - g_mean_penalty,
        baseline_realized_energy_cost_usd=b_energy,
        gpu_routing_realized_energy_cost_usd=g_energy,
        realized_energy_cost_delta_usd=g_energy - b_energy,
        baseline_goodput_per_dollar=b_gpd,
        gpu_routing_goodput_per_dollar=g_gpd,
        goodput_per_dollar_delta=g_gpd - b_gpd,
        baseline_lc_goodput_per_dollar=b_lc_gpd,
        gpu_routing_lc_goodput_per_dollar=g_lc_gpd,
        lc_goodput_per_dollar_delta=g_lc_gpd - b_lc_gpd,
        best_gpu_type=BEST_GPU_TYPE,
        best_gpu_region=BEST_GPU_REGION,
        region_gpu_types=dict(rgt),
    )
