"""SRTF-under-contention backtest — the queue-contention evaluation of the
``predicted_output_tokens`` greedy sort key merged in run 2026-06-20-f.

Run 2026-06-20-f showed the SRTF sort key is **neutral** on the canonical
26-day energy trace.  The reason was diagnosed there: that trace has no queue
contention — each job independently finds a cheap hour over a 26-day window, so
the *order* in which jobs are placed never changes the outcome.  SRTF only pays
off when jobs **compete for a shared, capacity-limited resource** at the same
time (arXiv:2604.06970).

This module builds exactly that contended scenario and measures whether the
merged sort key produces a real improvement:

  * **Real workload sizes.**  Job runtimes are derived from the *real*
    Azure LLM 2024 ``GeneratedTokens`` distribution (public trace, heavy-tailed:
    p50≈90, p99≈479, max≈1346 output tokens).  Short-vs-long heterogeneity is
    what SRTF exploits.
  * **Real energy prices.**  Cost is scored on the real CAISO (us-west) hourly
    day-ahead price series used by the canonical backtest.
  * **A binding power cap.**  A single region with a power cap below the
    aggregate job demand forces temporal contention: only a few jobs run
    concurrently, so the *processing order* decides which jobs claim feasible
    slots before the common deadline.

Under a common deadline with a binding capacity cap, minimizing the number of
late jobs is the classic Moore–Hodgson result: **process shortest-first**.  The
merged sort key does exactly this when ``predicted_output_tokens`` is set, so we
expect SRTF to complete more jobs before deadline (higher goodput numerator)
and pack cheap hours more efficiently (lower-cost denominator) than FIFO.

Three variants are compared through the **identical** ``JobScheduler``:

  ``fifo``
      ``predicted_output_tokens=None`` → sort degrades to arrival/deadline order
      (the pre-run-f behaviour).
  ``srtf_perfect``
      ``predicted_output_tokens`` = real ``GeneratedTokens`` (clairvoyant prior —
      the achievable ceiling).
  ``srtf_forecast``
      ``predicted_output_tokens`` = real tokens perturbed by lognormal forecast
      noise (a realistic ``OutputLengthForecastBundle``-quality prior).  This is
      the honest, deployable number — it is NEVER the actual value used as a
      scheduling label leak; the actual length is used only to compute runtime
      physics, identically across all three variants.

Honesty / non-goals (mirrors ``docs/RESULTS.md`` §8):
- Simulator / public-trace directional result — **not** a production savings claim.
- Azure LLM 2024 is a public serving trace, not customer telemetry.
- Output-token → runtime mapping is an explicit documented proxy applied
  identically to every variant; wins come only from the **sort order**.
"""

from __future__ import annotations

import dataclasses
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..models import Job, OptimizationConfig, ScheduleDecision
from .canonical_backtests import CANONICAL_DA_FILES, REGION_CAISO, _load_price_csv

# ---------------------------------------------------------------------------
# Proxy constants (documented; identical across every variant)
# ---------------------------------------------------------------------------

# Real Azure LLM 2024 output tokens → batch-inference runtime hours.  Divisor
# chosen so the real heavy-tailed token distribution spans a meaningful range of
# integer hours (p50≈90 tok → 1h, p99≈479 tok → 4h, max≈1346 tok → ~11h),
# giving the short-vs-long heterogeneity SRTF exploits.  Applied identically to
# all variants — only the SORT ORDER differs.
TOKENS_PER_RUNTIME_HOUR: float = 120.0

# Per-job GPU board power (kW) and the binding regional cap.  With 40 kW/job and
# a 120 kW cap only THREE jobs run concurrently, so processing order strongly
# determines which jobs claim feasible slots before the deadline.
JOB_POWER_KW: float = 40.0
REGION_POWER_CAP_KW: float = 120.0

# GPU compute cost ($/GPU-hour) for the infra-dollar denominator (matches
# srtf_backtest.py).
GPU_HOUR_USD: float = 2.0

# Default Azure LLM 2024 public-trace sample (real per-request output tokens).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AZURE_FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv"
)

# The contended scenario is anchored inside the canonical CAISO price window so
# real day-ahead prices cover every scheduling hour.
_CONTENTION_WINDOW_START = datetime(2026, 2, 2, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Real Azure LLM 2024 output-token loading
# ---------------------------------------------------------------------------

def load_azure_output_tokens(
    path: str = DEFAULT_AZURE_FIXTURE,
    limit: Optional[int] = None,
) -> list[int]:
    """Return real Azure LLM 2024 ``GeneratedTokens`` (output length per request).

    Uses the project's own Azure loader so the schema guard and the
    ``GeneratedTokens == 0`` failure convention are honoured.  Failures are
    excluded (they carry no output-length signal).
    """
    from ..traces.azure_llm import load_csv

    requests = load_csv(path, include_failures=False)
    tokens = [r.output_tokens for r in requests if r.output_tokens > 0]
    if limit is not None:
        tokens = tokens[:limit]
    return tokens


def _runtime_hours_from_tokens(output_tokens: int) -> int:
    """Map real output-token count → integer batch runtime hours (min 1h)."""
    return max(1, int(round(output_tokens / TOKENS_PER_RUNTIME_HOUR)))


# ---------------------------------------------------------------------------
# Contended job construction
# ---------------------------------------------------------------------------

def build_contended_jobs(
    output_tokens: list[int],
    horizon_hours: int,
    region: str = REGION_CAISO,
    window_start: datetime = _CONTENTION_WINDOW_START,
) -> list[Job]:
    """Build a contended batch-inference workload from real output-token sizes.

    Every job shares the same ``earliest_start`` and the same ``deadline``
    (``window_start + horizon_hours``), so they all compete for the same
    capacity-limited hours — the queue-contention condition SRTF needs.  Jobs
    are returned in **arrival order** (the order they appear in the trace) so the
    FIFO baseline reflects realistic submission order.

    ``predicted_output_tokens`` is left as ``None`` here; the variant builders
    set it.  The job's ``runtime_hours`` (the physics) is derived from the real
    output-token count identically for every variant.
    """
    deadline = window_start + timedelta(hours=horizon_hours)
    jobs: list[Job] = []
    for i, tok in enumerate(output_tokens):
        runtime = _runtime_hours_from_tokens(tok)
        jobs.append(Job(
            job_id=f"azure-{i:05d}",
            submit_time=window_start,
            runtime_hours=float(runtime),
            deadline=deadline,
            power_kw=JOB_POWER_KW,
            earliest_start=window_start,
            region_options=[region],
            workload_type="llm_batch_inference",
            sla_class="best_effort",
            gpu_count=1,
            # Stash the true output length so variant builders / metrics can read
            # it without re-deriving; not used by the scheduler directly.
            data_transfer_gb=float(tok),
        ))
    return jobs


def with_srtf_prior(
    jobs: list[Job],
    noise_cv: float = 0.0,
    seed: int = 20260201,
) -> list[Job]:
    """Return a copy of ``jobs`` with ``predicted_output_tokens`` populated.

    The prior is the job's true output-token count (stashed in
    ``data_transfer_gb``) optionally perturbed by lognormal multiplicative noise
    with coefficient of variation ``noise_cv`` (0.0 → clairvoyant ceiling).  The
    perturbed value models a real ``OutputLengthForecastBundle`` prediction; the
    true length is never used as the sort key when ``noise_cv > 0``.
    """
    rng = random.Random(seed)
    sigma = math.sqrt(math.log(1.0 + noise_cv * noise_cv)) if noise_cv > 0 else 0.0
    out: list[Job] = []
    for j in jobs:
        true_tokens = j.data_transfer_gb
        if sigma > 0:
            # lognormal with unit median multiplier, mean-preserving-ish
            factor = math.exp(rng.gauss(0.0, sigma))
            prior = max(1.0, true_tokens * factor)
        else:
            prior = true_tokens
        out.append(dataclasses.replace(j, predicted_output_tokens=prior))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _completion_time(decision: ScheduleDecision) -> datetime:
    return decision.all_segments[-1].end_time


def compute_contention_metrics(
    schedule: list[ScheduleDecision],
    jobs: list[Job],
    rt_prices: dict,
    region: str = REGION_CAISO,
    gpu_hour_usd: float = GPU_HOUR_USD,
) -> dict:
    """Return SLA-safe goodput/$ and supporting KPIs for one schedule.

    Goodput numerator counts GPU-hours of jobs that complete **on or before**
    their deadline (SLA-safe).  Denominator is realized energy cost (real CAISO
    prices) + GPU compute cost.  Also returns deadline misses and flow-time
    statistics that explain *why* the goodput moves.
    """
    from ..backtesting.evaluator import evaluate_schedule

    realized = evaluate_schedule(
        schedule, jobs, {region: rt_prices}, {region: {}}, warn_on_missing=False,
    )
    energy_cost = realized.total_energy_cost_usd

    job_by_id = {j.job_id: j for j in jobs}
    gpu_infra = 0.0
    deadline_misses = 0
    sla_safe_goodput = 0.0
    flow_hours: list[float] = []
    late_short = 0   # late jobs whose true size is below median (the ones SRTF protects)
    sizes = sorted(j.runtime_hours for j in jobs)
    median_runtime = sizes[len(sizes) // 2] if sizes else 0.0

    for d in schedule:
        job = job_by_id.get(d.job_id)
        if job is None:
            continue
        gpu_infra += gpu_hour_usd * max(1, job.gpu_count) * job.runtime_hours
        completion = _completion_time(d)
        flow = (completion - job.earliest_start).total_seconds() / 3600.0
        flow_hours.append(flow)
        if completion > job.deadline:
            deadline_misses += 1
            if job.runtime_hours <= median_runtime:
                late_short += 1
        else:
            sla_safe_goodput += max(1, job.gpu_count) * job.runtime_hours

    total_infra = energy_cost + gpu_infra
    gp_per_dollar = sla_safe_goodput / total_infra if total_infra > 0 else 0.0
    flow_hours.sort()
    n = len(flow_hours)
    mean_flow = sum(flow_hours) / n if n else 0.0
    p90_flow = flow_hours[min(n - 1, int(0.9 * n))] if n else 0.0

    return {
        "goodput_per_dollar": gp_per_dollar,
        "sla_safe_goodput": sla_safe_goodput,
        "energy_cost_usd": energy_cost,
        "gpu_infra_usd": gpu_infra,
        "total_infra_usd": total_infra,
        "deadline_misses": deadline_misses,
        "late_short_jobs": late_short,
        "mean_flow_hours": mean_flow,
        "p90_flow_hours": p90_flow,
    }


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class SRTFContentionReport:
    """Baseline (FIFO) vs SRTF goodput/$ under a binding power cap."""

    total_jobs: int
    horizon_hours: int
    power_cap_kw: float
    concurrent_slots: int
    total_job_hours: float
    capacity_job_hours: float
    contention_ratio: float          # demand / capacity (>1 means binding)

    fifo: dict
    srtf_perfect: dict
    srtf_forecast: dict

    # Headline deltas (srtf_perfect vs fifo)
    goodput_delta_pct: float
    miss_reduction: int
    forecast_goodput_delta_pct: float

    shadow_tag: str = "shadow_only_simulator_result_not_production_savings"

    def to_dict(self) -> dict:
        def _round(d: dict) -> dict:
            return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
        return {
            "total_jobs": self.total_jobs,
            "horizon_hours": self.horizon_hours,
            "power_cap_kw": self.power_cap_kw,
            "concurrent_slots": self.concurrent_slots,
            "total_job_hours": round(self.total_job_hours, 2),
            "capacity_job_hours": round(self.capacity_job_hours, 2),
            "contention_ratio": round(self.contention_ratio, 3),
            "fifo": _round(self.fifo),
            "srtf_perfect": _round(self.srtf_perfect),
            "srtf_forecast": _round(self.srtf_forecast),
            "goodput_delta_pct": round(self.goodput_delta_pct, 4),
            "miss_reduction": self.miss_reduction,
            "forecast_goodput_delta_pct": round(self.forecast_goodput_delta_pct, 4),
            "shadow_tag": self.shadow_tag,
        }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_srtf_contention_backtest(
    horizon_hours: int = 24,
    job_limit: int = 200,
    forecast_noise_cv: float = 0.30,
    azure_fixture: str = DEFAULT_AZURE_FIXTURE,
    region: str = REGION_CAISO,
    seed: int = 20260201,
) -> SRTFContentionReport:
    """Evaluate the merged SRTF sort key under queue contention.

    Loads real Azure LLM 2024 output tokens, builds a capacity-contended
    batch-inference workload priced on real CAISO day-ahead energy, then runs
    FIFO vs SRTF (perfect prior) vs SRTF (noisy forecast prior) through the same
    ``JobScheduler`` and scores SLA-safe goodput/$.

    Args:
        horizon_hours: Common deadline window. Tighter → more deadline pressure.
        job_limit: Number of real Azure requests to admit (in arrival order).
        forecast_noise_cv: Lognormal CV of the realistic forecast prior.
        azure_fixture: Path to the real Azure LLM 2024 CSV.
        region: Canonical region whose real prices form the cost basis.
        seed: Forecast-noise seed (physics are deterministic).

    Returns:
        ``SRTFContentionReport`` with per-variant KPIs and headline deltas.

    Raises:
        FileNotFoundError: If the canonical CAISO price CSV is absent.
    """
    # Phase 3: route through the canonical AureliusOptimizer (energy policy).
    from ..optimizer import AureliusOptimizer

    # 1. Real Azure LLM 2024 output tokens → contended jobs (FIFO/no prior).
    tokens = load_azure_output_tokens(azure_fixture, limit=job_limit)
    base_jobs = build_contended_jobs(tokens, horizon_hours=horizon_hours, region=region)

    # 2. SRTF priors.
    srtf_perfect_jobs = with_srtf_prior(base_jobs, noise_cv=0.0)
    srtf_forecast_jobs = with_srtf_prior(base_jobs, noise_cv=forecast_noise_cv, seed=seed)

    # 3. Real CAISO day-ahead prices (cost basis for every variant).
    da_prices = _load_price_csv(CANONICAL_DA_FILES[region])
    price_map = {region: da_prices}
    carbon = {region: {}}

    # 4. Binding power cap on the single contended region.
    cfg = OptimizationConfig(
        default_region=region,
        min_power_fraction=1.0,
        region_power_caps={region: REGION_POWER_CAP_KW},
    )

    def _run(js_jobs: list[Job]) -> list[ScheduleDecision]:
        optimizer = AureliusOptimizer(config=cfg)
        return optimizer.optimize(js_jobs, price_map, carbon, method="greedy").schedule

    fifo_sched = _run(base_jobs)
    perfect_sched = _run(srtf_perfect_jobs)
    forecast_sched = _run(srtf_forecast_jobs)

    fifo_m = compute_contention_metrics(fifo_sched, base_jobs, da_prices, region)
    perfect_m = compute_contention_metrics(perfect_sched, srtf_perfect_jobs, da_prices, region)
    forecast_m = compute_contention_metrics(forecast_sched, srtf_forecast_jobs, da_prices, region)

    # 5. Contention diagnostics.
    concurrent = int(REGION_POWER_CAP_KW // JOB_POWER_KW)
    total_job_hours = sum(j.runtime_hours for j in base_jobs)
    capacity_job_hours = float(concurrent * horizon_hours)
    contention_ratio = (
        total_job_hours / capacity_job_hours if capacity_job_hours > 0 else float("inf")
    )

    base_gp = fifo_m["goodput_per_dollar"]
    gp_delta_pct = (
        (perfect_m["goodput_per_dollar"] - base_gp) / base_gp * 100.0 if base_gp > 0 else 0.0
    )
    fc_delta_pct = (
        (forecast_m["goodput_per_dollar"] - base_gp) / base_gp * 100.0 if base_gp > 0 else 0.0
    )

    return SRTFContentionReport(
        total_jobs=len(base_jobs),
        horizon_hours=horizon_hours,
        power_cap_kw=REGION_POWER_CAP_KW,
        concurrent_slots=concurrent,
        total_job_hours=total_job_hours,
        capacity_job_hours=capacity_job_hours,
        contention_ratio=contention_ratio,
        fifo=fifo_m,
        srtf_perfect=perfect_m,
        srtf_forecast=forecast_m,
        goodput_delta_pct=gp_delta_pct,
        miss_reduction=fifo_m["deadline_misses"] - perfect_m["deadline_misses"],
        forecast_goodput_delta_pct=fc_delta_pct,
    )
