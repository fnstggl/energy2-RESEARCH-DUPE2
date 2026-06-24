"""Phase 1 parity guard — the canonical ``AureliusOptimizer`` wrapper produces
byte-identical results to the existing productized ``JobScheduler``.

This pins the Phase 1 contract of the canonical-optimizer unification
(``research/OPTIMIZER_UNIFICATION_PLAN.md``):

  * the wrapper *delegates* to ``JobScheduler`` (no reimplementation),
  * no runtime behavior changes (schedule + objective identical across methods),
  * it reproduces the *pinned* energy-core snapshot
    (same fixture/constants as ``tests/test_energy_core_preservation.py``),
  * it reproduces the scheduler step that drives the canonical energy benchmark
    (=> 0% KPI drift, since the benchmark KPI is a deterministic function of the
    schedule), and
  * NO serving/SRTF, placement, admission, or replica-scaling code is wired in
    (those policies raise ``NotImplementedError``).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from aurelius.backtesting.evaluator import evaluate_schedule
from aurelius.models import Job, OptimizationConfig
from aurelius.optimization.scheduler import JobScheduler
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.policies import (
    POLICY_REGISTRY,
    EnergySchedulingPolicy,
)

# Same frozen reference as tests/test_energy_core_preservation.py so the wrapper
# is tied to the pinned energy core, not just to a live JobScheduler.
W = datetime(2026, 2, 1, tzinfo=timezone.utc)
STANDALONE_COST = 153.0
STANDALONE_REGIONS_HASH = (
    "6a5a7078d315b2715ee45499469662a2473369814352d879dd9b023ae4ad12e0"
)


def _price_curve(base: float) -> dict[datetime, float]:
    return {
        W + timedelta(hours=h): base + 30.0 * (8 <= (h % 24) < 20)
        for h in range(0, 72)
    }


def _fixture():
    da = {
        "us-west": _price_curve(30.0),
        "us-east": _price_curve(60.0),
        "us-south": _price_curve(45.0),
    }
    carbon = {r: {} for r in da}
    jobs = []
    for i in range(12):
        rt_h = [2, 4, 6][i % 3]
        slack = [6, 12, 24][i % 3]
        es = W + timedelta(hours=i)
        jobs.append(Job(
            job_id=f"job-{i:03d}", submit_time=es, runtime_hours=rt_h,
            deadline=es + timedelta(hours=rt_h + slack), power_kw=100.0,
            earliest_start=es, region_options=["us-west", "us-east", "us-south"],
            gpu_count=2, workload_type="llm_batch_inference", migration_cost_hours=0.1,
        ))
    return jobs, da, carbon


def _regions_hash(schedule) -> str:
    payload = [
        (d.job_id, d.region, d.start_time.isoformat(), round(d.power_fraction, 3))
        for d in schedule
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _objective_tuple(obj):
    """Stable comparable view of the objective components."""
    return (
        round(obj.total, 9),
        round(getattr(obj, "energy_cost", 0.0), 9),
        round(getattr(obj, "carbon_cost", 0.0), 9),
        round(getattr(obj, "risk_penalty", 0.0), 9),
    )


# --------------------------------------------------------------------------
# 1. The wrapper delegates and is behavior-identical across every solve method
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method",
    ["greedy", "local_search", "greedy_migrate", "greedy_migrate_dp", "milp"],
)
def test_wrapper_matches_jobscheduler_across_methods(method):
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)

    direct = JobScheduler(cfg).solve(jobs, da, carbon, method=method)
    wrapped = AureliusOptimizer(cfg).optimize(jobs, da, carbon, method=method)

    assert _regions_hash(wrapped.schedule) == _regions_hash(direct.schedule), (
        f"wrapper changed placement decisions for method={method}"
    )
    assert _objective_tuple(wrapped.objective) == _objective_tuple(direct.objective), (
        f"wrapper changed objective for method={method}"
    )
    assert wrapped.violations == direct.violations
    assert len(wrapped.schedule) == len(direct.schedule)


# --------------------------------------------------------------------------
# 2. The wrapper reproduces the PINNED energy-core snapshot
# --------------------------------------------------------------------------

def test_wrapper_reproduces_pinned_energy_core():
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)

    res = AureliusOptimizer(cfg).optimize(jobs, da, carbon, method="greedy")

    cost = round(
        evaluate_schedule(res.schedule, jobs, da, carbon).total_energy_cost_usd, 6
    )
    assert cost == STANDALONE_COST, (
        f"canonical wrapper realized cost drifted: {cost} != {STANDALONE_COST}"
    )
    assert _regions_hash(res.schedule) == STANDALONE_REGIONS_HASH, (
        "canonical wrapper placement decisions drifted from the pinned core"
    )


# --------------------------------------------------------------------------
# 3. Delegation plumbing: injected scheduler, baseline passthrough, scheduler prop
# --------------------------------------------------------------------------

def test_wrapper_uses_injected_scheduler_identity():
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    js = JobScheduler(cfg)
    opt = AureliusOptimizer(scheduler=js)
    assert opt.scheduler is js
    assert isinstance(opt.policy, EnergySchedulingPolicy)
    assert opt.policy_name == "energy"


def test_baseline_schedule_delegates():
    jobs, da, carbon = _fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    js = JobScheduler(cfg)
    direct = {d.job_id: d.region for d in js.create_baseline_schedule(jobs)}
    wrapped = {
        d.job_id: d.region
        for d in AureliusOptimizer(cfg).create_baseline_schedule(jobs)
    }
    assert wrapped == direct


def test_injected_scheduler_rejects_extra_args():
    cfg = OptimizationConfig()
    with pytest.raises(ValueError):
        AureliusOptimizer(config=cfg, scheduler=JobScheduler(cfg))


# --------------------------------------------------------------------------
# 4. 0% drift on the canonical energy benchmark's scheduler step
#    (the benchmark KPI is a deterministic function of this schedule)
# --------------------------------------------------------------------------

def test_wrapper_matches_canonical_benchmark_scheduler_step():
    from aurelius.benchmarks.canonical_backtests import (
        CANONICAL_METHOD,
        CANONICAL_SEED,
        REGION_PJM,
        build_canonical_jobs,
        load_canonical_price_data,
    )

    jobs = build_canonical_jobs(CANONICAL_SEED, 1000)
    da, _rt = load_canonical_price_data()
    carbon = {r: {} for r in da}
    cfg = OptimizationConfig(default_region=REGION_PJM, min_power_fraction=1.0)

    direct = JobScheduler(cfg).solve(jobs, da, carbon, method=CANONICAL_METHOD)
    wrapped = AureliusOptimizer(cfg).optimize(jobs, da, carbon, method=CANONICAL_METHOD)

    assert _regions_hash(wrapped.schedule) == _regions_hash(direct.schedule)
    assert _objective_tuple(wrapped.objective) == _objective_tuple(direct.objective)

    # Baseline (FIFO/ASAP) step the benchmark also consumes.
    base_direct = {d.job_id: d.region for d in JobScheduler(cfg).create_baseline_schedule(jobs)}
    base_wrap = {d.job_id: d.region for d in AureliusOptimizer(cfg).create_baseline_schedule(jobs)}
    assert base_wrap == base_direct


# --------------------------------------------------------------------------
# 5. Serving/SRTF + the other policies are NOT wired in (Phase 1 guard)
# --------------------------------------------------------------------------

# serving_queue and replica_scaling are implemented as of Phase 2; the
# remaining seams still raise.
@pytest.mark.parametrize(
    "policy", ["placement", "admission"]
)
def test_unimplemented_policies_raise_on_use(policy):
    assert policy in POLICY_REGISTRY  # the seam exists...
    opt = AureliusOptimizer(policy=policy)
    with pytest.raises(NotImplementedError):
        opt.optimize([], {}, {})  # ...but using it fails loudly


def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        AureliusOptimizer(policy="does_not_exist")


def test_default_policy_is_energy():
    assert AureliusOptimizer().policy_name == "energy"
