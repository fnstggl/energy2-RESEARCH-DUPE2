"""Tests for the GPU routing benchmark (aurelius/benchmarks/gpu_routing_backtest.py).

Structure
---------
TestSyntheticPriorBuilder (8 tests)
    Verifies the synthetic TTFTShadowPrior is built correctly from CARA-calibrated
    TTFT values and that its by_gpu / by_gpu_counts fields satisfy the scorer's
    min_subgroup_rows threshold.

TestJobAugmentation (6 tests)
    Verifies augment_jobs_with_sla_class() correctly maps workload_type to sla_class
    without mutating the original jobs.

TestGpuRoutingReportStructure (4 tests)
    Verifies the GpuRoutingReport dataclass and to_dict() serialisation.

TestTtftPenaltyComputation (6 tests)
    Verifies _compute_ttft_penalty() returns correct mean_penalty / pct_on_best_gpu
    for known schedules.

TestGpuRoutingBacktestIntegration (6 tests)
    End-to-end: runs run_gpu_routing_backtest() on a small synthetic job set
    (requires no data files; uses a patched load_canonical_price_data).
    Verifies GPU routing routes more latency_critical jobs to H100.

TestGpuRoutingRegression (4 tests)
    Invariant checks that must hold regardless of seed/price data.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from aurelius.benchmarks.gpu_routing_backtest import (
    BEST_GPU_REGION,
    BEST_GPU_TYPE,
    CANONICAL_REGION_GPU_TYPES,
    SYNTHETIC_GPU_TTFT_P50_S,
    GpuRoutingReport,
    _compute_ttft_penalty,
    augment_jobs_with_sla_class,
    build_synthetic_prior,
)
from aurelius.models import Job, WORKLOAD_DEFAULT_SLA_CLASS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc

def _make_job(
    job_id: str = "j-0",
    workload_type: str = "realtime_inference",
    sla_class: str = "best_effort",
    region_options: list | None = None,
    runtime_hours: float = 1.0,
    gpu_count: int = 1,
) -> Job:
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=_UTC)
    return Job(
        job_id=job_id,
        submit_time=t0,
        runtime_hours=runtime_hours,
        deadline=t0 + timedelta(hours=runtime_hours + 4),
        power_kw=50.0,
        earliest_start=t0,
        region_options=region_options or ["us-west", "us-east", "us-south"],
        workload_type=workload_type,
        sla_class=sla_class,
        gpu_count=gpu_count,
    )


def _make_schedule(job_id: str, region: str, runtime_hours: float = 1.0) -> "ScheduleDecision":
    from aurelius.models import ScheduleDecision
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=_UTC)
    return ScheduleDecision(
        job_id=job_id,
        start_time=t0,
        region=region,
        power_fraction=1.0,
        actual_runtime_hours=runtime_hours,
    )


# ---------------------------------------------------------------------------
# TestSyntheticPriorBuilder
# ---------------------------------------------------------------------------

class TestSyntheticPriorBuilder:

    def test_by_gpu_has_all_three_gpu_types(self):
        prior = build_synthetic_prior()
        assert "h100" in prior.by_gpu
        assert "a100" in prior.by_gpu
        assert "t4" in prior.by_gpu

    def test_by_gpu_counts_meets_min_subgroup_rows(self):
        prior = build_synthetic_prior(rows_per_gpu=200)
        for gtype in ("h100", "a100", "t4"):
            assert prior.by_gpu_counts.get(gtype, 0) >= 50, (
                f"by_gpu_counts[{gtype}]={prior.by_gpu_counts.get(gtype)} < 50"
            )

    def test_h100_fastest_a100_middle_t4_slowest(self):
        prior = build_synthetic_prior()
        h = prior.by_gpu["h100"]
        a = prior.by_gpu["a100"]
        t = prior.by_gpu["t4"]
        assert h < a < t, f"Expected h100<a100<t4 but got h100={h}, a100={a}, t4={t}"

    def test_h100_approximately_012s(self):
        prior = build_synthetic_prior()
        h = prior.by_gpu["h100"]
        assert 0.05 < h < 0.25, f"H100 p50 {h:.3f} s outside expected range"

    def test_t4_approximately_095s(self):
        prior = build_synthetic_prior()
        t = prior.by_gpu["t4"]
        assert 0.50 < t < 2.0, f"T4 p50 {t:.3f} s outside expected range"

    def test_fit_row_count_is_rows_per_gpu_times_three(self):
        prior = build_synthetic_prior(rows_per_gpu=100)
        assert prior.fit_row_count == 300

    def test_predict_model_size_none_returns_by_gpu(self):
        prior = build_synthetic_prior()
        h100_pred = prior.predict(model_size=None, gpu_type="h100", prompt_tokens=None)
        assert h100_pred is not None
        assert not math.isnan(h100_pred)
        assert abs(h100_pred - prior.by_gpu["h100"]) < 1e-9

    def test_subgroup_n_model_size_none_uses_by_gpu_counts(self):
        prior = build_synthetic_prior(rows_per_gpu=150)
        n = prior.subgroup_n(model_size=None, gpu_type="h100", prompt_tokens=None)
        assert n == 150


# ---------------------------------------------------------------------------
# TestJobAugmentation
# ---------------------------------------------------------------------------

class TestJobAugmentation:

    def test_realtime_inference_gets_latency_critical(self):
        jobs = [_make_job(workload_type="realtime_inference")]
        aug = augment_jobs_with_sla_class(jobs)
        assert aug[0].sla_class == "latency_critical"

    def test_training_gets_best_effort(self):
        jobs = [_make_job(workload_type="training")]
        aug = augment_jobs_with_sla_class(jobs)
        assert aug[0].sla_class == "best_effort"

    def test_llm_batch_inference_gets_deadline(self):
        jobs = [_make_job(workload_type="llm_batch_inference")]
        aug = augment_jobs_with_sla_class(jobs)
        assert aug[0].sla_class == "deadline"

    def test_original_jobs_not_mutated(self):
        jobs = [_make_job(workload_type="realtime_inference", sla_class="best_effort")]
        _ = augment_jobs_with_sla_class(jobs)
        assert jobs[0].sla_class == "best_effort"

    def test_all_canonical_workload_types_mapped(self):
        wt_jobs = [
            _make_job(job_id=f"j-{i}", workload_type=wt)
            for i, wt in enumerate(WORKLOAD_DEFAULT_SLA_CLASS)
        ]
        aug = augment_jobs_with_sla_class(wt_jobs)
        for j, expected_wt in zip(aug, WORKLOAD_DEFAULT_SLA_CLASS):
            expected = WORKLOAD_DEFAULT_SLA_CLASS[expected_wt]
            assert j.sla_class == expected, (
                f"{expected_wt}: expected {expected}, got {j.sla_class}"
            )

    def test_empty_list_returns_empty(self):
        assert augment_jobs_with_sla_class([]) == []


# ---------------------------------------------------------------------------
# TestGpuRoutingReportStructure
# ---------------------------------------------------------------------------

class TestGpuRoutingReportStructure:

    def _make_report(self, **overrides) -> GpuRoutingReport:
        defaults = dict(
            total_jobs=1000, latency_critical_jobs=150, latency_critical_pct=15.0,
            baseline_pct_on_best_gpu=0.33, gpu_routing_pct_on_best_gpu=0.75,
            routing_improvement_pp=42.0,
            baseline_mean_gpu_penalty=0.18, gpu_routing_mean_gpu_penalty=0.07,
            penalty_reduction=0.11,
            baseline_realized_energy_cost_usd=12000.0,
            gpu_routing_realized_energy_cost_usd=12300.0,
            realized_energy_cost_delta_usd=300.0,
            baseline_goodput_per_dollar=0.00123, gpu_routing_goodput_per_dollar=0.00122,
            goodput_per_dollar_delta=-0.00001,
            baseline_lc_goodput_per_dollar=0.00150, gpu_routing_lc_goodput_per_dollar=0.00160,
            lc_goodput_per_dollar_delta=0.00010,
            region_gpu_types=dict(CANONICAL_REGION_GPU_TYPES),
        )
        defaults.update(overrides)
        return GpuRoutingReport(**defaults)

    def test_to_dict_contains_required_keys(self):
        report = self._make_report()
        d = report.to_dict()
        required = [
            "total_jobs", "latency_critical_jobs", "routing_improvement_pp",
            "baseline_pct_on_best_gpu", "gpu_routing_pct_on_best_gpu",
            "baseline_mean_gpu_penalty", "gpu_routing_mean_gpu_penalty",
            "penalty_reduction", "lc_goodput_per_dollar_delta",
            "shadow_tag", "region_gpu_types",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"

    def test_shadow_tag_present(self):
        report = self._make_report()
        d = report.to_dict()
        assert "shadow" in d["shadow_tag"].lower()

    def test_best_gpu_fields_match_constants(self):
        report = self._make_report()
        assert report.best_gpu_type == BEST_GPU_TYPE
        assert report.best_gpu_region == BEST_GPU_REGION

    def test_routing_improvement_pp_computed_correctly(self):
        report = self._make_report(
            baseline_pct_on_best_gpu=0.33,
            gpu_routing_pct_on_best_gpu=0.75,
        )
        # Manually: (0.75 - 0.33) * 100 = 42
        assert abs(report.routing_improvement_pp - 42.0) < 1e-6


# ---------------------------------------------------------------------------
# TestTtftPenaltyComputation
# ---------------------------------------------------------------------------

class TestTtftPenaltyComputation:

    def _make_scorer(self, enabled=True):
        from aurelius.forecasting.gpu_placement_scorer import (
            GpuPlacementConfig,
            GpuPlacementScorer,
        )
        prior = build_synthetic_prior()
        return GpuPlacementScorer(
            prior=prior,
            config=GpuPlacementConfig(
                enabled=enabled, min_subgroup_rows=50,
                penalty_floor=0.05, penalty_ceil=0.50,
            ),
        )

    def test_all_on_h100_gives_zero_or_floor_penalty(self):
        jobs = [
            _make_job(job_id=f"j-{i}", workload_type="realtime_inference",
                      sla_class="latency_critical")
            for i in range(3)
        ]
        schedule = [_make_schedule(j.job_id, "us-east") for j in jobs]
        scorer = self._make_scorer()
        mean_pen, pct = _compute_ttft_penalty(schedule, jobs, CANONICAL_REGION_GPU_TYPES, scorer)
        # All on best GPU (H100=rank 0) → penalty = floor = 0.05
        assert pct == pytest.approx(1.0)
        assert mean_pen == pytest.approx(0.05, abs=0.01)

    def test_all_on_t4_gives_ceil_penalty(self):
        jobs = [
            _make_job(job_id=f"j-{i}", workload_type="realtime_inference",
                      sla_class="latency_critical")
            for i in range(3)
        ]
        schedule = [_make_schedule(j.job_id, "us-south") for j in jobs]  # T4 region
        scorer = self._make_scorer()
        mean_pen, pct = _compute_ttft_penalty(schedule, jobs, CANONICAL_REGION_GPU_TYPES, scorer)
        # All on worst GPU (T4=rank 1) → penalty = ceil = 0.50
        assert pct == pytest.approx(0.0)
        assert mean_pen == pytest.approx(0.50, abs=0.01)

    def test_empty_lc_jobs_returns_zero(self):
        scorer = self._make_scorer()
        mean_pen, pct = _compute_ttft_penalty([], [], CANONICAL_REGION_GPU_TYPES, scorer)
        assert mean_pen == 0.0
        assert pct == 0.0

    def test_mixed_placement_penalty_is_between_floor_and_ceil(self):
        jobs = [
            _make_job(job_id="j-0", workload_type="realtime_inference",
                      sla_class="latency_critical"),
            _make_job(job_id="j-1", workload_type="realtime_inference",
                      sla_class="latency_critical"),
        ]
        # One on H100 (best), one on T4 (worst)
        schedule = [
            _make_schedule("j-0", "us-east"),   # H100
            _make_schedule("j-1", "us-south"),  # T4
        ]
        scorer = self._make_scorer()
        mean_pen, pct = _compute_ttft_penalty(schedule, jobs, CANONICAL_REGION_GPU_TYPES, scorer)
        assert pct == pytest.approx(0.5)  # 1 of 2 on best
        assert 0.05 < mean_pen < 0.50     # mix of floor and ceil

    def test_non_latency_critical_jobs_not_scored(self):
        jobs = [_make_job(job_id="j-0", workload_type="training", sla_class="best_effort")]
        schedule = [_make_schedule("j-0", "us-south")]
        scorer = self._make_scorer()
        # best_effort jobs should get sla_neutral → penalty = 0.0
        mean_pen, pct = _compute_ttft_penalty(schedule, jobs, CANONICAL_REGION_GPU_TYPES, scorer)
        # Since all jobs are be, penalties list is [0.0], mean = 0.0
        assert mean_pen == pytest.approx(0.0, abs=0.01)

    def test_disabled_scorer_gives_zero_penalty(self):
        jobs = [
            _make_job(job_id="j-0", workload_type="realtime_inference",
                      sla_class="latency_critical"),
        ]
        schedule = [_make_schedule("j-0", "us-south")]  # T4 — worst
        scorer = self._make_scorer(enabled=False)
        mean_pen, pct = _compute_ttft_penalty(schedule, jobs, CANONICAL_REGION_GPU_TYPES, scorer)
        # Disabled scorer returns 0.0 for all
        assert mean_pen == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestGpuRoutingBacktestIntegration
# ---------------------------------------------------------------------------

class TestGpuRoutingBacktestIntegration:
    """Run the benchmark on a tiny synthetic dataset (no real price files needed)."""

    def _build_small_jobs(self, n: int = 30, seed: int = 99) -> list[Job]:
        import random
        rng = random.Random(seed)
        t0 = datetime(2026, 2, 1, 0, 0, tzinfo=_UTC)
        workload_types = [
            "realtime_inference", "realtime_inference",
            "llm_batch_inference", "training", "scheduled_batch",
        ]
        jobs = []
        for i in range(n):
            wt = rng.choice(workload_types)
            runtime = rng.choice([1, 2])
            slack = rng.choice([0, 4, 8]) if wt != "realtime_inference" else 0
            submit = t0 + timedelta(hours=i * 0.5)
            jobs.append(Job(
                job_id=f"j-{i:03d}",
                submit_time=submit,
                runtime_hours=float(runtime),
                deadline=submit + timedelta(hours=runtime + slack + 1),
                power_kw=50.0,
                earliest_start=submit,
                region_options=["us-west", "us-east", "us-south"],
                workload_type=wt,
                gpu_count=rng.choice([1, 2]),
                migration_cost_hours=0.1 if wt != "realtime_inference" else None,
            ))
        return augment_jobs_with_sla_class(jobs)

    def _build_flat_price_data(self, regions, value=30.0):
        t0 = datetime(2026, 2, 1, 0, 0, tzinfo=_UTC)
        return {
            r: {t0 + timedelta(hours=h): value for h in range(24 * 30)}
            for r in regions
        }

    def _run_with_synthetic_data(self, price_value=30.0):
        """Run run_gpu_routing_backtest patching the canonical data loaders."""
        from aurelius.benchmarks.gpu_routing_backtest import run_gpu_routing_backtest

        regions = ["us-west", "us-east", "us-south"]
        flat_da = self._build_flat_price_data(regions, value=price_value)
        flat_rt = self._build_flat_price_data(regions, value=price_value)
        jobs = self._build_small_jobs(n=30)

        with (
            patch(
                "aurelius.benchmarks.gpu_routing_backtest.load_canonical_price_data",
                return_value=(flat_da, flat_rt),
            ),
            patch(
                "aurelius.benchmarks.gpu_routing_backtest.build_canonical_jobs",
                return_value=[
                    # strip augmentation for raw_jobs (augment_jobs_with_sla_class is called inside)
                    j for j in jobs
                ],
            ),
        ):
            return run_gpu_routing_backtest(
                seed=99, job_count=30, method="greedy", prior_rows_per_gpu=100
            )

    def test_report_has_correct_total_jobs(self):
        report = self._run_with_synthetic_data()
        assert report.total_jobs == 30

    def test_latency_critical_jobs_are_nonzero(self):
        report = self._run_with_synthetic_data()
        assert report.latency_critical_jobs > 0

    def test_gpu_routing_routes_more_lc_jobs_to_h100(self):
        """Key invariant: GPU routing must improve H100 routing rate for lc jobs."""
        report = self._run_with_synthetic_data()
        # With equal energy prices, H100 routing rate must strictly improve.
        assert report.gpu_routing_pct_on_best_gpu > report.baseline_pct_on_best_gpu, (
            f"Expected GPU routing to improve H100 rate: "
            f"baseline={report.baseline_pct_on_best_gpu:.2%}, "
            f"gpu_routing={report.gpu_routing_pct_on_best_gpu:.2%}"
        )

    def test_gpu_routing_reduces_mean_penalty(self):
        report = self._run_with_synthetic_data()
        assert report.gpu_routing_mean_gpu_penalty <= report.baseline_mean_gpu_penalty, (
            f"Expected GPU routing to reduce mean penalty: "
            f"baseline={report.baseline_mean_gpu_penalty:.4f}, "
            f"gpu_routing={report.gpu_routing_mean_gpu_penalty:.4f}"
        )

    def test_routing_improvement_pp_is_nonnegative(self):
        report = self._run_with_synthetic_data()
        assert report.routing_improvement_pp >= 0.0

    def test_to_dict_serialises_without_error(self):
        report = self._run_with_synthetic_data()
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["shadow_tag"].startswith("shadow")


# ---------------------------------------------------------------------------
# TestGpuRoutingRegression
# ---------------------------------------------------------------------------

class TestGpuRoutingRegression:
    """Invariants that must hold regardless of specific numbers."""

    def test_canonical_region_gpu_types_has_all_three_regions(self):
        assert "us-west" in CANONICAL_REGION_GPU_TYPES
        assert "us-east" in CANONICAL_REGION_GPU_TYPES
        assert "us-south" in CANONICAL_REGION_GPU_TYPES

    def test_best_gpu_region_is_in_region_gpu_types(self):
        assert BEST_GPU_REGION in CANONICAL_REGION_GPU_TYPES
        assert CANONICAL_REGION_GPU_TYPES[BEST_GPU_REGION] == BEST_GPU_TYPE

    def test_synthetic_ttft_values_are_ordered(self):
        h = SYNTHETIC_GPU_TTFT_P50_S["h100"]
        a = SYNTHETIC_GPU_TTFT_P50_S["a100"]
        t = SYNTHETIC_GPU_TTFT_P50_S["t4"]
        assert h < a < t, f"Expected h100<a100<t4 but got {h}, {a}, {t}"

    def test_ttft_spread_is_at_least_5x(self):
        h = SYNTHETIC_GPU_TTFT_P50_S["h100"]
        t = SYNTHETIC_GPU_TTFT_P50_S["t4"]
        spread = t / h
        assert spread >= 5.0, (
            f"TTFT spread t4/h100 = {spread:.1f}× — expected ≥5× (CARA cites 9×)"
        )
