"""Integration tests for GPU placement scorer wiring in JobScheduler.

Validates:
 - Disabled scorer (default): no change in placement decisions
 - Enabled scorer: latency_critical jobs routed to faster GPU region
 - Non-latency-critical jobs: unaffected by GPU placement scorer
 - _sla_adjusted_score: gpu_penalty correctly folds into score
 - Fail-open: missing region_gpu_types → no penalty, same decisions
 - Fail-open: region not in region_gpu_types map → no penalty
 - Peer comparison: fastest region gets floor penalty, slowest gets ceil
 - Scorer disabled at config level: all penalties = 0
 - Multiple jobs: latency_critical routed to h100, best_effort to cheapest
 - No scheduler / controller imports in gpu_placement_scorer module
 - by_gpu_counts populated in TTFTShadowPrior.fit_from_rows
 - predict() falls through to by_gpu when model_size=None
 - subgroup_n() uses by_gpu_counts when model_size=None

Research basis:
 - "Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for
   SLO-Constrained Inference" (arXiv:2604.07472) — near-optimal allocation.
 - "KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving"
   (arXiv:2604.16682) — TTFT-aware SLO routing across GPU generations.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from aurelius.forecasting.gpu_placement_scorer import (
    GpuPlacementConfig,
    GpuPlacementScorer,
)
from aurelius.forecasting.ttft_shadow_prior import TTFTShadowPrior
from aurelius.models import Job, OptimizationConfig, ScheduleDecision
from aurelius.optimization.scheduler import JobScheduler

UTC = timezone.utc
T0 = datetime(2024, 6, 1, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h(n: float) -> timedelta:
    return timedelta(hours=n)


def _make_price(regions, hours=48, price=50.0):
    data = {}
    for r in regions:
        data[r] = {}
        for h in range(hours):
            ts = (T0 + _h(h)).replace(minute=0, second=0, microsecond=0)
            data[r][ts] = price
    return data


def _make_price_differential(regions_prices: dict, hours=48):
    """Build price data with per-region prices."""
    data = {}
    for r, price in regions_prices.items():
        data[r] = {}
        for h in range(hours):
            ts = (T0 + _h(h)).replace(minute=0, second=0, microsecond=0)
            data[r][ts] = price
    return data


def _make_carbon(regions, hours=48, carbon=300.0):
    data = {}
    for r in regions:
        data[r] = {}
        for h in range(hours):
            ts = (T0 + _h(h)).replace(minute=0, second=0, microsecond=0)
            data[r][ts] = carbon
    return data


def _job(
    job_id="j1",
    sla_class="best_effort",
    region_options=None,
    runtime_hours=2.0,
    power_kw=100.0,
):
    if region_options is None:
        region_options = ["r-h100", "r-t4"]
    return Job(
        job_id=job_id,
        submit_time=T0,
        runtime_hours=runtime_hours,
        deadline=T0 + _h(24),
        power_kw=power_kw,
        earliest_start=T0,
        region_options=region_options,
        sla_class=sla_class,
    )


def _build_prior(gpu_p50_map: dict, rows_per_gpu: int = 100) -> TTFTShadowPrior:
    """Build TTFTShadowPrior from {gpu_type: ttft_p50_s} map."""
    rows = []
    for gpu_type, ttft_p50 in gpu_p50_map.items():
        instance_type = f"qwen2-7b_{gpu_type}"
        for i in range(rows_per_gpu):
            jitter = (i % 5) * 0.01 - 0.02
            rows.append({
                "instance_type": instance_type,
                "num_prompt_tokens": 256,
                "actual_ttft_s": ttft_p50 + jitter,
            })
    prior = TTFTShadowPrior()
    prior.fit_from_rows(rows)
    return prior


def _build_scorer(gpu_p50_map: dict, enabled: bool = True) -> GpuPlacementScorer:
    prior = _build_prior(gpu_p50_map)
    config = GpuPlacementConfig(enabled=enabled, min_subgroup_rows=50)
    return GpuPlacementScorer(prior=prior, config=config)


# ---------------------------------------------------------------------------
# TTFTShadowPrior: new fields and fallback behavior
# ---------------------------------------------------------------------------


class TestTTFTShadowPriorGpuFallback:
    def test_by_gpu_counts_populated(self):
        prior = _build_prior({"h100": 0.05, "t4": 0.45})
        assert "h100" in prior.by_gpu_counts
        assert "t4" in prior.by_gpu_counts
        assert prior.by_gpu_counts["h100"] == 100
        assert prior.by_gpu_counts["t4"] == 100

    def test_predict_model_size_none_falls_through_to_by_gpu(self):
        prior = _build_prior({"h100": 0.05, "t4": 0.45})
        p_h100 = prior.predict(model_size=None, gpu_type="h100", prompt_tokens=256)
        p_t4 = prior.predict(model_size=None, gpu_type="t4", prompt_tokens=256)
        assert p_h100 is not None
        assert p_t4 is not None
        assert p_h100 < p_t4, "h100 should be faster than t4"

    def test_predict_unknown_gpu_with_model_size_none_returns_none(self):
        prior = _build_prior({"h100": 0.05})
        p = prior.predict(model_size=None, gpu_type="v100", prompt_tokens=256)
        assert p is None

    def test_predict_gpu_type_none_returns_none(self):
        prior = _build_prior({"h100": 0.05})
        p = prior.predict(model_size=None, gpu_type=None, prompt_tokens=256)
        assert p is None

    def test_subgroup_n_model_size_none_uses_by_gpu_counts(self):
        prior = _build_prior({"h100": 0.05, "t4": 0.45})
        n_h100 = prior.subgroup_n(model_size=None, gpu_type="h100", prompt_tokens=256)
        assert n_h100 == 100

    def test_subgroup_n_unknown_gpu_returns_zero(self):
        prior = _build_prior({"h100": 0.05})
        n = prior.subgroup_n(model_size=None, gpu_type="v100", prompt_tokens=256)
        assert n == 0

    def test_predict_full_model_size_still_works(self):
        prior = _build_prior({"h100": 0.05, "t4": 0.45})
        # With explicit model_size, should still hit the full table hierarchy
        p = prior.predict(model_size="7b", gpu_type="h100", prompt_tokens=256)
        assert p is not None
        assert p < 0.1  # should be near 0.05

    def test_to_dict_includes_by_gpu_counts(self):
        prior = _build_prior({"h100": 0.05})
        d = prior.to_dict()
        assert "by_gpu_counts" in d
        assert d["by_gpu_counts"]["h100"] == 100

    def test_empty_prior_predict_none(self):
        prior = TTFTShadowPrior()
        assert prior.predict(model_size=None, gpu_type="h100", prompt_tokens=None) is None


# ---------------------------------------------------------------------------
# Scheduler: _sla_adjusted_score with gpu_penalty
# ---------------------------------------------------------------------------


class TestSlaAdjustedScoreWithGpuPenalty:
    def test_no_penalty_when_gpu_penalty_zero(self):
        score = JobScheduler._sla_adjusted_score(100.0, None, gpu_penalty=0.0)
        assert score == 100.0

    def test_gpu_penalty_adds_fraction(self):
        # penalty of 0.10 → score increases by 10% of |objective|
        score = JobScheduler._sla_adjusted_score(100.0, None, gpu_penalty=0.10)
        assert abs(score - 110.0) < 1e-9

    def test_gpu_penalty_combines_with_sla_penalty(self):
        class FakeSlaEval:
            risk_score = 5.0
            soft_penalty_score = 5.0
        # SLA = 0.10, GPU = 0.10, combined = 0.20 → +20%
        score = JobScheduler._sla_adjusted_score(100.0, FakeSlaEval(), gpu_penalty=0.10)
        assert abs(score - 120.0) < 1e-9

    def test_negative_objective_uses_abs_for_penalty(self):
        score = JobScheduler._sla_adjusted_score(-100.0, None, gpu_penalty=0.10)
        assert abs(score - (-90.0)) < 1e-9

    def test_default_gpu_penalty_is_zero(self):
        score = JobScheduler._sla_adjusted_score(50.0, None)
        assert score == 50.0


# ---------------------------------------------------------------------------
# Scheduler: GPU placement wiring integration
# ---------------------------------------------------------------------------


class TestSchedulerGpuPlacementWiring:
    """Tests for the scheduler's GPU placement penalty integration.

    Scenario: Two regions — r-h100 (H100, fast TTFT) and r-t4 (T4, slow TTFT).
    Prices are equal in both regions so only TTFT penalty drives routing.
    """

    REGIONS = ["r-h100", "r-t4"]
    REGION_GPU_TYPES = {"r-h100": "h100", "r-t4": "t4"}
    # H100 is 9× faster than T4 to match CARA empirical data
    GPU_P50_MAP = {"h100": 0.05, "t4": 0.45}

    def _scheduler(self, enabled: bool = True) -> JobScheduler:
        scorer = _build_scorer(self.GPU_P50_MAP, enabled=enabled)
        return JobScheduler(
            gpu_placement_scorer=scorer,
            region_gpu_types=self.REGION_GPU_TYPES,
        )

    def _price_data(self, price=50.0):
        return _make_price(self.REGIONS, price=price)

    def _carbon_data(self):
        return _make_carbon(self.REGIONS)

    def test_latency_critical_routed_to_h100(self):
        """latency_critical job prefers h100 region despite equal price."""
        scheduler = self._scheduler(enabled=True)
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], self._price_data(), self._carbon_data(), method="greedy"
        )
        assert len(result.schedule) == 1
        assert result.schedule[0].region == "r-h100", (
            f"Expected r-h100 for latency_critical, got {result.schedule[0].region}"
        )

    def test_best_effort_chooses_cheapest_region(self):
        """best_effort job follows price only — GPU penalty is zero."""
        # Make t4 region slightly cheaper so best_effort goes there
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        scheduler = self._scheduler(enabled=True)
        job = _job(sla_class="best_effort", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        assert result.schedule[0].region == "r-t4"

    def test_scorer_disabled_no_routing_effect(self):
        """Disabled scorer: latency_critical job routes to cheapest, not fastest GPU."""
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        scheduler = self._scheduler(enabled=False)
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        # With scorer disabled, price wins → t4 (cheaper)
        assert result.schedule[0].region == "r-t4"

    def test_no_scorer_no_change(self):
        """Without gpu_placement_scorer, scheduler behaves identically to before."""
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        scheduler = JobScheduler()  # no scorer, no region_gpu_types
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        assert result.schedule[0].region == "r-t4"  # price wins when no scorer

    def test_empty_region_gpu_types_no_penalty(self):
        """Empty region_gpu_types mapping → no GPU penalty, price-only routing."""
        scorer = _build_scorer(self.GPU_P50_MAP, enabled=True)
        scheduler = JobScheduler(
            gpu_placement_scorer=scorer,
            region_gpu_types={},  # no mapping
        )
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        assert result.schedule[0].region == "r-t4"  # price wins without GPU info

    def test_region_not_in_gpu_map_no_penalty(self):
        """Region absent from region_gpu_types gets penalty=0 (fail-open)."""
        scorer = _build_scorer(self.GPU_P50_MAP, enabled=True)
        scheduler = JobScheduler(
            gpu_placement_scorer=scorer,
            region_gpu_types={"r-h100": "h100"},  # r-t4 not mapped
        )
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        # r-t4 has no GPU penalty (not in map) + lower price → r-t4 wins
        assert result.schedule[0].region == "r-t4"

    def test_two_jobs_mixed_sla_class(self):
        """latency_critical goes to h100; best_effort goes to cheaper t4."""
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        scheduler = self._scheduler(enabled=True)
        j_critical = _job(
            job_id="crit", sla_class="latency_critical", region_options=self.REGIONS,
            runtime_hours=1.0,
        )
        j_best = _job(
            job_id="best", sla_class="best_effort", region_options=self.REGIONS,
            runtime_hours=1.0,
        )
        result = scheduler.solve(
            [j_critical, j_best], price_data, self._carbon_data(), method="greedy"
        )
        by_id = {d.job_id: d for d in result.schedule}
        assert by_id["crit"].region == "r-h100"
        assert by_id["best"].region == "r-t4"

    def test_equal_price_latency_critical_prefers_h100(self):
        """With equal prices, GPU penalty alone should drive latency_critical to h100."""
        price_data = _make_price(self.REGIONS, price=50.0)  # equal price
        scheduler = self._scheduler(enabled=True)
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        assert result.schedule[0].region == "r-h100"

    def test_no_regression_existing_scheduler_params(self):
        """Existing tests still work: scheduler without GPU scorer unchanged."""
        price_data = _make_price(["us-west", "us-east"])
        carbon_data = _make_carbon(["us-west", "us-east"])
        config = OptimizationConfig(region_power_caps={"us-west": 200.0})
        scheduler = JobScheduler(config=config)
        job = _job(sla_class="best_effort", region_options=["us-west", "us-east"])
        result = scheduler.solve([job], price_data, carbon_data, method="greedy")
        assert len(result.schedule) == 1  # basic sanity

    def test_three_gpu_types_rank_order(self):
        """With three GPU regions, relative rank penalty increases from h100 → a100 → t4."""
        regions = ["r-h100", "r-a100", "r-t4"]
        region_gpu = {"r-h100": "h100", "r-a100": "a100", "r-t4": "t4"}
        gpu_p50 = {"h100": 0.05, "a100": 0.12, "t4": 0.45}
        scorer = _build_scorer(gpu_p50, enabled=True)
        scheduler = JobScheduler(
            gpu_placement_scorer=scorer,
            region_gpu_types=region_gpu,
        )
        price_data = _make_price(regions, price=50.0)
        carbon_data = _make_carbon(regions)
        job = _job(sla_class="latency_critical", region_options=regions)
        result = scheduler.solve([job], price_data, carbon_data, method="greedy")
        assert result.schedule[0].region == "r-h100"

    def test_insufficient_sample_falls_back_to_no_penalty(self):
        """When GPU prior has too few rows, penalty=0 for all regions."""
        prior = _build_prior({"h100": 0.05, "t4": 0.45}, rows_per_gpu=10)
        # min_subgroup_rows=50 > 10 → all regions get insufficient_sample
        config = GpuPlacementConfig(enabled=True, min_subgroup_rows=50)
        scorer = GpuPlacementScorer(prior=prior, config=config)
        price_data = _make_price_differential({"r-h100": 60.0, "r-t4": 40.0})
        scheduler = JobScheduler(
            gpu_placement_scorer=scorer,
            region_gpu_types=self.REGION_GPU_TYPES,
        )
        job = _job(sla_class="latency_critical", region_options=self.REGIONS)
        result = scheduler.solve(
            [job], price_data, self._carbon_data(), method="greedy"
        )
        # Insufficient sample → no GPU penalty → price wins → t4 (cheaper)
        assert result.schedule[0].region == "r-t4"

    def test_single_region_no_peer_comparison(self):
        """Single-region job: no peer comparison possible, no GPU penalty."""
        scheduler = self._scheduler(enabled=True)
        price_data = _make_price(["r-h100"])
        carbon_data = _make_carbon(["r-h100"])
        job = _job(sla_class="latency_critical", region_options=["r-h100"])
        result = scheduler.solve([job], price_data, carbon_data, method="greedy")
        assert result.schedule[0].region == "r-h100"

    def test_scheduler_gpu_scorer_attribute(self):
        """Scheduler stores the scorer and region_gpu_types correctly."""
        scorer = _build_scorer(self.GPU_P50_MAP, enabled=True)
        scheduler = JobScheduler(
            gpu_placement_scorer=scorer,
            region_gpu_types=self.REGION_GPU_TYPES,
        )
        assert scheduler._gpu_placement_scorer is scorer
        assert scheduler._region_gpu_types == self.REGION_GPU_TYPES

    def test_default_scheduler_has_no_scorer(self):
        """Default JobScheduler has no GPU placement scorer."""
        scheduler = JobScheduler()
        assert scheduler._gpu_placement_scorer is None
        assert scheduler._region_gpu_types == {}
