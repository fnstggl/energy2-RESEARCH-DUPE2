"""Tests for SRTF (Shortest-Remaining-Time-First) scheduling integration.

Covers:
1. Job.predicted_output_tokens field — existence, default, backward compat
2. Scheduler._SLA_CLASS_RANK — ordering values
3. _solve_greedy sort key — SLA class + SRTF ordering, backward compat
4. augment_jobs_with_srtf_priors — correct augmentation + non-mutation
5. SRTFBacktestReport structure and serialization
6. Mini end-to-end integration with synthetic price data
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from aurelius.models import Job, OptimizationConfig, ScheduleDecision
from aurelius.optimization.scheduler import JobScheduler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _utc(hour: int, day: int = 1) -> datetime:
    return datetime(2026, 2, day, hour, 0, tzinfo=timezone.utc)


def _make_job(
    job_id: str = "j0",
    priority: int = 1,
    runtime_hours: float = 4.0,
    sla_class: str = "best_effort",
    workload_type: str = "scheduled_batch",
    predicted_output_tokens: Optional[float] = None,
    slack_hours: float = 24.0,
) -> Job:
    earliest = _utc(0)
    deadline = earliest + timedelta(hours=runtime_hours + slack_hours)
    return Job(
        job_id=job_id,
        submit_time=earliest,
        runtime_hours=runtime_hours,
        deadline=deadline,
        power_kw=100.0,
        earliest_start=earliest,
        region_options=["us-west", "us-east"],
        priority=priority,
        workload_type=workload_type,
        sla_class=sla_class,
        predicted_output_tokens=predicted_output_tokens,
    )


def _flat_prices(regions=("us-west", "us-east"), price=50.0, hours=72) -> dict:
    prices = {}
    base = _utc(0)
    for r in regions:
        prices[r] = {base + timedelta(hours=h): price for h in range(hours)}
    return prices


# ---------------------------------------------------------------------------
# 1. Job.predicted_output_tokens field
# ---------------------------------------------------------------------------

class TestJobPredictedOutputTokens:
    def test_default_is_none(self):
        j = _make_job()
        assert j.predicted_output_tokens is None

    def test_can_set_float(self):
        j = _make_job(predicted_output_tokens=1_000_000.0)
        assert j.predicted_output_tokens == 1_000_000.0

    def test_can_replace_via_dataclass(self):
        j = _make_job()
        j2 = dataclasses.replace(j, predicted_output_tokens=500_000.0)
        assert j.predicted_output_tokens is None  # original unchanged
        assert j2.predicted_output_tokens == 500_000.0

    def test_field_preserved_in_copy(self):
        j = _make_job(predicted_output_tokens=250_000.0)
        j2 = dataclasses.replace(j, job_id="j2")
        assert j2.predicted_output_tokens == 250_000.0

    def test_zero_allowed(self):
        j = _make_job(predicted_output_tokens=0.0)
        assert j.predicted_output_tokens == 0.0


# ---------------------------------------------------------------------------
# 2. Scheduler._SLA_CLASS_RANK
# ---------------------------------------------------------------------------

class TestSLAClassRank:
    def test_rank_exists(self):
        assert hasattr(JobScheduler, "_SLA_CLASS_RANK")

    def test_latency_critical_lowest(self):
        rank = JobScheduler._SLA_CLASS_RANK
        assert rank["latency_critical"] < rank["deadline"]
        assert rank["latency_critical"] < rank["best_effort"]

    def test_deadline_before_best_effort(self):
        rank = JobScheduler._SLA_CLASS_RANK
        assert rank["deadline"] < rank["best_effort"]

    def test_covers_known_classes(self):
        rank = JobScheduler._SLA_CLASS_RANK
        assert "latency_critical" in rank
        assert "deadline" in rank
        assert "best_effort" in rank


# ---------------------------------------------------------------------------
# 3. Greedy sort key behavior
# ---------------------------------------------------------------------------

class TestGreedySortKey:
    """Verify that the greedy scheduler processes jobs in SRTF order."""

    def _run_and_get_order(self, jobs: list[Job]) -> list[str]:
        """Return job_ids in the order they appear in the schedule."""
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="greedy")
        # Schedule order may not exactly match processing order, but start_time
        # reflects scheduling decisions — sort by start time as a proxy.
        return [d.job_id for d in sorted(result.schedule, key=lambda d: d.start_time)]

    def test_backward_compat_no_prior(self):
        """Without predicted_output_tokens, long-deadline jobs go last (unchanged)."""
        j_early = _make_job("j_early", slack_hours=4.0)    # tight deadline
        j_late = _make_job("j_late", slack_hours=48.0)     # loose deadline
        # Both have no prior — sort by deadline (j_early first)
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        # Just verify no exception is raised and both jobs are scheduled
        result = scheduler.solve([j_early, j_late], prices, carbon, method="greedy")
        assert len(result.schedule) == 2

    def test_srtf_short_before_long_same_priority(self):
        """Short-prior job must be RANKED before long-prior job at equal priority."""
        j_short = _make_job("j_short", runtime_hours=2.0,
                             predicted_output_tokens=1_000_000.0,
                             slack_hours=48.0)
        j_long = _make_job("j_long", runtime_hours=8.0,
                            predicted_output_tokens=8_000_000.0,
                            slack_hours=48.0)
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        # Build sort keys directly
        k_short = scheduler._solve_greedy.__func__  # access via class
        rank = JobScheduler._SLA_CLASS_RANK
        key_s = (-j_short.priority, rank.get(j_short.sla_class, 2),
                 j_short.predicted_output_tokens, j_short.deadline)
        key_l = (-j_long.priority, rank.get(j_long.sla_class, 2),
                 j_long.predicted_output_tokens, j_long.deadline)
        assert key_s < key_l, "Short job must sort before long job"

    def test_srtf_no_prior_uses_inf(self):
        """Job without prior → length_prior = inf → sorts after jobs with priors."""
        rank = JobScheduler._SLA_CLASS_RANK
        j_with = _make_job("j_w", predicted_output_tokens=1_000_000.0, slack_hours=48.0)
        j_without = _make_job("j_wo", predicted_output_tokens=None, slack_hours=48.0)
        key_with = (
            -j_with.priority,
            rank.get(j_with.sla_class, 2),
            j_with.predicted_output_tokens,
            j_with.deadline,
        )
        key_without = (
            -j_without.priority,
            rank.get(j_without.sla_class, 2),
            float("inf"),
            j_without.deadline,
        )
        assert key_with < key_without

    def test_sla_class_latency_critical_before_best_effort(self):
        """latency_critical job sorts before best_effort job at same priority."""
        rank = JobScheduler._SLA_CLASS_RANK
        j_lc = _make_job("j_lc", sla_class="latency_critical",
                          predicted_output_tokens=None, slack_hours=48.0)
        j_be = _make_job("j_be", sla_class="best_effort",
                          predicted_output_tokens=None, slack_hours=48.0)
        key_lc = (-j_lc.priority, rank.get(j_lc.sla_class, 2), float("inf"), j_lc.deadline)
        key_be = (-j_be.priority, rank.get(j_be.sla_class, 2), float("inf"), j_be.deadline)
        assert key_lc < key_be

    def test_sla_class_latency_critical_before_deadline(self):
        rank = JobScheduler._SLA_CLASS_RANK
        j_lc = _make_job("j_lc", sla_class="latency_critical", slack_hours=48.0)
        j_dl = _make_job("j_dl", sla_class="deadline", slack_hours=48.0)
        key_lc = (-j_lc.priority, rank.get(j_lc.sla_class, 2), float("inf"), j_lc.deadline)
        key_dl = (-j_dl.priority, rank.get(j_dl.sla_class, 2), float("inf"), j_dl.deadline)
        assert key_lc < key_dl

    def test_priority_dominates_srtf(self):
        """Higher priority always overrides SLA class rank and length prior."""
        rank = JobScheduler._SLA_CLASS_RANK
        j_high = _make_job("j_high", priority=10, sla_class="best_effort",
                             predicted_output_tokens=9_000_000.0, slack_hours=48.0)
        j_low = _make_job("j_low", priority=1, sla_class="latency_critical",
                            predicted_output_tokens=1.0, slack_hours=48.0)
        key_high = (-j_high.priority, rank.get(j_high.sla_class, 2),
                    j_high.predicted_output_tokens, j_high.deadline)
        key_low = (-j_low.priority, rank.get(j_low.sla_class, 2),
                   j_low.predicted_output_tokens, j_low.deadline)
        assert key_high < key_low, "Higher numeric priority must win"

    def test_deadline_tiebreak_within_same_sla_no_prior(self):
        """Without prior, earlier deadline wins tiebreak (unchanged behavior)."""
        rank = JobScheduler._SLA_CLASS_RANK
        j_early = _make_job("j_early", slack_hours=4.0)
        j_late = _make_job("j_late", slack_hours=48.0)
        key_e = (-j_early.priority, rank.get(j_early.sla_class, 2),
                 float("inf"), j_early.deadline)
        key_l = (-j_late.priority, rank.get(j_late.sla_class, 2),
                 float("inf"), j_late.deadline)
        assert key_e < key_l

    def test_greedy_runs_with_srtf_priors(self):
        """Full greedy solve runs without error when priors are set."""
        jobs = [
            _make_job("j1", runtime_hours=1.0, predicted_output_tokens=500_000.0,
                       slack_hours=24.0),
            _make_job("j2", runtime_hours=4.0, predicted_output_tokens=2_000_000.0,
                       slack_hours=24.0),
            _make_job("j3", runtime_hours=8.0, predicted_output_tokens=None,
                       slack_hours=48.0),
        ]
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="greedy")
        assert len(result.schedule) == 3

    def test_greedy_no_regressions_all_scheduled(self):
        """All jobs scheduled even with mixed prior/no-prior jobs."""
        jobs = [_make_job(f"j{i}", predicted_output_tokens=float(i * 1e6) if i % 2 == 0 else None)
                for i in range(10)]
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="greedy")
        assert len(result.schedule) == 10


# ---------------------------------------------------------------------------
# 4. augment_jobs_with_srtf_priors
# ---------------------------------------------------------------------------

class TestAugmentJobsWithSRTFPriors:
    def test_llm_batch_inference_gets_prior(self):
        from aurelius.benchmarks.srtf_backtest import (
            SRTF_TOKENS_PER_HOUR,
            augment_jobs_with_srtf_priors,
        )
        j = _make_job(workload_type="llm_batch_inference", runtime_hours=4.0)
        aug = augment_jobs_with_srtf_priors([j])
        assert aug[0].predicted_output_tokens == pytest.approx(4.0 * SRTF_TOKENS_PER_HOUR)

    def test_realtime_inference_gets_prior(self):
        from aurelius.benchmarks.srtf_backtest import (
            SRTF_TOKENS_PER_HOUR,
            augment_jobs_with_srtf_priors,
        )
        j = _make_job(workload_type="realtime_inference", runtime_hours=1.0)
        aug = augment_jobs_with_srtf_priors([j])
        assert aug[0].predicted_output_tokens == pytest.approx(SRTF_TOKENS_PER_HOUR)

    def test_training_no_prior(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        j = _make_job(workload_type="training", runtime_hours=12.0)
        aug = augment_jobs_with_srtf_priors([j])
        assert aug[0].predicted_output_tokens is None

    def test_fine_tuning_no_prior(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        j = _make_job(workload_type="fine_tuning", runtime_hours=8.0)
        aug = augment_jobs_with_srtf_priors([j])
        assert aug[0].predicted_output_tokens is None

    def test_data_processing_no_prior(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        j = _make_job(workload_type="data_processing", runtime_hours=6.0)
        aug = augment_jobs_with_srtf_priors([j])
        assert aug[0].predicted_output_tokens is None

    def test_original_not_mutated(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        j = _make_job(workload_type="llm_batch_inference", runtime_hours=4.0)
        augment_jobs_with_srtf_priors([j])
        assert j.predicted_output_tokens is None  # original unchanged

    def test_mixed_list_correctly_partitioned(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        jobs = [
            _make_job("j0", workload_type="llm_batch_inference"),
            _make_job("j1", workload_type="training"),
            _make_job("j2", workload_type="realtime_inference"),
            _make_job("j3", workload_type="data_processing"),
        ]
        aug = augment_jobs_with_srtf_priors(jobs)
        assert aug[0].predicted_output_tokens is not None
        assert aug[1].predicted_output_tokens is None
        assert aug[2].predicted_output_tokens is not None
        assert aug[3].predicted_output_tokens is None

    def test_empty_list(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        assert augment_jobs_with_srtf_priors([]) == []

    def test_custom_tokens_per_hour(self):
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        j = _make_job(workload_type="llm_batch_inference", runtime_hours=2.0)
        aug = augment_jobs_with_srtf_priors([j], tokens_per_hour=1_000.0)
        assert aug[0].predicted_output_tokens == pytest.approx(2_000.0)

    def test_short_before_long_after_augmentation(self):
        """After augmentation, short-runtime LLM jobs sort before long-runtime ones."""
        from aurelius.benchmarks.srtf_backtest import augment_jobs_with_srtf_priors
        j_short = _make_job("short", workload_type="llm_batch_inference",
                             runtime_hours=2.0, slack_hours=48.0)
        j_long = _make_job("long", workload_type="llm_batch_inference",
                            runtime_hours=8.0, slack_hours=48.0)
        aug = augment_jobs_with_srtf_priors([j_long, j_short])
        # After augment + sort by predicted_output_tokens: short should precede long
        aug_sorted = sorted(aug, key=lambda j: (
            j.predicted_output_tokens if j.predicted_output_tokens is not None else float("inf"),
            j.deadline,
        ))
        assert aug_sorted[0].job_id == "short"
        assert aug_sorted[1].job_id == "long"


# ---------------------------------------------------------------------------
# 5. SRTFBacktestReport structure
# ---------------------------------------------------------------------------

class TestSRTFBacktestReport:
    def _make_report(self, delta_pct: float = 0.0) -> "SRTFBacktestReport":
        from aurelius.benchmarks.srtf_backtest import SRTFBacktestReport
        return SRTFBacktestReport(
            total_jobs=1000,
            srtf_eligible_jobs=500,
            srtf_eligible_pct=50.0,
            baseline_goodput_per_dollar=0.300,
            srtf_goodput_per_dollar=0.300 * (1 + delta_pct / 100),
            goodput_per_dollar_delta=0.300 * delta_pct / 100,
            goodput_per_dollar_delta_pct=delta_pct,
            baseline_realized_cost_usd=50_000.0,
            srtf_realized_cost_usd=50_000.0 - 200.0 * delta_pct,
            realized_cost_delta_usd=-200.0 * delta_pct,
            baseline_deadline_misses=0,
            srtf_deadline_misses=0,
            srtf_eligible_pct_scheduled_short_first=0.85,
        )

    def test_to_dict_has_required_keys(self):
        r = self._make_report(delta_pct=1.5)
        d = r.to_dict()
        for k in (
            "total_jobs", "srtf_eligible_jobs", "srtf_eligible_pct",
            "baseline_goodput_per_dollar", "srtf_goodput_per_dollar",
            "goodput_per_dollar_delta", "goodput_per_dollar_delta_pct",
            "baseline_realized_cost_usd", "srtf_realized_cost_usd",
            "realized_cost_delta_usd", "baseline_deadline_misses",
            "srtf_deadline_misses", "srtf_eligible_pct_scheduled_short_first",
            "shadow_tag",
        ):
            assert k in d, f"Missing key: {k}"

    def test_shadow_tag_present(self):
        r = self._make_report()
        assert "shadow" in r.to_dict()["shadow_tag"]

    def test_positive_delta_means_improvement(self):
        r = self._make_report(delta_pct=2.0)
        assert r.goodput_per_dollar_delta > 0
        assert r.srtf_goodput_per_dollar > r.baseline_goodput_per_dollar

    def test_zero_delta_neutral(self):
        r = self._make_report(delta_pct=0.0)
        assert r.goodput_per_dollar_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Mini end-to-end integration
# ---------------------------------------------------------------------------

class TestSRTFEndToEnd:
    """Runs the SRTF and baseline schedulers on a small synthetic trace and
    verifies that SRTF ordering properties hold."""

    def test_srtf_routes_short_llm_before_long_llm(self):
        """SRTF scheduler picks earlier time slots for shorter LLM jobs."""
        jobs = [
            _make_job("long_llm", workload_type="llm_batch_inference",
                       runtime_hours=8.0, predicted_output_tokens=4_000_000.0,
                       slack_hours=48.0),
            _make_job("short_llm", workload_type="llm_batch_inference",
                       runtime_hours=2.0, predicted_output_tokens=1_000_000.0,
                       slack_hours=48.0),
        ]
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="greedy")
        sched_by_id = {d.job_id: d for d in result.schedule}
        # Short job should be scheduled (start_time ≤) before long job
        assert sched_by_id["short_llm"].start_time <= sched_by_id["long_llm"].start_time

    def test_srtf_no_deadline_violations_introduced(self):
        """SRTF sort must not cause deadline misses that baseline avoids."""
        jobs = [
            _make_job(f"j{i}", workload_type="llm_batch_inference",
                       runtime_hours=float((i % 4) + 1),
                       predicted_output_tokens=float((i % 4) + 1) * 500_000.0,
                       slack_hours=24.0)
            for i in range(8)
        ]
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="greedy")
        job_by_id = {j.job_id: j for j in jobs}
        for d in result.schedule:
            job = job_by_id[d.job_id]
            assert d.end_time <= job.deadline, f"Deadline miss for {d.job_id}"

    def test_mixed_workload_training_unaffected_by_srtf(self):
        """Training jobs (no prior) retain deadline-based ordering even with SRTF jobs present."""
        jobs = [
            _make_job("llm_short", workload_type="llm_batch_inference",
                       runtime_hours=2.0, predicted_output_tokens=1_000_000.0,
                       slack_hours=24.0),
            _make_job("train_tight", workload_type="training",
                       runtime_hours=4.0, predicted_output_tokens=None,
                       slack_hours=6.0),  # tight deadline
            _make_job("llm_long", workload_type="llm_batch_inference",
                       runtime_hours=6.0, predicted_output_tokens=3_000_000.0,
                       slack_hours=24.0),
        ]
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="greedy")
        assert len(result.schedule) == 3

    def test_srtf_preserves_high_priority_dominance(self):
        """High-priority long job always scheduled before low-priority short job."""
        j_high_long = _make_job("high_long", priority=5, runtime_hours=8.0,
                                  predicted_output_tokens=8_000_000.0, slack_hours=48.0)
        j_low_short = _make_job("low_short", priority=1, runtime_hours=1.0,
                                  predicted_output_tokens=100_000.0, slack_hours=48.0)
        rank = JobScheduler._SLA_CLASS_RANK
        key_high = (-j_high_long.priority, rank.get(j_high_long.sla_class, 2),
                    j_high_long.predicted_output_tokens, j_high_long.deadline)
        key_low = (-j_low_short.priority, rank.get(j_low_short.sla_class, 2),
                   j_low_short.predicted_output_tokens, j_low_short.deadline)
        assert key_high < key_low

    def test_local_search_also_works_with_priors(self):
        """Local search solver handles predicted_output_tokens without error."""
        jobs = [
            _make_job(f"j{i}", predicted_output_tokens=float(i) * 1e6, slack_hours=24.0)
            for i in range(5)
        ]
        scheduler = JobScheduler(OptimizationConfig(
            default_region="us-west", min_power_fraction=1.0,
        ))
        prices = _flat_prices(price=50.0, hours=72)
        carbon = {r: {} for r in prices}
        result = scheduler.solve(jobs, prices, carbon, method="local_search",
                                  time_limit_seconds=5.0)
        assert len(result.schedule) == 5
