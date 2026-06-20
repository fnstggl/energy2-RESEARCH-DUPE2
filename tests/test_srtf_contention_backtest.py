"""Tests for the SRTF-under-contention probe on the batch ``JobScheduler``.

This module documents a NEGATIVE finding: the merged ``predicted_output_tokens``
sort key cannot produce a goodput/$ improvement in the greedy batch scheduler,
even at high capacity-contention ratios, because that scheduler has no
queue-wait semantics (it falls back to ``earliest_start`` rather than making a
job wait).  These tests lock in that boundary so a future change that claims a
batch-scheduler SRTF win has to update them deliberately.

The request-level SRTF benefit is evaluated separately in
``test_srtf_serving_backtest.py`` (a discrete-event queue, where ordering does
change completion times).
"""

from __future__ import annotations

import logging

import pytest

from aurelius.benchmarks.srtf_contention_backtest import (
    JOB_POWER_KW,
    REGION_POWER_CAP_KW,
    SRTFContentionReport,
    build_contended_jobs,
    load_azure_output_tokens,
    run_srtf_contention_backtest,
    with_srtf_prior,
)

logging.disable(logging.CRITICAL)


class TestAzureTokenLoading:
    def test_loads_real_tokens(self):
        toks = load_azure_output_tokens()
        assert len(toks) > 1000
        assert all(t > 0 for t in toks)

    def test_limit(self):
        assert len(load_azure_output_tokens(limit=100)) == 100


class TestContendedJobs:
    def test_jobs_share_window_and_region(self):
        jobs = build_contended_jobs([90, 224, 479], horizon_hours=24)
        assert len(jobs) == 3
        assert len({j.earliest_start for j in jobs}) == 1
        assert len({j.deadline for j in jobs}) == 1
        assert all(j.region_options == ["us-west"] for j in jobs)
        assert all(j.power_kw == JOB_POWER_KW for j in jobs)

    def test_runtime_scales_with_tokens(self):
        jobs = build_contended_jobs([60, 600], horizon_hours=48)
        assert jobs[1].runtime_hours > jobs[0].runtime_hours

    def test_base_jobs_have_no_prior(self):
        jobs = build_contended_jobs([90, 224], horizon_hours=24)
        assert all(j.predicted_output_tokens is None for j in jobs)


class TestSrtfPrior:
    def test_perfect_prior_equals_true_tokens(self):
        jobs = build_contended_jobs([90, 240], horizon_hours=24)
        primed = with_srtf_prior(jobs, noise_cv=0.0)
        # true tokens are stashed in data_transfer_gb
        for j in primed:
            assert j.predicted_output_tokens == pytest.approx(j.data_transfer_gb)

    def test_does_not_mutate_input(self):
        jobs = build_contended_jobs([90, 240], horizon_hours=24)
        with_srtf_prior(jobs, noise_cv=0.3)
        assert all(j.predicted_output_tokens is None for j in jobs)

    def test_noisy_prior_is_positive(self):
        jobs = build_contended_jobs([90, 240, 500], horizon_hours=24)
        primed = with_srtf_prior(jobs, noise_cv=0.5, seed=1)
        assert all(j.predicted_output_tokens >= 1.0 for j in primed)


class TestContentionNegativeFinding:
    def test_scenario_is_actually_contended(self):
        r = run_srtf_contention_backtest(horizon_hours=24, job_limit=200)
        # demand exceeds capacity — the cap is genuinely binding
        assert r.contention_ratio > 1.0
        assert r.concurrent_slots == int(REGION_POWER_CAP_KW // JOB_POWER_KW)

    def test_batch_scheduler_srtf_is_neutral(self):
        # The documented negative result: |goodput delta| is negligible because
        # the greedy batch path has no queue-wait semantics for ordering to move.
        r = run_srtf_contention_backtest(horizon_hours=24, job_limit=200)
        assert abs(r.goodput_delta_pct) < 1.0

    def test_report_serializes(self):
        r = run_srtf_contention_backtest(horizon_hours=24, job_limit=120)
        d = r.to_dict()
        assert isinstance(r, SRTFContentionReport)
        assert d["shadow_tag"].startswith("shadow_only")
        assert "fifo" in d and "srtf_perfect" in d
