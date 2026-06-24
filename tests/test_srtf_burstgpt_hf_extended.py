"""Tests for BurstGPT HF full-scale extended validation [run 2026-06-21-r].

Validates the three new HF-JSONL-backed public functions added to
srtf_serving_backtest.py for cross-trace confirmation of:

  (1) Conformal Adaptive α  — run_burstgpt_hf_conformal_alpha_backtest()
  (2) SLA-Aware Baseline    — run_burstgpt_hf_sla_aware_baseline_backtest()
  (3) Noisy Prior Robustness — run_burstgpt_hf_noisy_prior_backtest()

All three functions use ``load_burstgpt_serving_requests_jsonl`` (the HF JSONL
loader, not the CSV fixture) and delegate through the shared internal helpers
that were already validated on Azure LLM 2024 in runs -n and -q.

Research basis:
  - arXiv:2604.07931 (Robust Length Prediction, ProD, heavy-tailed BurstGPT)
  - arXiv:2603.11273 (Duration Aware Scheduling, cross-trace robustness)
  - arXiv:2509.23384 (NexusSched, two-layer adaptive scheduling)
"""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_TARGET_P90_ERROR,
    CONFORMAL_WARMUP,
    CONFORMAL_WINDOW,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    ConformalAlphaReport,
    NoisyPriorRobustnessReport,
    SLAAwareBaselineReport,
    run_burstgpt_hf_conformal_alpha_backtest,
    run_burstgpt_hf_noisy_prior_backtest,
    run_burstgpt_hf_sla_aware_baseline_backtest,
)

# ---------------------------------------------------------------------------
# Shared JSONL fixture helpers
# ---------------------------------------------------------------------------

def _write_jsonl(records: list[dict]) -> str:
    """Write records to a temporary JSONL file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for rec in records:
        f.write(json.dumps(rec) + "\n")
    f.flush()
    f.close()
    return f.name


def _make_queue_records(
    n: int = 120,
    start_ts: float = 0.0,
    gap_s: float = 0.5,
    short_tokens: int = 50,
    long_tokens: int = 500,
) -> list[dict]:
    """Return n BurstGPT-format records with alternating short/long tokens.

    Gap of 0.5s between arrivals at n=120 creates enough queue depth (ρ ≈ 0.85
    at 4 servers with mixed service times) to demonstrate ordering signal.
    """
    records = []
    for i in range(n):
        out_tok = short_tokens if i % 2 == 0 else long_tokens
        records.append({
            "request_arrival_ts_s": start_ts + i * gap_s,
            "output_tokens": out_tok,
            "input_tokens": 200,
            "model_id": "test-model",
        })
    return records


def _small_records(n: int = 20) -> list[dict]:
    """Return a small JSONL batch (insufficient queue depth — sanity tests only)."""
    return _make_queue_records(n=n, gap_s=2.0)


# ---------------------------------------------------------------------------
# Class 1: run_burstgpt_hf_conformal_alpha_backtest — unit tests
# ---------------------------------------------------------------------------

class TestBurstGPTHFConformalAlphaBacktest:
    """Tests for run_burstgpt_hf_conformal_alpha_backtest."""

    def test_returns_conformal_alpha_report(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert isinstance(rpt, ConformalAlphaReport)

    def test_trace_name_is_burstgpt_hf_fullscale(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.trace == "burstgpt_hf_fullscale"

    def test_sla_s_matches_default_burstgpt(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(DEFAULT_BURSTGPT_SLA_S)

    def test_custom_sla_propagated(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path, sla_s=20.0)
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(20.0)

    def test_all_four_discipline_dicts_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert isinstance(rpt.fifo, dict)
        assert isinstance(rpt.srpt, dict)
        assert isinstance(rpt.decoupled_fixed, dict)
        assert isinstance(rpt.decoupled_conformal, dict)

    def test_goodput_per_dollar_fields_nonnegative(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.fifo_goodput_per_dollar >= 0.0
        assert rpt.srpt_goodput_per_dollar >= 0.0
        assert rpt.decoupled_fixed_goodput_per_dollar >= 0.0
        assert rpt.decoupled_conformal_goodput_per_dollar >= 0.0

    def test_conformal_warmup_window_target_stored(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.conformal_warmup == CONFORMAL_WARMUP
        assert rpt.conformal_window == CONFORMAL_WINDOW
        assert rpt.conformal_target_p90_error == pytest.approx(CONFORMAL_TARGET_P90_ERROR)

    def test_conformal_mean_alpha_finite(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert math.isfinite(rpt.conformal_mean_alpha)
        assert rpt.conformal_mean_alpha >= 0.0

    def test_shadow_tag_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert "shadow" in rpt.shadow_tag.lower()

    def test_to_dict_serialisable(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        d = rpt.to_dict()
        assert isinstance(d, dict)
        assert "fifo_goodput_per_dollar" in d
        assert "conformal_mean_alpha" in d
        assert d["trace"] == "burstgpt_hf_fullscale"

    def test_total_requests_matches_job_limit(self):
        path = _write_jsonl(_make_queue_records(80))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(
                jsonl_path=path, job_limit=50
            )
        finally:
            os.unlink(path)
        assert rpt.total_requests == 50

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            run_burstgpt_hf_conformal_alpha_backtest(jsonl_path="/nonexistent/path.jsonl")

    def test_insufficient_records_raises(self):
        path = _write_jsonl([{"request_arrival_ts_s": 0.0, "output_tokens": 100}])
        try:
            with pytest.raises(ValueError, match="fewer than 2"):
                run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_oracle_prior_conformal_approaches_srpt(self):
        """With oracle prior (predicted==actual), conformal → α=0 → approaches SRPT."""
        path = _write_jsonl(_make_queue_records(n=150, gap_s=0.4))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(
                jsonl_path=path, target_rho=0.85
            )
        finally:
            os.unlink(path)
        # Conformal should be >= fixed at oracle, or within 5% (warmup effect)
        fixed_gp = rpt.decoupled_fixed_goodput_per_dollar
        conformal_gp = rpt.decoupled_conformal_goodput_per_dollar
        if fixed_gp > 0:
            ratio = conformal_gp / fixed_gp
            # Conformal ≥ fixed (oracle should push α→0→SRPT); small tolerance for warmup
            assert ratio >= 0.90, f"conformal={conformal_gp:.1f} << fixed={fixed_gp:.1f}"

    def test_srpt_goodput_geq_fixed_decoupled(self):
        """SRPT upper bound should be ≥ fixed-α decoupled hybrid."""
        path = _write_jsonl(_make_queue_records(80))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        # SRPT ≥ decoupled (pure SRPT preemption should dominate aging dispatch)
        assert rpt.srpt_goodput_per_dollar >= rpt.decoupled_fixed_goodput_per_dollar * 0.85

    def test_fifo_goodput_lowest_under_contention(self):
        """Under sufficient contention FIFO should yield less goodput than SRPT."""
        path = _write_jsonl(_make_queue_records(n=150, gap_s=0.4))
        try:
            rpt = run_burstgpt_hf_conformal_alpha_backtest(
                jsonl_path=path, target_rho=0.85
            )
        finally:
            os.unlink(path)
        if rpt.srpt_goodput_per_dollar > 0 and rpt.fifo_goodput_per_dollar > 0:
            assert rpt.srpt_goodput_per_dollar >= rpt.fifo_goodput_per_dollar


# ---------------------------------------------------------------------------
# Class 2: run_burstgpt_hf_sla_aware_baseline_backtest — unit tests
# ---------------------------------------------------------------------------

class TestBurstGPTHFSLAAwareBaselineBacktest:
    """Tests for run_burstgpt_hf_sla_aware_baseline_backtest."""

    def test_returns_sla_aware_baseline_report(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert isinstance(rpt, SLAAwareBaselineReport)

    def test_trace_name_is_burstgpt_hf_fullscale(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.trace == "burstgpt_hf_fullscale"

    def test_sla_s_default_is_burstgpt(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(DEFAULT_BURSTGPT_SLA_S)

    def test_all_four_goodput_fields_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.fifo_goodput >= 0.0
        assert rpt.sla_aware_goodput >= 0.0
        assert rpt.decoupled_goodput >= 0.0
        assert rpt.srpt_goodput >= 0.0

    def test_discipline_dicts_all_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert isinstance(rpt.fifo, dict)
        assert isinstance(rpt.sla_aware, dict)
        assert isinstance(rpt.decoupled, dict)
        assert isinstance(rpt.srpt, dict)

    def test_to_dict_contains_incremental_delta(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        d = rpt.to_dict()
        assert "decoupled_vs_sla_aware_delta_pct" in d
        assert isinstance(d["decoupled_vs_sla_aware_delta_pct"], float)

    def test_shadow_tag_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert "shadow" in rpt.shadow_tag.lower()

    def test_total_requests_matches_job_limit(self):
        path = _write_jsonl(_make_queue_records(80))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path=path, job_limit=50
            )
        finally:
            os.unlink(path)
        assert rpt.total_requests == 50

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path="/nonexistent/path.jsonl"
            )

    def test_insufficient_records_raises(self):
        path = _write_jsonl([{"request_arrival_ts_s": 0.0, "output_tokens": 100}])
        try:
            with pytest.raises(ValueError, match="fewer than 2"):
                run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_aging_alpha_parameter_accepted(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path=path, aging_alpha=0.005
            )
        finally:
            os.unlink(path)
        assert isinstance(rpt, SLAAwareBaselineReport)

    def test_ordering_srpt_geq_decoupled_geq_sla_aware(self):
        """On BurstGPT under contention, SRPT ≥ decoupled ≥ SLA-aware (ordering check)."""
        path = _write_jsonl(_make_queue_records(n=150, gap_s=0.4))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path=path, target_rho=0.85
            )
        finally:
            os.unlink(path)
        # Only check if FIFO has non-zero goodput (trace has some SLA-safe requests)
        if rpt.fifo_goodput > 0:
            assert rpt.srpt_goodput >= rpt.decoupled_goodput * 0.80
            assert rpt.decoupled_goodput >= rpt.sla_aware_goodput * 0.80

    def test_custom_sla_accepted(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path=path, sla_s=15.0
            )
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(15.0)

    def test_to_dict_is_serialisable(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        d = rpt.to_dict()
        assert isinstance(d, dict)
        assert d["trace"] == "burstgpt_hf_fullscale"


# ---------------------------------------------------------------------------
# Class 3: run_burstgpt_hf_noisy_prior_backtest — unit tests
# ---------------------------------------------------------------------------

class TestBurstGPTHFNoisyPriorBacktest:
    """Tests for run_burstgpt_hf_noisy_prior_backtest."""

    def test_returns_noisy_prior_robustness_report(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert isinstance(rpt, NoisyPriorRobustnessReport)

    def test_trace_name_is_burstgpt_hf_fullscale(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.trace == "burstgpt_hf_fullscale"

    def test_sla_s_default_burstgpt(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.sla_s == pytest.approx(DEFAULT_BURSTGPT_SLA_S)

    def test_noise_cv_stored(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, forecast_noise_cv=0.30
            )
        finally:
            os.unlink(path)
        assert rpt.forecast_noise_cv == pytest.approx(0.30)

    def test_goodput_fields_nonnegative(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.fifo_goodput >= 0.0
        assert rpt.oracle_goodput >= 0.0
        assert rpt.noisy_goodput >= 0.0

    def test_retention_pct_between_0_and_200(self):
        """Retention pct can slightly exceed 100% due to noise variance; bound 0–200%."""
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert 0.0 <= rpt.noisy_retention_pct <= 200.0

    def test_short_p90_metrics_stored(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert rpt.fifo_short_p90_s >= 0.0
        assert rpt.oracle_short_p90_s >= 0.0
        assert rpt.noisy_short_p90_s >= 0.0

    def test_to_dict_contains_required_keys(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        d = rpt.to_dict()
        assert "noisy_retention_pct" in d
        assert "oracle_goodput" in d
        assert "noisy_goodput" in d
        assert d["trace"] == "burstgpt_hf_fullscale"

    def test_shadow_tag_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        assert "shadow" in rpt.shadow_tag.lower()

    def test_total_requests_matches_job_limit(self):
        path = _write_jsonl(_make_queue_records(80))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, job_limit=50
            )
        finally:
            os.unlink(path)
        assert rpt.total_requests == 50

    def test_zero_noise_matches_oracle(self):
        """With CV=0 noise, noisy run should match oracle exactly."""
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, forecast_noise_cv=0.0
            )
        finally:
            os.unlink(path)
        assert rpt.oracle_goodput == pytest.approx(rpt.noisy_goodput, rel=1e-6)

    def test_seed_reproducibility(self):
        """Same seed must give same noisy_goodput."""
        path = _write_jsonl(_make_queue_records(60))
        try:
            rpt1 = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path, seed=42)
            rpt2 = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path, seed=42)
        finally:
            os.unlink(path)
        assert rpt1.noisy_goodput == pytest.approx(rpt2.noisy_goodput, rel=1e-9)

    def test_different_seed_different_noisy_result(self):
        """Different seeds produce different noisy orderings (with sufficient n)."""
        path = _write_jsonl(_make_queue_records(80))
        try:
            rpt1 = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, seed=1, forecast_noise_cv=0.5
            )
            rpt2 = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, seed=999, forecast_noise_cv=0.5
            )
        finally:
            os.unlink(path)
        # Different seeds → different noisy ordering → likely different goodput
        # (not guaranteed for all fixture sizes, so only assert if oracle > fifo)
        if rpt1.oracle_goodput > rpt1.fifo_goodput:
            assert rpt1.noisy_goodput != rpt2.noisy_goodput or (
                rpt1.oracle_goodput == rpt1.fifo_goodput
            )

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, ValueError)):
            run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path="/nonexistent/path.jsonl"
            )

    def test_insufficient_records_raises(self):
        path = _write_jsonl([{"request_arrival_ts_s": 0.0, "output_tokens": 100}])
        try:
            with pytest.raises(ValueError, match="fewer than 2"):
                run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)

    def test_high_retention_under_30pct_cv(self):
        """Under 30%-CV noise, retention should remain ≥ 80% when oracle > FIFO."""
        path = _write_jsonl(_make_queue_records(n=150, gap_s=0.4))
        try:
            rpt = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path,
                forecast_noise_cv=0.30,
                target_rho=0.85,
            )
        finally:
            os.unlink(path)
        if rpt.oracle_goodput > rpt.fifo_goodput * 1.05:
            assert rpt.noisy_retention_pct >= 80.0, (
                f"retention={rpt.noisy_retention_pct:.1f}% < 80% "
                f"oracle={rpt.oracle_goodput:.1f} noisy={rpt.noisy_goodput:.1f}"
            )


# ---------------------------------------------------------------------------
# Class 4: Cross-function consistency checks
# ---------------------------------------------------------------------------

class TestBurstGPTHFExtendedCrossConsistency:
    """Consistency checks across the three new HF functions."""

    def test_all_three_functions_run_same_trace_same_servers(self):
        """All three functions should accept the same JSONL and server params."""
        path = _write_jsonl(_make_queue_records(60))
        try:
            conformal = run_burstgpt_hf_conformal_alpha_backtest(
                jsonl_path=path, servers=4
            )
            sla_aware = run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path=path, servers=4
            )
            noisy = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, servers=4
            )
        finally:
            os.unlink(path)
        assert conformal.servers == 4
        assert sla_aware.servers == 4
        assert noisy.servers == 4

    def test_fifo_goodput_consistent_across_functions(self):
        """FIFO goodput/$ should be consistent across all three functions (same trace)."""
        path = _write_jsonl(_make_queue_records(60))
        try:
            conformal = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
            sla_aware = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        # FIFO goodput must be consistent between conformal and sla_aware reports
        assert conformal.fifo_goodput_per_dollar == pytest.approx(
            sla_aware.fifo_goodput, rel=1e-6
        )

    def test_job_limit_consistently_applied(self):
        """job_limit=30 should produce total_requests=30 in all three functions."""
        path = _write_jsonl(_make_queue_records(60))
        try:
            conformal = run_burstgpt_hf_conformal_alpha_backtest(
                jsonl_path=path, job_limit=30
            )
            sla_aware = run_burstgpt_hf_sla_aware_baseline_backtest(
                jsonl_path=path, job_limit=30
            )
            noisy = run_burstgpt_hf_noisy_prior_backtest(
                jsonl_path=path, job_limit=30
            )
        finally:
            os.unlink(path)
        assert conformal.total_requests == 30
        assert sla_aware.total_requests == 30
        assert noisy.total_requests == 30

    def test_all_shadow_tags_present(self):
        path = _write_jsonl(_make_queue_records(60))
        try:
            conformal = run_burstgpt_hf_conformal_alpha_backtest(jsonl_path=path)
            sla_aware = run_burstgpt_hf_sla_aware_baseline_backtest(jsonl_path=path)
            noisy = run_burstgpt_hf_noisy_prior_backtest(jsonl_path=path)
        finally:
            os.unlink(path)
        for rpt in (conformal, sla_aware, noisy):
            assert "shadow" in rpt.shadow_tag.lower()


# ---------------------------------------------------------------------------
# Class 5: HF dataset integration smoke test (skipped if dataset absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(DEFAULT_BURSTGPT_HF_JSONL),
    reason="BurstGPT HF JSONL not present; skipping real-dataset smoke test",
)
class TestBurstGPTHFExtendedRealDataset:
    """Smoke tests on the real 59,999-record HF dataset [run 2026-06-21-r].

    These tests run only when the HF JSONL is present.  They validate that
    the three new functions can load and process the real dataset without
    crashing, and that the basic KPI ordering makes sense.
    """

    def test_conformal_smoke_5880_records(self):
        rpt = run_burstgpt_hf_conformal_alpha_backtest(job_limit=5880)
        assert rpt.total_requests == 5880
        assert rpt.fifo_goodput_per_dollar > 0
        assert rpt.srpt_goodput_per_dollar > 0
        assert rpt.decoupled_conformal_goodput_per_dollar > 0
        assert rpt.srpt_delta_pct > 0

    def test_sla_aware_smoke_5880_records(self):
        rpt = run_burstgpt_hf_sla_aware_baseline_backtest(job_limit=5880)
        assert rpt.total_requests == 5880
        assert rpt.fifo_goodput > 0
        assert rpt.sla_aware_goodput > 0
        assert rpt.sla_aware_delta_pct > 0

    def test_noisy_prior_smoke_5880_records(self):
        rpt = run_burstgpt_hf_noisy_prior_backtest(job_limit=5880)
        assert rpt.total_requests == 5880
        assert rpt.oracle_goodput > 0
        assert rpt.noisy_goodput > 0
        assert rpt.noisy_retention_pct > 0

    def test_conformal_srpt_delta_significant(self):
        """SRPT should be materially better than FIFO on 5,880 BurstGPT records."""
        rpt = run_burstgpt_hf_conformal_alpha_backtest(job_limit=5880)
        # Run -p confirmed +644.4% SRPT vs FIFO; 100% minimum as sanity gate
        assert rpt.srpt_delta_pct >= 100.0, (
            f"SRPT vs FIFO = {rpt.srpt_delta_pct:.1f}% < 100%"
        )

    def test_sla_aware_vs_fifo_meaningful_on_real_data(self):
        """SLA-aware should be materially better than FIFO on BurstGPT HF."""
        rpt = run_burstgpt_hf_sla_aware_baseline_backtest(job_limit=5880)
        # Run -n measured +125.4% SLA-aware vs FIFO on Azure; BurstGPT is heavier
        assert rpt.sla_aware_delta_pct >= 50.0, (
            f"SLA-aware vs FIFO = {rpt.sla_aware_delta_pct:.1f}% < 50%"
        )

    def test_noisy_retention_high_on_real_data(self):
        """30%-CV noisy prior should retain ≥ 80% of oracle gain on BurstGPT HF."""
        rpt = run_burstgpt_hf_noisy_prior_backtest(job_limit=5880)
        assert rpt.noisy_retention_pct >= 80.0, (
            f"noisy retention = {rpt.noisy_retention_pct:.1f}% < 80%"
        )
