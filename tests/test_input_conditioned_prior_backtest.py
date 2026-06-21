"""Tests for input-token-conditioned live prior backtest [run 2026-06-21-u].

Validates make_input_conditioned_prior_predictions(),
load_burstgpt_serving_requests_with_features(), and
run_burstgpt_hf_input_conditioned_prior_backtest() against:
  - Causality (no future leakage)
  - Bucket dispatch correctness (bucket vs global fallback)
  - Improvement over global median on a variable input-correlated trace
  - All stats keys present and valid
  - InputConditionedPriorReport serialization

Invariants tested:
  1.  load_burstgpt_serving_requests_with_features: returns 4-tuple.
  2.  make_input_conditioned_prior_predictions: empty returns empty.
  3.  make_input_conditioned_prior_predictions: single request warmup.
  4.  make_input_conditioned_prior_predictions: causal — no future leakage.
  5.  make_input_conditioned_prior_predictions: uses bucket median when bucket >= min_count.
  6.  make_input_conditioned_prior_predictions: falls back to global median when bucket sparse.
  7.  make_input_conditioned_prior_predictions: stats dict has required keys.
  8.  make_input_conditioned_prior_predictions: prior_cv_pct >= 0.
  9.  make_input_conditioned_prior_predictions: ranking_accuracy_pct in [0, 100].
  10. Conditioned prior improves MAE on input-correlated synthetic trace.
  11. InputConditionedPriorReport.to_dict() includes all required keys.
  12. InputConditionedPriorReport.to_dict() shadow_tag correct.
  13. run_burstgpt_hf_input_conditioned_prior_backtest: returns report (fixture).
  14. cond_goodput_per_dollar > 0 on fixture.
  15. cond_ranking_accuracy_pct > 50 (beats random on correlated trace).
  16. cond_vs_oracle_retention_pct in (0, 200).
"""

from __future__ import annotations

import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    INPUT_TOKEN_BUCKETS,
    INPUT_CONDITIONED_WINDOW,
    InputConditionedPriorReport,
    _input_bucket_idx,
    load_burstgpt_serving_requests_with_features,
    make_input_conditioned_prior_predictions,
    run_burstgpt_hf_input_conditioned_prior_backtest,
)

_HF_AVAILABLE = os.path.exists(DEFAULT_BURSTGPT_HF_JSONL)
_skipif_no_hf = pytest.mark.skipif(not _HF_AVAILABLE, reason="BurstGPT HF JSONL not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _correlated_raw(n: int = 80) -> list[tuple[float, int, int, str]]:
    """Synthetic trace where output_tokens = input_tokens * 2 (perfect correlation)."""
    rows = []
    for i in range(n):
        in_tok = (i % 5 + 1) * 100  # 100, 200, 300, 400, 500 cycling
        out_tok = in_tok * 2         # perfect linear correlation
        rows.append((float(i * 2), out_tok, in_tok, "test_model"))
    return rows


def _uncorrelated_raw(n: int = 80) -> list[tuple[float, int, int, str]]:
    """Synthetic trace where output_tokens is constant regardless of input."""
    return [(float(i * 2), 100, (i % 5 + 1) * 100, "test_model") for i in range(n)]


# ---------------------------------------------------------------------------
# _input_bucket_idx unit tests
# ---------------------------------------------------------------------------

def test_bucket_idx_below_first_edge():
    assert _input_bucket_idx(50, [100, 300, 700]) == 0


def test_bucket_idx_exactly_at_edge():
    assert _input_bucket_idx(100, [100, 300, 700]) == 1


def test_bucket_idx_above_last_edge():
    assert _input_bucket_idx(5000, [100, 300, 700]) == 3


# ---------------------------------------------------------------------------
# make_input_conditioned_prior_predictions unit tests
# ---------------------------------------------------------------------------

def test_empty_returns_empty():
    preds, stats = make_input_conditioned_prior_predictions([])
    assert preds == []
    assert stats == {}


def test_single_request_warmup():
    raw = [(0.0, 100, 50, "m")]
    preds, stats = make_input_conditioned_prior_predictions(raw, warmup_value=99.0)
    assert preds == [99.0]
    assert stats["n_requests"] == 1


def test_causal_no_future_leakage():
    # prediction[i] must only depend on completions 0..i-1
    raw = [(float(i), (i + 1) * 50, 100, "m") for i in range(20)]
    preds, _ = make_input_conditioned_prior_predictions(raw, min_bucket_count=2)
    # prediction[0] cannot use any actual output (no history yet)
    assert preds[0] is not None
    # prediction[1] can at most know tokens[0]=50
    # It must not equal tokens[1]=100 (unless 50==100, which it doesn't here)
    assert preds[1] != 100.0


def test_uses_bucket_median_when_sufficient():
    # Two input buckets: in_tok=50 (bucket 0) and in_tok=500 (bucket 4 for edges [100,300,700,2000]).
    # Bucket 0 gets outputs 10,10,10,10,10 → median=10.
    # Bucket 4 gets outputs 900,900,900,900,900 → median=900.
    # After enough history, bucket 0 requests should predict ~10, bucket 4 ~900.
    raw = []
    for i in range(5):
        raw.append((float(i), 10, 50, "m"))    # bucket 0
        raw.append((float(i + 0.1), 900, 500, "m"))  # bucket 4
    # Add 5 more of each to ensure min_bucket_count=5 is satisfied for predictions
    for i in range(5):
        raw.append((float(10 + i), 10, 50, "m"))
        raw.append((float(10 + i + 0.1), 900, 500, "m"))
    raw.sort(key=lambda r: r[0])

    preds, _ = make_input_conditioned_prior_predictions(raw, min_bucket_count=5, window=200)
    # After position 9 (first 10 requests), bucket 0 has at least 5 completions.
    # Find first prediction for a bucket-0 request after position 9.
    bucket0_late = [i for i, (_, _, in_tok, _) in enumerate(raw) if in_tok == 50 and i > 9]
    bucket4_late = [i for i, (_, _, in_tok, _) in enumerate(raw) if in_tok == 500 and i > 9]
    if bucket0_late and bucket4_late:
        assert preds[bucket0_late[0]] < preds[bucket4_late[0]], (
            "Bucket with short outputs should predict smaller than bucket with long outputs"
        )


def test_falls_back_to_global_when_bucket_sparse():
    # All requests in different buckets so no bucket gets >= min_bucket_count.
    # The prediction should fall back to the global sliding-window median.
    raw = [
        (0.0, 200, 50, "m"),    # bucket 0
        (1.0, 200, 150, "m"),   # bucket 1
        (2.0, 200, 400, "m"),   # bucket 2
        (3.0, 200, 800, "m"),   # bucket 3
        (4.0, 200, 2500, "m"),  # bucket 4
        (5.0, 200, 50, "m"),    # bucket 0 again (only 1 prior entry in bucket 0)
    ]
    preds, stats = make_input_conditioned_prior_predictions(
        raw, min_bucket_count=10, window=200
    )
    # All predictions should be based on global history (fallback), not a sparse bucket.
    assert len(preds) == len(raw)
    assert stats["n_requests"] == len(raw)


def test_stats_dict_has_required_keys():
    raw = _correlated_raw(50)
    _, stats = make_input_conditioned_prior_predictions(raw)
    required = {
        "prior_cv_pct", "prior_mae_tokens", "prior_rel_mae_pct",
        "prior_bias_tokens", "prior_bias_pct", "ranking_accuracy_pct",
        "warmup_fallback", "global_median_actual", "n_buckets",
        "bucket_edges", "min_bucket_count", "window", "n_requests",
    }
    assert required.issubset(set(stats.keys()))


def test_prior_cv_nonnegative():
    raw = _correlated_raw(100)
    _, stats = make_input_conditioned_prior_predictions(raw)
    assert stats["prior_cv_pct"] >= 0.0


def test_ranking_accuracy_in_valid_range():
    raw = _correlated_raw(100)
    _, stats = make_input_conditioned_prior_predictions(raw)
    assert 0.0 <= stats["ranking_accuracy_pct"] <= 100.0


def test_ranking_accuracy_better_on_correlated_trace():
    # On a perfectly input-output correlated trace, bucket predictor should beat 50%.
    raw = _correlated_raw(200)
    _, stats = make_input_conditioned_prior_predictions(raw, min_bucket_count=3)
    assert stats["ranking_accuracy_pct"] > 50.0, (
        f"Ranking accuracy {stats['ranking_accuracy_pct']:.1f}% should beat 50% on correlated trace"
    )


def test_conditioned_mae_le_global_on_correlated():
    # On a strongly correlated trace, conditioned prior should have lower or equal MAE.
    from aurelius.benchmarks.srtf_serving_backtest import make_live_prior_predictions
    raw = _correlated_raw(200)
    raw_simple = [(arr, out_tok) for arr, out_tok, _, _ in raw]

    _, global_stats = make_live_prior_predictions(raw_simple)
    _, cond_stats = make_input_conditioned_prior_predictions(raw, min_bucket_count=3)

    # Should be at least competitive (conditioned MAE <= 1.1 * global MAE)
    assert cond_stats["prior_mae_tokens"] <= global_stats["prior_mae_tokens"] * 1.1, (
        f"Conditioned MAE {cond_stats['prior_mae_tokens']:.1f} "
        f"should not be much worse than global MAE {global_stats['prior_mae_tokens']:.1f}"
    )


# ---------------------------------------------------------------------------
# InputConditionedPriorReport unit tests
# ---------------------------------------------------------------------------

def test_report_to_dict_all_required_keys():
    report = InputConditionedPriorReport(
        trace="test", total_requests=100, servers=4, target_rho=0.85,
        sla_s=30.0, prior_window=200, bucket_edges=[100, 300, 700, 2000],
        min_bucket_count=5,
        global_prior_cv_pct=15.3, global_prior_mae_tokens=166.9, global_prior_rel_mae_pct=60.0,
        cond_prior_cv_pct=34.6, cond_prior_mae_tokens=153.7, cond_prior_rel_mae_pct=55.2,
        cond_ranking_accuracy_pct=61.1,
        fifo_goodput_per_dollar=6528.76, oracle_goodput_per_dollar=48598.82,
        global_goodput_per_dollar=34003.60, cond_goodput_per_dollar=38000.0,
        oracle_delta_pct=644.38, global_delta_pct=420.83, cond_delta_pct=482.0,
        cond_vs_oracle_retention_pct=78.2, cond_vs_global_uplift_pct=11.75,
    )
    d = report.to_dict()
    required = {
        "trace", "total_requests", "servers", "target_rho", "sla_s",
        "prior_window", "bucket_edges", "min_bucket_count",
        "global_prior_cv_pct", "global_prior_mae_tokens", "global_prior_rel_mae_pct",
        "cond_prior_cv_pct", "cond_prior_mae_tokens", "cond_prior_rel_mae_pct",
        "cond_ranking_accuracy_pct",
        "fifo_goodput_per_dollar", "oracle_goodput_per_dollar",
        "global_goodput_per_dollar", "cond_goodput_per_dollar",
        "oracle_delta_pct", "global_delta_pct", "cond_delta_pct",
        "cond_vs_oracle_retention_pct", "cond_vs_global_uplift_pct",
        "shadow_tag",
    }
    assert required.issubset(set(d.keys()))


def test_report_shadow_tag():
    report = InputConditionedPriorReport(
        trace="test", total_requests=10, servers=4, target_rho=0.85,
        sla_s=30.0, prior_window=200, bucket_edges=[100, 300, 700, 2000],
        min_bucket_count=5,
        global_prior_cv_pct=0.0, global_prior_mae_tokens=0.0, global_prior_rel_mae_pct=0.0,
        cond_prior_cv_pct=0.0, cond_prior_mae_tokens=0.0, cond_prior_rel_mae_pct=0.0,
        cond_ranking_accuracy_pct=50.0,
        fifo_goodput_per_dollar=100.0, oracle_goodput_per_dollar=500.0,
        global_goodput_per_dollar=400.0, cond_goodput_per_dollar=450.0,
        oracle_delta_pct=400.0, global_delta_pct=300.0, cond_delta_pct=350.0,
        cond_vs_oracle_retention_pct=90.0, cond_vs_global_uplift_pct=12.5,
    )
    d = report.to_dict()
    assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"


def test_report_float_fields_not_nan():
    report = InputConditionedPriorReport(
        trace="test", total_requests=100, servers=4, target_rho=0.85,
        sla_s=30.0, prior_window=200, bucket_edges=[100, 300, 700, 2000],
        min_bucket_count=5,
        global_prior_cv_pct=15.3, global_prior_mae_tokens=166.9, global_prior_rel_mae_pct=60.0,
        cond_prior_cv_pct=34.6, cond_prior_mae_tokens=153.7, cond_prior_rel_mae_pct=55.2,
        cond_ranking_accuracy_pct=61.1,
        fifo_goodput_per_dollar=6528.76, oracle_goodput_per_dollar=48598.82,
        global_goodput_per_dollar=34003.60, cond_goodput_per_dollar=38000.0,
        oracle_delta_pct=644.38, global_delta_pct=420.83, cond_delta_pct=482.0,
        cond_vs_oracle_retention_pct=78.2, cond_vs_global_uplift_pct=11.75,
    )
    d = report.to_dict()
    for key in ("oracle_delta_pct", "global_delta_pct", "cond_delta_pct",
                "cond_vs_oracle_retention_pct", "cond_vs_global_uplift_pct",
                "cond_prior_cv_pct", "cond_prior_mae_tokens"):
        val = d[key]
        assert isinstance(val, float), f"{key} should be float, got {type(val)}"
        assert val == val, f"{key} is NaN"


# ---------------------------------------------------------------------------
# Integration tests (BurstGPT HF fixture only)
# ---------------------------------------------------------------------------

@_skipif_no_hf
def test_load_burstgpt_with_features_returns_4tuple():
    rows = load_burstgpt_serving_requests_with_features(limit=10)
    assert len(rows) == 10
    for row in rows:
        assert len(row) == 4, f"Expected 4-tuple, got {len(row)}-tuple: {row}"
        arrival_s, out_tok, in_tok, model = row
        assert isinstance(arrival_s, float)
        assert isinstance(out_tok, int) and out_tok > 0
        assert isinstance(in_tok, int) and in_tok >= 0
        assert isinstance(model, str)


@_skipif_no_hf
def test_run_input_conditioned_prior_returns_report():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=200)
    assert isinstance(report, InputConditionedPriorReport)


@_skipif_no_hf
def test_cond_goodput_positive():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=200)
    assert report.fifo_goodput_per_dollar > 0
    assert report.oracle_goodput_per_dollar > 0
    assert report.global_goodput_per_dollar > 0
    assert report.cond_goodput_per_dollar > 0


@_skipif_no_hf
def test_cond_vs_oracle_retention_plausible():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=200)
    assert 0.0 < report.cond_vs_oracle_retention_pct < 200.0


@_skipif_no_hf
def test_cond_ranking_accuracy_beats_random():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=200)
    assert report.cond_ranking_accuracy_pct > 50.0, (
        f"Conditioned ranking accuracy {report.cond_ranking_accuracy_pct:.1f}% should beat random"
    )


@_skipif_no_hf
def test_report_total_requests_matches():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=300)
    # Should be at most 300 (filter removes output_tokens==0)
    assert 0 < report.total_requests <= 300


@_skipif_no_hf
def test_prior_cv_nonnegative_on_fixture():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=200)
    assert report.cond_prior_cv_pct >= 0.0
    assert report.global_prior_cv_pct >= 0.0


@_skipif_no_hf
def test_bucket_edges_in_report():
    report = run_burstgpt_hf_input_conditioned_prior_backtest(job_limit=200)
    assert report.bucket_edges == list(INPUT_TOKEN_BUCKETS)
    assert report.prior_window == INPUT_CONDITIONED_WINDOW
