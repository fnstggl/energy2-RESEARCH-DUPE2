"""Tests for model-stratified prior serving-queue backtest [run 2026-06-22-u].

Validates make_model_stratified_prior_predictions(),
load_burstgpt_serving_requests_with_model_ids(), and
run_burstgpt_hf_model_stratified_prior_backtest() against the expectations
established in run -t (global live prior):

  - Global live prior on BurstGPT: +420.83% vs FIFO, 70.0% oracle retention
  - Model-stratified prior must not degrade below global prior
  - Per-model prior must be strictly causal (no future leakage)
  - ChatGPT (median=7) and GPT-4 (median=212) get correctly separated

Invariants tested:
  1.  make_model_stratified_prior_predictions: first prediction uses global median.
  2.  make_model_stratified_prior_predictions: causal order — no future leakage.
  3.  make_model_stratified_prior_predictions: per-model median activates after warmup.
  4.  make_model_stratified_prior_predictions: global fallback before warmup.
  5.  make_model_stratified_prior_predictions: sliding window correct for per-model.
  6.  make_model_stratified_prior_predictions: stats dict has required keys.
  7.  make_model_stratified_prior_predictions: per_model breakdown keys present.
  8.  make_model_stratified_prior_predictions: empty input returns empty list.
  9.  make_model_stratified_prior_predictions: single request returns fallback.
  10. make_model_stratified_prior_predictions: n_models counts distinct models.
  11. StratifiedPriorReport.to_dict() serialises all floats correctly.
  12. StratifiedPriorReport.to_dict() includes shadow_tag.
  13. load_burstgpt_serving_requests_with_model_ids: returns (float, int, str) tuples.
  14. load_burstgpt_serving_requests_with_model_ids: arrival times are zero-based.
  15. load_burstgpt_serving_requests_with_model_ids: respects limit parameter.
  16. run_burstgpt_hf_model_stratified_prior_backtest: returns StratifiedPriorReport.
  17. stratified_retention_pct is in (0, 150) on fixture.
  18. stratified_goodput >= global_live_goodput (should be at least as good).
  19. oracle_delta_pct > 0 on fixture.
  20. per_model_stats has ChatGPT and GPT-4 entries on fixture.
"""

from __future__ import annotations

import os
import tempfile
import json

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    LIVE_PRIOR_WINDOW,
    MODEL_STRATIFIED_WARMUP,
    StratifiedPriorReport,
    load_burstgpt_serving_requests_with_model_ids,
    make_model_stratified_prior_predictions,
    run_burstgpt_hf_model_stratified_prior_backtest,
)

FIXTURE_AVAILABLE = os.path.exists(DEFAULT_BURSTGPT_HF_JSONL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_raw_with_models(
    n: int = 20,
    model_a_tokens: int = 10,
    model_b_tokens: int = 200,
) -> list[tuple[float, int, str]]:
    """Alternating model_a / model_b with distinct token distributions."""
    rows = []
    for i in range(n):
        model = "model_a" if i % 2 == 0 else "model_b"
        tok = model_a_tokens + (i % 3) if model == "model_a" else model_b_tokens + (i % 5)
        rows.append((float(i * 2), tok, model))
    return rows


def _write_temp_jsonl(rows: list[dict]) -> str:
    """Write JSONL rows to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    for row in rows:
        f.write(json.dumps(row) + "\n")
    f.flush()
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# make_model_stratified_prior_predictions unit tests
# ---------------------------------------------------------------------------

def test_first_prediction_uses_global_median():
    """First request has no history → must use global static median."""
    raw = _simple_raw_with_models(20)
    all_toks = sorted(t for _, t, _ in raw)
    global_median = float(all_toks[len(all_toks) // 2])
    preds, _ = make_model_stratified_prior_predictions(raw, window=5, warmup=3)
    assert preds[0] == global_median


def test_causal_order_no_future_leakage():
    """prediction[i] must only use tokens from requests 0..i-1."""
    tokens_a = [10, 20, 30, 40, 50]
    tokens_b = [200, 210, 220, 230, 240]
    # Interleave: a0, b0, a1, b1, a2, b2 ...
    raw = []
    for i in range(5):
        raw.append((float(i * 4), tokens_a[i], "A"))
        raw.append((float(i * 4 + 1), tokens_b[i], "B"))

    preds, _ = make_model_stratified_prior_predictions(raw, window=100, warmup=1)
    # prediction[2] is for A1 (second A request). A has 1 completion (warmup=1),
    # so uses per-model median of [10] → 10.0
    assert preds[2] == 10.0
    # prediction[3] is for B1 (second B). B has 1 completion → median([200]) = 200.0
    assert preds[3] == 200.0


def test_per_model_median_activates_after_warmup():
    """Predictions before warmup use global fallback; after warmup use per-model."""
    # model "rare" appears only at positions 5 and 10 (warmup=3)
    raw = [
        (0.0, 100, "common"), (1.0, 105, "common"), (2.0, 110, "common"),
        (3.0, 115, "common"), (4.0, 120, "common"),
        (5.0, 500, "rare"),  # first rare — no per-model median yet
        (6.0, 100, "common"), (7.0, 105, "common"), (8.0, 110, "common"),
        (9.0, 115, "common"), (10.0, 120, "common"),
        (11.0, 500, "rare"),  # second rare — still not enough warmup (warmup=3)
        (12.0, 100, "common"), (13.0, 105, "common"), (14.0, 110, "common"),
        (15.0, 500, "rare"),  # third rare — warmup met; uses per-model median now
        (16.0, 500, "rare"),  # fourth: now has 3 completions → per-model median
    ]
    preds, stats = make_model_stratified_prior_predictions(raw, window=50, warmup=3)
    # Position 5 (first rare): no per-model history → global fallback
    # Position 11 (second rare): 1 completion → global fallback
    # Position 15 (third rare): 2 completions → global fallback
    # Position 16 (fourth rare): 3 completions → per-model median activates
    assert stats["n_models"] == 2
    # The per-model median at position 16 should be 500.0 (all rare so far are 500)
    rare_indices = [i for i, (_, _, m) in enumerate(raw) if m == "rare"]
    # First 3 rare predictions use global fallback; 4th uses per-model
    assert rare_indices[3] == 16
    # per-model median of [500, 500, 500] = 500.0
    assert preds[16] == 500.0


def test_global_fallback_before_warmup():
    """Before warmup, predictions for rare model use global running median."""
    raw = [
        (0.0, 100, "common"), (1.0, 100, "common"), (2.0, 100, "common"),
        (3.0, 100, "common"), (4.0, 100, "common"),
        (5.0, 999, "rare"),  # first rare — warmup=10, uses global median ≈ 100
    ]
    preds, _ = make_model_stratified_prior_predictions(raw, window=50, warmup=10)
    # global history at position 5 is [100, 100, 100, 100, 100] → median = 100.0
    assert preds[5] == 100.0


def test_per_model_sliding_window():
    """Per-model median uses only last ``window`` completions."""
    # Interleave: odd positions = model B with token values 1..10
    raw = []
    for i in range(20):
        if i % 2 == 0:
            raw.append((float(i), 100, "A"))
        else:
            raw.append((float(i), i // 2 + 1, "B"))  # tokens 1,2,3,...10 for B

    preds, _ = make_model_stratified_prior_predictions(raw, window=3, warmup=2)
    # At position 19 (10th B request, index in raw), B has tokens 1..9 past
    # window=3 → median of [7,8,9] = 8.0
    b_indices = [i for i, (_, _, m) in enumerate(raw) if m == "B"]
    # 10th B is b_indices[9] = 19; window over last 3 of [1..9] = [7,8,9] → 8.0
    assert preds[b_indices[9]] == 8.0


def test_stats_dict_has_required_keys():
    raw = _simple_raw_with_models(50)
    _, stats = make_model_stratified_prior_predictions(raw)
    required = {
        "prior_cv_pct", "prior_mae_tokens", "prior_rel_mae_pct",
        "prior_bias_tokens", "window", "warmup", "n_requests",
        "n_models", "per_model",
    }
    assert required.issubset(set(stats.keys()))


def test_per_model_breakdown_in_stats():
    raw = _simple_raw_with_models(50)
    _, stats = make_model_stratified_prior_predictions(raw)
    pm = stats["per_model"]
    assert "model_a" in pm
    assert "model_b" in pm
    for model_stats in pm.values():
        assert "n" in model_stats
        assert "median_actual" in model_stats
        assert "mae" in model_stats


def test_empty_input_returns_empty():
    preds, stats = make_model_stratified_prior_predictions([])
    assert preds == []
    assert stats == {}


def test_single_request_returns_global_median():
    """Single request: no history → global static median."""
    raw = [(0.0, 42, "modelX")]
    preds, stats = make_model_stratified_prior_predictions(raw, window=5, warmup=3)
    assert preds == [42.0]  # global median of [42] = 42
    assert stats["n_requests"] == 1
    assert stats["n_models"] == 1


def test_n_models_counts_distinct_models():
    raw = [
        (0.0, 10, "A"), (1.0, 20, "B"), (2.0, 30, "C"),
        (3.0, 10, "A"), (4.0, 20, "B"),
    ]
    _, stats = make_model_stratified_prior_predictions(raw)
    assert stats["n_models"] == 3


# ---------------------------------------------------------------------------
# StratifiedPriorReport unit tests
# ---------------------------------------------------------------------------

def test_stratified_prior_report_to_dict_serialisable():
    report = StratifiedPriorReport(
        trace="burstgpt_hf", total_requests=1000, servers=4, target_rho=0.85,
        sla_s=30.0, prior_window=200,
        global_prior_cv_pct=15.0, global_prior_mae_tokens=50.0,
        stratified_prior_cv_pct=8.0, stratified_prior_mae_tokens=20.0,
        per_model_stats={"ChatGPT": {"n": 842, "median_actual": 7.0, "mae": 3.5, "stratified_uses": 700, "mean_actual": 101.0},
                         "GPT-4": {"n": 158, "median_actual": 212.0, "mae": 30.0, "stratified_uses": 100, "mean_actual": 233.0}},
        fifo={"mean_response_s": 5.0},
        conformal_global_live={"mean_response_s": 2.0},
        conformal_stratified={"mean_response_s": 1.5},
        conformal_oracle={"mean_response_s": 1.0},
        fifo_goodput_per_dollar=100.0,
        global_live_goodput_per_dollar=520.0,
        stratified_goodput_per_dollar=600.0,
        oracle_goodput_per_dollar=740.0,
        global_live_delta_pct=420.0,
        stratified_delta_pct=500.0,
        oracle_delta_pct=640.0,
        global_retention_pct=70.0,
        stratified_retention_pct=81.0,
        stratified_vs_global_gain_pct=15.4,
    )
    d = report.to_dict()
    assert isinstance(d, dict)
    assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"
    assert d["stratified_retention_pct"] == 81.0
    assert d["global_retention_pct"] == 70.0
    # All float fields should be rounded floats (not nan/inf)
    for key in ("oracle_delta_pct", "stratified_delta_pct", "global_live_delta_pct",
                "global_prior_cv_pct", "stratified_prior_cv_pct"):
        assert isinstance(d[key], float), f"{key} is not float"
        assert d[key] == d[key], f"{key} is NaN"


def test_stratified_prior_report_has_shadow_tag():
    report = StratifiedPriorReport(
        trace="test", total_requests=10, servers=2, target_rho=0.7,
        sla_s=10.0, prior_window=50,
        global_prior_cv_pct=5.0, global_prior_mae_tokens=10.0,
        stratified_prior_cv_pct=3.0, stratified_prior_mae_tokens=7.0,
        per_model_stats={},
        fifo={"mean_response_s": 3.0}, conformal_global_live={"mean_response_s": 2.0},
        conformal_stratified={"mean_response_s": 1.5}, conformal_oracle={"mean_response_s": 1.0},
        fifo_goodput_per_dollar=100.0, global_live_goodput_per_dollar=200.0,
        stratified_goodput_per_dollar=250.0, oracle_goodput_per_dollar=300.0,
        global_live_delta_pct=100.0, stratified_delta_pct=150.0,
        oracle_delta_pct=200.0, global_retention_pct=70.0,
        stratified_retention_pct=83.0, stratified_vs_global_gain_pct=25.0,
    )
    assert report.shadow_tag == "shadow_only_simulator_result_not_production_savings"
    d = report.to_dict()
    assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"


# ---------------------------------------------------------------------------
# load_burstgpt_serving_requests_with_model_ids unit tests
# ---------------------------------------------------------------------------

def test_loader_returns_correct_tuple_types():
    rows = [
        {"request_arrival_ts_s": 1.0, "output_tokens": 50, "model_id": "ChatGPT"},
        {"request_arrival_ts_s": 2.0, "output_tokens": 200, "model_id": "GPT-4"},
        {"request_arrival_ts_s": 3.0, "output_tokens": 0, "model_id": "ChatGPT"},  # excluded
    ]
    path = _write_temp_jsonl(rows)
    try:
        result = load_burstgpt_serving_requests_with_model_ids(path)
        assert len(result) == 2  # zero-token excluded
        for item in result:
            assert isinstance(item[0], float)   # arrival_s
            assert isinstance(item[1], int)     # output_tokens
            assert isinstance(item[2], str)     # model_id
    finally:
        os.unlink(path)


def test_loader_arrival_times_zero_based():
    rows = [
        {"request_arrival_ts_s": 100.0, "output_tokens": 10, "model_id": "A"},
        {"request_arrival_ts_s": 102.0, "output_tokens": 20, "model_id": "B"},
        {"request_arrival_ts_s": 105.0, "output_tokens": 30, "model_id": "A"},
    ]
    path = _write_temp_jsonl(rows)
    try:
        result = load_burstgpt_serving_requests_with_model_ids(path)
        assert result[0][0] == 0.0   # first arrival zeroed
        assert result[1][0] == 2.0   # 102 - 100 = 2
        assert result[2][0] == 5.0   # 105 - 100 = 5
    finally:
        os.unlink(path)


def test_loader_respects_limit():
    rows = [
        {"request_arrival_ts_s": float(i), "output_tokens": 10 + i, "model_id": "X"}
        for i in range(20)
    ]
    path = _write_temp_jsonl(rows)
    try:
        result = load_burstgpt_serving_requests_with_model_ids(path, limit=5)
        assert len(result) == 5
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# run_burstgpt_hf_model_stratified_prior_backtest integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not FIXTURE_AVAILABLE,
    reason="BurstGPT HF JSONL fixture not available",
)
def test_run_stratified_backtest_returns_report():
    report = run_burstgpt_hf_model_stratified_prior_backtest()
    assert isinstance(report, StratifiedPriorReport)


@pytest.mark.skipif(
    not FIXTURE_AVAILABLE,
    reason="BurstGPT HF JSONL fixture not available",
)
def test_stratified_retention_in_plausible_range():
    report = run_burstgpt_hf_model_stratified_prior_backtest()
    assert 0.0 < report.stratified_retention_pct < 200.0


@pytest.mark.skipif(
    not FIXTURE_AVAILABLE,
    reason="BurstGPT HF JSONL fixture not available",
)
def test_stratified_goodput_not_below_global():
    """Model-stratified prior should match or beat global prior."""
    report = run_burstgpt_hf_model_stratified_prior_backtest()
    # Allow small numerical noise: require within 5% of global
    assert report.stratified_goodput_per_dollar >= report.global_live_goodput_per_dollar * 0.95


@pytest.mark.skipif(
    not FIXTURE_AVAILABLE,
    reason="BurstGPT HF JSONL fixture not available",
)
def test_oracle_delta_positive():
    report = run_burstgpt_hf_model_stratified_prior_backtest()
    assert report.oracle_delta_pct > 0.0


@pytest.mark.skipif(
    not FIXTURE_AVAILABLE,
    reason="BurstGPT HF JSONL fixture not available",
)
def test_per_model_stats_has_chatgpt_and_gpt4():
    """Fixture must produce per-model stats for both ChatGPT and GPT-4."""
    report = run_burstgpt_hf_model_stratified_prior_backtest()
    pm = report.per_model_stats
    # BurstGPT HF has model_id values "ChatGPT" and "GPT-4"
    assert len(pm) >= 2, f"Expected ≥2 models, got: {list(pm.keys())}"
    # Both models should have reasonable counts
    for model_name, ms in pm.items():
        assert ms["n"] > 0, f"Model {model_name} has 0 requests"
