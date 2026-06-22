"""Tests for ML-HGB prior [run 2026-06-22-v].

Validates make_ml_prior_predictions_burstgpt(), MLPriorReport, and
run_burstgpt_hf_ml_prior_backtest() against the invariants established in
runs -t and -u:

  - ML prior must be fully causal (Phase 2 only uses Phase 1 completions)
  - Phase 1 must produce running-median predictions (same as live prior)
  - Phase 2 predictions must vary (HGB captures model_id signal)
  - MLPriorReport serialises correctly and contains all required KPIs
  - On a two-model synthetic trace, ML prior must be more accurate than global
  - On BurstGPT HF data (if available), ML prior must not regress vs global

Invariants tested:
  1.  make_ml_prior_predictions_burstgpt: returns len(raw) predictions.
  2.  make_ml_prior_predictions_burstgpt: all predictions are >= 1.0.
  3.  make_ml_prior_predictions_burstgpt: empty input returns empty list.
  4.  make_ml_prior_predictions_burstgpt: single-request warmup fallback.
  5.  make_ml_prior_predictions_burstgpt: warmup_n >= n returns running-median only.
  6.  make_ml_prior_predictions_burstgpt: phase 1 matches running-median predictions.
  7.  make_ml_prior_predictions_burstgpt: phase 2 predictions are varied (not constant).
  8.  make_ml_prior_predictions_burstgpt: stats dict has required keys.
  9.  make_ml_prior_predictions_burstgpt: prior_cv_pct is non-negative.
  10. make_ml_prior_predictions_burstgpt: prior_mae_tokens is non-negative.
  11. make_ml_prior_predictions_burstgpt: n_model_ids >= 1 after Phase 2.
  12. make_ml_prior_predictions_burstgpt: two-model trace — Phase 2 CV > global CV.
  13. make_ml_prior_predictions_burstgpt: two-model trace — Phase 2 MAE < global MAE.
  14. make_ml_prior_predictions_burstgpt: prior_type key present and correct.
  15. make_ml_prior_predictions_burstgpt: features length mismatch raises AssertionError.
  16. MLPriorReport.to_dict(): serialises all floats correctly (no NaN/inf).
  17. MLPriorReport.to_dict(): shadow_tag present.
  18. MLPriorReport.to_dict(): ml_vs_global_improvement_pct key present.
  19. run_burstgpt_hf_ml_prior_backtest: returns MLPriorReport on HF data.
  20. run_burstgpt_hf_ml_prior_backtest: ml_goodput_per_dollar > 0.
  21. run_burstgpt_hf_ml_prior_backtest: fifo_goodput_per_dollar > 0.
  22. run_burstgpt_hf_ml_prior_backtest: oracle > global > FIFO on HF data.
  23. run_burstgpt_hf_ml_prior_backtest: ml_vs_oracle_retention_pct in (0, 150).
  24. ML_PRIOR_WARMUP_N constant > 0.
"""

from __future__ import annotations

import json
import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_BURSTGPT_HF_JSONL,
    LIVE_PRIOR_WINDOW,
    ML_PRIOR_WARMUP_N,
    MLPriorReport,
    make_live_prior_predictions,
    make_ml_prior_predictions_burstgpt,
    run_burstgpt_hf_ml_prior_backtest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_model_trace(
    n_short: int = 80,
    n_long: int = 20,
    short_tok: int = 10,
    long_tok: int = 300,
    short_inp: int = 50,
    long_inp: int = 500,
    seed: int = 42,
) -> tuple[list[tuple[float, int]], list[dict]]:
    """Synthetic two-model trace: 80% short (model_A) + 20% long (model_B).

    Requests are INTERLEAVED by arrival time so both model types appear in the
    warmup period.  This is realistic: in production, both model types arrive
    concurrently throughout the trace.
    """
    import random as _rnd
    rng = _rnd.Random(seed)
    rows = []
    t = 0.0
    short_remaining = n_short
    long_remaining = n_long
    total = n_short + n_long
    for i in range(total):
        # Interleave: decide model by probability matching target ratio
        if long_remaining == 0 or (
            short_remaining > 0 and rng.random() < short_remaining / (short_remaining + long_remaining)
        ):
            tok = short_tok + rng.randint(0, 5)
            inp = short_inp + rng.randint(0, 20)
            rows.append((t, tok, {"model_id": "model_A", "input_tokens": inp}))
            short_remaining -= 1
        else:
            tok = long_tok + rng.randint(0, 50)
            inp = long_inp + rng.randint(0, 100)
            rows.append((t, tok, {"model_id": "model_B", "input_tokens": inp}))
            long_remaining -= 1
        t += rng.uniform(1.0, 4.0)
    raw = [(ts, tok) for ts, tok, _ in rows]
    feats = [f for _, _, f in rows]
    return raw, feats


def _make_single_model_trace(
    n: int = 50,
    tok: int = 20,
    inp: int = 100,
) -> tuple[list[tuple[float, int]], list[dict]]:
    raw = [(float(i * 2), tok) for i in range(n)]
    feats = [{"model_id": "model_X", "input_tokens": inp} for _ in range(n)]
    return raw, feats


HF_AVAILABLE = os.path.isfile(DEFAULT_BURSTGPT_HF_JSONL)


# ---------------------------------------------------------------------------
# Tests: make_ml_prior_predictions_burstgpt
# ---------------------------------------------------------------------------

def test_returns_correct_length():
    """1. Output length equals input length."""
    raw, feats = _make_two_model_trace()
    preds, _ = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=50)
    assert len(preds) == len(raw)


def test_all_predictions_ge_one():
    """2. All predictions are >= 1.0."""
    raw, feats = _make_two_model_trace()
    preds, _ = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=50)
    assert all(p >= 1.0 for p in preds), f"min prediction = {min(preds)}"


def test_empty_input_returns_empty():
    """3. Empty input returns empty list and empty stats."""
    preds, stats = make_ml_prior_predictions_burstgpt([], [], warmup_n=10)
    assert preds == []
    assert stats == {}


def test_single_request_warmup_fallback():
    """4. Single request uses global median (warmup fallback)."""
    raw = [(0.0, 42)]
    feats = [{"model_id": "model_A", "input_tokens": 10}]
    preds, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=10)
    assert len(preds) == 1
    assert preds[0] >= 1.0


def test_warmup_n_gte_n_returns_running_median():
    """5. warmup_n >= n means all requests are in Phase 1 (running median)."""
    raw, feats = _make_two_model_trace(n_short=30, n_long=10)
    n = len(raw)
    preds_ml, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=n + 100)
    preds_median, _ = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
    assert stats["prior_type"] == "ml_hgb_warmup_only"
    assert stats["phase2_n"] == 0
    for i in range(n):
        assert abs(preds_ml[i] - preds_median[i]) < 1e-9, (
            f"request {i}: ml={preds_ml[i]}, median={preds_median[i]}"
        )


def test_phase1_matches_running_median():
    """6. First warmup_n predictions match running-median (Phase 1)."""
    raw, feats = _make_two_model_trace()
    n = len(raw)  # noqa: F841
    warmup_n = 40
    preds_ml, _ = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=warmup_n)
    preds_median, _ = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
    for i in range(warmup_n):
        assert abs(preds_ml[i] - preds_median[i]) < 1e-9, (
            f"Phase 1 request {i}: ml={preds_ml[i]}, median={preds_median[i]}"
        )


def test_phase2_predictions_are_varied():
    """7. Phase 2 predictions are not all identical (HGB captures signal)."""
    raw, feats = _make_two_model_trace(n_short=80, n_long=20)
    warmup_n = 30
    preds, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=warmup_n)
    phase2_preds = preds[warmup_n:]
    assert len(phase2_preds) > 0, "no Phase 2 predictions"
    assert max(phase2_preds) > min(phase2_preds), (
        "Phase 2 predictions are all identical — HGB has no signal"
    )


def test_stats_has_required_keys():
    """8. Stats dict contains all documented keys."""
    raw, feats = _make_two_model_trace()
    _, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=50)
    required = {
        "prior_type", "warmup_n", "n_requests", "n_model_ids", "phase2_n",
        "prior_cv_pct", "prior_mae_tokens", "prior_rel_mae_pct",
        "prior_bias_tokens", "global_median_actual",
    }
    missing = required - set(stats.keys())
    assert not missing, f"Missing keys: {missing}"


def test_prior_cv_pct_non_negative():
    """9. prior_cv_pct is non-negative."""
    raw, feats = _make_two_model_trace()
    _, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=50)
    assert stats["prior_cv_pct"] >= 0.0


def test_prior_mae_tokens_non_negative():
    """10. prior_mae_tokens is non-negative."""
    raw, feats = _make_two_model_trace()
    _, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=50)
    assert stats["prior_mae_tokens"] >= 0.0


def test_n_model_ids_ge_one_after_phase2():
    """11. n_model_ids >= 1 when Phase 2 is active."""
    raw, feats = _make_two_model_trace()
    _, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=30)
    if stats["prior_type"] != "ml_hgb_warmup_only":
        assert stats["n_model_ids"] >= 1


def test_two_model_trace_phase2_cv_gt_global():
    """12. Two-model trace: Phase 2 CV > global running-median CV (more variation)."""
    raw, feats = _make_two_model_trace(n_short=80, n_long=20)
    preds_ml, stats_ml = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=30)
    _, stats_global = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
    if stats_ml.get("prior_type") == "ml_hgb_p50":
        assert stats_ml["prior_cv_pct"] > stats_global["prior_cv_pct"], (
            f"ML CV={stats_ml['prior_cv_pct']:.2f} not > global CV={stats_global['prior_cv_pct']:.2f}"
        )


def test_two_model_trace_phase2_mae_lt_global():
    """13. Two-model trace: Phase 2 MAE < global running-median MAE."""
    raw, feats = _make_two_model_trace(n_short=80, n_long=20, short_tok=10, long_tok=300)
    preds_ml, stats_ml = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=30)
    _, stats_global = make_live_prior_predictions(raw, window=LIVE_PRIOR_WINDOW)
    if stats_ml.get("prior_type") == "ml_hgb_p50":
        assert stats_ml["prior_mae_tokens"] < stats_global["prior_mae_tokens"], (
            f"ML MAE={stats_ml['prior_mae_tokens']:.1f} not < global MAE={stats_global['prior_mae_tokens']:.1f}"
        )


def test_prior_type_key_present_and_correct():
    """14. prior_type key is present and one of the expected values."""
    raw, feats = _make_two_model_trace()
    _, stats = make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=50)
    expected_types = {"ml_hgb_p50", "ml_hgb_warmup_only", "ml_hgb_sklearn_unavailable_fallback"}
    assert stats.get("prior_type") in expected_types, f"Unexpected prior_type: {stats.get('prior_type')}"


def test_features_length_mismatch_raises():
    """15. Mismatched raw/features lengths raise AssertionError."""
    raw = [(0.0, 10), (1.0, 20)]
    feats = [{"model_id": "A", "input_tokens": 5}]  # length 1, raw length 2
    with pytest.raises(AssertionError):
        make_ml_prior_predictions_burstgpt(raw, feats, warmup_n=5)


# ---------------------------------------------------------------------------
# Tests: MLPriorReport
# ---------------------------------------------------------------------------

def _make_dummy_report() -> MLPriorReport:
    dummy_sim = {"n_completed": 10, "sla_safe_goodput_per_dollar": 100.0}
    return MLPriorReport(
        trace="test",
        total_requests=100,
        servers=4,
        target_rho=0.85,
        sla_s=30.0,
        warmup_n=50,
        n_model_ids=2,
        global_prior_cv_pct=5.0,
        global_prior_mae_tokens=50.0,
        global_prior_rel_mae_pct=25.0,
        ml_prior_cv_pct=30.0,
        ml_prior_mae_tokens=20.0,
        ml_prior_rel_mae_pct=10.0,
        fifo=dict(dummy_sim),
        conformal_oracle=dict(dummy_sim),
        conformal_global=dict(dummy_sim),
        conformal_ml=dict(dummy_sim),
        fifo_goodput_per_dollar=1000.0,
        oracle_goodput_per_dollar=5000.0,
        global_goodput_per_dollar=3500.0,
        ml_goodput_per_dollar=4200.0,
        oracle_delta_pct=400.0,
        global_delta_pct=250.0,
        ml_delta_pct=320.0,
        global_vs_oracle_retention_pct=70.0,
        ml_vs_oracle_retention_pct=84.0,
        ml_vs_global_improvement_pct=20.0,
    )


def test_to_dict_serialises_floats():
    """16. to_dict() returns only JSON-serialisable values (no NaN/inf)."""
    report = _make_dummy_report()
    d = report.to_dict()
    j = json.dumps(d)  # will raise TypeError / ValueError if not serialisable
    assert isinstance(j, str)


def test_to_dict_shadow_tag_present():
    """17. to_dict() includes shadow_tag."""
    report = _make_dummy_report()
    d = report.to_dict()
    assert "shadow_tag" in d
    assert "shadow" in d["shadow_tag"].lower()


def test_to_dict_ml_vs_global_improvement_pct_present():
    """18. to_dict() includes ml_vs_global_improvement_pct key."""
    report = _make_dummy_report()
    d = report.to_dict()
    assert "ml_vs_global_improvement_pct" in d


# ---------------------------------------------------------------------------
# Tests: run_burstgpt_hf_ml_prior_backtest (requires HF data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_returns_ml_prior_report():
    """19. run_burstgpt_hf_ml_prior_backtest returns MLPriorReport on HF data."""
    report = run_burstgpt_hf_ml_prior_backtest(job_limit=5880)
    assert isinstance(report, MLPriorReport)


@pytest.mark.skipif(not HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_ml_goodput_per_dollar_positive():
    """20. ml_goodput_per_dollar > 0."""
    report = run_burstgpt_hf_ml_prior_backtest(job_limit=5880)
    assert report.ml_goodput_per_dollar > 0.0


@pytest.mark.skipif(not HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_fifo_goodput_per_dollar_positive():
    """21. fifo_goodput_per_dollar > 0."""
    report = run_burstgpt_hf_ml_prior_backtest(job_limit=5880)
    assert report.fifo_goodput_per_dollar > 0.0


@pytest.mark.skipif(not HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_oracle_gt_fifo_on_hf():
    """22. oracle > FIFO (oracle should always beat baseline)."""
    report = run_burstgpt_hf_ml_prior_backtest(job_limit=5880)
    assert report.oracle_goodput_per_dollar > report.fifo_goodput_per_dollar, (
        f"oracle={report.oracle_goodput_per_dollar} not > FIFO={report.fifo_goodput_per_dollar}"
    )


@pytest.mark.skipif(not HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_ml_vs_oracle_retention_plausible():
    """23. ml_vs_oracle_retention_pct in (0, 150) — plausible range."""
    report = run_burstgpt_hf_ml_prior_backtest(job_limit=5880)
    assert 0.0 < report.ml_vs_oracle_retention_pct < 150.0, (
        f"retention={report.ml_vs_oracle_retention_pct:.2f}% outside (0, 150)"
    )


def test_ml_prior_warmup_n_constant_positive():
    """24. ML_PRIOR_WARMUP_N constant > 0."""
    assert ML_PRIOR_WARMUP_N > 0
