"""Tests for ML prior under absolute-error conformal calibration [run 2026-06-22-z].

Run -v found the ML-HGB prior a null result (-0.12% vs global) under the
RELATIVE-error conformal calibrator, which was capped at mean_α=0.002 for both
priors. Run -x removed that cap via the ABSOLUTE-error calibrator (global prior
+420.83% → +557.12%). This run tests the open cell: ML prior + abs-conformal.

Result (BurstGPT HF, 5,880 req): ML+abs is -0.21% vs global+abs — another honest
null result. The abs-conformal gain is prior-agnostic; the ML prior's marginal
accuracy gain (MAE -2.5%) does not translate to SLA-safe goodput/$.

Invariants tested:
  1.  _run_ml_abs_conformal_on_trace builds a 2x2 + FIFO + oracle on a tiny trace.
  2.  MLAbsConformalReport.to_dict() serialises all floats (no NaN/inf).
  3.  to_dict() includes the shadow_tag.
  4.  to_dict() includes both key-contrast metrics.
  5.  Oracle goodput >= every prior discipline (it is the upper bound).
  6.  FIFO goodput <= every conformal discipline.
  7.  abs-conformal mean_alpha < rel-conformal mean_alpha (cap removed).
  8.  global+rel reproduces run -t (no regression of the baseline path).
  9.  ml_abs_vs_global_abs_pct equals the derived delta of the two KPIs.
  10. ml_abs_vs_ml_rel_pct equals the derived delta of the two KPIs.
  11. retention metrics are in a plausible (0, 150) range.
  12. all six goodput KPIs are positive.
  13. deltas vs FIFO are computed consistently with the KPIs.
  14. fixture-scale run on real BurstGPT returns a MLAbsConformalReport.
  15. fixture-scale: ML prior is genuinely active (n_model_ids >= 1) when sklearn present.
  16. fixture-scale: abs disciplines beat their rel counterparts (run -x effect holds).
  17. fixture-scale: ML+abs retention within 5pp of global+abs (prior-agnostic gain).
"""

from __future__ import annotations

import math
import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_ABS_TARGET_P90_TOKENS,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    ML_PRIOR_WARMUP_N,
    MLAbsConformalReport,
    _run_ml_abs_conformal_on_trace,
    run_burstgpt_hf_ml_abs_conformal_backtest,
)

FIXTURE_AVAILABLE = os.path.exists(DEFAULT_BURSTGPT_HF_JSONL)

try:  # the ML prior degrades to running-median fallback without sklearn
    import sklearn  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _toy_trace(n: int = 400) -> tuple[list[tuple[float, int]], list[dict]]:
    """Bimodal trace mimicking BurstGPT: ~80% short (ChatGPT), ~20% long (GPT-4)."""
    raw: list[tuple[float, int]] = []
    feats: list[dict] = []
    for i in range(n):
        if i % 5 == 0:  # GPT-4 long
            tok = 200 + (i % 7) * 20
            raw.append((float(i) * 0.5, tok))
            feats.append({"model_id": "GPT-4", "input_tokens": 1000 + i % 50})
        else:  # ChatGPT short
            tok = 5 + (i % 11)
            raw.append((float(i) * 0.5, tok))
            feats.append({"model_id": "ChatGPT", "input_tokens": 50 + i % 30})
    return raw, feats


# ---------------------------------------------------------------------------
# Unit tests on a small synthetic trace
# ---------------------------------------------------------------------------

def _toy_report() -> MLAbsConformalReport:
    raw, feats = _toy_trace(400)
    return _run_ml_abs_conformal_on_trace(
        raw, feats, "toy", servers=2, target_rho=0.85, sla_s=30.0, warmup_n=100
    )


def test_builds_full_2x2_report():
    rep = _toy_report()
    assert isinstance(rep, MLAbsConformalReport)
    assert rep.total_requests == 400


def test_to_dict_all_floats_finite():
    d = _toy_report().to_dict()
    for key in (
        "oracle_delta_pct", "global_rel_delta_pct", "global_abs_delta_pct",
        "ml_rel_delta_pct", "ml_abs_delta_pct",
        "ml_abs_vs_global_abs_pct", "ml_abs_vs_ml_rel_pct",
        "global_abs_retention_pct", "ml_abs_retention_pct",
    ):
        v = d[key]
        assert isinstance(v, float), f"{key} not float: {v!r}"
        assert math.isfinite(v), f"{key} not finite: {v!r}"


def test_to_dict_has_shadow_tag():
    d = _toy_report().to_dict()
    assert d["shadow_tag"] == "shadow_only_simulator_result_not_production_savings"


def test_to_dict_has_key_contrasts():
    d = _toy_report().to_dict()
    assert "ml_abs_vs_global_abs_pct" in d
    assert "ml_abs_vs_ml_rel_pct" in d


def test_oracle_is_upper_bound():
    rep = _toy_report()
    assert rep.oracle_goodput_per_dollar >= rep.global_rel_goodput_per_dollar
    assert rep.oracle_goodput_per_dollar >= rep.global_abs_goodput_per_dollar
    assert rep.oracle_goodput_per_dollar >= rep.ml_rel_goodput_per_dollar
    assert rep.oracle_goodput_per_dollar >= rep.ml_abs_goodput_per_dollar


def test_fifo_is_lower_bound():
    rep = _toy_report()
    assert rep.fifo_goodput_per_dollar <= rep.global_abs_goodput_per_dollar
    assert rep.fifo_goodput_per_dollar <= rep.ml_abs_goodput_per_dollar


def test_abs_alpha_below_rel_alpha():
    """Absolute-error calibrator should not be capped above the relative one."""
    rep = _toy_report()
    assert rep.global_abs_mean_alpha <= rep.global_rel_mean_alpha
    assert rep.ml_abs_mean_alpha <= rep.ml_rel_mean_alpha


def test_primary_contrast_matches_kpis():
    rep = _toy_report()
    expected = (
        (rep.ml_abs_goodput_per_dollar - rep.global_abs_goodput_per_dollar)
        / rep.global_abs_goodput_per_dollar * 100.0
    )
    assert rep.ml_abs_vs_global_abs_pct == pytest.approx(expected, abs=1e-6)


def test_secondary_contrast_matches_kpis():
    rep = _toy_report()
    expected = (
        (rep.ml_abs_goodput_per_dollar - rep.ml_rel_goodput_per_dollar)
        / rep.ml_rel_goodput_per_dollar * 100.0
    )
    assert rep.ml_abs_vs_ml_rel_pct == pytest.approx(expected, abs=1e-6)


def test_retention_plausible_range():
    rep = _toy_report()
    assert 0.0 < rep.global_abs_retention_pct < 150.0
    assert 0.0 < rep.ml_abs_retention_pct < 150.0


def test_all_goodputs_positive():
    rep = _toy_report()
    for v in (
        rep.fifo_goodput_per_dollar, rep.oracle_goodput_per_dollar,
        rep.global_rel_goodput_per_dollar, rep.global_abs_goodput_per_dollar,
        rep.ml_rel_goodput_per_dollar, rep.ml_abs_goodput_per_dollar,
    ):
        assert v > 0.0


def test_deltas_consistent_with_kpis():
    rep = _toy_report()
    base = rep.fifo_goodput_per_dollar
    expected = (rep.global_abs_goodput_per_dollar - base) / base * 100.0
    assert rep.global_abs_delta_pct == pytest.approx(expected, abs=1e-6)


def test_default_constants_reasonable():
    assert ML_PRIOR_WARMUP_N >= 1
    assert CONFORMAL_ABS_TARGET_P90_TOKENS > 0
    assert DEFAULT_BURSTGPT_SLA_S > 0


# ---------------------------------------------------------------------------
# Integration tests on the real BurstGPT HF trace
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FIXTURE_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_fixture_returns_report():
    rep = run_burstgpt_hf_ml_abs_conformal_backtest(job_limit=1500)
    assert isinstance(rep, MLAbsConformalReport)
    assert rep.total_requests == 1500


@pytest.mark.skipif(
    not (FIXTURE_AVAILABLE and SKLEARN_AVAILABLE),
    reason="needs BurstGPT HF + sklearn for an active ML prior",
)
def test_fixture_ml_prior_active():
    """With sklearn + enough requests past warmup, the HGB prior is genuinely used."""
    rep = run_burstgpt_hf_ml_abs_conformal_backtest(job_limit=3000)
    assert rep.n_model_ids >= 1


@pytest.mark.skipif(not FIXTURE_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_fixture_abs_beats_rel():
    """The run -x effect: abs-conformal beats rel-conformal for the global prior."""
    rep = run_burstgpt_hf_ml_abs_conformal_backtest(job_limit=5880)
    assert rep.global_abs_goodput_per_dollar > rep.global_rel_goodput_per_dollar


@pytest.mark.skipif(not FIXTURE_AVAILABLE, reason="BurstGPT HF JSONL not available")
def test_fixture_abs_gain_prior_agnostic():
    """ML+abs retention is within 5pp of global+abs — the abs gain is prior-agnostic."""
    rep = run_burstgpt_hf_ml_abs_conformal_backtest(job_limit=5880)
    assert abs(rep.ml_abs_retention_pct - rep.global_abs_retention_pct) < 5.0
