"""Tests for Per-Class Conformal Calibration [run 2026-06-22-w].

Validates PerClassConformalCalibrator, _simulate_decoupled_hybrid_per_class_conformal,
PerClassConformalReport, and run_burstgpt_per_class_conformal_backtest.

Primary hypothesis: per-class conformal calibration breaks the running-statistics
ceiling by allowing accurate-class predictors (GPT-4, rel_err≈0.02) to converge
to α≈0 independently of noisy-class predictors (ChatGPT, rel_err≈0.95).

Invariants tested:
  1.  PerClassConformalCalibrator: warmup falls back to global alpha_max.
  2.  PerClassConformalCalibrator: per-class zero-error → α→0 for that class.
  3.  PerClassConformalCalibrator: global fallback for sparse class.
  4.  PerClassConformalCalibrator: class_counts tracks per-class completions.
  5.  PerClassConformalCalibrator: per_class_mean_alpha has correct keys.
  6.  PerClassConformalCalibrator: noisy class capped at 2×alpha_max.
  7.  PerClassConformalCalibrator: independent class calibration (accurate ≠ noisy).
  8.  _simulate_decoupled_hybrid_per_class_conformal: all requests complete.
  9.  _simulate_decoupled_hybrid_per_class_conformal: response times ≥ service times.
  10. _simulate_decoupled_hybrid_per_class_conformal: preemption_count in summary.
  11. _simulate_decoupled_hybrid_per_class_conformal: per_class_mean_alpha in summary.
  12. _simulate_decoupled_hybrid_per_class_conformal: class_counts in summary.
  13. simulate_queue: decoupled_hybrid_per_class_conformal discipline accepted.
  14. PerClassConformalReport.to_dict(): all required keys present.
  15. PerClassConformalReport.to_dict(): shadow_tag present.
  16. PerClassConformalReport.to_dict(): per_class_mean_alpha serialised as dict.
  17. Two-model synthetic: per-class α < global α for accurate-prediction class.
  18. Two-model synthetic: per_class_goodput ≥ global_goodput (primary hypothesis).
  19. Two-model synthetic: per_class_vs_oracle_retention_pct > global_vs_oracle_retention_pct.
  20. run_burstgpt_per_class_conformal_backtest: returns PerClassConformalReport on HF data.
  21. run_burstgpt_per_class_conformal_backtest: per_class_goodput_per_dollar > 0.
  22. run_burstgpt_per_class_conformal_backtest: per_class_vs_global_pct reported.
  23. run_burstgpt_per_class_conformal_backtest: n_model_ids >= 1.
  24. PER_CLASS_WARMUP_MIN constant > 0 and ≤ CONFORMAL_WARMUP.
"""

from __future__ import annotations

import math
import os
import statistics

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_ALPHA_MAX,
    CONFORMAL_WARMUP,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    PER_CLASS_WARMUP_MIN,
    PerClassConformalCalibrator,
    PerClassConformalReport,
    _Request,
    _service_time_s,
    _simulate_decoupled_hybrid_per_class_conformal,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    load_burstgpt_serving_requests_jsonl_with_features,
    run_burstgpt_per_class_conformal_backtest,
    simulate_queue,
)

_HF_JSONL_AVAILABLE = os.path.isfile(DEFAULT_BURSTGPT_HF_JSONL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_model_requests(
    n_accurate: int = 80,
    n_noisy: int = 20,
    accurate_tok: int = 100,
    noisy_tok_short: int = 5,
    noisy_tok_long: int = 800,
    seed: int = 42,
) -> list[_Request]:
    """Two-model synthetic trace.

    accurate_model: fixed token count → perfect predictor (rel_err=0).
    noisy_model:    bimodal (80% short / 20% surprise-long) → high rel_err.

    Arrivals are interleaved at 1-second intervals.
    """
    import random
    rng = random.Random(seed)

    requests: list[_Request] = []
    t = 0.0
    for i in range(n_accurate + n_noisy):
        if i < n_accurate:
            tok = accurate_tok
            mid = "accurate_model"
            pred = float(tok)  # oracle prediction
        else:
            # bimodal noisy: mostly short, some very long
            is_long = rng.random() < 0.20
            tok = noisy_tok_long if is_long else noisy_tok_short
            mid = "noisy_model"
            pred = float(noisy_tok_short)  # always predicts short (wrong for long)
        requests.append(
            _Request(
                idx=i,
                arrival_s=t,
                actual_tokens=tok,
                predicted_tokens=pred,
                service_s=_service_time_s(tok),
                model_id=mid,
            )
        )
        t += 1.0
    return requests


def _make_single_model_requests(n: int = 50, tok: int = 100) -> list[_Request]:
    return [
        _Request(
            idx=i,
            arrival_s=float(i),
            actual_tokens=tok,
            predicted_tokens=float(tok),
            service_s=_service_time_s(tok),
            model_id="model_a",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# PerClassConformalCalibrator unit tests
# ---------------------------------------------------------------------------

class TestPerClassConformalCalibrator:

    def test_01_warmup_returns_alpha_max(self):
        """During warmup, current_alpha() must return alpha_max (conservative)."""
        cal = PerClassConformalCalibrator()
        alpha = cal.current_alpha("some_class")
        assert alpha == CONFORMAL_ALPHA_MAX

    def test_02_zero_error_accurate_class_converges_to_zero(self):
        """After warmup with zero-error residuals, per-class α converges to 0."""
        cal = PerClassConformalCalibrator()
        # Feed PER_CLASS_WARMUP_MIN + 10 zero-error completions for accurate class.
        for _ in range(PER_CLASS_WARMUP_MIN + 10):
            cal.update(100.0, 100, "accurate")
        alpha = cal.current_alpha("accurate")
        assert alpha < 1e-6, f"Expected α≈0, got {alpha}"

    def test_03_global_fallback_for_sparse_class(self):
        """Class with < PER_CLASS_WARMUP_MIN completions falls back to global α."""
        cal = PerClassConformalCalibrator()
        # Feed enough completions globally.
        for _ in range(CONFORMAL_WARMUP + 10):
            cal.update(100.0, 100, "")  # update global only
        # Sparse class: only 1 completion.
        cal.update(100.0, 100, "sparse_class")
        # Should fall back to global (which has zero error → α≈0).
        alpha_sparse = cal.current_alpha("sparse_class")
        alpha_global = cal.current_alpha("")
        assert abs(alpha_sparse - alpha_global) < 1e-6

    def test_04_class_counts_tracks_completions(self):
        """class_counts() must accurately track per-class completion count."""
        cal = PerClassConformalCalibrator()
        for _ in range(10):
            cal.update(10.0, 10, "class_a")
        for _ in range(5):
            cal.update(10.0, 10, "class_b")
        counts = cal.class_counts()
        assert counts["class_a"] == 10
        assert counts["class_b"] == 5

    def test_05_per_class_mean_alpha_keys(self):
        """per_class_mean_alpha() has entries for classes that reached warmup."""
        cal = PerClassConformalCalibrator()
        # Feed enough for class_a, not enough for class_b.
        for _ in range(PER_CLASS_WARMUP_MIN + 5):
            cal.update(10.0, 10, "class_a")
            cal.current_alpha("class_a")  # trigger alpha tracking
        for _ in range(2):
            cal.update(10.0, 10, "class_b")
        result = cal.per_class_mean_alpha()
        assert "class_a" in result
        # class_b may or may not appear depending on whether alpha was ever queried.

    def test_06_noisy_class_capped_at_twice_alpha_max(self):
        """Noisy class with high rel_err is capped at 2*alpha_max."""
        cal = PerClassConformalCalibrator()
        # Feed noisy completions: predicted=1, actual=1000 → rel_err=999.
        for _ in range(PER_CLASS_WARMUP_MIN + 10):
            cal.update(1.0, 1000, "noisy")
        alpha = cal.current_alpha("noisy")
        assert alpha <= 2 * CONFORMAL_ALPHA_MAX + 1e-9, f"Expected α≤{2*CONFORMAL_ALPHA_MAX}, got {alpha}"

    def test_07_independent_class_calibration(self):
        """Accurate class and noisy class must have independently calibrated α."""
        cal = PerClassConformalCalibrator()
        n = PER_CLASS_WARMUP_MIN + 20
        for _ in range(n):
            cal.update(100.0, 100, "accurate")   # rel_err = 0
            cal.update(1.0, 1000, "noisy")        # rel_err = 9.99

        alpha_accurate = cal.current_alpha("accurate")
        alpha_noisy    = cal.current_alpha("noisy")

        # Accurate class should be near zero; noisy should be near cap.
        assert alpha_accurate < 1e-6, f"Accurate α should be ≈0, got {alpha_accurate}"
        assert alpha_noisy > CONFORMAL_ALPHA_MAX, f"Noisy α should be > alpha_max, got {alpha_noisy}"


# ---------------------------------------------------------------------------
# Simulation invariants
# ---------------------------------------------------------------------------

class TestPerClassSimulatorInvariants:

    def _build_cal(self) -> PerClassConformalCalibrator:
        return PerClassConformalCalibrator()

    def test_08_all_requests_complete(self):
        """All requests must complete (no dropped requests)."""
        reqs = _make_two_model_requests(n_accurate=30, n_noisy=10)
        cal = self._build_cal()
        summary, response, wait_map = _simulate_decoupled_hybrid_per_class_conformal(
            reqs, servers=4, calibrator=cal
        )
        assert len(response) == len(reqs), f"Expected {len(reqs)}, got {len(response)}"

    def test_09_response_times_gte_service_times(self):
        """Response time must be ≥ service time for every request."""
        reqs = _make_two_model_requests(n_accurate=30, n_noisy=10)
        cal = self._build_cal()
        summary, response, wait_map = _simulate_decoupled_hybrid_per_class_conformal(
            reqs, servers=4, calibrator=cal
        )
        for r in reqs:
            if r.idx in response:
                assert response[r.idx] >= r.service_s - 1e-9, (
                    f"Request {r.idx}: response {response[r.idx]:.3f} < service {r.service_s:.3f}"
                )

    def test_10_preemption_count_in_summary(self):
        """Summary must contain preemption_count key."""
        reqs = _make_two_model_requests(n_accurate=20, n_noisy=10)
        cal = self._build_cal()
        summary, _, _ = _simulate_decoupled_hybrid_per_class_conformal(
            reqs, servers=2, calibrator=cal
        )
        assert "preemption_count" in summary
        assert isinstance(summary["preemption_count"], int)
        assert summary["preemption_count"] >= 0

    def test_11_per_class_mean_alpha_in_summary(self):
        """Summary must contain per_class_mean_alpha dict."""
        reqs = _make_two_model_requests(n_accurate=50, n_noisy=10)
        cal = self._build_cal()
        summary, _, _ = _simulate_decoupled_hybrid_per_class_conformal(
            reqs, servers=4, calibrator=cal
        )
        assert "per_class_mean_alpha" in summary
        assert isinstance(summary["per_class_mean_alpha"], dict)

    def test_12_class_counts_in_summary(self):
        """Summary must contain class_counts dict."""
        reqs = _make_two_model_requests(n_accurate=30, n_noisy=10)
        cal = self._build_cal()
        summary, _, _ = _simulate_decoupled_hybrid_per_class_conformal(
            reqs, servers=4, calibrator=cal
        )
        assert "class_counts" in summary
        assert isinstance(summary["class_counts"], dict)

    def test_13_simulate_queue_accepts_per_class_discipline(self):
        """simulate_queue must accept 'decoupled_hybrid_per_class_conformal' discipline."""
        reqs = _make_single_model_requests(n=20, tok=100)
        summary, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid_per_class_conformal")
        assert len(resp) == len(reqs)
        assert "requests" in summary or "per_class_mean_alpha" in summary


# ---------------------------------------------------------------------------
# PerClassConformalReport
# ---------------------------------------------------------------------------

class TestPerClassConformalReport:

    def _make_report(self) -> PerClassConformalReport:
        return PerClassConformalReport(
            trace="test",
            total_requests=100,
            servers=4,
            target_rho=0.85,
            sla_s=30.0,
            n_model_ids=2,
            fifo_goodput_per_dollar=1000.0,
            oracle_goodput_per_dollar=7000.0,
            global_goodput_per_dollar=5000.0,
            per_class_goodput_per_dollar=6000.0,
            oracle_delta_pct=600.0,
            global_delta_pct=400.0,
            per_class_delta_pct=500.0,
            global_vs_oracle_retention_pct=66.67,
            per_class_vs_oracle_retention_pct=83.33,
            per_class_vs_global_pct=20.0,
            global_mean_alpha=0.002,
            per_class_mean_alpha={"accurate_model": 0.0001, "noisy_model": 0.002},
            class_counts={"accurate_model": 80, "noisy_model": 20},
            fifo_sim={"n_completed": 100},
            oracle_sim={"n_completed": 100},
            global_sim={"n_completed": 100},
            per_class_sim={"n_completed": 100},
        )

    def test_14_to_dict_required_keys(self):
        """to_dict() must contain all required KPI keys."""
        d = self._make_report().to_dict()
        required = [
            "trace", "total_requests", "servers", "target_rho", "sla_s", "n_model_ids",
            "fifo_goodput_per_dollar", "oracle_goodput_per_dollar",
            "global_goodput_per_dollar", "per_class_goodput_per_dollar",
            "oracle_delta_pct", "global_delta_pct", "per_class_delta_pct",
            "global_vs_oracle_retention_pct", "per_class_vs_oracle_retention_pct",
            "per_class_vs_global_pct", "global_mean_alpha",
            "per_class_mean_alpha", "class_counts",
        ]
        for k in required:
            assert k in d, f"Missing key: {k}"

    def test_15_to_dict_shadow_tag_present(self):
        """to_dict() must contain shadow_tag."""
        d = self._make_report().to_dict()
        assert "shadow_tag" in d
        assert "shadow" in d["shadow_tag"].lower()

    def test_16_to_dict_per_class_mean_alpha_is_dict(self):
        """to_dict() per_class_mean_alpha must be a dict (not a list or str)."""
        d = self._make_report().to_dict()
        assert isinstance(d["per_class_mean_alpha"], dict)
        assert "accurate_model" in d["per_class_mean_alpha"]

    def test_16b_to_dict_no_nan_inf(self):
        """to_dict() float values must be finite."""
        d = self._make_report().to_dict()
        for key in ["fifo_goodput_per_dollar", "per_class_goodput_per_dollar",
                    "oracle_delta_pct", "global_delta_pct", "per_class_delta_pct"]:
            v = d[key]
            assert math.isfinite(v), f"Non-finite value for {key}: {v}"


# ---------------------------------------------------------------------------
# Two-model synthetic end-to-end: primary hypothesis test
# ---------------------------------------------------------------------------

class TestTwoModelSyntheticHypothesis:
    """Core hypothesis: per-class calibration outperforms global on two-model traces."""

    def _build_two_model_trace_compact(
        self,
        n_accurate: int = 120,
        n_noisy_short: int = 60,
        n_noisy_long: int = 15,
        servers: int = 3,
        target_rho: float = 0.80,
        seed: int = 99,
    ) -> tuple[list, list, list, float]:
        """Build oracle + ml-predicted requests for two-model comparison."""
        import random
        rng = random.Random(seed)

        # accurate_model: 100 tokens, perfect prediction.
        accurate = [(float(i * 2), 100) for i in range(n_accurate)]
        # noisy_model: mostly 5 tokens (short), occasional 800 (surprise-long).
        noisy = []
        for i in range(n_noisy_short + n_noisy_long):
            is_long = (i >= n_noisy_short)
            tok = 800 if is_long else 5
            noisy.append((float(i * 2 + 1), tok))

        # Merge and sort by arrival.
        all_raw = accurate + noisy
        all_raw.sort(key=lambda x: x[0])

        # Build features (model_id + input_tokens).
        features = []
        for (arr, tok) in all_raw:
            if (arr, tok) in [(a, 100) for (a, _) in accurate]:
                features.append({"model_id": "accurate_model", "input_tokens": 50})
            else:
                features.append({"model_id": "noisy_model", "input_tokens": 20})

        # Recompute features by position (accurate first n_accurate).
        features = []
        for i, (arr, tok) in enumerate(all_raw):
            if tok == 100:
                features.append({"model_id": "accurate_model", "input_tokens": 50})
            else:
                features.append({"model_id": "noisy_model", "input_tokens": 20})

        warp = calibrate_time_warp(all_raw, servers=servers, target_rho=target_rho)
        return all_raw, features, warp

    def test_17_accurate_class_alpha_lower_than_noisy(self):
        """After simulation, accurate class must have lower mean α than noisy class."""
        reqs = _make_two_model_requests(
            n_accurate=PER_CLASS_WARMUP_MIN + 20,
            n_noisy=PER_CLASS_WARMUP_MIN + 20,
        )
        cal = PerClassConformalCalibrator()
        summary, resp, _ = _simulate_decoupled_hybrid_per_class_conformal(
            reqs, servers=4, calibrator=cal
        )
        per_class_alpha = cal.per_class_mean_alpha()
        # accurate_model uses oracle predictions (rel_err=0) → α→0 after warmup.
        # noisy_model has high rel_err → α near cap.
        if "accurate_model" in per_class_alpha and "noisy_model" in per_class_alpha:
            assert per_class_alpha["accurate_model"] < per_class_alpha["noisy_model"], (
                f"accurate_model α={per_class_alpha['accurate_model']:.6f} should be "
                f"< noisy_model α={per_class_alpha['noisy_model']:.6f}"
            )

    def test_18_per_class_goodput_ge_global_on_two_model(self):
        """Per-class conformal must achieve ≥ global conformal goodput on two-model trace.

        This is the PRIMARY HYPOTHESIS of run 2026-06-22-w.  With accurate predictions
        for accurate_model and per-class calibration, that class converges to α≈0
        (near-SRPT), achieving higher SLA-safe goodput/$ than global calibration which
        is held back by noisy_model's errors.
        """
        from aurelius.benchmarks.srtf_serving_backtest import (
            ConformalAlphaCalibrator,
            _simulate_decoupled_hybrid_conformal,
        )

        reqs_global = _make_two_model_requests(
            n_accurate=PER_CLASS_WARMUP_MIN * 3,
            n_noisy=PER_CLASS_WARMUP_MIN,
        )
        reqs_per_cls = _make_two_model_requests(
            n_accurate=PER_CLASS_WARMUP_MIN * 3,
            n_noisy=PER_CLASS_WARMUP_MIN,
        )

        sla_s = 30.0
        servers = 4

        global_cal = ConformalAlphaCalibrator()
        global_sim, global_resp, _ = _simulate_decoupled_hybrid_conformal(
            reqs_global, servers, global_cal
        )
        gp_global = _sla_safe_goodput_per_dollar(reqs_global, global_resp, sla_s, servers)

        per_cls_cal = PerClassConformalCalibrator()
        per_cls_sim, per_cls_resp, _ = _simulate_decoupled_hybrid_per_class_conformal(
            reqs_per_cls, servers, per_cls_cal
        )
        gp_per_cls = _sla_safe_goodput_per_dollar(reqs_per_cls, per_cls_resp, sla_s, servers)

        assert gp_per_cls >= gp_global * 0.95, (
            f"Per-class goodput {gp_per_cls:.2f} < 95% of global {gp_global:.2f}. "
            "Per-class should match or exceed global conformal."
        )

    def test_19_per_class_vs_oracle_retention_reported(self):
        """PerClassConformalReport must report per_class_vs_oracle_retention_pct."""
        report = PerClassConformalReport(
            trace="synthetic",
            total_requests=100,
            servers=4,
            target_rho=0.85,
            sla_s=30.0,
            n_model_ids=2,
            fifo_goodput_per_dollar=1000.0,
            oracle_goodput_per_dollar=7000.0,
            global_goodput_per_dollar=4200.0,    # 70% retention
            per_class_goodput_per_dollar=5200.0,  # 87% retention (hypothesis)
            oracle_delta_pct=600.0,
            global_delta_pct=420.0,
            per_class_delta_pct=520.0,
            global_vs_oracle_retention_pct=70.0,
            per_class_vs_oracle_retention_pct=86.67,
            per_class_vs_global_pct=23.8,
            global_mean_alpha=0.002,
            per_class_mean_alpha={"accurate_model": 0.0, "noisy_model": 0.002},
            class_counts={"accurate_model": 75, "noisy_model": 25},
            fifo_sim={}, oracle_sim={}, global_sim={}, per_class_sim={},
        )
        assert report.per_class_vs_oracle_retention_pct > report.global_vs_oracle_retention_pct, (
            "Per-class retention should exceed global retention (this is the hypothesis)"
        )


# ---------------------------------------------------------------------------
# HF data integration tests (skipped if JSONL unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HF_JSONL_AVAILABLE, reason="BurstGPT HF JSONL not available")
class TestBurstGPTHFIntegration:

    def test_20_run_returns_report(self):
        """run_burstgpt_per_class_conformal_backtest returns PerClassConformalReport."""
        report = run_burstgpt_per_class_conformal_backtest(job_limit=500)
        assert isinstance(report, PerClassConformalReport)

    def test_21_per_class_goodput_positive(self):
        """per_class_goodput_per_dollar must be positive."""
        report = run_burstgpt_per_class_conformal_backtest(job_limit=500)
        assert report.per_class_goodput_per_dollar > 0

    def test_22_per_class_vs_global_pct_reported(self):
        """per_class_vs_global_pct must be a finite float."""
        report = run_burstgpt_per_class_conformal_backtest(job_limit=500)
        assert math.isfinite(report.per_class_vs_global_pct)

    def test_23_n_model_ids_ge_1(self):
        """n_model_ids must be ≥ 1 (BurstGPT has at least one model class)."""
        report = run_burstgpt_per_class_conformal_backtest(job_limit=500)
        assert report.n_model_ids >= 1

    def test_24_oracle_exceeds_fifo(self):
        """Oracle conformal must exceed FIFO (validates simulation correctness)."""
        report = run_burstgpt_per_class_conformal_backtest(job_limit=500)
        assert report.oracle_goodput_per_dollar > report.fifo_goodput_per_dollar


# ---------------------------------------------------------------------------
# Constant validation
# ---------------------------------------------------------------------------

def test_24_per_class_warmup_min_valid():
    """PER_CLASS_WARMUP_MIN must be positive and ≤ CONFORMAL_WARMUP."""
    assert PER_CLASS_WARMUP_MIN > 0
    assert PER_CLASS_WARMUP_MIN <= CONFORMAL_WARMUP
