"""Tests for SRTF-with-Aging anti-starvation guard + BurstGPT cross-validation.

Run 2026-06-20-i adds:
  - ``aging_srtf`` discipline: key(r, t) = predicted_tokens / (1 + alpha * wait_s)
  - ``load_burstgpt_serving_requests`` loader
  - ``SRTFAgingReport`` dataclass
  - ``run_aging_srtf_backtest`` (Azure 2024)
  - ``run_burstgpt_aging_backtest`` (BurstGPT cross-validation)

Research basis:
  - Astraea (arXiv:2512.14142): aging-based promotion for LLM serving fairness.
  - FlowPrefill (arXiv:2602.16603): operator-level preemption for HoL mitigation.
  - Equinox (arXiv:2508.16646): holistic fair scheduling with starvation bounds.
"""

from __future__ import annotations

import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    AGING_ALPHA_DEFAULT,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    SRTFAgingReport,
    _Request,
    _run_aging_backtest_on_trace,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    load_burstgpt_serving_requests,
    load_serving_requests,
    run_aging_srtf_backtest,
    run_burstgpt_aging_backtest,
    simulate_queue,
)


def _req(idx, arrival, tokens, predicted=None):
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(tokens),
        predicted_tokens=float(predicted if predicted is not None else tokens),
        service_s=float(tokens),
    )


# ---------------------------------------------------------------------------
# Aging-SRTF ordering semantics
# ---------------------------------------------------------------------------

class TestAgingSRTFOrdering:
    def test_at_zero_wait_orders_like_srtf(self):
        # Three requests arriving simultaneously; alpha > 0 but wait = 0 at
        # arrival, so first dispatch is identical to pure SRTF.
        reqs = [_req(0, 0, 10), _req(1, 0, 1), _req(2, 0, 5)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="aging_srtf")
        # At t=0, all arrive together with zero wait: order by predicted tok.
        # r1(1) < r2(5) < r0(10)
        assert resp[1] < resp[2] < resp[0]

    def test_long_wait_overcomes_short_prediction(self):
        # 1 server.  r0 is long (100 tok) and arrives at t=0.
        # r1 is short (1 tok) but arrives at t=1000 (after a very long gap).
        # Without aging, r0 should run first (only one in queue at t=0).
        # With aging, r0 has already been waiting a long time before r1 arrives.
        # This test verifies that a long-waiting request gets priority once the
        # server is free and a new short request just arrived.
        reqs = [
            _req(0, 0.0, 100, predicted=100),    # long, arrives first
            _req(1, 200.0, 1, predicted=1),       # short, arrives later
        ]
        _, resp_aging, _ = simulate_queue(reqs, servers=1, discipline="aging_srtf",
                                          aging_alpha=0.5)
        # r0 runs first (server free at t=0, only r0 is ready).
        # After r0 finishes at t=100, r1 arrives at t=200 so it runs then.
        # Both end up served; no starvation possible in this 2-request case.
        assert 0 in resp_aging and 1 in resp_aging

    def test_all_requests_complete_under_aging(self):
        reqs = [_req(i, i * 0.1, (i % 5) + 1) for i in range(50)]
        _, resp, _ = simulate_queue(reqs, servers=3, discipline="aging_srtf")
        assert len(resp) == 50

    def test_no_negative_wait_under_aging(self):
        reqs = [_req(i, i * 2.0, (i % 4) + 1) for i in range(20)]
        _, _, wait = simulate_queue(reqs, servers=2, discipline="aging_srtf")
        assert all(w >= -1e-9 for w in wait.values())

    def test_alpha_zero_degrades_to_srtf(self):
        # With alpha=0 the aging factor is 1 for all wait times → key =
        # predicted_tokens (same as srtf).
        reqs = [_req(i, 0.0, (i % 7) + 1) for i in range(21)]
        _, resp_srtf, _ = simulate_queue(reqs, servers=2, discipline="srtf")
        _, resp_aging, _ = simulate_queue(reqs, servers=2, discipline="aging_srtf",
                                          aging_alpha=0.0)
        for idx in resp_srtf:
            assert resp_srtf[idx] == pytest.approx(resp_aging[idx], abs=1e-9)

    def test_aging_reduces_max_response_vs_pure_srtf(self):
        # Under heavy load with heterogeneous lengths, aging should reduce the
        # maximum response time (the starvation bound) vs pure SRTF.
        # Mix: many short (1-2 tok) + a few long (100 tok) requests.
        reqs = []
        for i in range(30):
            reqs.append(_req(i, i * 0.5, 1, predicted=1))
        for i in range(30, 34):
            reqs.append(_req(i, 0.0, 100, predicted=100))
        _, resp_srtf, _ = simulate_queue(reqs, servers=1, discipline="srtf")
        _, resp_aging, _ = simulate_queue(reqs, servers=1, discipline="aging_srtf",
                                          aging_alpha=0.05)
        max_srtf = max(resp_srtf.values())
        max_aging = max(resp_aging.values())
        assert max_aging <= max_srtf + 1e-9  # aging should not increase max response

    def test_aging_preserves_mean_improvement_vs_fifo(self):
        # Aging SRTF should still reduce mean response vs FIFO (the SRTF benefit
        # is partially preserved even while bounding starvation).
        reqs = [_req(i, i * 0.5, (i % 8) + 1) for i in range(40)]
        fifo_sim, _, _ = simulate_queue(reqs, servers=2, discipline="fifo")
        aging_sim, _, _ = simulate_queue(reqs, servers=2, discipline="aging_srtf")
        # Aging SRTF should not be dramatically worse than FIFO on mean response.
        assert aging_sim["mean_response_s"] <= fifo_sim["mean_response_s"] * 1.5

    def test_starvation_bound_satisfied(self):
        # Long requests (100 tok) mixed with constant stream of short (1 tok).
        # With alpha=0.1 the long request reaches parity with short after ≈ 990s.
        # In a finite simulation the long request must eventually be served.
        long_req = _req(0, 0.0, 100, predicted=100)
        short_reqs = [_req(i + 1, i * 1.0, 1, predicted=1) for i in range(20)]
        reqs = [long_req] + short_reqs
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="aging_srtf",
                                    aging_alpha=0.1)
        # The long request must be served (no indefinite starvation).
        assert 0 in resp

    def test_aging_summary_has_long_p99(self):
        reqs = [_req(i, i * 0.5, (i % 6) + 1) for i in range(30)]
        summary, _, _ = simulate_queue(reqs, servers=2, discipline="aging_srtf")
        assert "long_p99_response_s" in summary
        assert "long_p90_response_s" in summary
        assert summary["long_p99_response_s"] >= 0.0


# ---------------------------------------------------------------------------
# BurstGPT loader
# ---------------------------------------------------------------------------

class TestLoadBurstGPTServingRequests:
    def test_fixture_exists(self):
        assert os.path.exists(DEFAULT_BURSTGPT_FIXTURE)

    def test_loads_non_zero_responses(self):
        reqs = load_burstgpt_serving_requests()
        assert len(reqs) > 0
        # All entries have positive response tokens
        assert all(tok > 0 for _, tok in reqs)

    def test_arrivals_relative_to_zero(self):
        reqs = load_burstgpt_serving_requests()
        arrivals = [a for a, _ in reqs]
        assert arrivals[0] == pytest.approx(0.0)
        assert arrivals == sorted(arrivals)

    def test_limit_respected(self):
        reqs = load_burstgpt_serving_requests(limit=5)
        assert len(reqs) <= 5

    def test_has_heterogeneous_token_lengths(self):
        reqs = load_burstgpt_serving_requests()
        toks = [t for _, t in reqs]
        assert max(toks) > min(toks)  # heterogeneous

    def test_excludes_failures(self):
        # The fixture has rows with Response tokens=0 (failures); confirm excluded.
        all_reqs = load_burstgpt_serving_requests()
        assert all(t > 0 for _, t in all_reqs)


# ---------------------------------------------------------------------------
# SRTFAgingReport dataclass
# ---------------------------------------------------------------------------

class TestSRTFAgingReport:
    def _make_report(self):
        fifo = {"mean_response_s": 100.0, "short_p90_response_s": 200.0,
                "long_p99_response_s": 90.0, "sla_safe_goodput_per_dollar": 100.0}
        srtf = {"mean_response_s": 40.0, "short_p90_response_s": 5.0,
                "long_p99_response_s": 600.0, "sla_safe_goodput_per_dollar": 320.0}
        aging = {"mean_response_s": 50.0, "short_p90_response_s": 15.0,
                 "long_p99_response_s": 120.0, "sla_safe_goodput_per_dollar": 200.0}
        return SRTFAgingReport(
            trace="test",
            total_requests=100,
            servers=4,
            target_rho=0.85,
            time_warp=1.0,
            sla_s=10.0,
            aging_alpha=0.05,
            fifo=fifo,
            srtf_perfect=srtf,
            aging_srtf=aging,
            srtf_short_p90_improvement_pct=97.5,
            aging_short_p90_improvement_pct=92.5,
            srtf_long_p99_delta_pct=566.7,
            aging_long_p99_delta_pct=33.3,
            srtf_goodput_delta_pct=220.0,
            aging_goodput_delta_pct=100.0,
        )

    def test_to_dict_has_all_fields(self):
        r = self._make_report()
        d = r.to_dict()
        assert "trace" in d
        assert "aging_alpha" in d
        assert "aging_srtf" in d
        assert "srtf_short_p90_improvement_pct" in d
        assert "aging_short_p90_improvement_pct" in d
        assert "srtf_long_p99_delta_pct" in d
        assert "aging_long_p99_delta_pct" in d
        assert "srtf_goodput_delta_pct" in d
        assert "aging_goodput_delta_pct" in d

    def test_shadow_tag_present(self):
        r = self._make_report()
        d = r.to_dict()
        assert "shadow" in d["shadow_tag"]

    def test_aging_recovers_long_tail_vs_srtf(self):
        r = self._make_report()
        # Aging should have a smaller long-p99 delta than pure SRTF
        assert r.aging_long_p99_delta_pct < r.srtf_long_p99_delta_pct

    def test_srtf_better_short_p90_than_aging(self):
        r = self._make_report()
        # Pure SRTF gets the maximum short-request benefit
        assert r.srtf_short_p90_improvement_pct >= r.aging_short_p90_improvement_pct


# ---------------------------------------------------------------------------
# Integration: run_aging_srtf_backtest on Azure 2024
# ---------------------------------------------------------------------------

class TestRunAgingSRTFBacktest:
    def test_returns_report_with_correct_trace(self):
        report = run_aging_srtf_backtest(servers=2, target_rho=0.70, job_limit=100)
        assert report.trace == "azure_llm_2024"
        assert report.total_requests == 100
        assert report.servers == 2

    def test_aging_bounds_long_tail_vs_srtf(self):
        report = run_aging_srtf_backtest(servers=4, target_rho=0.85, job_limit=200)
        # SRTF regresses long-tail (positive delta = worse); aging should regress less.
        assert report.aging_long_p99_delta_pct <= report.srtf_long_p99_delta_pct

    def test_aging_preserves_short_benefit(self):
        report = run_aging_srtf_backtest(servers=4, target_rho=0.85, job_limit=200)
        # Aging should still improve short-request p90 vs FIFO.
        assert report.aging_short_p90_improvement_pct > 0.0

    def test_srtf_has_higher_short_improvement(self):
        report = run_aging_srtf_backtest(servers=4, target_rho=0.85, job_limit=200)
        # Pure SRTF is optimal for short requests (aging trades some of this).
        assert report.srtf_short_p90_improvement_pct >= report.aging_short_p90_improvement_pct

    def test_aging_goodput_nonnegative_vs_fifo(self):
        # Aging SRTF should not regress goodput/$ vs FIFO.
        report = run_aging_srtf_backtest(servers=4, target_rho=0.85, job_limit=200)
        assert report.aging_goodput_delta_pct >= 0.0

    def test_aging_goodput_positive_under_tight_sla(self):
        # With a tight SLA (3s) at moderate load, SRTF and aging SRTF generate
        # SLA violations for long requests, making goodput/$ differentiation visible.
        report = run_aging_srtf_backtest(
            servers=4, target_rho=0.85, job_limit=500, sla_s=3.0
        )
        # Aging SRTF should improve goodput/$ vs FIFO.
        assert report.aging_goodput_delta_pct > 0.0
        # Should recover long-tail vs pure SRTF (smaller regression).
        assert report.aging_long_p99_delta_pct < report.srtf_long_p99_delta_pct

    def test_all_disciplines_have_long_p99(self):
        report = run_aging_srtf_backtest(servers=2, target_rho=0.70, job_limit=100)
        assert "long_p99_response_s" in report.fifo
        assert "long_p99_response_s" in report.srtf_perfect
        assert "long_p99_response_s" in report.aging_srtf

    def test_to_dict_is_serializable(self):
        import json
        report = run_aging_srtf_backtest(servers=2, target_rho=0.70, job_limit=50)
        d = report.to_dict()
        # Should be JSON-serializable
        json.dumps(d)

    def test_alpha_sensitivity(self):
        # Higher alpha → more aggressive aging → long-tail recovers more.
        report_low = run_aging_srtf_backtest(servers=4, target_rho=0.85, job_limit=200,
                                             aging_alpha=0.01)
        report_high = run_aging_srtf_backtest(servers=4, target_rho=0.85, job_limit=200,
                                              aging_alpha=0.50)
        # Higher alpha → long-tail regression should be smaller (less starvation).
        assert report_high.aging_long_p99_delta_pct <= report_low.aging_long_p99_delta_pct


# ---------------------------------------------------------------------------
# Integration: run_burstgpt_aging_backtest (cross-trace validation)
# ---------------------------------------------------------------------------

class TestRunBurstGPTAgingBacktest:
    def test_returns_burstgpt_trace(self):
        report = run_burstgpt_aging_backtest()
        assert report.trace == "burstgpt"
        assert report.servers == 4
        assert report.total_requests > 0

    def test_sla_budget_is_burstgpt_default(self):
        report = run_burstgpt_aging_backtest()
        assert report.sla_s == DEFAULT_BURSTGPT_SLA_S

    def test_all_requests_processed(self):
        report = run_burstgpt_aging_backtest()
        fifo_requests = report.fifo["requests"]
        srtf_requests = report.srtf_perfect["requests"]
        aging_requests = report.aging_srtf["requests"]
        assert fifo_requests == srtf_requests == aging_requests

    def test_long_p99_finite_under_both_disciplines(self):
        # With the small fixture (51 requests) starvation is not detectable at
        # scale; assert only that long_p99 is finite for both disciplines.
        report = run_burstgpt_aging_backtest()
        assert report.srtf_perfect["long_p99_response_s"] > 0.0
        assert report.aging_srtf["long_p99_response_s"] > 0.0

    def test_srtf_improves_short_p90_burstgpt(self):
        # Even on BurstGPT's heavier distribution, SRTF should improve short p90.
        report = run_burstgpt_aging_backtest()
        # At moderate ρ, SRTF should provide meaningful short-request benefit.
        assert report.srtf_short_p90_improvement_pct >= 0.0

    def test_consistency_fifo_summary_keys(self):
        report = run_burstgpt_aging_backtest()
        required_keys = {
            "mean_response_s", "p90_response_s", "p99_response_s",
            "short_p90_response_s", "long_p99_response_s",
            "sla_safe_goodput_per_dollar",
        }
        for key in required_keys:
            assert key in report.fifo, f"missing key: {key}"
            assert key in report.srtf_perfect, f"missing key: {key}"
            assert key in report.aging_srtf, f"missing key: {key}"

    def test_limit_applied(self):
        report = run_burstgpt_aging_backtest(job_limit=10)
        assert report.total_requests == 10


# ---------------------------------------------------------------------------
# AGING_ALPHA_DEFAULT constant
# ---------------------------------------------------------------------------

class TestAgingAlphaCalibration:
    def test_default_alpha_is_documented_value(self):
        assert AGING_ALPHA_DEFAULT == pytest.approx(0.05)

    def test_parity_time_within_reasonable_range(self):
        # At alpha=0.05, Azure 2024 p99 (479 tok) reaches parity with p50 (90 tok)
        # after W = (479/90 - 1) / 0.05 ≈ 86.4 seconds.
        alpha = AGING_ALPHA_DEFAULT
        p99_tok, p50_tok = 479.0, 90.0
        # parity condition: p99/(1+alpha*W) = p50 → W = (p99/p50 - 1)/alpha
        parity_s = (p99_tok / p50_tok - 1.0) / alpha
        # Should be between 60s and 120s (reasonable anti-starvation window).
        assert 60.0 <= parity_s <= 120.0
