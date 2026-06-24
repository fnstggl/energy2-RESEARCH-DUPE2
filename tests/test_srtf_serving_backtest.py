"""Tests for the SRTF serving-queue backtest (request-level SJF evaluation).

Covers the discrete-event M/G/c simulator correctness, the FIFO-vs-SRTF
ordering semantics, the time-warp calibration, the leakage guard (ordering uses
predicted tokens; physics uses actual tokens), and the headline-improvement
direction on the real Azure LLM 2024 public trace.
"""

from __future__ import annotations

import logging
import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    GPU_HOUR_USD,
    TPOT_S,
    TTFT_BASE_S,
    SRTFServingReport,
    _Request,
    _service_time_s,
    _sla_safe_goodput,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    load_serving_requests,
    run_srtf_serving_backtest,
    simulate_queue,
)

# Silence the scheduler/loader warning spam during the queue runs.
logging.disable(logging.CRITICAL)


def _req(idx, arrival, tokens, predicted=None):
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(tokens),
        predicted_tokens=float(predicted if predicted is not None else tokens),
        service_s=float(tokens),  # service == tokens for clean hand-math in tests
    )


# ---------------------------------------------------------------------------
# Service physics
# ---------------------------------------------------------------------------

class TestServiceTime:
    def test_service_time_monotonic_in_tokens(self):
        assert _service_time_s(10) < _service_time_s(100) < _service_time_s(1000)

    def test_service_time_formula(self):
        assert _service_time_s(0) == pytest.approx(TTFT_BASE_S)
        assert _service_time_s(100) == pytest.approx(TTFT_BASE_S + 100 * TPOT_S)


# ---------------------------------------------------------------------------
# Real-trace loading
# ---------------------------------------------------------------------------

class TestLoadServingRequests:
    def test_fixture_exists(self):
        assert os.path.exists(DEFAULT_AZURE_FIXTURE)

    def test_loads_real_requests(self):
        reqs = load_serving_requests()
        assert len(reqs) > 1000
        # every entry is (arrival_s, output_tokens) with positive tokens
        assert all(tok > 0 for _, tok in reqs)
        # arrivals are sorted and start at 0 (relative)
        arrivals = [a for a, _ in reqs]
        assert arrivals == sorted(arrivals)
        assert arrivals[0] == pytest.approx(0.0)

    def test_limit_respected(self):
        reqs = load_serving_requests(limit=50)
        assert len(reqs) == 50

    def test_real_distribution_is_heavy_tailed(self):
        # The SRTF benefit comes from output-length heterogeneity.
        toks = sorted(tok for _, tok in load_serving_requests())
        p50 = toks[len(toks) // 2]
        p99 = toks[int(0.99 * len(toks))]
        assert p99 > 2 * p50  # genuinely heavy-tailed


# ---------------------------------------------------------------------------
# Time-warp calibration
# ---------------------------------------------------------------------------

class TestCalibrateTimeWarp:
    def test_warp_hits_target_rho(self):
        raw = load_serving_requests()
        servers, rho = 4, 0.85
        warp = calibrate_time_warp(raw, servers=servers, target_rho=rho)
        # Reconstruct rho from the warped arrival rate and verify it matches.
        span = (raw[-1][0] - raw[0][0]) / warp
        lam = len(raw) / span
        mean_service = sum(_service_time_s(t) for _, t in raw) / len(raw)
        assert lam * mean_service / servers == pytest.approx(rho, rel=1e-6)

    def test_higher_rho_means_more_compression(self):
        raw = load_serving_requests()
        w_low = calibrate_time_warp(raw, servers=4, target_rho=0.5)
        w_high = calibrate_time_warp(raw, servers=4, target_rho=0.9)
        assert w_high > w_low

    def test_degenerate_input_returns_unit_warp(self):
        assert calibrate_time_warp([], servers=4, target_rho=0.85) == 1.0
        assert calibrate_time_warp([(0.0, 10)], servers=4, target_rho=0.85) == 1.0


# ---------------------------------------------------------------------------
# Discrete-event simulator correctness
# ---------------------------------------------------------------------------

class TestSimulateQueue:
    def test_single_server_fifo_hand_computed(self):
        # 1 server, three jobs arriving together: long, short, short (arrival seq).
        reqs = [_req(0, 0, 1), _req(1, 0, 10), _req(2, 0, 1)]
        summary, resp, wait = simulate_queue(reqs, servers=1, discipline="fifo")
        # FIFO order 0,1,2: resp = 1, 11, 12
        assert resp[0] == pytest.approx(1)
        assert resp[1] == pytest.approx(11)
        assert resp[2] == pytest.approx(12)

    def test_single_server_srtf_reorders_short_first(self):
        reqs = [_req(0, 0, 1), _req(1, 0, 10), _req(2, 0, 1)]
        summary, resp, wait = simulate_queue(reqs, servers=1, discipline="srtf")
        # SRTF picks shortest predicted first: r0(1), r2(1), then r1(10).
        assert resp[0] == pytest.approx(1)   # [0,1]
        assert resp[2] == pytest.approx(2)   # [1,2]  -- short job no longer blocked
        assert resp[1] == pytest.approx(12)  # [2,12] -- long job served last

    def test_srtf_reduces_mean_response_vs_fifo(self):
        reqs = [_req(0, 0, 1), _req(1, 0, 10), _req(2, 0, 1)]
        fifo, _, _ = simulate_queue(reqs, servers=1, discipline="fifo")
        srtf, _, _ = simulate_queue(reqs, servers=1, discipline="srtf")
        assert srtf["mean_response_s"] < fifo["mean_response_s"]

    def test_all_requests_complete(self):
        reqs = [_req(i, i * 0.1, (i % 5) + 1) for i in range(50)]
        for disc in ("fifo", "srtf"):
            _, resp, _ = simulate_queue(reqs, servers=3, discipline=disc)
            assert len(resp) == 50

    def test_no_request_starts_before_arrival(self):
        reqs = [_req(i, i * 2.0, 1) for i in range(10)]
        _, resp, wait = simulate_queue(reqs, servers=2, discipline="fifo")
        assert all(w >= -1e-9 for w in wait.values())

    def test_idle_system_has_zero_wait(self):
        # arrivals far apart relative to service → no queueing on 2 servers
        reqs = [_req(i, i * 100.0, 1) for i in range(5)]
        _, _, wait = simulate_queue(reqs, servers=2, discipline="fifo")
        assert all(w == pytest.approx(0.0) for w in wait.values())

    def test_more_servers_never_increase_mean_response(self):
        reqs = [_req(i, i * 0.05, (i % 7) + 1) for i in range(60)]
        prev = None
        for c in (1, 2, 4, 8):
            summary, _, _ = simulate_queue(reqs, servers=c, discipline="fifo")
            if prev is not None:
                assert summary["mean_response_s"] <= prev + 1e-9
            prev = summary["mean_response_s"]


# ---------------------------------------------------------------------------
# Leakage guard: ordering uses PREDICTED tokens; physics uses ACTUAL tokens
# ---------------------------------------------------------------------------

class TestLeakageGuard:
    def test_ordering_follows_predicted_not_actual(self):
        # r1 is actually short (service 1) but PREDICTED long (100): SRTF should
        # defer it; r2 predicted short runs first even though both arrive together.
        reqs = [
            _Request(0, 0.0, 10, predicted_tokens=10.0, service_s=10.0),   # long, predicted long
            _Request(1, 0.0, 1, predicted_tokens=100.0, service_s=1.0),    # short, predicted long
            _Request(2, 0.0, 1, predicted_tokens=1.0, service_s=1.0),      # short, predicted short
        ]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srtf")
        # Order by predicted: r2(1) < r0(10) < r1(100).
        assert resp[2] == pytest.approx(1)    # [0,1]
        assert resp[0] == pytest.approx(11)   # [1,11]
        assert resp[1] == pytest.approx(12)   # [11,12] deferred despite being short


# ---------------------------------------------------------------------------
# Goodput accounting
# ---------------------------------------------------------------------------

class TestGoodput:
    def test_sla_safe_goodput_counts_only_in_budget(self):
        reqs = [_req(0, 0, 5), _req(1, 0, 7)]
        resp = {0: 3.0, 1: 20.0}  # only r0 meets a 10s SLA
        assert _sla_safe_goodput(reqs, resp, sla_s=10.0) == 5.0

    def test_goodput_denominator_is_discipline_invariant(self):
        # Same request set, same servers → identical denominator regardless of
        # which response map is supplied. The metric moves only via the numerator.
        reqs = [_req(i, 0, (i % 4) + 1) for i in range(20)]
        all_fast = {r.idx: 0.0 for r in reqs}
        gp = _sla_safe_goodput_per_dollar(reqs, all_fast, sla_s=10.0, servers=4)
        busy_hours = sum(r.service_s for r in reqs) / 3600.0
        expected = sum(r.actual_tokens for r in reqs) / (busy_hours * GPU_HOUR_USD)
        assert gp == pytest.approx(expected)


# ---------------------------------------------------------------------------
# End-to-end on the real Azure LLM 2024 trace
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_report_structure(self):
        r = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        assert isinstance(r, SRTFServingReport)
        d = r.to_dict()
        for key in ("fifo", "srtf_perfect", "srtf_forecast"):
            assert "mean_response_s" in d[key]
            assert "short_p90_response_s" in d[key]
            assert "sla_safe_goodput_per_dollar" in d[key]
        assert d["shadow_tag"].startswith("shadow_only")

    def test_srtf_improves_short_request_p90_under_contention(self):
        r = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        # Short-request p90 latency should drop substantially under SRTF.
        assert r.srtf_perfect["short_p90_response_s"] < r.fifo["short_p90_response_s"]
        assert r.short_p90_improvement_pct > 20.0  # well beyond noise

    def test_srtf_improves_mean_response(self):
        r = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        assert r.mean_response_improvement_pct > 0.0

    def test_forecast_prior_remains_beneficial(self):
        # A realistic noisy forecast prior should still help short requests.
        r = run_srtf_serving_backtest(servers=4, target_rho=0.85, forecast_noise_cv=0.30)
        assert r.forecast_short_p90_improvement_pct > 20.0

    def test_sla_goodput_improves_under_contention(self):
        r = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        assert r.sla_goodput_delta_pct > 0.0

    def test_long_request_tail_regresses_honest_cost(self):
        # Non-preemptive SJF starves long requests: p99 should get WORSE. We
        # assert this so the documented trade-off cannot silently disappear.
        r = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        assert r.srtf_perfect["p99_response_s"] > r.fifo["p99_response_s"]

    def test_determinism(self):
        a = run_srtf_serving_backtest(servers=4, target_rho=0.85).to_dict()
        b = run_srtf_serving_backtest(servers=4, target_rho=0.85).to_dict()
        assert a == b

    def test_low_load_has_small_effect(self):
        # With a light load and plenty of servers, queueing is rare so SRTF and
        # FIFO should be close (sanity: the win is a contention phenomenon).
        r = run_srtf_serving_backtest(servers=64, target_rho=0.10)
        assert abs(r.mean_response_improvement_pct) < 50.0
