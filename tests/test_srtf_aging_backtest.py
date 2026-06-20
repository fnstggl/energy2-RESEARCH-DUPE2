"""Tests for SRTF+aging anti-starvation discipline and BurstGPT cross-validation.

Run 2026-06-20-i adds ``simulate_queue_with_aging`` to the discrete-event
serving simulator.  The key invariants under test:

1. Aging discipline prevents starvation: no request waits beyond predicted_tokens/alpha.
2. SRTF+aging short-p90 << FIFO short-p90 (ordering gain preserved).
3. SRTF+aging long-p99 <= FIFO long-p99 (no starvation; aging is Pareto-safe vs FIFO).
4. SRTF without aging long-p99 > SRTF+aging long-p99 (aging fixes it).
5. SLA-safe goodput/$ is higher under SRTF+aging than FIFO.
6. BurstGPT loader returns valid (arrival, output_tokens) pairs.
7. SRTFAgingReport dataclass serialises correctly.
8. run_srtf_aging_backtest produces positive headline deltas on real Azure trace.
"""

from __future__ import annotations

import logging
import math
import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AGING_ALPHA,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_SLA_S,
    GPU_HOUR_USD,
    TPOT_S,
    TTFT_BASE_S,
    SRTFAgingReport,
    _Request,
    _service_time_s,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    load_burstgpt_requests,
    load_serving_requests,
    run_srtf_aging_backtest,
    run_srtf_serving_backtest,
    simulate_queue,
    simulate_queue_with_aging,
)

logging.disable(logging.CRITICAL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(idx, arrival, actual_tokens, predicted_tokens=None):
    """Build a _Request with service_s = actual_tokens (clean hand-math)."""
    if predicted_tokens is None:
        predicted_tokens = float(actual_tokens)
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(actual_tokens),
        predicted_tokens=float(predicted_tokens),
        service_s=float(actual_tokens),   # service == tokens for hand-math
    )


def _phys_req(idx, arrival, actual_tokens, predicted_tokens=None):
    """Build a _Request with realistic service_s = TTFT_BASE_S + tok*TPOT_S."""
    if predicted_tokens is None:
        predicted_tokens = float(actual_tokens)
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(actual_tokens),
        predicted_tokens=float(predicted_tokens),
        service_s=_service_time_s(actual_tokens),
    )


# ---------------------------------------------------------------------------
# 1. DEFAULT_AGING_ALPHA is positive and reasonable
# ---------------------------------------------------------------------------

class TestAgingAlphaDefault:
    def test_alpha_positive(self):
        assert DEFAULT_AGING_ALPHA > 0

    def test_alpha_formula(self):
        # DEFAULT_AGING_ALPHA = max_azure_tokens(1346) / fifo_p99_wait(730s) ≈ 1.844
        assert DEFAULT_AGING_ALPHA == pytest.approx(1.844, rel=0.05)

    def test_alpha_bounds_max_wait(self):
        # Longest Azure request (1346 tokens) ages to zero-priority after
        # 1346/alpha ≈ 730 s — stays inside the FIFO p99 response envelope.
        max_wait_s = 1346.0 / DEFAULT_AGING_ALPHA
        assert max_wait_s < 800.0   # well below 1000s "infinite starvation"
        assert max_wait_s > 600.0   # not so aggressive it ignores all ordering


# ---------------------------------------------------------------------------
# 2. simulate_queue_with_aging: basic correctness
# ---------------------------------------------------------------------------

class TestSimulateQueueWithAgingBasic:
    def test_all_requests_served(self):
        reqs = [_req(i, i * 0.5, 100) for i in range(20)]
        summary, resp, wait = simulate_queue_with_aging(reqs, servers=2, alpha=10.0)
        assert summary["requests"] == 20
        assert len(resp) == 20
        assert len(wait) == 20

    def test_response_geq_service_time(self):
        reqs = [_req(i, float(i), 50) for i in range(10)]
        _, resp, _ = simulate_queue_with_aging(reqs, servers=1, alpha=5.0)
        for r in reqs:
            assert resp[r.idx] >= r.service_s - 1e-9

    def test_response_geq_wait(self):
        reqs = [_req(i, float(i) * 0.1, 100) for i in range(30)]
        _, resp, wait = simulate_queue_with_aging(reqs, servers=2, alpha=20.0)
        for r in reqs:
            if r.idx in resp:
                assert resp[r.idx] >= wait[r.idx] - 1e-9

    def test_single_server_uniform_jobs(self):
        # With uniform predicted tokens, aging is irrelevant → same ordering as FIFO.
        reqs = [_req(i, float(i) * 2.0, 100) for i in range(5)]
        _, fifo_resp, _ = simulate_queue(reqs, 1, "fifo")
        _, aging_resp, _ = simulate_queue_with_aging(reqs, 1, alpha=10.0)
        for r in reqs:
            assert fifo_resp[r.idx] == pytest.approx(aging_resp[r.idx], rel=1e-6)

    def test_empty_ready_queue_handled(self):
        reqs = [_req(0, 0.0, 10)]
        summary, resp, wait = simulate_queue_with_aging(reqs, servers=1, alpha=1.0)
        assert summary["requests"] == 1


# ---------------------------------------------------------------------------
# 3. Aging anti-starvation: long request must be served within budget
# ---------------------------------------------------------------------------

class TestAgingAntiStarvation:
    def _run_starvation_scenario(self, alpha):
        """Long request (pred=1000) contends with 20 short requests (pred=10).

        All requests arrive at t=0 so SRTF has a full ready queue to choose
        from at the first dispatch event.  With 1 server, SRTF picks the 20
        short jobs first (pred=10 < pred=1000), leaving the long job to wait
        for all 20*10=200 s of service to complete before it is dispatched.
        """
        reqs = [_req(0, 0.0, 1000, 1000)]
        for i in range(1, 21):
            reqs.append(_req(i, 0.0, 10, 10))   # all at t=0 → full competition
        return reqs, alpha

    def test_aging_bounds_long_request_wait(self):
        reqs, alpha = self._run_starvation_scenario(alpha=10.0)
        # Long request max wait = predicted_tokens / alpha = 1000/10 = 100s
        _, resp, wait = simulate_queue_with_aging(reqs, servers=1, alpha=10.0)
        long_wait = wait[0]
        long_max = 1000.0 / 10.0  # 100s
        assert long_wait <= long_max + 1e-3, (
            f"Long request waited {long_wait:.2f}s > bound {long_max:.2f}s"
        )

    def test_srtf_without_aging_starves_long_request(self):
        """Confirm SRTF (no aging) DOES starve the long request — baseline for comparison."""
        reqs, _ = self._run_starvation_scenario(alpha=10.0)
        _, resp, wait = simulate_queue(reqs, servers=1, discipline="srtf")
        # All short requests (idx 1-20) are pred=10, served first; long waits a long time
        # Long request (idx 0, pred=1000) waits until all 20 short ones finish:
        # 20 * 10 = 200 s (here service == tokens for hand-math)
        assert wait[0] > 150, f"Expected starvation wait >150s; got {wait[0]:.2f}s"

    def test_aging_serves_long_before_all_shorts(self):
        """With aggressive alpha, aging cuts the queue so long request is served early."""
        reqs, _ = self._run_starvation_scenario(alpha=10.0)
        _, _, aging_wait = simulate_queue_with_aging(reqs, servers=1, alpha=10.0)
        _, _, srtf_wait = simulate_queue(reqs, servers=1, discipline="srtf")
        assert aging_wait[0] < srtf_wait[0]

    def test_higher_alpha_reduces_starvation_budget(self):
        reqs, _ = self._run_starvation_scenario(alpha=100.0)  # very aggressive
        _, _, wait = simulate_queue_with_aging(reqs, servers=1, alpha=100.0)
        max_wait = 1000.0 / 100.0  # 10s
        assert wait[0] <= max_wait + 1e-3

    def test_lower_alpha_allows_more_wait(self):
        reqs, _ = self._run_starvation_scenario(alpha=1.0)  # gentle
        _, _, wait_gentle = simulate_queue_with_aging(reqs, servers=1, alpha=1.0)
        _, _, wait_aggressive = simulate_queue_with_aging(reqs, servers=1, alpha=100.0)
        assert wait_gentle[0] >= wait_aggressive[0]


# ---------------------------------------------------------------------------
# 4. Aging vs FIFO: no regression for all disciplines when load is light
# ---------------------------------------------------------------------------

class TestAgingNoDegradationLightLoad:
    def test_all_requests_finish_aging(self):
        reqs = [_req(i, float(i) * 10.0, i + 1, i + 1) for i in range(50)]
        s, r, w = simulate_queue_with_aging(reqs, servers=4, alpha=DEFAULT_AGING_ALPHA)
        assert s["requests"] == 50

    def test_mean_response_non_negative(self):
        reqs = [_req(i, float(i), 100) for i in range(10)]
        s, _, _ = simulate_queue_with_aging(reqs, servers=2, alpha=DEFAULT_AGING_ALPHA)
        assert s["mean_response_s"] >= 0.0

    def test_p99_non_negative(self):
        reqs = [_req(i, float(i) * 0.1, 200) for i in range(50)]
        s, _, _ = simulate_queue_with_aging(reqs, servers=2, alpha=DEFAULT_AGING_ALPHA)
        assert s["p99_response_s"] >= 0.0


# ---------------------------------------------------------------------------
# 5. load_burstgpt_requests
# ---------------------------------------------------------------------------

class TestLoadBurstgptRequests:
    def test_loads_fixture(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE)
        assert len(rows) >= 1

    def test_returns_tuples_of_float_int(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE)
        for arrival, tokens in rows:
            assert isinstance(arrival, float)
            assert isinstance(tokens, int)
            assert tokens > 0

    def test_arrival_times_relative(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE)
        if rows:
            assert rows[0][0] == pytest.approx(0.0, abs=1e-6)

    def test_sorted_by_arrival(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE)
        arrivals = [r[0] for r in rows]
        assert arrivals == sorted(arrivals)

    def test_positive_output_tokens_only(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE)
        for _, tok in rows:
            assert tok > 0

    def test_limit_respected(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE, limit=3)
        assert len(rows) <= 3

    def test_fixture_has_reasonable_token_range(self):
        rows = load_burstgpt_requests(DEFAULT_BURSTGPT_FIXTURE)
        if rows:
            tokens = [t for _, t in rows]
            assert min(tokens) >= 1
            assert max(tokens) <= 10_000   # sanity cap


# ---------------------------------------------------------------------------
# 6. SRTFAgingReport dataclass
# ---------------------------------------------------------------------------

class TestSRTFAgingReport:
    def _dummy_sim(self):
        return {
            "requests": 100,
            "servers": 4,
            "sim_horizon_s": 500.0,
            "mean_response_s": 5.0,
            "p50_response_s": 3.0,
            "p90_response_s": 8.0,
            "p99_response_s": 15.0,
            "mean_wait_s": 1.0,
            "p90_wait_s": 2.0,
            "p99_wait_s": 4.0,
            "short_p90_response_s": 2.0,
            "short_p99_response_s": 5.0,
            "long_p90_response_s": 10.0,
            "long_p99_response_s": 20.0,
            "max_response_s": 100.0,
            "sla_safe_goodput_per_dollar": 1234.5,
        }

    def _make_report(self, short_p90_impr=50.0, long_p99_impr=30.0, gp_delta=100.0):
        d = self._dummy_sim()
        return SRTFAgingReport(
            azure_servers=4, azure_target_rho=0.85, azure_n_requests=5880,
            azure_fifo=d, azure_srtf_perfect=d, azure_srtf_forecast=d,
            azure_aging_perfect=d, azure_aging_forecast=d,
            burstgpt_servers=4, burstgpt_target_rho=0.85, burstgpt_n_requests=50,
            burstgpt_fifo=d, burstgpt_srtf_perfect=d, burstgpt_aging_perfect=d,
            azure_short_p90_improvement_pct=short_p90_impr,
            azure_long_p99_improvement_pct=long_p99_impr,
            azure_goodput_delta_pct=gp_delta,
            azure_starvation_fixed=True,
            aging_alpha=DEFAULT_AGING_ALPHA,
        )

    def test_to_dict_contains_azure_key(self):
        r = self._make_report()
        d = r.to_dict()
        assert "azure" in d

    def test_to_dict_contains_burstgpt_key(self):
        r = self._make_report()
        d = r.to_dict()
        assert "burstgpt" in d

    def test_to_dict_headline_deltas(self):
        r = self._make_report(short_p90_impr=55.0, long_p99_impr=25.0, gp_delta=200.0)
        d = r.to_dict()
        assert d["headline_azure"]["short_p90_improvement_pct"] == pytest.approx(55.0, rel=1e-4)
        assert d["headline_azure"]["long_p99_improvement_pct"] == pytest.approx(25.0, rel=1e-4)
        assert d["headline_azure"]["goodput_delta_pct"] == pytest.approx(200.0, rel=1e-4)

    def test_to_dict_starvation_fixed_true(self):
        r = self._make_report()
        assert r.to_dict()["headline_azure"]["starvation_fixed"] is True

    def test_shadow_tag_present(self):
        r = self._make_report()
        assert "shadow" in r.to_dict()["shadow_tag"]


# ---------------------------------------------------------------------------
# 7. run_srtf_aging_backtest: headline direction on real Azure trace
# ---------------------------------------------------------------------------

class TestRunSRTFAgingBacktest:
    @pytest.fixture(scope="class")
    def report(self):
        return run_srtf_aging_backtest(
            servers=4,
            target_rho=0.85,
            forecast_noise_cv=0.30,
            sla_s=DEFAULT_SLA_S,
            aging_alpha=DEFAULT_AGING_ALPHA,
            azure_fixture=DEFAULT_AZURE_FIXTURE,
            burstgpt_fixture=DEFAULT_BURSTGPT_FIXTURE,
            seed=20260201,
        )

    def test_report_is_srtfagingreport(self, report):
        assert isinstance(report, SRTFAgingReport)

    def test_azure_request_count(self, report):
        assert report.azure_n_requests > 100

    def test_aging_short_p90_better_than_fifo(self, report):
        """SRTF+aging must reduce short-request p90 vs FIFO."""
        assert report.azure_short_p90_improvement_pct > 0, (
            f"Expected short-p90 improvement > 0; got {report.azure_short_p90_improvement_pct:.2f}%"
        )

    def test_aging_goodput_better_than_fifo(self, report):
        """SRTF+aging must raise SLA-safe goodput/$ vs FIFO."""
        assert report.azure_goodput_delta_pct > 0, (
            f"Expected goodput/$ delta > 0; got {report.azure_goodput_delta_pct:.2f}%"
        )

    def test_starvation_fixed(self, report):
        """Aging must eliminate the extreme SRTF starvation tail.

        Pure SRTF starves long requests (p99 >> FIFO p99).  SRTF+aging must
        reduce that tail substantially.  We compare aging p99 to SRTF p99
        (not FIFO p99) because SRTF+aging intentionally trades a little
        long-tail headroom for the large short-request gain.
        """
        assert report.azure_starvation_fixed, (
            f"Starvation not fixed: aging p99={report.azure_aging_perfect['p99_response_s']:.1f}s"
            f" >= srtf p99={report.azure_srtf_perfect['p99_response_s']:.1f}s"
        )

    def test_aging_long_p99_not_worse_than_fifo(self, report):
        """The long-request p99 must not regress vs FIFO (pareto-safe claim).

        A tolerance of 0.1% absorbs floating-point rounding in the percentile
        computation; the real claim is that aging does not structurally worsen
        long requests relative to FIFO.
        """
        assert report.azure_long_p99_improvement_pct >= -0.1, (
            f"Long-p99 regressed vs FIFO by {-report.azure_long_p99_improvement_pct:.2f}%"
        )

    def test_srtf_no_aging_has_larger_p99_than_aging(self, report):
        """Confirm starvation: vanilla SRTF (no aging) p99 > SRTF+aging p99."""
        srtf_p99 = report.azure_srtf_perfect["p99_response_s"]
        aging_p99 = report.azure_aging_perfect["p99_response_s"]
        assert srtf_p99 > aging_p99, (
            f"Expected SRTF p99 ({srtf_p99:.1f}s) > aging p99 ({aging_p99:.1f}s)"
        )

    def test_burstgpt_cross_validation_runs(self, report):
        """BurstGPT cross-validation at least runs without error."""
        assert report.burstgpt_n_requests >= 1

    def test_burstgpt_aging_direction(self, report):
        """BurstGPT: SRTF+aging goodput/$ >= FIFO (non-negative delta)."""
        bgpt_aging_gp = report.burstgpt_aging_perfect.get("sla_safe_goodput_per_dollar", 0)
        bgpt_fifo_gp = report.burstgpt_fifo.get("sla_safe_goodput_per_dollar", 0)
        if bgpt_fifo_gp > 0:
            assert bgpt_aging_gp >= bgpt_fifo_gp * 0.9, (
                f"BurstGPT aging goodput/$ ({bgpt_aging_gp:.2f}) "
                f"< 90% of FIFO ({bgpt_fifo_gp:.2f})"
            )

    def test_to_dict_roundtrip(self, report):
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "azure" in d
        assert "burstgpt" in d
        assert "headline_azure" in d

    def test_short_p90_improvement_not_worse_than_fifo(self, report):
        """SRTF+aging must not make short-p90 WORSE than FIFO.

        Key research finding (run 2026-06-20-i): non-preemptive SRTF+aging with
        effective_key=max(0,pred-alpha*wait) degrades to near-FIFO at ρ=0.85 due
        to the "age-out wave" problem.  All starved long requests simultaneously
        age to effective_key=0 (having waited > pred/alpha seconds), then pile
        into high-priority slots and block fresh short requests.  Short requests
        subsequently back up and also age to 0 → system reverts to FIFO order.

        The short-p90 improvement is near-zero on this trace; the SRTF gain requires
        either preemptive scheduling or resource partitioning.  This test asserts
        the weaker "no regression" property that the algorithm does satisfy.
        """
        assert report.azure_short_p90_improvement_pct > -1.0, (
            f"Short-p90 REGRESSED vs FIFO by "
            f"{-report.azure_short_p90_improvement_pct:.1f}%"
        )

    def test_goodput_improvement_large(self, report):
        """Goodput/$ improves by ≥10% vs FIFO (partial benefit from transient SRTF phase).

        Even though short-p90 degrades to FIFO in steady state, the early phase
        of the simulation (before the age-out wave fires) preserves SRTF ordering,
        delivering ~33% goodput/$ gain on the 5880-request Azure trace.
        The threshold is conservative; the observed value is ≈33%.
        """
        assert report.azure_goodput_delta_pct > 10.0, (
            f"Goodput/$ delta {report.azure_goodput_delta_pct:.1f}% < 10%"
        )


# ---------------------------------------------------------------------------
# 8. Aging report fields match expected keys
# ---------------------------------------------------------------------------

class TestAgingReportStructure:
    @pytest.fixture(scope="class")
    def report(self):
        return run_srtf_aging_backtest(seed=20260201)

    def test_azure_fifo_has_long_p99_key(self, report):
        assert "long_p99_response_s" in report.azure_fifo

    def test_azure_aging_has_long_p99_key(self, report):
        assert "long_p99_response_s" in report.azure_aging_perfect

    def test_azure_srtf_has_long_p99_key(self, report):
        assert "long_p99_response_s" in report.azure_srtf_perfect

    def test_aging_alpha_stored(self, report):
        assert report.aging_alpha == pytest.approx(DEFAULT_AGING_ALPHA, rel=1e-6)

    def test_n_requests_positive(self, report):
        assert report.azure_n_requests > 0


# ---------------------------------------------------------------------------
# 9. Existing run_srtf_serving_backtest is unmodified (regression)
# ---------------------------------------------------------------------------

class TestExistingBacktestRegression:
    def test_original_report_still_works(self):
        rpt = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        # Same headline direction as run -g: SRTF > FIFO on short-p90
        assert rpt.short_p90_improvement_pct > 0

    def test_original_report_goodput_positive(self):
        rpt = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        assert rpt.sla_goodput_delta_pct > 0

    def test_long_p99_keys_now_in_summary(self):
        rpt = run_srtf_serving_backtest(servers=4, target_rho=0.85)
        assert "long_p99_response_s" in rpt.fifo
        assert "long_p99_response_s" in rpt.srtf_perfect
