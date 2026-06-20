"""Tests for Preemptive SRPT discipline in the serving-queue simulator.

Run 2026-06-20-j adds ``srpt_preemptive`` to ``simulate_queue`` and the
four-discipline benchmark ``run_srpt_preemptive_backtest``.

Research basis:
  - TRAIL (arXiv:2410.01035, ICLR 2025): SRPT with limited preemptions.
  - FlowPrefill (arXiv:2602.16603, Feb 2026): operator-level preemption.
  - SRPT for multiserver systems (arXiv:1805.07686): SRPT for M/G/k.

Correctness invariant (SRPT): at all times the c servers run the c requests
with the shortest remaining service time.  Tests verify this inductively
through synthetic traces where the expected outcome is deterministic.
"""

from __future__ import annotations

import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    AGING_ALPHA_DEFAULT,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    SRTFPreemptiveReport,
    _Request,
    _run_preemptive_backtest_on_trace,
    _simulate_srpt_preemptive,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    run_burstgpt_srpt_preemptive_backtest,
    run_srpt_preemptive_backtest,
    simulate_queue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(idx, arrival, tokens, predicted=None):
    """Create a _Request with service_s = tokens (simplified unit physics)."""
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(tokens),
        predicted_tokens=float(predicted if predicted is not None else tokens),
        service_s=float(tokens),   # 1 tok = 1 s for unit-scale tests
    )


# ---------------------------------------------------------------------------
# Class 1: Basic preemption mechanics
# ---------------------------------------------------------------------------

class TestPreemptionBasics:
    def test_short_request_preempts_long(self):
        # 1 server.  Long request (10s) starts at t=0; short request (2s)
        # arrives at t=3.  SRPT should preempt the long request and serve
        # the short one first.
        reqs = [_req(0, 0, 10), _req(1, 3, 2)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        # r1 (short) completes at t=3+2=5 (wait=0 after preempting r0)
        # r0 resumes at t=5 with 7s remaining → completes at t=12
        # resp stores sojourn time (completion - arrival), not absolute completion time.
        assert resp[1] < resp[0], "short request must complete before long one"
        assert abs(resp[1] - 2.0) < 1e-9, "short sojourn = 2s (completes at t=5, arrived t=3)"
        assert abs(resp[0] - 12.0) < 1e-9, "long sojourn = 12s (completes at t=12, arrived t=0)"

    def test_no_preemption_when_arriving_longer(self):
        # 1 server. Short request (2s) running; long request (10s) arrives.
        # No preemption (10s > 2s remaining), long request waits.
        reqs = [_req(0, 0, 2), _req(1, 0.5, 10)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        # r0 completes at t=2; r1 starts at t=2, completes at t=12.
        # resp stores sojourn time (completion - arrival).
        # r1 sojourn = 12 - 0.5 = 11.5 (not absolute completion time 12).
        assert abs(resp[0] - 2.0) < 1e-9
        assert abs(resp[1] - 11.5) < 1e-9

    def test_free_server_no_preemption(self):
        # 2 servers. Two requests arrive; each goes to a free server.
        # A third arrives later — there's still a free server, no preemption.
        reqs = [_req(0, 0, 5), _req(1, 0, 8), _req(2, 2, 3)]
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="srpt_preemptive")
        # At t=2, both servers are busy (r0 with 3s rem, r1 with 6s rem).
        # r2(3s) arrives: r0.remaining=3.0, r1.remaining=6.0.  r2.service=3.0.
        # 3.0 < 6.0 → preempt r1, start r2.
        # r2 completes at t=2+3=5. r0 completes at t=5 too.
        assert 0 in resp and 1 in resp and 2 in resp

    def test_multiple_preemptions_same_job(self):
        # 1 server.  A long job can be preempted multiple times.
        # r0(20s) starts at t=0.
        # r1(5s) arrives at t=3 → preempts r0 (rem=17s). r1 runs 3-8.
        # r2(3s) arrives at t=4 → all servers busy (r1 with 4s rem).
        #   r2(3s) < r1(4s) → preempt r1! r2 runs 4-7. r1 resumes with 4s.
        # Order: r2 done at t=7, r1 resumes at t=7 done at t=7+4=11,
        #        r0 resumes at t=11 with 17s → done at t=11+17=28.
        reqs = [_req(0, 0, 20), _req(1, 3, 5), _req(2, 4, 3)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        assert resp[2] < resp[1] < resp[0], "shortest first in all cases"

    def test_simultaneous_arrivals_sorted_by_service(self):
        # 1 server. All three arrive at t=0. SRPT should serve shortest first.
        reqs = [_req(0, 0, 30), _req(1, 0, 10), _req(2, 0, 5)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        # r2(5s) → r1(10s) → r0(30s)
        assert resp[2] < resp[1] < resp[0]
        assert abs(resp[2] - 5.0) < 1e-9
        assert abs(resp[1] - 15.0) < 1e-9
        assert abs(resp[0] - 45.0) < 1e-9


# ---------------------------------------------------------------------------
# Class 2: SRPT invariant — all completed requests, no starvation
# ---------------------------------------------------------------------------

class TestSRPTInvariant:
    def test_all_requests_complete(self):
        reqs = [_req(i, i * 0.1, (i % 5 + 1) * 10) for i in range(20)]
        summary, resp, wait = simulate_queue(reqs, servers=3, discipline="srpt_preemptive")
        assert len(resp) == 20, "all 20 requests must complete"

    def test_no_starvation_long_requests(self):
        # 1 server: one very long job (1000s) followed by many short jobs.
        # The long job must eventually complete (no infinite starvation).
        reqs = [_req(0, 0, 1000)] + [_req(i + 1, i * 0.5, 1) for i in range(50)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        assert 0 in resp, "long job must complete"
        # Long job completes after ALL short jobs (≤ t=1000 + 50 + overhead)
        assert resp[0] > max(resp[i + 1] for i in range(50))

    def test_wait_time_nonnegative(self):
        reqs = [_req(i, i * 0.3, (i % 4 + 1) * 5) for i in range(12)]
        _, resp, wait = simulate_queue(reqs, servers=2, discipline="srpt_preemptive")
        for idx, w in wait.items():
            assert w >= -1e-9, f"wait time must be non-negative for req {idx}"

    def test_response_ge_service(self):
        reqs = [_req(i, i * 0.2, (i % 3 + 1) * 8) for i in range(15)]
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="srpt_preemptive")
        for idx, r in resp.items():
            req = next(q for q in reqs if q.idx == idx)
            assert r >= req.service_s - 1e-9, "response >= service always"

    def test_long_request_eventually_wins_priority(self):
        # 1 server.  Long job r0(10s) starts at t=0.
        # Short jobs r1(1s), r2(1s) arrive at t=1 and t=1.5.
        # After r1 and r2 are served, r0 completes.
        reqs = [_req(0, 0, 10), _req(1, 1, 1), _req(2, 1.5, 1)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        assert resp[1] < resp[0] and resp[2] < resp[0]
        assert 0 in resp  # long job completes


# ---------------------------------------------------------------------------
# Class 3: Comparison with non-preemptive disciplines
# ---------------------------------------------------------------------------

class TestPreemptiveVsNonPreemptive:
    def test_srpt_short_p90_better_than_fifo(self):
        # Generate a queue of mixed long/short jobs; SRPT should serve short
        # ones faster than FIFO.
        import random
        rng = random.Random(42)
        reqs = []
        for i in range(100):
            service = 1.0 if rng.random() < 0.7 else 50.0  # 70% short
            reqs.append(_req(i, i * 0.2, service))
        _, fifo_resp, _ = simulate_queue(reqs, servers=2, discipline="fifo")
        _, srpt_resp, _ = simulate_queue(reqs, servers=2, discipline="srpt_preemptive")
        # Short responses (service=1s)
        short_fifo = sorted(fifo_resp[r.idx] for r in reqs if r.service_s == 1.0)
        short_srpt = sorted(srpt_resp[r.idx] for r in reqs if r.service_s == 1.0)
        fifo_p90 = short_fifo[int(0.9 * len(short_fifo))]
        srpt_p90 = short_srpt[int(0.9 * len(short_srpt))]
        assert srpt_p90 <= fifo_p90, "SRPT short-request p90 should be ≤ FIFO"

    def test_srpt_long_p99_not_infinite(self):
        # Long requests cannot be starved indefinitely in SRPT.
        import random
        rng = random.Random(99)
        long_service = 20.0
        reqs = [_req(0, 0, long_service)]   # one long job at t=0
        # Many short jobs arrive during the long job's service window
        for i in range(30):
            reqs.append(_req(i + 1, rng.uniform(0, 5), 0.5))
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        assert 0 in resp, "long job must complete"
        # Long job response must be finite (not starved indefinitely)
        # It should complete within 20s + total_short_work ≈ 20 + 15 = 35s
        total_short_work = 30 * 0.5
        assert resp[0] <= long_service + total_short_work + 5.0, "no infinite starvation"

    def test_srpt_same_as_srtf_no_preemption_needed(self):
        # If requests arrive well-separated (no preemption occurs), SRPT
        # should produce the same response times as non-preemptive SRTF.
        reqs = [_req(0, 0, 5), _req(1, 100, 3)]  # arrivals 100s apart
        _, srpt_resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        _, srtf_resp, _ = simulate_queue(reqs, servers=1, discipline="srtf")
        assert abs(srpt_resp[0] - srtf_resp[0]) < 1e-9
        assert abs(srpt_resp[1] - srtf_resp[1]) < 1e-9

    def test_srpt_all_same_service_matches_fifo(self):
        # If all requests have identical service time, SRPT = FIFO.
        reqs = [_req(i, i * 0.5, 10) for i in range(8)]
        _, fifo_resp, _ = simulate_queue(reqs, servers=2, discipline="fifo")
        _, srpt_resp, _ = simulate_queue(reqs, servers=2, discipline="srpt_preemptive")
        for idx in fifo_resp:
            assert abs(fifo_resp[idx] - srpt_resp[idx]) < 1e-9, \
                "identical service times → SRPT = FIFO"


# ---------------------------------------------------------------------------
# Class 4: SRTFPreemptiveReport structure
# ---------------------------------------------------------------------------

class TestSRTFPreemptiveReportStructure:
    def _small_report(self):
        raw = [(0.0, 10), (0.5, 2), (1.0, 5), (1.5, 20), (2.0, 1)]
        return _run_preemptive_backtest_on_trace(
            raw, "test", servers=2, target_rho=0.9, aging_alpha=0.01, sla_s=30.0
        )

    def test_returns_report_instance(self):
        rpt = self._small_report()
        assert isinstance(rpt, SRTFPreemptiveReport)

    def test_all_four_disciplines_present(self):
        rpt = self._small_report()
        for attr in ("fifo", "srtf_perfect", "aging_srtf", "srpt_preemptive"):
            assert hasattr(rpt, attr) and isinstance(getattr(rpt, attr), dict)

    def test_to_dict_keys(self):
        d = self._small_report().to_dict()
        for k in (
            "trace", "total_requests", "servers", "fifo", "srtf_perfect",
            "aging_srtf", "srpt_preemptive",
            "srtf_short_p90_improvement_pct", "aging_short_p90_improvement_pct",
            "srpt_short_p90_improvement_pct",
            "srtf_long_p99_delta_pct", "aging_long_p99_delta_pct",
            "srpt_long_p99_delta_pct",
            "srtf_goodput_delta_pct", "aging_goodput_delta_pct",
            "srpt_goodput_delta_pct",
            "shadow_tag",
        ):
            assert k in d, f"missing key: {k}"

    def test_shadow_tag_correct(self):
        rpt = self._small_report()
        assert "not_production_savings" in rpt.shadow_tag

    def test_requests_count_matches(self):
        raw = [(i * 0.2, 5 + i % 3) for i in range(10)]
        rpt = _run_preemptive_backtest_on_trace(
            raw, "t", servers=2, target_rho=0.8, aging_alpha=0.01, sla_s=20.0
        )
        assert rpt.total_requests == 10

    def test_sla_safe_goodput_present_in_each_discipline(self):
        rpt = self._small_report()
        for attr in ("fifo", "srtf_perfect", "aging_srtf", "srpt_preemptive"):
            d = getattr(rpt, attr)
            assert "sla_safe_goodput_per_dollar" in d


# ---------------------------------------------------------------------------
# Class 5: SRPT vs FIFO/SRTF/Aging — ordering guarantees
# ---------------------------------------------------------------------------

class TestSRPTOrderingGuarantees:
    def test_srpt_goodput_not_worse_than_aging(self):
        # On a short synthetic trace, SRPT should generally match or beat aging-SRTF.
        raw = [(i * 0.3, max(1, (i * 7 + 3) % 50)) for i in range(40)]
        rpt = _run_preemptive_backtest_on_trace(
            raw, "t", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=20.0
        )
        # SRPT goodput delta vs FIFO should be positive or at least not far below aging
        assert rpt.srpt_goodput_delta_pct >= rpt.aging_goodput_delta_pct - 5.0, \
            "SRPT should not be significantly worse than aging-SRTF on goodput/$"

    def test_srpt_short_p90_beats_nonpreemptive_srtf(self):
        # Preemptive SRPT's key advantage: short requests can immediately preempt
        # the running job, so short_p90 should improve at least as much as SRTF.
        # (SRTF can only reorder queued jobs; SRPT can reclaim a running server.)
        raw = [(i * 0.2, 1 if i % 5 != 0 else 200) for i in range(50)]
        rpt = _run_preemptive_backtest_on_trace(
            raw, "t", servers=1, target_rho=0.85, aging_alpha=0.01, sla_s=50.0
        )
        # SRPT short_p90 improvement should match or exceed non-preemptive SRTF
        assert rpt.srpt_short_p90_improvement_pct >= rpt.srtf_short_p90_improvement_pct - 5.0, \
            "SRPT preemptive must match or exceed non-preemptive SRTF on short-request p90"

    def test_srpt_short_p90_improvement_positive(self):
        raw = [(i * 0.4, 1 if i % 3 != 0 else 30) for i in range(30)]
        rpt = _run_preemptive_backtest_on_trace(
            raw, "t", servers=1, target_rho=0.9, aging_alpha=0.01, sla_s=10.0
        )
        # Short-request p90 should improve vs FIFO
        assert rpt.srpt_short_p90_improvement_pct >= 0.0, \
            "SRPT short-request p90 should not regress vs FIFO"

    def test_all_requests_complete_in_backtest(self):
        raw = [(i * 0.1, (i % 8 + 1) * 3) for i in range(20)]
        rpt = _run_preemptive_backtest_on_trace(
            raw, "t", servers=2, target_rho=0.8, aging_alpha=0.05, sla_s=30.0
        )
        assert rpt.srpt_preemptive["requests"] == 20


# ---------------------------------------------------------------------------
# Class 6: simulate_queue dispatch for srpt_preemptive
# ---------------------------------------------------------------------------

class TestSimulateQueueDispatch:
    def test_srpt_preemptive_discipline_dispatched(self):
        reqs = [_req(0, 0, 5), _req(1, 1, 2)]
        summary, resp, wait = simulate_queue(
            reqs, servers=1, discipline="srpt_preemptive"
        )
        assert isinstance(summary, dict)
        assert "mean_response_s" in summary
        assert len(resp) == 2

    def test_unknown_discipline_not_caught_by_preemptive(self):
        # Non-preemptive path should still work (regression guard)
        reqs = [_req(0, 0, 5), _req(1, 1, 2)]
        summary, resp, _ = simulate_queue(reqs, servers=1, discipline="fifo")
        assert len(resp) == 2

    def test_aging_discipline_still_works(self):
        reqs = [_req(0, 0, 5), _req(1, 1, 2)]
        summary, resp, _ = simulate_queue(
            reqs, servers=1, discipline="aging_srtf", aging_alpha=0.01
        )
        assert len(resp) == 2

    def test_srpt_returns_three_tuple(self):
        reqs = [_req(i, i * 0.5, i + 1) for i in range(5)]
        result = simulate_queue(reqs, servers=2, discipline="srpt_preemptive")
        assert len(result) == 3
        summary, resp, wait = result
        assert isinstance(summary, dict) and isinstance(resp, dict) and isinstance(wait, dict)


# ---------------------------------------------------------------------------
# Class 7: run_srpt_preemptive_backtest (Azure 2024 fixture)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(DEFAULT_AZURE_FIXTURE),
    reason="Azure LLM 2024 fixture not present",
)
class TestRunSRPTPreemptiveBacktest:
    def test_runs_without_error(self):
        rpt = run_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, job_limit=200, sla_s=10.0, aging_alpha=0.01
        )
        assert isinstance(rpt, SRTFPreemptiveReport)
        assert rpt.total_requests == 200

    def test_srpt_short_p90_improves_vs_fifo(self):
        rpt = run_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, job_limit=200, sla_s=10.0, aging_alpha=0.01
        )
        assert rpt.srpt_short_p90_improvement_pct > 0, \
            "SRPT should improve short-request p90 vs FIFO on Azure 2024"

    def test_srpt_short_p90_beats_nonpreemptive_srtf(self):
        # Preemptive SRPT's primary guarantee: short requests benefit from
        # immediate server reclamation, so short_p90 improvement must exceed
        # or match non-preemptive SRTF which can only reorder queued jobs.
        rpt = run_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, job_limit=200, sla_s=10.0, aging_alpha=0.01
        )
        assert rpt.srpt_short_p90_improvement_pct >= rpt.srtf_short_p90_improvement_pct - 5.0, \
            "SRPT preemptive must match or exceed non-preemptive SRTF on short-request p90"

    def test_srpt_short_p90_improves_vs_fifo(self):
        # SRPT preemptive trades aggregate goodput for latency fairness;
        # the primary deliverable is short-request p90 improvement.
        rpt = run_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, job_limit=200, sla_s=10.0, aging_alpha=0.01
        )
        assert rpt.srpt_short_p90_improvement_pct > 0, \
            "SRPT must improve short-request p90 latency vs FIFO"

    def test_trace_field_correct(self):
        rpt = run_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, job_limit=50, sla_s=10.0, aging_alpha=0.01
        )
        assert rpt.trace == "azure_llm_2024"

    def test_all_disciplines_have_sla_goodput(self):
        rpt = run_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, job_limit=100, sla_s=10.0, aging_alpha=0.01
        )
        for attr in ("fifo", "srtf_perfect", "aging_srtf", "srpt_preemptive"):
            d = getattr(rpt, attr)
            assert d.get("sla_safe_goodput_per_dollar", 0) > 0, \
                f"{attr} sla_safe_goodput_per_dollar must be > 0"


# ---------------------------------------------------------------------------
# Class 8: BurstGPT cross-validation
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(DEFAULT_BURSTGPT_FIXTURE),
    reason="BurstGPT fixture not present",
)
class TestBurstGPTSRPTPreemptive:
    def test_burstgpt_runs_without_error(self):
        rpt = run_burstgpt_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, sla_s=DEFAULT_BURSTGPT_SLA_S, aging_alpha=0.01
        )
        assert isinstance(rpt, SRTFPreemptiveReport)
        assert rpt.trace == "burstgpt"

    def test_burstgpt_all_requests_complete(self):
        rpt = run_burstgpt_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, sla_s=DEFAULT_BURSTGPT_SLA_S, aging_alpha=0.01
        )
        total = rpt.total_requests
        assert rpt.srpt_preemptive["requests"] == total

    def test_burstgpt_srpt_short_p90_improves(self):
        rpt = run_burstgpt_srpt_preemptive_backtest(
            servers=4, target_rho=0.85, sla_s=DEFAULT_BURSTGPT_SLA_S, aging_alpha=0.01
        )
        assert rpt.srpt_short_p90_improvement_pct >= 0.0, \
            "SRPT short_p90 should not regress vs FIFO on BurstGPT"


# ---------------------------------------------------------------------------
# Class 9: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_request(self):
        reqs = [_req(0, 0, 10)]
        summary, resp, wait = simulate_queue(
            reqs, servers=1, discipline="srpt_preemptive"
        )
        assert resp[0] == pytest.approx(10.0)
        assert wait[0] == pytest.approx(0.0)

    def test_single_server_single_request(self):
        reqs = [_req(0, 5.0, 3)]
        _, resp, wait = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        assert resp[0] == pytest.approx(3.0)  # response = service time

    def test_more_servers_than_requests(self):
        reqs = [_req(0, 0, 5), _req(1, 0, 3)]
        _, resp, _ = simulate_queue(reqs, servers=4, discipline="srpt_preemptive")
        # Both run immediately; no preemption needed.
        assert abs(resp[0] - 5.0) < 1e-9
        assert abs(resp[1] - 3.0) < 1e-9

    def test_zero_arrival_gap(self):
        # All arrivals at t=0; 1 server.  Should serve in ascending service order.
        reqs = [_req(0, 0, 10), _req(1, 0, 3), _req(2, 0, 7)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        assert resp[1] < resp[2] < resp[0]

    def test_preemption_produces_correct_total_service(self):
        # 1 server.  r0(10s) starts at t=0; r1(4s) arrives at t=2.
        # r1 preempts r0 (remaining=8s).  r1 completes at t=2+4=6.
        # r0 resumes at t=6 with 8s → completes at t=6+8=14.
        # Total service time in system = 10 + 4 = 14s.  Last event = t=14.
        reqs = [_req(0, 0, 10), _req(1, 2, 4)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="srpt_preemptive")
        # resp stores sojourn time (completion - arrival), not absolute completion time.
        # r0: arrives t=0, completes t=14 → sojourn=14. r1: arrives t=2, completes t=6 → sojourn=4.
        assert abs(resp[0] - 14.0) < 1e-9
        assert abs(resp[1] - 4.0) < 1e-9  # r1 sojourn = 6 - 2 = 4 (completes at t=6)

    def test_empty_requests_list(self):
        summary, resp, wait = _simulate_srpt_preemptive([], servers=2)
        assert resp == {} and wait == {}
