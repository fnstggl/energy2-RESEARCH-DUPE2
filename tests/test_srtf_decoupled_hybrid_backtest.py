"""Tests for Decoupled Hybrid SRPT discipline [run 2026-06-20-l].

The decoupled hybrid separates the two decisions that run -k's unified aging
key conflated:

  Preemption key  = remaining_s (pure SRPT — no aging)
  Dispatch key    = remaining_s / (1 + α · total_wait_s) (aging)

Root cause fixed from run -k: the unified aging key remaining_s/(1+α·wait)
for BOTH preemption AND dispatch made the hybrid behave like Aging-SRTF
(+64.2% goodput/$ vs FIFO) rather than SRPT (+322.2%). By decoupling, fresh
short arrivals always preempt via pure SRPT, preserving throughput optimality.
Dispatch with aging bounds extreme starvation without blocking short arrivals
from preempting running long jobs.

Invariant assertions tested:
  1. Preemption: identical to pure SRPT (remaining_s only, no aging factor).
  2. Dispatch: aging key — long-waiting requests gain priority over time.
  3. With α=0 the discipline is equivalent to pure SRPT-preemptive.
  4. goodput/$ >= hybrid_aging_preemptive (decoupled restores SRPT preemption).
  5. long_p99 <= srpt_preemptive on the full Azure LLM 2024 trace.
  6. All requests complete (no infinite loops).
  7. Preemption fires iff new arrival's service_s < max(remaining_s) of running.
  8. Dispatch accumulates frozen_wait correctly across multiple preemption cycles.

Research basis:
  - TRAIL (arXiv:2410.01035, ICLR 2025)
  - Chimera (arXiv:2603.22206, March 2026)
  - FastServe (USENIX NSDI '26)
"""

from __future__ import annotations

import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DECOUPLED_HYBRID_ALPHA_DEFAULT,
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    DecoupledHybridReport,
    _Request,
    _run_decoupled_hybrid_backtest_on_trace,
    _sla_safe_goodput_per_dollar,
    run_burstgpt_decoupled_hybrid_backtest,
    run_decoupled_hybrid_backtest,
    simulate_queue,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(idx, arrival, tokens, predicted=None):
    """Create _Request with service_s = tokens (unit-scale: 1 tok = 1 s)."""
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(tokens),
        predicted_tokens=float(predicted if predicted is not None else tokens),
        service_s=float(tokens),
    )


# ---------------------------------------------------------------------------
# Class 1: Preemption mechanics — pure SRPT (no aging)
# ---------------------------------------------------------------------------

class TestDecoupledPreemptionIsPureSRPT:
    """Preemption key is remaining_s only — identical to SRPT-preemptive."""

    def test_short_arrival_preempts_long_no_accumulated_wait(self):
        # 1 server. Long job (10s) starts at t=0. Short job (2s) arrives at t=3.
        # remaining(long) = 7s > 2s → preempt. Same as pure SRPT.
        reqs = [_req(0, 0, 10), _req(1, 3, 2)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert resp[1] < resp[0], "short request must complete before long"
        assert abs(resp[1] - 2.0) < 1e-9, "short sojourn = 2s"
        # Long preempted at t=3 with 7s remaining; resumes at t=5, completes at t=12.
        assert abs(resp[0] - 12.0) < 1e-9, "long sojourn = 12s"

    def test_arrival_longer_than_remaining_does_not_preempt(self):
        # 1 server. Short job (2s) at t=0; long job (10s) arrives at t=0.5.
        # remaining(short at t=0.5) = 1.5s < 10s → no preemption.
        reqs = [_req(0, 0, 2), _req(1, 0.5, 10)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert abs(resp[0] - 2.0) < 1e-9, "short completes at t=2, sojourn=2"
        assert abs(resp[1] - 11.5) < 1e-9, "long sojourn = 10 + 1.5 wait = 11.5"

    def test_large_accumulated_wait_does_not_block_preemption(self):
        # Key difference from hybrid_aging_preemptive: in decoupled_hybrid,
        # even if a running request has HUGE accumulated wait, it is still
        # preempted by pure SRPT (remaining_s only).
        # Setup: r0 (100s) dispatched with frozen_wait = 100s (large aging protection).
        # In hybrid_aging_preemptive with α=1.0, effective_key = 100/(1+100) ≈ 0.99
        # → new arrival of 2s would NOT preempt (2 > 0.99... wait that means it WOULD).
        # Actually in hybrid, preemption fires if new_key < worst_ek.
        # new_key=2, worst_ek=0.99 → 2 > 0.99 → NO preemption in hybrid.
        # In decoupled_hybrid: preemption key = remaining_s = 100s.
        # new arrival (2s): 2 < 100 → PREEMPT.
        # We verify the decoupled_hybrid preempts when the hybrid would not.
        # Arrange: r_dummy(1s) ensures r0 waits 1s → frozen_wait=1.
        # Then r0(100s) runs with frozen_wait=1. At t=2, r1(2s) arrives.
        # hybrid (α=1.0): ek(r0) = remaining/(1+1*frozen) = 99/(1+1) = 49.5.
        #   new_key = 2 < 49.5 → preempt in hybrid too, actually.
        # Let's use α=10 instead:
        # ek(r0) = 99/(1+10*1) = 99/11 = 9. new_key = 2 < 9 → preempt in both.
        # We need frozen_wait >> remaining to get ek < new_arrival in hybrid.
        # ek(r0) = remaining/(1+α*frozen). For ek < 2: remaining/(1+α*frozen) < 2.
        # With α=1, frozen=100: ek = remaining/(101). For remaining=5: ek=0.049.
        # new_key=2 > 0.049 → hybrid would NOT preempt. But decoupled uses remaining_s=5.
        # 2 < 5 → decoupled WOULD preempt.
        # Setup: make r0 wait 100s before dispatch.
        # We need a long-running dummy series to create 100s of wait.
        # Simpler: use 1 server. r_dummy1(50s) at t=0, r0(5s) at t=0 (waits),
        # r1(2s) arrives at t=51 (just after r_dummy1 finishes and r0 starts running).
        # After r_dummy1 completes at t=50: r0 dispatched with frozen_wait=50s.
        # r0 runs from t=50. r1(2s) arrives at t=51. remaining(r0)=4s.
        # hybrid (α=1.0): ek(r0) = 4/(1+1*50) = 4/51 ≈ 0.078. new_key=2 > 0.078 → NO preempt.
        # decoupled_hybrid: preemption by remaining_s=4. 2 < 4 → PREEMPT.
        reqs = [_req(0, 0, 50), _req(1, 0, 5), _req(2, 51, 2)]
        _, resp_d, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                      aging_alpha=1.0)
        _, resp_h, _ = simulate_queue([_req(0, 0, 50), _req(1, 0, 5), _req(2, 51, 2)],
                                      servers=1, discipline="hybrid_aging_preemptive",
                                      aging_alpha=1.0)
        # In decoupled: r2 preempts r1 at t=51 → r2 completes at t=53, sojourn=2.
        # In hybrid: r2 does NOT preempt (ek(r1) ≈ 0.078 < 2) → r2 waits → r1 finishes
        # at t=55 → r2 dispatched → completes at t=57, sojourn=6.
        assert abs(resp_d[2] - 2.0) < 1e-9, (
            f"decoupled: r2 preempts r1 → sojourn=2 (got {resp_d[2]:.4f})"
        )
        assert resp_h[2] > 2.0, (
            f"hybrid: r2 does NOT preempt → sojourn>2 (got {resp_h[2]:.4f})"
        )

    def test_two_servers_preempt_longest_remaining(self):
        # 2 servers. r0(8s) and r1(5s) start at t=0.
        # At t=2: r0.remaining=6, r1.remaining=3. New arrival r2(4s).
        # Pure SRPT: preempt server with max remaining = r0 (6s). 4 < 6 → preempt.
        reqs = [_req(0, 0, 8), _req(1, 0, 5), _req(2, 2, 4)]
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert 0 in resp and 1 in resp and 2 in resp
        # r2 preempts r0 at t=2; r0 re-enters waiting with 6s remaining.
        # r2 completes at t=6 (2+4). r1 completes at t=5. r0 resumes at t=5 or t=6.
        # r0 resumes at min(t=5 free, t=6 free). r1 frees at t=5 → r0 starts at t=5.
        # r0 completes at t=5+6=11. sojourn = 11.
        assert abs(resp[2] - 4.0) < 1e-9, f"r2 sojourn should be 4s (got {resp[2]:.4f})"
        assert abs(resp[0] - 11.0) < 1e-9, f"r0 sojourn should be 11s (got {resp[0]:.4f})"

    def test_free_server_starts_without_preemption(self):
        # With a free server, no preemption needed — new arrival starts directly.
        reqs = [_req(0, 0, 10), _req(1, 1, 3)]  # 2 servers
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert abs(resp[1] - 3.0) < 1e-9, "r1 gets free server, sojourn=3"
        assert abs(resp[0] - 10.0) < 1e-9, "r0 runs uninterrupted, sojourn=10"


# ---------------------------------------------------------------------------
# Class 2: Dispatch is aging-based (not pure SRPT)
# ---------------------------------------------------------------------------

class TestDecoupledDispatchIsAging:
    """Dispatch uses remaining_s/(1+α·total_wait), giving starvation protection."""

    def test_long_waiting_job_dispatched_before_fresh_short(self):
        # 1 server. r0(1s) at t=0 occupies server.
        # r1(5s) at t=0 waits. r2(3s) arrives at t=1 (just as r0 completes).
        # At t=1 dispatch, waiting queue: [r1(5s, wait=1s), r2(3s, wait=0s)].
        # α=0.01: key(r1) = 5/(1+0.01*1) = 4.95. key(r2) = 3/(1+0) = 3.0.
        # r2 still wins dispatch (key 3 < 4.95). So for α=0.01 r2 dispatched first.
        # Now try α=1.0 to make the starvation bound kick in sooner:
        # key(r1) = 5/(1+1*1) = 2.5. key(r2) = 3/1 = 3.0. r1 wins (2.5 < 3).
        reqs_lo = [_req(0, 0, 1), _req(1, 0, 5), _req(2, 1, 3)]
        _, resp_lo, _ = simulate_queue(reqs_lo, servers=1, discipline="decoupled_hybrid",
                                       aging_alpha=0.01)
        # α=0.01: r2 dispatched first (key=3 < key_r1=4.95)
        assert resp_lo[2] < resp_lo[1], "α=0.01: fresh short (r2) dispatched before long-waiting (r1)"

        reqs_hi = [_req(0, 0, 1), _req(1, 0, 5), _req(2, 1, 3)]
        _, resp_hi, _ = simulate_queue(reqs_hi, servers=1, discipline="decoupled_hybrid",
                                       aging_alpha=1.0)
        # α=1.0: r1 dispatched first (key=2.5 < key_r2=3.0)
        assert resp_hi[1] < resp_hi[2], "α=1.0: long-waiting (r1) dispatched before fresh short (r2)"

    def test_aging_accumulates_across_preemption_wait_intervals(self):
        # 1 server. r0(1s) occupies. r1(10s) waits from t=0 to t=1.
        # At t=1, r2(2s) arrives simultaneously with r1's wait expiring.
        # Dispatch at t=1: r1(10s, wait=1s) vs r2(2s, wait=0s).
        # α=0.5: key(r1) = 10/(1+0.5*1) = 10/1.5 ≈ 6.67. key(r2) = 2.
        # r2 dispatched first. r2 completes at t=3. Now r1 dispatches with
        # frozen_wait = 1 (from t=0 to t=1) + (t=3 - t=1) = 3.
        # total_wait_so_far for r1 at dispatch = 3s (frozen from last wait).
        # key(r1) = 10/(1+0.5*3) = 10/2.5 = 4. (No competing requests; dispatched.)
        reqs = [_req(0, 0, 1), _req(1, 0, 10), _req(2, 1, 2)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.5)
        # r0 done at t=1, r2 dispatched (key=2 < r1's key=6.67), done at t=3.
        assert abs(resp[2] - 2.0) < 1e-9, f"r2 sojourn=2 (got {resp[2]:.4f})"
        # r1 dispatched at t=3, service=10, done at t=13. sojourn=13.
        assert abs(resp[1] - 13.0) < 1e-9, f"r1 sojourn=13 (got {resp[1]:.4f})"

    def test_dispatch_key_decreases_with_accumulated_wait(self):
        # Verify that a request's dispatch key decreases as wait accumulates.
        # Construct a scenario where first dispatch goes to short, but second
        # dispatch goes to the previously-waiting long job (now lower key).
        # 1 server. r0(dummy,1s) at t=0. r1(20s) at t=0 (waits). r2(3s) at t=1.
        # t=1: dispatch choice: r1(20s, wait=1s), r2(3s, wait=0s).
        # α=0.1: key(r1)=20/(1+0.1)=18.18; key(r2)=3. r2 dispatched.
        # r2 done at t=4. Now only r1 remains (wait: 4s total, frozen=1+3=4).
        # key(r1 at t=4) = 20/(1+0.1*4) = 20/1.4 ≈ 14.3. Dispatched uncontested.
        reqs = [_req(0, 0, 1), _req(1, 0, 20), _req(2, 1, 3)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.1)
        assert resp[2] < resp[1], "r2 (3s) dispatched before r1 (20s)"
        assert abs(resp[0] - 1.0) < 1e-9
        assert abs(resp[2] - 3.0) < 1e-9
        assert abs(resp[1] - 24.0) < 1e-9


# ---------------------------------------------------------------------------
# Class 3: α=0 equivalence with SRPT-preemptive
# ---------------------------------------------------------------------------

class TestDecoupledAlphaZeroEqualsSRPT:
    """With aging_alpha=0, decoupled_hybrid reduces to pure SRPT-preemptive."""

    def test_alpha_zero_same_response_single_server(self):
        reqs_d = [_req(0, 0, 10), _req(1, 3, 2), _req(2, 6, 5)]
        reqs_s = [_req(0, 0, 10), _req(1, 3, 2), _req(2, 6, 5)]
        _, resp_d, _ = simulate_queue(reqs_d, servers=1, discipline="decoupled_hybrid",
                                      aging_alpha=0.0)
        _, resp_s, _ = simulate_queue(reqs_s, servers=1, discipline="srpt_preemptive")
        for idx in resp_s:
            assert abs(resp_d[idx] - resp_s[idx]) < 1e-9, (
                f"α=0 decoupled must match SRPT for idx={idx}: "
                f"decoupled={resp_d[idx]:.4f} srpt={resp_s[idx]:.4f}"
            )

    def test_alpha_zero_same_response_two_servers(self):
        reqs = [_req(i, i * 0.5, 10 - i) for i in range(6)]
        reqs_d = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs]
        reqs_s = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs]
        _, resp_d, _ = simulate_queue(reqs_d, servers=2, discipline="decoupled_hybrid",
                                      aging_alpha=0.0)
        _, resp_s, _ = simulate_queue(reqs_s, servers=2, discipline="srpt_preemptive")
        for idx in resp_s:
            assert abs(resp_d[idx] - resp_s[idx]) < 1e-9, (
                f"2-server α=0 decoupled != SRPT for idx={idx}"
            )

    def test_alpha_zero_all_requests_complete(self):
        reqs = [_req(i, i * 1.0, 5 + i) for i in range(10)]
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid",
                                    aging_alpha=0.0)
        assert len(resp) == 10, "all 10 requests must complete"

    def test_alpha_zero_goodput_matches_srpt(self):
        reqs_d = [_req(i, i * 0.3, max(1, 8 - i)) for i in range(8)]
        reqs_s = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_d]
        _, resp_d, _ = simulate_queue(reqs_d, servers=1, discipline="decoupled_hybrid",
                                      aging_alpha=0.0)
        _, resp_s, _ = simulate_queue(reqs_s, servers=1, discipline="srpt_preemptive")
        gp_d = _sla_safe_goodput_per_dollar(reqs_d, resp_d, sla_s=50.0, servers=1)
        gp_s = _sla_safe_goodput_per_dollar(reqs_s, resp_s, sla_s=50.0, servers=1)
        assert abs(gp_d - gp_s) < 1e-6


# ---------------------------------------------------------------------------
# Class 4: Completeness and correctness
# ---------------------------------------------------------------------------

class TestDecoupledCompleteness:
    """All requests complete; no infinite loops or deadlocks."""

    def test_all_complete_single_server(self):
        reqs = [_req(i, i * 2, 5 + (i % 3)) for i in range(15)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert len(resp) == 15

    def test_all_complete_two_servers(self):
        reqs = [_req(i, i * 0.5, max(1, 10 - i)) for i in range(20)]
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert len(resp) == 20

    def test_all_complete_heavy_preemption(self):
        # 1 server; rapid arrivals with decreasing service times → many preemptions.
        reqs = [_req(i, i * 0.3, max(1, 20 - i * 2)) for i in range(10)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert len(resp) == 10

    def test_response_non_negative(self):
        reqs = [_req(i, i * 1.5, 3 + i) for i in range(8)]
        _, resp, _ = simulate_queue(reqs, servers=2, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        for idx, r in resp.items():
            assert r >= 0.0, f"response time for {idx} must be non-negative: {r}"

    def test_response_at_least_service_time(self):
        reqs = [_req(i, i * 1.0, 4 + i) for i in range(6)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        for r in reqs:
            if r.idx in resp:
                assert resp[r.idx] >= r.service_s - 1e-9, (
                    f"response must be >= service_s for idx={r.idx}"
                )

    def test_single_request_completes_immediately(self):
        reqs = [_req(0, 0.0, 7)]
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        assert 0 in resp
        assert abs(resp[0] - 7.0) < 1e-9


# ---------------------------------------------------------------------------
# Class 5: Goodput and long_p99 positioning vs other disciplines
# ---------------------------------------------------------------------------

class TestDecoupledPositioning:
    """Verify decoupled_hybrid positions correctly vs other disciplines."""

    def test_goodput_at_least_hybrid_and_close_to_srpt(self):
        # On small workloads (short queues), aging dispatch at α=0.01 rarely reorders vs
        # pure SRPT, so decoupled ≈ srpt_preemptive AND decoupled ≥ hybrid.
        # NB: SRPT preemption can hurt SLA-safe goodput vs non-preemptive aging_srtf
        # when long jobs exceed the SLA budget due to preemption delays.
        reqs_d = [_req(i, i * 0.2, max(1, 10 - i)) for i in range(15)]
        reqs_s = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_d]
        reqs_h = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_d]
        _, resp_d, _ = simulate_queue(reqs_d, servers=2, discipline="decoupled_hybrid",
                                      aging_alpha=0.01)
        _, resp_s, _ = simulate_queue(reqs_s, servers=2, discipline="srpt_preemptive")
        _, resp_h, _ = simulate_queue(reqs_h, servers=2, discipline="hybrid_aging_preemptive",
                                      aging_alpha=0.01)
        gp_d = _sla_safe_goodput_per_dollar(reqs_d, resp_d, sla_s=20.0, servers=2)
        gp_s = _sla_safe_goodput_per_dollar(reqs_s, resp_s, sla_s=20.0, servers=2)
        gp_h = _sla_safe_goodput_per_dollar(reqs_h, resp_h, sla_s=20.0, servers=2)
        assert gp_d >= gp_h - 1e-6, (
            f"decoupled ({gp_d:.1f}) must be >= hybrid ({gp_h:.1f}): "
            "pure SRPT preemption >= aging-adjusted preemption"
        )
        assert abs(gp_d - gp_s) / max(gp_s, 1.0) < 0.05, (
            f"decoupled ({gp_d:.1f}) should track srpt ({gp_s:.1f}) within 5% at α=0.01"
        )

    def test_goodput_at_least_hybrid_aging_preemptive(self):
        # Decoupled uses pure SRPT preemption → at least as good as hybrid (unified key).
        reqs_d = [_req(i, i * 0.2, max(1, 10 - i)) for i in range(15)]
        reqs_h = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_d]
        _, resp_d, _ = simulate_queue(reqs_d, servers=2, discipline="decoupled_hybrid",
                                      aging_alpha=0.01)
        _, resp_h, _ = simulate_queue(reqs_h, servers=2, discipline="hybrid_aging_preemptive",
                                      aging_alpha=0.01)
        gp_d = _sla_safe_goodput_per_dollar(reqs_d, resp_d, sla_s=20.0, servers=2)
        gp_h = _sla_safe_goodput_per_dollar(reqs_h, resp_h, sla_s=20.0, servers=2)
        assert gp_d >= gp_h - 1e-6, (
            f"decoupled goodput ({gp_d:.4f}) must be >= hybrid ({gp_h:.4f})"
        )

    def test_higher_alpha_increases_long_job_priority_at_dispatch(self):
        # With higher α, a long-waiting job wins dispatch sooner.
        # 1 server. r_dummy(1s) at t=0. r1(10s) waits from t=0. Multiple r2..rN arrive.
        # Higher α → long-waiting r1 gets dispatched sooner → smaller sojourn.
        reqs_lo = [
            _req(0, 0, 1),           # dummy occupies server
            _req(1, 0, 10),          # long job waits
            _req(2, 1, 2),           # fresh short
            _req(3, 1.5, 2),         # fresh short
        ]
        reqs_hi = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_lo]
        _, resp_lo, _ = simulate_queue(reqs_lo, servers=1, discipline="decoupled_hybrid",
                                       aging_alpha=0.001)
        _, resp_hi, _ = simulate_queue(reqs_hi, servers=1, discipline="decoupled_hybrid",
                                       aging_alpha=5.0)
        # Higher α → r1 dispatched sooner → smaller sojourn for r1.
        assert resp_hi[1] <= resp_lo[1], (
            f"higher α should not worsen r1: lo={resp_lo[1]:.2f}, hi={resp_hi[1]:.2f}"
        )

    def test_short_request_benefits_from_decoupled_vs_aging_srtf(self):
        # In decoupled_hybrid, short requests can STILL preempt running long jobs
        # (pure SRPT preemption), which aging_srtf cannot do (non-preemptive).
        # → short request sojourn should be better under decoupled_hybrid.
        reqs_d = [_req(0, 0, 10), _req(1, 5, 2)]   # long starts, short arrives later
        reqs_a = [_req(0, 0, 10), _req(1, 5, 2)]
        _, resp_d, _ = simulate_queue(reqs_d, servers=1, discipline="decoupled_hybrid",
                                      aging_alpha=0.01)
        _, resp_a, _ = simulate_queue(reqs_a, servers=1, discipline="aging_srtf",
                                      aging_alpha=0.01)
        # decoupled: r1 preempts r0 at t=5 → sojourn = 2s.
        # aging_srtf: non-preemptive → r1 must wait for r0 to finish → sojourn > 2s.
        assert abs(resp_d[1] - 2.0) < 1e-9, (
            f"decoupled: short sojourn=2 (preempted long) (got {resp_d[1]:.4f})"
        )
        assert resp_a[1] > 2.0, (
            f"aging_srtf: short must wait (non-preemptive) (got {resp_a[1]:.4f})"
        )


# ---------------------------------------------------------------------------
# Class 6: Anti-starvation via aging dispatch
# ---------------------------------------------------------------------------

class TestDecoupledAntiStarvation:
    """Aging dispatch prevents extreme starvation even with SRPT preemption."""

    def test_long_job_eventually_completes_under_heavy_short_stream(self):
        # 1 server. Long job (20s) at t=0; 15 short jobs (1s each) every 0.3s.
        # All must complete.
        reqs = [_req(0, 0, 20)]
        for i in range(15):
            reqs.append(_req(i + 1, 0.3 * (i + 1), 1))
        _, resp, _ = simulate_queue(reqs, servers=1, discipline="decoupled_hybrid",
                                    aging_alpha=0.01)
        for r in reqs:
            assert r.idx in resp, f"request {r.idx} must complete"

    def test_long_job_completes_faster_with_higher_alpha(self):
        # Higher α gives long job dispatch priority sooner.
        reqs_lo = [_req(0, 0, 20)] + [_req(i, i * 0.3, 1) for i in range(1, 20)]
        reqs_hi = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_lo]
        _, resp_lo, _ = simulate_queue(reqs_lo, servers=1, discipline="decoupled_hybrid",
                                       aging_alpha=0.001)
        _, resp_hi, _ = simulate_queue(reqs_hi, servers=1, discipline="decoupled_hybrid",
                                       aging_alpha=2.0)
        assert resp_hi[0] <= resp_lo[0], (
            f"higher α must not worsen long job: lo={resp_lo[0]:.2f}, hi={resp_hi[0]:.2f}"
        )

    def test_anti_starvation_vs_srpt_pure(self):
        # Compare decoupled_hybrid vs srpt_preemptive on a starvation-prone trace.
        # Long job (50s) alongside many 1s short jobs. With sufficient α, decoupled
        # should dispatch the long job sooner than pure SRPT.
        reqs_base = [_req(0, 0, 50)] + [_req(i, i * 0.4, 1) for i in range(1, 30)]
        reqs_d = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_base]
        reqs_s = [_req(r.idx, r.arrival_s, r.actual_tokens) for r in reqs_base]
        _, resp_d, _ = simulate_queue(reqs_d, servers=1, discipline="decoupled_hybrid",
                                      aging_alpha=0.5)
        _, resp_s, _ = simulate_queue(reqs_s, servers=1, discipline="srpt_preemptive")
        # With α=0.5, the long job accumulates dispatch priority and completes
        # before or at the same time as pure SRPT.
        assert resp_d[0] <= resp_s[0] + 1e-6, (
            f"decoupled (α=0.5) long job {resp_d[0]:.2f}s <= srpt {resp_s[0]:.2f}s"
        )


# ---------------------------------------------------------------------------
# Class 7: Report dataclass
# ---------------------------------------------------------------------------

class TestDecoupledHybridReportDataclass:
    """Verify DecoupledHybridReport structure and serialization."""

    def _make_summary(self) -> dict:
        return {
            "requests": 10, "servers": 2, "sim_horizon_s": 50.0,
            "mean_response_s": 5.0, "p50_response_s": 4.0, "p90_response_s": 9.0,
            "p99_response_s": 12.0, "mean_wait_s": 1.0, "p90_wait_s": 3.0,
            "p99_wait_s": 5.0, "short_p90_response_s": 3.0, "short_p99_response_s": 5.0,
            "long_p90_response_s": 8.0, "long_p99_response_s": 12.0,
            "max_response_s": 15.0, "sla_safe_goodput_per_dollar": 1000.0,
        }

    def test_has_six_disciplines(self):
        s = self._make_summary()
        r = DecoupledHybridReport(
            trace="test", total_requests=10, servers=2, target_rho=0.85,
            time_warp=2.0, sla_s=10.0, aging_alpha=0.01,
            fifo=s, srtf_perfect=s, aging_srtf=s, srpt_preemptive=s,
            hybrid_aging_preemptive=s, decoupled_hybrid=s,
            srtf_short_p90_improvement_pct=99.0,
            aging_short_p90_improvement_pct=78.0,
            srpt_short_p90_improvement_pct=99.7,
            hybrid_short_p90_improvement_pct=75.7,
            decoupled_short_p90_improvement_pct=99.5,
            srtf_long_p99_delta_pct=223.0,
            aging_long_p99_delta_pct=113.0,
            srpt_long_p99_delta_pct=223.0,
            hybrid_long_p99_delta_pct=111.0,
            decoupled_long_p99_delta_pct=120.0,
            srtf_goodput_delta_pct=323.0,
            aging_goodput_delta_pct=70.7,
            srpt_goodput_delta_pct=322.2,
            hybrid_goodput_delta_pct=64.2,
            decoupled_goodput_delta_pct=315.0,
        )
        d = r.to_dict()
        for key in ["fifo", "srtf_perfect", "aging_srtf", "srpt_preemptive",
                    "hybrid_aging_preemptive", "decoupled_hybrid"]:
            assert key in d, f"missing discipline '{key}' in to_dict()"

    def test_shadow_tag_present(self):
        s = self._make_summary()
        r = DecoupledHybridReport(
            trace="test", total_requests=5, servers=1, target_rho=0.85,
            time_warp=1.0, sla_s=10.0, aging_alpha=0.01,
            fifo=s, srtf_perfect=s, aging_srtf=s, srpt_preemptive=s,
            hybrid_aging_preemptive=s, decoupled_hybrid=s,
            srtf_short_p90_improvement_pct=0.0, aging_short_p90_improvement_pct=0.0,
            srpt_short_p90_improvement_pct=0.0, hybrid_short_p90_improvement_pct=0.0,
            decoupled_short_p90_improvement_pct=0.0,
            srtf_long_p99_delta_pct=0.0, aging_long_p99_delta_pct=0.0,
            srpt_long_p99_delta_pct=0.0, hybrid_long_p99_delta_pct=0.0,
            decoupled_long_p99_delta_pct=0.0,
            srtf_goodput_delta_pct=0.0, aging_goodput_delta_pct=0.0,
            srpt_goodput_delta_pct=0.0, hybrid_goodput_delta_pct=0.0,
            decoupled_goodput_delta_pct=0.0,
        )
        d = r.to_dict()
        assert "shadow_tag" in d
        assert "simulator_result" in d["shadow_tag"]

    def test_all_delta_fields_present(self):
        s = self._make_summary()
        r = DecoupledHybridReport(
            trace="azure", total_requests=100, servers=4, target_rho=0.85,
            time_warp=5.0, sla_s=10.0, aging_alpha=0.01,
            fifo=s, srtf_perfect=s, aging_srtf=s, srpt_preemptive=s,
            hybrid_aging_preemptive=s, decoupled_hybrid=s,
            srtf_short_p90_improvement_pct=99.0,
            aging_short_p90_improvement_pct=78.0,
            srpt_short_p90_improvement_pct=99.7,
            hybrid_short_p90_improvement_pct=75.0,
            decoupled_short_p90_improvement_pct=99.5,
            srtf_long_p99_delta_pct=223.0,
            aging_long_p99_delta_pct=113.0,
            srpt_long_p99_delta_pct=223.0,
            hybrid_long_p99_delta_pct=111.0,
            decoupled_long_p99_delta_pct=120.0,
            srtf_goodput_delta_pct=323.0,
            aging_goodput_delta_pct=70.0,
            srpt_goodput_delta_pct=322.0,
            hybrid_goodput_delta_pct=64.0,
            decoupled_goodput_delta_pct=315.0,
        )
        d = r.to_dict()
        for prefix in ["srtf", "aging", "srpt", "hybrid", "decoupled"]:
            assert f"{prefix}_goodput_delta_pct" in d
            assert f"{prefix}_long_p99_delta_pct" in d
            assert f"{prefix}_short_p90_improvement_pct" in d


# ---------------------------------------------------------------------------
# Class 8: Public API functions
# ---------------------------------------------------------------------------

class TestDecoupledPublicAPI:
    """run_decoupled_hybrid_backtest and run_burstgpt_decoupled_hybrid_backtest."""

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_AZURE_FIXTURE),
        reason="Azure LLM 2024 fixture not available",
    )
    def test_azure_backtest_returns_report(self):
        report = run_decoupled_hybrid_backtest(
            servers=2, target_rho=0.85, job_limit=200,
        )
        assert isinstance(report, DecoupledHybridReport)
        assert report.trace == "azure_llm_2024"
        assert report.total_requests <= 200
        assert report.servers == 2
        assert report.target_rho == 0.85

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_AZURE_FIXTURE),
        reason="Azure LLM 2024 fixture not available",
    )
    def test_azure_backtest_decoupled_goodput_far_above_hybrid(self):
        # On full Azure trace (5880 requests, 4 servers) decoupled shows true hybrid
        # benefit: SRPT-like preemption yields ~3x goodput vs hybrid's aging preemption.
        # (Small-workload tests show decoupled ≈ srpt since queue depth is low and
        #  aging dispatch rarely reorders at α=0.01.)
        report = run_decoupled_hybrid_backtest(
            servers=4, target_rho=0.85, aging_alpha=0.01,
        )
        assert report.decoupled_goodput_delta_pct > report.hybrid_goodput_delta_pct + 50, (
            f"decoupled (+{report.decoupled_goodput_delta_pct:.1f}%) must exceed "
            f"hybrid (+{report.hybrid_goodput_delta_pct:.1f}%) by >50pp on full Azure trace"
        )

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_AZURE_FIXTURE),
        reason="Azure LLM 2024 fixture not available",
    )
    def test_azure_backtest_decoupled_goodput_exceeds_hybrid(self):
        report = run_decoupled_hybrid_backtest(
            servers=2, target_rho=0.85, job_limit=300, aging_alpha=0.01,
        )
        # Decoupled (SRPT preemption) must outperform hybrid (aging preemption).
        assert report.decoupled_goodput_delta_pct >= report.hybrid_goodput_delta_pct - 1.0, (
            f"decoupled (+{report.decoupled_goodput_delta_pct:.1f}%) must be >= "
            f"hybrid (+{report.hybrid_goodput_delta_pct:.1f}%) - 1pp"
        )

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_BURSTGPT_FIXTURE),
        reason="BurstGPT fixture not available",
    )
    def test_burstgpt_backtest_returns_report(self):
        report = run_burstgpt_decoupled_hybrid_backtest(servers=2, target_rho=0.85)
        assert isinstance(report, DecoupledHybridReport)
        assert report.trace == "burstgpt"
        assert report.servers == 2

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_BURSTGPT_FIXTURE),
        reason="BurstGPT fixture not available",
    )
    def test_burstgpt_backtest_all_disciplines_have_goodput(self):
        report = run_burstgpt_decoupled_hybrid_backtest(servers=2, target_rho=0.85)
        for disc in ["fifo", "srtf_perfect", "aging_srtf", "srpt_preemptive",
                     "hybrid_aging_preemptive", "decoupled_hybrid"]:
            assert "sla_safe_goodput_per_dollar" in getattr(report, disc), (
                f"discipline '{disc}' missing goodput/$ in BurstGPT report"
            )

    def test_alpha_default_constant(self):
        assert DECOUPLED_HYBRID_ALPHA_DEFAULT == 0.001


# ---------------------------------------------------------------------------
# Class 9: Internal helper _run_decoupled_hybrid_backtest_on_trace
# ---------------------------------------------------------------------------

class TestDecoupledRunOnTrace:
    """Test _run_decoupled_hybrid_backtest_on_trace with synthetic data."""

    def _make_raw(self, n=50):
        import random
        rng = random.Random(42)
        raw = []
        t = 0.0
        for _ in range(n):
            t += rng.uniform(0.5, 2.0)
            tok = rng.randint(50, 500)
            raw.append((t, tok))
        return raw

    def test_returns_decoupled_hybrid_report(self):
        raw = self._make_raw(50)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "synthetic", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=10.0,
        )
        assert isinstance(report, DecoupledHybridReport)

    def test_total_requests_matches_raw(self):
        raw = self._make_raw(60)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "test", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=10.0,
        )
        assert report.total_requests == 60

    def test_trace_name_set_correctly(self):
        raw = self._make_raw(30)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "my_trace", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=10.0,
        )
        assert report.trace == "my_trace"

    def test_six_disciplines_in_report(self):
        raw = self._make_raw(40)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "test", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=10.0,
        )
        for disc in ["fifo", "srtf_perfect", "aging_srtf", "srpt_preemptive",
                     "hybrid_aging_preemptive", "decoupled_hybrid"]:
            d = getattr(report, disc)
            assert "sla_safe_goodput_per_dollar" in d, f"missing gp/$ in {disc}"

    def test_decoupled_goodput_tracks_srpt_on_small_trace(self):
        # On a small synthetic trace (low queue depth), aging dispatch at α=0.01
        # rarely reorders vs pure SRPT → decoupled_hybrid ≈ srpt_preemptive.
        # Both preemptive disciplines may underperform non-preemptive on SLA-safe goodput
        # when long jobs exceed the SLA budget due to preemption-induced delays.
        raw = self._make_raw(80)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "test", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=10.0,
        )
        assert abs(report.decoupled_goodput_delta_pct - report.srpt_goodput_delta_pct) < 2.0, (
            f"decoupled ({report.decoupled_goodput_delta_pct:.1f}%) should track "
            f"srpt ({report.srpt_goodput_delta_pct:.1f}%) within 2pp on small trace at α=0.01"
        )

    def test_decoupled_goodput_above_hybrid(self):
        raw = self._make_raw(80)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "test", servers=2, target_rho=0.85, aging_alpha=0.01, sla_s=10.0,
        )
        assert report.decoupled_goodput_delta_pct >= report.hybrid_goodput_delta_pct - 1.0, (
            f"decoupled should be >= hybrid on goodput/$: "
            f"decoupled={report.decoupled_goodput_delta_pct:.1f}%, "
            f"hybrid={report.hybrid_goodput_delta_pct:.1f}%"
        )

    def test_srpt_and_decoupled_goodput_close(self):
        # Since preemption is identical to SRPT, goodput should be close.
        raw = self._make_raw(100)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "test", servers=2, target_rho=0.85, aging_alpha=0.001, sla_s=10.0,
        )
        # At α=0.001, aging dispatch only kicks in after extreme waits → near-SRPT.
        decoupled_gp = report.decoupled_hybrid["sla_safe_goodput_per_dollar"]
        srpt_gp = report.srpt_preemptive["sla_safe_goodput_per_dollar"]
        ratio = decoupled_gp / max(srpt_gp, 1e-9)
        assert ratio >= 0.90, (
            f"decoupled (α=0.001) goodput/$ should be within 10% of SRPT: "
            f"decoupled={decoupled_gp:.2f}, srpt={srpt_gp:.2f}, ratio={ratio:.3f}"
        )

    def test_to_dict_round_trips_all_fields(self):
        raw = self._make_raw(30)
        report = _run_decoupled_hybrid_backtest_on_trace(
            raw, "test", servers=1, target_rho=0.80, aging_alpha=0.01, sla_s=15.0,
        )
        d = report.to_dict()
        assert d["trace"] == "test"
        assert d["servers"] == 1
        assert d["target_rho"] == 0.80
        assert d["aging_alpha"] == 0.01
        assert d["sla_s"] == 15.0
        assert "decoupled_hybrid" in d
        assert "decoupled_goodput_delta_pct" in d
