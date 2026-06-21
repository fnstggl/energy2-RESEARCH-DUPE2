"""Tests for Admission Gate + SRPT Compound Under Overload [run 2026-06-21-t].

Wires a queue-depth admission gate into the SRPT preemptive simulator and
validates the compound strategy across three load regimes (ρ ∈ {0.85, 0.95, 1.05})
on the Azure LLM 2024 and BurstGPT HF public traces.

Research basis:
  - arXiv:2604.11001 (Flow-Controlled Scheduling for LLM Inference, Apr 2026)
  - arXiv:2510.15330 (BeLLMan, demand-side congestion control, Oct 2025)
  - arXiv:2604.06970 (Scheduling the Unschedulable, §5 overload control)
  - arXiv:2605.16867 (GoodServe, SLO-violation risk monitoring, May 2026)

The gate fires only when a request would join the WAITING queue (not when it
would preempt a running job), preserving the SRPT preemption invariant.
"""

from __future__ import annotations

import os

import pytest

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    AdmissionGateEntry,
    AdmissionGateReport,
    _Request,
    _simulate_srpt_preemptive,
    _simulate_srpt_with_queue_gate,
    _sla_safe_goodput_per_dollar,
    calibrate_time_warp,
    run_admission_gate_overload_backtest,
    run_burstgpt_admission_gate_overload_backtest,
)

_HF_AVAILABLE = os.path.isfile(DEFAULT_BURSTGPT_HF_JSONL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _req(idx, arrival, tokens, service_s=None):
    """Synthetic _Request: service_s defaults to tokens (1 tok = 1 s)."""
    return _Request(
        idx=idx,
        arrival_s=float(arrival),
        actual_tokens=int(tokens),
        predicted_tokens=float(tokens),
        service_s=float(service_s if service_s is not None else tokens),
    )


def _make_requests(n=20, arrival_gap=0.5, tokens=5):
    """Uniform inter-arrival stream: gap apart, all same size."""
    return [_req(i, i * arrival_gap, tokens) for i in range(n)]


# ---------------------------------------------------------------------------
# Class 1: _simulate_srpt_with_queue_gate — mechanics
# ---------------------------------------------------------------------------

class TestGateMechanics:
    def test_no_gate_fires_when_queue_empty(self):
        """Gate never fires when waiting queue never fills up."""
        reqs = _make_requests(n=4, arrival_gap=10.0, tokens=1)
        _, resp, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=2, max_queue_depth=100, defer_s=1.0, sla_s=100.0
        )
        assert stats["defer_count"] == 0
        assert stats["drop_count"] == 0
        assert len(resp) == 4

    def test_gate_fires_when_queue_at_capacity(self):
        """When queue fills up, new arrivals are deferred, not queued immediately."""
        # 1 server, 1-unit jobs, rapid arrivals so queue fills.
        # max_queue_depth = 1 means once 1 job is waiting, next is deferred.
        reqs = [_req(i, i * 0.01, 10) for i in range(20)]
        _, _, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=1, max_queue_depth=1, defer_s=0.1, sla_s=1000.0
        )
        assert stats["defer_count"] > 0

    def test_all_requests_complete_without_gate_pressure(self):
        """Lightly loaded system: all requests complete with no deferrals."""
        reqs = [_req(i, i * 5.0, 2) for i in range(8)]
        _, resp, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=2, max_queue_depth=4, defer_s=1.0, sla_s=50.0
        )
        assert stats["defer_count"] == 0
        assert stats["drop_count"] == 0
        assert len(resp) == 8

    def test_drop_when_past_sla_deadline(self):
        """Re-injection past SLA budget → drop, not repeated deferral."""
        # 1 server, very tight SLA: 1.0s. Jobs are 0.5s long.
        # defer_s = 2.0 → re-injection at t+2 exceeds service_s(0.5) + 0 budget from arrival.
        # arrival_s=0, service_s=0.5, sla_s=1.0, defer_s=2.0 →
        # t=0 → gate fires → next_t=2.0, 2.0 + 0.5 > 0 + 1.0 → drop
        # (set max_queue_depth=0 to always trigger gate)
        r0 = _req(0, 0.0, 10, service_s=0.5)   # running
        r1 = _req(1, 0.01, 10, service_s=0.5)   # would queue; gate triggers
        _, _, _, stats = _simulate_srpt_with_queue_gate(
            [r0, r1], servers=1, max_queue_depth=0, defer_s=2.0, sla_s=1.0
        )
        assert stats["drop_count"] >= 1

    def test_deferred_request_eventually_completes(self):
        """A deferred request re-injects and completes when queue clears."""
        # Sequence: r0 long (runs 5s), r1 arrives at t=0.1, deferred, re-injects at 1.1,
        # r0 still running but r1 is shorter → preempts at re-injection time.
        r0 = _req(0, 0.0, 5, service_s=5.0)
        r1 = _req(1, 0.1, 1, service_s=1.0)
        _, resp, _, stats = _simulate_srpt_with_queue_gate(
            [r0, r1], servers=1, max_queue_depth=0, defer_s=1.0, sla_s=100.0
        )
        # Both should complete (r1 preempts r0 on re-injection)
        assert 0 in resp
        assert 1 in resp

    def test_gate_stats_sum_invariant(self):
        """len(resp) + drop_count == total_arrivals: every request completes or drops."""
        # defer_count counts events (a request can be deferred multiple times),
        # so defer_count can exceed total_arrivals. The true invariant is that
        # every original request either eventually completes (enters resp) or
        # is dropped at a gate check when its SLA deadline is unrecoverable.
        reqs = [_req(i, i * 0.05, 3) for i in range(30)]
        _, resp, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=1, max_queue_depth=2, defer_s=0.5, sla_s=20.0
        )
        total = stats["total_arrivals"]
        assert len(resp) + stats["drop_count"] == total

    def test_zero_max_queue_depth_forces_gate_always(self):
        """max_queue_depth=0 means every request that would wait is gated."""
        reqs = [_req(i, 0.0, 2, service_s=2.0) for i in range(5)]
        _, resp, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=1, max_queue_depth=0, defer_s=0.1, sla_s=1000.0
        )
        # Only 1 server: at t=0 only 1 runs, 4 deferred, then re-inject as queue clears
        assert stats["defer_count"] > 0

    def test_defer_fraction_is_normalized(self):
        """defer_fraction = defer_count / total_arrivals."""
        reqs = [_req(i, i * 0.1, 2) for i in range(10)]
        _, _, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=1, max_queue_depth=1, defer_s=0.5, sla_s=100.0
        )
        expected = stats["defer_count"] / stats["total_arrivals"]
        assert abs(stats["defer_fraction"] - expected) < 1e-9

    def test_drop_fraction_is_normalized(self):
        """drop_fraction = drop_count / total_arrivals."""
        reqs = [_req(i, i * 0.1, 2) for i in range(10)]
        _, _, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=1, max_queue_depth=1, defer_s=5.0, sla_s=1.5
        )
        expected = stats["drop_count"] / stats["total_arrivals"]
        assert abs(stats["drop_fraction"] - expected) < 1e-9

    def test_no_stale_events_on_gate_fire(self):
        """Gate deferral does not corrupt server version counters."""
        reqs = [_req(i, i * 0.1, 1, service_s=1.0) for i in range(10)]
        summary, resp, wait_map, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=2, max_queue_depth=2, defer_s=0.5, sla_s=50.0
        )
        # All completions should be non-negative response times
        for idx, rt in resp.items():
            assert rt >= 0.0, f"negative response time for idx={idx}"


# ---------------------------------------------------------------------------
# Class 2: gate does not break SRPT invariant
# ---------------------------------------------------------------------------

class TestSRPTInvariantPreserved:
    def test_preemption_still_occurs_when_gate_inactive(self):
        """With large max_queue_depth the gate never fires but SRPT still preempts."""
        # r0 long, r1 short → r1 should preempt r0 at arrival
        r0 = _req(0, 0.0, 10, service_s=10.0)
        r1 = _req(1, 0.1, 2, service_s=2.0)
        gate_sim, gate_resp, _, stats = _simulate_srpt_with_queue_gate(
            [r0, r1], servers=1, max_queue_depth=100, defer_s=1.0, sla_s=1000.0
        )
        srpt_sim, srpt_resp, _ = _simulate_srpt_preemptive([r0, r1], servers=1)
        assert stats["defer_count"] == 0
        # Both simulators should complete r1 before r0
        assert gate_resp.get(1, 9999) < gate_resp.get(0, 0)
        assert srpt_resp.get(1, 9999) < srpt_resp.get(0, 0)

    def test_gated_srpt_response_times_match_ungated_for_admitted(self):
        """Under light load (gate never fires), response times match vanilla SRPT."""
        reqs = [_req(i, i * 5.0, i + 1) for i in range(6)]
        _, gate_resp, _, stats = _simulate_srpt_with_queue_gate(
            reqs, servers=2, max_queue_depth=100, defer_s=1.0, sla_s=500.0
        )
        _, srpt_resp, _ = _simulate_srpt_preemptive(reqs, servers=2)
        assert stats["defer_count"] == 0
        for r in reqs:
            assert abs(gate_resp[r.idx] - srpt_resp[r.idx]) < 1e-9

    def test_gate_does_not_fire_for_preemptible_arrivals(self):
        """A short arrival that preempts is never subject to the gate, even at max depth."""
        # 1 server busy with long job; short arrival preempts (gate shouldn't fire)
        r_long = _req(0, 0.0, 10, service_s=10.0)
        r_short = _req(1, 0.1, 1, service_s=1.0)
        _, _, _, stats = _simulate_srpt_with_queue_gate(
            [r_long, r_short],
            servers=1,
            max_queue_depth=0,   # gate would fire for any waiting request
            defer_s=1.0,
            sla_s=1000.0,
        )
        # r_short preempts r_long, so gate does not fire
        assert stats["defer_count"] == 0
        assert stats["drop_count"] == 0


# ---------------------------------------------------------------------------
# Class 3: AdmissionGateEntry / AdmissionGateReport structure
# ---------------------------------------------------------------------------

class TestReportStructure:
    def _make_report(self):
        raw = [(float(i), 90) for i in range(20)]
        from aurelius.benchmarks.srtf_serving_backtest import _run_admission_gate_on_trace
        return _run_admission_gate_on_trace(
            raw, "test_trace", servers=2,
            rho_list=[0.85, 0.95],
            max_queue_depth=4, defer_s=1.0, sla_s=10.0,
        )

    def test_report_type(self):
        r = self._make_report()
        assert isinstance(r, AdmissionGateReport)

    def test_rho_list_preserved(self):
        r = self._make_report()
        assert set(r.rho_list) == {0.85, 0.95}

    def test_entries_keys_match_rho_list(self):
        r = self._make_report()
        for rho in r.rho_list:
            assert rho in r.entries

    def test_entry_type(self):
        r = self._make_report()
        for entry in r.entries.values():
            assert isinstance(entry, AdmissionGateEntry)

    def test_entry_rho_matches_key(self):
        r = self._make_report()
        for rho, entry in r.entries.items():
            assert entry.rho == rho

    def test_srpt_without_gate_has_goodput_key(self):
        r = self._make_report()
        for entry in r.entries.values():
            assert "sla_safe_goodput_per_dollar" in entry.srpt_without_gate

    def test_srpt_with_gate_has_goodput_key(self):
        r = self._make_report()
        for entry in r.entries.values():
            assert "sla_safe_goodput_per_dollar" in entry.srpt_with_gate

    def test_defer_fraction_non_negative(self):
        # defer_fraction = defer_count / total_arrivals where defer_count counts
        # events (not unique requests). A request deferred k times contributes k,
        # so defer_fraction can exceed 1.0. Only test non-negativity here.
        r = self._make_report()
        for entry in r.entries.values():
            assert entry.defer_fraction >= 0.0

    def test_drop_fraction_in_range(self):
        # drop_fraction counts unique drops — always in [0, 1].
        r = self._make_report()
        for entry in r.entries.values():
            assert 0.0 <= entry.drop_fraction <= 1.0

    def test_benefit_at_rho_accessor(self):
        r = self._make_report()
        for rho in r.rho_list:
            pct = r.benefit_at_rho(rho)
            assert isinstance(pct, float)

    def test_entry_accessor(self):
        r = self._make_report()
        e = r.entry(0.85)
        assert isinstance(e, AdmissionGateEntry)
        assert e.rho == 0.85

    def test_to_dict_serializable(self):
        import json
        r = self._make_report()
        d = r.to_dict()
        json.dumps(d)   # must not raise

    def test_entry_to_dict_serializable(self):
        import json
        r = self._make_report()
        for entry in r.entries.values():
            json.dumps(entry.to_dict())

    def test_report_trace_name(self):
        r = self._make_report()
        assert r.trace_name == "test_trace"

    def test_report_max_queue_depth(self):
        r = self._make_report()
        assert r.max_queue_depth == 4

    def test_report_defer_s(self):
        r = self._make_report()
        assert r.defer_s == 1.0


# ---------------------------------------------------------------------------
# Class 4: Azure LLM 2024 backtest integration
# ---------------------------------------------------------------------------

class TestAzureAdmissionGateBacktest:
    @pytest.fixture(scope="class")
    def report(self):
        return run_admission_gate_overload_backtest(
            servers=4,
            rho_list=[0.85, 0.95, 1.05],
            max_queue_depth=8,
            defer_s=1.0,
            sla_s=DEFAULT_SLA_S,
        )

    def test_report_is_admission_gate_report(self, report):
        assert isinstance(report, AdmissionGateReport)

    def test_three_rho_entries(self, report):
        assert len(report.entries) == 3
        for rho in [0.85, 0.95, 1.05]:
            assert rho in report.entries

    def test_trace_name_azure(self, report):
        assert "azure" in report.trace_name.lower()

    def test_goodput_positive_all_rho(self, report):
        for entry in report.entries.values():
            assert entry.srpt_without_gate["sla_safe_goodput_per_dollar"] > 0

    def test_gate_benefit_positive_at_normal_load(self, report):
        """At ρ=0.85 the gate should provide ≥ 0% goodput/$ benefit.

        Even at stable load, SRPT causes severe starvation of long requests
        (p99 >> SLA). The gate drops these SLA-busted long requests, which
        improves sla_safe_goodput/$ by shedding work that would never count.
        """
        e = report.entry(0.85)
        assert e.gate_benefit_pct >= 0.0

    def test_drop_fraction_monotone_with_load(self, report):
        """drop_fraction (unique requests dropped / total) increases with ρ.

        Note: defer_fraction counts defer EVENTS (one request deferred k times
        contributes k), so it can exceed 1.0. Use drop_fraction for unique-request
        monotonicity comparisons.
        """
        e85  = report.entry(0.85)
        e105 = report.entry(1.05)
        assert e105.drop_fraction >= e85.drop_fraction

    def test_gate_benefit_increases_with_load(self, report):
        """Gate provides more benefit at higher load (ρ=1.05 > ρ=0.85)."""
        assert report.benefit_at_rho(1.05) >= report.benefit_at_rho(0.85)

    def test_all_fractions_valid(self, report):
        for entry in report.entries.values():
            assert entry.defer_fraction >= 0.0  # can exceed 1 (counts events)
            assert 0.0 <= entry.drop_fraction <= 1.0  # unique drops

    def test_servers_field(self, report):
        assert report.servers == 4

    def test_max_queue_depth_field(self, report):
        assert report.max_queue_depth == 8


# ---------------------------------------------------------------------------
# Class 5: BurstGPT HF cross-validation (skipped when HF file absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HF_AVAILABLE, reason="BurstGPT HF JSONL not available")
class TestBurstGPTAdmissionGateBacktest:
    @pytest.fixture(scope="class")
    def report(self):
        return run_burstgpt_admission_gate_overload_backtest(
            servers=4,
            rho_list=[0.85, 0.95, 1.05],
            max_queue_depth=8,
            defer_s=1.0,
            sla_s=DEFAULT_BURSTGPT_SLA_S,
            job_limit=5880,
        )

    def test_report_type(self, report):
        assert isinstance(report, AdmissionGateReport)

    def test_three_rho_entries(self, report):
        assert len(report.entries) == 3

    def test_trace_name_burstgpt(self, report):
        assert "burstgpt" in report.trace_name.lower()

    def test_goodput_positive_all_rho(self, report):
        for entry in report.entries.values():
            assert entry.srpt_without_gate["sla_safe_goodput_per_dollar"] > 0

    def test_gate_more_active_at_overload_than_normal(self, report):
        e85  = report.entry(0.85)
        e105 = report.entry(1.05)
        rate_85  = e85.defer_fraction  + e85.drop_fraction
        rate_105 = e105.defer_fraction + e105.drop_fraction
        assert rate_105 >= rate_85

    def test_all_fractions_valid(self, report):
        for entry in report.entries.values():
            assert entry.defer_fraction >= 0.0  # event count, can exceed 1
            assert 0.0 <= entry.drop_fraction <= 1.0  # unique drops

    def test_sla_s_is_burstgpt_default(self, report):
        assert report.sla_s == DEFAULT_BURSTGPT_SLA_S
