# Admission Gate + SRPT Compound Under Overload — Backtest Results

**Run:** 2026-06-21-t  
**Category:** FRONTIER IMPROVEMENT  
**Status:** CLOSED — all validation gates passed  
**Date:** 2026-06-21

## Summary

This run wires a **queue-depth admission gate** into the SRPT preemptive simulator
and measures the compound strategy (SRPT + gate) against SRPT alone at three load
levels (ρ ∈ {0.85, 0.95, 1.05}) on both the Azure LLM 2024 and BurstGPT HF public traces.

**Key finding:** The admission gate provides consistent, monotonically increasing
sla_safe_goodput/$ benefit across all load levels — even at stable load (ρ=0.85).
The mechanism: pure SRPT causes extreme starvation of long requests (p99 >> SLA).
The queue-depth gate sheds these SLA-busted long requests early, recovering the
goodput that SRPT wastes on work that will never count toward the SLA-safe metric.

**NOT production savings.** Simulator / public-trace directional result only.

---

## Results — Azure LLM 2024

**Trace:** `tests/fixtures/azure_llm_2024_sample.csv` (5,880 requests)  
**SLA:** 10s | **Servers:** 4 | **Queue depth:** 8 = servers × 2 | **Defer:** 1.0s

| ρ (load) | SRPT gp/$ | SRPT+Gate gp/$ | Gate benefit | Drop rate | SRPT p99 | Gate p99 |
|-----------|-----------|----------------|--------------|-----------|-----------|----------|
| 0.85 | 56,311 | **58,418** | **+3.74%** | 9.0% | 2,197s | 33s |
| 0.95 | 51,725 | **53,990** | **+4.38%** | 12.2% | 2,338s | 35s |
| 1.05 | 47,717 | **51,188** | **+7.28%** | 15.3% | 2,402s | 31s |

p99 latency improvement: from 2,197–2,402s → 31–35s at all load levels.

---

## Results — BurstGPT HF Cross-Validation

**Trace:** `data/external/hf/lzzmm__BurstGPT/.../normalized_sample.jsonl` (5,880 records sampled)  
**SLA:** 30s | **Servers:** 4 | **Queue depth:** 8 | **Defer:** 1.0s

| ρ (load) | SRPT gp/$ | SRPT+Gate gp/$ | Gate benefit | Drop rate | SRPT p99 | Gate p99 |
|-----------|-----------|----------------|--------------|-----------|-----------|----------|
| 0.85 | 48,599 | **53,939** | **+10.99%** | 12.9% | 2,760s | 244s |
| 0.95 | 44,350 | **50,207** | **+13.21%** | 16.0% | 2,986s | 251s |
| 1.05 | 40,834 | **46,416** | **+13.67%** | 18.8% | 3,446s | 247s |

BurstGPT shows larger gate benefits (+11–14% vs +4–7%) due to its heavier
output-token distribution (p50=236 vs 90, p99=934 vs 479 tokens). Heavier tails
→ more severe SRPT starvation → larger gate uplift.

---

## Gate Mechanics

**Gate trigger condition:** len(waiting) ≥ max_queue_depth AND arriving request
would join waiting queue (not preempt). The gate does NOT apply to preemptible
arrivals — SRPT's preemption invariant is fully preserved.

**Deferral:** Request re-injected at t + 1.0s as a new arrival.  
**Drop condition:** If re-injection time + service_s > original_arrival_s + sla_s,
the request cannot complete within SLA even if served immediately → dropped.

**defer_fraction:** defer events / total_arrivals. Can exceed 1.0 since a single
request may be deferred multiple times. Ranges 0.79–1.24 on Azure, 0.85–1.27 on BurstGPT.

**drop_fraction:** Unique requests dropped / total_arrivals. In [0, 1].
Ranges 9–15% on Azure, 13–19% on BurstGPT.

---

## Interpretation

### Why the gate helps even at ρ = 0.85

At ρ=0.85, pure SRPT WITHOUT the gate already produces p99=2,197s (Azure) and
p99=2,760s (BurstGPT). This is severe starvation: long requests wait hundreds
of seconds in the queue, far past their SLA deadline.

The gate with `max_queue_depth=8` hits a full queue during burst periods even at
ρ=0.85. When the queue is full, new arrivals that would extend the backlog are
deferred. If re-injection still finds a full queue and the request has fallen past
its SLA deadline, it is dropped.

These dropped requests would have counted as 0 goodput anyway (they'd miss SLA).
By shedding them early, the gate:
1. Reduces queue depth → remaining requests see shorter waits
2. Concentrates service on requests that can still complete within SLA
3. Improves mean response time (5.85s vs 129.6s at ρ=0.85 on Azure)

### Monotonic benefit increase with load

Gate benefit increases with ρ: +3.74% → +4.38% → +7.28% (Azure) and
+10.99% → +13.21% → +13.67% (BurstGPT). At higher load:
- Queue fills more frequently → gate triggers more
- More requests fall past SLA deadline under uncontrolled SRPT
- Gate sheds more doomed work → larger uplift

### BurstGPT > Azure benefit gap

BurstGPT benefit (+11–14%) > Azure benefit (+4–7%) because:
- BurstGPT p99=934 vs Azure p99=479 tokens (heavier tail)
- Heavier tail → longer service times → more starvation under SRPT
- SLA=30s vs 10s: proportionally more headroom, but long BurstGPT requests
  still blow past 30s SLA under SRPT (p99=2,760–3,446s)

---

## Research Papers (Run 2026-06-21-t)

Three new papers identified this run (not in ROADMAP before this run):

1. **BeLLMan** (arXiv:2510.15330, Oct 2025): Demand-side congestion control for LLM
   serving. Re-injection (deferral) rather than immediate drop minimizes starvation.
   8× E2E latency reduction, 25% energy reduction, 19% more requests served at peak.

2. **GoodServe** (arXiv:2605.16867, May 2026): High-goodput agentic LLM serving with
   SLO-violation risk monitoring and runtime request migrations. +27.4% goodput.
   Confirms "drop doomed requests early" is sound at production scale.

3. **"Scheduling the Unschedulable" overload control** (arXiv:2604.06970, Apr 2026):
   §5 overload control component (admit/defer/reject) — previously only the ORDERING
   component (SRTF sort key) was exploited in Aurelius. This run closes that gap.

---

## Test Coverage

46 tests in `tests/test_admission_gate_overload_backtest.py`, all passing:

- **TestGateMechanics** (10 tests): basic gate trigger, defer/drop mechanics,
  stats invariants, stale-event safety
- **TestSRPTInvariantPreserved** (3 tests): preemption still occurs, response
  times match ungated SRPT when gate inactive
- **TestReportStructure** (15 tests): dataclass structure, serialization, accessors
- **TestAzureAdmissionGateBacktest** (9 tests): integration on Azure LLM 2024
- **TestBurstGPTAdmissionGateBacktest** (9 tests): cross-validation on BurstGPT HF

---

## New Code Artifacts

- **`aurelius/benchmarks/srtf_serving_backtest.py`**:
  - `_simulate_srpt_with_queue_gate()` — SRPT + queue-depth gate simulator
  - `AdmissionGateEntry` — frozen dataclass for one ρ level result
  - `AdmissionGateReport` — frozen dataclass for multi-ρ comparison
  - `_run_admission_gate_on_trace()` — shared inner loop
  - `run_admission_gate_overload_backtest()` — Azure LLM 2024 backtest
  - `run_burstgpt_admission_gate_overload_backtest()` — BurstGPT HF backtest

- **`tests/test_admission_gate_overload_backtest.py`** (new, 46 tests)

---

## Shadow / Production Status

**Shadow-mode result only.** The `WorkloadAdmissionGate` in
`aurelius/frontier/admission.py` (KV-cache utilization + queue-p99 signals) remains
unconnected to any serving runtime. This backtest uses queue depth as a tractable
proxy for admission control pressure. Live pilot telemetry calibration is required
before cluster deployment (`docs/RESULTS.md` §8).

---

## What This Unlocks

- **Closes ROADMAP rank #6:** "Admission gate → cluster simulator | Wire into cluster
  simulator + Azure 2024 replay" — now fully wired and cross-validated.
- **Opens new gap:** Wire `WorkloadAdmissionGate` (richer KV-cache + queue-p99
  signals) into the simulator as a second admission strategy — compare depth-gate
  vs. KV-pressure gate under matched ρ conditions.
- **Confirmed pattern:** Gate shedding + SRPT preemption = "cut losses early, serve
  what you can" → consistent 4–14% goodput/$ improvement across both traces and all
  load levels tested.
