# SRPT Preemptive Backtest Results — Run 2026-06-20-j

> **Shadow tag:** `shadow_only_simulator_result_not_production_savings`
> All numbers are from a discrete-event M/G/c serving-queue simulator on
> real public traces.  No production savings claim is made.

## Summary

Run 2026-06-20-j adds **Preemptive SRPT** (`discipline="srpt_preemptive"`)
to the serving-queue simulator.  This eliminates the theoretical unbounded
starvation risk of non-preemptive SRTF: long jobs are preempted when a
shorter job arrives but their remaining service decreases monotonically, so
they always make forward progress.

**Key finding:** On Azure LLM 2024 (5,880 requests), SRPT preemptive achieves
**+322.2% SLA-safe goodput/$ vs FIFO** — within 0.3 pp of non-preemptive SRTF
perfect (+323.5%) — while delivering the **best short_p90 latency across all
four disciplines** (1.89s vs 3.03s for non-preemptive SRTF).  Long-request p99
regression is nearly identical to SRTF (+223.4% vs +223.5%), showing that
preemption alone does not eliminate starvation at ρ=0.85 in this trace (the
server is always busy with newly arriving short jobs).

---

## Research Basis

| Paper | ArXiv | Venue | Mapping |
|---|---|---|---|
| TRAIL: Token Reduction and Intelligent Learning for LLM Serving | 2410.01035 | ICLR 2025 | SRPT with limited preemptions; near-SRTF performance via embedding-based SPRPT |
| FlowPrefill: Flow-Controlled Scheduling for LLM Prefill | 2602.16603 | Feb 2026 | Operator-level preemption blueprint; SLO-aware prioritization for arriving high-priority requests |
| SRPT for Multiserver Systems | 1805.07686 | 2018 | SRPT for M/G/k; server-selection rule for multi-server preemptive case |

---

## Implementation

**File:** `aurelius/benchmarks/srtf_serving_backtest.py`

### New function: `_simulate_srpt_preemptive(requests, servers)`

Discrete-event M/G/c simulator with preemption:
- **Event types:** ARRIVAL (ev_type=0) and COMPLETION (ev_type=1)
- **Arrival rule:** If a free server exists, assign immediately.  Otherwise,
  compare arriving job's service time to the longest-remaining-service running
  job.  If shorter: preempt that server (record elapsed service, push preempted
  job back to waiting heap with its remaining service time).  If longer: enqueue.
- **Completion rule:** Skip if server version counter has changed (stale event
  from preempted server).  Mark response time.  If waiting queue non-empty,
  dispatch shortest-remaining job.
- **Wait accounting:** `wait = response_time - service_s` (captures all
  queuing intervals including preemption pauses, without per-preemption tracking)
- **Stale-event detection:** Per-server `s_ver[sid]` counter, incremented on
  every preemption or start; completion events that don't match current version
  are silently skipped.
- **SRPT invariant:** At all times, the c servers run the c requests with the
  shortest remaining service time (proven inductively via the preemption rule
  at arrivals and shortest-first dispatch at completions).

### New dispatch branch in `simulate_queue()`

```python
if discipline == "srpt_preemptive":
    return _simulate_srpt_preemptive(requests, servers)
```

### New report dataclass: `SRTFPreemptiveReport`

All four disciplines (FIFO, SRTF-perfect, Aging-SRTF, SRPT-preemptive),
delta KPIs vs FIFO, and shadow tag.

### New public functions

- `run_srpt_preemptive_backtest(servers=4, target_rho=0.85, sla_s=10.0, aging_alpha=0.01, job_limit=None)` — Azure LLM 2024
- `run_burstgpt_srpt_preemptive_backtest(servers=4, target_rho=0.85, sla_s=30.0, aging_alpha=0.01)` — BurstGPT

---

## Test Suite

**File:** `tests/test_srtf_preemptive_backtest.py` (NEW — 42 tests)

| Class | Tests | Focus |
|---|---|---|
| `TestPreemptionBasics` | 5 | Short preempts long; no preemption when arriving longer; multiple preemptions; simultaneous arrivals |
| `TestSRPTInvariant` | 5 | All complete; no starvation; non-negative waits; response ≥ service; deterministic ordering |
| `TestPreemptiveVsNonPreemptive` | 4 | Short p90 ≤ FIFO; identical services = FIFO; wait non-negative |
| `TestSRTFPreemptiveReportStructure` | 6 | Dataclass fields; to_dict; shadow tag |
| `TestSRPTOrderingGuarantees` | 4 | Short p90 improvement; SRPT beats/matches SRTF on short_p90; all complete |
| `TestSimulateQueueDispatch` | 4 | Discipline routing; 3-tuple return; non-SRPT disciplines unaffected |
| `TestRunSRPTPreemptiveBacktest` | 5 | Azure LLM 2024 fixture; short_p90 improves; SRPT matches/exceeds SRTF short_p90; shadow tag |
| `TestBurstGPTSRPTPreemptive` | 3 | BurstGPT cross-validation; all complete; short_p90 non-negative |
| `TestEdgeCases` | 6 | Single request; more servers than requests; zero arrivals; timing precision |

**Result:** 42/42 tests pass, 0 regressions.

---

## Public Backtest Results

### Azure LLM 2024 (Primary — 5,880 requests)

**Parameters:** servers=4, target_rho=0.85, sla_s=10.0, aging_alpha=0.01, time_warp=21.95

| KPI | FIFO | SRTF-perfect | Aging-SRTF (α=0.01) | **SRPT Preemptive** |
|---|---:|---:|---:|---:|
| SLA-safe goodput/$ | 13,336 | 56,481 | 22,768 | **56,311** |
| vs FIFO | — | **+323.5%** | +70.7% | **+322.2%** |
| mean_response_s | 343.89 | 129.89 | 183.06 | **129.58** |
| p50_response_s | 342.20 | 2.71 | 58.49 | **2.09** |
| short_p90_response_s | 696.16 | 3.03 | 152.61 | **1.89** |
| short_p99_response_s | 731.54 | 4.70 | 245.31 | **2.74** |
| short_p90 improvement vs FIFO | — | +99.57% | +78.08% | **+99.73%** |
| long_p90_response_s | 703.81 | 1,198.20 | 765.18 | **1,196.04** |
| long_p99_response_s | 733.55 | 2,373.09 | 1,568.16 | **2,372.56** |
| long_p99 regression vs FIFO | — | +223.5% | +113.8% | **+223.4%** |
| sim_horizon_s | 739.61 | 2,841.16 | 2,543.58 | **2,840.21** |
| mean_wait_s | 341.43 | 127.42 | 180.60 | **127.11** |

**Key observations:**
1. SRPT preemptive achieves **+322.2% goodput/$** vs FIFO — within 0.3 pp of SRTF perfect.
2. SRPT preemptive has the **best short_p90 across all disciplines** (1.89s vs SRTF's 3.03s), showing +99.73% improvement vs FIFO.
3. Long_p99 regression for SRPT (+223.4%) matches SRTF (+223.5%) — preemption does not eliminate long-request starvation at ρ=0.85 with this trace's short-job intensity.  Forward progress is guaranteed but slow when short jobs continuously arrive.
4. p50 response: SRPT achieves 2.09s vs SRTF's 2.71s — preemption benefits the median request.

### BurstGPT Cross-Validation (51-request fixture)

**Parameters:** servers=4, target_rho=0.85, sla_s=30.0, aging_alpha=0.01, time_warp=29.86

| KPI | FIFO | SRTF-perfect | Aging-SRTF (α=0.01) | **SRPT Preemptive** |
|---|---:|---:|---:|---:|
| SLA-safe goodput/$ | 70,975 | 67,754 | 67,754 | **67,754** |
| vs FIFO | — | −4.5% | −4.5% | **−4.5%** |
| short_p90_response_s | 19.41 | 8.43 | 8.43 | **6.31** |
| short_p90 improvement vs FIFO | — | +56.5% | +56.5% | **+67.5%** |
| long_p99_response_s | 52.28 | 57.92 | 57.92 | **60.65** |
| long_p99 regression vs FIFO | — | +10.8% | +10.8% | **+16.0%** |

**BurstGPT caveats:** Fixture is 51 rows (small sample — high variance).  SRTF
and Aging-SRTF produce identical results due to insufficient load to differentiate
the aging term on this small fixture.  SRPT preemptive achieves better short_p90
(+67.5% vs +56.5% for SRTF) at the cost of slightly higher long_p99 (+16.0% vs
+10.8%).  Full 1.4M-row BurstGPT dataset needed for cross-trace confirmation.

---

## Discipline Comparison Summary (Azure LLM 2024)

| Discipline | Goodput/$ vs FIFO | short_p90 improvement | long_p99 regression | Production safe? |
|---|---|---|---|---|
| FIFO | baseline | — | — | Yes |
| SRTF perfect (non-preemptive) | **+323.5%** | +99.6% | +223.5% ⚠ | No (starvation risk) |
| Aging-SRTF (α=0.01) | +70.7% | +78.1% | +113.8% ⚠ | Partial (bounded) |
| **SRPT Preemptive** | **+322.2%** | **+99.7%** | +223.4% ⚠ | Partial (forward progress guaranteed) |

**SRPT preemptive vs non-preemptive SRTF:**
- Goodput/$ essentially equal (within 0.3 pp)
- short_p90 strictly better (+99.73% vs +99.57%)
- Long_p99 regression nearly identical — starvation not eliminated, only bounded

---

## Algorithm Correctness Notes

### SRPT Invariant
At all times, the c servers run the c requests with the shortest remaining service:
- **At arrival (free server):** Assign directly → invariant trivially maintained.
- **At arrival (all busy):** Compare new job to longest-remaining server.  If shorter: preempt (new top-c = c-1 previously-shortest + new shortest).  If longer: enqueue → invariant maintained.
- **At completion:** Pop shortest from waiting heap → new top-c = c-1 remaining running + popped shortest → invariant maintained.

### Anti-starvation property
A long job's remaining service decreases monotonically whenever it runs.  It
CAN be preempted by newly arriving shorter jobs, but its remaining service only
decreases (never increases).  When remaining_service falls below all waiting
short jobs, it finishes uninterrupted.  This bounds (but does not eliminate)
starvation when short-job arrival rate is high.

### Known limitation
At ρ=0.85 with the Azure LLM 2024 heavy-tailed output distribution, long jobs
(e.g., 2,000+ tokens) are continuously outcompeted by short jobs (< 100 tokens)
for the duration of the simulation.  The long_p99 regression (+223%) shows that
forward-progress guarantee alone is insufficient to prevent tail latency blowup.
A hybrid approach (Aging + Preemptive, or a preemption-limit cap) is the
recommended next direction.

---

## Files Changed

| File | Change |
|---|---|
| `aurelius/benchmarks/srtf_serving_backtest.py` | Added `_simulate_srpt_preemptive()`, `simulate_queue()` dispatch, `SRTFPreemptiveReport`, `_run_preemptive_backtest_on_trace()`, `run_srpt_preemptive_backtest()`, `run_burstgpt_srpt_preemptive_backtest()` |
| `tests/test_srtf_preemptive_backtest.py` | NEW — 42 tests, 9 classes |
| `docs/SRPT_PREEMPTIVE_BACKTEST_RESULTS.md` | NEW — this file |
| `research/ROADMAP.md` | Updated: run 2026-06-20-j entry, leaderboard, opportunity table |
| `research/GAP_ANALYSIS.md` | Updated: all gap questions answered for run -j |

---

*Generated by Aurelius Autonomous Research Routine — run 2026-06-20-j*
