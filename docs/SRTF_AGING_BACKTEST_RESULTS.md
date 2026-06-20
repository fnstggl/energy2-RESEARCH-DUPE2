# SRTF+Aging Anti-Starvation Backtest Results

**Run:** 2026-06-20-i
**Classification:** RESEARCH DISCOVERY + INFRASTRUCTURE
**Module:** `aurelius/benchmarks/srtf_serving_backtest.py`
**Tests:** `tests/test_srtf_aging_backtest.py` (48 tests, all passing)

> **Binding honesty label:** Discrete-event simulator result on public Azure LLM 2024
> trace (5,880 requests).  Directional only.  Baseline is FIFO (not SLA-aware).
> Not production savings.

---

## Objective

Add an anti-starvation guard to the SRTF serving queue (run 2026-06-20-g).
Pure SRTF improved short-request p90 by −99.6% and SLA-safe goodput/$ by +323%
vs FIFO, but at the cost of severe long-request starvation (p99: 733s → 2189s).
Goal: preserve the SRTF gain for short requests while bounding the long-request tail.

---

## Algorithm Implemented

`simulate_queue_with_aging()` — non-preemptive M/G/c discrete-event simulator
with linear-aging anti-starvation:

```
effective_key(t) = max(0.0, predicted_tokens − alpha × (t − arrival_s))
```

When a request has waited `predicted_tokens / alpha` seconds, its effective_key
reaches 0 — it is treated as a "zero-length" job and dispatched next (bounded
starvation).  Alpha = 1.844 tokens/second, derived as:

```
alpha = max_azure_tokens(1346) / fifo_p99_wait_s(730) ≈ 1.844
```

At this alpha:
- Short request (p50 = 90 tokens): ages to 0 after 90/1.844 ≈ 49s
- Long request (max = 1346 tokens): ages to 0 after 1346/1.844 ≈ 730s ≈ FIFO p99

---

## Results: Azure LLM 2024 (c=4 servers, ρ=0.85, SLA=10s)

| Discipline | short_p90 (s) | long_p99 (s) | overall_p99 (s) | goodput/$ | vs FIFO |
|---|---:|---:|---:|---:|---:|
| FIFO | 696.2 | 733.6 | 732.7 | 13,336 | — |
| SRTF (no aging) | **3.0** | 2373.1 | 2188.7 | **56,481** | **+323%** |
| SRTF+aging (α=1.844) | 696.2 | **733.6** | **732.7** | 17,783 | **+33%** |

---

## Key Finding: Age-Out Wave Failure Mode

**SRTF+aging degrades to FIFO at ρ=0.85.**  The algorithm fails to preserve
the SRTF short-p90 benefit.  Mechanism:

1. Under pure SRTF, long requests are severely starved — they accumulate unserved.
2. After `pred/alpha` seconds (≈542s for a 1000-token request), each starved long
   request ages to `effective_key=0` and jumps to high-priority.
3. At ρ=0.85 with the Azure trace, MANY long requests starve simultaneously.
   Around t≈500–730s of simulation, ALL of them fire the starvation trigger at
   nearly the same time — a **"wave"** of `effective_key=0` long requests floods
   the dispatch queue.
4. Fresh short requests (effective_key ≈ 90, having just arrived) **lose** to aged
   long requests (effective_key=0, winning the arrival-time tiebreak).
5. Short requests back up, wait > 49s → they also age to 0 → system settles into
   FIFO order.

The 33% goodput improvement is real but **transient**: it reflects the initial
SRTF phase (t=0 to first age-out wave) before the system collapses to FIFO.

---

## What Works vs. What Doesn't

| Property | Result |
|---|---|
| Starvation bounded vs pure SRTF | ✅ overall p99 = 732.7s vs SRTF's 2188.7s |
| Long-request p99 not worse than FIFO | ✅ long_p99 = 733.6s = FIFO long_p99 |
| Short-request p90 benefit preserved | ❌ short_p90 = 696.2s = FIFO short_p90 |
| Meaningful goodput improvement | ✅ +33% vs FIFO (transient, not steady-state) |
| BurstGPT cross-validation runs | ✅ (54 requests fixture, direction preserved) |

---

## Why Non-Preemptive Aging Cannot Work at High Load

The self-consistent SRTF equilibrium (short requests served in ~3s) requires:
- Short requests never exceed their aging threshold (90/alpha = 49s) during normal operation
- Under SRTF, this holds: short requests are dispatched in ~3s << 49s

But the FIFO equilibrium is **also stable**:
- All requests wait ~341s (FIFO mean wait) >> 49s → all short requests age to 0
- System is FIFO → all wait 341s → cycle holds

At ρ=0.85, the system is initialized in an empty state (cold-start transient),
briefly enjoys SRTF ordering, then collapses into the stable FIFO equilibrium
once the age-out wave fires.  Preemption breaks this cycle by serving short
requests mid-flight through a long request's execution.

---

## Path Forward

### Option A: Preemptive SRPT
Suspend a running long request when a shorter request arrives.  Resume when a
server is free.  In LLM serving this requires KV-cache checkpoint (paged attention
makes save/restore practical at page granularity).

- **PecSched** (arXiv:2409.15104): preemptive scheduling → 92% p99 reduction,
  595% throughput improvement for short inputs.
- **Equinox** (arXiv:2508.16646): holistic fair scheduling → 1.3× throughput,
  60% TTFT reduction.

Implementation: extend `simulate_queue` with preemption events; add a KV-overhead
model (checkpoint_cost_s = pages × checkpoint_us_per_page).

### Option B: Resource Partitioning
Reserve K servers exclusively for long requests (FIFO); run SRTF on (c-K) servers
for short requests.  Short requests get their SRTF benefit on dedicated servers;
long requests get guaranteed throughput on K servers.

- No preemption required.
- Capacity waste: K servers idle when long queue is empty.
- Optimize K via grid search on Azure trace.

---

## Infrastructure Added

| Component | Description |
|---|---|
| `simulate_queue_with_aging()` | O(n) linear scan at each dispatch event |
| `SRTFAgingReport` | Dataclass covering Azure + BurstGPT metrics |
| `run_srtf_aging_backtest()` | FIFO / SRTF / SRTF+aging on both traces |
| `load_burstgpt_requests()` | CSV loader for BurstGPT fixture |
| `_summarize()` extended | Now includes `long_p90_response_s`, `long_p99_response_s` |
| 48 new tests | 9 test classes covering algorithm, starvation, BurstGPT, report |
