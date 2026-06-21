# Decoupled Hybrid SRPT Backtest Results

> **Run:** 2026-06-21-l  
> **Shadow-only:** All results are M/G/c discrete-event simulator results, not
> production savings. No production decisions changed by this run.  
> `shadow_only_simulator_result_not_production_savings`

---

## Summary

Decoupled Hybrid implements **SRPT preemption** (pure `remaining_s` trigger, no
aging) combined with **aging dispatch** (`remaining_s / (1 + α·total_wait_s)` queue
selection). This decouples two previously conflated decisions:

- **Preemption (arrival):** should the new arrival displace a running job? → SRPT rule
- **Dispatch (completion):** which waiting job starts next? → aging key

The run -k finding was that a unified aging key for both decisions caused the hybrid
to behave like Aging-SRTF (+64.2% goodput) rather than SRPT (+322.2%). Decoupling
recovers **+184.5% goodput/$ vs FIFO** — 2.6× the hybrid gain — while moderating
long_p99 regression vs pure SRPT.

---

## Benchmark: Azure LLM 2024

**Parameters:** 5,880 requests, ρ=0.85, SLA=10s, c=4 servers, α=0.01, oracle prior

**Command:**
```python
from aurelius.benchmarks.srtf_serving_backtest import run_decoupled_hybrid_backtest
r = run_decoupled_hybrid_backtest(servers=4, target_rho=0.85, aging_alpha=0.01)
```

| Discipline | SLA-safe goodput/$ | vs FIFO | short_p90 (s) | short_p90 impr | long_p99 (s) | long_p99 regr |
|---|---:|---:|---:|---:|---:|---:|
| FIFO (baseline) | 13,336 | — | 696.16 | — | 733.55 | — |
| SRTF-perfect | 56,481 | +323.5% | 3.03 | +99.6% | 2,373 | +223.5% |
| Aging-SRTF (α=0.01) | 22,768 | +70.7% | 152.61 | +78.1% | 1,568 | +113.8% |
| SRPT Preemptive | 56,311 | +322.2% | 1.89 | +99.7% | 2,373 | +223.4% |
| Hybrid (α=0.01) | 21,899 | +64.2% | 169.26 | +75.7% | 1,550 | +111.3% |
| **Decoupled (α=0.01)** | **37,945** | **+184.5%** | **14.41** | **+97.9%** | **1,703** | **+132.3%** |

### Key Observations

1. **+184.5% goodput/$ vs FIFO** — more than 2.6× the gain of both Aging-SRTF
   (+70.7%) and Hybrid (+64.2%). Confirms the root cause: aging dispatch in run -k
   was suppressing SRPT preemption benefit.

2. **long_p99 regression +132.3%** — better than pure SRPT (+223.4%) but worse than
   Aging-SRTF (+113.8%). The aging dispatch provides partial starvation protection
   but fires less systematically than Aging-SRTF's non-preemptive queue discipline.

3. **short_p90 improvement +97.9%** — nearly as good as pure SRPT (+99.7%),
   confirming pure SRPT preemption is the dominant factor for short-job latency.

4. **Decoupling works:** Decoupled uses the same preemption logic as SRPT preemptive
   but adds aging at dispatch time. The net result is between Aging-SRTF and SRPT on
   both goodput and long_p99 — exactly the expected theoretical positioning.

---

## Benchmark: BurstGPT

**Parameters:** 51 requests, ρ=0.85, SLA=30s, c=4 servers

**Command:**
```python
from aurelius.benchmarks.srtf_serving_backtest import run_burstgpt_decoupled_hybrid_backtest
r = run_burstgpt_decoupled_hybrid_backtest(servers=4, target_rho=0.85)
```

| Discipline | SLA-safe goodput/$ | vs FIFO | short_p90 impr | long_p99 regr |
|---|---:|---:|---:|---:|
| FIFO | 70,975 | — | — | — |
| SRPT Preemptive | 67,754 | −4.5% | +67.5% | +16.0% |
| **Decoupled (α=0.01)** | **67,754** | **−4.5%** | **+67.5%** | **+16.1%** |

BurstGPT's 51-request fixture is too small for queue dynamics to manifest. All
preemptive disciplines converge to identical results. Full 1.4M-row dataset needed.

---

## Implementation Notes

### Decoupled Hybrid Algorithm

```
On ARRIVAL of request r at time t:
  If any server is FREE:
    → Start r immediately (no preemption needed)
  Else:
    Find server s* = argmax(remaining_s)          # SRPT rule: no aging factor
    If r.service_s < remaining_s[s*]:
      → Preempt s* (put preempted job back in queue with frozen_wait accumulated)
      → Start r on s*
    Else:
      → Add r to waiting queue

On COMPLETION of request at server s at time t:
  If waiting queue is non-empty:
    For each waiting job j with (remaining_s_j, frozen_wait_j, last_queue_entry_j):
      total_wait_j = frozen_wait_j + (t - last_queue_entry_j)
      key_j = remaining_s_j / (1 + α * total_wait_j)   # aging dispatch key
    Select j* = argmin(key_j)                            # aging-priority dispatch
    → Start j* on s, update frozen_wait for j*'s next preemption
```

Key invariant: `frozen_wait_s` tracks total accumulated queue time across
multiple preemption intervals. This ensures aging priority is computed correctly
even for multiply-preempted requests.

### Why Decoupled Outperforms Hybrid at α=0.01

The run -k hybrid used `remaining_s / (1 + α·total_wait)` for BOTH preemption
and dispatch. At α=0.01, the "flip point" where a waiting 5s job beats a fresh 3s
arrival is `total_wait = (5/3 - 1) / 0.01 = 66.7s`. On the Azure LLM 2024 trace
at ρ=0.85, many requests wait >66.7s before a server frees. When the hybrid
evaluated preemption using this key, it would NOT preempt a long-waiting job (high
effective key from aging denominator) even when a new short arrival appeared.

Decoupled removes aging from preemption entirely. A new 3s arrival ALWAYS preempts
a server running a 5s job (3 < 5), regardless of the running job's accumulated wait.
This restores SRPT's throughput-optimal preemption rule.

---

## Test Suite

**File:** `tests/test_srtf_decoupled_hybrid_backtest.py`  
**Tests:** 42 (9 classes), all passing

| Class | Tests | What it verifies |
|---|---|---|
| `TestDecoupledPreemptionIsPureSRPT` | 5 | Preemption fires on remaining_s < server_remaining_s, ignoring accumulated wait |
| `TestDecoupledDispatchIsAging` | 3 | Dispatch selects by `remaining_s/(1+α·wait)`, favoring long-waiting jobs |
| `TestDecoupledAlphaZeroEqualsSRPT` | 4 | At α=0, decoupled ≡ srpt_preemptive exactly |
| `TestDecoupledCompleteness` | 6 | All requests complete, no deadlocks, sojourn ≥ service_s |
| `TestDecoupledPositioning` | 4 | Decoupled ≥ hybrid (pure SRPT preemption is better), close to srpt at α=0.01 |
| `TestDecoupledAntiStarvation` | 3 | Long jobs complete under heavy short-job stream; higher α helps |
| `TestDecoupledHybridReportDataclass` | 3 | Report structure: 6 disciplines, shadow_tag, all delta fields |
| `TestDecoupledPublicAPI` | 6 | Azure + BurstGPT functions return valid reports, full-trace decoupled >> hybrid |
| `TestDecoupledRunOnTrace` | 8 | Internal helper: completeness, naming, discipline count, small-trace convergence |

---

## Caveats

1. **FIFO baseline, not SLA-aware** — all deltas vs FIFO. An SLA-aware baseline
   would show smaller gains.
2. **Oracle prior** — actual token counts used as predicted. 30%-CV robustness not
   yet tested for decoupled hybrid (tested for non-preemptive SRTF in run -g).
3. **Zero preemption overhead** — real LLM preemption requires KV-cache eviction
   (memory reallocation, potential prefill recompute). Net goodput would be lower.
4. **Single α value** — only α=0.01 benchmarked. Alpha sweep needed to map
   goodput/long_p99 Pareto frontier.
5. **Simulator fidelity** — no batching, speculative decoding, CUDA graph overhead.
   Real serving systems would see different absolute numbers.

---

## Next Steps

1. **Alpha sweep:** α ∈ {0.001, 0.005, 0.01, 0.05} to find Pareto-optimal config.
   Expected: α=0.001 → near-SRPT goodput (+315%) with mild starvation reduction.
2. **Runtime deployment:** Connect decoupled_hybrid (α=0.01 or best-α) to serving
   runtime path driven by `OutputLengthForecastBundle.p50`.
3. **Full BurstGPT:** Run on 1.4M-row dataset; 51-row fixture too small.
