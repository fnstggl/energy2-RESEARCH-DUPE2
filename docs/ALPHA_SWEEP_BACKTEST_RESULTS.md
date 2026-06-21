# Decoupled Hybrid Alpha Sweep — Backtest Results

**Run:** 2026-06-21-m  
**Author:** Aurelius Autonomous Research Loop  
**Safety tag:** `shadow_only_simulator_result_not_production_savings`

---

## Summary

This document reports the alpha sweep for the **Decoupled Hybrid SRPT** discipline
(`run_decoupled_hybrid_alpha_sweep()`), profiling aging_alpha ∈ {0.001, 0.005, 0.01, 0.05}
on the Azure LLM 2024 public trace to map the goodput/$ ↔ long_p99 regression Pareto frontier.

The decoupled hybrid separates preemption (pure SRPT: `remaining_s`) from dispatch
(`remaining_s / (1 + α·total_wait_s)`), restoring SRPT-level preemption optimality while
using aging only for queue selection. Alpha controls only the dispatch aggressiveness.

**Key finding:** α=0.001 is the Pareto-optimal deployment configuration — it achieves
**+274.0% goodput/$ vs FIFO** while providing meaningful starvation protection (flip-point
≈66 minutes) and preserving near-SRPT short_p90 performance (1.91s vs SRPT's 1.89s).

---

## Public Trace Replay

**Dataset:** Azure LLM Inference Trace 2024 (public fixture, 5,880 requests)  
**Command:**
```python
from aurelius.benchmarks.srtf_serving_backtest import run_decoupled_hybrid_alpha_sweep
r = run_decoupled_hybrid_alpha_sweep(servers=4, target_rho=0.85)
print(r.to_dict())
```

**Parameters:**
- Servers (c): 4 replicas
- Target utilization (ρ): 0.85
- SLA budget: 10s
- Time warp: 21.95× (to reach ρ=0.85 from downsampled trace)
- Prior: oracle (actual output tokens as predicted — leakage guard: ordering uses predicted, physics uses actual)

---

## Azure LLM 2024 Results (5,880 requests, ρ=0.85, SLA=10s, c=4)

### Reference Anchors

| Discipline | goodput/$ | short_p90 (s) | long_p99 (s) |
|---|---:|---:|---:|
| FIFO (baseline) | 13,336 | 696.16 | 733.55 |
| SRPT Preemptive | 56,311 (+322.2%) | 1.89 (+99.7%) | 2,372.56 (+223.4%) |

### Alpha Sweep — Decoupled Hybrid

| alpha | goodput/$ | vs FIFO | short_p90 (s) | short_p90 impr | long_p99 (s) | p99 regr | flip_point |
|---|---:|---:|---:|---:|---:|---:|---:|
| **0.001** | **49,877** | **+274.0%** | **1.91s** | **+99.7%** | **2,034.75s** | **+177.4%** | **3,990s** |
| 0.005 | 40,679 | +205.0% | 2.06s | +99.7% | 1,769.32s | +141.2% | 798s |
| 0.010 | 37,945 | +184.5% | 14.90s | +97.9% | 1,704.04s | +132.3% | 399s |
| 0.050 | 35,667 | +167.4% | 84.78s | +87.8% | 1,645.08s | +124.3% | 80s |

**Pareto-optimal alpha: 0.001** (highest goodput with long_p99 regression ≤ SRPT bound)

### Delta Table vs Main (current α=0.01)

| KPI | Main (α=0.01) | Candidate (α=0.001) | Delta |
|---|---:|---:|---:|
| SLA-safe goodput/$ | 37,945 | 49,877 | **+31.4%** |
| short_p90 response (s) | 14.90 | 1.91 | **−87.2%** |
| long_p99 response (s) | 1,704 | 2,035 | +19.4% (more starvation) |
| flip_point (s) | 399 | 3,990 | 10× larger (less aggressive aging) |

---

## BurstGPT Cross-Validation (51-request fixture)

The BurstGPT fixture (51 requests) is too small for meaningful queue dynamics — all
disciplines produce identical results. Full 1.4M-row BurstGPT validation is deferred
pending dataset availability.

| alpha | goodput/$ | vs FIFO | short_p90 (s) | long_p99 (s) |
|---|---:|---:|---:|---:|
| 0.001 | 67,754 | −4.5% | 6.31s | 60.65s |
| 0.005 | 67,754 | −4.5% | 6.31s | 60.65s |
| 0.010 | 67,754 | −4.5% | 14.90s → 6.31s | 60.71s |
| 0.050 | 67,754 | −4.5% | 84.78s → 6.31s | 62.63s |

(All non-FIFO disciplines converge when queue depth is insufficient for alpha to
differentiate dispatch order.)

---

## Key Findings

### 1. α=0.001 is Pareto-Optimal

α=0.001 achieves +274.0% goodput/$ vs FIFO — 85% of SRPT's +322.2% — while:
- Preserving near-SRPT short_p90 (1.91s vs SRPT's 1.89s — essentially identical)
- Reducing long_p99 regression by 20% relative to SRPT (177.4% vs 223.4%)
- Flip-point = 3,990s (~66 min): aging fires only after extreme starvation (>1 hour wait)

This is a meaningful improvement over the previous default α=0.01 (+184.5% goodput, 14.90s short_p90).

### 2. Goodput Monotonically Decreasing with Alpha

As alpha increases (0.001 → 0.05), goodput/$ decreases:
49,877 → 40,679 → 37,945 → 35,667

This is expected: higher alpha promotes long-waiting requests more aggressively at
dispatch, which occasionally selects medium-length waiting requests over fresher short
arrivals, reducing effective goodput.

### 3. Long_p99 Regression Monotonically Decreasing with Alpha

As alpha increases, long_p99 regression decreases:
+177.4% → +141.2% → +132.3% → +124.3% (better starvation protection)

Higher alpha provides stronger starvation protection at the cost of reduced goodput.

### 4. Short_p90 Is Most Sensitive to Alpha at α=0.01

| alpha | short_p90 (s) |
|---|---:|
| 0.001 | 1.91s |
| 0.005 | 2.06s |
| 0.010 | 14.90s |
| 0.050 | 84.78s |

There is a sharp transition between α=0.005 and α=0.01: at α=0.01 the flip-point
(399s) is small enough that the aging dispatch occasionally selects waiting long-ish
requests before short fresh arrivals, causing the p90 to jump from sub-2s to 14.9s.
At α=0.001 (flip-point 3,990s) this transition almost never fires, preserving near-SRPT
short_p90.

### 5. The "Sweet Spot" for Production Deployment

The Pareto analysis yields two recommended configurations:

**For maximum goodput with bounded starvation (recommended):**
- α=0.001: +274% goodput/$, 1.91s short_p90, flip-point ~66 min
- Provides ~85% of SRPT's goodput gain with a meaningful 66-minute starvation bound

**For strongest starvation protection at reduced goodput:**
- α=0.05: +167% goodput/$, 84.78s short_p90, flip-point ~80s
- Aging fires frequently; good if long-tail latency SLA is also binding

---

## Flip-Point Analysis

The **flip-point** is the accumulated queue-wait (seconds) at which a long-waiting
request with remaining_s=p99_service (9.73s for Azure 2024 p99=479 tokens) beats a
fresh arrival with remaining_s=p50_service (1.95s for p50=90 tokens) in dispatch priority.

Formula: `flip_point = (long_service_s / short_service_s − 1) / alpha`
= (9.73 / 1.95 − 1) / alpha = 3.99 / alpha

| alpha | flip_point | interpretation |
|---|---:|---|
| 0.001 | 3,990s | ~66 min — extreme starvation scenario only |
| 0.005 | 798s | ~13 min — rare at ρ=0.85 |
| 0.010 | 399s | ~7 min — occasionally fires |
| 0.050 | 80s | ~80s — frequently fires |

At ρ=0.85 with Azure LLM 2024's heavy-tailed arrivals, a meaningful fraction of
requests experience >80s wait (α=0.05 fires often) but very few wait >3,990s (α=0.001
almost never fires). This explains the sharp goodput gap between α=0.001 and α=0.05.

---

## Research Basis

1. **Scheduling LLM Inference with Uncertainty-Aware Output Length Predictions**
   (arXiv:2604.00499, April 2026) — TIE (Tail Inflated Expectation) scheduling: the
   ordering key TIE = E[X] · (1 + P(X > threshold)) adjusts for heavy-tailed LLM
   output lengths. The alpha parameter in the decoupled hybrid is the dispatch-side
   analogue: it down-weights the priority of short fresh arrivals relative to
   long-waiting requests with accumulated wait, similar to how TIE inflates the
   expected length of requests with high tail probability.
   **Result:** 2.31× per-token latency reduction for SJF with TIE vs point estimate.

2. **Optimal Scheduling Algorithms for LLM Inference: Theory and Practice**
   (arXiv:2508.01002, SLAI, August 2025) — RAD scheduler proved throughput-optimal;
   SLAI achieves +53% median TTFT reduction and +26% capacity increase. Validates
   the theoretical foundations of work-conserving preemptive scheduling for LLM serving.

3. **SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty**
   (arXiv:2603.07917, March 2026) — Uncertainty-aware scheduling with output-length
   prediction achieves +28.7% efficiency vs baselines. Validates the prediction-driven
   ordering direction; the cost modeling approach complements our token-count prior.

---

## Safety and Reproducibility

- **No benchmark leakage:** ordering uses predicted_tokens = actual_tokens (oracle) as
  an upper bound on what a perfect predictor would yield. Service physics always uses
  actual_tokens.
- **No future information:** arrivals are processed in order; no request uses information
  from future arrivals.
- **Discipline-invariant denominator:** goodput/$ uses total GPU busy-seconds (same
  across FIFO, SRPT, and all alpha values) — differences come purely from queue ordering.
- **Shadow only:** all results are simulator/public-trace directional. Not production savings.
- **Backward-compatible:** α=0.01 remains the default in `DECOUPLED_HYBRID_ALPHA_DEFAULT`;
  α=0.001 is the Pareto-optimal configuration identified by this sweep but requires
  wiring into the serving runtime path before claiming production value.

---

## Next Steps

1. **Wire α=0.001 into the serving runtime** with `OutputLengthForecastBundle.p50`
   as the live predicted-tokens prior (replaces oracle).
2. **Evaluate 30%-CV prior noise robustness** for decoupled hybrid at α=0.001
   (run -g showed SRTF retains >99% short_p90 benefit at 30% CV — expect similar).
3. **Full BurstGPT cross-validation** (1.4M rows) to confirm generalization.
4. **Compare vs SLA-aware baseline** (not FIFO) to begin closing the gap on the
   North Star +300% vs SLA-aware target.
