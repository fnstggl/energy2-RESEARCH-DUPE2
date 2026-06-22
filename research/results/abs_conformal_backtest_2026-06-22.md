# Absolute-Error Conformal Calibration Backtest Results

**Run:** 2026-06-22-x  
**Date:** 2026-06-22  
**Hypothesis:** Replacing p90 relative prediction error with p90 absolute error in the conformal alpha calibrator eliminates the spurious α cap caused by short-request over-predictions, driving α closer to the Pareto-optimal 0.001 and improving SLA-safe goodput/$ on both public LLM traces.

---

## Summary

**FRONTIER IMPROVEMENT CONFIRMED ON BOTH PUBLIC TRACES**

| Trace | Discipline | Goodput/$ | vs FIFO | vs Rel-conformal | α | Oracle Retention |
|---|---|---|---|---|---|---|
| Azure LLM 2024 | FIFO baseline | 13,336 | — | — | — | — |
| Azure LLM 2024 | Oracle (upper bound) | 56,311 | +322.24% | — | →0 | 100% |
| Azure LLM 2024 | Rel-conformal (live) | 45,933 | +244.42% | — | 0.00200 (CAPPED) | 81.6% |
| Azure LLM 2024 | **Abs-conformal (live) NEW** | **55,097** | **+313.14%** | **+19.95%** | **0.000222** | **97.8%** |
| BurstGPT HF | FIFO baseline | 6,529 | — | — | — | — |
| BurstGPT HF | Oracle (upper bound) | 48,599 | +644.38% | — | →0 | 100% |
| BurstGPT HF | Rel-conformal (live) | 34,004 | +420.83% | — | 0.00199 (CAPPED) | 70.0% |
| BurstGPT HF | **Abs-conformal (live) NEW** | **42,902** | **+557.12%** | **+26.17%** | **0.000562** | **88.3%** |

---

## Root Cause (Addressed)

The rel-conformal calibrator was capped at `α = 2 × alpha_max = 0.002` on both traces. The formula:

```
α = alpha_max × min(2.0, p90_rel_err / target_p90_error)
```

with `target_p90_error = 0.40` caps whenever `p90_rel_err ≥ 0.80`. On BurstGPT HF with running-median prior (~18 tokens), short ChatGPT requests (actual=7, predicted=18) produce `rel_err = |18-7|/7 = 1.57`, which is scheduling-irrelevant (11-token misprediction ≈ 1 second service) but dominates the p90 tail, forcing `α = 0.002` — 2× above the Pareto-optimal 0.001.

## Fix

Replace relative error with absolute error:

```python
abs_err = abs(predicted_tokens - actual_tokens)
p90_abs_err = np.percentile(residuals, 90)
α = alpha_max × min(2.0, p90_abs_err / target_p90_abs_tokens)
```

With `target_p90_abs_tokens = 500`: p90 abs_err is driven by genuinely uncertain long requests (GPT-4 + surprise-long ChatGPT ≈ 300-600 tokens). Short over-predictions (11 tokens) no longer dominate.

## Mechanism

| Signal | Rel-error value | Abs-error value | Scheduling relevance |
|---|---|---|---|
| ChatGPT short (actual=7, pred=18) | 1.57 (DOMINATES p90) | 11 tokens (≈1s) | LOW — scheduling irrelevant |
| GPT-4 long (actual=600, pred=350) | 0.42 | 250 tokens (≈20s) | HIGH — genuine uncertainty |
| ChatGPT surprise-long (actual=800, pred=18) | 43.4 | 782 tokens (≈63s) | HIGH — genuine uncertainty |

Rel-error p90 is controlled by the first row (spurious). Abs-error p90 is controlled by the second and third rows (real uncertainty).

## Results Details

### Azure LLM 2024 (5,880 requests, 4 servers, ρ=0.85, SLA=10s)

- p90 abs_err tokens (abs calibrator): **509 tokens**
- abs_mean_alpha: **0.000222** (vs rel_mean_alpha 0.001999 — **11× lower**)
- Oracle retention: **97.8%** (was 81.6% — **+16.2pp**)
- Goodput/$: 55,097 vs FIFO 13,336 → **+313.14%** (was +244.42%)

### BurstGPT HF (5,880 requests, 4 servers, ρ=0.85, SLA=30s)

- p90 abs_err tokens (abs calibrator): **632 tokens**
- abs_mean_alpha: **0.000562** (vs rel_mean_alpha 0.001990 — **3.5× lower**)
- Oracle retention: **88.3%** (was 70.0% — **+18.3pp**)
- Goodput/$: 42,902 vs FIFO 6,529 → **+557.12%** (was +420.83%)

## Implementation

- **New class**: `AbsoluteErrorConformalCalibrator` in `aurelius/benchmarks/srtf_serving_backtest.py`
- **New constant**: `CONFORMAL_ABS_TARGET_P90_TOKENS = 500.0`
- **New discipline**: `"decoupled_hybrid_abs_conformal"` in `simulate_queue()`
- **New simulator**: `_simulate_decoupled_hybrid_abs_conformal()`
- **New report**: `AbsConformalReport`, `run_abs_conformal_azure_backtest()`, `run_abs_conformal_burstgpt_backtest()`
- **New tests**: 28 unit + integration tests (all passing)
- **New script**: `scripts/run_abs_conformal_backtest.py`

## Research Basis

- arXiv:2302.07675 — Conformal Prediction for Scheduling (Cohen et al. 2023): formal scheduling guarantees via online conformal prediction
- arXiv:2604.07931 — Robust Length Prediction (2026): heavy-tailed distribution model for output length
- arXiv:2509.23384 — NexusSched: scheduling with absolute error bounds
- arXiv:1902.00732 — Scheduling with Predictions (Mitzenmacher & Vassilvitskii 2020)

## Decision

**FRONTIER IMPROVEMENT — Merge to main.**

Abs-conformal achieves 97.8% oracle retention on Azure (near-SRPT with running-median prior) and 88.3% on BurstGPT. This closes the previously-binding constraint across 5 consecutive non-improvement runs (-s through -w). The running-statistics retention ceiling (81.6%/70.0%) is broken.

## Shadow Tag

`shadow_only_simulator_result_not_production_savings`

Results are discrete-event M/G/c simulator on public traces. Not a production measurement.
