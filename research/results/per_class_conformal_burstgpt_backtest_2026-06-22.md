# Per-Class Conformal Calibration Backtest — Run 2026-06-22-w

**Date:** 2026-06-22  
**Trace:** BurstGPT HF (JSONL, 5,880 requests, CC-BY-4.0)  
**Benchmark command:**
```python
from aurelius.benchmarks.srtf_serving_backtest import run_burstgpt_per_class_conformal_backtest
report = run_burstgpt_per_class_conformal_backtest(job_limit=5880)
```

## Motivation

Runs -t/-u/-v established the running-statistics ceiling: global conformal calibration
achieves ~70% oracle retention regardless of prediction quality (global, stratified, or
ML-HGB priors all converge at α≈0.002 capped). The root cause: BurstGPT's ChatGPT class
(70% of traffic) has extreme within-class variance (p5=1, p95=800+ tokens), keeping global
p90 relative error ≥ 0.80 and capping α for ALL requests.

This run tests per-class conformal calibration (separate ConformalAlphaCalibrator per
model_id) — the #1 ranked opportunity from the gap analysis. Hypothesis: GPT-4 requests
(30% of traffic, tight distribution) should get α → 0 independently, achieving near-SRPT
dispatch for the GPT-4 class.

## Method

**Per-class conformal calibration (new `PerClassConformalCalibrator`):**
- One `ConformalAlphaCalibrator` per model_id class (ChatGPT, GPT-4)
- Per-class warmup: 50 completions (half of global 100) before switching from global fallback
- On completion: update per-class residual window for req.model_id
- On dispatch: use per-class α for each waiting request's class
- Fallback to global calibrator if class has < 50 completions

**Prior:** ML-HGB (same as run -v): trained on Phase 1 completions (first 300), predicts
GPT-4 accurately (~235 tokens) but not ChatGPT surprise-long requests.

**Simulator:** M/G/c discrete-event, 4 servers, ρ=0.85, SLA=30s.

## Results

| Discipline | Goodput/$ | vs FIFO | vs Oracle retention |
|---|---:|---:|---:|
| FIFO | 6,528.76 | — | — |
| Oracle conformal | 48,598.82 | +644.38% | 100.00% |
| Global conformal (ML-HGB) | 34,003.60 | +420.83% | 65.31% |
| **Per-class conformal (ML-HGB)** | **34,100.59** | **+422.31%** | **65.54%** |

**Per-class vs global: +0.29%**

## Per-Class α Diagnostics

| Class | Mean α | Completions |
|---|---:|---:|
| ChatGPT | 0.001994 | 4,137 |
| GPT-4 | 0.002000 | 1,743 |
| Global | 0.001990 | — |

**Critical finding:** Both classes hit the α cap (0.002). GPT-4 per-class α never
drops below the cap, even with accurate ML-HGB predictions and per-class calibration.

## Root Cause Analysis: Within-Class Variance Ceiling

The within-class variance ceiling confirmed — the running-statistics ceiling is not
between-class (ChatGPT dominating GPT-4's global calibrator), but within-class:

**GPT-4 intra-class token distribution:**
- GPT-4 mean ≈ 235 tokens with large spread (CV estimated 40-60%)
- ML-HGB predicts GPT-4 p50 ≈ 235 tokens
- Short GPT-4 requests (100 tokens): rel_err = |235-100|/100 = 1.35
- Long GPT-4 requests (500 tokens): rel_err = |235-500|/500 = 0.53
- Per-class p90 rel_err ≈ 0.40-0.80 → ratio ≥ 1.0 → α ≥ alpha_max (0.001)

Even with per-class calibration, GPT-4's within-class variance ensures its per-class
p90 relative error remains at or above the target threshold (0.40), preventing α from
converging to 0.

**Phase 1 contamination:** Additional issue — Phase 1 running-median predictions for
GPT-4 have rel_err ≈ 0.97 (predicted 7 tokens for 235-token GPT-4 outputs). These
Phase 1 residuals persist in the per-class window during the early Phase 2 transition,
but even after the window flushes Phase 1 residuals, GPT-4's within-class variance
keeps per-class p90 rel_err above target.

## Test Suite

26 new tests in `tests/test_per_class_conformal_backtest.py`:
- 7 unit tests for `PerClassConformalCalibrator`
- 6 simulation invariant tests
- 4 `PerClassConformalReport` serialization tests
- 3 two-model synthetic hypothesis tests (key: independent class calibration confirmed)
- 5 BurstGPT HF integration tests
- 1 constant validation test

All 26 pass. No regression in existing 157 SRTF tests.

## Comparison Table

| KPI | Main (Global) | Candidate (Per-Class) | Delta |
|---|---:|---:|---:|
| SLA-safe goodput/$ | 34,003.60 | 34,100.59 | +0.29% |
| vs FIFO delta | +420.83% | +422.31% | +1.48pp |
| Oracle retention | 65.31% | 65.54% | +0.23pp |
| per_class_vs_global_pct | — | +0.29% | — |

## Classification

**Category:** Research Discovery + Infrastructure Improvement  
**Frontier improvement:** No (+0.29% is real but below noise threshold for frontier claim)  
**Infrastructure value:** Yes — `PerClassConformalCalibrator` class + new simulator + 26 tests

## Next Recommended Directions

The confirmed diagnosis (within-class variance is the binding constraint) points to:

1. **Change calibrator formula**: Use absolute error percentile instead of relative error.
   - Absolute error for GPT-4 (e.g., p90_abs = 117 tokens) is more stable than relative
   - Maps to α via p90_abs / target_p90_abs_tokens (e.g., 50 tokens)
   - Expected: GPT-4 calibrator converges for predictions of the right scale

2. **Classification-based prediction**: Predict "short" (≤ median) vs "long" (> median)
   - Binary classification has lower prediction error than continuous regression
   - Binary SLA-aware baseline already shows +125-211% vs FIFO [run -n]
   - Per-class classification (GPT-4 vs ChatGPT × short vs long) could provide 4-class dispatch

3. **Absolute-error conformal calibration**: New formula:
   `α = alpha_max × min(2.0, p90_abs_err / target_p90_abs_err_tokens)`
   This decouples prediction scale from calibration sensitivity — works better for
   heavy-tailed output lengths where relative errors are inherently high.
