# ML-HGB Prior BurstGPT HF Backtest — Run 2026-06-22-v

**Date:** 2026-06-22  
**Trace:** BurstGPT HF normalized sample (lzzmm__BurstGPT, CC-BY-4.0)  
**Requests:** 5,880 (job_limit=5880)  
**Servers:** 4 | ρ=0.85 | SLA=30s  
**Prior:** HistGradientBoostingRegressor (quantile p50, loss="quantile")  
**Prior type:** Causal two-phase — Phase 1: running median (warmup_n=1000); Phase 2: HGB trained on Phase 1 data  
**Features:** [model_id_encoded, input_tokens]  
**HGB params:** max_iter=200, max_leaf_nodes=31, min_samples_leaf=max(5, warmup_n//20)=50, lr=0.1  

---

## Results

| Condition | Goodput/$ | vs FIFO | Oracle Retention |
|-----------|-----------|---------|-----------------|
| FIFO | 6,528.76 | baseline | — |
| Oracle (perfect predictions) | 48,598.82 | +644.38% | 100.0% |
| Global prior (live running median) | 34,003.60 | +420.83% | 69.97% |
| **ML-HGB prior** | **33,962.29** | **+420.2%** | **69.88%** |

**ml_vs_global_improvement_pct: −0.12%**

---

## Prior Accuracy Comparison

| Metric | Global prior | ML-HGB prior |
|--------|-------------|-------------|
| CV (%) | 15.34 | 43.03 |
| MAE (tokens) | 166.93 | 162.82 |
| Relative MAE (%) | 64.48 | 62.89 |
| n_model_ids | — | 2 (ChatGPT, GPT-4) |
| warmup_n | — | 1000 |
| conformal_mean_alpha | 0.002 | 0.002 |

The ML prior IS more accurate (MAE −2.5%, CV +28pp showing it distinguishes model types).
Both hit conformal_mean_alpha=0.002 (maximum cap).

---

## Root Cause Analysis

**The conformal calibrator formula is the binding constraint, not prediction accuracy.**

Formula: `α = alpha_max × min(2.0, p90_err / target_p90_error)`  
- alpha_max = 0.001 → 2× cap = 0.002  
- target_p90_error = 0.40  
- Cap triggers when p90_relative_error ≥ 0.80  

With ML-HGB prior, GPT-4 predictions improve (CV 43% vs 15%), but ChatGPT intra-class variance
remains large (p5=1 tok, p95=800+ tok). The p90 of the combined relative-error distribution
remains ≥ 0.80 because long-tail ChatGPT requests still have high per-request error.

**Three prior classes tested, all negative:**
1. Global running median [run -t]: α=0.002 cap, 70.0% retention  
2. Stratified running median [run -u]: α=0.002 cap, 70.0% retention (−0.12% vs global)  
3. ML-HGB quantile p50 [run -v]: α=0.002 cap, 69.88% retention (−0.12% vs global)  

**Conclusion:** To break the 70% retention ceiling on BurstGPT HF, the calibrator formula
itself must change — not the predictor. Specific opportunities:
1. Per-class calibration (separate α for ChatGPT vs GPT-4)
2. Calibrator metric: absolute error instead of p90 relative error
3. Quantile-hedged calibration targeting p75 instead of p90

---

## Simulation Detail

```
conformal_ml:
  sim_horizon_s:     3948.2
  mean_response_s:   225.6
  p50_response_s:    5.33
  p90_response_s:    837.1
  p99_response_s:    1992.3
  short_p90_response_s: 574.9   (vs 840.4 global — ML improved short ordering)
  long_p90_response_s:  1010.4  (vs 837.0 global — ML worsened long ordering slightly)
  preemption_count:  3328       (vs 3329 global — essentially identical)
  conformal_mean_alpha: 0.002
  sla_safe_goodput_per_dollar: 33962.29

conformal_global (reference):
  sim_horizon_s:     3949.4
  mean_response_s:   225.4
  short_p90_response_s: 840.4
  long_p90_response_s:  837.0
  preemption_count:  3329
  conformal_mean_alpha: 0.002
  sla_safe_goodput_per_dollar: 34003.60
```

Note: short_p90 improved 574.9 vs 840.4 (−31.5%) but both are ~20× above SLA=30s threshold.
Since no additional requests fall within SLA, the goodput/$ is statistically identical.

---

## Test Coverage

24 unit tests in `tests/test_ml_prior_backtest.py` — all pass.  
39 existing stratified-prior tests — all pass (no regression).

---

## Infrastructure Added

- `make_ml_prior_predictions_burstgpt()` in `srtf_serving_backtest.py`
- `MLPriorReport` dataclass
- `run_burstgpt_hf_ml_prior_backtest()` public API
- `tests/test_ml_prior_backtest.py` (24 tests)

**Adaptive `min_samples_leaf=max(5, warmup_n//20)`:** Prevents over-regularization when
warmup_n is small (e.g., test traces with warmup_n=30 → min_samples_leaf=5, not 20).

---

## Next Steps

1. Per-class conformal calibration (highest EV, feasible)
2. End-to-end compound backtest (economic × queue)
3. Calibrator metric change: absolute error instead of p90 relative error
