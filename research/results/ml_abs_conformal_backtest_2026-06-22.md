# Run 2026-06-22-z: ML Prior under Absolute-Error Conformal — BurstGPT HF

## Question

Run **-v** found the ML-HGB prior (`model_id` + `input_tokens`, HistGradientBoosting
p50) to be a **null result** on BurstGPT: −0.12% vs the global running-median prior,
under the **relative-error** conformal calibrator. Run -v explicitly attributed this to
the calibrator, not the prior: the relative-error formula was **capped at mean_α=0.002**
for *both* priors because ChatGPT short-request relative error (actual=7, pred≈18,
rel_err=1.57) kept p90 relative error ≥ 0.80.

Run **-x** then removed that cap with the **absolute-error** conformal calibrator,
lifting the global running-median prior from +420.83% (70.0% retention) to **+557.12%
(88.3% retention)** on BurstGPT.

This run closes the obvious open cell: **does the ML prior beat the running median once
the absolute-error calibrator can actually exploit its better accuracy?**

## Design: 2×2 (prior × calibrator) + FIFO + oracle

| | rel-conformal | abs-conformal |
|---|---|---|
| **global (running median)** | run -t baseline | run -x result |
| **ML-HGB** | run -v null result | **NEW (this run)** |

## Setup

| Parameter | Value |
|-----------|-------|
| Trace | BurstGPT HF (5,880 requests) |
| Servers | 4 |
| Target ρ | 0.85 |
| SLA | 30 s |
| ML warmup_n | 1,000 (Phase 1 running median → Phase 2 HGB p50) |
| abs target p90 | 500 tokens |

## Results

| Discipline | SLA-safe goodput/$ | Δ vs FIFO | Retention |
|-----------|---:|---:|---:|
| FIFO | 6,528.8 | (baseline) | — |
| Oracle (abs) | 48,598.8 | +644.38% | 100% |
| global + rel (run -t) | 34,003.6 | +420.83% | 69.97% |
| global + abs (run -x) | 42,901.6 | +557.12% | **88.28%** |
| ML + rel (run -v) | 33,962.3 | +420.20% | 69.88% |
| **ML + abs (this run)** | **42,810.7** | **+555.72%** | **88.09%** |

### Key contrasts

- **PRIMARY — ML+abs vs global+abs: −0.21%** (flat / marginally negative — **NULL**)
- SECONDARY — ML+abs vs ML+rel: **+26.05%** (abs-conformal helps the ML prior just as
  much as it helps the global prior — the +26% abs gain is **prior-agnostic**)

### Prior quality and calibration

| | global | ML-HGB |
|---|---:|---:|
| CV | 15.3% | 43.0% |
| MAE (tokens) | 166.9 | 162.8 |
| abs mean α | 0.000562 | 0.000595 |
| rel mean α | 0.001990 | 0.001994 |

The ML prior is genuinely active (`n_model_ids=2`, CV and MAE differ from global). It
reduces MAE by 2.5% but **increases** prediction CV (15.3% → 43.0%) — it is a sharper
but more volatile predictor.

## Conclusion: Honest Null Result

**The ML-HGB prior does not beat the trivial running-median prior, even under the
absolute-error conformal calibrator that uncapped α.** Three reinforcing reasons:

1. **The abs-conformal gain is prior-agnostic.** Both priors jump ~+26% from rel→abs and
   land at ~88% retention. The improvement comes from the *calibrator metric*, not from
   any property of the prediction — exactly the structural finding run -v anticipated.

2. **The ML prior's marginal accuracy gain doesn't reach the SLA threshold.** MAE drops
   only 2.5% (166.9→162.8), while CV *rises* (15.3→43.0%). The remaining error is
   dominated by ChatGPT "surprise-long" requests that no `model_id`/`input_tokens`
   feature can predict — the same intra-class variance ceiling diagnosed in runs -u/-v/-w.

3. **α barely moves between priors** (0.000562 vs 0.000595): the abs calibrator sees
   nearly identical p90 absolute error for both, so dispatch behaviour is essentially
   identical.

### Cross-validation of the harness

This 2×2 reproduces the three already-merged results exactly: global+rel = 34,003.6
(= run -t's 34,004), global+abs = 42,901.6 (= run -x's 42,902), ml+rel = 33,962.3
(= run -v's 33,962). The new ml+abs cell is the only previously-unmeasured condition.

## Implication for the roadmap

This **rules out an entire class of future work**: investing in better causal output-
length *predictors* (ML, stratified, model-aware) will not move SLA-safe goodput/$ on
BurstGPT, because (a) the abs-conformal calibrator already extracts the available
scheduling signal prior-agnostically, and (b) the residual error is irreducible
intra-class variance. The remaining lever for the north-star (+300% vs SLA-aware) is the
**compound economic × queue scheduling** integration (run -y Q2/Q6), not prediction
accuracy.

> Shadow-only simulator result; not a production-savings claim.
