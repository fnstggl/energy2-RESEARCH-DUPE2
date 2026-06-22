# Run 2026-06-22-u: Model-Stratified Prior — BurstGPT HF Backtest

## Hypothesis

Per-model sliding-window median should beat global median on BurstGPT HF because
ChatGPT (median=7 tokens) and GPT-4 (median=212 tokens) have a 30× output-length
difference. The global running median ≈ 14 collapses both distributions, predicting
GPT-4 requests as short and giving them SRPT priority over shorter ChatGPT requests —
exactly the wrong ordering.

**Research basis:**
- EWSJF (arXiv:2601.21758, KDD 2026): per-group SJF achieves 30%+ throughput, 4× TTFT
- Entropy-Guided Output Length Prediction (arXiv:2602.11812, ICLR 2026): model-specific p95/p50 ratios 1.7–20.5
- BurstGPT (arXiv:2401.17644, KDD 2025): dataset with ChatGPT/GPT-4 heterogeneity

## Setup

| Parameter | Value |
|-----------|-------|
| Trace | BurstGPT HF full-scale (58,042 requests) |
| Servers | 4 |
| Target ρ | 0.85 |
| SLA | 30 s |
| Prior window | 200 |
| Model warmup | 50 completions |

## Results

| Condition | Goodput/$ | Δ vs FIFO | Oracle retention |
|-----------|-----------|-----------|-----------------|
| FIFO | 11,354.98 | — | — |
| Global live prior (run -t) | 34,376.84 | **+202.75%** | 72.76% |
| Model-stratified prior (this run) | 34,418.15 | **+203.11%** | 72.85% |
| Oracle | 47,244.91 | **+316.07%** | 100% |

**Stratified vs global gain: +0.12% (NULL RESULT)**

## Prior Quality

| Metric | Global | Stratified |
|--------|--------|-----------|
| CV (%) | 94.11 | 94.33 |
| MAE (tokens) | 78.56 | 78.51 |

## Per-Model Breakdown

| Model | n | median_actual | mean_actual | MAE | stratified_uses |
|-------|---|--------------|------------|-----|----------------|
| ChatGPT | 49,302 | 7.0 | 103.9 | 66.0 | 49,252 |
| GPT-4 | 8,740 | 235.0 | 253.8 | 149.2 | 8,690 |

## Conclusion: Honest Null Result

Model stratification provides only **+0.12%** gain over global prior. The hypothesis
was falsified because:

1. **Within-model variability dominates**: ChatGPT has a heavy tail (median=7, mean=103.9,
   MAE=66.0); GPT-4 is broadly variable (median=235, mean=253.8, MAE=149.2). Even knowing
   the model type, token predictions are nearly as wrong as global predictions.

2. **CV unchanged**: Both global and stratified have CV≈94%. Per-model median is a better
   expected-value estimator, but variance is unchanged, so SRPT ordering is not materially
   improved.

3. **Conformal α already compensates**: The ConformalAlphaCalibrator's adaptive α absorbs
   systematic prediction bias. The model-level correction (which is systematic) was already
   being partially corrected by calibration.

## Next Best Opportunity

Request-specific output-token predictor using **context length + request type + model ID**
features. A gradient-boosted model (CARA HGB) could reduce within-model CV from 94% to
potentially 40–60%. Historical result on Azure: low-context-length bin had CV=7%.

If BurstGPT within-model CV can be halved (94% → 47%), expected oracle retention gain:
~5–10pp (72.85% → ~78–83%).
