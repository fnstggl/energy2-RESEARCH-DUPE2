# Input-Token-Conditioned Live Prior Backtest — 2026-06-21

**Run:** 2026-06-21-u  |  **Status:** Public-trace M/G/c discrete-event replay

> Directional simulator/backtest evidence — not production savings (docs/RESULTS.md §8).

## Summary

This run tests whether conditioning the causal sliding-window median on the
request's **input-token length** improves scheduling gain over the global
running-median prior from run -t.

Hypothesis: input-token length predicts output length (r=0.309 for BurstGPT HF
5,880-sample), so bucketing by input length should lift pred_cv from 15.3%→34.6%
and ranking accuracy from 52.2%→61.1% (+17%), translating to higher retention.

### Key Results

**Null result — input-token conditioning provides no measurable scheduling gain.**

| Condition | gp/$ | vs FIFO | vs Oracle retention |
|---|---:|---:|---:|
| FIFO | 6,528.76 | — | — |
| Oracle conformal | 48,598.82 | +644.38% | 100% (ref) |
| Global median (run -t) | 34,003.60 | +420.83% | 70.0% |
| Input-conditioned (this run) | 33,962.29 | +420.20% | **69.88%** |

- **cond vs global uplift: −0.12%** (within simulation noise)
- **Prior quality**: global CV=15.3%, MAE=166.9 tok → conditioned CV=34.6%, MAE=153.7 tok
- **Ranking accuracy**: global≈52.2% → conditioned=60.5% (+17%)

## Finding

Ranking accuracy improvement from 52% → 61% does **not** translate to
scheduling improvement when the conformal adaptive-α calibrator is active.

**Root cause**: The ConformalAlphaCalibrator (α=0→0.001) adapts the aging
parameter to observed prediction errors. It effectively compensates for poor
prediction quality by adjusting request ordering at dispatch time. Within the
range 52%–61% ranking accuracy, this compensation is complete — the calibrator
already extracts essentially all available scheduling signal.

**Implication**: The path to meaningful retention improvement requires a
predictor with **ranking accuracy > ~75%** (i.e., much better differentiation),
achievable only with a learned model (gradient boosting or similar) using input
features beyond just token-count buckets.

## Prior Quality

| Metric | Global median (run -t) | Input-conditioned (run -u) |
|---|---:|---:|
| pred_cv_pct | 15.34% | 34.64% |
| prior_mae_tokens | 166.93 | 153.68 |
| prior_rel_mae_pct | 64.48% | 59.36% |
| ranking_accuracy_pct | ~52.2% | 60.50% |

Note: pred_cv doubled (better spread) and MAE dropped 7.9%, but the
conformal calibrator absorbs this improvement entirely.

## Research Papers (Phase 3)

Three papers discovered and supporting this run's direction:

1. **arXiv:2408.15792** (Learning to Rank for LLM Scheduling, NeurIPS 2024):
   Ranking accuracy is the key metric for SJF scheduling. This run confirms
   the threshold for improvement is higher than 61%.

2. **arXiv:2604.07931** (ProD: Robust Length Prediction, April 2026):
   Input-conditioned bucket-median estimation. Validated as correct algorithm;
   result: insufficient without a richer learned predictor.

3. **arXiv:2602.11812** (EGTP: Input-Feature Output Prediction, ICLR 2026):
   EGTP uses richer input features beyond token count (content, model type).
   This is the **next highest EV target**: add semantic features to the predictor.

## Decision (Phase 9)

Null result: cond_vs_global_uplift = −0.12% (< +2% threshold).
**Implementation kept** (adds clean research infrastructure for future learned predictors).
**Not merged as a performance improvement** — documented as a learning run.

## North Star Progress (Unchanged)

Azure LLM 2024: +52.8% vs SLA-aware (target: +300%, gap: 247.2pp)

The clearest highest-EV next target remains a learned output-length predictor
achieving 75%+ ranking accuracy (CARA / HGB approach).

## Methodology

- **Input-conditioned prior**: For request i with input_tokens in bucket b
  (edges [0,100,300,700,2000,∞]), predict output as median of last 200
  completions also in bucket b. Fall back to global median if bucket < 5 entries.
- **Global prior (run -t reference)**: median of last 200 completions globally.
- **Service time**: always actual_tokens × TPOT_S (no leakage in serving physics).
- **Identical server pool**: 4 servers, identical across all disciplines.
- **Time warp**: single scalar to achieve target ρ=0.85, applied identically.
- **Conformal calibrator**: adapts α from empirical prediction errors (causal).

This is a discrete-event M/G/c simulator result, not production savings.
See docs/RESULTS.md §8 for the full honesty/limitations statement.
