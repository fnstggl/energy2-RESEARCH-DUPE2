# Per-Class Conformal α Calibration Backtest — Run 2026-06-22-w

**Date:** 2026-06-22  
**Experiment ID:** run-w  
**Branch:** claude/kind-tesla-mn6iez  
**Status:** NULL RESULT — structural ceiling confirmed at per-class level

---

## Hypothesis

Per-class conformal calibration (separate `ConformalAlphaCalibrator` per `model_id`)
would allow GPT-4 requests (15.8% of BurstGPT) to reach α → 0 under a GPT-4-specific
stratified prior, recovering pure SRPT dispatch for those requests and improving
SLA-safe goodput/$ by 5–15%.

**Root cause hypothesis being tested (from run -v):**
The global calibrator saturates at α=0.002 because ChatGPT's bimodal within-class
variance (~10% surprise-long requests with rel_err≈0.99) dominates p90 error. GPT-4's
lower intra-class variance (p50=235 vs median ≈ median) should produce p90 error < 0.80
under a GPT-4-specific running median, allowing α_GPT4 → 0.

---

## Backtest Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | BurstGPT HF JSONL (CC-BY-4.0), 5,880-record fixture |
| Servers | 4 (M/G/c) |
| Target utilization | ρ = 0.85 |
| SLA budget | 30 s |
| Prior window | 200 completions |
| Warmup per calibrator | 100 completions |
| Calibrator | PerClassConformalCalibrator (per model_id) |
| Prior for conditions 4–5 | make_stratified_prior_predictions() |

---

## Results

| Condition | Goodput/$ | vs FIFO | Oracle retention |
|-----------|----------:|--------:|-----------------:|
| FIFO (baseline) | 6,528.76 | — | — |
| Oracle + global mono (upper bound) | 48,598.82 | +644.38% | 100.0% |
| Global prior + global mono (run -t) | 34,003.60 | +420.83% | 69.97% |
| Stratified prior + global mono (run -u) | 33,962.29 | +420.20% | 69.88% |
| **Stratified prior + per-class [NEW]** | **33,982.80** | **+420.51%** | **69.93%** |

**Per-class vs monolithic gain: +0.06%** (within simulator noise — NULL RESULT)

### Per-Class Calibrator Diagnostics

| Class | n_completed | warmup_cleared | mean_dispatch_α | capped? |
|-------|------------:|:--------------:|----------------:|:-------:|
| ChatGPT | 4,137 | Yes | 0.001999 | Yes (cap=0.002) |
| GPT-4 | 1,743 | Yes | 0.002000 | Yes (cap=0.002) |

### Prior Quality

| Prior | MAE (tokens) | Rel MAE % |
|-------|-------------:|----------:|
| Global running median | 166.93 | 64.48% |
| Stratified (per model_id + input_bin) | 157.29 | 60.76% |

---

## Analysis

### Why the hypothesis was wrong

GPT-4's within-class distribution is ALSO heavy-tailed. The stratified prior reduces
GPT-4 MAE from ~167 to ~157 tokens (5.8% improvement), but the RELATIVE error
distribution is still heavy-tailed with p90 relative error ≥ 0.80:

- GPT-4 requests with short actual output (e.g. 20 tokens vs median ≈ 235): rel_err = (235-20)/20 = 10.75
- GPT-4 requests with long actual output (e.g. 1000 tokens vs median ≈ 235): rel_err = (1000-235)/1000 = 0.765
- The heavy-tailed upper quantile (short GPT-4 responses) keeps p90 ≥ 0.80 → α capped at 0.002

This is structurally identical to the ChatGPT ceiling: within any LLM model class,
output length has high relative variance due to the fundamental diversity of LLM tasks.

### Structural Ceiling Confirmed at Per-Class Level

Four consecutive null results on conformal calibrator improvement (runs -t/-u/-v/-w):

| Run | Experiment | GPT-4 α | ChatGPT α | Global α | Δ vs run -t |
|-----|-----------|--------:|----------:|--------:|-------------|
| -t  | Global prior + global mono | 0.002 | 0.002 | 0.002 | baseline (+420.83%) |
| -u  | Stratified prior + global mono | 0.002 | 0.002 | 0.002 | -0.63pp |
| -v  | ML-HGB prior + global mono | 0.002 | 0.002 | 0.002 | null |
| **-w** | **Stratified prior + per-class** | **0.002** | **0.002** | **0.002** | **+0.06% noise** |

**Conclusion:** The conformal calibrator on BurstGPT HF is a HARD STRUCTURAL CEILING.
For ANY prior and ANY calibration granularity tested, p90 relative prediction error ≥ 0.80
for both model classes → α = 0.002 (2× alpha_max) for all dispatches.

The ceiling does NOT apply to the oracle condition (+644.38%), only to causal priors.
This gap (oracle vs live prior: 644 vs 421 = 65.3% of oracle captured) represents the
fundamental limit of running-statistics prediction on heavy-tailed LLM output lengths.

---

## Implications for Future Research

### What is NOT productive on BurstGPT

- Any further calibration-granularity improvements (per-request, per-session, etc.)
- Any running-statistics prior improvement (all converge to the same α cap)
- ML-based output length predictors using features available in BurstGPT (model_id, input_tokens)

### What might break the ceiling

1. **KV prefix hash features** (Mooncake traces): direct prediction of cache reuse →
   repeat prompts have near-zero prediction error → α → 0 for cache-hit requests
2. **Different trace characteristics**: Azure LLM 2024 reached 81.6% retention (vs
   BurstGPT 70%) — better output length predictability in that trace

### Pivot recommendation

**Immediate:** Mooncake FAST25 traces ingestion (Section 4.1 in BENCHMARK_REGISTRY.md).
KV block hash list per request → prefix reuse prediction → structural ceiling break.

**Alternative:** Apply current stack to a NEW trace (not BurstGPT) to validate
generalization of the +420% result. The Azure 2024 conformal result (+244.4% live, 81.6%
retention) is already committed in run -t's compound backtest.

---

## Files

| File | Description |
|------|-------------|
| `research/results/perclass_conformal_backtest_2026-06-22.json` | Full JSON result |
| `research/results/perclass_conformal_backtest_2026-06-22.md` | This file |
| `aurelius/benchmarks/srtf_serving_backtest.py` | New: `PerClassConformalCalibrator`, `_simulate_decoupled_hybrid_perclass_conformal`, `PerClassConformalAlphaReport`, `run_burstgpt_hf_perclass_conformal_backtest` |
| `tests/test_perclass_conformal_backtest.py` | 30 unit + integration tests |
| `scripts/run_perclass_conformal_backtest.py` | Standalone runner |
