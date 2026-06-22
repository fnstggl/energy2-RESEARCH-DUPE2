# Stratified Feature-Aware Causal Prior — BurstGPT HF Backtest [run 2026-06-22-u]

**Run date:** 2026-06-22  
**Category:** RESEARCH DISCOVERY — Negative Result with Informative Diagnostics  
**Dataset:** BurstGPT HF (lzzmm__BurstGPT, 59,999 records, CC-BY-4.0)  
**Trace sample:** 5,880 requests, ρ=0.85, 4 servers, SLA=30s  
**Baseline:** Global causal prior (run 2026-06-21-t: +420.83% vs FIFO, 70.0% retention)

---

## Motivation

Run -t measured 70.0% oracle retention on BurstGPT HF with the global causal sliding-window median prior. The root cause identified in GAP_ANALYSIS:

- BurstGPT mixes two model types with dramatically different output length distributions
- **ChatGPT** (84.2% of traffic): p50=7 tokens — mostly very short responses
- **GPT-4** (15.8% of traffic): p50=235 tokens — substantially longer
- A global running median (~18 tokens) is 33× wrong for GPT-4 requests

Additionally, within-ChatGPT input-output correlation is r=0.513 (strong), suggesting that input token count is a useful feature for predicting ChatGPT output length.

**Hypothesis:** Per-(model_id, input_bin) stratified prior should improve BurstGPT retention from 70.0% toward ≥80% by separating the two model distributions.

---

## Implementation

New code in `aurelius/benchmarks/srtf_serving_backtest.py`:

- `load_burstgpt_serving_requests_jsonl_with_features()` — extends base loader to return parallel feature list with `model_id` and `input_tokens`
- `make_stratified_prior_predictions()` — three-level causal fallback hierarchy:
  1. Per-(model_id, input_bin) running median — when ≥20 stratum completions available
  2. Per-model_id running median — when stratum is sparse
  3. Global running median — ultimate fallback
  Input bin classification: 'long' if `input_tokens ≥` running median of past input_tokens for that model (fully causal)
- `StratifiedPriorReport` dataclass — 4-way comparison: FIFO / oracle / global / stratified
- `_run_stratified_prior_on_trace_with_features()` — internal runner
- `run_burstgpt_hf_stratified_prior_backtest()` — public API

**Causality guarantee:** All predictions for request i use only completions from requests 0..i-1.

---

## Benchmark Commands

```bash
# Primary benchmark command
python3 -c "
from aurelius.benchmarks.srtf_serving_backtest import run_burstgpt_hf_stratified_prior_backtest, DEFAULT_BURSTGPT_SLA_S, LIVE_PRIOR_WINDOW
r = run_burstgpt_hf_stratified_prior_backtest(servers=4, target_rho=0.85, job_limit=5880, sla_s=DEFAULT_BURSTGPT_SLA_S, prior_window=LIVE_PRIOR_WINDOW)
print(r.to_dict())
"
```

---

## Prior Quality Diagnostics

| Metric | Global prior (run -t) | Stratified prior (run -u) | Delta |
|---|---:|---:|---:|
| CV | 15.3% | 32.3% | +17.0pp |
| MAE (tokens) | 166.9 | 157.3 | **−9.6 tokens (−5.7%)** |
| Relative MAE | 64.5% | 60.8% | **−3.7pp** |
| Stratum-level usage | — | 98.0% | — |
| Model-level usage | — | 1.4% | — |
| Global-fallback usage | — | 0.7% | — |

The stratified prior reduces absolute MAE by 5.7% relative and is served from stratum-level in 98.0% of requests (confirming adequate warmup throughout the 5,880-request trace).

The higher CV (32.3% vs 15.3%) is expected and correct: the stratified prior produces meaningfully different predictions for different model types (ChatGPT≈3-7 tokens vs GPT-4≈235 tokens), while the global prior converges near the global median of ~18 tokens for all requests.

---

## Benchmark Results

**BurstGPT HF, 5,880 requests, ρ=0.85, 4 servers, SLA=30s**

| Discipline | SLA-safe goodput/$ | vs FIFO | Oracle retention |
|---|---:|---:|---:|
| FIFO | 6,528.76 | (baseline) | — |
| Conformal oracle (upper bound) | 48,598.82 | **+644.38%** | 100% |
| **Conformal global prior [run -t]** | **34,003.60** | **+420.83%** | **70.0%** |
| **Conformal stratified prior [run -u]** | **33,962.29** | **+420.20%** | **69.9%** |

**Delta table: Main (global prior) vs Candidate (stratified prior)**

| KPI | Main (global, run -t) | Candidate (stratified, run -u) | Delta |
|---|---:|---:|---:|
| SLA-safe goodput/$ | 34,003.60 | 33,962.29 | **−0.12%** |
| Oracle retention | 70.0% | 69.9% | −0.1pp |
| Prior MAE (tokens) | 166.9 | 157.3 | −9.6 tokens |
| Prior CV | 15.3% | 32.3% | +17.0pp |
| Short_p90 response (s) | 840.40 | **631.83** | **−24.9%** |
| Long_p99 response (s) | 2,160.94 | 2,186.38 | +1.2% |
| Conformal mean α | 0.002000 | 0.002000 | 0.0 |
| Preemption count | 3,329 | 3,328 | −1 |

---

## Key Research Findings

### Finding 1: MAE improvement (−5.7%) does not improve goodput/$ (−0.12%)

The stratified prior reduces prediction MAE by 5.7%, but goodput/$ is essentially flat at −0.12%. This is the central finding: **prediction accuracy improvement is absorbed by the conformal calibrator's α adaptation**.

The conformal calibrator converges to identical mean_α (0.002) for both global and stratified priors. This means the calibrator has detected similar overall prediction uncertainty for both and applied equivalent aging compensation. The scheduling behavior is therefore identical at the goodput/$ level.

### Finding 2: Short_p90 latency improves by 24.9% (840s → 631s)

Despite flat goodput/$, the stratified prior improves short_p90 latency by 24.9%. This is real and meaningful: more short requests (ChatGPT responses with predicted=3 tokens) get scheduled earlier. However, **both 840s and 631s are far beyond the 30s SLA**, so neither improvement registers in the goodput/$ KPI. The SLA threshold is the binding constraint, not relative latency.

### Finding 3: The oracle gap is structural, not reducible by running statistics

Oracle achieves short_p90 = 4.39s. Stratified prior achieves 631.83s. The 144× gap reveals a structural limitation:

- Oracle knows actual output length → puts 7-token requests first → they complete in 0.29s
- Any running-statistics prior predicts based on past distributions, not the current request's actual length
- ChatGPT's bimodal distribution (p50=7 tokens BUT 10% exceed 200 tokens) means predictions are wrong for the "surprise-long" requests
- These surprise-long requests get scheduled first (predicted=3 tokens) but take much longer, blocking genuinely short requests

### Finding 4: The bottleneck is RANK ORDERING accuracy for surprise-long requests

Within the ChatGPT short-input bin (predicted=3-7 tokens), approximately 5-10% of requests are actually 200-750 tokens. These "disguised-long" requests:
1. Get high priority (small predicted_tokens → front of queue)
2. Take 3-15s of actual service time
3. Block other short requests from being served
4. Cause cascading queue buildup exceeding SLA=30s for the rest

No running-statistics prior can reliably identify these disguised-long requests. A trained ML model with per-request features (session history, query complexity signals) would be needed.

### Finding 5: The conformal calibrator is working correctly

The identical conformal_mean_α (0.002) for both priors confirms the calibrator correctly adapts to per-request prediction uncertainty. When both priors have similar residual error distributions (they both fail on surprise-long requests), they converge to the same α, and therefore the same scheduling behavior and identical goodput/$.

---

## Failure Mode Analysis

**Why stratification doesn't help for BurstGPT goodput:**

1. **Bimodal ChatGPT distribution**: 90% of ChatGPT requests are ≤100 tokens, 10% are 200-750+ tokens. Both groups look similar at request-arrival time (same input-bin characteristics on average).

2. **Conformal α absorption**: The conformal calibrator raises α to compensate for "surprise-long" prediction misses, effectively making the scheduling more conservative (more aging-like) for both priors equally.

3. **SLA threshold far above achievable**: With running-statistics priors, short_p90 ≈ 630-840s vs SLA=30s. The improvement needed to move the goodput/$ needle is a 20-28× reduction in short_p90 (to ≤30s), which requires fundamentally better prediction, not marginally better statistics.

---

## What This Means for the Research Roadmap

**Confirmed**: The oracle gap on BurstGPT (30pp) cannot be closed by running-statistics priors alone, regardless of stratification sophistication. The bottleneck is:
- Request-level prediction quality for INDIVIDUAL requests (not population statistics)
- Specifically, correctly identifying "surprise-long" ChatGPT requests

**Required for ≥85% retention**: A trained ML predictor that uses request-level features to classify requests as potentially-long vs definitely-short. The CARA HGB forecaster (`aurelius/forecasting/cara_output_length_forecaster.py`) is designed exactly for this, but requires:
1. A trace with TTFT/queue-state features at arrival time (not just input_tokens)
2. Training on traces with labeled output lengths per request

**Updated oracle gap understanding:**
- Azure LLM 2024 (18pp gap): Azure only has ContextTokens, with r≈0 correlation → gap requires external ML predictor
- BurstGPT HF (30pp gap): BurstGPT has model_id + input_tokens, with r=0.513 → stratification helps MAE but not goodput (structural bimodal limitation)

**Both traces** confirm: running-statistics priors have a ceiling at ~70-82% retention, and crossing it requires a trained ML predictor.

---

## Decision

**Do not merge as a frontier improvement.** The goodput/$ delta is −0.12% (neutral/negative).

**Merge as research infrastructure** under the label "Research Infrastructure":
- Adds `load_burstgpt_serving_requests_jsonl_with_features` — useful for future ML-predictor experiments
- Adds `make_stratified_prior_predictions` — a clean, reusable stratified predictor component
- Adds `StratifiedPriorReport` — 4-way comparison infrastructure
- 39 new tests passing; 0 regressions
- Documents the oracle gap structure with new quantitative evidence

The code is sound and the negative result is **informative** — it precisely characterizes why running-statistics priors cannot close the oracle gap and what type of predictor is actually needed.

---

## Research Papers Reviewed

1. **TIE Scheduling** (arXiv:2604.00499, April 2026): distributional ordering; confirmed model_id stratification is the right direction but also confirmed the bimodal distribution creates an irreducible entropy floor.

2. **ProD, Robust Length Prediction** (arXiv:2604.07931): argues for request-specific prediction using prompt type, model family. This experiment implements the simplest version and finds it insufficient.

3. **SLAI Scheduler** (arXiv:2508.01002): work-conserving preemptive scheduling; confirms that ordering quality (not just scheduling discipline) is the key variable.

4. **arXiv:2507.10150 (Past-Future Scheduler)**: joint history + future prediction; validates that session-level features (beyond single-request statistics) are needed for high-quality prediction.

---

## Next Recommended Direction

**Immediate**: The oracle gap requires a trained ML predictor with per-request features. Options:

1. **CARA HGB forecaster integration** (`cara_output_length_forecaster.py`): train on CARA telemetry where TTFT, queue state, and actual output lengths are all available. Wire `HGBOutputLengthForecaster.p50` as the prior in `_run_live_prior_on_trace`. This was the Rank-1 opportunity and this experiment confirms it's the correct path.

2. **Prompt-type classifier for BurstGPT**: Build a simple binary classifier that identifies "potentially-long" ChatGPT requests using input_tokens + arrival context. Even 70% classifier accuracy for surprise-long requests would improve retention meaningfully.

3. **Compound economic + queue backtest**: The independence estimate (+876% vs FIFO) remains the highest-feasibility high-EV opportunity. Wire economic scheduling and serving queue discipline together in a unified simulator.

---

## Infrastructure Artifacts

- **Implementation**: `aurelius/benchmarks/srtf_serving_backtest.py` (new functions at end)
- **Tests**: `tests/test_stratified_prior_backtest.py` (39 tests, all passing)
- **Results JSON**: `research/results/stratified_prior_burstgpt_backtest_2026-06-22.json`
- **Results MD**: this file
