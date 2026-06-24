# Aging SRTF + AMCSG Compound Backtest — 2026-06-24

**Classification: HONEST NULL RESULT**  
**Five-Failure Rule Integration Experiment — Run 6/5 (counter still active)**  
`production_claim: false` | `five_failure_rule_integration: true`

---

## Summary

Non-preemptive aging-SRTF queue discipline combined with AMCSG optimal variable-c capacity schedule produces **no measurable improvement** over FIFO+AMCSG on either trace.

| Condition | Azure gp/$ | BurstGPT gp/$ |
|---|---|---|
| FIFO + AMCSG (strongest fair baseline) | 150,630 | 168,270 |
| Aging SRTF + AMCSG (candidate) | 150,630 | 168,421 |
| **Delta vs baseline** | **+0.00%** | **+0.09%** |
| OSOTSS frontier | 159,578 | 178,109 |
| vs OSOTSS | -5.61% | -5.44% |

**Verdict:** Aging SRTF + AMCSG is indistinguishable from FIFO+AMCSG within measurement noise. Does NOT meet the merge threshold. Five-Failure Rule remains active.

---

## Root Cause: Prediction Degeneracy

The aging-SRTF priority key is:

```
key = predicted_tokens / (1 + aging_alpha × max(0, t - arrived_t))
```

With `aging_alpha=0.05` and the running-median live prior (`LIVE_PRIOR_WINDOW=200`):

| Metric | Value |
|---|---|
| Predicted tokens stdev | 8.1 |
| Actual tokens stdev | 93.1 |
| Unique predicted values | 37 |
| Mode predicted value | ~91 tokens |

The running-median window of 200 converges to a near-constant prediction ≈91 tokens. With nearly identical `predicted_tokens` across all requests, the aging factor dominates and the key degenerates to near-FIFO ordering.

**Fixed-c degeneracy test (< 1% delta):** Confirmed at fixed-c, aging SRTF ≈ FIFO — identical to expectation.

**Implication:** No queue discipline improvement can outperform FIFO until prediction accuracy is substantially improved (e.g., per-request token prediction as in Trail/NP-SRPT).

---

## Experiment Design

### 2×2 Factorial

| | Fixed-c | AMCSG variable-c |
|---|---|---|
| FIFO | Condition A | Condition B (canonical baseline) |
| Aging SRTF | Condition C | Condition D (candidate) |

- **Same-conditions rule**: All four conditions share identical c_effective sequences (same stochastic interruption seed=42 per condition pair).
- **Non-preemptive**: `preemption_count=0` confirmed by test.
- **No oracle**: Predictions from running median prior only; no future-arrival information used.
- **GSF spot-fleet cost model**: 95% spot @ $0.80/hr, 10%/hr interrupt rate, ZFHC threshold=8, Binomial stochastic (seed=42).

### New Functions Added (research/benchmark only)

- `_simulate_aging_srtf_variable_c()` — non-preemptive aging SRTF in variable-c event loop
- `_apply_gsf_spot_interruptions()` — stochastic spot interruption pre-computation
- `AgingSRTFAMCSGReport` — 4-condition result dataclass
- `run_aging_srtf_amcsg_azure_backtest()` / `run_aging_srtf_amcsg_burstgpt_backtest()` — entry points

**No production modules modified.**

---

## Test Suite: `tests/test_aging_srtf_amcsg_compound.py`

24 tests total: **23 passed, 1 skipped** (no production main ref in CI).

Key tests:
- `test_aging_srtf_variable_c_non_preemptive` — preemption_count=0 ✓
- `test_azure_fifo_amcsg_reproduces_canonical` — 150,630 in [149k, 152k] ✓
- `test_burstgpt_fifo_amcsg_reproduces_canonical` — 168,270 in [166k, 170k] ✓
- `test_azure_aging_srtf_amcsg_not_worse_than_fifo_amcsg` — delta ≥ -0.5% ✓
- `test_azure_fixed_c_discipline_degeneracy` — |delta| < 1.0% ✓
- `test_below_osotss_frontier` — both traces below OSOTSS ✓
- `test_completion_rates_one` — all 4 conditions complete all 5,880 requests ✓

---

## Five-Failure Rule Status

| Run # | Date | Experiment | Result |
|---|---|---|---|
| 1 | 2026-06-23 | ABS Conformal + variable-c | Null |
| 2 | 2026-06-24 | Borderline OSOTSS | Null |
| 3 | 2026-06-24 | Stochastic safety margin OSOTSS | Null |
| 4 | 2026-06-24 | ABS floor + spot fleet | Null |
| 5 | 2026-06-24 | Adaptive EWMA OSOTSS | Null |
| **6** | **2026-06-24** | **Aging SRTF + AMCSG** | **Null** |

**Counter: 6/5 — Five-Failure Rule ACTIVE**

Allowed: integration, validation, diagnosis, architecture simplification.  
Forbidden: new modules, new optimizer paths.

---

## Papers Surveyed

- **Trail** (arXiv:2410.01035, ICLR 2025) — Per-request token prediction; addresses prediction degeneracy directly. Most relevant next step if pilot telemetry becomes available.
- **NP-SRPT** (arXiv:2411.06348) — Non-preemptive SRPT for LLM inference; same family as this experiment; also requires accurate prediction.
- **FastServe** (arXiv:2305.05920) — Skip-join MLFQ; prediction-dependent.
- **Queueing+LLMs** (arXiv:2503.07545) — Theoretical grounding for SJF/SRTF in LLM workloads.

---

## Third-Trace Cross-Validation: BLOCKED

| Trace | Reason blocked |
|---|---|
| Alibaba GenAI 2026 | Image generation workload (TXT_2_IMG, IMG_2_IMG); OSOTSS not applicable |
| ShareGPT | No timestamps; cannot do per-tick replay |
| LMSYS | No processed data available |

---

## Conclusion

Aging SRTF + AMCSG is a valid integration experiment that correctly confirms the non-preemptive variable-c compound architecture. The null result is scientifically honest and reproducible.

**The binding constraint is prediction accuracy, not queue discipline.** Until per-request token predictions significantly exceed running-median accuracy (stdev 8.1 vs actual 93.1), queue discipline optimizations cannot escape the prediction-degeneracy collapse.

The OSOTSS frontier (+5.94%/+5.85% vs AMCSG) was achieved via causal EWMA-based capacity scheduling, not queue discipline — and it remains the ceiling under the current live prior regime.
