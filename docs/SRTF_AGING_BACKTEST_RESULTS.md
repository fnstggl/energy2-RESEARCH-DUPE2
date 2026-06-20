# SRTF-with-Aging Anti-Starvation Backtest Results — Run 2026-06-20-i

> **Directional simulator result — NOT production savings.**
> Baseline is FIFO, not SLA-aware. See `docs/RESULTS.md` §8.

---

## Summary

Run 2026-06-20-g proved that SRTF (shortest-predicted-job-first) gives
**+323% SLA-safe goodput/$ vs FIFO** on the real Azure LLM 2024 serving queue —
but at the cost of a severe long-request starvation regression: p99 grows from
733s → 2373s (+223.5%). This prevents production deployment.

Run 2026-06-20-i adds **SRTF-with-Aging**, an anti-starvation discipline:

```
key(r, t) = predicted_tokens / (1 + α × wait_s)
```

As a request waits, its effective priority key falls — long-waiting jobs gain
priority over newly-arriving short ones, bounding starvation. Research basis:
- **Astraea** (arXiv:2512.14142): aging-based promotion for LLM serving fairness.
- **FlowPrefill** (arXiv:2602.16603): operator-level preemption for HoL mitigation.
- **Equinox** (arXiv:2508.16646): holistic fair scheduling with dual-counter starvation
  prevention.

---

## Primary Benchmark: Azure LLM 2024 (ρ=0.85, SLA=10s, c=4)

**Dataset:** Real Azure LLM 2024 request-level trace — 5,880 requests, real
output-token distribution (p50≈90, p99≈479, max≈1346). Same setup as run -g.

**Command:**
```python
from aurelius.benchmarks.srtf_serving_backtest import run_aging_srtf_backtest
report = run_aging_srtf_backtest(servers=4, target_rho=0.85, sla_s=10.0)
```

### Result Table

| KPI | FIFO (baseline) | SRTF-perfect | Aging-SRTF (α=0.05) |
|---|---:|---:|---:|
| SLA-safe goodput/$ | 13,336 | 56,481 | 16,317 |
| gp/$ delta vs FIFO | — | **+323.5%** | **+22.4%** |
| short_p90 response (s) | 696.16 | 3.03 | 204.02 |
| short_p90 improvement vs FIFO | — | **+99.6%** | **+70.7%** |
| long_p99 response (s) | 733.55 | 2,373.09 | 1,479.49 |
| long_p99 delta vs FIFO | — | **+223.5%** ← starvation | **+101.7%** |
| mean response (s) | 343.89 | 129.89 | 200.74 |

**Key finding:** Aging-SRTF (α=0.05) cuts long-request starvation by **55%**
(long_p99: +223.5% → +101.7% regression vs FIFO) while retaining 22.4%
goodput/$ gain vs FIFO and 70.7% short_p90 improvement.

---

## Alpha Sensitivity: The Fairness–Efficiency Trade-Off Curve

**Azure LLM 2024, ρ=0.85, SLA=10s, c=4.**

| alpha | short_p90 impr% | long_p99 delta% | gp/$ delta% | Notes |
|---|---:|---:|---:|---|
| 0.00 | +99.6% | +223.5% | +323.5% | Pure SRTF (starvation risk) |
| 0.01 | +78.1% | +113.8% | +70.7% | **Sweet spot** — 49% starvation reduction |
| 0.05 | +70.7% | +101.7% | +22.4% | Default — 55% starvation reduction |
| 0.10 | +69.7% | +100.9% | +15.6% | Diminishing returns |
| 0.50 | +69.2% | +100.7% | +10.0% | Near-FIFO fairness |
| 999 | +69.0% | +100.4% | +8.8% | ≈ FIFO |

**Recommended α=0.01 (production sweet spot):**
- Retains **+70.7% goodput/$** vs FIFO (vs +323.5% pure SRTF)
- Cuts long_p99 starvation by **49%** (from +223.5% to +113.8% vs FIFO)
- short_p90 still improves +78.1% (vs +99.6% pure SRTF)
- Parity time for p99 request (479 tok) vs median (90 tok): ≈ 430 seconds
  → long request gets priority before starvation becomes production-visible

---

## High-Contention Scenario (ρ=0.92, SLA=10s)

| Discipline | short_p90 impr% | long_p99 delta% | gp/$ delta% |
|---|---:|---:|---:|
| SRTF vs FIFO | +99.6% | +183.0% | +314.0% |
| Aging-SRTF (α=0.05) vs FIFO | +71.8% | +96.4% | +17.3% |

At ρ=0.92, aging-SRTF still improves goodput/$ by +17.3% vs FIFO while
holding long_p99 regression below +100%.

---

## Tight SLA Scenario (SLA=3.0s)

| Discipline | short_p90 impr% | long_p99 delta% | gp/$ delta% |
|---|---:|---:|---:|
| SRTF vs FIFO | +99.6% | +223.5% | +362.7% |
| Aging-SRTF (α=0.05) vs FIFO | +70.7% | +101.7% | +37.8% |

Tighter SLA increases goodput/$ differentiation: aging-SRTF achieves +37.8%
vs FIFO under a 3s response budget.

---

## BurstGPT Cross-Trace Validation

**Dataset:** BurstGPT sample fixture — 51 requests, avg response ≈ 340 tokens.
Sample is too small (c=4 leaves minimal queue contention) for robust starvation
characterization. Full 1.4M-row BurstGPT dataset would confirm generalization.

| Discipline | short_p90 impr% | long_p99 delta% | gp/$ delta% |
|---|---:|---:|---:|
| SRTF vs FIFO | +56.6% | +10.8% | −4.5% |
| Aging-SRTF vs FIFO | +55.1% | +19.7% | −0.7% |

The BurstGPT sample shows the SRTF short-request ordering is confirmed (+56.6%
short_p90 improvement). The small sample size means goodput/$ differentiation
is minimal. This is consistent with run -h: scale is required for meaningful
queue effects. Cross-validation on the full BurstGPT trace is the next step.

---

## Implementation

**Module:** `aurelius/benchmarks/srtf_serving_backtest.py` (extended)

New components:
- `AGING_ALPHA_DEFAULT = 0.05` — calibrated for Azure 2024 p99 parity in ~87s
- `DEFAULT_BURSTGPT_FIXTURE` / `DEFAULT_BURSTGPT_SLA_S = 30.0`
- `simulate_queue(..., discipline="aging_srtf", aging_alpha=...)` — O(|ready|) dispatch
- `load_burstgpt_serving_requests()` — BurstGPT CSV loader
- `SRTFAgingReport` — FIFO / SRTF / aging_SRTF comparison dataclass
- `run_aging_srtf_backtest()` — Azure LLM 2024 multi-discipline benchmark
- `run_burstgpt_aging_backtest()` — BurstGPT cross-validation benchmark
- `_summarize` extended: `long_p90_response_s`, `long_p99_response_s` added

**Tests:** `tests/test_srtf_aging_backtest.py` — 37 new tests, all passing.
Zero regressions (27 existing serving tests all pass; 14 pre-existing failures
in `test_srtf_scheduling.py` unchanged).

---

## Honest Caveats

1. **FIFO baseline** — the +323.5% goodput/$ for SRTF and +22.4% for aging-SRTF
   are vs FIFO, not vs an SLA-aware scheduler. The +300% north-star target is
   vs SLA-aware. These results do not claim to reach that target.

2. **Non-preemptive model** — long requests cannot be interrupted once started.
   A preemptive SRPT would further reduce long-tail regression.

3. **Single trace** — Azure 2024 is the only full-size validation. BurstGPT
   requires the full 1.4M-row dataset for confirmatory evidence.

4. **Time-warp model** — arrivals are compressed to hit target ρ; real burst
   micro-structure is preserved but absolute burst magnitudes may differ.

5. **Simulator directional** — the discrete-event M/G/c model is a faithful
   physics representation but does not account for continuous-batching throughput
   scaling or KV-cache-driven service-time variation.

---

## Next Steps

1. **Lower α to 0.01** in the default and re-validate: retains +70.7% goodput/$
   vs FIFO while reducing starvation by 49%.
2. **Add preemptive SRPT** to the simulator: when a shorter job arrives, preempt
   the current job at operator boundaries (FlowPrefill arXiv:2602.16603).
3. **Full BurstGPT replay** (1.4M rows) to cross-validate aging-SRTF gain at scale.
4. **Wire into serving path**: expose aging_srtf as an ordering option in the
   Aurelius serving runtime, driven by OutputLengthForecastBundle.p50 as the
   predicted_tokens prior.
