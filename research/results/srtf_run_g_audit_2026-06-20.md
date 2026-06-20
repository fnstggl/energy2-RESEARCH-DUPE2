# Audit of Run 2026-06-20-g's "+323% SLA-safe goodput/$" SRTF claim

> **Verdict: the +323% is NOT a verified improvement vs the real (SLA-aware /
> autoscaling) baseline on the public rollup.** It is a regime-specific
> SLA-compliance gain measured *only vs FIFO*, at a *fixed, forced-overloaded*
> capacity, in a queue model that *omits continuous batching*. Against the
> utilization an autoscaler actually maintains, the gain is **−0.1% to −1.0%**
> (i.e. ≈ 0, or slightly negative). Do not merge any runtime SRTF change on the
> basis of this claim.

Source under audit: `aurelius/benchmarks/srtf_serving_backtest.py`,
`docs/SRTF_SERVING_BACKTEST_RESULTS.md`. Reproduced from a clean checkout at
`servers=4, target_rho=0.85` → `sla_goodput_delta_pct = 323.51` (exact match).

## 7-point verification

| # | Question | Finding | Pass? |
|---|---|---|---|
| 1 | vs FIFO or vs SLA-aware? | **vs FIFO only.** The benchmark defines exactly two disciplines (`fifo`, `srtf`); there is no `sla_aware` or `constraint_aware` discipline anywhere in the module or its tests. The module's own Caveat 1 admits this. | ⚠️ vs FIFO |
| 2 | on public trace replay? | **Partly.** Real Azure-2024 *output-token* counts (5,880 reqs) are used, but **arrivals are time-warped ×22** to force ρ=0.85, and **service times are modeled** (`TTFT_BASE + tokens·TPOT`), not measured (Azure has no latency labels). So: real tokens, synthetic arrival rate, modeled latency. | ⚠️ semi-synthetic |
| 3 | uses SLA-safe goodput/$? | **A different, weaker definition.** Numerator = tokens of requests with E2E ≤ 10 s (binary cutoff, not the locked fractional `timeout_rate`). Denominator = busy GPU-seconds × \$2 only (no energy/network) and is **byte-identical across FIFO and SRTF** (same requests, same service times → 4.0248 GPU-h, \$8.05 both). **The "/$" cancels** → the +323% is purely an SLA-compliance *count* delta, not a cost-efficiency delta. | ❌ not the locked metric; "/$" is cosmetic |
| 4 | free of future leakage? | **Headline uses a clairvoyant ("perfect") prior** (orders by *actual* service time = optimal SRPT — that is leakage). The leakage-free `srtf_forecast` variant (30%-CV *synthetic* lognormal noise) gives a near-identical number, so leakage isn't the driver — **but** the real `OutputLengthForecastBundle` accuracy is never used; "robust to forecast error" rests on an *assumed* 30%-CV synthetic prior. | ⚠️ headline leaks; forecast variant synthetic |
| 5 | realistic per-request queueing? | **No.** The M/G/c model serves **one request per GPU at a time — no continuous batching**, the dominant LLM-serving mechanism (and the one the repo's own `serving.py` models via `batching_efficiency`). Without batching, a single long request fully blocks a server (head-of-line blocking), which is what makes FIFO catastrophic (84% SLA violations, 341 s mean wait). It also uses fixed `c=4` with **no autoscaling** and a forced ρ=0.85. | ❌ omits batching + autoscaling |
| 6 | robust across Azure 2024, BurstGPT, rollup? | **No.** Runs on **Azure-2024 only**. Never run on BurstGPT or the public rollup. And it **vanishes at the utilization an autoscaler maintains** (see ρ-sweep). | ❌ single trace, regime-fragile |
| 7 | reproducible from clean checkout? | **Yes.** Uses the committed `tests/fixtures/azure_llm_2024_sample.csv` + fixed seed; reproduced +323.51% exactly. | ✅ |

## ρ-sweep — the +323% is an overload artifact

Same code, same trace, only `target_rho` varies (the time-warp adjusts arrivals).
`constraint_aware` targets ρ≈0.65 and an SLA-aware autoscaler keeps it lower to
meet SLA:

| ρ | FIFO mean resp | FIFO meets 10 s SLA? | SRTF goodput/$ Δ |
|---|---:|---|---:|
| 0.30 | 2.5 s | **yes** | **−0.1%** |
| 0.50 | 3.1 s | **yes** | **−1.0%** |
| 0.65 | 101.6 s | no | +174.9% |
| 0.75 | 228.2 s | no | +235.7% |
| 0.85 | 343.9 s | no | **+323.5%** (headline) |
| 0.92 | 416.9 s | no | +314.0% |

At the load an autoscaling baseline holds (ρ ≤ 0.5), FIFO already meets the SLA and
**SRTF is −0.1% to −1.0%**. The headline exists only because the benchmark pins a
fixed pool at ρ=0.85, a state the real system avoids by adding replicas.

## Before / after KPI table

**(A) The SRTF benchmark's own model** — fixed `c=4`, ρ=0.85, M/G/c, **no batching**,
real Azure tokens, ×22 warped arrivals, SLA=10 s E2E:

| policy | SLA-safe goodput/$ | GPU-hours | cost | SLA violations | queue delay (mean wait) | deadline misses | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FIFO** | 13,336 | 4.0248 | \$8.05 | 4,917 / 5,880 (84%) | 341.4 s | 4,917 | ~20 ms |
| **SLA-aware** | — not in benchmark — | — | — | — | — | — | — |
| **constraint-aware (current main)** | — not in benchmark — | — | — | — | — | — | — |
| **SRTF (perfect prior)** | 56,481 | 4.0248 | \$8.05 | 833 / 5,880 (14%) | 127.4 s | 833 | ~21 ms |
| **SRTF (30%-CV forecast)** | 56,855 | 4.0248 | \$8.05 | ~820 | ~141 s | ~820 | ~21 ms |

GPU-hours, cost identical → goodput/$ Δ = SLA-safe-token Δ. SRTF's p99 *wait*
**regresses 730 s → 2,182 s** (long-request starvation).

**(B) The autoscaling model the real Aurelius policy uses** — `aurelius/traces/backtest.py`,
**with continuous batching**, same Azure trace (50×):

| policy | SLA-safe goodput/$ | GPU-hours | cost | timeout % (SLA viol) | queue p99 | scale events |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 604,601 | 0.533 | \$1.088 | 3.317% | 240.8 ms | 0 |
| SLA-aware | 604,601 | 0.533 | \$1.088 | 3.317% | 240.8 ms | 0 |
| constraint-aware | 604,601 | 0.533 | \$1.088 | 3.317% | 240.8 ms | 0 |

With batching, **FIFO is not catastrophic** (3.3% timeout, 241 ms queue — not 84% /
341 s). The two models disagree by ~100× on FIFO latency; the entire SRTF gain
lives in the gap created by *omitting batching*.

## "Actual improvement vs SLA-aware"

SLA-aware/constraint-aware are **absent from the SRTF benchmark**, so a like-for-like
row cannot be produced from it. But the answer is bounded by the ρ-sweep: an
SLA-aware autoscaler keeps utilization in the ρ ≤ 0.5 regime to meet SLA, where
**SRTF Δ = −0.1% to −1.0%**. Therefore the actual improvement vs an SLA-aware
(autoscaling) baseline is **≈ 0% (slightly negative)** — not +323%.

## Which benchmark assumptions changed (vs the locked public rollup)

`srtf_serving_backtest.py` is a **new, separate** benchmark; it did not alter the
locked `backtest.py` / `serving.py` / `economics.py`. But its result rests on
assumptions that **differ from the locked public-trace rollup**, and each one
inflates the gap:

1. **Baseline = FIFO only** (rollup headline baseline is `sla_aware`).
2. **Goodput/$ redefined**: binary 10 s E2E cutoff (vs fractional `timeout_rate`);
   denominator = busy-GPU-seconds only and *identical across disciplines* (so "/$"
   measures nothing the decision changes).
3. **No continuous batching** (rollup `serving.py` models `batching_efficiency`).
4. **Fixed capacity, no autoscaling** (the rollup's Aurelius policy *is* the
   autoscaling provisioning decision).
5. **Arrivals time-warped ×22** to force ρ=0.85 (vs the trace's native rate).
6. **New SLA = 10 s E2E** (vs the rollup's TTFT-2000 ms + 50 ms/token decomposition).

## Bottom line

- The **SRTF *ordering principle* is real** under genuine, un-batched, fixed-pool
  contention — short requests stop queueing behind long ones. The module is honest
  about being vs-FIFO and regime-dependent, and (correctly) calls the merged
  scheduler sort key "inert for serving."
- **The "+323% SLA-safe goodput/$" headline does not transfer** to (a) the real
  SLA-aware/autoscaling baseline (≈ 0%), (b) a continuous-batching server (FIFO
  isn't catastrophic), or (c) any trace beyond Azure-2024. It is a vs-FIFO,
  fixed-overload, no-batching, binary-SLA artifact, and the "/$" denominator
  cancels.
- **Recommendation:** do not enable SRTF in the serving runtime or claim a
  goodput/$ improvement on its basis. If pursued, it must be measured vs the
  autoscaling baseline, with continuous batching, and on ≥2 public traces.

*Directional simulator audit — not production savings (`docs/RESULTS.md` §8).*
