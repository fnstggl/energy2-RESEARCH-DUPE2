# Forecasted MCS Spot Fleet Backtest — 2026-06-24

**Status:** COMPLETE — NEUTRAL/NEGATIVE result (neither sub-mode beats AMCSG)

## Summary

First apples-to-apples evaluation of `forecasted_mcs` (the only fully deployable
replica-scaling mode) under the GSF spot-fleet cost model. Prior runs only evaluated
`forecasted_mcs` under on-demand pricing; this run uses the same conditions as the
AMCSG/OSOTSS spot-fleet leaderboard.

**Finding:** Both forecasted_mcs sub-modes (lag1, ewma) are BELOW AMCSG on both
traces, with BurstGPT showing severe SLA degradation. The modes are NOT competitive
with arrival-oracle AMCSG under spot-fleet economics.

## Conditions

| Parameter | Value |
|---|---|
| Entry point | `run_forecasted_mcs_spot_{azure,burstgpt}_backtest` |
| Routing | `AureliusOptimizer(policy="replica_scaling", mode="forecasted_mcs")` |
| Spot fraction | 0.95 |
| Spot price | $0.80/GPU-hr |
| On-demand price | $2.00/GPU-hr |
| Interruption rate | 10%/hr (Binomial model) |
| ZF-HC threshold | 8 |
| MCS gate | 12.5% (same as AMCSG) |
| Seed | 42 |
| Azure SLA | 10s |
| BurstGPT SLA | 30s |
| Requests | 5,880 (both traces) |

## Results

### Azure LLM 2024

| Policy | Goodput/$ | vs AMCSG | n_sla_safe | p99 (s) | North-star? |
|---|---|---|---|---|---|
| AMCSG (gate=12.5%) | **150,630** | — | 5,823 | 9.95 | — |
| forecasted_mcs lag1 | 149,110 | **−1.01%** | 5,821 (−2) | 10.03 | ❌ |
| forecasted_mcs ewma | 150,162 | **−0.31%** | 5,808 (−15) | 10.37 | ❌ |
| SLA oracle (ref) | 25,208 | — | — | — | — |
| North-star ×6 (ref) | 151,248 | — | — | — | — |

Azure verdict: Both modes within ~1% of AMCSG but marginally below on both goodput/$ and n_sla_safe. The ewma mode comes closest (−0.31%).

### BurstGPT HF

| Policy | Goodput/$ | vs AMCSG | n_sla_safe | p99 (s) | North-star? |
|---|---|---|---|---|---|
| AMCSG (gate=12.5%) | **168,270** | — | 5,864 | 22.92 | — |
| forecasted_mcs lag1 | 147,181 | **−12.5%** | 5,344 (−520) | 47.47 | ❌ |
| forecasted_mcs ewma | 103,192 | **−38.7%** | 3,840 (−2,024) | 67.44 | ❌ |
| SLA oracle (ref) | 20,280 | — | — | — | — |
| North-star ×6 (ref) | 121,680 | — | — | — | — |

BurstGPT verdict: Both modes severely underperform AMCSG. Lag1 p99=47.5s (exceeds 30s SLA), ewma p99=67.4s (more than 2× SLA). AMCSG p99=22.9s safely within SLA.

## Root Cause Analysis

BurstGPT's namesake is burst traffic. The forecasted_mcs modes can only use past data (≤ t-1); they cannot see the burst that is arriving in tick t. AMCSG uses the actual tick-t arrival counts (arrival-oracle), so it provisions exactly the right capacity for each burst.

Lag1 (reactive from t-1) misses sharp burst onsets by exactly one tick — on BurstGPT, where burst onset causes the largest SLA risk, this one-tick gap causes 520 additional SLA violations.

EWMA (smoothed) spreads out corrections over multiple ticks, which is even worse for sharp bursts — 2,024 additional SLA violations and p99 > 2× the SLA limit.

Azure's smoother traffic pattern means the one-tick lag is less severe (only 2-15 additional violations vs AMCSG), but still not enough to achieve north-star or match AMCSG.

## Classification

| Claim | Result |
|---|---|
| forecasted_mcs lag1 beats AMCSG (Azure) | **FAIL** (−1.01%) |
| forecasted_mcs ewma beats AMCSG (Azure) | **FAIL** (−0.31%) |
| forecasted_mcs lag1 beats AMCSG (BurstGPT) | **FAIL** (−12.5%) |
| forecasted_mcs ewma beats AMCSG (BurstGPT) | **FAIL** (−38.7%) |
| North-star achieved (any sub-mode, any trace) | **FAIL** |
| n_sla_safe_safe (any sub-mode, any trace) | **FAIL** |

**Verdict: NEUTRAL/NEGATIVE.** Forecasted MCS cannot match the arrival-oracle advantage
of AMCSG on either trace. The gap is structural, not stochastic — forecasted_mcs has no
mechanism to see tick-t arrivals before they occur. AMCSG's oracle access to actual
tick-t counts is the source of its advantage.

## Decision

This confirms forecasted_mcs is deployment-realistic but weaker than AMCSG for
spot-fleet scheduling. No leaderboard update. The OSOTSS result (which also uses
actual tick-t arrivals in the oracle loop, making it arrival-oracle rather than
fully deployable) remains the spot-fleet frontier.

Forecasted_mcs remains the only FULLY deployable mode (no oracle) but operates
in a different tradeoff regime — it is appropriate for scenarios where actual
tick-t arrival data is truly unavailable before scheduling decisions are made.

## Files

- `research/results/forecasted_mcs_spot_backtest_2026-06-24.json` — machine-readable results
- `aurelius/benchmarks/srtf_serving_backtest.py` — `ForecastedMCSSpotReport`,
  `_run_forecasted_mcs_spot_backtest`, `run_forecasted_mcs_spot_azure_backtest`,
  `run_forecasted_mcs_spot_burstgpt_backtest`
- `tests/test_forecasted_mcs_spot_backtest.py` — 18 tests pass (28 skipped if numpy absent)
