# Forecasted MCS Backtest — deployable vs oracle capacity provisioning

> **Directional simulator evidence only — NOT production savings** (`docs/RESULTS.md` §8).

- Generated: 2026-06-23
- Physics: TTFT_BASE_S(0.150) + output_tokens*TPOT_S(0.020); gate = Erlang-C M/M/c SLA-timeout < mcs_gate%
- Cost denominator: provisioned GPU-hours over fixed trace window = sum(c)*tick_hr * $2.00/hr
- Goodput: SLA-safe output tokens (response <= sla_s)
- Config: {"job_limit": 5880, "fixed_c_sweep": [4, 5, 6, 7, 8, 10, 12], "target_rho": 0.85, "tick_seconds": 60.0, "mcs_gate": 9.5}

All policies share one physics model, one cost denominator (provisioned GPU-hours over the **fixed** trace window — a backed-up queue does not extend the billing window), one SLA definition, one trace, and one discrete-event simulator family. Only the per-tick capacity schedule and the queue discipline differ.

## azure_llm_2024 — 5,880 req, 72 ticks, SLA 10.0s, warp 21.9519

- Strongest deployable **fixed** SLA-aware baseline (no MCS): **sla_aware_fixed_c7** = 38,351 goodput/$
- Documented leaderboard baseline (sla_aware @ c=4): **sla_aware_fixed_c4** = 25,208 goodput/$ (under-provisioned — see SLA violations below)
- Strongest deployable **MCS** baseline: **reactive_lag1_mcs** = 59,097 goodput/$

### KPI table (Phase 5 required format)

`Δ vs SLA-aware fixed` is vs the strongest swept fixed-c SLA-aware baseline; `Δ vs forecasted-MCS baseline` is vs the naive lag-1 forecast.

| Policy | Goodput/$ | Δ vs SLA-aware fixed | Δ vs forecasted-MCS baseline | GPU-hours | Cost $ | SLA violations | p99 queue |
|---|---:|---:|---:|---:|---:|---:|---:|
| fifo_fixed_c4 | 11,183 | -70.8% | -81.1% | 4.800 | 9.60 | 4917 | 730.29s |
| sla_aware_fixed_c4 | 25,208 | -34.3% | -57.3% | 4.800 | 9.60 | 2475 | 843.82s |
| fifo_fixed_c5 | 14,768 | -61.5% | -75.0% | 6.000 | 12.00 | 4262 | 251.89s |
| sla_aware_fixed_c5 | 24,383 | -36.4% | -58.7% | 6.000 | 12.00 | 2156 | 324.75s |
| fifo_fixed_c6 | 27,420 | -28.5% | -53.6% | 7.200 | 14.40 | 2340 | 31.63s |
| sla_aware_fixed_c6 | 31,880 | -16.9% | -46.1% | 7.200 | 14.40 | 1174 | 44.01s |
| fifo_fixed_c7 | 38,403 | +0.1% | -35.0% | 8.400 | 16.80 | 56 | 2.63s |
| sla_aware_fixed_c7 | 38,351 | +0.0% | -35.1% | 8.400 | 16.80 | 59 | 2.66s |
| fifo_fixed_c8 | 33,646 | -12.3% | -43.1% | 9.600 | 19.20 | 54 | 1.12s |
| sla_aware_fixed_c8 | 33,628 | -12.3% | -43.1% | 9.600 | 19.20 | 55 | 1.07s |
| fifo_fixed_c10 | 26,917 | -29.8% | -54.5% | 12.000 | 24.00 | 54 | 0.00s |
| sla_aware_fixed_c10 | 26,917 | -29.8% | -54.5% | 12.000 | 24.00 | 54 | 0.00s |
| fifo_fixed_c12 | 22,431 | -41.5% | -62.0% | 14.400 | 28.80 | 54 | 0.00s |
| sla_aware_fixed_c12 | 22,431 | -41.5% | -62.0% | 14.400 | 28.80 | 54 | 0.00s |
| reactive_lag1_mcs | 59,097 | +54.1% | +0.0% | 5.450 | 10.90 | 59 | 3.90s |
| forecast_mcs_ewma | 59,097 | +54.1% | +0.0% | 5.450 | 10.90 | 59 | 3.90s |
| forecast_mcs_quantile_p90 | 50,668 | +32.1% | -14.3% | 6.367 | 12.73 | 56 | 3.18s |
| oracle_mcs | 59,694 | +55.6% | +1.0% | 5.400 | 10.80 | 57 | 2.32s |
| forecast_mcs_ewma+abs_conformal | 57,139 | +49.0% | -3.3% | 5.450 | 10.90 | 116 | 4.17s |
| oracle_mcs+abs_conformal | 58,323 | +52.1% | -1.3% | 5.400 | 10.80 | 96 | 3.18s |

_Documented leaderboard baseline reference: sla_aware_fixed_c4 = 25,208 goodput/$ (sla_aware @ c=4, under-provisioned)._

### Classification

| Policy | Forecast type | Uses future info? | Deployable? | Classification |
|---|---|---|---|---|
| fifo_fixed_c4 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c4 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c5 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c5 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c6 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c6 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c7 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c7 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c8 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c8 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c10 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c10 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c12 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c12 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| reactive_lag1_mcs | naive lag-1 | no | YES | Deployable (forecast MCS) |
| forecast_mcs_ewma | EWMA point | no | YES | Deployable (forecast MCS) |
| forecast_mcs_quantile_p90 | rolling p90 + 1σ | no | YES | Deployable (forecast MCS) |
| oracle_mcs | clairvoyant | YES | NO | Oracle upper bound |
| forecast_mcs_ewma+abs_conformal | EWMA point | no | YES | Deployable (forecast MCS + SRTF) |
| oracle_mcs+abs_conformal | clairvoyant | YES | NO | Oracle upper bound (+SRTF) |

### Forecast error (causal, vs realised ticks)

| Policy | arrival MAE | arrival rel-MAE | arrival bias | service MAE (s) |
|---|---:|---:|---:|---:|
| forecast_mcs_ewma | 7.98 | 9.8% | +0.29 | 0.212 |
| forecast_mcs_quantile_p90 | 22.47 | 27.5% | +20.58 | 0.212 |
| forecast_mcs_ewma+abs_conformal | 7.98 | 9.8% | +0.29 | 0.212 |

### Decomposition

**Best deployable forecast MCS (reactive_lag1_mcs) vs strongest fixed SLA-aware (sla_aware_fixed_c7):**
- Goodput/$: 38,351 → 59,097 (+54.1%)
- GPU-hours: 8.400 → 5.450 (-35.1% — less capacity)
- SLA violations: 59 → 59 (Δ +0)

**Oracle MCS → best deployable forecast MCS (reactive_lag1_mcs):**
- Goodput/$: 59,694 → 59,097 (-1.0% — forecast retains 99.0% of oracle)
- GPU-hours: oracle 5.400 → forecast 5.450 (+0.9%)
- SLA violations: oracle 57 → forecast 59 (Δ +2)

**North-star (+300% vs strongest fixed SLA-aware): NOT ACHIEVED** (best deployable forecast MCS is +54.1%).

## burstgpt_hf — 5,880 req, 154 ticks, SLA 30.0s, warp 17.9955

- Strongest deployable **fixed** SLA-aware baseline (no MCS): **sla_aware_fixed_c8** = 34,468 goodput/$
- Documented leaderboard baseline (sla_aware @ c=4): **sla_aware_fixed_c4** = 17,189 goodput/$ (under-provisioned — see SLA violations below)
- Strongest deployable **MCS** baseline: **reactive_lag1_mcs** = 58,953 goodput/$

### KPI table (Phase 5 required format)

`Δ vs SLA-aware fixed` is vs the strongest swept fixed-c SLA-aware baseline; `Δ vs forecasted-MCS baseline` is vs the naive lag-1 forecast.

| Policy | Goodput/$ | Δ vs SLA-aware fixed | Δ vs forecasted-MCS baseline | GPU-hours | Cost $ | SLA violations | p99 queue |
|---|---:|---:|---:|---:|---:|---:|---:|
| fifo_fixed_c4 | 5,534 | -84.0% | -90.6% | 10.267 | 20.53 | 5435 | 1,120.60s |
| sla_aware_fixed_c4 | 17,189 | -50.1% | -70.8% | 10.267 | 20.53 | 2741 | 1,205.11s |
| fifo_fixed_c5 | 9,491 | -72.5% | -83.9% | 12.833 | 25.67 | 4867 | 410.09s |
| sla_aware_fixed_c5 | 17,411 | -49.5% | -70.5% | 12.833 | 25.67 | 2509 | 511.47s |
| fifo_fixed_c6 | 16,938 | -50.9% | -71.3% | 15.400 | 30.80 | 3746 | 198.46s |
| sla_aware_fixed_c6 | 21,484 | -37.7% | -63.6% | 15.400 | 30.80 | 1975 | 224.68s |
| fifo_fixed_c7 | 30,906 | -10.3% | -47.6% | 17.967 | 35.93 | 1473 | 79.25s |
| sla_aware_fixed_c7 | 32,036 | -7.1% | -45.7% | 17.967 | 35.93 | 798 | 81.92s |
| fifo_fixed_c8 | 34,534 | +0.2% | -41.4% | 20.533 | 41.07 | 275 | 55.99s |
| sla_aware_fixed_c8 | 34,468 | +0.0% | -41.5% | 20.533 | 41.07 | 195 | 58.79s |
| fifo_fixed_c10 | 28,612 | -17.0% | -51.5% | 25.667 | 51.33 | 87 | 24.39s |
| sla_aware_fixed_c10 | 28,345 | -17.8% | -51.9% | 25.667 | 51.33 | 120 | 26.38s |
| fifo_fixed_c12 | 24,389 | -29.2% | -58.6% | 30.800 | 61.60 | 11 | 10.62s |
| sla_aware_fixed_c12 | 24,389 | -29.2% | -58.6% | 30.800 | 61.60 | 11 | 11.31s |
| reactive_lag1_mcs | 58,953 | +71.0% | +0.0% | 11.133 | 22.27 | 526 | 39.11s |
| forecast_mcs_ewma | 41,006 | +19.0% | -30.4% | 11.183 | 22.37 | 2020 | 56.55s |
| forecast_mcs_quantile_p90 | 40,235 | +16.7% | -31.8% | 18.300 | 36.60 | 63 | 21.54s |
| oracle_mcs | 67,107 | +94.7% | +13.8% | 11.150 | 22.30 | 16 | 11.10s |
| forecast_mcs_ewma+abs_conformal | 56,861 | +65.0% | -3.5% | 11.183 | 22.37 | 372 | 151.51s |
| oracle_mcs+abs_conformal | 64,546 | +87.3% | +9.5% | 11.150 | 22.30 | 87 | 18.46s |

_Documented leaderboard baseline reference: sla_aware_fixed_c4 = 17,189 goodput/$ (sla_aware @ c=4, under-provisioned)._

### Classification

| Policy | Forecast type | Uses future info? | Deployable? | Classification |
|---|---|---|---|---|
| fifo_fixed_c4 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c4 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c5 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c5 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c6 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c6 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c7 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c7 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c8 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c8 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c10 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c10 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| fifo_fixed_c12 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| sla_aware_fixed_c12 | none (static, swept) | no | YES | Deployable (fixed, no MCS) |
| reactive_lag1_mcs | naive lag-1 | no | YES | Deployable (forecast MCS) |
| forecast_mcs_ewma | EWMA point | no | YES | Deployable (forecast MCS) |
| forecast_mcs_quantile_p90 | rolling p90 + 1σ | no | YES | Deployable (forecast MCS) |
| oracle_mcs | clairvoyant | YES | NO | Oracle upper bound |
| forecast_mcs_ewma+abs_conformal | EWMA point | no | YES | Deployable (forecast MCS + SRTF) |
| oracle_mcs+abs_conformal | clairvoyant | YES | NO | Oracle upper bound (+SRTF) |

### Forecast error (causal, vs realised ticks)

| Policy | arrival MAE | arrival rel-MAE | arrival bias | service MAE (s) |
|---|---:|---:|---:|---:|
| forecast_mcs_ewma | 12.92 | 33.8% | -0.33 | 1.163 |
| forecast_mcs_quantile_p90 | 32.91 | 86.2% | +29.79 | 1.163 |
| forecast_mcs_ewma+abs_conformal | 12.92 | 33.8% | -0.33 | 1.163 |

### Decomposition

**Best deployable forecast MCS (reactive_lag1_mcs) vs strongest fixed SLA-aware (sla_aware_fixed_c8):**
- Goodput/$: 34,468 → 58,953 (+71.0%)
- GPU-hours: 20.533 → 11.133 (-45.8% — less capacity)
- SLA violations: 195 → 526 (Δ +331)

**Oracle MCS → best deployable forecast MCS (reactive_lag1_mcs):**
- Goodput/$: 67,107 → 58,953 (-12.2% — forecast retains 87.8% of oracle)
- GPU-hours: oracle 11.150 → forecast 11.133 (-0.1%)
- SLA violations: oracle 16 → forecast 526 (Δ +510)

**North-star (+300% vs strongest fixed SLA-aware): NOT ACHIEVED** (best deployable forecast MCS is +71.0%).

