# Forecasted-MCS Component Ablation

> **Directional simulator evidence only — NOT production savings** (`docs/RESULTS.md` §8).

- Generated: 2026-06-23
- One physics model, one SLA, one cost denominator (provisioned GPU-hours over the fixed trace window). Only the listed component varies.
- **Keep rule:** a component is kept only if it improves SLA-safe goodput/$ vs the forecasted-MCS baseline by >0.5% **without** material SLA regression, and (for cost levers) helps MCS more than it helps the fixed baseline.
- GPU menu (real median on-demand $/gpu-hr × documented decode throughput): H100 $9.73/hr, A100 $3.52/hr, A10 $2.00/hr, L4 $1.18/hr, T4 $1.15/hr

## azure_llm_2024 — 5,880 req, 72 ticks, SLA 10.0s

Forecasted-MCS baseline: **59,097 goodput/$**

| Condition | Goodput/$ | Δ vs forecasted MCS | GPU-h | Cost $ | SLA viol | p99 queue |
|---|---:|---:|---:|---:|---:|---:|
| no_mcs_best_fixed(fixed_c7) | 38,403 | -35.0% | 8.400 | 16.80 | 56 | 2.63s |
| oracle_mcs | 59,694 | +1.0% | 5.400 | 10.80 | 57 | 2.32s |
| osotss_arrival_oracle | 63,831 | +8.0% | 5.050 | 10.10 | 57 | 3.38s |
| forecasted_mcs | 59,097 | +0.0% | 5.450 | 10.90 | 59 | 3.90s |
| forecasted_mcs+queue | 57,139 | -3.3% | 5.450 | 10.90 | 116 | 4.17s |
| forecasted_mcs+energy_routing | 58,996 | -0.2% | 5.450 | 10.92 | 59 | 3.90s |
| forecasted_mcs+placement | 58,383 | -1.2% | 5.533 | 10.82 | 131 | 5.79s |

### Component verdicts (vs forecasted-MCS baseline)

| Component | Δ goodput/$ | SLA viol | Notes | Verdict |
|---|---:|---:|---|---|
| queue policy (abs-conformal SRTF) | -3.3% | 116 |  | **DROP** |
| energy routing (CAISO/PJM/ERCOT) | -0.2% | 59 | energy=0.171% of cost; routing gain +0.45% (MCS) vs +0.45% (fixed) | **DROP** |
| placement (heterogeneous GPU menu) | -1.2% | 131 | GPU mix: A10×56, A100×9, L4×7 | **DROP** |

## burstgpt_hf — 5,880 req, 154 ticks, SLA 30.0s

Forecasted-MCS baseline: **41,006 goodput/$**

| Condition | Goodput/$ | Δ vs forecasted MCS | GPU-h | Cost $ | SLA viol | p99 queue |
|---|---:|---:|---:|---:|---:|---:|
| no_mcs_best_fixed(fixed_c8) | 34,534 | -15.8% | 20.533 | 41.07 | 275 | 55.99s |
| oracle_mcs | 67,107 | +63.6% | 11.150 | 22.30 | 16 | 11.10s |
| osotss_arrival_oracle | 71,244 | +73.7% | 10.450 | 20.90 | 31 | 15.84s |
| forecasted_mcs | 41,006 | +0.0% | 11.183 | 22.37 | 2020 | 56.55s |
| forecasted_mcs+queue | 56,861 | +38.7% | 11.183 | 22.37 | 372 | 151.51s |
| forecasted_mcs+energy_routing | 40,901 | -0.2% | 11.183 | 22.42 | 2020 | 56.55s |
| forecasted_mcs+placement | 39,803 | -2.9% | 11.267 | 21.92 | 2259 | 73.38s |

### Component verdicts (vs forecasted-MCS baseline)

| Component | Δ goodput/$ | SLA viol | Notes | Verdict |
|---|---:|---:|---|---|
| queue policy (abs-conformal SRTF) | +38.7% | 372 |  | **KEEP** |
| energy routing (CAISO/PJM/ERCOT) | -0.2% | 2020 | energy=0.254% of cost; routing gain +0.37% (MCS) vs +0.36% (fixed) | **DROP** |
| placement (heterogeneous GPU menu) | -2.9% | 2259 | GPU mix: A10×112, L4×17, A100×9, T4×16 | **DROP** |

