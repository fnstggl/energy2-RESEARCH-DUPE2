# SRTF Serving Queue Backtest Results

Generated: 2026-06-22T02:32:39Z

## Summary

Request-level SRTF/conformal queue discipline evaluation on two public LLM serving traces.
Service physics: M/G/c discrete-event queue, identical across disciplines.
All differences in goodput/$ come purely from queue ordering.

**Disciplines:**
- `fifo` — arrival order (no prediction)
- `conformal_oracle` — decoupled hybrid SRPT with oracle token counts (upper bound)
- `conformal_live` — decoupled hybrid SRPT with causal sliding-window median prior

## Results Summary

| KPI | Azure LLM 2024 | BurstGPT HF | Unit |
|-----|---------------:|------------:|------|
| Oracle vs FIFO | 322.24 | 644.38 | % vs FIFO |
| Live prior vs FIFO | 244.42 | 420.83 | % vs FIFO |
| Oracle retention | 81.6 | 70.0 | % of oracle |
| FIFO goodput/$ | 13,336 | 6,529 | tokens/$ |
| Oracle goodput/$ | 56,311 | 48,599 | tokens/$ |
| Live goodput/$ | 45,933 | 34,004 | tokens/$ |
| Prior CV | 7.0 | 15.3 | % |
| Prior MAE | 60.5 | 166.9 | tokens |

## Detailed Results


### Azure LLM 2024
- Requests: 5,880 | Servers: 4 | ρ=0.85 | SLA=10.0s | Window=200
- Prior quality: CV=7.0% | MAE=60.5 tok | RelMAE=52.3%

| Discipline            | SLA-safe goodput/$ | vs FIFO      |
|----------------------|-------------------:|-------------:|
| FIFO (baseline)      |             13,336 | —            |
| Conformal oracle     |             56,311 |     +322.24% |
| Conformal live prior |             45,933 |     +244.42% |

**Oracle retention: 81.6%** (live prior retains 81.6% of oracle gain; production-viable threshold ≥83%)

> Shadow tag: `shadow_only_simulator_result_not_production_savings`

### BurstGPT HF
- Requests: 5,880 | Servers: 4 | ρ=0.85 | SLA=30.0s | Window=200
- Prior quality: CV=15.3% | MAE=166.9 tok | RelMAE=64.5%

| Discipline            | SLA-safe goodput/$ | vs FIFO      |
|----------------------|-------------------:|-------------:|
| FIFO (baseline)      |              6,529 | —            |
| Conformal oracle     |             48,599 |     +644.38% |
| Conformal live prior |             34,004 |     +420.83% |

**Oracle retention: 70.0%** (live prior retains 70.0% of oracle gain; production-viable threshold ≥83%)

> Shadow tag: `shadow_only_simulator_result_not_production_savings`

## Prior Benchmarks (from ROADMAP)

| Run | Trace | Discipline | Result vs FIFO | Oracle Retention |
|-----|-------|------------|---------------:|-----------------|
| 2026-06-21-t | Azure LLM 2024 | Conformal live prior | +244.42% | 81.6% |
| 2026-06-21-t | BurstGPT 5,880 | Conformal live prior | +420.83% | 70.0% |
| 2026-06-21-q | Azure LLM 2024 | Conformal oracle     | +322.24% | 100%  |
| 2026-06-21-r | BurstGPT 5,880 | Conformal oracle     | +644.4%  | 100%  |

## Methodology

- **SLA-safe goodput/$**: `Σ actual_tokens[i where e2e_latency ≤ sla_s] / (Σ service_s / 3600 × GPU_HOUR_USD)`
- **Infra-dollar denominator**: `GPU_HOUR_USD = $2.00/replica-hour` × total service seconds — identical across disciplines
- **Service time**: `TTFT_BASE_S (0.15s) + output_tokens × TPOT_S (0.02s)`
- **Time warp**: arrivals rescaled to `target_rho=0.85` cluster utilization, applied identically to all disciplines
- **Causal prior**: prediction[i] = median of actual_tokens[0..i-1] (last 200 completions), no future leakage

Directional simulator evidence — **not** production savings (docs/RESULTS.md §8).