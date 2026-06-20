# Baseline Public Backtest — current main (no module integration)

> **Directional simulator/backtest evidence only — NOT production savings** (`docs/RESULTS.md` §8). Phase-3 snapshot of current-main KPIs before any research module is wired in.

- Generated: 2026-06-20

## Serving traces (aggregate autoscaling replay)

`constraint_aware` is the Aurelius policy; KPIs are its values. Migration count = autoscaler scale events. Deadline-miss / migration-cost / optimizer-runtime do not apply to the trivial autoscaler (serving path).

| dataset | load | SLA-safe goodput/$ | GPU-hours | cost | SLA viol (timeout %) | queue p99 (ms) | migration (scale ev) | CA vs sla_aware |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| burstgpt (100,000) | 1.0× | 3,879.37 | 1,463.17 | 3,108.93 | 1.441 | 4.20 | 0 | +0.08% |
| burstgpt (100,000) | 300.0× | 789,331.10 | 6.47 | 15.22 | 2.355 | 134.48 | 35 | +22.10% |
| azure_llm_2024 (5,880) | 1.0× | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 0 | +0.00% |
| azure_llm_2024 (5,880) | 50.0× | 604,601.10 | 0.53 | 1.09 | 3.317 | 240.78 | 0 | +0.00% |

## Canonical energy backtest (JobScheduler path)

- Solve runtime: 27.149 s · jobs: 1000

| policy | SLA-safe goodput/$ | total cost $ | energy $ | deadline misses | migrations | migration cost $ |
|---|---:|---:|---:|---:|---:|---:|
| constraint_aware_with_energy_adapter | 0.33730 | 51,725.58 | 16,485.58 | 0 | 692 | 346.00 |
| current_price_only | 0.30368 | 57,452.65 | 22,133.15 | 0 | 851 | 425.50 |
| fifo | 0.16578 | 105,241.13 | 70,347.13 | 0 | 0 | 0.00 |

## GPU routing baseline (GpuPlacementScorer DISABLED)

- baseline goodput/$: 0.300667
- baseline latency_critical goodput/$: 0.371870
- baseline realized energy $: 14,561.38
- baseline % latency_critical on best GPU: 0.033

