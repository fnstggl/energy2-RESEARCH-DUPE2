# Live Causal Prior + Compound Serving/Economic Backtest — 2026-06-21

**Run:** 2026-06-21-t  |  **Status:** Public-trace M/G/c discrete-event replay

> Directional simulator/backtest evidence — not production savings (docs/RESULTS.md §8).

## Summary

This run closes the oracle gap by replacing the oracle prediction prior with
a **causal sliding-window median estimator** — the minimum viable production
prior that uses only the trace's own historical completion statistics.

### Key Results

- **Azure LLM 2024 live prior:** +244.42% vs FIFO (81.6% retention vs oracle)
- **Azure prior quality:** CV=7.0%, MAE=60.5 tokens
- **BurstGPT HF live prior:** +420.83% vs FIFO (70.0% retention vs oracle)
- **BurstGPT prior quality:** CV=15.3%, MAE=166.9 tokens

## Public Trace Backtest Results

### Azure LLM 2024 (5,880 requests, ρ=0.85, 4 servers, SLA=10s)

- Dataset: Azure LLM Inference Trace 2024 (public, DynamoLLM HPCA 2025)
- Requests: 5880  |  Servers: 4
- Target ρ: 0.85  |  SLA: 10.0s

| Discipline | SLA-safe goodput/$ | vs FIFO |
|---|---:|---:|
| FIFO (baseline) | 13336.33 | — |
| Conformal oracle | 56311.40 | +322.24% |
| Conformal live prior | 45932.71 | +244.42% |

**Live vs oracle retention: 81.6%**
Prior CV: 7.0%  |  Prior MAE: 60.5 tokens
(30%-CV lognormal floor from run -n: 83.1% retention)

### BurstGPT HF (5,880-record sample, ρ=0.85, 4 servers, SLA=30s)

- Dataset: BurstGPT HF (59,999 records, CC-BY-4.0, 5,880 sampled)
- Requests: 5880  |  Servers: 4
- Target ρ: 0.85  |  SLA: 30.0s

| Discipline | SLA-safe goodput/$ | vs FIFO |
|---|---:|---:|
| FIFO (baseline) | 6528.76 | — |
| Conformal oracle | 48598.82 | +644.38% |
| Conformal live prior | 34003.60 | +420.83% |

**Live vs oracle retention: 70.0%**
Prior CV: 15.3%  |  Prior MAE: 166.9 tokens

## Compound Gain Table

Economic scheduling gain (constraint_aware vs FIFO) is from published
benchmark results in BENCHMARK_REGISTRY.md / run 2026-06-21-s.
Serving queue gain (live prior) is measured in this run.

| Lever | vs FIFO | Source |
|---|---:|---|
| SLA-aware binary priority | +125.4% | run -n |
| Economic scheduling only (constraint_aware) | +183.4% | BENCHMARK_REGISTRY |
| Conformal queue only (oracle) | +322.24% | run -q |
| Conformal queue only (live prior) | +244.42% | **this run** |
| Compound: live queue + economic (est.) | +876.2% (estimated) | independence estimate |

Independence assumption: economic (provisioning) and serving queue (request ordering)
improvements operate on orthogonal dimensions. The compound estimate is a product
of their individual multipliers. A true end-to-end integrated backtest remains
the highest expected value next step.

## North Star Progress

North Star target: +300% SLA-safe goodput/$ vs SLA-aware schedulers.

Current best (live prior conformal vs SLA-aware):
  = conformal_live_vs_fifo / (sla_aware_vs_fifo + 1) × 100 − 100

  Azure LLM 2024: +52.8% vs SLA-aware (target: +300%, gap: 247.2pp)

## Methodology

- **Live prior**: For request i, predict output_tokens as median of actual tokens
  from requests 0..i-1 (causal, no future leakage).
- **Service time**: always actual_tokens × TPOT_S (no leakage in serving physics).
- **Identical server pool**: 4 servers, identical across all disciplines.
- **Time warp**: single scalar to achieve target ρ=0.85, applied identically.
- **Conformal calibrator**: adapts α from empirical prediction errors observed
  during the simulation (causal: error measured after completion).

This is a discrete-event M/G/c simulator result, not production savings.
See docs/RESULTS.md §8 for the full honesty/limitations statement.