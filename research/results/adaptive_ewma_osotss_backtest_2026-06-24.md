# Adaptive EWMA Online SOTSS Backtest — 2026-06-24

## Summary

**NEGATIVE RESULT.** Adaptive EWMA alpha does not produce a frontier improvement on
either canonical public trace. The BurstGPT 15-request SLA gap persists under all
tested hyperparameter configurations. Five-Failure counter advances to **3/5**.

## Hypothesis

The BurstGPT 15-request SLA gap (OSOTSS n_sla_safe=5849 vs AMCSG 5864) is caused by
EWMA alpha=0.1 being too slow to track burst patterns, causing the oracle to add capacity
to wrong ticks. An adaptive EWMA (alpha boost when tick_mean > burst_threshold × ewma_val)
should close this gap without oracle access.

## Same-Conditions Checklist

| Dimension | Status |
|---|---|
| Trace | Azure LLM 2024 (5880 req, SLA=10s), BurstGPT HF (5880 req, SLA=30s) |
| SLA definition | Identical: E2E latency ≤ sla_s |
| Cost denominator | GPU-hr ($0.80 spot, $2.00 on-demand) |
| GPU-hour accounting | GSF spot-fleet simulation, seed=42 |
| Decision-time information | Both use only past observations (causal EWMA) |
| Baseline | OSOTSS fixed alpha=0.1 |
| Evaluation method | Stochastic GSF simulation, seed=42 (same as baseline) |

All same-conditions requirements satisfied.

## Results

### Azure LLM 2024 (SLA=10s)

| Method | Goodput/$ | n_sla_safe | vs fixed |
|---|---|---|---|
| AMCSG gate=12.5% | 150,630 | 5,823 | (reference) |
| OSOTSS fixed α=0.1 | **159,578** | 5,823 | — |
| OSOTSS adaptive (any config) | **159,578** | 5,823 | +0.00% |

Adaptive EWMA **never triggers** on Azure LLM 2024. The smooth workload keeps tick_mean
below 2.0 × ewma_val at every tick under all tested thresholds (1.5, 2.0, 3.0).
No regression; no improvement.

### BurstGPT HF (SLA=30s)

| Method | Goodput/$ | n_sla_safe | vs fixed |
|---|---|---|---|
| AMCSG gate=12.5% | 168,270 | 5,864 | (reference) |
| OSOTSS fixed α=0.1 | **178,109** | 5,849 | — |
| Adaptive threshold=2.0 | **178,109** | 5,849 | +0.00% |
| Adaptive threshold=1.5, burst_α=0.15, cd=1 | 177,415 | 5,851 | −0.39% |
| Adaptive threshold=1.5, burst_α=0.25, cd=1 | 176,295 | 5,851 | −1.02% |
| Adaptive threshold=1.5, burst_α=0.50, cd=2 | 174,838 | 5,853 | −1.84% |

**Pattern:** At threshold=2.0+, adaptive never triggers (identical to fixed). At
threshold=1.5, adaptive triggers on BurstGPT burst ticks and adds 2–4 n_sla_safe
requests, but at the cost of goodput/$ regressions of 0.39–1.84% due to
over-provisioning.

## Hyperparameter Sweep

```
threshold | burst_α | cd | BurstGPT gp/$  | n_sla  | Azure gp/$ | n_sla
----------|---------|----|-----------     |--------|------------|------
1.5       | 0.15    | 1  | 177,415 (-694) | 5851   | 159,578    | 5823
1.5       | 0.20    | 1  | 176,853 (-1256)| 5851   | 159,578    | 5823
1.5       | 0.25    | 1  | 176,295 (-1814)| 5851   | 159,578    | 5823
2.0       | 0.20    | 1  | 178,109 (=)    | 5849   | 159,578    | 5823  (no trigger)
2.0       | 0.30    | 1  | 178,109 (=)    | 5849   | 159,578    | 5823  (no trigger)
2.0       | 0.50    | 1  | 178,109 (=)    | 5849   | 159,578    | 5823  (no trigger)
3.0       | 0.30    | 1  | 178,109 (=)    | 5849   | 159,578    | 5823  (no trigger)
3.0       | 0.50    | 1  | 178,109 (=)    | 5849   | 159,578    | 5823  (no trigger)
1.5       | 0.15    | 0  | 178,109 (=)    | 5849   | 159,578    | 5823  (cd=0 = no effect)
2.0       | 0.20    | 0  | 178,109 (=)    | 5849   | 159,578    | 5823  (cd=0 = no effect)
```

No configuration achieves goodput/$ ≥ fixed AND n_sla_safe ≥ fixed on both traces.

## Root Cause Analysis — Revised Diagnosis

The **original hypothesis was falsified.** The BurstGPT 15-request gap is not caused by
EWMA predictions guiding the oracle to wrong ticks. The revised diagnosis:

**Stochastic/deterministic simulation mismatch.** The oracle's convergence check uses
deterministic FIFO simulation (no spot interruptions). The stochastic GSF evaluation
includes Binomial interruptions (p_survive ≈ 0.9982/tick) that occasionally reduce
effective capacity by 1 on heavy-load ticks, exposing 15 borderline requests.

**Why adaptive EWMA helps n_sla_safe but hurts goodput/$:** Adaptive EWMA overestimates
service time on burst ticks → oracle adds more servers → covers some of the stochastic
interrupt buffer. But this is pure over-provisioning: extra servers are added to more
ticks than necessary, raising cost and lowering goodput/$.

**Implication:** Closing the 15-request gap requires accounting for the stochastic
interruption buffer in the oracle convergence target (e.g., set baseline_n_sla_safe to
amcsg_n_sla_safe + safety_margin), not improving EWMA predictions.

## Implementation

The adaptive EWMA infrastructure is correctly wired into the canonical optimizer:
- `compute_online_sotss_schedule`: `ewma_mode`, `burst_threshold`, `burst_alpha`, `burst_cooldown_ticks` params
- `ReplicaScalingConfig`: same fields with sane defaults
- `ReplicaScalingPolicy.optimize()`: forwards new params on the `online_sotss` path
- `run_online_sotss_azure_backtest`, `run_online_sotss_burstgpt_backtest`: forwarding added

The infrastructure is preserved for future experiments (e.g., different oracle target
strategies that may benefit from better EWMA predictions).

## Verdict

- **Frontier improvement:** NO
- **Regression introduced:** NO (ewma_mode="fixed" default is byte-identical to pre-change behavior)
- **Five-Failure counter:** 3/5

## Tests

`tests/test_adaptive_ewma_backtest.py` — 8 tests, all passing:
- Fixed mode is byte-identical to default (regression guard)
- Adaptive mode runs without error on linear and burst traces
- Burst-detection logic triggers correctly
- `ReplicaScalingConfig` new fields wired through
- `AureliusOptimizer(policy="replica_scaling")` routes correctly
- `AdaptiveEWMAReport` dataclass importable
