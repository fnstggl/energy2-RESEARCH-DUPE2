# Stochastic Safety Margin OSOTSS Backtest — 2026-06-24

## Summary

**NEGATIVE RESULT.** Stochastic safety margin (`interrupt_safety_margin` ∈ {0, 10, 15, 20, 25, 30})
has zero effect on either canonical public trace. The BurstGPT 15-request SLA gap persists for all
tested margin values. Five-Failure counter advances to **4/5**.

Root cause: the oracle's secondary termination condition (`violators = []` in deterministic FIFO)
fires at n_sla_safe=5849 on BurstGPT before the margin-adjusted convergence threshold is tested.
The oracle has no mechanism to add capacity beyond the "no deterministic violations" floor — the
margin adjusts the convergence threshold but the loop exits via the secondary break before testing it.

## Hypothesis

The BurstGPT 15-request SLA gap arises because the oracle convergence check uses deterministic FIFO
(n_sla_safe counted in no-interruption simulation) while the final evaluation uses stochastic Binomial
interruptions (p_survive≈0.9982/tick). Adding `interrupt_safety_margin` to the oracle's convergence
target should force the oracle to over-provision enough to absorb the expected stochastic capacity
reductions, closing the 15-request gap without future-token access.

## Same-Conditions Checklist

| Dimension | Status |
|---|---|
| Trace | Azure LLM 2024 (5880 req, SLA=10s), BurstGPT HF (5880 req, SLA=30s) |
| SLA definition | Identical: E2E latency ≤ sla_s |
| Cost denominator | GPU-hr ($0.80 spot, $2.00 on-demand) |
| GPU-hour accounting | GSF spot-fleet simulation, seed=42 |
| Decision-time information | Causal EWMA (past observations only) |
| Baseline | OSOTSS margin=0 (default) |
| Candidate | OSOTSS margin ∈ {10, 15, 20, 25, 30} |
| Evaluation method | Stochastic GSF simulation, seed=42 (same as baseline) |

All same-conditions requirements satisfied.

## Results

### Azure LLM 2024 (SLA=10s)

| margin | Goodput/$ | n_sla_safe | vs AMCSG | vs margin=0 | n_iters |
|---|---|---|---|---|---|
| AMCSG gate=12.5% | 150,630 | 5,823 | — | — | — |
| margin=0 | **159,578** | 5,823 | +5.94% | 0.00% | 35 |
| margin=10 | **159,578** | 5,823 | +5.94% | 0.00% | 35 |
| margin=15 | **159,578** | 5,823 | +5.94% | 0.00% | 35 |
| margin=20 | **159,578** | 5,823 | +5.94% | 0.00% | 35 |
| margin=25 | **159,578** | 5,823 | +5.94% | 0.00% | 35 |
| margin=30 | **159,578** | 5,823 | +5.94% | 0.00% | 35 |

**All margin values identical.** No regression on Azure.

### BurstGPT HF (SLA=30s)

| margin | Goodput/$ | n_sla_safe | vs AMCSG n_sla | vs AMCSG gp/$ | vs margin=0 | n_iters |
|---|---|---|---|---|---|---|
| AMCSG gate=12.5% | 168,270 | 5,864 | — | — | — | — |
| margin=0 | **178,109** | 5,849 | −15 | +5.85% | 0.00% | 11 |
| margin=10 | **178,109** | 5,849 | −15 | +5.85% | 0.00% | 11 |
| margin=15 | **178,109** | 5,849 | −15 | +5.85% | 0.00% | 11 |
| margin=20 | **178,109** | 5,849 | −15 | +5.85% | 0.00% | 11 |
| margin=25 | **178,109** | 5,849 | −15 | +5.85% | 0.00% | 11 |
| margin=30 | **178,109** | 5,849 | −15 | +5.85% | 0.00% | 11 |

**All margin values identical.** The BurstGPT 15-request SLA gap persists under all margins.

## Root Cause Analysis — Deepened Diagnosis

### Why the margin has no effect

The oracle loop in `compute_online_sotss_schedule` has two termination conditions:

1. **Primary:** `if n_sla_safe >= baseline_n_sla_safe + interrupt_safety_margin: break` (convergence)
2. **Secondary:** `if not violators: break` (no more violations to fix in predicted/actual FIFO)

On BurstGPT, the oracle reaches deterministic FIFO state where no predicted or actual violations
remain at n_sla_safe=5849 (iteration 11), activating the secondary termination condition. At this
point, the primary convergence check `5849 >= 5864 + margin` is never reached — the loop exits
before testing it.

The oracle has no mechanism to add capacity beyond the "no deterministic violations" floor. The
margin correctly adjusts the primary convergence criterion but the secondary break fires first.

### The structural gap

The oracle achieves n_sla_safe=5849 in deterministic FIFO (no interruptions). AMCSG achieves
n_sla_safe=5864 in stochastic GSF (10%/hr interruptions). The gap is 15 requests.

Counter-intuitively, AMCSG achieves *more* SLA-safe requests in the stochastic evaluation than
OSOTSS achieves in the deterministic oracle. This is because AMCSG uses a fixed higher-c schedule
(gate=12.5% Erlang-C floor) that provides more total server capacity than OSOTSS's optimized
minimum-violation schedule. The extra capacity in AMCSG absorbs stochastic interruptions on
borderline ticks — OSOTSS's leaner schedule leaves those ticks exposed.

### What would be needed

To close the BurstGPT gap, one of the following architectural changes would be needed:

1. **Continue iterating beyond "no violations"**: Remove the secondary `violators=[]` break and
   allow the oracle to add capacity to preemptively buffer borderline ticks. This requires a new
   signal for "which ticks are borderline" — e.g., response times within ε of the SLA.

2. **Minimum c floor from stochastic analysis**: Compute a per-tick c floor from the stochastic
   interruption model and use it as the starting point (instead of the aggressive gate). This
   ensures the oracle never goes below stochastic-safe capacity on heavy ticks.

3. **Higher aggressive gate**: Use a less aggressive starting gate (e.g., gate=25% instead of
   gate=100%) to give the oracle more initial capacity. But this loses the cost savings that
   motivated OSOTSS's aggressive starting gate.

All three options require architectural changes beyond a single parameter sweep.

### Previous hypothesis status

**Revised 2026-06-23**: "Stochastic/deterministic mismatch causes the gap; adding interrupt_safety_margin
to the convergence target will force the oracle to over-provision enough to absorb the interruption
buffer."

**FALSIFIED by this run.** The mismatch diagnosis is correct, but the fix mechanism is wrong. The margin
cannot be leveraged because the oracle's secondary termination condition (`violators=[]`) prevents it from
reaching the margin-adjusted threshold. The oracle cannot over-provision beyond what deterministic FIFO
violations require.

## Implementation

The `interrupt_safety_margin` parameter is correctly wired through the entire call chain:
- `compute_online_sotss_schedule`: `interrupt_safety_margin` param in signature, used in convergence check
- `ReplicaScalingConfig`: `interrupt_safety_margin: int = 0` field with docstring
- `ReplicaScalingPolicy.optimize()`: forwards `cfg.interrupt_safety_margin`
- `run_online_sotss_azure_backtest`, `run_online_sotss_burstgpt_backtest`: parameter forwarded
- `_run_online_sotss_backtest`, `_online_sotss_cost_schedule`: parameter forwarded

Default=0 preserves byte-identical behavior for all existing callers. The infrastructure is correct
and retained; the approach is ineffective.

## Verdict

- **Frontier improvement:** NO (identical results across all margin values)
- **Regression introduced:** NO (margin=0 default is byte-identical to pre-change behavior)
- **Hypothesis:** FALSIFIED (mechanism misdiagnosed — secondary break fires before margin test)
- **Five-Failure counter:** **4/5** (one away from the architectural focus rule)

## Tests

`tests/test_stochastic_safety_margin_backtest.py` — 10 tests:
- `margin=0` is byte-identical to default (regression guard)
- Various margin values run without error
- `ReplicaScalingConfig.interrupt_safety_margin` field wired through
- `AureliusOptimizer(policy="replica_scaling")` routes correctly
- `MarginSweepEntry` and `StochasticSafetyMarginReport` dataclasses importable

## Five-Failure Counter Status

| Run | Result | Counter |
|---|---|---|
| C1PGS (2026-06-23) | NEGATIVE RESULT | 1/5 |
| SOTSS-GSF (2026-06-23) | NULL RESULT | 2/5 |
| Adaptive EWMA (2026-06-24) | NEGATIVE RESULT | 3/5 |
| Stochastic Safety Margin (2026-06-24) | NEGATIVE RESULT | **4/5** |

**One more consecutive negative result will trigger the Five-Failure architectural focus rule.**
Future runs must focus on integration, replay validation, and architecture simplification.
