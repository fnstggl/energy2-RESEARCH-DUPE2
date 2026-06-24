# Joint OSOTSS × Abs-Conformal SRPT Compound Backtest — Run 2026-06-24

**Status: NULL RESULT — Frontier Leaderboard Unchanged**

## Summary

This is a Five-Failure Rule **integration experiment**, combining two existing production-deployed components — the OSOTSS capacity provisioner and the abs-conformal SRPT queue discipline — via a 6-condition 2×3 factorial backtest. The compound hypothesis was **refuted**: conformal SRPT has a **negative interaction** with variable-c capacity scheduling.

**Headline**: conformal+OSOTSS vs conformal+AMCSG = **+4.18% Azure, +2.98% BurstGPT** — but this is the wrong comparison. The correct question is whether conformal+OSOTSS beats **FIFO+OSOTSS** (the existing frontier). It does not:

- Azure: conformal+OSOTSS = 61,262 gp/$ < **FIFO+OSOTSS = 63,831 gp/$** (−4.0%)
- BurstGPT: conformal+OSOTSS = 66,667 gp/$ < **FIFO+OSOTSS = 71,244 gp/$** (−6.4%)

**Frontier leaderboard unchanged**: OSOTSS (= FIFO+OSOTSS) remains at 159,578 gp/$ (Azure, GSF spot-fleet cost model).

## Motivation

The prior joint MCS+conformal backtest (2026-06-23) found a **negative compound** because MCS over-provisioned (c_mean≈4.9 vs fixed-c=4), reducing queue depth and making conformal SRPT less useful. The hypothesis for this run:

> OSOTSS under-provisions vs AMCSG (c_mean≈4.2 vs 4.46, −5.6% GPU-hours), creating deeper queues → conformal SRPT should be more valuable → positive compound.

The hypothesis was **wrong**.

## Results: 6-Condition 2×3 Factorial

### Azure LLM 2024 (SLA=10s, 5880 requests, provisioned cost)

| Condition | goodput/$ | n_sla_safe | cost ($) | vs FIFO+fixed |
|-----------|-----------|-----------|---------|--------------|
| **FIFO + fixed-c=4** | 11,183 | 963 | 9.60 | — |
| FIFO + AMCSG | 60,252 | 5,823 | 10.70 | +438.8% |
| **FIFO + OSOTSS** | **63,831** | **5,823** | 10.10 | **+470.8%** |
| Conformal + fixed-c=4 | 46,199 | 4,995 | 9.60 | +313.1% |
| Conformal + AMCSG | 58,803 | 5,782 | 10.70 | +425.8% |
| **Conformal + OSOTSS** | **61,262** | **5,749** | 10.10 | **+447.8%** |

**Best combination: FIFO+OSOTSS (63,831 gp/$)**

### BurstGPT HF (SLA=30s, 5880 requests, provisioned cost)

| Condition | goodput/$ | n_sla_safe | cost ($) | vs FIFO+fixed |
|-----------|-----------|-----------|---------|--------------|
| **FIFO + fixed-c=4** | 5,534 | 445 | 20.53 | — |
| FIFO + AMCSG | 67,308 | 5,864 | 22.23 | +1116.3% |
| **FIFO + OSOTSS** | **71,244** | **5,849** | 20.90 | **+1187.4%** |
| Conformal + fixed-c=4 | 36,363 | 4,410 | 20.53 | +557.1% |
| Conformal + AMCSG | 64,740 | 5,793 | 22.23 | +1069.9% |
| **Conformal + OSOTSS** | **66,667** | **5,729** | 20.90 | **+1104.7%** |

**Best combination: FIFO+OSOTSS (71,244 gp/$)**

## Key Findings

### 1. Conformal SRPT is NEGATIVE under variable-c capacity scheduling

At fixed-c=4, conformal SRPT is excellent: **+313% over FIFO** on Azure. But with AMCSG or OSOTSS variable-c:

- conformal+AMCSG < FIFO+AMCSG: **−2.4% Azure, −3.8% BurstGPT**
- conformal+OSOTSS < FIFO+OSOTSS: **−4.0% Azure, −6.4% BurstGPT**

This is the opposite of the expected positive compound.

### 2. Conformal SRPT also reduces n_sla_safe under variable-c

This is the most concerning finding. Conformal SRPT does not just fail to improve goodput/$; it **actively reduces SLA-safe request counts**:

- Azure: FIFO+AMCSG=5,823 vs conformal+AMCSG=5,782 (−41 requests)
- Azure: FIFO+OSOTSS=5,823 vs conformal+OSOTSS=5,749 (−74 requests)
- BurstGPT: FIFO+AMCSG=5,864 vs conformal+AMCSG=5,793 (−71 requests)
- BurstGPT: FIFO+OSOTSS=5,849 vs conformal+OSOTSS=5,729 (−120 requests)

### 3. Mechanism: preemption × capacity-change interaction

When variable-c capacity drops at a tick boundary, servers at indices ≥ c(t) complete their current request but do not accept new work. Preempted long jobs (displaced by conformal SRPT at arrival time) now wait in the queue during a capacity-reduced period, accumulating total response times that exceed the SLA budget.

At fixed-c=4, this capacity-drop effect doesn't exist, so conformal SRPT's ordering benefit is pure gain. At variable-c, the capacity oscillation creates starvation pockets that eliminate the ordering benefit.

### 4. The hypothesis about deeper queues was wrong

OSOTSS c_mean=4.208 vs AMCSG c_mean=4.458 (−5.6% capacity). The hypothesis was that fewer servers → deeper queues → more room for conformal SRPT to add value. In practice, the effect is the opposite: fewer servers → more capacity drops → more preemption-starvation interactions. The "deeper queues" are not deeper in a steady-state sense; they're created by bursty capacity reductions that conformal SRPT cannot safely navigate.

## Comparison with Prior Joint Backtest (MCS, 2026-06-23)

| Run | Configuration | Compound? | Root cause |
|-----|--------------|-----------|------------|
| 2026-06-23 | FIFO+MCS vs conformal+MCS | **Negative** | MCS over-provisions (c_mean≈4.9); low queue depth, conformal has little to work with |
| 2026-06-24 | FIFO+OSOTSS vs conformal+OSOTSS | **Negative** | OSOTSS under-provisions (c_mean≈4.2); capacity drops cause preemption starvation |

Both directions of capacity deviation (over and under) produce negative conformal×variable-c compound. This suggests the interaction is **structural** and not tunable by capacity amount: any capacity variability creates preemption starvation that hurts SLA compliance.

## Architecture Insight

The `serving_queue` (conformal SRPT) and `replica_scaling` (OSOTSS/AMCSG) policies in `AureliusOptimizer` occupy **different optimization regimes**:

- `serving_queue` excels when capacity is the bottleneck (queue is always saturated → ordering matters greatly) → fixed-c regime
- `replica_scaling` excels by controlling utilization over time → variable-c regime
- **When `replica_scaling` is active, the system is never persistently saturated**, removing the condition that makes `serving_queue` valuable

This is an important architectural finding: the two policies are **not additively composable** via the current preemptive ordering mechanism.

## Capacity Statistics

| Trace | AMCSG c_mean | OSOTSS c_mean | OSOTSS vs AMCSG cost |
|-------|-------------|--------------|---------------------|
| Azure | 4.458 | 4.208 | −5.61% |
| BurstGPT | 4.331 | 4.071 | −6.00% |

## p99 Response Times

Under conformal SRPT, p99 is **elevated** relative to FIFO (because long jobs get starved):
- Azure conformal+OSOTSS p99 = 13.5s vs FIFO+OSOTSS p99 = 9.9s (SLA=10s)
- BurstGPT conformal+OSOTSS p99 = 45.3s vs FIFO+OSOTSS p99 = 26.8s (SLA=30s)

The p99 EXCEEDS the SLA budget under conformal+variable-c, confirming that conformal SRPT is starving long jobs in a way that violates SLA.

## Classification

**NULL RESULT** — Do not merge as a frontier improvement.

The implementation (benchmark functions + dataclass + test file) is valid research infrastructure and should be merged as documentation. The result clarifies an important architectural constraint: conformal SRPT and variable-c capacity provisioning have a negative interaction that prevents compound optimization.

## Five-Failure Rule Update

This is the **6th consecutive run without a frontier improvement** (all integration work under the already-triggered Five-Failure Rule):

| Run | Type | Result |
|-----|------|--------|
| C1PGS | New module | Fail (−1) |
| SOTSS-GSF | New module | Fail (−2) |
| Adaptive EWMA | New module | Fail (−3) |
| Stochastic Safety Margin | New module | Fail (−4) |
| OSSC | New module | Fail (−5) ← **Five-Failure Rule triggered** |
| Joint MCS+Conformal | Integration | Null result |
| Joint OSOTSS+Conformal | Integration | Null result |

**Direction**: Architecture simplification. The negative conformal×variable-c finding suggests the `serving_queue` policy should be **disabled** (not just non-composed) when `replica_scaling` is active. The two policies should not be combined by default in `AureliusOptimizer`.

## Implementation Notes

New code added to `aurelius/benchmarks/srtf_serving_backtest.py`:
- `JointOSOTSSAbsConformalReport` dataclass (6-condition 2×3 KPIs)
- `_run_joint_osotss_abs_conformal_backtest()` internal helper
- `run_joint_osotss_abs_conformal_azure_backtest()` public function
- `run_joint_osotss_abs_conformal_burstgpt_backtest()` public function

Tests: `tests/test_joint_osotss_abs_conformal_backtest.py` (29 tests, all pass).

All existing functionality unchanged. Five-Failure Rule compliance: no new module added; integration work only (combines existing `compute_mcs_c_schedule`, `compute_online_sotss_schedule`, `_simulate_fifo_variable_c`, `_simulate_abs_conformal_variable_c`).
