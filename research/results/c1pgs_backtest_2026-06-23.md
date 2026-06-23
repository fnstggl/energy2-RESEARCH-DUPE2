# C1PGS Backtest — run 2026-06-23

## Algorithm: C1-Protected Gate Sweep (C1PGS)

**Hypothesis**: At Erlang-C gate=25%, protecting c=1 ticks with on-demand instances (0 spot) eliminates the BurstGPT spot-interruption cliff and yields higher goodput/$ than AMCSG gate=12.5% while maintaining SLA safety.

**Verdict: FALSIFIED**

---

## Results

### Azure LLM 2024 (5,880 requests, SLA=10s)

| Metric | AMCSG gate=12.5% | C1PGS gate=25% | Delta |
|---|---|---|---|
| goodput/$ | 150,629.91 | 153,960.38 | **+2.21%** |
| cost ($) | 4.2800 | 4.1733 | -2.49% |
| c_mean | 4.458 | 4.306 | -0.152 |
| n_sla_safe | 5,823 | 5,818 | **-5** |
| p99 (s) | 9.946 | 10.12 | +0.17s |
| n_ticks_c1 | — | 2 | — |
| SLA-safe? | ✓ baseline | **✗ FAILS** | — |
| North-star? | ✓ yes | **✗ NO** | — |

C1PGS Azure: raw goodput/$ is higher (+2.21%) but **5 more SLA violations**. Gate=25% violates the SLA safety criterion (n_sla_safe < AMCSG baseline). Not a valid product improvement.

---

### BurstGPT HF (5,880 requests, SLA=30s)

| Metric | AMCSG gate=12.5% | C1PGS gate=25% | Delta |
|---|---|---|---|
| goodput/$ | 168,269.98 | 155,786.06 | **-7.42%** |
| cost ($) | 8.8933 | 9.5867 | **+7.80%** |
| c_mean | 4.331 | 4.221 | -0.110 |
| n_sla_safe | 5,864 | 5,859 | **-5** |
| p99 (s) | 22.918 | 23.89 | +0.97s |
| n_ticks_c1 | — | 46 | — |
| SLA-safe? | ✓ baseline | **✗ FAILS** | — |
| North-star? | ✓ yes | **✗ NO** | — |

C1PGS BurstGPT: **worse on every metric**. More expensive, lower goodput/$, fewer SLA-safe requests.

---

## Root Cause Analysis

### Why the mechanism was wrong

The pre-implementation hypothesis stated:
> "BurstGPT at gate=25% gets 3-4 SLA violations from spot interruptions at c=1 ticks (c_effective=0)."

This was incorrect. The simulation contains a numerical guard:
```python
c_effective.append(max(1, c_demand + survived))
```

This guard prevents `c_effective=0` in **all cases**, including c=1 spot ticks. With the guard active:
- GSF at c=1 (c_spot=1): `c_effective = max(1, 0 + survived) = 1` always
- C1PGS at c=1 (c_spot=0, c_demand=1): `c_effective = max(1, 1 + 0) = 1` always

Both produce **identical effective capacity** at c=1 ticks. The C1PGS spot-allocation change has no effect on SLA safety.

The SLA violations at gate=25% come from **Erlang-C over-optimism** (the M/M/c model is too optimistic vs the actual M/G/c load), not from spot interruptions.

### Why BurstGPT costs MORE with C1PGS

C1PGS at gate=25% uses on-demand at c=1 ticks ($2.00/hr). With BurstGPT's SLA=30s budget (vs Azure's 10s), Erlang-C at gate=12.5% assigns c=2 on the same ticks (all-spot, $1.60/hr). On-demand is MORE expensive than 2 spot instances:

```
AMCSG gate=12.5% c=2 all-spot: 2 × $0.80 = $1.60/hr
C1PGS gate=25%   c=1 on-demand: 1 × $2.00 = $2.00/hr  ← more expensive!
```

46 ticks × ($2.00 − $1.60) × (60/3600) hr ≈ $0.31 extra from c=1 OD premium, compounded by other differences in the schedule.

---

## Constitutional Classification

Per the Aurelius constitutional rules:
- C1PGS does **NOT** beat the strongest fair baseline (AMCSG) on either trace
- SLA safety is **NOT** preserved (5 more violations on both traces)
- **Not merged** as a product claim

The code remains in the codebase as research infrastructure (`_c1pgs_spot_replicas`, `_simulate_fifo_c1pgs_spot_fleet`, `run_c1pgs_*_backtest`). The canonical `compute_c1pgs_spot_replicas` is retained in `replica_scaling.py` for completeness.

---

## Next Directions

Given this falsification, the next candidates from GAP_ANALYSIS are:

1. **SOTSS-MIN tuning** — SOTSS-MIN gate=100% already achieves 160,107 goodput/$ on Azure (+6.29% vs AMCSG). Investigate whether SOTSS-MIN can close the 5-request SLA gap on Azure with fewer oracle iterations.

2. **BurstGPT SOTSS-MIN** — SOTSS-MIN on BurstGPT achieves 170,572 (+1.37% vs AMCSG). Both traces are above north-star. Focus may shift to production integration (ReplicaScalingPolicy convergence) rather than further algorithm search.

3. **ShareGPT cross-validation** — Validate SOTSS-MIN on a third public trace to confirm generalization.

The Five-Failure Rule counter: This is the **1st consecutive run** without a Frontier Improvement (SOTSS-MIN from run 2026-06-23 was the last frontier). Still 4 more attempts before triggering the "stop adding modules" rule.
