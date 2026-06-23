# Fair MCS Comparison — Does Aurelius Add Value After MCS? — 2026-06-23

## Question

Once the baseline already has MCS-style per-tick capacity scaling, does
Aurelius's queue intelligence (abs-conformal SRTF) add value over a strong
SLA-aware baseline?

## Method — controlled experiment

All conditions use the **same trace** (Azure LLM 2024, 5,880 req), **same SLA**
(10s), **same physics** (TPOT_S=0.020 s/tok sequential decoding), **same
arrival process** (warp-calibrated to ρ=0.85 on fixed c=4), and **same cost
denominator** (provisioned GPU-hours × $2.00/hr).

The MCS conditions all receive the **identical** per-tick capacity schedule
(`_joint_mcs_c_schedule`, mean=4.5, range [1,8], 72 ticks). Holding capacity,
cost, and GPU-hours constant isolates the **queue-ordering** contribution.

## Results

| Policy | gp/$ | SLA tok | tok% | GPU-hr | cost | qP50 | qP95 | qP99 | viol | preempt |
|--------|------|---------|------|--------|------|------|------|------|------|---------|
| FIFO + fixed c=4 | 11,183 | 107,353 | 16% | 4.80 | $9.60 | 340.6 | 719.9 | 730.3 | 4,917 | 0 |
| SLA-aware + fixed c=4 | 16,596 | 159,326 | 23% | 4.80 | $9.60 | 122.7 | 1,615.7 | 1,683.9 | 4,413 | 0 |
| SLA-aware + MCS | 59,676 | 644,499 | 95% | 5.40 | $10.80 | 0.00 | 1.2 | 2.5 | 58 | 0 |
| FIFO + MCS (= constraint-aware + MCS) | 59,694 | 644,696 | 95% | 5.40 | $10.80 | 0.00 | 1.3 | 2.3 | 57 | 0 |
| **Abs-conformal + MCS (Aurelius)** | **58,323** | **629,888** | **93%** | **5.40** | **$10.80** | **0.00** | **0.8** | **3.2** | **96** | **1,228** |
| SLA-aware **oracle** + MCS (strongest) | 59,661 | 644,344 | 95% | 5.40 | $10.80 | 0.00 | — | — | 58 | 0 |
| Abs-conformal **oracle** + MCS (best candidate) | 58,323 | 629,888 | 93% | 5.40 | $10.80 | 0.00 | 0.8 | 3.2 | 96 | 1,228 |

Runtime: all conditions < 0.2s on the 5,880-req trace.

### A note on "constraint-aware + MCS" and "Aurelius main + MCS"

"Constraint-aware" and "safe_high_utilization" are **provisioning policies**
(how many replicas), not **queue disciplines** (what order to serve). They live
in `aurelius/traces/backtest.py` and run on continuous-batching physics
(2,500 tok/s/replica), incompatible with the sequential-decoding queue
simulation. When capacity is fixed to the MCS schedule, "constraint-aware + MCS"
has no distinct queue-ordering meaning and collapses to **FIFO + MCS**. The
Aurelius serving headline IS abs-conformal SRTF, so "Aurelius main + MCS" =
"best Aurelius candidate + MCS" = abs-conformal + MCS.

## Decomposition — where every point of gain comes from

Baseline: FIFO + fixed c=4 = 11,183 gp/$.

| Lever | From → To | Δ gp/$ | Source |
|-------|-----------|--------|--------|
| **Queue ordering** (at fixed cap) | FIFO+fixed → SLA-aware+fixed | **+48.4%** | reorder short-first; more SLA-compliant tokens at same cost |
| **MCS capacity** | SLA-aware+fixed → SLA-aware+MCS | **+259.6%** | +501% SLA-compliant tokens (queue drains), −12.5% offset from higher cost |
| **Aurelius ordering on top of MCS** | SLA-aware+MCS → abs-conformal+MCS | **−2.27%** | preemption overhead with no queue to optimize |
| **Lower cost** | — | **0% (negative)** | MCS costs +12.5% MORE GPU-hours, not less |
| **Fewer SLA misses (Aurelius vs baseline at MCS)** | 58 → 96 violations | **WORSE** | abs-conformal misses MORE at MCS capacity |

### Why Aurelius loses at MCS capacity

MCS sizes the fleet so the queue barely forms: median queue wait = **0.00s**,
p99 = 2–3s. With essentially no queue, there is nothing to reorder — FIFO+MCS
already captures 94.8% of tokens. Abs-conformal's preemptive SRPT then performs
1,228 preemptions that add overhead and push a handful of near-complete long
requests past the 10s SLA (96 violations vs 58), costing **−2.27%**. Prediction
quality is irrelevant: live-prior and oracle-prior abs-conformal both land at
exactly 58,323 gp/$ — the loss is structural (preemption), not predictive.

## Conclusion

**Does Aurelius improve over a strong MCS-enabled SLA-aware baseline?**

**No.** At equal capacity, cost, GPU-hours, SLA, physics, and arrival process,
Aurelius (abs-conformal + MCS) scores **58,323 gp/$ vs SLA-aware + MCS at
59,676 — a delta of −2.27%** (−2.24% vs the oracle SLA-aware baseline). Even the
strongest Aurelius candidate (oracle-prior abs-conformal) does not beat it.

The compound gain previously attributed to "Aurelius + MCS" comes almost
entirely from **MCS capacity scaling (+260%)**, not from Aurelius's queue
ordering. Aurelius's queue intelligence is **regime-dependent**: it delivers
+48% over FIFO when capacity is fixed and the queue is deep, but once MCS
supplies enough capacity to drain the queue, ordering becomes a no-op and the
preemptive variant is a slight net negative.

**North-star (+300% vs SLA-aware + MCS): NOT ACHIEVED.** Aurelius is below the
SLA-aware + MCS baseline, let alone +300% above it.

## Implication for the roadmap

1. The headline serving result (+313%/+557% abs-conformal vs FIFO) is real but
   only at **fixed, constrained capacity**. It is not additive with capacity
   scaling — the two levers target the same bottleneck (SLA-compliant goodput
   under load) and capacity scaling dominates.
2. A deployable system should pick ONE lever per regime: **abs-conformal when
   capacity is hard-capped**, **MCS capacity scaling when elastic**. Running both
   wastes preemption overhead.
3. To beat SLA-aware + MCS, Aurelius needs a lever that operates where MCS
   cannot: **cost reduction** (spot/preemptible pricing — MCS currently costs
   +12.5% more), not more queue ordering.
