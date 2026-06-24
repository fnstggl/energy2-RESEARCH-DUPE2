# Oracle Soft-SLA Continuation (OSSC) Backtest — 2026-06-24

## Summary

**NEGATIVE RESULT.** OSSC (borderline_margin_s sweep ∈ {0.5, 1.0, 2.0, 3.0, 5.0}s) narrows the
BurstGPT 15-request SLA gap to -3 requests at 5.0s margin but never closes it, and causes monotone
goodput/$ regression on Azure at every tested margin.

**Five-Failure Rule TRIGGERED (5/5).**

---

## Hypothesis

After primary oracle convergence (`violators=[]`), identify requests whose deterministic-FIFO
response time is within `borderline_margin_s` seconds of the SLA limit and add capacity to their
ticks. These ticks are most vulnerable to stochastic spot interruptions.

Root motivation: OSOTSS misses AMCSG by 15 requests on BurstGPT (n_sla_safe=5849 vs 5864). The
SSM run (PR #62) confirmed `interrupt_safety_margin` is ineffective — the secondary `violators=[]`
break fires before the primary convergence check. OSSC avoids this by operating as a separate
post-convergence phase on ticks already past primary convergence.

---

## Results

### Azure LLM 2024 (SLA=10s, 5880 requests)

| borderline_margin_s | goodput/$ | n_sla_safe | vs AMCSG | vs baseline | c_mean | iters |
|---------------------|-----------|------------|----------|-------------|--------|-------|
| 0.0 (baseline)      | 159,578   | 5823       | +5.94%   | 0.00%       | 4.21   | 35    |
| 0.5                 | 158,532   | 5823       | +5.25%   | -0.66%      | 4.24   | 37    |
| 1.0                 | 157,499   | 5823       | +4.56%   | -1.30%      | 4.26   | 39    |
| 2.0                 | 153,988   | 5823       | +2.23%   | -3.50%      | 4.36   | 46    |
| 3.0                 | 152,531   | 5823       | +1.26%   | -4.42%      | 4.40   | 49    |
| 5.0                 | 151,101   | 5823       | +0.31%   | -5.31%      | 4.44   | 52    |

### BurstGPT HF (SLA=30s, 5880 requests)

| borderline_margin_s | goodput/$ | n_sla_safe | vs AMCSG (5864) | vs AMCSG gpd | vs baseline | c_mean | iters |
|---------------------|-----------|------------|-----------------|--------------|-------------|--------|-------|
| 0.0 (baseline)      | 178,109   | 5849       | -15             | +5.85%       | 0.00%       | 4.07   | 11    |
| 0.5                 | 177,825   | 5849       | -15             | +5.68%       | -0.16%      | 4.08   | 12    |
| 1.0                 | 177,697   | 5851       | -13             | +5.60%       | -0.23%      | 4.08   | 13    |
| 2.0                 | 176,459   | 5853       | -11             | +4.87%       | -0.93%      | 4.12   | 18    |
| 3.0                 | 176,181   | 5853       | -11             | +4.70%       | -1.08%      | 4.12   | 19    |
| 5.0                 | 175,753   | 5861       | -3              | +4.45%       | -1.32%      | 4.14   | 22    |

---

## Key Findings

1. **OSSC narrows but never closes the BurstGPT gap**: at 5.0s margin, n_sla_safe=5861 (+12 vs
   baseline), still -3 vs AMCSG=5864.

2. **Monotone Azure regression**: every positive margin reduces Azure goodput/$ (-0.66% at 0.5s,
   -5.31% at 5.0s). Azure n_sla_safe remains 5823 throughout (no SLA regression — OSSC only adds
   capacity, never removes it).

3. **No joint frontier**: no single margin achieves both goodput/$≥baseline AND n_sla_safe≥AMCSG
   on both traces simultaneously.

4. **BurstGPT gap is stochastic-structural**: even with 5.0s of borderline pre-provisioning, 3
   requests remain below AMCSG's stochastic n_sla_safe. These are likely on ticks where stochastic
   interruption reduces c_effective below even the OSSC-augmented c_sched.

5. **Cost of closing the gap increases superlinearly**: narrowing from -15 to -3 requires
   increasing borderline_margin_s from 0 to 5.0s (10× extra margin), but only achieves 80% gap
   closure, at cost of -5.31% Azure goodput/$.

---

## Root Cause Analysis

OSSC adds capacity to ticks with borderline response times after primary convergence. The mechanism
works in the direction intended (narrowing the gap) but doesn't converge because:

- The 15 vulnerable requests are on ticks where stochastic interruptions reduce c_effective by 1
- Adding +1 deterministic capacity to those ticks raises c_sched but the probability of a
  Binomial(c_spot, 0.9982) shortfall remains non-zero
- At margin=5.0s, the OSSC phase finds only a few additional borderline ticks to augment (iters
  11→22) because the oracle already eliminated most response-time slack in the primary phase
- The remaining 3 requests at margin=5.0s may be on ticks already at c_ceil (no room to add)
  or on ticks with c_effective failures that 1 extra replica cannot absorb

---

## Five-Failure Rule

This is the 5th consecutive non-frontier-improvement run:
1. **C1PGS** (run 2026-06-23): hypothesis falsified; GSF guard prevents c_effective=0
2. **SOTSS-GSF** (run 2026-06-23): stochastic oracle degenerates to deterministic at p_survive=0.9982
3. **Adaptive EWMA** (run 2026-06-24): oracle ceiling structural, not EWMA error
4. **Stochastic Safety Margin** (run 2026-06-24): secondary violators=[] break fires before primary check
5. **OSSC/Borderline** (run 2026-06-24): narrows gap but never closes; Azure regression

**ARCHITECTURAL FOCUS RULE NOW ACTIVE**: Stop adding new modules. Focus exclusively on:
- Integrating existing modules (energy+serving+replica policy chain)
- Replay validation (cross-trace generalization audit)
- Benchmark realism (arrival model, capacity model, cost model audits)
- Bottleneck diagnosis (what prevents OSOTSS from closing the BurstGPT gap structurally)
- Architecture simplification (reduce code complexity, improve maintainability)

---

## Infrastructure

- `borderline_margin_s=0.0` default: byte-identical to pre-OSSC `compute_online_sotss_schedule`
- `ReplicaScalingConfig.borderline_margin_s: float = 0.0`
- All backtest runners accept `borderline_margin_s` with default 0.0
- 10 tests in `tests/test_borderline_osotss_backtest.py` (all passing)
- Implementation in `compute_online_sotss_schedule` post-convergence OSSC loop (lines ~714-740)
