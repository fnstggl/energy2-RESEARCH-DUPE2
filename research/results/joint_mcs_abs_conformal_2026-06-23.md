# Joint Economic × Queue Compound Backtest — 2026-06-23

## Run Summary

**Date:** 2026-06-23  
**Branch:** `claude/practical-cori-690mw0`  
**Trace:** Azure LLM 2024 (5,880 requests, ρ=0.85, SLA=10s, fixed_c=4)  
**Method:** TRUE compound — MCS per-tick variable-c provisioning + abs-conformal SRTF in a single discrete-event 2×2 factorial simulation (replaces run-z independence estimate)

---

## Full Comparison Table (all conditions, provisioned-hours cost)

**Cost model:** provisioned GPU-hours × $2.00/hr — what the fleet actually costs, not just compute consumed.

| Condition | Goodput/$ | vs FIFO+fixed | vs SLA-aware oracle |
|-----------|-----------|---------------|---------------------|
| FIFO + fixed c=4 | 11,183 | +0% (baseline) | −56% |
| SLA-aware live + fixed | 16,596 | +48% | −34% |
| **SLA-aware oracle + fixed** | **25,208** | **+125%** | **0% (north-star base)** |
| Rel-conformal + fixed | 38,515 | +244% | +53% |
| Abs-conformal + fixed | 46,199 | +313% | +83% |
| Oracle conformal + fixed (ceil) | 47,218 | +322% | +87% |
| FIFO + MCS | 59,694 | +434% | **+137%** |
| **Abs-conformal + MCS (TRUE compound)** | **58,323** | **+422%** | **+131%** |

**North-star threshold: +300% vs SLA-aware oracle = 4× SLA-oracle = 100,832 goodput/$**  
**Status: NOT ACHIEVED. Abs+MCS at 58,323 is 57.8% of threshold. Economic factor still needed: 1.73×.**

---

## GPU-Hours and Capacity Impact

| Fleet | GPU-hours | Cost | vs fixed-c |
|-------|-----------|------|------------|
| Fixed c=4 (always) | 4.80 hr | $9.60 | baseline |
| **MCS variable** | **5.40 hr** | **$10.80** | **+12.5% MORE** |
| Service time consumed (compute only) | 4.02 hr | $8.05 | −16% |

**MCS uses MORE capacity, not less.** On this diurnal trace, MCS scales up to c=8 at peak to maintain SLA — the savings at off-peak (c=1) are outweighed by the peak scaling. Mean replicas = 4.5 vs fixed 4.0. The goodput gain from MCS comes entirely from serving more SLA-compliant requests, not from cost reduction.

---

## Key Structural Findings

### 1. North-star NOT achieved
Abs-conformal+MCS reaches +131% vs SLA-aware oracle. The target is +300%. Factor still needed: 1.73×.

### 2. MCS raises capacity and cost (+12.5%)
The claim that MCS saves money is wrong for this diurnal trace. Fixed c=4 is calibrated for average load (ρ=0.85), but MCS must scale above 4 at peak to maintain the SLA gate — net result is higher provisioned hours. The goodput gain (+131% vs SLA-oracle) is real but comes from better SLA compliance under higher capacity, not from cost savings.

### 3. FIFO+MCS slightly beats Abs+MCS (+137% vs +131%)
When MCS controls queue depth by scaling capacity, the queue is short enough that SRPT ordering provides no benefit. Abs-conformal preemption overhead dominates, making it marginally worse than FIFO in the MCS context. The queue discipline axis only pays off in the fixed-capacity overloaded regime.

### 4. TRUE compound vs independence estimate
- Independence estimate: `gp_abs_fixed × (cost_fixed/cost_mcs)` = 46,199 × 0.889 = 41,066
- TRUE compound: 58,323 (+42% above independence estimate)
- The independence estimate is biased low because it ignores the large goodput numerator increase from MCS scaling

### 5. +422% vs FIFO+fixed is a misleading baseline
The correct comparison is vs SLA-aware oracle (+131%), not vs FIFO+fixed c=4 (+422%). FIFO+fixed c=4 with SLA=10s on a diurnal trace is catastrophically overloaded (p99=732s) — it is the weakest possible baseline.

---

## What the run-z independence estimate got wrong

Run-z applied `economic_cost_factor = 1.2575` (from the weekly provisioning benchmark) to the abs-conformal result. This factor came from the provisioning model using `FALLBACK_TOKENS_PER_S=2500 tok/s` (continuous batching), which predicted c=1 for all ticks in the queue simulation. The corrected physics (Erlang-C with μ = 1/service_time_s) shows MCS requires c=4.5 mean, increasing cost rather than decreasing it.

---

## What needs to happen for north-star

From abs-conformal+MCS at 58,323 to threshold at 100,832:
- Factor needed: 1.73×
- This requires 42% GPU-hour cost reduction (spot/preemptible pricing) on top of the current MCS schedule
- OR 42% improvement in SLA-compliant goodput (e.g., smarter queue discipline that also works in underloaded regime)

---

## Simulation Configuration

```
trace           : azure_llm_2024 (5,880 requests)
fixed_c         : 4 replicas
target_rho      : 0.85
sla_s           : 10.0 s
tick_seconds    : 60.0 s
mcs_gate        : 9.5% Erlang-C timeout rate
n_ticks         : 72
c_schedule      : mean=4.5, min=1, max=8
cost_fixed_c    : $9.60 (4.80 GPU-hr)
cost_mcs_c      : $10.80 (5.40 GPU-hr) — MCS costs MORE
```

---

## Test Suite

15 unit tests in `tests/test_joint_mcs_abs_conformal.py`, all passing on the 200-request fixture.  
Small-sample tolerance (±10%) applied to tests 9 and 11 to account for 200-req calibration artifacts.
