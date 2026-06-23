# SOTSS-GSF Backtest — 2026-06-23

**Status: NULL RESULT — Hypothesis Falsified**
**Five-Failure Rule Counter: 2/5**

## Summary

SOTSS-GSF (Stochastic Oracle SOTSS) replaces the deterministic FIFO oracle in
the SOTSS-MIN fix-up loop with a stochastic Binomial oracle (seed-matched to the
final evaluation). The hypothesis: spot-interruption-vulnerable ticks missed by
the deterministic oracle are detected and fixed by the stochastic oracle, yielding
a cheaper schedule with equal or better safety.

**Hypothesis falsified.** SOTSS-GSF produces the same c_schedule as SOTSS-MIN on
both traces. The stochastic oracle degenerates to the deterministic oracle at
`p_interrupt=10%/hr` because the per-tick survival probability is ~99.82%.

---

## Root Cause

At `p_interrupt_hourly=0.10` and `tick_seconds=60.0`:
```
p_survive = (1 - 0.10)^(60/3600) = 0.90^(1/60) ≈ 0.9982
```

Each spot instance survives each tick with 99.82% probability. The Binomial draw
`survived = Binomial(c_spot, 0.9982)` almost always returns `c_spot`, making
`c_effective ≈ c` in every oracle iteration. Both seeds (42 and 99) produce
virtually identical c_effective schedules, so the oracle sees the same queue
dynamics as the deterministic oracle and converges to the same c_schedule.

The stochastic oracle would only differentiate from the deterministic oracle if
`p_survive << 1` per tick (e.g., `p_interrupt ≥ 50%/hr` at 60s ticks, or at
longer tick durations where more interruptions accumulate per tick).

---

## Results

### Azure LLM 2024

| Metric                    | AMCSG (baseline) | SOTSS-MIN (frontier) | SOTSS-GSF (this run) |
|---------------------------|-----------------|---------------------|---------------------|
| Goodput/$                 | 150,630         | 160,107             | 160,107             |
| vs AMCSG                  | baseline        | +6.29%              | +6.29%              |
| vs SOTSS-MIN              | —               | baseline            | **0.00%**           |
| Cost                      | $4.2800         | $4.0267             | $4.0267             |
| c_mean                    | 4.458           | 4.194               | 4.194               |
| n_sla_safe                | 5823            | 5823                | 5823                |
| Oracle iters              | —               | 34                  | 34                  |
| North-star 500% achieved  | ✓               | ✓                   | ✓                   |
| Safe                      | ✓               | ✓                   | ✓                   |

**Verdict: NEUTRAL** — SOTSS-GSF is identical to SOTSS-MIN. Azure frontier unchanged.

### BurstGPT HF

| Metric                    | AMCSG (baseline) | SOTSS gate=20% (frontier) | SOTSS-MIN gate=100% | SOTSS-GSF gate=100% |
|---------------------------|-----------------|--------------------------|--------------------|--------------------|
| Goodput/$                 | 168,270         | 170,572                  | 178,462            | 178,462            |
| vs AMCSG                  | baseline        | +1.37%                   | +6.06%             | +6.06%             |
| vs SOTSS-MIN gate=100%    | —               | —                        | baseline           | **0.00%**          |
| n_sla_safe                | 5864            | ≥5864                    | 5860               | 5860               |
| Safe                      | ✓               | ✓                        | ✗ (−4)             | ✗ (−4)             |
| North-star 500% achieved  | ✓               | ✓                        | ✗                  | ✗                  |

**Verdict: UNSAFE** — SOTSS-GSF (gate=100%) is 4 requests short of AMCSG safety
criterion. This is the oracle-simulation gap: the oracle uses simplified stochastic
FIFO (c_effective then deterministic queue), while the evaluator uses the full
stochastic FIFO simulation. The same gap affects SOTSS-MIN at gate=100% on BurstGPT.

**BurstGPT frontier remains SOTSS gate=20% at 170,572 goodput/$.**

---

## Same-Conditions Checklist

- ✓ Same trace (5880 req each), same SLA (Azure: 10s, BurstGPT: 30s)
- ✓ Same cost denominator and GPU-hour accounting
- ✓ Same pricing ($0.80 spot, $2.00 OD), same warp scalar
- ✓ Same stochastic simulator (`_simulate_fifo_gsf_spot_fleet`, seed=42)
- ✓ Same telemetry class: actual tick arrival counts, no future-arrival oracle
- ✓ Oracle class: uses actual token counts (valid offline capacity planner)
- ✓ Flows through `AureliusOptimizer(policy="replica_scaling", mode="sotss_gsf")`

---

## What This Rules Out

The experiment falsifies the hypothesis that **spot-interruption detection** is the
binding gap between SOTSS-MIN and the stochastic evaluation. At 10%/hr interruption
rate with 60s ticks, spot interruptions are so rare per tick that the stochastic
oracle degenerates to the deterministic one. The gap on BurstGPT (4 missing requests)
is an oracle-simulation mismatch in queue dynamics, not an interruption detection gap.

---

## Five-Failure Rule

After 2/5 consecutive non-frontier runs (C1PGS, SOTSS-GSF), the rule recommends
focusing on production-readying the existing SOTSS-MIN frontier rather than adding
new modules. Specifically:

1. **Close the oracle-simulation gap on BurstGPT**: SOTSS-MIN gate=100% gives 178,462
   goodput/$ but is 4 requests short. Closing this gap may unlock a new BurstGPT frontier.
2. **Production-ready SOTSS-MIN**: Extract the discipline from the benchmark into a
   serving-runtime path behind `AureliusOptimizer` (canonical architecture per ROADMAP §5).

---

## Implementation Details

- **Policy**: `AureliusOptimizer(policy="replica_scaling", mode="sotss_gsf")`
- **Functions added**: `_oracle_stochastic_response_times`, `compute_sotss_gsf_schedule`
  in `aurelius/optimizer/policies/replica_scaling.py`
- **Benchmark runner**: `run_sotss_gsf_azure_backtest`, `run_sotss_gsf_burstgpt_backtest`
  in `aurelius/benchmarks/srtf_serving_backtest.py`
- **Tests**: 49 tests in `tests/test_sotss_gsf.py` — all passing
