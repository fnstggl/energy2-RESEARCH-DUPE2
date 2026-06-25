# Joint Optimization — Combination Search (measured interaction) — 2026-06-25

> First increment of the unified joint loop (`aurelius/optimizer/joint.py`,
> `AureliusOptimizer.optimize_joint`). Composes the deployable serving levers on
> ONE trace through one on-demand evaluation and **measures** whether combining
> them compounds. Directional simulator only — not production savings
> (`docs/RESULTS.md` §8). Reproducible: seed + SHA-256 trace-content hash in
> `research/results/joint_combination_search_public_traces.json`.

## The question

"If we combine constraint-aware + energy (or any two optimizers), shouldn't we
get more goodput/$ than each alone?" This run answers it empirically for the
composable **serving** levers (capacity / ordering / admission), on a pure
on-demand denominator (no spot).

## Result — combining does NOT compound (measured)

**Azure LLM 2024 (5,880 reqs · on-demand · seed=42 · trace_hash=c23d19679c3ecdc5):**

| levers | goodput/$ | SLA viol | vs base |
|---|---|---|---|
| base (reactive_lag1 + FIFO) | 59,096.7 | 59 | +0.00% |
| C (forecasted_mcs capacity) | 59,096.7 | 59 | **+0.00%** |
| O (abs-conformal SRPT) | 57,338.9 | 110 | **−2.97%** |
| C+O | 57,139.2 | 116 | −3.31% |
| A (peak-shave admission) | 25,875.8 | 2,582 | **−56.2%** |
| any combo with A | 26–27k | ~2,300+ | −53…−56% |

**Best combination = the base. INTERACTION = SUBSTITUTIVE** — combining does not
beat the best single lever. (BurstGPT's committed fixture is 51 reqs — too small
to read.)

## Why (this is the honest core)

1. **Capacity (forecasting) ties the reactive baseline** (+0.00%). Azure's
   arrival rate is autocorrelated enough that lag-1 ≈ forecast — confirms Phase C.
2. **Ordering (SRPT) is substitutive — it HURTS under good capacity (−3%).** Once
   capacity is sized to meet the 10 s SLA, there is almost no queue left for
   reordering to optimise; the conformal machinery just shuffles an already-met
   queue and loses a little goodput. This reproduces `BENCHMARK_REGISTRY §2A`
   (`conformal+OSOTSS < FIFO+OSOTSS`) on the deployable on-demand regime.
3. **Admission (naive peak-shave) is catastrophic (−56%)** because the public
   trace has **no workload-class labels** — every request is latency-critical, so
   deferring *anything* pushes it past the 10 s SLA. Real flow control only ever
   defers **best-effort/batch** load (`frontier/admission.py` never defers
   latency-critical), which this trace doesn't contain.

## What this means for "combining"

- **Overlapping levers don't compound.** Capacity, ordering, and admission all
  compete for the same queue slack on a single latency-SLA workload — stacking
  them is at best neutral, at worst negative. "More optimizers = more savings" is
  empirically false here.
- **Compounding requires levers on DIFFERENT cost terms**, on workloads that have
  the structure to use them: placement-affinity (numerator: cold-start goodput),
  energy time-shift (denominator: price), utilization-ceiling (denominator:
  GPU-hours). The Azure trace is single-model + all-interactive, so affinity and
  admission have nothing to act on.
- **The blocker is data, not the loop.** The joint loop is built and works; what
  it needs to *find* compounding is a multi-model / multi-class / multi-region
  workload (Mooncake-class data or pilot telemetry), not more serving levers.

## Nothing is wrong

The 0%/negative combination result is correct and honest — it's the expected
outcome of composing **overlapping** levers on a **single-class** workload. The
joint loop's value is precisely that it *measures* this instead of assuming a
combination compounds. Next: combine non-overlapping cost-term levers
(placement + energy + utilization) once a workload/trace (or pilot) carries the
signals to exercise them.
