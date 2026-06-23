# Aurelius Canonical Frontier (Phase 4 — Discovery Only)

> **Discovery run. No optimization claim. No new policy, optimizer, benchmark,
> dataset, objective, SLA, pricing, trace, or eval logic changed.** All numbers
> are existing committed/re-run benchmarks, directional simulator only
> (`docs/RESULTS.md §8` production-claim gate unchanged and unmet). This document
> answers: *what is the strongest validated `AureliusOptimizer` configuration
> achievable today using only existing implemented policies?*
>
> Note: Step 1 named `research/OPTIMIZER_ARCHITECTURE.md`; the actual file is
> `research/OPTIMIZER_ARCHITECTURE_AUDIT.md` (read, with
> `CANONICAL_AURELIUS_OPTIMIZER.md` and `OPTIMIZER_UNIFICATION_PLAN.md`).

## Step 1 — Canonical optimizer inventory

`AureliusOptimizer` (`aurelius/optimizer/`) exposes a policy registry of **five**
declared policies; only **two are implemented** (`IMPLEMENTED_POLICIES =
{energy, serving_queue}`). The other three raise `NotImplementedError` and
therefore **cannot be ablated, combined, or validated** through the optimizer.

| Policy | Status | Decision type | Inputs | Outputs | Objective contribution | Replay coverage | Validation |
|---|---|---|---|---|---|---|---|
| **`energy`** (EnergySchedulingPolicy) | **Active** | when/where/how-fast a batch job runs (time-shift, region route, power throttle, migrate) | jobs, DA/RT price, carbon, `OptimizationConfig` | `ScheduleDecision[]` (verbatim `JobScheduler.solve`) | minimizes weighted **cost** (denominator); goodput/$ scored on top | canonical energy backtest (1000 jobs, real CAISO/PJM/ERCOT); routed via `AureliusOptimizer` (Phase 3) | **Validated**: +11.1% gpd/$ vs strongest safe baseline, 0 deadline misses |
| **`serving_queue`** (ServingQueuePolicy) | **Active** | request dispatch/preemption order (Decoupled-Hybrid SRPT + abs-error conformal α) | per-request (arrival, predicted/actual tokens, service_s), servers | `(summary, response, wait)` | improves SLA-compliant goodput (numerator) at **fixed** capacity | abs-conformal serving backtest (Azure 2024 + BurstGPT HF, fixed c=4); routed via `AureliusOptimizer` (Phase 3) | **Validated**: +313%/+557% vs FIFO, **+83% vs SLA-oracle** (Azure, fixed-c) |
| `replica_scaling` | **Stub (NotImplemented)** | per-tick replica count (autoscaling) | — | raises | — | — | not in optimizer (the real provisioning work — SHU/MCS — lives un-routed in `traces/backtest.py`) |
| `placement` | **Stub (NotImplemented)** | GPU/region routing for latency-critical | — | raises | — | — | not in optimizer (shadow `GpuPlacementScorer` exists, **off**) |
| `admission` | **Stub (NotImplemented)** | admit/defer/reject under pressure | — | raises | — | — | not in optimizer (shadow `WorkloadAdmissionGate` exists, **off**) |

**Workload-class split (decisive for composition):** `energy` operates on
**batch jobs over real energy-price traces** (cost objective); `serving_queue`
operates on a **request queue over LLM serving traces** (goodput/$ objective).
They share **no** benchmark, trace, or replay loop.

## Step 5 — Frontier validation (four-way comparison)

Per `docs/RESULTS.md §3`, **FIFO is sanity-only**; the headline comparator is the
strongest *relevant safe* baseline (`current_price_only`/`sla_aware` for energy;
**SLA-aware oracle** for serving). Numbers re-run on current main (`353efd9`).

### Energy world (canonical 1000-job backtest, real prices)
| Configuration | gpd/$ | deadline misses | infra $ | vs strongest-safe baseline |
|---|---:|---:|---:|---|
| FIFO (sanity only) | 0.165781 | 0 | 105,241 | — |
| `current_price_only` (**strongest safe baseline**) | 0.303676 | 0 | 57,453 | 0% |
| `sla_aware` / `greedy_energy` / `robust_energy` | 0.298–0.301 | **119–143 (UNSAFE)** | ~50k | excluded (unsafe) |
| **`energy` policy (CA) = Best AO energy config = Current Main** | **0.337299** | **0** | 51,726 | **+11.1%** |

### Serving world (Azure LLM 2024, fixed c=4, ρ=0.85; vs SLA-aware oracle)
| Configuration | gpd/$ (script basis) | vs FIFO | vs SLA-oracle |
|---|---:|---:|---:|
| FIFO (sanity only) | 13,336 | 0% | −56% |
| SLA-aware oracle (**north-star base**) | (25,208, joint basis) | — | 0% |
| rel-conformal | 45,933 | +244% | +53% |
| **`serving_queue` policy (abs-conformal) = Best AO serving config = Current Main** | **55,097** | **+313%** | **+83%** |
| conformal **oracle** (ceiling; clairvoyant — never headline) | 56,311 | +322% | +87% |

### The four-way verdict
| | Energy | Serving |
|---|---|---|
| **Current Main** | `energy` (CA) 0.337299 | `serving_queue` (abs-conformal) +83% vs oracle |
| **Best historical Aurelius** | same (CA +11%) | **Spot Fleet MCS +304.7%/+381.2% vs SLA-oracle** — but this is **FIFO + provisioning + spot pricing, NOT an `AureliusOptimizer` policy** |
| **Best current AO combination** | — none — | the two policies **cannot compose** (disjoint workloads; other 3 are stubs) |
| **Strongest external baseline** | `current_price_only` (safe) | SLA-aware oracle |

**Strongest validated `AureliusOptimizer` configuration today =** each implemented
policy run on its own workload at its own operating point:
- **`energy`**: +11.1% gpd/$ vs the strongest safe baseline, 0 deadline misses.
- **`serving_queue`**: +83% gpd/$ vs SLA-aware oracle at **fixed capacity** (+313% vs the FIFO sanity baseline).

There is **no combined AO configuration** — see `POLICY_INTERACTION_ANALYSIS.md`.
The repo's strongest *overall* serving result (+304.7%) comes from provisioning +
spot pricing that does **not** run through `AureliusOptimizer` — see
`FRONTIER_RECOMMENDATIONS.md` (claims audit).
