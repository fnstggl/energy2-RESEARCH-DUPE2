# Optimizer Integration Dependency Graph (Phase 5 — Planning Only)

> Planning only. Shows which canonical integrations must precede others. No code
> changed. Directional simulator only.

## Why a dependency graph (not independent integrations)
Phase 4 proved policies do **not** compose independently: `serving_queue` ×
provisioning is **negative/substitutive**, and the biggest lever (cost
denominator / spot pricing) is **un-routed**. So integration order matters: you
cannot honestly measure a *combination* frontier until the replay layer hosts all
deciders on one workload, and you cannot treat spot pricing as a canonical policy
until the objective layer exposes a cost interface.

## Dependency edges (A → B means "A must precede B")

```
                         ┌─────────────────────────────┐
                         │ [DONE] EnergySchedulingPolicy│
                         │ [DONE] ServingQueuePolicy    │
                         │ [DONE] ReplicaScalingPolicy  │  (MCS/SOTSS-MIN +
                         └──────────────┬──────────────┘   GSF/C1PGS schedules)
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                          ▼                          ▼
   ┌────────────────────┐   ┌───────────────────────┐   ┌────────────────────┐
   │ P5.1 ObjectiveLayer│   │ P5.3 Consolidate dup  │   │ P5.2 Route          │
   │ cost interface     │   │ provisioning          │   │ BacktestEngine      │
   │ (spot/preemptible) │   │ (traces ↔ replica_*)  │   │ (energy policy)     │
   └─────────┬──────────┘   └───────────┬───────────┘   └─────────┬──────────┘
             │                          │                         │
             ▼                          │                         │
   ┌────────────────────┐               │                         │
   │ P5.1b Spot policy   │               │                         │
   │ → ReplicaScaling    │               │                         │
   │ (GSF spot mode)     │               │                         │
   └─────────┬──────────┘               │                         │
             └──────────────┬───────────┴─────────────────────────┘
                            ▼
                ┌────────────────────────────┐
                │ P5.4 Unified ReplayLayer    │  ◄── hard prerequisite
                │ (one discrete-event engine) │
                └──────────────┬─────────────┘
                               ▼
                ┌────────────────────────────┐
                │ P5.5 Honest combination     │
                │ search: energy ⊕ serving ⊕  │
                │ replica ⊕ cost(spot)        │
                └──────────────┬─────────────┘
                               ▼
                ┌────────────────────────────┐
                │ P5.6 ConstraintLayer        │  (frontier safe-ρ; SLA gate)
                │ P5.7 Deprecate dead /       │  (independent, can run anytime)
                │      keep shadow            │
                └────────────────────────────┘
```

## Critical-path edges (must-precede), with rationale

| Edge | Why it is required |
|---|---|
| ReplicaScalingPolicy → **ObjectiveLayer cost interface** | Spot pricing changes the **denominator**; it must be a first-class cost input the objective reads, not a benchmark-local overlay, before any spot policy can be "canonical." |
| ObjectiveLayer cost interface → **GSF spot policy canonical** | The GSF/ZFHC/AFMS spot policies only mean something economically once the optimizer's objective can price spot vs on-demand. Spot-fraction decisions (ReplicaScaling) and spot pricing (Objective) are two halves of one lever. |
| ReplicaScalingPolicy → **consolidate trace-replay provisioning** | `traces/backtest.py:_min_cost_safe_replicas` duplicates the now-canonical provisioning; consolidation must follow (not precede) the canonical owner existing. |
| {ObjectiveLayer, ReplicaScaling, consolidation, BacktestEngine routing} → **Unified ReplayLayer** | A single replay engine can only host all deciders once each decider is canonical and the cost interface exists; otherwise unification would freeze in duplicated/benchmark-local logic. |
| **Unified ReplayLayer → honest combination search** | Phase 4 showed combinations are undefined without a shared workload/replay. The negative `ordering × provisioning` interaction and the additive `cost` lever can only be *optimized* (not estimated) once all run in one loop. |
| (independent) **Deprecate dead / keep shadow** | Removing `frontier` EVAL_WORKLOAD/BATCH_INFERENCE and keeping GpuPlacementScorer/admission shadow has **no** dependency — safe anytime; do early to reduce surface. |

## What does NOT depend on anything (can be done in parallel / early)
- **BacktestEngine routing** (energy parity) — independent, low-risk.
- **Deprecate dead frontier families** — independent.
- **ForecastLayer contract** for advisory forecasters — independent, but
  *promotion* of any forecaster to decision-feeding depends on the unified replay
  + a public-replay gate.

## What is blocked until the unified replay exists
- Any **multi-policy combination claim** (energy ⊕ serving ⊕ replica ⊕ cost).
- Promoting **PlacementPolicy** (it must be measured against the real KPI in the
  same loop; today the shadow scorer is harmful, so it stays shadow regardless).
- A single **canonical frontier number** spanning provisioning + ordering + cost.

## One-line takeaway
The **ObjectiveLayer cost interface (spot pricing)** is the highest-value
*unblocked* integration; the **Unified ReplayLayer** is the highest-value
*enabling* integration (it gates all composition). Placement/admission stay
shadow (harmful/neutral); dead frontier families can be deprecated anytime.
