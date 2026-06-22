# Canonical Optimizer — Phase 1 Parity Report (2026-06-22)

> **Phase 1 of `research/OPTIMIZER_UNIFICATION_PLAN.md`.** Establishes the
> permanent top-level `AureliusOptimizer` interface + decision-layer policy seam,
> with the energy policy as a **thin, behavior-preserving delegate** to the
> existing productized `JobScheduler`. **No runtime behavior changed; no
> serving/SRTF code touched; no benchmark definition changed; no duplicate
> system deleted.** Directional simulator evidence only (`docs/RESULTS.md §8`
> production gate unchanged and unmet).

## What was built (thinnest safe version)
- `aurelius/optimizer/aurelius_optimizer.py` — `AureliusOptimizer` facade. Default
  policy `"energy"`. `optimize(...)` delegates verbatim to `JobScheduler.solve(...)`
  and returns the unchanged `SchedulerResult`. Exposes `.scheduler`, `.policy`,
  `create_baseline_schedule(...)`.
- `aurelius/optimizer/policies.py` — the permanent decision-layer seam:
  - `EnergySchedulingPolicy` — **implemented** (delegates to `JobScheduler`).
  - `ServingQueuePolicy`, `ReplicaScalingPolicy`, `PlacementPolicy`,
    `AdmissionPolicy` — **declared but raise `NotImplementedError`** (Phase ≥2).
    Importable so the architecture exists; using one fails loudly and points at
    the migration plan. Nothing can silently route through an unbuilt policy.
- `aurelius/optimizer/__init__.py` — package exports. Distinct from
  `aurelius.optimization` (the unchanged energy solver it wraps).
- `tests/test_canonical_optimizer_parity.py` — 21 parity/guard tests.

The Forecast / Constraint / Objective / Replay / Evaluation layers of the target
architecture are intentionally **not** built in Phase 1.

## Parity evidence — wrapper == JobScheduler
`tests/test_canonical_optimizer_parity.py`: **21 passed**, ruff clean, mypy clean
on the new package.

| Check | Result |
|---|---|
| Schedule (regions hash) + objective identical across `greedy`, `local_search`, `greedy_migrate`, `greedy_migrate_dp`, `milp` | ✅ identical |
| Reproduces the **pinned** energy-core snapshot (same fixture/constants as `tests/test_energy_core_preservation.py`: realized cost `153.0`, regions hash `6a5a…12e0`) | ✅ identical |
| Reproduces the canonical benchmark's 1000-job scheduler step (schedule hash + objective + ASAP baseline) | ✅ identical |
| Injected-scheduler identity; baseline delegation; ambiguous-arg rejection | ✅ pass |
| `serving_queue` / `replica_scaling` / `placement` / `admission` raise `NotImplementedError`; unknown policy raises `ValueError` | ✅ pass |
| `tests/test_energy_core_preservation.py` (pinned core) still green | ✅ 5 passed |

## 0% KPI drift — canonical energy benchmark
`python scripts/run_canonical_backtests.py --json`, before vs. after this change.
The benchmark is **untouched** (it still constructs `JobScheduler` directly); the
output is **byte-for-byte identical**, and the wrapper is independently proven to
reproduce that scheduler step (so a future switch to the wrapper would be 0%
drift too).

Policy `constraint_aware_with_energy_adapter` (the headline canonical policy):

| KPI | Before | After | Drift |
|---|---:|---:|---|
| `sla_safe_goodput_per_infra_dollar` | 0.337299 | 0.337299 | **0%** |
| `total_infra_cost_usd` | 51725.58 | 51725.58 | **0%** |
| `realized_energy_cost_usd` | 16485.58 | 16485.58 | **0%** |
| `deadline_misses` | 0 | 0 | **0%** |
| `sla_compliant_goodput` | 17447.0 | 17447.0 | **0%** |
| `net_energy_savings_vs_fifo_usd` | 53515.55 | 53515.55 | **0%** |
| `migrations` | 692 | 692 | **0%** |

Full JSON byte-diff: **IDENTICAL**. Matches `aurelius/benchmarks/golden/canonical_energy_backtest.json`.

## What was explicitly NOT done (Phase 1 scope guards)
- ❌ No optimizer rewritten; `aurelius/optimization/scheduler.py` is unmodified.
- ❌ No serving/SRTF code touched (`srtf_serving_backtest.py` untouched; serving
  policy is a stub that raises).
- ❌ No benchmark definition changed (no edits to `benchmarks/`, scenario hashes,
  `economics.py`, or `canonical_backtests.py`).
- ❌ No duplicate system deleted.
- ❌ No full unification attempted.

## Phase 1 merge gate (per task)
| Gate | Status |
|---|---|
| Wrapper delegates to existing `JobScheduler` | ✅ |
| No runtime behavior changes | ✅ |
| Parity tests pass | ✅ (21 + 5 pinned-core) |
| Canonical energy benchmark 0% KPI drift | ✅ (byte-identical) |
| Benchmark definitions unchanged | ✅ |
| `main` verified after merge | _pending merge_ |
