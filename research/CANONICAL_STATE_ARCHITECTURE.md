# Canonical State Architecture (Phase 11)

The canonical world model after this PR: one authoritative owner per evolving piece of state, the new belief +
lifecycle states promoted from existing data (not duplicated), and a clear transition + causal graph. See
`CANONICAL_STATE_COVERAGE_AUDIT.md` for the FULL/PARTIAL/ABSENT classification this builds on.

## The canonical states

| state | owner / module | persists | clones | mutated by | affects reward |
|--|--|--|--|--|--|
| ReplicaState / ServerState / RackState | `world_state.py` (`CanonicalWorldState`) | yes | yes (deepcopy) | `world_simulator._advance` | via cost/SLA |
| KVState | `ReplicaState` + `world_serving.py` (`_kv_cache`) | yes | yes | `simulate_residency_serving` | service time → SLA/cost |
| MigrationState / WarmState | `world_state.py` (`.migrations`/`.warm_state`) | yes | yes | `_advance` | warm-hold/migration cost |
| NetworkState | `world_state.NetworkPressureState` + `RackState` | yes (static) | yes | init only | topology penalty → service time |
| ElectricityState / PowerState | `electricity.py` + `CanonicalWorldState` | yes | yes | `simulate_period` | energy×price → cost |
| DeferrableWorkState | `deferrable.py` + `CanonicalWorldState` | yes | yes | `schedule_deferrable` | separate energy ledger |
| QualityState | `roofline_actions` → `PeriodOutcome.quality_sla_risk` | transient | n/a | precision action | SLA-safe goodput discount |
| **RequestState** (new) | `request_state.RequestLifecycleState` | yes | yes (deepcopy) | `ingest_period` (promoted from replay) | diagnostic (records goodput/SLA outcome) |
| **ForecastState** (new) | `forecast_state.ForecastState` (controller-held) | yes | yes (deepcopy) | `decide` (belief) + `run_period_episode` (realized) | **none — belief record, never a reward term** |
| **RooflineRecord** (new) | `request_state.RooflineRecord` (snapshot of `roofline_diag`) | yes (record) | yes | `from_diag` per period | diagnostic; physics already in the reward |
| QueueState | `world_state.QueueState` (consolidated via `RequestLifecycleState.queue_summary`) | yes | yes | promoted from RequestState | via SLA |

## Transition graph (one timestep)

```
ForecastState.record_belief   (decide: what the planner believes about period p, BEFORE it runs)
        │
        ▼
forecast → action chosen by MPC search (reward = gp/$ over the world rollout; UNCHANGED)
        │
        ▼
world_simulator.simulate_period(period p)         ← the physics (replay + stateful effects)
   ├─ run_unified_replay        → per-Job lifecycle (admit/start/done), KPI, SLA, queue tail
   ├─ roofline/precision/clock   → service time + power  (RooflineRecord snapshot)
   ├─ KV/Warm/Placement/Migration→ service time + capacity
   └─ Electricity/Power          → energy × price → cost
        │
        ▼
_advance(): commit the chosen action to CanonicalWorldState (warm/migration/electricity ledgers, period++)
        │
        ▼
RequestState.ingest_period(p)   (promote the period's requests + realised SLA outcome → conservation)
ForecastState.record_realized(p) (realized arrival/tokens/price → forecast_error; causal, post-hoc)
```

Reward emerges from goodput / cost / SLA exactly as before — **the new states observe and record; they do not
own the physics and do not change the objective or the Pareto gate.**

## Causal graph (what affects what)

- `clock/precision/batch/model` → RooflineState (regime) → service time → SLA + power → cost → gp/$.
- `prewarm/placement/migration` → Warm/Placement/Migration → service time + capacity → SLA + cost.
- `electricity_price_aware` → clock vs the ForecastState price belief → power → cost (N2; PR #118).
- ForecastState **belief→action** is causal, but **ForecastState itself is inert to reward** (a record).
- RequestState is a **sink** (records the outcome); it feeds QueueState's consolidated summary.

## Invariants (validated — `state_validation.py`, `test_canonical_state.py`)

request conservation (`arrived = running + completed + dropped`) · completed-not-queued · placement-ref
validity · ForecastState no-future-leakage (belief before target; error only after realized) ·
forecast-error-correctness · clone isolation. PASS/WARN/FAIL.

## Provenance

- **TRACE_DERIVED:** request identity (Azure/Mooncake arrival/tokens/prompt), electricity price (PJM/ERCOT/CAISO).
- **FORECAST_DERIVED:** ForecastState beliefs (the planner's causal forecaster outputs).
- **SIMULATOR_INFERENCE:** request fine-grained lifecycle timestamps, RooflineState power/regime, DVFS curve,
  deferrable workload.
- **NEEDS_PRODUCTION_TELEMETRY:** thermal/power-caps, demand charges, real per-request output length, real
  cache-hit rates — ABSENT, labelled, never fabricated.

## What was ported / referenced (not imported as deps)

- **Request lifecycle / queue discipline** — the LLMServingSim / BLIS / vLLM/Orca request-state + queue patterns
  informed `RequestState`'s lifecycle stages and the conservation invariant (we record the existing replay's
  lifecycle rather than re-implementing a scheduler).
- **Roofline state** — InferSim / llm-analysis / LLM-Viewer arithmetic-intensity/ridge concepts are already in
  `roofline.py`/`roofline_external.py` (PR #110/#119); `RooflineRecord` snapshots them.
- **Forecast belief-vs-realized** — standard MPC practice (record belief, measure realized error); no external
  sim needed.
- **V2** is audit-doc + the #119 roofline-timing promotion only (no `aurelius/environment/v2/` package on this
  branch); the all-knobs runner marks a `v2_reference` arm SKIPPED_TOO_HEAVY rather than inventing one.

## What remains PARTIAL / ABSENT (honest)

- PlacementState per-request mapping is recorded on RequestState but not yet driven by a per-request router in
  the canonical path (EXTEND-after).
- CostState placeholder on `CanonicalWorldState` is still superseded by `PeriodOutcome` (consolidate or delete
  in a follow-up — low risk, no reward effect).
- thermal/power-cap/demand-charge states remain ABSENT (NEEDS_PRODUCTION_TELEMETRY).
