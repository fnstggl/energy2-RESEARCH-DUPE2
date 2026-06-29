# Canonical State Coverage Audit (Phase 0)

A **normalization-first** audit of every candidate canonical state. The guiding principle (from the PR brief):
*promote existing functionality into canonical ownership; do not duplicate working logic; introduce new state
only where it genuinely does not exist.* So the headline is **how much already exists**, not how many classes
to add. Verdicts are evidence-based with file:line citations (no UNKNOWN).

## Summary verdict

| state | verdict | owner | recommended action |
|--|--|--|--|
| ReplicaState | **FULL** | `world_state.py` (on `CanonicalWorldState.replicas`) | KEEP_AS_IS |
| ServerState | **FULL** | `world_state.py` (`.servers`) | KEEP_AS_IS |
| RackState | **FULL** | `world_state.py` (`.racks`) | KEEP_AS_IS |
| KVState | **FULL** | `ReplicaState.{kv_warm_frac,model_id}` + `world_serving.py` (`_kv_cache`) | KEEP_AS_IS |
| ElectricityState | **FULL** | `electricity.py` + `CanonicalWorldState.electricity_state` | KEEP_AS_IS |
| PowerState | **FULL** | `electricity.py` + `.power_state` | KEEP_AS_IS |
| DeferrableWorkState | **FULL** | `deferrable.py` + `.deferrable_state` | KEEP_AS_IS |
| MigrationState | **FULL** | `world_state.py` + `.migrations` | KEEP_AS_IS |
| WarmState | **FULL** | `world_state.py` + `.warm_state` | KEEP_AS_IS |
| NetworkState | **FULL** (macro; static after init) | `world_state.NetworkPressureState` + `RackState` | KEEP_AS_IS |
| PlacementState | **FULL** at replica granularity; **PARTIAL** per-request | `world_state.PlacementState` + `ReplicaState.{server_id,rack_id}` | EXTEND_NOW (per-request map via RequestState) |
| QualityState | **FULL** (transient, single-source) | `roofline_actions.py` → `PeriodOutcome.quality_sla_risk` | KEEP_AS_IS (promote a record in RooflineState) |
| QueueState | **DUPLICATED_BUT_NOT_CANONICAL** (placeholder ≠ live queue) | `world_state.QueueState` (placeholder) + `unified_replay._State.wait_queue` (authoritative) | EXTEND_NOW (consolidate) |
| CostState | **DUPLICATED_BUT_NOT_CANONICAL** (placeholder ≠ `PeriodOutcome`) | `world_state.CostState` (placeholder) + `PeriodOutcome` (authoritative) | EXTEND_NOW (consolidate or delete placeholder) |
| RooflineState | **DIAGNOSTIC_ONLY** (transient `roofline_diag`) | `roofline.py`/`roofline_actions.py` → `PeriodOutcome.roofline_diag` | EXTEND_NOW (persist a record) |
| DecodeState | **DIAGNOSTIC_ONLY** (transient; 2 classifiers) | `prefill_decode.py` (phase) + `roofline.py` (regime) | KEEP_AS_IS (fold into RooflineState record) |
| RequestState | **ABSENT** as canonical (lifecycle is ephemeral in the replay) | `unified_replay.Job` (transient) | BUILD_NOW (promote the Job lifecycle) |
| ForecastState | **ABSENT** as canonical (belief is transient per decision; no belief-vs-realized) | `forecast_trajectory.py` (transient) | **BUILD_NOW (highest priority)** |

**11 states are already FULL canonical.** The genuine gaps are **ForecastState** and **RequestState**; **QueueState
/ CostState / RooflineState** are present-but-not-canonical (a consolidation/promotion opportunity, not a rebuild).

## Per-state detail

### RequestState — ABSENT (canonical) → BUILD_NOW (promote)
- **Current:** `unified_replay.Job` (`unified_replay.py:74-88`) carries `idx, arrival_s, actual_tokens,
  predicted_tokens, service_s, cls, admit_s, start_s, done_s, deferred_ticks`. Mutated inside
  `run_unified_replay` (admit/start/done); **never persisted** on `CanonicalWorldState` (the replay is
  closed-loop and ephemeral by design).
- **Missing (canonical):** persistent per-request lifecycle (arrived→queued→admitted→dispatched→prefill→decode
  →completed/dropped/missed_sla), placement (replica/gpu/server/rack), deadline/SLA target, reuse keys.
- **Persists across timesteps:** no. **Clones for MPC:** n/a (lives in the replay). **Actions mutate:** the
  serving knobs change dispatch/admission. **Affects reward causally:** yes (goodput = SLA-safe Jobs).
- **Validation:** the replay is tested; no request-conservation invariant on a persistent object.
- **Action: BUILD_NOW** — a canonical, clone-safe `RequestState` **promoted from** the Job lifecycle (a
  snapshot/record, not a re-implementation), with the conservation invariant `arrived = queued + running +
  completed + dropped`. SIMULATOR_INFERENCE for any lifecycle event not in the trace.

### QueueState — DUPLICATED_BUT_NOT_CANONICAL → EXTEND_NOW (consolidate)
- **Current:** `world_state.QueueState` (`world_state.py:139-145`: `pending_requests, per_replica_queue,
  queue_delay_p50/p95/p99`) is a **placeholder** — `per_replica_queue` is never populated; the **authoritative**
  queue is `unified_replay._State.wait_queue` (transient, recomputed each period). `world_simulator` computes the
  delay percentiles post-replay (`world_simulator.py:405-408`) but does not write them back to `QueueState`.
- **Missing:** backlog/arrival-rate/CV estimates, class mix, SLA-slack distribution, forecasted pressure, and
  the persisted percentiles.
- **Action: EXTEND_NOW** — populate the canonical `QueueState` from the realised replay each period (backlog,
  percentiles, class mix, SLA-slack) so it is the single authoritative summary; keep the live heap in the replay
  (do not duplicate the heap into the world state). Resolves the placeholder duplication.

### ForecastState — ABSENT (canonical) → BUILD_NOW (highest priority)
- **Current:** `forecast_trajectory.ForecastTrajectory` (`forecast_trajectory.py:27-74`) is built fresh **per
  decision** from history (`build_trajectory`, causal). The Decision carries the belief snapshot
  (`Decision.forecast`), and `decision_diagnostics` can compute leave-one-out attribution **offline** — but
  **no persistent record of belief-vs-realized exists**; forecast error is never stored.
- **Missing:** a persistent, evolving record of *what the planner believed* (per decision/horizon, with
  provenance/confidence) **and** *what actually happened* (realized) → forecast error → oracle comparison →
  regret attribution.
- **Persists:** no. **Affects reward:** indirectly (belief drives action) — but **ForecastState itself is a
  belief record, NOT a reward term** (no bonus).
- **Action: BUILD_NOW** — a first-class `ForecastState` that **references existing forecaster outputs**
  (no new forecasting model, no duplicated forecast code), persists belief + realized + error + oracle/regret,
  and is the canonical source of planner belief for diagnostics. This is the highest-leverage state for regret.

### PlacementState — FULL (replica) / PARTIAL (per-request) → EXTEND_NOW
- **Current:** `world_state.PlacementState` (`world_state.py:119-126`) summarizes replica→server/rack +
  `rack_spread, locality_score, topology_penalty`; the **master** is `ReplicaState.{server_id, rack_id}`
  (persistent). KV/model residency routing (`world_serving.py`) is causal per request. **No per-request →
  replica → gpu → server → rack record** (that needs RequestState).
- **Action: EXTEND_NOW** — once RequestState exists, record per-request placement on it; keep the replica-level
  master where it is (do not duplicate).

### RooflineState — DIAGNOSTIC_ONLY → EXTEND_NOW (persist a record)
- **Current:** roofline regime / arithmetic intensity / ridge / phase-bound are computed **per period** in
  `roofline.py`/`roofline_actions.py` and recorded transiently as `PeriodOutcome.roofline_diag`
  (`world_simulator.py:85`); the PR #119 roofline timing is promoted behind a flag.
- **Missing:** a persisted per-replica/period record (gpu_type, model, precision, batch, AI, regime, timing
  model, optimal regions).
- **Action: EXTEND_NOW** — a `RooflineState` record snapshotting the already-computed `roofline_diag` per
  period (diagnostic + planning state, **no reward bonus** — actions change it only through precision/clock/
  batch/model). Folds DecodeState's phase classification in (resolves the two-classifier overlap).

### States that are KEEP_AS_IS (already FULL)
ReplicaState, ServerState, RackState, KVState, ElectricityState, PowerState, DeferrableWorkState, MigrationState,
WarmState, NetworkState, QualityState — all stored on `CanonicalWorldState` (or `ReplicaState`), persist across
periods, clone safely (`CanonicalWorldState.clone()` deep-copies), are mutated by `world_simulator._advance` /
`simulate_period` on the chosen action, and affect reward causally through cost/SLA. No rebuild.

## NEEDS_PRODUCTION_TELEMETRY (cannot be made FULL offline)
- **thermal state / true power caps** — no thermal model; the DVFS curve is the only power physics.
- **demand charges** — only day-ahead energy price is modeled.
- **real per-request output length / cache-hit rates** — planning uses a causal prior / synthetic prefixes.
These remain ABSENT-by-necessity and are labelled, never fabricated.

## What this PR builds (minimal, normalization-driven)
1. **ForecastState** (BUILD_NOW) — the highest-priority genuine gap.
2. **RequestState** (BUILD_NOW) — promote the Job lifecycle into a canonical, conservation-checked record.
3. **QueueState consolidation** (EXTEND_NOW) — populate the canonical summary from the realised replay.
4. **RooflineState record** (EXTEND_NOW) — persist the existing `roofline_diag` (folds DecodeState).
5. A **tractable all-knobs checkpointed runner** + bounded validation.

Everything else is KEEP_AS_IS. No state is rebuilt that already exists; the reward path stays byte-identical.
