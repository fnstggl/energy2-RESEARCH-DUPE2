# Aurelius Persistent World State + Stateful Actions (prewarm / placement / migration)

PRs #99/#100 connected routing, capacity_multiplier and batching on the **stateless** serving loop
and proved the next bottleneck was not forecasting — it was **missing persistent simulator state**.
This work builds that state and connects the three actions it makes honestly simulatable.

## Why these actions needed persistent world state

The stateless loop (`run_unified_replay`) represents the fleet as a scalar replica pool `c`, rebuilt
every period. You cannot score prewarming without knowing *which* replica is already warm; you
cannot score placement without knowing *which rack* a replica sits on; you cannot score migration
without an in-flight move that costs now and pays later. All three need state that **persists across
periods** and **evolves with the chosen action** — see
`research/AURELIUS_PERSISTENT_WORLD_STATE_AUDIT.md` for the per-state-object gap analysis.

## What state was added (`aurelius/environment/world_state.py`)

A typed, persistent `CanonicalWorldState`:

- **ServerState / RackState** — a representative cluster of GPU servers across racks, **sampled from
  the v2026 `server_hourly` marginals** (gpu_type fractions, gpu_count distribution, ASW locality)
  and `network_hourly` rx/tx for per-rack macro pressure.
- **ReplicaState** — replicas with a home server/rack, a warm/cold flag, last-used period, and
  cold-start budget. **WarmState / PlacementState / MigrationState / QueueState /
  NetworkPressureState / CostState** carry the per-period ledgers.
- `clone()` for candidate isolation; the whole state **persists period→period** and advances once for
  the chosen action.

## What is trace-derived vs inferred (fidelity, stated honestly)

| Piece | Fidelity | Source |
|---|---|---|
| gpu_type / gpu_count / ASW locality of the cluster | **TRACE_DERIVED_SAMPLE** | v2026 `server_hourly` calibration (FULL_TRACE_EXACT marginals) — preserves DISTRIBUTIONS, not machine identities |
| per-rack macro network pressure | **TRACE_EXACT (macro)** | v2026 `network_hourly` rx/tx means |
| cold-start magnitude (`COLD_START_S=30s`) | **BENCHMARK_DERIVED** | vLLM/TGI model-load regime; order-checked vs the v2026 `ready_delay_s` distribution (whose mean conflates batch-queue waits, so it is an upper bound, not a serving cold start) |
| warm-hold GPU-hours, migration cost/duration, topology discount magnitude | **BENCHMARK_DERIVED / INFERRED** | public-prior magnitudes, sanity-banded |
| per-link / NVLink / NVSwitch / PFC-ECN / congestion / hardware health | **ABSENT** | not in any public trace — **not claimed** |

The cluster is a **representative sample for relative (policy-vs-policy) scoring**, not an inventory
reconstruction (the trace does not support one: `asw_locality` is 55% unlabelled). Labelled
TRACE_DERIVED_SAMPLE throughout; never laundered as measured per-machine telemetry.

## What was connected (`world_simulator.py`) — and how each pays out

All three layer **on top of** the proven serving engine via a per-period service-time factor, a
warm-capacity ramp, and extra cost terms; with every knob at its no-op the result is **bit-for-bit
the stateless PR-#100 path** (verified).

- **prewarm_policy** {off, conservative, aggressive} — a cold-start ramp: cold replicas cannot serve
  for `cold_start_s`, so a period needing more replicas than are warm eats a warm-up queue spike.
  Prewarming warms ahead from the forecast (paying warm-hold GPU-hours, wasted if load doesn't
  arrive). Reactive `off` cools to actual usage → no warm-hold floor.
- **placement_policy** {topology_blind, rack_local, network_aware} — a **macro** service-time
  discount for exploiting rack locality + low network pressure (the same channel KV routing uses).
  `topology_blind` = no discount (the baseline); the discount vanishes when racks share pressure
  (no free lunch).
- **migration_policy** {off, conservative, aggressive} — a live move: operator cost + capacity loss
  + KV cache invalidation **this** period, a locality benefit only **after** it lands next period
  (persistent `MigrationState`). Never a free win.

## Simulator effect per action (direct-effect tests, `tests/test_world_state.py`)

Each effect is proven in a controlled fixture, in the right regime, deterministically:

- **prewarm** reduces cold-start events (e.g. 35→9 under a heavy forecast) and lifts gp/$ when the
  load materialises; but with a wrong (too-heavy) forecast it pays warm-hold and **loses gp/$**.
- **placement** `network_aware` drops the topology factor below 1.0 and changes gp/$; a flat-pressure
  cluster yields ~no discount.
- **migration** starts in-flight moves with non-zero cost + a cache-invalidation service penalty,
  **hurting** the single period; the move persists and lands the next period.

## Fair baselines

- prewarm: reactive `off` (no prewarm) vs `prewarm_always` (aggressive) — the upper bound that
  exposes warm-hold cost.
- placement: `topology_blind` vs `network_aware`; `world_static_best` is a competent static operator
  that **already** places topology-aware (so the MPC must beat adaptation, not lever access).
- migration: `off` (no migration) vs the MPC's adaptive use.

## Incremental held-out results (Azure 2024 week, persistent world)

<!-- WS_INCREMENT_TABLE -->

## Final fair backtest (Azure 2024 week, persistent world)

<!-- WS_FULL_TABLE -->

## Safe vs unsafe claims

**Safe (supported):** *Aurelius now has a persistent canonical world state with server, rack,
replica, warm/cold, placement, and migration state, sampled from public v2026 trace marginals. The
MPC can simulate and optimise prewarming, topology-aware placement, and migration as stateful
infrastructure actions behind a Pareto-aware claim gate, with each action's effect proven in tests
and each able to HURT (so none is a fake knob).*

**Unsafe (NOT claimed):** that Aurelius has production telemetry for live placement, warm state, or
operator migration decisions; that the cluster is a real machine inventory; or that any per-link /
NVLink / micro-congestion behaviour is modelled. All numbers are **SIMULATED directional evidence
on a TRACE_DERIVED_SAMPLE cluster**, not production telemetry.

## Next recommended action

Energy-aware temporal shifting of deferrable (best-effort) work — it needs the one piece this batch
did not build: cross-period **DeferrableWorkState** (a deferral window + deadline model). The price
is already in the objective and best-effort admission already exists, so it is the smallest honest
next step.
