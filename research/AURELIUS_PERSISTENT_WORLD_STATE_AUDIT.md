# Aurelius Persistent World-State Audit (post-PR #100)

PRs #99/#100 connected routing, capacity_multiplier and batching. They also proved the next
bottleneck is **not forecasting** — it is **missing persistent simulator state**. Every action
deferred so far (prewarming, placement/topology, migration, energy-shifting) needs the world
model to remember *which* replica is warm, *where* it sits, and *what* a move costs. Today the
simulator has none of that: capacity is a scalar pool, each period starts from scratch.

This audit records exactly what state exists, what is missing, and which **public-trace
artifact** can calibrate each missing piece — so Phase 2 builds only honestly-calibratable state
and Phases 4–6 connect only actions that the new state makes real.

## How the simulator represents the world TODAY

- **`run_unified_replay`** (`aurelius/optimizer/unified_replay.py`): a discrete-event token
  serving loop whose entire fleet state is `_State.c` — an **integer pooled replica count** — plus
  ephemeral `wait_queue` / `defer_buffer`. Server slots `sid` exist only within a period; **no
  server identity, no rack, no warm/cold, no cross-period memory.**
- **`run_period_episode`** (`controller.py`): loops periods, calls `run_unified_replay` fresh each
  period. **Nothing persists hour→hour.**
- **`FleetState`** (`fleet_plane_v2026.py`, `schemas.py`): immutable hourly **marginals** —
  `gpu_type_mix`, `rack_locality = {asw_id: gpu_count}` (aggregate), `net_pressure` (one 0..1
  scalar), `capacity_envelope`, `util_target`, `mem_pressure`. Real, trace-derived, but **macro,
  not per-server**.
- **`StatefulKVCache`** (`kv_cache.py`): genuinely per-server stateful **within** a period (LRU
  paged cache), but **reset at the period boundary**.
- **`CostModel.operator_cost`** (`cost_model.py`): already *accepts* `migrations`, `egress_gb`,
  `queue_delay_s` and prices them — but the simulator **never passes non-zero** values (the hooks
  exist, nothing drives them).

## State-object table

Fidelity tags: **TRACE_EXACT** (a FULL_TRACE_EXACT v2026 marginal), **TRACE_DERIVED_SAMPLE** (a
representative cluster sampled from trace marginals — preserves distributions, not real identities),
**BENCHMARK_DERIVED** (public-prior magnitude, not from our trace), **INFERRED/HEURISTIC**, **ABSENT**.

| STATE | EXISTS TODAY? | PERSISTENT? | TRACE SOURCE (for build) | ACTIONS ENABLED | MISSING PIECES |
|---|---|---|---|---|---|
| **CanonicalWorldState** | NO | — | composed below | all stateful | the cross-period container itself |
| **ServerState** | NO (only scalar `c`) | — | v2026 `server_hourly`: `gpu_type`(13 frac), `gpu_count`(μ4.1, 1–16), `cluster`(17) → **TRACE_DERIVED_SAMPLE** | placement, migration | per-server identity + slot inventory |
| **RackState / ASW topology** | MARGINAL (`rack_locality` aggregate) | NO | v2026 `asw_locality` (2389 ASWs; **55% unlabeled "None"**, rest sparse μ≈9e-5) → **TRACE_DERIVED_SAMPLE** | placement, topology | per-server→rack map; cross-rack penalty model |
| **GPUState** | NO (homogeneous slots) | — | `gpu_type` frac (TRACE_EXACT) | placement (capacity/cost by type) | per-GPU residency; not needed beyond type/count |
| **ReplicaState** | NO (ephemeral `sid`) | — | derived from ServerState | prewarm, migration, placement | warm flag, last_used, cold-start remaining, home server |
| **WarmState** | NO (cold-start = 0) | — | cold-start magnitude **BENCHMARK_DERIVED** (tens of s), order-checked vs `ready_delay_s` | **prewarm** | warm pool, cold→ready lag, warm-hold accounting |
| **PlacementState** | NO | — | ServerState + RackState | **placement/topology** | replica→server map; locality/spread score |
| **MigrationState** | cost hook only (always 0) | NO | move cost **BENCHMARK_DERIVED**; benefit from RackState | **migration** | in-flight move state; duration; cap-loss; cache-invalidation |
| **QueueState** | YES (ephemeral) | NO (reset/period) | synthetic from job stream | admission (connected) | optional cross-period carryover (not required here) |
| **KVState** | YES (per-server, per-period) | NO | Mooncake reuse (TRACE_DERIVED) | routing (connected) | per-replica persistence + invalidation-on-migrate |
| **NetworkPressureState** | MARGINAL (`net_pressure` scalar) | NO | v2026 `network_hourly` `rx/tx_gibps` (TRACE_EXACT; μ0.13/0.07, max 922/674) | topology | per-rack pressure (macro only; **per-link ABSENT**) |
| **CostState** | per-period rollup | NO | `cost_model` (+ warm-hold/migration extensions) | all (reward) | warm-hold + migration + topology-penalty terms wired live |
| **PowerState** | static service time | NO | — | clock/DVFS (**not in scope**) | power-vs-perf curve — **left ABSENT/PLANNED** |
| **DeferrableWorkState** | class tag only | NO | `job_type_public` (online/offline) | energy-shift (**deferred**) | cross-period deferral window + deadline model |

## What this enables — and the honest limits

**Cold-start magnitude.** The v2026 `pod_hourly` `ready_delay_s` / `schedule_delay_s` are
FULL_TRACE_EXACT (μ 347s / 314s) — but those means **conflate batch-job scheduling queues** with
container/model load, so they are an **upper bound, not a serving cold-start**. We therefore set
the serving cold-start penalty as **BENCHMARK_DERIVED** (tens of seconds, the vLLM/TGI model-load
regime) and use the trace only to confirm order-of-magnitude. Labeled honestly; never called
"measured serving cold-start."

**Topology is real but thin.** `asw_locality` exists (2389 ASWs) but **55% of GPUs carry no ASW
label** and the labeled racks are individually tiny. So per-rack structure is **sparse**: we build
a representative **TRACE_DERIVED_SAMPLE** cluster preserving the gpu_type / gpu_count / rack-
concentration marginals, and treat the **macro `net_pressure`** (TRACE_EXACT) as the dominant
topology signal. We claim **macro** network/locality penalties only — **no NVLink, NVSwitch,
PFC/ECN, per-link congestion, or hardware-health** (all ABSENT).

**Migration cost** has no trace source (operator move reasons are pilot-only), so its
cost/duration are **BENCHMARK_DERIVED** and it stays behind the Pareto gate; if it cannot move
the simulator with non-zero cost honestly, it stays **PLANNED**.

## Design decision for Phase 2

Build a **persistent `CanonicalWorldState`** holding a **TRACE_DERIVED_SAMPLE** cluster (a few
dozen `ServerState` across a handful of `RackState`, sampled from the v2026 marginals), with
`ReplicaState` (warm/cold, home server) living on servers, plus `MigrationState`, `PlacementState`,
`WarmState`, `QueueState`, `NetworkPressureState`, `CostState`. The state **persists across
periods** and is **cloned per candidate** so MPC search never contaminates the real timeline. The
existing stateless `run_unified_replay` path is preserved unchanged; the world-state path adds the
stateful penalties (cold-start, topology, migration) on top of the same serving economics, and
collapses to the stateless result when all stateful knobs are at their no-op.

Not built (honest deferral): `PowerState`/clock (no curve), `DeferrableWorkState`/energy-shift
(needs cross-period deferral semantics + deadline model — recommended as the *next* batch).
