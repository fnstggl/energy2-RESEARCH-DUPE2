# Aurelius Simulatable-Action Connection Audit (post-PR #98)

PR #98 built the canonical action-surface architecture (typed `ActionBundle` + registry).
This audit decides, for every SIMULATED_ONLY / PLANNED surface, whether the canonical
environment has **enough state to model its economic effect honestly today**, and connects
the ones that do. The rule is strict: *a surface becomes CONNECTED only when changing it
alters the simulated reward in a causally defensible way that the simulator can score* — no
fake knobs.

## The data constraint that shapes everything

The serving reward path (`run_unified_replay`) operates on Azure `Job`s with
`(arrival_s, tokens, service_s, class)`. **Azure jobs carry no per-request prefix hashes** —
the prefix-reuse dynamic comes from a *separate* trace (Mooncake `hash_ids`). So an action
that needs per-request prefixes to route *inside the Azure loop* cannot be honestly simulated
there. KV-aware routing is connectable, but through the **fleet channel**: replay the Mooncake
reuse trace across N server caches under a routing policy → fleet prefix-reuse depth → a
service-time discount that feeds the Azure economics. That is causal, validatable (Mooncake
held-out), and real.

## Decision table

| ACTION | CURRENT STATUS | CAN SIMULATE TODAY? | CONNECT NOW? | REASON | MISSING PIECES |
|---|---|---|---|---|---|
| **Routing (req→replica)** | SIMULATED_ONLY | **YES** | **YES ✅** | `fleet_kv_routing` replays Mooncake across server caches; kv_aware reuses ~50% more prefix depth than round-robin → lower service factor → moves goodput/$ | — (connected this PR) |
| **KV-aware routing** | SIMULATED_ONLY | **YES (fleet level)** | **YES ✅ (via routing_policy=kv_aware)** | same channel; the *per-Azure-request* version still needs Azure prefixes | per-request Azure prefix ids for the finer version (kept SIMULATED_ONLY) |
| Replica count / capacity | **already CONNECTED** | yes | already on | `CapacityController` sizes replicas → GPU-hours/queue (the `capacity_policy` lever) | — |
| Batching / composition | PLANNED | no | no | the loop is per-request discrete-event; no throughput/latency/KV-memory batch model — a batch knob today would be a guess | a roofline batch model (effective service rate vs latency vs KV memory) |
| Placement / packing | REQUIRES_PILOT_TELEMETRY | partial | no | serving servers are homogeneous; v2026 topology is anchored marginals, not a per-server rack map in the loop | a topology placement simulator; live residency/health (ABSENT → pilot) |
| Migration / consolidation | PLANNED | no | no | no replica-assignment state or live-move cost in the loop | replica-placement state + a migration cost/penalty model |
| Prewarming | PLANNED | no | no (next) | the arrival forecast exists (and beats naive), but there is no warm-pool state or cold-start tax | warm-pool state + a cold-start-tax term; forecast-driven trigger |
| Energy / price shifting | PLANNED | partial | no | best-effort *deferral* exists (admission), and the objective is already price-aware, but cross-period price-shifting needs cross-period state + a deferrable-class model | cross-period deferral state + deferrable/SLA constraint model |
| Clock / DVFS | PLANNED | no | no | service time is fixed (TTFT+tokens·TPOT); no power-vs-performance curve | a clock action + a phase-aware power/perf curve + SLA-slack model |
| Precision / model routing | PLANNED | no | no | service time is precision-agnostic; no quality model | a quality/difficulty proxy + quality floor + per-precision service/quality model |
| Speculative decoding | PLANNED | no | no | no acceptance-rate or roofline (mem/compute-bound) model | acceptance-rate model + roofline indicator + draft-token overhead |
| tenant spot/arbitrage | REJECTED | n/a | no | out of product scope (operator, not tenant) | — |

## What this PR connects

**Routing (`routing_policy`: round_robin / shortest_queue / kv_aware) → CONNECTED**, via the
`kv_service_factor` channel:
- `fleet_kv_routing` (kv_cache.py) replays the Mooncake prefix trace across N server caches
  under each policy; `kv_aware` co-locates shared prefixes → more reuse depth.
- `routing_service_factors` yields a per-policy service factor (the share of service time a
  routed hit avoids); the controller scores each candidate bundle with its routing's factor,
  and `run_period_episode` replays the chosen routing's factor — so a routing decision changes
  the replayed service times and therefore goodput/$.
- The registry now enumerates 4 connected surfaces (capacity × ordering × admission × routing
  = **36 bundles**); the `CandidateBundleGenerator` searches that space and reports an ablation.

**Validation (causal, no oracle):** on the committed Mooncake fixture, `kv_aware` reuses more
prefix depth and saves ~2× the prefill tokens of `round_robin` → a strictly smaller service
factor (`tests/test_action_connection.py`). The routing **baselines** in the eval
(`sla_aware_kv_routing`, `aurelius_canonical_kv_routing`) also use kv-aware routing, so the MPC
must beat a routing-enabled baseline — it cannot win merely by *discovering* routing.

## What this PR deliberately does NOT connect (and why)

Batching, placement, migration, prewarming, energy-shifting, clock/DVFS, precision, and
speculative decoding remain PLANNED/SIMULATED_ONLY because the canonical simulator lacks the
state/model each needs (see the table). Each has an exact next step in
`research/AURELIUS_ACTION_SURFACE_AND_MPC_ARCHITECTURE.md`. Connecting any of them today would
be a fake knob — the one thing the architecture exists to prevent. **Recommended next:
prewarming** (the forecast already exists; it needs only a warm-pool state + cold-start tax).
