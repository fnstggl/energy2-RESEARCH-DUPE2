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

## Incremental held-out results (Azure 2024 week, persistent world, 42 held-out periods)

Each stateful action isolated by freezing the other two to their no-op (the *other six* connected
surfaces vary in every rung, so the capacity/batching/routing behaviour is identical across rungs and
cancels in the marginal — the delta is the stateful action alone).

| Rung | gp/$ | SLA viol | queue p95 | cold-starts | warm-hold (GPU-h) | migration $ | Δ gp/$ vs core |
|---|--:|--:|--:|--:|--:|--:|--:|
| core (all stateful no-op) | 96,627 | 0.0151 | 8.3s | 31 | 30.0 | 0 | — |
| +prewarm | 96,627 | 0.0151 | 8.3s | 31 | 30.0 | 0 | **+0.00%** |
| +placement | 93,950 | 0.0143 | 9.0s | 37 | 36.0 | 0 | **−2.77%** |
| +migration | 96,627 | 0.0151 | 8.3s | 31 | 30.0 | 0 | **+0.00%** |
| full | 93,950 | 0.0143 | 9.0s | 37 | 36.0 | 0 | −2.77% |

**Honest reading.** On the Azure serving trace the MPC **does not select prewarm or migration**
(both rungs == core): reactive scaling is adequate so prewarming's warm-hold is not worth it, and
the single-period decision horizon cannot amortise a migration whose benefit only lands next period
(so it never pays the up-front cost). **Placement is selected** (the MPC picks rack_local /
network_aware) and trades **−2.77% gp/$ for a slightly better SLA** (0.0151→0.0143; Pareto-not-worse
True). So none of the three is a gp/$ win on this trace — the effects are real (proven below) but
the workload doesn't reward them. They stay CONNECTED (they meet every bar: change output + reward,
tested, fair baseline, behind the gate) exactly as capacity_multiplier did in PR #100 despite being
net-negative alone; this is reported, not hidden.

## Final fair backtest (Azure 2024 week, persistent world, 42 held-out periods)

| Arm | gp/$ | SLA viol | queue p95 | GPU-h | note |
|---|--:|--:|--:|--:|---|
| `aurelius_canonical_kv_routing` ⟵ **fair baseline** | 162,965 | 0.0162 | 17.7s | 121.8 | no stateful actions (capacity 1.0×) |
| `world_static_best` | 157,739 | 0.0124 | 8.8s | — | static operator that **already** places topology-aware |
| `aurelius_static_full` | 149,849 | 0.0141 | 8.8s | — | |
| `sla_aware` | 143,861 | 0.0210 | 6.2s | — | |
| **mpc_controller** | **93,950** | 0.0143 | 9.0s | 188.7 | prewarm off ×42, migration off ×42, placement topology-aware, **capacity 1.5× ×42** |
| `prewarm_always` | 32,877 | 0.0124 | 0.0s | — | aggressive prewarm everywhere → warm-hold dominates |

```
fair_baseline          = aurelius_canonical_kv_routing
beats_fair_baseline    = False  (mpc −42.35% gp/$)
pareto_sla_not_worse   = True   (mpc 0.0143 ≤ fair 0.0162)
headline_claim_allowed = False
```

**Honest reading.** The world-path MPC **regresses 42% on gp/$** — but the cause is **not** the new
stateful actions (prewarm/migration stay off). It is `capacity_multiplier=1.5` chosen in every period
(GPU-hours 188.7 vs the baseline's 121.8): the tuner, run on the now-riskier cold-start world, selected
a **risk-averse config (risk_weight=1.0)** that over-provisions to suppress SLA risk, trading gp/$ for a
slightly better SLA. A competent static operator (no stateful actions, capacity 1.0×) beats it. The
`prewarm_always` row is the other end of the honesty ledger — holding everything warm craters gp/$ to
32,877. So on this trace the persistent-world MPC is a Pareto *re-balance* (cheaper SLA, much worse
gp/$), not a win — the gate blocks the headline, correctly. The stateless PR-#100 path (world_state
off) is unchanged and still posts its +35% (its result is not affected by this opt-in path).

## Verdict on each connected action (Azure trace)

- **placement_policy** — CONNECTED, selected, **Pareto-neutral-to-slightly-negative** here (−2.77%
  gp/$, +SLA). Real macro discount; the trace's locality/pressure structure is thin.
- **prewarm_policy** — CONNECTED, effect proven in tests, **not selected** (reactive scaling adequate).
- **migration_policy** — CONNECTED + honestly costed, **not selected**: the single-period MPC horizon
  cannot amortise a deferred benefit. The honest fix is multi-period lookahead (next step), not a knob.

## Safe vs unsafe claims

**Safe (supported):** *Aurelius now has a persistent canonical world state with server, rack,
replica, warm/cold, placement, and migration state, sampled from public v2026 trace marginals. The
MPC can simulate and optimise prewarming, topology-aware placement, and migration as stateful
infrastructure actions behind a Pareto-aware claim gate, with each action's effect proven in tests
and each able to HURT (so none is a fake knob).* **On the Azure 2024 serving trace these actions do
NOT produce a goodput/$ win** — prewarm/migration are left off and placement is gp/$-negative — and
the gate blocks the headline; this is reported honestly, not forced into a claim.

**Unsafe (NOT claimed):** that Aurelius has production telemetry for live placement, warm state, or
operator migration decisions; that the cluster is a real machine inventory; or that any per-link /
NVLink / micro-congestion behaviour is modelled. All numbers are **SIMULATED directional evidence
on a TRACE_DERIVED_SAMPLE cluster**, not production telemetry.

## Next recommended action

Energy-aware temporal shifting of deferrable (best-effort) work — it needs the one piece this batch
did not build: cross-period **DeferrableWorkState** (a deferral window + deadline model). The price
is already in the objective and best-effort admission already exists, so it is the smallest honest
next step.
