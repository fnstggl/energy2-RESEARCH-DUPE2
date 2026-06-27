# Aurelius Action-Surface + MPC Controller Architecture

This documents the canonical action-surface interface that lets the MPC economic controller
grow from today's connected levers to all first-principles actions **without a redesign** —
while never pretending a not-yet-real knob is active. Code: `aurelius/environment/actions.py`
(schema), `aurelius/environment/action_registry.py` (registry),
`aurelius/environment/controller.py` (planner). Evidence: the Phase-1 audit
`research/AURELIUS_ACTION_SURFACE_AUDIT.md`.

## Why Aurelius is not "choose among 12 fixed policies"

The end product is a predictive world model + model-predictive economic controller:

```
historical telemetry → forecasting layer → canonical simulator/world model
  → economic planner → candidate action bundles → simulated outcomes
  → maximize SLA-safe goodput/$ → emit infrastructure action bundle
```

The controller must *understand* every major infrastructure action surface, but it must only
**optimize actions that actually exist and that the simulator can score**. The previous
controller hard-coded a 12-way product of three levers; this architecture replaces that with
a typed `ActionBundle` over **15 canonical surfaces** and a registry that decides — from the
honest audit status — which surfaces are optimized, which are opt-in, and which are merely
represented.

## What is connected today

| surface | field | status | simulator hook |
|---|---|---|---|
| replica count / capacity | `capacity_policy` | **CONNECTED** | `run_unified_replay(capacity=)` |
| ordering / scheduling | `ordering_policy` | **CONNECTED** | `run_unified_replay(ordering=)` |
| admission / defer | `admission_policy` | **CONNECTED** | `run_unified_replay(admission=)` |

These three change the scored reward (SLA-safe goodput/$). The registry's default
`enumerate_candidate_bundles(connected_only=True)` produces exactly their 3×2×2 = **12**
bundles — every other surface stays at its no-op default, so the bundle is byte-for-byte
today's behaviour plus a richer type.

## What is represented but NOT optimized

- **SIMULATED_ONLY (3):** `routing_policy`, `kv_routing_policy`, `topology_policy`. A real
  model exists (`KVAwareRouter`, `StatefulKVCache`) but it is **not wired into
  `run_unified_replay`'s reward path** (dispatch is round-robin `_free_sid`). These are
  enumerated only behind `controller.optimize_simulated=True`, and even then have **no reward
  effect until wired** — the flag is the extension point, not a claim.
- **PLANNED (8) + REQUIRES_PILOT_TELEMETRY (1):** `batching_policy`, `kv_placement_policy`,
  `prewarm_policy`, `clock_policy`, `precision_policy`, `spec_decode_policy`, `energy_policy`,
  `migration_policy`, `placement_policy`. Represented (with their conceivable future option
  sets) so the interface is future-proof, but **never enumerated** and **rejected by
  `validate_action_bundle` if set away from their no-op** ("not actuatable").
- **REJECTED:** tenant-side spot/reserved/on-demand arbitrage — out of product scope
  (`cost_model.py:13`; the serving cost basis is pure on-demand, "no spot, no oracle").

## Why fake knobs are dangerous

The audit found `optimizer_adapter.ACTION_SPACE` already advertised `kv_routing: [True,False]`
that **nothing consumes** — toggling it changes no KPI. A planner that "optimizes" such a knob
will report a decision that does nothing, or worse, manufacture an apparent win from noise. The
registry structurally prevents this: only `CONNECTED` surfaces are optimizable
(`ActionSpec.optimizable`), `affects_reward` is true **iff** `status == CONNECTED`, and every
non-connected surface maps to `sim_param = None` (it cannot reach the simulator). Tests
(`test_action_surface.py`) assert no non-connected surface can change the simulator kwargs.

## How the MPC controller optimizes action bundles

`ModelPredictiveEconomicController.decide()` now:
1. builds the causal `ForecastBundle` (unchanged);
2. asks the registry for candidate `ActionBundle`s (`connected_only` by default);
3. simulates each bundle's `connected_kwargs()` on the forecast via `unified_replay` + cost
   model, scoring risk-adjusted SLA-safe goodput/$ (unchanged scoring);
4. returns a `Decision` whose `.action` is still the legacy `{capacity,ordering,admission}`
   dict (back-compat) **and** whose `.bundle` is the full `ActionBundle`;
5. exposes `understood_but_unavailable()` → the SIMULATED_ONLY + PLANNED surfaces, reported
   **separately** so planned knobs are never mistaken for active ones.

The claim gate is unchanged and remains honest (Pareto-aware; PR #96): no headline unless the
controller beats the fair baseline without trading away SLA.

## How a new action becomes available (the lifecycle)

`PLANNED → SIMULATED_ONLY → CONNECTED`:
1. **Simulate:** add the action's effect to `run_unified_replay` (or a sibling simulator) so a
   non-default value changes a KPI. Flip the spec to `SIMULATED_ONLY`, set its `sim_param`.
2. **Validate:** show the simulated effect matches a held-out reference (or a documented
   benchmark) — and pick its fair baseline.
3. **Connect:** when the effect is in the reward path and validated, flip to `CONNECTED`; the
   registry then enumerates it by default. No controller change required — only the spec.

## Phase 5 — concrete implementation path per PLANNED action

Each row is one future PR: shadow-sim first, behind the same Pareto-aware gate, kill if it
does not beat its fair baseline on held-out.

| action | simulator change needed | telemetry needed | forecast target | reward term | validation | first fair baseline | min shadow-sim test |
|---|---|---|---|---|---|---|---|
| **fleet-global KV routing** (N4, *do first* — already SIMULATED_ONLY) | one `StatefulKVCache` per server inside `run_unified_replay`; route via `KVAwareRouter.route()` instead of `_free_sid` | none extra (Mooncake reuse already FULL_TRACE) | prefix-reuse rate (have) | service-time discount on cache hits → goodput/$ | fleet hit-rate vs per-node cache (held-out Mooncake) | per-node cache (round-robin) | hit-rate(routed) > hit-rate(round-robin) on the committed fixture |
| **clock / DVFS** (N2) | `clock_factor` scales `service_s` ↑ and power ↓ (convex) per phase | per-clock energy/perf curve (BENCHMARK_DERIVED ok) | SLA slack per period (derivable) | energy term in `CostModel` responds to clock | $/token at equal SLA vs fixed max-clock | fixed nominal clock | $/token(clocked) < $/token(nominal) at equal SLA violations |
| **precision / model routing** (N5) | per-precision `service_s` + a quality score; route by difficulty | quality/difficulty proxy; quality floor | request difficulty (PLANNED) | quality-constrained goodput/$ | quality within floor; held-out quality | static full precision / RouteLLM | $/quality(routed) < $/quality(full) with no quality breach |
| **speculative decoding** | draft-token overhead + acceptance model gated by a roofline (mem/compute-bound) indicator | roofline state (arith-intensity) | batch memory/compute regime | throughput term at fixed latency | latency/throughput vs no-spec | reactive (spec off) | throughput(spec) > throughput(off) at equal p95 latency |
| **prewarming** (N7) | warm-pool state + a cold-start tax avoided when pre-warmed | none extra | arrival + prefix forecast (have, beats naive) | cold-start cost avoided − warm-hold cost | prewarm cost < avoided cold-start on held-out | reactive (no prewarm) | net cost(prewarm) < net cost(reactive) on a bursty hour |
| **batching / composition** (N1) | a roofline batch model (throughput vs latency vs KV memory) | none extra | arrival burst + token mix | goodput at the batch-size frontier | tokens/joule vs measured ceiling | best fixed batch size | gp/$(composed) > gp/$(fixed) on held-out |
| **migration / consolidation** | replica-assignment state + a live-move cost | replica placement state | load drift forecast | consolidation savings − migration cost | migration cost < idle savings | no-migration | net(migrate) < net(static) without SLA breach |
| **placement / packing** | a topology-aware placement simulator (asw/rack) over the serving servers | **live residency / hardware health (ABSENT → pilot)** | fragmentation forecast | locality/fragmentation cost | needs pilot telemetry to validate fidelity | no topology awareness | (gated on pilot) |
| **energy / price shifting** (N2) | a temporal-shift action the sim honours (defer best-effort to cheap hours) | none extra (price in objective) | electricity-price forecast | price-weighted cost (already in objective) | $/token reduction at equal SLA | no shifting | $/token(shift) < $/token(none) at equal SLA |

**Recommended order:** **fleet-global KV routing (N4) first** — it is the only PLANNED-class
gain whose model already exists (it is SIMULATED_ONLY, not PLANNED); wiring `KVAwareRouter`
into the dispatch loop is the smallest step from "represented" to "connected" and has real
public-trace validation (Mooncake). Then **clock/DVFS (N2)** and **prewarming (N7)**, which
reuse the forecasts that already beat naive.

## Claims

**Safe:** "Aurelius has a canonical action-surface architecture for MPC over infrastructure
action bundles. The controller optimizes the currently connected actions
(capacity/ordering/admission) and explicitly tracks SIMULATED_ONLY and PLANNED first-principles
actions without pretending they are active."

**Unsafe:** "Aurelius optimizes all GPU-fleet knobs today." (It optimizes three; the rest are
represented, audited, and gated — with a concrete path to connect each.)
