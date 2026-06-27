# Aurelius Next-Batch Action Simulability Audit (post-PR #99)

PR #99 connected routing/KV-aware routing. This audit decides which **remaining** action
surfaces the canonical simulator can score **causally and honestly today**, and connects only
those. The bar is unchanged: a CONNECTED action must change simulator output + reward, have a
test proving the effect, a fair baseline, documented fidelity, and stay behind the Pareto gate.

## Decision table

| ACTION | STATUS | CAN SIMULATE TODAY? | CONNECT NOW? | REQUIRED STATE | SIM EFFECT | REWARD EFFECT | FAIR BASELINE | VALIDATION | MISSING PIECES |
|---|---|---|---|---|---|---|---|---|---|
| **capacity_multiplier** (replica level) | new field | **YES** | **YES ✅** | sized replica count `c` (exists) | scales `c` per tick | GPU-hours/cost ↑, queue ↓, SLA ↓ | fixed `1.0x`; backlog-aware sizing | more replicas cut queue + raise GPU-hours (loaded fixture) | — |
| **batching_policy** (conservative/balanced/aggressive) | PLANNED→**YES** | **YES** | per-replica concurrency (added) | more slots/replica + service inflation | queue ↓ from concurrency; latency/SLA ↑ from inflation; cost ~flat | fixed `conservative` (today's 1/replica) | balanced helps; aggressive raises SLA viol (no free win) | INFERRED concurrency/latency magnitudes (public prior; no trace calibration) |
| routing_policy | CONNECTED (PR #99) | yes | already on | — | — | — | — | — | — |
| capacity_policy / ordering / admission | CONNECTED | yes | already on | — | — | — | — | — | — |
| **prewarm_policy** | PLANNED | partial | **no** | per-replica warm/cold state + availability time | cold-start delay on scale-up; warm-hold GPU-hours | would change latency + cost | reactive / always-prewarm | — | per-replica availability-time tracking + a cold-start-tax model in run_unified_replay (next PR) |
| **placement_policy / topology_policy** | SIMULATED_ONLY | partial (macro) | **no** | per-server rack/asw map in the loop | cross-rack net penalty on dispatch | small service/cost penalty | topology-blind | — | per-server rack assignment in the serving loop (v2026 topology is anchored marginals, not per-server); net pressure is a macro scalar |
| **migration_policy** | PLANNED | no | **no** | replica-assignment state across periods + move cost | — | — | no-migration | — | cross-period replica-placement state + a live-move cost/penalty model (the loop has no replica identity across periods) |
| **energy_policy** (price shifting) | PLANNED | partial | **no** | deferrable-class + cross-period deferral state | — | — | no-shift / current-price | — | cross-period deferral state + a deferrable/SLA constraint model (admission defers only WITHIN a period; the objective is already price-aware) |
| kv_placement_policy | PLANNED | no | **no** | an eviction-policy lever | — | — | LRU | — | a counterfactual eviction sim (cache is simulated STATE, not an action) |
| clock_policy (DVFS) | PLANNED | no | **no** (per instruction) | clock action + power/perf curve | — | — | nominal | — | a phase-aware power-vs-performance curve + SLA-slack model |
| precision_policy | PLANNED | no | **no** (per instruction) | quality/difficulty proxy + floor | — | — | full precision | — | a per-precision service/quality model + quality floor |
| spec_decode_policy | PLANNED | no | **no** (per instruction) | acceptance-rate + roofline model | — | — | spec off | — | an acceptance-rate model + memory/compute-bound roofline indicator |

## What this PR connects (2 new) + why the rest are deferred

**CONNECT — `capacity_multiplier`** (scales the sized replica count): the cleanest honest lever.
`run_unified_replay` already sizes `c` per tick; the multiplier scales it. Verified: 0.75×→1.5×
cuts queue p95 (14.0s→5.6s) and SLA violations (327→92) while raising GPU-hours (0.80→1.37).
Monotone, no free capacity. Fidelity: the sizing model is the existing CapacityController; the
multiplier is a deterministic scale.

**CONNECT — `batching_policy`** (per-replica continuous batching): `batch_concurrency` slots per
replica (throughput ↑, queue ↓ at the SAME GPU-hours) traded against `batch_service_factor`
(shared compute → per-request latency ↑ → SLA risk ↑). Verified: `balanced` (2× / 1.15×) cuts
queue p95 10.4s→3.3s with fewer violations; `aggressive` (4× / 1.5×) **raises** violations
150→224 — it cannot fake a win by violating SLA. **Fidelity: INFERRED** (the concurrency/latency
magnitudes are public-prior heuristics, like the cost model — not trace-calibrated; documented
and sanity-banded).

**DEFER — prewarming, placement/topology, migration, energy, kv-placement, clock, precision,
spec-decode.** Each needs simulator state/physics the canonical stack does not have yet (see
table). Connecting any today would be a fake knob. **Recommended next: prewarming** — it needs
only per-replica availability-time tracking + a cold-start tax, and the arrival forecast already
beats naive.

## Tractability note

Connecting both new levers grows the connected bundle space to capacity(3)×ordering(2)×
admission(2)×routing(3)×capacity_multiplier(3)×batching(3) = **324 bundles** (> the 256
exhaustive budget). The `CandidateBundleGenerator.search` therefore switches from full
enumeration to **coordinate descent** from the no-op incumbent (each free surface moved to its
best option, repeated to convergence) — it touches every connected dimension at ≈ surfaces×
options×passes evaluations (~50–60, not 324), so no connected knob is silently dropped. The
planner reports the method used, the theoretical combination count, and how many candidates it
actually evaluated; `latin_hypercube` remains available for even larger spaces.
