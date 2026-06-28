# World-Model Realism Gap Audit

Audit of the persistent world model (`world_state.py`, `world_simulator.py`, `world_calibration.py`,
`kv_cache.py`, `controller.py`, `optimizer/unified_replay.py`, `training.py`, `scripts/sweep_mpc_horizon.py`)
**before** the realism work of this PR. The goal: find why prewarm / migration / placement / batching
still do not become valuable (PR #104), and ground each fix in public data with a fidelity tier and a
validation test — **no fake knobs, no result-tuning, no weakening of the Pareto gate**.

## The one-sentence diagnosis

The MPC machinery is correct (PR #103) and warm state now persists across sub-hour steps (PR #104 bug
fix), but the **deferred actions have no realistic benefit channel**: migration pays a cost and a KV
*penalty* for only a tiny topology discount (so it is never worth it); cold-start is a single flat 30 s
constant (so prewarm cannot avoid a *specific* component and migration cannot *preserve* the expensive
component a cold-start-elsewhere would pay); and the rich per-replica KV residency that `kv_cache.py`
already models is computed **offline as a fleet-scalar** and never enters the persistent replicas.

---

## Gap 1 — Replica identity & cache identity (Phase 1)

**Current behavior.** `ReplicaState` carries `warm/cold`, `last_used_period`, `server_id`, `rack_id`,
`gpu_type`, `assigned_capacity`. It has **no** `model_id`/loaded-weights, **no** resident KV / prefix
set, **no** cache occupancy, **no** `warm_until`. Migration (`_advance`) only rewrites `rack_id` when a
move lands; the replica's warm/KV state is not moved — and `_migration_plan` actually applies a KV
*penalty* (`cache_factor = 1.04`).

**Why not production-realistic.** A real replica is an *identity*: specific weights resident in HBM, a
specific paged KV cache, a home server/rack. Live migration (Llumnix) **moves that identity** — weights
stay loaded, KV is pipelined to the destination (append-only, near-zero downtime) — which is precisely
why migrating a hot replica beats cold-starting a fresh one elsewhere. Modelling migration as
cost-plus-penalty with no state preservation makes it strictly dominated, so the MPC (correctly) never
picks it. That is a *model* artifact, not a finding about migration.

**Public source.** Llumnix live migration (pipelined KV copy, near-zero downtime, append-only KV) —
arXiv 2406.03243 (already cited in `world_calibration.py`). KV transfer bandwidth (RDMA 12–50 GB/s,
PCIe5 ≈ 63 GB/s, 1–10 GB caches) — arXiv 2504.11816 (cited).

**Calibration method.** Give `ReplicaState` identity fields (`model_id`, `warm_until_period`,
`kv_warm_frac`, `weights_loaded`). Make migration **move the identity**: destination inherits
`weights_loaded` (no model-load cold-start) and, per migration mode, inherits KV warmth (pipelined
copy) or pays a calibrated KV re-warm; source capacity is withheld while in-flight; **no duplicate
replica** is created. The benefit is the *avoided model-load cold-start at the destination*, valued by
the Gap-4 decomposition — not a heuristic bonus.

**Expected simulator effect.** Migration becomes a real trade: pay transfer + capacity-loss now, keep a
warm hot replica on a better rack (avoiding a destination cold-start) later. Whether that nets positive
on the Azure trace is the empirical question the dt=60 diagnostic answers, behind the unchanged gate.

**Validation test.** Replica identity persists across periods; migration moves identity source→dest;
migration never duplicates a replica or capacity; warm/KV conservation across a move; clone isolation.

## Gap 2 — Request→prefix locality from Mooncake (Phase 2)

**Current behavior.** `kv_cache.py` has a full causal paged-LRU per-server cache, a KV-aware router, and
Mooncake `hash_ids` reuse — but `build_mpc_inputs` runs it **offline** and reduces it to one scalar
`service_factor` per routing policy (`routing_service_factors`). The persistent replicas hold **no**
prefixes; Azure serving requests have token counts but **no prefix signatures**, so within the world
simulator routing/placement/migration cannot exploit *which* replica holds a reusable prefix.

**Why not production-realistic.** Prefix reuse is *local* to whichever replica served the conversation;
routing to that replica skips prefill, and migrating it preserves that locality. A fleet-scalar erases
the per-replica locality that makes placement and KV-aware routing matter.

**Public source.** Mooncake `hash_ids` block-level prefix-reuse trace (committed) — TRACE_DERIVED. No
row-join with Azure: assign **trace-derived** prefix signatures by sampling Mooncake reuse sequences
(exact/partial reuse, depth, reuse distance) onto Azure arrivals → label `TRACE_DERIVED_REUSE_MODEL`.

**Calibration method.** Carry a compact resident-prefix summary on each warm `ReplicaState`; assign each
Azure request a Mooncake-sampled prefix signature (causal, no future leakage); route to the replica with
the most resident leading-prefix reuse; migration moves residency, fresh cold-start loses it.

**Expected simulator effect.** KV-aware routing/placement gain a *per-replica* channel; migration that
preserves KV beats cold-start-elsewhere by the saved prefill.

**Validation test.** Generated prefix stream matches Mooncake held-out distribution (KS / histogram-L1
on reuse depth & hit rate); no future-prefix leakage; KV hit depth changes routing outcome; migration
preserving identity raises future KV hit rate vs a fresh replica.

> **Scope note.** This is the largest change (touches the serving replay inside `simulate_period`). This
> PR lands the *identity + cold-start* benefit channels (Gaps 1, 4) that make migration/prewarm
> testable, and **defers** the full per-replica prefix routing inside the world simulator to the next
> increment with the calibration plan above — rather than ship it half-validated.

## Gap 3 — Topology-calibrated network penalties (Phase 3)

**Current behavior.** `_placement_plan` already derives a macro service-time factor from where warm
replicas sit, using the v2026 per-rack macro `rx+tx` pressure, capped at `TOPOLOGY_MAX_DISCOUNT = 0.08`
(TRACE_DERIVED). Same-rack vs cross-rack is implicit via the rack-pressure spread; there is no explicit
KV-transfer-across-racks penalty.

**Why not (fully) production-realistic.** It models *macro* pressure relief but not the *cost of moving
KV across racks* (cross-ASW transfer is slower than same-rack), so migration cost is rack-distance-blind.

**Public source.** v2026 `server_hourly` `asw_id` locality + `network_hourly` rx/tx marginals
(TRACE_DERIVED); KV transfer bandwidth bands (arXiv 2504.11816). **No** NVLink/NVSwitch/RDMA/PFC/ECN or
per-link congestion — ABSENT from any committed trace, must not be invented.

**Calibration method.** Add a same-rack < cross-rack migration-cost multiplier from the KV-transfer-time
band (cross-rack pays the macro-pressure-scaled transfer; same-rack is cheap). Keep the existing macro
discount. Magnitude documented, capped, flat-pressure → ~no benefit.

**Expected effect.** Cross-rack migration costs more than same-rack; placement benefit only where the
pressure spread is real.

**Validation test.** same-rack penalty < cross-rack; flat pressure → no placement win; penalty
magnitude has provenance.

> **Scope note.** The macro placement channel already exists and is calibrated; the cross-rack-transfer
> refinement is **audited here and deferred** with the rest of Gap-2's KV-movement physics (they share
> the transfer-bandwidth source) to keep this PR coherent and fully validated.

## Gap 4 — Cold-start calibration (Phase 4) — **implemented this PR**

**Current behavior.** A single `cold_start_s = 30 s` constant; prewarm avoids "a cold start," migration
pays a flat cache penalty. No components.

**Why not production-realistic.** A cold start is several distinct costs — **engine/process init**,
**model-weight load** (the dominant term), **KV warm-up**, **scheduler/ready delay** — that different
actions avoid differently: a *warm* replica skips engine-init + model-load; a *migrated* replica keeps
weights loaded (skips model-load) but may lose KV-warmup; *prewarming* pays warm-hold to skip the future
model-load + engine-init. A single constant cannot express these.

**Public source (already cited in `world_calibration.py`).** vLLM/GKE startup decomposition (engine init
2–5 s; weight load dominates) — dudeperf3ct startup post; model-load 10–72 s by storage, 8–32 B ≈ 60 s
on A100 — gigagpu; warm-pool / sleep-mode resume 2–8 s — vLLM sleep-mode; ServerlessLLM cold-start survey
— arXiv 2401.14351. These already describe the components; this PR *decomposes the existing band* into
them (sum-preserving), it does **not** lower cold-start to improve results.

**Calibration method.** Split `cold_start_s` into `engine_init_s` (2–5, base 3), `model_load_s`
(10–60, base 22), `kv_warmup_s` (0–8, base 3), `ready_delay_s` (0–4, base 2) with low/base/high bands;
`total = sum` reconciles to the existing 8/30/60 band. Warm replicas avoid engine+model+ready; a
migrated replica avoids engine+model (weights move) but pays `kv_warmup` unless KV is pipelined;
prewarming pays warm-hold to avoid the forecast period's `model_load + engine_init`.

**Expected effect.** Prewarm pays off on a forecasted up-ramp (avoids the big model-load on the cold
replicas the ramp would otherwise need); over-prewarm on a wrong forecast still wastes warm-hold;
migration's avoided-model-load becomes a quantified benefit vs cold-start-elsewhere.

**Validation test.** Components sum to the total band; bands sane vs sources; prewarm helps in a
calibrated burst fixture; over-prewarm hurts on a wrong forecast; cold-start not tuned down.

## Gap 5 — Roofline-aware batching (Phase 5)

**Current behavior.** Batching is a fixed `batch_concurrency` × `batch_service_factor` pair per policy
(`conservative`/`balanced`/`aggressive`) in `unified_replay` — concurrency speedup and a flat service
inflation, independent of GPU type, sequence length, prefill/decode mix, or KV/memory pressure.

**Why not production-realistic.** LLM serving has two regimes — **prefill** (compute-bound) and
**decode** (memory-bandwidth-bound, low arithmetic intensity) — so optimal batch size depends on the
GPU's FLOPs:HBM-bandwidth ratio, sequence lengths, and KV pressure. A fixed pair cannot make batching
*hurt* under SLA pressure for the right reason, nor differentiate H100 from A100.

**Public source.** NVIDIA public specs — A100: 312 TFLOP/s BF16, 1.555–2.0 TB/s HBM; H100: ~990 TFLOP/s
BF16, ~3.35 TB/s HBM (BENCHMARK_DERIVED). Decode memory-bound roofline + continuous batching — Orca
(OSDI'22), vLLM/PagedAttention (SOSP'23), Sarathi-Serve chunked prefill (OSDI'24). Use benchmark-derived
*bands*, not vendor-private curves.

**Calibration method.** A `roofline.py` `RooflineState` mapping (gpu_type, batch, prompt/gen lengths,
KV pressure) → tokens/s + latency via a memory-bound vs compute-bound estimate; batching helps until KV
/ bandwidth saturation, then latency rises; SLA slack gates the benefit.

**Expected effect.** Batching becomes regime-dependent and can hurt; H100 vs A100 differ plausibly.

**Validation test.** throughput rises to saturation then latency rises; KV/memory pressure constrains
batch; no free batching; roofline changes action selection in a fixture.

> **Scope note.** Audited here with the calibration plan and sources; **deferred** to the next increment
> so this PR ships the identity + cold-start benefit channels fully validated rather than five
> half-done domains. The honest sequencing is benefit-channel-first.

## Gap 6 — Transition validation suite (Phase 6) — **implemented this PR (for landed mechanisms)**

**Current behavior.** `validation_suite.py` / `validators.py` exist for the multi-plane environment;
there is no single suite asserting the *world-model transitions* (cold-start decomposition, replica/
migration conservation) against public evidence with PASS/WARN/FAIL.

**Calibration method.** `world_validation.py` checks: cold-start components sum to band & sane vs
sources; replica-count / warm / capacity conservation across a migration; no replica duplication; clone
isolation; determinism. Distribution checks (Mooncake KS, Alibaba locality, roofline curves) are stubbed
as `SKIPPED` with the reason until Gaps 2/3/5 land — explicit, never silently "passing".

---

## What this PR implements vs defers (honest scope)

| Phase | Gap | This PR |
|------|-----|--------|
| 1 | Replica + cache identity, migration moves state | **Implemented** (identity fields; migration moves warm/model identity, no duplication; KV-warmth moved/lost per mode) |
| 4 | Cold-start decomposition | **Implemented** (4 components, sum-preserving, sources cited) |
| 6 | Transition validation suite | **Implemented** for landed mechanisms; distribution checks `SKIPPED` with reasons |
| 8 | dt=60 diagnostic | **Run** (bounded 6 h window) |
| 2 | Per-replica Mooncake prefix routing in the world sim | **Audited + deferred** (calibration plan above) |
| 3 | Cross-rack KV-transfer penalty | **Audited + deferred** (shares Gap-2 transfer source) |
| 5 | Roofline batching | **Audited + deferred** (calibration plan + sources above) |

Rationale: the deferred actions' *benefit channels* are identity + cold-start (Gaps 1, 4). Landing those
two **fully calibrated, validated, and tested** — then measuring at dt=60 — answers the question more
honestly than shipping five partially-validated domains. The deferred gaps have concrete calibration
plans and sources here and are the named next increment.
