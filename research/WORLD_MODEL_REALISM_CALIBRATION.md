# World-Model Realism Calibration (PR #105)

Per-transition calibration for the realism this PR lands — **cold-start decomposition**, **replica/
migration identity**, and the **migration KV-preservation correction** — each with its model, source,
fitted band, confidence, validation, and limitation. Every parameter lives in
`aurelius/environment/world_calibration.py` with a fidelity tier and a URL; the validation suite is
`aurelius/environment/world_validation.py` (run via `run_world_validation()`); the gap audit and the
deferred gaps are in `research/WORLD_MODEL_REALISM_GAP_AUDIT.md`.

**Rule honored:** no parameter was tuned to make an action profitable. The aggregate cold-start base is
**unchanged at 30 s** (decomposed, not lowered); the migration change *corrects an unrealistic flat KV
surcharge* using an already-cited source, it does not zero out migration's cost.

## Fidelity tiers

`TRACE_DERIVED` (our committed trace) · `PUBLIC_PAPER` (published measurement) · `BENCHMARK_DERIVED`
(public vendor/blog magnitude) · `SIMULATOR_INFERENCE` (explicit modelling assumption, no external
measurement).

---

## Transition 1 — Cold-start decomposition (Phase 4)

**Model.** `total_cold_start_s = engine_init_s + model_load_s + kv_warmup_s + ready_delay_s`. A *warm*
replica skips engine+model+ready (and keeps its cache); a *migrated* replica skips engine+model (weights
move) and keeps `migration_kv_preserved_frac` of `kv_warmup`; *prewarming* pays warm-hold to skip the
forecast period's cold replicas' engine+model.

| Component | low/base/high (s) | tier | source |
|---|---|---|---|
| `cold_start_engine_init_s` | 2 / 3 / 5 | BENCHMARK_DERIVED | vLLM/GKE startup (engine init 2–5 s) |
| `cold_start_model_load_s` | 10 / **22** / 60 | BENCHMARK_DERIVED | model-weight load 10 s NVMe…60–72 s remote; 8–32 B ≈ 60 s A100 (gigagpu) |
| `cold_start_kv_warmup_s` | 0 / 3 / 8 | SIMULATOR_INFERENCE | empty-cache first-batch prefill; overlaps sleep-mode resume 2–8 s |
| `cold_start_ready_delay_s` | 0 / 2 / 4 | SIMULATOR_INFERENCE | k8s readiness + endpoint registration |

**Fitted base sum = 30.0 s**, reconciled to the pre-existing `cold_start_s` base (30 s). Bands bracket
the aggregate (low 12 vs 8; high 72 vs 60 — the model-load high is the SATA/remote 72 s prior).

**Validation (PASS).** `components_sum_to_aggregate_base` (30.0 == 30.0); every band ordered;
`model_load_dominates` (22 > 3). **Confidence:** medium (engine/model from public benchmarks;
kv_warmup/ready_delay are inference). **Limitation:** storage tier dominates `model_load` and is
deployment-specific. **Production telemetry that would improve it:** per-replica cold-start traces
(engine-up → first-token) by storage backend.

## Transition 2 — Replica & migration identity (Phase 1)

**Model.** `ReplicaState` carries `model_id`, `weights_loaded`, `kv_warm_frac`, `warm_until_period`.
A migration **moves the identity** (same `replica_id`): source capacity is withheld while in-flight; on
landing the replica adopts the target rack, **lands warm with `weights_loaded=True`** (no destination
model-load), and keeps `kv_warm_frac × migration_kv_preserved_frac` of its cache. A cooled (idle past
the 300 s timeout) replica unloads weights and clears its cache (`weights_loaded=False`,
`kv_warm_frac=0`), so a future use legitimately cold-starts. **No replica is ever duplicated.**

**Source.** Llumnix live migration (pipelined KV copy, near-zero downtime, append-only KV) — arXiv
2406.03243 (PUBLIC_PAPER); KV transfer bandwidth bands — arXiv 2504.11816.

**Validation (PASS).** `replica_count_conserved` (60→60 across a move); `no_replica_duplication` (id set
stable); `landed_replica_keeps_weights` (16/16 landed moves kept weights); clone isolation; determinism.
**Confidence:** medium (conservation is exact; the *value* of preserved state depends on Gap 2).
**Limitation:** `kv_warm_frac` is carried but **not yet consumed** by the serving service-time (that is
Gap 2 — per-replica prefix routing in the world sim, deferred). **Production telemetry:** per-replica KV
residency + migration event logs.

## Transition 3 — Migration KV-preservation correction (Phase 1)

**Model.** A live move keeps a mode-dependent share of KV warmth; the service surcharge is the **lost**
fraction only: `cache_factor = 1 + migration_cache_penalty × (1 − preserved)`.

| mode | `kv_preserved_frac` | `cache_factor` | meaning |
|---|---|---|---|
| conservative (pipelined / Llumnix) | 0.90 | ≈ **1.004** | most KV moves with the replica |
| aggressive (bulk) | 0.60 | ≈ 1.016 | more KV re-warmed at the destination |

`migration_kv_preserved_frac` band 0.5 / **0.9** / 1.0 (PUBLIC_PAPER, Llumnix). This **replaces the old
flat `cache_factor = 1.04`** — a 4 % surcharge on *every* migrated period that, combined with the move
cost, made migration strictly dominated regardless of horizon. The move still pays
`migration_cost_per_replica` ($0.40) + `migration_capacity_loss_frac` (0.10) — migration is **not** free.

**Validation (PASS).** `pipelined_keeps_more_kv_than_bulk` (0.9 > 0.6); `kv_mostly_preserved_not_
surcharged` (≥ 0.5); conservative `cache_factor < 1.01`. **Confidence:** low (no $-denominated public
migration benchmark; preservation quality varies). **Limitation:** the *benefit* of the preserved KV is
not yet a serving-time saving (Gap 2). **Production telemetry:** migration downtime + post-move TTFT.

---

## What this calibration does and does not change behaviorally (honest)

- **Behavioral change:** the migration `cache_factor` correction (1.04 → ≈1.004 conservative) removes the
  surcharge that made migration strictly dominated. This is the one change that can shift an MPC choice.
- **Fidelity-only (no behavioral change this PR):** the cold-start *decomposition* does not change the
  *total* a warm/prewarmed replica avoids (still ~30 s), so it does not by itself change prewarm
  economics; it makes the cost auditable by component and enables the migration-vs-cold-start
  distinction that **Gap 2** will turn into a serving-time saving. The identity fields
  (`weights_loaded`, `kv_warm_frac`) are conserved scaffolding consumed by the deferred Gap-2 routing.

## Deferred transitions (audited, with plans + sources)

| transition | status | source ready | becomes live when |
|---|---|---|---|
| Per-replica Mooncake prefix routing (Gap 2) | SKIPPED | Mooncake `hash_ids` (committed) | residency consumed by serving service-time |
| Cross-rack KV-transfer penalty (Gap 3) | SKIPPED | v2026 asw locality + KV-BW band | same-rack < cross-rack move cost wired |
| Roofline batching (Gap 5) | SKIPPED | NVIDIA specs + Orca/vLLM/Sarathi | `roofline.py` lands |
| Migration future-KV-hit-rate | SKIPPED | — | Gap 2 provides a per-replica hit rate to measure |

The validation suite emits these as **SKIPPED with the reason**, never silently passing.
