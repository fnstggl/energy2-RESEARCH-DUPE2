# Full Serving-Physics Integration Plan (Phase 0)

Derived from the PR #110 audit (`OPEN_LLM_INFERENCE_SIMULATOR_REUSE_AUDIT.md`, `..._CAPABILITY_MATRIX.md`,
`AURELIUS_VS_OPEN_SIMULATORS_GAP_ANALYSIS.md`, `..._CODE_PATHS.md`, `..._REUSE_DECISIONS.md`,
`ROOFLINE_REUSE_DECISION.md`, `FULL_WORLD_MODEL_FEASIBILITY_ASSESSMENT.md`,
`FULL_AI_INFERENCE_WORLD_MODEL_RECOMMENDATION.md`).

## Decision recap from #110

No open simulator is a drop-in. Aurelius stays canonical (persistent ClusterState, clone-safe MPC,
routing/placement/migration/prewarm, per-replica KV/model residency, goodput/$, trace fusion). This PR
**ports the strongest external-derived physics into a live V2 serving world model built *beside* V1** —
not over it.

## Architecture: V2 beside V1 (hard rule #1)

V1 (`world_state.py`, `world_simulator.py`, `prefill_decode.py`, `kv_cache.py`, `serving_plane.py`,
`controller.py`) is **unchanged and remains the canonical/default**. V2 lives in a new subpackage
`aurelius/environment/v2/` and shares only **read-only** primitives:

- `roofline_external.py` (timing formulas — already ported in #110),
- `cost_model.CostModel` / `GPUEconomics` (operator economics — reused verbatim),
- the fidelity-tier vocabulary from `schemas.py`.

V2 imports nothing mutable from V1 and never mutates V1 state. Both run on identical deterministic inputs
for an honest V1-vs-V2 comparison (Phase X / dt60).

## What this PR implements (no core item deferred)

| component | module | ports from | mechanism |
|--|--|--|--|
| `RooflineServingModelV2` | `v2/roofline_serving.py` | InferSim/llm-analysis/LLM-Viewer | live FLOP/bandwidth timing; precision, spec-decode, clock as byte/FLOP/step modifiers |
| `TieredKVStateV2` | `v2/tiered_kv.py` | vLLM/LMCache/Mooncake/Splitwise | GPU_HBM→CPU_DRAM→REMOTE_KV→SSD tiers; eviction; remote-vs-recompute decision; transfer = bytes/bw |
| `PrefillDecodeSchedulerV2` | `v2/prefill_decode_scheduler.py` | Splitwise/DistServe/Sarathi/Orca | shared vs disaggregated pools; phase queues (M/D/1); KV handoff; token-budget continuous batching; chunked prefill |
| `CanonicalWorldStateV2` | `v2/world_state.py` | Aurelius V1 | persistent clone-safe state: servers/racks/replicas + pools + tiered KV |
| `WorldSimulatorV2` | `v2/world_simulator.py` | Aurelius V1 spine | arrival→route/admit→KV tier→roofline timing→batching→handoff→GPU-s/energy/cost→goodput/$ |
| `AdaptiveMPCSearchV2` | `v2/mpc_search.py` | — | exhaustive_cartesian / beam_search / coordinate_descent + search_regret_audit (no silent 256 cap) |
| roofline-aware candidates | `v2/candidate_generator.py` | ROOFLINE_REUSE_DECISION | regime-conditioned **soft prior** (never forces selection) |
| external baselines | `external_sim_validation.py` (extend) | all of the above | InferSim/llm-analysis/LLM-Viewer/Splitwise/DistServe/vLLM/LMCache/Mooncake/BLIS sanity checks |

### MPC action surfaces added (Phase 5)
`precision_mode` (bf16/fp8/int4), `spec_decode_mode` (off/shallow/medium/aggressive),
`clock_power_state` (low/base/high), `colocation_mode` (off/conservative/aggressive — **pruned to off
unless real/trace-derived background work exists**), `prefill_decode_allocation`
(shared/p40_d60/p50_d50/p60_d40 [+p20_d80/p80_d20 diagnostic]).

## Causal-only law (hard rules #4–#8)

Every mechanism affects reward **only** through TTFT, completion latency, queueing, GPU-seconds, energy,
power, memory pressure, bandwidth pressure, capacity, SLA, or cost. There is **no** reward bonus, no
action-specific scalar, no "roofline bonus". The roofline regime is a **soft prior on candidate
generation only**; selection is always by the simulated causal outcome under the unchanged Pareto gate.

Examples of the causal chain:
- `precision=fp8` → halves KV/weight bytes → lowers `memory_time` in the roofline → faster memory-bound
  decode + lower HBM pressure → more batching headroom → higher goodput; **but** an `int4` quality/risk
  penalty raises the risk-adjusted SLA miss, so it only wins where memory pressure dominates.
- `spec_decode` → fewer serial decode steps × acceptance, but extra draft+verify compute → wins only when
  decode is memory-bound *and* acceptance is high (compute headroom exists).
- `clock=low` → lower peak FLOPS + lower power → cheaper energy but slower compute → wins on memory-bound
  decode (compute not binding), loses on compute-bound prefill.
- `disaggregated` → separate pools + KV handoff bytes/bw → helps phase-skewed load, hurts when handoff
  overhead dominates or allocation is wrong.
- `remote KV hit` → transfer bytes/bw + network pressure vs recompute prefill cost → cheapest causal path wins.

## Guard rails (hard rules #2,3,9–12)

- No heavy external repo as a runtime dependency; **port equations/patterns** only; no downloads; deterministic.
- Bias to implementation; SKIPPED only where a conservative approximation would mislead (documented per item).
- Pareto claim gate unchanged; no optimistic legacy baseline used as a fair comparator (the legacy KV scalar
  stays a labelled reference only, as in #107/#110).

## Honest scope notes (first-cut approximations, labelled)

- **Continuous batching** is an *analytical occupancy* approximation (token-budget + Little's law + saturation
  tail), not a per-iteration event loop — sufficient for causal direction, labelled SIMULATOR_INFERENCE.
- **Spec-decode acceptance**, **int4 quality risk**, **co-location background work** are BENCHMARK_DERIVED /
  INFERRED bands; without real telemetry they are conservative and explicitly tiered, never MEASURED.
- **Per-link/NVLink** transfer stays macro (hourly rx/tx), as in V1 — per-link fidelity is PROP (pilot-only).

## Success criterion (from the brief)

V2 succeeds if it runs side-by-side with V1, reproduces the controlled fixtures, and explains action value
with more physical detail. It need **not** beat V1 on gp/$ immediately — it must beat V1 on **causal
fidelity** and explain any difference honestly (Phase X + dt60). V2 does **not** become default unless the
comparison proves it more realistic, deterministic, tested, and not materially slower.

## Deliverables

Code: `aurelius/environment/v2/{__init__,roofline_serving,tiered_kv,prefill_decode_scheduler,world_state,
world_simulator,mpc_search,candidate_generator}.py`; extended `external_sim_validation.py`; scripts
`compare_external_roofline.py` (exists), `compare_kv_tier_fixture.py`, `compare_disaggregation_fixture.py`,
`compare_batching_fixture.py`, `compare_mpc_search_strategies.py`, `run_v1_v2_comparison.py`,
`run_dt60_full_serving_physics.py`. Tests under `tests/` per module. Docs: this plan,
`FULL_SERVING_PHYSICS_CONTROLLED_FIXTURES.md`, `CANONICAL_WORLD_STATE_V2_COMPARISON.md`,
`DT60_FULL_SERVING_PHYSICS_DIAGNOSTIC.md`, `ROOFLINE_MPC_ACTION_CALIBRATION.md`, plus updates to
`PREFILL_DECODE_ECONOMICS_CALIBRATION.md` / `CACHE_LOCALITY_WORLD_MODEL_CALIBRATION.md`.
