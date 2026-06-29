# Production-Default Decisions

Every serving-physics mechanism, with its canonical-default status and a **specific technical** justification
for anything not default-on. Per policy, *"to preserve benchmark compatibility"* alone is **not** a sufficient
reason — each non-default row names a concrete blocker (architecture gap, weak calibration that would mislead,
or missing telemetry). Status legend: **ON** = canonical default; **OFF (config)** = implemented, opt-in;
**Legacy-only** = retained solely for explicit regression reproduction; **Absent** = not implemented in V1.

| Mechanism | Default ON | Default OFF (config) | Legacy-only | Why | Validation evidence | Remaining blocker | Required telemetry |
|--|:--:|:--:|:--:|--|--|--|--|
| **Roofline GPU/model base timing** | ✅ | | | Production-realistic, BENCHMARK_DERIVED (public specs+arch), deterministic, clone-safe, fixture-validated. **Now the default** in `compute_phase_serving` / `simulate_period`. | `test_v1_roofline_timing.py` (24); `world_validation` 45 PASS; `compare_v1_legacy_vs_v1_roofline.py` (H100 phantom SLA 0.700→0.092) | — | real per-(GPU,model) MFU for tighter floor |
| **GPU/model-aware prefill timing** | ✅ | | | Same roofline path (compute-bound leg). | `test_roofline_70b_decode_slower_than_8b`; `_prefill_decode_checks` | — | — |
| **GPU/model-aware decode timing** | ✅ | | | Same roofline path (memory-bound leg); fixes the L40S-class scalar. | `test_roofline_h100_decode_faster_than_l40s` | — | — |
| **Arithmetic-intensity compute/memory classification** | ✅ | | | `roofline_analyze` ridge-point; drives regime-aware candidates. | `_roofline_checks`; `roofline_external` tests | — | — |
| **Timing provenance labels** | ✅ | | | `TIMING_PROVENANCE` + `timing_model` in `PhaseResult.summary()`. | `test_provenance_labels_present` | — | — |
| **Legacy scalar timing** | | | ✅ | Retained ONLY as `timing_model="legacy_scalar"` / `AURELIUS_TIMING_MODEL=legacy_scalar` for bit-for-bit regression reproduction. | `test_explicit_legacy_is_bit_for_bit_scalar` | — | — |
| **Precision (bf16/fp8/int4)** | ✅* | | | CONNECTED action; at neutral policy factor=1.0, the MPC selects it causally. *On as an *available* action, neutral by default. int4 carries quality-risk (never free). | `_roofline_action_checks` (fp8 helps memory-bound; int4 quality risk) | — | per-task accuracy-vs-precision (for int4 risk magnitude) |
| **Speculative decoding** | ✅* | | | CONNECTED action; helps memory-bound decode, pays compute tax / hurts compute-bound. | `spec_latency_win_pays_compute_tax`, `spec_hurts_compute_bound` | — | measured acceptance per draft/target pair |
| **Clock / DVFS (power)** | ✅* | | | CONNECTED action; compute leg + power, not bandwidth. Energy via `power_scale`. | `clock_changes_power_not_memory_bw` | — | measured power-vs-clock curve per GPU |
| **Roofline-aware batching factor** | ✅ | | | Existing `BATCH_DECODE_FACTOR` + saturation tail is the canonical batching model. | `test_batching_changes_decode_work` | — | continuous-batching throughput/latency curve |
| **Continuous-batching token budget** | | ✅ | | A *different* batching mechanism (vLLM/Orca token budget + active-seq accounting) than V1's factor model. Implemented in V2 only; promoting it into V1 would replace the calibrated factor model → needs its own validation. | V2 fixtures (`test_v2_serving_physics`) | V1 has no token-budget scheduler; integrating it is an additive scheduler change, not a default flip | continuous-batching iteration trace |
| **Chunked prefill** | | ✅ | | V1 has no chunked-prefill path; the V2 model reduces a decode-stall term that V1's analytical queue doesn't represent. | V2 fixtures | requires a per-iteration scheduler in V1 (architecture) | Sarathi-style iteration trace |
| **Prefill/decode pool allocation** | | ✅ | | SIMULATED_ONLY/frozen: **the live cluster replay has no disaggregated pools**. Default-on would model a capacity split that does not exist in V1 → misleading. | `connected_live_simulated_frozen_split` (frozen with reason) | V1 world state has no prefill/decode pools (architecture) | a real disaggregated-pool deployment |
| **Tiered KV cache (HBM/CPU/remote/SSD)** | | | | **Absent in V1**: `StatefulKVCache` is a single GPU-HBM LRU tier. The multi-tier cascade + remote-vs-recompute is new state/physics. | V2 `tiered_kv` fixtures; `world_validation` SKIPPED rows (multi-tier remote cache) | architectural migration (new tier state in `world_state`) → separate PR | real per-replica KV residency + cross-node transfer bandwidth |
| **Remote-vs-recompute KV decision** | | | | Depends on tiered KV (above). | V2 fixtures | same as tiered KV | same as tiered KV |
| **Cache lookup overhead** | ✅ | | | Already in `world_serving` (`LOOKUP_OVERHEAD_S`). | `_kv_residency_checks` | — | — |
| **Cache transfer overhead** | | ✅ | | Cross-rack/tier transfer cost is SKIPPED in `world_validation` (no tier model in V1). | `world_validation` SKIPPED (cross-rack KV transfer) | needs tiered KV | production transfer bandwidth |
| **Cache pollution / eviction pressure** | ✅ | | | LRU + memory-pressure eviction in `StatefulKVCache`. | `test_kv_cache` (eviction) | — | — |
| **Co-location** | | ✅ | | SIMULATED_ONLY/frozen: **no background-work trace**. Default-on credits ZERO useful work and only interference → would understate or invent value. Implemented but off unless `background_work_gpu_seconds` supplied. | `no_imaginary_background_goodput` | no background-work stream in public traces | a batch/flex job stream with class + SLA |
| **Routing (round_robin/shortest_queue/kv_aware)** | ✅ | | | CONNECTED; kv_aware is the canonical routing. | `_kv_residency_checks`; routing parity tests | — | — |
| **Placement / migration / prewarm** | ✅ | | | CONNECTED persistent-state actions (PR #99–#107). | `_conservation_checks`, `_migration_realism_checks` | — | — |
| **Production candidate generation (regime-aware)** | ✅ | | | `search_planner` prunes int4/spec to the regime that can use them; pruning is logged + auditable. | `candidate_pruning_is_regime_aware` | — | — |
| **Adaptive MPC search (beam/exhaustive + regret)** | ✅ | | | `AdaptiveSearchPlanner` reports raw/evaluated/strategy/**estimated regret** vs exhaustive — never a silent cap. Already wired in the controller. | `search_regret_is_measured` | — | — |
| **Production diagnostics + provenance fields** | ✅ | | | `timing_model`, roofline regime, prefill/decode seconds, queue, KV diag, search regret all surfaced. | `PeriodOutcome` diagnostics; `SearchReport.to_dict()` | — | — |

\* Precision/spec/clock are *connected actions*: available and chosen causally by the MPC, neutral (factor
1.0) at the default policy. They were promoted in earlier PRs and are not gated off.

## Summary of non-default mechanisms (the only ones not ON)

- **Legacy scalar timing** — Legacy-only by design (explicit regression mode). Concrete reason: it is the
  *old, less realistic* model; keeping it default would contradict the production-physics objective.
- **Continuous-batching token budget / chunked prefill** — OFF: V1 has no per-iteration scheduler; these are
  additive scheduler mechanisms (architecture), not a default flip. Tracked for a follow-up PR.
- **Prefill/decode pools** — OFF (frozen): the live replay has no disaggregated pools; default-on would model
  capacity that doesn't exist. Needs a pool-aware world state.
- **Tiered KV / remote-vs-recompute / cross-tier transfer** — Absent: V1 has a single GPU-HBM LRU tier;
  multi-tier is an architectural migration requiring per-replica residency telemetry. Separate PR.
- **Co-location** — OFF (frozen): no background-work trace; default-on would invent or understate goodput.

Every non-default row has a **specific technical blocker** (architecture gap or missing telemetry), not mere
benchmark preservation.
