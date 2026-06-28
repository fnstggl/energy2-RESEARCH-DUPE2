# Prefill/Decode + Economics Calibration (PR #107)

Per-transition calibration for the prefill/decode serving model (`prefill_decode.py`) and the cost
modes. Every effect flows through prefill work / realized GPU-seconds; nothing touches reward. Validated
by `world_validation.py` (`run_world_validation()` → 27 PASS / 0 FAIL / 3 SKIPPED) +
`tests/test_prefill_decode_economics.py`.

## Transitions

| transition | equation | source / tier | confidence | limitation |
|--|--|--|--|--|
| prefill work | `TTFT_BASE_S + (prompt−prefix_hit)·PREFILL_S_PER_TOKEN + model_cold` | BENCHMARK_DERIVED (vLLM/Sarathi prefill 5–15k tok/s) | medium | one per-token rate; full roofline deferred |
| prefill tokens remaining | `max(prompt − prefix_hit, 0)` | TRACE_DERIVED_REUSE_MODEL (PR #106) | high | positional Mooncake bridge |
| decode work | `out · TPOT_S · batch_factor` | BENCHMARK_DERIVED (`TPOT_S=0.02`, 50 tok/s/seq) | medium | KV-insensitive (correct) |
| TTFT | `prefill_queue + prefill_work` (service-only here; cluster queue from replay) | PUBLIC_PAPER (DistServe/Splitwise) | medium | prefill queue is light-model |
| realized GPU-seconds | `Σ(prefill_work + decode_work)` (+ batching tail past saturation) | SIMULATOR_INFERENCE | medium | shared-GPU prefill/decode interference approximate |
| active decode occupancy | Little's law `Σ decode_work / period` | PUBLIC_PAPER (vLLM continuous batching) | low | mean only |

### Calibration bands
- `PREFILL_S_PER_TOKEN`: 0.00007 / **0.00015** / 0.0004 s/token — BENCHMARK_DERIVED (A100/H100 prefill throughput).
- `TPOT_S = 0.020`, `TTFT_BASE_S = 0.150` — existing BENCHMARK_DERIVED constants (unchanged).
- decode `batch_factor`: conservative 1.0 / balanced 0.92 / aggressive 0.82, + a 10%/over-saturation tail.

## Cost modes (the economic bridge) — equations + claim safety

| mode | billable GPU-seconds | operational meaning | claim safety |
|--|--|--|--|
| `provisioned_capacity` | `provisioned` (capacity integral) | existing behaviour; faster service does NOT cut cost | **reproduces PR #106** (verify) |
| `realized_serving_work` | `max(realized, 0.05·provisioned)` | cost follows realized serving GPU-seconds | **upper-bound counterfactual** — NOT a production claim |
| `hybrid_capacity_work` | `max(realized, 0.5·provisioned)` | provisioned warm-idle floor; realized work above it earns a bounded discount | **defensible default** |

`realized ≤ provisioned` by construction; no mode is free (a warm-idle floor always bills). A headline
is allowed only under `provisioned` or `hybrid` (defensible), vs a fair baseline that **also pays
realistic prefill**.

## Baseline labels (Phase 6 cleanup)
| baseline | label | use |
|--|--|--|
| `fair_oldmodel_no_cache` | reference | the pre-PR constant-prefill model (NOT a fair comparator — too optimistic on long prompts) |
| `fair_phase_no_cache` | **fair** | round_robin, phase model, unique prefixes → realistic prefill, **no reuse** (the apples-to-apples comparator) |
| `legacy_kv_scalar_optimistic` | **unsafe** | offline fleet scalar credits reuse to cold requests — reference only, never a headline |
| `residency_*` | candidate | per-replica KV + phase model (PR #106 + #107) |

## FINAL CLOSURE — production-realism claim table

Every closure-phase mechanism, with status and the honest reason. **No UNKNOWN.**

| # | mechanism | status | basis / why |
|--|--|--|--|
| — | prefill/decode disaggregation (model) | **implemented** | prefill vs decode are separate work terms; KV cuts prefill only |
| — | service-time-sensitive economics | **implemented** | 3 cost modes; realized work drives cost (hybrid/realized) |
| 1 | shared/disaggregated KV pool (HBM/DRAM/SSD) | **deferred** | PUBLIC_PAPER (LMCache); needs a tier model — a conservative single-tier (HBM) is what we have; multi-tier would need transfer-latency calibration we can add next, not misleading to omit |
| 2 | remote KV transfer latency/bandwidth | **deferred** | PUBLIC_PAPER (KV-BW band exists); not material at single-tier; would need topology-pressure-scaled transfer |
| 3 | staged migration (bulk+dirty+sync) | **deferred** | Llumnix; PR #105 models move cost + KV-preserved fraction; staged timing is a refinement, omitting it is conservative (we already charge a move cost) |
| 4 | block-granular prefix caching | **implemented** | `StatefulKVCache` is block/hash-granular (PR #106); one-token divergence preserves only common ancestors |
| 5 | multi-model/LoRA/precision residency | **partial** | model_id + model-switch cold-start implemented (PR #106); adapter/precision deferred (BENCHMARK_DERIVED available, not material to single-model Azure) |
| 6 | tenant/region cache-sharing boundaries | **partial** | model_id already gates cross-model reuse; explicit tenant_id deferred — single-tenant Azure, so omitting does not overstate hit rate |
| 7 | memory fragmentation / allocator pressure | **deferred** | SIMULATOR_INFERENCE; finite LRU cache proxies memory pressure; explicit fragmentation would be inference-on-inference — omitting is more truthful than a fabricated fragmentation curve |
| 8 | active-sequence memory reservation / max concurrency | **partial** | concurrency cap from cache capacity + decode occupancy; explicit max-active-seq reservation deferred |
| 9 | prewarm forecast uncertainty (FP/FN) | **deferred** | ForecastTrajectory quantiles exist; a prefix-reuse forecast for prewarm is the next channel (Phase 6 prefix-prewarm) |
| 10 | autoscaling lifecycle (scale-up/drain/cooldown) | **partial** | warm pool + idle-timeout cooldown (PR #102/#105); scale-up delay = cold-start (modeled); explicit drain deferred |
| 11 | cancellation / timeout / early EOS | **deferred** | TRACE: Azure output lengths drive decode; cancellation needs a cancellation trace we do not have — modeling it would be fabricated |
| 12 | fair baseline cleanup (no optimistic scalar) | **implemented** | `fair_phase_no_cache` added; legacy scalar relabeled unsafe |
| 13 | validation matrix PASS/WARN/FAIL/SKIPPED | **implemented** | `world_validation.py` 27/0/3 |
| 14 | final claim table | **implemented** | this table |

**SKIPPED-with-reason** (would be misleading to approximate, or need absent telemetry): remote-KV
multi-tier transfer (needs per-tier bandwidth telemetry), cancellation (needs a cancellation trace),
fragmentation (inference-on-inference). Each SKIP states the minimum telemetry that unblocks it. No
other simulator path silently depends on these (validated: cost modes + phase model run without them).
