# Cache / Locality / Prewarm / Migration Physics Audit (PR #106, Phase 1)

The complete domain, each mechanism classified by fidelity (`TRACE_DERIVED` · `TRACE_DERIVED_REUSE_MODEL`
· `BENCHMARK_DERIVED` · `PUBLIC_PAPER` · `SIMULATOR_INFERENCE` · `ABSENT`) and marked **implemented** /
**deferred** in this PR. The chain built here: request prefix locality → per-replica KV/model residency →
routing/placement/migration → cache hit depth / model warmness → **service time / TTFT** → SLA-safe
goodput/$ (all causal — `world_serving.py`). Every benefit flows through service time, never a bonus.

## A. Request locality physics
| mechanism | fidelity | this PR |
|--|--|--|
| exact / partial prefix reuse, reuse depth | TRACE_DERIVED (Mooncake `hash_ids`) | **implemented** (per-request, causal) |
| reuse distance / conversation recurrence | TRACE_DERIVED_REUSE_MODEL | **implemented** (cursor over the pool across periods) |
| shared system prompts / RAG / agent context | TRACE_DERIVED_REUSE_MODEL | implemented via Mooncake reuse process (not separately tagged) |
| Azure prompt/output lengths/timing | TRACE_DERIVED (Azure) | **implemented** (Azure supplies these; no row-join with Mooncake) |

## B. KV / cache physics
| mechanism | fidelity | this PR |
|--|--|--|
| paged block residency, exact/partial hits, LRU eviction | TRACE_DERIVED + BENCHMARK_DERIVED (vLLM PagedAttention 16-token blocks) | **implemented** (`StatefulKVCache` per warm replica) |
| prefill tokens avoided → TTFT/service reduction | BENCHMARK_DERIVED (`PREFILL_SAVINGS_FRAC=0.9`) | **implemented** (`service_factor = 1 − 0.9·prefix_frac`) |
| HBM/KV capacity + memory pressure → eviction | BENCHMARK_DERIVED (GPU mem) / INFERRED (budget split) | **implemented** (finite cache; concentration thrashes) |
| cache transfer on migration | PUBLIC_PAPER (Llumnix/KV-BW) | partial (KV preserved-fraction on move; staged dirty-block transfer **deferred**, 8B-C) |
| CPU/NVMe offload, multi-tier remote KV | PUBLIC_PAPER (LMCache/CacheBlend) | **deferred** (8B-B) |

## C. Model warmness physics
| mechanism | fidelity | this PR |
|--|--|--|
| model weights loaded / engine ready | BENCHMARK_DERIVED (cold-start decomposition, PR #105) | **implemented** (`model_id`, `weights_loaded`) |
| model-switch cold-start (affinity channel) | BENCHMARK_DERIVED (genai medians) | **implemented** (ported `genai_effective_service_s`: a mismatch reloads weights, invalidates KV) |
| adapter/LoRA, precision, TP-size residency | BENCHMARK_DERIVED / SIMULATOR_INFERENCE | **deferred** (8B-E) |
| CUDA-graph/kernel warmup | SIMULATOR_INFERENCE | ABSENT (not modelled) |

## D. Routing & placement physics
| mechanism | fidelity | this PR |
|--|--|--|
| route to best prefix/model locality vs queue/mem/topology | SIMULATED (router) | **implemented** (`kv_aware` exploits both channels; round_robin/shortest_queue baselines) |
| same-rack vs cross-rack macro penalty | TRACE_DERIVED (v2026 rx/tx) | **implemented** (per-replica topology in the residency score; macro only) |
| cross-rack KV-transfer cost | PUBLIC_PAPER (KV-BW) | **deferred** (8B-C / Gap 3) |

## E. Migration physics
| mechanism | fidelity | this PR |
|--|--|--|
| move replica identity (no duplication), preserve model + KV fraction | PUBLIC_PAPER (Llumnix) | **implemented** (PR #105 + residency cache moves with the replica) |
| temporary capacity loss, move cost, benefit only via future service | PUBLIC_PAPER / BENCHMARK_DERIVED | **implemented** |
| staged dirty-block copy, final-sync pause, bandwidth sharing | PUBLIC_PAPER (Llumnix) | **deferred** (8B-C) |

## F. Prewarm physics
| mechanism | fidelity | this PR |
|--|--|--|
| pre-load model, warm-hold cost, future cold-start avoidance | BENCHMARK_DERIVED | **implemented** (PR #105) |
| pre-position hot prefixes from a causal reuse forecast | TRACE_DERIVED_REUSE_MODEL | **deferred** (8B / Phase 6 prefix-prewarm — needs the synthetic reuse forecast) |
| memory reservation / eviction opportunity cost | INFERRED | partial (cache capacity is finite; explicit reservation deferred) |

## Phase 8B — additional production-serving physics (audit + status)
| item | fidelity | this PR |
|--|--|--|
| A. prefill/decode disaggregation (TTFT vs total) | PUBLIC_PAPER (Splitwise/DistServe) | **deferred** — the *named next mechanism* (see results doc): without it a TTFT win isn't a goodput win |
| B. shared/remote KV tiers (HBM/host/rack/cross-rack) | PUBLIC_PAPER (LMCache) | **deferred** (validation stub present) |
| C. staged dirty-block KV migration | PUBLIC_PAPER (Llumnix) | **deferred** |
| D. block-granular prefix hashing (parent-hash, partial blocks) | TRACE_DERIVED (Mooncake) | **implemented** (`StatefulKVCache` is block/hash-granular; one-token divergence preserves only common ancestors) |
| E. multi-model/adapter/precision residency | BENCHMARK_DERIVED | partial (model_id implemented; adapter/precision deferred) |
| F. HBM fragmentation / allocator pressure | SIMULATOR_INFERENCE | **deferred** (cache capacity proxies memory; fragmentation deferred) |
| G. cache-aware admission (future-reuse value) | SIMULATOR_INFERENCE | **deferred** (must flow through future service, no bonus) |
| H. streaming decode / cancellation | TRACE_DERIVED (Azure output lengths) | **deferred** (output tokens drive decode today; cancellation deferred) |
| I. autoscaling / replica lifecycle | BENCHMARK_DERIVED | partial (warm pool + idle-timeout cooldown from PR #102/#105; scale-up delay deferred) |
| J. tenant / residency cache-sharing eligibility | SIMULATOR_INFERENCE | **deferred** (single-tenant Azure; model_id already gates cross-model reuse) |
| K. eviction policy variants (LRU/LFU/TTL) | HEURISTIC | LRU implemented; variants **deferred** |
| L. forecast uncertainty for prewarm/migration | SIMULATOR_INFERENCE | partial (ForecastTrajectory quantiles exist; prefix-reuse forecast deferred) |
| M. non-cache interference (tokenizer CPU, NIC) | SIMULATOR_INFERENCE | **deferred** (bounded terms, only if sourced) |
| N. cache retention opportunity cost | SIMULATOR_INFERENCE | partial (finite cache: retention competes via eviction; explicit batching-headroom cost deferred) |

## Honest scope summary

**Implemented (the payoff channel + its honest limits):** per-replica KV-prefix locality and
model-affinity, both causal through service time, block-granular, with finite-cache eviction, persistent
across periods, moved by migration, with the macro topology penalty — proven in 12 fixtures + 6
validation checks. **Deferred (audited, with sources + the reason):** prefill/decode disaggregation
(the named blocker for converting TTFT wins to goodput), remote KV tiers, staged migration, roofline
batching, and the smaller 8B refinements. The deferral is sequencing, not hand-waving: each has a
fidelity tag, a source, and a place in the build order, and the validation suite marks the unbuilt ones
`SKIPPED` with reasons.
