# Cache-Locality World-Model Calibration (PR #106)

Per-transition calibration for the residency serving channel (`world_serving.py`). Every benefit flows
through **service time / TTFT**; none through a reward bonus. Validated by `world_validation.py`
(`run_world_validation()` → 21 PASS / 0 FAIL / 3 SKIPPED) and `tests/test_cache_locality_physics.py`.

## Transition 1 — prefix hit → prefill skipped → service time

**Equation.** For a request routed to replica *r*: `prefix_frac = matched_leading_blocks / request_blocks`
(block = 16 tokens, vLLM PagedAttention); `service_factor = (1 − PREFILL_SAVINGS_FRAC · prefix_frac) ·
topo + lookup_overhead`, with `PREFILL_SAVINGS_FRAC = 0.9` (prefill share of service that a hit skips —
same constant as the offline `kv_cache.fleet_kv_routing`), `lookup_overhead = 0.002` (hashing cost; can
erase a marginal hit on tiny prompts).
**Source.** Mooncake `hash_ids` block reuse — TRACE_DERIVED; prefill-savings fraction — BENCHMARK_DERIVED
(vLLM/PagedAttention prefix-cache). **Validation:** full hit ≈ 0.094, partial strictly between, cold ≈ 0.92.
**Limitation:** uses one prefill/decode share; true split is sequence-length-dependent (Phase 8B-A).
**Telemetry needed:** per-request TTFT-with/without-cache.

## Transition 2 — request→prefix assignment (TRACE_DERIVED_REUSE_MODEL)

**Model.** Azure supplies `(arrival, out_tok, in_tok)`; the Mooncake `hash_ids` sequence supplies the
**reuse process**, assigned by **position** (request *k* → `pool[(cursor+k) % |pool|]`, cursor advancing
across periods) — **no row-join**, no key matched across traces. Preserves Mooncake's exact/partial reuse,
reuse depth, and reuse distance. Deterministic. **Source:** Mooncake (committed) — TRACE_DERIVED_REUSE_MODEL.
**Validation:** Azure tokens/timing preserved; hashes attached positionally; reusing stream → hits > 0;
no future leakage (request 0 cannot hit). **Limitation:** the join is positional, so the *correlation*
between an Azure request's size and its prefix is synthetic; only the reuse *dynamic* is real.
**Telemetry needed:** prefix hashes on the production serving trace itself.

## Transition 3 — model-affinity → switch cold-start (ported genai)

**Equation.** Routing to a replica whose `model_id ≠ request.model_id` adds `model_load_s` (the
`cold_start_model_load_s` band base, 22 s) to that request's service time and **invalidates** the
replica's KV (the weights changed); the replica adopts the new model. `kv_aware`/`affinity` routing
scores a model match (+`len(prefix)+1`) so it co-locates models and avoids switches.
**Source.** `aurelius/optimizer/policies/genai_serving.py:genai_effective_service_s` (Alibaba GenAI
cold-start medians) — BENCHMARK_DERIVED. **Validation:** a mismatch adds 22 s + clears the cache; a
multi-model stream has fewer switches under model-aware routing than round-robin; a single-model stream
has zero switches (the Azure regime). **Limitation:** amortisation is per-request (not the tick-level
`distinct_models/n` of the original); adapters/precision not yet distinguished (8B-E).

## Transition 4 — finite cache + eviction (no free hit)

**Model.** Each warm replica holds a `StatefulKVCache(capacity_blocks)` with LRU eviction; admitting a
request's blocks evicts the LRU pages when full. Over-concentrating reuse on few replicas thrashes their
caches (evictions ↑, hit rate ↓) — the natural penalty that lets `shortest_queue` beat cache-affinity
when the working set exceeds capacity. **Source.** GPU HBM capacity — BENCHMARK_DERIVED; LRU — HEURISTIC
(deployable default; trace does not expose the real policy). **Validation:** small cache → many evictions;
an evicted prefix returns to a miss; a tiny-cache concentration test. **Limitation:** LRU only;
fragmentation/allocator overhead deferred (8B-F).

## Transition 5 — per-replica topology in routing (macro only)

**Model.** A replica's rack macro pressure (v2026 rx/tx) discounts a hit on a low-pressure rack
(`topo = 1 − TOPOLOGY_MAX_DISCOUNT·(1 − rack_pressure)`, `TOPOLOGY_MAX_DISCOUNT = 0.08`); routing
subtracts the topology penalty, so placement/migration to low-pressure racks helps **only** when the
pressure spread is real. **Source.** v2026 `network_hourly` — TRACE_DERIVED. **No** per-link/NVLink claims
(ABSENT from any trace). **Validation:** flat pressure → ~no placement benefit. **Limitation:** macro
only; cross-rack KV-transfer cost deferred (8B-C).

## Persistence & isolation (the channel prewarm/migration act through)

Each warm replica's cache is attached as `_kv_cache` and **persists across periods**; a **scoring**
(`mutate=False`) pass routes on **copies** so the rollout's risk/point double-eval never pollutes the
real residency; a **committed** (`mutate=True`) pass writes through. Migration moves the cache with the
replica; cooldown clears it. **Validation:** scoring leaves the persistent caches empty;
clone isolation; deterministic replay (PR #105/#106 suites).

## Calibration honesty

No constant here was chosen to make an action win. `PREFILL_SAVINGS_FRAC` matches the pre-existing
offline value; `model_load_s` is the existing cold-start band; `TOPOLOGY_MAX_DISCOUNT` is the existing
TRACE_DERIVED cap. The held-out diagnostic shows **no Pareto-safe gain** (results doc) — the realistic
per-replica model in fact *lowers* the headline vs the optimistic uniform scalar, which is a fidelity
gain, not a tuned win.
