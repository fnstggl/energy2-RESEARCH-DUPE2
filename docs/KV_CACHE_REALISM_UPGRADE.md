# KV-Cache / Prefix-Affinity / Memory-Pressure Realism Upgrade

Status: **simulator-only** realism upgrade. All outputs carry `is_sandbox=True`
and are excluded from economic claims. Real/customer environments remain
`recommendation_only`. This document is deliberately conservative — it does
**not** claim production accuracy; it claims the simulator's *dynamics* are now
operationally believable.

---

## 1. Architectural diff

New modules:

| File | Purpose |
|---|---|
| `aurelius/simulation/cluster/kv_cache.py` | Pure, deterministic KV-memory / prefix / locality / preemption functions. All magnitudes from calibration; all randomness caller-supplied (seedable). |
| `aurelius/simulation/cluster/cache_model.py` | Explicit mutable state models: `KVCacheState`, `KVPressureState`, `PrefixCacheState`, `CacheWarmupState`, `LocalityConfidenceState`, `RoutingLocalityState`, `CacheFragmentationState`, `CacheEvictionState`, `PreemptionState`, `CacheAffinityState`, aggregated into `WorkloadCacheState`. |

Changed modules:

| File | Change |
|---|---|
| `calibration.py` | New `KV_CACHE_PARAMS` registry (20 calibrated params), `MODEL_KV_PROFILES` (KV architectures w/ provenance), `kv_value()`, `resolve_kv_profile()`, `model_profile_table()`; `calibration_table()` now returns serving **and** KV groups. |
| `model.py` | `SimWorkload` gains `model_kv_profile`, `prefix_overlap`, `avg_seq_len_tokens`, event override fields, and a `cache: WorkloadCacheState`. `SimQueue` gains `kv_pressure`, `kv_pressure_region`, `preemptions_total`, `cache_fragmentation_frac` telemetry. |
| `engine.py` | New `_update_kv_cache()` (runs **before** `_update_queues`); `_update_queues()` now uses cache-aware batching, prefix-reuse prefill savings, KV-pressure TTFT amplification, and cold-route + recompute penalties; `migrate_workload()` prices the cold reroute; `get_cluster_state()` emits `preemptions_total`; `TickMetrics` gains 12 cache KPIs. Old `_update_cache_proxy()` removed. |
| `scenarios.py` | New validation scenarios `prefix_affinity_energy_arbitrage`, `kv_exhaustion_preemption_storm`. |
| `report.py`, `constraint_runner.py` | `TickKPI`/`AggregatedKPI` carry the new cache KPIs through the benchmark. |

Tick order (the key determinism change):

```
… → _update_thermal → _update_kv_cache → _update_queues → _update_cost_accounting
```

`_update_kv_cache` computes this tick's pressure/hit-rate/preemption from **last
tick's** offered concurrency (`cache.active_seqs_prev`), then `_update_queues`
consumes them and stores the new concurrency for next tick. This keeps the
pre/post ordering deterministic under a fixed seed.

---

## 2. KV-state model diagram

```
                 SimWorkload.cache : WorkloadCacheState
   ┌───────────────────────────────────────────────────────────────────┐
   │ KVCacheState        allocated_bytes, reserved_budget, batch, seq    │
   │     │  KV_bytes = batch · seq · layers · kv_heads · head_dim · 2 ·  │
   │     │             bytes_per_elem          (GQA/MQA via kv_heads)    │
   │     ▼                                                               │
   │ KVPressureState     pressure = allocated / reserved_budget          │
   │     │   region ∈ { LOW <0.70 | ELEVATED | THROTTLING ≥0.90 |        │
   │     │              PREEMPTION ≥0.97 }      (operational heuristics) │
   │     ├──────────────► PreemptionState   p(preempt)↑ near 1.0         │
   │     │                  recompute_penalty_ms = preempted·ctx·k       │
   │     ├──────────────► CacheEvictionState  LRU under pressure         │
   │     ├──────────────► CacheFragmentationState  block slack (paged)   │
   │     ▼                                                               │
   │ PrefixCacheState    hit = sigmoid(a·(overlap−b)) · locality         │
   │     │   prefill_savings = hit · cap   (shrinks TTFT prefill ONLY)   │
   │     ▼                                                               │
   │ LocalityConfidenceState  logistic warm-up if reused; decay if not   │
   │     │                                                               │
   │ CacheWarmupState (reuse-driven, NOT time-driven)                    │
   │ RoutingLocalityState  home_region, home_gpus, affinity_score, tier  │
   │ CacheAffinityState  cold_reroute_count, cold_route_penalty_ms       │
   └───────────────────────────────────────────────────────────────────┘

   TTFT = queue_wait
        + (prefill·(1−prefill_savings) + active_seq_contention + kv_stall)
              · pressure_ttft_multiplier
        + cold_route_penalty_ms            (lost reusable prefill on a reroute)
        + recompute_penalty_ms             (preempted sequences re-prefilled)
```

---

## 3. Cache-affinity routing model

A reroute/migration is **cache-aware**. Two layers:

1. **Physics (always on, all policies).** `migrate_workload()` prices the cold
   reroute: `lost_prefill_tokens = shared_prefix_tokens · hit_rate_before`,
   `cold_route_penalty_ms = lost_prefill_tokens · prefill_cost_per_token`. It
   resets locality confidence to `cold_route_confidence` (≈0.05) and starts a
   reuse-driven re-warm. So **any** policy that reroutes pays the cache cost; a
   churny policy pays it repeatedly.

2. **Policy helper (opt-in).** `kv_cache.should_preserve_affinity(...)` blocks a
   reroute when `expected_cache_loss_ms > expected_queue_gain_ms`. Preserve when
   overlap is high, prefixes long, queue gains modest, confidence high. Break
   only under severe imbalance/overload or when locality confidence is already
   low (nothing warm to protect). Telemetry-tier damping
   (`routing_aggressiveness`) makes routing more conservative as KV/cache
   visibility drops — missing telemetry **lowers** confidence; it is never read
   as "no pressure".

---

## 4. Calibration table

20 KV-cache parameters, every one carrying `{value, source, source_type,
confidence, calibration_notes}` (see `KV_CACHE_PARAMS` in `calibration.py`;
inspect at runtime via `calibration_table()`). Highlights:

| name | value | source_type | confidence |
|---|---|---|---|
| kv_bytes_per_elem | 2.0 (FP16; 1.0 FP8; 0.5 FP4) | documented | high |
| kv_reserved_budget_frac | 0.80 | inferred | medium |
| kv_block_size_tokens | 16 | documented | high |
| kv_pressure_elevated | 0.70 | inferred | low |
| kv_pressure_throttling | 0.90 | inferred | low |
| kv_pressure_preemption | 0.97 | inferred | low |
| kv_pressure_ttft_max_mult | 4.0 | heuristic | low |
| kv_pressure_batch_floor | 0.35 | inferred | low |
| preemption_prob_max | 0.6 | heuristic | low |
| recompute_ms_per_token | 0.30 | inferred | low |
| prefix_hit_sigmoid_a | 8.0 | heuristic | low |
| prefix_hit_sigmoid_b | 0.45 | heuristic | low |
| prefix_max_prefill_savings_frac | 0.85 | inferred | medium |
| locality_confidence_growth | 0.35 | inferred | low |
| locality_confidence_decay | 0.15 | heuristic | low |
| locality_confidence_init | 0.5 | heuristic | low |
| cold_route_confidence | 0.05 | inferred | medium |
| prefill_cost_per_token_ms | 0.25 | benchmark_derived | low |
| telemetry_missing_routing_damp | 0.5 | heuristic | low |

Model KV profiles (`MODEL_KV_PROFILES`, `model_profile_table()`): `llama3-8b`
(GQA, 8 KV heads), `llama2-7b` (MHA, 32 KV heads), `llama3-70b` (GQA, 80
layers), `mistral-7b` (GQA), `mqa-7b` (single KV head). Layer/head counts are
config-card values (`documented`); they are **not** measured serving numbers.

Every value is overridable per run via `serving_config`
(e.g. `{"kv_pressure_throttling": 0.85, "kv_bytes_per_elem": 1.0}`) — the
FP8/FP4 quantization lever and pressure thresholds are configurable priors.

---

## 5. Source-confidence table

| confidence | count | meaning |
|---|---|---|
| high | 3 | element sizes, block size — these are definitional/documented constants |
| medium | 4 | budget fraction, prefill-savings cap, cold-route confidence, model layer counts |
| low | 13 | pressure thresholds, curve shapes, penalty magnitudes — engineering priors |

**Honest grade: the realism is qualitative, not quantitative.** The thresholds
(0.70/0.90/0.97) and curve shapes are *operational heuristics inferred from
documented vLLM PagedAttention/preemption behaviour*, not fitted to a live
cluster. They make the dynamics believable; they do not assert production
accuracy. None are MEASURED.

---

## 6. Realism-gap report

What is now modelled (was missing/over-simplified before):

- KV memory **scaling law** with GQA/MQA via `kv_heads` (not hidden-size); KV
  quantization lever (FP16/FP8/FP4).
- KV **pressure** as first-class state with four operational regions.
- **PagedAttention** internal block slack (correctly *not* heap fragmentation).
- Prefix hit rate as `sigmoid(overlap) · locality` — reuse gated by **both**
  content overlap and routing locality.
- **Cold-reroute penalty** (lost reusable prefill) + reuse-driven warm-up/decay.
- **Preemption/recompute** under exhaustion; eviction; decode instability.
- **Cache-aware batching**; TTFT decomposition with KV amplification + penalties.
- **Telemetry-confidence tiers** → routing aggressiveness; missing telemetry
  lowers confidence (never implies "no pressure").

What is still a proxy (the gap):

- Magnitudes in the **collapse region** (preemption storms) are directionally
  right but quantitatively uncalibrated — `kv_exhaustion_preemption_storm`
  produces multi-minute TTFT p99 that signals "timeout/collapse", not a precise
  number.
- Single representative prompt length per workload (heavy-tailed length
  distribution not modelled).
- Cold-route penalty prices lost prefill only — batch-packing churn and
  scheduler thrash after a reroute are under-priced (penalty is a lower bound).
- Orphaned-queue mechanic: a workload migrated away from its queue's region
  leaves that queue under-served (crude proxy for "replicas moved away").

---

## 7. Before / after benchmark comparison

`prefix_affinity_energy_arbitrage`, 24 ticks, seed 42 (high overlap, migratable):

| policy | migrations | cold reroutes | prefix hit | locality conf | TTFT p99 (ms) | energy cost | SLA viol |
|---|---|---|---|---|---|---|---|
| fifo (stays warm) | 0 | 0 | 0.90 | 0.93 | 107 | 2.58 | 300 |
| **greedy_energy** | 2 | 2 | **0.08** | **0.13** | **1132** | 1.90 | 70 |
| constraint_aware | 0 | 0 | 0.90 | 0.93 | 1372* | 3.95 | 300 |

Naive energy-greedy saves ~26% energy but **destroys its prefix cache** (hit
0.90 → 0.08, locality 0.93 → 0.13) and pays >10× TTFT-prefill on cold routes.
*(constraint_aware's TTFT p99 here is a transient from a non-migration action,
not cache loss — its cache stays warm identically to fifo.)*

`kv_exhaustion_preemption_storm`, 24 ticks, seed 42 (long context, thin KV):

| KPI | value |
|---|---|
| KV pressure (max) | 1.5 (saturated → over-demand) |
| preemptions (total) | 392 |
| prefix hit (mean) | 0.08 |
| TTFT p99 | ~2.25M ms (timeout/collapse) |

KV exhaustion now produces an emergent **preemption storm** with decode
collapse — the simulator can no longer pretend pressure only affects throughput.

**Before this upgrade** these scenarios produced stable p50 and a fixed-multiple
p99; rerouting was nearly free and energy arbitrage looked unconditionally
profitable.

---

## 8. New failure modes introduced

- **Cache collapse / cold-route disaster** — repeated rerouting collapses prefix
  hit rate and spikes TTFT.
- **Preemption storms** — KV pressure → 1.0 triggers preemption + recompute, p99
  tail collapse, timeout risk.
- **Locality thrash** — churny migration keeps locality confidence pinned low.
- **Decode instability under memory pressure** — pressure thins batches and
  amplifies the active-sequence TTFT component.
- **Telemetry-blind over-routing** — guarded by tier damping, not silently
  treated as healthy.

## 9. Strategies that now fail realistically

- **Naive energy arbitrage / greedy rerouting** — wins on energy, loses on
  TTFT/p99 via cold-route cache loss (demonstrated above).
- **Aggressive autoscale-by-spreading** — more replicas thin batches *and*
  dilute cache locality; cache-aware batching now penalizes it.
- **"p50 is fine, ship it"** — p50 can look healthy while preemption-driven p99
  collapses; the benchmark surfaces TTFT p99 and preemption counts.

## 10. Remaining realism limitations

1. No value is MEASURED against a live cluster; thresholds/curves are priors.
2. Collapse-region magnitudes are uncalibrated (directional only).
3. Single prompt length per workload (no heavy-tailed distribution).
4. Cold-route penalty is a lower bound (packing/scheduler thrash under-priced).
5. Orphaned-queue mechanic is a crude stand-in for replica relocation.

## 11. Honest readiness assessment

**Believable, not validated.** The simulator now exhibits the right *qualitative*
couplings — KV pressure ↔ preemption ↔ TTFT tails, prefix overlap ↔ locality ↔
reuse, reroute churn ↔ cache loss ↔ p99 — so naive arbitrage can fail,
cache-aware orchestration matters, and locality preservation is economically
meaningful. Estimated savings are **substantially more trustworthy** as
*directional* signals because the dominant downside risks (cache loss, TTFT
spikes, preemption) are no longer invisible.

It is **not** a validated quantitative predictor. Before any economic claim,
the `low`-confidence parameters (pressure thresholds, curve shapes, penalty
magnitudes) must be calibrated against real vLLM telemetry: KV-usage vs TTFT
curves, preemption counts, prefix hit-rate vs overlap scatter, and post-reroute
hit-rate recovery. Until then: treat outputs as believable scenario dynamics,
not production forecasts.
