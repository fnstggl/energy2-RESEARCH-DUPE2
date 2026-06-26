# Canonical Multi-Plane Environment — built from first principles — 2026-06-26

> The bespoke, trace-grounded training/evaluation environment for Aurelius. NOT a
> wrapper over the old simulation stack. Old code is reused **only** where it
> passed the strict quality test; everything else is built fresh. Ships as a
> working end-to-end skeleton on sample/real slices (`aurelius/environment/`).

## 1. KEEP / ADAPT / REPLACE / DELETE audit of old modules

Strict test: trace-grounded (not heuristic-first), causal (no oracle),
production-relevant, cleanly composable, **no hidden M/M/1 serving proxy**, no
stale assumptions vs the v2026 schema, fidelity-tagged, testable via held-out
distribution matching.

| old module | verdict | why |
|---|---|---|
| `optimizer/unified_replay.py` | **KEEP** | token-level discrete-event, causal, deployable, fidelity-aware → the ServingPlane's engine |
| `benchmarks/srtf_serving_backtest._service_time_s`, Azure loader | **KEEP** | real Azure spine + TTFT+TPOT physics |
| `benchmarks/economics.InfrastructureCostConfig` | **ADAPT** | per-GPU-type price kept as a *rental cross-check*; the new CostModel adds PUE + depreciation + region (the owned-hardware basis) |
| `datasets/calibration.alibaba_v2026_serving_class_mix` | **KEEP** | already distribution-derived from v2026 → reused by the FleetPlane + bridge |
| `datasets/signal_matrix.py` | **KEEP** | the signal audit; the manifest's sibling |
| `simulation/cluster/engine.py` (hourly cluster sim) | **REPLACE** | heuristic-first, single-timescale hourly, **M/M/1 queue proxy**, statically calibrated, disconnected from v2026 → replaced by a v2026-native FleetPlane (informs the *state-variable checklist*, not reused) |
| `simulation/cluster/calibration.py` (static `CalibratedParam` table) | **REPLACE** | static engineering guesses; fails "trace-derived." Its *provenance-ladder vocabulary* is adapted (we add `TRACE_DERIVED`) |
| `simulation/cluster/kv_cache.py` | **ADAPT (later)** | generic per-model structure is good, but its prefix-reuse curve is an unfitted sigmoid; the bridge fits the hit-rate from Mooncake instead. Full KV-state reuse is a later PR |
| `simulation/cluster/topology_model.py` | **ADAPT (later)** | `RackLocalityState` maps to `asw_id`; the deeper fabric tiers stay heuristic. The FleetPlane consumes `asw_id` directly for now |
| `benchmarks/realism_audit.py` | **ADAPT** | its honesty-gate vocabulary (`NOT_PRODUCTION_REALISTIC_YET`) is adopted by the ValidationSuite |
| the earlier thin `environment/canonical.py` wrapper | **DELETE** | superseded — this is the first-principles build, not a wrapper |

## 2. New architecture (`aurelius/environment/`)

```
schemas.py            CalibratedParam (full provenance), SignalProvenance,
                      FleetState, ServingRequest, EnvObservation, EnvStep,
                      fidelity ladder MEASURED>TRACE_DERIVED>BENCHMARK>INFERRED>HEURISTIC>ABSENT
fleet_plane_v2026.py  V2026FleetPlane — reads pod/server/network_hourly DIRECTLY;
                      every field TRACE_DERIVED (util/mem/priority/queue-delay/
                      topology asw_id/network/capacity/fragmentation/price)
serving_plane.py      ServingPlane — Azure token-level via unified_replay (NO M/M/1);
                      KVReuseModel (Mooncake-calibrated prefill-savings)
calibration_bridge.py distribution-derived params w/ provenance: Azure tokens+
                      burstiness, Mooncake prefix-hit (from hash_ids), v2026, ISO;
                      time-split train/holdout
cost_model.py         per-GPU-type depreciation + power draw + PUE + region + network
fidelity_manifest.py  per-signal provenance + 5 ABSENT proprietary signals + gate
validation_suite.py   KS + Wasserstein-1 + histogram-L1 + percentile; PASS/WARN/FAIL;
                      capped at NOT_PRODUCTION_REALISTIC_YET
canonical.py          CanonicalMultiPlaneEnvironment — owns the two-clock loop,
                      policy-pluggable, emits (observation, action, reward, metrics)
```

## 3. What the first PR proves end-to-end (on sample + real slices)

Running `CanonicalMultiPlaneEnvironment.run({hour: azure_slice})` on the real
Azure trace + the v2026/Mooncake/CAISO sample slices:

- **Two synchronized clocks** — per-second token-level serving nested inside each
  v2026 fleet hour; per-hour `(observation, action, reward, metrics)` emitted.
- **Trace-derived calibration** — Mooncake prefix-hit rate **0.625** computed from
  `hash_ids`; Azure token p50/p95/p99 + inter-arrival CV fitted; v2026 fleet state
  (util 0.59, queue-delay 5.5→11.8 s, asw racks, CAISO price) all `TRACE_DERIVED`.
- **No M/M/1** — serving is the per-request discrete-event loop; `kpi.n_total ==`
  the request count, not an aggregate proxy.
- **Cost** — owned-hardware basis (depreciation + PUE energy at the hour's ISO
  price), per GPU type; cloud rental kept as a cross-check.
- **Held-out validation** — env's served token distribution vs the Azure holdout:
  **KS 0.042 → PASS**; inter-arrival WARN (honest).
- **Honesty gate** — overall verdict `NOT_PRODUCTION_REALISTIC_YET` (cost params
  are HEURISTIC); manifest `is_production_grade=False`; 5 ABSENT proprietary signals.
- **No row-join** — `ServingRequest` and `FleetState` field sets are disjoint;
  planes couple only via a calibrated scalar (best-effort fraction). Tested.

## 4. Honest framing

Production-LIKE multi-plane environment grounded by real public traces, every
signal fidelity-tagged — **not** real production telemetry. It becomes
production-grade only when a pilot fills the ABSENT tier (user/operator intent,
hardware health, live KV memory state, migration/rejection reasons, internal cost
model). The validation suite structurally cannot say "production realistic."

## 5. Next PRs (sequenced)

1. **CalibrationBridge depth** — wire the remaining v2026 hooks (per-class util,
   mem→KV capacity, fragmentation from real placement) once the multi-GB trace is
   downloaded; raise each from sample to full-trace TRACE_DERIVED.
2. **ValidationSuite breadth** — add the GPU-util / memory / priority / queue-delay
   / topology / network / cost held-out checks (metrics already shipped).
3. **KV depth** — replace the prefill-savings discount with the adapted
   `kv_cache.py` state model, fitted to Mooncake.
4. **AureliusOptimizer integration** — feed `EnvStep` (state/action/reward) to the
   optimizer for training + fair backtesting against this environment.
