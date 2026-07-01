# Batch-1 claim safety & fidelity (Phase 8)

For each new knob, the ten questions. **Hard rule honoured:** where a knob's win depends on weak simulator
inference, it is kept opt-in or labelled directional-only.

---

## 1. KV-cache precision (`kv_cache_precision_policy`) — fp8 / int8 / int4

1. **Causal mechanism implemented?** Yes. KV dtype is decoupled from weights in `roofline.ServingConfig`
   (`kv_precision` + `KV_PRECISION_BYTES`); fewer KV bytes/token → higher decode tokens/s in the
   memory-bandwidth-bound regime → lower service time/cost (reward via `roofline_actions` →
   `compute_phase_serving`). `kv_precision.py` adds the HBM/active-sequence-capacity/eviction channel.
2. **Production-like enough?** For **direction**, yes — vLLM/TensorRT-LLM ship fp8/int8 KV; the
   memory-bound-only throughput benefit matches the public result (FP8 KV did *not* help vLLM prefill-heavy).
3. **Trace-derived?** The KV bytes/token formula is BENCHMARK_DERIVED from published architecture (shared with
   `kv_cache.KVFootprint`); prefix-reuse dynamics remain TRACE_DERIVED (Mooncake).
4. **Benchmark/spec-derived?** GPU HBM/bandwidth (PUBLIC_SPEC); "fp8/int8 KV ≈ lossless, decode-bound
   throughput" (PUBLIC_BENCHMARK).
5. **Simulator-inferred?** The latency **magnitude** (roofline band) and the active-sequence-capacity / HBM /
   eviction deltas.
6. **Needs production telemetry?** Live KV residency / eviction rate; and a per-model accuracy harness to
   retire the int4 quality risk.
7. **Magnitude headline-safe?** fp8/int8: borderline — direction is benchmark-grounded, magnitude is
   simulator-inferred → safe as *directional*. int4: **no**.
8. **Direction headline-safe?** fp8/int8: **yes**. int4: **no** (unmodelled quality risk).
9. **Default-on?** **No (corrected in the Batch-1 corrective PR).** KV-cache precision is an
   `OPTIONAL_SERVING_ENGINE_INTEGRATION` and is **default-OFF** (requires `enable_kv_cache_precision`). When
   enabled it is additionally regime-gated + Pareto-gated; default value is the no-op
   `inherit_weight_precision`. See `BATCH1_ACTION_SURFACE_PRODUCT_BOUNDARY.md`.
10. **Diagnostic-only?** Only `kv_int4_diagnostic_only` (opt-in via `allow_quality_risk`; quality-risk channel
    + excluded from the headline planner). fp8/int8 are deployable **once the operator opts in**.

## 2. Heterogeneous GPU assignment (`gpu_assignment_policy`)

1. **Causal mechanism implemented?** Yes, as a **fixture model** (`gpu_assignment.py`): per-GPU roofline timing
   + per-type cost → class→GPU mapping → gp/$ / SLA / HBM pressure. **Not** wired into the production reward.
2. **Production-like enough?** Not for the current benchmark — the cost path is single-dominant-GPU and
   `gpu_type` is constant per server. **NOT_APPLICABLE** to Benchmark v1 (stated, not hidden).
3. **Trace-derived?** None (no per-replica assignment trace).
4. **Benchmark/spec-derived?** Per-GPU FLOPs/bandwidth/HBM and per-type CapEx/lease/power (PUBLIC_SPEC /
   public-list), from `roofline.GPU_SPECS` + `cost_model`.
5. **Simulator-inferred?** The routing outcome and per-class HBM pressure.
6. **Needs production telemetry?** A per-replica assignment mechanism in the cluster/cost path + per-type
   measured latency for the served model.
7. **Magnitude headline-safe?** No — fixture-only.
8. **Direction headline-safe?** Direction is benchmark/literature-grounded (Helix/AIBrix/ThunderServe), but it
   is **not on the production benchmark**, so no headline is taken from it.
9. **Default-on?** No — frozen off in the production planner (NOT_APPLICABLE, recorded reason). On a
   homogeneous fleet it provably ties (no fake gain).
10. **Diagnostic-only?** Yes (SIMULATED_ONLY) until the fleet/cost path exposes per-replica assignment;
    `diagnostic_oracle_assignment` is additionally NON-deployable.

## 3. Prefill/decode disaggregation (`prefill_decode_policy`)

1. **Causal mechanism implemented?** Yes. Roofline `disaggregated_static` (split inflates one phase + fixed
   handoff) + `pd_disaggregation.py` (two M/M/c phase pools, shared-pool interference penalty, KV-handoff
   bytes/latency, idle GPU-seconds). Reward via the roofline service-time / GPU-seconds channel.
2. **Production-like enough?** For **direction**, yes — DistServe/Splitwise/Dynamo show disaggregation removes
   P/D interference (up to 4.48× at equal SLO) and pays a KV handoff. The cluster replay has **no persistent
   phase queues**, so the magnitude is a conservative approximation.
3. **Trace-derived?** None for phase queues; KV-handoff *bytes* are BENCHMARK_DERIVED (architecture × context).
4. **Benchmark/spec-derived?** Interference-removal benefit + handoff mechanism (DistServe/Splitwise/Dynamo);
   interconnect bandwidth band (RDMA/IB, public).
5. **Simulator-inferred?** Queue waits, the interference penalty magnitude, the handoff latency.
6. **Needs production telemetry?** Persistent per-pool queue depths, measured handoff bytes/latency,
   per-phase utilization.
7. **Magnitude headline-safe?** No — directional only.
8. **Direction headline-safe?** Yes (benchmark-grounded), but reported directional given no live phase queues.
9. **Default-on?** **No (corrected in the Batch-1 corrective PR).** PD disaggregation is an
   `OPTIONAL_SERVING_ENGINE_INTEGRATION` and is **default-OFF** (requires `enable_prefill_decode_disagg`, and a
   serving stack that supports disaggregation). When enabled it is regime-gated + Pareto-gated; default value
   is the no-op `shared`.
10. **Diagnostic-only?** No (CONNECTED when opted in), but every PD result is labelled **directional /
    SIMULATOR_INFERENCE** until pilot disaggregated-pool telemetry. The DistServe-shaped fixtures confirm the
    model reproduces a DistServe-ORDER goodput win in the genuine regime.

---

## Net claim-safety posture

- **No headline is claimed from any new knob on Benchmark v1.** The ablation (Phase 7) shows all three are
  neutral on the production window (regime-gated off or Pareto-not-selected), and the unchanged +204.6 % vs
  production_scheduler is carried entirely by pre-existing knobs.
- **fp8/int8 KV precision** is the closest to headline-safe (benchmark-grounded direction, deployable, default
  no-op, regime-gated); we still label its *magnitude* simulator-inferred.
- **int4 KV** is diagnostic-only (no quality model).
- **heterogeneous GPU assignment** is NOT_APPLICABLE to the benchmark → fixture-only / SIMULATED_ONLY.
- **PD disaggregation** is **default-off / optional** (serving-engine integration) and **directional-only**
  until live phase-queue telemetry; it reproduces a DistServe-order win in the genuine high-load skewed regime.

This satisfies the hard rule: every knob whose win rests on weak simulator inference is opt-in (int4),
not-on-benchmark (GPU assignment), or labelled directional (PD); only the benchmark-grounded, regime-gated,
Pareto-safe behaviour (fp8/int8 KV direction) is treated as deployable.
