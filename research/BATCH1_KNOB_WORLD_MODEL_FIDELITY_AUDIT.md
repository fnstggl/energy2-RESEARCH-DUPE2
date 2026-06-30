# Batch-1 knob world-model fidelity audit (Phase 1)

For each new knob: who ships it, what traces would calibrate it, what public data we can use now, what the
simulator already had, what had to be added to make it causal, what would be misleading to model, and the
fidelity tier. Fidelity tiers (from the build spec + the user's ladder):
`TRACE_DERIVED` > `PUBLIC_BENCHMARK_DERIVED` > `PUBLIC_SPEC_DERIVED` > `SIMULATOR_INFERENCE` >
`NEEDS_PRODUCTION_TELEMETRY`.

This is not a README skim: it is anchored in the repo's existing roofline/KV/world models
(`roofline.py`, `kv_cache.py`, `cost_model.py`, `world_simulator.py`) and the public systems/papers below.

---

## 1. KV-cache precision (`kv_cache_precision_policy`)

**Public systems / papers.** vLLM ships **FP8 KV cache** (E4M3/E5M2) decoupled from weight precision; INT8 KV
is an open vLLM feature request and is shipped today by **TensorRT-LLM** (FP8 *and* INT8). Quality: public
evals find FP8 KV + FP8 attention **nearly lossless** (≤ 0.7 pt aggregate, ~99 % recovery); GPU-accelerated
INT8 KV reports **4× memory reduction with minimal degradation**; KVTuner (mixed-precision KV) reports nearly
lossless INT4-ish only with per-layer sensitivity tuning. **Crucial nuance:** FP8 KV did **not** improve vLLM
throughput in a *prefill-heavy* scenario — the throughput win is **decode / memory-bandwidth-bound only**.
That is exactly the regime gate we implemented.

**Traces needed for calibration.** Per-deployment KV-residency telemetry (live bytes/token, eviction rate,
active-sequence occupancy) and an accuracy harness for the served model at each KV dtype. **ABSENT** in public
traces.

**Public data usable now.** Model architecture (layers · kv_heads · head_dim · dtype) → KV bytes/token
(BENCHMARK_DERIVED, already in `kv_cache.KVFootprint`); GPU HBM capacity + bandwidth (vendor specs,
`roofline.GPU_SPECS` / `kv_cache.GPU_MEM_GIB`); the public "FP8/INT8 KV ≈ lossless, decode-bound throughput"
result above.

**What the simulator already had.** `roofline.serving_point` modeled KV bytes — but **tied to the weight
precision** (`kv_bytes = kv_bytes_per_token·(weight_pb/2)`). KV could not be quantized independently.

**What had to be added.** A **separate** `kv_precision` on `ServingConfig` + `KV_PRECISION_BYTES`, so KV bytes
decouple from weights (`inherit` reproduces the old behaviour bit-for-bit). The decode-bandwidth roofline term
now uses it → faster decode in the memory-bound regime, neutral otherwise. `kv_precision.py` adds the
memory/HBM channel the fixed-batch roofline omits: active-sequence capacity, HBM pressure, eviction delta.

**What would be misleading to model.** Booking an INT4-KV throughput win as headline-safe (no quality model →
we attach a quality-risk channel and exclude it from the headline planner); claiming a throughput gain in the
*prefill-heavy / compute-bound* regime (public result says there is none → our regime gate freezes it there);
claiming a *live* eviction/occupancy number (residency telemetry is ABSENT → reported as a diagnostic).

**Fidelity tier:** **PUBLIC_BENCHMARK_DERIVED** for the byte/quality facts (fp8/int8 KV ≈ lossless, bytes/token
formula) and the *direction*; **SIMULATOR_INFERENCE** for the latency *magnitude* (roofline band) and the
active-sequence/HBM-pressure deltas. INT4 KV: **NEEDS_PRODUCTION_TELEMETRY** (no quality model) → diagnostic.

---

## 2. Heterogeneous GPU assignment (`gpu_assignment_policy`)

**Public systems / papers.** **Helix** (serve LLMs over heterogeneous + geo-distributed GPUs via max-flow;
commodity L4/T4 matching H100/A100 on cost/energy/memory), **AIBrix** Heterogeneous GPU Inference (lower-cost
GPUs to cut spend), **ThunderServe / Mélange** (automate heterogeneous handover, KV over RDMA). Production
pattern: "route long prompts to H100 buckets, short bursty chat to L40S → near-perfect fleet utilization."

**Traces needed for calibration.** Per-GPU-type measured serving latency/throughput for the served model, and
a per-replica assignment mechanism in the cluster/cost path. **ABSENT** — and structurally so in the current
benchmark.

**Public data usable now.** Per-GPU FLOPs / bandwidth / HBM (vendor specs, `roofline.GPU_SPECS`,
`kv_cache.GPU_MEM_GIB`); per-GPU-type CapEx / power / lease (`cost_model.OWNED_ECONOMICS`,
`LEASE_USD_PER_GPU_HOUR`); public price points (H100 on-demand ≈ $3.58/h, June 2026).

**What the simulator already had.** A fleet `gpu_type_mix` (heterogeneous fleet-wide, e.g. 50/50 H100/A100 in
the sample) and per-type cost — but the reward path **costs the whole period at ONE dominant GPU type**, and
`gpu_type` is **constant per server** (set at cluster build, never a decision). There is **no per-workload
assignment mechanism** in `run_unified_replay`.

**What had to be added.** A standalone causal model `gpu_assignment.py` (per-GPU roofline timing + per-type
cost → class→GPU mapping → gp/$ / SLA / HBM-pressure), exercised by **controlled fixtures**. It is **not**
wired into the production reward path.

**What would be misleading to model.** Booking a heterogeneous gp/$ win on the production benchmark, where the
cost path is single-dominant-GPU and there is no assignment mechanism — that would be a **fake fleet**. We
therefore label the production benchmark **NOT_APPLICABLE** for this knob, freeze it off in the production
planner with a recorded reason, and demonstrate it only on fixtures with an explicit GPU mix. On a homogeneous
fleet every policy provably ties (structural no-fake-gain guarantee).

**Fidelity tier:** **PUBLIC_SPEC_DERIVED** for the per-GPU FLOPs/bandwidth/HBM and per-type cost; the routing
*outcome* is **SIMULATOR_INFERENCE**; production-benchmark applicability is **NEEDS_PRODUCTION_TELEMETRY** (a
per-replica assignment + per-type calibration) — until then it is fixture-only / SIMULATED_ONLY.

---

## 3. Prefill/decode disaggregation (`prefill_decode_policy`)

**Public systems / papers.** **DistServe** and **Splitwise** physically separate prefill and decode → up to
**4.48× more requests at the same SLO** by **eliminating prefill/decode interference**. **NVIDIA Dynamo**,
llm-d, SGLang, vLLM, LMCache, Mooncake all ship disaggregation; KV is transferred prefill→decode over
**NIXL/RDMA** (non-blocking). **Sarathi** mitigates the same interference via chunked prefill within one pool.

**Traces needed for calibration.** Persistent per-phase pool queue depths, measured KV-handoff bytes/latency
on the deployment's interconnect, and per-phase utilization. **ABSENT** — the cluster replay has a single
capacity pool, no persistent phase queues.

**Public data usable now.** The interference→disaggregation benefit and the KV-handoff mechanism (papers
above); KV bytes/token for handoff size (BENCHMARK_DERIVED); interconnect bandwidth band (RDMA/IB class,
public).

**What the simulator already had.** `roofline.serving_point` already modeled the split *analytically*
(`disaggregated_static`: a wrong split inflates one phase's work + a fixed `handoff_s`). `prefill_decode_policy`
existed but was `SIMULATED_ONLY`.

**What had to be added.** Promotion to **CONNECTED** (it reaches reward through the roofline service-time /
GPU-seconds channel) plus `pd_disaggregation.py`: a **conservative phase-pool queueing approximation** — two
M/M/c-style pools, the **shared-pool interference penalty** (the DistServe/Splitwise benefit, gated to
high-load + skewed mixes), KV-handoff bytes/latency, and idle GPU-seconds by pool. This is the "conservative
causal approximation, labelled honestly" the spec asked for.

**What would be misleading to model.** Free disaggregation (we charge KV handoff + idle-pool GPU-seconds + a
statistical-multiplexing penalty on two smaller pools, so a balanced/light workload prefers the shared pool);
a benefit with no interference present (gated to high-load + skewed regimes); claiming *measured* phase queue
depths (there are none → SIMULATOR_INFERENCE).

**Fidelity tier:** **PUBLIC_BENCHMARK_DERIVED** for the *direction* (disaggregation removes P/D interference;
handoff has a real cost) and the handoff bytes; **SIMULATOR_INFERENCE** for the queue/interference/handoff
*magnitudes*; live phase-queue dynamics are **NEEDS_PRODUCTION_TELEMETRY**.

---

## Summary table

| knob | direction fidelity | magnitude fidelity | production-benchmark status | headline-safe? |
|--|--|--|--|--|
| KV-cache precision (fp8/int8) | PUBLIC_BENCHMARK_DERIVED | SIMULATOR_INFERENCE | CONNECTED, regime-gated | yes (fp8/int8) |
| KV-cache precision (int4) | — | NEEDS_PRODUCTION_TELEMETRY | diagnostic-only | **no** |
| heterogeneous GPU assignment | PUBLIC_SPEC_DERIVED | SIMULATOR_INFERENCE | **NOT_APPLICABLE** (fixture-only) | n/a (not on benchmark) |
| prefill/decode disaggregation | PUBLIC_BENCHMARK_DERIVED | SIMULATOR_INFERENCE | CONNECTED, regime-gated | directional-only |

**Sources:** [vLLM FP8 KV-cache blog](https://vllm.ai/blog/2026-04-22-fp8-kvcache) ·
[vLLM Quantized KV docs](https://docs.vllm.ai/en/v0.10.1/features/quantization/quantized_kvcache.html) ·
[vLLM INT8 KV issue #33480](https://github.com/vllm-project/vllm/issues/33480) ·
[GPU-accelerated INT8 KV (arXiv)](https://arxiv.org/pdf/2601.04719) ·
[KVTuner (arXiv 2502.04420)](https://arxiv.org/pdf/2502.04420) ·
[DistServe retrospective (Hao AI Lab)](https://haoailab.com/blogs/distserve-retro/) ·
[NVIDIA Dynamo disaggregated serving](https://docs.nvidia.com/dynamo/user-guides/disaggregated-serving) ·
[Helix (arXiv 2406.01566)](https://arxiv.org/html/2406.01566v2) ·
[AIBrix heterogeneous GPU inference](https://aibrix.readthedocs.io/latest/features/heterogeneous-gpu.html) ·
[Cost-efficiency in heterogeneous GPU serving](https://www.decodesfuture.com/articles/cost-efficiency-heterogeneous-gpu-llm-serving)
