# Roofline MPC Action Calibration (Phase 12)

Calibration + provenance for the V2 action surfaces. Fidelity tiers: TRACE_DERIVED · BENCHMARK_DERIVED ·
PUBLIC_PAPER · INFERRED · SIMULATOR_INFERENCE · ABSENT(PROP). Every constant feeds a physical quantity
(bytes moved, FLOPs, peak throughput, power, transfer seconds), never reward directly.

## Roofline timing (RooflineServingModelV2)

| parameter | value | tier | source |
|--|--|--|--|
| peak FP16 TFLOPS / HBM GB/s / HBM GiB per GPU | spec table | BENCHMARK_DERIVED | public spec sheets; cross-checked llm-analysis `gpu_configs`, InferSim `hardware/gpu.py` |
| KV bytes/token = 2·L·kv_heads·head_dim·dtype | per arch | BENCHMARK_DERIVED | identical to `kv_cache.KVFootprint`; vLLM/InferSim/llm-analysis confirm |
| gemm FLOPs = 2·m·n·k | — | PUBLIC_PAPER | InferSim `flops/flops.py` |
| achievable-bandwidth derate | 0.8 | BENCHMARK_DERIVED/INFERRED | InferSim convention |
| prefill MFU / decode MFU | 0.7 / 0.35 | BENCHMARK_DERIVED | Megatron/InferSim band; constant (profiled MFU optional) |
| ridge = peak/BW; bound = AI<ridge?memory:compute | — | PUBLIC_PAPER | LLM-Viewer `roofline_analyze`, llm-analysis `get_pivot` |

Validation: `test_roofline_external.py` (14), `external_formula_checks` (7). The constants are a physical
*floor* at constant MFU — optimistic on small batches; tile/wave quantization + per-kernel MFU are ABSENT
(would need Vidur-style profiling).

## precision_mode (bf16 / fp8 / int4)

| precision | byte-scale (vs bf16) | compute-peak scale | quality-risk | tier |
|--|--|--|--|--|
| bf16 | 1.0 | 1.0 | 0.0 | BENCHMARK_DERIVED |
| fp8 | 0.5 | 2.0 | 0.01 | BENCHMARK_DERIVED (tensor-core fp8 ≈2× FLOPS; tiny risk band) |
| int4 | 0.25 | 2.0 | 0.06 | INFERRED (weight-only 4-bit; conservative 6 % quality/risk surcharge) |

Chain: byte-scale → memory leg of the roofline (helps memory-bound decode); compute-peak → compute leg;
quality-risk → reduces SLA-safe goodput (so int4 only wins where memory pressure dominates). The fp8 ≈2× and
int4 risk are bands, **not** measured per-model quality — labelled, never claimed as MEASURED.

## spec_decode_mode (off / shallow / medium / aggressive)

| mode | memory-leg acceptance speedup | compute-leg draft+verify overhead | tier |
|--|--|--|--|
| off | 1.0 | 1.0 | — |
| shallow | 1.3 | 1.15 | INFERRED |
| medium | 1.6 | 1.35 | INFERRED |
| aggressive | 2.0 | 1.7 | INFERRED |

Chain: `decode = max(compute·overhead, memory/speedup)` — divides the memory leg (fewer weight reloads),
multiplies the compute leg (draft+verify). So it helps memory-bound, high-acceptance decode and hurts
compute-bound or low-acceptance decode (fixtures #5/#6). Real acceptance is workload/model-specific →
SIMULATOR_INFERENCE; a measured acceptance trace is ABSENT (PROP).

## clock_power_state (low / base / high)

| state | peak-FLOPS scale | power scale | tier |
|--|--|--|--|
| low | 0.75 | 0.7 | INFERRED (DVFS band) |
| base | 1.0 | 1.0 | — |
| high | 1.15 | 1.35 | INFERRED |

Chain: peak-FLOPS scales the compute leg only (HBM BW fixed) → down-clock is neutral on memory-bound decode
and slows compute-bound prefill; power scale feeds the energy term. dt60 config F shows the energy win
(−37 %) from down-clocking memory-bound decode. Real DVFS curves are GPU/firmware-specific → INFERRED.

## colocation_mode (off / conservative / aggressive)

| mode | idle reclaim frac | contention surcharge | tier |
|--|--|--|--|
| off | 0.0 | 0.0 | — |
| conservative | 0.5 | 0.01 | INFERRED |
| aggressive | 0.9 | 0.05 | INFERRED |

Chain: reclaim applies to `min(background_work_gpu_seconds, idle)` → reduces billed GPU-seconds; contention
inflates completion (can push borderline requests over SLA). **Hard guard:** with no real/trace-derived
`background_work_gpu_seconds`, reclaim is 0 and the candidate generator prunes co-location to `off` — no
imaginary background goodput (fixtures #9/#10). Real background SM-headroom is ABSENT (PROP) until a pilot.

## prefill_decode_allocation (shared / p40_d60 / p50_d50 / p60_d40 [/ p20_d80 / p80_d20 diag])

| parameter | value | tier | source |
|--|--|--|--|
| KV handoff = prompt_tokens·kv_bytes / handoff_bw | bw=50 GB/s | BENCHMARK_DERIVED | Splitwise/Mooncake (RDMA band) |
| phase queue (M/D/1 per pool) | ρ·D/(2(1−ρ)) | PUBLIC_PAPER | DistServe M/D/1; deterministic service |
| mixed-batch / interference | chunked-prefill 0.4× reduction | INFERRED | Sarathi Algorithm 3 (qualitative) |

Chain: split → per-pool capacity + queue + handoff latency → TTFT/completion + idle GPU-s. Wrong allocation
starves a pool (fixture #13). The M/D/1 near-saturation behaviour is a cliff (SIMULATOR_INFERENCE) — a
per-iteration calibration (Vidur) would smooth it; that is the named next calibration step.

## Adaptive search (AdaptiveMPCSearchV2)

No silent cap: raw/evaluated counts, strategy, runtime, regret all reported. `auto` exhausts when raw ≤
threshold (default 4096), else beam (k=6). Regret audit compares approximate vs exhaustive on the same space.
Fixture #20/#21: coordinate descent regret 0.50 (warns), beam regret 0.0 (matches exhaustive). Runtime is
bounded but config F's ~1146 ms/decision flags that large memory-regime spaces (~144 candidates) warrant a
smaller beam or tighter regime pruning — a documented tuning knob, not a silent truncation.

## What stays simulator-inferred / proprietary

SIMULATOR_INFERENCE: M/D/1 phase queue, continuous-batching occupancy, spec acceptance, int4 quality-risk,
co-location contention, MFU constants. ABSENT (PROP, pilot-only): real per-replica KV residency, measured
cache hit rates, production KV-transfer bandwidth, real DVFS/power curves, real background SM-headroom, true
per-request model identity, per-link/NVLink contention, internal operator $/energy/carbon. None is claimed as
MEASURED; each is named here so no headline overstates fidelity.
