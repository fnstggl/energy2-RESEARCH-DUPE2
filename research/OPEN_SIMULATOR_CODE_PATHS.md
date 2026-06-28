# Open Simulator Code Paths (Phase 3)

Actual files / functions / equations that implement each mechanism, from source inspection (not READMEs).
For each project: key files, key functions/classes, inputs, outputs, equations/logic, assumptions,
reusable-code candidates, reusable-equation candidates, mismatch with Aurelius, integration difficulty,
license risk.

---

## Alibaba InferSim (`alibaba/InferSim`, Apache-2.0)

| concern | file · symbol |
|--|--|
| roofline core | `layers/attn.py` · `MHA.decode_attn_core` / `prefill_attn_core` → `max(attn_core_time, kv_load_time)` |
| GEMM FLOPs | `flops/flops.py` · `gemm_flops(m,n,k)=2·m·n·k`; `get_mha_gflops`, `get_mla_*`, `get_moe_gflops` |
| KV bytes | `kvcache/kvcache.py` · `get_mha_kvcache_size = 2·layers·kv_heads·head_dim·dtype` |
| GPU specs | `hardware/gpu.py` · `GPU(fp16_tflops, fp8_tflops, mfu, mem, mem_bw·0.8)` |
| TTFT/TPOT assembly | `models/model.py` · per-layer sum ×layers ×1000 +30ms(prefill)/+5ms(decode); `TGS=tokens/tp/latency` |
| KV capacity budget | `models/model.py` · `kvcache_mem = mem − params_per_gpu − 15 − 5` GB |

- **Inputs:** HF `config.json`, device, tp/world size, isl/osl, batch, fp8 flags. **Outputs:** weights/KV GB,
  per-token GFLOPs, TTFT/TPOT ms, throughput TGS, MFU, per-stage compute-vs-KV-load breakdown.
- **Equations (verbatim):** `attn_core_time = bs·gflops/(fp16_tflops·1024·mfu)`;
  `kv_load_time = kvcache_bytes·kv_len·bs/layers/1024³/mem_bw`; prefill divides compute by extra `1.8`
  (kernel efficiency). `get_moe_gflops = act·3·gemm_flops(1,hidden,inter/tp)/1e9`, `act = shared + experts_per_tok`.
- **Assumptions:** profiled MFU from `bench_data/` CSVs (optional; constant MFU substitutable); `mem_bw·0.8`
  achievable-bandwidth derate; `+30/+5 ms` scheduler overheads.
- **Reusable code:** `flops/`, `kvcache/`, `params/`, `layers/` — dependency-free, Apache-2.0, directly liftable.
- **Reusable equations:** all of the above. **Mismatch with Aurelius:** none structurally — KV-byte formula is
  *identical* to `kv_cache.py`; InferSim is stateless (composes cleanly). **Integration: LOW–MEDIUM. License: LOW.**
- **Ported in this PR:** `aurelius/environment/roofline_external.py` (re-implemented, not copied).

## llm-analysis (`cli99/llm-analysis`, Apache-2.0)

| concern | file · symbol |
|--|--|
| attention FLOPs (GQA) | `analysis.py` · `4·b·s·h² + 4·b·s·h²/kv_groups + 4·b·s²·h` |
| MLP FLOPs | `(6 if gated else 4)·b·s·h²·expansion` |
| KV bytes/layer | `2·b·s·head_dim·heads_per_gpu·kv_dtype_bytes` |
| roofline | `compute=flops/tp/(TFLOPS·1e12)`, `memory=act_mem/(BW·1e9)`, `max(...)` |
| ridge point | `get_pivot() = peak_TFLOPS·1e3·bits/8 / hbm_bw / 2` |
| GPU config | `gpu_configs/*.json` · `{mem_per_GPU_in_GB, hbm_bandwidth_in_GB_per_sec, peak_fp16_TFLOPS, ...}` |

- **Reusable equations:** GQA FLOPs, `get_pivot`, GPU JSON schema *with HBM capacity*, `get_TFLOPS_per_gpu =
  peak·flops_efficiency`. **Mismatch:** methods coupled to its dataclasses → port formulas, not classes.
- **Integration: MEDIUM. License: LOW–MEDIUM** (Apache NOTICE). **Ported:** ridge-point label + GPU table.

## LLM-Viewer (`hahnyuan/LLM-Viewer`, MIT)

| concern | file · symbol |
|--|--|
| ridge classifier | `roofline_model.py` · `roofline_analyze(bandwidth, max_OPS, OPs, memory_access)` |
| hardware table | `hardwares/hardware_params.py` · `{bandwidth, FP16, INT8, onchip_buffer}` (no capacity) |
| layer OPs | prefill `ic·oc·b·s·2`; decode `ic·oc·b·2`; attn prefill `s²·hd·heads·b·2` |

- **Equation (verbatim):** `turning_point=max_OPS/bandwidth`; `ai<turning → memory, perf=ai·bw`; else
  `compute, perf=max_OPS`; `time=OPs/perf`. **Reusable:** the cleanest standalone ridge classifier (liftable).
- **Mismatch:** none. **Integration: LOW. License: LOW.** **Ported:** `roofline_analyze` in `roofline_external.py`.

## BLIS / inference-sim (`inference-sim/inference-sim`, Apache-2.0, Go)

| concern | file · symbol |
|--|--|
| DES engine | `sim/simulator.go` · `EventQueue` (heap), `ProcessNextEvent`, `Step` (schedule→execute→completions→next) |
| latency iface | `sim/latency_model.go` · `StepTime(batch)`, `QueueingTime`, `PostDecodeFixedOverhead` |
| roofline | `sim/latency/roofline.go` · `rooflineStepTime`: `max(compute_time, memory_time)`, FP8/FP16 select |
| trained-physics | `sim/latency/trained_physics_model.go` · 10-β step-time (5 roofline basis + TP/MoE comm) |
| KV store | `sim/kv_store.go` · block KV + prefix caching + chunking + GPU/CPU tiers |
| batching | `sim/batch.go` (vLLM-style); routing `sim/routing.go`; admission `sim/admission.go` |

- **Equations (verbatim):** `compute_time = FLOPs/(tp·peak·MFU)` (separate `MFU_prefill`/`MFU_decode`);
  `memory_time = (weight_bytes + dynamic_bytes)/peak_bw`; `KVCacheGrowth = 2·layers·kv_heads·head_dim·bytes·new_tokens`.
  Trained-physics: `StepTime = β1·max(T_pf_compute,T_pf_kv) + β2·max(T_dc_compute,T_dc_kv) + β3·T_weight + … +
  β5·L + β6·B + β7 + β8·moe_scaling·n_moe_layers` (+α₀/α₁/α₂ per-request/token overheads).
- **Reusable:** roofline + trained-physics formulas (Go → re-implement in Python). DES engine entangled with
  Go `container/heap` → don't port. **Mismatch:** owns its world, Go. **Integration: MEDIUM (math) / HIGH (engine).
  License: LOW.**

## SplitwiseSim (`Mutinifni/splitwise-sim`, MIT, Python)

| concern | file · symbol |
|--|--|
| DES engine | `simulator.py` · `Simulator` (heapq), `TraceSimulator.load_trace` |
| perf model | `performance_model.py` · `DatabasePerformanceModel` (scipy `interp1d` over `batch_tokens`) |
| iteration time | `get_iteration_duration`: all-prompt→`prompt_predictor`; all-token→`token_predictor`; mixed→`·1.1` |
| batching | `instance.py` · `ORCAInstance.select_batch`, `SplitwiseInstance` (preempt, `max_batch_tokens`) |
| schedulers | `scheduler.py` · `JSQ/TokenJSQ/KV*Scheduler`; `KVScheduler.add_kv_cache_transfer` (Flow over DummyLink) |
| KV size | `request.py` · `estimate_kv_cache_size = 2·B·T·H·L·dtype` |

- **Equations:** KV transfer `flow_time = kv_bytes/bandwidth` (overlap = 10× bandwidth); mixed-batch ×1.1;
  contiguous-iteration collapse `delay = iter_dur·num_contiguous`. **Reusable:** `DatabasePerformanceModel`
  (drop-in profiled interpolator), schedulers, `metrics.py` (TTFT/TBT/SLO). **Mismatch:** module-level
  singletons (not re-entrant for MPC); KV = bulk bytes (no LRU/prefix). **Integration: MEDIUM. License: LOW.**

## LLMServingSim 2.0 (`casys-kaist/LLMServingSim`, MIT, Python+C++)

| concern | file · symbol |
|--|--|
| scheduler/batching | `serving/core/scheduler.py` · `schedule_with_prefix`, chunked-prefill token budget, preempt-to-CPU |
| routing/disagg | `router.py` · `_least_load_select`, `transfer_prefill_request` (prefill→decode) |
| KV/mem | `memory_model.py` · `get_kv = 2·kv_dim·seq·layers·kv_fp//npus`; `get_block_kv` (new blocks only) |
| prefix cache | `radix_tree.py` · `RadixCache.match_prefix/evict` (lock-ref) |
| timing | `trace_generator.py` · profiled-table interp (`_lookup_attention` log-space 4D, `_lookup_moe`) |
| power | `power_model.py` · `energy_j = (active−idle)·latency_s` (NPU); DRAM/Link `energy_per_bit·bits` |
| ASTRA-sim IPC | `controller.py`, `graph_generator.py` (Chakra), `config_builder.py` |

- **Reusable:** `request.py`, `scheduler.py` (chunked prefill + preemption), `memory_model.py`+`radix_tree.py`
  (radix prefix cache), `power_model.py`, `gate_function.py` (MoE). **Hard to reuse:** `controller/graph/config`
  (ASTRA-sim bound). **Mismatch:** no rack hierarchy/warm-cold/migration; state ephemeral per run; C++ build.
  **Integration: HIGH (full) / MEDIUM (lift Python modules). License: LOW** (verify ASTRA-sim/Chakra subcomponents).

## Vidur (`microsoft/vidur`, MIT, Python)

| concern | file · symbol |
|--|--|
| predictor | `sklearn_execution_time_predictor.py` · RandomForest; features below |
| registry | `execution_time_predictor_registry.py` · `random_forest` (default) / `linear` |
| DES | `simulator.py` (heap event loop); schedulers `vidur/scheduler/replica_scheduler/{vllm,sarathi,orca,...}` |
| KV admission | `num_required_blocks = ceil(prefill_tokens/block_size)` + watermark; decode +1 block/iter |

- **Features:** token matmuls `[num_tokens]`; prefill-attn `[kv_cache_size, prefill_chunk_size²]`; decode-attn
  `[batch_size, kv_cache_size]`; TP comm `all_reduce + nccl_skew·tp^1.25`. **Reusable:** profile→GridSearchCV→
  cache→O(1) lookup recipe; the A100/A40/H100 profiling CSVs (MIT). **Mismatch:** predictor coupled to Vidur
  `Batch`/`Request` + CSV schema; needs profiling run per new GPU/model. **Integration: MEDIUM. License: LOW.**

## Mooncake (`kvcache-ai/Mooncake`, Apache-2.0)

| concern | symbol |
|--|--|
| block hashing | `hash = hash(current_512tok_block + previous_hash)` (prefix-chained) |
| Conductor (Alg.1) | `TTFT = T_queue + T_prefill` (low hit) / `T_transfer + T_queue + T_prefill` (remote); `EstimateKVCacheTransferTime()` |
| early reject | admit iff `TTFT ≤ TTFT_SLO ∧ predicted TBT ≤ TBT_SLO`; SLO `TTFT_P90≤10×base`, `TBT_P90≤5×base` |
| trace | JSONL: `timestamp`(ms), `input_length`, `output_length`, `hash_ids`(512-tok block ints) |

- **Reusable equation:** the TTFT decomposition + early-reject test (portable). **Reject:** Transfer Engine
  (C++/RDMA). **Mismatch:** 512-tok blocks vs Aurelius/vLLM 16-tok paging (document the mapping).
  **Integration: LOW (trace/eqs) / HIGH (engine). License: LOW.**

## vLLM / PagedAttention (`vllm-project/vllm`, Apache-2.0)

| concern | file · symbol |
|--|--|
| KV bytes/block | `v1/kv_cache_interface.py` · `2·block_size·num_kv_heads·head_dim·dtype_bytes` |
| num blocks | `v1/core/kv_cache_utils.py` · `available_memory // page_size // num_layers` |
| scheduler | `v1/core/sched/scheduler.py` · unified prefill/decode, `max_num_batched_tokens` budget, chunked clamp |
| block pool / LRU | `v1/core/block_pool.py` · free-queue, reverse-order eviction (tail first), full-block-only caching |

- **Reusable equations:** KV bytes/token (GQA), max-blocks, `max_seqs ≈ num_blocks/ceil(avg_seq/block)` (a clean
  memory-bound batch ceiling). **Mismatch:** CUDA/torch engine → re-implement formulas. **Integration: LOW (eqs)
  / HIGH (engine). License: LOW.**

## DistServe / Sarathi / Orca (papers + artifacts)

| item | reusable logic |
|--|--|
| DistServe (`LLMServe/DistServe`, Apache-2.0; `simdistserve` simpy) | prefill **M/D/1** `Avg_TTFT = D + R·D²/(2(1−R·D))`; 2-way pipeline `/(4(2−R·D))`; tensor `D/K + R·D²/(2K(K−R·D))` |
| Sarathi (`microsoft/sarathi-serve`, Apache-2.0) | **Algorithm 3** stall-free batching: pack decodes → fill `token_budget` with partial prefills → admit new last; budget via Vidur (512 strict/2048 relaxed); tile-quant (chunk 257 ~32% slower than 256) |
| Orca (no licensed artifact) | iteration-level scheduling + selective batching (batch non-attn GEMMs token-wise, run attention per-request) — **pattern only**, clone code unlicensed |

- **Integration: LOW–MEDIUM (eqs/patterns). License: LOW** (Orca clone HIGH — do not import).

---

## Integration-difficulty / license-risk roll-up

| project | reuse mode | integration | license risk |
|--|--|--|--|
| InferSim | PORT-EQ (done) | LOW–MED | LOW |
| llm-analysis | PORT-EQ (done) | MED | LOW–MED |
| LLM-Viewer | PORT-EQ (done) | LOW | LOW |
| BLIS | PORT-EQ / VALIDATION | MED (math) / HIGH (Go engine) | LOW |
| SplitwiseSim | PORT-EQ + PORT-PATTERN | MEDIUM | LOW |
| LLMServingSim 2.0 | PORT-PATTERN + VALIDATION | HIGH (full) / MED (modules) | LOW |
| Vidur | IMPORT-OPTIONAL / VALIDATION | MEDIUM | LOW |
| Mooncake (eqs/trace) | PORT-EQ + VENDOR-trace | LOW | LOW |
| vLLM | PORT-EQ + PORT-PATTERN | LOW (eqs) / HIGH (engine) | LOW |
| DistServe/Sarathi | PORT-EQ / PORT-PATTERN | LOW–MED | LOW |
| LLMRoofline | REJECT | — | **HIGH (no license)** |
| Orca clone | REJECT code | — | **HIGH (no license)** |
| LMCache / KVServe / Mooncake-TE | REJECT dep | HIGH | LOW |
| Alibaba clusterdata (trace) | reference at build time | MED | **MEDIUM (research-use only)** |
