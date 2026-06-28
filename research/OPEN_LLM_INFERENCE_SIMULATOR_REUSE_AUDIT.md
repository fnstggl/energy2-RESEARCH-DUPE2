# Open LLM-Inference Simulator Reuse Audit (Phase 0)

**Question.** Before Aurelius builds more custom serving physics, can existing open-source LLM inference
simulators / roofline tools / serving-system simulators be imported, ported, or combined to make
Aurelius' world model much more production-realistic?

**Method.** Each project below was inspected at the **source level** (repo file tree + raw source files
on `raw.githubusercontent.com`, papers, configs, tests), not just READMEs. Where a fact could not be
verified at the source it is flagged `UNVERIFIED`. `UNKNOWN` is not used — every cell is a decision.

**Aurelius baseline (for reference).** A Python hybrid analytical + token-level discrete-event simulator:
Azure request spine (`unified_replay`), Mooncake `hash_ids` KV prefix-reuse, Alibaba `cluster-trace-gpu-v2026`
hourly fleet/topology spine, ISO electricity cost. A **persistent `CanonicalWorldState`** (servers/racks/
replicas, warm/cold, migration) that is **cloned per MPC candidate** and is deterministic/clone-safe. The
roofline is currently *approximate* (`prefill = TTFT_BASE + remaining_prompt·0.00015 s/tok`;
`decode = out·0.020 s/tok·batch_factor`); see "Aurelius gaps" in `WORLD_MODEL_REALISM_GAP_AUDIT.md`.

Verdicts use: **IMPORT** (add as dependency) · **VENDOR** (copy a module) · **PORT-EQ** (re-implement
equations) · **PORT-PATTERN** (copy a design) · **VALIDATION** (reference baseline only) · **REJECT**.

---

## Summary table

| # | Project | Type | License | Status | Lang | Standalone | Counterfactual | Verdict |
|--|--|--|--|--|--|--|--|--|
| 1 | **LLMServingSim 2.0** | profiled-table + ASTRA-sim DES | MIT | active (2026-06) | Py + C++ | needs C++ build | yes (strong) | PORT-PATTERN + VALIDATION |
| 2 | **BLIS / inference-sim** | DES + roofline/trained-physics | Apache-2.0 | active (2026-06) | Go | yes (CPU) | yes | PORT-EQ + VALIDATION |
| 3 | **llm-d-inference-sim** | runtime mock (sleeps) | Apache-2.0 | active (2026-06) | Go | yes | params only | PORT-EQ (load-factor) / REJECT engine |
| 4 | **SplitwiseSim** | DES + profiled table | MIT | stale (2024-04) | Py | yes (CPU) | yes (strong) | PORT-EQ + PORT-PATTERN + VALIDATION |
| 5 | **Alibaba InferSim** | analytical roofline | Apache-2.0 | active (2026-05) | Py (no deps) | yes | yes | **PORT-EQ (primary)** ✅ |
| 6 | **LLMRoofline** | roofline plot (decode-only) | **none** | stale (2024-03) | Notebook | partial | no | **REJECT** |
| 7 | **LMCache** | KV-cache runtime layer | Apache-2.0 | active (2026-06) | Py+CUDA | no (GPU) | no | PORT-PATTERN (tiering/CacheBlend) / REJECT dep |
| 8 | **Mooncake** | KVCache-centric runtime + **trace** | Apache-2.0 | active (2026-05) | C++/Py | engine no / trace yes | no | VENDOR (trace, in use) + PORT-EQ (Conductor) |
| 9 | **KVServe** | KV-compression vLLM plugin | Apache-2.0 | active (2026-05) | Py | no | no | REJECT (not a sim) |
| 10 | **vLLM / PagedAttention** | serving runtime + scheduler | Apache-2.0 | active (2026-06) | Py+CUDA | no (GPU) | no | PORT-EQ + PORT-PATTERN (LRU reference) |
| 11 | **Vidur** (MS) | DES + RandomForest predictor | MIT | lightly stale (2025-07) | Py (CPU) | yes | yes | IMPORT-OPTIONAL / VALIDATION |
| 12 | **llm-analysis** | analytical calculator | Apache-2.0 | stale (2024-11) | Py | yes | yes | PORT-EQ (ridge point + GPU JSON) |
| 13 | **LLM-Viewer** | roofline calculator | MIT | stale (2024-09) | Py | yes | yes | PORT-EQ (`roofline_analyze` label) |
| — | DistServe / Sarathi / Orca | papers + artifacts | Apache-2.0 / — | mixed | Py | mixed | sim artifacts | PORT-EQ / PORT-PATTERN / VALIDATION |

---

## 1. LLMServingSim 2.0 — `casys-kaist/LLMServingSim` (MIT, active)

- **Paper:** IISWC 2024 (arXiv 2408.05499), CAL 2025, ISPASS 2026. Reports <14.7% error vs real GPU serving.
- **Type:** Hybrid — **profiled-latency-table** compute timing (CSV interpolation, *no* roofline/FLOP) +
  **ASTRA-sim** event/cycle-level network simulation of a Chakra execution graph + a vLLM-style serving
  scheduler (continuous batching, chunked prefill, radix-tree prefix cache).
- **Standalone?** No — requires building the **ASTRA-sim** C++ submodule + **Chakra**; `vLLM`/`ns-3` optional;
  `gem5` not needed. Run = `python -m serving --cluster-config <path>`. Validation harness under `bench/`
  compares against real vLLM; CI present; no unit-test dir surfaced.
- **Inputs:** cluster JSON (topology, TP/PP/EP/DP), model-arch YAML, **profiled latency CSVs**
  (`profiler/perf/<hw>/<model>/...`), workload JSONL (flat or agentic sessions; optional `input_tok_ids`
  for prefix matching). **Outputs:** per-request TTFT/TPOT/ITL/queue-delay, throughput, NPU mem/util,
  prefix-cache hit ratios (NPU/CPU/CXL), **power/energy in kJ**.
- **Counterfactual:** Strong — change cluster JSON / CLI and re-simulate (instance count, parallelism,
  `max-num-seqs`, prefix-cache on/off, chunked prefill, routing policy, dtype, network backend, mem tiers).
- **World model:** Owns its own, rebuilt per run from JSON; ASTRA-sim holds the network world. Models
  replicas/instances, per-replica KV + radix prefix cache, prefill→decode disaggregation, decode→CPU
  preemption. **Missing for Aurelius:** rack/server hierarchy (flat NPU map), warm/cold lifecycle,
  autoscaling, cross-host live migration, persistent externally-owned state.
- **Verdict:** **PORT-PATTERN** (radix prefix cache + tier-aware eviction-cost + chunked-prefill scheduler) +
  **VALIDATION**. Importing wholesale is HIGH effort (C++ build, per-hw profiling corpus) and would not
  participate in Aurelius MPC rollouts (subprocess, owns its state). License risk LOW (verify ASTRA-sim
  subcomponent if vendored — observed MIT).

## 2. BLIS / Blackbox Inference Simulator — `inference-sim/inference-sim` (Apache-2.0, active)

- **Found** (the "BLIS" the brief asked about). Closest paper: `inference-fleet-sim` (arXiv 2603.16054,
  M/G/c queueing + DES + physics-informed GPU model) — adjacent, does not name BLIS verbatim (UNVERIFIED link).
- **Type:** Discrete-event simulator (Go) with **two pluggable latency back-ends**: (a) **Roofline**
  (`sim/latency/roofline.go`, pure FLOPs/bandwidth) and (b) **Trained-Physics** (default — physics basis
  functions with *learned* β/α coefficients + MoE/TP terms; an ML-surrogate/calibrated hybrid). CPU-only,
  deterministic (partitioned RNG → byte-identical results).
- **Standalone?** Yes — single Go binary; optional HuggingFace `config.json` fetch (works offline with
  cached configs). Tests + `examples/` (policy YAML, ServeGen workloads, disaggregation/autoscaling demos).
- **Inputs:** model spec, hardware/parallelism, workload (Poisson/Gamma/Weibull/ServeGen/multiturn/trace),
  policy YAML. **Outputs (JSON):** TTFT (mean/P90/P95/P99), ITL/TPOT, E2E percentiles, throughput, KV peak +
  time-integral usage, cache hit/thrash rate, KV alloc failures, preemptions, drops/timeouts.
- **Counterfactual:** Yes — capacity planning (sweep instances/traffic), independent policy swap, decision
  tracing + top-k candidate ranking (regret machinery lives in the cluster layer, not `metrics.go` — UNVERIFIED).
- **World model:** Owns its own per-run snapshot; `Finalize()` tears down. A recent `RequestSource` interface
  + per-arrival trace hooks hint at pluggable inputs. Not built for an external persistent ClusterState.
- **Verdict:** **PORT-EQ** (the `roofline.go` / `trained_physics_model.go` formulas are the highest-value,
  cleanly-factored pieces) + **VALIDATION**. Go ⇒ re-implement the math in Python (it is plain arithmetic),
  don't FFI the engine. License risk LOW.

## 3. llm-d-inference-sim — `llm-d/llm-d-inference-sim` (Apache-2.0, active)

- **Type:** A **runtime mock/emulator**, not a physics or DES simulator. An OpenAI/vLLM-compatible HTTP+gRPC
  server that returns realistically-shaped fake responses with realistically-*timed* `time.Sleep` delays —
  **no model, no event queue, no simulated clock.** Part of the K8s-native llm-d stack (tests schedulers/
  gateway/PD-disaggregation/LoRA without GPUs). Deps MEDIUM-HIGH (K8s controller-runtime, fasthttp, grpc, zmq).
- **Latency logic (`latencies.go`, verbatim):** linear load factor
  `1 + (timeFactorUnderLoad-1)·(nRunning-1)/(maxNumSeqs-1)`; prefill
  `overhead·LF + (prompt − cached)·perToken·LF`; ITL `interTokenLatency·LF` with Gaussian jitter. No queueing
  theory, no roofline, no batch-execution model. Concurrency = fixed worker pool of size `max-num-seqs`.
- **World model:** No persistent global state; each process is one stateless emulated replica;
  multi-replica behaviour is external (llm-d gateway + ZMQ KV events).
- **Verdict:** **PORT-EQ** for the trivial load-factor + prefill-uncached-token formulas (an afternoon to
  re-derive in Python); **REJECT** the engine. License risk LOW.

## 4. SplitwiseSim — `Mutinifni/splitwise-sim` (MIT, stale)

- **Paper:** Splitwise, ISCA 2024 (arXiv 2311.18677, Microsoft). **Type:** Discrete-event sim with a **custom
  heapq event engine** (NOT SimPy, despite common belief) + a **profiled-latency table** model
  (`DatabasePerformanceModel`, scipy `interp1d` over `batch_tokens = prompt·batch`). CPU-only, no ML framework.
- **Standalone?** Yes (Hydra configs); no unit tests (example shell scripts only); ships `traces/test_trace.csv`.
- **Inputs:** request-trace CSV (`arrival_timestamp, prompt_size, token_size, …`), Hydra YAML (cluster, hardware
  a100/h100, models, schedulers, start-state). **Outputs:** per-request `ttft_times`, `tbt_times` (=TPOT),
  e2e/queue times, throughput, power. **Counterfactual:** Strong — its entire purpose (fleet composition,
  prompt/token-pool disaggregation, 11 schedulers incl. KV-aware, power caps, Ray multirun sweeps).
- **Key equations:** iteration time = `prompt_predictor(batch_tokens)` (all-prompt) / `token_predictor`
  (all-token) / `prompt_predictor·1.1` (mixed); contiguous-iteration collapsing for decode; KV bytes
  `2·B·T·H·L·dtype`; **KV transfer time `= kv_bytes / bandwidth`** with a 10× overlap trick (layer-wise);
  TTFT/TBT definitions.
- **World model:** Owns it, rebuilt per run. No racks/topology hierarchy (flat `DummyLink`), no warm/cold or
  migration, KV is **bulk byte accounting (no LRU/eviction/prefix reuse)**. Module-level singletons (`sim`,
  `performance_model`) ⇒ not re-entrant for MPC.
- **Verdict:** **PORT-EQ** (layer-wise KV-transfer cost, mixed-batch penalty, profiled-table interpolation) +
  **PORT-PATTERN** (prompt/token pool disaggregation) + **VALIDATION**. License risk LOW (MIT; profiling data CC-BY).

## 5. Alibaba InferSim — `alibaba/InferSim` (Apache-2.0, active)  ✅ primary roofline source

- **Type:** Analytical **per-stage roofline** + profiled-MFU lookup. **Pure Python, zero third-party deps.**
  Explicitly separates prefill (TTFT) vs decode (TPOT). Calibrated vs real hw (DeepSeek-V3/H800, Qwen3/H20/H800;
  ~4–15% of measured throughput).
- **Standalone?** Yes (`python3 main.py --config-path <hf config.json> --device-type ...`), runnable examples
  per model; no formal unit-test dir (examples are smoke tests).
- **Core formula (`layers/attn.py`, verbatim):** `return max(attn_core_time, kv_load_time)` where
  `attn_core_time = bs·gflops/(fp16_tflops·1024·mfu)` and `kv_load_time = kvcache_bytes·kv_len·bs/layers/.../mem_bw`.
  `gemm_flops = 2·m·n·k`. KV bytes `2·layers·kv_heads·head_dim·dtype` — **identical to Aurelius**. GPU table
  (`hardware/gpu.py`) includes **capacity** and an empirical `mem_bw·0.8` derate + prefill `÷1.8` efficiency.
- **Counterfactual:** Yes (device, world/tp size, fp8, batch, isl/osl). **World model:** stateless calculator —
  composes cleanly into Aurelius (no world to own).
- **Verdict:** **PORT-EQ (primary).** Re-implemented in `aurelius/environment/roofline_external.py`
  (see Phase 5/7). License risk LOW.

## 6. LLMRoofline — `feifeibear/LLMRoofline`  ❌ reject

- **License: NONE** (`GET /license` → null, no `LICENSE` file → all-rights-reserved). **Stale** (15 commits,
  last 2024-03). Decode-only; outputs a relative arithmetic-intensity *ratio* (`min(peak_flop, bw·AI)`), **no
  latency in seconds, no HBM capacity**; hardcodes local Mac paths; needs the LLM-Viewer submodule for the
  useful path. Its one good idea (ridge-point selector) exists license-clean in LLM-Viewer and InferSim.
- **Verdict:** **REJECT.** License risk HIGH; do not copy. (Math is textbook — re-derive from clean sources.)

## 7. LMCache — `LMCache/LMCache` (Apache-2.0, active)

- **Type:** Real KV-cache management **runtime layer** (daemon), not a simulator. Tiered KV (GPU→pinned CPU
  DRAM→local SSD→remote: Redis/S3/Mooncake/NIXL). LRU eviction; **256-token chunk** granularity (vs vLLM 16).
  **CacheBlend** (EuroSys'25, arXiv 2405.16444): non-prefix positional KV reuse with ~10–15% selective
  recompute (HKVD top-r% per layer), recompute/load pipelining `T_recompute(r,L)=r·Prefill(L)`.
- **World model:** runtime, CUDA-bound. **Verdict:** **PORT-PATTERN** (tiering hierarchy) + **PORT-EQ**
  (CacheBlend overlap — *only if* Aurelius adds non-prefix/RAG reuse; Mooncake `hash_ids` are prefix-only so
  it is out of scope today); **REJECT** as dependency. License risk LOW.

## 8. Mooncake — `kvcache-ai/Mooncake` (Apache-2.0, active)

- **Type:** KVCache-centric **disaggregated runtime** (prefill/decode clusters + pooled DRAM/SSD KV store +
  RDMA Transfer Engine) and the **Mooncake trace** (already an Aurelius input). FAST'25 (arXiv 2407.00079).
- **Trace fields (verbatim JSONL):** `timestamp` (ms), `input_length`, `output_length`, `hash_ids`
  (array of 512-token block-hash ints; shared leading run = prefix hit). No `block_size` field (512 is a
  convention). No model id, no measured hit rate (the ~50% reuse ceiling is an aggregate trace property).
- **Conductor scheduling (Algorithm 1):** `TTFT = T_queue + T_prefill` (low hit) or
  `T_transfer + T_queue + T_prefill` (remote reuse); **prediction-based early rejection**: admit iff
  `TTFT ≤ TTFT_SLO ∧ predicted TBT ≤ TBT_SLO`; goodput counts only fully-completed requests.
- **Verdict:** **VENDOR** the trace (in use, Apache-2.0) + **PORT-EQ** (Conductor TTFT decomposition + early-
  reject SLO test); **REJECT** the Transfer Engine (C++/RDMA). **Granularity mismatch to document:** Mooncake
  512-token blocks vs Aurelius/vLLM 16-token paging. License risk LOW.

## 9. KVServe — `hpdps-group/KVServe` (Apache-2.0, active)

- A **vLLM KV-compression connector plugin** (SIGCOMM 2026), not a simulator. **Verdict:** **REJECT** (at most
  a VALIDATION baseline if Aurelius ever models KV-transfer *compression*). License risk LOW.

## 10. vLLM / PagedAttention — `vllm-project/vllm` (Apache-2.0, active)

- The de-facto reference for the **LRU paged KV cache** Aurelius already mirrors. V1 paths:
  `vllm/v1/core/{block_pool,kv_cache_manager}.py`, `kv_cache_interface.py` (legacy `core/scheduler.py` removed).
  KV bytes/block `= 2·block_size·num_kv_heads·head_dim·dtype` (default block_size 16); `num_blocks =
  available_memory // page_size // num_layers`; V1 unifies prefill/decode under a `max_num_batched_tokens`
  token budget with chunked-prefill clamping; LRU free-queue evicts the least-reusable tail first.
- **Verdict:** **PORT-EQ + PORT-PATTERN** (re-implement, don't import — the engine is CUDA/torch-bound).
  License risk LOW.

## 11. Vidur — `microsoft/vidur` (MIT, lightly stale 2025-07)

- **Type:** CPU-only **DES + data-driven runtime predictor** (RandomForest over profiled operator CSVs for
  A100/A40/H100; *not* a roofline). Features: token matmuls `[num_tokens]`; prefill-attn
  `[kv_cache_size, prefill_chunk_size²]`; decode-attn `[batch_size, kv_cache_size]`; TP comm
  `all_reduce + nccl_skew·tp^1.25`. Real schedulers (vllm/sarathi/orca/lightllm). <9% mean latency error.
- **Verdict:** The best MIT-clean *accuracy* reference. **IMPORT-OPTIONAL** (CPU-only, deep-copyable Python —
  could in principle feed an Aurelius timing oracle) / **VALIDATION**. Caveat: predictor coupled to Vidur's
  `Batch`/`Request` entities + CSV schema; new GPU/model needs a one-time profiling run. License risk LOW.

## 12. llm-analysis — `cli99/llm-analysis` (Apache-2.0, stale 2024-11)

- Analytical calculator with explicit ridge point `get_pivot() = peak_TFLOPS·bits/8 / hbm_bw / 2`, GQA-aware
  attention/MLP FLOPs, and **GPU-config JSON with `mem_per_GPU_in_GB`** (the capacity column). Has real tests/CI.
- **Verdict:** **PORT-EQ** (ridge-point classifier + GPU JSON schema with HBM capacity, complementing InferSim).
  License risk LOW–MEDIUM (Apache NOTICE).

## 13. LLM-Viewer — `hahnyuan/LLM-Viewer` (MIT, stale 2024-09)

- The cleanest standalone `roofline_analyze(bandwidth, max_OPS, OPs, memory_access)` →
  `(arithmetic_intensity, performance, bound)` with `turning_point = max_OPS/bandwidth`. Per-layer prefill/decode
  OPs; `memory_access = load_weight+load_act+store_act+load_kv+store_kv`. **Verdict:** **PORT-EQ** (the
  compute/memory-bound LABEL). License risk LOW (MIT).

## Disaggregation / scheduling papers (artifacts)

- **DistServe** (OSDI'24, arXiv 2401.09670; `LLMServe/DistServe` Apache-2.0, ships `simdistserve` simpy DES):
  prefill **M/D/1** TTFT `D + R·D²/(2(1−R·D))`; goodput under per-phase SLO. **PORT-EQ + PORT-PATTERN.**
- **Splitwise** (see #4): layer-wise KV-transfer overlap cost. **PORT-EQ + VALIDATION.**
- **Sarathi-Serve** (OSDI'24, arXiv 2403.02310; `microsoft/sarathi-serve` Apache-2.0 + Vidur): **chunked-prefill
  stall-free batching** (Algorithm 3: pack decodes → fill budget with partial prefills → admit new last).
  Token budget chosen empirically via Vidur (512 strict / 2048 relaxed). Naive hybrid inflates TBT up to 28×.
  **PORT-PATTERN (Algorithm 3)** + **VALIDATION.**
- **Orca** (OSDI'22): **iteration-level scheduling (continuous batching) + selective batching.** No official
  artifact (FriendliAI proprietary); third-party clone is **unlicensed — do not copy.** **PORT-PATTERN** (idea) only.

---

## Cross-cutting conclusions

1. **No single open simulator is a drop-in replacement for Aurelius** (see Phase 6). Each owns its own
   ephemeral world, rebuilt per run; none exposes a persistent, externally-owned, clone-per-candidate
   `ClusterState` that an MPC controller can roll forward. They are *serving-microsimulators*, not *fleet
   world models with counterfactual control*.
2. **Port equations, reference engines.** The clean, permissively-licensed, directly-portable pieces are:
   InferSim/llm-analysis/LLM-Viewer roofline; vLLM KV-bytes + LRU design; Mooncake Conductor TTFT + early-
   reject; DistServe M/D/1; Splitwise layer-wise transfer; Sarathi Algorithm 3; Orca continuous batching.
   The runtime engines (LMCache, Mooncake TE, KVServe, vLLM, Sarathi-Serve, llm-d) are reject-as-dependency.
3. **The genuinely proprietary signals — identical across every public trace and repo — are:** real
   per-replica KV residency/eviction state, real measured per-request cache hit rates, real cross-node KV
   transfer bandwidth under production congestion, true model identity per request, and prompt content. These
   must be *modelled and labelled* in Aurelius, never *sourced* (see `REQUIRES_PROPRIETARY_DATA` cells in the
   capability matrix).
4. **One elevated license risk:** Alibaba `clusterdata` (research-use only, no formal redistribution license) —
   link/download at build time, do not vendor raw rows. Everything else Aurelius depends on (Azure CC-BY-4.0,
   Mooncake/vLLM/InferSim/Vidur permissive) is LOW risk.
