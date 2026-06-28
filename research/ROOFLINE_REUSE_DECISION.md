# Roofline Reuse Decision (Phase 5)

**Question.** Should Aurelius use **LLMRoofline** or **Alibaba InferSim** for roofline modeling?

**Decision.** **Port formulas from Alibaba InferSim (primary) + llm-analysis (ridge point + GPU spec
table with HBM capacity) + LLM-Viewer (clean compute/memory-bound classifier). REJECT LLMRoofline.** This
is decision option **3 (port formulas) + 4 (validation baseline)** — done in this PR as a *reference*
model; wiring it into the live service path is the recommended next build.

---

## Why not LLMRoofline

| factor | LLMRoofline | verdict |
|--|--|--|
| license | **none** (`GET /license` → null, no `LICENSE`) → all-rights-reserved | **disqualifying** (HIGH risk) |
| maintenance | 15 commits, last 2024-03-13 | stale |
| output | relative arithmetic-intensity *ratio* `min(peak_flop, bw·AI)` — **no latency in seconds** | unusable for service time |
| prefill/decode | decode-only (vary `kv_len`); no explicit prefill | insufficient |
| HBM capacity | absent | insufficient |
| portability | hardcodes a local Mac path; needs the LLM-Viewer submodule | poor |

Its one good idea (the ridge-point selector) is the *same* textbook `min(peak, bw·AI)` available
license-clean in LLM-Viewer (MIT) and InferSim (Apache-2.0). There is no reason to touch LLMRoofline.

## Why InferSim (primary)

- **Apache-2.0**, **actively maintained (2026-05)**, **pure Python with zero third-party deps**.
- Explicitly separates **prefill (TTFT)** vs **decode (TPOT)** and implements the exact
  `time = max(compute_time, memory_bandwidth_time)` roofline Aurelius wants, per operator.
- Its KV-cache footprint formula `2·layers·kv_heads·head_dim·dtype` is **identical** to Aurelius'
  `kv_cache.KVFootprint.bytes_per_token` — so the roofline and the live KV model stay consistent by construction.
- **Calibrated vs real hardware** (DeepSeek-V3/H800, Qwen3/H20/H800; within ~4–15% of measured throughput) →
  usable as a validation anchor, not just a formula source.

llm-analysis supplies the **explicit ridge point** (`get_pivot = peak·bits/8 / bw`) and a **GPU-config schema
with `mem_per_GPU_in_GB`** (InferSim's hardware dataclass omits capacity). LLM-Viewer supplies the **cleanest
standalone classifier** (`roofline_analyze` → memory/compute LABEL). The three cross-confirm each other's FLOP
and KV-byte formulas, which is why porting (re-implementing) is safe.

---

## Exact formulas ported (and where they live now)

Ported into `aurelius/environment/roofline_external.py` (re-implemented, not copied — no source text reused):

| formula | source (file) | Aurelius symbol |
|--|--|--|
| `gemm_flops(m,n,k) = 2·m·n·k` | InferSim `flops/flops.py` | `_gemm_flops` |
| KV bytes/token `= 2·layers·kv_heads·head_dim·dtype` | InferSim `kvcache/kvcache.py` = vLLM = llm-analysis | `ModelArch.kv_bytes_per_token` (== `kv_cache.KVFootprint`) |
| linear FLOPs/token `= 2·(q+kv+o+mlp params)` | llm-analysis `analysis.py` (GQA-aware) | `ModelArch.linear_params_per_layer` |
| attention FLOPs/token `≈ 4·S·(heads·head_dim)` (QK^T+AV) | InferSim `get_mha_gflops` / LLM-Viewer | `prefill_estimate`/`decode_estimate` |
| roofline `time = max(FLOPs/(peak·MFU), bytes/(BW·derate))` | InferSim `layers/attn.py` | `prefill_estimate`/`decode_estimate` |
| ridge `turning_point = peak/BW`; bound = AI<ridge?memory:compute | LLM-Viewer `roofline_analyze`; llm-analysis `get_pivot` | `roofline_analyze` |
| `mem_bw·0.8` achievable-bandwidth derate | InferSim `hardware/gpu.py` | `GPUSpec.bw_derate` |
| GPU table with **HBM capacity** | llm-analysis `gpu_configs/*.json` | `GPU_SPECS` (incl. `hbm_gib`) |

**Source file references for audit:** InferSim `layers/attn.py` (`decode_attn_core`/`prefill_attn_core`),
`flops/flops.py` (`gemm_flops`, `get_mha_gflops`, `get_moe_gflops`), `kvcache/kvcache.py`
(`get_mha_kvcache_size`), `hardware/gpu.py`; llm-analysis `analysis.py` (`get_num_flops_fwd_per_layer_attn/
mlp`, `get_pivot`), `gpu_configs/*.json`; LLM-Viewer `roofline_model.py` (`roofline_analyze`).

## Assumptions (and their fidelity tier)

| assumption | value | tier |
|--|--|--|
| peak FP16 TFLOPS / HBM GB/s / HBM GiB per GPU | public spec sheets | BENCHMARK_DERIVED |
| achievable-bandwidth derate | 0.8 (InferSim convention) | BENCHMARK_DERIVED / INFERRED |
| prefill MFU / decode MFU | 0.7 / 0.35 (Megatron/InferSim band) | BENCHMARK_DERIVED (constant; profiled MFU optional) |
| attention ≈ full prompt (causal upper band) | conservative | INFERRED (≤2× the causal-½ exact) |
| real per-kernel MFU, tile/wave quantization | not modelled | **ABSENT (PROP)** — needs profiling (Vidur) |

## What Aurelius must add itself (not in any roofline source)

- **Batch/iteration coupling:** the roofline gives a per-token floor at a given batch; mapping it onto the
  continuous-batching occupancy (Little's law / Sarathi Algorithm 3) is Aurelius' job (next PR).
- **Queueing:** the roofline is service-time only; the per-request DES queue (`unified_replay`) stays authoritative.
- **TP/PP/MoE timing:** `_gemm_flops` and the GQA arch make these DERIVABLE, but the comm terms (InferSim
  `_tp_comm`, Vidur `tp^1.25`) are not yet wired — deferred.
- **Profiled-MFU realism:** the constant-MFU floor is honest but optimistic on small batches; closing it needs
  Vidur-style profiled tables (calibration, not formula) — ABSENT without a profiling run.

## Tests / validation against controlled fixtures

`tests/test_roofline_external.py` (12 PASS) proves the ported physics is self-consistent:
- KV-byte formula **identical** to `kv_cache.KVFootprint` (no divergence between roofline and live cache).
- ridge-point classification correct; decode **memory-bound** at batch=1, prefill **compute-bound** at 2k prompt.
- monotonicities: longer context → slower decode; bigger model → slower; faster HBM → faster decode; batching
  amortises decode weights; ideal-MFU helps *only* the compute-bound stage.
- determinism (clone-safe).

`scripts/compare_external_roofline.py` prints the roofline floor vs Aurelius' constants. **Key validation
finding (measured):**

| model · GPU | roofline decode | Aurelius `TPOT_S` | roofline prefill | Aurelius `PREFILL_S_PER_TOKEN` |
|--|--|--|--|--|
| 8B · H100 | **5.3 ms/tok** | 20 ms | 0.021 ms/tok | 0.15 ms |
| 8B · A100 | 8.7 ms/tok | 20 ms | 0.065 ms/tok | 0.15 ms |
| 8B · **L40S** | **20.6 ms/tok** | 20 ms | 0.056 ms/tok | 0.15 ms |
| 70B · H100 | 51 ms/tok | 20 ms | **0.20 ms/tok** | 0.15 ms |

**Interpretation (honest).** Aurelius' single constants are *inside* the physical band but do not resolve it:
`TPOT_S=0.020` corresponds to an **L40S-class** single-stream 8B decode, and `PREFILL_S_PER_TOKEN=0.00015`
to a **70B-class** prefill. On an H100-heavy fleet the constants are ~4× too slow for 8B decode; on a 70B
workload they are ~2.5× too fast for decode. The roofline closes this by making service time a function of
(model, GPU, context, batch) instead of a fleet-wide scalar — the realism gain is the **4–40× spread** the
constant cannot express. This is exactly the deficit the audit set out to close, and it is now quantified.

## Decision recap

- **Use:** InferSim + llm-analysis + LLM-Viewer (port formulas) — implemented as a reference model this PR.
- **Reject:** LLMRoofline (license HIGH, decode-only, no seconds).
- **Next:** wire `roofline_external.prefill_estimate/decode_estimate` into `prefill_decode.compute_phase_serving`
  as a per-(model,GPU) service-time source behind a flag, with the constants kept as the conservative fallback;
  add profiled-MFU calibration (Vidur) as a validation baseline. No new runtime dependency is required.
