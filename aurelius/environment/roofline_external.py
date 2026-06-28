"""External-roofline performance model — ported formulas (PR audit, Phase 5/7).

A *physically-grounded* prefill/decode latency model to sit beside the existing benchmark-constant
band in :mod:`prefill_decode` (``PREFILL_S_PER_TOKEN`` / ``TPOT_S``). This is the "port equations,
validate against controlled fixtures" outcome of
``research/OPEN_LLM_INFERENCE_SIMULATOR_REUSE_AUDIT.md`` — it is NOT a vendored external repo and
adds **no dependencies** (pure Python, deterministic, clone-safe).

Formulas are ported (re-implemented, not copied) from three permissively-licensed sources, cross-checked
against each other (see ``research/ROOFLINE_REUSE_DECISION.md`` for exact source files):

  * **Alibaba InferSim** (Apache-2.0) — per-stage roofline ``time = max(compute_time, mem_time)`` with
    ``gemm_flops = 2·m·n·k``; GQA/MoE-aware FLOPs; KV bytes ``2·layers·kv_heads·head_dim·dtype``;
    GPU spec table with the empirical ``mem_bw·0.8`` derate and the prefill kernel-efficiency divisor.
  * **llm-analysis** (Apache-2.0) — GQA-aware attention/MLP FLOPs; the explicit ridge point
    ``pivot = peak_FLOPS·dtype_bytes / mem_BW``; GPU-config JSON schema *with HBM capacity*.
  * **LLM-Viewer** (MIT) — the clean ``roofline_analyze`` ridge-point classifier returning a
    compute/memory-bound LABEL.

This module computes the *physical floor* (ideal-MFU) and an MFU-derated estimate. It is exposed as a
validation/calibration reference; it does NOT replace the existing service-time path. Every Aurelius KV
footprint number it uses is the SAME ``2·layers·kv_heads·head_dim·dtype`` formula already in
``kv_cache.py`` (all three external sources confirm it), so the two stay consistent by construction.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# GPU spec table (BENCHMARK_DERIVED public spec sheets).
# peak_fp16_tflops: dense FP16/BF16 TFLOPS · hbm_bw_gbps: peak HBM bandwidth GB/s
# hbm_gib: HBM capacity (matches kv_cache.GPU_MEM_GIB) · mfu_prefill/decode: realistic utilisation.
# Sources cross-checked: llm-analysis gpu_configs/*.json, InferSim hardware/gpu.py, public spec sheets.
# The 0.8 HBM-bandwidth derate (InferSim convention) is applied at call time, not baked here.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GPUSpec:
    name: str
    peak_fp16_tflops: float
    hbm_bw_gbps: float
    hbm_gib: float
    mfu_prefill: float = 0.7      # realistic compute-bound prefill MFU (InferSim/Megatron band ~0.5–0.8)
    mfu_decode: float = 0.35      # decode is memory-bound; "MFU" only caps the compute leg
    bw_derate: float = 0.8        # achievable HBM fraction (InferSim empirical derate)


GPU_SPECS = {
    # name             TFLOPS   HBM GB/s  HBM GiB
    "H100": GPUSpec("H100", 989.0, 3350.0, 80.0),
    "H800": GPUSpec("H800", 989.0, 3430.0, 80.0),
    "H200": GPUSpec("H200", 989.0, 4800.0, 141.0),
    "H20":  GPUSpec("H20",  148.0, 4096.0, 96.0),
    "A100": GPUSpec("A100", 312.0, 2039.0, 80.0),
    "A800": GPUSpec("A800", 312.0, 2039.0, 80.0),
    "A10":  GPUSpec("A10",  125.0,  600.0, 24.0),
    "L40S": GPUSpec("L40S", 362.0,  864.0, 48.0),
    "L40":  GPUSpec("L40",  181.0,  864.0, 48.0),
    "L20":  GPUSpec("L20",  119.0,  864.0, 48.0),
    "V100": GPUSpec("V100", 125.0,  900.0, 32.0),
    "T4":   GPUSpec("T4",    65.0,  320.0, 16.0),
}
DEFAULT_GPU = "H100"


# ---------------------------------------------------------------------------
# Model architecture (BENCHMARK_DERIVED public configs). Extends kv_cache.FOOTPRINTS with the
# hidden / intermediate / n_heads needed for the FLOP roofline. dtype_bytes = 2 (fp16/bf16).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelArch:
    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    gated_mlp: bool = True        # SwiGLU (3 matmuls) vs vanilla (2)
    dtype_bytes: int = 2

    @property
    def gqa_groups(self) -> int:
        return max(1, self.n_heads // self.n_kv_heads)

    @property
    def kv_bytes_per_token(self) -> int:
        # IDENTICAL to kv_cache.KVFootprint.bytes_per_token (2·layers·kv_heads·head_dim·dtype).
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def linear_params_per_layer(self) -> int:
        """Params in the per-layer linear projections (q,k,v,o + MLP). 2·this = linear FLOPs/token."""
        h, hd = self.hidden, self.head_dim
        q = h * (self.n_heads * hd)                      # q_proj
        kv = 2 * h * (self.n_kv_heads * hd)              # k_proj + v_proj (GQA-reduced)
        o = (self.n_heads * hd) * h                      # o_proj
        mlp = (3 if self.gated_mlp else 2) * h * self.intermediate
        return q + kv + o + mlp

    @property
    def weight_bytes(self) -> int:
        return self.n_layers * self.linear_params_per_layer * self.dtype_bytes


ARCHS = {
    # name             L   hidden heads kv  hd   inter
    "llama-8b-gqa":  ModelArch("llama-8b-gqa",  32, 4096, 32, 8, 128, 14336),
    "llama-7b-mha":  ModelArch("llama-7b-mha",  32, 4096, 32, 32, 128, 11008),
    "llama-70b-gqa": ModelArch("llama-70b-gqa", 80, 8192, 64, 8, 128, 28672),
    "qwen-7b-gqa":   ModelArch("qwen-7b-gqa",   28, 3584, 28, 4, 128, 18944),
}
DEFAULT_ARCH = "llama-8b-gqa"


def _gemm_flops(m: int, n: int, k: int) -> float:
    """InferSim: a matmul (m,k)·(k,n) costs 2·m·n·k FLOPs (multiply + add)."""
    return 2.0 * m * n * k


def roofline_analyze(peak_flops: float, bw_bytes_s: float, ops: float, mem_bytes: float):
    """LLM-Viewer ridge-point classifier.

    ``turning_point = peak_FLOPS / BW`` (FLOPs/byte). If arithmetic intensity < ridge → memory-bound
    (achievable = AI·BW), else compute-bound (achievable = peak). Returns
    ``(arithmetic_intensity, achievable_flops_s, bound, time_s)``.
    """
    ridge = peak_flops / bw_bytes_s if bw_bytes_s > 0 else float("inf")
    ai = ops / mem_bytes if mem_bytes > 0 else float("inf")
    if ai < ridge:
        bound = "memory"
        achievable = ai * bw_bytes_s
    else:
        bound = "compute"
        achievable = peak_flops
    time_s = ops / achievable if achievable > 0 else 0.0
    return ai, achievable, bound, time_s


@dataclass
class StageEstimate:
    seconds_per_token: float
    bound: str
    arithmetic_intensity: float
    compute_time_s: float
    memory_time_s: float


def prefill_estimate(arch: ModelArch, gpu: GPUSpec, *, prompt_tokens: int = 512,
                     ideal_mfu: bool = False) -> StageEstimate:
    """Per-prompt-token prefill time via roofline (compute-bound regime expected).

    Compute (per token) = linear FLOPs (2·linear_params) + attention score/value FLOPs over the prompt
    (causal ≈ ½·prompt average, approximated with the full prompt as the conservative upper band).
    Memory (per token) = weights streamed + KV written. Time = max(compute, memory)/peak — the roofline.
    """
    mfu = 1.0 if ideal_mfu else gpu.mfu_prefill
    peak = gpu.peak_fp16_tflops * 1e12 * mfu
    bw = gpu.hbm_bw_gbps * 1e9 * gpu.bw_derate
    # attention FLOPs per token: QK^T + AV ≈ 2·(2·S·H) where H = n_heads·head_dim ≈ hidden.
    s = max(1, prompt_tokens)
    attn_per_tok = 4.0 * s * (arch.n_heads * arch.head_dim)
    linear_per_tok = 2.0 * arch.linear_params_per_layer
    ops = arch.n_layers * (linear_per_tok + attn_per_tok)
    # prefill is throughput work: weights amortise across the whole prompt, so per-token weight traffic
    # is weight_bytes/prompt; KV write is kv_bytes_per_token. Compute dominates for non-trivial prompts.
    mem_bytes = arch.weight_bytes / s + arch.kv_bytes_per_token
    ai, _, bound, _ = roofline_analyze(peak, bw, ops, mem_bytes)
    compute_t = ops / peak
    mem_t = mem_bytes / bw
    return StageEstimate(max(compute_t, mem_t), bound, ai, compute_t, mem_t)


def decode_estimate(arch: ModelArch, gpu: GPUSpec, *, context_tokens: int = 2048, batch: int = 1,
                    ideal_mfu: bool = False) -> StageEstimate:
    """Per-output-token decode time via roofline (memory-bound regime expected at batch=1).

    One token, context S. Compute (per token) = linear + attention over S. Memory = weights streamed
    once per token (batch=1) + KV read over S. At batch=1 memory dominates → the classic decode floor;
    larger batches amortise weights (``/batch``) until the cache/compute limit.
    """
    mfu = 1.0 if ideal_mfu else gpu.mfu_decode
    peak = gpu.peak_fp16_tflops * 1e12 * mfu
    bw = gpu.hbm_bw_gbps * 1e9 * gpu.bw_derate
    s = max(1, context_tokens)
    attn_per_tok = 4.0 * s * (arch.n_heads * arch.head_dim)
    linear_per_tok = 2.0 * arch.linear_params_per_layer
    ops = arch.n_layers * (linear_per_tok + attn_per_tok) * batch
    # weights streamed once per step (amortised across the batch); KV read is per-sequence.
    mem_bytes = arch.weight_bytes + arch.kv_bytes_per_token * s * batch
    ai, _, bound, _ = roofline_analyze(peak, bw, ops, mem_bytes)
    compute_t = ops / peak
    mem_t = mem_bytes / bw
    per_token = max(compute_t, mem_t) / batch
    return StageEstimate(per_token, bound, ai, compute_t / batch, mem_t / batch)


def compare_to_aurelius_constants(arch_name: str = DEFAULT_ARCH, gpu_name: str = DEFAULT_GPU, *,
                                  prompt_tokens: int = 512, context_tokens: int = 2048) -> dict:
    """Side-by-side of the roofline floor vs the Aurelius benchmark constants (the validation use)."""
    from .prefill_decode import PREFILL_S_PER_TOKEN, TPOT_S
    arch = ARCHS[arch_name]
    gpu = GPU_SPECS[gpu_name]
    pf = prefill_estimate(arch, gpu, prompt_tokens=prompt_tokens)
    pf_ideal = prefill_estimate(arch, gpu, prompt_tokens=prompt_tokens, ideal_mfu=True)
    dc = decode_estimate(arch, gpu, context_tokens=context_tokens)
    dc_ideal = decode_estimate(arch, gpu, context_tokens=context_tokens, ideal_mfu=True)
    return {
        "arch": arch_name, "gpu": gpu_name,
        "prefill_roofline_s_per_token": round(pf.seconds_per_token, 8),
        "prefill_roofline_ideal_s_per_token": round(pf_ideal.seconds_per_token, 8),
        "prefill_bound": pf.bound,
        "aurelius_prefill_s_per_token": PREFILL_S_PER_TOKEN,
        "decode_roofline_s_per_token": round(dc.seconds_per_token, 8),
        "decode_roofline_ideal_s_per_token": round(dc_ideal.seconds_per_token, 8),
        "decode_bound": dc.bound,
        "aurelius_decode_s_per_token": TPOT_S,
    }


__all__ = ["GPUSpec", "GPU_SPECS", "ModelArch", "ARCHS", "StageEstimate", "roofline_analyze",
           "prefill_estimate", "decode_estimate", "compare_to_aurelius_constants",
           "DEFAULT_ARCH", "DEFAULT_GPU"]
