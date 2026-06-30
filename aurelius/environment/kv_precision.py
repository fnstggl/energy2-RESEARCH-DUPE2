"""KV-cache precision physics — the memory / HBM-pressure / active-sequence-capacity channel for the
``kv_cache_precision_policy`` action (Batch-1 Phase 2).

The roofline (``roofline.serving_point`` via ``roofline_actions``) already carries the *latency / GPU-second*
effect of a smaller KV cache: in the memory-bandwidth-bound decode regime, fewer KV bytes/token raise decode
tokens/s → lower decode service time → cost/SLA. That is the reward channel and it is bit-for-bit unchanged
at the no-op (``inherit_weight_precision``).

This module supplies the *other* causal effect of KV quantization — the one the roofline's fixed-batch model
does not express: **KV bytes/token set how many active sequences (and how much prefix cache) fit in HBM.**
Halving KV bytes doubles the KV budget in blocks → more concurrent decode sequences before HBM saturates,
lower eviction pressure, lower HBM pressure. These are reported as honest diagnostics (and used by the
controlled fixtures + the planner's HBM-pressure regime gate); they are NOT a second reward bonus.

Fidelity: KV bytes/token = 2·layers·kv_heads·head_dim·dtype_bytes is BENCHMARK_DERIVED (published model
architecture; same formula as ``kv_cache.KVFootprint``). The *latency* magnitude of KV quantization is
SIMULATOR_INFERENCE (roofline band). The quality risk of low-precision KV is SIMULATOR_INFERENCE: fp8/int8 KV
are ~lossless in public results (PUBLIC_BENCHMARK_DERIVED → headline-eligible); **int4 KV carries an
UNMODELLED quality risk → labelled unsafe / diagnostic-only**, never a headline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kv_cache import FOOTPRINTS, KVFootprint, gpu_mem_for
from .roofline import KV_PRECISION_BYTES, PRECISION_BYTES

# action policy → roofline KV precision string. "inherit_weight_precision" is the no-op (KV follows weights).
KV_CACHE_PRECISION_TO_ROOFLINE = {
    "inherit_weight_precision": "inherit",
    "default": "inherit",
    "kv_bf16": "bf16",
    "kv_fp8": "fp8",
    "kv_int8": "int8",
    "kv_int4_diagnostic_only": "int4",
}
KV_CACHE_PRECISION_OPTIONS = (
    "inherit_weight_precision", "kv_bf16", "kv_fp8", "kv_int8", "kv_int4_diagnostic_only")

# headline-safe KV precisions (quality risk modelled/negligible). int4 KV is excluded → diagnostic-only.
# fp8/int8 KV ≈ lossless in public results (NVIDIA TensorRT-LLM, vLLM fp8 KV); int4 has no quality model.
KV_PRECISION_QUALITY_RISK = {
    "inherit": 0.0, "bf16": 0.0, "fp16": 0.0, "fp8": 0.0, "int8": 0.0, "int4": 0.06}
HEADLINE_SAFE_KV = frozenset({"inherit_weight_precision", "default", "kv_bf16", "kv_fp8", "kv_int8"})


def _kv_dtype_bytes(kv_precision: str, *, weight_precision: str = "bf16") -> float:
    """KV bytes/element for a roofline KV-precision string. ``inherit`` follows the weight precision."""
    b = KV_PRECISION_BYTES.get(kv_precision)
    if b is None:
        b = PRECISION_BYTES.get(weight_precision, 2)
    return float(b)


def kv_bytes_per_token(footprint: KVFootprint, kv_precision: str, *, weight_precision: str = "bf16") -> float:
    """KV bytes/token at ``kv_precision`` (the fp16 footprint scaled by the KV dtype)."""
    return footprint.bytes_per_token * (_kv_dtype_bytes(kv_precision, weight_precision=weight_precision) / 2.0)


@dataclass
class KVPrecisionEffect:
    """The memory/capacity effect of a KV precision vs the inherit (no-op) baseline."""
    kv_precision: str
    kv_bytes_per_token: float
    baseline_kv_bytes_per_token: float
    kv_memory_saved_pct: float
    active_sequence_capacity_before: int
    active_sequence_capacity_after: int
    capacity_gain_pct: float
    hbm_pressure_before: float
    hbm_pressure_after: float
    cache_eviction_delta: float           # change in occupancy/eviction pressure (<0 = less eviction)
    quality_risk: float
    headline_safe: bool

    def to_dict(self) -> dict:
        return {
            "kv_precision": self.kv_precision,
            "kv_bytes_per_token": round(self.kv_bytes_per_token, 1),
            "kv_memory_saved_pct": round(self.kv_memory_saved_pct, 2),
            "active_sequence_capacity_before": self.active_sequence_capacity_before,
            "active_sequence_capacity_after": self.active_sequence_capacity_after,
            "capacity_gain_pct": round(self.capacity_gain_pct, 2),
            "hbm_pressure_before": round(self.hbm_pressure_before, 4),
            "hbm_pressure_after": round(self.hbm_pressure_after, 4),
            "cache_eviction_delta": round(self.cache_eviction_delta, 4),
            "quality_risk": round(self.quality_risk, 4),
            "headline_safe": self.headline_safe}


def kv_precision_memory_effect(
    policy: str, *, gpu_type: str = "A100", model: str = "llama-8b-gqa",
    context_tokens: int = 832, active_sequences: int = 64, weight_precision: str = "bf16",
    mem_pressure: float = 0.0) -> KVPrecisionEffect:
    """Causal memory effect of a ``kv_cache_precision_policy`` value on one GPU/model.

    ``active_sequences`` is the offered concurrent-decode demand; ``context_tokens`` the per-sequence KV
    length. The KV budget (HBM − weights) holds ``capacity = budget / (bytes_per_token·context)`` sequences;
    a lower KV precision raises ``capacity`` (→ HBM pressure ``offered/capacity`` falls). The no-op
    (``inherit_weight_precision``) returns zero deltas. Deterministic."""
    fp = FOOTPRINTS.get(model, FOOTPRINTS["llama-8b-gqa"])
    kv_precision = KV_CACHE_PRECISION_TO_ROOFLINE.get(policy, "inherit")
    base_bpt = kv_bytes_per_token(fp, "inherit", weight_precision=weight_precision)
    act_bpt = kv_bytes_per_token(fp, kv_precision, weight_precision=weight_precision)
    gpu_mem_gib = gpu_mem_for(gpu_type)
    free_bytes = max(0.0, gpu_mem_gib - fp.weight_gib) * 0.9 * (1.0 - max(0.0, min(1.0, mem_pressure)))
    free_bytes *= 1024 ** 3
    per_seq_base = base_bpt * max(1, context_tokens)
    per_seq_act = act_bpt * max(1, context_tokens)
    cap_before = max(1, int(free_bytes / per_seq_base))
    cap_after = max(1, int(free_bytes / per_seq_act))
    saved_pct = 100.0 * (1.0 - act_bpt / base_bpt) if base_bpt else 0.0
    gain_pct = 100.0 * (cap_after - cap_before) / cap_before if cap_before else 0.0
    hbm_before = min(1.0, active_sequences / cap_before)
    hbm_after = min(1.0, active_sequences / cap_after)
    # eviction pressure proxy: occupancy above a 0.9 working-set threshold drives LRU eviction.
    evict_before = max(0.0, hbm_before - 0.9)
    evict_after = max(0.0, hbm_after - 0.9)
    return KVPrecisionEffect(
        kv_precision=kv_precision, kv_bytes_per_token=act_bpt, baseline_kv_bytes_per_token=base_bpt,
        kv_memory_saved_pct=saved_pct, active_sequence_capacity_before=cap_before,
        active_sequence_capacity_after=cap_after, capacity_gain_pct=gain_pct,
        hbm_pressure_before=hbm_before, hbm_pressure_after=hbm_after,
        cache_eviction_delta=evict_after - evict_before,
        quality_risk=KV_PRECISION_QUALITY_RISK.get(kv_precision, 0.0),
        headline_safe=policy in HEADLINE_SAFE_KV)


def is_headline_safe_kv(policy: str) -> bool:
    return policy in HEADLINE_SAFE_KV


__all__ = ["KV_CACHE_PRECISION_TO_ROOFLINE", "KV_CACHE_PRECISION_OPTIONS", "KV_PRECISION_QUALITY_RISK",
           "HEADLINE_SAFE_KV", "KVPrecisionEffect", "kv_bytes_per_token", "kv_precision_memory_effect",
           "is_headline_safe_kv"]
