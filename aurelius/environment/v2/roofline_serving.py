"""RooflineServingModelV2 — live FLOP/bandwidth serving timing (V2, ports from #110 audit).

Replaces the scalar `PREFILL_S_PER_TOKEN` / `TPOT_S` / fixed-batch-factor timing as the PRIMARY model with
a per-request roofline derived from GPU type, model profile, precision, prompt/output tokens, batch size,
active-sequence count, and context length — using the formulas ported in `roofline_external.py`
(InferSim / llm-analysis / LLM-Viewer). The scalar model is preserved as `legacy_scalar` for the validation
baseline and as a fallback when roofline inputs are missing.

Precision / spec-decode / clock are modelled as **physical modifiers on the roofline**, never as bonuses:
  * precision  → bytes moved (weights + KV) and compute peak (fp8 doubles tensor-core FLOPS); int4 carries a
    conservative quality/risk surcharge (raises risk-adjusted SLA miss), so it only wins when memory pressure
    dominates.
  * spec-decode → `decode = max(compute·draft_overhead, memory/accept_speedup)`: it divides the memory-bound
    leg (fewer weight reloads) but multiplies the compute leg (draft+verify) — so it helps memory-bound,
    high-acceptance decode and hurts compute-bound or low-acceptance decode.
  * clock      → scales peak FLOPS and power; touches the compute leg only (HBM BW fixed), so down-clock helps
    memory-bound energy and hurts compute-bound latency.

All effects flow into prefill/decode service seconds, HBM pressure, and energy — the scheduler/simulator turn
those into TTFT, completion latency, GPU-seconds, SLA, and cost. Nothing here touches reward directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..prefill_decode import PREFILL_S_PER_TOKEN, TPOT_S, TTFT_BASE_S
from ..roofline_external import ARCHS, GPU_SPECS, GPUSpec, ModelArch, roofline_analyze

# precision: (bytes-per-element scale vs bf16, compute-peak scale, quality_risk surcharge fraction)
PRECISION = {
    "bf16": (1.0, 1.0, 0.0),
    "fp8":  (0.5, 2.0, 0.01),     # fp8 ~2x tensor-core FLOPS; tiny risk (BENCHMARK_DERIVED band)
    "int4": (0.25, 2.0, 0.06),    # int4 weights 0.5B/elem-ish; conservative 6% quality/risk surcharge
}
# spec-decode: (memory-leg acceptance speedup, compute-leg draft+verify overhead)
SPEC_DECODE = {
    "off":        (1.0, 1.0),
    "shallow":    (1.3, 1.15),
    "medium":     (1.6, 1.35),
    "aggressive": (2.0, 1.7),
}
# clock/power: (peak-FLOPS scale, power scale)
CLOCK = {"low": (0.75, 0.7), "base": (1.0, 1.0), "high": (1.15, 1.35)}

PRECISIONS = tuple(PRECISION)
SPEC_MODES = tuple(SPEC_DECODE)
CLOCK_STATES = tuple(CLOCK)


@dataclass
class TimingResult:
    """Per-request serving timing + roofline diagnostics from V2."""
    timing_model_used: str
    prefill_time_s: float
    decode_time_s: float
    ttft_s: float
    tpot_s: float
    completion_latency_s: float
    arithmetic_intensity: float
    ridge_point: float
    roofline_regime: str                  # "compute" | "memory" | "mixed"
    compute_util_estimate: float
    memory_bandwidth_util_estimate: float
    hbm_pressure: float
    quality_risk: float
    provenance_label: str = "BENCHMARK_DERIVED"
    extra: dict = field(default_factory=dict)


@dataclass
class RooflineServingModelV2:
    """Live roofline timing model. ``mode='roofline'`` (default) or ``'legacy_scalar'`` (baseline/fallback)."""

    gpu_type: str = "H100"
    arch_name: str = "llama-8b-gqa"
    mode: str = "roofline"               # "roofline" | "legacy_scalar"
    prefill_mfu: float | None = None     # override arch/gpu default
    decode_mfu: float | None = None

    def _gpu(self) -> GPUSpec:
        return GPU_SPECS.get(self.gpu_type, GPU_SPECS["H100"])

    def _arch(self) -> ModelArch:
        return ARCHS.get(self.arch_name, ARCHS["llama-8b-gqa"])

    # -- per-token roofline legs (precision/clock aware) -----------------
    def _decode_legs(self, arch, gpu, *, context, batch, byte_scale, peak_scale, mfu):
        """Return (compute_time_s, memory_time_s, ai, ridge) for ONE decode token at this batch/context."""
        peak = gpu.peak_fp16_tflops * 1e12 * mfu * peak_scale
        bw = gpu.hbm_bw_gbps * 1e9 * gpu.bw_derate
        attn = 4.0 * max(1, context) * (arch.n_heads * arch.head_dim)
        linear = 2.0 * arch.linear_params_per_layer
        ops = arch.n_layers * (linear + attn) * batch
        mem = arch.weight_bytes * byte_scale + arch.kv_bytes_per_token * byte_scale * max(1, context) * batch
        ai, _, _, _ = roofline_analyze(peak, bw, ops, mem)
        ridge = peak / bw if bw > 0 else float("inf")
        return ops / peak, mem / bw, ai, ridge

    def _prefill_legs(self, arch, gpu, *, prompt, byte_scale, peak_scale, mfu):
        peak = gpu.peak_fp16_tflops * 1e12 * mfu * peak_scale
        bw = gpu.hbm_bw_gbps * 1e9 * gpu.bw_derate
        s = max(1, prompt)
        attn = 4.0 * s * (arch.n_heads * arch.head_dim)
        linear = 2.0 * arch.linear_params_per_layer
        ops = arch.n_layers * (linear + attn)
        mem = arch.weight_bytes * byte_scale / s + arch.kv_bytes_per_token * byte_scale
        ai, _, _, _ = roofline_analyze(peak, bw, ops, mem)
        ridge = peak / bw if bw > 0 else float("inf")
        return ops / peak, mem / bw, ai, ridge

    def estimate(self, *, prompt_tokens: int, output_tokens: int, prefill_tokens_remaining: int | None = None,
                 context_tokens: int | None = None, batch: int = 1, active_sequences: int = 1,
                 precision: str = "bf16", spec_decode: str = "off", clock: str = "base") -> TimingResult:
        """Per-request prefill/decode timing. ``prefill_tokens_remaining`` (after a KV hit) defaults to
        the full prompt. ``context_tokens`` defaults to prompt + half the output (mean decode context)."""
        arch, gpu = self._arch(), self._gpu()
        remaining = prompt_tokens if prefill_tokens_remaining is None else max(0, prefill_tokens_remaining)
        ctx = context_tokens if context_tokens is not None else prompt_tokens + output_tokens // 2

        if self.mode == "legacy_scalar":
            prefill = TTFT_BASE_S + remaining * PREFILL_S_PER_TOKEN
            decode = output_tokens * TPOT_S
            return TimingResult("legacy_scalar", prefill, decode, prefill, TPOT_S, prefill + decode,
                                arithmetic_intensity=0.0, ridge_point=0.0, roofline_regime="mixed",
                                compute_util_estimate=0.0, memory_bandwidth_util_estimate=0.0,
                                hbm_pressure=0.0, quality_risk=0.0, provenance_label="BENCHMARK_DERIVED_LEGACY")

        byte_scale, peak_scale, qrisk = PRECISION.get(precision, PRECISION["bf16"])
        spec_speedup, spec_overhead = SPEC_DECODE.get(spec_decode, SPEC_DECODE["off"])
        clk_flops, _clk_pwr = CLOCK.get(clock, CLOCK["base"])
        pmfu = self.prefill_mfu if self.prefill_mfu is not None else gpu.mfu_prefill
        dmfu = self.decode_mfu if self.decode_mfu is not None else gpu.mfu_decode

        # prefill (compute-bound regime expected); precision lowers bytes, clock scales compute
        pc, pm, p_ai, p_ridge = self._prefill_legs(arch, gpu, prompt=max(1, remaining),
                                                   byte_scale=byte_scale, peak_scale=peak_scale * clk_flops, mfu=pmfu)
        prefill_per_tok = max(pc, pm)
        prefill = TTFT_BASE_S + remaining * prefill_per_tok

        # decode (memory-bound regime expected at low batch); spec divides memory leg, multiplies compute leg
        dc, dm, d_ai, d_ridge = self._decode_legs(arch, gpu, context=ctx, batch=max(1, batch),
                                                 byte_scale=byte_scale, peak_scale=peak_scale * clk_flops, mfu=dmfu)
        dc_eff = dc * spec_overhead
        dm_eff = dm / spec_speedup
        decode_per_tok = max(dc_eff, dm_eff) / max(1, batch)   # weight cost amortised across the batch
        decode = output_tokens * decode_per_tok

        regime = "compute" if d_ai >= d_ridge else "memory"
        compute_util = min(1.0, dc_eff / max(dc_eff, dm_eff)) if max(dc_eff, dm_eff) > 0 else 0.0
        mem_util = min(1.0, dm_eff / max(dc_eff, dm_eff)) if max(dc_eff, dm_eff) > 0 else 0.0
        # HBM pressure: KV bytes for the active set vs capacity (precision lowers it)
        kv_active = arch.kv_bytes_per_token * byte_scale * max(1, ctx) * max(1, active_sequences)
        hbm_bytes = gpu.hbm_gib * (1024 ** 3)
        hbm_pressure = min(1.5, kv_active / max(1.0, hbm_bytes - arch.weight_bytes * byte_scale))

        return TimingResult(
            "roofline", round(prefill, 6), round(decode, 6), round(prefill, 6), round(decode_per_tok, 8),
            round(prefill + decode, 6), arithmetic_intensity=round(d_ai, 3), ridge_point=round(d_ridge, 3),
            roofline_regime=regime, compute_util_estimate=round(compute_util, 4),
            memory_bandwidth_util_estimate=round(mem_util, 4), hbm_pressure=round(hbm_pressure, 4),
            quality_risk=round(qrisk, 4), provenance_label="BENCHMARK_DERIVED",
            extra={"precision": precision, "spec_decode": spec_decode, "clock": clock,
                   "prefill_per_tok": round(prefill_per_tok, 8), "decode_per_tok": round(decode_per_tok, 8),
                   "prefill_ai": round(p_ai, 3), "prefill_ridge": round(p_ridge, 3),
                   "byte_scale": byte_scale, "peak_scale": peak_scale * clk_flops})

    def power_scale(self, clock: str = "base") -> float:
        """Clock→power multiplier (feeds the energy term in the simulator)."""
        return CLOCK.get(clock, CLOCK["base"])[1]


__all__ = ["RooflineServingModelV2", "TimingResult", "PRECISION", "SPEC_DECODE", "CLOCK",
           "PRECISIONS", "SPEC_MODES", "CLOCK_STATES"]
