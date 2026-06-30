"""Roofline serving-physics model + mechanism sensitivity sweeps (PR #109).

A conservative, analytical roofline model that distinguishes the two things PR #107 conflated:
**decode-PHASE-bound** (decode dominates time/work) vs **memory-BANDWIDTH-bound** (arithmetic intensity
below the GPU ridge point). Every serving mechanism — batching, prefill/decode allocation, speculative
decoding, clock/DVFS, precision, co-location — participates in the SAME physics (FLOPs, bytes, arithmetic
intensity, tokens/s, TTFT, completion latency, GPU-seconds, energy, cost) so each produces a real
**sensitivity curve**, never a bonus. The MPC may only *select* a mechanism that already has a connected
action surface (batching); the rest are swept **diagnostically** with explicit claim-safety labels.

Fidelity: GPU peak FLOPs / HBM bandwidth are PUBLIC_SPEC (vendor sheets); the roofline reduction (tokens/s
= min(compute, bandwidth)) is PUBLIC_PAPER (Williams roofline; vLLM/DistServe decode-is-bandwidth-bound);
the mechanism deltas (spec-decode/clock/precision/co-location) are SIMULATOR_INFERENCE with conservative
bands. Nothing here is UNKNOWN; nothing here touches reward directly.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- GPU peak compute + HBM bandwidth (PUBLIC_SPEC, vendor sheets, dense BF16) ----------------------
GPU_SPECS = {
    "H100": {"peak_flops": 989e12, "mem_bw": 3350e9, "tdp_w": 700.0},     # SXM 80GB
    "H800": {"peak_flops": 989e12, "mem_bw": 3350e9, "tdp_w": 700.0},
    "A100": {"peak_flops": 312e12, "mem_bw": 2039e9, "tdp_w": 400.0},     # 80GB
    "A800": {"peak_flops": 312e12, "mem_bw": 2039e9, "tdp_w": 400.0},
    "L20": {"peak_flops": 119e12, "mem_bw": 864e9, "tdp_w": 275.0},
    "A10": {"peak_flops": 125e12, "mem_bw": 600e9, "tdp_w": 150.0},
    "H20": {"peak_flops": 148e12, "mem_bw": 4000e9, "tdp_w": 400.0},      # bandwidth-rich, compute-light
}
DEFAULT_GPU = "A100"

# --- model FLOPs + byte footprint (BENCHMARK_DERIVED from published architecture) -------------------
# llama-8b-gqa: 8.03e9 params, 32 layers, 8 kv heads, head_dim 128 (matches kv_cache.FOOTPRINTS).
MODEL_SPECS = {
    "llama-8b-gqa": {"params": 8.03e9, "kv_bytes_per_token": 2 * 32 * 8 * 128 * 2},   # 131072 B/tok (fp16)
    "llama-70b-gqa": {"params": 70e9, "kv_bytes_per_token": 2 * 80 * 8 * 128 * 2},
}
DEFAULT_MODEL = "llama-8b-gqa"
PRECISION_BYTES = {"fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
# KV-cache precision is a SEPARATE dtype from the model weights (vLLM ships fp8/int8 KV independent of the
# weight precision). "inherit" follows the weight precision (the no-op: KV bytes == today's behaviour).
KV_PRECISION_BYTES = {"inherit": None, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}
GPU_HOUR_USD = 2.0
ENERGY_USD_PER_KWH = 0.06


@dataclass
class Workload:
    prompt_tokens: int = 512
    decode_tokens: int = 128
    prefix_hit_frac: float = 0.0        # share of prompt skipped by KV reuse (PR #106)
    context_len: int = 512              # KV context per active sequence
    n_requests: int = 1


@dataclass
class ServingConfig:
    """All mechanism settings. Defaults are NEUTRAL (a sweep varies one at a time)."""
    gpu: str = DEFAULT_GPU
    model: str = DEFAULT_MODEL
    precision: str = "fp16"
    kv_precision: str = "inherit"       # KV-cache dtype, SEPARATE from weights; "inherit" = follow `precision`
    batch_size: int = 16
    prefill_decode_ratio: float = 0.5   # share of replicas given to PREFILL (disaggregated); 0.5 = balanced
    serving_mode: str = "shared_gpu"    # shared_gpu | disaggregated_static
    # diagnostic-only mechanisms (no live action surface):
    spec_decode_accept: float = 0.0     # 0 = off; else expected token-acceptance rate of the draft
    spec_decode_draft_frac: float = 0.2 # draft-model FLOPs as a fraction of target per proposed token
    clock_factor: float = 1.0           # DVFS: compute scales with clock; bandwidth ~constant
    colocation_frac: float = 0.0        # background compute-bound work offered to idle SMs (0 = none)


def _ridge_point(gpu: str) -> float:
    g = GPU_SPECS.get(gpu, GPU_SPECS[DEFAULT_GPU])
    return g["peak_flops"] / g["mem_bw"]


def _kv_bytes_per_token(cfg: "ServingConfig", m: dict) -> float:
    """KV bytes/token at the config's KV precision. The footprint number is fp16 (2 B/elem); scale it by the
    KV dtype's bytes. ``kv_precision='inherit'`` follows the WEIGHT precision (the no-op: identical to the
    historical ``kv_bytes = kv_bytes_per_token·(weight_pb/2)``). An explicit KV dtype decouples KV from
    weights — the whole point of the kv_cache_precision knob (fp8/int8 KV with bf16 weights)."""
    kv_pb = KV_PRECISION_BYTES.get(cfg.kv_precision)
    if kv_pb is None:                                  # "inherit" (or unknown) → follow the weight precision
        kv_pb = PRECISION_BYTES.get(cfg.precision, 2)
    return m["kv_bytes_per_token"] * (kv_pb / 2.0)


def arithmetic_intensity(phase: str, cfg: ServingConfig, wl: Workload) -> float:
    """FLOPs per byte for one phase (the roofline x-axis). Decode at low batch is memory-bound (weights
    streamed per token); batching amortises weight bytes over the batch → AI rises. Prefill processes
    many prompt tokens per weight load → higher AI."""
    m = MODEL_SPECS.get(cfg.model, MODEL_SPECS[DEFAULT_MODEL])
    pb = PRECISION_BYTES.get(cfg.precision, 2)
    weight_bytes = m["params"] * pb
    kv_bytes = _kv_bytes_per_token(cfg, m)               # KV dtype is separate from weights (inherit = follow)
    flops_per_token = 2 * m["params"]                    # 1 MAC/param/token ≈ 2 FLOPs
    b = max(1, cfg.batch_size)
    if phase == "prefill":
        # prefill processes `prompt_tokens` per weight load (per request); batch amortises further.
        tok = max(1, wl.prompt_tokens) * b
        bytes_moved = weight_bytes + kv_bytes * tok
        return (flops_per_token * tok) / max(bytes_moved, 1.0)
    # decode: 1 new token/seq/step; weights streamed once per step, amortised over the batch.
    bytes_moved = weight_bytes + kv_bytes * wl.context_len * b
    return (flops_per_token * b) / max(bytes_moved, 1.0)


def roofline_regime(phase: str, cfg: ServingConfig, wl: Workload) -> dict:
    """Classify compute-bound vs memory-bandwidth-bound from arithmetic intensity vs the ridge point."""
    ai = arithmetic_intensity(phase, cfg, wl)
    ridge = _ridge_point(cfg.gpu)
    regime = "compute_bound" if ai >= ridge else "memory_bandwidth_bound"
    return {"phase": phase, "arithmetic_intensity": round(ai, 4), "ridge_point": round(ridge, 2),
            "roofline_regime": regime, "headroom": round(ai / ridge, 4)}


def _tokens_per_s(phase: str, cfg: ServingConfig, wl: Workload) -> float:
    """Roofline throughput = min(compute-limited, bandwidth-limited) tokens/s, scaled by clock (compute
    only) + precision. Conservative; PUBLIC_PAPER roofline reduction."""
    g = GPU_SPECS.get(cfg.gpu, GPU_SPECS[DEFAULT_GPU])
    m = MODEL_SPECS.get(cfg.model, MODEL_SPECS[DEFAULT_MODEL])
    pb = PRECISION_BYTES.get(cfg.precision, 2)
    flops_per_token = 2 * m["params"]
    compute = g["peak_flops"] * cfg.clock_factor / flops_per_token            # tokens/s if compute-bound
    weight_bytes = m["params"] * pb
    kv_bytes = _kv_bytes_per_token(cfg, m)               # KV dtype is separate from weights (inherit = follow)
    b = max(1, cfg.batch_size)
    if phase == "prefill":
        bw_tokens = g["mem_bw"] / max(weight_bytes / (wl.prompt_tokens * b) + kv_bytes, 1.0)
    else:
        bw_tokens = g["mem_bw"] / max(weight_bytes / b + kv_bytes * wl.context_len, 1.0)
    return max(1.0, min(compute, bw_tokens))


def _power_w(cfg: ServingConfig) -> float:
    """DVFS: power scales ~ clock^2.4 of TDP (conservative cubic-ish); memory power roughly flat."""
    g = GPU_SPECS.get(cfg.gpu, GPU_SPECS[DEFAULT_GPU])
    return g["tdp_w"] * (0.4 + 0.6 * (cfg.clock_factor ** 2.4))


def serving_point(wl: Workload, cfg: ServingConfig) -> dict:
    """Full serving physics for one (workload, config): TTFT, completion latency, GPU-seconds, energy,
    cost, SLA risk, roofline regime — every mechanism flows through here. Analytical (no event sim),
    deterministic, conservative. The single source the sensitivity sweeps call."""
    prefill_tok = max(0, int(wl.prompt_tokens * (1.0 - wl.prefix_hit_frac)))   # KV reuse cuts prefill only
    decode_tok = max(1, wl.decode_tokens)

    pf_tps = _tokens_per_s("prefill", cfg, wl)
    dc_tps = _tokens_per_s("decode", cfg, wl)

    # speculative decoding (diagnostic): a draft proposes k tokens, target verifies in one pass; accepted
    # tokens skip serial decode steps but every proposal pays draft + verify FLOPs. Helps ONLY when decode
    # is memory-bandwidth-bound (spare compute) AND acceptance is high; hurts when compute-bound.
    dc_regime = roofline_regime("decode", cfg, wl)["roofline_regime"]
    spec_speedup, spec_compute_overhead = 1.0, 0.0
    if cfg.spec_decode_accept > 0:
        a = cfg.spec_decode_accept
        spec_compute_overhead = cfg.spec_decode_draft_frac + a   # extra draft + verify FLOPs
        serial_reduction = 1.0 + a * 2.0                          # accepted tokens reduce serial steps
        if dc_regime == "memory_bandwidth_bound":
            spec_speedup = serial_reduction / (1.0 + 0.1 * spec_compute_overhead)  # spare compute → wins
        else:
            # compute-bound: the extra draft+verify FLOPs compete for the scarce resource; the serial
            # reduction cannot help when you are FLOP-limited → spec decode SLOWS wall-clock.
            spec_speedup = 1.0 / (1.0 + spec_compute_overhead)
    dc_tps_eff = dc_tps * spec_speedup

    prefill_work_s = prefill_tok / pf_tps
    decode_work_s = decode_tok / dc_tps_eff
    # disaggregation: split capacity; a wrong split queues one phase. handoff overhead on disaggregation.
    handoff_s = 0.0
    if cfg.serving_mode == "disaggregated_static":
        pf_share = min(0.95, max(0.05, cfg.prefill_decode_ratio))
        # effective per-phase capacity = total × its share; under-provisioning a phase inflates its work.
        prefill_work_s /= max(pf_share * 2.0, 0.1)               # 2.0 normaliser so 0.5/0.5 ≈ shared
        decode_work_s /= max((1.0 - pf_share) * 2.0, 0.1)
        handoff_s = 0.004                                         # KV handoff prefill→decode (PUBLIC_PAPER)

    ttft_s = prefill_work_s + handoff_s
    completion_s = ttft_s + decode_work_s
    gpu_seconds = (prefill_work_s + decode_work_s) * (1.0 + spec_compute_overhead * 0.0)  # time-based GPU-s
    # speculative decoding raises COMPUTE demand (extra FLOPs) even when it cuts latency → more GPU-s.
    gpu_seconds *= (1.0 + (spec_compute_overhead if cfg.spec_decode_accept > 0 else 0.0) * 0.5)

    # co-location (diagnostic): background compute-bound work uses idle SMs ONLY in memory-bound decode;
    # it adds memory pressure → a foreground latency penalty. Credits extra useful GPU-seconds only when
    # SM headroom is real (memory-bound) and bounded by colocation_frac.
    coloc_useful_gpu_s, coloc_penalty = 0.0, 1.0
    if cfg.colocation_frac > 0:
        if dc_regime == "memory_bandwidth_bound":
            coloc_useful_gpu_s = decode_work_s * cfg.colocation_frac      # idle SMs do real work
            coloc_penalty = 1.0 + 0.15 * cfg.colocation_frac             # but adds memory pressure
        else:
            coloc_penalty = 1.0 + 0.6 * cfg.colocation_frac             # compute-bound → pure interference
    completion_s *= coloc_penalty
    ttft_s *= coloc_penalty

    energy_j = gpu_seconds * _power_w(cfg)
    cost_usd = gpu_seconds * GPU_HOUR_USD / 3600.0 + (energy_j / 3.6e6) * ENERGY_USD_PER_KWH
    pf_gpu_s, dc_gpu_s = prefill_work_s, decode_work_s
    _m = MODEL_SPECS.get(cfg.model, MODEL_SPECS[DEFAULT_MODEL])
    kv_bpt = _kv_bytes_per_token(cfg, _m)                 # KV bytes/token at the config's KV precision
    share = dc_gpu_s / max(pf_gpu_s + dc_gpu_s, 1e-9)
    phase_bottleneck = ("decode_phase_bound" if share > 0.66 else
                        ("prefill_phase_bound" if share < 0.34 else "mixed_phase_bound"))
    return {
        "ttft_s": round(ttft_s, 5), "completion_s": round(completion_s, 5),
        "gpu_seconds": round(gpu_seconds + coloc_useful_gpu_s, 5),
        "serving_gpu_seconds": round(gpu_seconds, 5), "coloc_useful_gpu_seconds": round(coloc_useful_gpu_s, 5),
        "prefill_gpu_seconds": round(pf_gpu_s, 5), "decode_gpu_seconds": round(dc_gpu_s, 5),
        "decode_gpu_sec_share": round(share, 4), "phase_bottleneck": phase_bottleneck,
        "decode_regime": dc_regime, "prefill_regime": roofline_regime("prefill", cfg, wl)["roofline_regime"],
        "decode_arithmetic_intensity": round(arithmetic_intensity("decode", cfg, wl), 4),
        "ridge_point": round(_ridge_point(cfg.gpu), 2),
        "energy_j": round(energy_j, 3), "cost_usd": round(cost_usd, 8),
        "tokens_per_s_decode": round(dc_tps_eff, 1), "tokens_per_s_prefill": round(pf_tps, 1),
        "spec_speedup": round(spec_speedup, 4), "power_w": round(_power_w(cfg), 1),
        "coloc_penalty": round(coloc_penalty, 5), "kv_bytes_per_token": round(kv_bpt, 1)}


def _set(cfg: ServingConfig, **kw) -> ServingConfig:
    return ServingConfig(**{**cfg.__dict__, **kw})


def _classify(curve, key, baseline_idx) -> dict:
    """help/hurt/neutral regions for a swept metric vs the baseline (lower cost/latency = help)."""
    base = curve[baseline_idx][key]
    out = []
    for pt in curve:
        d = (pt[key] - base) / abs(base) if base else 0.0
        out.append("help" if d < -0.02 else ("hurt" if d > 0.02 else "neutral"))
    return out


def sweep_mechanism(mechanism: str, wl: Workload, base: ServingConfig | None = None) -> dict:
    """Sweep one mechanism across a reasonable operating range → a sensitivity curve of
    (setting → roofline regime, TTFT, completion, GPU-seconds, energy, SLA-risk proxy, cost) and the
    region where it helps / hurts / is neutral (vs the neutral baseline). Diagnostic — the MPC only
    selects mechanisms whose action surface already exists (batching)."""
    base = base or ServingConfig()
    # mechanisms now wired as LIVE MPC actions via roofline_actions.py (this PR). co_location +
    # prefill_decode_allocation remain diagnostic (SIMULATED_ONLY, frozen off — no background-work trace /
    # no disaggregated capacity pools). Batching was the only live one at PR #109.
    live_mechanisms = ("batching", "precision", "speculative_decoding", "clock_dvfs")
    grids = {
        "batching": ("batch_size", [1, 2, 4, 8, 16, 32, 64, 128]),
        "prefill_decode_allocation": ("prefill_decode_ratio", [0.2, 0.4, 0.5, 0.6, 0.8]),
        "speculative_decoding": ("spec_decode_accept", [0.0, 0.3, 0.5, 0.7, 0.9]),
        "clock_dvfs": ("clock_factor", [0.7, 0.85, 1.0, 1.15]),
        "precision": ("precision", ["fp16", "fp8", "int4"]),
        "co_location": ("colocation_frac", [0.0, 0.25, 0.5, 0.75]),
    }
    if mechanism not in grids:
        raise ValueError(f"unknown mechanism {mechanism}")
    field_name, values = grids[mechanism]
    if mechanism == "prefill_decode_allocation":
        base = _set(base, serving_mode="disaggregated_static")
    curve = []
    for v in values:
        pt = serving_point(wl, _set(base, **{field_name: v}))
        pt["setting"] = v
        curve.append(pt)
    base_idx = next((i for i, v in enumerate(values) if v in (0.0, 1.0, "fp16", 16, 0.5)), 0)
    verdict = {k: _classify(curve, k, base_idx) for k in ("completion_s", "gpu_seconds", "cost_usd", "energy_j")}
    # the action-surface status of this mechanism (precision/spec/clock are live via roofline_actions).
    live = mechanism in live_mechanisms
    return {"mechanism": mechanism, "field": field_name, "settings": values, "baseline_index": base_idx,
            "curve": curve, "help_hurt_neutral": verdict,
            "action_surface": "live_mpc_action" if live else "diagnostic_sweep_only",
            "claim_safety": "simulator_inferred"}


def all_sensitivity_curves(wl: Workload, base: ServingConfig | None = None) -> dict:
    """Every mechanism's sensitivity curve for a workload (the Phase-9 unified diagnostic table)."""
    return {mech: sweep_mechanism(mech, wl, base) for mech in
            ("batching", "prefill_decode_allocation", "speculative_decoding", "clock_dvfs",
             "precision", "co_location")}


__all__ = ["Workload", "ServingConfig", "GPU_SPECS", "MODEL_SPECS", "PRECISION_BYTES",
           "KV_PRECISION_BYTES", "arithmetic_intensity", "roofline_regime", "serving_point",
           "sweep_mechanism", "all_sensitivity_curves", "DEFAULT_GPU", "DEFAULT_MODEL"]
