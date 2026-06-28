"""Roofline action physics — the bridge that makes precision / speculative-decoding / clock / co-location
/ prefill-decode-allocation **causal MPC actions** instead of diagnostic-only sweeps.

The honesty problem this solves: ``roofline.serving_point`` already simulates every mechanism, but it was
only ever called from a diagnostic script — the live reward path (``world_simulator.simulate_period`` →
``prefill_decode.compute_phase_serving`` → ``run_unified_replay``) never saw precision/spec/clock. This
module connects them **without** a reward bonus and **without** double-modelling the serving physics:

  * It maps a bundle's action policies → a single :class:`roofline.ServingConfig`.
  * It evaluates ``serving_point`` at the **action** config and the **neutral** (all-defaults) config for
    the period's representative workload, and returns the **ratios** (``decode_factor``, ``prefill_factor``,
    ``gpu_seconds_factor``, ``power_factor`` …). At default policies the two configs are identical, so every
    factor is exactly ``1.0`` and the live simulation is bit-for-bit unchanged.
  * ``compute_phase_serving`` multiplies its calibrated per-request prefill/decode times by those factors.
    **The live model keeps its calibrated absolute level (``TPOT_S``/``PREFILL_S_PER_TOKEN``); roofline
    supplies only the relative mechanism delta.** Every effect therefore reaches reward through service
    time → queue/SLA/goodput and GPU-seconds/energy → cost. Never a scalar bonus.

Fidelity: the roofline law (FLOPs/byte vs ridge) is PUBLIC_PAPER; the GPU peak/bandwidth/TDP are
PUBLIC_SPEC; the precision/spec/clock/co-location **magnitudes** are SIMULATOR_INFERENCE with conservative
bands. ``int4`` carries a conservative quality/SLA risk (no quality model → any int4 win is labelled unsafe
/ diagnostic). Co-location credits **no** background goodput (no background-work trace exists); it only ever
adds interference. Nothing here is UNKNOWN.
"""

from __future__ import annotations

from .roofline import (
    DEFAULT_GPU,
    GPU_SPECS,
    ServingConfig,
    Workload,
    serving_point,
)

# --- action policy → roofline ServingConfig parameter (the SINGLE physics mapping) ------------------
# precision_policy → roofline precision string (PRECISION_BYTES). "bf16" is the no-op baseline.
PRECISION_TO_ROOFLINE = {"bf16": "bf16", "fp8": "fp8", "int4": "int4"}
# spec_decode_policy → (expected acceptance rate, draft FLOPs fraction). "off" = acceptance 0 (no spec).
SPEC_DECODE_TO_ROOFLINE = {
    "off": (0.0, 0.0), "shallow": (0.3, 0.2), "medium": (0.5, 0.25), "aggressive": (0.7, 0.3)}
# clock_policy → DVFS clock factor (compute scales; bandwidth ~flat). "base" = nominal.
CLOCK_TO_ROOFLINE = {"base": 1.0, "low": 0.85, "high": 1.15}
# colocation_policy → background-work fraction offered to idle SMs. "off" = none.
COLOCATION_TO_ROOFLINE = {"off": 0.0, "conservative": 0.25, "aggressive": 0.5}
# prefill_decode_policy → (serving_mode, prefill share). "shared" = no disaggregation (the no-op).
PREFILL_DECODE_TO_ROOFLINE = {
    "shared": ("shared_gpu", 0.5), "p40_d60": ("disaggregated_static", 0.4),
    "p60_d40": ("disaggregated_static", 0.6)}
# precision quality/SLA risk: fraction of requests at quality risk (a wrong answer is an SLA failure).
# fp8 is ~lossless for inference (0, conservative); int4 carries a conservative risk with NO quality model
# → any int4 win is labelled unsafe/diagnostic. SIMULATOR_INFERENCE.
PRECISION_QUALITY_RISK = {"bf16": 0.0, "fp8": 0.0, "int4": 0.05}

NEUTRAL_POLICIES = {"precision_policy": "bf16", "spec_decode_policy": "off", "clock_policy": "base",
                    "colocation_policy": "off", "prefill_decode_policy": "shared"}


def _pol(bundle, name: str) -> str:
    """Read an action policy off a bundle / SimpleNamespace / dict, defaulting to its no-op."""
    if isinstance(bundle, dict):
        return str(bundle.get(name, NEUTRAL_POLICIES[name]))
    return str(getattr(bundle, name, NEUTRAL_POLICIES[name]))


def action_serving_config(bundle, *, gpu: str = DEFAULT_GPU, batch_size: int = 16,
                          model: str | None = None, prefix_hit_frac: float = 0.0) -> ServingConfig:
    """The roofline :class:`ServingConfig` implied by a bundle's roofline-action policies. A default
    bundle yields the neutral config (precision bf16, no spec, base clock, no co-location, shared)."""
    accept, draft = SPEC_DECODE_TO_ROOFLINE.get(_pol(bundle, "spec_decode_policy"), (0.0, 0.0))
    mode, ratio = PREFILL_DECODE_TO_ROOFLINE.get(_pol(bundle, "prefill_decode_policy"), ("shared_gpu", 0.5))
    cfg = ServingConfig(
        gpu=gpu if gpu in GPU_SPECS else DEFAULT_GPU, batch_size=max(1, int(batch_size)),
        precision=PRECISION_TO_ROOFLINE.get(_pol(bundle, "precision_policy"), "bf16"),
        spec_decode_accept=accept, spec_decode_draft_frac=draft if draft > 0 else 0.2,
        clock_factor=CLOCK_TO_ROOFLINE.get(_pol(bundle, "clock_policy"), 1.0),
        colocation_frac=COLOCATION_TO_ROOFLINE.get(_pol(bundle, "colocation_policy"), 0.0),
        serving_mode=mode, prefill_decode_ratio=ratio)
    if model is not None:
        cfg = ServingConfig(**{**cfg.__dict__, "model": model})
    return cfg


def representative_workload(recs, *, prefix_hit_frac: float = 0.0) -> Workload:
    """Median prompt / output tokens for a period's request records ``(arrival, out_tok, in_tok)``.
    Context ≈ prompt + half the output (KV grows during decode)."""
    if not recs:
        return Workload()
    outs = sorted(int(r[1]) for r in recs)
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for r in recs)
    om = outs[len(outs) // 2]
    pm = ins[len(ins) // 2]
    return Workload(prompt_tokens=max(1, pm), decode_tokens=max(1, om),
                    context_len=max(1, pm + om // 2), prefix_hit_frac=max(0.0, min(1.0, prefix_hit_frac)),
                    n_requests=len(recs))


def _ratio(act: dict, neutral: dict, key: str) -> float:
    nv = neutral.get(key, 0.0)
    av = act.get(key, 0.0)
    return (av / nv) if nv else 1.0


def roofline_action_factors(bundle, workload: Workload, *, gpu: str = DEFAULT_GPU, batch_size: int = 16,
                            background_work: bool = False) -> dict:
    """No-op-anchored causal multipliers for a bundle's roofline actions on ``workload``.

    Returns ratios vs the neutral (all-default) config — exactly ``1.0`` each at default policies. The
    factors are applied to the live per-request prefill/decode service times + realized GPU-seconds +
    power, so the action reaches reward only through physics. ``quality_sla_risk`` is the precision quality
    failure fraction (an SLA failure). Co-location credits background goodput **only** when
    ``background_work`` is True (no background trace → False → interference-only)."""
    neutral_cfg = action_serving_config({}, gpu=gpu, batch_size=batch_size)
    act_cfg = action_serving_config(bundle, gpu=gpu, batch_size=batch_size)
    neutral = serving_point(workload, neutral_cfg)
    act = serving_point(workload, act_cfg)
    coloc_useful = act.get("coloc_useful_gpu_seconds", 0.0) if background_work else 0.0
    return {
        "prefill_factor": _ratio(act, neutral, "prefill_gpu_seconds"),
        "decode_factor": _ratio(act, neutral, "decode_gpu_seconds"),
        "gpu_seconds_factor": _ratio(act, neutral, "serving_gpu_seconds"),
        "ttft_factor": _ratio(act, neutral, "ttft_s"),
        "completion_factor": _ratio(act, neutral, "completion_s"),
        "power_factor": _ratio(act, neutral, "power_w"),
        # co-location foreground interference penalty (>1 hurts; 1.0 at no co-location). The prefill/decode
        # factors are per-phase GPU-seconds ratios that EXCLUDE this completion-level penalty, so the phase
        # path applies it separately (else forced co-location would look free).
        "interference_factor": _ratio(act, neutral, "coloc_penalty"),
        "quality_sla_risk": PRECISION_QUALITY_RISK.get(_pol(bundle, "precision_policy"), 0.0),
        "coloc_useful_gpu_seconds": round(coloc_useful, 5),
        "decode_regime": act.get("decode_regime", "memory_bandwidth_bound"),
        "decode_arithmetic_intensity": act.get("decode_arithmetic_intensity", 0.0),
        "ridge_point": act.get("ridge_point", 0.0),
        "power_w": act.get("power_w", 0.0),
        "is_neutral": all(_pol(bundle, k) == v for k, v in NEUTRAL_POLICIES.items()),
    }


def is_neutral_roofline_bundle(bundle) -> bool:
    """True iff every roofline action is at its no-op default (→ all factors 1.0, live path unchanged)."""
    return all(_pol(bundle, k) == v for k, v in NEUTRAL_POLICIES.items())


# batching_policy → a representative active-batch size for the roofline (continuous-batching concurrency
# band). Used so the precision/spec interaction is evaluated AT the chosen batch (batching ↔ precision).
BATCH_SIZE_FOR_ROOFLINE = {"conservative": 8, "balanced": 16, "aggressive": 32}


def _batching(bundle) -> str:
    if isinstance(bundle, dict):
        return str(bundle.get("batching_policy", "balanced"))
    return str(getattr(bundle, "batching_policy", "balanced"))


def period_action_modulation(bundle, recs, *, gpu: str = DEFAULT_GPU, prefix_hit_frac: float = 0.0,
                             background_work: bool = False) -> dict | None:
    """Representative-workload roofline factors for a period's ``bundle`` + request records. Returns
    ``None`` when every roofline action is at its no-op default (the caller then skips modulation, so the
    live path is bit-for-bit unchanged). The roofline batch is the batching policy's concurrency band, so
    precision/spec are evaluated AT the chosen batch."""
    if is_neutral_roofline_bundle(bundle):
        return None
    batch = BATCH_SIZE_FOR_ROOFLINE.get(_batching(bundle), 16)
    wl = representative_workload(recs, prefix_hit_frac=prefix_hit_frac)
    return roofline_action_factors(bundle, wl, gpu=gpu, batch_size=batch, background_work=background_work)


__all__ = ["PRECISION_TO_ROOFLINE", "SPEC_DECODE_TO_ROOFLINE", "CLOCK_TO_ROOFLINE",
           "COLOCATION_TO_ROOFLINE", "PREFILL_DECODE_TO_ROOFLINE", "PRECISION_QUALITY_RISK",
           "NEUTRAL_POLICIES", "BATCH_SIZE_FOR_ROOFLINE", "action_serving_config",
           "representative_workload", "roofline_action_factors", "is_neutral_roofline_bundle",
           "period_action_modulation"]
