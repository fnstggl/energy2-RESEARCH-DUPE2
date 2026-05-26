"""KV-cache, prefix-affinity, and memory-pressure realism for the simulator.

Pure, deterministic functions (all randomness is caller-supplied → seedable)
that upgrade the simulator from a memory-pressure *proxy* to a believable model
of KV-cache growth, PagedAttention block slack, prefix-cache reuse, locality-
aware routing, cold-reroute penalties, and preemption/recompute under KV
exhaustion.

Every magnitude comes from ``calibration.KV_CACHE_PARAMS`` /
``calibration.MODEL_KV_PROFILES`` (inspectable provenance + confidence) and is
overridable via a per-run ``config`` dict. These are intentionally proxies, not
a serving-engine simulation — see the realism-gap report. Specifically:

- pressure thresholds (ELEVATED/THROTTLING/PREEMPTION) are OPERATIONAL
  HEURISTICS inferred from documented vLLM behaviour, NOT universal constants;
- the prefix hit-rate curve is a configurable sigmoid prior, NOT a fitted
  industry curve;
- the cold-reroute penalty prices lost prefill, which is a lower bound on the
  true cost (batch-packing churn, scheduler thrash are not fully modelled).

Do NOT read any value here as production-accurate. They make the *dynamics*
believable so that naive arbitrage can lose, cache-aware orchestration matters,
and locality preservation becomes economically meaningful.
"""

from __future__ import annotations

import math
from typing import Optional

from .calibration import ModelKVProfile, kv_value, resolve_kv_profile

__all__ = [
    "PressureRegion",
    "kv_bytes_per_token",
    "kv_bytes",
    "kv_pressure",
    "pressure_region",
    "block_slack_bytes",
    "fragmentation_frac",
    "sigmoid",
    "prefix_hit_rate",
    "prefill_savings_frac",
    "lost_prefill_tokens",
    "cold_route_penalty_ms",
    "locality_confidence_step",
    "preemption_probability",
    "recompute_penalty_ms",
    "pressure_ttft_multiplier",
    "pressure_batch_efficiency",
    "cache_aware_batch_efficiency",
    "telemetry_confidence_tier",
    "routing_aggressiveness",
    "should_preserve_affinity",
    "resolve_kv_profile",
    "ModelKVProfile",
]


# ---------------------------------------------------------------------------
# Operational pressure regions (heuristics inferred from vLLM behaviour)
# ---------------------------------------------------------------------------

class PressureRegion:
    LOW = "low"
    ELEVATED = "elevated"
    THROTTLING = "throttling_risk"
    PREEMPTION = "preemption"


# ---------------------------------------------------------------------------
# KV memory scaling law
# ---------------------------------------------------------------------------

def kv_bytes_per_token(profile: ModelKVProfile) -> float:
    """Bytes of KV cache per token for a model architecture.

    = layers · kv_heads · head_dim · 2 (K and V) · bytes_per_elem.
    GQA/MQA/reduced-KV are captured via ``kv_heads`` directly — NOT via
    hidden_size, which would over-count KV for grouped-query attention.
    """
    return profile.kv_bytes_per_token()


def kv_bytes(profile: ModelKVProfile, batch_size: float, seq_len: float) -> float:
    """Total KV-cache bytes for a batch of sequences at a given length.

    KV_bytes = batch_size · seq_len · kv_bytes_per_token(profile).
    """
    b = max(0.0, batch_size)
    s = max(0.0, seq_len)
    return b * s * kv_bytes_per_token(profile)


def kv_pressure(kv_allocated_bytes: float, reserved_kv_budget_bytes: float) -> float:
    """KV pressure = allocated / reserved budget, clamped to [0, 1.5].

    Allowed above 1.0 (would-be over-allocation) so the caller can see how far
    into the preemption region demand pushes; downstream effects clamp at 1.0.
    """
    if reserved_kv_budget_bytes <= 0:
        return 1.5 if kv_allocated_bytes > 0 else 0.0
    return max(0.0, min(1.5, kv_allocated_bytes / reserved_kv_budget_bytes))


def pressure_region(pressure: float, config: Optional[dict] = None) -> str:
    """Classify KV pressure into an operational region.

    Thresholds are operational heuristics (configurable), NOT universal.
    """
    elevated = kv_value("kv_pressure_elevated", config)
    throttling = kv_value("kv_pressure_throttling", config)
    preemption = kv_value("kv_pressure_preemption", config)
    if pressure >= preemption:
        return PressureRegion.PREEMPTION
    if pressure >= throttling:
        return PressureRegion.THROTTLING
    if pressure >= elevated:
        return PressureRegion.ELEVATED
    return PressureRegion.LOW


# ---------------------------------------------------------------------------
# PagedAttention fragmentation (internal slack, NOT heap fragmentation)
# ---------------------------------------------------------------------------

def block_slack_bytes(
    batch_size: float, avg_partial_block_tokens: float, kv_bytes_per_token_: float
) -> float:
    """Wasted KV bytes from partially-filled PagedAttention blocks.

    Each active sequence has one partially-filled tail block; the unused token
    slots in that block are internal slack. This is the *only* fragmentation in
    a paged allocator — there is no external/heap fragmentation.
    """
    return max(0.0, batch_size) * max(0.0, avg_partial_block_tokens) * max(0.0, kv_bytes_per_token_)


def fragmentation_frac(
    batch_size: float, seq_len: float, profile: ModelKVProfile, config: Optional[dict] = None
) -> float:
    """Fraction of allocated KV that is internal block slack, in [0, 1).

    avg wasted tokens per sequence ≈ (block_size - 1)/2 (uniform tail position).
    """
    block = kv_value("kv_block_size_tokens", config)
    avg_partial = max(0.0, (block - 1.0) / 2.0)
    bpt = kv_bytes_per_token(profile)
    slack = block_slack_bytes(batch_size, avg_partial, bpt)
    allocated = kv_bytes(profile, batch_size, seq_len) + slack
    if allocated <= 0:
        return 0.0
    return max(0.0, min(0.999, slack / allocated))


# ---------------------------------------------------------------------------
# Prefix-cache reuse
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def prefix_hit_rate(
    overlap: float, locality_factor: float, config: Optional[dict] = None
) -> float:
    """Prefix-cache hit rate = sigmoid(a·(overlap − b)) · locality_factor.

    overlap         shared-prefix overlap of the workload (workload-family /
                    tenant property), in [0, 1].
    locality_factor route-affinity quality (cache/locality confidence), in
                    [0, 1] — a cold route has low locality even at high overlap.

    Reuse is therefore gated by BOTH content overlap and routing locality:
    high-overlap traffic on a cold route still misses.
    """
    a = kv_value("prefix_hit_sigmoid_a", config)
    b = kv_value("prefix_hit_sigmoid_b", config)
    o = max(0.0, min(1.0, overlap))
    loc = max(0.0, min(1.0, locality_factor))
    return max(0.0, min(1.0, sigmoid(a * (o - b)) * loc))


def prefill_savings_frac(hit_rate: float, config: Optional[dict] = None) -> float:
    """Fraction of prefill TTFT removed by prefix reuse, in [0, max].

    Capped below 1.0: the uncached suffix + scheduling still cost even at a
    full hit. The benefit lands on prefill/TTFT, NOT decode throughput.
    """
    cap = kv_value("prefix_max_prefill_savings_frac", config)
    return max(0.0, min(cap, hit_rate * cap))


def lost_prefill_tokens(shared_prefix_tokens: float, hit_rate_before: float) -> float:
    """Reusable prefix tokens lost on a cold reroute.

    The destination cannot reuse the prefix it never cached, so the previously
    reused tokens must be re-prefilled.
    """
    return max(0.0, shared_prefix_tokens) * max(0.0, min(1.0, hit_rate_before))


def cold_route_penalty_ms(lost_tokens: float, config: Optional[dict] = None) -> float:
    """Cold-reroute TTFT penalty = lost_prefill_tokens · prefill_cost_per_token.

    A lower bound on the true cost: batch-packing churn and scheduler thrash
    after a reroute are not fully priced here (see realism-gap report).
    """
    cost = kv_value("prefill_cost_per_token_ms", config)
    return max(0.0, lost_tokens) * cost


# ---------------------------------------------------------------------------
# Locality / cache-confidence dynamics (reuse-driven warmup, not time-driven)
# ---------------------------------------------------------------------------

def locality_confidence_step(
    confidence: float, reused: bool, config: Optional[dict] = None
) -> float:
    """Advance locality/cache confidence one tick toward 1 (reuse) or 0 (idle).

    Logistic growth when affinity is sustained (repeated shared prefixes on the
    same route); decay when affinity is broken / cache idles (stale maps, LRU).
    Warmup is REUSE-driven, not purely elapsed-time-driven.
    """
    c = max(0.0, min(1.0, confidence))
    if reused:
        g = kv_value("locality_confidence_growth", config)
        # Logistic: dc = g·c·(1−c); seed a small floor so a cold (c≈0) route can
        # still begin warming when it starts being reused.
        c_seed = max(c, 0.05)
        return max(0.0, min(1.0, c_seed + g * c_seed * (1.0 - c_seed)))
    d = kv_value("locality_confidence_decay", config)
    return max(0.0, c * (1.0 - d))


# ---------------------------------------------------------------------------
# Preemption / recompute under KV exhaustion
# ---------------------------------------------------------------------------

def preemption_probability(pressure: float, config: Optional[dict] = None) -> float:
    """Per-tick probability of a preemption event, in [0, max].

    Zero below the throttling threshold; ramps quadratically to
    ``preemption_prob_max`` as pressure → 1.0. Above 1.0 demand it saturates.
    """
    throttling = kv_value("kv_pressure_throttling", config)
    pmax = kv_value("preemption_prob_max", config)
    p = max(0.0, pressure)
    if p <= throttling:
        return 0.0
    span = max(1e-6, 1.0 - throttling)
    frac = min(1.0, (p - throttling) / span)
    return max(0.0, min(pmax, pmax * frac * frac))


def recompute_penalty_ms(
    preempted_seqs: float, avg_context_tokens: float, config: Optional[dict] = None
) -> float:
    """Recompute cost (ms) of re-prefilling preempted sequences.

    Preemption-by-recompute ≈ re-prefilling the preempted context → not free.
    """
    per_tok = kv_value("recompute_ms_per_token", config)
    return max(0.0, preempted_seqs) * max(0.0, avg_context_tokens) * per_tok


# ---------------------------------------------------------------------------
# Pressure effects on TTFT / batching
# ---------------------------------------------------------------------------

def pressure_ttft_multiplier(pressure: float, config: Optional[dict] = None) -> float:
    """Multiplier on the active-sequence TTFT component as pressure rises.

    1.0 below ELEVATED; ramps convexly to ``kv_pressure_ttft_max_mult`` as
    pressure → 1.0 (allocation stalls under contention).
    """
    elevated = kv_value("kv_pressure_elevated", config)
    mmax = kv_value("kv_pressure_ttft_max_mult", config)
    p = max(0.0, min(1.0, pressure))
    if p <= elevated:
        return 1.0
    span = max(1e-6, 1.0 - elevated)
    frac = (p - elevated) / span
    return 1.0 + (mmax - 1.0) * frac * frac


def pressure_batch_efficiency(pressure: float, config: Optional[dict] = None) -> float:
    """Batching-efficiency multiplier in (0, 1] that DEGRADES under KV pressure.

    1.0 below ELEVATED; falls toward ``kv_pressure_batch_floor`` as pressure →
    1.0 (the scheduler runs thinner batches to fit KV).
    """
    elevated = kv_value("kv_pressure_elevated", config)
    floor = kv_value("kv_pressure_batch_floor", config)
    p = max(0.0, min(1.0, pressure))
    if p <= elevated:
        return 1.0
    span = max(1e-6, 1.0 - elevated)
    frac = (p - elevated) / span
    return max(floor, 1.0 - (1.0 - floor) * frac)


def cache_aware_batch_efficiency(
    base_efficiency: float, hit_rate: float, pressure: float, config: Optional[dict] = None
) -> float:
    """Combine batching-knee efficiency with prefix-reuse boost and KV pressure.

    - high prefix hit rate packs batches better (shared prefix blocks) → a small
      multiplicative boost;
    - KV pressure thins batches → degradation.
    Result clamped to (0, 1].
    """
    boost = 1.0 + 0.15 * max(0.0, min(1.0, hit_rate))  # up to +15% from shared prefixes
    eff = base_efficiency * boost * pressure_batch_efficiency(pressure, config)
    return max(0.01, min(1.0, eff))


# ---------------------------------------------------------------------------
# Telemetry confidence tiers → routing aggressiveness
# ---------------------------------------------------------------------------

def telemetry_confidence_tier(
    has_kv_usage: bool,
    has_preemptions: bool,
    has_prefix_hit_rate: bool,
    has_latency: bool,
) -> str:
    """Map available telemetry to a confidence tier.

    HIGH    KV usage + preemptions + prefix hit rate all visible.
    MEDIUM  latency/throughput visible but KV/cache internals missing.
    LOW     only request-level SLOs.
    Missing KV telemetry LOWERS confidence — it must NOT be read as 'no pressure'.
    """
    if has_kv_usage and has_preemptions and has_prefix_hit_rate:
        return "high"
    if has_latency:
        return "medium"
    return "low"


def routing_aggressiveness(tier: str, config: Optional[dict] = None) -> float:
    """Routing aggressiveness multiplier in (0, 1] by telemetry tier.

    Aggressiveness decreases as confidence drops: with poor cache visibility we
    must assume reroutes are riskier than they look.
    """
    if tier == "high":
        return 1.0
    damp = kv_value("telemetry_missing_routing_damp", config)
    if tier == "medium":
        return max(0.0, min(1.0, 0.5 + 0.5 * damp))
    return max(0.0, min(1.0, damp))


# ---------------------------------------------------------------------------
# Cache-aware migration policy
# ---------------------------------------------------------------------------

def should_preserve_affinity(
    overlap: float,
    shared_prefix_tokens: float,
    expected_queue_gain_ms: float,
    locality_confidence: float,
    severe_imbalance: bool = False,
    config: Optional[dict] = None,
) -> bool:
    """Decide whether to PRESERVE affinity rather than reroute/migrate.

    Block the reroute when expected cache loss > expected queue gain:

        expected_cache_loss_ms = cold_route_penalty(lost_prefill_tokens)
        lost_prefill_tokens    = shared_prefix_tokens · hit_rate(overlap, conf)

    Preserve when overlap is high, prefixes are long, queue differences are
    modest, and cache confidence is high. Break affinity only under severe load
    imbalance / overload, or when locality confidence is already low (little to
    lose). Deterministic; no randomness.
    """
    if severe_imbalance:
        return False
    # Low locality confidence → the cache is not warm here, so there is little
    # reuse to protect; don't pay to preserve a route that isn't warm.
    if locality_confidence < 0.15:
        return False
    hit = prefix_hit_rate(overlap, locality_confidence, config)
    lost = lost_prefill_tokens(shared_prefix_tokens, hit)
    expected_cache_loss_ms = cold_route_penalty_ms(lost, config)
    return expected_cache_loss_ms > max(0.0, expected_queue_gain_ms)
