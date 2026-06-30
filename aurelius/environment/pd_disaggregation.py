"""Prefill/decode disaggregation — conservative phase-pool queueing model (Batch-1 Phase 4).

The ``prefill_decode_policy`` action already reaches reward through the roofline (``serving_point``
``disaggregated_static`` branch: a split inflates one phase's per-request work + a fixed KV handoff). That
captures the *service-time* effect but not the **phase-pool queueing**: when prefill and decode run in
SEPARATE capacity pools, a wrong split saturates one pool (its queue blows up) while the other sits idle.
This module adds that missing structure as a **conservative causal approximation** (the live cluster replay
has no persistent disaggregated pools), so the disaggregation knob can be evaluated honestly:

  * two pools (prefill ``c_p``, decode ``c_d``) sized by the split; each an M/M/c-style queue;
  * a wrong split → one pool's utilization → 1 → its queue wait explodes (disaggregation HURTS);
  * KV handoff: every request ships its context KV bytes prefill→decode pool (bytes + latency) — the cost
    that makes disaggregation NOT free and can ERASE the gain on small/balanced workloads;
  * idle GPU-seconds by pool (the capacity a wrong split strands);
  * allocation efficiency = how balanced the two pools' utilizations are.

``shared`` (the no-op) is a single pool of all replicas with NO handoff — the baseline disaggregation must
beat. Everything is deterministic and analytical (no event sim).

Fidelity: the queueing law (M/M/c wait) is PUBLIC_THEORY; the KV-handoff bytes are BENCHMARK_DERIVED (model
architecture × context); the interconnect bandwidth + the per-phase work are SIMULATOR_INFERENCE bands. There
are NO persistent disaggregated-pool traces → the whole knob is labelled SIMULATOR_INFERENCE / directional
until pilot telemetry. Nothing here adds a reward bonus; effects flow through completion time + GPU-seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

# split policy → prefill share of the replica pool. "shared" = single undivided pool (the no-op baseline).
PD_POLICY_TO_PREFILL_SHARE = {
    "shared": None,            # no disaggregation (one pool, no handoff)
    "p40_d60": 0.4,
    "p60_d40": 0.6,
    "prefill_heavy": 0.7,
    "decode_heavy": 0.3,
    "balanced_pd": 0.5,
}
PD_POLICY_OPTIONS = ("shared", "disaggregated_static", "prefill_heavy", "decode_heavy", "balanced_pd")

# KV handoff interconnect bandwidth (B/s) prefill→decode pool. DistServe §"Placement": intra-node NVLink
# (~hundreds of GB/s) makes the KV handoff negligible; cross-node IB/RoCE (tens of GB/s) makes BANDWIDTH-AWARE
# placement essential — below a threshold the handoff dominates and disaggregation HURTS. Default ≈ NVLink.
HANDOFF_BANDWIDTH_BYTES_PER_S = 100e9
# A handoff that consumes more than this fraction of the per-request decode budget makes disaggregation a net
# loss (the KV transfer stalls the decode worker). DistServe's bandwidth/placement condition, as a guard.
HANDOFF_BUDGET_FRACTION_HURT = 0.5
# Shared-pool prefill/decode INTERFERENCE (the loss DistServe/Splitwise/Sarathi disaggregate to remove):
# mixing prefill and decode on the same replicas costs efficiency in proportion to how BUSY the pool is and
# how SKEWED the phase mix is (a near-balanced mix multiplexes cleanly; a dominant phase starves the other /
# spikes inter-token latency). Disaggregated pools are phase-isolated → no interference, but pay the KV
# handoff + the statistical-multiplexing penalty of two smaller pools. SIMULATOR_INFERENCE band.
SHARED_INTERFERENCE_K = 0.9
# DistServe's dominant colocation cost: in a shared pool a DECODE token waits behind in-flight PREFILL chunks
# (the inter-token-latency / TPOT spike). Chunked prefill (Sarathi) mitigates but does NOT remove it → a
# residual blocking. Scales with how much of the shared pool is doing prefill (its prefill utilization).
# Disaggregated pools are phase-isolated → no TPOT blocking (but pay handoff + multiplexing). SIMULATOR_INFERENCE.
PREFILL_DECODE_BLOCK_K = 1.5          # max decode slowdown from prefill blocking ≈ 1 + K at full prefill load
CHUNKED_PREFILL_RESIDUAL = 0.5        # share of the blocking that survives chunked-prefill mitigation
_GIB = 1024 ** 3


@dataclass
class PDWorkload:
    """One period's phase workload. Times are per-request SERVICE seconds for each phase."""
    arrival_rate: float            # requests / second
    prefill_work_s: float          # per-request prefill service time
    decode_work_s: float           # per-request decode service time
    context_tokens: int = 832      # KV context handed off prefill→decode
    kv_bytes_per_token: float = 131072.0   # llama-8b-gqa fp16 (BENCHMARK_DERIVED)
    decode_tokens: int = 128       # output tokens (for TPOT/decode-budget SLO attainment)
    kv_bandwidth_bytes_per_s: float = HANDOFF_BANDWIDTH_BYTES_PER_S  # interconnect for the prefill→decode handoff


def _mmc_wait(offered: float, c: int, service_s: float) -> float:
    """Conservative M/M/c-style mean queue wait. ``offered`` = arrival·service (Erlangs); ``c`` servers.
    ρ = offered/c. Below saturation the wait ~ service·ρ^c/(c(1−ρ)) (Erlang-C upper-bounded); at/above
    saturation (ρ≥1) the queue is unstable → a large but finite penalty proportional to the overload."""
    c = max(1, int(c))
    rho = offered / c
    if rho >= 0.999:
        return service_s * (1.0 + 50.0 * (rho - 0.9))      # unstable: steep, finite overload penalty
    # Erlang-C-ish: probability of wait ≈ rho^c, mean wait ≈ that · service/(c(1−rho)).
    pw = rho ** c
    return service_s * pw / (c * (1.0 - rho))


@dataclass
class PDResult:
    selected_pd_policy: str
    prefill_pool: int
    decode_pool: int
    prefill_queue_wait: float
    decode_queue_wait: float
    prefill_pool_utilization: float
    decode_pool_utilization: float
    kv_handoff_bytes: float
    kv_handoff_latency: float
    idle_gpu_seconds_prefill: float
    idle_gpu_seconds_decode: float
    idle_gpu_seconds_total: float
    allocation_efficiency: float
    mean_completion_s: float       # TTFT(+queue+handoff) + decode(+queue)
    ttft_s: float
    decode_latency_s: float

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def pd_serving_point(wl: PDWorkload, policy: str, *, n_replicas: int, period_seconds: float = 60.0) -> PDResult:
    """Phase-pool serving point for ``policy`` on ``wl`` over ``n_replicas``. ``shared`` is one pool, no
    handoff (the baseline). A split sizes two pools; each queues independently; a KV handoff is paid per
    request on the disaggregated path. Deterministic."""
    share = PD_POLICY_TO_PREFILL_SHARE.get(policy)
    lam = max(0.0, wl.arrival_rate)
    offered_p = lam * wl.prefill_work_s
    offered_d = lam * wl.decode_work_s

    if share is None:                                       # shared: one undivided pool, no handoff
        c = max(1, n_replicas)
        # a shared pool serves both phases on the same replicas → combined offered load, one queue.
        offered = offered_p + offered_d
        wait = _mmc_wait(offered, c, wl.prefill_work_s + wl.decode_work_s)
        util = offered / c
        idle = max(0.0, c - offered) * period_seconds
        ttft = wl.prefill_work_s + wait * (wl.prefill_work_s / max(wl.prefill_work_s + wl.decode_work_s, 1e-9))
        dec = wl.decode_work_s + wait * (wl.decode_work_s / max(wl.prefill_work_s + wl.decode_work_s, 1e-9))
        # prefill/decode interference: phase mixing costs efficiency ∝ pool busyness × phase SKEW (distance
        # of the prefill share from a balanced 0.5). Balanced mixes multiplex cleanly (skew≈0 → no penalty);
        # a dominant phase spikes the other's latency. Disaggregated (isolated) pools avoid it. Applied to the
        # whole shared completion (both TTFT and decode suffer under a skewed, busy shared pool).
        prefill_share = offered_p / max(offered_p + offered_d, 1e-9)
        skew = abs(prefill_share - 0.5) * 2.0
        contention = max(0.0, min(1.0, util) - 0.25)        # HoL blocking is a high-load phenomenon
        interference = 1.0 + SHARED_INTERFERENCE_K * contention * skew
        ttft *= interference
        # DistServe TPOT blocking: each decode token waits behind in-flight prefill chunks. Scales with the
        # shared pool's prefill utilization (how often a prefill is co-scheduled) and only bites under load
        # (the contention floor). Chunked prefill leaves a residual. This is the dominant colocation TPOT cost.
        prefill_intensity = min(1.0, offered_p / c) * (contention / max(util, 1e-9) if util > 0 else 0.0)
        tpot_block = 1.0 + PREFILL_DECODE_BLOCK_K * CHUNKED_PREFILL_RESIDUAL * prefill_intensity
        dec *= interference * tpot_block
        return PDResult(
            selected_pd_policy="shared", prefill_pool=c, decode_pool=c,
            prefill_queue_wait=round(wait, 6), decode_queue_wait=round(wait, 6),
            prefill_pool_utilization=round(util, 4), decode_pool_utilization=round(util, 4),
            kv_handoff_bytes=0.0, kv_handoff_latency=0.0,
            idle_gpu_seconds_prefill=round(idle, 3), idle_gpu_seconds_decode=round(idle, 3),
            idle_gpu_seconds_total=round(idle, 3), allocation_efficiency=1.0,
            mean_completion_s=round(ttft + dec, 6), ttft_s=round(ttft, 6), decode_latency_s=round(dec, 6))

    c_p = max(1, round(n_replicas * share))
    c_d = max(1, n_replicas - c_p)
    wait_p = _mmc_wait(offered_p, c_p, wl.prefill_work_s)
    wait_d = _mmc_wait(offered_d, c_d, wl.decode_work_s)
    util_p = offered_p / c_p
    util_d = offered_d / c_d
    # KV handoff: ship the full context KV from the prefill pool to the decode pool (DistServe/Splitwise),
    # over the (possibly cross-node) interconnect. Low bandwidth → the transfer stalls the decode worker.
    handoff_bytes = wl.kv_bytes_per_token * max(1, wl.context_tokens)
    handoff_latency = handoff_bytes / max(1.0, wl.kv_bandwidth_bytes_per_s)
    # DistServe bandwidth/placement guard: if the handoff eats more than HANDOFF_BUDGET_FRACTION_HURT of the
    # decode budget, the transfer dominates and disaggregation is a net loss (extra stall on decode start).
    decode_budget = max(wl.decode_work_s, 1e-9)
    if handoff_latency > HANDOFF_BUDGET_FRACTION_HURT * decode_budget:
        wait_d += (handoff_latency - HANDOFF_BUDGET_FRACTION_HURT * decode_budget)  # excess stalls decode
    idle_p = max(0.0, c_p - offered_p) * period_seconds
    idle_d = max(0.0, c_d - offered_d) * period_seconds
    # balanced pools (equal utilization) waste no capacity; divergence strandes one pool.
    alloc_eff = 1.0 - abs(util_p - util_d) / max(util_p, util_d, 1e-9)
    ttft = wl.prefill_work_s + wait_p + handoff_latency
    dec = wl.decode_work_s + wait_d
    return PDResult(
        selected_pd_policy=policy, prefill_pool=c_p, decode_pool=c_d,
        prefill_queue_wait=round(wait_p, 6), decode_queue_wait=round(wait_d, 6),
        prefill_pool_utilization=round(util_p, 4), decode_pool_utilization=round(util_d, 4),
        kv_handoff_bytes=round(handoff_bytes, 1), kv_handoff_latency=round(handoff_latency, 6),
        idle_gpu_seconds_prefill=round(idle_p, 3), idle_gpu_seconds_decode=round(idle_d, 3),
        idle_gpu_seconds_total=round(idle_p + idle_d, 3), allocation_efficiency=round(max(0.0, alloc_eff), 4),
        mean_completion_s=round(ttft + dec, 6), ttft_s=round(ttft, 6), decode_latency_s=round(dec, 6))


def _slo_attainment(mean_latency_s: float, slo_s: float) -> float:
    """P(latency ≤ slo) under an exponential-sojourn assumption (M/M/1-class): 1 − exp(−slo/mean). Monotone
    in the slack slo/mean — a busier pool (higher mean) attains less; an idle pool attains ~1. DistServe
    scores SLO *attainment*; this is the smooth per-phase attainment from the phase-pool model's mean."""
    import math
    if slo_s <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - math.exp(-slo_s / max(mean_latency_s, 1e-9))))


def pd_slo_goodput(wl: PDWorkload, policy: str, *, n_replicas: int, ttft_slo_s: float,
                   tpot_slo_s: float, period_seconds: float = 60.0) -> dict:
    """SLO-attainment goodput for a PD policy (the DistServe metric). A request is good only if it meets BOTH
    the TTFT SLO (first token) AND the TPOT SLO (per output token → a decode-budget = tpot·decode_tokens).
    goodput = attainment · throughput. Captures DistServe's mechanism: colocation interference lowers both
    means → lower attainment; a correct disaggregated split removes interference → higher attainment; a wrong
    split or insufficient KV bandwidth stalls a pool → its mean explodes → attainment → 0."""
    r = pd_serving_point(wl, policy, n_replicas=n_replicas, period_seconds=period_seconds)
    decode_budget_s = max(1e-9, tpot_slo_s * max(1, wl.decode_tokens))
    attain_ttft = _slo_attainment(r.ttft_s, ttft_slo_s)
    attain_tpot = _slo_attainment(r.decode_latency_s, decode_budget_s)
    attainment = attain_ttft * attain_tpot
    throughput = wl.arrival_rate * max(1, wl.decode_tokens)
    return {"policy": policy, "attainment": round(attainment, 5),
            "attain_ttft": round(attain_ttft, 5), "attain_tpot": round(attain_tpot, 5),
            "slo_safe_goodput": round(attainment * throughput, 3),
            "ttft_s": r.ttft_s, "decode_latency_s": r.decode_latency_s,
            "kv_handoff_latency": r.kv_handoff_latency, "prefill_pool_util": r.prefill_pool_utilization,
            "decode_pool_util": r.decode_pool_utilization, "allocation_efficiency": r.allocation_efficiency}


def distserve_goodput_comparison(wl: PDWorkload, *, n_replicas: int, ttft_slo_s: float, tpot_slo_s: float,
                                 splits=("p40_d60", "p60_d40", "balanced_pd", "prefill_heavy", "decode_heavy"),
                                 period_seconds: float = 60.0) -> dict:
    """DistServe-shaped sanity check: SLO-attainment goodput of the shared (colocated continuous-batching)
    pool vs the BEST disaggregated split, and their ratio. Used to ask 'can our PD model reproduce a
    DistServe-like win?' — NOT a reward path. Returns the ratio and a `distserve_like` flag (≥1.5× goodput)."""
    shared = pd_slo_goodput(wl, "shared", n_replicas=n_replicas, ttft_slo_s=ttft_slo_s,
                            tpot_slo_s=tpot_slo_s, period_seconds=period_seconds)
    cand = {p: pd_slo_goodput(wl, p, n_replicas=n_replicas, ttft_slo_s=ttft_slo_s, tpot_slo_s=tpot_slo_s,
                              period_seconds=period_seconds) for p in splits}
    best = max(cand, key=lambda p: cand[p]["slo_safe_goodput"])
    ratio = cand[best]["slo_safe_goodput"] / max(shared["slo_safe_goodput"], 1e-9)
    return {"shared": shared, "best_split": best, "best": cand[best], "all_splits": cand,
            "goodput_ratio_disagg_over_shared": round(ratio, 3),
            "distserve_like": ratio >= 1.5}


def compare_pd_policies(wl: PDWorkload, *, n_replicas: int, policies=PD_POLICY_OPTIONS,
                        period_seconds: float = 60.0) -> dict:
    """``{policy: PDResult.to_dict()}`` for each policy + the completion-time winner. The shared baseline is
    always included so a disaggregated win must beat the no-handoff single pool (no free disaggregation)."""
    pols = list(dict.fromkeys(("shared", *policies)))
    out = {p: pd_serving_point(wl, p, n_replicas=n_replicas, period_seconds=period_seconds).to_dict()
           for p in pols}
    best = min(out, key=lambda p: out[p]["mean_completion_s"])
    return {"results": out, "best_policy_by_completion": best,
            "shared_completion_s": out["shared"]["mean_completion_s"]}


__all__ = ["PD_POLICY_TO_PREFILL_SHARE", "PD_POLICY_OPTIONS", "HANDOFF_BANDWIDTH_BYTES_PER_S",
           "HANDOFF_BUDGET_FRACTION_HURT", "PDWorkload", "PDResult", "pd_serving_point",
           "compare_pd_policies", "pd_slo_goodput", "distserve_goodput_comparison"]
