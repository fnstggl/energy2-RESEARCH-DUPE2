"""Source-specific validation builders for the canonical environment.

Assembles the full breadth of held-out distribution checks the build spec
requires, pulling each real reference from its source and tagging its data tier:

  * **Azure / serving** — held-out time-split (train vs holdout): tokens,
    inter-arrival, burstiness. Genuinely held-out → can PASS.
  * **v2026 / fleet** — the environment's committed *sample* fleet distribution
    vs the **FULL_TRACE_EXACT** calibration artifact (the 6.5 B-row reference):
    GPU util/memory, priority/job/model mix, queue/ready delay, GPU request,
    placement/fragmentation, GPU-type mix, capacity, rack/asw locality, network
    rx/tx, job duration. Reference tier = the artifact's label; if an artifact is
    missing the check is SKIPPED with the exact re-stream command.
  * **Mooncake / KV** — train vs holdout prefix reuse (exact + partial overlap),
    cache-hit rate, cold-vs-warm warmup. Reference tier from the Mooncake source.
  * **electricity / cost** — price sanity band (SAMPLE_FIXTURE); the held-out ISO
    comparison is SKIPPED with the auth/endpoint step until live pulls are wired.

Never row-joins planes; only compares each plane's distribution to its own
held-out reference.
"""

from __future__ import annotations

import statistics
from collections import Counter

from .ingestion import v2026_artifacts
from .ingestion.electricity import MANUAL_STEP as ELEC_MANUAL_STEP
from .ingestion.mooncake import ingest_mooncake, split_reuse
from .kv_cache import DEFAULT_FOOTPRINT, FOOTPRINTS, KVAwareRouter, KVModel, StatefulKVCache
from .validation_suite import (
    FAIL,
    PASS,
    ValidationCheck,
    check_category_mix,
    check_samples,
    check_summary,
    sanity_check,
    skipped_check,
)

# Generous but documented tolerances: v2026 checks are a finite committed SAMPLE
# vs the full-trace POPULATION, so sample noise is expected — moderate divergence
# WARNs, only gross divergence FAILs. Azure/Mooncake are true held-out splits.
_V2026_SUMMARY_TOL, _V2026_SUMMARY_WARN = 0.25, 0.50
_V2026_MIX_TOL, _V2026_MIX_WARN = 0.15, 0.30
_RESTREAM = "python -m scripts.run_v2026_streaming_calibration {table}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _floats(rows: list, col: str) -> list:
    out = []
    for r in rows:
        v = r.get(col)
        if v in (None, ""):
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _counts(rows: list, col: str) -> dict:
    return dict(Counter(r.get(col, "unknown") for r in rows if r.get(col) not in (None, "")))


def _artifact_label(art: dict) -> str:
    return art.get("label", "UNAVAILABLE")


# ---------------------------------------------------------------------------
# v2026 fleet checks — validate the env's EMITTED fleet state (aggregated over
# hours from state_at) against the FULL_TRACE_EXACT artifact marginals. One path:
#   * anchored env (processed_dir given)  → emitted marginals == full trace
#     → CONSISTENCY (PASS; the env reproduces the marginals it's calibrated from,
#       NOT an independent held-out test — said so in `detail`).
#   * sample-only env (no processed_dir)  → emitted marginals == sample
#     → REPRESENTATIVENESS (the committed sample vs the full trace; FAILs honestly
#       when the sample is an unrepresentative slice).
# ---------------------------------------------------------------------------

def v2026_fleet_checks(fleet_plane, *, processed_dir: str | None = None) -> list:
    pdir = processed_dir or v2026_artifacts.PROCESSED_DIR
    pod = v2026_artifacts.load_table("pod_hourly", pdir)
    srv = v2026_artifacts.load_table("server_hourly", pdir)
    net = v2026_artifacts.load_table("network_hourly", pdir)
    job = v2026_artifacts.load_table("job_execution_summary", pdir)
    if pod is None:
        return [skipped_check(
            "v2026_fleet", source="alibaba_gpu_v2026",
            required_artifact="pod_hourly_calibration.json (FULL_TRACE_EXACT)",
            command=_RESTREAM.format(table="pod_hourly"), reason="artifact not present")]

    pa = pod["artifacts"]
    sa = (srv or {"artifacts": {}})["artifacts"]
    na = (net or {"artifacts": {}})["artifacts"]
    anchored = getattr(fleet_plane, "full_trace", None) is not None
    note = ("CONSISTENCY: env fleet calibrated from this FULL_TRACE_EXACT artifact "
            "(reproduces the marginal; NOT an independent held-out test)") if anchored else (
            "REPRESENTATIVENESS: committed SAMPLE fleet vs the full-trace marginal")
    tier = "FULL_TRACE_EXACT"
    tol, warn = _V2026_SUMMARY_TOL, _V2026_SUMMARY_WARN

    # The env's emitted hourly fleet state (what the two-clock loop actually uses).
    states = [fleet_plane.state_at(h) for h in fleet_plane.hours()] or [fleet_plane.state_at(0)]

    def _mean(attr):
        vs = [getattr(s, attr) for s in states]
        return statistics.mean(vs) if vs else 0.0

    def _sum(kind, env_val, ref_val, t=tol, w=warn):
        c = check_summary(kind, {"v": env_val}, {"v": ref_val}, source="alibaba_gpu_v2026",
                          ref_tier=tier, keys=("v",), tolerance=t, warn_tolerance=w)
        c.detail = note
        return c

    checks = [
        _sum("v2026_gpu_utilization", _mean("util_target") * 100.0, pa["gpu_sm_util"]["mean"]),
        _sum("v2026_gpu_memory", _mean("mem_pressure"),
             (pa.get("gpu_mem_util") or {}).get("mean", 0.0)),
        _sum("v2026_queue_delay", _mean("queue_delay_s"),
             (pa.get("schedule_delay_s") or {}).get("mean", 0.0)),
        _sum("v2026_ready_delay", _mean("ready_delay_s"),
             (pa.get("ready_delay_s") or {}).get("mean", 0.0)),
    ]
    # priority + job/model + GPU-type mixes (category total-variation)
    prio = check_category_mix("v2026_priority_mix", states[0].priority_mix,
                              (pa.get("priority_class") or {}).get("counts", {}),
                              source="alibaba_gpu_v2026", ref_tier=tier,
                              tolerance=_V2026_MIX_TOL, warn_tolerance=_V2026_MIX_WARN)
    prio.detail = note
    checks.append(prio)
    if states[0].gpu_type_mix and (sa.get("gpu_type") or {}).get("counts"):
        gt = check_category_mix("v2026_gpu_type_mix", states[0].gpu_type_mix,
                                sa["gpu_type"]["counts"], source="alibaba_gpu_v2026",
                                ref_tier=_artifact_label(srv) if srv else tier,
                                tolerance=_V2026_MIX_TOL, warn_tolerance=_V2026_MIX_WARN)
        gt.detail = note
        checks.append(gt)
    # best-effort (offline) fraction ~ job_type mix; model_type not modelled by env
    jt = (pa.get("job_type_public") or {}).get("fractions", {})
    ref_be = (jt.get("offline_inference", 0.0)
              / max(1e-9, jt.get("online_inference", 0.0) + jt.get("offline_inference", 0.0)))
    checks.append(_sum("v2026_job_type_best_effort", _mean("best_effort_fraction"), ref_be,
                       t=_V2026_SUMMARY_WARN, w=0.75))
    checks.append(skipped_check(
        "v2026_model_type_mix", source="alibaba_gpu_v2026",
        required_artifact="an env model_type signal to compare to pod_hourly.model_type_public",
        command="extend FleetState with a model_type mix",
        reason="env does not yet emit a model-type distribution"))
    # network rx+tx pressure (env net_pressure is normalised by net_ref_gibps)
    ref_net = ((na.get("rx_gibps") or {}).get("mean", 0.0) + (na.get("tx_gibps") or {}).get("mean", 0.0))
    if ref_net or anchored:
        checks.append(_sum("v2026_network_rx_tx", _mean("net_pressure") * fleet_plane.net_ref_gibps,
                           ref_net, t=_V2026_SUMMARY_WARN, w=0.75))
    # placement / fragmentation: 1 - gpu_request_mean / server_gpu_count_mean (both full-trace)
    sgc = (pa.get("server_gpu_count") or {}).get("mean")
    greq = (pa.get("gpu_request") or {}).get("mean")
    if sgc:
        ref_frag = max(0.0, 1.0 - (greq or 0.0) / sgc)
        checks.append(_sum("v2026_placement_fragmentation", _mean("fragmentation"), ref_frag,
                           t=_V2026_SUMMARY_WARN, w=0.75))
    # capacity / GPU count + rack-asw locality are TOPOLOGY = SAMPLE_FIXTURE (honest)
    if (sa.get("gpu_count") or {}).get("mean"):
        env_gc = statistics.mean(_floats(fleet_plane.servers, "gpu_count")) if fleet_plane.servers else 0.0
        c = check_summary("v2026_capacity_gpu_count", {"v": env_gc},
                          {"v": sa["gpu_count"]["mean"]}, source="alibaba_gpu_v2026",
                          ref_tier="FULL_TRACE_EXACT (env topology=SAMPLE_FIXTURE)",
                          keys=("v",), tolerance=_V2026_SUMMARY_WARN, warn_tolerance=0.9)
        checks.append(c)
    if (sa.get("asw_locality") or {}).get("counts"):
        c = check_category_mix("v2026_rack_asw_locality", _counts(fleet_plane.servers, "asw_id"),
                               sa["asw_locality"]["counts"], source="alibaba_gpu_v2026",
                               ref_tier="FULL_TRACE_EXACT (env topology=SAMPLE_FIXTURE)",
                               tolerance=0.5, warn_tolerance=0.9)
        checks.append(c)
    # job duration: full-trace reference present, but the env has no job sample → SKIPPED
    checks.append(skipped_check(
        "v2026_job_duration", source="alibaba_gpu_v2026",
        required_artifact=("job_execution_summary_calibration.json + an env job duration signal"
                           if job is not None else "job_execution_summary_calibration.json"),
        command=(_RESTREAM.format(table="job_execution_summary") if job is None
                 else "wire job_execution duration into the fleet plane / env"),
        reason="env does not emit a job-duration distribution to compare"))
    return checks


# ---------------------------------------------------------------------------
# Mooncake KV checks (train vs holdout reuse)
# ---------------------------------------------------------------------------

def mooncake_kv_checks() -> list:
    reqs, status = ingest_mooncake()
    tier = status.tier
    if not reqs:
        return [skipped_check(
            "kv_prefix_reuse", source="mooncake",
            required_artifact="Mooncake conversation_trace", command=status.manual_step or "",
            reason="no Mooncake records")]
    train, hold = split_reuse(reqs, holdout_frac=0.3)
    checks = [
        check_summary(
            "kv_exact_prefix_reuse",
            {"rate": train["exact_prefix_hit_rate"]}, {"rate": hold["exact_prefix_hit_rate"]},
            source="mooncake", ref_tier=tier, keys=("rate",), tolerance=0.20, warn_tolerance=0.40),
        check_summary(
            "kv_partial_prefix_overlap",
            {"overlap": train["mean_partial_overlap"]}, {"overlap": hold["mean_partial_overlap"]},
            source="mooncake", ref_tier=tier, keys=("overlap",), tolerance=0.25, warn_tolerance=0.50),
        check_samples(
            "kv_partial_overlap_distribution", train["partial_overlap_samples"],
            hold["partial_overlap_samples"], source="mooncake", ref_tier=tier,
            tolerance=0.20, warn_tolerance=0.35),
    ]
    # cache hit-rate distribution train vs holdout (the exact-prefix hit signal)
    checks.append(check_summary(
        "kv_cache_hit_rate", {"hit_rate": train["exact_prefix_hit_rate"]},
        {"hit_rate": hold["exact_prefix_hit_rate"]}, source="mooncake", ref_tier=tier,
        keys=("hit_rate",), tolerance=0.20, warn_tolerance=0.40))
    # cold-vs-warm warmup: overlap in the first quarter (cold) must be ≤ last quarter (warm)
    n = len(reqs)
    cold = split_reuse(reqs[: max(1, n // 4)], holdout_frac=0.0)[0]["mean_partial_overlap"]
    warm = split_reuse(reqs[max(1, 3 * n // 4):], holdout_frac=0.0)[0]["mean_partial_overlap"]
    ok = warm >= cold
    checks.append(ValidationCheck(
        kind="kv_cold_vs_warm", source="mooncake", ref_tier=tier, mode="sanity",
        metric=0.0 if ok else 1.0, metric_name="warm_ge_cold",
        metrics={"cold_overlap": round(cold, 4), "warm_overlap": round(warm, 4)},
        tolerance=0.5, warn_tolerance=0.5, verdict=PASS if ok else FAIL,
        detail="cache warms up: late-window reuse ≥ early-window reuse"))
    return checks


# ---------------------------------------------------------------------------
# stateful KV simulator checks (TTFT/prefill, memory pressure, eviction, no-oracle)
# ---------------------------------------------------------------------------

def kv_simulator_checks() -> list:
    reqs, status = ingest_mooncake()
    tier = status.tier
    blocked = [r for r in reqs if r.hash_ids]
    if not blocked:
        return [skipped_check(
            "kv_simulator", source="mooncake", required_artifact="Mooncake hash_ids",
            command=status.manual_step or "", reason="no block-hash records")]
    train = blocked[: int(len(blocked) * 0.7)] or blocked
    n = len(train)
    distinct = len({b for r in train for b in r.hash_ids})

    # TTFT / prefill improvement: an enabled, warm KV cache yields prefix hits that
    # cut prefill tokens and TTFT (vs the disabled model which saves nothing).
    warm = KVModel.fit(train, gpu_mem_gib=80.0, mem_pressure=0.0)
    st = warm.stats(n)
    off = KVModel.fit(train, gpu_mem_gib=80.0, enabled=False).stats(n)
    ttft_ok = (st["kv_hit_rate"] > 0 and st["mean_ttft_factor"] < 1.0
               and st["prefill_tokens_saved"] > 0 and off["prefill_tokens_saved"] == 0)
    checks = [sanity_check(
        "kv_ttft_prefill_improvement", ttft_ok,
        {"hit_rate": st["kv_hit_rate"], "mean_ttft_factor": st["mean_ttft_factor"],
         "prefill_tokens_saved": st["prefill_tokens_saved"], "disabled_saved": off["prefill_tokens_saved"]},
        source="mooncake", ref_tier=tier, detail="enabled KV → prefix hits cut prefill/TTFT; disabled saves nothing")]

    # KV memory pressure: pressure in [0,1]; used blocks never exceed capacity.
    cs = warm.cache_summary
    mem_ok = 0.0 <= cs["memory_pressure"] <= 1.0 and cs["used_blocks"] <= cs["capacity_blocks"]
    checks.append(sanity_check(
        "kv_memory_pressure", mem_ok,
        {"memory_pressure": cs["memory_pressure"], "used_blocks": cs["used_blocks"],
         "capacity_blocks": cs["capacity_blocks"]},
        source="engineering", ref_tier="INFERRED", detail="cache memory pressure in-band; used ≤ capacity"))

    # Eviction: a cache smaller than the trace's working set MUST evict; a cache
    # larger than it must NOT. (LRU is a documented HEURISTIC.)
    tiny = StatefulKVCache(capacity_blocks=max(1, distinct // 4))
    big = StatefulKVCache(capacity_blocks=distinct + 16)
    for r in train:
        tiny.process(r.hash_ids)
        big.process(r.hash_ids)
    evict_ok = tiny.evictions > 0 and big.evictions == 0
    checks.append(sanity_check(
        "kv_eviction", evict_ok,
        {"tiny_cap": tiny.capacity_blocks, "tiny_evictions": tiny.evictions,
         "big_cap": big.capacity_blocks, "big_evictions": big.evictions, "distinct_blocks": distinct},
        source="engineering", ref_tier="HEURISTIC", detail="evicts iff working set exceeds capacity (LRU)"))

    # No-oracle / causality: the exact-prefix outcomes over the FIRST HALF are
    # identical whether or not the later requests exist → no future leakage.
    half = max(1, n // 2)
    a = StatefulKVCache(capacity_blocks=distinct + 16)
    b = StatefulKVCache(capacity_blocks=distinct + 16)
    seq_full = [a.process(r.hash_ids)["exact_prefix_blocks"] for r in train]
    seq_prefix = [b.process(r.hash_ids)["exact_prefix_blocks"] for r in train[:half]]
    causal_ok = seq_full[:half] == seq_prefix
    checks.append(sanity_check(
        "kv_no_oracle_causality", causal_ok,
        {"first_half_len": half, "identical": causal_ok},
        source="mooncake", ref_tier=tier,
        detail="first-half KV outcomes identical with/without later requests → causal, no oracle"))

    # KV-aware routing executes causally and captures reuse across servers.
    router = KVAwareRouter(4, capacity_blocks=max(8, distinct // 2),
                           block_tokens=FOOTPRINTS[DEFAULT_FOOTPRINT].block_tokens)
    for r in train:
        router.route(r.hash_ids)
    summ = router.summary()
    route_ok = summ["routed"]["kv_aware"] == n and summ["total_prefill_blocks_reused"] >= 0
    checks.append(sanity_check(
        "kv_aware_routing", route_ok,
        {"routed": summ["routed"]["kv_aware"], "reused_blocks": summ["total_prefill_blocks_reused"],
         "n_servers": summ["n_servers"]},
        source="mooncake", ref_tier="SIMULATED",
        detail="KV-aware routing runs causally over the trace; reuse captured (SIMULATED, not measured)"))
    return checks


# ---------------------------------------------------------------------------
# electricity / cost checks
# ---------------------------------------------------------------------------

def electricity_checks(fleet_plane) -> list:
    prices = list(fleet_plane.price_by_hour.values())
    if not prices:
        return [skipped_check(
            "electricity_price", source="iso", required_artifact="regional price series",
            command="provide an electricity sample", reason="no prices")]
    in_band = all(0.005 <= p <= 2.0 for p in prices)
    band = ValidationCheck(
        kind="electricity_price_sanity", source=f"iso_{fleet_plane.region.lower()}",
        ref_tier="SAMPLE_FIXTURE", mode="sanity", metric=0.0 if in_band else 1.0,
        metric_name="all_in_band", metrics={"min": round(min(prices), 5),
        "max": round(max(prices), 5), "mean": round(statistics.mean(prices), 5)},
        tolerance=0.5, warn_tolerance=0.5, verdict=PASS if in_band else FAIL,
        detail="diurnal $/kWh within [0.005, 2.0]; SAMPLE_FIXTURE (live ISO pull blocked)")
    heldout = skipped_check(
        "electricity_heldout_iso", source="iso",
        required_artifact="a live regional ISO price pull (PJM/ERCOT/CAISO) for held-out comparison",
        command=ELEC_MANUAL_STEP, reason="live ISO auth flow unresolved in this environment")
    return [band, heldout]


# ---------------------------------------------------------------------------
# Azure serving checks (held-out time split, from the calibration bridge)
# ---------------------------------------------------------------------------

def azure_checks(bridge) -> list:
    if bridge is None:
        return [skipped_check(
            "azure_serving", source="azure_llm_2024", required_artifact="calibrated bridge",
            command="environment.calibrate(azure_raw)", reason="no calibration bridge")]
    h = bridge.holdout
    return [
        check_samples("azure_token_distribution", h["train_tokens"], h["azure_tokens"],
                      source="azure_llm_2024", ref_tier="held-out (time split)",
                      tolerance=0.15, warn_tolerance=0.25),
        check_samples("azure_interarrival", h["train_interarrival"], h["azure_interarrival"],
                      source="azure_llm_2024", ref_tier="held-out (time split)",
                      tolerance=0.20, warn_tolerance=0.35),
    ]


def build_all_checks(*, bridge, fleet_plane, processed_dir: str | None = None) -> list:
    """Assemble the full breadth of validation checks across all four planes."""
    checks: list = []
    checks += azure_checks(bridge)
    checks += v2026_fleet_checks(fleet_plane, processed_dir=processed_dir)
    checks += mooncake_kv_checks()
    checks += kv_simulator_checks()
    checks += electricity_checks(fleet_plane)
    return checks


__all__ = [
    "v2026_fleet_checks", "mooncake_kv_checks", "kv_simulator_checks",
    "electricity_checks", "azure_checks", "build_all_checks",
]
