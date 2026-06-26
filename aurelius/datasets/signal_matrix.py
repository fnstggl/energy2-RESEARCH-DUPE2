"""Canonical production-like dataset — the SIGNAL MATRIX (machine-readable audit).

The joint optimizer has surfaces on five cost terms (capacity, ordering,
admission, energy, placement, plus the thermal/topology/KV physics that gate
them). Each surface needs specific signals to act on. Today those signals live in
**stratified** public traces: inference traces (Azure / BurstGPT / Mooncake)
carry arrivals+tokens but no system state; training traces (Alibaba PAI / Philly /
Acme) carry GPU state but no serving. No single public trace carries all of them,
which is why the joint optimizer can only ever be tested one lever at a time.

This module is the **audit**: every signal the canonical trace would need, which
public dataset supplies it, the field it maps to in our normalized schema
(:mod:`aurelius.traces.schema`), and — critically — its **fidelity tier**: is the
signal MEASURED on the right hardware/workload, a defensible PROXY, a SYNTHETIC
overlay, only obtainable in a SIMULATOR, or simply ABSENT from all public data?

It is deliberately structured (not prose) so it can be queried, tested, and kept
honest: ``coverage_by_tier()`` reports how much of the canonical trace is real
vs. proxy vs. simulator-only, so we never quietly present a stitched-together
dataset as if it were production telemetry. The companion prose analysis (stitch
risks, "monstrous dataset" failure modes, build sequence) is in
``research/CANONICAL_PRODUCTION_DATASET_DESIGN.md``.

Nothing here is a production claim. The whole point of the tiering is to make the
gap between "production-like" and "production" explicit and measurable.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Fidelity tiers (ordered best → worst) ---------------------------------
TIER_MEASURED = "MEASURED_REAL"        # real telemetry, right hardware + workload
TIER_PROXY = "PROXY"                   # derived from a real but indirect signal
TIER_SYNTHETIC = "SYNTHETIC"           # parameterized overlay, documented + reproducible
TIER_SIMULATOR = "SIMULATOR_ONLY"      # only a simulator/analytic model exists publicly
TIER_ABSENT = "ABSENT"                 # no public source at any fidelity

_TIER_ORDER = {
    TIER_MEASURED: 0, TIER_PROXY: 1, TIER_SYNTHETIC: 2,
    TIER_SIMULATOR: 3, TIER_ABSENT: 4,
}


@dataclass(frozen=True)
class CanonicalSignal:
    """One signal the canonical trace must carry, and where it would come from."""

    name: str
    layer: str                 # optimizer layer that consumes it
    lever: str                 # which surface needs it
    source_dataset: str        # best public source (or "—" if none)
    source_field: str          # field in aurelius.traces.schema (or proposed)
    tier: str                  # fidelity tier (one of TIER_*)
    in_repo: bool              # is a loader for this source already in the repo?
    stitch_risk: str           # the mismatch risk when combined with the spine
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "layer": self.layer, "lever": self.lever,
            "source_dataset": self.source_dataset, "source_field": self.source_field,
            "tier": self.tier, "in_repo": self.in_repo,
            "stitch_risk": self.stitch_risk, "note": self.note,
        }


# ---------------------------------------------------------------------------
# THE MATRIX
# ---------------------------------------------------------------------------
# Spine = the real demand backbone everything else attaches to: the Azure LLM
# 2024 inference trace (real arrivals + real output tokens). Every other signal
# is either carried by the spine, derivable from it, sourced from a SECOND real
# trace and time-aligned, overlaid synthetically, or left to the simulator.

CANONICAL_SIGNAL_MATRIX: tuple = (
    # --- serving spine (MEASURED on the right workload) --------------------
    CanonicalSignal(
        "arrival_time", "Replay", "all", "Azure LLM 2024", "timestamp_s",
        TIER_MEASURED, True,
        "none — this is the spine",
        "Real inter-arrival structure (bursts/diurnal). The one signal every "
        "public serving trace agrees on.",
    ),
    CanonicalSignal(
        "output_tokens", "Objective", "ordering/capacity", "Azure LLM 2024",
        "output_tokens", TIER_MEASURED, True,
        "none — this is the spine",
        "Drives service time (TTFT+tokens·TPOT) and the goodput numerator.",
    ),
    CanonicalSignal(
        "prompt_tokens", "Constraint", "kv/capacity", "Azure LLM 2024",
        "prompt_tokens", TIER_MEASURED, True,
        "none — this is the spine",
        "Prefill cost + KV footprint proxy.",
    ),
    CanonicalSignal(
        "model_id", "Decision", "placement/routing", "BurstGPT / Mooncake",
        "model", TIER_MEASURED, True,
        "Azure spine is single-model; multi-model must come from a 2nd trace "
        "whose arrival base differs → time-align or treat as a separate tier",
        "Needed for placement affinity + per-class conformal. Azure trace is "
        "effectively single-model, so this is the first real stratification gap.",
    ),

    # --- workload class (the lever that unlocks admission) -----------------
    CanonicalSignal(
        "workload_class", "Constraint", "admission",
        "Alibaba cluster-trace-gpu-v2026", "pod_hourly.job_type_public",
        TIER_MEASURED, False,
        "v2026 gives REAL online_inference vs offline_inference labels (on-domain "
        "serving best-effort ratio) — hourly pod aggregates, distribution-level, "
        "not a per-record join to the token spine",
        "latency_critical vs best_effort. THE missing dimension. The compounding "
        "magnitude is BOUND to this ratio (2026-06-26 correction). v2026's online/"
        "offline-inference split is the correct on-domain number; the v2023 QoS "
        "(LS/BE) used so far is a training-pod PROXY. Use "
        "calibration.alibaba_v2026_serving_class_mix once the trace is downloaded.",
    ),
    CanonicalSignal(
        "best_effort_overlay", "Decision", "admission/energy", "—",
        "(synthetic)", TIER_SYNTHETIC, True,
        "synthetic by construction — must be labeled, parameterized, reproducible "
        "and never reported as real demand",
        "Minimal realizable slice: a documented batch/offline tier alongside the "
        "real interactive spine, so admission + energy time-shift have deferrable "
        "load to act on. This is what makes compounding testable today.",
    ),

    # --- KV cache (Mooncake is the one real source) ------------------------
    CanonicalSignal(
        "kv_prefix_hit", "Constraint", "kv/placement", "Mooncake (hash_ids)",
        "cache_affinity_key", TIER_PROXY, True,
        "Mooncake hash_ids are block-level prefix hashes — a real reuse SIGNAL "
        "but on Kimi/Moonshot traffic, not Azure; prefix-hit RATE is computable "
        "but preemption/recompute/routing remain simulator-only",
        "schema already documents cache_affinity_key as a PROXY, not a measured "
        "hit rate. Mooncake upgrades the proxy from session-id to real prefix "
        "hashes; the memory-pressure dynamics still need the simulator.",
    ),
    CanonicalSignal(
        "kv_block_reuse_dynamics", "Replay", "kv", "—", "(simulator)",
        TIER_SIMULATOR, False,
        "live KV eviction / recompute / per-instance routing affinity are never "
        "published — only inferable in a serving simulator",
        "The difference between 'prefix-hit rate' (computable from Mooncake) and "
        "'realized KV benefit under memory pressure' (simulator-only).",
    ),

    # --- energy / power (Zeus has power, M100 has thermal) -----------------
    CanonicalSignal(
        "gpu_power_w", "Objective", "energy", "Zeus / ML.ENERGY v3",
        "NormalizedGPUUtilizationSample.power_w", TIER_PROXY, False,
        "Zeus power is real but on H100/B200 micro-benchmarks, not the Azure "
        "serving mix → power-vs-utilization CURVE transfers, absolute watts do not",
        "Use Zeus to calibrate a power(utilization) curve, then drive it from the "
        "spine's utilization — a calibrated model, not a measured series.",
    ),
    CanonicalSignal(
        "energy_price", "Objective", "energy", "public grid / ISO (EIA, CAISO)",
        "(exogenous series)", TIER_MEASURED, False,
        "grid price is real and public but must be aligned to the trace's wall-"
        "clock; the Azure trace is time-anonymized → align by relative hour-of-day",
        "Energy time-shift needs a price signal AND deferrable (best-effort) load. "
        "Real price series exist; the join is by hour-of-day, not absolute time.",
    ),

    # --- thermal (the systematic blind spot) -------------------------------
    CanonicalSignal(
        "gpu_temperature_c", "Constraint", "thermal", "M100 ExaData (CINECA)",
        "NormalizedGPUUtilizationSample.temperature_c", TIER_PROXY, False,
        "M100 ExaData is V100/HPC, not H100 inference — thermal_alpha/beta "
        "constants transfer in FORM, the absolute temperatures do not; throttle/"
        "clock-event ground truth is never published at all",
        "Temperature is the canonical blind spot. ExaData gives a real temp(power,"
        "util,cooling) shape to fit thermal_alpha/beta; absolute throttle behavior "
        "stays a calibrated model.",
    ),
    CanonicalSignal(
        "throttle_events", "Replay", "thermal", "—", "(simulator)",
        TIER_SIMULATOR, False,
        "clock-throttle / thermal-violation events are not in any public dataset",
        "Only obtainable from a thermal model or a real DCGM shadow pilot.",
    ),

    # --- topology / collectives (half real, half simulator) ----------------
    CanonicalSignal(
        "collective_cost", "Constraint", "placement/topology",
        "MLCommons Chakra + nccl-tests", "(alpha-beta model)", TIER_PROXY, False,
        "Chakra gives real collective TRACES + nccl-tests give real bandwidth "
        "sweeps → an alpha-beta cost model transfers; the value depends on the "
        "exact fabric, so it is a calibrated model not a measured cost",
        "The 'cost of a collective on a given placement' half is calibratable.",
    ),
    CanonicalSignal(
        "fabric_congestion", "Constraint", "placement/topology", "—",
        "(simulator: ASTRA-sim)", TIER_SIMULATOR, False,
        "congestion / incast / straggler data (Alibaba HPN/C4, ByteDance "
        "MegaScale, Meta) is paper-only — no public trace; only ASTRA-sim "
        "(sim-vs-sim) exists",
        "The 'what actually happens under contention' half is absent. Hard ceiling "
        "on topology realism from public data alone.",
    ),

    # --- utilization / packing (training traces, wrong workload) -----------
    CanonicalSignal(
        "gpu_utilization", "Forecast", "capacity/packing",
        "Alibaba cluster-trace-gpu-v2026", "pod_hourly.avg_gpu_sm_util",
        TIER_PROXY, False,
        "v2026 gives real hourly SM-util per pod LABELLED by job_type "
        "(online/offline inference) — finally inference util, not just training; "
        "still hourly-aggregate, sanity-check not drive",
        "v2026 upgrades this over Acme/PAI (which are training-only): real "
        "online_inference utilization distribution to calibrate the simulator's "
        "implied util. Hourly aggregates cannot drive the per-request serving loop.",
    ),
    CanonicalSignal(
        "network_traffic", "Constraint", "placement/topology",
        "Alibaba cluster-trace-gpu-v2026", "network_hourly.rx/tx_gibps_avg",
        TIER_PROXY, False,
        "real per-server hourly rx/tx — MACRO traffic only; no per-link congestion/"
        "incast/PFC-ECN (those stay simulator-only)",
        "NEW real signal in v2026: macro node-level network utilization, joinable "
        "to pods by server_id+hour. Upgrades network-load reasoning from absent to "
        "PROXY; micro-congestion (fabric_congestion) is still simulator-only.",
    ),
    CanonicalSignal(
        "fleet_topology", "Decision", "placement/topology",
        "Alibaba cluster-trace-gpu-v2026", "server_hourly.asw_id/gpu_spec_public",
        TIER_MEASURED, False,
        "real rack/access-switch topology + heterogeneous GPU inventory at "
        "155k-GPU scale — server-hour granularity",
        "NEW in v2026: real ASW (rack) topology + GPU-type inventory for "
        "topology-aware placement / ASW-local packing. The fleet-scheduling "
        "substrate; a different product surface from token serving.",
    ),
    CanonicalSignal(
        "gpu_fragmentation", "Decision", "placement/packing",
        "Alibaba cluster-trace-gpu (gpu_milli)", "NormalizedGPUJob.gpu_milli",
        TIER_MEASURED, True,
        "real fractional-GPU packing requests — but a SEPARATE problem from the "
        "serving spine; joined at the FLEET level (shared GPUs), not per-request",
        "Fractional-GPU sharing requests are real; they bound the packing lever.",
    ),
    CanonicalSignal(
        "inference_autoscaling_truth", "Evaluation", "capacity", "—",
        "(absent)", TIER_ABSENT, False,
        "online-inference autoscaling / batching / migration ground truth "
        "(SageServe, Aegaeon, DynamoLLM) is paper-only — nothing public",
        "We cannot validate the capacity lever against real autoscaler decisions; "
        "only against the Erlang-C/queueing model. A genuine ceiling.",
    ),
)


# ---------------------------------------------------------------------------
# Coverage reporting (keeps the audit honest)
# ---------------------------------------------------------------------------

def coverage_by_tier(matrix: tuple = CANONICAL_SIGNAL_MATRIX) -> dict:
    """Count signals per fidelity tier — the honesty gauge for the dataset."""
    out: dict = {t: 0 for t in _TIER_ORDER}
    for s in matrix:
        out[s.tier] = out.get(s.tier, 0) + 1
    return out


def coverage_by_lever(matrix: tuple = CANONICAL_SIGNAL_MATRIX) -> dict:
    """Best (lowest) fidelity tier achievable per lever — what's testable today."""
    best: dict = {}
    for s in matrix:
        cur = best.get(s.lever)
        if cur is None or _TIER_ORDER[s.tier] < _TIER_ORDER[cur]:
            best[s.lever] = s.tier
    return best


def realizable_today(matrix: tuple = CANONICAL_SIGNAL_MATRIX) -> tuple:
    """Signals we can supply NOW at MEASURED/PROXY/SYNTHETIC fidelity (in-repo)."""
    ok = {TIER_MEASURED, TIER_PROXY, TIER_SYNTHETIC}
    return tuple(s for s in matrix if s.tier in ok and s.in_repo)


def simulator_or_absent(matrix: tuple = CANONICAL_SIGNAL_MATRIX) -> tuple:
    """Signals that are simulator-only or absent — the hard ceiling on realism."""
    return tuple(s for s in matrix if s.tier in (TIER_SIMULATOR, TIER_ABSENT))


def matrix_as_dicts(matrix: tuple = CANONICAL_SIGNAL_MATRIX) -> list:
    return [s.to_dict() for s in matrix]


__all__ = [
    "TIER_MEASURED", "TIER_PROXY", "TIER_SYNTHETIC", "TIER_SIMULATOR", "TIER_ABSENT",
    "CanonicalSignal", "CANONICAL_SIGNAL_MATRIX",
    "coverage_by_tier", "coverage_by_lever", "realizable_today",
    "simulator_or_absent", "matrix_as_dicts",
]
