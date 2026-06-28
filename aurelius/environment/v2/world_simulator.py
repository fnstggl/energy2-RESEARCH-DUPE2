"""WorldSimulatorV2 — the V2 integration spine (everything → SLA-safe goodput/$).

Ties the V2 components into one causal period:

    arrival → routing/admission → per-replica tiered-KV hit-or-recompute → roofline prefill/decode timing
    (precision/spec/clock) → continuous batching / pool queues / KV handoff → GPU-seconds / energy / cost
    → SLA-safe goodput/$.

Every mechanism reaches reward ONLY through those physical quantities — there is no reward bonus, no action
scalar, no "roofline bonus" (hard rules #4–#8). Candidate evaluation runs on a clone of the persistent
:class:`CanonicalWorldStateV2`, so the MPC search never contaminates the real timeline (the V1 guarantee).

Economics reuse V1's :class:`cost_model.CostModel` verbatim (owned-depreciation + PUE-scaled ISO energy),
billed on a hybrid capacity/work basis (a warm-idle floor is never free — matches PR #107). Co-location
only reclaims idle GPU-seconds when there is real/trace-derived background work; with none it is inert and an
aggressive setting only adds a contention penalty, so the search prunes it (hard rule: no imaginary goodput).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cost_model import CostModel
from ..prefill_decode import HYBRID_IDLE_FLOOR_FRAC
from ..roofline_external import ARCHS
from .prefill_decode_scheduler import PrefillDecodeSchedulerV2, SchedRequest
from .roofline_serving import RooflineServingModelV2
from .world_state import CanonicalWorldStateV2, clone_state_v2

DEFAULT_ACTION = {
    "serving_mode": "shared_pool", "prefill_frac": None, "routing": "kv_aware",
    "precision": "bf16", "spec_decode": "off", "clock": "base", "colocation_mode": "off",
    "max_num_batched_tokens": 2048, "max_active_sequences": 64, "chunked_prefill": True,
    "timing_model": "roofline",          # "roofline" (V2 primary) | "legacy_scalar" (V1-equivalent baseline)
}
COLOC_RECLAIM = {"off": 0.0, "conservative": 0.5, "aggressive": 0.9}
COLOC_CONTENTION = {"off": 0.0, "conservative": 0.01, "aggressive": 0.05}


@dataclass
class PeriodOutcomeV2:
    metrics: dict = field(default_factory=dict)
    reward: float = 0.0                 # SLA-safe goodput/$
    sla_violation_rate: float = 0.0
    serving: dict = field(default_factory=dict)
    tiered_kv: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)


def _route(replicas, hash_ids, policy, idx):
    if policy == "round_robin" or not hash_ids:
        return idx % len(replicas)
    # kv_aware: route to the replica with the longest resident leading prefix (across tiers), causal.
    best_i, best_run = 0, -1
    for i, rep in enumerate(replicas):
        run = max(rep.kv._leading_run(t, hash_ids) for t in ("GPU_HBM", "CPU_DRAM", "REMOTE_KV", "SSD_NVME"))
        if run > best_run:
            best_i, best_run = i, run
    return best_i


def simulate_period_v2(state: CanonicalWorldStateV2, reqs, hash_seq, action: dict, *,
                       sla_s: float = 5.0, period_s: float = 60.0, cost_model: CostModel | None = None,
                       mutate: bool = True, scenario: str = "owned") -> PeriodOutcomeV2:
    """One causal V2 period. ``reqs`` = list of (arrival_s, out_tok, in_tok); ``hash_seq`` = Mooncake-style
    block-hash lists assigned by position (no row-join, as in V1). ``mutate=False`` runs on a clone."""
    a = {**DEFAULT_ACTION, **(action or {})}
    cost_model = cost_model or CostModel()
    if not mutate:
        state = clone_state_v2(state)
    replicas = [r for r in state.replicas if r.warm] or state.replicas
    n_replicas = len(replicas)
    arch = ARCHS.get(state.model, ARCHS["llama-8b-gqa"])
    kv_bpt = arch.kv_bytes_per_token

    tm = RooflineServingModelV2(gpu_type=state.gpu_type, arch_name=state.model,
                                mode=a.get("timing_model", "roofline"))
    # representative per-token prefill cost for the tier decision (precision/clock aware)
    rep_t = tm.estimate(prompt_tokens=512, output_tokens=128, precision=a["precision"],
                        spec_decode=a["spec_decode"], clock=a["clock"])
    pf_per_tok = rep_t.extra.get("prefill_per_tok", 0.00015)

    # route + tiered-KV serve (causal, mutates replica residency)
    ordered = sorted(enumerate(reqs), key=lambda kv: kv[1][0])
    sched_reqs = []
    quality_risk = rep_t.quality_risk
    m = len(hash_seq) if hash_seq else 0
    for pos, (orig_i, r) in enumerate(ordered):
        out = int(r[1])
        prompt = int(r[2]) if len(r) > 2 else out
        hids = list(hash_seq[orig_i % m]) if m else []
        i = _route(replicas, hids, a["routing"], pos)
        dec = replicas[i].kv.serve(hids, prefill_s_per_token=pf_per_tok)
        sched_reqs.append(SchedRequest(arrival_s=float(r[0]), prompt_tokens=prompt, output_tokens=out,
                                       saved_prefill_tokens=dec.saved_prefill_tokens,
                                       transfer_latency_s=dec.transfer_latency_s,
                                       recompute=(dec.tier == "RECOMPUTE")))

    sched = PrefillDecodeSchedulerV2(max_num_batched_tokens=a["max_num_batched_tokens"],
                                     max_active_sequences=a["max_active_sequences"],
                                     chunked_prefill=a["chunked_prefill"])
    sr = sched.simulate(sched_reqs, timing_model=tm, n_replicas=n_replicas,
                        serving_mode=a["serving_mode"], prefill_frac=a["prefill_frac"], sla_s=sla_s,
                        period_s=period_s, precision=a["precision"], spec_decode=a["spec_decode"],
                        clock=a["clock"], kv_bytes_per_token=kv_bpt)

    # SLA-safe goodput (tokens of requests completing within SLA), minus quality risk (int4)
    sla_safe_tokens = sum(req.output_tokens for req, comp in zip(sched_reqs, sr.completion_s) if comp <= sla_s)
    coloc = a["colocation_mode"]
    contention = COLOC_CONTENTION.get(coloc, 0.0)
    # co-location contention can push borderline requests over SLA (a service-time hit)
    if contention > 0:
        infl = [c * (1.0 + contention) for c in sr.completion_s]
        sla_safe_tokens = sum(req.output_tokens for req, comp in zip(sched_reqs, infl) if comp <= sla_s)
    sla_safe_tokens *= (1.0 - quality_risk)
    n_safe = sum(1 for c in sr.completion_s if c <= sla_s)
    sla_violation_rate = 1.0 - n_safe / max(1, len(sched_reqs))

    # economics: hybrid-billed GPU-seconds, co-location reclaims idle ONLY with real background work
    # Operator economics are PROVISIONED-capacity dominated (owned hardware): you pay for the GPU-seconds
    # you run, NOT for realized work — so faster service does NOT cut cost within a period (the honest V1
    # finding from PR #106/#107). Realized work above capacity is queued (→ SLA failures), never billed as
    # extra GPU-seconds. Cost varies only via clock→power (energy) and co-location reclaim of idle by REAL
    # background work. This deliberately removes the period_s/arrival-window artifact that would otherwise
    # bill realized>provisioned and manufacture a fake "faster service is cheaper" win (hard rule #4).
    provisioned = n_replicas * period_s
    realized = sr.realized_gpu_seconds
    idle = max(0.0, provisioned - realized)
    reclaim = min(state.background_work_gpu_seconds, idle) * COLOC_RECLAIM.get(coloc, 0.0)
    billed_gpu_s = max(provisioned - reclaim, HYBRID_IDLE_FLOOR_FRAC * provisioned)
    gpu_hours = billed_gpu_s / 3600.0
    util = min(1.0, realized / max(provisioned, 1e-9))
    breakdown = cost_model.operator_cost(gpu_hours=gpu_hours, gpu_type=state.gpu_type,
                                          energy_price_per_kwh=state.energy_price_per_kwh,
                                          utilization=max(0.05, util), scenario=scenario,
                                          power_scale=tm.power_scale(a["clock"]),
                                          sla_violations=int(sla_violation_rate * len(sched_reqs)))
    cost = breakdown.total_operator_cost
    reward = sla_safe_tokens / cost if cost > 0 else 0.0

    if mutate:
        state.period += 1

    metrics = {"sla_safe_goodput_tokens": round(sla_safe_tokens, 1), "cost_usd": round(cost, 5),
               "goodput_per_dollar": round(reward, 4), "billed_gpu_hours": round(gpu_hours, 5),
               "realized_gpu_seconds": round(realized, 3), "idle_gpu_seconds": round(idle, 3),
               "coloc_reclaim_gpu_seconds": round(reclaim, 3), "utilization": round(util, 4),
               "energy_cost": round(breakdown.energy_cost, 5), "sla_violation_rate": round(sla_violation_rate, 4),
               "quality_risk": round(quality_risk, 4)}
    return PeriodOutcomeV2(metrics=metrics, reward=reward, sla_violation_rate=sla_violation_rate,
                           serving=sr.summary(),
                           tiered_kv={rid: rep.kv.summary() for rid, rep in
                                      [(r.replica_id, r) for r in replicas[:1]]},  # sample replica
                           timing={"regime": rep_t.roofline_regime, "ridge": rep_t.ridge_point,
                                   "arithmetic_intensity": rep_t.arithmetic_intensity,
                                   "hbm_pressure": rep_t.hbm_pressure, "timing_model": rep_t.timing_model_used},
                           action=a)


class WorldSimulatorV2:
    """Thin OO wrapper so the MPC search and comparison harness share one entry point."""

    def __init__(self, cost_model: CostModel | None = None):
        self.cost_model = cost_model or CostModel()

    def evaluate(self, state, reqs, hash_seq, action, *, sla_s=5.0, period_s=60.0, mutate=False,
                 scenario="owned") -> PeriodOutcomeV2:
        return simulate_period_v2(state, reqs, hash_seq, action, sla_s=sla_s, period_s=period_s,
                                  cost_model=self.cost_model, mutate=mutate, scenario=scenario)


__all__ = ["WorldSimulatorV2", "simulate_period_v2", "PeriodOutcomeV2", "DEFAULT_ACTION"]
