"""Request-level GenAI serving-replay backtest for the Alibaba GenAI 2026 trace.

Drives a serving replay from the **application layer ONLY** (real arrivals +
measured ``e2e_latency_s`` as the per-request service demand) through the
UNCHANGED serving physics in ``aurelius/simulation/cluster/serving.py`` (Erlang-C
+ saturation + tail multipliers) and scores the canonical KPI
(``docs/RESULTS.md`` §1). goodput_unit = **completed_requests** (no output-token
field exists; honest).

Cross-layer honesty: the application layer cannot be joined to the metric layers
(incompatible anonymized time bases, no container_ip in requests — see
``alibaba_genai.classify_linkage``). The pipeline cold-start latency layers are
used only as a DISTRIBUTION CALIBRATION for the replay's model-cold-start cost
(``cold_start_s`` medians passed in), never as a per-request causal join.

The decisive lever on this trace is **model-affinity / prewarm**: the trace has
many base models with a large measured base-model load latency, so a router that
keeps requests on warm replicas avoids repeated cold starts. ``constraint_aware``
uses affinity; the baselines load-balance without it. Same physics, calibration
and cost basis across all policies — only the provisioning/routing decision
differs. Not a production claim; no constants tuned to favour a policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

from aurelius.benchmarks.economics import (
    InfrastructureCostConfig,
    compute_economic_kpi,
)
from aurelius.optimizer.aurelius_optimizer import AureliusOptimizer
from aurelius.optimizer.policies.genai_serving import (
    genai_effective_service_s,
    genai_size_for_sla,
    genai_size_for_target,
)
from aurelius.simulation.cluster import serving

from .schema import NormalizedGenAIRequest

# Phase 3d: route constraint_aware through the canonical AureliusOptimizer facade.
_GENAI_OPTIMIZER = AureliusOptimizer(policy="genai_serving")

# Documented priors (identical across policies; override before any claim).
GPU_HOUR_PRICE = 3.0                 # SD serving GPU ($/hr) — public-list ballpark
SLA_LATENCY_MULT = 2.0               # e2e p99 budget = mult × request exec + abs
SLA_LATENCY_ABS_S = 30.0
TARGET_RHO_SLA = 0.65
TARGET_RHO_UTIL = 0.85               # utilization_aware runs hotter (fewer replicas)
MIN_REPLICAS = 1

# Cold-start calibration defaults (s) if the pipeline layer is absent. When the
# pipeline layer IS ingested these are overridden by measured medians.
DEFAULT_COLD_START = {
    "basemodel_load": 22.0, "lora_load": 4.4, "controlnet_load": 3.9,
}

POLICIES = ("fifo", "sla_aware", "queue_aware", "utilization_aware", "constraint_aware")
HEADLINE_BASELINE = "sla_aware"      # interactive inference (docs/RESULTS.md §3 r5)


@dataclass
class TickAgg:
    tick_index: int
    start_s: float
    n: int
    arrival_rate: float
    mean_exec_s: float
    distinct_models: int
    lora_frac: float
    controlnet_frac: float
    failures: int


def _aggregate_ticks(requests, tick_seconds: float) -> list[TickAgg]:
    if not requests:
        return []
    t0 = requests[0].timestamp_s
    t1 = requests[-1].timestamp_s
    n_ticks = max(1, int((t1 - t0) // tick_seconds) + 1)
    buckets: list[list] = [[] for _ in range(n_ticks)]
    for r in requests:
        idx = min(n_ticks - 1, int((r.timestamp_s - t0) / tick_seconds))
        buckets[idx].append(r)
    ticks = []
    for i, b in enumerate(buckets):
        if not b:
            ticks.append(TickAgg(i, t0 + i * tick_seconds, 0, 0.0, 0.0, 0, 0.0, 0.0, 0))
            continue
        served = [r for r in b if not r.is_failed] or b
        execs = [r.e2e_latency_s for r in served if r.e2e_latency_s]
        mean_exec = (sum(execs) / len(execs)) if execs else 1.0
        models = {r.service_id for r in b}
        lora = sum(1 for r in b if (r.num_lora or 0) > 0)
        cnet = sum(1 for r in b if r.request_type in ("IMG_2_IMG", "INPAINTING"))
        ticks.append(TickAgg(
            tick_index=i, start_s=t0 + i * tick_seconds, n=len(b),
            arrival_rate=len(b) / tick_seconds, mean_exec_s=max(0.1, mean_exec),
            distinct_models=len(models), lora_frac=lora / len(b),
            controlnet_frac=cnet / len(b),
            failures=sum(1 for r in b if r.is_failed)))
    return ticks


def _effective_service_s(tick: TickAgg, cold: dict, affinity: bool) -> float:
    """Per-request mean service time incl. model cold-start.

    Delegates to the canonical physics owner
    :func:`~aurelius.optimizer.policies.genai_serving.genai_effective_service_s`.
    """
    return genai_effective_service_s(
        tick.mean_exec_s, tick.n, tick.distinct_models,
        tick.lora_frac, tick.controlnet_frac, cold, affinity,
    )


@dataclass
class PolicyResult:
    policy: str
    kpi: object
    completed_requests: int
    sla_compliant_requests: int
    e2e_p95_s: float
    e2e_p99_s: float
    queue_p95_s: float
    queue_p99_s: float
    timeout_rate_pct: float
    replica_hours: float
    scale_events: int
    affinity: bool
    mean_cold_start_s: float

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "goodput_unit": "completed_requests",
            "sla_safe_goodput_per_infra_dollar": self.kpi.sla_safe_goodput_per_infra_dollar,
            "sla_compliant_requests": self.sla_compliant_requests,
            "completed_requests": self.completed_requests,
            "total_infrastructure_cost": round(self.kpi.total_infrastructure_cost, 2),
            "replica_gpu_hours": round(self.replica_hours, 2),
            "e2e_latency_s_p95": round(self.e2e_p95_s, 2),
            "e2e_latency_s_p99": round(self.e2e_p99_s, 2),
            "queue_wait_s_p95": round(self.queue_p95_s, 2),
            "queue_wait_s_p99": round(self.queue_p99_s, 2),
            "timeout_rate_pct": round(self.timeout_rate_pct, 3),
            "scale_events": self.scale_events,
            "affinity_routing": self.affinity,
            "mean_cold_start_s": round(self.mean_cold_start_s, 2),
        }


def _eval_tick(tick, replicas, cold, affinity):
    """Erlang-C serving evaluation for one tick (reuses serving.py physics)."""
    replicas = max(MIN_REPLICAS, int(replicas))
    service_s = _effective_service_s(tick, cold, affinity)
    mu = 1.0 / service_s if service_s > 0 else 1.0
    lam = tick.arrival_rate
    rho = lam / (replicas * mu) if replicas > 0 else 1.0
    wait_s = serving.erlang_c_wait_s(lam, mu, replicas)
    if not math.isfinite(wait_s):
        wait_s = 60.0
    wait_s = min(600.0, wait_s * serving.saturation_amplifier(rho))
    p95m, p99m = serving.tail_multipliers(rho)
    e2e_p95 = wait_s * (p95m / 2 + 1) + service_s
    e2e_p99 = wait_s * (p99m / 2 + 1) + service_s
    sla = SLA_LATENCY_ABS_S + SLA_LATENCY_MULT * tick.mean_exec_s
    timeout = min(50.0, (e2e_p99 - sla) / sla * 10.0) if e2e_p99 > sla else 0.0
    return {"rho": rho, "wait_s": wait_s, "e2e_p95": e2e_p95, "e2e_p99": e2e_p99,
            "timeout": timeout, "service_s": service_s,
            "cold_s": service_s - tick.mean_exec_s, "sla": sla}


def _size_for_target(tick: TickAgg, cold: dict, affinity: bool, target_rho: float) -> int:
    """Delegates to the canonical physics owner :func:`genai_size_for_target`."""
    return genai_size_for_target(
        tick.n, tick.arrival_rate, tick.mean_exec_s,
        tick.distinct_models, tick.lora_frac, tick.controlnet_frac,
        cold, affinity, target_rho,
    )


def _size_for_sla(tick: TickAgg, cold: dict, affinity: bool) -> int:
    """Delegates to the canonical physics owner :func:`genai_size_for_sla`."""
    return genai_size_for_sla(
        tick.n, tick.arrival_rate, tick.mean_exec_s,
        tick.distinct_models, tick.lora_frac, tick.controlnet_frac,
        cold, affinity,
    )


def _run_policy(policy, ticks, cold, tick_hours) -> PolicyResult:
    affinity = policy == "constraint_aware"
    # global fixed sizing for fifo
    active = [t for t in ticks if t.n > 0]
    if active:
        peak = max(active, key=lambda t: t.arrival_rate)
        fifo_replicas = _size_for_target(peak, cold, False, TARGET_RHO_SLA)
    else:
        fifo_replicas = MIN_REPLICAS

    # Phase 3d: pre-compute constraint_aware replica counts via AureliusOptimizer.
    _ca_counts: list[int] = []
    if policy == "constraint_aware":
        _ca_counts = _GENAI_OPTIMIZER.optimize(ticks, cold).replica_counts

    tokens_per_tick = []
    timeout_per_tick = []
    gpu_hours_tick = []
    e2e95 = []
    e2e99 = []
    q95 = []
    q99 = []
    weights = []
    replica_hours = 0.0
    scale_events = 0
    prev_r = None
    cold_sum = 0.0
    cold_w = 0
    prev_tick = None

    for i, t in enumerate(ticks):
        if policy == "fifo":
            r = fifo_replicas
        elif policy == "sla_aware":
            src = prev_tick if prev_tick is not None else t   # reactive lag
            r = _size_for_sla(src, cold, False) if (src and src.n) else MIN_REPLICAS
        elif policy == "queue_aware":
            src = prev_tick if prev_tick is not None else t
            r = _size_for_target(src, cold, False, TARGET_RHO_SLA) if (src and src.n) else MIN_REPLICAS
        elif policy == "utilization_aware":
            r = _size_for_target(t, cold, False, TARGET_RHO_UTIL) if t.n else MIN_REPLICAS
        elif policy == "constraint_aware":
            r = _ca_counts[i]
        else:  # pragma: no cover
            raise ValueError(policy)

        ev = _eval_tick(t, r, cold, affinity)
        tokens_per_tick.append(t.n)          # 1 "token" == 1 request (goodput unit)
        timeout_per_tick.append(ev["timeout"])
        gpu_hours = r * tick_hours
        gpu_hours_tick.append({"genai-gpu": gpu_hours})
        replica_hours += gpu_hours
        if t.n > 0:
            e2e95.append(ev["e2e_p95"])
            e2e99.append(ev["e2e_p99"])
            q95.append(ev["wait_s"])
            q99.append(ev["wait_s"])
            weights.append(t.n)
            cold_sum += ev["cold_s"] * t.n
            cold_w += t.n
        if prev_r is not None and r != prev_r:
            scale_events += 1
        prev_r = r
        prev_tick = t

    cfg = InfrastructureCostConfig(gpu_hour_prices={"genai-gpu": GPU_HOUR_PRICE},
                                   fallback_gpu_hour_price=GPU_HOUR_PRICE)
    kpi = compute_economic_kpi(
        tokens_per_tick=tokens_per_tick, timeout_rate_pct_per_tick=timeout_per_tick,
        energy_cost_per_tick=[0.0] * len(ticks),
        active_gpu_hours_by_type_per_tick=gpu_hours_tick,
        migration_count=scale_events, config=cfg)

    def wmean(vals):
        tw = sum(weights)
        return sum(v * w for v, w in zip(vals, weights)) / tw if tw else 0.0

    return PolicyResult(
        policy=policy, kpi=kpi,
        completed_requests=sum(tokens_per_tick),
        sla_compliant_requests=kpi.sla_compliant_goodput,
        e2e_p95_s=wmean(e2e95), e2e_p99_s=wmean(e2e99),
        queue_p95_s=wmean(q95), queue_p99_s=wmean(q99),
        timeout_rate_pct=wmean(timeout_per_tick),
        replica_hours=replica_hours, scale_events=scale_events, affinity=affinity,
        mean_cold_start_s=cold_sum / cold_w if cold_w else 0.0)


@dataclass
class GenAIOutcome:
    outcome: str
    margin_pct: float
    headline: str
    safety_evidence: list = field(default_factory=list)
    notes: str = ""


def _classify(results: dict) -> GenAIOutcome:
    ca = results["constraint_aware"]
    base = results.get(HEADLINE_BASELINE)
    cg = ca.kpi.sla_safe_goodput_per_infra_dollar or 0.0
    bg = (base.kpi.sla_safe_goodput_per_infra_dollar or 0.0) if base else 0.0
    margin = ((cg - bg) / bg * 100.0) if bg > 0 else 0.0
    safety = []
    for name, r in results.items():
        if name in ("constraint_aware", HEADLINE_BASELINE):
            continue
        if r.e2e_p99_s > 0 and ca.e2e_p99_s <= 0.5 * r.e2e_p99_s:
            safety.append(f"e2e_p99<=0.5x_{name}")
    if margin > 1.0:
        return GenAIOutcome("ALPHA_WIN", margin, HEADLINE_BASELINE, safety)
    if abs(margin) <= 1.0 and safety:
        return GenAIOutcome("SAFETY_WIN", margin, HEADLINE_BASELINE, safety)
    if abs(margin) <= 1.0:
        return GenAIOutcome("TIE", margin, HEADLINE_BASELINE)
    return GenAIOutcome("LOSS", margin, HEADLINE_BASELINE,
                        notes="constraint_aware below sla_aware headline")


@dataclass
class GenAIBacktestResult:
    n_requests: int
    n_ticks: int
    tick_seconds: float
    cold_start_s: dict
    policy_results: dict
    outcome: GenAIOutcome

    def to_summary_dict(self) -> dict:
        return {
            "primary_kpi": "sla_safe_goodput_per_infrastructure_dollar",
            "goodput_unit": "completed_requests",
            "headline_baseline": self.outcome.headline,
            "n_requests": self.n_requests, "n_ticks": self.n_ticks,
            "tick_seconds": self.tick_seconds,
            "cold_start_calibration_s": self.cold_start_s,
            "policies": {p: r.summary() for p, r in self.policy_results.items()},
            "outcome": {
                "constraint_aware_vs_headline": self.outcome.outcome,
                "margin_pct": round(self.outcome.margin_pct, 4),
                "safety_evidence": self.outcome.safety_evidence,
                "notes": self.outcome.notes,
            },
        }


def run_backtest(requests: Sequence[NormalizedGenAIRequest], *,
                 tick_seconds: float = 3600.0,
                 cold_start_s: Optional[dict] = None,
                 policies: Sequence[str] = POLICIES) -> GenAIBacktestResult:
    cold = dict(DEFAULT_COLD_START)
    if cold_start_s:
        for k in ("basemodel_load", "lora_load", "controlnet_load"):
            if cold_start_s.get(k):
                cold[k] = cold_start_s[k]
    ticks = _aggregate_ticks(list(requests), tick_seconds)
    tick_hours = tick_seconds / 3600.0
    results = {p: _run_policy(p, ticks, cold, tick_hours) for p in policies}
    outcome = _classify(results) if "constraint_aware" in results else GenAIOutcome(
        "TIE", 0.0, HEADLINE_BASELINE)
    return GenAIBacktestResult(
        n_requests=len(requests), n_ticks=len(ticks), tick_seconds=tick_seconds,
        cold_start_s=cold, policy_results=results, outcome=outcome)


def predictive_layer_analysis(cold_start_s: dict, gateway_summary: dict,
                              app_summary: dict) -> dict:
    """Which layer's latency magnitude most plausibly dominates p99 (calibration
    comparison, NOT a per-request join)."""
    basemodel = cold_start_s.get("basemodel_load", 0.0)
    gateway_wait = (gateway_summary.get("waiting_time_s_p99") or 0.0)
    exec_p99 = (app_summary.get("e2e_latency_s_p99") or 0.0)
    exec_p50 = (app_summary.get("e2e_latency_s_p50") or 0.0)
    contributions = {
        "scheduler_pipeline_cold_start_s": round(basemodel, 2),
        "gateway_queue_wait_s": round(gateway_wait, 4),
        "request_exec_variance_s": round(max(0.0, exec_p99 - exec_p50), 2),
    }
    dominant = max(contributions, key=contributions.get)
    # The intrinsic request-exec variance is NOT addressable by orchestration;
    # among the ADDRESSABLE layers (gateway vs pipeline cold-start) report which
    # an optimizer can actually act on.
    addressable = {k: v for k, v in contributions.items()
                   if k != "request_exec_variance_s"}
    most_addressable = max(addressable, key=addressable.get) if addressable else None
    return {"contributions_s": contributions, "most_predictive_of_p99": dominant,
            "most_addressable_of_p99": most_addressable}
