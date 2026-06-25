"""Canonical Aurelius optimization policies (decision-layer seam).

All six decision-layer surfaces of the comprehensive ``AureliusOptimizer`` are
now implemented and owned here — the canonical optimizer is no longer a single
energy delegate, it is a multi-surface fleet optimizer (Phase B consolidation
over the Phase 1/2/3/3d extractions):

    EnergySchedulingPolicy   — thin delegate to JobScheduler (when/where/how-fast)
    ServingQueuePolicy       — extracted abs-conformal SRPT request ordering (Phase 2)
    ReplicaScalingPolicy     — extracted AMCSG/SOTSS + deployable forecasted_mcs (Phase 2/3)
    GenAIServingPolicy       — extracted multi-model GenAI constraint_aware sizing (Phase 3d)
    PlacementPolicy          — delegates to the residency decision engine (routing)
    AdmissionPolicy          — delegates to the frontier admission gate (flow control)

PlacementPolicy and AdmissionPolicy are **parity wirings** of existing, tested,
recommendation-only surfaces (``aurelius/residency/decision.py`` and
``aurelius/frontier/admission.py``) into the canonical seam — no new optimization
logic, no benchmark-assumption change. Every surface targets the one objective in
``docs/RESULTS.md`` §1: SLA-safe goodput per infrastructure dollar.
"""

from __future__ import annotations

from typing import Optional

from ...optimization.scheduler import JobScheduler, SchedulerResult
from .base import OptimizationPolicy
from .genai_serving import (
    GENAI_EWMA_ALPHA,
    GENAI_MIN_REPLICAS,
    GENAI_SLA_LATENCY_ABS_S,
    GENAI_SLA_LATENCY_MULT,
    GENAI_TARGET_RHO_SLA,
    GENAI_TARGET_RHO_UTIL,
    GenAIServingPolicy,
    GenAIServingResult,
    genai_effective_service_s,
    genai_eval_tick_timeout,
    genai_size_for_sla,
    genai_size_for_target,
)
from .replica_scaling import (
    REPLICA_AGGRESSIVE_GATE,
    REPLICA_MAX_ORACLE_ITERS,
    REPLICA_SAFE_GATE,
    REPLICA_TPOT_S,
    REPLICA_TTFT_BASE_S,
    ReplicaScalingConfig,
    ReplicaScalingPolicy,
    ReplicaScalingResult,
    compute_c1pgs_spot_replicas,
    compute_constraint_aware_schedule,
    compute_frontier_rho_schedule,
    compute_mcs_c_schedule,
    compute_shu_schedule,
    compute_sotss_min_schedule,
)
from .serving_queue import (
    CONFORMAL_ABS_TARGET_P90_TOKENS,
    CONFORMAL_ALPHA_MAX,
    CONFORMAL_WARMUP,
    CONFORMAL_WINDOW,
    AbsoluteErrorConformalCalibrator,
    ServingQueuePolicy,
    simulate_decoupled_hybrid_abs_conformal,
)


class EnergySchedulingPolicy(OptimizationPolicy):
    """Energy batch-scheduling policy — a thin delegate to ``JobScheduler``.

    Wraps the existing productized energy optimizer WITHOUT changing its
    behavior. ``optimize`` forwards verbatim to :meth:`JobScheduler.solve` and
    returns the unchanged :class:`SchedulerResult`.
    """

    name = "energy"

    def __init__(
        self,
        scheduler: Optional[JobScheduler] = None,
        *,
        config=None,
        **scheduler_kwargs,
    ):
        if scheduler is not None:
            if config is not None or scheduler_kwargs:
                raise ValueError(
                    "EnergySchedulingPolicy: pass either a prebuilt `scheduler` "
                    "or constructor args (`config`/kwargs), not both."
                )
            self.scheduler = scheduler
        else:
            self.scheduler = JobScheduler(config, **scheduler_kwargs)

    def optimize(self, jobs, price_data, carbon_data, *args, **kwargs) -> SchedulerResult:
        """Delegate verbatim to ``JobScheduler.solve`` (behavior-preserving).

        ``*args`` are forwarded positionally (e.g. ``risk_data``) so every
        production energy call site can route through the canonical optimizer
        without losing ``JobScheduler.solve``'s positional parameters.
        """
        return self.scheduler.solve(jobs, price_data, carbon_data, *args, **kwargs)

    def create_baseline_schedule(self, jobs):
        """Delegate to ``JobScheduler.create_baseline_schedule`` (ASAP home)."""
        return self.scheduler.create_baseline_schedule(jobs)


class PlacementPolicy(OptimizationPolicy):
    """GPU/region model-placement & routing policy — canonical owner.

    Delegates to the validated, recommendation-only Model Residency decision
    engine (``aurelius/residency/decision.py::choose_residency_decision``), which
    already maximizes the ``docs/RESULTS.md`` §1 KPI (SLA-safe goodput/$) over a
    set of candidate serving locations and is ``executable_in_real_cluster=False``
    by construction. This is a parity wiring of an existing, tested surface into
    the canonical optimizer seam — no new placement logic.

    ``optimize`` returns a :class:`aurelius.residency.models.ResidencyDecision`
    (route / prewarm / evict / keep / reject), never a substitution of the
    requested model. Constructs with no args (facade-constructible); a default
    :class:`SafetyContext` is built lazily.
    """

    name = "placement"

    def __init__(self, *, safety_context=None, cost_config=None):
        # Defaulted lazily in optimize() to avoid importing residency at import
        # time (keeps the optimizer package import-light).
        self._safety_context = safety_context
        self._cost_config = cost_config

    def optimize(
        self,
        request,
        locations,
        *,
        load_profiles=None,
        cost_config=None,
        safety_context=None,
    ):
        """Recommend where to place/route ``request`` (max SLA-safe goodput/$).

        Args:
            request: a ``ModelResidencyRequest`` (model/adapter + SLA + region).
            locations: candidate ``ModelLocationState`` objects.
            load_profiles: ``{model_id | (model_id, adapter_id): ModelLoadProfile}``.
            cost_config / safety_context: ``SafetyContext`` gates + cost basis;
                default to the policy's configured context (or a fresh default).
        """
        from ...residency.decision import (
            SafetyContext,
            choose_residency_decision,
        )

        safety = safety_context or self._safety_context or SafetyContext()
        cost = cost_config or self._cost_config or safety
        return choose_residency_decision(
            request, list(locations), dict(load_profiles or {}), cost, safety
        )


class AdmissionPolicy(OptimizationPolicy):
    """Flow-control admission policy — canonical owner.

    Delegates to the validated ``aurelius/frontier/admission.py::evaluate_admission``
    gate: a deterministic, telemetry-driven ADMIT/DEFER/REJECT control that is
    shadow-mode by default (``enabled=False`` → always ADMIT) and never defers or
    rejects latency-critical SLA classes. Parity wiring of an existing, tested
    surface — no new admission logic.

    ``optimize`` returns an
    :class:`aurelius.frontier.admission.AdmissionDecision`. Constructs with no
    args (facade-constructible).
    """

    name = "admission"

    def __init__(self, *, config=None):
        self._config = config

    def optimize(self, *, sla_class, window, config=None):
        """Decide ADMIT/DEFER/REJECT for one incoming workload class."""
        from ...frontier.admission import evaluate_admission

        return evaluate_admission(
            sla_class=sla_class, window=window, config=config or self._config
        )

    def optimize_batch(self, *, workloads, window, config=None):
        """Evaluate admission for many ``(workload_id, sla_class)`` at once."""
        from ...frontier.admission import evaluate_admission_batch

        return evaluate_admission_batch(
            workloads=workloads, window=window, config=config or self._config
        )


#: Registry of the canonical decision-layer policies.
POLICY_REGISTRY: dict[str, type[OptimizationPolicy]] = {
    EnergySchedulingPolicy.name: EnergySchedulingPolicy,
    ServingQueuePolicy.name: ServingQueuePolicy,
    ReplicaScalingPolicy.name: ReplicaScalingPolicy,
    GenAIServingPolicy.name: GenAIServingPolicy,
    PlacementPolicy.name: PlacementPolicy,
    AdmissionPolicy.name: AdmissionPolicy,
}

#: Policies that are actually implemented. All six decision-layer surfaces are
#: now live (Phase B over Phase 1/2/3/3d): the canonical optimizer covers energy
#: scheduling, serving ordering, replica capacity, GenAI multi-model sizing,
#: placement/routing, and admission control.
IMPLEMENTED_POLICIES: frozenset[str] = frozenset(
    {
        EnergySchedulingPolicy.name,
        ServingQueuePolicy.name,
        ReplicaScalingPolicy.name,
        GenAIServingPolicy.name,
        PlacementPolicy.name,
        AdmissionPolicy.name,
    }
)

__all__ = [
    "OptimizationPolicy",
    "EnergySchedulingPolicy",
    "ServingQueuePolicy",
    "ReplicaScalingPolicy",
    "GenAIServingPolicy",
    "GenAIServingResult",
    "PlacementPolicy",
    "AdmissionPolicy",
    "AbsoluteErrorConformalCalibrator",
    "simulate_decoupled_hybrid_abs_conformal",
    "CONFORMAL_ALPHA_MAX",
    "CONFORMAL_WARMUP",
    "CONFORMAL_WINDOW",
    "CONFORMAL_ABS_TARGET_P90_TOKENS",
    "compute_c1pgs_spot_replicas",
    "compute_frontier_rho_schedule",
    "compute_constraint_aware_schedule",
    "compute_shu_schedule",
    "compute_mcs_c_schedule",
    "compute_sotss_min_schedule",
    "ReplicaScalingConfig",
    "ReplicaScalingResult",
    "REPLICA_TTFT_BASE_S",
    "REPLICA_TPOT_S",
    "REPLICA_SAFE_GATE",
    "REPLICA_AGGRESSIVE_GATE",
    "REPLICA_MAX_ORACLE_ITERS",
    "genai_effective_service_s",
    "genai_eval_tick_timeout",
    "genai_size_for_sla",
    "genai_size_for_target",
    "GENAI_MIN_REPLICAS",
    "GENAI_SLA_LATENCY_MULT",
    "GENAI_SLA_LATENCY_ABS_S",
    "GENAI_TARGET_RHO_SLA",
    "GENAI_TARGET_RHO_UTIL",
    "GENAI_EWMA_ALPHA",
    "POLICY_REGISTRY",
    "IMPLEMENTED_POLICIES",
]
