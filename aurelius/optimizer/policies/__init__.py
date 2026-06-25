"""Canonical Aurelius optimization policies (decision-layer seam).

Phase 1 stood up this seam with the energy policy as a thin delegate to
``JobScheduler``; Phase 2 implements ``ServingQueuePolicy`` by extracting the
strongest validated serving-queue discipline out of the benchmark monolith into
:mod:`aurelius.optimizer.policies.serving_queue`; Phase 2/3 implements
``ReplicaScalingPolicy`` by extracting the per-tick provisioning logic (AMCSG
MCS gate sweep and SOTSS-MIN oracle loop) into
:mod:`aurelius.optimizer.policies.replica_scaling`; Phase 3d implements
``GenAIServingPolicy`` by extracting the multi-model GenAI constraint_aware
replica-sizing decision (EWMA anticipatory + model-affinity cold-start routing)
from the genai_backtest benchmark monolith into
:mod:`aurelius.optimizer.policies.genai_serving`.

    EnergySchedulingPolicy   — implemented (Phase 1): thin delegate to JobScheduler
    ServingQueuePolicy       — implemented (Phase 2): extracted abs-conformal SRPT
    ReplicaScalingPolicy     — implemented (Phase 2/3): extracted AMCSG/SOTSS-MIN
    GenAIServingPolicy       — implemented (Phase 3d): extracted constraint_aware GenAI
    PlacementPolicy          — NOT implemented (Phase 3)
    AdmissionPolicy          — NOT implemented (Phase 3)

The not-yet-built policies remain importable stubs that raise
``NotImplementedError`` so nothing can silently route a decision through an
unbuilt policy.
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

    def optimize(self, jobs, price_data, carbon_data, **kwargs) -> SchedulerResult:
        """Delegate verbatim to ``JobScheduler.solve`` (behavior-preserving)."""
        return self.scheduler.solve(jobs, price_data, carbon_data, **kwargs)

    def create_baseline_schedule(self, jobs):
        """Delegate to ``JobScheduler.create_baseline_schedule`` (ASAP home)."""
        return self.scheduler.create_baseline_schedule(jobs)


class _UnimplementedPolicy(OptimizationPolicy):
    """Shared base for declared-but-not-yet-built policies (Phase >= 2)."""

    phase: str = "a later phase"

    def optimize(self, *args, **kwargs):
        raise NotImplementedError(
            f"The {self.name!r} policy is not implemented yet. It is reserved "
            f"for {self.phase}; see research/OPTIMIZER_UNIFICATION_PLAN.md. "
            "Nothing routes a decision through an unbuilt policy."
        )


class PlacementPolicy(_UnimplementedPolicy):
    """GPU/region placement-and-routing policy (Phase 3)."""

    name = "placement"
    phase = "Phase 3 (placement / routing)"


class AdmissionPolicy(_UnimplementedPolicy):
    """Flow-control admission policy (Phase 3)."""

    name = "admission"
    phase = "Phase 3 (admission control)"


#: Registry of the canonical decision-layer policies.
POLICY_REGISTRY: dict[str, type[OptimizationPolicy]] = {
    EnergySchedulingPolicy.name: EnergySchedulingPolicy,
    ServingQueuePolicy.name: ServingQueuePolicy,
    ReplicaScalingPolicy.name: ReplicaScalingPolicy,
    GenAIServingPolicy.name: GenAIServingPolicy,
    PlacementPolicy.name: PlacementPolicy,
    AdmissionPolicy.name: AdmissionPolicy,
}

#: Policies that are actually implemented in the current phase.
IMPLEMENTED_POLICIES: frozenset[str] = frozenset(
    {
        EnergySchedulingPolicy.name,
        ServingQueuePolicy.name,
        ReplicaScalingPolicy.name,
        GenAIServingPolicy.name,
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
