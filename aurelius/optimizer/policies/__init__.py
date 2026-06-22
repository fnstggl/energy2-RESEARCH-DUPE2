"""Canonical Aurelius optimization policies (decision-layer seam).

Phase 1 stood up this seam with the energy policy as a thin delegate to
``JobScheduler``; Phase 2 implements ``ServingQueuePolicy`` by extracting the
strongest validated serving-queue discipline out of the benchmark monolith into
:mod:`aurelius.optimizer.policies.serving_queue`.

    EnergySchedulingPolicy   — implemented (Phase 1): thin delegate to JobScheduler
    ServingQueuePolicy       — implemented (Phase 2): extracted abs-conformal SRPT
    ReplicaScalingPolicy     — NOT implemented (Phase 2/3)
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


class ReplicaScalingPolicy(_UnimplementedPolicy):
    """Replica/autoscaling provisioning policy (Phase 2/3)."""

    name = "replica_scaling"
    phase = "Phase 2/3 (replica scaling)"


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
    PlacementPolicy.name: PlacementPolicy,
    AdmissionPolicy.name: AdmissionPolicy,
}

#: Policies that are actually implemented in the current phase.
IMPLEMENTED_POLICIES: frozenset[str] = frozenset(
    {EnergySchedulingPolicy.name, ServingQueuePolicy.name}
)

__all__ = [
    "OptimizationPolicy",
    "EnergySchedulingPolicy",
    "ServingQueuePolicy",
    "ReplicaScalingPolicy",
    "PlacementPolicy",
    "AdmissionPolicy",
    "AbsoluteErrorConformalCalibrator",
    "simulate_decoupled_hybrid_abs_conformal",
    "CONFORMAL_ALPHA_MAX",
    "CONFORMAL_WARMUP",
    "CONFORMAL_WINDOW",
    "CONFORMAL_ABS_TARGET_P90_TOKENS",
    "POLICY_REGISTRY",
    "IMPLEMENTED_POLICIES",
]
