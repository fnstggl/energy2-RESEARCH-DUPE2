"""Canonical Aurelius optimization policies (decision-layer seam).

Phase 1 of the canonical-optimizer unification
(``research/OPTIMIZER_UNIFICATION_PLAN.md``).

This module declares the *permanent* decision-layer policy seam that the
canonical optimizer will eventually route every decision through:

    EnergySchedulingPolicy   — implemented (Phase 1): thin delegate to JobScheduler
    ServingQueuePolicy       — NOT implemented (Phase 2)
    ReplicaScalingPolicy     — NOT implemented (Phase 2/3)
    PlacementPolicy          — NOT implemented (Phase 3)
    AdmissionPolicy          — NOT implemented (Phase 3)

ONLY ``EnergySchedulingPolicy`` is implemented in Phase 1, and it is a
behavior-preserving delegate to the existing productized energy
``JobScheduler`` (``aurelius/optimization/scheduler.py``). The other policies
are intentionally left as ``NotImplementedError`` stubs so that:

  * the long-term architecture (decision layer with pluggable policies) has a
    real, importable home from day one, and
  * nothing can *silently* route a decision through an unbuilt policy — any
    attempt fails loudly and points at the migration plan.

Phase 1 changes NO runtime behavior and touches NO serving/SRTF code.
"""

from __future__ import annotations

import abc
from typing import Optional

from ..optimization.scheduler import JobScheduler, SchedulerResult


class OptimizationPolicy(abc.ABC):
    """Base class for a canonical decision-layer policy.

    A policy is a thin strategy object the :class:`AureliusOptimizer` delegates
    to. The contract is intentionally minimal in Phase 1: ``optimize`` takes the
    decision inputs and returns a decision artifact. Each concrete policy
    documents its own return type.
    """

    #: Stable, machine-readable policy name (registry key).
    name: str = "abstract"

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        """Produce a decision for this policy's workload class."""
        raise NotImplementedError


class EnergySchedulingPolicy(OptimizationPolicy):
    """Energy batch-scheduling policy — a thin delegate to ``JobScheduler``.

    This wraps the existing productized energy optimizer WITHOUT changing its
    behavior. ``optimize`` forwards verbatim to :meth:`JobScheduler.solve` and
    returns the unchanged :class:`SchedulerResult`. The wrapped scheduler is
    exposed as :attr:`scheduler` for callers that need the underlying object.
    """

    name = "energy"

    def __init__(
        self,
        scheduler: Optional[JobScheduler] = None,
        *,
        config=None,
        **scheduler_kwargs,
    ):
        """Wrap a ``JobScheduler``.

        Args:
            scheduler: A pre-built ``JobScheduler`` to delegate to. Mutually
                exclusive with ``config``/``scheduler_kwargs``.
            config: ``OptimizationConfig`` forwarded to a new ``JobScheduler``.
            **scheduler_kwargs: Any other ``JobScheduler`` constructor kwargs
                (``sla_registry``, ``region_contexts``, ``current_states``,
                ``sla_block_on_unknown``, ``gpu_placement_scorer``,
                ``region_gpu_types``) forwarded verbatim.
        """
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
        """Delegate verbatim to ``JobScheduler.solve`` (behavior-preserving).

        All positional inputs and keyword arguments (``risk_data``, ``method``,
        ``time_limit_seconds``, ``queue_data``, ``gpu_health_data``) are forwarded
        unchanged, so the returned ``SchedulerResult`` is byte-for-byte what the
        scheduler would have produced when called directly.
        """
        return self.scheduler.solve(jobs, price_data, carbon_data, **kwargs)

    def create_baseline_schedule(self, jobs):
        """Delegate to ``JobScheduler.create_baseline_schedule`` (ASAP home)."""
        return self.scheduler.create_baseline_schedule(jobs)


class _UnimplementedPolicy(OptimizationPolicy):
    """Shared base for declared-but-not-yet-built policies (Phase >= 2)."""

    phase: str = "a later phase"

    def optimize(self, *args, **kwargs):
        raise NotImplementedError(
            f"The {self.name!r} policy is not implemented in Phase 1 of the "
            f"canonical-optimizer unification. It is reserved for {self.phase}; "
            "see research/OPTIMIZER_UNIFICATION_PLAN.md. Phase 1 deliberately "
            "ships only the energy policy and touches no serving/SRTF, "
            "placement, admission, or replica-scaling code."
        )


class ServingQueuePolicy(_UnimplementedPolicy):
    """SRPT/aging/decoupled-hybrid serving-queue discipline (Phase 2)."""

    name = "serving_queue"
    phase = "Phase 2 (serving-queue discipline)"


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


#: Registry of the canonical decision-layer policies. Phase 1 implements only
#: ``"energy"``; the rest are importable seams that raise on use.
POLICY_REGISTRY: dict[str, type[OptimizationPolicy]] = {
    EnergySchedulingPolicy.name: EnergySchedulingPolicy,
    ServingQueuePolicy.name: ServingQueuePolicy,
    ReplicaScalingPolicy.name: ReplicaScalingPolicy,
    PlacementPolicy.name: PlacementPolicy,
    AdmissionPolicy.name: AdmissionPolicy,
}

#: Policies that are actually implemented in the current phase.
IMPLEMENTED_POLICIES: frozenset[str] = frozenset({EnergySchedulingPolicy.name})

__all__ = [
    "OptimizationPolicy",
    "EnergySchedulingPolicy",
    "ServingQueuePolicy",
    "ReplicaScalingPolicy",
    "PlacementPolicy",
    "AdmissionPolicy",
    "POLICY_REGISTRY",
    "IMPLEMENTED_POLICIES",
]
