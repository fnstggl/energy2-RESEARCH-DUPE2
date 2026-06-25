"""Canonical Aurelius optimizer package — the comprehensive fleet optimizer.

Single top-level seam for the unified optimizer
(``research/CANONICAL_AURELIUS_OPTIMIZER.md``,
``research/OPTIMIZER_UNIFICATION_PLAN.md``). ``AureliusOptimizer`` now holds every
implemented decision surface (energy scheduling, serving ordering, replica
capacity, placement/routing, admission) and orchestrates them via
``optimize_fleet`` against SLA-safe goodput per infrastructure dollar — it is no
longer a single energy delegate.

Note: distinct from ``aurelius.optimization`` (the energy solver package,
unchanged and pinned). This package wraps and orchestrates it; it does not
replace it.
"""

from .aurelius_optimizer import (
    CANONICAL_OBJECTIVE,
    AureliusOptimizer,
    FleetOptimizationResult,
)
from .policies import (
    IMPLEMENTED_POLICIES,
    POLICY_REGISTRY,
    AdmissionPolicy,
    EnergySchedulingPolicy,
    OptimizationPolicy,
    PlacementPolicy,
    ReplicaScalingPolicy,
    ServingQueuePolicy,
)

__all__ = [
    "AureliusOptimizer",
    "FleetOptimizationResult",
    "CANONICAL_OBJECTIVE",
    "OptimizationPolicy",
    "EnergySchedulingPolicy",
    "ServingQueuePolicy",
    "ReplicaScalingPolicy",
    "PlacementPolicy",
    "AdmissionPolicy",
    "POLICY_REGISTRY",
    "IMPLEMENTED_POLICIES",
]
