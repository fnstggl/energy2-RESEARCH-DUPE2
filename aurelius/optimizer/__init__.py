"""Canonical Aurelius optimizer package (Phase 1).

Top-level seam for the long-term unified optimizer
(``research/CANONICAL_AURELIUS_OPTIMIZER.md``). Phase 1 ships only a thin,
behavior-preserving delegate to the existing energy ``JobScheduler``; see
``research/OPTIMIZER_UNIFICATION_PLAN.md``.

Note: distinct from ``aurelius.optimization`` (the existing energy solver
package, unchanged). This package wraps it; it does not replace it.
"""

from .aurelius_optimizer import AureliusOptimizer
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
from .replay_harness import ReplayHarness, ReplayHarnessConfig, ReplayHarnessError
from .replay_result import (
    BENCHMARK_IDS,
    ReplayEvaluationResult,
    from_backtest_policy_result,
    from_canonical_policy_metrics,
    from_genai_policy_result,
    from_srtf_sim_dict,
)

__all__ = [
    "AureliusOptimizer",
    "OptimizationPolicy",
    "EnergySchedulingPolicy",
    "ServingQueuePolicy",
    "ReplicaScalingPolicy",
    "PlacementPolicy",
    "AdmissionPolicy",
    "POLICY_REGISTRY",
    "IMPLEMENTED_POLICIES",
    "ReplayEvaluationResult",
    "BENCHMARK_IDS",
    "from_backtest_policy_result",
    "from_genai_policy_result",
    "from_canonical_policy_metrics",
    "from_srtf_sim_dict",
    # Phase 1b-A: unified replay harness
    "ReplayHarness",
    "ReplayHarnessConfig",
    "ReplayHarnessError",
]
