"""AureliusOptimizer — the canonical top-level optimization interface (Phase 1).

This is the thinnest safe first version of the canonical optimizer described in
``research/CANONICAL_AURELIUS_OPTIMIZER.md``. It establishes the permanent
top-level seam through which all future Aurelius optimization gains will
eventually flow:

    AureliusOptimizer
    └── DecisionLayer (pluggable policies)
        ├── EnergySchedulingPolicy   ← implemented in Phase 1 (delegate)
        ├── ServingQueuePolicy       ← Phase 2
        ├── ReplicaScalingPolicy     ← Phase 2/3
        ├── PlacementPolicy          ← Phase 3
        └── AdmissionPolicy          ← Phase 3

The Forecast / Constraint / Objective / Replay / Evaluation layers from the
target architecture are NOT built here — Phase 1 only stands up the decision
layer with its single implemented policy.

Phase 1 contract (verified by ``tests/test_canonical_optimizer_parity.py`` and
``research/results/canonical_optimizer_phase1_parity_2026-06-22.md``):

  * ``AureliusOptimizer(...).optimize(...)`` delegates verbatim to the existing
    productized ``JobScheduler.solve(...)`` and returns the unchanged
    ``SchedulerResult``.
  * NO runtime behavior changes; the energy core is untouched.
  * NO serving/SRTF, placement, admission, or replica-scaling code is added —
    those policies raise ``NotImplementedError``.

Example (identical result to constructing ``JobScheduler`` directly)::

    from aurelius.optimizer import AureliusOptimizer
    opt = AureliusOptimizer(config)                 # default policy = "energy"
    result = opt.optimize(jobs, price_data, carbon_data, method="greedy")
    # result is exactly JobScheduler(config).solve(jobs, price_data, carbon_data,
    #                                              method="greedy")
"""

from __future__ import annotations

from typing import Optional

from ..optimization.scheduler import JobScheduler, SchedulerResult
from .policies import (
    POLICY_REGISTRY,
    EnergySchedulingPolicy,
    OptimizationPolicy,
)


class AureliusOptimizer:
    """Canonical optimization facade. Phase 1 = thin energy delegate.

    The optimizer holds exactly one active decision-layer policy. In Phase 1 the
    default (and only implemented) policy is ``"energy"``, which delegates to
    the existing ``JobScheduler`` without changing behavior.
    """

    #: The decision-layer policy active by default in Phase 1.
    DEFAULT_POLICY: str = "energy"

    def __init__(
        self,
        config=None,
        *,
        policy: str = DEFAULT_POLICY,
        scheduler: Optional[JobScheduler] = None,
        **scheduler_kwargs,
    ):
        """Construct the canonical optimizer.

        Args:
            config: ``OptimizationConfig`` forwarded to a new ``JobScheduler``
                when the energy policy builds its own scheduler.
            policy: Decision-layer policy name. Phase 1 implements only
                ``"energy"``; any other *known* policy is constructed but will
                raise ``NotImplementedError`` on use; an unknown name raises
                ``ValueError``.
            scheduler: Optional pre-built ``JobScheduler`` to delegate to
                (energy policy only). Mutually exclusive with
                ``config``/``scheduler_kwargs``.
            **scheduler_kwargs: Additional ``JobScheduler`` constructor kwargs
                forwarded verbatim (energy policy only).
        """
        if policy not in POLICY_REGISTRY:
            raise ValueError(
                f"Unknown policy {policy!r}. Known policies: "
                f"{sorted(POLICY_REGISTRY)}."
            )

        self.policy_name: str = policy

        if policy == EnergySchedulingPolicy.name:
            if scheduler is not None:
                if config is not None or scheduler_kwargs:
                    raise ValueError(
                        "AureliusOptimizer: pass either a prebuilt `scheduler` "
                        "or `config`/constructor kwargs, not both."
                    )
                self._policy: OptimizationPolicy = EnergySchedulingPolicy(
                    scheduler=scheduler
                )
            else:
                self._policy = EnergySchedulingPolicy(
                    config=config, **scheduler_kwargs
                )
        else:
            # Declared-but-unimplemented policy seam (Phase >= 2). Constructing
            # is allowed (so the architecture is importable); using it raises.
            if scheduler is not None or scheduler_kwargs:
                raise ValueError(
                    f"Policy {policy!r} is not implemented in Phase 1 and takes "
                    "no scheduler/constructor arguments."
                )
            self._policy = POLICY_REGISTRY[policy]()

    # ------------------------------------------------------------------
    # Canonical interface
    # ------------------------------------------------------------------

    def optimize(self, *args, **kwargs):
        """Run the active decision-layer policy.

        For the Phase 1 energy policy this delegates verbatim to
        ``JobScheduler.solve`` and returns the unchanged ``SchedulerResult``.
        """
        return self._policy.optimize(*args, **kwargs)

    def create_baseline_schedule(self, jobs):
        """Energy-policy convenience: ASAP/home baseline via the wrapped engine.

        Delegates to ``JobScheduler.create_baseline_schedule``. Only meaningful
        for the energy policy.
        """
        baseline = getattr(self._policy, "create_baseline_schedule", None)
        if baseline is None:
            raise NotImplementedError(
                f"create_baseline_schedule is not available for policy "
                f"{self.policy_name!r}."
            )
        return baseline(jobs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def policy(self) -> OptimizationPolicy:
        """The active decision-layer policy object."""
        return self._policy

    @property
    def scheduler(self) -> Optional[JobScheduler]:
        """The wrapped ``JobScheduler`` (energy policy), else ``None``."""
        return getattr(self._policy, "scheduler", None)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AureliusOptimizer(policy={self.policy_name!r})"


__all__ = ["AureliusOptimizer", "SchedulerResult"]
