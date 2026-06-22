"""Base type for canonical decision-layer policies.

Separated into its own module so policy submodules (e.g. ``serving_queue``) can
import the base class without a circular dependency on the package ``__init__``.
"""

from __future__ import annotations

import abc


class OptimizationPolicy(abc.ABC):
    """Base class for a canonical decision-layer policy.

    A policy is a thin strategy object the :class:`AureliusOptimizer` delegates
    to. The contract is intentionally minimal: ``optimize`` takes the decision
    inputs and returns a decision artifact. Each concrete policy documents its
    own input/return types.
    """

    #: Stable, machine-readable policy name (registry key).
    name: str = "abstract"

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        """Produce a decision for this policy's workload class."""
        raise NotImplementedError
