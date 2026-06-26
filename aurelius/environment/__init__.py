"""Canonical multi-plane environment — one production-like training/eval env over
separate raw traces (Azure serving spine + v2026 fleet plane + electricity cost),
coupled by calibrated state variables, never by row-joins.

PR-1 scaffold (see ``research/CANONICAL_ENVIRONMENT_PLAN.md``): the API + the
seams made explicit. Composes the serving sim + economics + the v2026 class-mix
hook; fences the heavy cluster-engine fusion (PR-5). Every field fidelity-tagged;
honest framing preserved — this is production-LIKE, not production telemetry.
"""

from .canonical import (
    CanonicalMultiPlaneEnvironment,
    CostBreakdown,
    CostModel,
    EnvironmentResult,
    FidelityManifest,
    FleetPlane,
    FleetState,
    ServingPlane,
)
from .validation import DistributionMatch, match_distribution

__all__ = [
    "CanonicalMultiPlaneEnvironment", "FleetPlane", "FleetState", "ServingPlane",
    "CostModel", "CostBreakdown", "FidelityManifest", "EnvironmentResult",
    "DistributionMatch", "match_distribution",
]
