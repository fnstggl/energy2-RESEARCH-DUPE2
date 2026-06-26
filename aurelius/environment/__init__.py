"""Aurelius canonical multi-plane environment (built from first principles).

ONE production-like training/evaluation environment over SEPARATE raw traces —
Azure serving spine (per-second) + Alibaba cluster-trace-gpu-v2026 fleet plane
(hourly) + Mooncake KV calibration + ISO electricity cost — coupled by calibrated
state variables, **never** by row-joins. Every emitted signal is provenance-tagged;
the environment is never "production grade" while any signal is below TRACE_DERIVED
or the proprietary decision-intent tier is ABSENT (pilot only).

See ``research/CANONICAL_ENVIRONMENT_PLAN.md`` for the KEEP/ADAPT/REPLACE/DELETE
audit and the architecture.
"""

from .calibration_bridge import CalibrationBridge, build_bridge
from .canonical import CanonicalMultiPlaneEnvironment, EnvironmentResult
from .cost_model import CostBreakdown, CostModel
from .fidelity_manifest import FidelityManifest
from .fleet_plane_v2026 import V2026FleetPlane
from .schemas import (
    CalibratedParam,
    EnvObservation,
    EnvStep,
    FleetState,
    ServingRequest,
    SignalProvenance,
)
from .serving_plane import KVReuseModel, ServingPlane
from .validation_suite import ValidationReport, check_distribution, run_validation

__all__ = [
    "CanonicalMultiPlaneEnvironment", "EnvironmentResult",
    "V2026FleetPlane", "ServingPlane", "KVReuseModel",
    "CalibrationBridge", "build_bridge", "CostModel", "CostBreakdown",
    "FidelityManifest", "ValidationReport", "check_distribution", "run_validation",
    "CalibratedParam", "SignalProvenance", "FleetState", "ServingRequest",
    "EnvObservation", "EnvStep",
]
