"""Aurelius serving world-model **V2** — production-like serving physics, built *beside* V1.

This subpackage ports the strongest external-derived serving physics from the PR #110 audit into a live,
integrated, clone-safe world model. It does NOT replace the canonical V1 world model
(`aurelius.environment.world_state` / `world_simulator`) — V1 remains the default and is unchanged. V2
shares only read-only primitives with V1 (`roofline_external`, `cost_model`, `schemas` fidelity tiers).

Components:
  * :class:`RooflineServingModelV2`  — live FLOP/bandwidth timing + precision/spec-decode/clock physics
  * :class:`TieredKVStateV2`         — GPU_HBM→CPU_DRAM→REMOTE_KV→SSD tiers + remote-vs-recompute
  * :class:`PrefillDecodeSchedulerV2`— shared/disaggregated pools + phase queues + continuous batching
  * :class:`CanonicalWorldStateV2`   — persistent, clone-safe fleet state
  * :class:`WorldSimulatorV2`        — the integration spine (everything → goodput/$)
  * :class:`AdaptiveMPCSearchV2`     — exhaustive/beam/coordinate search + regret audit

Hard law (see research/FULL_SERVING_PHYSICS_INTEGRATION_PLAN.md): every mechanism affects reward ONLY
through TTFT / completion latency / queueing / GPU-seconds / energy / power / memory pressure / bandwidth
pressure / capacity / SLA / cost. No reward bonus, no action scalar, no "roofline bonus".
"""

from __future__ import annotations

__all__ = [
    "RooflineServingModelV2",
    "TieredKVStateV2",
    "PrefillDecodeSchedulerV2",
    "CanonicalWorldStateV2",
    "WorldSimulatorV2",
    "AdaptiveMPCSearchV2",
]
