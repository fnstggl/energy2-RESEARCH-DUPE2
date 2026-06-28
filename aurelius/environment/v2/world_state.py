"""CanonicalWorldStateV2 — persistent, clone-safe fleet state for the V2 world model.

Mirrors V1's persistence/clone discipline (`world_state.CanonicalWorldState`) but carries the V2 serving
physics: a fleet of replicas each with a persistent :class:`TieredKVStateV2`, a per-rack macro network
pressure, warm/cold counts, and the economic context (gpu mix, energy price, background work for
co-location). It is deep-copyable so the MPC search clones it per candidate and never contaminates the real
timeline — exactly the V1 guarantee.

This is STATE only; the physics that evolves it lives in :mod:`world_simulator`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from ..roofline_external import ARCHS
from .tiered_kv import TieredKVStateV2


@dataclass
class ReplicaV2:
    replica_id: str
    gpu_type: str
    model: str
    rack_id: str
    warm: bool = True
    kv: TieredKVStateV2 = None


@dataclass
class CanonicalWorldStateV2:
    """Persistent V2 fleet state. Clone with :func:`clone_state_v2`."""
    period: int = 0
    gpu_type: str = "H100"
    model: str = "llama-8b-gqa"
    energy_price_per_kwh: float = 0.08
    network_pressure: float = 0.0
    background_work_gpu_seconds: float = 0.0     # real/trace-derived co-location headroom (0 ⇒ no co-loc value)
    replicas: list = field(default_factory=list)

    @property
    def n_warm(self) -> int:
        return sum(1 for r in self.replicas if r.warm)

    def to_dict(self) -> dict:
        return {"period": self.period, "n_replicas": len(self.replicas), "n_warm": self.n_warm,
                "gpu_type": self.gpu_type, "model": self.model,
                "network_pressure": round(self.network_pressure, 4),
                "background_work_gpu_seconds": round(self.background_work_gpu_seconds, 3)}


def build_fleet_v2(*, n_replicas: int = 8, gpu_type: str = "H100", model: str = "llama-8b-gqa",
                   n_racks: int = 2, network_pressure: float = 0.0, energy_price_per_kwh: float = 0.08,
                   background_work_gpu_seconds: float = 0.0, cap_hbm: int = 512) -> CanonicalWorldStateV2:
    """Deterministic V2 fleet. KV block bytes are derived from the model's public KV footprint."""
    arch = ARCHS.get(model, ARCHS["llama-8b-gqa"])
    block_bytes = arch.kv_bytes_per_token * 16
    reps = []
    for i in range(n_replicas):
        kv = TieredKVStateV2(block_bytes=block_bytes, cap_hbm=cap_hbm, network_pressure=network_pressure)
        reps.append(ReplicaV2(replica_id=f"r{i}", gpu_type=gpu_type, model=model,
                              rack_id=f"rack{i % max(1, n_racks)}", warm=True, kv=kv))
    return CanonicalWorldStateV2(gpu_type=gpu_type, model=model, energy_price_per_kwh=energy_price_per_kwh,
                                 network_pressure=network_pressure,
                                 background_work_gpu_seconds=background_work_gpu_seconds, replicas=reps)


def clone_state_v2(state: CanonicalWorldStateV2) -> CanonicalWorldStateV2:
    """Deep clone for candidate evaluation — the real timeline is never mutated by the MPC search."""
    return copy.deepcopy(state)


__all__ = ["CanonicalWorldStateV2", "ReplicaV2", "build_fleet_v2", "clone_state_v2"]
