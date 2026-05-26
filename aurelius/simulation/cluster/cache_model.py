"""Explicit, mutable KV-cache / prefix-affinity / locality state models.

These are the first-class simulator states required by the KV-cache realism
upgrade. They are mutable (updated each tick by the engine) and intentionally
separate from the frozen canonical ClusterState. A single composite
``WorkloadCacheState`` aggregates them and is attached to each SimWorkload.

All values are bounded proxies, not a serving-engine simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KVCacheState:
    """KV-cache memory occupancy for a workload (PagedAttention-style)."""
    allocated_bytes: float = 0.0
    reserved_budget_bytes: float = 0.0
    occupancy_frac: float = 0.0          # allocated / total GPU mem (telemetry-facing)
    batch_size: float = 0.0
    avg_seq_len: float = 0.0


@dataclass
class KVPressureState:
    """KV pressure (allocated / reserved budget) and its operational region."""
    pressure: float = 0.0                # 0..1.5 (clamped); >1 = over-demand
    region: str = "low"                  # low | elevated | throttling_risk | preemption


@dataclass
class PrefixCacheState:
    """Prefix-cache reuse for a workload."""
    overlap: float = 0.5                 # shared-prefix overlap (workload property)
    hit_rate: float = 0.0                # sigmoid(overlap)·locality
    shared_prefix_tokens: float = 0.0    # length of the shared prefix (tokens)
    prefill_savings_frac: float = 0.0    # fraction of prefill removed by reuse


@dataclass
class CacheWarmupState:
    """Reuse-driven cache warmup (NOT purely time-driven)."""
    warm: bool = False
    ticks_warm: int = 0                  # consecutive ticks of sustained reuse


@dataclass
class LocalityConfidenceState:
    """Confidence that the route preserves cache locality (warmup/decay)."""
    confidence: float = 0.5


@dataclass
class RoutingLocalityState:
    """Routing-locality bookkeeping for affinity-aware decisions."""
    home_region: Optional[str] = None    # region where the cache is warm
    home_gpu_ids: tuple[str, ...] = ()   # GPUs holding the warm cache
    affinity_score: float = 1.0          # 1 = on home/warm route, lower = strayed
    telemetry_tier: str = "high"         # high | medium | low (cache visibility)


@dataclass
class CacheFragmentationState:
    """PagedAttention internal block slack (NOT heap fragmentation)."""
    slack_bytes: float = 0.0
    slack_frac: float = 0.0


@dataclass
class CacheEvictionState:
    """LRU / pressure-driven cache eviction bookkeeping."""
    cumulative_evictions: int = 0
    last_tick_evictions: int = 0


@dataclass
class PreemptionState:
    """Preemption / recompute under KV exhaustion."""
    cumulative_count: int = 0
    last_tick_count: int = 0
    recompute_penalty_ms: float = 0.0    # pending recompute cost feeding TTFT


@dataclass
class CacheAffinityState:
    """Cold-reroute bookkeeping and the pending cold-route TTFT penalty."""
    cold_reroute_count: int = 0
    cold_route_penalty_ms: float = 0.0   # pending penalty, decays over warmup
    cold_warmup_ticks_remaining: int = 0


@dataclass
class WorkloadCacheState:
    """Composite per-workload cache/affinity state (all sub-states above).

    Also carries cross-tick scheduling memory: ``active_seqs_prev`` lets the
    engine compute this tick's KV pressure from last tick's offered concurrency,
    keeping the pre-queue / post-queue update order deterministic.
    """
    kv: KVCacheState = field(default_factory=KVCacheState)
    pressure: KVPressureState = field(default_factory=KVPressureState)
    prefix: PrefixCacheState = field(default_factory=PrefixCacheState)
    warmup: CacheWarmupState = field(default_factory=CacheWarmupState)
    locality: LocalityConfidenceState = field(default_factory=LocalityConfidenceState)
    routing: RoutingLocalityState = field(default_factory=RoutingLocalityState)
    fragmentation: CacheFragmentationState = field(default_factory=CacheFragmentationState)
    eviction: CacheEvictionState = field(default_factory=CacheEvictionState)
    preemption: PreemptionState = field(default_factory=PreemptionState)
    affinity: CacheAffinityState = field(default_factory=CacheAffinityState)

    active_seqs_prev: float = 0.0
