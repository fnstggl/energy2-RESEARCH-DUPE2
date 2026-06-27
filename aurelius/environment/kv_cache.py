"""Stateful KV cache + KV-aware routing — calibrated from the Mooncake trace.

Replaces the old "KV improves latency by X%" discount with a *stateful, paged,
LRU* KV cache whose dynamics are fitted to the real Mooncake prefix-reuse trace
(``hash_ids`` = block-level prefix hashes). The cache is causal — every decision
uses only the blocks admitted by EARLIER requests, never future ones — and is
coupled to the v2026 GPU-memory calibration through its capacity (a memory budget
in KV blocks that shrinks under fleet memory pressure → more eviction → lower hit
rate).

Fidelity (per the build spec):
  * prefix/hash reuse (rate, exact vs partial, depth)  — TRACE_DERIVED (Mooncake)
  * per-model KV footprint (bytes/token, block size)   — BENCHMARK_DERIVED (public
    model architecture: 2·layers·kv_heads·head_dim·dtype)
  * cache capacity by GPU memory budget                — INFERRED (budget split)
  * eviction (LRU) + memory-pressure eviction          — HEURISTIC/INFERRED
    (LRU is a deployable default; the trace does not expose the real policy)
  * live KV memory residency                           — ABSENT (pilot telemetry)
  * cache-routing outcome                              — SIMULATED (not measured)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from .schemas import (
    ABSENT,
    BENCHMARK_DERIVED,
    HEURISTIC,
    INFERRED,
    TRACE_DERIVED,
    CalibratedParam,
)

_GIB = 1024 ** 3

# Per-GPU-type HBM capacity (GiB), public spec sheets (BENCHMARK_DERIVED). The KV
# budget is (this − model weights) · kv_fraction, shrunk by live memory pressure.
GPU_MEM_GIB = {
    "H100": 80.0, "H800": 80.0, "H20": 96.0, "A100": 80.0, "A10": 24.0,
    "A800": 80.0, "L20": 48.0, "L40S": 48.0, "L40": 48.0, "XPU-B": 64.0,
    "V100": 32.0, "T4": 16.0,
}
DEFAULT_GPU_MEM_GIB = 40.0


def gpu_mem_for(gpu_type: str) -> float:
    return GPU_MEM_GIB.get(gpu_type, DEFAULT_GPU_MEM_GIB)


# ---------------------------------------------------------------------------
# Per-model KV footprint (BENCHMARK_DERIVED from public model architecture)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVFootprint:
    """KV bytes/token = 2 (K+V) · n_layers · n_kv_heads · head_dim · dtype_bytes.

    Block (page) size follows vLLM PagedAttention (16 tokens). Numbers are derived
    from published model architecture (BENCHMARK_DERIVED), not measured residency."""

    model: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2               # fp16/bf16
    block_tokens: int = 16
    weight_gib: float = 14.0           # model weights resident (fp16) — leaves the KV budget

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    @property
    def block_bytes(self) -> int:
        return self.bytes_per_token * self.block_tokens

    def tokens_per_gib(self) -> float:
        return _GIB / self.bytes_per_token

    def capacity_blocks(self, gpu_mem_gib: float, *, kv_fraction: float = 0.9,
                        mem_pressure: float = 0.0) -> int:
        """KV blocks that fit in (GPU memory − weights) · kv_fraction, shrunk by the
        live fleet memory pressure (less free memory → smaller cache)."""
        free_gib = max(0.0, gpu_mem_gib - self.weight_gib) * kv_fraction
        free_gib *= max(0.0, 1.0 - mem_pressure)
        return max(1, int(free_gib * _GIB / self.block_bytes))


# Public architectures (BENCHMARK_DERIVED). MHA = n_kv_heads == n_heads; GQA fewer.
FOOTPRINTS = {
    "llama-7b-mha": KVFootprint("llama-7b-mha", n_layers=32, n_kv_heads=32, head_dim=128, weight_gib=14.0),
    "llama-8b-gqa": KVFootprint("llama-8b-gqa", n_layers=32, n_kv_heads=8, head_dim=128, weight_gib=16.0),
    "llama-70b-gqa": KVFootprint("llama-70b-gqa", n_layers=80, n_kv_heads=8, head_dim=128, weight_gib=140.0),
}
DEFAULT_FOOTPRINT = "llama-8b-gqa"


# ---------------------------------------------------------------------------
# Stateful paged LRU KV cache (causal)
# ---------------------------------------------------------------------------

@dataclass
class StatefulKVCache:
    """One GPU/server's paged KV cache: ``capacity_blocks`` pages, LRU eviction.

    ``process(hash_ids)`` is causal: it scores the request against blocks admitted
    by EARLIER requests, then admits this request's blocks (evicting LRU pages when
    full). Reports exact-prefix reuse (contiguous leading blocks already cached →
    the prefill that can be skipped) and partial overlap (any cached block)."""

    capacity_blocks: int
    block_tokens: int = 16
    eviction: str = "lru"
    _lru: "OrderedDict" = field(default_factory=OrderedDict)
    lookups: int = 0
    block_hits: int = 0
    block_misses: int = 0
    evictions: int = 0
    exact_prefix_hits: int = 0          # requests whose leading block was cached
    prefill_blocks_saved: int = 0       # total contiguous leading blocks reused

    def process(self, hash_ids: list) -> dict:
        self.lookups += 1
        exact_prefix = 0
        in_prefix = True
        partial = 0
        for b in hash_ids:
            present = b in self._lru
            if present:
                self.block_hits += 1
                partial += 1
                self._lru.move_to_end(b)               # LRU touch
                if in_prefix:
                    exact_prefix += 1
            else:
                self.block_misses += 1
                in_prefix = False
                self._admit(b)
        if exact_prefix:
            self.exact_prefix_hits += 1
            self.prefill_blocks_saved += exact_prefix
        return {
            "exact_prefix_blocks": exact_prefix,
            "partial_blocks": partial,
            "n_blocks": len(hash_ids),
            "prefill_tokens_saved": exact_prefix * self.block_tokens,
            "used_blocks": len(self._lru),
        }

    def _admit(self, block: str) -> None:
        if block in self._lru:
            self._lru.move_to_end(block)
            return
        while len(self._lru) >= self.capacity_blocks and self._lru:
            self._lru.popitem(last=False)              # evict LRU
            self.evictions += 1
        self._lru[block] = True

    def used_blocks(self) -> int:
        return len(self._lru)

    def memory_pressure(self) -> float:
        return min(1.0, len(self._lru) / self.capacity_blocks) if self.capacity_blocks else 1.0

    def reset(self) -> None:
        self._lru.clear()
        for a in ("lookups", "block_hits", "block_misses", "evictions",
                  "exact_prefix_hits", "prefill_blocks_saved"):
            setattr(self, a, 0)

    def to_dict(self) -> dict:
        bl = self.block_hits + self.block_misses
        return {
            "capacity_blocks": self.capacity_blocks, "used_blocks": self.used_blocks(),
            "lookups": self.lookups, "block_hit_rate": round(self.block_hits / bl, 4) if bl else 0.0,
            "exact_prefix_hit_rate": round(self.exact_prefix_hits / self.lookups, 4) if self.lookups else 0.0,
            "evictions": self.evictions, "memory_pressure": round(self.memory_pressure(), 4),
            "eviction_rate": round(self.evictions / bl, 4) if bl else 0.0,
        }


# ---------------------------------------------------------------------------
# KVModel — fit the cache on Mooncake, parameterise the serving plane
# ---------------------------------------------------------------------------

@dataclass
class KVOutcome:
    hit: bool
    prefill_tokens_saved: int
    ttft_factor: float                 # multiply prefill/TTFT by this (<=1 on a hit)
    service_factor: float              # multiply total service time by this


@dataclass
class KVModel:
    """KV behaviour fitted by replaying a real prefix-reuse trace through the cache.

    The per-request reuse OUTCOMES (exact-prefix blocks, hit/miss, warmup, eviction)
    are TRACE_DERIVED from Mooncake; applying that outcome sequence to the serving
    plane's requests is SIMULATED (no row-join — only the reuse dynamic crosses).
    Capacity is coupled to the fleet GPU-memory budget + live memory pressure."""

    footprint: KVFootprint
    prefill_savings_frac: float = 0.9   # share of a request's service that is prefill-skippable
    enabled: bool = True
    capacity_blocks: int = 0
    outcomes: list = field(default_factory=list)   # per-Mooncake-request dicts (from cache.process)
    cache_summary: dict = field(default_factory=dict)
    block_tokens: int = 16

    @classmethod
    def fit(
        cls, mooncake_requests: list, *, gpu_mem_gib: float = 80.0,
        footprint: KVFootprint | None = None, mem_pressure: float = 0.0,
        prefill_savings_frac: float = 0.9, enabled: bool = True,
    ) -> "KVModel":
        fp = footprint or FOOTPRINTS[DEFAULT_FOOTPRINT]
        cap = fp.capacity_blocks(gpu_mem_gib, mem_pressure=mem_pressure)
        cache = StatefulKVCache(capacity_blocks=cap, block_tokens=fp.block_tokens)
        outcomes = [cache.process(r.hash_ids) for r in mooncake_requests if r.hash_ids]
        return cls(footprint=fp, prefill_savings_frac=prefill_savings_frac, enabled=enabled,
                   capacity_blocks=cap, outcomes=outcomes, cache_summary=cache.to_dict(),
                   block_tokens=fp.block_tokens)

    def warm_hit_rate(self) -> float:
        return self.cache_summary.get("exact_prefix_hit_rate", 0.0)

    def outcome(self, idx: int, tokens: int) -> KVOutcome:
        """KV outcome for the serving request at position ``idx`` (causal: a function
        of the fitted reuse sequence up to its own position, never future requests)."""
        if not self.enabled or not self.outcomes:
            return KVOutcome(False, 0, 1.0, 1.0)
        o = self.outcomes[idx % len(self.outcomes)]
        exact = o["exact_prefix_blocks"]
        if exact <= 0:
            return KVOutcome(False, 0, 1.0, 1.0)
        # prefill tokens saved is the cached leading prefix, capped by the request size
        saved = min(exact * self.block_tokens, max(0, tokens))
        frac = (saved / tokens) if tokens > 0 else 0.0
        # a hit skips re-prefilling the cached prefix → TTFT/prefill shrinks by that
        # fraction of the prefill-skippable share; service time shrinks proportionally
        ttft_factor = max(0.0, 1.0 - frac)
        service_factor = max(0.0, 1.0 - self.prefill_savings_frac * frac)
        return KVOutcome(True, saved, ttft_factor, service_factor)

    def stats(self, n_requests: int, sample_tokens: int = 256) -> dict:
        """Aggregate KV metrics over ``n_requests`` serving requests (enabled/disabled)."""
        if not self.enabled or not self.outcomes:
            return {"enabled": self.enabled, "kv_hit_rate": 0.0, "prefill_tokens_saved": 0,
                    "kv_memory_used_gib": 0.0, "evictions": 0, "mean_ttft_factor": 1.0,
                    "capacity_blocks": self.capacity_blocks}
        hits = saved = 0
        ttft_sum = 0.0
        for i in range(n_requests):
            o = self.outcome(i, sample_tokens)
            hits += int(o.hit)
            saved += o.prefill_tokens_saved
            ttft_sum += o.ttft_factor
        used_blocks = self.cache_summary.get("used_blocks", 0)
        return {
            "enabled": True,
            "kv_hit_rate": round(hits / n_requests, 4) if n_requests else 0.0,
            "prefill_tokens_saved": saved,
            "kv_memory_used_gib": round(used_blocks * self.footprint.block_bytes / _GIB, 3),
            "kv_memory_capacity_gib": round(self.capacity_blocks * self.footprint.block_bytes / _GIB, 3),
            "evictions": self.cache_summary.get("evictions", 0),
            "eviction_rate": self.cache_summary.get("eviction_rate", 0.0),
            "cache_memory_pressure": self.cache_summary.get("memory_pressure", 0.0),
            "mean_ttft_factor": round(ttft_sum / n_requests, 4) if n_requests else 1.0,
            "capacity_blocks": self.capacity_blocks,
            "model": self.footprint.model,
        }

    def params(self) -> list:
        fp = self.footprint
        return [
            CalibratedParam(
                "kv_prefix_reuse", self.warm_hit_rate(), "mooncake", "trace.hash_ids",
                "stateful LRU cache replay (exact leading-prefix)", "train split",
                "mooncake", TRACE_DERIVED,
                "reuse RATE/depth are real; applying them to Azure serving is SIMULATED", True),
            CalibratedParam(
                "kv_footprint_bytes_per_token", fp.bytes_per_token, "model_architecture",
                f"{fp.model}: 2·{fp.n_layers}·{fp.n_kv_heads}·{fp.head_dim}·{fp.dtype_bytes}",
                "published architecture", "n/a", "n/a", BENCHMARK_DERIVED,
                "architecture-derived; per-deployment quantization/MQA may differ", False),
            CalibratedParam(
                "kv_cache_capacity_blocks", self.capacity_blocks, "engineering",
                "(gpu_mem_gib − weights)·kv_fraction / block_bytes", "memory-budget split",
                "n/a", "n/a", INFERRED, "budget split assumption; operator config in pilot", False),
            CalibratedParam(
                "kv_eviction_policy", self.eviction_label(), "engineering", "cache.eviction",
                "LRU (deployable default)", "n/a", "n/a", HEURISTIC,
                "trace does not expose the real eviction policy; LRU is an assumption", False),
            CalibratedParam(
                "live_kv_memory_residency", "ABSENT", "—", "—", "pilot telemetry", "n/a", "n/a",
                ABSENT, "per-instance live KV residency needs operator telemetry", False),
        ]

    def eviction_label(self) -> str:
        return "lru"


# ---------------------------------------------------------------------------
# KV-aware router (causal): route to the cache holding the most reusable prefix,
# traded off against queue delay + memory pressure + network/topology penalty.
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    server: int
    reason: str
    reuse_blocks: int
    score: float


class KVAwareRouter:
    """N server caches; route each request to the server maximising
    ``reuse_blocks − w_queue·queue − w_mem·mem_pressure − w_net·net_penalty``.

    Strictly causal: scoring sees only blocks admitted by EARLIER requests (the
    live cache state), never the request's own or any future request's blocks.
    Provides fastest-available and shortest-queue baselines for comparison."""

    def __init__(self, n_servers: int, *, capacity_blocks: int, block_tokens: int = 16,
                 net_penalty: list | None = None, w_queue: float = 0.5,
                 w_mem: float = 4.0, w_net: float = 1.0) -> None:
        self.caches = [StatefulKVCache(capacity_blocks=capacity_blocks, block_tokens=block_tokens)
                       for _ in range(n_servers)]
        self.queue = [0] * n_servers
        self.net_penalty = net_penalty or [0.0] * n_servers
        self.w_queue, self.w_mem, self.w_net = w_queue, w_mem, w_net
        self.routed = {"kv_aware": 0, "would_fastest": 0, "would_shortest_queue": 0,
                       "kv_matched_fastest": 0}

    def _reuse_blocks(self, cache: StatefulKVCache, hash_ids: list) -> int:
        """Count contiguous leading prefix blocks already resident (no mutation)."""
        n = 0
        for b in hash_ids:
            if b in cache._lru:
                n += 1
            else:
                break
        return n

    def route(self, hash_ids: list) -> RouteDecision:
        reuse = [self._reuse_blocks(c, hash_ids) for c in self.caches]
        scores = [reuse[i] - self.w_queue * self.queue[i]
                  - self.w_mem * self.caches[i].memory_pressure()
                  - self.w_net * self.net_penalty[i] for i in range(len(self.caches))]
        best = max(range(len(scores)), key=lambda i: scores[i])
        shortest_q = min(range(len(self.queue)), key=lambda i: self.queue[i])
        # "fastest available" baseline ≈ lowest queue + lowest mem pressure, KV-blind
        fastest = min(range(len(self.caches)),
                      key=lambda i: self.queue[i] + self.caches[i].memory_pressure())
        self.routed["kv_aware"] += 1
        self.routed["would_shortest_queue"] += int(best == shortest_q)
        self.routed["kv_matched_fastest"] += int(best == fastest)
        # commit: admit on the chosen server, advance its queue
        self.caches[best].process(hash_ids)
        self.queue[best] += 1
        for i in range(len(self.queue)):          # drain one unit of queue per step
            if i != best and self.queue[i] > 0:
                self.queue[i] -= 1
        return RouteDecision(server=best, reason="kv_reuse" if reuse[best] > 0 else "load_balanced",
                             reuse_blocks=reuse[best], score=scores[best])

    def summary(self) -> dict:
        total_reuse = sum(c.prefill_blocks_saved for c in self.caches)
        return {
            "n_servers": len(self.caches), "routed": dict(self.routed),
            "total_prefill_blocks_reused": total_reuse,
            "per_server": [c.to_dict() for c in self.caches],
        }


__all__ = [
    "KVFootprint", "FOOTPRINTS", "DEFAULT_FOOTPRINT", "GPU_MEM_GIB", "gpu_mem_for",
    "StatefulKVCache", "KVModel", "KVOutcome", "KVAwareRouter", "RouteDecision",
]
