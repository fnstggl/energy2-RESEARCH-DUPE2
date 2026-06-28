"""TieredKVStateV2 — production-like tiered KV-cache hierarchy (V2, ports vLLM/LMCache/Mooncake/Splitwise).

Extends V1's single per-replica LRU cache (`kv_cache.StatefulKVCache`) into a four-tier hierarchy:
``GPU_HBM → CPU_DRAM → REMOTE_KV → SSD_NVME``. For every request's prefix the cache chooses the **cheapest
causal path** among {HBM hit, CPU-DRAM hit, REMOTE_KV hit, SSD hit, recompute} by net prefill saving:

    net_benefit(tier) = saved_prefill_s(prefix_len) − transfer_cost_s(prefix_bytes, tier)
    transfer_cost_s   = lookup_overhead + tier_latency + bytes / effective_bandwidth
    effective_bandwidth(REMOTE_KV) = base · (1 − network_pressure)        # congestion makes remote worse

The tier with the highest positive net benefit serves the hit; if none beats recompute (net ≤ 0), the
prefill is recomputed (saved = 0). This makes the design patterns causal and economically consequential:
HBM fastest; CPU beats remote at equal pressure; remote beats recompute for LONG prefixes; recompute beats
remote under HIGH network pressure; SSD helps only very long prefixes under low load. Ports:
vLLM PagedAttention block model, LMCache tier hierarchy + LRU, Mooncake disaggregated remote KV, Splitwise
``transfer = bytes / bandwidth``.

Eviction cascades down the tiers (HBM LRU-evicts to CPU, CPU to REMOTE, REMOTE to SSD, SSD drops) so capacity
and eviction change FUTURE hit rates. Strictly causal: a request sees only blocks admitted by earlier
requests. Cache-sharing is gated by a ``domain`` key (tenant-safe: no cross-domain reuse unless equal).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

BLOCK_TOKENS = 16

# per-tier: (effective bandwidth GB/s, fixed latency s, lookup overhead s). HBM = resident (no transfer).
TIER_SPECS = {
    "GPU_HBM":  (1e9, 0.0, 0.0005),     # resident; "bandwidth" effectively infinite (no copy)
    "CPU_DRAM": (64.0, 0.0010, 0.0010),  # PCIe gen4/5 host transfer
    "REMOTE_KV": (50.0, 0.0030, 0.0020),  # RDMA base; scaled by (1 − network_pressure)
    "SSD_NVME": (5.0, 0.0050, 0.0020),   # local NVMe
}
TIERS = ("GPU_HBM", "CPU_DRAM", "REMOTE_KV", "SSD_NVME")


@dataclass
class TierDecision:
    tier: str                       # tier that served the prefix, or "RECOMPUTE"
    saved_prefill_tokens: int
    transfer_bytes: int
    transfer_latency_s: float
    recompute_tokens: int
    net_benefit_s: float


@dataclass
class TieredKVStateV2:
    """One replica's tiered KV residency. Capacities are in blocks (16 tok). ``block_bytes`` from the model
    KV footprint (passed in by the simulator). Deterministic, clone-safe (deep-copyable)."""

    block_bytes: int = 256 * 1024
    cap_hbm: int = 512
    cap_cpu: int = 4096
    cap_remote: int = 32768
    cap_ssd: int = 131072
    network_pressure: float = 0.0       # 0..1 macro congestion (from v2026 rx/tx)
    domain: str = "default"             # cache-sharing domain (tenant-safe)
    _tiers: dict = field(default_factory=dict)
    # diagnostics
    hbm_hits: int = 0
    cpu_hits: int = 0
    remote_hits: int = 0
    ssd_hits: int = 0
    recompute_count: int = 0
    transfer_bytes_total: int = 0
    transfer_latency_total: float = 0.0
    evictions_by_tier: dict = field(default_factory=dict)
    remote_vs_recompute_remote_won: int = 0
    remote_vs_recompute_recompute_won: int = 0

    def __post_init__(self):
        if not self._tiers:
            self._tiers = {t: OrderedDict() for t in TIERS}
        if not self.evictions_by_tier:
            self.evictions_by_tier = {t: 0 for t in TIERS}

    def _cap(self, tier: str) -> int:
        return {"GPU_HBM": self.cap_hbm, "CPU_DRAM": self.cap_cpu,
                "REMOTE_KV": self.cap_remote, "SSD_NVME": self.cap_ssd}[tier]

    def _eff_bw(self, tier: str) -> float:
        bw, _, _ = TIER_SPECS[tier]
        if tier == "REMOTE_KV":
            return max(1.0, bw * (1.0 - min(0.95, self.network_pressure)))
        return bw

    def _leading_run(self, tier: str, hash_ids) -> int:
        """Contiguous leading blocks of ``hash_ids`` resident in ``tier`` (causal; no mutation)."""
        store = self._tiers[tier]
        n = 0
        for b in hash_ids:
            if (self.domain, b) in store:
                n += 1
            else:
                break
        return n

    def _transfer_cost_s(self, tier: str, blocks: int) -> tuple:
        bw, lat, lookup = TIER_SPECS[tier]
        if tier == "GPU_HBM":
            return lookup, 0  # resident: lookup only, no copy
        nbytes = blocks * self.block_bytes
        cost = lookup + lat + nbytes / (self._eff_bw(tier) * 1e9)
        return cost, nbytes

    def decide(self, hash_ids, *, prefill_s_per_token: float) -> TierDecision:
        """Pick the cheapest causal path for this prefix. ``prefill_s_per_token`` is the per-token recompute
        cost (from the roofline timing model). Does NOT mutate; call :meth:`admit` after."""
        best = TierDecision("RECOMPUTE", 0, 0, 0.0, len(hash_ids) * BLOCK_TOKENS, 0.0)
        for tier in TIERS:
            run = self._leading_run(tier, hash_ids)
            if run <= 0:
                continue
            saved_tokens = run * BLOCK_TOKENS
            cost_s, nbytes = self._transfer_cost_s(tier, run)
            net = saved_tokens * prefill_s_per_token - cost_s
            if net > best.net_benefit_s:
                best = TierDecision(tier, saved_tokens, nbytes, cost_s,
                                    (len(hash_ids) - run) * BLOCK_TOKENS, net)
        return best

    def serve(self, hash_ids, *, prefill_s_per_token: float) -> TierDecision:
        """Decide + record diagnostics + admit the request's blocks (LRU, cascading eviction)."""
        d = self.decide(hash_ids, prefill_s_per_token=prefill_s_per_token)
        # remote-vs-recompute bookkeeping (the headline tradeoff)
        remote_run = self._leading_run("REMOTE_KV", hash_ids)
        if remote_run > 0:
            if d.tier == "REMOTE_KV":
                self.remote_vs_recompute_remote_won += 1
            elif d.tier == "RECOMPUTE":
                self.remote_vs_recompute_recompute_won += 1
        if d.tier == "GPU_HBM":
            self.hbm_hits += 1
        elif d.tier == "CPU_DRAM":
            self.cpu_hits += 1
        elif d.tier == "REMOTE_KV":
            self.remote_hits += 1
        elif d.tier == "SSD_NVME":
            self.ssd_hits += 1
        else:
            self.recompute_count += 1
        self.transfer_bytes_total += d.transfer_bytes
        self.transfer_latency_total += d.transfer_latency_s
        self.admit(hash_ids)
        return d

    def admit(self, hash_ids):
        """Admit blocks into HBM (LRU); cascade evictions HBM→CPU→REMOTE→SSD→drop."""
        store = self._tiers["GPU_HBM"]
        for b in hash_ids:
            key = (self.domain, b)
            if key in store:
                store.move_to_end(key)
            else:
                store[key] = True
            while len(store) > self.cap_hbm:
                old, _ = store.popitem(last=False)
                self.evictions_by_tier["GPU_HBM"] += 1
                self._demote("CPU_DRAM", old)

    def _demote(self, tier: str, key):
        if tier is None:
            return
        store = self._tiers[tier]
        if key in store:
            store.move_to_end(key)
        else:
            store[key] = True
        nxt = {"CPU_DRAM": "REMOTE_KV", "REMOTE_KV": "SSD_NVME", "SSD_NVME": None}[tier]
        while len(store) > self._cap(tier):
            old, _ = store.popitem(last=False)
            self.evictions_by_tier[tier] += 1
            self._demote(nxt, old)

    def summary(self) -> dict:
        total = self.hbm_hits + self.cpu_hits + self.remote_hits + self.ssd_hits + self.recompute_count
        hits = total - self.recompute_count
        return {"n": total, "tier_hit_rate": round(hits / total, 4) if total else 0.0,
                "HBM_hits": self.hbm_hits, "CPU_DRAM_hits": self.cpu_hits,
                "REMOTE_KV_hits": self.remote_hits, "SSD_hits": self.ssd_hits,
                "recompute_count": self.recompute_count,
                "transfer_bytes": self.transfer_bytes_total,
                "transfer_latency_s": round(self.transfer_latency_total, 6),
                "evictions_by_tier": dict(self.evictions_by_tier),
                "remote_vs_recompute": {"remote_won": self.remote_vs_recompute_remote_won,
                                        "recompute_won": self.remote_vs_recompute_recompute_won},
                "network_pressure": round(self.network_pressure, 4),
                "cache_sharing_domain": self.domain}


__all__ = ["TieredKVStateV2", "TierDecision", "TIER_SPECS", "TIERS", "BLOCK_TOKENS"]
