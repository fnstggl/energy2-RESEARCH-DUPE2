"""Per-replica cache / model residency → causal serving-time payoff (PR #106).

This is the payoff channel PR #105 left as scaffolding: a request's prefix locality and a replica's
**persistent** KV / model residency now change the request's **service time** (and TTFT), which flows
through the existing cluster replay into SLA-safe goodput/$ — never through a reward bonus or a scalar
"action value". Two residency channels, same shape (residency on a replica → route to a match → avoid a
cost → lower service time), both causal:

1. **KV-prefix locality** — a request routed to a replica holding its leading prefix skips that prefill.
   ``service_factor = 1 − PREFILL_SAVINGS_FRAC · matched_prefix_fraction`` (the SAME formula the offline
   ``fleet_kv_routing`` used, now PER-REQUEST and PERSISTENT). Source: Mooncake ``hash_ids`` reuse
   (TRACE_DERIVED_REUSE_MODEL — Azure supplies arrival/tokens/timing, Mooncake supplies the reuse
   process; no row-join). Cache capacity is finite + LRU, so over-concentrating load thrashes a
   replica's cache (evictions ↑, hit rate ↓) — there is no free hit.
2. **Model-affinity** — a request whose model is not the routed replica's incurs a model-switch
   cold-start (weights reload), amortised exactly as the prior Alibaba-GenAI winner did
   (``genai_effective_service_s``): ``switch adds model_load_s`` and the replica adopts the new model
   (its KV is invalidated). Source: genai cold-start medians (BENCHMARK_DERIVED). On a single-model
   stream this channel is inert (no switches); it needs model heterogeneity (a multi-model fixture).

Routing policies (``round_robin`` / ``shortest_queue`` / ``kv_aware`` / ``affinity``) decide WHICH
replica each request hits, trading prefix/model reuse against per-replica load and the macro topology
penalty of the replica's rack. Strictly causal: a replica's residency at request *i* reflects only
requests ``< i`` (live cache state), never future requests.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from .kv_cache import StatefulKVCache

# the prefill share of service time that a prefix hit can skip (matches kv_cache.prefill_savings_frac).
PREFILL_SAVINGS_FRAC = 0.9
BLOCK_TOKENS = 16
LOOKUP_OVERHEAD_S = 0.002          # cache lookup / hashing overhead per request (small; hurts tiny reqs)
ROUTING_POLICIES = ("round_robin", "shortest_queue", "kv_aware", "affinity")


@dataclass
class RequestSig:
    """A serving request with a TRACE_DERIVED_REUSE_MODEL prefix/model signature."""
    arrival_s: float
    out_tokens: int
    in_tokens: int
    model_id: str
    hash_ids: tuple                 # block-level prefix hashes (Mooncake-derived), leading = shared prefix


@dataclass
class ReplicaResidency:
    """The live residency of one warm replica: which model + which prefix blocks it holds."""
    replica_id: str
    rack_id: str
    model_id: str
    cache: StatefulKVCache
    topology_penalty: float = 0.0   # macro rack pressure (0 = lowest)
    served: int = 0                 # requests routed here this period (load signal for shortest_queue)
    model_switches: int = 0


@dataclass
class ResidencyResult:
    """Per-request service factors + fleet diagnostics from a residency-routed period."""
    service_factor: list = field(default_factory=list)   # per-request multiplier on base service time
    ttft_factor: list = field(default_factory=list)       # per-request multiplier on prefill/TTFT
    model_switch_s: list = field(default_factory=list)    # per-request added cold-start seconds
    saved_tokens: list = field(default_factory=list)      # per-request prefill tokens skipped by a hit (PR #107)
    exact_prefix_hits: int = 0
    partial_prefix_hits: int = 0
    prefill_tokens_saved: int = 0
    reuse_depth_sum: int = 0
    evictions: int = 0
    model_switch_events: int = 0
    routed: dict = field(default_factory=dict)

    def summary(self, n: int) -> dict:
        return {"n": n, "exact_prefix_hit_rate": round(self.exact_prefix_hits / n, 4) if n else 0.0,
                "partial_prefix_hit_rate": round(self.partial_prefix_hits / n, 4) if n else 0.0,
                "mean_reuse_depth_blocks": round(self.reuse_depth_sum / n, 3) if n else 0.0,
                "prefill_tokens_saved": self.prefill_tokens_saved, "evictions": self.evictions,
                "model_switch_events": self.model_switch_events,
                "mean_service_factor": round(sum(self.service_factor) / n, 5) if n else 1.0,
                "routed": dict(self.routed)}


def build_request_signatures(recs, hash_seq, *, model_ids=("llama-8b-gqa",), model_seq=None):
    """Assign each Azure request ``(arrival, out_tok, in_tok)`` a prefix/model signature.

    TRACE_DERIVED_REUSE_MODEL: Azure supplies arrival/tokens/timing; the **prefix-reuse process** is
    Mooncake's ``hash_ids`` sequence (``hash_seq``), assigned by POSITION (request *k* → ``hash_seq[k %
    len]``) so the Mooncake reuse structure — exact/partial reuse, reuse depth, reuse distance — is
    preserved exactly, with **no row-join** (no key is matched across the two traces). ``model_seq`` (or
    a single-element ``model_ids``) assigns the model — single-model by default (the affinity channel is
    then inert); a multi-model ``model_seq`` exercises model-affinity. Deterministic."""
    sigs = []
    m = len(hash_seq) if hash_seq else 0
    for k, r in enumerate(recs):
        hid = tuple(hash_seq[k % m]) if m else ()
        if model_seq:
            mid = model_seq[k % len(model_seq)]
        else:
            mid = model_ids[0]
        sigs.append(RequestSig(arrival_s=float(r[0]), out_tokens=int(r[1]),
                               in_tokens=int(r[2]) if len(r) > 2 else int(r[1]),
                               model_id=mid, hash_ids=hid))
    return sigs


def _leading_reuse(cache: StatefulKVCache, hash_ids) -> int:
    """Contiguous leading prefix blocks already resident (no mutation) — the skippable prefill."""
    n = 0
    for b in hash_ids:
        if b in cache._lru:
            n += 1
        else:
            break
    return n


def _route(replicas, sig, policy, idx, *, w_load, w_net):
    """Pick a replica index for ``sig`` under ``policy`` (causal: sees only live cache state)."""
    n = len(replicas)
    if policy == "round_robin":
        return idx % n
    if policy == "shortest_queue":
        return min(range(n), key=lambda i: (replicas[i].served, replicas[i].topology_penalty))
    # kv_aware / affinity score: prefix reuse + model match − load − topology. Routing to a
    # model-matched replica avoids a model-switch cold-start (the affinity channel); routing to a
    # prefix-matched replica avoids prefill (the KV channel). kv_aware exploits BOTH (no new action).
    def score(i):
        rep = replicas[i]
        reuse = _leading_reuse(rep.cache, sig.hash_ids)
        model_bonus = (len(sig.hash_ids) + 1) if rep.model_id == sig.model_id else 0
        return reuse + model_bonus - w_load * rep.served - w_net * rep.topology_penalty
    return max(range(n), key=score)


def simulate_residency_serving(replicas, sigs, *, policy="kv_aware", model_load_s=22.0,
                               w_load=0.25, w_net=1.0, topology_max_discount=0.08) -> ResidencyResult:
    """Route each request over the warm replicas' PERSISTENT residency and return per-request service
    factors. Mutates each replica's cache (admits the request's blocks) and model_id (on a switch) — the
    persistence that lets prewarm/migration/placement change future periods. Causal in request order."""
    res = ResidencyResult(routed={p: 0 for p in range(len(replicas))})
    if not replicas:
        # no warm replica → every request is a full cold serve (no reuse, model load each)
        for s in sigs:
            res.service_factor.append(1.0)
            res.ttft_factor.append(1.0)
            res.model_switch_s.append(model_load_s)
            res.saved_tokens.append(0)
            res.model_switch_events += 1
        return res
    sigs = sorted(sigs, key=lambda s: s.arrival_s)
    for idx, s in enumerate(sigs):
        i = _route(replicas, s, policy, idx, w_load=w_load, w_net=w_net)
        rep = replicas[i]
        nb = len(s.hash_ids)
        # model-affinity: a mismatch reloads weights (KV invalid) — the genai cold-start channel.
        switch_s = 0.0
        if rep.model_id != s.model_id:
            switch_s = model_load_s
            rep.model_switches += 1
            res.model_switch_events += 1
            rep.cache.reset()                       # weights changed → resident KV is invalid
            rep.model_id = s.model_id
        # KV-prefix locality: leading resident blocks skip that prefill (causal: pre-admit).
        exact = _leading_reuse(rep.cache, s.hash_ids)
        prefix_frac = (exact / nb) if nb else 0.0
        saved_tokens = exact * BLOCK_TOKENS
        # topology: a hit on a higher-pressure rack is worth slightly less (macro relief only).
        topo = 1.0 - topology_max_discount * (1.0 - rep.topology_penalty)
        service_factor = (1.0 - PREFILL_SAVINGS_FRAC * prefix_frac) * topo
        # lookup overhead as a small fraction floor (hurts tiny prompts / erases marginal hits)
        service_factor = min(1.0, service_factor + LOOKUP_OVERHEAD_S)
        res.service_factor.append(round(service_factor, 6))
        res.ttft_factor.append(round(1.0 - prefix_frac, 6))
        res.model_switch_s.append(round(switch_s, 4))
        res.saved_tokens.append(saved_tokens)
        if exact > 0:
            res.exact_prefix_hits += 1
            res.prefill_tokens_saved += saved_tokens
            res.reuse_depth_sum += exact
        # partial overlap (any resident block beyond the leading run) — measured, not mutated yet
        if any(b in rep.cache._lru for b in s.hash_ids):
            res.partial_prefix_hits += 1
        before = rep.cache.evictions
        rep.cache.process(list(s.hash_ids))         # admit this request's blocks (LRU eviction)
        res.evictions += rep.cache.evictions - before
        rep.served += 1
        res.routed[i] += 1
    return res


def _copy_cache(cache: StatefulKVCache) -> StatefulKVCache:
    c = StatefulKVCache(capacity_blocks=cache.capacity_blocks, block_tokens=cache.block_tokens)
    c._lru = OrderedDict(cache._lru)            # same residency, independent for a throwaway scoring pass
    return c


def replica_residency_view(ws, *, topology_max_pressure=1.0, capacity_blocks=512, commit=True):
    """Build a routing view of the WARM replicas from the persistent world state. Each warm replica
    keeps a persistent ``StatefulKVCache`` (attached as ``_kv_cache``) so prewarm/migration/placement
    move its KV across periods. ``commit=False`` routes on COPIES of those caches (a read-only scoring
    pass — the rollout's risk/point double-eval must not pollute the persistent residency); ``commit=True``
    routes on the persistent caches and the admissions stick. Cold/migrating replicas are not servable."""
    view = []
    for rid, r in ws.replicas.items():
        if not r.warm or r.migrating:
            continue
        cache = getattr(r, "_kv_cache", None)
        if cache is None or cache.capacity_blocks != capacity_blocks:
            cache = StatefulKVCache(capacity_blocks=capacity_blocks, block_tokens=BLOCK_TOKENS)
            r._kv_cache = cache
        press = ws.racks[r.rack_id].macro_network_pressure if r.rack_id in ws.racks else 0.0
        view.append(ReplicaResidency(replica_id=rid, rack_id=r.rack_id,
                                     model_id=getattr(r, "model_id", "llama-8b-gqa"),
                                     cache=(cache if commit else _copy_cache(cache)),
                                     topology_penalty=min(1.0, press / max(topology_max_pressure, 1e-9))))
    return view


__all__ = ["RequestSig", "ReplicaResidency", "ResidencyResult", "simulate_residency_serving",
           "replica_residency_view", "build_request_signatures", "PREFILL_SAVINGS_FRAC",
           "BLOCK_TOKENS", "ROUTING_POLICIES"]
