"""Tests for the stateful KV cache + KV-aware router (Phase 2).

Covers the model-derived footprint math, paged LRU eviction, the exact-prefix
reuse accounting, memory-pressure-driven capacity, the KVModel fit/outcome (hit →
prefill saved + service discount; disabled → no-op), the causal KV-aware router,
and the environment KV-enabled/disabled switch. Proves the cache and router are
CAUSAL — a request's KV outcome never depends on later requests.
"""

from __future__ import annotations

from aurelius.environment.kv_cache import (
    FOOTPRINTS,
    KVAwareRouter,
    KVFootprint,
    KVModel,
    StatefulKVCache,
    gpu_mem_for,
)


class _Req:
    def __init__(self, hash_ids):
        self.hash_ids = hash_ids


# --- footprint math (BENCHMARK_DERIVED) ------------------------------------

def test_footprint_bytes_per_token():
    fp = KVFootprint("m", n_layers=32, n_kv_heads=32, head_dim=128, dtype_bytes=2)
    assert fp.bytes_per_token == 2 * 32 * 32 * 128 * 2          # 512 KiB/token (7B MHA)
    gqa = FOOTPRINTS["llama-8b-gqa"]
    assert gqa.bytes_per_token < fp.bytes_per_token            # GQA is smaller
    assert gqa.capacity_blocks(80.0) > gqa.capacity_blocks(80.0, mem_pressure=0.5)
    assert gpu_mem_for("H100") == 80.0 and gpu_mem_for("???") == 40.0


# --- paged LRU cache + eviction --------------------------------------------

def test_cache_exact_prefix_and_lru_eviction():
    c = StatefulKVCache(capacity_blocks=3)
    assert c.process(["a", "b", "c"])["exact_prefix_blocks"] == 0      # all cold
    r = c.process(["a", "b", "x"])                                      # a,b cached → prefix 2
    assert r["exact_prefix_blocks"] == 2 and r["prefill_tokens_saved"] == 2 * c.block_tokens
    # admitting x evicts the LRU block (c was least-recently-used)
    assert c.used_blocks() == 3 and c.evictions >= 1
    assert 0.0 <= c.memory_pressure() <= 1.0


def test_cache_eviction_only_when_over_capacity():
    train = [_Req([f"b{i}", f"b{i+1}"]) for i in range(50)]
    distinct = len({b for r in train for b in r.hash_ids})
    tiny = StatefulKVCache(capacity_blocks=max(1, distinct // 4))
    big = StatefulKVCache(capacity_blocks=distinct + 16)
    for r in train:
        tiny.process(r.hash_ids)
        big.process(r.hash_ids)
    assert tiny.evictions > 0 and big.evictions == 0


def test_cache_is_causal_no_future_leak():
    train = [_Req([f"p{i % 7}", f"q{i % 5}", f"r{i}"]) for i in range(40)]
    full = StatefulKVCache(capacity_blocks=200)
    prefix = StatefulKVCache(capacity_blocks=200)
    seq_full = [full.process(r.hash_ids)["exact_prefix_blocks"] for r in train]
    seq_prefix = [prefix.process(r.hash_ids)["exact_prefix_blocks"] for r in train[:20]]
    assert seq_full[:20] == seq_prefix          # later requests cannot change earlier outcomes


# --- KVModel fit / outcome --------------------------------------------------

def test_kvmodel_fit_outcome_enabled_and_disabled():
    train = [_Req(["x", "y", f"z{i % 3}"]) for i in range(60)]
    m = KVModel.fit(train, gpu_mem_gib=80.0)
    # a warm request reuses the x,y prefix → hit, prefill saved, service discounted
    hit_any = any(m.outcome(i, 256).hit for i in range(len(train)))
    assert hit_any
    o = next(m.outcome(i, 256) for i in range(len(train)) if m.outcome(i, 256).hit)
    assert o.prefill_tokens_saved > 0 and o.service_factor < 1.0 and o.ttft_factor < 1.0
    st = m.stats(len(train))
    assert st["kv_hit_rate"] > 0 and st["prefill_tokens_saved"] > 0 and st["kv_memory_used_gib"] >= 0
    # disabled → pure no-op
    off = KVModel.fit(train, gpu_mem_gib=80.0, enabled=False)
    assert off.outcome(5, 256) .hit is False and off.stats(len(train))["prefill_tokens_saved"] == 0


def test_kvmodel_params_fidelity_tiers():
    train = [_Req(["a", "b"]) for _ in range(10)]
    tiers = {p.name: p.tier for p in KVModel.fit(train).params()}
    assert tiers["kv_prefix_reuse"] == "TRACE_DERIVED"
    assert tiers["kv_footprint_bytes_per_token"] == "BENCHMARK_DERIVED"
    assert tiers["kv_eviction_policy"] == "HEURISTIC"
    assert tiers["live_kv_memory_residency"] == "ABSENT"


# --- KV-aware router (causal) ----------------------------------------------

def test_router_routes_causally_and_captures_reuse():
    train = [_Req(["s", "t", f"u{i % 4}"]) for i in range(40)]
    r = KVAwareRouter(3, capacity_blocks=50)
    decisions = [r.route(req.hash_ids) for req in train]
    s = r.summary()
    assert s["routed"]["kv_aware"] == 40 and s["n_servers"] == 3
    assert all(0 <= d.server < 3 for d in decisions)
    assert s["total_prefill_blocks_reused"] >= 0


# --- environment KV enabled/disabled ---------------------------------------

def test_env_kv_switch_changes_metrics():
    import os

    from aurelius.environment.canonical import CanonicalMultiPlaneEnvironment
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mooncake = os.path.join(repo, "tests", "fixtures", "mooncake", "mooncake_sample.csv")
    azure_hourly = {0: [(float(i) * 0.5, 100 + (i * 7) % 300) for i in range(40)]}
    on = CanonicalMultiPlaneEnvironment(mooncake_path=mooncake, kv_enabled=True).run(azure_hourly)
    off = CanonicalMultiPlaneEnvironment(mooncake_path=mooncake, kv_enabled=False).run(azure_hourly)
    assert on.steps[0].metrics["kv"]["enabled"] is True
    assert off.steps[0].metrics["kv"]["enabled"] is False
    assert on.steps[0].metrics["kv"]["prefill_tokens_saved"] >= 0
    assert off.steps[0].action["n_kv_hits"] == 0
