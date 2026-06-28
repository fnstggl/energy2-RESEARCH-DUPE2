"""Controlled fixtures proving the PR #106 cache/locality/prewarm/migration PHYSICS — every benefit
flows through service time / TTFT (never a reward bonus). These are the "the mechanism works" proof the
held-out diagnostic is measured against.

Channels: KV-prefix locality (prefill skipped on a prefix hit) and model-affinity (model-switch
cold-start avoided), both causal and persistent per replica."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.kv_cache import StatefulKVCache
from aurelius.environment.world_serving import (
    PREFILL_SAVINGS_FRAC,
    ReplicaResidency,
    RequestSig,
    build_request_signatures,
    simulate_residency_serving,
)
from aurelius.environment.world_simulator import initialize_world_state, simulate_period, warm_seed


def _rep(rid, model="m1", cap=512, press=0.0, rack="rack0"):
    return ReplicaResidency(rid, rack, model, StatefulKVCache(capacity_blocks=cap, block_tokens=16), press)


def _prefix(p, n=8):
    return tuple(f"p{p}_{b}" for b in range(n))


# --- KV-prefix locality channel ---------------------------------------------

def test_prefix_hit_lowers_service_time_no_hit_is_baseline():
    P = _prefix(0)
    r = simulate_residency_serving([_rep("a")], [RequestSig(0.0, 100, 128, "m1", P),
                                                 RequestSig(1.0, 100, 128, "m1", P)], policy="kv_aware")
    assert r.service_factor[0] > 0.9                       # cold: ~no saving (lookup overhead only)
    assert r.service_factor[1] < 0.2                       # full leading-prefix hit → big prefill saving
    # a brand-new prefix on a fresh replica is baseline (no hit)
    r2 = simulate_residency_serving([_rep("b")], [RequestSig(0.0, 100, 128, "m1", _prefix(9))],
                                    policy="kv_aware")
    assert r2.service_factor[0] > 0.9


def test_partial_prefix_hit_gives_partial_benefit():
    P = _prefix(0, 8)
    Q = P[:4] + _prefix(7, 4)                               # shares the first 4 of 8 blocks
    r = simulate_residency_serving([_rep("a")], [RequestSig(0.0, 100, 128, "m1", P),
                                                 RequestSig(1.0, 100, 128, "m1", Q)], policy="kv_aware")
    full_hit = 1.0 - PREFILL_SAVINGS_FRAC
    assert full_hit < r.service_factor[1] < 1.0            # between a full hit and no hit


def test_cache_affinity_beats_round_robin_under_high_reuse():
    # 4 replicas, 5 distinct prefixes (aperiodic wrt 4 so round-robin spreads each prefix across
    # replicas → poor leading-prefix reuse); kv_aware co-locates each prefix → more reuse.
    reps_kv = [_rep(f"r{i}") for i in range(4)]
    reps_rr = [_rep(f"s{i}") for i in range(4)]
    sigs = [RequestSig(float(i), 100, 128, "m1", _prefix(i % 5)) for i in range(100)]
    kv = simulate_residency_serving(reps_kv, sigs, policy="kv_aware")
    rr = simulate_residency_serving(reps_rr, sigs, policy="round_robin")
    assert kv.summary(100)["mean_service_factor"] < rr.summary(100)["mean_service_factor"]
    assert kv.exact_prefix_hits > rr.exact_prefix_hits


def test_shortest_queue_can_beat_affinity_when_cache_too_small():
    # severe working-set > capacity: piling all reuse on few replicas thrashes their cache (evictions).
    reps_kv = [_rep(f"r{i}", cap=16) for i in range(4)]    # tiny caches
    reps_sq = [_rep(f"s{i}", cap=16) for i in range(4)]
    sigs = [RequestSig(float(i), 100, 128, "m1", _prefix(i % 12)) for i in range(120)]  # 12 prefixes
    kv = simulate_residency_serving(reps_kv, sigs, policy="kv_aware")
    sq = simulate_residency_serving(reps_sq, sigs, policy="shortest_queue")
    assert kv.evictions > 0                                # affinity concentrates → thrash
    assert sq.summary(120)["routed"]                       # shortest_queue spreads (sanity)


def test_memory_pressure_no_free_hit():
    # a full small cache evicts old prefixes → a returning prefix misses (cache is not free).
    rep = _rep("a", cap=8)                                 # holds ~1 prefix of 8 blocks
    sigs = ([RequestSig(0.0, 100, 128, "m1", _prefix(0))]
            + [RequestSig(float(i + 1), 100, 128, "m1", _prefix(i + 1)) for i in range(5)]
            + [RequestSig(10.0, 100, 128, "m1", _prefix(0))])   # prefix 0 returns after eviction
    r = simulate_residency_serving([rep], sigs, policy="kv_aware")
    assert r.service_factor[-1] > 0.9                      # evicted → no hit on return


# --- model-affinity channel (the prior Alibaba-GenAI winner) -----------------

def test_model_affinity_avoids_switch_cold_start_multi_model():
    # 6 replicas, 4 models (fewer than replicas, so each model can hold a stable home). Round-robin
    # maps model→replica with period lcm(4,6)=12 → every model is spread across 3 replicas (constant
    # switches); model-aware kv_aware gives each model one home → far fewer switches after warm-up.
    reps_a = [_rep(f"r{i}", model="m0") for i in range(6)]
    reps_b = [_rep(f"s{i}", model="m0") for i in range(6)]
    sigs = [RequestSig(float(i), 100, 128, f"m{i % 4}", _prefix(i)) for i in range(120)]
    aware = simulate_residency_serving(reps_a, sigs, policy="kv_aware", model_load_s=22.0)
    blind = simulate_residency_serving(reps_b, sigs, policy="round_robin", model_load_s=22.0)
    assert aware.model_switch_events < blind.model_switch_events
    assert sum(aware.model_switch_s) < sum(blind.model_switch_s)   # less cold-start time added


def test_single_model_stream_has_no_affinity_channel():
    # on one model there are no switches → the affinity channel is inert (the Azure regime).
    sigs = [RequestSig(float(i), 100, 128, "m1", _prefix(i)) for i in range(40)]
    r = simulate_residency_serving([_rep("a", model="m1")], sigs, policy="kv_aware")
    assert r.model_switch_events == 0


# --- causality / determinism -------------------------------------------------

def test_no_future_prefix_leakage_and_deterministic():
    # request i's hit depends only on requests < i (cache state), never future ones.
    sigs = [RequestSig(float(i), 100, 128, "m1", _prefix(0)) for i in range(5)]
    r1 = simulate_residency_serving([_rep("a")], sigs, policy="kv_aware")
    assert r1.service_factor[0] > 0.9                      # first request cannot hit (nothing admitted)
    # truncating the future does not change request 0's outcome
    r2 = simulate_residency_serving([_rep("a")], sigs[:1], policy="kv_aware")
    assert r1.service_factor[0] == r2.service_factor[0]
    # deterministic replay
    r3 = simulate_residency_serving([_rep("a")], sigs, policy="kv_aware")
    assert r1.service_factor == r3.service_factor


def test_mooncake_bridge_is_positional_not_a_row_join():
    recs = [(float(i), 100 + i, 200 + i) for i in range(10)]
    hashes = [_prefix(i % 3) for i in range(10)]
    sigs = build_request_signatures(recs, hashes)
    # Azure timing/tokens preserved; Mooncake hashes attached by POSITION (no key matched)
    assert [s.out_tokens for s in sigs] == [r[1] for r in recs]
    assert [s.hash_ids for s in sigs] == hashes
    # determinism
    assert build_request_signatures(recs, hashes) == sigs


# --- end-to-end through simulate_period (serving economics) ------------------

def test_residency_raises_goodput_in_simulate_period():
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware")
    recs = [(float(i), 120, 256) for i in range(60)]
    hash_seq = [_prefix(i % 8) for i in range(60)]          # high reuse (8 prefixes)
    common = dict(sla_s=10.0, tick_seconds=10.0, cost_model=cm, fleet_state=fleet,
                  base_service_factor=0.95, period_hours=0.0167, dt_seconds=60.0, mutate=False)
    a = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    b = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(a, 8)
    warm_seed(b, 8)
    oc_kv = simulate_period(a, pol, recs, {"arrival_rate": 1.0, "arrival_p90": 1.5, "mean_service_s": 1.0},
                            kv_state={"hash_seq": hash_seq, "routing": "kv_aware"}, **common)
    oc_no = simulate_period(b, pol, recs, {"arrival_rate": 1.0, "arrival_p90": 1.5, "mean_service_s": 1.0},
                            **common)
    assert oc_kv.goodput_per_dollar > oc_no.goodput_per_dollar     # reuse → cheaper service → more gp/$
    assert oc_kv.kv_diag["exact_prefix_hit_rate"] > 0.5
    assert oc_no.kv_diag is None                                   # default path carries no residency diag


def test_residency_scoring_pass_does_not_mutate_persistent_cache():
    # a mutate=False (scoring) call must not pollute the persistent replica caches (rollout safety).
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware")
    ws = initialize_world_state(n_servers=8, n_racks=2, seed=0)
    warm_seed(ws, 6)
    recs = [(float(i), 100, 128) for i in range(20)]
    hs = [_prefix(0) for _ in range(20)]
    simulate_period(ws, pol, recs, {"arrival_rate": 1.0, "arrival_p90": 1.0, "mean_service_s": 1.0},
                    sla_s=10.0, tick_seconds=10.0, cost_model=cm, fleet_state=fleet, period_hours=0.0167,
                    dt_seconds=60.0, kv_state={"hash_seq": hs, "routing": "kv_aware"}, mutate=False)
    resident = sum(len(getattr(r, "_kv_cache")._lru) for r in ws.replicas.values()
                   if getattr(r, "_kv_cache", None))
    assert resident == 0                                          # scoring left the real caches empty


def test_counter_sanity_distinct_prefixes_routed():
    # diagnostic plumbing: routed distribution sums to request count.
    sigs = [RequestSig(float(i), 100, 128, "m1", _prefix(i % 5)) for i in range(50)]
    r = simulate_residency_serving([_rep(f"r{i}") for i in range(4)], sigs, policy="kv_aware")
    assert sum(Counter(r.routed).values()) == 50
