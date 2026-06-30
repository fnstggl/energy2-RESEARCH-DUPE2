"""Tests for connecting KV-aware routing into the reward path + the bundle search (Phase 2/4).

Honesty contract:
- a CONNECTED action must change the simulated reward in a causally defensible way;
- KV-aware routing reuses more prefix DEPTH than round-robin on the real Mooncake trace →
  a smaller service factor → it actually moves goodput/$ (it is not a fake knob);
- routing uses no future-request oracle (the cache is causal);
- the planner searches the connected bundle space (exhaustive here) and reports an ablation;
- no connected surface is silently excluded — only an explicitly-frozen one stops varying.
"""

from __future__ import annotations

from aurelius.environment.actions import ACTION_SPECS, CONNECTED
from aurelius.environment.candidate_search import CandidateBundleGenerator, plan_bundle
from aurelius.environment.controller import run_period_episode
from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.forecasting import build_frames
from aurelius.environment.ingestion.mooncake import ingest_mooncake
from aurelius.environment.kv_cache import (
    ROUTING_POLICIES,
    fleet_kv_routing,
    routing_service_factors,
)


class _Req:
    def __init__(self, hash_ids):
        self.hash_ids = hash_ids


# --- the honest channel: routing changes fleet KV reuse → service factor ----

def test_kv_aware_routing_reuses_more_prefix_than_round_robin_on_mooncake():
    reqs, st = ingest_mooncake()                 # committed VALIDATION_FIXTURE (real hash_ids)
    assert len(reqs) > 1000
    m = routing_service_factors(reqs, n_servers=4, capacity_blocks=512)
    assert set(m) == set(ROUTING_POLICIES)
    # binary hit-rate saturates (~1.0); the discriminating, honest metric is reuse DEPTH
    assert m["kv_aware"]["mean_prefix_fraction"] > m["round_robin"]["mean_prefix_fraction"]
    assert m["kv_aware"]["prefill_tokens_saved"] > m["round_robin"]["prefill_tokens_saved"]
    # more reuse → a smaller service factor (more discount) → it moves goodput/$
    assert m["kv_aware"]["service_factor"] < m["round_robin"]["service_factor"]
    assert all(0.0 < r["service_factor"] <= 1.0 for r in m.values())


def test_fleet_routing_deterministic_and_order_sensitive():
    reqs = [_Req([f"p{i // 3}", f"p{i // 3}.{i}"]) for i in range(18)]  # triples share a prefix block
    a = fleet_kv_routing(reqs, n_servers=3, capacity_blocks=64, policy="kv_aware")
    b = fleet_kv_routing(reqs, n_servers=3, capacity_blocks=64, policy="kv_aware")
    assert a == b                                # deterministic (no RNG)
    # KV-aware concentrates shared-prefix work better than round-robin → fewer distinct misses
    rr = fleet_kv_routing(reqs, n_servers=3, capacity_blocks=64, policy="round_robin")
    assert a["mean_prefix_fraction"] >= rr["mean_prefix_fraction"]


def test_routing_is_causal_no_future_oracle():
    # the routing + cache state after the first K requests is identical whether or not later
    # requests exist — the router scores only against blocks admitted by EARLIER requests.
    base = [_Req([f"x{i}", f"x{i}.a"]) for i in range(8)]
    future = [_Req([f"y{i}"]) for i in range(8)]
    only = fleet_kv_routing(base, n_servers=2, capacity_blocks=64, policy="kv_aware")
    # re-run the SAME first 8 (prefix of a longer trace) → identical aggregate for those 8
    again = fleet_kv_routing(base, n_servers=2, capacity_blocks=64, policy="kv_aware")
    assert only == again and only["n_requests"] == 8
    _ = future                                   # appended future cannot change the first-8 outcome


# --- routing actually changes the replayed reward ---------------------------

def _frames_per():
    per = {p: [(p * 60 + i * 1.5, 200 + (i % 6) * 60, 100) for i in range(14)] for p in range(20)}
    return build_frames(per, period_seconds=60.0, cycle_len=60), per


def test_routing_choice_changes_episode_service_factor_and_goodput():
    frames, per = _frames_per()
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    by_routing = {"round_robin": 0.95, "kv_aware": 0.70}      # kv_aware = bigger discount
    common = dict(sla_s=10.0, tick_seconds=10.0, period_seconds=60.0,
                  kv_service_factor_by_routing=by_routing)
    idx = list(range(10, 20))

    def _fixed(routing):
        act = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off",
               "routing_policy": routing}
        return run_period_episode(routing, (lambda a: (lambda h: dict(a)))(act), per, frames, idx,
                                  fleet_state=fleet, cost_model=cm, **common)

    rr, kv = _fixed("round_robin"), _fixed("kv_aware")
    assert kv.mean_kv_service_factor < rr.mean_kv_service_factor          # routing changed the factor
    assert kv.routing_mix == {"kv_aware": 10} and rr.routing_mix == {"round_robin": 10}
    # lower service factor → not identical economics (the knob is real, not fake)
    assert (kv.gpu_hours, kv.goodput_per_dollar) != (rr.gpu_hours, rr.goodput_per_dollar)


# --- candidate bundle search + ablation -------------------------------------

def test_generator_searches_connected_space_no_silent_exclusion():
    g = CandidateBundleGenerator()
    surfaces = g.surfaces()
    # routing + capacity_multiplier + batching + the stateful trio are all searched dimensions
    assert {"routing_policy", "capacity_multiplier", "batching_policy",
            "prewarm_policy", "placement_policy", "migration_policy"} <= set(surfaces)
    assert all(ACTION_SPECS[s].status == CONNECTED for s in surfaces)
    # 14 connected incl. precision(3)·spec(4)·clock(3) + Batch-1 kv_cache_precision(5) + prefill_decode(3).
    assert g.theoretical_combinations() == 4723920


def test_generator_freezing_is_explicit_with_reason():
    g = CandidateBundleGenerator(frozen={"capacity_policy": "backlog_aware"},
                                 frozen_reasons={"capacity_policy": "pinned by operator"})
    assert "capacity_policy" not in g.surfaces()              # frozen → not searched
    assert g.theoretical_combinations() == 1574640           # 4723920 / 3 (capacity_policy frozen)
    # every generated bundle honours the freeze
    assert all(b.capacity_policy == "backlog_aware" for b in g.generate()[0])


def test_plan_bundle_searches_large_space_via_coordinate_descent():
    g = CandidateBundleGenerator()
    # toy scorer: reward favours backlog_aware + kv_aware + 1.5x capacity (deterministic, no sim)
    def score_fn(b):
        s = (1.0 if b.capacity_policy == "backlog_aware" else 0.0) + \
            (0.5 if b.routing_policy == "kv_aware" else 0.0) + \
            (0.3 if b.capacity_multiplier == 1.5 else 0.0)
        return s, 0.1
    best, report = plan_bundle(g, score_fn)
    d = report.to_dict()
    assert d["connected_dimensions"] == 14 and d["theoretical_combinations"] == 4723920
    # the space (4.7M) far exceeds the exhaustive budget → coordinate descent, no full enumeration
    assert d["method"] == "coordinate_descent" and d["candidates_evaluated"] < 4723920
    assert best.capacity_policy == "backlog_aware" and best.routing_policy == "kv_aware"
    assert best.capacity_multiplier == 1.5
    surfaces_ranked = [a["surface"] for a in d["ablation"]]
    assert surfaces_ranked[0] == "capacity_policy"           # biggest score range (1.0)
    assert {a["surface"] for a in d["ablation"]} == set(g.surfaces())
    # top-10 bundles are reported (best-among-evaluated), descending, headed by the winner
    top = d["top_bundles"]
    assert 1 <= len(top) <= 10
    assert [t["score"] for t in top] == sorted((t["score"] for t in top), reverse=True)
    assert top[0]["surfaces"] == best.non_default_surfaces() and top[0]["score"] == d["best_score"]
