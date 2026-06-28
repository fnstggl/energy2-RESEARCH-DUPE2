"""Tests for the PR #105 world-model realism: replica identity, migration moving identity (not
duplicating it), cold-start decomposition, and the migration KV-preservation correction.

These guard the *benefit-channel* realism — they assert the transitions are physically conserved and
calibrated, NOT that any action becomes profitable (that is the dt=60 diagnostic's empirical question,
behind the unchanged Pareto gate)."""

from __future__ import annotations

from types import SimpleNamespace

from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.world_calibration import (
    COLD_START_COMPONENTS,
    cold_start_components,
    world_calibration,
)
from aurelius.environment.world_simulator import (
    MIGRATION_KV_PRESERVED_BY_MODE,
    initialize_world_state,
    simulate_period,
    warm_seed,
)
from aurelius.environment.world_validation import run_world_validation


def _world(n_servers=16, n_racks=4, seed=0, warm=8):
    ws = initialize_world_state(n_servers=n_servers, n_racks=n_racks, seed=seed)
    warm_seed(ws, warm)
    return ws


def _step(ws, *, migration="off", placement="topology_blind", prewarm="off", periods=1):
    fleet = V2026FleetPlane().state_at(0)
    recs = [(i * 5.0, 200, 100) for i in range(6)]
    fcast = {"arrival_rate": 0.1, "arrival_p90": 0.2, "mean_service_s": 1.0}
    pol = SimpleNamespace(prewarm_policy=prewarm, placement_policy=placement, migration_policy=migration)
    for _ in range(periods):
        simulate_period(ws, pol, recs, fcast, sla_s=10.0, tick_seconds=10.0, cost_model=CostModel(),
                        fleet_state=fleet, period_hours=1.0, dt_seconds=60.0, mutate=True)
    return ws


# --- cold-start decomposition (Phase 4) -------------------------------------

def test_cold_start_components_sum_to_aggregate_band():
    r = world_calibration()
    comp = cold_start_components(r)
    assert abs(comp["total_s"] - r.base("cold_start_s")) <= 1.0      # decomposition, not a reduction
    # model-load is the dominant component (public prior); every band is ordered
    assert comp["cold_start_model_load_s"] == max(comp[k] for k in COLD_START_COMPONENTS)
    for k in COLD_START_COMPONENTS:
        p = r.parameters[k]
        assert p.low <= p.base <= p.high


def test_cold_start_not_tuned_down():
    # the aggregate cold-start base is unchanged from the pre-PR value (30s) — decomposed, not lowered.
    assert world_calibration().base("cold_start_s") == 30.0


# --- replica identity (Phase 1) ---------------------------------------------

def test_replica_identity_persists_across_periods():
    ws = _world()
    ids0 = set(ws.replicas)
    _step(ws, periods=4)
    assert set(ws.replicas) == ids0                                  # the same replicas persist
    # a warm replica carries loaded weights + a hot cache (identity, not just a bool)
    warm = [r for r in ws.replicas.values() if r.warm]
    assert warm and all(r.weights_loaded for r in warm)


def test_migration_moves_identity_without_duplicating():
    ws = _world(warm=12)
    n0, ids0 = ws.total_replicas(), set(ws.replicas)
    racks0 = {rid: r.rack_id for rid, r in ws.replicas.items()}
    _step(ws, migration="aggressive", placement="network_aware", periods=3)
    assert ws.total_replicas() == n0 and set(ws.replicas) == ids0    # no replica created/destroyed
    moved = [rid for rid in ws.replicas if ws.replicas[rid].rack_id != racks0[rid]]
    assert moved                                                     # at least one replica relocated
    # a moved replica is the SAME object (id stable) and lands warm with weights resident
    for rid in moved:
        assert ws.replicas[rid].weights_loaded and ws.replicas[rid].warm


def test_cooled_replica_unloads_weights_and_cache():
    # a replica idle past the timeout cools: weights unloaded, cache gone (so a future use cold-starts).
    ws = _world(warm=4)
    # serve a tiny load (peak_c small) at hourly dt so idle replicas exceed the 300s timeout and cool.
    fleet = V2026FleetPlane().state_at(0)
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind", migration_policy="off")
    for _ in range(3):
        simulate_period(ws, pol, [(0.0, 100, 50)], {"arrival_rate": 0.001, "arrival_p90": 0.002,
                        "mean_service_s": 1.0}, sla_s=10.0, tick_seconds=10.0, cost_model=CostModel(),
                        fleet_state=fleet, period_hours=1.0, dt_seconds=3600.0, mutate=True)
    cold = [r for r in ws.replicas.values() if not r.warm]
    assert cold and all((not r.weights_loaded) and r.kv_warm_frac == 0.0 for r in cold)


# --- migration KV-preservation correction (Phase 1) -------------------------

def test_migration_kv_preservation_is_mode_dependent_not_a_flat_surcharge():
    cons = MIGRATION_KV_PRESERVED_BY_MODE["conservative"]
    aggr = MIGRATION_KV_PRESERVED_BY_MODE["aggressive"]
    assert 0.5 <= aggr < cons <= 1.0                                 # pipelined keeps more than bulk
    # the conservative move's service surcharge is tiny (KV preserved), not the old flat 1.04
    from aurelius.environment.world_simulator import _migration_plan
    ws = _world(warm=12)
    mg = _migration_plan(ws, "conservative", placement={})
    assert mg["n_migrations"] > 0 and 1.0 <= mg["cache_factor"] < 1.01


# --- clone isolation + determinism (carry identity) -------------------------

def test_clone_isolation_with_identity_fields():
    ws = _world(warm=8)
    clone = ws.clone()
    for r in clone.replicas.values():
        r.weights_loaded = False
        r.kv_warm_frac = 0.0
    assert any(r.weights_loaded for r in ws.replicas.values())       # original untouched
    assert all(not r.weights_loaded for r in clone.replicas.values())


def test_deterministic_world_and_migration():
    a, b = _world(seed=3, warm=10), _world(seed=3, warm=10)
    _step(a, migration="aggressive", placement="network_aware", periods=3)
    _step(b, migration="aggressive", placement="network_aware", periods=3)
    assert [r.rack_id for r in a.replicas.values()] == [r.rack_id for r in b.replicas.values()]
    assert a.warm_count() == b.warm_count()


# --- validation suite (Phase 6) ---------------------------------------------

def test_world_validation_suite_has_no_failures_and_marks_deferred_skipped():
    rep = run_world_validation()
    assert rep["counts"]["FAIL"] == 0 and rep["all_landed_pass"]
    assert rep["counts"]["PASS"] >= 12
    assert rep["counts"]["SKIPPED"] >= 4                             # deferred gaps honestly skipped
    skipped = [c for c in rep["checks"] if c["status"] == "SKIPPED"]
    assert all(c["detail"] for c in skipped)                        # every SKIP carries a reason
