"""Tests for the persistent canonical world state + the stateful actions it makes simulatable
(prewarm / placement / migration).

Honesty contract (mirrors PR #99/#100):
- each newly-CONNECTED action changes the SIMULATED reward in a causally defensible way;
- none is a free win — prewarming pays warm-hold + can be wasted; migration pays cost + cache
  penalty before any benefit; placement only discounts when locality/pressure structure exists;
- candidate evaluation never contaminates the real timeline (clone isolation / mutate=False purity);
- the state persists across periods; the chosen action — and only it — advances the real state;
- no per-link / NVLink / micro-congestion is claimed (macro topology only).
"""

from __future__ import annotations

from aurelius.environment.actions import ActionBundle
from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.world_simulator import (
    clone_world_state_for_candidate,
    initialize_world_state,
    simulate_period,
    warm_seed,
)
from aurelius.environment.world_state import (
    TRACE_DERIVED_SAMPLE,
    CanonicalWorldState,
    build_sample_cluster,
)

_FLEET = V2026FleetPlane().state_at(0)
_CM = CostModel()


def _common(**over):
    c = dict(sla_s=8.0, tick_seconds=10.0, cost_model=_CM, fleet_state=_FLEET,
             best_effort_fraction=0.0, period_hours=1.0)
    c.update(over)
    return c


def _bundle(**kw):
    return ActionBundle().with_overrides(**kw)


def _sim(ws, bundle, recs, forecast, **over):
    return simulate_period(ws, bundle, recs, forecast, base_service_factor=over.pop("bsf", 1.0),
                           replay_kwargs=bundle.replay_kwargs(), **_common(**over))


# --- world state: structure, calibration, persistence ------------------------

def test_world_state_initializes_from_v2026_sample_distributions():
    ws = build_sample_cluster(n_servers=24, n_racks=4, seed=0)
    assert isinstance(ws, CanonicalWorldState)
    assert len(ws.servers) == 24 and len(ws.racks) == 4 and ws.total_replicas() > 24
    # GPU types are drawn from the real v2026 server_hourly fractions (A10/L20/… dominate)
    types = {s.gpu_type for s in ws.servers.values()}
    assert types & {"A10", "L20", "XPU-B", "H20", "A100"}
    # every replica sits on a real server/rack (placement state exists)
    for r in ws.replicas.values():
        assert r.server_id in ws.servers and r.rack_id in ws.racks
    # honestly labelled a sample, not measured per-machine telemetry
    assert ws.fidelity["cluster"] == TRACE_DERIVED_SAMPLE


def test_world_state_is_deterministic_and_clone_is_isolated():
    a = build_sample_cluster(n_servers=16, n_racks=4, seed=1)
    b = build_sample_cluster(n_servers=16, n_racks=4, seed=1)
    assert a.summary() == b.summary()                       # seeded → reproducible
    clone = clone_world_state_for_candidate(a)
    list(clone.replicas.values())[0].warm = True
    clone.period = 99
    assert a.warm_count() == 0 and a.period == 0            # mutating the clone never touched a


def test_chosen_action_mutates_real_state_and_persists_across_periods():
    ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(ws, 6)
    recs = [(i * 0.3, 220, 100) for i in range(300)]
    fc = {"arrival_rate": 3.0, "arrival_p90": 4.0, "mean_service_s": 2.0}
    assert ws.period == 0
    _sim(ws, _bundle(prewarm_policy="conservative"), recs, fc, mutate=True)
    assert ws.period == 1                                    # the chosen action advanced the clock
    _sim(ws, _bundle(prewarm_policy="conservative"), recs, fc, mutate=True)
    assert ws.period == 2                                    # state persists & keeps advancing


def test_candidate_scoring_is_pure_read_no_contamination():
    ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(ws, 6)
    recs = [(i * 0.3, 220, 100) for i in range(300)]
    fc = {"arrival_rate": 3.0, "arrival_p90": 4.0, "mean_service_s": 2.0}
    before = (ws.period, ws.warm_count(), len(ws.migrations))
    for pol in ("aggressive", "off"):
        for mg in ("conservative", "aggressive"):
            _sim(ws, _bundle(prewarm_policy=pol, placement_policy="network_aware",
                             migration_policy=mg), recs, fc, mutate=False)
    assert (ws.period, ws.warm_count(), len(ws.migrations)) == before   # mutate=False never wrote


# --- prewarming --------------------------------------------------------------

def _heavy_recs(n=900):
    return [(i * 0.2, 240, 100) for i in range(n)]          # heavy load from t=0 (coincides w/ warm-up)


def test_prewarm_reduces_cold_starts_and_helps_under_forecast_load():
    fc = {"arrival_rate": 8.0, "arrival_p90": 11.0, "mean_service_s": 2.0}   # forecast says heavy
    recs = _heavy_recs()
    # a cluster with a SMALL warm pool → off eats cold starts; prewarming warms ahead.
    def run(pol):
        ws = initialize_world_state(n_servers=20, n_racks=4, seed=0)
        warm_seed(ws, 2)
        return _sim(ws, _bundle(prewarm_policy=pol), recs, fc, mutate=False)
    off, cons, aggr = run("off"), run("conservative"), run("aggressive")
    assert aggr.cold_start_events < cons.cold_start_events < off.cold_start_events   # fewer cold starts
    assert aggr.warm_capacity > off.warm_capacity                       # it warmed ahead
    assert aggr.goodput_per_dollar > off.goodput_per_dollar             # and it pays off here


def test_prewarm_is_not_free_when_forecast_is_wrong():
    # forecast says HEAVY, actual load is LIGHT → prewarming holds idle replicas warm = wasted.
    fc = {"arrival_rate": 15.0, "arrival_p90": 22.0, "mean_service_s": 2.0}
    light = [(i * 2.0, 240, 100) for i in range(40)]
    def run(pol):
        ws = initialize_world_state(n_servers=24, n_racks=4, seed=0)
        warm_seed(ws, 2)
        return _sim(ws, _bundle(prewarm_policy=pol), light, fc, mutate=False)
    off, aggr = run("off"), run("aggressive")
    assert aggr.warm_hold_cost > off.warm_hold_cost and aggr.warm_hold_cost > 0   # it paid warm-hold
    assert aggr.wasted_prewarm_hours > 0
    assert aggr.goodput_per_dollar < off.goodput_per_dollar            # and it lost gp/$ for nothing


def test_prewarm_uses_only_the_forecast_no_future_arrivals():
    # the warm pool is sized from the forecast dict ALONE — identical recs with the same forecast
    # give the same warm_capacity regardless of what the recs actually contain afterwards.
    ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(ws, 2)
    fc = {"arrival_rate": 6.0, "arrival_p90": 9.0, "mean_service_s": 2.0}
    a = _sim(ws, _bundle(prewarm_policy="aggressive"), _heavy_recs(400), fc, mutate=False)
    b = _sim(ws, _bundle(prewarm_policy="aggressive"), _heavy_recs(900), fc, mutate=False)
    assert a.warm_capacity == b.warm_capacity              # sizing saw only the forecast, not the load


# --- placement / topology ----------------------------------------------------

def test_placement_discounts_service_with_locality_and_changes_reward():
    # heavy, fully-warm load so cold-start noise is out of the way → the topology lever is clean.
    ws = initialize_world_state(n_servers=20, n_racks=4, seed=0)
    warm_seed(ws, 20)
    recs = _heavy_recs(1200)
    fc = {"arrival_rate": 6.0, "arrival_p90": 9.0, "mean_service_s": 2.0}
    blind = _sim(ws, _bundle(placement_policy="topology_blind"), recs, fc, mutate=False)
    aware = _sim(ws, _bundle(placement_policy="network_aware"), recs, fc, mutate=False)
    assert blind.topology_factor == 1.0                    # blind = the no-op baseline (no discount)
    assert aware.topology_factor < 1.0                     # exploiting locality/pressure discounts service
    assert aware.goodput_per_dollar != blind.goodput_per_dollar   # it changes the reward


def test_placement_discount_needs_pressure_spread_no_free_lunch():
    # a cluster whose racks all carry the SAME network pressure → nothing to exploit → no discount.
    ws = build_sample_cluster(n_servers=16, n_racks=4, seed=0)
    for rk in ws.racks.values():
        rk.macro_network_pressure = 0.3                    # flatten the topology
    warm_seed(ws, 16)
    recs = _heavy_recs(800)
    fc = {"arrival_rate": 6.0, "arrival_p90": 9.0, "mean_service_s": 2.0}
    aware = _sim(ws, _bundle(placement_policy="network_aware"), recs, fc, mutate=False)
    # network_aware's pressure-relief term vanishes; only the (smaller) locality term can apply
    assert aware.topology_factor >= 1.0 - 0.05             # bounded, no large free discount


def test_placement_is_macro_only_no_microcongestion_fields():
    # the state carries macro rack pressure only — no per-link / NVLink / congestion fields exist.
    ws = build_sample_cluster(n_servers=8, n_racks=2, seed=0)
    rack = next(iter(ws.racks.values()))
    fields = set(vars(rack))
    assert "macro_network_pressure" in fields
    assert not (fields & {"nvlink", "nvswitch", "per_link_congestion", "pfc", "ecn", "link_bw"})


# --- migration ---------------------------------------------------------------

def test_migration_costs_now_and_can_hurt_in_a_single_period():
    ws = initialize_world_state(n_servers=20, n_racks=4, seed=0)
    warm_seed(ws, 16)                                       # warm replicas exist to migrate
    recs = _heavy_recs(900)
    fc = {"arrival_rate": 6.0, "arrival_p90": 9.0, "mean_service_s": 2.0}
    off = _sim(ws, _bundle(placement_policy="network_aware", migration_policy="off"), recs, fc, mutate=False)
    aggr = _sim(ws, _bundle(placement_policy="network_aware", migration_policy="aggressive"), recs, fc, mutate=False)
    assert aggr.migrations_started > 0 and aggr.migration_cost > 0      # a move costs money
    assert aggr.service_factor > off.service_factor                    # + KV cache invalidation penalty
    assert aggr.goodput_per_dollar < off.goodput_per_dollar            # cost lands NOW → hurts this period


def test_migration_persists_across_periods():
    ws = initialize_world_state(n_servers=20, n_racks=4, seed=0)
    warm_seed(ws, 16)
    recs = _heavy_recs(600)
    fc = {"arrival_rate": 5.0, "arrival_p90": 7.0, "mean_service_s": 2.0}
    _sim(ws, _bundle(placement_policy="network_aware", migration_policy="aggressive"), recs, fc, mutate=True)
    assert any(m.status == "in_flight" for m in ws.migrations)         # the move is in flight
    assert any(r.migrating for r in ws.replicas.values())
    _sim(ws, _bundle(migration_policy="off"), recs, fc, mutate=True)   # next period it lands
    assert all(m.status == "completed" for m in ws.migrations)
    assert not any(r.migrating for r in ws.replicas.values())


# --- MPC integration: no fake knobs, no silent exclusion ---------------------

def test_controller_world_path_can_select_stateful_actions():
    from aurelius.environment.controller import ModelPredictiveEconomicController
    from aurelius.environment.forecasting import ForecastingModel, build_frames
    from aurelius.environment.training import make_world_state
    per = {p: [(p * 60 + i * 0.3, 220 + (i % 5) * 60, 100) for i in range(50 + 30 * (p % 3))]
           for p in range(40)}
    frames = build_frames(per, period_seconds=60.0, cycle_len=60)
    fm = ForecastingModel().fit(frames[:24], train_frac=0.7)
    ws = make_world_state({"n_servers": 16, "n_racks": 4, "seed": 0, "warm": 6})
    ctrl = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=_FLEET, cost_model=_CM, horizon=1, risk_weight=0.5,
        confidence_min=0.05, sla_s=8.0, period_seconds=60.0, tick_seconds=10.0,
        kv_service_factor_by_routing={"round_robin": 0.95, "kv_aware": 0.7}, sim_seconds=60.0,
        world_state=ws)
    d = ctrl.decide(frames[:24])
    # the chosen bundle exposes the stateful surfaces (they are in the searched space, not dropped)
    assert d.bundle.prewarm_policy in ("off", "conservative", "aggressive")
    assert d.bundle.placement_policy in ("topology_blind", "rack_local", "network_aware")
    assert d.bundle.migration_policy in ("off", "conservative", "aggressive")
    dd = d.to_dict()
    assert {"prewarm_policy", "placement_policy", "migration_policy"} <= set(dd)
