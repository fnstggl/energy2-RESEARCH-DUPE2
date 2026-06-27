"""Tests for the canonical action-surface architecture (schema + registry + MPC).

Locks in the honesty contract from the Phase-1 audit
(`research/AURELIUS_ACTION_SURFACE_AUDIT.md`):

- the schema represents ALL surfaces but defaults to no-ops (an empty bundle == today);
- the registry optimizes ONLY connected surfaces by default; SIMULATED_ONLY only on opt-in;
- PLANNED / REQUIRES_PILOT_TELEMETRY surfaces are never enumerated and are rejected if set;
- no fake knob can change the scored reward (a non-connected surface never alters the
  simulator kwargs);
- the MPC controller optimizes ActionBundles, keeps the legacy dict action, and reports
  understood-but-unavailable surfaces separately;
- existing fixed policies still run.
"""

from __future__ import annotations

from aurelius.environment.action_registry import (
    enumerate_candidate_bundles,
    list_connected_actions,
    status_counts,
    validate_action_bundle,
)
from aurelius.environment.actions import (
    ACTION_SPECS,
    CONNECTED,
    CONNECTED_SURFACES,
    PLANNED,
    SIMULATED_ONLY,
    ActionBundle,
)

# --- schema -----------------------------------------------------------------

def test_default_bundle_is_all_noops_and_reproduces_today():
    b = ActionBundle()
    # every field sits at its surface's no-op default
    assert all(getattr(b, n) == ACTION_SPECS[n].default for n in ACTION_SPECS)
    assert b.non_default_surfaces() == {}
    # connected levers that are direct run_unified_replay kwargs (capacity_multiplier included)
    assert set(b.connected_kwargs()) == {"capacity", "ordering", "admission", "capacity_multiplier"}
    assert b.connected_kwargs() == {"capacity": "reactive_lag1", "ordering": "fifo",
                                    "admission": "off", "capacity_multiplier": 1.0}
    # a default bundle's full replay kwargs reproduce today's no-op behaviour
    assert b.replay_kwargs() == {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off",
                                 "capacity_multiplier": 1.0, "batch_concurrency": 1.0,
                                 "batch_service_factor": 1.0}


def test_legacy_roundtrip_and_serialization():
    act = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "class_aware"}
    b = ActionBundle.from_legacy(act)
    assert b.legacy_action() == act                       # the 3-key back-compat view
    assert ActionBundle(**b.to_dict()) == b               # to_dict round-trips
    desc = {d["field"]: d for d in b.describe()}
    assert desc["capacity_policy"]["status"] == CONNECTED and desc["capacity_policy"]["affects_reward"]
    assert desc["clock_policy"]["status"] == PLANNED and not desc["clock_policy"]["affects_reward"]


# --- registry enumeration ---------------------------------------------------

def test_status_counts_match_audit():
    counts = status_counts()
    # CONNECTED: capacity, ordering, admission, routing, capacity_multiplier, batching
    assert counts["CONNECTED"] == 6
    assert counts["SIMULATED_ONLY"] == 2                  # per-request kv-routing, topology
    assert counts.get("PLANNED", 0) + counts.get("REQUIRES_PILOT_TELEMETRY", 0) == 8
    assert {s.name for s in list_connected_actions()} == set(CONNECTED_SURFACES)


def test_enumerate_connected_only_varies_only_connected():
    bundles = enumerate_candidate_bundles(connected_only=True)
    assert len(bundles) == 324                            # 3 x 2 x 2 x 3(routing) x 3(cap_mult) x 3(batching)
    # no candidate moves a non-connected surface off its default
    for b in bundles:
        assert all(k in CONNECTED_SURFACES for k in b.non_default_surfaces())
        assert validate_action_bundle(b)["ok"]


def test_enumerate_with_simulated_opt_in():
    # opting in adds the 2 SIMULATED_ONLY surfaces (2 options each) -> 324 * 4 = 1296
    assert len(enumerate_candidate_bundles(connected_only=False)) == 1296
    # PLANNED surfaces are STILL never enumerated
    for b in enumerate_candidate_bundles(connected_only=False):
        assert b.clock_policy == "nominal" and b.precision_policy == "full" and b.migration_policy == "off"


# --- validation: planned/fake knobs rejected --------------------------------

def test_planned_surface_cannot_be_actuated():
    for field, val in [("clock_policy", "low"), ("precision_policy", "fp8"),
                       ("spec_decode_policy", "on"), ("migration_policy", "consolidate"),
                       ("placement_policy", "topology_aware")]:
        v = validate_action_bundle(ActionBundle().with_overrides(**{field: val}))
        assert not v["ok"] and any(field in p and "not actuatable" in p for p in v["problems"])


def test_invalid_option_rejected():
    assert not validate_action_bundle(ActionBundle().with_overrides(capacity_policy="warp9"))["ok"]


# --- no fake knob changes the reward ----------------------------------------

def test_non_connected_surface_never_changes_simulator_kwargs():
    base = ActionBundle()
    # flip every SIMULATED_ONLY / PLANNED surface to a non-default value (NOT routing / batching /
    # capacity_multiplier, which are now CONNECTED)
    flips = {"kv_routing_policy": "prefix_affinity", "topology_policy": "net_aware",
             "clock_policy": "high", "precision_policy": "int8", "migration_policy": "consolidate"}
    flipped = base.with_overrides(**flips)
    # none of these reach the simulator → identical replay kwargs by construction
    assert flipped.replay_kwargs() == base.replay_kwargs()
    for f in flips:
        assert not ACTION_SPECS[f].affects_reward and ACTION_SPECS[f].status in (SIMULATED_ONLY, PLANNED)


def test_no_action_spec_outside_connected_claims_reward_effect():
    for spec in ACTION_SPECS.values():
        assert spec.affects_reward == (spec.status == CONNECTED)
        if spec.status == CONNECTED:
            assert spec.reward_channel                    # every connected surface names HOW it pays out
            if spec.sim_param is not None:                # a direct run_unified_replay kwarg
                assert spec.reward_channel == "run_unified_replay"
        else:
            assert spec.sim_param is None                 # non-connected map to NO simulator kwarg


def test_routing_is_connected_via_kv_channel_not_replay_kwargs():
    spec = ACTION_SPECS["routing_policy"]
    assert spec.status == CONNECTED and spec.reward_channel == "kv_service_factor"
    assert spec.sim_param is None                         # routing is NOT a run_unified_replay kwarg
    # a routing flip changes no replay kwarg, but IS a reward-affecting connected surface
    b = ActionBundle().with_overrides(routing_policy="kv_aware")
    assert b.connected_kwargs() == ActionBundle().connected_kwargs()
    assert spec.affects_reward and "routing_policy" in dict(b.non_default_surfaces())


# --- MPC controller uses ActionBundle ---------------------------------------

def _fitted_ctrl(**kw):
    from aurelius.environment.controller import ModelPredictiveEconomicController
    from aurelius.environment.cost_model import CostModel
    from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
    from aurelius.environment.forecasting import ForecastingModel, build_frames
    per = {p: [(p * 60 + i * 2.0, 200 + (i % 7) * 50, 100) for i in range(8 + p % 5)] for p in range(40)}
    frames = build_frames(per, period_seconds=60.0, cycle_len=60)
    fm = ForecastingModel().fit(frames[:24], train_frac=0.7)
    ctrl = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=V2026FleetPlane().state_at(0), cost_model=CostModel(),
        horizon=2, period_seconds=60.0, tick_seconds=10.0, **kw)
    return ctrl, frames


def test_controller_optimizes_bundles_keeps_legacy_action_and_reports_planned():
    from aurelius.environment.controller import enumerate_actions
    ctrl, frames = _fitted_ctrl()
    d = ctrl.decide(frames[:20])
    assert d.action in enumerate_actions()                       # legacy 3-key dict preserved
    assert isinstance(d.bundle, ActionBundle)                    # full bundle attached
    assert d.bundle.legacy_action() == d.action                  # the 3-key view agrees
    # connected_kwargs is the legacy action PLUS the newly-connected capacity_multiplier knob
    assert d.action.items() <= d.bundle.connected_kwargs().items()
    assert "capacity_multiplier" in d.bundle.connected_kwargs()
    # planned/simulated surfaces are reported separately, never as the chosen action
    rep = {p["field"] for p in ctrl.understood_but_unavailable()}
    assert "clock_policy" in rep and "kv_routing_policy" in rep   # still planned/simulated
    assert "routing_policy" not in rep                            # routing is now CONNECTED
    assert d.bundle.non_default_surfaces().keys() <= set(CONNECTED_SURFACES)


def test_fixed_policy_dict_still_runs_through_controller_candidates():
    # injecting explicit legacy-dict candidates (a fixed policy) still works (back-compat)
    fixed = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off"}
    ctrl, frames = _fitted_ctrl(candidates=[fixed])
    d = ctrl.decide(frames[:20])
    assert d.action == fixed                                     # the only candidate
