"""Scenario forecaster + ensemble/oracle planning tests (small PR — a focused few)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aurelius.environment.scenario_forecaster import SCENARIOS, build_scenarios


def _pt(**kw):
    return SimpleNamespace(**kw)


def test_build_scenarios_includes_sla_pressure_futures():
    ar = _pt(mean=10.0, p90=18.0, p10=4.0)
    tm = _pt(mean=100.0, p90=300.0)
    tp = _pt(value=400.0, p99=1200.0)
    cv = _pt(mean=1.0, p90=2.0)
    scs = build_scenarios(ar, tm, tp, cv, prompt_tokens=800)
    by = {s["label"]: s for s in scs}
    assert {"base", "burst", "long_output", "long_prompt", "tight_sla", "calm"} <= set(by)
    assert by["burst"]["arrival_rate"] > by["base"]["arrival_rate"]        # burst stresses arrival
    assert by["tight_sla"]["arrival_rate"] > by["base"]["arrival_rate"]    # …and the combined-stress future
    assert by["long_output"]["tm"] > by["base"]["tm"]                      # decode-heavy
    assert by["long_prompt"]["prompt_mult"] > 1.0                          # prefill-heavy
    assert by["calm"]["arrival_rate"] < by["base"]["arrival_rate"]         # low load
    assert by["base"]["weight"] == max(s["weight"] for s in scs)           # base is the dominant prior


def test_build_scenarios_is_robust_to_missing_percentiles():
    # a point object carrying only a mean must still yield every scenario (falls back to the mean/value)
    bare = _pt(mean=5.0)
    scs = build_scenarios(bare, bare, _pt(value=50.0), bare, prompt_tokens=None)
    assert len(scs) == len(SCENARIOS) and all(s["arrival_rate"] == 5.0 for s in scs)


def test_scenario_and_oracle_planning_off_by_default():
    from aurelius.environment.controller import ModelPredictiveEconomicController
    flds = ModelPredictiveEconomicController.__dataclass_fields__
    assert flds["planning_scenarios"].default is False
    assert flds["planning_oracle_records"].default is None


def test_ensemble_planning_decision_runs_and_is_valid():
    """Smoke: a controller with the scenario ensemble produces a valid bundle (the path is exercised)."""
    from aurelius.environment.action_registry import validate_action_bundle
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import build_mpc_inputs, make_world_state, train_forecasters
    inp = build_mpc_inputs(hourly_stride=96, sim_seconds=180.0, use_world_state=True, control_dt_seconds=60.0)
    if inp is None:
        pytest.skip("no Azure serving data available")
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    fm, _ = train_forecasters(frames, len(frames) - 20)
    ctrl = build_controller(fm, fleet, cm, {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15},
                            common, world_state=make_world_state(common.get("world_state_params")))
    ctrl.horizon_steps = 1
    ctrl.planning_kv_cost_mode = "hybrid_capacity_work"
    ctrl.planning_prompt_tokens = 800
    ctrl.planning_scenarios = True
    d = ctrl.decide(frames[: len(frames) - 1])
    assert validate_action_bundle(d.bundle)["ok"]
