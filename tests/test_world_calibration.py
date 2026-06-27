"""Tests for the evidence-based world-state calibration + the regression fix it encodes.

Locks in the root-cause fix (research/WORLD_STATE_REGRESSION_ROOT_CAUSE_AUDIT.md):
- every calibrated transition has a low/base/high band + a public source + a known fidelity tier
  (never UNKNOWN); the simulator uses the band's base;
- warm-hold belongs to the PREWARM decision, not capacity — reactive (off) carries zero intentional
  warm-hold, so capacity economics are no longer inverted (gp/$ decreases with capacity on
  comfortably-served load);
- cold-start magnitude stays in the evidence band (NOT tuned to results);
- SLA risk is still respected (the cold-start ramp still makes under-provisioning miss SLA on a burst).
"""

from __future__ import annotations

from aurelius.benchmarks.srtf_serving_backtest import _service_time_s
from aurelius.environment.actions import ActionBundle
from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.world_calibration import (
    _TIERS,
    SIMULATOR_INFERENCE,
    world_calibration,
)
from aurelius.environment.world_simulator import (
    COLD_START_S,
    WARM_IDLE_TIMEOUT_S,
    initialize_world_state,
    simulate_period,
    warm_seed,
)

_FLEET = V2026FleetPlane().state_at(0)
_CM = CostModel()


def _common(**over):
    c = dict(sla_s=10.0, tick_seconds=10.0, cost_model=_CM, fleet_state=_FLEET,
             best_effort_fraction=0.0, period_hours=1.0)
    c.update(over)
    return c


def _sim(ws, bundle, recs, fc, **over):
    return simulate_period(ws, bundle, recs, fc, base_service_factor=over.pop("bsf", 1.0),
                           replay_kwargs=bundle.replay_kwargs(), **_common(**over))


# --- calibration provenance --------------------------------------------------

def test_every_calibrated_parameter_has_provenance_no_unknown():
    rep = world_calibration()
    assert rep.parameters                                    # non-empty
    for name, p in rep.parameters.items():
        assert p.fidelity in _TIERS                          # known tier, never UNKNOWN
        assert p.low <= p.base <= p.high                     # band ordered
        assert p.method and p.limitation                     # documented method + limitation
        # a measured tier must cite at least one source; pure modelling assumptions may stand alone
        if p.fidelity != SIMULATOR_INFERENCE:
            assert p.sources, f"{name} claims {p.fidelity} but cites no source"
            assert all(s.url for s in p.sources)
    assert rep.to_dict()["any_unknown_provenance"] is False


def test_simulator_uses_the_calibrated_base_values():
    rep = world_calibration()
    assert COLD_START_S == rep.base("cold_start_s")
    assert WARM_IDLE_TIMEOUT_S == rep.base("warm_idle_timeout_s")


def test_cold_start_stays_in_the_evidence_band_not_tuned_to_results():
    # the fix must NOT have quietly lowered cold-start to make results look good
    p = world_calibration().parameters["cold_start_s"]
    assert p.base == 30.0 and 8.0 <= p.base <= 60.0          # serving-startup public band
    assert WARM_IDLE_TIMEOUT_S == 300.0                      # default scale-down delay


# --- the fix: warm-hold no longer inverts capacity economics -----------------

def _served_load():
    # a load comfortably served at every capacity (no SLA pressure) → only COST should drive gp/$
    recs = [(i * 1.2, 160, 100) for i in range(120)]
    fc = {"arrival_rate": 0.8, "arrival_p90": 1.1, "mean_service_s": _service_time_s(160)}
    return recs, fc


def test_capacity_economics_not_inverted_under_reactive_prewarm():
    ws = initialize_world_state(n_servers=24, n_racks=4, seed=0)
    warm_seed(ws, 14)                                        # a carried pool larger than this load
    recs, fc = _served_load()
    gpd = []
    for m in (0.75, 1.0, 1.5):
        b = ActionBundle().with_overrides(capacity_policy="backlog_aware", capacity_multiplier=m)
        o = _sim(ws, b, recs, fc, sla_s=15.0, bsf=0.9)
        assert o.sla_violation_rate == 0.0                   # comfortably served at every capacity
        assert o.warm_hold_cost == 0.0                       # reactive off carries no warm-hold
        gpd.append(o.goodput_per_dollar)
    # gp/$ must DECREASE as capacity rises (more GPU-hours = more cost) — the inversion is gone
    assert gpd[0] > gpd[1] > gpd[2]


def test_reactive_off_has_zero_warm_hold_but_prewarm_pays():
    recs, fc = _served_load()
    ws = initialize_world_state(n_servers=24, n_racks=4, seed=0)
    warm_seed(ws, 14)
    off = _sim(ws, _bundle_off(), recs, fc, sla_s=15.0, bsf=0.9)
    assert off.warm_hold_cost == 0.0                         # reactive cooling → no intentional hold
    # a proactive prewarm that over-warms (forecast says heavy, load is light) DOES pay warm-hold
    fc_heavy = {"arrival_rate": 12.0, "arrival_p90": 18.0, "mean_service_s": 2.0}
    ws2 = initialize_world_state(n_servers=24, n_racks=4, seed=0)
    warm_seed(ws2, 2)
    aggr = _sim(ws2, ActionBundle().with_overrides(prewarm_policy="aggressive"), recs, fc_heavy,
                sla_s=15.0, bsf=0.9)
    assert aggr.warm_hold_cost > 0.0


def _bundle_off():
    return ActionBundle().with_overrides(capacity_policy="backlog_aware", capacity_multiplier=1.0)


# --- SLA risk is still respected (we did not just zero it out) ----------------

def test_under_provisioning_still_misses_sla_on_a_burst():
    # the calibrated cold-start ramp must still make low capacity miss SLA on a burst, so the risk
    # term remains meaningful (the fix did not remove SLA risk, it fixed the cost inversion).
    ws = initialize_world_state(n_servers=24, n_racks=4, seed=0)
    warm_seed(ws, 2)                                          # tiny warm pool → cold start on the burst
    recs = [(i * 0.15, 240, 100) for i in range(900)]        # heavy burst from t=0
    fc = {"arrival_rate": 9.0, "arrival_p90": 13.0, "mean_service_s": 2.0}
    lo = _sim(ws, ActionBundle().with_overrides(capacity_policy="backlog_aware", capacity_multiplier=0.75),
              recs, fc, sla_s=6.0)
    hi = _sim(ws, ActionBundle().with_overrides(capacity_policy="backlog_aware", capacity_multiplier=1.5),
              recs, fc, sla_s=6.0)
    assert lo.sla_violation_rate > hi.sla_violation_rate     # under-provisioning still costs SLA
    assert lo.sla_violation_rate > 0.05                      # and it is a real, non-trivial miss
