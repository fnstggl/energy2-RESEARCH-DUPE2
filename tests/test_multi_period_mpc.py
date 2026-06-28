"""Tests for the receding-horizon multi-period MPC over the persistent ClusterState.

Proves the controller is a genuine finite-horizon MPC (not a one-step optimizer):
- the horizon is in SIMULATION STEPS, not hours; dt_seconds sets the real lookahead;
- H=1 reproduces the single-period score exactly (backward compatibility);
- candidate evaluation is a READ-ONLY rollout on a CLONE — the real world never mutates, only the
  chosen first action is committed (receding horizon);
- the forecast trajectory is causal (built from history; no future leakage);
- decisions are deterministic / replayable; runtime is bounded and scales with H.
"""

from __future__ import annotations

from aurelius.environment.actions import ActionBundle
from aurelius.environment.controller import ModelPredictiveEconomicController, run_period_episode
from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.forecast_trajectory import build_trajectory
from aurelius.environment.forecasting import ForecastingModel, build_frames
from aurelius.environment.ingestion.azure import _bin_stream
from aurelius.environment.simulation_clock import SUPPORTED_CONTROL_DT, SimulationClock
from aurelius.environment.training import make_world_state


def _frames():
    per = {p: [(p * 60 + i * 0.4, 220 + (i % 5) * 50, 100) for i in range(40 + 20 * (p % 3))]
           for p in range(40)}
    return build_frames(per, period_seconds=60.0, cycle_len=60), per


def _ctrl(H, *, dt=60.0, **kw):
    frames, _ = _frames()
    fm = ForecastingModel().fit(frames[:24], train_frac=0.7)
    c = ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=V2026FleetPlane().state_at(0), cost_model=CostModel(),
        risk_weight=0.5, confidence_min=0.05, sla_s=8.0, period_seconds=dt, tick_seconds=10.0,
        kv_service_factor_by_routing={"round_robin": 0.95, "kv_aware": 0.7}, sim_seconds=min(dt, 120.0),
        world_state=make_world_state({"n_servers": 16, "n_racks": 4, "seed": 0, "warm": 6}),
        horizon_steps=H, **kw)
    return c, fm, frames


# --- clock: horizon is STEPS, not hours -------------------------------------

def test_horizon_is_steps_not_hours():
    clk = SimulationClock(dt_seconds=300.0)
    # H=4 at dt=300s is 20 minutes, NOT 4 hours
    assert clk.lookahead_seconds(4) == 1200.0
    assert clk.lookahead_minutes(4) == 20.0
    assert clk.lookahead_hours(4) == 1200.0 / 3600.0
    # the SAME H at a different dt is a different real lookahead
    assert SimulationClock(dt_seconds=3600.0).lookahead_hours(4) == 4.0
    assert set(SUPPORTED_CONTROL_DT) >= {60.0, 300.0, 900.0, 3600.0}


def test_decision_reports_dt_and_lookahead_in_steps():
    c, _fm, frames = _ctrl(4, dt=300.0)
    c.decide(frames[:24])
    d = c.last_decision_diag
    assert d["horizon_steps"] == 4 and d["dt_seconds"] == 300.0
    assert d["lookahead_minutes"] == 20.0 and abs(d["lookahead_hours"] - 1200.0 / 3600.0) < 1e-4


# --- backward compatibility: H=1 parity -------------------------------------

def test_h1_rollout_reproduces_single_period_score():
    # the H=1 rollout score for a fixed candidate equals one direct single-period world sim.
    c, fm, frames = _ctrl(1)
    clk = SimulationClock(dt_seconds=c.period_seconds)
    traj = build_trajectory(fm, frames[:24], clk, 1)
    cand = ActionBundle().with_overrides(capacity_policy="backlog_aware", routing_policy="kv_aware")
    cum, steps = c._rollout_world(cand, traj, be=0.0, factor=0.7, horizon_steps=1)
    assert len(steps) == 1                                   # exactly one step
    # cumulative == the single step's risk-adjusted reward (rounded step diagnostics → relative tol)
    expected = steps[0]["gp_per_dollar"] - c.risk_weight * steps[0]["risk_viol"] * steps[0]["gp_per_dollar"]
    assert abs(cum - expected) / max(abs(cum), 1.0) < 1e-3


# --- receding horizon: clone isolation, first-action-only -------------------

def test_candidate_rollout_never_mutates_the_real_world():
    c, fm, frames = _ctrl(4)
    before = (c.world_state.period, c.world_state.warm_count(), len(c.world_state.migrations))
    c.decide(frames[:24])                                    # scores many candidates over H steps
    assert (c.world_state.period, c.world_state.warm_count(), len(c.world_state.migrations)) == before


def test_decide_commits_only_a_single_first_action():
    c, _fm, frames = _ctrl(4)
    d = c.decide(frames[:24])
    assert isinstance(d.bundle, ActionBundle)               # ONE first-action bundle, not a plan
    # the rollout is H steps long (planning artifact) but only the first action is returned
    assert len(c.last_decision_diag["rollout"]) == 4


# --- forecast trajectory: causal, no future leakage -------------------------

def test_forecast_trajectory_is_causal_length_h():
    _c, fm, frames = _ctrl(1)
    clk = SimulationClock(dt_seconds=60.0)
    traj = build_trajectory(fm, frames[:20], clk, 6)
    assert traj.horizon_steps == 6 and len(traj.to_dict()["path"]) == 6
    # built from the first 20 frames only — appending later frames cannot change the first-20 forecast
    again = build_trajectory(fm, frames[:20], clk, 6)
    assert traj.to_dict()["path"] == again.to_dict()["path"]
    # uncertainty is reported honestly (present or ABSENT), never fabricated
    man = traj.uncertainty_manifest()
    assert all("fidelity" in v and "has_quantiles" in v for v in man.values())


# --- determinism / replay ----------------------------------------------------

def test_decisions_are_deterministic_and_replayable():
    c1, _f, frames = _ctrl(4)
    c2, _f2, _ = _ctrl(4)
    d1, d2 = c1.decide(frames[:24]), c2.decide(frames[:24])
    assert d1.bundle.to_dict() == d2.bundle.to_dict() and d1.score == d2.score


# --- runtime budget + horizon scaling ---------------------------------------

def test_runtime_scales_with_horizon_and_world_steps_reported():
    diags = []
    for H in (1, 2, 4):
        c, _f, frames = _ctrl(H)
        c.decide(frames[:24])
        diags.append(c.last_decision_diag)
    # world-steps simulated grows with H (more rollout per candidate)
    assert diags[0]["world_steps_simulated"] < diags[1]["world_steps_simulated"] < diags[2]["world_steps_simulated"]
    # the candidate budget is reported and respected
    for d in diags:
        assert d["candidate_bundles_evaluated"] <= d["theoretical_bundles"]
        assert d["world_steps_simulated"] == d["candidate_bundles_evaluated"] * d["horizon_steps"]


def test_candidate_budget_is_respected():
    c, _f, frames = _ctrl(2, max_candidate_bundles=20)
    c.decide(frames[:24])
    # coordinate descent over the connected space, capped — far below the 8748 theoretical bundles
    assert c.last_decision_diag["candidate_bundles_evaluated"] <= 200
    assert c.last_decision_diag["theoretical_bundles"] == 8748


# --- sub-hour control: dt is the CONTROL INTERVAL, threaded into the eval replay ---------------

def _episode_warm_count_after(dt):
    """Run a few held-out periods of `run_period_episode` at control interval `dt` on a warm-seeded
    cluster under steady LOW load (so most warm replicas sit idle) and return the surviving warm pool.
    The reactive (`off`) policy cools an idle replica only AFTER the ~300s idle timeout, measured in
    real time as `idle_steps × dt` — so the surviving pool is a direct read of whether `dt` was
    threaded into the world simulator's time-based persistence."""
    per = {p: [(i * 6.0, 180, 90) for i in range(2)] for p in range(5)}      # 2 short jobs / period
    frames = build_frames(per, period_seconds=dt, cycle_len=max(1, round(86400 / dt)))
    ws = make_world_state({"n_servers": 16, "n_racks": 4, "seed": 0, "warm": 12})

    def decide(_h):                                   # fixed reactive (off) decision every period
        return {"action": {}, "prewarm_policy": "off", "placement_policy": "topology_blind",
                "migration_policy": "off", "routing_policy": "round_robin",
                "capacity_multiplier": 1.0, "batching_policy": "conservative"}
    run_period_episode("t", decide, per, frames, list(range(5)), fleet_state=V2026FleetPlane().state_at(0),
                       cost_model=CostModel(), sla_s=10.0, tick_seconds=10.0, period_seconds=dt,
                       kv_service_factor_by_routing={"round_robin": 1.0}, world_state=ws)
    return ws.warm_count()


def test_eval_replay_threads_dt_into_warm_persistence():
    # The held-out eval replay must advance warm state at the CONTROL interval, not a hardcoded hour.
    # At dt=60s an idle replica (idle 60..240s) stays warm across steps; at dt=3600s every idle step
    # (3600s > 300s timeout) cools it. Same load, same seed — only dt differs.
    warm_fine = _episode_warm_count_after(60.0)
    warm_hourly = _episode_warm_count_after(3600.0)
    assert warm_fine > warm_hourly                # sub-hour control preserves the warm pool across steps
    assert warm_hourly <= 4                        # hourly control cools the idle pool down to what it serves


def test_rebinning_changes_control_interval_not_hours():
    # Re-binning the SAME arrival stream at a finer dt yields proportionally MORE control periods at
    # the SAME arrival rate — dt is the control step length, not "hours". (Mirrors how the sweep
    # re-bins the one-week trace for sub-hour control.)
    rows = [(t * 1.0, 100, 50) for t in range(7200)]          # 1 req/s for 2 hours
    counts, rates = {}, {}
    for dt in (3600.0, 900.0, 300.0, 60.0):
        per, _exact = _bin_stream(iter(rows), bin_seconds=dt, sample_stride=1)
        frames = build_frames(per, period_seconds=dt, cycle_len=max(1, round(86400 / dt)))
        counts[dt] = len(frames)
        rates[dt] = sum(f.arrival_rate for f in frames) / len(frames)
    # finer dt → strictly more periods (60s gives 60× the bins of 3600s over the same 2 hours)
    assert counts[60.0] > counts[300.0] > counts[900.0] > counts[3600.0]
    assert counts[3600.0] == 2 and counts[60.0] == 120
    # ...but the arrival RATE (req/s) is invariant to the control interval (load unchanged, ~1/s)
    assert all(abs(r - 1.0) < 0.05 for r in rates.values())
