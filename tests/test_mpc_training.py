"""Tests for MPC training/tuning + the honest claim gate (Phase 3).

Proves: disjoint train/val/eval splits (no leakage), deterministic training, the tuned
controller loads + evaluates, the claim gate blocks a headline unless the controller
beats the strongest NON-weak baseline with disjoint splits and no oracle.
"""

from __future__ import annotations

from aurelius.environment.controller import EpisodeReport
from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.forecasting import build_frames
from aurelius.environment.training import (
    claim_gate,
    evaluate_mpc,
    split_cuts,
    train_mpc_policy,
)


def _synth():
    per = {p: [(p * 60 + i * 1.5, 200 + (i % 6) * 60, 100) for i in range(10 + p % 6)]
           for p in range(40)}
    frames = build_frames(per, period_seconds=60.0, cycle_len=60)
    return frames, per


def _arm(gpd):
    return EpisodeReport(name="x", n_periods=5, sla_safe_goodput=gpd, total_operator_cost=1.0,
                         goodput_per_dollar=gpd, sla_violation_rate=0.1, gpu_hours=1.0,
                         energy_cost=0.1, n_sla_safe=10, queue_delay_p95=1.0)


def test_split_cuts_disjoint_and_ordered():
    t1, t2 = split_cuts(40, train=0.5, val=0.25)
    assert 0 < t1 < t2 < 40                       # train | val | eval strictly ordered


def test_claim_gate_blocks_when_not_beating_fair_baseline():
    arms = {"mpc_controller": _arm(90.0), "sla_aware": _arm(100.0), "fifo_weak": _arm(50.0)}
    g = claim_gate(arms)
    assert g["fair_baseline"] == "sla_aware"          # strongest non-weak (never fifo)
    assert not g["beats_fair_baseline"] and not g["headline_claim_allowed"]


def test_claim_gate_allows_only_when_beats_and_clean():
    arms = {"mpc_controller": _arm(120.0), "sla_aware": _arm(100.0), "fifo_weak": _arm(50.0)}
    g = claim_gate(arms)
    assert g["fair_baseline"] == "sla_aware" and g["beats_fair_baseline"]
    assert g["no_oracle"] and g["splits_disjoint"] and g["headline_claim_allowed"]
    assert g["fair_baseline"] not in ("fifo_weak",)   # weak is never the fair baseline


def test_train_and_evaluate_pipeline_deterministic_and_gated():
    frames, per = _synth()
    fleet = V2026FleetPlane().state_at(0)
    cm = CostModel()
    common = {"sla_s": 10.0, "period_seconds": 60.0, "tick_seconds": 10.0}
    grid = {"horizon": [1, 2], "risk_weight": [0.0, 0.5], "confidence_min": [0.1]}
    tr1, fm1 = train_mpc_policy(frames, per, fleet_state=fleet, cost_model=cm, grid=grid, common=common)
    tr2, _ = train_mpc_policy(frames, per, fleet_state=fleet, cost_model=cm, grid=grid, common=common)
    assert tr1["controller_config"] == tr2["controller_config"]    # deterministic
    s = tr1["splits"]
    assert s["train_cut"] <= s["val"][0] and s["val"][1] <= s["eval"][0]   # disjoint
    rep = evaluate_mpc(tr1, fm1, frames, per, fleet_state=fleet, cost_model=cm, common=common)
    assert "mpc_controller" in rep["arms"] and "gate" in rep
    g = rep["gate"]
    # the gate is internally consistent and never claims on a weak baseline
    assert g["fair_baseline"] != "fifo_weak"
    assert g["headline_claim_allowed"] == (g["beats_fair_baseline"] and g["fair_baseline_not_weak"]
                                           and g["no_oracle"] and g["splits_disjoint"])
