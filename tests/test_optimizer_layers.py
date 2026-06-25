"""Tests for the first-class optimization layers (Phase B+).

Verifies that ForecastLayer / ConstraintLayer / ObjectiveLayer / ReplayLayer /
EvaluationLayer are real, optimizer-owned components (not just scattered code),
each a thin wrapper over the existing implementation, and that the live-cluster
orchestration surface is a first-class participant in optimize_fleet.
"""

from __future__ import annotations

import pytest

from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.layers import (
    ConstraintLayer,
    EvaluationLayer,
    ForecastLayer,
    ObjectiveLayer,
    ReplayLayer,
)

# --------------------------------------------------------------------------
# Layers are first-class, optimizer-owned components
# --------------------------------------------------------------------------

def test_optimizer_exposes_all_layers():
    opt = AureliusOptimizer()
    assert isinstance(opt.objective, ObjectiveLayer)
    assert isinstance(opt.constraints, ConstraintLayer)
    assert isinstance(opt.forecast, ForecastLayer)
    assert isinstance(opt.replay, ReplayLayer)
    assert isinstance(opt.evaluation, EvaluationLayer)
    # cached (same object each access)
    assert opt.objective is opt.objective


def test_decision_surfaces_enumeration_includes_live_orchestration():
    s = AureliusOptimizer.DECISION_SURFACES
    for name in ("energy", "serving_queue", "replica_scaling", "genai_serving",
                 "placement", "admission", "live_orchestration"):
        assert name in s


# --------------------------------------------------------------------------
# ObjectiveLayer — SLA-safe goodput/$ as a first-class scorer
# --------------------------------------------------------------------------

def test_objective_layer_scores_goodput_per_dollar():
    obj = ObjectiveLayer()
    assert obj.score(sla_compliant_goodput=1000, total_infrastructure_cost=4.0) == 250.0
    # no cost basis -> None (undefined)
    assert obj.score(sla_compliant_goodput=1000, total_infrastructure_cost=0.0) is None
    # from a kpi-like object / dict
    assert obj.score(kpi={"sla_safe_goodput_per_infra_dollar": 7.5}) == 7.5


def test_objective_layer_compare_ranks_descending_none_last():
    obj = ObjectiveLayer()
    ranked = obj.compare({
        "current_main": 250.0, "best_aurelius": 300.0,
        "candidate": 280.0, "broken": None,
    })
    assert [name for name, _ in ranked] == [
        "best_aurelius", "candidate", "current_main", "broken"
    ]


def test_objective_layer_score_requires_inputs():
    with pytest.raises(ValueError):
        ObjectiveLayer().score()


# --------------------------------------------------------------------------
# ForecastLayer — honest taxonomy + the one causal forecast
# --------------------------------------------------------------------------

def test_forecast_layer_taxonomy_is_honest():
    fl = ForecastLayer()
    tax = fl.classify()
    # the only decision-feeding forecaster is the causal capacity forecast
    assert tax["decision_feeding"] == ("forecasted_mcs_capacity",)
    # output-length / gpu_placement are research-only (they HURT the KPI)
    assert "output_length" in tax["research_only"]
    assert "gpu_placement" in tax["research_only"]


def test_forecast_layer_causal_capacity_forecast_runs():
    fl = ForecastLayer()
    raw = [(float(i), 100 + (i % 4) * 40) for i in range(300)]
    sched = fl.causal_capacity_forecast(raw, 60.0, 1.0, method="ewma")
    assert isinstance(sched, list) and len(sched) >= 1
    assert all(isinstance(c, int) and c >= 1 for c in sched)


# --------------------------------------------------------------------------
# ConstraintLayer — wraps the (recommendation-only) ConstraintAwareEngine
# --------------------------------------------------------------------------

def test_constraint_layer_reuses_live_engine_and_is_recommendation_only():
    opt = AureliusOptimizer()
    assert opt.constraints.engine is opt.serving_orchestration
    assert opt.constraints.engine.implementation_mode == "recommendation_only"


# --------------------------------------------------------------------------
# ReplayLayer — cross-loop result normalization
# --------------------------------------------------------------------------

def test_replay_layer_benchmark_ids_and_unknown_kind():
    rl = ReplayLayer()
    assert "serving_queue" in rl.benchmark_ids
    with pytest.raises(ValueError):
        rl.normalize("not_a_loop", "p", object())


# --------------------------------------------------------------------------
# live_orchestration surface is a first-class optimize_fleet participant
# --------------------------------------------------------------------------

class _StubEngine:
    implementation_mode = "recommendation_only"

    def __init__(self):
        self.calls = []

    def run(self, state, sla_registry=None):
        self.calls.append((state, sla_registry))
        return {"recommendations": [], "stub": True}


def test_optimize_fleet_live_surface_routes_through_recommend_live():
    opt = AureliusOptimizer()
    stub = _StubEngine()
    opt._constraint_engine = stub  # inject (avoids building a full ClusterState)
    sentinel_state = object()
    res = opt.optimize_fleet(live={"state": sentinel_state})
    assert "live_orchestration" in res.surfaces_used
    assert res.live == {"recommendations": [], "stub": True}
    assert stub.calls == [(sentinel_state, None)]
    assert any("recommendation-only" in n for n in res.notes)
