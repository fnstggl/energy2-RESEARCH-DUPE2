"""Decision Diagnostics Engine tests (focused — online observability + offline attribution correctness)."""

from __future__ import annotations

from aurelius.environment.actions import ActionBundle
from aurelius.environment.decision_diagnostics import (
    ABSENT_FORECASTS,
    CONSUMED_FORECASTS,
    LeaveOneOutAttributor,
    counterfactual_sensitivity,
    explain_decision,
    generate_roadmap,
    planner_confidence,
    regret_decomposition,
)


def _scored():
    # winner clearly ahead of the field (robust decision)
    return [(ActionBundle(precision_policy="fp8", spec_decode_policy="medium"), 180.0),
            (ActionBundle(precision_policy="fp8"), 150.0),
            (ActionBundle(), 100.0)]


# --- ONLINE: explanation from already-computed scores (no solves) ------------
def test_explain_decision_is_pure_over_scored_candidates():
    chosen = ActionBundle(precision_policy="fp8", spec_decode_policy="medium")
    e = explain_decision(7, chosen, _scored(), expected_gpd=180.0, expected_sla=0.01, expected_cost=5.0,
                         expected_reward=180.0, n_evaluated=3, planning_horizon=4, forecast_horizon=4,
                         search_strategy="beam_search")
    d = e.to_dict()
    assert d["decision_index"] == 7 and d["chosen_bundle"]["precision_policy"] == "fp8"
    assert d["decision_margin"] == 30.0 and d["decision_margin_pct"] > 0  # winner 180 − runner 150
    assert len(d["competing_candidates"]) == 3 and d["search_strategy"] == "beam_search"
    # why-won names the surface that distinguishes the winner from the runner-up (spec_decode)
    assert "spec_decode_policy" in d["why_won"]


def test_decision_margin_and_robustness():
    e = explain_decision(0, ActionBundle(precision_policy="fp8", spec_decode_policy="medium"), _scored(),
                         expected_gpd=180.0, expected_sla=0.0, expected_cost=0.0, expected_reward=180.0)
    assert e.robustness_score > 0.05 and e.switching_thresholds["stable"]   # 30/180 ≈ 0.17 → robust
    # a near-tie is fragile
    tie = [(ActionBundle(precision_policy="fp8"), 100.0), (ActionBundle(), 99.9)]
    cf = counterfactual_sensitivity(tie)
    assert not cf["stable"] and cf["robustness_score"] < 0.05


def test_planner_confidence_uses_only_computed_scores():
    assert 0.0 <= planner_confidence(_scored(), forecast_confidence=0.5) <= 1.0
    # a single candidate → falls back to the forecast confidence
    assert planner_confidence([(ActionBundle(), 1.0)], forecast_confidence=0.42) == 0.42


# --- OFFLINE: leave-one-out attribution correctness -------------------------
def test_leave_one_out_normalizes_and_counts_calls():
    drops = {"arrival_rate": 10.0, "output_length": 5.0, "prompt_length": 3.0, "interarrival_cv": 2.0}
    calls = []

    def evaluate(var):                              # None = full oracle (100); degraded = 100 − drop
        calls.append(var)
        return 100.0 - (drops.get(var, 0.0) if var else 0.0)

    res = LeaveOneOutAttributor().attribute(tuple(drops), evaluate)
    # oracle (None) + one eval per variable, in order → no future leakage beyond these isolations
    assert calls[0] is None and len(calls) == 1 + len(drops)
    c = res["contributions_pct"]
    assert res["method"] == "leave_one_out"
    assert abs(sum(c[v] for v in drops) - 100.0) < 1e-6        # normalised to 100%
    assert c["arrival_rate"] == 50.0                            # 10/(10+5+3+2)
    # ABSENT variables are reported 0 by construction, never fabricated
    assert all(c[v] == 0.0 for v in ABSENT_FORECASTS)


def test_consumed_and_absent_are_disjoint_and_documented():
    assert not (set(CONSUMED_FORECASTS) & set(ABSENT_FORECASTS))
    assert all(reason for reason in ABSENT_FORECASTS.values())  # every ABSENT var records WHY


def test_regret_decomposition_is_honest_about_world_model():
    r = regret_decomposition(current_gpd=149164, scenario_gpd=171485, oracle_gpd=174062)
    assert r["forecast_quality_pct"] == 100.0 and r["search_pct"] == 0.0
    # world-model fidelity is NOT fabricated — it is unmeasurable in pure simulation
    assert "UNMEASURABLE" in r["world_model_fidelity_pct"]
    assert r["within_forecast"]["workload_model_gain_gpd"] == 22321.0


def test_roadmap_is_ranked_from_attribution():
    attr = {"contributions_pct": {"arrival_rate": 50.0, "output_length": 30.0, "prompt_length": 20.0},
            "method": "leave_one_out"}
    rm = generate_roadmap(attr, {})
    assert rm[0]["improvement"].startswith("arrival_rate") and rm[1]["improvement"].startswith("output_length")
    assert rm[0]["estimated_impact_pct_of_forecast_regret"] == 50.0
    # the standing world-model item (collect real telemetry) is always present last
    assert any("telemetry" in r["improvement"] for r in rm)


def test_determinism():
    a = explain_decision(1, ActionBundle(precision_policy="fp8"), _scored(), expected_gpd=1.0, expected_sla=0.0,
                         expected_cost=0.0, expected_reward=180.0).to_dict()
    b = explain_decision(1, ActionBundle(precision_policy="fp8"), _scored(), expected_gpd=1.0, expected_sla=0.0,
                         expected_cost=0.0, expected_reward=180.0).to_dict()
    assert a == b
