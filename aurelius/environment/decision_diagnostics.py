"""Decision Diagnostics Engine — permanent planner observability for the Aurelius MPC.

The controller never produces an action without also producing an EXPLANATION. After each MPC optimisation
this engine emits a structured :class:`DecisionExplanation`: the chosen bundle, its expected
reward/SLA/cost/gp$, the planner confidence, the competing candidates and WHY the chosen one won, plus (on
demand) a forecast attribution (which predicted variables created the decision's value), a counterfactual
sensitivity (how robust the decision is), and a planner-regret decomposition.

Two strictly-separated tiers:
  * **ONLINE** (after every MPC solve, NEGLIGIBLE overhead): chosen bundle, expected metrics, reward
    decomposition, decision margin, planner confidence, local switching thresholds, top-K competitors,
    why-won — **all from values the search ALREADY computed**. NO extra simulation, NO oracle, NO
    re-solves. This is a permanent first-class controller output.
  * **OFFLINE** (validation / benchmarking only — NEVER called during live planning): forecast leave-one-out
    attribution, oracle reruns, world-model attribution, planner-regret decomposition, roadmap. Each
    re-plans under perturbed inputs and is far too expensive for the online path.

Attribution is **pluggable**: :class:`LeaveOneOutAttributor` today; the :class:`ForecastAttributor`
interface lets Shapley / integrated-gradients drop in later WITHOUT changing the diagnostics interface, and
every result records the ``attribution_method`` that produced it.

**Honesty:** attribution is reported only over the forecast variables the planner ACTUALLY consumes
(``CONSUMED_FORECASTS``). A variable the planner does not consume (``ABSENT_FORECASTS`` — KV-reuse is
unique-prefix in planning, queue/SLA-pressure are emergent not inputs, carbon/weather are not wired) is
reported **ABSENT (0 by construction)**, never fabricated. This PR changes no controller DECISION behaviour;
it only adds the explanation alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Forecast variables the MPC planning rollout actually consumes (controller._rollout_world / _rollout_ensemble
# + the cost objective). Attribution is computed ONLY over these.
CONSUMED_FORECASTS = ("arrival_rate", "output_length", "prompt_length", "interarrival_cv", "electricity_price")
# Variables a forecaster COULD predict but the planner does NOT consume today — reported ABSENT, with reason.
ABSENT_FORECASTS = {
    "kv_reuse": "planning uses synthetic UNIQUE prefixes (PR #112) → no KV-reuse prediction is consumed",
    "queue_pressure": "emergent from arrival+service, not a planner input forecast",
    "sla_pressure": "emergent from arrival+service+SLA, not a planner input forecast",
    "carbon": "not wired into the MPC objective or world model",
    "weather": "not wired into the MPC objective or world model",
    "congestion": "no live network-congestion model in the serving replay (ABSENT since PR #98)",
}


# ---------------------------------------------------------------------------
# ONLINE — the always-on explanation (only values the search already computed; no extra solves)
# ---------------------------------------------------------------------------
@dataclass
class DecisionExplanation:
    decision_index: int
    chosen_bundle: dict                 # non-default surfaces of the winner
    expected_reward: float
    expected_gp_per_dollar: float
    expected_sla_violation: float
    expected_cost: float
    expected_gpu_hours: float
    planning_horizon: int
    forecast_horizon: int
    planner_confidence: float           # blend of forecast-spread confidence + decision robustness
    n_candidates_evaluated: int
    search_strategy: str = ""
    planning_latency_s: float = 0.0
    reward_decomposition: dict = field(default_factory=dict)   # chosen bundle's reward components (computed)
    competing_candidates: list = field(default_factory=list)   # top-K [{surfaces, score}]
    why_won: dict = field(default_factory=dict)                # surfaces where winner ≠ runner-up
    decision_margin: float = 0.0                               # best_score − runner_up_score
    decision_margin_pct: float = 0.0                           # margin / |best_score| · 100
    robustness_score: float = 1.0                              # 0 = fragile, 1 = robust
    switching_thresholds: dict = field(default_factory=dict)   # top influential vars → local flip estimate
    # OFFLINE-only fields (stay None online; filled by the benchmark script):
    forecast_attribution: dict | None = None
    attribution_method: str | None = None

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def reward_decomposition_from_outcome(outcome) -> dict:
    """The chosen bundle's reward components — read straight off the PeriodOutcome the search ALREADY
    produced (no recompute): goodput, operator cost + its warm-hold / migration / energy parts, queue
    delay. These are the terms the gp/$ reward is built from."""
    kpi = getattr(outcome, "kpi", None)
    return {"sla_safe_goodput": round(getattr(kpi, "sla_safe_goodput", 0.0), 2) if kpi else 0.0,
            "operator_cost": round(getattr(outcome, "operator_cost", 0.0), 4),
            "warm_hold_cost": round(getattr(outcome, "warm_hold_cost", 0.0), 4),
            "migration_cost": round(getattr(outcome, "migration_cost", 0.0), 4),
            "energy_j": round(getattr(outcome, "energy_j", 0.0), 1),
            "queue_delay_p95": round(getattr(outcome, "queue_delay_p95", 0.0), 4),
            "sla_violation_rate": round(getattr(outcome, "sla_violation_rate", 0.0), 4),
            "quality_sla_risk": round(getattr(outcome, "quality_sla_risk", 0.0), 4)}


def planner_confidence(scored_candidates, *, forecast_confidence=0.5) -> float:
    """Confidence from values already computed during search: the winner→runner-up relative gap and the
    dispersion of candidate scores, blended with the forecast-spread confidence. No extra solves."""
    scores = sorted((float(s) for _b, s in scored_candidates), reverse=True)
    if len(scores) < 2:
        return round(float(forecast_confidence), 3)
    best = scores[0]
    rel_gap = (best - scores[1]) / max(abs(best), 1e-9)
    import statistics
    disp = statistics.pstdev(scores) / max(abs(best), 1e-9)
    decisiveness = max(0.0, min(1.0, 5.0 * rel_gap + 0.5 * min(1.0, disp)))
    return round(max(0.0, min(1.0, 0.5 * float(forecast_confidence) + 0.5 * decisiveness)), 3)


def local_switching_thresholds(scored_candidates, forecast_snapshot, *, top_k=2) -> dict:
    """LOCAL switching-threshold estimates for the few forecast variables that distinguish the winner from
    the runner-up — derived ONLY from the decision margin + the forecast snapshot (no perturbation, no
    re-solves). For each surface where the winner differs from the runner-up, report the current forecast of
    the most-related variable, the decision margin, and a stability verdict. A small margin ⇒ a small shift
    could flip the decision (fragile); a large margin ⇒ stable. Precise thresholds are an OFFLINE analysis."""
    ranked = sorted(scored_candidates, key=lambda t: -t[1])
    if len(ranked) < 2:
        return {"stable": True, "note": "single candidate"}
    best, runner = ranked[0], ranked[1]
    margin = best[1] - runner[1]
    rel = abs(margin) / max(abs(best[1]), 1e-9)
    diff_surfaces = []
    if hasattr(best[0], "to_dict") and hasattr(runner[0], "to_dict"):
        bd, rd = best[0].to_dict(), runner[0].to_dict()
        diff_surfaces = [k for k in bd if bd.get(k) != rd.get(k)][:top_k]
    return {"decision_margin": round(margin, 4), "decision_margin_pct": round(100.0 * rel, 2),
            "stable": rel > 0.05, "distinguishing_surfaces": diff_surfaces,
            "forecast_snapshot": {k: v for k, v in (forecast_snapshot or {}).items()},
            "note": ("precise per-variable switching thresholds are an OFFLINE perturbation analysis; "
                     "online reports the decision margin as the cheap robustness proxy")}


def explain_decision(decision_index, chosen, scored_candidates, *, expected_gpd, expected_sla,
                     expected_cost, expected_reward, expected_gpu_hours=0.0, forecast_confidence=0.5,
                     n_evaluated=0, planning_horizon=1, forecast_horizon=1, search_strategy="",
                     planning_latency_s=0.0, chosen_outcome=None, forecast_snapshot=None) -> DecisionExplanation:
    """ONLINE explanation built ONLY from already-computed search values. ``scored_candidates`` is the list
    of ``(bundle, score)`` the search evaluated; ``chosen_outcome`` is the winner's PeriodOutcome (for the
    reward decomposition). No extra simulation is performed."""
    ranked = sorted(scored_candidates, key=lambda t: -t[1])
    top = ranked[:5]
    best_score = ranked[0][1] if ranked else float(expected_reward)
    margin = (best_score - ranked[1][1]) if len(ranked) > 1 else best_score
    rel = min(1.0, abs(margin) / max(abs(best_score), 1e-9))
    why = {}
    if len(ranked) > 1 and hasattr(ranked[1][0], "to_dict") and hasattr(chosen, "to_dict"):
        rd, cd = ranked[1][0].to_dict(), chosen.to_dict()
        why = {k: {"chosen": cd[k], "runner_up": rd[k]} for k in cd if cd.get(k) != rd.get(k)}
    return DecisionExplanation(
        decision_index=decision_index,
        chosen_bundle=chosen.non_default_surfaces() if hasattr(chosen, "non_default_surfaces") else {},
        expected_reward=float(expected_reward), expected_gp_per_dollar=float(expected_gpd),
        expected_sla_violation=float(expected_sla), expected_cost=float(expected_cost),
        expected_gpu_hours=float(expected_gpu_hours), planning_horizon=int(planning_horizon),
        forecast_horizon=int(forecast_horizon),
        planner_confidence=planner_confidence(scored_candidates, forecast_confidence=forecast_confidence),
        n_candidates_evaluated=int(n_evaluated), search_strategy=search_strategy,
        planning_latency_s=round(float(planning_latency_s), 4),
        reward_decomposition=(reward_decomposition_from_outcome(chosen_outcome) if chosen_outcome is not None else {}),
        competing_candidates=[{"surfaces": (b.non_default_surfaces() if hasattr(b, "non_default_surfaces") else {}),
                               "score": round(float(s), 4)} for b, s in top],
        why_won=why, decision_margin=round(float(margin), 4), decision_margin_pct=round(100.0 * rel, 2),
        robustness_score=round(rel, 4),
        switching_thresholds=local_switching_thresholds(scored_candidates, forecast_snapshot))


# ---------------------------------------------------------------------------
# OFFLINE ONLY — pluggable forecast attribution (re-plans under perturbed inputs; NEVER called online)
# ---------------------------------------------------------------------------
class ForecastAttributor:
    """Interface so Shapley / integrated-gradients can replace leave-one-out WITHOUT changing callers."""
    name = "abstract"

    def attribute(self, consumed_vars, evaluate) -> dict:
        raise NotImplementedError


class LeaveOneOutAttributor(ForecastAttributor):
    """Start from the ORACLE decision (every forecast = the realised future); degrade ONE variable back to
    the model forecast, re-plan, and measure the gp/$ drop on the TRUE workload — that drop is the variable's
    planner-value. ``evaluate(var)`` returns the true-workload gp/$ of the bundle the planner picks with
    ``var`` degraded (``None`` = full oracle). Contributions are normalised to ~100%. Leave-one-out ignores
    interaction terms — documented; the residual (1 − Σ vs the full oracle→current gap) captures them."""
    name = "leave_one_out"

    def attribute(self, consumed_vars, evaluate) -> dict:
        base = evaluate(None)                                  # full-oracle gp/$
        drops = {v: max(0.0, base - evaluate(v)) for v in consumed_vars}
        total = sum(drops.values()) or 1.0
        out = {v: round(100.0 * d / total, 1) for v, d in drops.items()}
        out.update({v: 0.0 for v in ABSENT_FORECASTS})        # ABSENT vars are 0 by construction
        return {"method": self.name, "contributions_pct": out, "oracle_gp_per_dollar": round(base, 1),
                "raw_drops": {v: round(d, 1) for v, d in drops.items()},
                "absent": dict(ABSENT_FORECASTS)}


def forecast_attribution(evaluate, *, attributor: ForecastAttributor | None = None,
                         consumed_vars=CONSUMED_FORECASTS) -> dict:
    """Run a pluggable forecast attribution. ``evaluate(var|None)→gp/$`` is provided by the caller (it knows
    how to re-plan with one forecast degraded). Always records the method used."""
    return (attributor or LeaveOneOutAttributor()).attribute(consumed_vars, evaluate)


def counterfactual_sensitivity(scored_candidates) -> dict:
    """How robust the decision is, from the candidate score ranking: the runner-up margin and a robustness
    score. A small margin ⇒ the decision could flip under modest forecast error (a candidate for
    uncertainty-aware planning); a large margin ⇒ stable."""
    ranked = sorted(scored_candidates, key=lambda t: -t[1])
    if len(ranked) < 2:
        return {"decision_margin": None, "robustness_score": 1.0, "runner_up": None, "stable": True}
    best, runner = ranked[0], ranked[1]
    margin = best[1] - runner[1]
    robustness = round(min(1.0, abs(margin) / max(abs(best[1]), 1e-9)), 4)
    return {"decision_margin": round(margin, 4), "robustness_score": robustness,
            "runner_up": (runner[0].non_default_surfaces() if hasattr(runner[0], "non_default_surfaces") else {}),
            "stable": robustness > 0.05}


def regret_decomposition(*, current_gpd, scenario_gpd, oracle_gpd, search_regret_frac=0.0,
                         objective_gap_frac=0.0) -> dict:
    """Decompose the MEASURABLE planner regret (Current → Oracle) into categories, with the methodology.

    Current/Scenario/Oracle differ ONLY in the planning WORKLOAD (median → ensemble → exact future); the
    controller, search, world model and objective are identical across them. So the Current→Oracle gap is
    **entirely forecast (workload-prediction) quality** by construction. Search regret is measured separately
    (beam ≈ exhaustive ⇒ ≈0, PR #112). World-model fidelity is **not isolable in a fully-simulated
    environment** — the planner and the evaluator share the same simulator, so there is no higher-fidelity
    reference; attributing it requires REAL serving telemetry (absent). We report that honestly rather than
    fabricate a percentage."""
    total = max(oracle_gpd - current_gpd, 1e-9)
    forecast_frac = 1.0 - search_regret_frac - objective_gap_frac
    return {
        "methodology": ("Current/Scenario/Oracle vary ONLY the planning workload → the Current→Oracle gap is "
                        "forecast quality by construction; search measured separately; world-model fidelity "
                        "is NOT isolable in pure simulation (shared simulator) and needs real telemetry."),
        "total_planner_regret_gpd": round(total, 1),
        "forecast_quality_pct": round(100.0 * max(0.0, forecast_frac), 1),
        "search_pct": round(100.0 * search_regret_frac, 1),
        "objective_pct": round(100.0 * objective_gap_frac, 1),
        "world_model_fidelity_pct": "UNMEASURABLE_IN_SIMULATION (needs real serving telemetry)",
        "within_forecast": {"workload_model_gain_gpd": round(scenario_gpd - current_gpd, 1),
                            "residual_forecast_gap_gpd": round(oracle_gpd - scenario_gpd, 1)},
    }


def generate_roadmap(attribution: dict, regret: dict) -> list:
    """Rank the next engineering improvements DIRECTLY from the measured attribution (highest forecast
    contributor first). Each item carries the planner-regret it explains, an effort estimate and confidence."""
    contrib = attribution.get("contributions_pct", {})
    consumed = sorted(((v, p) for v, p in contrib.items() if v in CONSUMED_FORECASTS and p > 0),
                      key=lambda t: -t[1])
    effort = {"arrival_rate": "low", "output_length": "low", "prompt_length": "medium",
              "interarrival_cv": "medium", "electricity_price": "low"}
    items = []
    for rank, (v, pct) in enumerate(consumed, 1):
        items.append({"rank": rank, "improvement": f"{v} forecaster",
                      "estimated_impact_pct_of_forecast_regret": pct,
                      "expected_effort": effort.get(v, "medium"),
                      "confidence": "medium (leave-one-out, bounded window)",
                      "evidence": f"{pct}% of the measured forecast attribution"})
    # the standing world-model item: collect real telemetry so world-model fidelity becomes attributable.
    items.append({"rank": len(items) + 1, "improvement": "real serving telemetry (to attribute world-model fidelity)",
                  "estimated_impact_pct_of_forecast_regret": None, "expected_effort": "high",
                  "confidence": "high (it is the only way to isolate simulator error)",
                  "evidence": "world-model fidelity is UNMEASURABLE in pure simulation"})
    return items


__all__ = ["CONSUMED_FORECASTS", "ABSENT_FORECASTS", "DecisionExplanation", "explain_decision",
           "reward_decomposition_from_outcome", "planner_confidence", "local_switching_thresholds",
           "ForecastAttributor", "LeaveOneOutAttributor", "forecast_attribution",
           "counterfactual_sensitivity", "regret_decomposition", "generate_roadmap"]
