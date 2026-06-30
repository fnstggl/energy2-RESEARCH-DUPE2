"""Default-change gate (Phase C) — the honest rule for promoting a candidate planner to the benchmark default.

PR #123 found `hierarchical_search` wins the search-method tournament. This module is the *gate* that decides
whether that win is strong enough to flip the **benchmark default planner** away from the current deployable
default (the PR #122 physics-guided beam). It is a **pure function** of already-measured per-arm summaries — it
runs no search, no simulator, no oracle; it only reads numbers the ladder produced and applies the contract.

The contract (10 conditions, all must hold; from `research/HIERARCHICAL_PLANNER_DEFAULT_AUDIT.md` Q9):

  1.  gp/$ strictly higher than the current default
  2.  SLA-violation rate not worse than the current default
  3.  the `production_scheduler` Pareto gate passes (candidate beats it on gp/$ AND SLA not worse)
  4.  the `sla_aware` Pareto gate passes — OR the failure is explicitly documented (then the headline is held)
  5.  required anchors always contained (the anchor floor is never dropped)
  6.  search regret not higher than the current default where measurable
  7.  runtime bounded (max evaluations per decision ≤ the runtime budget)
  8.  timeout rate acceptable (≤ the allowed ceiling)
  9.  no oracle / future data used by the candidate
  10. no int4 / quality-risked lever in the headline-winning action (no quality model exists to license it)

`passed` is the AND of all conditions. If `passed`, the verdict is `flip_benchmark_default`; otherwise
`keep_opt_in` with the failing conditions named. Condition 4 has a documented-exception path: a `sla_aware`
Pareto miss does not block the flip (sla_aware is the *hardest* honest bar, not the production bar) but it
**does** suppress the sla_aware headline — recorded as `headline_vs_sla_aware_allowed=False`. Nothing here is
tuned to a benchmark; the thresholds are operator/contract constants.
"""

from __future__ import annotations

from dataclasses import dataclass

_EPS = 1e-9


@dataclass
class ArmSummary:
    """A planner/baseline arm's ladder summary (aggregated across the validation windows). Only the fields the
    gate reads. MPC-only fields default to None/False so a static baseline can be passed for the Pareto checks."""

    name: str
    gp_per_dollar: float                     # SLA-safe goodput / $ (higher is better)
    sla_violation_rate: float                # fraction of requests missing the SLA (lower is better)
    # --- MPC-arm-only (None when not applicable / not measurable) ---
    regret: float | None = None              # search regret as a fraction of the best contained reward (0 = none)
    anchors_contained: bool | None = None    # were the required anchors in the evaluated set on EVERY window?
    max_evals_per_decision: int | None = None
    timeout_rate: float | None = None        # fraction of decisions that hit the per-cell wall-clock cap
    uses_oracle: bool = False                # did this arm read future / oracle workload or prices?
    headline_uses_quality_risk: bool = False  # int4 / other quality-risked lever in the winning bundle?


def _pareto_pass(candidate: ArmSummary, bar: ArmSummary) -> tuple[bool, float]:
    """The Pareto clause vs one bar arm: candidate beats it on gp/$ AND SLA is no worse. Returns
    `(passed, pct_delta)` where pct_delta is the gp/$ improvement over the bar (signed, percent)."""
    base = bar.gp_per_dollar
    pct = 100.0 * (candidate.gp_per_dollar - base) / base if abs(base) > _EPS else 0.0
    beats = candidate.gp_per_dollar > base + _EPS
    sla_ok = candidate.sla_violation_rate <= bar.sla_violation_rate + _EPS
    return (beats and sla_ok), pct


def default_change_gate(*, candidate: ArmSummary, current_default: ArmSummary,
                        production_scheduler: ArmSummary, sla_aware: ArmSummary,
                        runtime_budget_evals: int, timeout_rate_max: float = 0.0,
                        sla_aware_failure_documented: bool = False) -> dict:
    """Apply the 10-condition default-change contract. Pure: reads the summaries, returns a verdict dict.

    `candidate` is the proposed new default (e.g. `aurelius_mpc_hierarchical_search`); `current_default` is the
    deployable default it must beat (the physics-guided beam). `production_scheduler` and `sla_aware` are the
    benchmark bars for the Pareto clauses. `runtime_budget_evals` is the per-decision evaluation cap;
    `timeout_rate_max` the allowed timeout fraction. Set `sla_aware_failure_documented=True` to take the
    documented-exception path on condition 4 (flip still allowed; the sla_aware headline is suppressed).
    """
    prod_pass, prod_pct = _pareto_pass(candidate, production_scheduler)
    sla_pass, sla_pct = _pareto_pass(candidate, sla_aware)
    def_pct = (100.0 * (candidate.gp_per_dollar - current_default.gp_per_dollar)
               / current_default.gp_per_dollar if abs(current_default.gp_per_dollar) > _EPS else 0.0)

    # condition 6 — regret is only a blocker where BOTH arms measured it; otherwise it passes with a note.
    regret_measurable = candidate.regret is not None and current_default.regret is not None
    regret_ok = (candidate.regret <= current_default.regret + _EPS) if regret_measurable else True

    conditions = {
        # 1. gp/$ strictly higher than the current default
        "gp_per_dollar_higher_than_current_default": {
            "passed": candidate.gp_per_dollar > current_default.gp_per_dollar + _EPS,
            "candidate": round(candidate.gp_per_dollar, 4),
            "current_default": round(current_default.gp_per_dollar, 4),
            "abs_delta": round(candidate.gp_per_dollar - current_default.gp_per_dollar, 4),
            "pct_delta": round(def_pct, 3)},
        # 2. SLA not worse than the current default
        "sla_not_worse_than_current_default": {
            "passed": candidate.sla_violation_rate <= current_default.sla_violation_rate + _EPS,
            "candidate_sla": round(candidate.sla_violation_rate, 4),
            "current_default_sla": round(current_default.sla_violation_rate, 4)},
        # 3. production_scheduler Pareto gate passes
        "production_scheduler_pareto_pass": {
            "passed": prod_pass,
            "abs_delta": round(candidate.gp_per_dollar - production_scheduler.gp_per_dollar, 4),
            "pct_delta": round(prod_pct, 3),
            "candidate_sla": round(candidate.sla_violation_rate, 4),
            "production_scheduler_sla": round(production_scheduler.sla_violation_rate, 4)},
        # 4. sla_aware Pareto gate passes OR documented (documented does not block, but suppresses its headline)
        "sla_aware_pareto_pass_or_documented": {
            "passed": sla_pass or sla_aware_failure_documented,
            "pareto_pass": sla_pass,
            "documented_exception": (not sla_pass) and sla_aware_failure_documented,
            "abs_delta": round(candidate.gp_per_dollar - sla_aware.gp_per_dollar, 4),
            "pct_delta": round(sla_pct, 3)},
        # 5. required anchors always contained
        "required_anchors_always_contained": {
            "passed": candidate.anchors_contained is True,
            "anchors_contained": candidate.anchors_contained},
        # 6. search regret not higher than the current default where measurable
        "search_regret_not_higher_than_current_default": {
            "passed": regret_ok,
            "measurable": regret_measurable,
            "candidate_regret": candidate.regret,
            "current_default_regret": current_default.regret},
        # 7. runtime bounded
        "runtime_bounded": {
            "passed": (candidate.max_evals_per_decision is not None
                       and candidate.max_evals_per_decision <= runtime_budget_evals),
            "max_evals_per_decision": candidate.max_evals_per_decision,
            "runtime_budget_evals": runtime_budget_evals},
        # 8. timeout rate acceptable
        "timeout_rate_acceptable": {
            "passed": (candidate.timeout_rate is not None
                       and candidate.timeout_rate <= timeout_rate_max + _EPS),
            "timeout_rate": candidate.timeout_rate,
            "timeout_rate_max": timeout_rate_max},
        # 9. no oracle data
        "no_oracle_data": {
            "passed": not candidate.uses_oracle,
            "uses_oracle": candidate.uses_oracle},
        # 10. no int4 / quality-risked action in the headline
        "no_quality_risked_action_in_headline": {
            "passed": not candidate.headline_uses_quality_risk,
            "headline_uses_quality_risk": candidate.headline_uses_quality_risk},
    }

    failed = [name for name, c in conditions.items() if not c["passed"]]
    passed = not failed
    verdict = {
        "candidate": candidate.name,
        "current_default": current_default.name,
        "passed": passed,
        "verdict": "flip_benchmark_default" if passed else "keep_opt_in",
        "failed_conditions": failed,
        "conditions": conditions,
        # the sla_aware headline is only allowed when its Pareto clause genuinely passed (not via the exception).
        "headline_vs_sla_aware_allowed": sla_pass,
        "headline_vs_production_scheduler_allowed": prod_pass,
        "reason": ("all 10 conditions pass → promote candidate to the benchmark default"
                   if passed else
                   f"{len(failed)} condition(s) failed → keep candidate opt-in: {', '.join(failed)}"),
        "note": "SIMULATED directional evidence (bounded simulator-inferred magnitudes), not production telemetry",
    }
    return verdict


__all__ = ["ArmSummary", "default_change_gate"]
