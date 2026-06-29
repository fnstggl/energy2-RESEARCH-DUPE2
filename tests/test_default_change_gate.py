"""The default-change gate (Phase C) — the contract tests.

The gate is a PURE function: it promotes the candidate planner to the benchmark default only when all 10
conditions hold, and otherwise keeps it opt-in with the failing conditions named. These tests pin that it
blocks the unsafe cases (worse SLA, fails the production_scheduler Pareto clause, over budget, oracle data,
int4 / quality-risked headline, dropped anchors, higher regret) and allows the genuinely-safe case — plus the
documented-exception path for the (hardest) sla_aware bar.
"""

from __future__ import annotations

from aurelius.environment.planner.default_change_gate import ArmSummary, default_change_gate

# a canonical PASS configuration (mirrors the PR #123 hierarchical win): beats the default and both bars on
# gp/$, SLA not worse, anchors contained, bounded, 0 regret, no oracle, no quality-risked lever.
_CAND = dict(gp_per_dollar=715000.0, sla_violation_rate=0.0, regret=0.0, anchors_contained=True,
             max_evals_per_decision=80, timeout_rate=0.0, uses_oracle=False, headline_uses_quality_risk=False)
_DEFAULT = dict(gp_per_dollar=619000.0, sla_violation_rate=0.02, regret=0.0)
_PROD = dict(gp_per_dollar=300000.0, sla_violation_rate=0.10)
_SLA = dict(gp_per_dollar=264000.0, sla_violation_rate=0.25)


def _gate(cand_over=None, default_over=None, prod_over=None, sla_over=None, **kw):
    cand = ArmSummary("aurelius_mpc_hierarchical_search", **{**_CAND, **(cand_over or {})})
    deflt = ArmSummary("aurelius_mpc_current_default", **{**_DEFAULT, **(default_over or {})})
    prod = ArmSummary("production_scheduler", **{**_PROD, **(prod_over or {})})
    sla = ArmSummary("sla_aware", **{**_SLA, **(sla_over or {})})
    return default_change_gate(candidate=cand, current_default=deflt, production_scheduler=prod, sla_aware=sla,
                               runtime_budget_evals=kw.pop("runtime_budget_evals", 120),
                               timeout_rate_max=kw.pop("timeout_rate_max", 0.0), **kw)


def test_safe_case_flips_default():
    v = _gate()
    assert v["passed"] is True
    assert v["verdict"] == "flip_benchmark_default"
    assert v["failed_conditions"] == []
    # abs AND pct deltas are reported for every comparison.
    c = v["conditions"]["gp_per_dollar_higher_than_current_default"]
    assert c["abs_delta"] == 96000.0 and c["pct_delta"] is not None
    assert v["conditions"]["production_scheduler_pareto_pass"]["pct_delta"] is not None


def test_worse_sla_blocks():
    v = _gate(cand_over={"sla_violation_rate": 0.30})         # worse than the default's 0.02
    assert v["passed"] is False and v["verdict"] == "keep_opt_in"
    assert "sla_not_worse_than_current_default" in v["failed_conditions"]


def test_not_beating_production_scheduler_blocks():
    v = _gate(cand_over={"gp_per_dollar": 290000.0})          # below production_scheduler's 300000
    assert v["passed"] is False
    assert "production_scheduler_pareto_pass" in v["failed_conditions"]


def test_over_runtime_budget_blocks():
    v = _gate(cand_over={"max_evals_per_decision": 5000}, runtime_budget_evals=120)
    assert v["passed"] is False and "runtime_bounded" in v["failed_conditions"]


def test_timeouts_block():
    v = _gate(cand_over={"timeout_rate": 0.2}, timeout_rate_max=0.0)
    assert v["passed"] is False and "timeout_rate_acceptable" in v["failed_conditions"]


def test_oracle_data_blocks():
    v = _gate(cand_over={"uses_oracle": True})
    assert v["passed"] is False and "no_oracle_data" in v["failed_conditions"]


def test_quality_risked_headline_blocks():
    v = _gate(cand_over={"headline_uses_quality_risk": True})   # int4 etc. — no quality model licenses it
    assert v["passed"] is False
    assert "no_quality_risked_action_in_headline" in v["failed_conditions"]


def test_dropped_anchors_block():
    v = _gate(cand_over={"anchors_contained": False})
    assert v["passed"] is False and "required_anchors_always_contained" in v["failed_conditions"]


def test_higher_regret_blocks_when_measurable():
    v = _gate(cand_over={"regret": 0.4}, default_over={"regret": 0.1})
    assert v["passed"] is False
    assert "search_regret_not_higher_than_current_default" in v["failed_conditions"]


def test_regret_passes_when_not_measurable():
    v = _gate(cand_over={"regret": None}, default_over={"regret": None})
    assert v["conditions"]["search_regret_not_higher_than_current_default"]["passed"] is True
    assert v["conditions"]["search_regret_not_higher_than_current_default"]["measurable"] is False


def test_sla_aware_documented_exception_allows_but_suppresses_headline():
    # candidate does NOT beat the (hardest) sla_aware bar, but DOES beat production_scheduler + default.
    v = _gate(cand_over={"gp_per_dollar": 280000.0}, sla_over={"gp_per_dollar": 300000.0},
              prod_over={"gp_per_dollar": 270000.0}, sla_aware_failure_documented=True)
    # the documented exception keeps condition 4 satisfied (does not block the flip) ...
    assert v["conditions"]["sla_aware_pareto_pass_or_documented"]["passed"] is True
    assert v["conditions"]["sla_aware_pareto_pass_or_documented"]["documented_exception"] is True
    # ... but the sla_aware headline itself is suppressed (honest: it did not actually beat that bar).
    assert v["headline_vs_sla_aware_allowed"] is False


def test_sla_aware_miss_without_documentation_blocks():
    v = _gate(cand_over={"gp_per_dollar": 280000.0}, sla_over={"gp_per_dollar": 300000.0},
              prod_over={"gp_per_dollar": 270000.0}, sla_aware_failure_documented=False)
    assert v["passed"] is False
    assert "sla_aware_pareto_pass_or_documented" in v["failed_conditions"]
