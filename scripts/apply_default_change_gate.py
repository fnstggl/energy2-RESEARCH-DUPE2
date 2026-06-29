#!/usr/bin/env python3
"""Apply the default-change gate (Phase C) to the real ladder + regret results.

Reads `ladder_benchmark.json` (Phase E) and `hierarchical_regret_validation.json` (Phase F), aggregates the
per-arm KPIs across the validation windows, builds the four `ArmSummary` inputs, and runs the pure
`default_change_gate`. Writes the verdict to `default_change_gate_verdict.json`. This is the gate's *application*
— the gate itself (the contract) is the pure function under test in `tests/test_default_change_gate.py`.

Usage: python -m scripts.apply_default_change_gate
"""

from __future__ import annotations

import json
import os
import statistics

from aurelius.environment.planner.default_change_gate import ArmSummary, default_change_gate

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_LADDER = os.path.join(_OUT, "ladder_benchmark.json")
_REGRET = os.path.join(_OUT, "hierarchical_regret_validation.json")
_VERDICT = os.path.join(_OUT, "default_change_gate_verdict.json")

_RUNTIME_BUDGET_EVALS = 120          # the controller's planner_budget (100) + slack; the runtime cap
_CANDIDATE = "aurelius_mpc_hierarchical_search"
_DEFAULT = "aurelius_mpc_current_default"


def _agg(ladder, arm, key):
    vals = [c["result"][key] for k, c in ladder["cells"].items()
            if c.get("status") == "COMPLETED" and k.endswith(f"|{arm}") and key in (c.get("result") or {})]
    return statistics.mean(vals) if vals else None


def _max_evals(ladder, arm):
    vals = [(c["result"].get("search") or {}).get("candidate_bundles_evaluated")
            for k, c in ladder["cells"].items()
            if c.get("status") == "COMPLETED" and k.endswith(f"|{arm}") and (c.get("result") or {}).get("search")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def _timeout_rate(ladder, arm):
    cells = [c for k, c in ladder["cells"].items() if k.endswith(f"|{arm}")]
    if not cells:
        return None
    return sum(1 for c in cells if c.get("status") == "TIMEOUT") / len(cells)


def main() -> None:
    ladder = json.load(open(_LADDER))
    regret = json.load(open(_REGRET)) if os.path.exists(_REGRET) else None

    # regret (Phase F, exhaustive-able synthetic) — the candidate's and default's max true regret %.
    cand_regret = default_regret = None
    anchors_ok = True
    if regret:
        rs = regret["summary"]
        cand_regret = rs.get("hierarchical_max_regret_pct")
        default_regret = rs.get("current_default_max_regret_pct")
        # condition 5 is the ANCHOR contract: the named known-good bundles were evaluated on every fixture.
        # (Distinct from "contains the regret reference": the true optimum can lie OUTSIDE the static core-grid
        # reachable set — on sla_tight it does, and hierarchical reaches it anyway → regret 0. That reach is the
        # advantage, not an anchor violation.)
        anchors_ok = bool(rs.get("anchors_held_everywhere"))

    candidate = ArmSummary(
        _CANDIDATE, gp_per_dollar=_agg(ladder, _CANDIDATE, "gp_per_dollar"),
        sla_violation_rate=_agg(ladder, _CANDIDATE, "sla_violation_rate"),
        regret=cand_regret, anchors_contained=anchors_ok,
        max_evals_per_decision=_max_evals(ladder, _CANDIDATE),
        timeout_rate=_timeout_rate(ladder, _CANDIDATE), uses_oracle=False,
        # quality-risked levers are excluded by construction; confirm via the simulated quality-risk KPI.
        headline_uses_quality_risk=bool((_agg(ladder, _CANDIDATE, "quality_sla_risk_mean") or 0.0) > 1e-9))
    current_default = ArmSummary(
        _DEFAULT, gp_per_dollar=_agg(ladder, _DEFAULT, "gp_per_dollar"),
        sla_violation_rate=_agg(ladder, _DEFAULT, "sla_violation_rate"), regret=default_regret)
    production = ArmSummary(
        "production_scheduler", gp_per_dollar=_agg(ladder, "production_scheduler", "gp_per_dollar"),
        sla_violation_rate=_agg(ladder, "production_scheduler", "sla_violation_rate"))
    sla_aware = ArmSummary(
        "sla_aware", gp_per_dollar=_agg(ladder, "sla_aware", "gp_per_dollar"),
        sla_violation_rate=_agg(ladder, "sla_aware", "sla_violation_rate"))

    verdict = default_change_gate(candidate=candidate, current_default=current_default,
                                  production_scheduler=production, sla_aware=sla_aware,
                                  runtime_budget_evals=_RUNTIME_BUDGET_EVALS, timeout_rate_max=0.0)
    out = {"verdict": verdict,
           "inputs": {a.name: a.__dict__ for a in (candidate, current_default, production, sla_aware)},
           "windows": sorted({k.rsplit("|", 1)[0] for k in ladder["cells"]}),
           "note": "Aggregated across the Phase E ladder windows; regret from the Phase F exhaustive-able "
                   "synthetic fixtures. SIMULATED directional evidence."}
    with open(_VERDICT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"VERDICT: {verdict['verdict']}  (passed={verdict['passed']})")
    if verdict["failed_conditions"]:
        print("  failed:", verdict["failed_conditions"])
    print(f"  candidate gp/$={candidate.gp_per_dollar:.0f} sla={candidate.sla_violation_rate:.4f} "
          f"regret={candidate.regret} evals={candidate.max_evals_per_decision}")
    print(f"  vs default {current_default.gp_per_dollar:.0f}: "
          f"+{candidate.gp_per_dollar - current_default.gp_per_dollar:.0f}")
    print(f"→ {_VERDICT}")


if __name__ == "__main__":
    main()
