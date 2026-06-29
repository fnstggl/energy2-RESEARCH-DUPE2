#!/usr/bin/env python3
"""Hierarchical-search regret & containment validation vs exhaustive_small (Phase F).

The default-change gate (condition 5/6) requires that the candidate planner (`hierarchical_search`) (a) never
drops the required anchors and (b) has search regret no higher than the current default WHERE MEASURABLE. On
the synthetic bottleneck fixtures the action space is small enough to ENUMERATE, so `exhaustive_small` gives
the TRUE optimum and regret is exact — the faithful test of "does the bounded hierarchical search actually find
the best bundle, or does it leave gp/$ on the table?".

This runs three methods on each fixture (`memory_bound_decode`, `compute_bound_prefill`, `sla_tight`,
`queue_bound` — i.e. the SLA-tight / memory-bound / compute-bound regimes) at an equal evaluation budget,
through the SAME unchanged reward path (`simulate_period`):

  • physics_guided_grid   the current deployable default's candidate set (core grid; the #122 beam searches it)
  • hierarchical_search   the candidate new default (slow/medium/fast timescale groups + coupling + polish)
  • exhaustive_small      the TRUE optimum (regret reference)

For each fixture it reports each method's true regret (abs + %), whether the true optimum was CONTAINED in the
method's reachable space and whether it was actually EVALUATED, and any anchor-contract violation. No tuning,
no gate weakening; this is a measurement. Fast (synthetic, ~40–96 jobs/fixture). Writes a JSON artifact.

Usage: python -m scripts.run_hierarchical_regret_validation
"""

from __future__ import annotations

import json
import os

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "hierarchical_regret_validation.json")

_METHODS = ("physics_guided_grid", "hierarchical_search", "exhaustive_small")
_BUDGET = 100


def run() -> dict:
    from aurelius.environment.physics_guided_candidates import SAFE_BASELINE_BUNDLE
    from aurelius.environment.planner.candidate_generators import named_anchor_keys
    from aurelius.environment.planner.planner_tournament import (
        SYNTHETIC_FIXTURES,
        run_tournament_window,
        synthetic_fixture_scorer,
    )
    from scripts.run_checkpointed_electricity_backtest import build_market

    # standard fleet / cost-model / world params (same build the tournament uses) — only the workload varies.
    ctx = build_market("pjm", req_cap=80, mooncake_limit=6000)
    fleet, cost_model = ctx["fleet"], ctx["cm"]
    world_params = ctx["common"].get("world_state_params")
    prev_best = None                                  # cold start (no prior winner) → the named anchors must hold

    per_fixture: dict = {}
    for fx in SYNTHETIC_FIXTURES:
        scorer, state = synthetic_fixture_scorer(fx, fleet, cost_model, world_params)
        wr = run_tournament_window(scorer, state, methods=list(_METHODS), budgets=[_BUDGET],
                                   baseline_bundle=SAFE_BASELINE_BUNDLE, exhaustive_for_regret=True,
                                   prev_best=prev_best)
        regret = wr["regret"]
        rtab = regret["per_method"]
        cells = wr["cells"]
        # the anchor contract: a non-exempt method that did NOT evaluate the named anchors is a violation.
        violations = [m for m in _METHODS if not rtab[m]["anchors_evaluated"]]
        per_fixture[fx.name] = {
            "expected_regime": fx.expected_regime, "active_regimes": wr["regimes"],
            "reference_kind": regret.get("reference_kind"), "baseline_gp": wr["baseline_gp"],
            "methods": {m: {
                "gp_per_dollar": cells[f"{m}|{_BUDGET}"]["gp_per_dollar"],
                "regret_abs": rtab[m]["regret_abs"], "regret_pct": rtab[m]["regret_pct"],
                "true_opt_contained": rtab[m]["true_opt_contained"],
                "true_opt_evaluated": rtab[m]["true_opt_evaluated"],
                "candidates_evaluated": cells[f"{m}|{_BUDGET}"]["candidates_evaluated"],
                "anchors_evaluated": rtab[m]["anchors_evaluated"],
            } for m in _METHODS},
            "anchor_contract_violations": violations,
        }

    # roll up the headline finding: hierarchical's regret vs the current default's, and containment everywhere.
    hsr = [per_fixture[f]["methods"]["hierarchical_search"] for f in per_fixture]
    cdr = [per_fixture[f]["methods"]["physics_guided_grid"] for f in per_fixture]
    summary = {
        "named_anchors": sorted(str(k) for k in named_anchor_keys(prev_best)),
        "hierarchical_max_regret_pct": max((r["regret_pct"] or 0.0) for r in hsr),
        "current_default_max_regret_pct": max((r["regret_pct"] or 0.0) for r in cdr),
        "hierarchical_contains_true_opt_everywhere": all(r["true_opt_contained"] for r in hsr),
        "hierarchical_evaluates_true_opt_everywhere": all(r["true_opt_evaluated"] for r in hsr),
        "hierarchical_regret_not_higher_than_default": all(
            (per_fixture[f]["methods"]["hierarchical_search"]["regret_abs"]
             <= per_fixture[f]["methods"]["physics_guided_grid"]["regret_abs"] + 1e-9) for f in per_fixture),
        "anchors_held_everywhere": all(not per_fixture[f]["anchor_contract_violations"] for f in per_fixture),
        "note": "TRUE regret (exhaustive reference) on synthetic exhaustive-able fixtures; SIMULATED magnitudes.",
    }
    return {"budget": _BUDGET, "methods": list(_METHODS), "per_fixture": per_fixture, "summary": summary}


def main() -> None:
    out = run()
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(out, f, indent=2)
    s = out["summary"]
    print(f"hierarchical max regret: {s['hierarchical_max_regret_pct']}%  "
          f"(current default: {s['current_default_max_regret_pct']}%)")
    print(f"contains true opt everywhere: {s['hierarchical_contains_true_opt_everywhere']}  "
          f"| evaluates: {s['hierarchical_evaluates_true_opt_everywhere']}  "
          f"| anchors held: {s['anchors_held_everywhere']}")
    print(f"→ {_ARTIFACT}")


if __name__ == "__main__":
    main()
