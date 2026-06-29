#!/usr/bin/env python3
"""Diagnostic: reconcile PR #123's "+1273% vs sla_aware" with PR #124's "+164% vs sla_aware / +148% vs
production_scheduler" — same window, same method, two DIFFERENT measurement harnesses.

This changes NOTHING about the simulator physics / reward / Pareto gate / cost model / action semantics /
baselines. It only MEASURES the two existing setups side by side on the SAME pjm·expensive window:

  • Setup A = the PR #123 search-method tournament harness: ONE planning decision (`market_window_scorer` →
    `_rollout_world`, horizon_steps=1) over the forecast trajectory, req_cap 80. gp/$ = the single rollout
    period's `gp_per_dollar`. Baseline = `SAFE_BASELINE_BUNDLE` (== `SLA_AWARE_FALLBACK`).
  • Setup B = the PR #124 ladder harness: a multi-period `run_period_episode` over the REAL trace requests
    through the persistent world simulator, req_cap 56, 3 decisions. gp/$ = episode goodput/$. (Read from the
    committed `ladder_benchmark.json`.)

It adds `production_scheduler` as an extra ARM to Setup A (task 2: "do not otherwise change the setup") so the
+1273% denominator can be compared against the production baseline too. Writes a JSON artifact with both
setups' rows + the numerator/denominator decomposition. Deterministic.

Usage: python -m scripts.diagnose_headline_reconciliation
"""

from __future__ import annotations

import json
import os

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_LADDER = os.path.join(_OUT, "ladder_benchmark.json")
_ARTIFACT = os.path.join(_OUT, "headline_reconciliation.json")

_MARKET, _WINDOW, _REQ_CAP_A, _BUDGET = "pjm", "expensive", 80, 100


def _bundle_from_action(d: dict):
    """production_scheduler's action dict → ActionBundle (same surfaces the ladder merges), defaults elsewhere."""
    from aurelius.environment.actions import ActionBundle
    return ActionBundle(
        capacity_policy=d.get("capacity", "reactive_lag1"),
        ordering_policy=d.get("ordering", "fifo"),
        admission_policy=d.get("admission", "off"),
        capacity_multiplier=float(d.get("capacity_multiplier", 1.0)),
        batching_policy=d.get("batching_policy", "conservative"),
        routing_policy=d.get("routing_policy", "round_robin"),
        placement_policy=d.get("placement_policy", "topology_blind"),
        prewarm_policy=d.get("prewarm_policy", "off"),
        precision_policy=d.get("precision_policy", "bf16"),
        clock_policy=d.get("clock_policy", "base"),
        migration_policy=d.get("migration_policy", "off"),
        spec_decode_policy=d.get("spec_decode_policy", "off"))


def _pct(cand, base):
    return round(100.0 * (cand - base) / base, 2) if base else None


def setup_a_single_decision_tournament() -> dict:
    """Reproduce the EXACT PR #123 pjm·expensive tournament scorer and score sla_aware + hierarchical_search +
    production_scheduler in it (the single-decision `_rollout_world` harness, req_cap 80)."""
    from aurelius.environment.physics_guided_candidates import SAFE_BASELINE_BUNDLE
    from aurelius.environment.planner.candidate_generators import named_anchor_keys
    from aurelius.environment.planner.planner_tournament import (
        market_window_scorer,
        run_tournament_window,
    )
    from aurelius.environment.production_baselines import ProductionScheduler
    from scripts.run_checkpointed_electricity_backtest import build_market, select_windows

    # the exact tournament scorer (req_cap 80, decision_index 0).
    scorer, state, meta = market_window_scorer(_MARKET, _WINDOW, req_cap=_REQ_CAP_A)
    period = meta["period"]

    # reproduce the committed hierarchical + baseline numbers (sanity: must match the JSON).
    wr = run_tournament_window(scorer, state, methods=["hierarchical_search"], budgets=[_BUDGET],
                               baseline_bundle=SAFE_BASELINE_BUNDLE, exhaustive_for_regret=False)
    base_gp, base_sla = scorer.gp_sla(SAFE_BASELINE_BUNDLE)
    hcell = wr["cells"][f"hierarchical_search|{_BUDGET}"]
    hreg = wr["regret"]["per_method"]["hierarchical_search"]

    # ADD production_scheduler as an arm: its causal action on this window's history, scored in the SAME harness.
    ctx = build_market(_MARKET, req_cap=_REQ_CAP_A, mooncake_limit=6000)
    wins = select_windows(ctx["prices"], ctx["n"], win_len=6, quick=False)
    win = wins.get(_WINDOW, next(iter(wins.values())))
    assert win[0] == period, f"window start {win[0]} != tournament period {period}"
    ps_action = ProductionScheduler().decide(ctx["frames"][:period])
    ps_bundle = _bundle_from_action(ps_action)
    ps_gp, ps_sla = scorer.gp_sla(ps_bundle)

    anchors = named_anchor_keys(None)
    rows = {
        "sla_aware": {"gp_per_dollar": round(base_gp, 1), "sla_violation_rate": round(base_sla, 4),
                      "selected_bundle": SAFE_BASELINE_BUNDLE.non_default_surfaces(),
                      "candidates_evaluated": 1, "is_baseline": True},
        "production_scheduler": {"gp_per_dollar": round(ps_gp, 1), "sla_violation_rate": round(ps_sla, 4),
                                 "selected_bundle": ps_bundle.non_default_surfaces(), "candidates_evaluated": 1},
        "aurelius_mpc_hierarchical_search": {
            "gp_per_dollar": hcell["gp_per_dollar"], "sla_violation_rate": hcell["sla_violation_rate"],
            "selected_bundle": hcell["selected_bundle"], "candidates_evaluated": hcell["candidates_evaluated"],
            "cpu_time_s": hcell["cpu_time_s"], "regret_pct": hreg["regret_pct"],
            "anchors_evaluated": hcell["anchors_evaluated"]},
    }
    # deltas vs each baseline, both abs and pct, with the Pareto clause.
    for arm in ("production_scheduler", "aurelius_mpc_hierarchical_search"):
        r = rows[arm]
        r["vs_sla_aware"] = {"abs": round(r["gp_per_dollar"] - base_gp, 1),
                             "pct": _pct(r["gp_per_dollar"], base_gp),
                             "sla_not_worse": r["sla_violation_rate"] <= base_sla + 1e-9}
    h = rows["aurelius_mpc_hierarchical_search"]
    h["vs_production_scheduler"] = {"abs": round(h["gp_per_dollar"] - ps_gp, 1),
                                    "pct": _pct(h["gp_per_dollar"], ps_gp),
                                    "sla_not_worse": h["sla_violation_rate"] <= ps_sla + 1e-9}
    return {"harness": "single_decision_forecast_rollout (_rollout_world, horizon_steps=1)",
            "req_cap": _REQ_CAP_A, "n_decisions": 1, "dt_seconds": 3600.0,
            "window": f"{_MARKET}|{_WINDOW}", "period": period, "median_prompt": meta["median_prompt"],
            "budget": _BUDGET, "named_anchor_count": len(anchors), "rows": rows,
            "anchors_note": "anchors evaluated; regret reference may lie outside the static core-grid set"}


def setup_b_ladder_episode() -> dict:
    """Read the committed PR #124 ladder rows for the SAME window (multi-period real-trace episode)."""
    if not os.path.exists(_LADDER):
        return {"error": "ladder_benchmark.json not found — run scripts.run_ladder_benchmark first"}
    d = json.load(open(_LADDER))
    arms = ("sla_aware", "production_scheduler", "aurelius_mpc_current_default",
            "aurelius_mpc_hierarchical_search", "oracle_diagnostic")
    rows = {}
    base_gp = base_sla = None
    ps_gp = ps_sla = None
    for arm in arms:
        cell = d["cells"].get(f"{_MARKET}|{_WINDOW}|{arm}", {})
        res = cell.get("result") or {}
        if not res:
            continue
        rows[arm] = {"gp_per_dollar": res.get("gp_per_dollar"), "sla_violation_rate": res.get("sla_violation_rate"),
                     "candidates_evaluated": (res.get("search") or {}).get("candidate_bundles_evaluated"),
                     "cpu_time_s": cell.get("seconds"),
                     "selected_bundle": {k: v for k, v in {
                         "routing_policy": _mix1(res.get("routing_mix")),
                         "placement_policy": _mix1(res.get("placement_mix")),
                         "precision_policy": _mix1(res.get("precision_mix")),
                         "clock_policy": _mix1(res.get("clock_mix")),
                         "batching_policy": _mix1(res.get("batching_mix")),
                         "capacity_multiplier": _mix1(res.get("capacity_multiplier_mix")),
                         "spec_decode_policy": _mix1(res.get("spec_decode_mix"))}.items() if v is not None}}
        if arm == "sla_aware":
            base_gp, base_sla = res.get("gp_per_dollar"), res.get("sla_violation_rate")
        if arm == "production_scheduler":
            ps_gp, ps_sla = res.get("gp_per_dollar"), res.get("sla_violation_rate")
    for arm, r in rows.items():
        if arm == "sla_aware" or r.get("gp_per_dollar") is None:
            continue
        if base_gp:
            r["vs_sla_aware"] = {"abs": round(r["gp_per_dollar"] - base_gp, 1), "pct": _pct(r["gp_per_dollar"], base_gp),
                                 "sla_not_worse": r["sla_violation_rate"] <= (base_sla or 0) + 1e-9}
        if ps_gp and arm != "production_scheduler":
            r["vs_production_scheduler"] = {"abs": round(r["gp_per_dollar"] - ps_gp, 1),
                                            "pct": _pct(r["gp_per_dollar"], ps_gp),
                                            "sla_not_worse": r["sla_violation_rate"] <= (ps_sla or 0) + 1e-9}
    cfg = d.get("config", {})
    return {"harness": "multi_period_episode (run_period_episode, persistent world simulator)",
            "req_cap": cfg.get("req_cap"), "n_decisions": cfg.get("max_decisions"), "dt_seconds": 3600.0,
            "window": f"{_MARKET}|{_WINDOW}", "rows": rows}


def _mix1(mix):
    """The single dominant key of a {value: count} mix (every period chose the same lever here)."""
    if not mix:
        return None
    return max(mix, key=mix.get)


def main() -> None:
    a = setup_a_single_decision_tournament()
    b = setup_b_ladder_episode()

    # the decomposition: why +1273% (A) vs +164% (B)?
    ha = a["rows"]["aurelius_mpc_hierarchical_search"]["gp_per_dollar"]
    ba = a["rows"]["sla_aware"]["gp_per_dollar"]
    decomp = {"setup_a_pct_vs_sla_aware": a["rows"]["aurelius_mpc_hierarchical_search"]["vs_sla_aware"]["pct"]}
    hb = b["rows"].get("aurelius_mpc_hierarchical_search", {}).get("gp_per_dollar")
    bb = b["rows"].get("sla_aware", {}).get("gp_per_dollar")
    if hb and bb:
        decomp.update({
            "setup_b_pct_vs_sla_aware": b["rows"]["aurelius_mpc_hierarchical_search"]["vs_sla_aware"]["pct"],
            "numerator_ratio_A_over_B": round(ha / hb, 3),       # how much higher hierarchical reads in A
            "denominator_ratio_B_over_A": round(bb / ba, 3),     # how much higher the baseline reads in B
            "ratio_of_ratios": round((ha / ba) / (hb / bb), 3),  # = numerator_ratio * denominator_ratio
            "explains": "the +1273% (A) vs +164% (B) gap = numerator_ratio × denominator_ratio; the SAME "
                        "sla_aware policy reads far LOWER in A's single forecast-rollout than in B's multi-"
                        "period real-trace episode, and hierarchical reads HIGHER in A — both are harness "
                        "effects (single idealized rollout @req_cap80 vs 3 real periods @req_cap56), not a "
                        "method regression."})
    out = {"setup_a_pr123_tournament": a, "setup_b_pr124_ladder": b, "decomposition": decomp,
           "note": "Diagnostic only. No physics/reward/gate/cost/action/baseline change. SIMULATED magnitudes."}
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(out, f, indent=2)

    print("=== Setup A (PR #123 single-decision tournament, req_cap 80) ===")
    for arm, r in a["rows"].items():
        d = r.get("vs_sla_aware", {})
        print(f"  {arm:34s} gp/$={r['gp_per_dollar']:>12} sla={r['sla_violation_rate']:.4f}"
              + (f"  vs_sla=+{d.get('pct')}% (safe={d.get('sla_not_worse')})" if d else "  [BASELINE]"))
    if "vs_production_scheduler" in a["rows"]["aurelius_mpc_hierarchical_search"]:
        v = a["rows"]["aurelius_mpc_hierarchical_search"]["vs_production_scheduler"]
        print(f"  hierarchical vs production_scheduler: +{v['pct']}% (safe={v['sla_not_worse']})")
    print("=== Setup B (PR #124 ladder episode, req_cap 56) ===")
    for arm, r in b["rows"].items():
        d = r.get("vs_sla_aware", {})
        print(f"  {arm:34s} gp/$={r.get('gp_per_dollar')!s:>12} sla={r.get('sla_violation_rate')}"
              + (f"  vs_sla=+{d.get('pct')}%" if d else "  [BASELINE]"))
    print("=== Decomposition ===")
    print("  " + json.dumps({k: v for k, v in decomp.items() if k != "explains"}))
    print(f"→ {_ARTIFACT}")


if __name__ == "__main__":
    main()
