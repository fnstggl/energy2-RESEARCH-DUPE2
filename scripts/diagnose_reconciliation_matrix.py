#!/usr/bin/env python3
"""Variable-isolation experiment matrix for the PR #123 (+1273%) vs PR #124 (+165%) reconciliation.

Scientific diagnostic ONLY — changes NOTHING (no simulator / reward / gate / cost / action / baseline / planner
edit, no tuning). It scores the SAME two FIXED action bundles (sla_aware == `SAFE_BASELINE_BUNDLE`; hierarchical
== the committed winning bundle) through the two existing evaluation harnesses, varying ONE experimental
variable at a time, to attribute the headline gap to each variable.

The two harnesses (both pre-existing, unchanged):
  • Harness A = PR #123 tournament: ONE planning decision via the controller's forecast rollout
    (`_rollout_world`, horizon_steps=1) — serves SYNTHETIC jobs drawn from the forecast distribution. gp/$ =
    the single rollout step's `gp_per_dollar`.
  • Harness B = PR #124 ladder: `run_period_episode` over REAL trace requests through the persistent world
    simulator. gp/$ = episode goodput/$.

Isolated variables: request_cap (80↔56), evaluation harness (A↔B, i.e. forecast-synthetic vs real-trace replay
+ state handling), episode horizon/aggregation (1↔3 periods, B only), electricity pricing (constant fleet price
↔ real diurnal prices). Holding the BUNDLE fixed means every gp/$ change is attributable to the measurement,
not to the planner choosing differently. Non-separable couplings are recorded, not forced.

Writes `data/external/mpc_controller/reconciliation_matrix.json`. Deterministic. ~2 market builds.

Usage: python -m scripts.diagnose_reconciliation_matrix
"""

from __future__ import annotations

import json
import os

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "reconciliation_matrix.json")
_MARKET, _WINDOW = "pjm", "expensive"


def _fixed_bundles():
    from aurelius.environment.actions import ActionBundle
    from aurelius.environment.physics_guided_candidates import SAFE_BASELINE_BUNDLE
    # the committed PR #123/#124 hierarchical winner on pjm·expensive (identical surfaces in both harnesses).
    hier = ActionBundle(capacity_policy="forecasted_mcs", ordering_policy="abs_conformal",
                        admission_policy="class_aware", routing_policy="kv_aware", capacity_multiplier=0.75,
                        batching_policy="aggressive", placement_policy="network_aware", precision_policy="fp8",
                        spec_decode_policy="aggressive", clock_policy="high")
    return {"sla_aware": SAFE_BASELINE_BUNDLE, "hierarchical": hier}


def _build(req_cap):
    from scripts.run_checkpointed_electricity_backtest import build_market, select_windows
    ctx = build_market(_MARKET, req_cap=req_cap, mooncake_limit=6000)
    wins = select_windows(ctx["prices"], ctx["n"], win_len=6, quick=False)
    ctx["win"] = wins.get(_WINDOW, next(iter(wins.values())))
    return ctx


def _harness_a(ctx, bundles, *, price_aware):
    """Harness A: the PR #123 single-decision forecast rollout. Replicates `market_window_scorer` exactly
    (build controller, forecast trajectory, `_rollout_world` H=1), optionally electricity-price-aware."""
    from aurelius.environment.forecast_trajectory import build_trajectory
    from aurelius.environment.simulation_clock import SimulationClock
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm, frames, per, fm = (ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"], ctx["per"],
                                          ctx["fm"])
    period = ctx["win"][0]
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for r in per.get(period, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    ws = make_world_state(common.get("world_state_params"))
    c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
    c.horizon_steps = 1
    c.planning_kv_cost_mode = "hybrid_capacity_work"
    c.planning_prompt_tokens = med_prompt
    c.electricity_price_aware = price_aware
    history = frames[:period]
    clock = SimulationClock(dt_seconds=c.period_seconds)
    traj = build_trajectory(c.forecasters, history, clock, 1, mode=c.uncertainty_mode)
    by_routing = c.kv_service_factor_by_routing or {}
    out = {}
    for name, ab in bundles.items():
        factor = by_routing.get(ab.routing_policy, c.kv_service_factor)
        _cum, steps = c._rollout_world(ab, traj, be=c.fleet_state.best_effort_fraction, factor=factor,
                                       horizon_steps=1)
        out[name] = {"gp_per_dollar": round(steps[0]["gp_per_dollar"], 1),
                     "sla_violation_rate": round(steps[0]["risk_viol"], 4)}
    return out, {"period": int(period), "median_prompt": med_prompt}


def _harness_b(ctx, bundles, *, periods, real_prices):
    """Harness B: the PR #124 multi-period real-trace episode (`run_period_episode`, persistent world)."""
    from aurelius.environment.controller import run_period_episode
    from aurelius.environment.training import make_world_state
    common, fleet, cm, frames, per, prices = (ctx["common"], ctx["fleet"], ctx["cm"], ctx["frames"], ctx["per"],
                                              ctx["prices"])
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    elec = prices if real_prices else None
    out = {}
    for name, ab in bundles.items():
        act = {"capacity": ab.capacity_policy, "ordering": ab.ordering_policy, "admission": ab.admission_policy,
               "capacity_multiplier": float(ab.capacity_multiplier), "batching_policy": ab.batching_policy,
               "routing_policy": ab.routing_policy, "placement_policy": ab.placement_policy,
               "prewarm_policy": ab.prewarm_policy, "migration_policy": ab.migration_policy,
               "precision_policy": ab.precision_policy, "spec_decode_policy": ab.spec_decode_policy,
               "clock_policy": ab.clock_policy}
        ws = make_world_state(common.get("world_state_params"))
        rep = run_period_episode(name, (lambda a: (lambda h: dict(a)))(act), per, frames, periods,
                                 fleet_state=fleet, cost_model=cm, world_state=ws, electricity_prices=elec,
                                 **common, **kv)
        out[name] = {"gp_per_dollar": round(rep.goodput_per_dollar, 1),
                     "sla_violation_rate": round(rep.sla_violation_rate, 4),
                     "operator_cost": round(rep.total_operator_cost, 5)}
    return out


def _pct(c, b):
    return round(100.0 * (c - b) / b, 2) if b else None


def _cell(label, rows, desc):
    h, s = rows["hierarchical"], rows["sla_aware"]
    return {"label": label, "desc": desc, "sla_aware": s, "hierarchical": h,
            "hier_vs_sla_aware_pct": _pct(h["gp_per_dollar"], s["gp_per_dollar"]),
            "hier_vs_sla_aware_abs": round(h["gp_per_dollar"] - s["gp_per_dollar"], 1),
            "hier_sla_not_worse": h["sla_violation_rate"] <= s["sla_violation_rate"] + 1e-9}


def main() -> None:
    b = _fixed_bundles()
    cells = {}
    ctx80 = _build(80)
    a80, meta = _harness_a(ctx80, b, price_aware=False)
    cells["E0_PR123_exact"] = _cell("E0", a80, "Harness A · req_cap 80 · 1 forecast-rollout period · const price (PR #123)")
    a80e, _ = _harness_a(ctx80, b, price_aware=True)
    cells["E_electricity_in_A"] = _cell("E_elec_A", a80e, "Harness A · req_cap 80 · 1 period · REAL price (only electricity changed)")
    b80_1 = _harness_b(ctx80, b, periods=[meta["period"]], real_prices=False)
    cells["E_harness_at_cap80_1period_const"] = _cell("E_harness", b80_1, "Harness B · req_cap 80 · 1 real-trace period · const price (only harness changed from E0)")
    b80_3 = _harness_b(ctx80, b, periods=list(range(meta["period"], meta["period"] + 3)), real_prices=False)
    cells["E_horizon_at_cap80_3period_const"] = _cell("E_horizon", b80_3, "Harness B · req_cap 80 · 3 real-trace periods · const price (only horizon/aggregation changed from E_harness)")
    del ctx80

    ctx56 = _build(56)
    a56, _ = _harness_a(ctx56, b, price_aware=False)
    cells["E_reqcap_in_A"] = _cell("E_reqcap_A", a56, "Harness A · req_cap 56 · 1 period · const price (only req_cap changed from E0)")
    b56_3 = _harness_b(ctx56, b, periods=list(range(meta["period"], meta["period"] + 3)), real_prices=False)
    cells["E_bridge_cap56_3period_const"] = _cell("E_bridge", b56_3, "Harness B · req_cap 56 · 3 periods · const price (req_cap 80→56 from E_horizon)")
    b56_3r = _harness_b(ctx56, b, periods=list(range(meta["period"], meta["period"] + 3)), real_prices=True)
    cells["E5_PR124_exact"] = _cell("E5", b56_3r, "Harness B · req_cap 56 · 3 periods · REAL price (PR #124)")
    del ctx56

    # marginal one-at-a-time-from-PR123 effects on the hierarchical-vs-sla_aware percent.
    base = cells["E0_PR123_exact"]["hier_vs_sla_aware_pct"]
    marginals = {
        "request_cap_80_to_56 (Harness A)": round(cells["E_reqcap_in_A"]["hier_vs_sla_aware_pct"] - base, 2),
        "electricity_const_to_real (Harness A)": round(cells["E_electricity_in_A"]["hier_vs_sla_aware_pct"] - base, 2),
        "harness_A_to_B (cap80, 1 period)": round(cells["E_harness_at_cap80_1period_const"]["hier_vs_sla_aware_pct"] - base, 2),
    }
    # cumulative bridge PR123 → PR124 (one variable per step; order: harness → horizon → req_cap → electricity).
    bridge = [
        ("PR123 (E0)", cells["E0_PR123_exact"]["hier_vs_sla_aware_pct"]),
        ("+ harness A→B (cap80,1p,const)", cells["E_harness_at_cap80_1period_const"]["hier_vs_sla_aware_pct"]),
        ("+ horizon 1→3 (cap80,const)", cells["E_horizon_at_cap80_3period_const"]["hier_vs_sla_aware_pct"]),
        ("+ req_cap 80→56 (3p,const)", cells["E_bridge_cap56_3period_const"]["hier_vs_sla_aware_pct"]),
        ("+ electricity const→real = PR124 (E5)", cells["E5_PR124_exact"]["hier_vs_sla_aware_pct"]),
    ]
    out = {"cells": cells, "marginal_effects_on_hier_vs_sla_pct_from_PR123": marginals,
           "cumulative_bridge_PR123_to_PR124": bridge,
           "non_separable_notes": [
               "episode horizon and cross-period aggregation are inseparable in run_period_episode (>1 period "
               "IS the aggregation); measured together as the horizon effect.",
               "request_cap acts differently per harness: in Harness A it shapes the forecast + median-prompt "
               "inputs (the rollout serves forecast-synthetic jobs, not the capped real requests); in Harness B "
               "it directly caps the served real requests. So the req_cap effect is harness-dependent, reported "
               "within each harness rather than as one cross-harness number.",
               "normalization (gp = goodput / operator_cost) is IDENTICAL in both harnesses; what differs is the "
               "goodput/cost BASIS (single forecast step vs real-trace episode), i.e. the harness, not a "
               "different normalization formula.",
               "the evaluation harness bundles workload-source (forecast-synthetic vs real-trace) + state "
               "handling (single clone vs persistent) + the replay path; these are one inseparable 'harness' "
               "variable, isolated as A→B at fixed req_cap and 1 period."],
           "note": "Diagnostic only. SIMULATED magnitudes. Same fixed bundles; no planner/sim/reward/baseline change."}
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(out, f, indent=2)

    print("=== cells (hierarchical gp/$ | sla_aware gp/$ | hier vs sla %) ===")
    for k, c in cells.items():
        print(f"  {k:36s} hier={c['hierarchical']['gp_per_dollar']:>11} sla={c['sla_aware']['gp_per_dollar']:>10} "
              f"→ +{c['hier_vs_sla_aware_pct']}%  (safe={c['hier_sla_not_worse']})")
    print("=== cumulative bridge PR123 → PR124 ===")
    for name, pct in bridge:
        print(f"  {name:42s} +{pct}%")
    print(f"→ {_ARTIFACT}")


if __name__ == "__main__":
    main()
