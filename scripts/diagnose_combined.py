#!/usr/bin/env python3
"""Track E — combined re-run: token-shape forecaster + price-aware clock, one bounded diagnostic.

Four arms on the same bounded Azure eval window, full action layer, same Pareto gate:
  A  sla_aware            — strongest deployable SLA-aware baseline
  B  current_mpc          — current main MPC (single-median planning, default fleet price)
  C  tokenshape_mpc       — MPC with the token-shape scenario forecaster (this PR), default fleet price
  D  tokenshape_mpc_p90   — arm C re-priced at the market p90 electricity price (price-aware clock stress):
                            does expensive power shift the MPC's clock mix toward `low`? (full-MPC version of
                            the Track D question — clock is already a live action, so no new knob)

Reports per arm: SLA-safe gp/$, SLA, GPU-hours, GPU-seconds, energy(J), cost, and the precision / spec-decode /
clock mixes; plus C's improvement over A and over B, and the C→D clock-mix shift. Forecast attribution AFTER
the improvement is read from the Track C artifact. Diagnostic only — no reward shaping, no Pareto weakening.

Usage: python -m scripts.diagnose_combined --mpc-periods 6 --market pjm
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
from aurelius.environment.price_series import load_price_series, price_percentiles
from aurelius.environment.token_shape_forecaster import TokenShapeForecaster
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import (
    build_mpc_inputs,
    claim_gate,
    make_world_state,
    train_forecasters,
)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _row(rep):
    return {"gp_per_dollar": round(rep.goodput_per_dollar, 1),
            "sla_violation_rate": round(rep.sla_violation_rate, 5), "gpu_hours": round(rep.gpu_hours, 3),
            "gpu_seconds": round(rep.realized_gpu_seconds, 1),
            "operator_cost": round(rep.total_operator_cost, 4), "energy_j": round(rep.total_energy_j, 1),
            "precision_mix": rep.precision_mix, "spec_decode_mix": rep.spec_decode_mix,
            "clock_mix": rep.clock_mix}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpc-periods", type=int, default=6)
    ap.add_argument("--fit-window", type=int, default=8)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--market", default="pjm")
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=180.0, use_world_state=True,
                           control_dt_seconds=60.0)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames, per = (inp["common"], inp["fleet_state"], inp["cost_model"],
                                      inp["frames"], inp["per"])
    n = len(frames)
    mpc_ev = list(range(max(8, n - args.mpc_periods), n))
    pool = _mooncake_pool(args.mooncake_limit)
    if not pool:
        raise SystemExit("no Mooncake prefix pool available")
    kv = dict(kv_state_pool=pool, kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in mpc_ev for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    fm, _ = train_forecasters(frames, max(8, n - args.mpc_periods - 8))
    p90 = price_percentiles(load_price_series(args.market))["p90"]

    def _ep(name, decide_fn, fleet_state):
        ws = make_world_state(common.get("world_state_params"))
        return run_period_episode(name, decide_fn, per, frames, mpc_ev, fleet_state=fleet_state,
                                  cost_model=cm, world_state=ws, **common, **kv)

    def _mpc(kind, fleet_state):
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet_state, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        if kind == "tokenshape":
            c.planning_scenarios = True
        cur = {"i": 0}

        def _dec(h):
            p = mpc_ev[min(cur["i"], len(mpc_ev) - 1)]
            cur["i"] += 1
            if kind == "tokenshape":
                recent = [q for q in range(max(8, p - args.fit_window), p) if per.get(q)]
                c.scenario_builder = TokenShapeForecaster.fit(per, recent,
                                                              period_seconds=common["period_seconds"])
            return c.decide(h).to_dict()
        return _dec

    fleet_p90 = dataclasses.replace(fleet, energy_price_per_kwh=p90)
    arms = {
        "A_sla_aware": _ep("sla_aware", lambda h: dict(SLA_AWARE_FALLBACK), fleet),
        "B_current_mpc": _ep("current_mpc", _mpc("current", fleet), fleet),
        "C_tokenshape_mpc": _ep("tokenshape_mpc", _mpc("tokenshape", fleet), fleet),
        "D_tokenshape_mpc_p90": _ep("tokenshape_mpc_p90", _mpc("tokenshape", fleet_p90), fleet_p90),
    }
    rows = {k: _row(v) for k, v in arms.items()}
    a, b, c = (rows["A_sla_aware"]["gp_per_dollar"], rows["B_current_mpc"]["gp_per_dollar"],
               rows["C_tokenshape_mpc"]["gp_per_dollar"])
    gate_C = claim_gate({"mpc_controller": arms["C_tokenshape_mpc"], "sla_aware": arms["A_sla_aware"]})

    def _low_frac(mix):
        tot = sum(mix.values()) or 1
        return round(mix.get("low", 0) / tot, 3)
    summary = {
        "C_vs_A_baseline_pct": round(100.0 * (c - a) / a, 2) if a else None,
        "C_vs_B_current_pct": round(100.0 * (c - b) / b, 2) if b else None,
        "C_pareto_sla_not_worse": bool(gate_C["pareto_sla_not_worse"]),
        "clock_low_frac": {"C_default_price": _low_frac(rows["C_tokenshape_mpc"]["clock_mix"]),
                           "D_p90_price": _low_frac(rows["D_tokenshape_mpc_p90"]["clock_mix"])},
        "price_default": round(float(getattr(fleet, "energy_price_per_kwh", 0.0)), 5), "price_p90": round(p90, 5)}
    # forecast attribution AFTER the improvement (from the Track C artifact, if present)
    gap_art = os.path.join(_OUT, "token_shape_gap.json")
    attr_after = (json.load(open(gap_art)).get("attribution_after", {}).get("contributions_pct")
                  if os.path.exists(gap_art) else None)

    out = {"mpc_periods": len(mpc_ev), "market": args.market, "arms": rows, "summary": summary,
           "forecast_attribution_after": attr_after}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "combined.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("ARM                       gp/$       SLA     GPUh    energy(J)   clock_mix")
    for k, r in rows.items():
        print(f"  {k:<22} {r['gp_per_dollar']:>9.0f} {r['sla_violation_rate']:>8} {r['gpu_hours']:>7} "
              f"{r['energy_j']:>11} {r['clock_mix']}")
    print(f"C token-shape MPC vs A baseline: {summary['C_vs_A_baseline_pct']}%  | vs B current main: "
          f"{summary['C_vs_B_current_pct']}%  | Pareto-safe={summary['C_pareto_sla_not_worse']}")
    print(f"clock 'low' fraction: default-price={summary['clock_low_frac']['C_default_price']}  "
          f"p90-price={summary['clock_low_frac']['D_p90_price']}  "
          f"(price {summary['price_default']}→{summary['price_p90']} $/kWh)")
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
