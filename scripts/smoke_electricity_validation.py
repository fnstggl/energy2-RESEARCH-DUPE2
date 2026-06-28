#!/usr/bin/env python3
"""FIXTURE-level electricity validation smoke — proves the causal path in SECONDS (no headline backtest).

The full historical all-arm hourly sweep AND even a few live MPC decisions are too heavy at hourly cadence
(the planning workload scales with the hourly arrival rate × the candidate search; see
ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md → deferred). This smoke instead proves the causal MECHANISMS at the
fixture level — direct `simulate_period` calls + the deferrable scheduler, no `build_mpc_inputs`, no MPC search:

  P1 the real diurnal price path that feeds the planner frames VARIES
  P2 a high electricity price increases operator cost (causal: energy × price)
  P3 the gp/$-optimal clock is price-dependent: at a high price, downclocking memory-bound decode is the
     Pareto-safe gp/$ winner (so a price-aware planner selects it) — its advantage shrinks at a low price
  P4 deferrable work shifts into cheaper periods under deadline slack (price_aware < asap cost)
  P5 a flat price produces NO fake energy-shifting value (price_aware == asap cost)
  P6 serving SLA is not violated for free (no spare ⇒ deferrable defers, never steals serving capacity)

No headline gp/$ is claimed. The full-MPC confirmation of P3 (downclock fraction 0.0→0.5 at PJM p90) is the
independent PR #115 Track D/E evidence. Usage: python -m scripts.smoke_electricity_validation --market pjm
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from types import SimpleNamespace

from aurelius.environment.cost_model import CostModel
from aurelius.environment.deferrable import generate_deferrable_pool, run_deferrable_episode
from aurelius.environment.electricity import build_price_profile
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.training import make_world_state
from aurelius.environment.world_simulator import simulate_period

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="pjm", choices=["pjm", "ercot", "caiso"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    fleet = V2026FleetPlane().state_at(0)
    cm = CostModel()
    wsp = {"n_servers": 24, "n_racks": 4, "seed": 0, "warm": 8, "processed_dir": None}
    prof = build_price_profile(args.market, 24)
    p10, p90 = prof.percentiles.get("p10", 0.03), prof.percentiles.get("p90", 0.20)
    res = {"market": args.market, "provenance": prof.provenance, "p10": round(p10, 5), "p90": round(p90, 5)}

    # P1 — the diurnal price path that feeds planner frames varies --------------
    distinct = len(set(round(v, 6) for v in prof.by_cycle.values()))
    res["P1_price_path_varies"] = {"distinct_hourly_prices": distinct, "pass": distinct > 1}

    # controlled memory-bound decode workload (short prompt, long output) -------
    wl = [(i * 4.0, 512, 64) for i in range(24)]
    fc = {"arrival_rate": 0.2, "arrival_p90": 0.26, "mean_service_s": 1.0}
    kw = dict(sla_s=12.0, tick_seconds=10.0, base_service_factor=1.0, cost_model=cm, fleet_state=fleet,
              cost_scenario="owned", best_effort_fraction=fleet.best_effort_fraction,
              period_hours=180 / 3600, dt_seconds=3600.0)

    def _sim(clock, price):
        ws = make_world_state(wsp)
        pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind", migration_policy="off",
                              batching_policy="conservative", precision_policy="bf16", spec_decode_policy="off",
                              clock_policy=clock, colocation_policy="off", prefill_decode_policy="shared")
        return simulate_period(ws, pol, wl, fc, energy_price_per_kwh=price, **kw)

    # P2 — high price increases cost (clock fixed at base) ---------------------
    c_lo, c_hi = _sim("base", p10).operator_cost, _sim("base", p90).operator_cost
    res["P2_high_price_costs_more"] = {"cost_p10": round(c_lo, 5), "cost_p90": round(c_hi, 5), "pass": c_hi > c_lo}

    # P3 — gp/$-optimal clock is price-dependent (downclock wins at high price, Pareto-safe) -----
    def _pareto_safe_low(price):
        b, lo = _sim("base", price), _sim("low", price)
        return {"gp_base": round(b.goodput_per_dollar, 1), "gp_low": round(lo.goodput_per_dollar, 1),
                "sla_base": round(b.sla_violation_rate, 5), "sla_low": round(lo.sla_violation_rate, 5),
                "low_wins_pareto_safe": lo.goodput_per_dollar > b.goodput_per_dollar
                and lo.sla_violation_rate <= b.sla_violation_rate + 1e-9,
                "gp_gain_low": round(lo.goodput_per_dollar - b.goodput_per_dollar, 1)}
    at_hi, at_lo = _pareto_safe_low(p90), _pareto_safe_low(p10)
    res["P3_price_aware_clock"] = {"at_p90": at_hi, "at_p10": at_lo,
                                   # downclock is a Pareto-safe gp/$ win at the HIGH price, and its advantage is
                                   # strictly larger at p90 than p10 → the price-aware planner downclocks more.
                                   "pass": at_hi["low_wins_pareto_safe"] and at_hi["gp_gain_low"] > at_lo["gp_gain_low"]}

    # P4/P5/P6 — deferrable shifting ------------------------------------------
    periods = list(range(12))
    vary = {p: (0.02 if p % 4 == 0 else 0.20) for p in periods}
    flat = {p: 0.10 for p in periods}
    ample = {p: 1e9 for p in periods}

    def _defer(prices, spare, policy):
        return run_deferrable_episode(copy.deepcopy(generate_deferrable_pool(8, horizon_periods=12)),
                                      periods=periods, prices=prices, spare_by_period=spare, policy=policy)
    pa, asap = _defer(vary, ample, "price_aware"), _defer(vary, ample, "asap")
    fa, fs = _defer(flat, ample, "price_aware"), _defer(flat, ample, "asap")
    nospare = _defer(vary, {p: 0.0 for p in periods}, "asap")
    res["P4_deferrable_delays_to_cheap"] = {"price_aware_cost": pa["electricity_cost"], "asap_cost": asap["electricity_cost"],
                                            "price_aware_avg_price": pa["avg_price_paid"], "shifted": pa["shifted"],
                                            "pass": pa["electricity_cost"] < asap["electricity_cost"] and pa["missed"] == 0}
    res["P5_flat_no_fake_shifting"] = {"price_aware_cost": fa["electricity_cost"], "asap_cost": fs["electricity_cost"],
                                       "pass": abs(fa["electricity_cost"] - fs["electricity_cost"]) < 1e-9}
    res["P6_serving_protected"] = {"no_spare_completed": nospare["completed"], "no_spare_missed": nospare["missed"],
                                   "pass": nospare["completed"] == 0 and nospare["missed"] > 0}

    res["all_pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, f"electricity_smoke_{args.market}.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"ELECTRICITY SMOKE [{args.market}] ({prof.provenance})  p10=${res['p10']} p90=${res['p90']}/kWh")
    for k in ("P1_price_path_varies", "P2_high_price_costs_more", "P3_price_aware_clock",
              "P4_deferrable_delays_to_cheap", "P5_flat_no_fake_shifting", "P6_serving_protected"):
        print(f"  {'PASS' if res[k]['pass'] else 'FAIL'}  {k}: {res[k]}")
    print(f"ALL PASS: {res['all_pass']}")
    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
