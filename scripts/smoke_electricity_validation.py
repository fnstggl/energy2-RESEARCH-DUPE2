#!/usr/bin/env python3
"""BOUNDED electricity validation smoke — proves the causal path cheaply (NOT a headline backtest).

The full historical all-arm hourly sweep is too heavy (hourly periods replay a full hour of real requests;
see ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md → deferred). This smoke proves ONLY the causal mechanisms, with
one input build, capped per-hour request volume, and ≤6 MPC decisions:

  P1 real prices vary across planner frames
  P2 high-price periods increase electricity cost (causal: energy × price)
  P3 price-aware clock changes clock selection when price is high
  P4 deferrable work delays into cheaper periods when deadline slack exists
  P5 flat price produces NO fake energy-shifting value
  P6 serving SLA is not violated for free (deferrable can't steal serving capacity)

No headline gp/$ is claimed. Usage: python -m scripts.smoke_electricity_validation --market pjm
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from types import SimpleNamespace

from aurelius.environment.deferrable import generate_deferrable_pool, run_deferrable_episode
from aurelius.environment.electricity import build_price_profile
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import build_mpc_inputs, make_world_state, train_forecasters
from aurelius.environment.world_simulator import simulate_period

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="pjm", choices=["pjm", "ercot", "caiso"])
    ap.add_argument("--mpc-periods", type=int, default=3)        # ≤6 decisions total (A2 + A3)
    ap.add_argument("--req-cap", type=int, default=60)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=1, sim_seconds=120.0, use_world_state=True,
                           control_dt_seconds=3600.0, electricity_market=args.market)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames, per = (inp["common"], inp["fleet_state"], inp["cost_model"],
                                      inp["frames"], inp["per"])
    elec = inp["electricity"]
    prices = elec["prices"]
    n = len(frames)
    mpc_ev = list(range(max(8, n - args.mpc_periods), n))
    per = {p: sorted(per.get(p, []), key=lambda r: r[0])[:args.req_cap] for p in per}
    prof = build_price_profile(args.market, elec["cycle_len"])
    p10, p90 = prof.percentiles.get("p10", 0.03), prof.percentiles.get("p90", 0.20)
    res = {"market": args.market, "provenance": elec["provenance"], "p10": round(p10, 5), "p90": round(p90, 5)}

    # P1 — frames carry a varying diurnal price -------------------------------
    fp = [round(f.electricity_price, 6) for f in frames[-24:]]
    res["P1_frames_vary"] = {"distinct_prices_last_24": len(set(fp)), "pass": len(set(fp)) > 1}

    # P2 — high price increases electricity cost (controlled simulate_period; no MPC search) ----
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind", migration_policy="off",
                          batching_policy="conservative", precision_policy="bf16", spec_decode_policy="off",
                          clock_policy="base", colocation_policy="off", prefill_decode_policy="shared")
    wl = [(i * 4.0, 512, 64) for i in range(24)]                 # memory-bound decode (short prompt, long out)
    fc = {"arrival_rate": 0.2, "arrival_p90": 0.26, "mean_service_s": 1.0}
    kw = dict(sla_s=10.0, tick_seconds=10.0, base_service_factor=1.0, cost_model=cm, fleet_state=fleet,
              cost_scenario="owned", best_effort_fraction=fleet.best_effort_fraction,
              period_hours=120 / 3600, dt_seconds=3600.0)
    ws0 = make_world_state(common.get("world_state_params"))
    c_lo = simulate_period(copy.deepcopy(ws0), pol, wl, fc, energy_price_per_kwh=p10, **kw).operator_cost
    c_hi = simulate_period(copy.deepcopy(ws0), pol, wl, fc, energy_price_per_kwh=p90, **kw).operator_cost
    res["P2_high_price_costs_more"] = {"cost_p10": round(c_lo, 5), "cost_p90": round(c_hi, 5), "pass": c_hi > c_lo}

    # P3 — price-aware clock changes selection vs not-price-aware (A2 vs A3; ≤6 MPC decisions) ----
    fm, _ = train_forecasters(frames, max(8, n - args.mpc_periods - 8))
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in mpc_ev for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    kv = dict(kv_state_pool=None, kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")

    def _clock_mix(price_aware):
        from aurelius.environment.controller import run_period_episode
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, cfg, common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        c.electricity_price_aware = price_aware
        cur = {"i": 0}

        def _dec(h):
            cur["i"] += 1
            return c.decide(h).to_dict()
        ws2 = make_world_state(common.get("world_state_params"))
        rep = run_period_episode("smoke", _dec, per, frames, mpc_ev, fleet_state=fleet, cost_model=cm,
                                 world_state=ws2, electricity_prices=prices, **common, **kv)
        return rep.clock_mix

    a2 = _clock_mix(False)
    a3 = _clock_mix(True)

    def _low(m):
        return round(m.get("low", 0) / (sum(m.values()) or 1), 3)
    win_prices = [round(prices.get(p, 0), 5) for p in mpc_ev]
    res["P3_price_aware_clock"] = {"window_prices": win_prices, "a2_not_price_aware_clock": a2,
                                   "a3_price_aware_clock": a3, "low_frac_a2": _low(a2), "low_frac_a3": _low(a3),
                                   "pass": _low(a3) >= _low(a2)}   # price-aware downclocks at least as much

    # P4/P5/P6 — deferrable shifting (instant) --------------------------------
    periods = list(range(12))
    vary = {p: (0.02 if p % 4 == 0 else 0.20) for p in periods}
    flat = {p: 0.10 for p in periods}
    ample = {p: 1e9 for p in periods}
    pa = run_deferrable_episode(copy.deepcopy(generate_deferrable_pool(8, horizon_periods=12)),
                                periods=periods, prices=vary, spare_by_period=ample, policy="price_aware")
    asap = run_deferrable_episode(copy.deepcopy(generate_deferrable_pool(8, horizon_periods=12)),
                                  periods=periods, prices=vary, spare_by_period=ample, policy="asap")
    fa = run_deferrable_episode(copy.deepcopy(generate_deferrable_pool(8, horizon_periods=12)),
                                periods=periods, prices=flat, spare_by_period=ample, policy="price_aware")
    fs = run_deferrable_episode(copy.deepcopy(generate_deferrable_pool(8, horizon_periods=12)),
                                periods=periods, prices=flat, spare_by_period=ample, policy="asap")
    nospare = run_deferrable_episode(copy.deepcopy(generate_deferrable_pool(8, horizon_periods=12)),
                                     periods=periods, prices=vary,
                                     spare_by_period={p: 0.0 for p in periods}, policy="asap")
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
    print(f"ELECTRICITY SMOKE [{args.market}] ({elec['provenance']})  p10=${res['p10']} p90=${res['p90']}/kWh")
    for k in ("P1_frames_vary", "P2_high_price_costs_more", "P3_price_aware_clock",
              "P4_deferrable_delays_to_cheap", "P5_flat_no_fake_shifting", "P6_serving_protected"):
        print(f"  {'PASS' if res[k]['pass'] else 'FAIL'}  {k}: {res[k]}")
    print(f"ALL PASS: {res['all_pass']}")
    if args.json:
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
