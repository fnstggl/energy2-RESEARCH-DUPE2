#!/usr/bin/env python3
"""Electricity economic controller backtest (Phases B/E/7/F): real PJM/ERCOT/CAISO prices + DVFS + deferrable.

HOURLY cadence (``control_dt_seconds=3600``) so the real diurnal price varies period-to-period across the eval
window — the regime where electricity can actually become decision-relevant (PR #115 Track E showed a constant
price keeps it at 0%). Five arms on the SAME serving workload + Pareto gate:

  A1 flat_no_actions   — constant fleet price, no electricity actions      (current main behaviour)
  A2 real_no_actions   — real diurnal price in frames + realized cost, MPC not price-aware
  A3 real_dvfs         — A2 + price-aware DVFS (controller chooses clock against the forecast price path)
  A4 real_deferrable   — A2 + price-aware deferrable energy shifting (separate ledger vs run-asap)
  A5 real_dvfs_deferrable — A3 + price-aware deferrable shifting

The honest DVFS value is A3 − A2 (SAME prices, price-aware vs not). The honest shifting value is
price_aware − asap deferrable electricity cost (same jobs, same deadlines). No reward bonus; serving SLA
dominates; flat price ⇒ no fake value. Usage: python -m scripts.diagnose_electricity_controller --market pjm
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
from aurelius.environment.deferrable import generate_deferrable_pool, run_deferrable_episode
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


def _row(rep, prices, periods):
    energy_kwh = rep.total_energy_j / 3.6e6
    return {"gp_per_dollar": round(rep.goodput_per_dollar, 1), "sla_violation_rate": round(rep.sla_violation_rate, 5),
            "gpu_hours": round(rep.gpu_hours, 3), "energy_kwh": round(energy_kwh, 3),
            "operator_cost": round(rep.total_operator_cost, 4), "clock_mix": rep.clock_mix,
            "precision_mix": rep.precision_mix, "avg_price_paid": round(
                statistics.mean(prices.get(p, 0.0) for p in periods) if periods else 0.0, 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="pjm", choices=["pjm", "ercot", "caiso"])
    ap.add_argument("--mpc-periods", type=int, default=8)        # consecutive HOURS scored (price varies hourly)
    ap.add_argument("--req-cap", type=int, default=150)          # bounded requests/hour for tractable eval
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--n-deferrable", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # ONE hourly build (control_dt=3600 → cycle_len 24 → diurnal price varies period-to-period). The flat
    # baseline arm just passes electricity_prices=None (constant fleet price); no second heavy build needed.
    real_in = build_mpc_inputs(hourly_stride=1, sim_seconds=180.0, use_world_state=True,
                               control_dt_seconds=3600.0, electricity_market=args.market)
    if real_in is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames, per = (real_in["common"], real_in["fleet_state"], real_in["cost_model"],
                                      real_in["frames"], real_in["per"])
    elec = real_in["electricity"]
    prices = elec["prices"]
    n = len(frames)
    mpc_ev = list(range(max(8, n - args.mpc_periods), n))
    full_counts = {p: len(per.get(p, [])) for p in mpc_ev}      # serving pressure (BEFORE capping)
    # cap each hour's request volume for a tractable hourly replay (all arms share the cap → comparison valid;
    # absolute gp/$ is a BOUNDED-VOLUME sample, labelled in the doc). Planning is unaffected (synthetic window).
    per = {p: sorted(per.get(p, []), key=lambda r: r[0])[:args.req_cap] for p in per}
    pool = _mooncake_pool(args.mooncake_limit)
    if not pool:
        raise SystemExit("no Mooncake prefix pool available")
    kv = dict(kv_state_pool=pool, kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in mpc_ev for r in per.get(p, []))
    med_prompt = ins[len(ins) // 2] if ins else 512
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    fm, _ = train_forecasters(frames, max(8, n - args.mpc_periods - 8))

    def _ep(name, decide_fn, elec_prices):
        ws = make_world_state(common.get("world_state_params"))
        return run_period_episode(name, decide_fn, per, frames, mpc_ev, fleet_state=fleet, cost_model=cm,
                                  world_state=ws, electricity_prices=elec_prices, **common, **kv)

    def _mpc(price_aware):
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
        return _dec

    # --- serving arms ---------------------------------------------------------
    arms = {}
    arms["A1_flat_no_actions"] = _row(_ep("A1", _mpc(False), None), {p: fleet.energy_price_per_kwh for p in mpc_ev}, mpc_ev)
    arms["A2_real_no_actions"] = _row(_ep("A2", _mpc(False), prices), prices, mpc_ev)
    arms["A3_real_dvfs"] = _row(_ep("A3", _mpc(True), prices), prices, mpc_ev)

    # --- deferrable energy shifting (separate ledger; serving dominates) ------
    # spare GPU-seconds per period: ample on low-arrival hours, 0 on the busiest quartile (serving protected).
    # Uses FULL serving pressure (pre-cap counts), not the bounded eval sample.
    busy = sorted(full_counts.values())
    busy_cut = busy[int(0.75 * (len(busy) - 1))] if busy else 0
    spare = {p: (0.0 if full_counts[p] >= busy_cut and busy_cut > 0 else 1e7) for p in mpc_ev}

    def _defer(policy):
        st = generate_deferrable_pool(args.n_deferrable, horizon_periods=len(mpc_ev))
        # remap job periods into the eval window indices [0, len(mpc_ev))
        return run_deferrable_episode(st, periods=list(range(len(mpc_ev))),
                                      prices={i: prices.get(p, 0.06) for i, p in enumerate(mpc_ev)},
                                      spare_by_period={i: spare[p] for i, p in enumerate(mpc_ev)}, policy=policy)
    defer_pa, defer_as = _defer("price_aware"), _defer("asap")

    # --- Pareto gate: each real arm vs sla_aware on the SAME (real) prices ----
    def _baseline():
        return _ep("sla_aware", lambda h: dict(SLA_AWARE_FALLBACK), prices)
    base_rep = _baseline()
    from types import SimpleNamespace
    gate_inputs = {"A3_real_dvfs": arms["A3_real_dvfs"], "A2_real_no_actions": arms["A2_real_no_actions"]}
    gates = {k: claim_gate({"mpc_controller": SimpleNamespace(goodput_per_dollar=v["gp_per_dollar"],
                                                              sla_violation_rate=v["sla_violation_rate"]),
                            "sla_aware": SimpleNamespace(goodput_per_dollar=base_rep.goodput_per_dollar,
                                                         sla_violation_rate=base_rep.sla_violation_rate)})
             for k, v in gate_inputs.items()}

    def _lowfrac(mix):
        t = sum(mix.values()) or 1
        return round(mix.get("low", 0) / t, 3)
    summary = {
        "market": args.market, "electricity_provenance": elec["provenance"],
        "price_min": round(min(prices.get(p, 0) for p in mpc_ev), 5),
        "price_max": round(max(prices.get(p, 0) for p in mpc_ev), 5),
        "dvfs_value_A3_minus_A2_gp$": round(arms["A3_real_dvfs"]["gp_per_dollar"] - arms["A2_real_no_actions"]["gp_per_dollar"], 1),
        "dvfs_clock_low_frac": {"A2_not_price_aware": _lowfrac(arms["A2_real_no_actions"]["clock_mix"]),
                                "A3_price_aware": _lowfrac(arms["A3_real_dvfs"]["clock_mix"])},
        "dvfs_A3_pareto_safe": bool(gates["A3_real_dvfs"]["pareto_sla_not_worse"]),
        "deferrable": {"price_aware": defer_pa, "asap": defer_as,
                       "shifting_saving": round(defer_as["electricity_cost"] - defer_pa["electricity_cost"], 5),
                       "deadlines_respected": defer_pa["missed"] == 0},
        "electricity_decision_relevant": (_lowfrac(arms["A3_real_dvfs"]["clock_mix"]) >
                                          _lowfrac(arms["A2_real_no_actions"]["clock_mix"])) or
                                         (defer_as["electricity_cost"] > defer_pa["electricity_cost"]),
    }
    out = {"market": args.market, "mpc_periods": len(mpc_ev), "arms": arms, "summary": summary,
           "baseline_sla_aware_gp$": round(base_rep.goodput_per_dollar, 1)}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, f"electricity_{args.market}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"MARKET {args.market} ({elec['provenance']}) price ${summary['price_min']}–${summary['price_max']}/kWh")
    for k, v in arms.items():
        print(f"  {k:<22} gp$={v['gp_per_dollar']:>8} sla={v['sla_violation_rate']:<7} "
              f"energy_kWh={v['energy_kwh']:<7} cost={v['operator_cost']:<8} clock={v['clock_mix']}")
    print(f"DVFS value A3−A2: {summary['dvfs_value_A3_minus_A2_gp$']} gp/$ | clock low frac {summary['dvfs_clock_low_frac']} "
          f"| Pareto-safe={summary['dvfs_A3_pareto_safe']}")
    d = summary["deferrable"]
    print(f"DEFERRABLE shifting saving (asap−price_aware): ${d['shifting_saving']} | price_aware avg ${defer_pa['avg_price_paid']} "
          f"vs asap ${defer_as['avg_price_paid']} | deadlines_respected={d['deadlines_respected']}")
    print(f"ELECTRICITY decision-relevant: {summary['electricity_decision_relevant']}")
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
