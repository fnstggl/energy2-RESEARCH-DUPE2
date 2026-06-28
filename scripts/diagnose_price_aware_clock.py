#!/usr/bin/env python3
"""Track D — price-aware clock/power shaping diagnostic (bounded, causal, Pareto-gated).

PR #114 measured electricity_price at 0.0% planner-value — but only because the Azure window feeds a CONSTANT
fleet price (~$0.06/kWh) and at that level energy is a small share of cost (depreciation dominates), so the
planner never trades latency for trivial energy savings. The causal pathway already exists and is wired
end-to-end: clock → power_factor (power_w = TDP·(0.4+0.6·clock^2.4)) → energy_j → operator_cost via
`energy = gpu_hours · power_kw · power_scale · pue · energy_price_per_kwh`, AND clock → decode/prefill service
time → completion latency → SLA. This diagnostic DRIVES that path with REAL day-ahead prices
(`price_series`, PJM/ERCOT/CAISO) across the required regimes and asks: when is downclocking memory-bound
decode Pareto-safe (gp/$ up AND SLA not worse)?

For each scenario (price level × SLA slack × decode-heavy vs prefill-heavy workload) we sweep
clock ∈ {base, low, high} through `run_period_episode` (no MPC search — a constant-clock decider, so ONLY the
clock varies) and report gp/$, energy(J), operator cost ($), SLA-violation rate, power (W), and the measured
roofline regime. Verdict per scenario: the Pareto-safe gp/$-optimal clock, and whether downclock is Pareto-safe.

Honesty: no new knob (the clock action already supports this end-to-end); no reward bonus; a saving is only
reported when operator cost falls through actual electricity_price × energy AND SLA is not worse. EIA / 5-min
real-time prices are NOT wired (`price_series.ABSENT_MARKETS`) and are not fabricated.

Usage: python -m scripts.diagnose_price_aware_clock --market pjm
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

from aurelius.environment.controller import run_period_episode
from aurelius.environment.price_series import (
    ABSENT_MARKETS,
    diurnal_profile,
    load_price_series,
    price_percentiles,
)
from aurelius.environment.training import build_mpc_inputs, make_world_state

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_CLOCKS = ("base", "low", "high")


def _workload(kind, n_req, window_s):
    """Controlled workload records (arrival_s, output_tokens, prompt_tokens), uniform arrivals."""
    if kind == "decode":          # decode-heavy → memory-bandwidth-bound (short prompt, long output)
        out, prompt = 512, 64
    else:                         # prefill-heavy → higher arithmetic intensity (long prompt, short output)
        out, prompt = 16, 4096
    return [(i * window_s / max(1, n_req), out, prompt) for i in range(n_req)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="pjm", choices=sorted(["pjm", "ercot", "caiso"]))
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--n-req", type=int, default=60)   # enough requests for fine SLA-rate resolution
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=180.0, use_world_state=True,
                           control_dt_seconds=60.0)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    n = len(frames)
    window_s = float(common.get("period_seconds", 60.0))
    idx = n - 1

    pct = {m: price_percentiles(load_price_series(m)) for m in ("pjm", "ercot", "caiso")}
    series = load_price_series(args.market)
    p = price_percentiles(series)
    price_cheap, price_exp, price_p95 = p["p10"], p["p90"], p["p95"]
    fleet_price = float(getattr(fleet, "energy_price_per_kwh", 0.06))

    def _episode(kind, price, sla_s, clock):
        per_local = {idx - 1: _workload(kind, args.n_req, window_s),
                     idx: _workload(kind, args.n_req, window_s)}
        fleet_p = dataclasses.replace(fleet, energy_price_per_kwh=price)
        ws = make_world_state(common.get("world_state_params"))
        common2 = {k: v for k, v in common.items() if k != "sla_s"}

        def _dec(h):
            return {"clock_policy": clock}
        rep = run_period_episode(f"{kind}_{clock}", _dec, per_local, frames, [idx], fleet_state=fleet_p,
                                 cost_model=cm, world_state=ws, sla_s=sla_s, **common2)
        rd = rep.decode_regime_mix or {}
        regime = max(rd, key=rd.get) if rd else "unknown"
        return {"gp_per_dollar": round(rep.goodput_per_dollar, 1),
                "sla_violation_rate": round(rep.sla_violation_rate, 5),
                "energy_j": round(rep.total_energy_j, 1), "operator_cost": round(rep.total_operator_cost, 6),
                "power_w": round(rep.mean_power_w, 1), "decode_regime": regime,
                "decode_arith_intensity": round(rep.mean_decode_arithmetic_intensity, 3),
                "clock_mix": rep.clock_mix}

    # adaptive SLA: from the decode workload at base clock, set slack (headroom) and tight (pressure) targets
    probe = _episode("decode", fleet_price, 1e9, "base")  # huge SLA → measure raw completion via gp/$ ref
    # use the simulator's own completion percentile by probing a sweep of sla_s on base clock
    def _sla_levels(kind):
        # find an sla that yields ~0 violations (slack) and one with material violations (tight)
        slack = tight = None
        for s in (40.0, 30.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0):
            v = _episode(kind, fleet_price, s, "base")["sla_violation_rate"]
            if v <= 0.001 and slack is None:
                slack = s
            if v >= 0.1 and tight is None:
                tight = s
        return (slack or 40.0), (tight or 4.0)
    slack_d, tight_d = _sla_levels("decode")
    slack_p, tight_p = _sla_levels("prefill")

    scenarios = [
        ("expensive_slack_decode", "decode", price_exp, slack_d),
        ("expensive_tight_decode", "decode", price_exp, tight_d),
        ("cheap_slack_decode", "decode", price_cheap, slack_d),
        ("cheap_tight_decode", "decode", price_cheap, tight_d),
        ("expensive_slack_prefill", "prefill", price_exp, slack_p),
        ("expensive_tight_prefill", "prefill", price_exp, tight_p),
    ]
    results = {}
    for name, kind, price, sla_s in scenarios:
        sweep = {clk: _episode(kind, price, sla_s, clk) for clk in _CLOCKS}
        base = sweep["base"]
        # Pareto-safe optimum: max gp/$ among clocks whose SLA is not worse than base
        safe = {c: r for c, r in sweep.items() if r["sla_violation_rate"] <= base["sla_violation_rate"] + 1e-9}
        opt = max(safe, key=lambda c: safe[c]["gp_per_dollar"]) if safe else "base"
        low, hi = sweep["low"], sweep["high"]
        downclock_pareto_safe = (low["gp_per_dollar"] > base["gp_per_dollar"]
                                 and low["sla_violation_rate"] <= base["sla_violation_rate"] + 1e-9)
        results[name] = {
            "price_per_kwh": round(price, 6), "sla_s": sla_s, "workload": kind,
            # the roofline regime/AI is read from the non-neutral `low` run (the neutral base run leaves the
            # roofline diagnostic off, so it would read "unknown")
            "decode_regime": low["decode_regime"], "decode_arith_intensity": low["decode_arith_intensity"],
            "sweep": sweep, "pareto_safe_optimal_clock": opt,
            "downclock_pareto_safe": bool(downclock_pareto_safe),
            "downclock_cost_delta": round(low["operator_cost"] - base["operator_cost"], 6),
            "downclock_energy_delta_j": round(low["energy_j"] - base["energy_j"], 1),
            "downclock_sla_delta": round(low["sla_violation_rate"] - base["sla_violation_rate"], 5),
            "downclock_gp_delta": round(low["gp_per_dollar"] - base["gp_per_dollar"], 1)}

    # causality check: at FIXED base clock, does operator cost rise with electricity price? (price × energy)
    cz = {p_lbl: _episode("decode", pv, slack_d, "base")["operator_cost"]
          for p_lbl, pv in (("p10", price_cheap), ("p90", price_exp), ("p95", price_p95))}
    cost_rises_with_price = cz["p10"] < cz["p90"] <= cz["p95"] + 1e-9
    # at FIXED expensive price, does lower clock cut energy AND power? (power_factor × energy)
    # NOTE: the energy_j/power_w DIAGNOSTIC is only populated under non-neutral roofline actions, so the
    # neutral base run reads 0 there (its energy is still booked in operator_cost at power_scale=1.0). We
    # therefore demonstrate clock→power→energy on the two NON-neutral clocks (low vs high), both populated.
    e_low = _episode("decode", price_exp, slack_d, "low")
    e_high = _episode("decode", price_exp, slack_d, "high")
    downclock_cuts_energy = e_low["energy_j"] < e_high["energy_j"] and e_low["power_w"] < e_high["power_w"]

    out = {"market": args.market, "fleet_constant_price": round(fleet_price, 6),
           "price_percentiles": pct, "diurnal_profile_market": diurnal_profile(series),
           "absent_markets": ABSENT_MARKETS, "probe_base_sla": probe["sla_violation_rate"],
           "sla_levels": {"decode": [slack_d, tight_d], "prefill": [slack_p, tight_p]},
           "scenarios": results,
           "causality": {"cost_vs_price_base_clock": cz, "cost_rises_with_price": bool(cost_rises_with_price),
                         "low_vs_high_energy_j": [e_low["energy_j"], e_high["energy_j"]],
                         "low_vs_high_power_w": [e_low["power_w"], e_high["power_w"]],
                         "downclock_cuts_energy": bool(downclock_cuts_energy)}}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "price_aware_clock.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"MARKET {args.market}  cheap(p10)={price_cheap:.4f}  expensive(p90)={price_exp:.4f}  "
          f"p95={price_p95:.4f} $/kWh  (fleet constant={fleet_price:.4f})")
    print(f"CAUSALITY: cost rises with price (base clock, p10<p90<p95): {cz}  -> {cost_rises_with_price}")
    print(f"CAUSALITY: low<high energy+power: {out['causality']['low_vs_high_energy_j']} J, "
          f"{out['causality']['low_vs_high_power_w']} W -> {downclock_cuts_energy}")
    for name, r in results.items():
        print(f"  [{name:>24}] regime={r['decode_regime']:<22} opt_clock={r['pareto_safe_optimal_clock']:<5} "
              f"downclock_pareto_safe={r['downclock_pareto_safe']}  "
              f"(Δcost={r['downclock_cost_delta']:+.5f} Δsla={r['downclock_sla_delta']:+.4f} "
              f"Δgp$={r['downclock_gp_delta']:+.0f})")
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
