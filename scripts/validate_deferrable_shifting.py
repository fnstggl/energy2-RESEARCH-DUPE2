#!/usr/bin/env python3
"""Dedicated deferrable energy-shifting validation (Phase 5) — a clean, deadline-respecting saving.

The in-backtest deferrable ledger runs over only the 3-period serving-decision window, which is too short
for the look-ahead scheduler to find a strictly-cheaper period inside each job's deadline (it either finds
none → 0 saving, or shifts into a deadline miss). This script measures the SAME `deferrable.py` scheduler
over the full 24-hour real diurnal price path with AMPLE spare capacity, so the only difference between
`asap` and `price_aware` is WHEN a job runs (timing → price) — isolating the valid shifting saving.

Honest by construction (see deferrable.py): work is conserved; a missed deadline is penalised; deadline
safety dominates price; with a FLAT price `price_aware == asap` (no fake shifting). Effects flow only through
timing → energy_kWh × electricity_price → $. This is SIMULATOR_INFERENCE (no real deferrable trace exists).

Usage: python -m scripts.validate_deferrable_shifting
"""

from __future__ import annotations

import json
import os
import statistics

from aurelius.environment.deferrable import generate_deferrable_pool, run_deferrable_episode
from aurelius.environment.price_series import diurnal_profile, load_price_series, price_percentiles

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "deferrable_shifting_validation.json")
_MARKETS = ("pjm", "ercot", "caiso")
_N_JOBS = 24
_HORIZON = 24


def _episode(prices, policy):
    # ample spare everywhere → capacity never blocks; only timing→price differs between policies
    spare = {p: 1e9 for p in range(_HORIZON)}
    pool = generate_deferrable_pool(_N_JOBS, horizon_periods=_HORIZON)
    return run_deferrable_episode(pool, periods=list(range(_HORIZON)), prices=prices,
                                  spare_by_period=spare, policy=policy)


def validate_market(market):
    series = load_price_series(market)
    pct = price_percentiles(series)
    diurnal = diurnal_profile(series)                       # hour 0..23 → mean $/kWh
    real = {p: diurnal[p % 24] for p in range(_HORIZON)}
    flat = {p: statistics.mean(diurnal.values()) for p in range(_HORIZON)}

    asap = _episode(real, "asap")
    pa = _episode(real, "price_aware")
    flat_asap = _episode(flat, "asap")
    flat_pa = _episode(flat, "price_aware")

    saving = round(asap["electricity_cost"] - pa["electricity_cost"], 5)
    saving_pct = round(100.0 * saving / asap["electricity_cost"], 2) if asap["electricity_cost"] else 0.0
    return {
        "market": market, "provenance": "TRACE_DERIVED price path; SIMULATOR_INFERENCE deferrable workload",
        "price_p10": pct.get("p10"), "price_p90": pct.get("p90"), "price_mean": round(pct.get("mean", 0), 5),
        "real_price": {
            "asap_cost": asap["electricity_cost"], "asap_avg_price": asap["avg_price_paid"],
            "price_aware_cost": pa["electricity_cost"], "price_aware_avg_price": pa["avg_price_paid"],
            "shifting_saving_$": saving, "shifting_saving_pct": saving_pct,
            "shifted": pa["shifted"], "completed": pa["completed"], "missed": pa["missed"],
            "deadlines_respected": pa["missed"] == 0,
        },
        "flat_price_control": {
            "asap_cost": flat_asap["electricity_cost"], "price_aware_cost": flat_pa["electricity_cost"],
            "no_fake_shifting": abs(flat_asap["electricity_cost"] - flat_pa["electricity_cost"]) < 1e-9,
        },
        # a valid result: price_aware strictly cheaper than asap AND 0 missed AND flat shows no fake saving
        "valid_shifting_saving": bool(saving > 0 and pa["missed"] == 0
                                      and abs(flat_asap["electricity_cost"] - flat_pa["electricity_cost"]) < 1e-9),
    }


def main():
    results = {m: validate_market(m) for m in _MARKETS}
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(results, f, indent=2)
    for m, r in results.items():
        rp = r["real_price"]
        print(f"{m:6} p10={r['price_p10']} p90={r['price_p90']}  asap=${rp['asap_cost']} "
              f"price_aware=${rp['price_aware_cost']}  saving=${rp['shifting_saving_$']} "
              f"({rp['shifting_saving_pct']}%)  shifted={rp['shifted']} missed={rp['missed']}  "
              f"deadlines_respected={rp['deadlines_respected']}  valid={r['valid_shifting_saving']}")
    print(f"→ {_ARTIFACT}")


if __name__ == "__main__":
    main()
