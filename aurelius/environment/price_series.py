"""Real electricity price series (PJM / ERCOT / CAISO) for the price-aware clock/power diagnostic.

Loads the committed day-ahead LMP/SPP series already in ``data/`` (hourly, $/MWh) and exposes them in $/kWh
with percentile + diurnal helpers, so the price-aware clock diagnostic is driven by REAL prices — never
fabricated. Markets that are NOT wired into the environment are listed in ``ABSENT_MARKETS`` and must not be
synthesised.

Units: source CSVs are ``price_per_mwh`` (USD/MWh); we divide by 1000 to the $/kWh the CostModel consumes
(``energy = gpu_hours · power_kw · power_scale · pue · energy_price_per_kwh``).
"""

from __future__ import annotations

import csv
import math
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# market → committed day-ahead series (hourly $/MWh). All three are real and wired.
MARKETS = {
    "pjm": "data/pjm_us_east_dam.csv",
    "ercot": "data/ercot_us_south_dam.csv",
    "caiso": "data/caiso_us_west_dam.csv",
}
# could be a price source but is NOT wired into the environment — documented, never fabricated.
ABSENT_MARKETS = {
    "eia": "aurelius/ingestion/grid_apis/eia.py exists but is not wired into the environment/cost model",
    "entsoe": "ENTSO-E adapter present but not wired into Aurelius",
    "realtime_5min": "only hourly day-ahead (DAM) is committed; sub-hourly real-time not available offline",
}


def load_price_series(market: str) -> list:
    """Return ``[(hour_index, price_per_kwh), ...]`` for a wired market (chronological).

    Raises if the market is not wired (so callers cannot silently fabricate a series).
    """
    key = market.lower()
    if key not in MARKETS:
        raise ValueError(f"market {market!r} is not wired (have {sorted(MARKETS)}); "
                         f"absent: {sorted(ABSENT_MARKETS)} — do not fabricate")
    path = os.path.join(_ROOT, MARKETS[key])
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                mwh = float(r["price_per_mwh"])
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(mwh / 1000.0)            # $/MWh → $/kWh
    return list(enumerate(rows))


def price_percentiles(series, qs=(0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)) -> dict:
    """Percentiles of the price series in $/kWh (plus min/max/mean), for cheap-vs-expensive levels."""
    vals = sorted(p for _, p in series)
    if not vals:
        return {}
    out = {"min": vals[0], "max": vals[-1], "mean": math.fsum(vals) / len(vals)}
    for q in qs:
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        out[f"p{int(q * 100):02d}"] = vals[idx]
    return {k: round(v, 6) for k, v in out.items()}


def diurnal_profile(series) -> dict:
    """Mean price ($/kWh) by hour-of-day (0..23) — the expensive-evening / cheap-night structure."""
    buckets: dict = {}
    for hour_idx, price in series:
        h = hour_idx % 24
        buckets.setdefault(h, []).append(price)
    return {h: round(math.fsum(v) / len(v), 6) for h, v in sorted(buckets.items())}
