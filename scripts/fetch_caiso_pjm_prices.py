"""Fetch CAISO day-ahead + PJM real-time LMP price data and combine into one CSV.

Produces the price files used by the combined CAISO+PJM real-time backtest:
    data/caiso_us_west_dam.csv      CAISO us-west day-ahead (hourly)
    data/pjm_us_east_rt.csv         PJM   us-east real-time  (hourly, total_lmp_rt)
    data/combined_caiso_pjm_rt.csv  both regions concatenated

Usage:
    PJM_API_KEY=... python scripts/fetch_caiso_pjm_prices.py \
        --start 2026-01-01 --end 2026-03-15 --out-dir data

Note: PJM RT five-minute data archives after ~6 months; the hourly RT feed
(rt_hrl_lmps, hourly=True here) retains longer and aligns with CAISO's hourly
day-ahead series — required because the backtest engine floors timestamps to
the hour.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider
from aurelius.ingestion.grid_apis.pjm import PJMRealtimePriceProvider


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2026-01-01", help="ISO start date (UTC)")
    ap.add_argument("--end", default="2026-03-15", help="ISO end date (UTC)")
    ap.add_argument("--out-dir", default="data", help="Output directory")
    args = ap.parse_args()

    start, end = _parse_utc(args.start), _parse_utc(args.end)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("Fetching CAISO us-west day-ahead (hourly)...", flush=True)
    caiso = CAISOPriceProvider().fetch_prices("us-west", start, end)
    print(f"  CAISO rows={len(caiso)} ({time.time() - t0:.0f}s)", flush=True)
    caiso.to_csv(out / "caiso_us_west_dam.csv", index=False)

    print("Fetching PJM us-east real-time (hourly)...", flush=True)
    pjm = PJMRealtimePriceProvider(hourly=True).fetch_prices("us-east", start, end)
    print(f"  PJM RT rows={len(pjm)} ({time.time() - t0:.0f}s)", flush=True)
    pjm.to_csv(out / "pjm_us_east_rt.csv", index=False)

    combined = pd.concat([caiso, pjm], ignore_index=True)
    combined.to_csv(out / "combined_caiso_pjm_rt.csv", index=False)
    print(f"COMBINED rows={len(combined)} regions={sorted(combined['region'].unique())}",
          flush=True)
    for reg, g in combined.groupby("region"):
        print(f"  {reg}: {len(g)} rows  {g['timestamp'].min()} .. {g['timestamp'].max()}  "
              f"mean=${g['price_per_mwh'].mean():.2f}/MWh", flush=True)


if __name__ == "__main__":
    main()
