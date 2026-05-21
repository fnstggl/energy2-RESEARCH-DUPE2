"""Fetch CAISO + PJM day-ahead (planning) and real-time (settlement) LMP prices.

Produces the DA-plan / RT-settle backtest inputs for both regions:
    data/caiso_us_west_dam.csv      CAISO us-west day-ahead   (hourly)
    data/caiso_us_west_rt.csv       CAISO us-west real-time   (5-min -> hourly mean)
    data/pjm_us_east_dam.csv        PJM   us-east  day-ahead   (hourly)
    data/pjm_us_east_rt.csv         PJM   us-east  real-time   (rt_hrl_lmps hourly)
    data/plan_da_caiso_pjm.csv      PLANNING prices: both regions' day-ahead
    data/settle_rt_caiso_pjm.csv    SETTLEMENT prices: both regions' real-time

Backtest usage (RT-exposed customer):
    aurelius backtest --price-provider csv --price-file data/plan_da_caiso_pjm.csv \
        --settlement-price-file data/settle_rt_caiso_pjm.csv \
        --regions us-west,us-east --forecaster ml_quantile --forecast-horizon-hours 24

The optimizer plans against day-ahead (known ~24h ahead) and is scored on the
realized real-time bill. CAISO real-time is published at 5-minute granularity
and is resampled to an hourly mean here because the backtest engine works at
hourly resolution.

Usage:
    PJM_API_KEY=... python scripts/fetch_caiso_pjm_prices.py \
        --start 2026-01-01 --end 2026-03-15 --out-dir data
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aurelius.ingestion.grid_apis.base import normalize_price_df
from aurelius.ingestion.grid_apis.caiso import CAISOPriceProvider, CAISORealtimePriceProvider
from aurelius.ingestion.grid_apis.pjm import PJMPriceProvider, PJMRealtimePriceProvider


def _parse_utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _to_hourly_mean(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Resample a sub-hourly price series to an hourly mean per region."""
    if df.empty:
        return df
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True).dt.floor("h")
    agg = (
        d.groupby(["region", "timestamp"], as_index=False)["price_per_mwh"]
        .mean()
    )
    return normalize_price_df(agg, source=source, currency="USD", granularity="hourly")


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

    def log(msg: str) -> None:
        print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    log("CAISO us-west day-ahead (hourly)...")
    caiso_da = CAISOPriceProvider().fetch_prices("us-west", start, end)
    caiso_da.to_csv(out / "caiso_us_west_dam.csv", index=False)
    log(f"  rows={len(caiso_da)}")

    log("CAISO us-west real-time (5-min -> hourly mean)...")
    caiso_rt_5min = CAISORealtimePriceProvider().fetch_prices("us-west", start, end)
    caiso_rt = _to_hourly_mean(caiso_rt_5min, source="caiso_oasis_rtm_hourly")
    caiso_rt.to_csv(out / "caiso_us_west_rt.csv", index=False)
    log(f"  rows={len(caiso_rt)} (from {len(caiso_rt_5min)} 5-min intervals)")

    log("PJM us-east day-ahead (hourly)...")
    pjm_da = PJMPriceProvider().fetch_prices("us-east", start, end)
    pjm_da.to_csv(out / "pjm_us_east_dam.csv", index=False)
    log(f"  rows={len(pjm_da)}")

    log("PJM us-east real-time (hourly)...")
    pjm_rt = PJMRealtimePriceProvider(hourly=True).fetch_prices("us-east", start, end)
    pjm_rt.to_csv(out / "pjm_us_east_rt.csv", index=False)
    log(f"  rows={len(pjm_rt)}")

    plan = pd.concat([caiso_da, pjm_da], ignore_index=True)
    settle = pd.concat([caiso_rt, pjm_rt], ignore_index=True)
    plan.to_csv(out / "plan_da_caiso_pjm.csv", index=False)
    settle.to_csv(out / "settle_rt_caiso_pjm.csv", index=False)

    print("\n--- summary (mean $/MWh) ---", flush=True)
    for label, df in (("PLAN (day-ahead)", plan), ("SETTLE (real-time)", settle)):
        print(f"{label}:", flush=True)
        for reg, g in df.groupby("region"):
            print(f"  {reg}: {len(g)} rows  mean=${g['price_per_mwh'].mean():.2f}  "
                  f"max=${g['price_per_mwh'].max():.2f}", flush=True)


if __name__ == "__main__":
    main()
