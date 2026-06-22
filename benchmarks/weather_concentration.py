#!/usr/bin/env python3
"""Per-fold RT-scored decomposition of weather savings (caveat 3: concentration).

Multi-year cannot be tested: all repo price data (DAM+RT) is a single window
2025-06..2026-03 (one winter). This instead quantifies how concentrated the
weather savings are across the available folds, under realistic RT-settlement
scoring — i.e. is the +9pp broad or driven by one cold-snap fold?
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from aurelius.backtesting.engine import BacktestEngine
from aurelius.ingestion.job_logs import JobLogIngester
from aurelius.models import OptimizationConfig
from aurelius.forecasting.price_model import PriceQuantileForecaster, PriceModelConfig
import logging; logging.disable(logging.WARNING)

REPO = Path(__file__).resolve().parent.parent
REGIONS = ["us-west", "us-east", "us-south"]; START, END = "2025-09-01", "2026-03-05"; PRIMARY = "current_price_only"

def load(p):
    d = pd.read_csv(p, parse_dates=["timestamp"]); d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    return d[d.region.isin(REGIONS)]
da, rt = load(REPO/"data/combined_2025_2026/3region_dam.csv"), load(REPO/"data/combined_2025_2026/3region_rt.csv")
w = pd.read_csv(REPO/"data/weather_openmeteo/forecast_day1.csv", parse_dates=["timestamp"]); w["timestamp"] = pd.to_datetime(w["timestamp"], utc=True)
carbon = pd.DataFrame(columns=["timestamp", "region", "gco2_per_kwh"])
sts, ets = pd.Timestamp(START, tz="UTC"), pd.Timestamp(END, tz="UTC")
dur = int((ets-sts).total_seconds()/3600/0.7)+24
jobs = JobLogIngester().generate_synthetic(start_time=sts.to_pydatetime(), duration_hours=dur, num_jobs=50, regions=REGIONS, seed=0, workload_mix="realistic", workload_filter="training")
def cfg(uw): return PriceModelConfig(seed=42, n_estimators=200, learning_rate=0.05, include_volatility_features=True, num_leaves=63, include_weather_features=uw)
def run(uw):
    e = BacktestEngine(method="greedy_migrate", train_days=90, eval_days=14, config=OptimizationConfig(),
        price_forecaster_cls=PriceQuantileForecaster, price_forecaster_config=cfg(uw), context_hours=336,
        weather_df=w if uw else None)
    return e.run(jobs, da, carbon, start=sts, end=ets, settle_price_df=rt)
off, on = run(False), run(True)
print(f"{'fold':>4} {'eval_start':>12} {'RT_OFFcost':>11} {'RT_ONcost':>10} {'extra_saved$':>12}")
extra = []
for i, (ro, rn) in enumerate(zip(off, on)):
    if not (ro.optimizer_metrics and rn.optimizer_metrics and PRIMARY in ro.baseline_metrics and PRIMARY in rn.baseline_metrics): continue
    bo, bn = ro.baseline_metrics[PRIMARY].total_energy_cost_usd, rn.baseline_metrics[PRIMARY].total_energy_cost_usd
    co, cn = ro.optimizer_metrics.total_energy_cost_usd, rn.optimizer_metrics.total_energy_cost_usd
    es = getattr(ro, "split", None); esd = es.eval_start.date() if es and hasattr(es, "eval_start") else "?"
    ex = (bn-cn)-(bo-co); extra.append(ex)
    print(f"{i:>4} {str(esd):>12} {co:11.1f} {cn:10.1f} {ex:12.1f}")
extra = np.array(extra)
pos = extra[extra > 0].sum() if (extra > 0).any() else 0
print(f"\ntotal extra saved (RT) = ${extra.sum():.1f}  | folds positive: {(extra>0).sum()}/{len(extra)}")
if extra.sum() > 0:
    print(f"top fold = {100*extra.max()/extra.sum():.0f}% of net extra savings; top-2 = {100*np.sort(extra)[-2:].sum()/extra.sum():.0f}%")
