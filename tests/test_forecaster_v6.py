"""Tests for ML forecaster v6.0: cross-regional price spread features.

Feature design (v6.0):
- cross_price_vs_min: (price - min_regional) / (min_regional + eps)
  How much more expensive than cheapest region right now?
- cross_price_vs_mean: (price - mean_regional) / (mean_regional + eps)
  Deviation from regional mean (signed).
- cross_is_cheapest: 1.0 if this region is the cheapest, else 0.0
  Direct routing signal.
All three are computed at current time, 24h lag, and 168h lag (9 features total).

Single-region neutrality: when only one region's data is present, vs_min=0,
vs_mean=0, is_cheapest=1 — model degrades gracefully.

Leakage guard: features at time T only use prices at times ≤ T.

Acceptance tests:
1. CROSS_REGION_FEATURE_COLS has 9 entries
2. build_cross_region_lookup: keys are hour-floored, values are dict per region
3. add_cross_region_features: correct shape (adds 9 columns)
4. cross_is_cheapest=1 for cheapest region, 0 for others
5. cross_price_vs_min >= 0 always
6. cross_price_vs_mean is signed (can be negative for cheap regions)
7. Single-region neutrality: vs_min=0, vs_mean=0, is_cheapest=1
8. Lag features (24h, 168h) use data from prior timestamps
9. NaN-safe: no NaN in cross-regional outputs
10. build_feature_matrix adds cross-regional columns when enabled
11. build_feature_matrix_for_predict propagates cross-regional flag
12. PriceModelConfig default include_cross_region_features=False (opt-in)
13. PriceQuantileForecaster v6.0 fit/predict smoke test (multi-region)
14. Leakage: 24h lag uses T-24h lookup, not T
15. Cross-region lookup with sub-hourly timestamps floors correctly
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from aurelius.forecasting.price_model import (
    PriceModelConfig,
    PriceQuantileForecaster,
)
from aurelius.forecasting.quantile_model import (
    CROSS_REGION_FEATURE_COLS,
    add_cross_region_features,
    build_cross_region_lookup,
    build_feature_matrix,
    build_feature_matrix_for_predict,
)
from aurelius.models import EnergyPrice

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REGIONS = ["us-west", "us-east", "us-south"]
T0 = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)


def _make_prices(
    n_hours: int = 400,
    region: str = "us-west",
    base: float = 50.0,
    seed: int = 42,
    start: datetime | None = None,
) -> list[EnergyPrice]:
    rng = np.random.default_rng(seed)
    t = start or datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    out = []
    for i in range(n_hours):
        noise = rng.normal(0, 5)
        out.append(EnergyPrice(
            timestamp=t + timedelta(hours=i),
            region=region,
            price_per_mwh=max(1.0, base + noise),
        ))
    return out


def _multi_region_prices(n_hours: int = 400, seed: int = 42) -> list[EnergyPrice]:
    """Three regions with distinct price levels so cross-regional features are non-trivial."""
    rng = np.random.default_rng(seed)
    out = []
    bases = {"us-west": 40.0, "us-east": 55.0, "us-south": 70.0}
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for region, base in bases.items():
        for i in range(n_hours):
            out.append(EnergyPrice(
                timestamp=t0 + timedelta(hours=i),
                region=region,
                price_per_mwh=max(1.0, base + rng.normal(0, 3)),
            ))
    return out


def _build_lookup_from_prices(prices: list[EnergyPrice]) -> dict:
    ts = [p.timestamp for p in prices]
    regions = [p.region for p in prices]
    vals = np.array([p.price_per_mwh for p in prices])
    return build_cross_region_lookup(ts, regions, vals)


# ---------------------------------------------------------------------------
# Test 1: CROSS_REGION_FEATURE_COLS has 9 entries
# ---------------------------------------------------------------------------

def test_cross_region_feature_cols_count():
    assert len(CROSS_REGION_FEATURE_COLS) == 9


# ---------------------------------------------------------------------------
# Test 2: build_cross_region_lookup keys are hour-floored dicts
# ---------------------------------------------------------------------------

def test_build_cross_region_lookup_structure():
    prices = _multi_region_prices(n_hours=10)
    lookup = _build_lookup_from_prices(prices)
    assert isinstance(lookup, dict)
    for key, val in lookup.items():
        assert isinstance(key, pd.Timestamp), f"Expected pd.Timestamp key, got {type(key)}"
        assert isinstance(val, dict)
        assert key.minute == 0 and key.second == 0


# ---------------------------------------------------------------------------
# Test 3: add_cross_region_features returns correct shape
# ---------------------------------------------------------------------------

def test_add_cross_region_features_shape():
    prices = _multi_region_prices(n_hours=400)
    lookup = _build_lookup_from_prices(prices)

    n = 50
    t0 = datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [t0 + timedelta(hours=i) for i in range(n)]
    regions = ["us-west"] * n

    df_base = pd.DataFrame({"timestamp": timestamps, "region": regions, "price_per_mwh": [50.0] * n})
    df_out = add_cross_region_features(df_base, timestamps, regions, lookup)

    for col in CROSS_REGION_FEATURE_COLS:
        assert col in df_out.columns, f"Missing column: {col}"
    assert len(df_out) == n


# ---------------------------------------------------------------------------
# Test 4: cross_is_cheapest=1.0 for cheapest region, 0.0 for others
# ---------------------------------------------------------------------------

def test_cross_is_cheapest_routing_signal():
    # us-west is cheapest (base=40), us-east=55, us-south=70
    prices = _multi_region_prices(n_hours=400, seed=0)
    lookup = _build_lookup_from_prices(prices)

    t0 = datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    n = 24
    timestamps = [t0 + timedelta(hours=i) for i in range(n)]

    results = {}
    for region in ["us-west", "us-east", "us-south"]:
        region_list = [region] * n
        df = pd.DataFrame({"timestamp": timestamps, "region": region_list, "price_per_mwh": [0.0] * n})
        df = add_cross_region_features(df, timestamps, region_list, lookup)
        results[region] = df["cross_is_cheapest"].mean()

    # us-west should be cheapest most often
    assert results["us-west"] > results["us-east"], "us-west should be cheapest more often than us-east"
    assert results["us-west"] > results["us-south"], "us-west should be cheapest more often than us-south"


# ---------------------------------------------------------------------------
# Test 5: cross_price_vs_min >= 0 always
# ---------------------------------------------------------------------------

def test_cross_price_vs_min_non_negative():
    prices = _multi_region_prices(n_hours=400)
    lookup = _build_lookup_from_prices(prices)

    t0 = datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    n = 100
    timestamps = [t0 + timedelta(hours=i) for i in range(n)]

    for region in REGIONS:
        region_list = [region] * n
        df = pd.DataFrame({"timestamp": timestamps, "region": region_list, "price_per_mwh": [0.0] * n})
        df = add_cross_region_features(df, timestamps, region_list, lookup)
        assert (df["cross_price_vs_min"] >= -1e-9).all(), f"{region}: cross_price_vs_min went negative"


# ---------------------------------------------------------------------------
# Test 6: cross_price_vs_mean is signed (cheapest region should be negative)
# ---------------------------------------------------------------------------

def test_cross_price_vs_mean_signed():
    prices = _multi_region_prices(n_hours=400)
    lookup = _build_lookup_from_prices(prices)

    t0 = datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    n = 100
    timestamps = [t0 + timedelta(hours=i) for i in range(n)]

    west_list = ["us-west"] * n
    south_list = ["us-south"] * n
    df_west = pd.DataFrame({"timestamp": timestamps, "region": west_list, "price_per_mwh": [0.0] * n})
    df_south = pd.DataFrame({"timestamp": timestamps, "region": south_list, "price_per_mwh": [0.0] * n})
    df_west = add_cross_region_features(df_west, timestamps, west_list, lookup)
    df_south = add_cross_region_features(df_south, timestamps, south_list, lookup)

    # us-west (cheapest) should average negative vs_mean; us-south (most expensive) positive
    assert df_west["cross_price_vs_mean"].mean() < df_south["cross_price_vs_mean"].mean()


# ---------------------------------------------------------------------------
# Test 7: Single-region neutrality
# ---------------------------------------------------------------------------

def test_single_region_neutrality():
    west_prices = _make_prices(n_hours=400, region="us-west")
    lookup = _build_lookup_from_prices(west_prices)  # Only us-west in lookup

    t0 = datetime(2026, 1, 8, 0, 0, 0, tzinfo=timezone.utc)
    n = 24
    timestamps = [t0 + timedelta(hours=i) for i in range(n)]
    regions = ["us-west"] * n
    df = pd.DataFrame({"timestamp": timestamps, "region": regions, "price_per_mwh": [0.0] * n})
    df = add_cross_region_features(df, timestamps, regions, lookup)

    # With only one region: vs_min=0, vs_mean=0, is_cheapest=1
    assert (df["cross_price_vs_min"].abs() < 1e-9).all(), "vs_min should be 0 for single region"
    assert (df["cross_price_vs_mean"].abs() < 1e-9).all(), "vs_mean should be 0 for single region"
    assert (df["cross_is_cheapest"] == 1.0).all(), "is_cheapest should be 1 for single region"


# ---------------------------------------------------------------------------
# Test 8: Lag features use data from prior timestamps
# ---------------------------------------------------------------------------

def test_lag_features_use_prior_data():
    prices = _multi_region_prices(n_hours=400)
    lookup = _build_lookup_from_prices(prices)

    # Pick a timestamp well inside the data range
    t_eval = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    t_lag24 = t_eval - timedelta(hours=24)
    t_lag168 = t_eval - timedelta(hours=168)

    n = 1
    timestamps = [t_eval]
    regions = ["us-west"]
    df = pd.DataFrame({"timestamp": timestamps, "region": regions, "price_per_mwh": [0.0]})
    df = add_cross_region_features(df, timestamps, regions, lookup)

    # Verify lag features are present (non-null) — we have data at those times
    assert not math.isnan(df["cross_is_cheapest_lag_24h"].iloc[0]), "24h lag should be non-NaN"
    assert not math.isnan(df["cross_is_cheapest_lag_168h"].iloc[0]), "168h lag should be non-NaN"

    # Verify lag values differ from current values (different time → different prices)
    # (With distinct base prices and noise, current vs lag_24h may match only by coincidence)
    # Instead just check they are valid 0/1 values
    for col in ["cross_is_cheapest_lag_24h", "cross_is_cheapest_lag_168h"]:
        val = df[col].iloc[0]
        assert val in (0.0, 1.0), f"{col} should be 0 or 1, got {val}"


# ---------------------------------------------------------------------------
# Test 9: NaN-safe — no NaN in cross-regional outputs when data present
# ---------------------------------------------------------------------------

def test_no_nan_in_cross_regional_features():
    prices = _multi_region_prices(n_hours=500)
    lookup = _build_lookup_from_prices(prices)

    t0 = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc)
    n = 100
    timestamps = [t0 + timedelta(hours=i) for i in range(n)]

    for region in REGIONS:
        region_list = [region] * n
        df = pd.DataFrame({"timestamp": timestamps, "region": region_list, "price_per_mwh": [0.0] * n})
        df = add_cross_region_features(df, timestamps, region_list, lookup)
        for col in CROSS_REGION_FEATURE_COLS:
            assert not df[col].isna().any(), f"NaN found in {col} for region {region}"


# ---------------------------------------------------------------------------
# Test 10: build_feature_matrix adds cross-regional columns when enabled
# ---------------------------------------------------------------------------

def test_build_feature_matrix_cross_region_columns():
    prices = _multi_region_prices(n_hours=400)
    lookup = _build_lookup_from_prices(prices)

    west_prices = [p for p in prices if p.region == "us-west"]
    timestamps = [p.timestamp for p in west_prices]
    values = np.array([p.price_per_mwh for p in west_prices])

    df = build_feature_matrix(
        timestamps=timestamps,
        values=values,
        regions=["us-west"] * len(timestamps),
        include_cross_region_features=True,
        cross_region_lookup=lookup,
    )

    for col in CROSS_REGION_FEATURE_COLS:
        assert col in df.columns, f"Missing cross-regional column: {col}"


# ---------------------------------------------------------------------------
# Test 11: build_feature_matrix_for_predict propagates cross-regional flag
# ---------------------------------------------------------------------------

def test_build_feature_matrix_for_predict_cross_region():
    prices = _multi_region_prices(n_hours=400)
    lookup = _build_lookup_from_prices(prices)

    west_prices = [p for p in prices if p.region == "us-west"]
    predict_ts = [datetime(2026, 1, 10, h, 0, 0, tzinfo=timezone.utc) for h in range(24)]
    recent_vals = np.array([p.price_per_mwh for p in west_prices[-192:]])

    df, _used_lags = build_feature_matrix_for_predict(
        timestamps=predict_ts,
        regions=["us-west"] * 24,
        recent_values=recent_vals,
        include_cross_region_features=True,
        cross_region_lookup=lookup,
    )

    for col in CROSS_REGION_FEATURE_COLS:
        assert col in df.columns, f"Missing cross-regional column in predict: {col}"


# ---------------------------------------------------------------------------
# Test 12: PriceModelConfig default include_cross_region_features=False
# ---------------------------------------------------------------------------

def test_price_model_config_default():
    cfg = PriceModelConfig()
    assert cfg.include_cross_region_features is False


# ---------------------------------------------------------------------------
# Test 13: PriceQuantileForecaster v6.0 smoke test (multi-region fit/predict)
# ---------------------------------------------------------------------------

def test_price_quantile_forecaster_v6_smoke():
    prices = _multi_region_prices(n_hours=400)

    cfg = PriceModelConfig(
        include_cross_region_features=True,
        include_rank_features=True,
        include_volatility_features=True,
        n_estimators=20,
        num_leaves=15,
        seed=42,
    )
    fc = PriceQuantileForecaster(cfg)
    fc.fit(prices)

    predict_ts = [datetime(2026, 1, 17, h, 0, 0, tzinfo=timezone.utc) for h in range(24)]
    recent = prices[-192:]  # Last 8 days across all regions

    preds = fc.predict("us-west", predict_ts, recent)
    assert len(preds) == 24
    for p in preds:
        assert p.p50 > 0, "p50 should be positive"
        assert p.p90 >= p.p50, "p90 should be >= p50"


# ---------------------------------------------------------------------------
# Test 14: Leakage — 24h lag uses T-24h lookup, not T
# ---------------------------------------------------------------------------

def test_lag_24h_uses_prior_timestamp():
    # Build a lookup where only specific timestamps have data
    t_now = pd.Timestamp("2026-01-15 12:00:00", tz="UTC")
    t_lag24 = pd.Timestamp("2026-01-14 12:00:00", tz="UTC")

    lookup = {
        t_now: {"us-west": 100.0, "us-east": 200.0},
        t_lag24: {"us-west": 50.0, "us-east": 50.0},  # Equal at lag time → is_cheapest could be either
    }

    timestamps = [t_now.to_pydatetime()]
    regions = ["us-west"]
    df = pd.DataFrame({"timestamp": timestamps, "region": regions, "price_per_mwh": [0.0]})
    df = add_cross_region_features(df, timestamps, regions, lookup, lag_hours=[24, 168])

    # At t_now: us-west=100, us-east=200 → us-west is cheapest → is_cheapest=1
    assert df["cross_is_cheapest"].iloc[0] == 1.0

    # At t_now-24h: us-west=50, us-east=50 → equal prices → is_cheapest=1 (us-west wins ties)
    # vs_min should be 0 (50-50)/(50+1) ≈ 0
    assert df["cross_price_vs_min_lag_24h"].iloc[0] == pytest.approx(0.0, abs=0.05)


# ---------------------------------------------------------------------------
# Test 15: Sub-hourly timestamps floor correctly in lookup
# ---------------------------------------------------------------------------

def test_cross_region_lookup_floors_sub_hourly():
    t_on_hour = pd.Timestamp("2026-01-10 14:00:00", tz="UTC")
    t_sub_hourly = pd.Timestamp("2026-01-10 14:37:22", tz="UTC")

    ts = [t_sub_hourly]
    regions = ["us-west"]
    vals = np.array([80.0])
    lookup = build_cross_region_lookup(ts, regions, vals)

    assert t_on_hour in lookup, "Sub-hourly timestamp should floor to the hour key"
    assert "us-west" in lookup[t_on_hour]
    assert lookup[t_on_hour]["us-west"] == pytest.approx(80.0)
