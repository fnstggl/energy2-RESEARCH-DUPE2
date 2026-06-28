"""Electricity economic controller tests (focused — prices, power, deferrable; no heavy inputs)."""

from __future__ import annotations

import copy

import pytest

from aurelius.environment.deferrable import (
    DeferrableWorkState,
    generate_deferrable_pool,
    run_deferrable_episode,
)
from aurelius.environment.electricity import (
    MARKET_BY_REGION,
    PowerState,
    build_price_profile,
    electricity_state_for_period,
)
from aurelius.environment.price_series import ABSENT_MARKETS, load_price_series, price_percentiles


# --- prices: real series, $/MWh→$/kWh, region mapping, flat-reproduces --------
def test_real_price_series_varies_and_units():
    s = load_price_series("pjm")
    assert len(s) > 500
    pct = price_percentiles(s)
    assert 0.0 < pct["p10"] < pct["p90"] < 5.0       # $/kWh range (MWh/1000), real variation
    with pytest.raises(ValueError):                  # EIA not wired → refuse, never fabricate
        load_price_series("eia")
    assert "eia" in ABSENT_MARKETS


def test_region_market_mapping_matches_registry():
    from aurelius.ingestion.region_registry import REGION_REGISTRY
    for region, market in MARKET_BY_REGION.items():
        assert REGION_REGISTRY[region].iso.lower() == market   # us-east→pjm, us-south→ercot, us-west→caiso


def test_flat_profile_reproduces_constant_real_varies():
    flat = build_price_profile(None, 24, flat_price=0.05)
    assert len(set(flat.by_cycle.values())) == 1 and flat.provenance == "SIMULATOR_INFERENCE"
    real = build_price_profile("pjm", 24)
    assert len(set(round(v, 6) for v in real.by_cycle.values())) > 1 and real.provenance == "TRACE_DERIVED"
    assert real.region == "us-east"


def test_electricity_state_deterministic_no_leakage():
    prof = build_price_profile("caiso", 24)
    a = electricity_state_for_period(prof, 18, 24)
    b = electricity_state_for_period(prof, 18, 24)
    assert a == b                                    # deterministic (period-indexed; no future input)
    assert 0.0 <= a.price_percentile <= 1.0 and a.market == "caiso"


# --- PowerState: energy accounting ------------------------------------------
def test_powerstate_energy_accounting():
    ps = PowerState()
    ps.accumulate(power_w=600.0, energy_j=3.6e6, price_per_kwh=0.10, clock_state="low")  # 3.6e6 J = 1 kWh
    assert ps.cumulative_energy_kwh == pytest.approx(1.0) and ps.cumulative_energy_cost == pytest.approx(0.10)
    ps.accumulate(power_w=800.0, energy_j=7.2e6, price_per_kwh=0.20, clock_state="high")  # +2 kWh × 0.20
    assert ps.cumulative_energy_kwh == pytest.approx(3.0) and ps.cumulative_energy_cost == pytest.approx(0.50)
    assert ps.clock_state == "high" and ps.lever == "clock_locking"


# --- DeferrableWorkState: persistence, conservation, deadlines, shifting ------
def _pool():
    return generate_deferrable_pool(8, horizon_periods=12)


def test_deferrable_persists_and_conserves_work():
    pool = _pool()
    periods = list(range(12))
    res = run_deferrable_episode(copy.deepcopy(pool), periods=periods,
                                 prices={p: 0.1 for p in periods}, spare_by_period={p: 1e9 for p in periods})
    assert res["completed"] + res["missed"] == len(pool.jobs)       # nothing vanishes


def test_deferrable_delayed_runs_or_misses_and_penalised():
    # zero spare everywhere → no job can run → all miss → penalty charged (deadline can't be dodged free)
    pool = _pool()
    periods = list(range(12))
    res = run_deferrable_episode(pool, periods=periods, prices={p: 0.1 for p in periods},
                                 spare_by_period={p: 0.0 for p in periods})
    assert res["missed"] == len(pool.jobs) and res["completed"] == 0
    assert res["missed_penalty_cost"] > 0.0


def test_shifting_saves_only_when_prices_vary():
    periods = list(range(12))
    vary = {p: (0.02 if p % 4 == 0 else 0.20) for p in periods}     # cheap hours 0,4,8
    flat = {p: 0.10 for p in periods}
    spare = {p: 1e9 for p in periods}
    pa_v = run_deferrable_episode(copy.deepcopy(_pool()), periods=periods, prices=vary, spare_by_period=spare, policy="price_aware")
    as_v = run_deferrable_episode(copy.deepcopy(_pool()), periods=periods, prices=vary, spare_by_period=spare, policy="asap")
    assert pa_v["electricity_cost"] < as_v["electricity_cost"]      # real shifting saving when prices vary
    pa_f = run_deferrable_episode(copy.deepcopy(_pool()), periods=periods, prices=flat, spare_by_period=spare, policy="price_aware")
    as_f = run_deferrable_episode(copy.deepcopy(_pool()), periods=periods, prices=flat, spare_by_period=spare, policy="asap")
    assert pa_f["electricity_cost"] == pytest.approx(as_f["electricity_cost"])   # FLAT → no fake shifting value


def test_serving_protected_from_deferrable():
    # busy periods (0 spare) cannot host deferrable work → it defers; serving capacity is never stolen
    periods = list(range(12))
    spare = {p: (0.0 if p < 10 else 1e9) for p in periods}          # only the last 2 periods have room
    res = run_deferrable_episode(_pool(), periods=periods, prices={p: 0.1 for p in periods},
                                 spare_by_period=spare, policy="asap")
    # any job whose entire deadline window falls in the no-spare span must miss (couldn't steal serving capacity)
    assert res["completed"] + res["missed"] == 8


def test_deferrable_region_shift_out_of_scope():
    # region shifting is documented OUT OF SCOPE (no multi-region fleet) → jobs are not region-shiftable
    assert all(not j.region_shiftable for j in generate_deferrable_pool(5, horizon_periods=10).jobs)


def test_deferrable_deterministic_replay():
    periods = list(range(12))
    kw = dict(periods=periods, prices={p: 0.03 * (p % 5) for p in periods},
              spare_by_period={p: 1e6 for p in periods}, policy="price_aware")
    a = run_deferrable_episode(_pool(), **kw)
    b = run_deferrable_episode(_pool(), **kw)
    assert a == b


def test_deferrable_state_dict_shape():
    st = DeferrableWorkState(jobs=generate_deferrable_pool(3, horizon_periods=6).jobs)
    d = st.to_dict()
    assert set(d) >= {"n_jobs", "waiting", "completed", "missed", "shifted", "electricity_cost"}
