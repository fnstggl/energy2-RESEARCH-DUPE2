"""Tests for the deployable forecasted MCS provisioner.

The single most important property is **causality**: the capacity chosen for
tick ``t`` must depend only on data from ticks ``0 .. t-1``.  ``test_*causal*``
falsify the oracle behaviour that this module exists to replace.
"""

from __future__ import annotations

import math

import pytest

from aurelius.benchmarks.forecasted_mcs import (
    bucketize,
    evaluate_c_schedule,
    forecast_mcs_c_schedule,
    reactive_lag1_c_schedule,
    sla_aware_fixed_c,
)
from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_SLA_S,
    _joint_mcs_c_schedule,
)


def _synthetic_trace(n_ticks: int = 12, per_tick: int = 30, tick_s: float = 60.0,
                     warp: float = 1.0, seed: int = 0) -> list[tuple[float, int]]:
    """Deterministic (arrival_s, output_tokens) trace, ``per_tick`` reqs/tick.

    Arrivals are spread evenly inside each (warped) tick; token counts vary
    tick-to-tick so service forecasts are non-trivial.
    """
    import random
    rng = random.Random(seed)
    raw: list[tuple[float, int]] = []
    for t in range(n_ticks):
        count = per_tick + (t % 5) * 4  # tick-varying demand
        for j in range(count):
            arr = (t + (j + 0.5) / count) * tick_s * warp
            tok = 40 + rng.randint(0, 120)
            raw.append((arr, tok))
    raw.sort(key=lambda r: r[0])
    return raw


# ---------------------------------------------------------------------------
# Causality — the core invariant
# ---------------------------------------------------------------------------

def test_forecast_mcs_is_causal_midtrace_perturbation():
    """Perturbing tick t's actuals must not change c[0..t]; only c[t+1..] may move."""
    tick_s, warp = 60.0, 1.0
    raw = _synthetic_trace(n_ticks=12, per_tick=30, tick_s=tick_s, warp=warp)
    c_base, _ = forecast_mcs_c_schedule(raw, tick_s, warp, sla_s=DEFAULT_SLA_S)

    # Inject a 10x demand spike into tick t=6 only.
    perturb_tick = 6
    spike = [((perturb_tick + 0.5) * tick_s, 300) for _ in range(300)]
    raw2 = sorted(raw + spike, key=lambda r: r[0])

    # Re-bucketize so both schedules align on the same grid length.
    c_pert, _ = forecast_mcs_c_schedule(raw2, tick_s, warp, sla_s=DEFAULT_SLA_S)

    # Decisions for ticks strictly before the perturbed tick are identical.
    for t in range(perturb_tick):
        assert c_base[t] == c_pert[t], (
            f"tick {t} changed after perturbing tick {perturb_tick} — NOT causal")

    # The perturbed tick's own capacity must also be unchanged (it is sized from
    # history < t, which the spike at tick t does not touch).
    assert c_base[perturb_tick] == c_pert[perturb_tick], (
        "perturbed tick's own capacity used current-tick actuals — leak!")


def test_reactive_lag1_is_causal():
    """Perturbing tick t must not change c[0..t]; lag-1 may change c[t+1] only."""
    tick_s, warp = 60.0, 1.0
    raw = _synthetic_trace(n_ticks=10, per_tick=25)
    c_base = reactive_lag1_c_schedule(raw, tick_s, warp, sla_s=DEFAULT_SLA_S)

    perturb_tick = 5
    spike = [((perturb_tick + 0.5) * tick_s, 250) for _ in range(200)]
    raw2 = sorted(raw + spike, key=lambda r: r[0])
    c_pert = reactive_lag1_c_schedule(raw2, tick_s, warp, sla_s=DEFAULT_SLA_S)

    for t in range(perturb_tick + 1):  # c[t] uses tick t-1, so c[<=t] unaffected
        assert c_base[t] == c_pert[t], f"lag-1 leaked future info at tick {t}"
    # c[t+1] should react to the spike at tick t.
    assert c_pert[perturb_tick + 1] >= c_base[perturb_tick + 1]


def test_oracle_differs_from_forecast_on_spikes():
    """Sanity: the oracle (clairvoyant) reacts to a spike *in the same tick*,
    while the causal forecast cannot.  This is exactly the gap we are auditing."""
    tick_s, warp = 60.0, 1.0
    raw = _synthetic_trace(n_ticks=10, per_tick=20)
    spike_tick = 7
    spike = [((spike_tick + 0.5) * tick_s, 200) for _ in range(150)]
    raw2 = sorted(raw + spike, key=lambda r: r[0])

    c_oracle = _joint_mcs_c_schedule(raw2, tick_s, warp, sla_s=DEFAULT_SLA_S)
    c_fc, _ = forecast_mcs_c_schedule(raw2, tick_s, warp, sla_s=DEFAULT_SLA_S)
    # Oracle sizes the spike tick from the spike itself; forecast does not.
    assert c_oracle[spike_tick] > c_fc[spike_tick]


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def test_bucketize_matches_oracle_grid():
    raw = _synthetic_trace(n_ticks=8, per_tick=15)
    counts, token_lists, n_ticks = bucketize(raw, 60.0, 1.0)
    c_oracle = _joint_mcs_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)
    assert n_ticks == len(c_oracle)
    assert sum(counts) == len(raw)
    assert sum(len(x) for x in token_lists) == len(raw)


def test_schedules_have_expected_length_and_floor():
    raw = _synthetic_trace(n_ticks=9, per_tick=18)
    _counts, _t, n_ticks = bucketize(raw, 60.0, 1.0)
    for c in (
        reactive_lag1_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S),
        forecast_mcs_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)[0],
        forecast_mcs_c_schedule(raw, 60.0, 1.0, method="quantile", safety_k=1.0,
                                sla_s=DEFAULT_SLA_S)[0],
    ):
        assert len(c) == n_ticks
        assert all(isinstance(x, int) and x >= 1 for x in c)


def test_quantile_buffer_is_at_least_point_forecast_capacity():
    """A p90 + 1σ safety buffer should never under-provision vs the EWMA point
    forecast on average (mean capacity higher)."""
    raw = _synthetic_trace(n_ticks=14, per_tick=30, seed=3)
    c_pt, _ = forecast_mcs_c_schedule(raw, 60.0, 1.0, method="ewma", sla_s=DEFAULT_SLA_S)
    c_q, _ = forecast_mcs_c_schedule(raw, 60.0, 1.0, method="quantile", quantile=0.90,
                                     safety_k=1.0, sla_s=DEFAULT_SLA_S)
    assert sum(c_q) >= sum(c_pt)


def test_sla_aware_fixed_c_positive():
    raw = _synthetic_trace(n_ticks=10, per_tick=30)
    c = sla_aware_fixed_c(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)
    assert isinstance(c, int) and c >= 1


def test_forecast_diag_reports_error():
    raw = _synthetic_trace(n_ticks=12, per_tick=30)
    _c, diag = forecast_mcs_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)
    d = diag.to_dict()
    assert d["arr_mae"] >= 0.0
    assert d["n_ticks"] == 12
    assert d["c_max"] >= d["c_min"] >= 1


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------

def test_evaluate_c_schedule_kpis_consistent():
    raw = _synthetic_trace(n_ticks=10, per_tick=25)
    _c, _ = forecast_mcs_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)
    kpi = evaluate_c_schedule(
        raw, _c, 60.0, 1.0, DEFAULT_SLA_S,
        policy="t", uses_future_info=False, deployable=True,
        classification="Deployable", discipline="fifo",
    )
    # provisioned GPU-hours and cost agree with the schedule (unrounded attrs).
    assert math.isclose(kpi.gpu_hours, sum(_c) * 60.0 / 3600.0, rel_tol=1e-9)
    assert math.isclose(kpi.cost_usd, kpi.gpu_hours * 2.0, rel_tol=1e-9)
    d = kpi.to_dict()
    assert d["goodput_per_dollar"] > 0
    assert 0 <= d["n_sla_safe"] <= d["n_total"] == len(raw)
    assert d["sla_violations"] == d["n_total"] - d["n_sla_safe"]


def test_evaluate_fixed_c_sla_aware_runs():
    raw = _synthetic_trace(n_ticks=8, per_tick=20)
    _c, _t, n_ticks = bucketize(raw, 60.0, 1.0)
    kpi = evaluate_c_schedule(
        raw, [5] * n_ticks, 60.0, 1.0, DEFAULT_SLA_S,
        policy="sla_aware_fixed_c5", uses_future_info=False, deployable=True,
        classification="Deployable (fixed, no MCS)", discipline="sla_aware",
    )
    assert kpi.goodput_per_dollar > 0
    assert kpi.c_mean == 5.0


def test_determinism():
    raw = _synthetic_trace(n_ticks=11, per_tick=22, seed=7)
    a, _ = forecast_mcs_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)
    b, _ = forecast_mcs_c_schedule(raw, 60.0, 1.0, sla_s=DEFAULT_SLA_S)
    assert a == b


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
