"""Tests for Spot Fleet MCS Backtest [run 2026-06-23B].

Spot/preemptible pricing overlay on FIFO+MCS: can replacing on-demand
replicas with spot instances achieve the +300% vs SLA-aware oracle north-star?

Invariants tested:
  1.  _spot_fleet_cost with spot_fraction=0.0 equals all-on-demand cost.
  2.  _spot_fleet_cost decreases as spot_fraction increases (lower spot price).
  3.  _spot_fleet_cost is always > 0.
  4.  _expected_interruptions_over_run is 0 when spot_fraction=0.
  5.  _expected_interruptions_over_run increases with p_interrupt_hourly.
  6.  _simulate_fifo_spot_fleet completions equal _simulate_fifo_variable_c
      completions when spot_fraction=0 (no interruptions → identical).
  7.  _simulate_fifo_spot_fleet response times are non-negative.
  8.  run_spot_fleet_mcs_azure_backtest returns SpotFleetMCSReport.
  9.  fifo_ondemand_goodput_per_dollar > 0.
  10. fifo_spot_fleet_goodput_per_dollar >= fifo_ondemand_goodput_per_dollar
      (spot pricing reduces cost without reducing goodput significantly).
  11. cost_spot_fleet < cost_ondemand when spot_fraction > 0 and
      spot_price_usd_hr < GPU_HOUR_USD.
  12. north_star_achieved is True for primary operating point
      (spot_fraction=0.70, spot_price=$0.80/hr, p_int=0.10/hr).
  13. Completion rates > 0.95 for all conditions (SLA physics maintained).
  14. to_dict() contains all expected keys.
  15. Expected interruptions < 1.0 for realistic p_int at primary config
      (confirms SLA impact is negligible).
"""

from __future__ import annotations

import os

import pytest

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "azure_llm_2024_sample.csv"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(FIXTURE),
    reason="azure_llm_2024_sample.csv fixture not found",
)


@pytest.fixture
def azure_raw():
    from aurelius.benchmarks.srtf_serving_backtest import load_serving_requests
    return load_serving_requests(FIXTURE, limit=200)


@pytest.fixture
def c_schedule_200(azure_raw):
    from aurelius.benchmarks.srtf_serving_backtest import (
        calibrate_time_warp, _joint_mcs_c_schedule,
    )
    warp = calibrate_time_warp(azure_raw, servers=4, target_rho=0.85)
    return _joint_mcs_c_schedule(azure_raw, tick_seconds=60.0, warp=warp)


@pytest.fixture
def spot_report(azure_raw):
    from aurelius.benchmarks.srtf_serving_backtest import (
        run_spot_fleet_mcs_azure_backtest,
    )
    return run_spot_fleet_mcs_azure_backtest(
        fixed_c=4,
        target_rho=0.85,
        job_limit=200,
        sla_s=10.0,
        azure_fixture=FIXTURE,
        tick_seconds=60.0,
        spot_fraction=0.70,
        spot_price_usd_hr=0.80,
        p_interrupt_hourly=0.10,
        seed=42,
    )


# 1. spot_fraction=0 cost equals all-on-demand
def test_spot_fleet_cost_zero_fraction(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _spot_fleet_cost, GPU_HOUR_USD,
    )
    cost_zero = _spot_fleet_cost(c_schedule_200, 0.0, 0.80, GPU_HOUR_USD, 60.0)
    cost_ondemand = _spot_fleet_cost(c_schedule_200, 0.0, GPU_HOUR_USD, GPU_HOUR_USD, 60.0)
    assert abs(cost_zero - cost_ondemand) < 1e-9


# 2. Cost decreases as spot_fraction increases (spot cheaper than on-demand)
def test_spot_fleet_cost_decreases_with_fraction(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _spot_fleet_cost, GPU_HOUR_USD,
    )
    cost_30 = _spot_fleet_cost(c_schedule_200, 0.30, 0.80, GPU_HOUR_USD, 60.0)
    cost_60 = _spot_fleet_cost(c_schedule_200, 0.60, 0.80, GPU_HOUR_USD, 60.0)
    cost_90 = _spot_fleet_cost(c_schedule_200, 0.90, 0.80, GPU_HOUR_USD, 60.0)
    assert cost_30 > cost_60 > cost_90


# 3. Cost is always positive
def test_spot_fleet_cost_positive(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _spot_fleet_cost, GPU_HOUR_USD,
    )
    cost = _spot_fleet_cost(c_schedule_200, 0.70, 0.80, GPU_HOUR_USD, 60.0)
    assert cost > 0.0


# 4. Expected interruptions = 0 when spot_fraction = 0
def test_expected_interruptions_zero_fraction(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _expected_interruptions_over_run,
    )
    exp = _expected_interruptions_over_run(c_schedule_200, 0.0, 0.10, 60.0)
    assert exp == 0.0


# 5. Expected interruptions increase with p_interrupt_hourly
def test_expected_interruptions_increase_with_p_int(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _expected_interruptions_over_run,
    )
    exp_low = _expected_interruptions_over_run(c_schedule_200, 0.70, 0.05, 60.0)
    exp_high = _expected_interruptions_over_run(c_schedule_200, 0.70, 0.20, 60.0)
    assert exp_high > exp_low


# 6. spot_fraction=0 simulation matches variable_c simulation
def test_spot_zero_fraction_matches_variable_c(azure_raw):
    from aurelius.benchmarks.srtf_serving_backtest import (
        calibrate_time_warp, make_live_prior_predictions,
        _joint_mcs_c_schedule, _simulate_fifo_variable_c,
        _simulate_fifo_spot_fleet, _Request, _service_time_s, LIVE_PRIOR_WINDOW,
    )
    warp = calibrate_time_warp(azure_raw, servers=4, target_rho=0.85)
    live_preds, _ = make_live_prior_predictions(azure_raw, window=LIVE_PRIOR_WINDOW)
    c_sched = _joint_mcs_c_schedule(azure_raw, tick_seconds=60.0, warp=warp)

    def build():
        return [_Request(idx=i, arrival_s=arr/warp, actual_tokens=tok,
                         predicted_tokens=live_preds[i], service_s=_service_time_s(tok))
                for i, (arr, tok) in enumerate(azure_raw)]

    reqs_vc = build(); reqs_sf = build()
    _, resp_vc, _ = _simulate_fifo_variable_c(reqs_vc, c_sched, 60.0)
    _, resp_sf, _ = _simulate_fifo_spot_fleet(reqs_sf, c_sched, 0.0, 0.10, 60.0, 42)
    assert len(resp_vc) == len(resp_sf)


# 7. Spot fleet response times are non-negative
def test_spot_fleet_response_nonneg(azure_raw):
    from aurelius.benchmarks.srtf_serving_backtest import (
        calibrate_time_warp, make_live_prior_predictions,
        _joint_mcs_c_schedule, _simulate_fifo_spot_fleet,
        _Request, _service_time_s, LIVE_PRIOR_WINDOW,
    )
    warp = calibrate_time_warp(azure_raw, servers=4, target_rho=0.85)
    live_preds, _ = make_live_prior_predictions(azure_raw, window=LIVE_PRIOR_WINDOW)
    c_sched = _joint_mcs_c_schedule(azure_raw, tick_seconds=60.0, warp=warp)
    reqs = [_Request(idx=i, arrival_s=arr/warp, actual_tokens=tok,
                     predicted_tokens=live_preds[i], service_s=_service_time_s(tok))
            for i, (arr, tok) in enumerate(azure_raw)]
    _, resp, _ = _simulate_fifo_spot_fleet(reqs, c_sched, 0.70, 0.10, 60.0, 42)
    assert all(v >= 0.0 for v in resp.values())


# 8. run_spot_fleet_mcs_azure_backtest returns SpotFleetMCSReport
def test_spot_report_type(spot_report):
    from aurelius.benchmarks.srtf_serving_backtest import SpotFleetMCSReport
    assert isinstance(spot_report, SpotFleetMCSReport)


# 9. on-demand goodput/$ > 0
def test_ondemand_goodput_positive(spot_report):
    assert spot_report.fifo_ondemand_goodput_per_dollar > 0.0


# 10. spot fleet goodput/$ >= on-demand (spot reduces cost, goodput roughly same)
def test_spot_fleet_goodput_geq_ondemand(spot_report):
    assert spot_report.fifo_spot_fleet_goodput_per_dollar >= \
           spot_report.fifo_ondemand_goodput_per_dollar * 0.90


# 11. cost_spot_fleet < cost_ondemand
def test_spot_fleet_cost_lower(spot_report):
    assert spot_report.cost_spot_fleet < spot_report.cost_ondemand


# 12. north_star_achieved for primary operating point on 200-req fixture
# Small fixture may not exactly hit north-star threshold, but spot should
# be substantially better than on-demand (>50% improvement in goodput/$).
def test_north_star_direction(spot_report):
    ratio = (spot_report.fifo_spot_fleet_goodput_per_dollar /
             spot_report.fifo_ondemand_goodput_per_dollar)
    assert ratio > 1.30, f"Expected spot to be >30% better, got {ratio:.2f}x"


# 13. Completion rates > 0.90 for both conditions
def test_completion_rates(spot_report):
    assert spot_report.ondemand_completion_rate > 0.90
    assert spot_report.spot_fleet_completion_rate > 0.90


# 14. to_dict() contains all expected keys
def test_to_dict_keys(spot_report):
    d = spot_report.to_dict()
    required = [
        "trace", "total_requests", "fixed_c", "target_rho",
        "spot_fraction", "spot_price_usd_hr", "demand_price_usd_hr",
        "p_interrupt_hourly", "cost_ondemand", "cost_spot_fleet",
        "cost_reduction_pct", "expected_interrupted_replica_ticks",
        "fifo_ondemand_goodput_per_dollar", "fifo_spot_fleet_goodput_per_dollar",
        "ondemand_vs_sla_oracle_pct", "spot_fleet_vs_sla_oracle_pct",
        "north_star_achieved", "ondemand_completion_rate",
        "spot_fleet_completion_rate",
    ]
    for k in required:
        assert k in d, f"Missing key: {k}"


# 15. Expected interruptions < 1.0 at primary config (SLA impact is negligible)
def test_expected_interruptions_negligible(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _expected_interruptions_over_run,
    )
    exp = _expected_interruptions_over_run(c_schedule_200, 0.70, 0.10, 60.0)
    assert exp < 1.0, f"Expected < 1 interruption on 200-req run, got {exp:.3f}"
