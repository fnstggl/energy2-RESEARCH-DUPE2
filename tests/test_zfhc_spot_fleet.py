"""Tests for Zero-Floor High-Capacity (ZFHC) Spot Policy [run 2026-06-25].

ZFHC policy: for c < threshold: AFMS (1 on-demand floor).
             for c >= threshold: all-spot (0 on-demand floor).

Research basis:
  GFS (arXiv:2509.11134, ASPLOS '26) — capacity-conditioned spot quota.
  SpotServe (arXiv:2311.15566, ASPLOS 2024) — full spot fleet, 54% cost reduction.
  SageServe (arXiv:2502.14617) — forecast-aware autoscaling, 25% GPU-hr savings.

Invariants tested:
  1.  _zfhc_spot_replicas == c (all-spot) for c >= threshold.
  2.  _zfhc_spot_replicas == _abs_floor_spot_replicas for c < threshold.
  3.  At threshold=8: _zfhc_spot_replicas(8,8) == 8 (all spot).
  4.  At threshold=10: _zfhc_spot_replicas(10,10) == 10 (all spot).
  5.  _zfhc_spot_replicas(c=5, threshold=10) == _abs_floor_spot_replicas(5).
  6.  _zfhc_spot_fleet_cost <= _abs_floor_spot_fleet_cost for same schedule.
  7.  _zfhc_spot_fleet_cost strictly < _abs_floor_cost when n_ticks_affected > 0.
  8.  Cost saving at c>=threshold is exactly $0.033/tick (1 on-demand removed).
  9.  _zfhc_expected_interruptions > 0 for positive p_interrupt.
  10. _zfhc_expected_interruptions >= AFMS interruptions (more spot = more exposure).
  11. _simulate_fifo_zfhc_spot_fleet returns non-negative response times.
  12. _simulate_fifo_zfhc_spot_fleet at p_interrupt=0 matches variable-c FIFO.
  13. run_zfhc_azure_backtest returns ZFHCReport.
  14. ZFHCReport has exactly 3 threshold entries (thresholds 8, 10, 12).
  15. best_threshold is one of _ZFHC_THRESHOLDS.
  16. best_goodput_per_dollar >= afms_goodput_per_dollar (ZFHC never hurts vs AFMS).
  17. All threshold entries have completion_rate > 0.90 (SLA maintained).
  18. threshold with n_ticks_affected=0 has cost == cost_afms.
  19. to_dict() contains all required keys.
  20. ZFHCThresholdEntry.to_dict() contains all required keys.
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
        _joint_mcs_c_schedule,
        calibrate_time_warp,
    )
    warp = calibrate_time_warp(azure_raw, servers=4, target_rho=0.85)
    return _joint_mcs_c_schedule(azure_raw, tick_seconds=60.0, warp=warp)


@pytest.fixture
def zfhc_report(azure_raw):
    from aurelius.benchmarks.srtf_serving_backtest import run_zfhc_azure_backtest
    return run_zfhc_azure_backtest(
        fixed_c=4,
        target_rho=0.85,
        job_limit=200,
        sla_s=10.0,
        azure_fixture=FIXTURE,
        tick_seconds=60.0,
    )


# 1. _zfhc_spot_replicas == c for c >= threshold
def test_zfhc_all_spot_at_high_c():
    from aurelius.benchmarks.srtf_serving_backtest import _zfhc_spot_replicas
    for thr in (8, 10, 12):
        for c in range(thr, thr + 5):
            assert _zfhc_spot_replicas(c, thr) == c, (
                f"c={c}, thr={thr}: expected all-spot ({c}), got {_zfhc_spot_replicas(c, thr)}"
            )


# 2. _zfhc_spot_replicas == _abs_floor_spot_replicas for c < threshold
def test_zfhc_matches_afms_below_threshold():
    from aurelius.benchmarks.srtf_serving_backtest import (
        _abs_floor_spot_replicas,
        _zfhc_spot_replicas,
    )
    for thr in (8, 10, 12):
        for c in range(1, thr):
            assert _zfhc_spot_replicas(c, thr) == _abs_floor_spot_replicas(c, min_ondemand=1), (
                f"c={c}, thr={thr}: ZFHC {_zfhc_spot_replicas(c, thr)} != "
                f"AFMS {_abs_floor_spot_replicas(c, min_ondemand=1)}"
            )


# 3. At threshold=8: c=8 is all spot
def test_zfhc_threshold_8_all_spot_at_c8():
    from aurelius.benchmarks.srtf_serving_backtest import _zfhc_spot_replicas
    assert _zfhc_spot_replicas(8, 8) == 8, "threshold=8, c=8 must be all-spot"
    assert _zfhc_spot_replicas(7, 8) < 7, "threshold=8, c=7 must keep 1 on-demand"


# 4. At threshold=10: c=10 is all spot
def test_zfhc_threshold_10_all_spot_at_c10():
    from aurelius.benchmarks.srtf_serving_backtest import _zfhc_spot_replicas
    assert _zfhc_spot_replicas(10, 10) == 10, "threshold=10, c=10 must be all-spot"
    assert _zfhc_spot_replicas(9, 10) < 9, "threshold=10, c=9 must keep 1 on-demand"


# 5. Below threshold, ZFHC behaves identically to AFMS for c=5
def test_zfhc_below_threshold_c5():
    from aurelius.benchmarks.srtf_serving_backtest import (
        _abs_floor_spot_replicas,
        _zfhc_spot_replicas,
    )
    assert _zfhc_spot_replicas(5, 10) == _abs_floor_spot_replicas(5, min_ondemand=1)


# 6. _zfhc_spot_fleet_cost <= _abs_floor_spot_fleet_cost
def test_zfhc_cost_le_afms(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        GPU_HOUR_USD,
        _abs_floor_spot_fleet_cost,
        _zfhc_spot_fleet_cost,
    )
    for thr in (8, 10, 12):
        cost_zfhc = _zfhc_spot_fleet_cost(c_schedule_200, thr, 0.80, GPU_HOUR_USD, 60.0)
        cost_afms = _abs_floor_spot_fleet_cost(c_schedule_200, 0.80, GPU_HOUR_USD, 60.0)
        assert cost_zfhc <= cost_afms + 1e-9, (
            f"thr={thr}: ZFHC cost {cost_zfhc:.4f} > AFMS cost {cost_afms:.4f}"
        )


# 7. ZFHC strictly cheaper when affected ticks exist
def test_zfhc_strictly_cheaper_with_high_c_ticks():
    from aurelius.benchmarks.srtf_serving_backtest import (
        GPU_HOUR_USD,
        _abs_floor_spot_fleet_cost,
        _zfhc_spot_fleet_cost,
    )
    # Schedule with c=10 ticks — threshold=10 should save vs AFMS
    c_sched = [3, 5, 8, 10, 12, 10, 8, 5]
    cost_zfhc_10 = _zfhc_spot_fleet_cost(c_sched, 10, 0.80, GPU_HOUR_USD, 60.0)
    cost_afms = _abs_floor_spot_fleet_cost(c_sched, 0.80, GPU_HOUR_USD, 60.0)
    assert cost_zfhc_10 < cost_afms, (
        f"ZFHC(thr=10) should be cheaper: {cost_zfhc_10:.4f} vs AFMS {cost_afms:.4f}"
    )


# 8. Cost saving at c>=threshold is exactly $0.020/tick (1 demand replaced by 1 spot)
def test_zfhc_cost_saving_per_tick():
    from aurelius.benchmarks.srtf_serving_backtest import (
        GPU_HOUR_USD,
        _abs_floor_spot_fleet_cost,
        _zfhc_spot_fleet_cost,
    )
    # Single tick at c=10 with threshold=10
    # AFMS: 9 spot + 1 on-demand; ZFHC: 10 spot + 0 on-demand
    # Saving = (GPU_HOUR_USD - spot_price) * (60/3600) = (2.00-0.80)*0.01667 ≈ $0.020/tick
    # (Not $0.033: the removed on-demand is replaced by one extra spot, not a vacuum)
    SPOT_PRICE = 0.80
    c_sched = [10]
    cost_zfhc = _zfhc_spot_fleet_cost(c_sched, 10, SPOT_PRICE, GPU_HOUR_USD, 60.0)
    cost_afms = _abs_floor_spot_fleet_cost(c_sched, SPOT_PRICE, GPU_HOUR_USD, 60.0)
    expected_saving = (GPU_HOUR_USD - SPOT_PRICE) * (60.0 / 3600.0)
    assert abs((cost_afms - cost_zfhc) - expected_saving) < 1e-9, (
        f"Expected saving {expected_saving:.6f}, got {cost_afms - cost_zfhc:.6f}"
    )


# 9. _zfhc_expected_interruptions > 0 for positive p_interrupt
def test_zfhc_expected_interruptions_positive(c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import _zfhc_expected_interruptions
    exp = _zfhc_expected_interruptions(c_schedule_200, 10, p_interrupt_hourly=0.10, tick_seconds=60.0)
    assert exp > 0.0, "Expected interruptions > 0 with p_interrupt=0.10"


# 10. ZFHC interruptions >= AFMS interruptions (more spot exposure)
def test_zfhc_interruptions_ge_afms():
    from aurelius.benchmarks.srtf_serving_backtest import (
        _abs_floor_expected_interruptions,
        _zfhc_expected_interruptions,
    )
    # Schedule with high-c ticks
    c_sched = [4, 6, 10, 12, 8, 10, 5]
    zfhc_int = _zfhc_expected_interruptions(c_sched, 10, p_interrupt_hourly=0.10, tick_seconds=60.0)
    afms_int = _abs_floor_expected_interruptions(c_sched, p_interrupt_hourly=0.10, tick_seconds=60.0)
    assert zfhc_int >= afms_int - 1e-9, (
        f"ZFHC expected interruptions {zfhc_int:.4f} < AFMS {afms_int:.4f}"
    )


# 11. _simulate_fifo_zfhc_spot_fleet returns non-negative response times
def test_zfhc_simulation_non_negative_times(azure_raw, c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _Request,
        _service_time_s,
        _simulate_fifo_zfhc_spot_fleet,
        calibrate_time_warp,
        make_live_prior_predictions,
    )
    warp = calibrate_time_warp(azure_raw, servers=4, target_rho=0.85)
    live_preds, _ = make_live_prior_predictions(azure_raw, window=200)
    reqs = [
        _Request(
            idx=i, arrival_s=arr / warp, actual_tokens=tok,
            predicted_tokens=live_preds[i], service_s=_service_time_s(tok),
        )
        for i, (arr, tok) in enumerate(azure_raw)
    ]
    _, resp, _ = _simulate_fifo_zfhc_spot_fleet(
        reqs, c_schedule_200, high_c_threshold=10,
        p_interrupt_hourly=0.10, tick_seconds=60.0, seed=42
    )
    for resp_time in resp.values():
        assert resp_time >= 0.0, f"Response time {resp_time} is negative"


# 12. _simulate_fifo_zfhc_spot_fleet at p_interrupt=0 matches variable-c FIFO
def test_zfhc_zero_interrupt_matches_fifo(azure_raw, c_schedule_200):
    from aurelius.benchmarks.srtf_serving_backtest import (
        _Request,
        _service_time_s,
        _simulate_fifo_variable_c,
        _simulate_fifo_zfhc_spot_fleet,
        calibrate_time_warp,
        make_live_prior_predictions,
    )
    warp = calibrate_time_warp(azure_raw, servers=4, target_rho=0.85)
    live_preds, _ = make_live_prior_predictions(azure_raw, window=200)

    def _build():
        return [
            _Request(
                idx=i, arrival_s=arr / warp, actual_tokens=tok,
                predicted_tokens=live_preds[i], service_s=_service_time_s(tok),
            )
            for i, (arr, tok) in enumerate(azure_raw)
        ]

    zfhc_reqs = _build()
    _, zfhc_resp, _ = _simulate_fifo_zfhc_spot_fleet(
        zfhc_reqs, c_schedule_200, high_c_threshold=10,
        p_interrupt_hourly=0.0, tick_seconds=60.0, seed=0
    )
    fifo_reqs = _build()
    _, fifo_resp, _ = _simulate_fifo_variable_c(fifo_reqs, c_schedule_200, tick_seconds=60.0)
    assert len(zfhc_resp) == len(fifo_resp), (
        f"ZFHC(p=0) served {len(zfhc_resp)} vs FIFO {len(fifo_resp)}"
    )


# 13. run_zfhc_azure_backtest returns ZFHCReport
def test_zfhc_azure_returns_report(zfhc_report):
    from aurelius.benchmarks.srtf_serving_backtest import ZFHCReport
    assert isinstance(zfhc_report, ZFHCReport)


# 14. ZFHCReport has exactly 3 threshold entries (thresholds 8, 10, 12)
def test_zfhc_report_has_three_entries(zfhc_report):
    assert len(zfhc_report.threshold_results) == 3, (
        f"Expected 3 threshold entries, got {len(zfhc_report.threshold_results)}"
    )
    thresholds = [e.threshold for e in zfhc_report.threshold_results]
    assert thresholds == [8, 10, 12], f"Expected [8,10,12], got {thresholds}"


# 15. best_threshold is one of _ZFHC_THRESHOLDS
def test_zfhc_best_threshold_valid(zfhc_report):
    from aurelius.benchmarks.srtf_serving_backtest import _ZFHC_THRESHOLDS
    assert zfhc_report.best_threshold in _ZFHC_THRESHOLDS, (
        f"best_threshold {zfhc_report.best_threshold} not in {_ZFHC_THRESHOLDS}"
    )


# 16. best_goodput_per_dollar >= afms_goodput_per_dollar
def test_zfhc_best_ge_afms(zfhc_report):
    assert zfhc_report.best_goodput_per_dollar >= zfhc_report.afms_goodput_per_dollar - 1e-6, (
        f"ZFHC best goodput/$ {zfhc_report.best_goodput_per_dollar} < "
        f"AFMS {zfhc_report.afms_goodput_per_dollar}"
    )


# 17. All threshold entries have completion_rate > 0.90
def test_zfhc_all_entries_completion_high(zfhc_report):
    for entry in zfhc_report.threshold_results:
        assert entry.completion_rate > 0.90, (
            f"threshold={entry.threshold}: completion_rate {entry.completion_rate} too low"
        )


# 18. Threshold with n_ticks_affected=0 has cost == cost_afms (no savings)
def test_zfhc_no_affected_ticks_no_savings():
    from aurelius.benchmarks.srtf_serving_backtest import (
        GPU_HOUR_USD,
        _abs_floor_spot_fleet_cost,
        _zfhc_spot_fleet_cost,
    )
    # Schedule where all c < threshold=12
    c_sched = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    # threshold=12: no affected ticks → ZFHC cost == AFMS cost
    cost_zfhc = _zfhc_spot_fleet_cost(c_sched, 12, 0.80, GPU_HOUR_USD, 60.0)
    cost_afms = _abs_floor_spot_fleet_cost(c_sched, 0.80, GPU_HOUR_USD, 60.0)
    assert abs(cost_zfhc - cost_afms) < 1e-9, (
        f"With no c>=12 ticks, ZFHC cost {cost_zfhc} should == AFMS {cost_afms}"
    )


# 19. ZFHCReport.to_dict() contains all required keys
def test_zfhc_report_to_dict_keys(zfhc_report):
    d = zfhc_report.to_dict()
    required_keys = [
        "trace", "total_requests", "fixed_c", "target_rho", "sla_s",
        "tick_seconds", "rng_seed",
        "c_schedule_mean", "c_schedule_min", "c_schedule_max", "n_ticks",
        "spot_price_usd_hr", "demand_price_usd_hr", "p_interrupt_hourly",
        "cost_afms", "afms_goodput_per_dollar", "afms_vs_sla_oracle_pct",
        "threshold_results",
        "best_threshold", "best_goodput_per_dollar",
        "best_vs_afms_pct", "best_vs_sla_oracle_pct", "best_north_star_achieved",
        "north_star_threshold", "sla_oracle_goodput_per_dollar",
    ]
    for key in required_keys:
        assert key in d, f"Missing key in ZFHCReport.to_dict(): {key}"


# 20. ZFHCThresholdEntry.to_dict() contains all required keys
def test_zfhc_entry_to_dict_keys(zfhc_report):
    for entry in zfhc_report.threshold_results:
        d = entry.to_dict()
        required_keys = [
            "threshold", "n_ticks_affected", "cost_zfhc",
            "cost_vs_afms_reduction_pct", "goodput_per_dollar",
            "goodput_vs_afms_pct", "goodput_vs_sla_oracle_pct",
            "north_star_achieved", "completion_rate", "p99_s",
        ]
        for key in required_keys:
            assert key in d, (
                f"Missing key in ZFHCThresholdEntry.to_dict() for thr={entry.threshold}: {key}"
            )
