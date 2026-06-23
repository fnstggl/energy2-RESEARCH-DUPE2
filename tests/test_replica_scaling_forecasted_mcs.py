"""Tests for the fully-deployable ``forecasted_mcs`` mode of ReplicaScalingPolicy.

This mode is the only ReplicaScalingPolicy mode that uses NO future information
(it forecasts both next-tick arrivals and service time from data <= t-1). The
other modes (amcsg / sotss_min / online_sotss) size from actual tick-t arrival
counts. See research/MCS_AUDIT.md.
"""

from __future__ import annotations

import pytest

from aurelius.benchmarks.forecasted_mcs import (
    forecast_mcs_c_schedule,
    reactive_lag1_c_schedule,
)
from aurelius.optimizer.policies.replica_scaling import (
    ReplicaScalingConfig,
    ReplicaScalingPolicy,
)


def _trace(n_ticks=12, per_tick=30, tick_s=60.0, warp=1.0, seed=0):
    import random
    rng = random.Random(seed)
    raw = []
    for t in range(n_ticks):
        count = per_tick + (t % 5) * 4
        for j in range(count):
            raw.append(((t + (j + 0.5) / count) * tick_s * warp, 40 + rng.randint(0, 120)))
    raw.sort()
    return raw


def test_forecasted_mcs_mode_runs_via_policy():
    raw = _trace()
    pol = ReplicaScalingPolicy(
        config=ReplicaScalingConfig(mode="forecasted_mcs", sla_s=10.0, tick_seconds=60.0)
    )
    res = pol.optimize(raw, warp=1.0)
    assert res.mode == "forecasted_mcs"
    assert res.n_ticks == 12
    assert len(res.c_schedule) == 12
    assert all(isinstance(c, int) and c >= 1 for c in res.c_schedule)


def test_forecasted_mcs_parity_with_direct_call():
    raw = _trace(seed=2)
    cfg = ReplicaScalingConfig(
        mode="forecasted_mcs", sla_s=10.0, tick_seconds=60.0,
        safe_gate_pct=12.5, forecast_method="ewma", forecast_ewma_alpha=0.5,
        forecast_warmup_c=4,
    )
    res = ReplicaScalingPolicy(config=cfg).optimize(raw, warp=1.0)
    c_direct, _ = forecast_mcs_c_schedule(
        raw, 60.0, 1.0, method="ewma", mcs_gate=12.5, sla_s=10.0,
        ewma_alpha=0.5, warmup_c=4,
    )
    assert res.c_schedule == c_direct


def test_forecasted_mcs_lag1_parity():
    raw = _trace(seed=3)
    cfg = ReplicaScalingConfig(
        mode="forecasted_mcs", sla_s=10.0, tick_seconds=60.0,
        safe_gate_pct=12.5, forecast_method="lag1", forecast_warmup_c=4,
    )
    res = ReplicaScalingPolicy(config=cfg).optimize(raw, warp=1.0)
    c_direct = reactive_lag1_c_schedule(raw, 60.0, 1.0, mcs_gate=12.5, sla_s=10.0, warmup_c=4)
    assert res.c_schedule == c_direct


def test_forecasted_mcs_is_causal_through_policy():
    """Perturbing tick t's demand must not change c[<=t] (the deployable invariant)."""
    raw = _trace(n_ticks=12, per_tick=30)
    cfg = ReplicaScalingConfig(mode="forecasted_mcs", sla_s=10.0, tick_seconds=60.0)
    base = ReplicaScalingPolicy(config=cfg).optimize(raw, warp=1.0).c_schedule

    perturb = 6
    spike = [((perturb + 0.5) * 60.0, 300) for _ in range(300)]
    raw2 = sorted(raw + spike)
    pert = ReplicaScalingPolicy(config=cfg).optimize(raw2, warp=1.0).c_schedule
    for t in range(perturb + 1):
        assert base[t] == pert[t], f"forecasted_mcs leaked future info at tick {t}"


def test_unknown_forecast_method_raises():
    cfg = ReplicaScalingConfig(mode="forecasted_mcs", forecast_method="transformer")
    with pytest.raises(ValueError):
        ReplicaScalingPolicy(config=cfg).optimize(_trace(), warp=1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
