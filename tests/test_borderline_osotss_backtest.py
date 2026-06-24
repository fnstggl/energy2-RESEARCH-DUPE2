"""Tests for Oracle Soft-SLA Continuation (OSSC) OSOTSS.

Verifies:
  1. compute_online_sotss_schedule with borderline_margin_s>0 runs without
     error and returns a valid schedule (same length, non-negative counts).
  2. With borderline_margin_s=0.0 (default), results are byte-identical to
     pre-OSSC baseline — no regression for existing deployments.
  3. Positive borderline_margin_s can only add capacity vs margin=0 (never
     fewer replicas per tick).
  4. ReplicaScalingConfig.borderline_margin_s is wired through to policy.
  5. AureliusOptimizer with ReplicaScalingPolicy + borderline_margin_s routes
     through the canonical optimizer interface without error.
  6. BorderlineSweepEntry and BorderlineOSOTSSReport dataclasses are importable
     and fields populate correctly.

These are unit/integration tests; they do NOT run the full backtest on public
traces (that is in the benchmark itself). All tests use synthetic data so they
run in < 5s without network access.
"""

from __future__ import annotations

import pytest

from aurelius.optimizer.policies.replica_scaling import (
    ReplicaScalingConfig,
    ReplicaScalingPolicy,
    compute_online_sotss_schedule,
)

# ---------------------------------------------------------------------------
# Synthetic trace helpers
# ---------------------------------------------------------------------------

def _linear_trace(n_reqs: int = 200, tokens_per_req: int = 500) -> list[tuple[float, int]]:
    """Uniform inter-arrival, constant tokens."""
    return [(float(i * 2), tokens_per_req) for i in range(n_reqs)]


def _heavy_trace(n_reqs: int = 300) -> list[tuple[float, int]]:
    """High-token trace with bursts to stress the oracle convergence."""
    reqs = []
    for i in range(n_reqs):
        tokens = 2000 if i % 20 == 0 else 400
        reqs.append((float(i * 1.5), tokens))
    return reqs


# ---------------------------------------------------------------------------
# 1. Positive borderline_margin_s runs without error
# ---------------------------------------------------------------------------

def test_borderline_margin_s_runs_without_error():
    raw = _linear_trace(200)
    c_sched, n_iters, init_viol, n_cheaper, baseline = compute_online_sotss_schedule(
        raw,
        tick_seconds=60.0,
        warp=1.0,
        sla_s=10.0,
        borderline_margin_s=2.0,
    )
    assert isinstance(c_sched, list)
    assert len(c_sched) > 0
    assert all(c >= 1 for c in c_sched)
    assert n_iters >= 0
    assert baseline >= 0


def test_borderline_margin_s_various_values():
    raw = _linear_trace(200)
    for margin in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        c_sched, n_iters, _, _, _ = compute_online_sotss_schedule(
            raw,
            tick_seconds=60.0,
            warp=1.0,
            sla_s=10.0,
            borderline_margin_s=margin,
        )
        assert len(c_sched) > 0, f"empty schedule for borderline_margin_s={margin}"
        assert all(c >= 1 for c in c_sched), f"non-positive c for margin={margin}"


# ---------------------------------------------------------------------------
# 2. borderline_margin_s=0.0 is byte-identical to default (no regression)
# ---------------------------------------------------------------------------

def test_zero_borderline_margin_matches_default():
    raw = _linear_trace(200)
    default_result = compute_online_sotss_schedule(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0,
    )
    zero_result = compute_online_sotss_schedule(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0,
        borderline_margin_s=0.0,
    )
    assert default_result[0] == zero_result[0], "borderline_margin_s=0 changed schedule vs default"
    assert default_result[1] == zero_result[1], "borderline_margin_s=0 changed n_iters vs default"
    assert default_result[4] == zero_result[4], "borderline_margin_s=0 changed baseline vs default"


# ---------------------------------------------------------------------------
# 3. Positive borderline_margin_s can only add capacity (monotone)
# ---------------------------------------------------------------------------

def test_borderline_margin_s_monotone_capacity():
    """OSSC only adds capacity after primary convergence; c_sched[tk] can only
    increase or stay the same compared to margin=0 result."""
    raw = _heavy_trace(300)
    c_base, _, _, _, _ = compute_online_sotss_schedule(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0, borderline_margin_s=0.0,
    )
    c_margin, _, _, _, _ = compute_online_sotss_schedule(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0, borderline_margin_s=3.0,
    )
    assert len(c_base) == len(c_margin)
    for i, (cb, cm) in enumerate(zip(c_base, c_margin)):
        assert cm >= cb, f"OSSC reduced capacity at tick {i}: {cm} < {cb}"


# ---------------------------------------------------------------------------
# 4. ReplicaScalingConfig.borderline_margin_s wired through
# ---------------------------------------------------------------------------

def test_replica_scaling_config_borderline_field():
    cfg = ReplicaScalingConfig(
        mode="online_sotss",
        borderline_margin_s=2.0,
    )
    assert cfg.borderline_margin_s == 2.0


def test_replica_scaling_config_default_zero():
    cfg = ReplicaScalingConfig(mode="online_sotss")
    assert cfg.borderline_margin_s == 0.0


def test_replica_scaling_policy_borderline_runs():
    raw = _linear_trace(200)
    cfg = ReplicaScalingConfig(
        mode="online_sotss",
        borderline_margin_s=2.0,
        sla_s=10.0,
    )
    policy = ReplicaScalingPolicy()
    result = policy.optimize(raw, config=cfg)
    assert result.mode == "online_sotss"
    assert len(result.c_schedule) > 0
    assert all(c >= 1 for c in result.c_schedule)


# ---------------------------------------------------------------------------
# 5. AureliusOptimizer routes through canonical interface
# ---------------------------------------------------------------------------

def test_aurelius_optimizer_replica_scaling_with_borderline():
    from aurelius.optimizer import AureliusOptimizer
    raw = _linear_trace(100)
    cfg = ReplicaScalingConfig(
        mode="online_sotss",
        borderline_margin_s=1.0,
        sla_s=10.0,
    )
    opt = AureliusOptimizer(policy="replica_scaling")
    result = opt.optimize(raw, config=cfg)
    assert result.mode == "online_sotss"
    assert len(result.c_schedule) > 0


# ---------------------------------------------------------------------------
# 6. Dataclass importable and fields populate
# ---------------------------------------------------------------------------

def test_borderline_sweep_entry_importable():
    from aurelius.benchmarks.borderline_osotss_backtest import BorderlineSweepEntry
    e = BorderlineSweepEntry(
        trace="azure_llm_2024",
        borderline_margin_s=2.0,
        goodput_per_dollar=160_000.0,
        n_sla_safe=5830,
        cost=0.5,
        c_mean=3.2,
        p99_s=8.5,
        n_iters=42,
        amcsg_goodput_per_dollar=150_630.0,
        amcsg_n_sla_safe=5823,
        vs_amcsg_pct=6.2,
        vs_osotss_baseline_pct=0.3,
        sla_safe_vs_amcsg=7,
    )
    assert e.borderline_margin_s == 2.0
    assert e.sla_safe_vs_amcsg == 7
    assert e.vs_amcsg_pct == pytest.approx(6.2)


def test_borderline_osotss_report_importable():
    from aurelius.benchmarks.borderline_osotss_backtest import (
        BorderlineOSOTSSReport,
        BorderlineSweepEntry,
    )
    entry = BorderlineSweepEntry(
        trace="burstgpt_hf",
        borderline_margin_s=0.0,
        goodput_per_dollar=178_109.0,
        n_sla_safe=5849,
        cost=0.4,
        c_mean=2.8,
        p99_s=22.0,
        n_iters=30,
        amcsg_goodput_per_dollar=168_270.0,
        amcsg_n_sla_safe=5864,
        vs_amcsg_pct=5.85,
        vs_osotss_baseline_pct=0.0,
        sla_safe_vs_amcsg=-15,
    )
    report = BorderlineOSOTSSReport(
        azure_sweep=[entry],
        burstgpt_sweep=[entry],
        best_azure_margin=0.0,
        best_burstgpt_margin=2.0,
        is_frontier_improvement=False,
        best_joint_margin=None,
        verdict="NEGATIVE_RESULT",
    )
    assert report.verdict == "NEGATIVE_RESULT"
    assert report.best_burstgpt_margin == 2.0
    assert report.is_frontier_improvement is False
