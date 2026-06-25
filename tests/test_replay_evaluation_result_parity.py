"""Parity tests for ReplayEvaluationResult Phase 1b-B.

Verifies that:
1. Each adapter produces a ReplayEvaluationResult with the correct KPI value
   (matches the source object exactly — 0% KPI drift by construction).
2. to_dict() emits all required keys.
3. Adapters do not mutate source objects.
4. All BENCHMARK_IDS are covered by at least one adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aurelius.optimizer.replay_result import (
    BENCHMARK_IDS,
    ReplayEvaluationResult,
    from_backtest_policy_result,
    from_canonical_policy_metrics,
    from_genai_policy_result,
    from_srtf_sim_dict,
)

# ---------------------------------------------------------------------------
# Minimal mock objects (duck-typed — same field names as real objects)
# ---------------------------------------------------------------------------

@dataclass
class _MockKPI:
    """Minimal stand-in for EconomicKPIResult."""
    sla_safe_goodput_per_infra_dollar: float
    sla_compliant_goodput: int
    total_infrastructure_cost: float
    active_gpu_hours: float


@dataclass
class _MockBacktestPolicyResult:
    """Minimal stand-in for aurelius.traces.backtest.PolicyResult."""
    policy: str
    kpi: Any
    timeout_rate_pct_mean: float
    scale_events: int
    latency_p99_ms: float


@dataclass
class _MockGenAIPolicyResult:
    """Minimal stand-in for aurelius.traces.genai_backtest.PolicyResult."""
    policy: str
    kpi: Any
    sla_compliant_requests: int
    replica_hours: float
    timeout_rate_pct: float
    scale_events: int
    e2e_p99_s: float


@dataclass
class _MockCanonicalPolicyMetrics:
    """Minimal stand-in for aurelius.benchmarks.canonical_backtests.PolicyMetrics."""
    policy: str
    sla_safe_goodput_per_infra_dollar: float
    sla_compliant_goodput: float
    total_infra_cost_usd: float
    deadline_misses: int
    realized_energy_cost_usd: float
    migrations: int


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_kpi():
    return _MockKPI(
        sla_safe_goodput_per_infra_dollar=150630.0,
        sla_compliant_goodput=5823,
        total_infrastructure_cost=0.0387,
        active_gpu_hours=229.25,
    )


@pytest.fixture
def mock_backtest_result(mock_kpi):
    return _MockBacktestPolicyResult(
        policy="constraint_aware",
        kpi=mock_kpi,
        timeout_rate_pct_mean=2.34,
        scale_events=12,
        latency_p99_ms=9800.0,
    )


@pytest.fixture
def mock_genai_kpi():
    return _MockKPI(
        sla_safe_goodput_per_infra_dollar=42.5,
        sla_compliant_goodput=850,
        total_infrastructure_cost=20.0,
        active_gpu_hours=8.5,
    )


@pytest.fixture
def mock_genai_result(mock_genai_kpi):
    return _MockGenAIPolicyResult(
        policy="constraint_aware",
        kpi=mock_genai_kpi,
        sla_compliant_requests=850,
        replica_hours=8.5,
        timeout_rate_pct=1.2,
        scale_events=3,
        e2e_p99_s=2.8,
    )


@pytest.fixture
def mock_canonical_metrics():
    return _MockCanonicalPolicyMetrics(
        policy="constraint_aware",
        sla_safe_goodput_per_infra_dollar=0.33730,
        sla_compliant_goodput=1024.0,
        total_infra_cost_usd=3035.0,
        deadline_misses=0,
        realized_energy_cost_usd=2910.0,
        migrations=5,
    )


@pytest.fixture
def mock_srtf_sim_dict():
    return {
        "requests": 5880,
        "servers": 4,
        "mean_response_s": 1.23,
        "short_p90_response_s": 0.87,
        "long_p99_response_s": 8.4,
        "sla_safe_goodput_per_dollar": 18543.7,
        "preemption_count": 0,
    }


# ---------------------------------------------------------------------------
# from_backtest_policy_result
# ---------------------------------------------------------------------------

def test_from_backtest_policy_result_type(mock_backtest_result):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    assert isinstance(r, ReplayEvaluationResult)


def test_from_backtest_policy_result_kpi_parity(mock_backtest_result, mock_kpi):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    assert r.kpi_sla_safe_goodput_per_dollar == mock_kpi.sla_safe_goodput_per_infra_dollar
    assert r.kpi_sla_compliant_goodput == float(mock_kpi.sla_compliant_goodput)
    assert r.kpi_gpu_hours == mock_kpi.active_gpu_hours
    assert r.kpi_total_cost == mock_kpi.total_infrastructure_cost


def test_from_backtest_policy_result_benchmark_id(mock_backtest_result):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    assert r.benchmark_id == "replica_scaling"
    assert r.benchmark_id in BENCHMARK_IDS


def test_from_backtest_policy_result_trace_meta(mock_backtest_result):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="burstgpt_v1", n_requests=51, n_ticks=55, tick_seconds=60.0,
    )
    assert r.trace_id == "burstgpt_v1"
    assert r.policy == "constraint_aware"
    assert r.n_requests == 51
    assert r.n_ticks == 55
    assert r.tick_seconds == 60.0


def test_from_backtest_policy_result_metadata(mock_backtest_result):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    assert r.metadata["goodput_unit"] == "tokens"
    assert r.metadata["timeout_rate_pct_mean"] == mock_backtest_result.timeout_rate_pct_mean
    assert r.metadata["scale_events"] == mock_backtest_result.scale_events


def test_from_backtest_does_not_mutate_source(mock_backtest_result):
    original_kpi = mock_backtest_result.kpi.sla_safe_goodput_per_infra_dollar
    from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    assert mock_backtest_result.kpi.sla_safe_goodput_per_infra_dollar == original_kpi


# ---------------------------------------------------------------------------
# from_genai_policy_result
# ---------------------------------------------------------------------------

def test_from_genai_policy_result_type(mock_genai_result):
    r = from_genai_policy_result(
        "constraint_aware", mock_genai_result,
        trace_id="alibaba_genai", n_requests=1000, n_ticks=1, tick_seconds=3600.0,
    )
    assert isinstance(r, ReplayEvaluationResult)


def test_from_genai_policy_result_kpi_parity(mock_genai_result, mock_genai_kpi):
    r = from_genai_policy_result(
        "constraint_aware", mock_genai_result,
        trace_id="alibaba_genai", n_requests=1000, n_ticks=1, tick_seconds=3600.0,
    )
    assert r.kpi_sla_safe_goodput_per_dollar == mock_genai_kpi.sla_safe_goodput_per_infra_dollar
    assert r.kpi_sla_compliant_goodput == float(mock_genai_result.sla_compliant_requests)
    assert r.kpi_gpu_hours == mock_genai_result.replica_hours
    assert r.kpi_total_cost == mock_genai_kpi.total_infrastructure_cost


def test_from_genai_policy_result_benchmark_id(mock_genai_result):
    r = from_genai_policy_result(
        "constraint_aware", mock_genai_result,
        trace_id="alibaba_genai", n_requests=1000, n_ticks=1, tick_seconds=3600.0,
    )
    assert r.benchmark_id == "genai_serving"
    assert r.benchmark_id in BENCHMARK_IDS


def test_from_genai_policy_result_metadata(mock_genai_result):
    r = from_genai_policy_result(
        "constraint_aware", mock_genai_result,
        trace_id="alibaba_genai", n_requests=1000, n_ticks=1, tick_seconds=3600.0,
    )
    assert r.metadata["goodput_unit"] == "requests"
    assert r.metadata["timeout_rate_pct"] == mock_genai_result.timeout_rate_pct
    assert r.metadata["e2e_p99_s"] == mock_genai_result.e2e_p99_s


# ---------------------------------------------------------------------------
# from_canonical_policy_metrics
# ---------------------------------------------------------------------------

def test_from_canonical_policy_metrics_type(mock_canonical_metrics):
    r = from_canonical_policy_metrics(
        "constraint_aware", mock_canonical_metrics,
        trace_id="canonical_energy", n_requests=100, n_ticks=720, tick_seconds=3600.0,
    )
    assert isinstance(r, ReplayEvaluationResult)


def test_from_canonical_policy_metrics_kpi_parity(mock_canonical_metrics):
    r = from_canonical_policy_metrics(
        "constraint_aware", mock_canonical_metrics,
        trace_id="canonical_energy", n_requests=100, n_ticks=720, tick_seconds=3600.0,
    )
    assert r.kpi_sla_safe_goodput_per_dollar == mock_canonical_metrics.sla_safe_goodput_per_infra_dollar
    assert r.kpi_sla_compliant_goodput == float(mock_canonical_metrics.sla_compliant_goodput)
    assert r.kpi_total_cost == mock_canonical_metrics.total_infra_cost_usd
    assert r.kpi_gpu_hours == 0.0  # not tracked in energy loop


def test_from_canonical_policy_metrics_benchmark_id(mock_canonical_metrics):
    r = from_canonical_policy_metrics(
        "constraint_aware", mock_canonical_metrics,
        trace_id="canonical_energy", n_requests=100, n_ticks=720, tick_seconds=3600.0,
    )
    assert r.benchmark_id == "energy"
    assert r.benchmark_id in BENCHMARK_IDS


def test_from_canonical_policy_metrics_metadata(mock_canonical_metrics):
    r = from_canonical_policy_metrics(
        "constraint_aware", mock_canonical_metrics,
        trace_id="canonical_energy", n_requests=100, n_ticks=720, tick_seconds=3600.0,
    )
    assert r.metadata["goodput_unit"] == "token_equivalent"
    assert r.metadata["deadline_misses"] == mock_canonical_metrics.deadline_misses
    assert r.metadata["migrations"] == mock_canonical_metrics.migrations


# ---------------------------------------------------------------------------
# from_srtf_sim_dict
# ---------------------------------------------------------------------------

def test_from_srtf_sim_dict_type(mock_srtf_sim_dict):
    r = from_srtf_sim_dict(
        "fifo", mock_srtf_sim_dict,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1, tick_seconds=3600.0,
        servers=4,
    )
    assert isinstance(r, ReplayEvaluationResult)


def test_from_srtf_sim_dict_kpi_parity(mock_srtf_sim_dict):
    r = from_srtf_sim_dict(
        "fifo", mock_srtf_sim_dict,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1, tick_seconds=3600.0,
        servers=4,
    )
    assert r.kpi_sla_safe_goodput_per_dollar == mock_srtf_sim_dict["sla_safe_goodput_per_dollar"]


def test_from_srtf_sim_dict_benchmark_id(mock_srtf_sim_dict):
    r = from_srtf_sim_dict(
        "srtf", mock_srtf_sim_dict,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1, tick_seconds=3600.0,
        servers=4,
    )
    assert r.benchmark_id == "serving_queue"
    assert r.benchmark_id in BENCHMARK_IDS


def test_from_srtf_sim_dict_metadata(mock_srtf_sim_dict):
    r = from_srtf_sim_dict(
        "fifo", mock_srtf_sim_dict,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1, tick_seconds=3600.0,
        servers=4,
    )
    assert r.metadata["servers"] == 4
    assert r.metadata["cost_basis"] == "busy_gpu_hours"
    assert r.metadata["mean_response_s"] == mock_srtf_sim_dict["mean_response_s"]


def test_from_srtf_sim_dict_missing_key():
    """Missing sla_safe_goodput_per_dollar defaults to 0.0 (pre-attachment dict)."""
    r = from_srtf_sim_dict(
        "fifo", {"mean_response_s": 1.5},
        trace_id="azure_llm_2024", n_requests=100, n_ticks=1, tick_seconds=3600.0,
        servers=4,
    )
    assert r.kpi_sla_safe_goodput_per_dollar == 0.0


# ---------------------------------------------------------------------------
# to_dict coverage
# ---------------------------------------------------------------------------

def test_to_dict_required_keys(mock_backtest_result):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    d = r.to_dict()
    for key in (
        "benchmark_id", "trace_id", "policy",
        "kpi_sla_safe_goodput_per_dollar", "kpi_sla_compliant_goodput",
        "kpi_gpu_hours", "kpi_total_cost",
        "n_requests", "n_ticks", "tick_seconds", "metadata",
    ):
        assert key in d, f"Missing key: {key}"


def test_to_dict_kpi_roundtrip(mock_backtest_result):
    r = from_backtest_policy_result(
        "constraint_aware", mock_backtest_result,
        trace_id="azure_llm_2024", n_requests=5880, n_ticks=1560, tick_seconds=60.0,
    )
    d = r.to_dict()
    assert abs(d["kpi_sla_safe_goodput_per_dollar"] - r.kpi_sla_safe_goodput_per_dollar) < 1e-3


# ---------------------------------------------------------------------------
# BENCHMARK_IDS coverage
# ---------------------------------------------------------------------------

def test_benchmark_ids_covered_by_adapters(
    mock_backtest_result, mock_genai_result, mock_canonical_metrics, mock_srtf_sim_dict,
):
    """Every BENCHMARK_ID is produced by exactly one adapter."""
    seen = set()
    seen.add(from_backtest_policy_result(
        "p", mock_backtest_result, trace_id="t", n_requests=1, n_ticks=1, tick_seconds=60.0,
    ).benchmark_id)
    seen.add(from_genai_policy_result(
        "p", mock_genai_result, trace_id="t", n_requests=1, n_ticks=1, tick_seconds=60.0,
    ).benchmark_id)
    seen.add(from_canonical_policy_metrics(
        "p", mock_canonical_metrics, trace_id="t", n_requests=1, n_ticks=1, tick_seconds=60.0,
    ).benchmark_id)
    seen.add(from_srtf_sim_dict(
        "p", mock_srtf_sim_dict, trace_id="t", n_requests=1, n_ticks=1, tick_seconds=60.0, servers=4,
    ).benchmark_id)
    assert seen == set(BENCHMARK_IDS)
