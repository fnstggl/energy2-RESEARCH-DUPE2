"""Unified per-policy replay evaluation result for cross-loop comparison.

Phase 1b-B: introduces the shared schema that allows results from different
Aurelius replay loops to be placed side by side.  No existing loop code is
modified — this module only reads from existing result objects (0% KPI drift).

Four replay loops are supported:
  replica_scaling — aurelius.traces.backtest (BurstGPT / Azure LLM 2024)
  genai_serving   — aurelius.traces.genai_backtest (Alibaba GenAI)
  energy          — aurelius.benchmarks.canonical_backtests (batch GPU jobs)
  serving_queue   — aurelius.benchmarks.srtf_serving_backtest (SRTF / SRPT)

Cross-loop comparison caveat: ``kpi_sla_safe_goodput_per_dollar`` is the
canonical metric, but its cost basis differs across loops:

  replica_scaling / genai_serving / energy:
      provisioned GPU-hours × GPU-hour rate (wall-clock × replicas)
  serving_queue:
      busy GPU-hours (sum of actual service times) × GPU-hour rate

Use ``kpi_sla_safe_goodput_per_dollar`` comparisons within the same loop only;
cross-loop ratios are informational and require explicit cost normalization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


#: Valid benchmark_id values — one per replay loop.
BENCHMARK_IDS = (
    "replica_scaling",   # aurelius.traces.backtest
    "serving_queue",     # aurelius.benchmarks.srtf_serving_backtest
    "genai_serving",     # aurelius.traces.genai_backtest
    "energy",            # aurelius.benchmarks.canonical_backtests
)


@dataclass
class ReplayEvaluationResult:
    """Normalized per-policy result from any Aurelius replay loop.

    Populated by the ``from_*`` adapter functions below.  No existing loop code
    is changed; 0% KPI drift is guaranteed by construction (adapters are
    read-only projections of the source result objects).

    Field units
    -----------
    kpi_sla_safe_goodput_per_dollar
        tokens/$ for replica_scaling, serving_queue, energy;
        requests/$ for genai_serving.  See ``metadata["goodput_unit"]``.
    kpi_sla_compliant_goodput
        Same unit as the goodput numerator (0.0 if unavailable for the loop).
    kpi_gpu_hours
        Provisioned GPU-hours (0.0 if unavailable).
    kpi_total_cost
        Total infrastructure cost in USD (0.0 if unavailable).
    """

    benchmark_id: str   # one of BENCHMARK_IDS
    trace_id: str       # e.g. "azure_llm_2024", "burstgpt_v1", "canonical_energy"
    policy: str         # policy name within this loop

    # Primary KPI — canonical across all loops (units differ; see above)
    kpi_sla_safe_goodput_per_dollar: float

    # Secondary KPIs
    kpi_sla_compliant_goodput: float  # tokens or requests (see metadata)
    kpi_gpu_hours: float              # GPU-hours provisioned
    kpi_total_cost: float             # total infrastructure cost ($)

    # Trace size
    n_requests: int
    n_ticks: int
    tick_seconds: float

    # Loop-specific supplementary data
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "benchmark_id": self.benchmark_id,
            "trace_id": self.trace_id,
            "policy": self.policy,
            "kpi_sla_safe_goodput_per_dollar": round(
                self.kpi_sla_safe_goodput_per_dollar, 6
            ),
            "kpi_sla_compliant_goodput": round(self.kpi_sla_compliant_goodput, 2),
            "kpi_gpu_hours": round(self.kpi_gpu_hours, 4),
            "kpi_total_cost": round(self.kpi_total_cost, 4),
            "n_requests": self.n_requests,
            "n_ticks": self.n_ticks,
            "tick_seconds": self.tick_seconds,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Adapter functions — one per replay loop
# ---------------------------------------------------------------------------

def from_backtest_policy_result(
    policy_name: str,
    result: Any,
    *,
    trace_id: str,
    n_requests: int,
    n_ticks: int,
    tick_seconds: float,
) -> ReplayEvaluationResult:
    """Adapter for ``aurelius.traces.backtest.PolicyResult``.

    The ``result.kpi`` is an ``EconomicKPIResult`` from
    ``aurelius.benchmarks.economics``.
    """
    kpi = result.kpi
    return ReplayEvaluationResult(
        benchmark_id="replica_scaling",
        trace_id=trace_id,
        policy=policy_name,
        kpi_sla_safe_goodput_per_dollar=kpi.sla_safe_goodput_per_infra_dollar,
        kpi_sla_compliant_goodput=float(kpi.sla_compliant_goodput),
        kpi_gpu_hours=kpi.active_gpu_hours,
        kpi_total_cost=kpi.total_infrastructure_cost,
        n_requests=n_requests,
        n_ticks=n_ticks,
        tick_seconds=tick_seconds,
        metadata={
            "goodput_unit": "tokens",
            "timeout_rate_pct_mean": result.timeout_rate_pct_mean,
            "scale_events": result.scale_events,
            "latency_p99_ms": result.latency_p99_ms,
        },
    )


def from_genai_policy_result(
    policy_name: str,
    result: Any,
    *,
    trace_id: str,
    n_requests: int,
    n_ticks: int,
    tick_seconds: float,
) -> ReplayEvaluationResult:
    """Adapter for ``aurelius.traces.genai_backtest.PolicyResult``.

    The ``result.kpi`` is an ``EconomicKPIResult``.
    """
    kpi = result.kpi
    return ReplayEvaluationResult(
        benchmark_id="genai_serving",
        trace_id=trace_id,
        policy=policy_name,
        kpi_sla_safe_goodput_per_dollar=kpi.sla_safe_goodput_per_infra_dollar,
        kpi_sla_compliant_goodput=float(result.sla_compliant_requests),
        kpi_gpu_hours=result.replica_hours,
        kpi_total_cost=kpi.total_infrastructure_cost,
        n_requests=n_requests,
        n_ticks=n_ticks,
        tick_seconds=tick_seconds,
        metadata={
            "goodput_unit": "requests",
            "timeout_rate_pct": result.timeout_rate_pct,
            "scale_events": result.scale_events,
            "e2e_p99_s": result.e2e_p99_s,
        },
    )


def from_canonical_policy_metrics(
    policy_name: str,
    metrics: Any,
    *,
    trace_id: str,
    n_requests: int,
    n_ticks: int,
    tick_seconds: float,
) -> ReplayEvaluationResult:
    """Adapter for ``aurelius.benchmarks.canonical_backtests.PolicyMetrics``.

    ``kpi_gpu_hours`` is 0.0 because the energy loop does not track provisioned
    GPU-hours directly; it tracks energy cost.
    """
    return ReplayEvaluationResult(
        benchmark_id="energy",
        trace_id=trace_id,
        policy=policy_name,
        kpi_sla_safe_goodput_per_dollar=metrics.sla_safe_goodput_per_infra_dollar,
        kpi_sla_compliant_goodput=float(metrics.sla_compliant_goodput),
        kpi_gpu_hours=0.0,
        kpi_total_cost=metrics.total_infra_cost_usd,
        n_requests=n_requests,
        n_ticks=n_ticks,
        tick_seconds=tick_seconds,
        metadata={
            "goodput_unit": "token_equivalent",
            "deadline_misses": metrics.deadline_misses,
            "realized_energy_cost_usd": metrics.realized_energy_cost_usd,
            "migrations": metrics.migrations,
        },
    )


def from_srtf_sim_dict(
    policy_name: str,
    sim_dict: dict,
    *,
    trace_id: str,
    n_requests: int,
    n_ticks: int,
    tick_seconds: float,
    servers: int,
) -> ReplayEvaluationResult:
    """Adapter for a ``srtf_serving_backtest`` per-discipline simulation dict.

    ``sim_dict`` is one of the per-discipline dicts (e.g. ``SRTFServingReport.fifo``)
    returned by any ``run_*_backtest`` in ``srtf_serving_backtest.py``, after
    ``sla_safe_goodput_per_dollar`` has been attached to it by the run function.

    ``kpi_sla_compliant_goodput``, ``kpi_gpu_hours``, and ``kpi_total_cost`` are
    0.0 because the SRTF loop does not store these in the sim dict.  The cost basis
    is busy_gpu_hours (sum of service times), not provisioned GPU-hours.
    """
    return ReplayEvaluationResult(
        benchmark_id="serving_queue",
        trace_id=trace_id,
        policy=policy_name,
        kpi_sla_safe_goodput_per_dollar=sim_dict.get("sla_safe_goodput_per_dollar", 0.0),
        kpi_sla_compliant_goodput=0.0,
        kpi_gpu_hours=0.0,
        kpi_total_cost=0.0,
        n_requests=n_requests,
        n_ticks=n_ticks,
        tick_seconds=tick_seconds,
        metadata={
            "goodput_unit": "tokens",
            "servers": servers,
            "cost_basis": "busy_gpu_hours",
            "mean_response_s": sim_dict.get("mean_response_s", 0.0),
            "short_p90_response_s": sim_dict.get("short_p90_response_s", 0.0),
            "long_p99_response_s": sim_dict.get("long_p99_response_s", 0.0),
        },
    )
