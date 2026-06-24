"""GenAI serving policy — Phase 3d architecture extraction.

Extracts the per-tick replica-sizing decision logic (``constraint_aware``:
EWMA anticipatory sizing + model-affinity cold-start routing) from the
``genai_backtest`` benchmark monolith into the canonical AureliusOptimizer
policy seam.  Follows the Phase 2/3 extraction pattern verbatim.

Decisions governed here:
  - Per-tick replica count for multi-model GenAI serving
  - EWMA anticipatory arrival smoothing (causal, alpha=0.5)
  - Affinity routing for cold-start amortisation (always True for this policy)
  - Erlang-C SLA-based minimum-replica sizing

Physics owned here (canonical; benchmark imports back):
  - ``genai_effective_service_s`` — mean per-request service time with cold-start
  - ``genai_eval_tick_timeout``   — Erlang-C p99 timeout rate for one tick
  - ``genai_size_for_sla``        — minimum SLA-safe replica count for one tick
  - ``genai_size_for_target``     — target-rho replica count for one tick

Zero behaviour change (verified by parity tests in
``tests/test_genai_canonical_routing_parity.py``).  No circular imports:
benchmark → policy (one direction only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .base import OptimizationPolicy

# ---------------------------------------------------------------------------
# Service / SLA constants — canonical owner (identical to genai_backtest.py)
# ---------------------------------------------------------------------------
GENAI_MIN_REPLICAS: int = 1
GENAI_SLA_LATENCY_MULT: float = 2.0
GENAI_SLA_LATENCY_ABS_S: float = 30.0
GENAI_TARGET_RHO_SLA: float = 0.65
GENAI_TARGET_RHO_UTIL: float = 0.85
GENAI_EWMA_ALPHA: float = 0.5


# ---------------------------------------------------------------------------
# Physics helpers — canonical owner; genai_backtest.py imports these back
# ---------------------------------------------------------------------------

def genai_effective_service_s(
    mean_exec_s: float,
    n: int,
    distinct_models: int,
    lora_frac: float,
    controlnet_frac: float,
    cold: dict,
    affinity: bool,
) -> float:
    """Per-request mean service time including model cold-start.

    Affinity routing routes requests to warm replicas, amortising the
    base-model reload over ``distinct_models`` arrivals in the tick.
    Non-affinity routing (load-balance) treats every request as a potential
    cold-start when multiple models are present.
    """
    if n == 0:
        return mean_exec_s
    if affinity:
        switch_rate = min(1.0, distinct_models / n)
    else:
        switch_rate = 1.0 if distinct_models > 1 else 0.0
    cold_s = (
        switch_rate * cold.get("basemodel_load", 0.0)
        + lora_frac * (switch_rate if affinity else 1.0) * cold.get("lora_load", 0.0)
        + controlnet_frac
        * (switch_rate if affinity else 1.0)
        * cold.get("controlnet_load", 0.0)
    )
    return mean_exec_s + cold_s


def genai_eval_tick_timeout(
    n: int,
    arrival_rate: float,
    mean_exec_s: float,
    distinct_models: int,
    lora_frac: float,
    controlnet_frac: float,
    replicas: int,
    cold: dict,
    affinity: bool,
) -> float:
    """Erlang-C p99 timeout rate (%) for one tick.

    Returns a value in [0, 50]: 0 means SLA is met; positive means the p99
    e2e latency exceeds the SLA budget.  The capped return of 50.0 matches
    the benchmark convention.
    """
    from aurelius.simulation.cluster import serving  # lazy to avoid heavy import at module load

    replicas = max(GENAI_MIN_REPLICAS, int(replicas))
    service_s = genai_effective_service_s(
        mean_exec_s, n, distinct_models, lora_frac, controlnet_frac, cold, affinity
    )
    mu = 1.0 / service_s if service_s > 0 else 1.0
    lam = arrival_rate
    rho = lam / (replicas * mu) if replicas > 0 else 1.0
    wait_s = serving.erlang_c_wait_s(lam, mu, replicas)
    if not math.isfinite(wait_s):
        wait_s = 60.0
    wait_s = min(600.0, wait_s * serving.saturation_amplifier(rho))
    _p95m, p99m = serving.tail_multipliers(rho)
    e2e_p99 = wait_s * (p99m / 2 + 1) + service_s
    sla = GENAI_SLA_LATENCY_ABS_S + GENAI_SLA_LATENCY_MULT * mean_exec_s
    return min(50.0, (e2e_p99 - sla) / sla * 10.0) if e2e_p99 > sla else 0.0


def genai_size_for_sla(
    n: int,
    arrival_rate: float,
    mean_exec_s: float,
    distinct_models: int,
    lora_frac: float,
    controlnet_frac: float,
    cold: dict,
    affinity: bool,
) -> int:
    """Minimum replica count satisfying the SLA for a given tick workload.

    Binary-searches from ``GENAI_MIN_REPLICAS`` to 4096 for the smallest
    ``r`` where the Erlang-C p99 timeout rate is exactly 0.
    """
    for r in range(GENAI_MIN_REPLICAS, 4096):
        if genai_eval_tick_timeout(
            n, arrival_rate, mean_exec_s, distinct_models,
            lora_frac, controlnet_frac, r, cold, affinity,
        ) <= 0.0:
            return r
    return 4096


def genai_size_for_target(
    n: int,
    arrival_rate: float,
    mean_exec_s: float,
    distinct_models: int,
    lora_frac: float,
    controlnet_frac: float,
    cold: dict,
    affinity: bool,
    target_rho: float,
) -> int:
    """Replica count that achieves ``target_rho`` utilisation for a given tick."""
    if arrival_rate <= 0:
        return GENAI_MIN_REPLICAS
    service_s = genai_effective_service_s(
        mean_exec_s, n, distinct_models, lora_frac, controlnet_frac, cold, affinity
    )
    mu = 1.0 / service_s if service_s > 0 else 1.0
    return max(GENAI_MIN_REPLICAS, int(math.ceil(arrival_rate / (mu * target_rho))))


# ---------------------------------------------------------------------------
# Policy result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GenAIServingResult:
    """Per-tick replica decisions returned by :class:`GenAIServingPolicy`.

    ``replica_counts[i]`` is the number of replicas Aurelius recommends for
    tick *i*.  ``affinity`` is always ``True`` for the ``constraint_aware``
    policy.
    """

    replica_counts: list = field(default_factory=list)
    affinity: bool = True
    mode: str = "constraint_aware"


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------

class GenAIServingPolicy(OptimizationPolicy):
    """GenAI serving policy — constraint_aware (EWMA anticipatory + affinity).

    Extracted verbatim from the ``constraint_aware`` branch of
    ``genai_backtest._run_policy``.  Behaviour is identical to calling
    that branch directly; the extraction makes the decision reachable through
    ``AureliusOptimizer(policy="genai_serving")``.

    Phase 3d contract (verified by parity tests):
      * ``optimize(ticks, cold)`` returns per-tick replica counts that are
        bit-identical to those computed inline in ``_run_policy``.
      * No new optimizer logic.  No new priors.
    """

    name: str = "genai_serving"

    def optimize(
        self,
        ticks: Sequence[Any],
        cold: dict,
        tick_hours: float = 1.0,
    ) -> GenAIServingResult:
        """Compute per-tick replica counts for the constraint_aware policy.

        Args:
            ticks: Sequence of tick aggregates.  Each element must expose
                ``n`` (int), ``arrival_rate`` (float), ``mean_exec_s`` (float),
                ``distinct_models`` (int), ``lora_frac`` (float),
                ``controlnet_frac`` (float).
            cold: Cold-start duration priors keyed by model type
                (``basemodel_load``, ``lora_load``, ``controlnet_load``).
            tick_hours: Tick duration in hours.  Not used in the sizing
                decision but kept for interface symmetry with the benchmark.

        Returns:
            :class:`GenAIServingResult` with per-tick replica counts.
        """
        ewma: float = 0.0
        replica_counts: list[int] = []

        for t in ticks:
            if t.n > 0:
                # causal EWMA — alpha=0.5, initialised from first non-zero tick
                ewma = (
                    GENAI_EWMA_ALPHA * t.arrival_rate
                    + (1.0 - GENAI_EWMA_ALPHA) * ewma
                    if ewma
                    else t.arrival_rate
                )

            if t.n:
                # constraint_aware: EWMA anticipatory sizing + affinity routing
                smoothed_rate = max(t.arrival_rate, ewma)
                r = genai_size_for_sla(
                    t.n,
                    smoothed_rate,
                    t.mean_exec_s,
                    t.distinct_models,
                    t.lora_frac,
                    t.controlnet_frac,
                    cold,
                    affinity=True,
                )
            else:
                r = GENAI_MIN_REPLICAS

            replica_counts.append(r)

        return GenAIServingResult(replica_counts=replica_counts, affinity=True)


__all__ = [
    "GenAIServingPolicy",
    "GenAIServingResult",
    "genai_effective_service_s",
    "genai_eval_tick_timeout",
    "genai_size_for_sla",
    "genai_size_for_target",
    "GENAI_MIN_REPLICAS",
    "GENAI_SLA_LATENCY_MULT",
    "GENAI_SLA_LATENCY_ABS_S",
    "GENAI_TARGET_RHO_SLA",
    "GENAI_TARGET_RHO_UTIL",
    "GENAI_EWMA_ALPHA",
]
