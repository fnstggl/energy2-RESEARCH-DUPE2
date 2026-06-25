"""First-class optimization layers for the canonical ``AureliusOptimizer``.

Target architecture (``research/CANONICAL_AURELIUS_OPTIMIZER.md``)::

    AureliusOptimizer
    ├── ForecastLayer
    ├── ConstraintLayer
    ├── ObjectiveLayer: SLA-safe goodput/$
    ├── DecisionLayer (the registered policies)
    ├── ReplayLayer
    └── EvaluationLayer

Before this module only the **DecisionLayer** (the policies) was a first-class,
optimizer-owned component; the other concerns existed but **scattered** across
the repo (audit 2026-06-25). These classes are **thin wrappers** over the
existing, tested implementations — **no new optimization logic, no rewrite, no
benchmark-assumption change** — so the layered architecture becomes real and
each layer has exactly one named owner. The energy core stays "do not modify".

Honest scope notes are inline. In particular the :class:`ForecastLayer`
documents that **no ML forecaster has a positive measured goodput/$ delta in
production** (output-length HURTS −7..−11%, gpu_placement −7.3%, admission
neutral, forecasting <0.3% of the Azure win) — so it surfaces only the one
*causal* forecast the optimizer actually uses for a decision, and labels the
rest advisory/research. This module does not pretend the advisory forecasters
drive decisions.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

# The objective every layer ultimately serves (docs/RESULTS.md §1).
CANONICAL_OBJECTIVE = "sla_safe_goodput_per_infrastructure_dollar"


# ---------------------------------------------------------------------------
# Objective Layer
# ---------------------------------------------------------------------------

class ObjectiveLayer:
    """First-class SLA-safe goodput/$ objective (``docs/RESULTS.md`` §1).

    Makes the canonical objective something the optimizer can **score and
    compare against**, closing the split-brain gap where the energy core
    minimizes a weighted *cost* (``optimization/objective.py``) while
    goodput/$ was only ever computed *after the fact* by the benchmarks. Wraps
    the frozen KPI math in ``aurelius/benchmarks/economics.py`` (unchanged).
    """

    name = CANONICAL_OBJECTIVE

    def __init__(self, cost_config=None, sla_filter=None):
        # InfrastructureCostConfig / SLAFilterConfig (operator-overridable cost
        # basis). Defaulted lazily so importing this layer stays light.
        self._cost_config = cost_config
        self._sla_filter = sla_filter

    def score(
        self,
        *,
        sla_compliant_goodput: Optional[int] = None,
        total_infrastructure_cost: Optional[float] = None,
        kpi: Any = None,
    ) -> Optional[float]:
        """Return SLA-safe goodput/$ for one result.

        Pass either ``(sla_compliant_goodput, total_infrastructure_cost)`` or a
        precomputed ``kpi`` (an ``EconomicKPIResult``, a dict, or any object
        exposing ``sla_safe_goodput_per_infra_dollar``).
        """
        if kpi is not None:
            if isinstance(kpi, Mapping):
                return kpi.get("sla_safe_goodput_per_infra_dollar")
            return getattr(kpi, "sla_safe_goodput_per_infra_dollar", None)
        if sla_compliant_goodput is None or total_infrastructure_cost is None:
            raise ValueError(
                "ObjectiveLayer.score needs either kpi= or both "
                "sla_compliant_goodput= and total_infrastructure_cost=."
            )
        from ..benchmarks.economics import (
            compute_sla_safe_goodput_per_infra_dollar,
        )

        return compute_sla_safe_goodput_per_infra_dollar(
            int(sla_compliant_goodput), float(total_infrastructure_cost)
        )

    def evaluate(self, **tick_series) -> Any:
        """Full ``EconomicKPIResult`` from per-tick simulator series.

        Forwards to ``economics.compute_economic_kpi`` with this layer's cost
        basis / SLA filter. ``tick_series`` keys are that function's kwargs
        (``tokens_per_tick``, ``timeout_rate_pct_per_tick``,
        ``energy_cost_per_tick``, ``active_gpu_hours_by_type_per_tick``,
        ``migration_count``, …).
        """
        from ..benchmarks.economics import compute_economic_kpi

        return compute_economic_kpi(
            config=self._cost_config, sla_filter=self._sla_filter, **tick_series
        )

    def compare(self, arms: Mapping[str, Any]) -> list[tuple[str, Optional[float]]]:
        """Rank candidate arms by goodput/$ (highest first; ``None`` last).

        ``arms`` maps an arm name to either a numeric goodput/$ score or a
        KPI object/dict this layer can ``score``. Use for the Phase C three-way
        Current-Main vs Best-Aurelius vs Candidate comparison.
        """
        scored: list[tuple[str, Optional[float]]] = []
        for name, value in arms.items():
            if value is None:
                s: Optional[float] = None
            elif isinstance(value, (int, float)):
                s = float(value)
            else:
                s = self.score(kpi=value)
            scored.append((name, s))
        scored.sort(key=lambda kv: (kv[1] is None, -(kv[1] if kv[1] is not None else 0.0)))
        return scored


# ---------------------------------------------------------------------------
# Constraint Layer
# ---------------------------------------------------------------------------

class ConstraintLayer:
    """First-class constraint gate.

    Unifies the binding-constraint classification + SLA gate + safe-utilization
    recommendations behind one named owner by wrapping
    ``aurelius/constraints/engine.py`` (``ConstraintAwareEngine``,
    recommendation-only by construction). The energy path's hard feasibility
    (deadline / power-cap / region) stays inside ``JobScheduler`` /
    ``optimization/constraints.py`` — this layer is the *serving/live* constraint
    seam the optimizer consults via ``recommend_live``.
    """

    def __init__(self, engine=None):
        self._engine = engine

    @property
    def engine(self):
        """The wrapped ``ConstraintAwareEngine`` (lazily built; recommendation-only)."""
        if self._engine is None:
            from ..constraints.engine import ConstraintAwareEngine

            self._engine = ConstraintAwareEngine()
        return self._engine

    def assess(self, state) -> Any:
        """Classify the binding constraint for a ``ClusterState`` (no actions)."""
        return self.engine.classifier.assess(state)

    def gate(self, state, sla_registry=None) -> Any:
        """Full constraint pass → gated recommendations (recommendation-only).

        Returns an ``EngineResult``; never mutates the cluster.
        """
        return self.engine.run(state, sla_registry)


# ---------------------------------------------------------------------------
# Forecast Layer
# ---------------------------------------------------------------------------

class ForecastLayer:
    """First-class forecast contract — with an HONEST scope (audit 2026-06-25).

    **No ML forecaster has a positive measured goodput/$ delta wired into
    production today**: output-length forecasting HURTS −7..−11%, the GPU
    placement scorer regressed −7.3%, admission is neutral, and demand
    forecasting is <0.3% of the Azure +25.75% (the KPI wins come from
    utilization/target-ρ and heuristic price corrections, not ML). The
    offline train→promote→drift pipeline is real code but operationally dormant
    (no persisted artifacts). So this layer:

    * surfaces the ONE causal-in-decision forecaster the optimizer actually uses
      (``forecasted_mcs``: next-tick arrivals + service from data ≤ t-1), and
    * classifies the rest honestly (advisory price/carbon that is wired but
      low-leverage; research-only ML that is neutral/negative or shadow).

    It does **not** pretend the advisory/shadow forecasters drive decisions.
    """

    #: Forecasters that feed a REAL decision today (causal, deployable).
    DECISION_FEEDING = ("forecasted_mcs_capacity",)
    #: Wired into the energy path but ADVISORY / low-leverage (<0.3% of KPI).
    ADVISORY = ("price_quantile", "carbon_quantile", "regime", "spread_risk")
    #: Trained-or-heuristic but RESEARCH-ONLY — neutral or NEGATIVE on the KPI,
    #: or shadow-only / un-persisted. Do NOT wire without new evidence.
    RESEARCH_ONLY = (
        "cara_latency", "output_length", "cache_prefix",
        "gpu_placement", "economic_ml",
    )

    def classify(self) -> dict[str, tuple]:
        """Return the honest forecaster taxonomy (decision/advisory/research)."""
        return {
            "decision_feeding": self.DECISION_FEEDING,
            "advisory": self.ADVISORY,
            "research_only": self.RESEARCH_ONLY,
        }

    def causal_capacity_forecast(
        self,
        raw: list,
        tick_seconds: float,
        warp: float,
        *,
        method: str = "ewma",
        mcs_gate: float = 12.5,
        sla_s: float = 10.0,
        **kwargs,
    ) -> list[int]:
        """The only causal-in-decision forecast: next-tick replica capacity.

        Forecasts BOTH next-tick arrivals AND service time from data ≤ t-1 (no
        oracle), then sizes per-tick replicas. Delegates to
        ``aurelius/benchmarks/forecasted_mcs.py`` — the same path
        ``ReplicaScalingPolicy(mode="forecasted_mcs")`` uses. ``method`` ∈
        {``"ewma"``, ``"quantile"``, ``"lag1"``}.
        """
        if method == "lag1":
            from ..benchmarks.forecasted_mcs import reactive_lag1_c_schedule

            return reactive_lag1_c_schedule(
                raw, tick_seconds, warp, mcs_gate=mcs_gate, sla_s=sla_s, **kwargs
            )
        from ..benchmarks.forecasted_mcs import forecast_mcs_c_schedule

        sched, _diag = forecast_mcs_c_schedule(
            raw, tick_seconds, warp, method=method, mcs_gate=mcs_gate,
            sla_s=sla_s, **kwargs,
        )
        return sched


# ---------------------------------------------------------------------------
# Replay Layer
# ---------------------------------------------------------------------------

class ReplayLayer:
    """First-class replay-result normalizer.

    Wraps the 4 cross-loop adapters in ``aurelius/optimizer/replay_result.py``
    (Phase 1b-B) so any backtest loop's native result normalizes to one
    ``ReplayEvaluationResult`` schema. **Honest limit:** this unifies the
    *result schema* across the 4 loops, NOT the loops themselves — a single
    unified replay *engine* (Phase 1b-A) remains future work, and is gated on a
    0%-delta parity harness. The 4 discrete-event loops still run separately.
    """

    _ADAPTERS = {
        "backtest": "from_backtest_policy_result",
        "genai": "from_genai_policy_result",
        "canonical": "from_canonical_policy_metrics",
        "srtf": "from_srtf_sim_dict",
    }

    def normalize(self, kind: str, policy_name: str, result: Any, **kwargs) -> Any:
        """Normalize a loop-native result to ``ReplayEvaluationResult``.

        ``kind`` ∈ {``"backtest"``, ``"genai"``, ``"canonical"``, ``"srtf"``};
        ``kwargs`` are the adapter's keyword args (``trace_id``, ``n_requests``,
        ``n_ticks``, ``tick_seconds``, and ``servers`` for ``srtf``).
        """
        if kind not in self._ADAPTERS:
            raise ValueError(
                f"ReplayLayer.normalize: unknown kind {kind!r}. "
                f"Valid: {sorted(self._ADAPTERS)}."
            )
        from . import replay_result as _rr

        adapter = getattr(_rr, self._ADAPTERS[kind])
        return adapter(policy_name, result, **kwargs)

    @property
    def benchmark_ids(self) -> tuple:
        """The canonical benchmark id tuple (from ``replay_result.BENCHMARK_IDS``)."""
        from .replay_result import BENCHMARK_IDS

        return BENCHMARK_IDS


# ---------------------------------------------------------------------------
# Evaluation Layer
# ---------------------------------------------------------------------------

class EvaluationLayer:
    """First-class evaluation layer.

    Owns the frozen KPI math (``economics.py``) + the fair-baseline selection
    and outcome classification (``per_workload.py``: ``select_headline_baseline``
    / ``analyze_outcome``). This is the substrate every benchmark already shares;
    wrapping it as a named layer lets the optimizer evaluate a decision against
    the SAME standard the public benchmarks use (``docs/RESULTS.md`` §3/§6/§7).
    """

    def compute_kpi(self, **tick_series) -> Any:
        """Canonical ``EconomicKPIResult`` from per-tick series (frozen math)."""
        from ..benchmarks.economics import compute_economic_kpi

        return compute_economic_kpi(**tick_series)

    def select_baseline(self, metadata, policy_results) -> tuple:
        """Pick the fair headline baseline for a scenario (``RESULTS.md`` §3).

        Returns ``(headline_name, rationale)`` — never silently FIFO when a
        stronger relevant safe baseline exists.
        """
        from ..benchmarks.per_workload import select_headline_baseline

        return select_headline_baseline(metadata, policy_results)

    def classify_outcome(
        self, metadata, ca_kpi, headline_kpi, all_baseline_kpis, **kwargs
    ) -> Any:
        """Classify ALPHA_WIN / SAFETY_WIN / TIE / LOSS / KEEP_CORRECT (§6)."""
        from ..benchmarks.per_workload import analyze_outcome

        return analyze_outcome(
            metadata, ca_kpi, headline_kpi, all_baseline_kpis, **kwargs
        )


__all__ = [
    "CANONICAL_OBJECTIVE",
    "ObjectiveLayer",
    "ConstraintLayer",
    "ForecastLayer",
    "ReplayLayer",
    "EvaluationLayer",
]
