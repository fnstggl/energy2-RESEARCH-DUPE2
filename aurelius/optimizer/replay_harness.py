"""Unified replay harness — Phase 1b-A.

Provides a single entry point (``ReplayHarness.run``) that dispatches to the
appropriate Aurelius replay backend and returns normalized
``ReplayEvaluationResult`` objects using the Phase 1b-B adapters.

Physics guarantee: all KPI computation stays in the individual backends.
This module is a pure routing facade — 0% KPI drift by construction.

Architecture::

    ReplayHarness.run(config, data)
        ├── "replica_scaling"  → aurelius.traces.backtest.run_backtest()
        │                      → from_backtest_policy_result()
        ├── "genai_serving"    → aurelius.traces.genai_backtest.run_backtest()
        │                      → from_genai_policy_result()
        ├── "energy"           → aurelius.benchmarks.canonical_backtests.run_canonical_backtest()
        │                      → from_canonical_policy_metrics()
        └── "serving_queue"    → pass-through adapter (pre-computed sim_dict)
                               → from_srtf_sim_dict()

Design notes
------------
- The ``serving_queue`` backend wraps the 14k-LOC SRTF engine via a pass-through
  interface: callers supply a ``dict`` with ``"sim_dicts"`` (``{policy: sim_dict}``)
  already computed by any ``run_*_backtest`` call in
  ``aurelius.benchmarks.srtf_serving_backtest``.  This avoids encoding 14k LOC of
  specialised physics behind a single entry-point signature while still normalising
  the output to ``ReplayEvaluationResult``.
- Backend imports are deferred (inside methods) so the harness can be imported
  without triggering heavyweight dependency loads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from .replay_result import (
    BENCHMARK_IDS,
    ReplayEvaluationResult,
    from_backtest_policy_result,
    from_canonical_policy_metrics,
    from_genai_policy_result,
    from_srtf_sim_dict,
)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class ReplayHarnessError(ValueError):
    """Raised when the harness cannot dispatch the given configuration."""


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ReplayHarnessConfig:
    """Unified configuration for any Aurelius replay loop.

    ``benchmark_id`` selects the backend; ``policies`` lists which optimizer
    policies to evaluate.  ``backend_kwargs`` are forwarded verbatim to the
    chosen backend's ``run_backtest()`` call so callers can pass backend-specific
    options (e.g. ``frontier_integration``) without the harness having to
    enumerate them explicitly.

    Parameters
    ----------
    benchmark_id:
        One of ``BENCHMARK_IDS`` — selects the replay backend.
    trace_id:
        Human-readable identifier for the trace (e.g. ``"azure_llm_2024"``).
        Used only for labelling ``ReplayEvaluationResult.trace_id``; does not
        affect physics.
    policies:
        Policy names to evaluate.  Forwarded to the backend's ``policies``
        kwarg where supported (``replica_scaling``, ``genai_serving``).  For
        ``energy`` and ``serving_queue`` the backend controls which policies run;
        the harness filters the result to this list.
    tick_seconds:
        Tick length in seconds.  Default 60 s (1 min) matches BurstGPT/Azure
        replica-scaling convention.
    backend_kwargs:
        Extra keyword arguments forwarded verbatim to the backend call.
    """

    benchmark_id: str
    trace_id: str
    policies: Sequence[str]
    tick_seconds: float = 60.0
    backend_kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.benchmark_id not in BENCHMARK_IDS:
            raise ReplayHarnessError(
                f"Unknown benchmark_id {self.benchmark_id!r}; "
                f"valid ids: {BENCHMARK_IDS}"
            )
        if not self.policies:
            raise ReplayHarnessError("policies must be non-empty")
        if self.tick_seconds <= 0:
            raise ReplayHarnessError("tick_seconds must be positive")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class ReplayHarness:
    """Unified entry point for all Aurelius replay loops.

    Phase 1b-A: creates the single dispatch layer that converts any replay
    backend's per-policy result into the normalized ``ReplayEvaluationResult``
    schema from Phase 1b-B.  No physics changes — all KPI computation remains
    in the individual backends.

    Usage::

        harness = ReplayHarness()
        cfg = ReplayHarnessConfig(
            benchmark_id="replica_scaling",
            trace_id="burstgpt_v1",
            policies=["constraint_aware", "amcsg"],
        )
        results = harness.run(cfg, requests)   # list[ReplayEvaluationResult]

    Each ``ReplayEvaluationResult`` has the same
    ``kpi_sla_safe_goodput_per_dollar`` value as calling the backend directly.
    KPI identity is verified by ``tests/test_phase1b_a_replay_harness_parity.py``.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        config: ReplayHarnessConfig,
        data: Any,
    ) -> list[ReplayEvaluationResult]:
        """Run a replay and return normalized results for all requested policies.

        ``data`` format depends on ``config.benchmark_id``:

        - ``"replica_scaling"``:
            ``Sequence[NormalizedLLMRequest]`` — forwarded to
            ``aurelius.traces.backtest.run_backtest()``.
        - ``"genai_serving"``:
            ``Sequence[NormalizedGenAIRequest]`` — forwarded to
            ``aurelius.traces.genai_backtest.run_backtest()``.
        - ``"energy"``:
            ``dict`` — forwarded to
            ``aurelius.benchmarks.canonical_backtests.run_canonical_backtest()``
            via ``config.backend_kwargs`` (``data`` is not used for this loop;
            pass ``{}`` or ``None``).
        - ``"serving_queue"``:
            ``dict`` with keys:

            - ``"sim_dicts"``: ``dict[str, dict]`` mapping policy name →
              simulation result dict (as returned by any
              ``run_*_backtest`` call in ``srtf_serving_backtest``).
            - ``"n_requests"``: ``int``
            - ``"n_ticks"``:   ``int``
            - ``"servers"``:   ``int``

        Returns a list of ``ReplayEvaluationResult``, one per evaluated policy,
        in the order ``config.policies`` is iterated.  Policies absent from the
        backend's output are silently skipped.
        """
        _dispatch = {
            "replica_scaling": self._run_replica_scaling,
            "genai_serving":   self._run_genai_serving,
            "energy":          self._run_energy,
            "serving_queue":   self._run_serving_queue,
        }
        return _dispatch[config.benchmark_id](config, data)

    # ------------------------------------------------------------------
    # Private dispatch methods — one per benchmark_id
    # ------------------------------------------------------------------

    def _run_replica_scaling(
        self,
        config: ReplayHarnessConfig,
        requests: Any,
    ) -> list[ReplayEvaluationResult]:
        """Dispatch to ``aurelius.traces.backtest.run_backtest``."""
        from aurelius.traces.backtest import run_backtest  # noqa: PLC0415

        bt = run_backtest(
            requests,
            tick_seconds=config.tick_seconds,
            policies=list(config.policies),
            **config.backend_kwargs,
        )
        results = []
        for policy in config.policies:
            if policy not in bt.policy_results:
                continue
            pr = bt.policy_results[policy]
            results.append(
                from_backtest_policy_result(
                    policy,
                    pr,
                    trace_id=config.trace_id,
                    n_requests=bt.n_requests,
                    n_ticks=bt.n_ticks,
                    tick_seconds=bt.tick_seconds,
                )
            )
        return results

    def _run_genai_serving(
        self,
        config: ReplayHarnessConfig,
        requests: Any,
    ) -> list[ReplayEvaluationResult]:
        """Dispatch to ``aurelius.traces.genai_backtest.run_backtest``."""
        from aurelius.traces.genai_backtest import (  # noqa: PLC0415
            run_backtest as _run_genai,
        )

        gt = _run_genai(
            requests,
            tick_seconds=config.tick_seconds,
            policies=list(config.policies),
            **config.backend_kwargs,
        )
        results = []
        for policy in config.policies:
            if policy not in gt.policy_results:
                continue
            pr = gt.policy_results[policy]
            results.append(
                from_genai_policy_result(
                    policy,
                    pr,
                    trace_id=config.trace_id,
                    n_requests=gt.n_requests,
                    n_ticks=gt.n_ticks,
                    tick_seconds=gt.tick_seconds,
                )
            )
        return results

    def _run_energy(
        self,
        config: ReplayHarnessConfig,
        data: Any,  # not used — energy loop builds its own data internally
    ) -> list[ReplayEvaluationResult]:
        """Dispatch to ``canonical_backtests.run_canonical_backtest``.

        The energy backend builds its own synthetic workload internally via
        ``build_canonical_jobs`` / ``load_canonical_price_data``, so ``data``
        is not consumed.  Pass ``{}`` or ``None``.

        Extra kwargs for ``run_canonical_backtest`` (``seed``, ``job_count``,
        ``method``) may be supplied via ``config.backend_kwargs``.
        """
        from aurelius.benchmarks.canonical_backtests import (  # noqa: PLC0415
            run_canonical_backtest,
        )

        summary = run_canonical_backtest(**config.backend_kwargs)
        requested = set(config.policies)
        results = []
        for policy in config.policies:
            if policy not in summary.policies:
                continue
            metrics = summary.policies[policy]
            results.append(
                from_canonical_policy_metrics(
                    policy,
                    metrics,
                    trace_id=config.trace_id,
                    n_requests=summary.job_count,
                    # Energy loop has no fixed tick count in CanonicalBacktestSummary
                    n_ticks=0,
                    tick_seconds=config.tick_seconds,
                )
            )
        _ = requested  # silence "unused" lint — used for ordering above
        return results

    def _run_serving_queue(
        self,
        config: ReplayHarnessConfig,
        data: dict,
    ) -> list[ReplayEvaluationResult]:
        """Pass-through adapter for SRTF serving backtest results.

        The ``srtf_serving_backtest`` engine exposes dozens of specialised
        entry-points with heterogeneous signatures.  Rather than encoding a
        single routing call for all of them, the harness accepts pre-computed
        simulation result dicts (``sim_dicts``) from any SRTF entry-point and
        normalises their output via ``from_srtf_sim_dict``.

        ``data`` must be a ``dict`` with:

        - ``"sim_dicts"``: ``dict[str, dict]`` — mapping ``policy_name`` → sim
          dict as returned by an SRTF ``run_*_backtest`` function.  Each sim
          dict must have a ``"sla_safe_goodput_per_dollar"`` key.
        - ``"n_requests"``: ``int`` — total request count.
        - ``"n_ticks"``:   ``int`` — total tick count (use 0 if unavailable).
        - ``"servers"``:   ``int`` — GPU server count for cost basis labelling.
        """
        sim_dicts: dict = data["sim_dicts"]
        n_requests: int = int(data.get("n_requests", 0))
        n_ticks: int = int(data.get("n_ticks", 0))
        servers: int = int(data.get("servers", 1))

        results = []
        for policy in config.policies:
            if policy not in sim_dicts:
                continue
            results.append(
                from_srtf_sim_dict(
                    policy,
                    sim_dicts[policy],
                    trace_id=config.trace_id,
                    n_requests=n_requests,
                    n_ticks=n_ticks,
                    tick_seconds=config.tick_seconds,
                    servers=servers,
                )
            )
        return results
