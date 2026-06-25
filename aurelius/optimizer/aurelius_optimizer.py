"""AureliusOptimizer — the canonical, comprehensive fleet optimization interface.

This is the single top-level seam through which all Aurelius optimization gains
flow. It is **no longer a one-policy-at-a-time energy delegate** — it is a
comprehensive GPU-fleet optimizer that holds every implemented decision surface
and orchestrates them against the one objective in ``docs/RESULTS.md`` §1:

    **SLA-safe goodput per infrastructure dollar.**

    AureliusOptimizer
    └── DecisionLayer (all surfaces live; selected by workload/decision context)
        ├── EnergySchedulingPolicy   — when / where / how-fast (batch on price traces)
        ├── ServingQueuePolicy       — request ordering / preemption (SRPT+conformal)
        ├── ReplicaScalingPolicy     — per-tick replica capacity (deployable MCS)
        ├── PlacementPolicy          — GPU/region model placement & routing
        └── AdmissionPolicy          — flow-control admission (ADMIT/DEFER/REJECT)
    └── serving_orchestration        — live-service recommendations (ConstraintAwareEngine)

Two entry points:

  * :meth:`optimize` — runs the single *active* policy (selected by ``policy=``,
    default ``"energy"``). Behavior-preserving and parity-pinned
    (``tests/test_canonical_optimizer_parity.py``, ``test_energy_core_preservation.py``):
    ``AureliusOptimizer(cfg).optimize(jobs, price, carbon, method=...)`` is byte-
    identical to ``JobScheduler(cfg).solve(...)``.
  * :meth:`optimize_fleet` — the **comprehensive** productized interface. Given
    whatever decision inputs an operator supplies, it routes each one through the
    relevant real optimization surface and returns a unified
    :class:`FleetOptimizationResult`. It does **not** fabricate a single combined
    cross-surface number: energy and serving operate on disjoint workloads
    (``research/OPTIMIZER_UNIFICATION_PLAN.md`` §"Policy combination search"), so
    the result carries each surface's decision plus honest provenance.

The energy core (``JobScheduler``) remains "do not modify"
(``docs/ENERGY_SYSTEM_MAP.md`` §8, pinned by ``tests/test_energy_core_preservation.py``).
PlacementPolicy / AdmissionPolicy are parity wirings of existing, tested,
recommendation-only surfaces (``residency/``, ``frontier/admission.py``).

Example::

    from aurelius.optimizer import AureliusOptimizer
    opt = AureliusOptimizer(config)                 # holds ALL surfaces

    # single-surface (parity path):
    sched = opt.optimize(jobs, price_data, carbon_data, method="greedy")

    # comprehensive fleet pass (only the surfaces you supply run):
    fleet = opt.optimize_fleet(
        workload_class="inference_standard",
        admission={"sla_class": "llm_batch_inference", "window": ticks},
        capacity={"raw": arrivals},                 # deployable forecasted_mcs
        placement={"request": req, "locations": locs, "load_profiles": profiles},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..optimization.scheduler import JobScheduler, SchedulerResult
from .policies import (
    POLICY_REGISTRY,
    EnergySchedulingPolicy,
    OptimizationPolicy,
)

#: The objective every canonical surface targets (``docs/RESULTS.md`` §1).
CANONICAL_OBJECTIVE = "sla_safe_goodput_per_infrastructure_dollar"


@dataclass
class FleetOptimizationResult:
    """Unified result of a comprehensive :meth:`AureliusOptimizer.optimize_fleet`.

    Carries each engaged decision surface's output plus provenance. Every surface
    targets the shared objective (:data:`CANONICAL_OBJECTIVE`). This result
    deliberately does **not** combine the surfaces into one number — energy and
    serving operate on disjoint workloads, so a single combined goodput/$ is not
    honestly measurable today (see the module docstring).

    Attributes:
        workload_class: optional operator-supplied workload tag for the pass.
        energy: ``SchedulerResult`` (when/where/how-fast) or ``None``.
        admission: ``AdmissionDecision`` (ADMIT/DEFER/REJECT) or ``None``.
        placement: ``ResidencyDecision`` (route/prewarm/evict/keep) or ``None``.
        capacity: ``ReplicaScalingResult`` (per-tick replica counts) or ``None``.
        serving_order: ``(summary, response_map, wait_map)`` discipline result
            (discrete-event simulator; advisory) or ``None``.
        surfaces_used: names of the surfaces that actually ran.
        objective: the shared objective string.
        notes: honest caveats attached during the pass (deployability, advisory
            status, defaults applied).
    """

    workload_class: Optional[str] = None
    energy: Optional[object] = None
    admission: Optional[object] = None
    placement: Optional[object] = None
    capacity: Optional[object] = None
    serving_order: Optional[object] = None
    genai: Optional[object] = None
    surfaces_used: tuple = ()
    objective: str = CANONICAL_OBJECTIVE
    notes: tuple = ()


class AureliusOptimizer:
    """Canonical, comprehensive fleet optimizer.

    Holds the single *active* policy (for the parity ``optimize`` path) and lazily
    constructs the other decision surfaces on demand for the comprehensive
    ``optimize_fleet`` path and the per-surface convenience methods.
    """

    #: The decision-layer policy active by default for :meth:`optimize`.
    DEFAULT_POLICY: str = "energy"

    def __init__(
        self,
        config=None,
        *,
        policy: str = DEFAULT_POLICY,
        scheduler: Optional[JobScheduler] = None,
        **scheduler_kwargs,
    ):
        """Construct the canonical optimizer.

        Args:
            config: ``OptimizationConfig`` forwarded to a new ``JobScheduler``
                when the energy surface builds its own scheduler.
            policy: The *active* decision-layer policy name for :meth:`optimize`
                (default ``"energy"``). All surfaces remain reachable via
                :meth:`optimize_fleet` / the convenience methods regardless of
                this selection. An unknown name raises ``ValueError``.
            scheduler: Optional pre-built ``JobScheduler`` to delegate to
                (energy surface only). Mutually exclusive with
                ``config``/``scheduler_kwargs``.
            **scheduler_kwargs: Additional ``JobScheduler`` constructor kwargs
                forwarded verbatim (energy surface only).
        """
        if policy not in POLICY_REGISTRY:
            raise ValueError(
                f"Unknown policy {policy!r}. Known policies: "
                f"{sorted(POLICY_REGISTRY)}."
            )

        self.policy_name: str = policy
        # Retained so the comprehensive path can lazily (re)build the energy
        # surface and the other surfaces with the operator's configuration.
        self._config = config
        self._scheduler_kwargs = dict(scheduler_kwargs)
        self._injected_scheduler = scheduler
        # Cache of constructed decision surfaces (the active policy is registered
        # under its own name so it is reused, never rebuilt).
        self._surfaces: dict[str, OptimizationPolicy] = {}
        self._constraint_engine = None  # lazily built ConstraintAwareEngine

        if policy == EnergySchedulingPolicy.name:
            if scheduler is not None:
                if config is not None or scheduler_kwargs:
                    raise ValueError(
                        "AureliusOptimizer: pass either a prebuilt `scheduler` "
                        "or `config`/constructor kwargs, not both."
                    )
                self._policy: OptimizationPolicy = EnergySchedulingPolicy(
                    scheduler=scheduler
                )
            else:
                self._policy = EnergySchedulingPolicy(
                    config=config, **scheduler_kwargs
                )
        else:
            # Non-energy active policy. All implemented policies construct with
            # no scheduler/constructor args (they are facade-constructible).
            if scheduler is not None or scheduler_kwargs:
                raise ValueError(
                    f"Policy {policy!r} takes no scheduler/constructor arguments "
                    "(those are energy-surface only)."
                )
            self._policy = POLICY_REGISTRY[policy]()

        self._surfaces[policy] = self._policy

    # ------------------------------------------------------------------
    # Active-policy interface (parity path)
    # ------------------------------------------------------------------

    def optimize(self, *args, **kwargs):
        """Run the active decision-layer policy (selected by ``policy=``).

        For the default energy policy this delegates verbatim to
        ``JobScheduler.solve`` and returns the unchanged ``SchedulerResult``.
        """
        return self._policy.optimize(*args, **kwargs)

    def create_baseline_schedule(self, jobs):
        """Energy convenience: ASAP/home baseline via the wrapped engine."""
        baseline = getattr(self._policy, "create_baseline_schedule", None)
        if baseline is None:
            baseline = getattr(self.surface("energy"), "create_baseline_schedule")
        return baseline(jobs)

    # ------------------------------------------------------------------
    # Surface access (comprehensive path)
    # ------------------------------------------------------------------

    def surface(self, name: str) -> OptimizationPolicy:
        """Return the decision surface ``name``, building & caching it on demand.

        The active policy is reused; other surfaces are constructed lazily with
        the operator's configuration (energy reuses ``config``/injected scheduler).
        """
        if name not in POLICY_REGISTRY:
            raise ValueError(
                f"Unknown surface {name!r}. Known surfaces: {sorted(POLICY_REGISTRY)}."
            )
        cached = self._surfaces.get(name)
        if cached is not None:
            return cached

        if name == EnergySchedulingPolicy.name:
            if self._injected_scheduler is not None:
                built: OptimizationPolicy = EnergySchedulingPolicy(
                    scheduler=self._injected_scheduler
                )
            else:
                built = EnergySchedulingPolicy(
                    config=self._config, **self._scheduler_kwargs
                )
        else:
            built = POLICY_REGISTRY[name]()

        self._surfaces[name] = built
        return built

    # --- per-surface convenience methods (all route through `surface`) -------

    def schedule_energy(self, jobs, price_data, carbon_data, *args, **kwargs):
        """Energy scheduling surface: when (time-shift) / where (region) / how-fast."""
        return self.surface("energy").optimize(
            jobs, price_data, carbon_data, *args, **kwargs
        )

    def order_serving_queue(self, requests, servers, **kwargs):
        """Serving-queue surface: request ordering / preemption (SRPT+conformal)."""
        return self.surface("serving_queue").optimize(requests, servers, **kwargs)

    def serve_genai(self, ticks, cold, **kwargs):
        """GenAI serving surface: multi-model constraint_aware replica sizing."""
        return self.surface("genai_serving").optimize(ticks, cold, **kwargs)

    def scale_replicas(self, raw, **kwargs):
        """Replica-capacity surface: per-tick replica count schedule."""
        return self.surface("replica_scaling").optimize(raw, **kwargs)

    def place(self, request, locations, **kwargs):
        """Placement surface: where to place/route a model request (max goodput/$)."""
        return self.surface("placement").optimize(request, locations, **kwargs)

    def admit(self, *, sla_class, window, config=None):
        """Admission surface: ADMIT / DEFER / REJECT one incoming workload class."""
        return self.surface("admission").optimize(
            sla_class=sla_class, window=window, config=config
        )

    @property
    def serving_orchestration(self):
        """Live-service orchestration surface (``ConstraintAwareEngine``).

        Lazily built; ``recommendation_only`` by construction (never mutates a
        cluster). This is the surface the constraint CLI / live serving path use
        to classify the binding constraint and emit gated SCALE / SPREAD / REROUTE
        / MIGRATE recommendations. Brought under the canonical optimizer so the
        live path is no longer a separate, un-owned decision engine.
        """
        if self._constraint_engine is None:
            from ..constraints.engine import ConstraintAwareEngine

            self._constraint_engine = ConstraintAwareEngine()
        return self._constraint_engine

    def recommend_live(self, state, sla_registry=None):
        """Run one live-service recommendation cycle over a ``ClusterState``.

        Delegates to :attr:`serving_orchestration` (``ConstraintAwareEngine.run``);
        always recommendation-only. Returns an ``EngineResult``.
        """
        return self.serving_orchestration.run(state, sla_registry)

    # ------------------------------------------------------------------
    # Comprehensive interface
    # ------------------------------------------------------------------

    def optimize_fleet(
        self,
        *,
        workload_class: Optional[str] = None,
        energy: Optional[dict] = None,
        admission: Optional[dict] = None,
        placement: Optional[dict] = None,
        capacity: Optional[dict] = None,
        serving: Optional[dict] = None,
        genai: Optional[dict] = None,
        notes=(),
    ) -> FleetOptimizationResult:
        """Comprehensive fleet optimization across every supplied surface.

        Each argument is a dict of inputs for one decision surface; only the
        surfaces you supply run. Every surface targets SLA-safe goodput/$. The
        capacity surface defaults to the **deployable** ``forecasted_mcs`` mode
        (no future-token / arrival oracle) unless you pass an explicit
        ``config``/``mode`` — productized callers should never silently use an
        oracle provisioner.

        Expected dict shapes:
            energy:    ``{"jobs", "price_data", "carbon_data", **solve_kwargs}``
            admission: ``{"sla_class", "window", "config"?}``
            placement: ``{"request", "locations", "load_profiles"?, ...}``
            capacity:  ``{"raw", "warp"?, "config"?, "mode"?}``
            serving:   ``{"requests", "servers", "summarize", ...}``  (advisory)
            genai:     ``{"ticks", "cold", "tick_hours"?}``  (multi-model sizing)

        Returns:
            :class:`FleetOptimizationResult` with each surface's decision +
            ``surfaces_used`` + honest ``notes``.
        """
        used: list[str] = []
        notes_list = list(notes)
        result = FleetOptimizationResult(workload_class=workload_class)

        if energy is not None:
            e = dict(energy)
            jobs = e.pop("jobs")
            price_data = e.pop("price_data")
            carbon_data = e.pop("carbon_data")
            result.energy = self.schedule_energy(jobs, price_data, carbon_data, **e)
            used.append("energy")

        if admission is not None:
            result.admission = self.admit(**admission)
            used.append("admission")

        if placement is not None:
            p = dict(placement)
            request = p.pop("request")
            locations = p.pop("locations")
            result.placement = self.place(request, locations, **p)
            used.append("placement")

        if capacity is not None:
            cap = dict(capacity)
            # `mode` is a convenience that is carried on the ReplicaScalingConfig,
            # not a policy kwarg. An explicit `config` always takes precedence.
            mode = cap.pop("mode", None)
            if "config" not in cap:
                from .policies.replica_scaling import ReplicaScalingConfig

                if mode is not None:
                    cap["config"] = ReplicaScalingConfig(mode=mode)
                else:
                    cap["config"] = ReplicaScalingConfig(mode="forecasted_mcs")
                    notes_list.append(
                        "capacity: defaulted to deployable 'forecasted_mcs' mode "
                        "(forecasts arrivals + service from data <= t-1; no oracle). "
                        "Pass config=/mode= to select another (oracle modes are "
                        "research-only — research/MCS_AUDIT.md)."
                    )
            raw = cap.pop("raw")
            result.capacity = self.scale_replicas(raw, **cap)
            used.append("replica_scaling")

        if serving is not None:
            s = dict(serving)
            requests = s.pop("requests")
            servers = s.pop("servers")
            result.serving_order = self.order_serving_queue(requests, servers, **s)
            used.append("serving_queue")
            notes_list.append(
                "serving_order: request-ordering discipline is a discrete-event "
                "simulator result (advisory, docs/RESULTS.md §8) — not a live-"
                "runtime guarantee."
            )

        if genai is not None:
            g = dict(genai)
            ticks = g.pop("ticks")
            cold = g.pop("cold")
            result.genai = self.serve_genai(ticks, cold, **g)
            used.append("genai_serving")

        result.surfaces_used = tuple(used)
        result.notes = tuple(notes_list)
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def policy(self) -> OptimizationPolicy:
        """The active decision-layer policy object (for :meth:`optimize`)."""
        return self._policy

    @property
    def scheduler(self) -> Optional[JobScheduler]:
        """The wrapped ``JobScheduler`` (energy surface), else ``None``."""
        return getattr(self._policy, "scheduler", None)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"AureliusOptimizer(policy={self.policy_name!r})"


__all__ = ["AureliusOptimizer", "FleetOptimizationResult", "CANONICAL_OBJECTIVE", "SchedulerResult"]
