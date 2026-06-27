"""Canonical action-surface schema for the MPC economic controller.

Aurelius is *not* limited to three control surfaces — but it must only **optimize** actions
that actually exist and that the simulator can score. This module defines:

- the canonical set of infrastructure action surfaces (``ACTION_SPECS``), each tagged with an
  honest **status** from the Phase-1 audit (``research/AURELIUS_ACTION_SURFACE_AUDIT.md``);
- a typed, immutable ``ActionBundle`` that can represent every current and future surface,
  whose **default values are the no-op / reward-path values** so an unspecified surface never
  changes the simulation;
- helpers to extract the subset of levers the serving simulator actually executes
  (``connected_kwargs`` → ``run_unified_replay(capacity=, ordering=, admission=)``).

**Hard rule (enforced by the registry, not just convention):** a surface whose ``status`` is
not ``CONNECTED`` does not affect the scored reward today, so the planner must not optimize
over it. ``SIMULATED_ONLY`` surfaces have a real model but are not yet wired into the reward
path (opt in explicitly); ``PLANNED`` surfaces are never optimized; ``REJECTED`` surfaces are
out of product scope. No fake knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# --- status taxonomy --------------------------------------------------------
CONNECTED = "CONNECTED"                      # implemented AND changes the scored KPI/reward
SIMULATED_ONLY = "SIMULATED_ONLY"            # modelled, but NOT in the reward path yet
PLANNED = "PLANNED"                          # desired, not simulatable today
REQUIRES_PILOT_TELEMETRY = "REQUIRES_PILOT_TELEMETRY"   # needs operator data (ABSENT)
REJECTED = "REJECTED"                        # out of product scope
STATUSES = (CONNECTED, SIMULATED_ONLY, PLANNED, REQUIRES_PILOT_TELEMETRY, REJECTED)


@dataclass(frozen=True)
class ActionSpec:
    """Canonical metadata for one action surface (the audit verdict, as data)."""

    name: str                  # ActionBundle field name (e.g. "capacity_policy")
    surface: str               # human label (e.g. "replica count / capacity")
    status: str                # one of STATUSES
    options: tuple             # allowed values; options[0] is the no-op / reward-path default
    sim_param: str | None      # the run_unified_replay kwarg it maps to (CONNECTED only)
    fidelity: str              # provenance of the model behind it
    limitation: str            # what is missing / why it is not CONNECTED
    roadmap: str = ""          # roadmap id (e.g. "N4"), if any

    @property
    def default(self):
        return self.options[0]

    @property
    def affects_reward(self) -> bool:
        """Does choosing a non-default value change the SCORED reward today?"""
        return self.status == CONNECTED

    @property
    def optimizable(self) -> bool:
        """May the planner optimize over this surface by default? (CONNECTED only.)"""
        return self.status == CONNECTED

    def validate(self, value) -> bool:
        return value in self.options


# --- the canonical action surfaces (encodes the Phase-1 audit verdicts) -----
# Order: CONNECTED first, then SIMULATED_ONLY, then PLANNED. options[0] is always the
# no-op (the value already on the reward path), so a default bundle == today's behaviour.
ACTION_SPECS: dict = {
    # ---- CONNECTED (optimized by default) ----
    "capacity_policy": ActionSpec(
        "capacity_policy", "replica count / capacity adjustment", CONNECTED,
        ("reactive_lag1", "backlog_aware", "forecasted_mcs"), "capacity",
        "CapacityController (unified_replay.py) — Erlang-C + live backlog",
        "", roadmap=""),
    "ordering_policy": ActionSpec(
        "ordering_policy", "ordering / scheduling", CONNECTED,
        ("fifo", "abs_conformal"), "ordering",
        "_dispatch_index (unified_replay.py) — class priority + SRPT", ""),
    "admission_policy": ActionSpec(
        "admission_policy", "admission / defer", CONNECTED,
        ("off", "class_aware"), "admission",
        "AdmissionController (unified_replay.py) — best-effort deferral", ""),
    # ---- SIMULATED_ONLY (opt in with an explicit flag) ----
    "routing_policy": ActionSpec(
        "routing_policy", "routing (request → replica)", SIMULATED_ONLY,
        ("round_robin", "kv_aware"), None,
        "KVAwareRouter (kv_cache.py) exists; dispatch is round-robin _free_sid",
        "router output is not fed into run_unified_replay's reward path", roadmap="N4"),
    "kv_routing_policy": ActionSpec(
        "kv_routing_policy", "KV-aware routing", SIMULATED_ONLY,
        ("off", "prefix_affinity"), None,
        "StatefulKVCache/KVModel (kv_cache.py); applied as a uniform service discount",
        "no per-server cache routing in the serving loop; the ACTION_SPACE kv_routing "
        "knob is currently inert", roadmap="N4"),
    "topology_policy": ActionSpec(
        "topology_policy", "network / topology-aware routing", SIMULATED_ONLY,
        ("off", "net_aware"), None,
        "KVAwareRouter scores a net_penalty; v2026 topology in the fleet plane",
        "no network model in run_unified_replay; net_penalty unused in reward", roadmap="N4"),
    # ---- PLANNED (never optimized; options[0] is the no-op, the rest are the conceivable
    # future values the surface WILL expose once it is simulatable — represented, not actuated)
    "batching_policy": ActionSpec(
        "batching_policy", "batching / batch composition", PLANNED,
        ("static", "roofline_aware"), None, "per-request discrete-event dispatch (no batch model)",
        "needs a roofline throughput/latency/memory batch model", roadmap="N1"),
    "kv_placement_policy": ActionSpec(
        "kv_placement_policy", "KV placement / eviction", PLANNED,
        ("lru", "reuse_aware"), None, "StatefulKVCache LRU is simulated STATE, not an action",
        "needs an eviction/placement lever + counterfactual sim"),
    "prewarm_policy": ActionSpec(
        "prewarm_policy", "prewarming / pre-positioning", PLANNED,
        ("off", "forecast_driven"), None, "no warm-pool state; cold start not modelled",
        "needs warm-pool state + cold-start tax (arrival forecast exists)", roadmap="N7"),
    "clock_policy": ActionSpec(
        "clock_policy", "clock / DVFS / power shaping", PLANNED,
        ("nominal", "low", "high"), None, "service time is fixed (TTFT + tokens·TPOT)",
        "needs a power-vs-performance curve + clock action", roadmap="N2"),
    "precision_policy": ActionSpec(
        "precision_policy", "precision / model routing", PLANNED,
        ("full", "fp8", "int8"), None, "service time is precision-agnostic; no quality model",
        "needs a quality/difficulty proxy + per-precision service/quality model", roadmap="N5"),
    "spec_decode_policy": ActionSpec(
        "spec_decode_policy", "speculative decoding control", PLANNED,
        ("off", "on"), None, "no speculative branch / draft-model overhead",
        "needs a roofline (mem/compute-bound) indicator + draft model", roadmap="N7"),
    "energy_policy": ActionSpec(
        "energy_policy", "energy / price-aware shifting", PLANNED,
        ("off", "defer_to_cheap"), None,
        "price IS in the objective (CostModel), but there is no shifting action",
        "needs a temporal-shift / power-shape action the simulator honours", roadmap="N2"),
    "migration_policy": ActionSpec(
        "migration_policy", "migration", PLANNED,
        ("off", "consolidate"), None, "no migration state/cost/simulator branch",
        "needs a live-move cost model + replica-assignment state"),
    # ---- PLANNED, with fidelity gated on pilot telemetry ----
    "placement_policy": ActionSpec(
        "placement_policy", "placement / packing (job → node/rack)", REQUIRES_PILOT_TELEMETRY,
        ("off", "topology_aware"), None,
        "v2026 topology is anchored marginals; serving servers are homogeneous",
        "needs a topology placement simulator; live residency/health is ABSENT (pilot)"),
}

# Derived views.
CONNECTED_SURFACES = tuple(n for n, s in ACTION_SPECS.items() if s.status == CONNECTED)
SIMULATED_SURFACES = tuple(n for n, s in ACTION_SPECS.items() if s.status == SIMULATED_ONLY)
PLANNED_SURFACES = tuple(n for n, s in ACTION_SPECS.items()
                         if s.status in (PLANNED, REQUIRES_PILOT_TELEMETRY))


@dataclass(frozen=True)
class ActionBundle:
    """A candidate infrastructure action bundle across all surfaces.

    Every field defaults to its surface's **no-op / reward-path value**, so an
    ``ActionBundle()`` reproduces today's behaviour and any unspecified surface is inert.
    Only the three CONNECTED fields reach the simulator (``connected_kwargs``)."""

    capacity_policy: str = "reactive_lag1"
    ordering_policy: str = "fifo"
    admission_policy: str = "off"
    routing_policy: str = "round_robin"
    kv_routing_policy: str = "off"
    topology_policy: str = "off"
    batching_policy: str = "static"
    kv_placement_policy: str = "lru"
    prewarm_policy: str = "off"
    clock_policy: str = "nominal"
    precision_policy: str = "full"
    spec_decode_policy: str = "off"
    energy_policy: str = "off"
    migration_policy: str = "off"
    placement_policy: str = "off"

    def connected_kwargs(self) -> dict:
        """The lever subset the serving simulator actually executes."""
        return {ACTION_SPECS[n].sim_param: getattr(self, n) for n in CONNECTED_SURFACES}

    def legacy_action(self) -> dict:
        """Back-compat dict for the existing controller/replay harness."""
        return {"capacity": self.capacity_policy, "ordering": self.ordering_policy,
                "admission": self.admission_policy}

    @classmethod
    def from_legacy(cls, action: dict) -> "ActionBundle":
        """Build from the legacy ``{capacity, ordering, admission}`` dict."""
        return cls(capacity_policy=action.get("capacity", "reactive_lag1"),
                   ordering_policy=action.get("ordering", "fifo"),
                   admission_policy=action.get("admission", "off"))

    def with_overrides(self, **kw) -> "ActionBundle":
        cur = {f.name: getattr(self, f.name) for f in fields(self)}
        cur.update(kw)
        return ActionBundle(**cur)

    def non_default_surfaces(self) -> dict:
        """Surfaces set away from their no-op default (what this bundle actually changes)."""
        return {n: getattr(self, n) for n in ACTION_SPECS if getattr(self, n) != ACTION_SPECS[n].default}

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def describe(self) -> list:
        """Per-field view: value + status + fidelity + affects_reward + limitation."""
        out = []
        for n, spec in ACTION_SPECS.items():
            out.append({"surface": spec.surface, "field": n, "value": getattr(self, n),
                        "status": spec.status, "affects_reward": spec.affects_reward,
                        "fidelity": spec.fidelity, "limitation": spec.limitation,
                        "roadmap": spec.roadmap})
        return out


__all__ = [
    "CONNECTED", "SIMULATED_ONLY", "PLANNED", "REQUIRES_PILOT_TELEMETRY", "REJECTED", "STATUSES",
    "ActionSpec", "ACTION_SPECS", "CONNECTED_SURFACES", "SIMULATED_SURFACES", "PLANNED_SURFACES",
    "ActionBundle",
]
