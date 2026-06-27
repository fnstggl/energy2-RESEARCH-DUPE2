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

# batching_policy → (batch_concurrency, batch_service_factor) for run_unified_replay.
# More concurrency packs more requests per replica (throughput ↑, queue ↓, same GPU-hours);
# the service factor inflates each batched request's latency (shared compute → SLA risk ↑).
# INFERRED magnitudes (public prior, not trace-calibrated); options[0] == today's behaviour.
BATCHING_MODELS = {
    "conservative": (1.0, 1.0),       # one request per replica — today's no-batching behaviour
    "balanced": (2.0, 1.15),
    "aggressive": (4.0, 1.5),
}


@dataclass(frozen=True)
class ActionSpec:
    """Canonical metadata for one action surface (the audit verdict, as data)."""

    name: str                  # ActionBundle field name (e.g. "capacity_policy")
    surface: str               # human label (e.g. "replica count / capacity")
    status: str                # one of STATUSES
    options: tuple             # allowed values; options[0] is the no-op / reward-path default
    sim_param: str | None      # the run_unified_replay kwarg it maps to (else None)
    fidelity: str              # provenance of the model behind it
    limitation: str            # what is missing / why it is not CONNECTED
    roadmap: str = ""          # roadmap id (e.g. "N4"), if any
    reward_channel: str = ""   # HOW a CONNECTED surface reaches the reward
    #                            ("run_unified_replay" kwarg, or "kv_service_factor", ...)

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
        "", roadmap="", reward_channel="run_unified_replay"),
    "ordering_policy": ActionSpec(
        "ordering_policy", "ordering / scheduling", CONNECTED,
        ("fifo", "abs_conformal"), "ordering",
        "_dispatch_index (unified_replay.py) — class priority + SRPT", "",
        reward_channel="run_unified_replay"),
    "admission_policy": ActionSpec(
        "admission_policy", "admission / defer", CONNECTED,
        ("off", "class_aware"), "admission",
        "AdmissionController (unified_replay.py) — best-effort deferral", "",
        reward_channel="run_unified_replay"),
    # routing — CONNECTED via the fleet-KV channel: a routing policy is replayed over the
    # real Mooncake prefix trace (fleet_kv_routing), and its fleet prefix-reuse depth sets
    # the serving service-time discount → it changes goodput/$ (kv_service_factor channel).
    "routing_policy": ActionSpec(
        "routing_policy", "routing (request → replica) / KV-aware routing", CONNECTED,
        ("round_robin", "shortest_queue", "kv_aware"), None,
        "fleet_kv_routing (kv_cache.py) over the Mooncake trace → routing-specific "
        "service factor (kv_aware reuses ~50% more prefix depth than round_robin)", "",
        roadmap="N4", reward_channel="kv_service_factor"),
    "capacity_multiplier": ActionSpec(
        "capacity_multiplier", "explicit replica count / capacity level", CONNECTED,
        (1.0, 0.75, 1.5), "capacity_multiplier",
        "scales the CapacityController-sized replica count in run_unified_replay",
        "", reward_channel="run_unified_replay"),
    "batching_policy": ActionSpec(
        "batching_policy", "batching / batch composition", CONNECTED,
        ("conservative", "balanced", "aggressive"), None,
        "per-replica continuous-batching concurrency + service inflation (run_unified_replay)",
        "INFERRED magnitudes (public prior; not trace-calibrated) — sanity-banded",
        roadmap="N1", reward_channel="run_unified_replay"),
    # ---- SIMULATED_ONLY (opt in with an explicit flag) ----
    "kv_routing_policy": ActionSpec(
        "kv_routing_policy", "per-request KV prefix routing (finer than routing_policy)",
        SIMULATED_ONLY, ("off", "prefix_affinity"), None,
        "StatefulKVCache exists; the FLEET effect is CONNECTED via routing_policy",
        "per-Azure-request prefix routing needs per-request prefix ids (Azure trace has "
        "none) — fleet-level KV routing is already CONNECTED via routing_policy", roadmap="N4"),
    "topology_policy": ActionSpec(
        "topology_policy", "network / topology-aware routing", SIMULATED_ONLY,
        ("off", "net_aware"), None,
        "KVAwareRouter scores a net_penalty; v2026 topology in the fleet plane",
        "no network model in run_unified_replay; net_penalty unused in reward", roadmap="N4"),
    # ---- PLANNED (never optimized; options[0] is the no-op, the rest are the conceivable
    # future values the surface WILL expose once it is simulatable — represented, not actuated)
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
    capacity_multiplier: float = 1.0
    batching_policy: str = "conservative"
    kv_routing_policy: str = "off"
    topology_policy: str = "off"
    kv_placement_policy: str = "lru"
    prewarm_policy: str = "off"
    clock_policy: str = "nominal"
    precision_policy: str = "full"
    spec_decode_policy: str = "off"
    energy_policy: str = "off"
    migration_policy: str = "off"
    placement_policy: str = "off"

    def connected_kwargs(self) -> dict:
        """The lever subset passed DIRECTLY to run_unified_replay (capacity/ordering/admission).
        CONNECTED surfaces that act through another channel (routing → kv_service_factor) are
        excluded here and applied by the controller via that channel (see reward_channel)."""
        return {ACTION_SPECS[n].sim_param: getattr(self, n)
                for n in CONNECTED_SURFACES if ACTION_SPECS[n].sim_param}

    def replay_kwargs(self) -> dict:
        """ALL run_unified_replay action kwargs this bundle implies: capacity/ordering/admission
        + capacity_multiplier + the batching translation. (Routing acts via the kv_service_factor
        channel applied to the jobs, not here.)"""
        conc, svc = BATCHING_MODELS.get(self.batching_policy, (1.0, 1.0))
        return {"capacity": self.capacity_policy, "ordering": self.ordering_policy,
                "admission": self.admission_policy, "capacity_multiplier": float(self.capacity_multiplier),
                "batch_concurrency": conc, "batch_service_factor": svc}

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


def replay_kwargs_from_action(action: dict) -> dict:
    """run_unified_replay action kwargs from a legacy/baseline action dict — any unset connected
    surface defaults to its no-op (capacity_multiplier 1.0, batching conservative)."""
    conc, svc = BATCHING_MODELS.get(action.get("batching_policy", "conservative"), (1.0, 1.0))
    return {"capacity": action.get("capacity", "reactive_lag1"),
            "ordering": action.get("ordering", "fifo"), "admission": action.get("admission", "off"),
            "capacity_multiplier": float(action.get("capacity_multiplier", 1.0)),
            "batch_concurrency": conc, "batch_service_factor": svc}


__all__ = [
    "CONNECTED", "SIMULATED_ONLY", "PLANNED", "REQUIRES_PILOT_TELEMETRY", "REJECTED", "STATUSES",
    "BATCHING_MODELS", "ActionSpec", "ACTION_SPECS", "CONNECTED_SURFACES", "SIMULATED_SURFACES",
    "PLANNED_SURFACES", "ActionBundle", "replay_kwargs_from_action",
]
