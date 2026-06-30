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

# --- product-boundary taxonomy (Batch-1 corrective) -------------------------
# Aurelius is primarily a GPU FLEET ORCHESTRATOR. It may optimize fleet-level decisions by default, but it
# must NOT silently take control of serving-engine internals unless the serving stack explicitly exposes them.
# Every surface is classified into exactly one product category:
CORE_ORCHESTRATION_DEFAULT = "CORE_ORCHESTRATION_DEFAULT"          # fleet decision, optimized by default
CORE_ORCHESTRATION_AUTO_NOOP = "CORE_ORCHESTRATION_AUTO_NOOP"      # fleet decision, deterministic no-op when N/A
OPTIONAL_SERVING_ENGINE_INTEGRATION = "OPTIONAL_SERVING_ENGINE_INTEGRATION"  # serving-engine internal; default OFF
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"                                # value/surface usable only as a diagnostic
PLANNED_ONLY = "PLANNED_ONLY"                                      # represented, not actuatable yet
PRODUCT_CATEGORIES = (CORE_ORCHESTRATION_DEFAULT, CORE_ORCHESTRATION_AUTO_NOOP,
                      OPTIONAL_SERVING_ENGINE_INTEGRATION, DIAGNOSTIC_ONLY, PLANNED_ONLY)
# Action values that are DIAGNOSTIC_ONLY regardless of their surface's category (no quality model exists for
# them, so a win on these values is never headline-safe).
DIAGNOSTIC_ONLY_VALUES = {
    "precision_policy": {"int4"},
    "kv_cache_precision_policy": {"kv_int4_diagnostic_only"},
}

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
    product_category: str = "" # product-boundary class (set explicitly, else derived in product_category())

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
    # stateful infrastructure actions — CONNECTED via the world_simulator channel (a persistent
    # cluster of server/rack/replica/warm/migration state; see world_state.py + world_simulator.py).
    # Each changes the per-period service factor / warm-capacity ramp / operator cost, so it moves
    # goodput/$ — and each can HURT (over-warm wastes warm-hold; migration costs before it pays),
    # so none can fake a win. Defaults are the no-op (collapse to the stateless PR-#100 path).
    "prewarm_policy": ActionSpec(
        "prewarm_policy", "prewarming / pre-positioning", CONNECTED,
        ("off", "conservative", "aggressive"), None,
        "warm-pool state + cold-start ramp in world_simulator (warm replicas avoid the cold_start_s "
        "serving gap; prewarming pays warm-hold GPU-hours, wasted if the load does not arrive)",
        "BENCHMARK_DERIVED cold-start magnitude (vLLM/TGI model-load regime, order-checked vs the "
        "v2026 ready_delay distribution); live warm state remains SIMULATED until pilot telemetry",
        roadmap="N7", reward_channel="world_simulator"),
    "placement_policy": ActionSpec(
        "placement_policy", "topology-aware placement (replica → rack)", CONNECTED,
        ("topology_blind", "rack_local", "network_aware"), None,
        "world_simulator macro topology service-time discount from rack locality + the v2026 macro "
        "network pressure (network_aware prefers the lowest-pressure racks); topology_blind = no-op",
        "MACRO ONLY — v2026 asw/rack + network_hourly rx/tx marginals; NO per-link / NVLink / "
        "NVSwitch / PFC-ECN / congestion / hardware-health (all ABSENT)",
        roadmap="N4", reward_channel="world_simulator"),
    "migration_policy": ActionSpec(
        "migration_policy", "migration / consolidation", CONNECTED,
        ("off", "conservative", "aggressive"), None,
        "world_simulator live move: operator cost + capacity loss + KV cache invalidation THIS "
        "period, locality benefit only AFTER the move lands next period (persistent MigrationState)",
        "BENCHMARK_DERIVED move cost / duration; real operator migration reasons ABSENT (pilot)",
        roadmap="N6", reward_channel="world_simulator"),
    # ---- CONNECTED roofline actions (PR roofline-economic): each maps to a roofline.ServingConfig and
    # reaches reward through roofline_actions.roofline_action_factors → per-request prefill/decode service
    # times + realized GPU-seconds + power (no bonus). No-op at default → today's behaviour reproduced. ----
    "precision_policy": ActionSpec(
        "precision_policy", "precision / quantization", CONNECTED,
        ("bf16", "fp8", "int4"), None,
        "roofline: lower precision cuts weight+KV bytes → less HBM/bandwidth pressure → higher tokens/s in "
        "the memory-bandwidth-bound regime (faster + cheaper). fp8 ~lossless; int4 carries quality risk",
        "SIMULATOR_INFERENCE magnitude; int4 quality/SLA risk is INFERRED (no quality model) so int4 wins "
        "are labelled unsafe/diagnostic — applied via roofline_actions modulation",
        roadmap="N5", reward_channel="roofline_serving"),
    # KV-cache precision — SEPARATE dtype from the model weights (vLLM ships fp8/int8 KV independent of the
    # weight precision). It sets KV bytes/token → decode-bandwidth roofline (memory-bound regime only) and
    # the KV-budget-in-HBM (active-sequence capacity / eviction). The no-op `inherit_weight_precision`
    # reproduces today's behaviour exactly. fp8/int8 KV ≈ lossless (headline-eligible); int4 KV is
    # diagnostic-only (no quality model → quality-risk channel + excluded from the headline planner).
    "kv_cache_precision_policy": ActionSpec(
        "kv_cache_precision_policy", "KV-cache precision (separate from weights)", CONNECTED,
        ("inherit_weight_precision", "kv_bf16", "kv_fp8", "kv_int8", "kv_int4_diagnostic_only"), None,
        "roofline: KV dtype sets KV bytes/token → higher decode tokens/s in the memory-bandwidth-bound "
        "regime (faster + cheaper) AND more active-sequence capacity / less HBM eviction (kv_precision.py)",
        "SIMULATOR_INFERENCE latency magnitude; fp8/int8 KV ≈ lossless (PUBLIC_BENCHMARK_DERIVED, "
        "headline-eligible); int4 KV quality risk is UNMODELLED → unsafe/diagnostic-only "
        "(excluded from the headline planner, gated by allow_quality_risk)",
        roadmap="PR-1", reward_channel="roofline_serving",
        product_category=OPTIONAL_SERVING_ENGINE_INTEGRATION),  # serving-engine internal → DEFAULT OFF
    "spec_decode_policy": ActionSpec(
        "spec_decode_policy", "speculative decoding depth", CONNECTED,
        ("off", "shallow", "medium", "aggressive"), None,
        "roofline: a draft proposes k tokens / target verifies in one pass — accepted tokens skip serial "
        "decode steps. Helps latency ONLY when decode is memory-bandwidth-bound (spare compute) AND "
        "acceptance is high; extra draft+verify FLOPs raise GPU-seconds → never a cost win; hurts compute-bound",
        "SIMULATOR_INFERENCE acceptance/overhead bands — applied via roofline_actions modulation",
        roadmap="N7", reward_channel="roofline_serving"),
    "clock_policy": ActionSpec(
        "clock_policy", "clock / DVFS / power shaping", CONNECTED,
        ("base", "low", "high"), None,
        "roofline DVFS: compute throughput scales with clock (bandwidth ~flat), power ~clock^2.4 — low "
        "clock saves energy in the memory-bandwidth-bound regime, costs latency/SLA when compute-bound",
        "SIMULATOR_INFERENCE conservative DVFS band; energy effect reported as a diagnostic (not booked as "
        "GPU-hour savings) — applied via roofline_actions modulation",
        roadmap="N2", reward_channel="roofline_serving"),
    # ---- SIMULATED_ONLY (modelled in the same roofline physics, but NOT a live reward action; opt in
    # explicitly. Swept diagnostically; the candidate generator freezes them with a recorded reason). ----
    "colocation_policy": ActionSpec(
        "colocation_policy", "co-location of background work on idle SMs", SIMULATED_ONLY,
        ("off", "conservative", "aggressive"), None,
        "roofline: background compute work uses idle SMs ONLY in the memory-bandwidth-bound decode regime; "
        "it adds memory pressure → a foreground latency penalty (modelled in serving_point)",
        "NO background-work trace exists (Azure is all latency-critical; ReplicaState.workload_class is "
        "unused) → co-location credits NO background goodput and can only HURT foreground SLA here; the "
        "generator prunes it off with this recorded reason", roadmap="N3"),
    # prefill/decode disaggregation — promoted to CONNECTED (Batch-1 Phase 4). The roofline already models
    # the split causally (serving_point disaggregated_static: a wrong split inflates one phase's work + a KV
    # handoff cost); pd_disaggregation.py adds the conservative phase-pool QUEUE approximation (idle
    # GPU-seconds by pool, handoff bytes/latency, allocation efficiency). Reaches reward only through the
    # roofline service-time / GPU-seconds channel. "shared" is the no-op (no disaggregation, no handoff).
    "prefill_decode_policy": ActionSpec(
        "prefill_decode_policy", "prefill/decode disaggregation allocation", CONNECTED,
        ("shared", "p40_d60", "p60_d40"), None,
        "roofline + pd_disaggregation: split capacity into prefill/decode pools (DistServe/Splitwise/Dynamo); "
        "a wrong split queues one phase; KV handoff overhead on disaggregation. Reaches reward via service "
        "time + GPU-seconds (no live persistent phase queues → conservative causal approximation)",
        "SIMULATOR_INFERENCE: the live cluster replay has NO persistent disaggregated capacity pools; the "
        "split + handoff + phase-queue effects are a conservative roofline/queueing approximation labelled "
        "directional until pilot disaggregated-pool telemetry (DistServe-style optional integration)",
        roadmap="N4", reward_channel="roofline_serving",
        product_category=OPTIONAL_SERVING_ENGINE_INTEGRATION),  # serving-engine internal → DEFAULT OFF
    # heterogeneous GPU-type assignment — route work to GPU classes (H100/A100/L40S/…) by their FLOPs /
    # bandwidth / HBM / power / cost. SIMULATED_ONLY: the production benchmark's reward path costs the whole
    # period at ONE dominant GPU type (gpu_type is constant per server, never a per-workload decision), so a
    # heterogeneous-assignment ACTION is NOT_APPLICABLE there → it cannot fake a benefit on a homogeneous-cost
    # fleet. It is a real causal lever in controlled fixtures (gpu_assignment.py: an explicit GPU mix + a
    # per-workload assignment simulator) and turns CONNECTED once the fleet/cost path exposes per-replica
    # GPU-type assignment. "homogeneous_default" is the no-op (everything on the dominant type).
    "gpu_assignment_policy": ActionSpec(
        "gpu_assignment_policy", "heterogeneous GPU-type assignment (workload → GPU class)", SIMULATED_ONLY,
        ("homogeneous_default", "fastest_for_latency_sensitive", "cheapest_for_batch",
         "memory_heavy_to_high_hbm", "balanced_heterogeneous", "diagnostic_oracle_assignment"), None,
        "gpu_assignment.py: per-GPU-type roofline (FLOPs/bandwidth/HBM) + per-type cost (cost_model "
        "OWNED_ECONOMICS / LEASE) route latency-sensitive→fast, batch→cheap, memory-heavy→high-HBM",
        "NOT_APPLICABLE to the production benchmark (single dominant GPU type in the cost path; GPU type is "
        "constant per server) → fixture-only until the fleet/cost path exposes per-replica assignment; "
        "diagnostic_oracle_assignment is NON-deployable (labelled)", roadmap="PR-2",
        product_category=CORE_ORCHESTRATION_AUTO_NOOP),  # core fleet orchestration; deterministic no-op when N/A
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
    "energy_policy": ActionSpec(
        "energy_policy", "energy / price-aware shifting", PLANNED,
        ("off", "defer_to_cheap"), None,
        "price IS in the objective (CostModel), but there is no shifting action",
        "needs a temporal-shift / power-shape action the simulator honours", roadmap="N2"),
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
    placement_policy: str = "topology_blind"
    migration_policy: str = "off"
    precision_policy: str = "bf16"
    kv_cache_precision_policy: str = "inherit_weight_precision"
    spec_decode_policy: str = "off"
    clock_policy: str = "base"
    colocation_policy: str = "off"
    prefill_decode_policy: str = "shared"
    gpu_assignment_policy: str = "homogeneous_default"
    energy_policy: str = "off"

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


def product_category(name: str) -> str:
    """The product-boundary category of a surface. Explicit on a spec when set, else derived from status:
    CONNECTED fleet knobs → CORE_ORCHESTRATION_DEFAULT; SIMULATED_ONLY → OPTIONAL_SERVING_ENGINE_INTEGRATION;
    PLANNED → PLANNED_ONLY. (The three Batch-1 knobs set it explicitly; gpu_assignment is core-auto-noop,
    kv_cache_precision + prefill_decode are optional serving-engine integrations.)"""
    spec = ACTION_SPECS[name]
    if spec.product_category:
        return spec.product_category
    if spec.status in (PLANNED, REQUIRES_PILOT_TELEMETRY):
        return PLANNED_ONLY
    if spec.status == SIMULATED_ONLY:
        return OPTIONAL_SERVING_ENGINE_INTEGRATION
    return CORE_ORCHESTRATION_DEFAULT


def value_is_diagnostic_only(name: str, value) -> bool:
    """True if a specific surface VALUE is diagnostic-only (no quality model → never headline-safe), e.g.
    ``precision_policy=int4`` or ``kv_cache_precision_policy=kv_int4_diagnostic_only``."""
    return value in DIAGNOSTIC_ONLY_VALUES.get(name, set())


# Optional serving-engine integrations are DEFAULT-OFF: the planner must not vary them unless the operator
# explicitly opts in (a serving stack that exposes the capability). Core orchestration knobs are not in here.
def optional_serving_engine_surfaces() -> tuple:
    """CONNECTED/SIMULATED surfaces classified as OPTIONAL_SERVING_ENGINE_INTEGRATION (default-off)."""
    return tuple(n for n in ACTION_SPECS
                 if product_category(n) == OPTIONAL_SERVING_ENGINE_INTEGRATION
                 and ACTION_SPECS[n].status in (CONNECTED, SIMULATED_ONLY))


def core_orchestration_surfaces() -> tuple:
    """Surfaces that are core fleet orchestration (default-on / auto-noop)."""
    return tuple(n for n in ACTION_SPECS
                 if product_category(n) in (CORE_ORCHESTRATION_DEFAULT, CORE_ORCHESTRATION_AUTO_NOOP))


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
    "CORE_ORCHESTRATION_DEFAULT", "CORE_ORCHESTRATION_AUTO_NOOP", "OPTIONAL_SERVING_ENGINE_INTEGRATION",
    "DIAGNOSTIC_ONLY", "PLANNED_ONLY", "PRODUCT_CATEGORIES", "DIAGNOSTIC_ONLY_VALUES",
    "product_category", "value_is_diagnostic_only", "optional_serving_engine_surfaces",
    "core_orchestration_surfaces",
]
