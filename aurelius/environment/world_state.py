"""Persistent canonical world state — the stateful infrastructure the serving loop lacked.

PRs #99/#100 connected routing, capacity_multiplier and batching on the *stateless* serving loop
(`run_unified_replay`): each period started from a scalar replica pool and no memory. Prewarming,
topology-aware placement and migration are impossible to score honestly without **persistent
server / rack / replica / warm / migration state** — which is what this module adds.

Design (see ``research/AURELIUS_PERSISTENT_WORLD_STATE_AUDIT.md``):

- a representative **TRACE_DERIVED_SAMPLE** cluster (a few dozen servers across a handful of racks)
  sampled from the v2026 ``server_hourly`` marginals (gpu_type / gpu_count / asw locality). It
  preserves the real DISTRIBUTIONS, not real machine identities — labelled as such, never as
  measured per-server telemetry;
- ``ReplicaState`` (warm/cold, home server) lives ON servers; ``MigrationState`` tracks in-flight
  moves; warm / placement / queue / network / cost states accumulate per period;
- the whole ``CanonicalWorldState`` **persists across periods** and is **cloned per candidate** so
  the MPC search never contaminates the real timeline.

This module holds only STATE + the (deterministic, seeded) sampler. The physics that evolves the
state and turns it into serving economics lives in ``world_simulator.py``.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from .ingestion import v2026_artifacts

# Fidelity tags (kept as strings so they serialize into the manifest verbatim).
TRACE_EXACT = "FULL_TRACE_EXACT"
TRACE_DERIVED_SAMPLE = "TRACE_DERIVED_SAMPLE"
BENCHMARK_DERIVED = "BENCHMARK_DERIVED"
INFERRED = "INFERRED"

# Fallback gpu_type / gpu_count marginals if the v2026 calibration artifact is absent (CI without
# the processed dir). These mirror the committed server_hourly_calibration.json head so the sampled
# cluster is stable either way; the real artifact (when present) overrides them.
_FALLBACK_GPU_TYPE_FRAC = {"A10": 0.2793, "L20": 0.1902, "XPU-B": 0.1416, "XPU-C": 0.1088,
                           "H20": 0.0869, "XPU-E": 0.0869, "XPU-A": 0.0695, "A100": 0.0167,
                           "H100": 0.0201}
_FALLBACK_GPU_COUNT_MEAN = 4.1


@dataclass
class ReplicaState:
    """One serving replica with a home server — warm (ready) or cold (needs a cold start)."""
    replica_id: str
    server_id: str
    rack_id: str
    gpu_type: str
    warm: bool = False
    last_used_period: int = -1              # period index it last served (for warm-hold/idle)
    cold_start_remaining_s: float = 0.0     # seconds of cold start still owed before it can serve
    assigned_capacity: int = 1              # GPU slots this replica occupies
    active: bool = False                    # currently serving this period
    migrating: bool = False                 # capacity withheld this period (mid-move)
    workload_class: str = "latency_critical"


@dataclass
class ServerState:
    """One physical GPU server (sampled from v2026 marginals) hosting replicas."""
    server_id: str
    rack_id: str
    gpu_type: str
    gpu_count: int
    available_gpu_slots: int                # gpu_count - sum(active replica slots)
    active_replicas: list = field(default_factory=list)   # replica_ids serving
    warm_replicas: list = field(default_factory=list)     # replica_ids warm but idle
    mem_pressure: float = 0.0
    net_pressure: float = 0.0
    placement_fidelity: str = TRACE_DERIVED_SAMPLE


@dataclass
class RackState:
    """One rack / ASW group — the topology unit for locality + macro network penalty."""
    rack_id: str
    server_ids: list = field(default_factory=list)
    macro_network_pressure: float = 0.0     # from v2026 network_hourly rx+tx (macro, per-rack)
    gpu_capacity: int = 0
    colocated_replicas: int = 0
    topology_fidelity: str = TRACE_DERIVED_SAMPLE


@dataclass
class MigrationState:
    """An in-flight replica move — non-zero cost + temporary capacity loss, benefit after it lands."""
    migration_id: str
    replica_id: str
    source_server_id: str
    target_server_id: str
    start_period: int
    end_period: int                         # period it completes (capacity restored, benefit on)
    remaining_penalty: float = 0.0          # service penalty still owed while migrating
    migration_cost: float = 0.0             # operator $ for the move (BENCHMARK_DERIVED)
    capacity_loss: int = 0                  # GPU slots withheld while migrating
    cache_invalidation_cost: float = 0.0    # KV warmth lost on the moved replica
    status: str = "in_flight"               # in_flight | completed


@dataclass
class PlacementState:
    """Derived view of where replicas sit — locality + the topology penalty it implies."""
    replica_to_server: dict = field(default_factory=dict)
    server_to_replicas: dict = field(default_factory=dict)
    rack_spread: int = 0                    # number of distinct racks serving load this period
    locality_score: float = 1.0            # 1.0 = fully rack-local; lower = more spread
    topology_penalty: float = 0.0          # service-time surcharge fraction (0 = none)


@dataclass
class WarmState:
    """Warm-pool accounting — the cost ledger that stops prewarming being a free win."""
    warm_replicas: int = 0
    cold_start_events: int = 0
    prewarm_events: int = 0
    warm_hold_gpu_hours: float = 0.0        # GPU-hours burned holding idle replicas warm
    wasted_prewarm_hours: float = 0.0       # warm-hold that no load arrived to use


@dataclass
class QueueState:
    """Pending-work + queue-delay estimates for the period (from the serving replay)."""
    pending_requests: int = 0
    per_replica_queue: dict = field(default_factory=dict)
    queue_delay_p50: float = 0.0
    queue_delay_p95: float = 0.0
    queue_delay_p99: float = 0.0


@dataclass
class NetworkPressureState:
    """Macro network pressure (v2026 network_hourly) → rack/topology penalty. No per-link claims."""
    rx_pressure: float = 0.0
    tx_pressure: float = 0.0
    rack_penalty: float = 0.0
    fidelity: str = TRACE_EXACT


@dataclass
class CostState:
    """Operator cost ledger for the period, including the new stateful terms."""
    gpu_hours: float = 0.0
    warm_hold_cost: float = 0.0
    migration_cost: float = 0.0
    network_penalty_cost: float = 0.0
    energy_cost: float = 0.0
    total_operator_cost: float = 0.0


@dataclass
class CanonicalWorldState:
    """The persistent world the MPC acts on. Cloned per candidate; advanced once per chosen action."""
    period: int = 0
    servers: dict = field(default_factory=dict)         # server_id -> ServerState
    racks: dict = field(default_factory=dict)           # rack_id -> RackState
    replicas: dict = field(default_factory=dict)        # replica_id -> ReplicaState
    migrations: list = field(default_factory=list)      # active MigrationState
    warm_state: WarmState = field(default_factory=WarmState)
    placement_state: PlacementState = field(default_factory=PlacementState)
    queue_state: QueueState = field(default_factory=QueueState)
    network_state: NetworkPressureState = field(default_factory=NetworkPressureState)
    cost_state: CostState = field(default_factory=CostState)
    metrics: dict = field(default_factory=dict)         # per-period accumulator (last simulate())
    fidelity: dict = field(default_factory=dict)        # provenance manifest references

    # -- inventory helpers ---------------------------------------------------
    def warm_count(self) -> int:
        return sum(1 for r in self.replicas.values() if r.warm and not r.migrating)

    def total_replicas(self) -> int:
        return len(self.replicas)

    def active_capacity_slots(self) -> int:
        """GPU slots NOT withheld by an in-flight migration."""
        return sum(r.assigned_capacity for r in self.replicas.values() if not r.migrating)

    def rack_of(self, replica_id: str) -> str:
        r = self.replicas.get(replica_id)
        return r.rack_id if r else ""

    def clone(self) -> "CanonicalWorldState":
        """Deep, independent copy for candidate evaluation — mutating the clone never touches the
        real timeline (the isolation guarantee the MPC search relies on)."""
        return copy.deepcopy(self)

    def summary(self) -> dict:
        return {"period": self.period, "n_servers": len(self.servers), "n_racks": len(self.racks),
                "n_replicas": len(self.replicas), "warm": self.warm_count(),
                "active_migrations": sum(1 for m in self.migrations if m.status == "in_flight"),
                "fidelity": self.fidelity.get("cluster", TRACE_DERIVED_SAMPLE)}


# ---------------------------------------------------------------------------
# TRACE_DERIVED_SAMPLE cluster builder
# ---------------------------------------------------------------------------

def _server_marginals(processed_dir: str | None) -> dict:
    """gpu_type fractions + gpu_count mean from the v2026 server_hourly calibration (or fallback)."""
    srv = v2026_artifacts.load_table("server_hourly", processed_dir or v2026_artifacts.PROCESSED_DIR)
    if not srv:
        return {"gpu_type": dict(_FALLBACK_GPU_TYPE_FRAC), "gpu_count_mean": _FALLBACK_GPU_COUNT_MEAN,
                "label": "fallback"}
    a = srv.get("artifacts", {})
    gt = (a.get("gpu_type") or {}).get("fractions") or dict(_FALLBACK_GPU_TYPE_FRAC)
    gc_mean = (a.get("gpu_count") or {}).get("mean", _FALLBACK_GPU_COUNT_MEAN)
    return {"gpu_type": gt, "gpu_count_mean": gc_mean, "label": srv.get("label", "n/a")}


def _net_marginals(processed_dir: str | None) -> dict:
    net = v2026_artifacts.load_table("network_hourly", processed_dir or v2026_artifacts.PROCESSED_DIR)
    a = (net or {}).get("artifacts", {})
    return {"rx_mean": (a.get("rx_gibps") or {}).get("mean", 0.13),
            "tx_mean": (a.get("tx_gibps") or {}).get("mean", 0.07),
            "label": (net or {}).get("label", "fallback")}


def _weighted_choice(rng: random.Random, frac: dict) -> str:
    items = sorted(frac.items())
    total = sum(v for _k, v in items) or 1.0
    x = rng.random() * total
    acc = 0.0
    for k, v in items:
        acc += v
        if x <= acc:
            return k
    return items[-1][0]


def build_sample_cluster(*, n_servers: int = 24, n_racks: int = 4, gpu_slots_per_replica: int = 1,
                         processed_dir: str | None = None, seed: int = 0,
                         net_ref_gibps: float = 1.0) -> CanonicalWorldState:
    """Build a deterministic TRACE_DERIVED_SAMPLE cluster: ``n_servers`` servers drawn from the
    v2026 gpu_type / gpu_count marginals, partitioned across ``n_racks`` racks, each rack carrying a
    macro network pressure scaled from the v2026 rx+tx means. Replicas are created idle/cold; the
    simulator warms and places them. Seeded → reproducible; preserves the real DISTRIBUTIONS only.

    The default cluster sits well below real fleet scale on purpose — it is a representative sample
    for relative (policy-vs-policy) scoring, not an inventory reconstruction (which the trace does
    not support: ``asw_locality`` is 55% unlabelled). Labelled TRACE_DERIVED_SAMPLE throughout."""
    rng = random.Random(seed)
    sm = _server_marginals(processed_dir)
    nm = _net_marginals(processed_dir)
    gpu_type_frac, gc_mean = sm["gpu_type"], max(1.0, sm["gpu_count_mean"])

    ws = CanonicalWorldState(period=0)
    # racks first, with a macro network pressure spread around the trace mean (rack 0 hottest).
    base_press = min(1.0, (nm["rx_mean"] + nm["tx_mean"]) / max(net_ref_gibps, 1e-9))
    for ri in range(n_racks):
        rid = f"rack{ri}"
        press = min(1.0, base_press * (1.0 + 0.5 * ri))   # deterministic spread, rack0 lightest
        ws.racks[rid] = RackState(rack_id=rid, macro_network_pressure=round(press, 4))
    rack_ids = list(ws.racks)

    for si in range(n_servers):
        rid = rack_ids[si % n_racks]
        gtype = _weighted_choice(rng, gpu_type_frac)
        gcount = max(1, int(round(rng.gauss(gc_mean, 1.5))))
        sid = f"srv{si}"
        ws.servers[sid] = ServerState(
            server_id=sid, rack_id=rid, gpu_type=gtype, gpu_count=gcount,
            available_gpu_slots=gcount, net_pressure=ws.racks[rid].macro_network_pressure)
        ws.racks[rid].server_ids.append(sid)
        ws.racks[rid].gpu_capacity += gcount

    # one cold replica per server slot-budget (sized to host the serving pool); all start cold.
    rep_i = 0
    for sid, srv in ws.servers.items():
        n_rep = max(1, srv.gpu_count // max(1, gpu_slots_per_replica))
        for _ in range(n_rep):
            rid_rack = srv.rack_id
            rep_id = f"rep{rep_i}"
            ws.replicas[rep_id] = ReplicaState(
                replica_id=rep_id, server_id=sid, rack_id=rid_rack, gpu_type=srv.gpu_type,
                warm=False, assigned_capacity=gpu_slots_per_replica)
            rep_i += 1

    ws.network_state = NetworkPressureState(
        rx_pressure=round(min(1.0, nm["rx_mean"] / max(net_ref_gibps, 1e-9)), 4),
        tx_pressure=round(min(1.0, nm["tx_mean"] / max(net_ref_gibps, 1e-9)), 4),
        rack_penalty=0.0, fidelity=TRACE_EXACT if sm["label"] != "fallback" else INFERRED)
    ws.fidelity = {"cluster": TRACE_DERIVED_SAMPLE, "server_marginals": sm["label"],
                   "network_marginals": nm["label"], "n_servers": n_servers, "n_racks": n_racks,
                   "note": "sampled from v2026 marginals; preserves distributions, not identities"}
    return ws


__all__ = [
    "TRACE_EXACT", "TRACE_DERIVED_SAMPLE", "BENCHMARK_DERIVED", "INFERRED",
    "ReplicaState", "ServerState", "RackState", "MigrationState", "PlacementState",
    "WarmState", "QueueState", "NetworkPressureState", "CostState", "CanonicalWorldState",
    "build_sample_cluster",
]
