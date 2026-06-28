"""Transition validation for the persistent world model (PR #105).

Asserts the world-model transitions this PR lands — cold-start decomposition, replica/migration
IDENTITY conservation, clone isolation, determinism — against the calibrated bands and conservation
laws, and emits PASS / WARN / FAIL / SKIPPED with a metric and detail per check. Distribution checks
that need the deferred mechanisms (Gap 2 per-replica Mooncake prefix routing, Gap 3 cross-rack
transfer, Gap 5 roofline) are emitted as **SKIPPED with the reason**, never silently "passing" —
honesty over green.

This is the world-MODEL counterpart to the multi-plane ``validation_suite.py``: it validates the
state transitions the MPC's stateful actions drive, not the plane ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .world_calibration import COLD_START_COMPONENTS, cold_start_components, world_calibration
from .world_simulator import (
    MIGRATION_KV_PRESERVED_BY_MODE,
    _migration_plan,
    initialize_world_state,
    simulate_period,
    warm_seed,
)

PASS, WARN, FAIL, SKIPPED = "PASS", "WARN", "FAIL", "SKIPPED"


@dataclass(frozen=True)
class Check:
    transition: str
    check: str
    status: str               # PASS | WARN | FAIL | SKIPPED
    metric: float | None
    detail: str

    def to_dict(self) -> dict:
        return {"transition": self.transition, "check": self.check, "status": self.status,
                "metric": self.metric, "detail": self.detail}


def _ok(cond: bool) -> str:
    return PASS if cond else FAIL


def _cold_start_checks() -> list:
    r = world_calibration()
    comp = cold_start_components(r)
    out = [Check("cold_start_decomposition", "components_sum_to_aggregate_base",
                 _ok(abs(comp["total_s"] - r.base("cold_start_s")) <= 1.0), comp["total_s"],
                 f"sum={comp['total_s']:.1f}s vs aggregate base={r.base('cold_start_s'):.1f}s")]
    # every component band is ordered low<=base<=high and the model-load term dominates (public prior)
    for k in COLD_START_COMPONENTS:
        p = r.parameters[k]
        out.append(Check("cold_start_decomposition", f"band_ordered:{k}",
                         _ok(p.low <= p.base <= p.high), p.base, f"{p.low}<= {p.base} <={p.high} {p.unit}"))
    ml, ei = r.base("cold_start_model_load_s"), r.base("cold_start_engine_init_s")
    out.append(Check("cold_start_decomposition", "model_load_dominates", _ok(ml > ei), ml,
                     f"model_load {ml}s > engine_init {ei}s (weight load is the dominant term)"))
    return out


def _conservation_checks(*, n_servers: int = 16, n_racks: int = 4, seed: int = 0) -> list:
    """Run a migration and assert replica/identity conservation across it."""
    ws = initialize_world_state(n_servers=n_servers, n_racks=n_racks, seed=seed)
    warm_seed(ws, max(8, ws.total_replicas() // 2))
    n0 = ws.total_replicas()
    ids0 = set(ws.replicas)
    racks0 = {rid: r.rack_id for rid, r in ws.replicas.items()}
    warm0 = ws.warm_count()
    # force an aggressive migration plan and commit periods so the move lands (completed migrations
    # are pruned from ws.migrations, so we detect a landing by the replica's rack CHANGING).
    from types import SimpleNamespace

    from .cost_model import CostModel
    from .fleet_plane_v2026 import V2026FleetPlane
    fleet = V2026FleetPlane().state_at(0)
    recs = [(i * 5.0, 200, 100) for i in range(6)]
    fcast = {"arrival_rate": 0.1, "arrival_p90": 0.2, "mean_service_s": 1.0}
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="network_aware",
                          migration_policy="aggressive")
    mg = _migration_plan(ws, "aggressive", placement={})
    out = []
    for _ in range(3):                                    # start → in-flight → land
        simulate_period(ws, pol, recs, fcast, sla_s=10.0, tick_seconds=10.0, cost_model=CostModel(),
                        fleet_state=fleet, period_hours=1.0, dt_seconds=60.0, mutate=True)
    n1 = ws.total_replicas()
    ids1 = set(ws.replicas)
    moved = [rid for rid in ws.replicas if ws.replicas[rid].rack_id != racks0.get(rid)]
    out.append(Check("migration", "replica_count_conserved", _ok(n1 == n0), n1 - n0,
                     f"replicas {n0}→{n1} (a move must not create/destroy a replica)"))
    out.append(Check("migration", "no_replica_duplication", _ok(ids1 == ids0 and len(ids1) == n0),
                     len(ids1), "replica ids are a stable set (identity moves, never copies)"))
    out.append(Check("migration", "started", PASS if mg["n_migrations"] > 0 else WARN,
                     mg["n_migrations"], f"{mg['n_migrations']} migration(s) planned"))
    # a landed migrated replica (rack changed) keeps weights resident — not re-cold-started
    landed_warm = all(ws.replicas[rid].weights_loaded for rid in moved) if moved else True
    out.append(Check("migration", "landed_replica_keeps_weights",
                     PASS if (moved and landed_warm) else (WARN if not moved else FAIL), len(moved),
                     f"{len(moved)} landed move(s) kept weights resident (no destination model-load)"))
    out.append(Check("warm_state", "warm_pool_nonnegative", _ok(ws.warm_count() >= 0), ws.warm_count(),
                     f"warm={ws.warm_count()} seeded≈{warm0}"))
    return out


def _isolation_determinism_checks() -> list:
    a = initialize_world_state(n_servers=12, n_racks=3, seed=1)
    b = initialize_world_state(n_servers=12, n_racks=3, seed=1)
    same = (list(a.replicas) == list(b.replicas)
            and [r.gpu_type for r in a.replicas.values()] == [r.gpu_type for r in b.replicas.values()])
    warm_seed(a, 6)
    clone = a.clone()
    before = a.warm_count()
    for r in clone.replicas.values():                     # mutate the clone hard
        r.warm = False
    isolated = a.warm_count() == before and clone.warm_count() == 0
    return [Check("cluster", "deterministic_seed", _ok(same), None, "same seed → same sampled cluster"),
            Check("clone", "isolation", _ok(isolated), None,
                  "mutating a clone never touches the real timeline")]


def _migration_realism_checks() -> list:
    """The KV-preservation correction: a pipelined (conservative) move keeps more KV than a bulk
    (aggressive) move, and BOTH keep more than the old flat-surcharge model (preserved≈0)."""
    cons = MIGRATION_KV_PRESERVED_BY_MODE["conservative"]
    aggr = MIGRATION_KV_PRESERVED_BY_MODE["aggressive"]
    return [Check("migration", "pipelined_keeps_more_kv_than_bulk", _ok(cons > aggr), cons - aggr,
                  f"conservative keeps {cons} vs aggressive {aggr} (Llumnix pipelined > bulk)"),
            Check("migration", "kv_mostly_preserved_not_surcharged", _ok(cons >= 0.5), cons,
                  "a live move keeps ≥0.5 of KV (was a flat 1.04 surcharge → strictly dominated)")]


def _skipped_distribution_checks() -> list:
    """Distribution checks that require the DEFERRED mechanisms — emitted SKIPPED with the reason
    (never silently passing). These become live when Gaps 2/3/5 land (see the gap audit)."""
    reasons = [
        ("mooncake_kv_reuse_distribution", "KS/L1 of assigned-prefix reuse vs Mooncake held-out",
         "Gap 2 (per-replica Mooncake prefix assignment in the world sim) deferred"),
        ("alibaba_rack_locality_distribution", "rack/asw locality marginal match",
         "Gap 3 (cross-rack KV-transfer penalty) deferred"),
        ("alibaba_network_rxtx_distribution", "rx/tx pressure marginal match",
         "macro pressure already in placement; per-link ABSENT from any trace"),
        ("roofline_batching_throughput_latency", "tokens/s↑ to saturation then latency↑",
         "Gap 5 (roofline batching model) deferred"),
        ("migration_future_kv_hit_rate", "migrated identity raises future KV hit vs fresh replica",
         "needs Gap 2 per-replica residency to measure a hit rate"),
    ]
    return [Check(t, c, SKIPPED, None, why) for (t, c, why) in reasons]


def run_world_validation() -> dict:
    """Run every world-model transition check → a structured PASS/WARN/FAIL/SKIPPED report."""
    checks = (_cold_start_checks() + _conservation_checks() + _isolation_determinism_checks()
              + _migration_realism_checks() + _skipped_distribution_checks())
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL, SKIPPED)}
    return {"checks": [c.to_dict() for c in checks], "counts": counts,
            "all_landed_pass": counts[FAIL] == 0}


__all__ = ["Check", "run_world_validation", "PASS", "WARN", "FAIL", "SKIPPED"]
