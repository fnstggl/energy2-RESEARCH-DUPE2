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


def _kv_residency_checks() -> list:
    """The PR #106 per-replica KV/model residency → causal service-time payoff (Gap 2, now LANDED in
    the serving path). Validates the realized transitions, not a bonus."""
    from .kv_cache import StatefulKVCache
    from .world_serving import (
        PREFILL_SAVINGS_FRAC,
        ReplicaResidency,
        RequestSig,
        simulate_residency_serving,
    )

    def rep(cap=512, model="m1"):
        return ReplicaResidency("a", "rack0", model, StatefulKVCache(capacity_blocks=cap, block_tokens=16))
    P = tuple(f"h{i}" for i in range(8))
    out = []
    # 1) a prefix hit lowers service time; a cold request does not.
    r = simulate_residency_serving([rep()], [RequestSig(0.0, 100, 128, "m1", P),
                                             RequestSig(1.0, 100, 128, "m1", P)], policy="kv_aware")
    out.append(Check("kv_residency", "prefix_hit_lowers_service_time",
                     _ok(r.service_factor[1] < r.service_factor[0] and r.service_factor[1] < 0.2),
                     r.service_factor[1], f"cold {r.service_factor[0]:.3f} → hit {r.service_factor[1]:.3f}"))
    # 2) partial hit is strictly between full hit and no hit.
    Q = P[:4] + tuple(f"z{i}" for i in range(4))
    rp = simulate_residency_serving([rep()], [RequestSig(0.0, 100, 128, "m1", P),
                                             RequestSig(1.0, 100, 128, "m1", Q)], policy="kv_aware")
    out.append(Check("kv_residency", "partial_hit_partial_benefit",
                     _ok((1.0 - PREFILL_SAVINGS_FRAC) < rp.service_factor[1] < 1.0), rp.service_factor[1],
                     f"partial factor {rp.service_factor[1]:.3f} in ({1 - PREFILL_SAVINGS_FRAC:.2f},1)"))
    # 3) no future leakage — the first request cannot hit (nothing admitted yet).
    out.append(Check("kv_residency", "no_future_prefix_leakage", _ok(r.service_factor[0] > 0.9),
                     r.service_factor[0], "request 0 sees an empty cache (causal)"))
    # 4) memory pressure → no free hit (a full small cache evicts a returning prefix).
    rep_s = rep(cap=8)
    seq = ([RequestSig(0.0, 100, 128, "m1", P)]
           + [RequestSig(float(i + 1), 100, 128, "m1", tuple(f"q{i}_{b}" for b in range(8)))
              for i in range(5)] + [RequestSig(10.0, 100, 128, "m1", P)])
    rm = simulate_residency_serving([rep_s], seq, policy="kv_aware")
    out.append(Check("kv_residency", "memory_pressure_no_free_hit", _ok(rm.service_factor[-1] > 0.9),
                     rm.evictions, f"evicted prefix returns to a miss ({rm.evictions} evictions)"))
    # 5) the Mooncake-derived bridge produces real reuse (hit rate > 0 on a reusing stream).
    sig = [RequestSig(float(i), 100, 128, "m1", P) for i in range(20)]
    rr = simulate_residency_serving([rep()], sig, policy="kv_aware")
    out.append(Check("kv_residency", "mooncake_reuse_produces_hits",
                     _ok(rr.summary(20)["exact_prefix_hit_rate"] > 0.5),
                     rr.summary(20)["exact_prefix_hit_rate"], "a reusing stream yields exact-prefix hits"))
    # 6) model-affinity: a model switch invalidates KV and adds a cold-start; matching avoids it.
    rs = simulate_residency_serving([rep(model="m1")], [RequestSig(0.0, 100, 128, "m1", P),
                                    RequestSig(1.0, 100, 128, "m2", P)], policy="kv_aware", model_load_s=22.0)
    out.append(Check("kv_residency", "model_switch_adds_cold_start", _ok(rs.model_switch_s[1] == 22.0),
                     rs.model_switch_s[1], "a model mismatch reloads weights (KV invalidated)"))
    return out


def _prefill_decode_checks() -> list:
    """PR #107 prefill/decode model + service-time-sensitive cost modes. Each effect flows through
    prefill work / realized GPU-seconds, never a bonus."""
    from .prefill_decode import compute_phase_serving, effective_gpu_hours
    recs = [(float(i), 100, 1000) for i in range(8)]
    cold = compute_phase_serving(recs, [0] * 8)
    hit = compute_phase_serving(recs, [1000] * 8)             # full prompt cached
    out = []
    out.append(Check("prefill_decode", "prefix_hit_reduces_prefill_only",
                     _ok(hit.prefill_gpu_seconds < cold.prefill_gpu_seconds
                         and hit.decode_gpu_seconds == cold.decode_gpu_seconds),
                     None, "KV reuse cuts prefill; decode (output-token) is unchanged"))
    out.append(Check("prefill_decode", "token_conservation",
                     _ok(cold.prefill_tokens_remaining == cold.prefill_tokens_total
                         and hit.prefill_tokens_remaining == 0), None,
                     "prefill_remaining = prompt − saved; decode tokens unaffected"))
    out.append(Check("prefill_decode", "ttft_falls_with_hit",
                     _ok(hit.summary()["ttft_p95"] < cold.summary()["ttft_p95"]),
                     hit.summary()["ttft_p95"], "a prefix hit lowers TTFT"))
    # decode-bound workload barely monetizes; cost-mode ordering holds; no free cost.
    dheavy_c = compute_phase_serving([(float(i), 2000, 64) for i in range(8)], [0] * 8)
    dheavy_h = compute_phase_serving([(float(i), 2000, 64) for i in range(8)], [64] * 8)
    rel = (dheavy_c.realized_gpu_seconds - dheavy_h.realized_gpu_seconds) / dheavy_c.realized_gpu_seconds
    out.append(Check("prefill_decode", "decode_bound_barely_monetizes",
                     _ok(dheavy_c.summary()["phase_bottleneck"] == "decode_bound" and rel < 0.05),
                     round(rel, 4), "decode-bound work: KV reuse cuts <5% of realized GPU-seconds"))
    prov = effective_gpu_hours("provisioned_capacity", provisioned_gpu_seconds=1000, realized_gpu_seconds=300)
    hyb = effective_gpu_hours("hybrid_capacity_work", provisioned_gpu_seconds=1000, realized_gpu_seconds=300)
    real = effective_gpu_hours("realized_serving_work", provisioned_gpu_seconds=1000, realized_gpu_seconds=300)
    out.append(Check("prefill_decode", "cost_mode_ordering_realized_le_hybrid_le_provisioned",
                     _ok(real <= hyb <= prov), None, f"realized {real:.3f} ≤ hybrid {hyb:.3f} ≤ prov {prov:.3f}"))
    out.append(Check("prefill_decode", "no_free_cost_warm_idle_floor",
                     _ok(real > 0.0 and hyb >= 0.5 * prov), None, "no mode is free; hybrid keeps a warm-idle floor"))
    return out


def _skipped_distribution_checks() -> list:
    """Checks that still require DEFERRED mechanisms — emitted SKIPPED with the reason (never silently
    passing). Gap 2 (per-replica KV routing) is now LANDED (see _kv_residency_checks); what remains:"""
    reasons = [
        ("alibaba_rack_locality_distribution", "cross-rack KV-transfer cost > same-rack",
         "Gap 3 (staged cross-rack KV-transfer model) deferred — macro placement penalty already live"),
        ("roofline_batching_throughput_latency", "tokens/s↑ to saturation then latency↑",
         "Gap 5 (roofline batching model) deferred"),
        ("multi_tier_remote_kv_cache", "local<host<rack<cross-rack hit cost ordering",
         "Phase 8B-B (remote KV tiers) deferred"),
    ]
    return [Check(t, c, SKIPPED, None, why) for (t, c, why) in reasons]


def run_world_validation() -> dict:
    """Run every world-model transition check → a structured PASS/WARN/FAIL/SKIPPED report."""
    checks = (_cold_start_checks() + _conservation_checks() + _isolation_determinism_checks()
              + _migration_realism_checks() + _kv_residency_checks() + _prefill_decode_checks()
              + _skipped_distribution_checks())
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL, SKIPPED)}
    return {"checks": [c.to_dict() for c in checks], "counts": counts,
            "all_landed_pass": counts[FAIL] == 0}


__all__ = ["Check", "run_world_validation", "PASS", "WARN", "FAIL", "SKIPPED"]
