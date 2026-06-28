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
                     _ok(dheavy_c.summary()["phase_bottleneck"] == "decode_phase_bound" and rel < 0.05),
                     round(rel, 4), "decode-bound work: KV reuse cuts <5% of realized GPU-seconds"))
    prov = effective_gpu_hours("provisioned_capacity", provisioned_gpu_seconds=1000, realized_gpu_seconds=300)
    hyb = effective_gpu_hours("hybrid_capacity_work", provisioned_gpu_seconds=1000, realized_gpu_seconds=300)
    real = effective_gpu_hours("realized_serving_work", provisioned_gpu_seconds=1000, realized_gpu_seconds=300)
    out.append(Check("prefill_decode", "cost_mode_ordering_realized_le_hybrid_le_provisioned",
                     _ok(real <= hyb <= prov), None, f"realized {real:.3f} ≤ hybrid {hyb:.3f} ≤ prov {prov:.3f}"))
    out.append(Check("prefill_decode", "no_free_cost_warm_idle_floor",
                     _ok(real > 0.0 and hyb >= 0.5 * prov), None, "no mode is free; hybrid keeps a warm-idle floor"))
    return out


def _roofline_checks() -> list:
    """PR #109 roofline serving model: compute-bound vs memory-bandwidth-bound is roofline-DERIVED (from
    arithmetic intensity vs the ridge point), distinct from decode-PHASE-bound; every mechanism is fully
    simulated and produces a sensitivity curve. No reward bonus."""
    from .roofline import (
        ServingConfig,
        Workload,
        all_sensitivity_curves,
        arithmetic_intensity,
        roofline_regime,
        serving_point,
    )
    out = []
    # 1) low-batch long-context decode is memory-bandwidth-bound (AI < ridge) — numerically.
    wl = Workload(decode_tokens=512, context_len=4096)
    rr = roofline_regime("decode", ServingConfig(batch_size=1), wl)
    out.append(Check("roofline", "low_batch_decode_is_memory_bandwidth_bound",
                     _ok(rr["roofline_regime"] == "memory_bandwidth_bound" and rr["arithmetic_intensity"] < rr["ridge_point"]),
                     rr["arithmetic_intensity"], f"AI {rr['arithmetic_intensity']} < ridge {rr['ridge_point']}"))
    # 2) batching raises arithmetic intensity (amortises weight bytes).
    lo = arithmetic_intensity("decode", ServingConfig(batch_size=1), wl)
    hi = arithmetic_intensity("decode", ServingConfig(batch_size=64), wl)
    out.append(Check("roofline", "batching_raises_arithmetic_intensity", _ok(hi > lo), round(hi / lo, 2),
                     f"AI {lo:.2f}→{hi:.2f} as batch 1→64"))
    # 3) decode-PHASE-bound is NOT the same label as memory-bandwidth-bound (the conflation fix).
    p = serving_point(Workload(prompt_tokens=128, decode_tokens=2000, context_len=2048), ServingConfig(batch_size=4))
    out.append(Check("roofline", "phase_bound_distinct_from_roofline_regime",
                     _ok(p["phase_bottleneck"] == "decode_phase_bound" and p["decode_regime"] == "memory_bandwidth_bound"),
                     None, "a decode-phase-bound load is still memory-bandwidth-bound by roofline"))
    # 4) spec decode hurts in compute-bound (extra FLOPs), helps in memory-bound high-accept.
    cb = ServingConfig(batch_size=128, gpu="H20")
    cbw = Workload(decode_tokens=256, context_len=64)
    spec_cb = serving_point(cbw, ServingConfig(batch_size=128, gpu="H20", spec_decode_accept=0.5))
    out.append(Check("roofline", "spec_decode_hurts_compute_bound",
                     _ok(roofline_regime("decode", cb, cbw)["roofline_regime"] == "compute_bound" and spec_cb["spec_speedup"] < 1.0),
                     spec_cb["spec_speedup"], "compute-bound spec decode slows wall-clock (FLOP-limited)"))
    # 5) every mechanism is fully simulated and yields a sensitivity curve with help/hurt/neutral.
    curves = all_sensitivity_curves(Workload(prompt_tokens=1024, decode_tokens=256, context_len=2048))
    full = all(len(c["curve"]) >= 3 and "completion_s" in c["help_hurt_neutral"] for c in curves.values())
    out.append(Check("roofline", "all_mechanisms_fully_simulated_with_curves", _ok(full and len(curves) == 6),
                     len(curves), "batching/alloc/spec/clock/precision/coloc each fully simulated + swept"))
    # 6) precision/spec/clock are now LIVE mpc actions (this PR, via roofline_actions); co-location stays
    # a fully-simulated diagnostic sweep (frozen off — no background-work trace).
    out.append(Check("roofline", "live_surfaces_match_connected_actions",
                     _ok(curves["batching"]["action_surface"] == "live_mpc_action"
                         and curves["precision"]["action_surface"] == "live_mpc_action"
                         and curves["co_location"]["action_surface"] == "diagnostic_sweep_only"),
                     None, "precision/spec/clock live via roofline_actions; co-location frozen off"))
    return out


def _roofline_action_checks() -> list:
    """Roofline-economic MPC actions: precision/spec/clock are LIVE (reward via roofline_serving);
    co-location + prefill/decode are SIMULATED and frozen off. Each helps/hurts in the correct regime
    through serving_point physics; neutral defaults reproduce; int4 carries a quality risk; co-location
    credits no background goodput; the Pareto gate still blocks SLA-shedding. No direct reward bonus."""
    from types import SimpleNamespace

    from .actions import ACTION_SPECS, ActionBundle
    from .roofline import Workload
    from .roofline_actions import roofline_action_factors
    from .search_planner import FROZEN_OFF, AdaptiveSearchPlanner, roofline_pruned_options
    from .training import claim_gate
    out = []
    mem = Workload(prompt_tokens=512, decode_tokens=256, context_len=2048)
    comp = Workload(prompt_tokens=512, decode_tokens=64, context_len=512)
    # 1) neutral defaults reproduce — every factor exactly 1.0, no quality risk.
    f0 = roofline_action_factors(ActionBundle(), mem, gpu="H100", batch_size=8)
    out.append(Check("roofline_action", "neutral_defaults_reproduce",
                     _ok(all(abs(f0[k] - 1.0) < 1e-9 for k in ("prefill_factor", "decode_factor",
                         "gpu_seconds_factor", "power_factor")) and f0["quality_sla_risk"] == 0.0),
                     None, "all factors 1.0 at default policies → live path bit-for-bit unchanged"))
    # 2) precision fp8 helps memory-bound (faster decode AND cheaper).
    f8 = roofline_action_factors(ActionBundle(precision_policy="fp8"), mem, gpu="H100", batch_size=8)
    out.append(Check("roofline_action", "precision_helps_memory_bound",
                     _ok(f8["decode_factor"] < 1.0 and f8["gpu_seconds_factor"] < 1.0),
                     round(f8["decode_factor"], 3), "fp8 fewer bytes → faster + cheaper decode"))
    # 3) spec helps latency but pays a compute tax (GPU-seconds fall LESS than wall-clock).
    fs = roofline_action_factors(ActionBundle(spec_decode_policy="medium"), mem, gpu="H100", batch_size=8)
    out.append(Check("roofline_action", "spec_latency_win_pays_compute_tax",
                     _ok(fs["decode_factor"] < 1.0 and fs["gpu_seconds_factor"] > fs["decode_factor"]),
                     round(fs["gpu_seconds_factor"], 3), "draft+verify FLOPs → spec is a latency lever, not a free cost win"))
    # 4) spec hurts compute-bound (FLOP-limited).
    fsc = roofline_action_factors(ActionBundle(spec_decode_policy="aggressive"), comp, gpu="H20", batch_size=128)
    out.append(Check("roofline_action", "spec_hurts_compute_bound", _ok(fsc["decode_factor"] >= 1.0 - 1e-9),
                     round(fsc["decode_factor"], 3), "compute-bound: extra FLOPs cannot speed a FLOP-limited decode"))
    # 5) clock changes power (DVFS) without faking memory-bandwidth throughput.
    fl = roofline_action_factors(ActionBundle(clock_policy="low"), mem, gpu="H100", batch_size=8)
    fh = roofline_action_factors(ActionBundle(clock_policy="high"), mem, gpu="H100", batch_size=8)
    out.append(Check("roofline_action", "clock_changes_power_not_memory_bw",
                     _ok(fl["power_factor"] < 1.0 < fh["power_factor"] and fl["decode_factor"] <= 1.0 + 1e-9),
                     None, "low clock saves power; memory-bandwidth-bound decode throughput is unmoved by clock"))
    # 6) co-location credits NO background goodput without a trace (no imaginary work).
    fco = roofline_action_factors(ActionBundle(colocation_policy="aggressive"), mem, gpu="H100",
                                  batch_size=8, background_work=False)
    out.append(Check("roofline_action", "no_imaginary_background_goodput",
                     _ok(fco["coloc_useful_gpu_seconds"] == 0.0 and fco["interference_factor"] >= 1.0),
                     None, "no background-work trace → co-location only adds interference, credits nothing"))
    # 7) int4 carries a quality/SLA risk (no free precision).
    fi = roofline_action_factors(ActionBundle(precision_policy="int4"), mem, gpu="H100", batch_size=8)
    out.append(Check("roofline_action", "int4_carries_quality_risk", _ok(fi["quality_sla_risk"] > 0.0),
                     fi["quality_sla_risk"], "int4 quality-failure fraction counts as SLA failures"))
    # 8) status split: precision/spec/clock CONNECTED via roofline_serving; co-location/pd frozen w/ reasons.
    conn = all(ACTION_SPECS[s].status == "CONNECTED" and ACTION_SPECS[s].reward_channel == "roofline_serving"
               for s in ("precision_policy", "spec_decode_policy", "clock_policy"))
    frozen = set(FROZEN_OFF) == {"colocation_policy", "prefill_decode_policy"} and all(FROZEN_OFF[s][1] for s in FROZEN_OFF)
    out.append(Check("roofline_action", "connected_live_simulated_frozen_split", _ok(conn and frozen),
                     None, "precision/spec/clock live; co-location + prefill/decode frozen with recorded reasons"))
    # 9) no direct reward bonus: affects_reward ⇔ CONNECTED (every effect flows through physics).
    out.append(Check("roofline_action", "no_direct_reward_bonus",
                     _ok(all((ACTION_SPECS[s].status == "CONNECTED") == ACTION_SPECS[s].affects_reward for s in ACTION_SPECS)),
                     None, "no surface adds a scalar reward bonus"))
    # 10) candidate pruning is regime-aware (int4/spec proposed only where they can help).
    mp = roofline_pruned_options(decode_regime="memory_bandwidth_bound")
    cp = roofline_pruned_options(decode_regime="compute_bound")
    out.append(Check("roofline_action", "candidate_pruning_is_regime_aware",
                     _ok("int4" in mp["precision_policy"] and "int4" not in cp["precision_policy"]
                         and cp["spec_decode_policy"] == ("off",)),
                     None, "int4/spec are search candidates only in the regime that can use them"))
    # 11) the adaptive planner MEASURES search regret vs exhaustive (never a silent cap).
    _b, plan = AdaptiveSearchPlanner(exhaustive_max=10).plan(
        lambda b: 100.0 + (5 if b.precision_policy == "fp8" else 0),
        surfaces={"precision_policy": ("bf16", "fp8", "int4"), "clock_policy": ("base", "low", "high")},
        frozen_reasons={}, regret_audit=True)
    out.append(Check("roofline_action", "search_regret_is_measured",
                     _ok(plan.regret_audited and plan.estimated_regret is not None and plan.estimated_regret <= 1e-9),
                     plan.estimated_regret, f"{plan.strategy}: regret vs exhaustive reported, not hidden"))
    # 12) the Pareto gate still blocks SLA-shedding (a cheaper-but-less-safe arm is not a headline).
    g = claim_gate({"mpc_controller": SimpleNamespace(goodput_per_dollar=150.0, sla_violation_rate=0.05),
                    "fair": SimpleNamespace(goodput_per_dollar=140.0, sla_violation_rate=0.02)})
    out.append(Check("roofline_action", "pareto_gate_blocks_sla_shedding",
                     _ok(g["beats_fair_baseline"] and not g["pareto_sla_not_worse"] and not g["headline_claim_allowed"]),
                     None, "higher gp/$ with worse SLA (e.g. int4) → headline blocked"))
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
              + _migration_realism_checks() + _kv_residency_checks() + _prefill_decode_checks() + _roofline_checks()
              + _roofline_action_checks() + _skipped_distribution_checks())
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL, SKIPPED)}
    return {"checks": [c.to_dict() for c in checks], "counts": counts,
            "all_landed_pass": counts[FAIL] == 0}


__all__ = ["Check", "run_world_validation", "PASS", "WARN", "FAIL", "SKIPPED"]
