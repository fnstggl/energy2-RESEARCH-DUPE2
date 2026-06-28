"""World-state-backed simulation path for the stateful infrastructure actions.

This is the physics that turns :mod:`world_state` into serving economics. It layers three
stateful effects **on top of** the proven stateless engine (`run_unified_replay`) — never
replacing it — so that with every stateful knob at its no-op the result is identical to the
PR-#100 path:

- **prewarm** → a warm-capacity ramp: cold replicas can't serve for ``COLD_START_S`` seconds, so a
  period that needs more replicas than are warm eats a warm-up queue spike. Prewarming raises the
  warm pool ahead of the period (paying warm-hold GPU-hours) to avoid it. Forecast-driven; causal.
- **placement / topology** → a macro service-time DISCOUNT for exploiting rack locality / low
  network pressure (exactly the channel KV-aware routing uses). ``topology_blind`` = no discount
  (the PR-#100 baseline); ``network_aware`` exploits the v2026 macro rx/tx pressure spread.
- **migration** → physically moving warm replicas onto better racks: a real cost + capacity loss +
  cache invalidation THIS period, a locality discount only AFTER the move lands next period.

All three flow through (a) a per-period service-time factor, (b) the warm-capacity ramp, and (c)
extra operator-cost terms — then the world state is advanced ONCE for the chosen action. Candidate
evaluation always runs on a CLONE (`clone_world_state_for_candidate`) so the MPC search can never
contaminate the real timeline.

Fidelity (see ``research/AURELIUS_PERSISTENT_WORLD_STATE_AUDIT.md``): the cluster is a
TRACE_DERIVED_SAMPLE of the v2026 marginals; the cold-start / migration magnitudes below are
BENCHMARK_DERIVED public-prior constants (order-checked against the trace's ready_delay regime),
NOT measured serving telemetry. The topology penalty is MACRO ONLY (no per-link / NVLink claims).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..optimizer.unified_replay import (
    CLASS_BEST_EFFORT,
    CLASS_LATENCY,
    Job,
    run_unified_replay,
)
from .world_calibration import world_calibration as _wc
from .world_state import CanonicalWorldState, MigrationState, build_sample_cluster

# All transition magnitudes are CALIBRATED with public-source provenance in world_calibration.py.
# (PR #101 used a full-period warm-hold here; that inverted the capacity economics — see
# research/WORLD_STATE_REGRESSION_ROOT_CAUSE_AUDIT.md. The idle-timeout warm-hold is the fix.)
_CAL = _wc()
COLD_START_S = _CAL.base("cold_start_s")                       # 30s — evidence-supported, not tuned
WARM_IDLE_TIMEOUT_S = _CAL.base("warm_idle_timeout_s")         # 300s — idle replicas cool after this
WARM_HOLD_GPU_FRACTION = _CAL.base("warm_hold_gpu_fraction")   # 1.0 — a warm replica occupies its GPU
COLD_START_RAMP = _CAL.base("cold_start_ramp")                 # 1.0 — progressive (staggered) ramp
TOPOLOGY_MAX_DISCOUNT = _CAL.base("topology_max_discount")     # 0.08 macro locality relief (no per-link)
MIGRATION_COST_PER_REPLICA = _CAL.base("migration_cost_per_replica")    # $0.40 per live move
MIGRATION_CAPACITY_LOSS_FRAC = _CAL.base("migration_capacity_loss_frac")  # 0.10 withheld while moving
MIGRATION_CACHE_PENALTY = _CAL.base("migration_cache_penalty")  # 0.04 KV-warmth-lost surcharge (bulk move)
MIGRATION_KV_PRESERVED_FRAC = _CAL.base("migration_kv_preserved_frac")  # 0.90 KV kept by a pipelined move
MIGRATION_DURATION_PERIODS = int(_CAL.base("migration_duration_periods"))  # lands next period
# share of KV warmth a move KEEPS, by mode: conservative ≈ pipelined (Llumnix), aggressive ≈ bulk.
MIGRATION_KV_PRESERVED_BY_MODE = {"conservative": MIGRATION_KV_PRESERVED_FRAC,
                                  "aggressive": max(0.0, MIGRATION_KV_PRESERVED_FRAC - 0.3)}
PREWARM_MARGIN = {"off": 0.0, "conservative": 0.15, "aggressive": 0.45}  # headroom over forecast

PREWARM_OPTIONS = ("off", "conservative", "aggressive")
PLACEMENT_OPTIONS = ("topology_blind", "rack_local", "network_aware")
MIGRATION_OPTIONS = ("off", "conservative", "aggressive")


@dataclass
class PeriodOutcome:
    """Result of simulating one period through the world state."""
    kpi: object                              # UnifiedKPI from the serving replay
    operator_cost: float                     # total $ incl. warm-hold + migration + base
    warm_hold_cost: float = 0.0
    migration_cost: float = 0.0
    service_factor: float = 1.0              # combined stateful service-time multiplier applied
    warm_capacity: int = 0
    cold_start_s: float = 0.0
    topology_factor: float = 1.0
    locality_score: float = 1.0
    rack_spread: int = 0
    cold_start_events: int = 0
    wasted_prewarm_hours: float = 0.0
    migrations_started: int = 0
    queue_delay_p95: float = 0.0
    queue_delay_p99: float = 0.0
    kv_diag: dict | None = None               # per-replica KV/model residency diagnostics (PR #106)
    roofline_diag: dict | None = None         # roofline action modulation (precision/spec/clock) diagnostics
    quality_sla_risk: float = 0.0             # precision quality-failure fraction (int4) → counts as SLA fails
    power_w: float = 0.0                       # mean GPU power under the clock/DVFS action (diagnostic)
    energy_j: float = 0.0                      # serving energy under the action (diagnostic)
    electricity_price_per_kwh: float = 0.0     # $/kWh applied to this period's energy cost (real or constant)
    metrics: dict = field(default_factory=dict)

    @property
    def goodput_per_dollar(self) -> float:
        # a precision quality failure is NOT sla-safe goodput (a wrong answer is worse than a slow one);
        # quality_sla_risk is 0 for bf16/fp8, conservative for int4. No bonus — only a penalty channel.
        return (self.kpi.sla_safe_goodput * (1.0 - self.quality_sla_risk)) / max(self.operator_cost, 1e-9)

    @property
    def sla_violation_rate(self) -> float:
        return (self.kpi.sla_violations + self.quality_sla_risk * self.kpi.n_total) / max(self.kpi.n_total, 1)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------

def initialize_world_state(*, n_servers: int = 24, n_racks: int = 4, processed_dir: str | None = None,
                           seed: int = 0, net_ref_gibps: float = 1.0) -> CanonicalWorldState:
    """Build the persistent TRACE_DERIVED_SAMPLE cluster the MPC will act on."""
    return build_sample_cluster(n_servers=n_servers, n_racks=n_racks, processed_dir=processed_dir,
                                seed=seed, net_ref_gibps=net_ref_gibps)


def clone_world_state_for_candidate(ws: CanonicalWorldState) -> CanonicalWorldState:
    """Independent deep copy — mutating it never touches the real timeline."""
    return ws.clone()


def warm_seed(ws: CanonicalWorldState, n_warm: int) -> None:
    """Mark the first ``n_warm`` replicas warm (a steady-state cluster that has been serving). Used
    at init so the reactive ``off`` policy starts from a realistic warm pool, not a cold boot — the
    condition under which ``off`` reproduces the stateless path on stable load."""
    n = max(0, min(n_warm, ws.total_replicas()))
    for i, rid in enumerate(ws.replicas):
        r = ws.replicas[rid]
        r.warm = i < n
        if i < n:
            r.last_used_period = ws.period
            r.weights_loaded = True            # a warm replica has its weights resident…
            r.kv_warm_frac = 1.0               # …and a hot cache (it has been serving)
    ws.warm_state.warm_replicas = ws.warm_count()


def reset_world_state(ws: CanonicalWorldState) -> None:
    """Return all replicas to cold/idle and clear migrations (fresh simulation)."""
    for r in ws.replicas.values():
        r.warm = False
        r.active = False
        r.migrating = False
        r.last_used_period = -1
        r.cold_start_remaining_s = 0.0
        r.weights_loaded = False
        r.kv_warm_frac = 0.0
        r.warm_until_period = -1
    ws.migrations = []
    ws.period = 0


# ---------------------------------------------------------------------------
# per-policy plans (pure: read state + bundle, return effect — no mutation)
# ---------------------------------------------------------------------------

def _forecast_capacity(forecast: dict, *, peak: bool = False) -> int:
    """Replicas the forecast implies for the period: offered load (Erlangs) + a safety unit.

    offered_load = arrival_rate · mean_service_s. ``peak`` uses the p90 arrival to size for spikes.
    Causal — uses only the forecast handed in (no future truth)."""
    ar = forecast.get("arrival_p90" if peak else "arrival_rate", forecast.get("arrival_rate", 0.0))
    svc = max(forecast.get("mean_service_s", 1.0), 1e-6)
    offered = ar * svc
    return max(1, int(math.ceil(offered + 1.0)))


def _racks_by_pressure(ws: CanonicalWorldState) -> list:
    """Rack ids sorted low→high macro network pressure (the placement preference order)."""
    return sorted(ws.racks, key=lambda r: ws.racks[r].macro_network_pressure)


def _prewarm_plan(ws: CanonicalWorldState, policy: str, forecast: dict) -> dict:
    """Decide the warm pool for THIS period. Returns warm_capacity + warm-hold accounting.

    - off: REACTIVE — keep exactly the warm pool carried from last period (what was used). On
      stable/declining load this already covers the need → no cold start → parity with the
      stateless path. Only an up-RAMP beyond the carried pool eats a cold start (the honest, real
      cost of reactive scaling, and exactly what prewarming exists to avoid).
    - conservative: prewarm up to the forecast mean (+15% headroom).
    - aggressive: prewarm up to the forecast p90 peak (+45% headroom).
    Prewarmed-but-idle replicas cost warm-hold GPU-hours; if load under-uses them that is
    wasted_prewarm (the ledger that stops prewarming being a free win)."""
    total = ws.total_replicas()
    reactive_warm = ws.warm_count()                    # pool carried from prior periods
    if policy == "off":
        target = max(1, reactive_warm)
    else:
        peak = policy == "aggressive"
        c_fc = _forecast_capacity(forecast, peak=peak)
        target = int(math.ceil(c_fc * (1.0 + PREWARM_MARGIN.get(policy, 0.0))))
        target = max(reactive_warm, target)            # never cool below what is already warm
    target = max(1, min(total, target))
    prewarm_events = max(0, target - reactive_warm)
    return {"warm_capacity": target, "reactive_warm": reactive_warm,
            "prewarm_events": prewarm_events, "cold_start_s": COLD_START_S}


def _placement_plan(ws: CanonicalWorldState, policy: str, *, c_used: int) -> dict:
    """Topology service-time factor (≤ 1.0) from WHERE THE WARM REPLICAS ACTUALLY SIT.

    The serving replicas are drawn from the warm pool, each on its home rack — you can only place on
    the racks your warm replicas occupy. ``network_aware`` activates the ``c_used`` warm replicas on
    the LOWEST-pressure home racks; the discount scales with the relief vs the warm pool's average
    rack pressure. ``rack_local`` rewards locality (fewest racks). ``topology_blind`` = 1.0 (baseline).

    This is the channel that makes MIGRATION worthwhile: if every warm replica sits on a high-pressure
    rack, network_aware finds no relief (no free lunch) — only physically MOVING replicas to a
    low-pressure rack (migration) creates low-pressure home racks for future placement to exploit.
    Macro only — no per-link claims."""
    if not ws.racks:
        return {"topology_factor": 1.0, "locality_score": 1.0, "rack_spread": 1, "used_racks": []}
    # warm replicas per home rack (what we can actually serve from this period)
    warm_by_rack: dict = {}
    for r in ws.replicas.values():
        if r.warm and not r.migrating:
            warm_by_rack[r.rack_id] = warm_by_rack.get(r.rack_id, 0) + 1
    if not warm_by_rack:                                # nothing warm yet → fall back to rack capacity
        warm_by_rack = {rid: max(1, rk.gpu_capacity) for rid, rk in ws.racks.items()}
    total_warm = sum(warm_by_rack.values())
    avg_press = (sum(ws.racks[rid].macro_network_pressure * n for rid, n in warm_by_rack.items())
                 / max(1, total_warm))                  # pressure topology_blind effectively serves at

    if policy == "topology_blind":
        return {"topology_factor": 1.0, "locality_score": 1.0,
                "rack_spread": len(warm_by_rack), "used_racks": list(warm_by_rack)}
    # activate c_used warm replicas from the lowest-pressure home racks first.
    order = sorted(warm_by_rack, key=lambda rid: ws.racks[rid].macro_network_pressure)
    used, acc, w_press = [], 0, 0.0
    for rid in order:
        take = min(warm_by_rack[rid], max(1, c_used) - acc)
        used.append(rid)
        w_press += ws.racks[rid].macro_network_pressure * take
        acc += take
        if acc >= max(1, c_used):
            break
    used_press = w_press / max(1, acc)
    relief = max(0.0, avg_press - used_press)           # pressure avoided by serving the best warm racks
    if policy == "network_aware":
        factor = 1.0 - TOPOLOGY_MAX_DISCOUNT * min(1.0, relief / max(avg_press, 1e-6))
    else:  # rack_local: locality (fewest racks) only, ignoring which racks are hot
        locality_bonus = (len(warm_by_rack) - len(used)) / max(1, len(warm_by_rack))
        factor = 1.0 - TOPOLOGY_MAX_DISCOUNT * 0.6 * locality_bonus
    factor = min(1.0, max(1.0 - TOPOLOGY_MAX_DISCOUNT, factor))
    return {"topology_factor": round(factor, 5),
            "locality_score": round(1.0 - (len(used) - 1) / max(1, len(warm_by_rack) - 1), 4)
            if len(warm_by_rack) > 1 else 1.0,
            "rack_spread": len(used), "used_racks": used}


def _migration_plan(ws: CanonicalWorldState, policy: str, *, placement: dict) -> dict:
    """Decide migrations to START this period. A move pays cost + capacity loss + cache penalty NOW
    and only yields its locality benefit AFTER it lands (next period), so it is never a free win.

    Migrations consolidate warm replicas that sit on HIGHER-pressure racks onto the lowest-pressure
    rack. ``conservative`` moves few, ``aggressive`` moves more. ``off`` moves none (no-op)."""
    if policy == "off" or not ws.racks:
        return {"n_migrations": 0, "capacity_loss_frac": 0.0, "migration_cost": 0.0,
                "cache_factor": 1.0, "targets": []}
    order = _racks_by_pressure(ws)
    best_rack = order[0]
    # warm replicas NOT already on the best rack are migration candidates.
    candidates = [r for r in ws.replicas.values()
                  if r.warm and not r.migrating and r.rack_id != best_rack]
    if not candidates:
        return {"n_migrations": 0, "capacity_loss_frac": 0.0, "migration_cost": 0.0,
                "cache_factor": 1.0, "targets": []}
    frac = 0.25 if policy == "conservative" else 0.6
    n = max(1, int(round(len(candidates) * frac)))
    targets = sorted(candidates, key=lambda r: -ws.racks[r.rack_id].macro_network_pressure)[:n]
    cap_loss = min(0.5, MIGRATION_CAPACITY_LOSS_FRAC * n / max(1, ws.warm_count()))
    # KV warmth KEPT across the move (Llumnix pipelined → conservative keeps ~0.9; bulk/aggressive
    # less). The service surcharge is the LOST fraction only — replacing the old flat 1.04 that made
    # migration strictly dominated (the moved replica preserves most of its cache, it is not recooled).
    preserved = MIGRATION_KV_PRESERVED_BY_MODE.get(policy, MIGRATION_KV_PRESERVED_FRAC)
    cache_factor = 1.0 + MIGRATION_CACHE_PENALTY * (1.0 - preserved) * (1.0 if n else 0.0)
    return {"n_migrations": n, "capacity_loss_frac": round(cap_loss, 4),
            "migration_cost": round(MIGRATION_COST_PER_REPLICA * n, 4),
            "cache_factor": round(cache_factor, 5), "kv_preserved_frac": preserved,
            "targets": [r.replica_id for r in targets], "target_rack": best_rack}


# ---------------------------------------------------------------------------
# the period simulation
# ---------------------------------------------------------------------------

def simulate_period(ws: CanonicalWorldState, bundle, recs: list, forecast: dict, *,
                    sla_s: float, tick_seconds: float, base_service_factor: float = 1.0,
                    replay_kwargs: dict | None = None, cost_model=None, fleet_state=None,
                    cost_scenario: str = "owned", best_effort_fraction: float = 0.0,
                    period_hours: float = 1.0, dt_seconds: float | None = None,
                    kv_state: dict | None = None, energy_price_per_kwh: float | None = None,
                    mutate: bool = False) -> PeriodOutcome:
    """Simulate one period under ``bundle`` on the world state.

    ``recs`` are the period's raw ``(arrival_s, tokens, ctx)`` request records. ``forecast`` carries
    at least ``arrival_rate`` / ``arrival_p90`` / ``mean_service_s`` (causal). Builds jobs with the
    combined stateful service factor, runs the serving replay under the warm-capacity ramp + any
    migration capacity loss, prices the result incl. warm-hold + migration, and (if ``mutate``)
    advances the real world state. Returns a :class:`PeriodOutcome`."""
    from ..benchmarks.srtf_serving_backtest import _service_time_s

    prewarm = str(getattr(bundle, "prewarm_policy", "off"))
    placement = str(getattr(bundle, "placement_policy", "topology_blind"))
    migration = str(getattr(bundle, "migration_policy", "off"))
    rkw = dict(replay_kwargs or {})
    # Note: migrations LAND (their benefit turns on) only on the real mutate path (_advance) — a
    # mutate=False scoring call is a PURE READ of the current state (no writes), so candidate
    # evaluation never contaminates the real timeline whether or not it is given a clone.

    # 1) prewarm plan → warm capacity + warm-hold accounting.
    pw = _prewarm_plan(ws, prewarm, forecast)
    warm_capacity = pw["warm_capacity"]

    # 2) capacity the period will actually drive (forecast-sized), for placement consolidation.
    c_used = _forecast_capacity(forecast, peak=False)

    # 3) placement → topology service factor. Placement activates the SERVING-sized subset of warm
    # replicas (not the whole pool), so network_aware can be selective about low-pressure home racks.
    pl = _placement_plan(ws, placement, c_used=max(1, c_used))

    # 4) migration → capacity loss + cost + cache penalty THIS period.
    mg = _migration_plan(ws, migration, placement=pl)

    # 5) combine factors and run the serving replay.
    service_factor = base_service_factor * pl["topology_factor"] * mg["cache_factor"]
    cap_mult = float(rkw.get("capacity_multiplier", 1.0)) * (1.0 - mg["capacity_loss_frac"])
    be_stride = max(1, round(1.0 / best_effort_fraction)) if best_effort_fraction > 0 else 0
    recs_sorted = sorted(recs, key=lambda r: r[0])
    t0 = recs_sorted[0][0] if recs_sorted else 0.0
    # roofline action modulation (precision / speculative-decoding / clock / ...). None when every
    # roofline action is at its no-op default → the live path below is bit-for-bit unchanged. Built from
    # the period's representative workload + the fleet's dominant GPU; reaches reward ONLY through service
    # time + realized GPU-seconds + power (no bonus). rl_svc is the latency factor for the non-phase path.
    from .roofline_actions import period_action_modulation
    _gpu_t = (max(fleet_state.gpu_type_mix, key=fleet_state.gpu_type_mix.get)
              if (fleet_state is not None and getattr(fleet_state, "gpu_type_mix", None)) else "A100")
    rl_mod = period_action_modulation(bundle, recs_sorted, gpu=_gpu_t)
    rl_svc = float(rl_mod["completion_factor"]) if rl_mod else 1.0
    # 5b) PER-REPLICA KV / model residency (PR #106) → per-request service factor. When kv_state is
    # given, routing over the warm replicas' PERSISTENT caches replaces the offline KV scalar AND the
    # macro topology scalar (the residency sim applies per-replica topology): a prefix hit skips that
    # request's prefill, a model match avoids a switch cold-start. Causal in request order; mutates the
    # replicas' caches (the persistence prewarm/migration/placement act through). See world_serving.py.
    res_diag = None
    per_req_factor = per_req_cold = None
    if kv_state and recs_sorted:
        from .world_serving import (
            build_request_signatures,
            replica_residency_view,
            simulate_residency_serving,
        )
        rview = replica_residency_view(ws, capacity_blocks=int(kv_state.get("capacity_blocks", 512)),
                                       commit=mutate)
        sigs = build_request_signatures(recs_sorted, kv_state.get("hash_seq") or [],
                                        model_ids=tuple(kv_state.get("model_ids", ("llama-8b-gqa",))),
                                        model_seq=kv_state.get("model_seq"))
        rr = simulate_residency_serving(
            rview, sigs, policy=str(rkw.get("routing_policy", kv_state.get("routing", "kv_aware"))),
            model_load_s=_CAL.base("cold_start_model_load_s"),
            topology_max_discount=TOPOLOGY_MAX_DISCOUNT)
        per_req_factor = rr.service_factor
        per_req_cold = rr.model_switch_s
        res_diag = rr.summary(len(sigs))

    # 5c) PREFILL/DECODE phase model (PR #107). When kv_state carries a cost_mode, a KV hit reduces
    # PREFILL ONLY (prompt-token-driven), decode stays output-token-driven — fixing the PR #106 bug
    # where the residency factor scaled the whole (decode-dominated) service. realized_gpu_seconds then
    # drives the cost mode. Default off → the PR #106 residency-factor path above is unchanged.
    phase = None
    cost_mode = (kv_state or {}).get("cost_mode")
    if cost_mode and per_req_factor is not None:
        from .prefill_decode import compute_phase_serving
        phase = compute_phase_serving(
            recs_sorted, rr.saved_tokens, model_cold_s=rr.model_switch_s,
            batching=str(getattr(bundle, "batching_policy", "balanced")),
            period_seconds=max(period_hours * 3600.0, 1.0), roofline_factors=rl_mod)
        if res_diag is not None:
            res_diag = {**res_diag, **phase.summary()}

    def _svc(i, r):
        if phase is not None and i < len(phase.service_s):
            return phase.service_s[i] * mg["cache_factor"]   # phase service (roofline factors applied inside)
        base = _service_time_s(int(r[1]))
        if per_req_factor is not None and i < len(per_req_factor):
            # residency factor REPLACES base_service_factor + topology (subsumed); migration KV penalty
            # (mg.cache_factor) still applies; model-switch cold-start adds seconds. The roofline action
            # latency factor (rl_svc; 1.0 at neutral) scales the served time when no phase model is active.
            return base * per_req_factor[i] * mg["cache_factor"] * rl_svc + (per_req_cold[i] if per_req_cold else 0.0)
        return base * service_factor * rl_svc

    jobs = [Job(idx=i, arrival_s=(r[0] - t0), actual_tokens=int(r[1]), predicted_tokens=float(r[1]),
                service_s=_svc(i, r),
                cls=(CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY))
            for i, r in enumerate(recs_sorted)]
    warmup_c = max(1, min(int(getattr(fleet_state, "capacity_envelope", 4) or 4), 4)) if fleet_state else 4
    # the warm ramp only matters when the warm pool is smaller than what the period drives; when it
    # covers the load (e.g. a warm-seeded cluster on stable load with prewarm=off) pass None so the
    # result is bit-for-bit the stateless path.
    ramp_capacity = warm_capacity if warm_capacity < max(c_used, warmup_c + 1) else None
    kpi = run_unified_replay(
        jobs, tick_seconds=tick_seconds, sla_s=sla_s,
        capacity=rkw.get("capacity", "reactive_lag1"), ordering=rkw.get("ordering", "fifo"),
        admission=rkw.get("admission", "off"), warmup_c=warmup_c,
        capacity_multiplier=cap_mult, batch_concurrency=float(rkw.get("batch_concurrency", 1.0)),
        batch_service_factor=float(rkw.get("batch_service_factor", 1.0)),
        warm_capacity=ramp_capacity, cold_start_s=(pw["cold_start_s"] if ramp_capacity is not None else 0.0))

    # queue-delay p95 (jobs carry start_s set by the replay) — the world path's latency signal.
    waits = sorted(max(0.0, j.start_s - j.arrival_s) for j in jobs if j.start_s >= 0)
    q_p95 = waits[min(len(waits) - 1, int(len(waits) * 0.95))] if waits else 0.0
    q_p99 = waits[min(len(waits) - 1, int(len(waits) * 0.99))] if waits else 0.0

    # 7) cost = serving operator cost + warm-hold + migration.
    peak_c = kpi.c_max
    # warm-hold belongs to the PREWARM decision, not the capacity decision. A reactive (off) operator
    # cools idle replicas down to what it serves (peak_c) within the idle timeout, so it carries no
    # INTENTIONAL warm-hold — only PROACTIVE prewarming (conservative/aggressive) holds replicas above
    # current usage and pays for them. Charging reactive idle against capacity_multiplier (via
    # peak_c) is exactly what inverted the capacity economics in PR #101 (see root-cause audit).
    prewarmed_idle = max(0, warm_capacity - peak_c) if prewarm != "off" else 0
    warm_hold_hours = min(period_hours, WARM_IDLE_TIMEOUT_S / 3600.0)   # idle replicas cool after timeout
    warm_hold_gpu_hours = prewarmed_idle * WARM_HOLD_GPU_FRACTION * warm_hold_hours
    warm_hold_cost = warm_hold_gpu_hours * _gpu_hour_usd()
    cold_started = max(0, peak_c - pw["reactive_warm"]) if prewarm == "off" else max(0, peak_c - warm_capacity)
    # COST MODE (PR #107): the provisioned-capacity GPU-hours are the period's billing baseline. When a
    # cost_mode + phase model is active, faster serving (KV-saved prefill → fewer realized GPU-seconds)
    # can reduce billable GPU-hours under realized_serving_work (upper bound) / hybrid (bounded, with a
    # warm-idle floor). provisioned_capacity reproduces the existing (PR #106) behaviour exactly.
    billable_gpu_hours = kpi.gpu_hours
    if phase is not None and cost_mode and cost_mode != "provisioned_capacity":
        from .prefill_decode import effective_gpu_hours
        billable_gpu_hours = effective_gpu_hours(
            cost_mode, provisioned_gpu_seconds=kpi.gpu_hours * 3600.0,
            realized_gpu_seconds=phase.realized_gpu_seconds)
    if cost_model is not None and fleet_state is not None:
        gpu_type = (max(fleet_state.gpu_type_mix, key=fleet_state.gpu_type_mix.get)
                    if getattr(fleet_state, "gpu_type_mix", None) else "H100")
        # per-period electricity price (real diurnal price when supplied) overrides the constant fleet
        # scalar; None reproduces the constant-price behaviour exactly. Cost still flows only through
        # energy = gpu_hours · power_kw · power_scale · pue · price.
        _price = energy_price_per_kwh if energy_price_per_kwh is not None else fleet_state.energy_price_per_kwh
        cb = cost_model.operator_cost(
            gpu_hours=billable_gpu_hours, gpu_type=gpu_type,
            energy_price_per_kwh=_price, utilization=fleet_state.util_target,
            scenario=cost_scenario, sla_violations=kpi.sla_violations,
            power_scale=(float(rl_mod["power_factor"]) if rl_mod else 1.0))   # clock/DVFS energy only (no GPU-hour fake)
        base_cost = cb.total_operator_cost
    else:
        base_cost = kpi.cost_usd
    # migration is a one-time operator $ for the move, amortised over the period it benefits (× the
    # window's period_hours) so it is COMMENSURATE with the window-scaled serving cost — otherwise a
    # fixed $ swamps a bounded scoring window and migration could never be worthwhile at any horizon.
    migration_cost = mg["migration_cost"] * max(period_hours, 1e-6)
    total_cost = base_cost + warm_hold_cost + migration_cost

    rl_power_w = float(rl_mod["power_w"]) if rl_mod else 0.0
    rl_gpu_s = phase.realized_gpu_seconds if phase is not None else billable_gpu_hours * 3600.0

    outcome = PeriodOutcome(
        kpi=kpi, operator_cost=total_cost, warm_hold_cost=round(warm_hold_cost, 5),
        migration_cost=mg["migration_cost"], service_factor=round(service_factor, 5),
        warm_capacity=warm_capacity, cold_start_s=pw["cold_start_s"],
        topology_factor=pl["topology_factor"], locality_score=pl["locality_score"],
        rack_spread=pl["rack_spread"], cold_start_events=cold_started,
        wasted_prewarm_hours=round(warm_hold_gpu_hours, 5), migrations_started=mg["n_migrations"],
        queue_delay_p95=round(q_p95, 4), queue_delay_p99=round(q_p99, 4), kv_diag=res_diag,
        roofline_diag=rl_mod, quality_sla_risk=(float(rl_mod["quality_sla_risk"]) if rl_mod else 0.0),
        power_w=round(rl_power_w, 1), energy_j=round(rl_power_w * rl_gpu_s, 1),
        electricity_price_per_kwh=round(
            energy_price_per_kwh if energy_price_per_kwh is not None
            else (fleet_state.energy_price_per_kwh if fleet_state is not None else 0.0), 6),
        metrics={"prewarm_policy": prewarm, "placement_policy": placement,
                 "migration_policy": migration, "warm_capacity": warm_capacity,
                 "peak_c": peak_c, "topology_factor": pl["topology_factor"],
                 "service_factor": round(service_factor, 5)})

    if mutate:
        _advance(ws, peak_c=peak_c, warm_capacity=warm_capacity, prewarm_events=pw["prewarm_events"],
                 cold_started=cold_started, warm_hold_gpu_hours=warm_hold_gpu_hours, mg=mg,
                 prewarm=prewarm, dt_seconds=(dt_seconds if dt_seconds is not None else 3600.0))
        # persist the electricity/power ledgers (DVFS energy economics) on the real timeline
        if getattr(ws, "power_state", None) is not None:
            ws.power_state.accumulate(power_w=rl_power_w, energy_j=outcome.energy_j,
                                      price_per_kwh=outcome.electricity_price_per_kwh,
                                      clock_state=getattr(bundle, "clock_policy", "base"))
        if getattr(ws, "electricity_state", None) is not None:
            ws.electricity_state.current_price = outcome.electricity_price_per_kwh
    return outcome


def _advance(ws: CanonicalWorldState, *, peak_c: int, warm_capacity: int, prewarm_events: int,
             cold_started: int, warm_hold_gpu_hours: float, mg: dict, prewarm: str = "off",
             dt_seconds: float = 3600.0) -> None:
    """Commit the chosen action to the REAL world state and step the clock one period.

    Reactive (``off``) cools idle replicas down to what was actually used (``peak_c``) so it never
    carries a warm pool it isn't paying off — that is what makes ``off`` the no-warm-hold baseline.
    Proactive prewarming holds the larger forecast-sized pool warm (and pays for it)."""
    # land any migrations that have now completed — the moved replica adopts its target rack (its
    # locality benefit turns on for subsequent periods). It is the SAME replica object (identity moves,
    # never duplicated): weights stay resident and it lands WARM; KV warmth is kept per the move mode
    # (pipelined ≈ 0.9, bulk less) — that is what makes a migrated replica cheaper than a cold-start.
    for m in ws.migrations:
        if m.status == "in_flight" and m.end_period <= ws.period:
            rep = ws.replicas.get(m.replica_id)
            if rep is not None:
                rep.rack_id = (m.target_server_id.split(":")[0]
                               if ":" in m.target_server_id else rep.rack_id)
                rep.migrating = False
                rep.warm = True                          # lands warm — not re-cold-started
                rep.weights_loaded = True                # weights moved with the replica
                rep.kv_warm_frac = round(rep.kv_warm_frac * m.kv_preserved_frac, 5)  # KV partly kept
            m.status = "completed"
    # warm the replicas that served (or were prewarmed); the rest cool down AFTER the idle timeout.
    # Persistence is TIME-based: a replica idle for less than WARM_IDLE_TIMEOUT_S stays warm. At a
    # coarse control interval (dt≥timeout, e.g. hourly) the pool cools every step (the PR-#102
    # behaviour); at a FINE interval (dt<timeout, e.g. 5-min) a prewarmed/used pool SURVIVES across
    # steps — which is what lets prewarming/migration pay off over a multi-step horizon. (See the
    # multi-period MPC architecture doc: deferred-benefit actions only span periods when the control
    # interval is shorter than the action's persistence timescale.)
    warm_target = peak_c if prewarm == "off" else max(peak_c, warm_capacity)
    rep_ids = list(ws.replicas)
    for i, rid in enumerate(rep_ids):
        r = ws.replicas[rid]
        idle_s = ((ws.period - r.last_used_period) * dt_seconds) if r.last_used_period >= 0 else 1e18
        held_by_timeout = r.warm and idle_s < WARM_IDLE_TIMEOUT_S
        if i < warm_target:
            r.warm = True
            r.weights_loaded = True            # warm ⇒ weights resident in HBM
            r.warm_until_period = ws.period + max(1, int(WARM_IDLE_TIMEOUT_S / max(dt_seconds, 1e-9)))
            r.active = i < peak_c
            if i < peak_c:
                r.last_used_period = ws.period
                r.kv_warm_frac = 1.0           # served this step ⇒ its cache is hot
        elif held_by_timeout:
            r.warm = True                     # still within the idle timeout → not cooled yet
            r.active = False
        else:
            r.warm = False                     # cooled past the idle timeout → weights unloaded, cache gone
            r.weights_loaded = False
            r.kv_warm_frac = 0.0
            r.active = False
    # start migrations: withhold those replicas and schedule their landing.
    for rid in mg.get("targets", []):
        r = ws.replicas.get(rid)
        if r is None:
            continue
        r.migrating = True
        ws.migrations.append(MigrationState(
            migration_id=f"mig{len(ws.migrations)}", replica_id=rid, source_server_id=r.server_id,
            target_server_id=f"{mg.get('target_rack', r.rack_id)}:srv", start_period=ws.period,
            end_period=ws.period + MIGRATION_DURATION_PERIODS,
            migration_cost=MIGRATION_COST_PER_REPLICA, capacity_loss=1,
            cache_invalidation_cost=MIGRATION_CACHE_PENALTY * (1.0 - mg.get("kv_preserved_frac", 1.0)),
            kv_preserved_frac=mg.get("kv_preserved_frac", 1.0)))
    # accumulate ledgers.
    ws.warm_state.warm_replicas = ws.warm_count()
    ws.warm_state.cold_start_events += cold_started
    ws.warm_state.prewarm_events += prewarm_events
    ws.warm_state.warm_hold_gpu_hours += warm_hold_gpu_hours
    ws.warm_state.wasted_prewarm_hours += warm_hold_gpu_hours if peak_c < warm_capacity else 0.0
    ws.cost_state.migration_cost += mg.get("migration_cost", 0.0)
    ws.migrations = [m for m in ws.migrations if m.status == "in_flight"]
    ws.period += 1


def score_simulated_outcome(outcome: PeriodOutcome) -> tuple:
    """``(goodput_per_dollar, sla_violation_rate)`` — the MPC's per-candidate score signal."""
    return outcome.goodput_per_dollar, outcome.sla_violation_rate


def _gpu_hour_usd() -> float:
    from ..optimizer.unified_replay import GPU_HOUR_USD
    return GPU_HOUR_USD


__all__ = [
    "PeriodOutcome", "initialize_world_state", "clone_world_state_for_candidate",
    "reset_world_state", "warm_seed", "simulate_period", "score_simulated_outcome",
    "PREWARM_OPTIONS", "PLACEMENT_OPTIONS", "MIGRATION_OPTIONS",
    "COLD_START_S", "TOPOLOGY_MAX_DISCOUNT", "MIGRATION_COST_PER_REPLICA",
]
