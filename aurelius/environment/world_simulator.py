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
from .world_state import CanonicalWorldState, MigrationState, build_sample_cluster

# --- BENCHMARK_DERIVED constants (documented public-prior magnitudes, not our trace) ----------
COLD_START_S = 30.0            # serving replica model-load / container warm time (vLLM/TGI regime)
WARM_HOLD_GPU_FRACTION = 1.0   # a warm idle replica still occupies its GPU → ~full GPU-hour held
TOPOLOGY_MAX_DISCOUNT = 0.08   # max service-time discount from perfect locality + lowest pressure
MIGRATION_COST_PER_REPLICA = 0.40   # operator $ to live-migrate one replica (move + reschedule)
MIGRATION_CAPACITY_LOSS_FRAC = 0.10 # capacity withheld per migrating replica, this period only
MIGRATION_CACHE_PENALTY = 0.04      # service-time surcharge from KV warmth lost on a moved replica
MIGRATION_DURATION_PERIODS = 1      # a move started in period p lands (benefit on) at p+1
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
    metrics: dict = field(default_factory=dict)

    @property
    def goodput_per_dollar(self) -> float:
        return self.kpi.sla_safe_goodput / max(self.operator_cost, 1e-9)

    @property
    def sla_violation_rate(self) -> float:
        return self.kpi.sla_violations / max(self.kpi.n_total, 1)


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
        ws.replicas[rid].warm = i < n
        if i < n:
            ws.replicas[rid].last_used_period = ws.period
    ws.warm_state.warm_replicas = ws.warm_count()


def reset_world_state(ws: CanonicalWorldState) -> None:
    """Return all replicas to cold/idle and clear migrations (fresh simulation)."""
    for r in ws.replicas.values():
        r.warm = False
        r.active = False
        r.migrating = False
        r.last_used_period = -1
        r.cold_start_remaining_s = 0.0
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
    """Topology service-time factor (≤ 1.0) from which racks the c_used replicas land on.

    Mirrors the KV-routing channel: topology_blind = 1.0 (no exploitation, the PR-#100 baseline);
    rack_local consolidates onto the fewest racks; network_aware prefers the lowest-pressure racks.
    Discount scales with how much the macro network pressure VARIES across racks (no spread → no
    free lunch). Macro only — no per-link claims."""
    if not ws.racks:
        return {"topology_factor": 1.0, "locality_score": 1.0, "rack_spread": 1, "used_racks": []}
    order = _racks_by_pressure(ws)
    pressures = [ws.racks[r].macro_network_pressure for r in order]
    p_min, p_max = min(pressures), max(pressures)
    spread_range = p_max - p_min                       # 0 → all racks equal → no topology lever
    per_rack_cap = {r: max(1, ws.racks[r].gpu_capacity) for r in order}

    if policy == "topology_blind":
        used = list(order)                              # spread across all racks (baseline)
        factor = 1.0
    else:
        # consolidate onto as few of the PREFERRED racks as hold c_used (network_aware prefers the
        # lowest-pressure racks; rack_local just minimises spread from the current order).
        used, acc = [], 0
        for r in order:
            used.append(r)
            acc += per_rack_cap[r]
            if acc >= c_used:
                break
        # discount: bigger when we concentrate onto low-pressure racks AND pressure varies.
        used_press = sum(ws.racks[r].macro_network_pressure for r in used) / len(used)
        avg_press = sum(pressures) / len(pressures)
        relief = max(0.0, avg_press - used_press)       # how much pressure we avoided
        locality_bonus = (len(order) - len(used)) / len(order)   # 0..1 consolidation
        if policy == "network_aware":
            factor = 1.0 - TOPOLOGY_MAX_DISCOUNT * (0.5 * locality_bonus + 0.5 * (relief + spread_range) / 2.0)
        else:  # rack_local: locality only, ignores which racks are hot
            factor = 1.0 - TOPOLOGY_MAX_DISCOUNT * 0.6 * locality_bonus
    factor = min(1.0, max(1.0 - TOPOLOGY_MAX_DISCOUNT, factor))
    return {"topology_factor": round(factor, 5), "locality_score": round(1.0 - (len(used) - 1) / max(1, len(order) - 1), 4) if len(order) > 1 else 1.0,
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
    return {"n_migrations": n, "capacity_loss_frac": round(cap_loss, 4),
            "migration_cost": round(MIGRATION_COST_PER_REPLICA * n, 4),
            "cache_factor": 1.0 + MIGRATION_CACHE_PENALTY * (1.0 if n else 0.0),
            "targets": [r.replica_id for r in targets], "target_rack": best_rack}


# ---------------------------------------------------------------------------
# the period simulation
# ---------------------------------------------------------------------------

def simulate_period(ws: CanonicalWorldState, bundle, recs: list, forecast: dict, *,
                    sla_s: float, tick_seconds: float, base_service_factor: float = 1.0,
                    replay_kwargs: dict | None = None, cost_model=None, fleet_state=None,
                    cost_scenario: str = "owned", best_effort_fraction: float = 0.0,
                    period_hours: float = 1.0, mutate: bool = False) -> PeriodOutcome:
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

    # 3) placement → topology service factor.
    pl = _placement_plan(ws, placement, c_used=max(c_used, warm_capacity))

    # 4) migration → capacity loss + cost + cache penalty THIS period.
    mg = _migration_plan(ws, migration, placement=pl)

    # 5) combine factors and run the serving replay.
    service_factor = base_service_factor * pl["topology_factor"] * mg["cache_factor"]
    cap_mult = float(rkw.get("capacity_multiplier", 1.0)) * (1.0 - mg["capacity_loss_frac"])
    be_stride = max(1, round(1.0 / best_effort_fraction)) if best_effort_fraction > 0 else 0
    recs_sorted = sorted(recs, key=lambda r: r[0])
    t0 = recs_sorted[0][0] if recs_sorted else 0.0
    jobs = [Job(idx=i, arrival_s=(r[0] - t0), actual_tokens=int(r[1]), predicted_tokens=float(r[1]),
                service_s=_service_time_s(int(r[1])) * service_factor,
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

    # 7) cost = serving operator cost + warm-hold + migration.
    peak_c = kpi.c_max
    idle_warm = max(0, warm_capacity - peak_c)                 # warm but never needed this period
    warm_hold_gpu_hours = idle_warm * WARM_HOLD_GPU_FRACTION * period_hours
    warm_hold_cost = warm_hold_gpu_hours * _gpu_hour_usd()
    cold_started = max(0, peak_c - pw["reactive_warm"]) if prewarm == "off" else max(0, peak_c - warm_capacity)
    if cost_model is not None and fleet_state is not None:
        gpu_type = (max(fleet_state.gpu_type_mix, key=fleet_state.gpu_type_mix.get)
                    if getattr(fleet_state, "gpu_type_mix", None) else "H100")
        cb = cost_model.operator_cost(
            gpu_hours=kpi.gpu_hours, gpu_type=gpu_type,
            energy_price_per_kwh=fleet_state.energy_price_per_kwh, utilization=fleet_state.util_target,
            scenario=cost_scenario, sla_violations=kpi.sla_violations, migrations=mg["n_migrations"])
        base_cost = cb.total_operator_cost
    else:
        base_cost = kpi.cost_usd
    total_cost = base_cost + warm_hold_cost + mg["migration_cost"]

    outcome = PeriodOutcome(
        kpi=kpi, operator_cost=total_cost, warm_hold_cost=round(warm_hold_cost, 5),
        migration_cost=mg["migration_cost"], service_factor=round(service_factor, 5),
        warm_capacity=warm_capacity, cold_start_s=pw["cold_start_s"],
        topology_factor=pl["topology_factor"], locality_score=pl["locality_score"],
        rack_spread=pl["rack_spread"], cold_start_events=cold_started,
        wasted_prewarm_hours=round(warm_hold_gpu_hours, 5), migrations_started=mg["n_migrations"],
        queue_delay_p95=round(q_p95, 4),
        metrics={"prewarm_policy": prewarm, "placement_policy": placement,
                 "migration_policy": migration, "warm_capacity": warm_capacity,
                 "peak_c": peak_c, "topology_factor": pl["topology_factor"],
                 "service_factor": round(service_factor, 5)})

    if mutate:
        _advance(ws, peak_c=peak_c, warm_capacity=warm_capacity, prewarm_events=pw["prewarm_events"],
                 cold_started=cold_started, warm_hold_gpu_hours=warm_hold_gpu_hours, mg=mg)
    return outcome


def _advance(ws: CanonicalWorldState, *, peak_c: int, warm_capacity: int, prewarm_events: int,
             cold_started: int, warm_hold_gpu_hours: float, mg: dict) -> None:
    """Commit the chosen action to the REAL world state and step the clock one period."""
    # land any migrations that have now completed — the moved replica adopts its target rack (its
    # locality benefit turns on for subsequent periods).
    for m in ws.migrations:
        if m.status == "in_flight" and m.end_period <= ws.period:
            rep = ws.replicas.get(m.replica_id)
            if rep is not None:
                rep.rack_id = (m.target_server_id.split(":")[0]
                               if ":" in m.target_server_id else rep.rack_id)
                rep.migrating = False
            m.status = "completed"
    # warm the replicas that served (or were prewarmed); the rest cool down.
    warm_target = max(peak_c, warm_capacity)
    rep_ids = list(ws.replicas)
    for i, rid in enumerate(rep_ids):
        r = ws.replicas[rid]
        if i < warm_target:
            r.warm = True
            r.last_used_period = ws.period if i < peak_c else r.last_used_period
            r.active = i < peak_c
        else:
            r.warm = False
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
            cache_invalidation_cost=MIGRATION_CACHE_PENALTY))
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
