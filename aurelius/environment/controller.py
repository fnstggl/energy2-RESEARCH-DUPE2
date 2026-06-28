"""ModelPredictiveEconomicController — forecast → simulate → choose (no deep RL).

A debuggable model-predictive controller over the **connected** environment actions
(``capacity`` / ``ordering`` / ``admission`` — see the Phase-0 audit; KV routing /
DVFS / placement are NOT connected and are NOT offered here). Each decision period:

    observe history  →  ForecastBundle over horizon H (causal, from train data)
      →  enumerate candidate action plans (connected actions only)
        →  simulate each plan on the FORECASTED load (point + p90 risk scenario)
          →  score by expected SLA-safe goodput / operator-$
            →  choose the best safe action  (fallback to SLA-aware if low confidence)

Strictly causal: the decision uses only the forecast (built from periods ≤ now) — never
the real next period's arrivals. The harness applies the chosen action to the real
period AFTER the decision. Savings are SIMULATED; the claim gate lives in
``training.py``.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field

from ..benchmarks.srtf_serving_backtest import _service_time_s
from ..optimizer.unified_replay import (
    CLASS_BEST_EFFORT,
    CLASS_LATENCY,
    Job,
    run_unified_replay,
)
from .action_registry import planned_report
from .actions import replay_kwargs_from_action
from .candidate_search import CandidateBundleGenerator, plan_bundle
from .cost_model import CostModel
from .forecast_trajectory import build_trajectory
from .forecasting import ForecastingModel
from .simulation_clock import SimulationClock
from .world_simulator import simulate_period

# Connected action levers (the only ones the environment actually executes).
CAPACITY = ("reactive_lag1", "backlog_aware", "forecasted_mcs")
ORDERING = ("fifo", "abs_conformal")
ADMISSION = ("off", "class_aware")
SLA_AWARE_FALLBACK = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off"}


def enumerate_actions() -> list:
    return [{"capacity": c, "ordering": o, "admission": a}
            for c in CAPACITY for o in ORDERING for a in ADMISSION]


@dataclass
class Decision:
    action: dict                       # legacy {capacity, ordering, admission} (back-compat)
    expected_gpd: float
    risk_gpd: float                    # gp/$ in the p90 high-load scenario
    score: float
    used_fallback: bool
    confidence: float
    forecast: dict = field(default_factory=dict)
    bundle: object = None              # the full ActionBundle chosen (connected levers)

    def to_dict(self) -> dict:
        d = {"action": self.action, "expected_gpd": round(self.expected_gpd, 2),
             "risk_gpd": round(self.risk_gpd, 2), "score": round(self.score, 2),
             "used_fallback": self.used_fallback, "confidence": round(self.confidence, 3)}
        if self.bundle is not None:
            d["bundle_changes"] = self.bundle.non_default_surfaces()
            d["routing_policy"] = self.bundle.routing_policy   # connected via kv_service_factor
            d["capacity_multiplier"] = self.bundle.capacity_multiplier
            d["batching_policy"] = self.bundle.batching_policy
            d["prewarm_policy"] = self.bundle.prewarm_policy   # connected via world_simulator
            d["placement_policy"] = self.bundle.placement_policy
            d["migration_policy"] = self.bundle.migration_policy
        return d


def _synth_jobs(arrival_rate: float, tok_mean: float, tok_p95: float, cv: float, *,
                window_seconds: float, best_effort_fraction: float, kv_service_factor: float) -> list:
    """Deterministic synthetic job set matching a forecasted period profile (no RNG).

    ``window_seconds`` is the simulation window the candidate actions are scored over —
    normally the full period, but for long (e.g. hourly) periods the controller scores a
    bounded representative window instead (same arrival rate + token/burst profile, so the
    queueing regime and therefore the action ranking are preserved at far less cost)."""
    n = max(0, int(round(arrival_rate * window_seconds)))
    if n == 0:
        return []
    be_stride = max(1, round(1.0 / best_effort_fraction)) if best_effort_fraction > 0 else 0
    # bursty arrivals: clump fraction grows with CV (deterministic interleave)
    burst = min(0.9, max(0.0, cv / 4.0))
    jobs = []
    for i in range(n):
        frac = i / n
        # tokens: most at mean, a deterministic 1-in-20 tail at p95
        tok = tok_p95 if (i % 20 == 0 and tok_p95 > 0) else tok_mean
        tok = max(1, int(tok))
        # arrival time: compress `burst` of the mass into the first 30% of the window
        arr = (frac * 0.3 if frac < burst else 0.3 + (frac - burst) / max(1e-9, 1 - burst) * 0.7)
        arr *= window_seconds
        cls = CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY
        jobs.append(Job(idx=i, arrival_s=arr, actual_tokens=tok, predicted_tokens=float(tok_mean),
                        service_s=_service_time_s(tok) * kv_service_factor, cls=cls))
    return jobs


@dataclass
class ModelPredictiveEconomicController:
    """MPC over connected actions, scored by expected SLA-safe goodput / operator-$."""

    forecasters: ForecastingModel
    fleet_state: object                # FleetState (constant anchored marginals)
    cost_model: CostModel
    horizon: int = 4
    sla_s: float = 10.0
    period_seconds: float = 60.0
    tick_seconds: float = 60.0
    risk_weight: float = 0.5           # penalty on the p90 high-load SLA-violation rate
    confidence_min: float = 0.15       # below this, fall back to the SLA-aware action
    kv_service_factor: float = 1.0     # default KV service discount (≤1) when no routing map
    kv_service_factor_by_routing: dict | None = None   # routing_policy → service factor
    #                                    (from fleet_kv_routing on Mooncake) — makes routing
    #                                    a CONNECTED action: kv_aware reuses more prefix → lower factor
    cost_scenario: str = "owned"
    sim_seconds: float | None = None   # bounded decision-sim window (default = period_seconds)
    optimize_simulated: bool = False   # opt-in to vary SIMULATED_ONLY surfaces (no reward
    #                                    effect until they are wired into run_unified_replay)
    candidates: list | None = None     # explicit candidate bundles; else generator-enumerated
    candidate_generator: object = None  # CandidateBundleGenerator (else a default exhaustive one)
    search_budget: int = 256           # exhaustive at/below this many bundles, else coordinate descent
    world_state: object = None         # CanonicalWorldState — when set, the stateful actions
    #                                    (prewarm/placement/migration) are scored through the
    #                                    world_simulator on a READ-ONLY (mutate=False) candidate sim
    # --- receding-horizon MPC (multi-period rollout over the persistent world) ---
    horizon_steps: int = 1             # MPC planning horizon in SIM STEPS (1 = single-period; >1
    #                                    rolls each candidate first-action H steps on a CLONE)
    gamma: float = 1.0                 # discount factor on future-step reward
    uncertainty_mode: str = "deterministic"   # forecast trajectory mode (point path)
    max_candidate_bundles: int = 256   # search-budget cap (bundles evaluated per decision)
    max_horizon_steps: int = 48        # hard cap on H (runtime safety)
    decision_timeout_s: float = 0.0    # >0 → abort the search after this wall-time, keep best so far
    last_decision_diag: dict = field(default_factory=dict)   # rollout/credit-assignment diagnostics

    def _gpd(self, jobs: list, replay_kw: dict, price: float) -> tuple:
        if not jobs:
            return 0.0, 0.0
        kpi = run_unified_replay(jobs, tick_seconds=self.tick_seconds, sla_s=self.sla_s,
                                 warmup_c=max(1, min(self.fleet_state.capacity_envelope, 4)),
                                 **replay_kw)
        gpu_type = (max(self.fleet_state.gpu_type_mix, key=self.fleet_state.gpu_type_mix.get)
                    if self.fleet_state.gpu_type_mix else "H100")
        cost = self.cost_model.operator_cost(
            gpu_hours=kpi.gpu_hours, gpu_type=gpu_type, energy_price_per_kwh=price,
            utilization=self.fleet_state.util_target, scenario=self.cost_scenario,
            sla_violations=kpi.sla_violations)
        gpd = kpi.sla_safe_goodput / max(cost.total_operator_cost, 1e-9)
        viol_rate = kpi.sla_violations / max(1, kpi.n_total)
        return gpd, viol_rate

    def _rollout_world(self, cand_ab, traj, *, be, factor, horizon_steps):
        """Receding-horizon rollout of ONE candidate first-action over the persistent world.

        Clones the real ClusterState, simulates ``horizon_steps`` future steps consuming the forecast
        TRAJECTORY (step k → ``traj.point(target, k)``), and accumulates the discounted risk-adjusted
        per-step reward. Step 0 applies the candidate; steps 1..H-1 apply a BASE CONTINUATION (the
        candidate's serving + placement levers held, prewarm/migration reverted to no-op) — we commit
        only the first action, so future stateful actions are not pre-committed, but the first action's
        STATE consequences (a moved replica's locality, a warm pool) persist and pay off downstream.
        READ-ONLY on the real world (operates on a clone). Returns (cumulative_return, step_diags)."""
        from .world_simulator import clone_world_state_for_candidate
        clone = clone_world_state_for_candidate(self.world_state)
        continuation = cand_ab.with_overrides(prewarm_policy="off", migration_policy="off")
        win = self.sim_seconds or self.period_seconds
        common0 = dict(sla_s=self.sla_s, tick_seconds=self.tick_seconds, base_service_factor=factor,
                       cost_model=self.cost_model, fleet_state=self.fleet_state,
                       cost_scenario=self.cost_scenario, best_effort_fraction=be,
                       period_hours=max(win, 1.0) / 3600.0, dt_seconds=self.period_seconds)
        cumulative, steps = 0.0, []
        for k in range(horizon_steps):
            bundle_k = cand_ab if k == 0 else continuation
            ar, tm = traj.point("arrival_rate", k), traj.point("output_token_mean", k)
            tp, cv = traj.point("output_token_p95", k), traj.point("interarrival_cv", k)
            if ar is None or tm is None:                     # forecast ran out → hold last step
                ar, tm, tp, cv = (traj.point(t, max(0, k - 1)) for t in
                                  ("arrival_rate", "output_token_mean", "output_token_p95",
                                   "interarrival_cv"))
            fc_k = {"arrival_rate": ar.mean, "arrival_p90": ar.p90,
                    "mean_service_s": max(_service_time_s(int(tm.mean)), 1e-3)}
            rkw = bundle_k.replay_kwargs()
            point = [(j.arrival_s, j.actual_tokens, j.actual_tokens) for j in
                     _synth_jobs(ar.mean, tm.mean, tp.value, cv.mean, window_seconds=win,
                                 best_effort_fraction=be, kv_service_factor=1.0)]
            risk = [(j.arrival_s, j.actual_tokens, j.actual_tokens) for j in
                    _synth_jobs(ar.p90, tm.p90, tp.p99, cv.p90, window_seconds=win,
                                best_effort_fraction=be, kv_service_factor=1.0)]
            risk_viol = simulate_period(clone, bundle_k, risk, fc_k, replay_kwargs=rkw,
                                        mutate=False, **common0).sla_violation_rate
            out = simulate_period(clone, bundle_k, point, fc_k, replay_kwargs=rkw,
                                  mutate=True, **common0)     # advance the cloned world one step
            exp_gpd = out.goodput_per_dollar
            reward = exp_gpd - self.risk_weight * risk_viol * exp_gpd
            cumulative += (self.gamma ** k) * reward
            steps.append({"step": k, "reward": round(reward, 2), "gp_per_dollar": round(exp_gpd, 1),
                          "risk_viol": round(risk_viol, 4), "warm_hold_cost": round(out.warm_hold_cost, 3),
                          "migration_cost": round(out.migration_cost, 3),
                          "cold_start_events": out.cold_start_events, "peak_c": out.kpi.c_max,
                          "topology_factor": out.topology_factor})
        return cumulative, steps

    def decide(self, history: list) -> Decision:
        """Choose the action for the next period from the causal forecast only."""
        if not self.forecasters.fitted or len(history) < 3:
            return Decision(dict(SLA_AWARE_FALLBACK), 0.0, 0.0, 0.0, True, 0.0)
        bundle = self.forecasters.predict(history, horizon=self.horizon)
        ar = bundle.at("arrival_rate", 0)
        tm = bundle.at("output_token_mean", 0)
        tp = bundle.at("output_token_p95", 0)
        cv = bundle.at("interarrival_cv", 0)
        pr = bundle.at("electricity_price", 0)
        # confidence: tighter band (relative) → higher confidence
        spread = (ar.p90 - ar.p10) / max(1e-9, ar.mean) if ar else 1.0
        confidence = max(0.0, 1.0 - spread)
        if confidence < self.confidence_min:
            return Decision(dict(SLA_AWARE_FALLBACK), 0.0, 0.0, 0.0, True, confidence,
                            forecast={"arrival_rate": ar.to_dict() if ar else {}})

        be = self.fleet_state.best_effort_fraction
        win = self.sim_seconds or self.period_seconds
        by_routing = self.kv_service_factor_by_routing or {}

        def _jobs(factor):                 # synth point+risk job sets at a given KV factor
            return (_synth_jobs(ar.mean, tm.mean, tp.value, cv.mean, window_seconds=win,
                                best_effort_fraction=be, kv_service_factor=factor),
                    _synth_jobs(ar.p90, tm.p90, tp.p99, cv.p90, window_seconds=win,
                                best_effort_fraction=be, kv_service_factor=factor))

        job_cache: dict = {}               # KV factor → (point, risk) jobs (routing changes the factor)

        # world-state scoring (stateful actions): a receding-horizon ROLLOUT over the persistent
        # world consuming the forecast TRAJECTORY. H=1 reproduces the single-period score exactly.
        world = self.world_state
        H = max(1, min(int(self.horizon_steps), int(self.max_horizon_steps)))
        clock = SimulationClock(dt_seconds=self.period_seconds)
        traj = (build_trajectory(self.forecasters, history, clock, H, mode=self.uncertainty_mode)
                if world is not None else None)
        rollout_cache: dict = {}           # id(ab) → (cumulative, step_diags)
        eval_count = [0]
        t_start = [None]

        def _resolve(cand):
            ab = cand if hasattr(cand, "replay_kwargs") else None
            if ab is not None:
                return ab, ab.legacy_action(), ab.replay_kwargs(), ab.routing_policy
            routing = cand.get("routing_policy", "round_robin") if isinstance(cand, dict) else "round_robin"
            return None, cand, replay_kwargs_from_action(cand if isinstance(cand, dict) else {}), routing

        def _eval(cand):                   # → (score, exp_gpd, risk_gpd, routing)
            ab, _act, rkw, routing = _resolve(cand)
            factor = by_routing.get(routing, self.kv_service_factor)
            if world is not None and ab is not None:
                # receding-horizon rollout (H=1 reproduces the single-period score exactly).
                eval_count[0] += 1
                cumulative, steps = self._rollout_world(ab, traj, be=be, factor=factor, horizon_steps=H)
                rollout_cache[id(ab)] = (cumulative, steps)
                exp_gpd = steps[0]["gp_per_dollar"] if steps else 0.0
                return cumulative, exp_gpd, exp_gpd, routing
            if factor not in job_cache:
                job_cache[factor] = _jobs(factor)
            point, risk = job_cache[factor]
            exp_gpd, _ = self._gpd(point, rkw, pr.value)
            risk_gpd, risk_viol = self._gpd(risk, rkw, pr.value)
            return exp_gpd - self.risk_weight * risk_viol * exp_gpd, exp_gpd, risk_gpd, routing

        # search the CONNECTED bundle space — exhaustive when small, coordinate descent when large
        # (no connected knob skipped). Each candidate is scored by the receding-horizon rollout.
        import time as _time
        t_start[0] = _time.monotonic()
        budget = min(int(self.search_budget), int(self.max_candidate_bundles))
        timed_out = [False]

        def _score(b):
            if self.decision_timeout_s > 0 and (_time.monotonic() - t_start[0]) > self.decision_timeout_s:
                timed_out[0] = True
                return -1e18                       # freeze the search; keep best-so-far
            return _eval(b)[0]

        if self.candidates is not None:
            method, theoretical = "explicit", len(self.candidates)
            best_cand = max(self.candidates, key=_score)
        else:
            gen = self.candidate_generator or CandidateBundleGenerator(
                include_simulated=self.optimize_simulated)
            theoretical = gen.theoretical_combinations()
            best_cand, _n, method = gen.search(_score, budget=budget)
        ab, act, _rkw, _routing = _resolve(best_cand)
        score, exp_gpd, risk_gpd, routing = _eval(best_cand)
        cumulative, win_steps = rollout_cache.get(id(ab), (score, []))
        self.last_decision_diag = {
            **clock.horizon_meta(H), "gamma": self.gamma, "uncertainty_mode": self.uncertainty_mode,
            "theoretical_bundles": theoretical, "candidate_bundles_evaluated": eval_count[0],
            "world_steps_simulated": eval_count[0] * H, "search_method": method,
            "runtime_s": round(_time.monotonic() - t_start[0], 4), "timed_out": timed_out[0],
            "cumulative_return": round(cumulative, 2), "rollout": win_steps,
            "trajectory": traj.to_dict() if traj is not None else None} if world is not None else {}
        return Decision(act, exp_gpd, risk_gpd, score, False, confidence,
                        forecast={"arrival_rate": ar.to_dict(), "price": pr.value,
                                  "routing_policy": routing}, bundle=ab)

    def understood_but_unavailable(self) -> list:
        """Action surfaces the controller REPRESENTS but does not optimize today
        (SIMULATED_ONLY + PLANNED) — reported separately so planned knobs are never
        mistaken for active ones. See research/AURELIUS_ACTION_SURFACE_AUDIT.md."""
        return planned_report()

    def search_report(self, history: list) -> dict | None:
        """Audit the planner's bundle search for ONE decision: total connected dimensions,
        theoretical combinations, candidates evaluated, search method, best bundle, and a
        per-surface ablation (how much each connected knob moves the score). Proves the search
        is over the connected action space — not a hand-picked preset list — and that no
        connected knob is silently excluded."""
        if not self.forecasters.fitted or len(history) < 3:
            return None
        fb = self.forecasters.predict(history, horizon=self.horizon)
        ar, tm, tp, cv, pr = (fb.at(t, 0) for t in
                              ("arrival_rate", "output_token_mean", "output_token_p95",
                               "interarrival_cv", "electricity_price"))
        be = self.fleet_state.best_effort_fraction
        win = self.sim_seconds or self.period_seconds
        by_routing = self.kv_service_factor_by_routing or {}

        def score_fn(b):
            factor = by_routing.get(b.routing_policy, self.kv_service_factor)
            point = _synth_jobs(ar.mean, tm.mean, tp.value, cv.mean, window_seconds=win,
                                best_effort_fraction=be, kv_service_factor=factor)
            risk = _synth_jobs(ar.p90, tm.p90, tp.p99, cv.p90, window_seconds=win,
                               best_effort_fraction=be, kv_service_factor=factor)
            rkw = b.replay_kwargs()
            exp_gpd, _ = self._gpd(point, rkw, pr.value)
            _r, risk_viol = self._gpd(risk, rkw, pr.value)
            return exp_gpd - self.risk_weight * risk_viol * exp_gpd, risk_viol

        gen = self.candidate_generator or CandidateBundleGenerator(
            include_simulated=self.optimize_simulated)
        _best, report = plan_bundle(gen, score_fn)
        return report.to_dict()


def _causal_pred(slice_sorted: list) -> list:
    """Running-median causal token prior (deployable, no oracle)."""
    n = len(slice_sorted)
    if n == 0:
        return []
    gmed = sorted(t for _, t, *_ in slice_sorted)[n // 2]
    pred, seen = [0.0] * n, []
    for i, rec in enumerate(slice_sorted):
        pred[i] = float(seen[(len(seen) - 1) // 2]) if seen else float(gmed)
        bisect.insort(seen, rec[1])
    return pred


@dataclass
class EpisodeReport:
    name: str
    n_periods: int
    sla_safe_goodput: float
    total_operator_cost: float
    goodput_per_dollar: float
    sla_violation_rate: float
    gpu_hours: float
    energy_cost: float
    n_sla_safe: int
    queue_delay_p95: float
    queue_delay_p99: float = 0.0                        # tail queueing delay (per-period averaged)
    used_fallback_frac: float = 0.0
    routing_mix: dict = field(default_factory=dict)     # routing_policy → periods chosen
    mean_kv_service_factor: float = 1.0                 # mean KV service factor applied
    capacity_multiplier_mix: dict = field(default_factory=dict)  # capacity_multiplier → periods
    batching_mix: dict = field(default_factory=dict)    # batching_policy → periods chosen
    prewarm_mix: dict = field(default_factory=dict)     # prewarm_policy → periods (world path)
    placement_mix: dict = field(default_factory=dict)   # placement_policy → periods
    migration_mix: dict = field(default_factory=dict)   # migration_policy → periods
    cold_start_events: int = 0                          # total cold starts incurred
    warm_hold_gpu_hours: float = 0.0                    # total GPU-hours held warm (prewarm cost)
    migration_cost: float = 0.0                         # total $ spent on live moves
    mean_topology_factor: float = 1.0                   # mean placement service-time factor

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def run_period_episode(name, decide_fn, real_per_period, frames, eval_indices, *,
                       fleet_state, cost_model, sla_s=10.0, tick_seconds=60.0,
                       period_seconds=60.0, kv_service_factor=1.0, cost_scenario="owned",
                       sim_seconds=None, kv_service_factor_by_routing=None, world_state=None,
                       world_state_params=None):
    """Run ``decide_fn(history_frames)`` over the REAL eval periods (causal: the action
    for period p is chosen from frames[:p], then applied to the real requests of p).

    The chosen ``routing_policy`` (CONNECTED via the fleet-KV channel) selects the period's
    KV service factor from ``kv_service_factor_by_routing`` — so a routing decision changes
    the replayed service times (and thus goodput/$). ``sim_seconds`` is accepted (and ignored)
    so the same ``common`` dict can be splatted here and into the controller.

    When ``world_state`` is given, the period is replayed through the PERSISTENT world simulator
    (``simulate_period``, ``mutate=True``) so prewarm / placement / migration take effect and the
    state evolves period→period; the warm pool is sized from a causal lag-1 forecast. With every
    stateful surface at its no-op this reproduces the stateless replay above."""
    from .world_simulator import simulate_period as _sim_period
    by_routing = kv_service_factor_by_routing or {}
    gpu_type = (max(fleet_state.gpu_type_mix, key=fleet_state.gpu_type_mix.get)
                if fleet_state.gpu_type_mix else "H100")
    be = fleet_state.best_effort_fraction
    be_stride = max(1, round(1.0 / be)) if be > 0 else 0
    tot_g = tot_cost = tot_energy = tot_gpu_h = 0.0
    tot_viol = tot_n = tot_safe = 0
    waits_p95: list = []
    waits_p99: list = []
    fb = 0
    routing_mix: dict = {}
    cap_mult_mix: dict = {}
    batch_mix: dict = {}
    prewarm_mix: dict = {}
    placement_mix: dict = {}
    migration_mix: dict = {}
    cold_starts = mig_cost = warm_hold = topo_sum = topo_n = 0.0
    factor_sum, factor_n = 0.0, 0
    period_hours = max(period_seconds, 1.0) / 3600.0
    for p in eval_indices:
        out = decide_fn(frames[:p])
        action = out["action"] if isinstance(out, dict) and "action" in out else out
        fb += int(bool(isinstance(out, dict) and out.get("used_fallback")))
        # merge the connected surfaces the MPC exposes at the top level onto the legacy 3-key
        # action; baselines carry their own keys (else the no-op default).
        merged = dict(action) if isinstance(action, dict) else {}
        if isinstance(out, dict):
            for _k in ("routing_policy", "capacity_multiplier", "batching_policy",
                       "prewarm_policy", "placement_policy", "migration_policy"):
                if _k in out:
                    merged[_k] = out[_k]
        routing = merged.get("routing_policy", "round_robin")
        factor = by_routing.get(routing, kv_service_factor)
        replay_kw = replay_kwargs_from_action(merged)
        routing_mix[routing] = routing_mix.get(routing, 0) + 1
        _cm = float(merged.get("capacity_multiplier", 1.0))
        cap_mult_mix[_cm] = cap_mult_mix.get(_cm, 0) + 1
        _bp = merged.get("batching_policy", "conservative")
        batch_mix[_bp] = batch_mix.get(_bp, 0) + 1
        _pw = merged.get("prewarm_policy", "off")
        _pl = merged.get("placement_policy", "topology_blind")
        _mg = merged.get("migration_policy", "off")
        prewarm_mix[_pw] = prewarm_mix.get(_pw, 0) + 1
        placement_mix[_pl] = placement_mix.get(_pl, 0) + 1
        migration_mix[_mg] = migration_mix.get(_mg, 0) + 1
        factor_sum += factor
        factor_n += 1
        recs = sorted(real_per_period.get(p, []), key=lambda r: r[0])
        if not recs:
            continue
        if world_state is not None:
            # causal lag-1 forecast (previous real period) sizes the warm pool.
            prev = real_per_period.get(p - 1, recs)
            fcast = {"arrival_rate": len(prev) / max(period_seconds, 1e-9),
                     "arrival_p90": 1.3 * len(prev) / max(period_seconds, 1e-9),
                     "mean_service_s": (statistics.mean(_service_time_s(int(r[1])) for r in prev)
                                        if prev else 1.0)}
            from types import SimpleNamespace
            pol = SimpleNamespace(prewarm_policy=_pw, placement_policy=_pl, migration_policy=_mg)
            t0 = recs[0][0]
            oc = _sim_period(world_state, pol, [(r[0] - t0, int(r[1]), r[2] if len(r) > 2 else r[1])
                                                for r in recs], fcast, sla_s=sla_s,
                             tick_seconds=tick_seconds, base_service_factor=factor,
                             replay_kwargs=replay_kw, cost_model=cost_model, fleet_state=fleet_state,
                             cost_scenario=cost_scenario, best_effort_fraction=be,
                             period_hours=period_hours, dt_seconds=period_seconds, mutate=True)
            kpi = oc.kpi
            tot_cost += oc.operator_cost
            cold_starts += oc.cold_start_events
            mig_cost += oc.migration_cost
            warm_hold += oc.wasted_prewarm_hours
            topo_sum += oc.topology_factor
            topo_n += 1
            if oc.queue_delay_p95 > 0:
                waits_p95.append(oc.queue_delay_p95)
                waits_p99.append(oc.queue_delay_p99)
            tot_g += kpi.sla_safe_goodput
            tot_gpu_h += kpi.gpu_hours
            tot_viol += kpi.sla_violations
            tot_n += kpi.n_total
            tot_safe += kpi.n_sla_safe
            continue
        t0 = recs[0][0]
        pred = _causal_pred(recs)
        jobs = [Job(idx=i, arrival_s=(r[0] - t0), actual_tokens=int(r[1]),
                    predicted_tokens=float(pred[i]),
                    service_s=_service_time_s(int(r[1])) * factor,
                    cls=(CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY))
                for i, r in enumerate(recs)]
        kpi = run_unified_replay(jobs, tick_seconds=tick_seconds, sla_s=sla_s,
                                 warmup_c=max(1, min(fleet_state.capacity_envelope, 4)), **replay_kw)
        cost = cost_model.operator_cost(
            gpu_hours=kpi.gpu_hours, gpu_type=gpu_type,
            energy_price_per_kwh=fleet_state.energy_price_per_kwh,
            utilization=fleet_state.util_target, scenario=cost_scenario,
            sla_violations=kpi.sla_violations)
        tot_g += kpi.sla_safe_goodput
        tot_cost += cost.total_operator_cost
        tot_energy += cost.energy_cost
        tot_gpu_h += kpi.gpu_hours
        tot_viol += kpi.sla_violations
        tot_n += kpi.n_total
        tot_safe += kpi.n_sla_safe
        waits = sorted(max(0.0, j.start_s - j.arrival_s) for j in jobs if j.start_s >= 0)
        if waits:
            waits_p95.append(waits[min(len(waits) - 1, int(len(waits) * 0.95))])
            waits_p99.append(waits[min(len(waits) - 1, int(len(waits) * 0.99))])
    ne = len(eval_indices)
    return EpisodeReport(
        name=name, n_periods=ne, sla_safe_goodput=tot_g, total_operator_cost=tot_cost,
        goodput_per_dollar=tot_g / max(tot_cost, 1e-9),
        sla_violation_rate=(tot_viol / tot_n if tot_n else 0.0), gpu_hours=tot_gpu_h,
        energy_cost=tot_energy, n_sla_safe=tot_safe,
        queue_delay_p95=(statistics.mean(waits_p95) if waits_p95 else 0.0),
        queue_delay_p99=(statistics.mean(waits_p99) if waits_p99 else 0.0),
        used_fallback_frac=(fb / ne if ne else 0.0), routing_mix=routing_mix,
        mean_kv_service_factor=(factor_sum / factor_n if factor_n else kv_service_factor),
        capacity_multiplier_mix=cap_mult_mix, batching_mix=batch_mix,
        prewarm_mix=prewarm_mix, placement_mix=placement_mix, migration_mix=migration_mix,
        cold_start_events=int(cold_starts), warm_hold_gpu_hours=warm_hold, migration_cost=mig_cost,
        mean_topology_factor=(topo_sum / topo_n if topo_n else 1.0))


__all__ = [
    "CAPACITY", "ORDERING", "ADMISSION", "SLA_AWARE_FALLBACK", "enumerate_actions",
    "Decision", "ModelPredictiveEconomicController", "EpisodeReport", "run_period_episode",
]
