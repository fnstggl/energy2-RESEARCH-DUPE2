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
from .actions import ActionBundle, replay_kwargs_from_action
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

# selectable planner modes (PR #123 tournament methods), driven through the planner package's run_method.
# `current_default` / `clock_only_diagnostic` / `physics_guided_beam` / `oracle_diagnostic` use the existing
# controller branches (set via candidates / physics_guided / planning_oracle_records); the rest map here.
_MODE_TO_METHOD = {
    "fixed_multi_knob_grid": "fixed_grid",
    "physics_guided_grid": "physics_guided_grid",
    "hierarchical_search": "hierarchical_search",
    "hierarchical_search_with_progressive_widening": "hierarchical_search",   # + a widening pass (see helper)
    "exhaustive_small_diagnostic": "exhaustive_small",
}
PACKAGE_PLANNER_MODES = tuple(_MODE_TO_METHOD)

# The DEFAULT benchmark planner. PR #124's default-change gate PASSED on the validation ladder (verdict
# `flip_benchmark_default`: hierarchical_search beats the prior physics-guided-beam default on gp/$ at no SLA
# cost, Pareto-dominates production_scheduler AND sla_aware, 0% search regret ≤ the prior default, anchors
# always contained, ~75 evals/decision bounded, 0 timeouts, no oracle, no quality-risked lever — see
# `data/external/mpc_controller/default_change_gate_verdict.json` and
# `research/HIERARCHICAL_PLANNER_PRODUCTION_COMPARISON.md`). So benchmark / standard-MPC reporting builds the
# controller with `planner_mode=DEFAULT_BENCHMARK_PLANNER_MODE` (e.g. via `training._controller(..., planner_mode
# =DEFAULT_BENCHMARK_PLANNER_MODE)`). It is intentionally NOT forced into the dataclass default: that keeps raw
# construction behaviour-preserving and leaves the specialized isolation backtests' explicit planner choices
# (clock-only / physics-guided / adaptive) intact, AND production-simulation runs stay on the physics-guided
# beam until the broader-window validation the audit asks for. The flip is real (declared + wired + evidenced),
# scoped honestly to benchmark reporting.
DEFAULT_BENCHMARK_PLANNER_MODE = "hierarchical_search"


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
            d["precision_policy"] = self.bundle.precision_policy   # connected via roofline_serving
            d["spec_decode_policy"] = self.bundle.spec_decode_policy
            d["clock_policy"] = self.bundle.clock_policy
            d["colocation_policy"] = self.bundle.colocation_policy
            d["prefill_decode_policy"] = self.bundle.prefill_decode_policy
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
    use_adaptive_search: bool = True   # adaptive planner (beam/CE + regret audit) over the fixed-256 cap
    search_planner_obj: object = None  # AdaptiveSearchPlanner instance (else a default one is built)
    # planning/eval fidelity (this PR): when set, the planning rollout runs the SAME phase + cost + roofline
    # model the evaluation path uses (a synthetic UNIQUE-prefix kv_state + this cost mode), so the planner
    # SEES the precision/spec/clock COST economics — not only their latency. None → today's latency-only
    # planning (default bundle still reproduces today's behaviour).
    planning_kv_cost_mode: str | None = None
    planning_capacity_blocks: int = 512
    planning_prompt_tokens: int | None = None   # representative prompt length for the planning workload
    # Batch-1 ablation mask: which new knobs (kv_cache_precision_policy / prefill_decode_policy /
    # gpu_assignment_policy) the planner may vary. None → use the product-boundary default (core orchestration
    # on; optional serving-engine integrations OFF unless their enable flag is set). An explicit set overrides
    # this entirely (used by the ablation runner).
    allowed_new_knobs: frozenset | None = None
    # OPTIONAL serving-engine integrations are DEFAULT-OFF (product boundary: Aurelius must not silently take
    # control of serving-engine internals). The operator opts in only when the serving stack exposes them.
    enable_kv_cache_precision: bool = False        # OPTIONAL_SERVING_ENGINE_INTEGRATION (vLLM/TRT-LLM fp8/int8 KV)
    enable_prefill_decode_disagg: bool = False     # OPTIONAL_SERVING_ENGINE_INTEGRATION (DistServe/Dynamo PD)
    # electricity (this PR): when True the horizon rollout prices each step at the FORECAST electricity price
    # path (traj.point("electricity_price", k)) so the controller chooses clock/DVFS against real diurnal
    # prices. Default False → every step uses the constant fleet price (today's behaviour, flat-price-identical).
    electricity_price_aware: bool = False
    # scenario forecaster (PR #113): plan across a small trace-derived workload ENSEMBLE (incl. SLA-pressure
    # futures) instead of one median, so the planning score is an expectation over realistic demand. Opt-in.
    planning_scenarios: bool = False
    scenario_tail_weight: float = 0.25          # risk-averse penalty on the worst-scenario SLA violation
    planning_oracle_records: list | None = None  # ORACLE diagnostic: plan against the EXACT future workload
    # online Decision Diagnostics (permanent, negligible overhead — exposes already-computed search values;
    # no leave-one-out / oracle / extra solves online). The controller never produces an action without one.
    emit_diagnostics: bool = True
    _decision_count: int = 0
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
    # physics-guided planner (opt-in): bounded-beam MPC over a physics-guided, anchor-guaranteed candidate
    # set + progressive widening (physics_guided_planner.py). Replaces clock-only / full-space search; the
    # known-good bundles are ALWAYS contained. Default False → behaviour unchanged.
    physics_guided: bool = False
    physics_planner_obj: object = None  # BoundedBeamPlanner instance (else a default one is built)
    current_price_percentile: float | None = None   # electricity price percentile for this decision (soft prior)
    _prev_best_bundle: object = None    # previous decision's winning bundle (continuity anchor)
    # selectable planner mode (PR #123 tournament methods). None → the existing branches (current default).
    # A package mode (hierarchical_search, …) drives the controller's per-decision rollout through the
    # planner package's run_method. Diagnostic / opt-in until the default-change gate passes.
    planner_mode: str | None = None
    planner_budget: int = 100           # evaluation budget for a package-mode planner

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
            # price-aware planning: each horizon step is priced at the FORECAST electricity price for that step
            # (causal — day-ahead prices are published ahead). None → constant fleet price (flat-price-identical).
            pr_k = None
            if self.electricity_price_aware:
                _pk = traj.point("electricity_price", k) or traj.point("electricity_price", 0)
                pr_k = _pk.value if _pk is not None else None
            rkw = bundle_k.replay_kwargs()
            if self.planning_scenarios or (self.planning_oracle_records is not None and k == 0):
                # plan across the scenario ensemble (or the exact future, in oracle mode)
                out, exp_gpd, reward, risk_viol = self._rollout_ensemble(clone, bundle_k, k, ar, tm, tp, cv,
                                                                         be, win, fc_k, rkw, common0)
            else:
                _pt = self.planning_prompt_tokens
                point = [(j.arrival_s, j.actual_tokens, _pt or j.actual_tokens) for j in
                         _synth_jobs(ar.mean, tm.mean, tp.value, cv.mean, window_seconds=win,
                                     best_effort_fraction=be, kv_service_factor=1.0)]
                risk = [(j.arrival_s, j.actual_tokens, _pt or j.actual_tokens) for j in
                        _synth_jobs(ar.p90, tm.p90, tp.p99, cv.p90, window_seconds=win,
                                    best_effort_fraction=be, kv_service_factor=1.0)]
                # planning/eval PARITY (PR #112): synthetic UNIQUE-prefix kv_state + the eval cost mode makes
                # the planning rollout run the same phase + hybrid-cost + roofline model the evaluator uses.
                kvp = kvr = None
                if self.planning_kv_cost_mode:
                    _cm, _cb = self.planning_kv_cost_mode, self.planning_capacity_blocks
                    kvp = {"hash_seq": [(f"plan{k}_{i}",) for i in range(len(point))],
                           "routing": bundle_k.routing_policy, "capacity_blocks": _cb, "cost_mode": _cm}
                    kvr = {**kvp, "hash_seq": [(f"plkr{k}_{i}",) for i in range(len(risk))]}
                risk_viol = simulate_period(clone, bundle_k, risk, fc_k, replay_kwargs=rkw,
                                            mutate=False, kv_state=kvr,
                                            energy_price_per_kwh=pr_k, **common0).sla_violation_rate
                out = simulate_period(clone, bundle_k, point, fc_k, replay_kwargs=rkw,
                                      mutate=True, kv_state=kvp,
                                      energy_price_per_kwh=pr_k, **common0)   # advance the cloned world one step
                exp_gpd = out.goodput_per_dollar
                reward = exp_gpd - self.risk_weight * risk_viol * exp_gpd
            cumulative += (self.gamma ** k) * reward
            steps.append({"step": k, "reward": round(reward, 2), "gp_per_dollar": round(exp_gpd, 1),
                          "risk_viol": round(risk_viol, 4), "warm_hold_cost": round(out.warm_hold_cost, 3),
                          "migration_cost": round(out.migration_cost, 3),
                          "cold_start_events": out.cold_start_events, "peak_c": out.kpi.c_max,
                          "topology_factor": out.topology_factor,
                          "sla_slack_ms": out.sla_slack_ms})   # N2: SLA headroom the chosen clock leaves (diagnostic)
        return cumulative, steps

    def _rollout_ensemble(self, clone, bundle_k, k, ar, tm, tp, cv, be, win, fc_k, rkw, common0):
        """Score the candidate across a small trace-derived workload ENSEMBLE (SLA-pressure aware), or — in
        ORACLE mode — against the EXACT realized future workload. Advances the clone on the BASE scenario
        only (single world track). Returns ``(base_outcome, expected_gp_per_dollar, risk_aware_reward)``
        where reward = E[gp/$] − risk·E[SLA] − tail·max(SLA), all scaled by E[gp/$]. No reward shaping."""
        from .scenario_forecaster import build_scenarios
        _pt = self.planning_prompt_tokens
        if self.planning_oracle_records is not None and k == 0:
            recs = self.planning_oracle_records                       # the EXACT future (oracle diagnostic)
            t0 = recs[0][0] if recs else 0.0
            scen = [("oracle", [(r[0] - t0, int(r[1]), int(r[2]) if len(r) > 2 else int(r[1])) for r in recs], 1.0)]
        else:
            scen = []
            for sc in build_scenarios(ar, tm, tp, cv, prompt_tokens=_pt):
                jobs = [(jb.arrival_s, jb.actual_tokens,
                         max(1, int((_pt or jb.actual_tokens) * sc["prompt_mult"])))
                        for jb in _synth_jobs(sc["arrival_rate"], sc["tm"], sc["tp"], sc["cv"],
                                              window_seconds=win, best_effort_fraction=be, kv_service_factor=1.0)]
                scen.append((sc["label"], jobs, sc["weight"]))
        g = v = wsum = vmax = 0.0
        base_out = None
        for j, (_lbl, jobs, wt) in enumerate(scen):
            mut = (j == 0)
            kvs = None
            if self.planning_kv_cost_mode:
                kvs = {"hash_seq": [(f"sc{k}_{j}_{i}",) for i in range(len(jobs))],
                       "routing": bundle_k.routing_policy, "capacity_blocks": self.planning_capacity_blocks,
                       "cost_mode": self.planning_kv_cost_mode}
            o = simulate_period(clone, bundle_k, jobs, fc_k, replay_kwargs=rkw, mutate=mut, kv_state=kvs, **common0)
            if mut:
                base_out = o
            sv = o.sla_violation_rate
            g += wt * o.goodput_per_dollar
            v += wt * sv
            wsum += wt
            vmax = max(vmax, sv)
        exp_gpd = g / wsum if wsum else 0.0
        exp_viol = v / wsum if wsum else 0.0
        tail_pen = self.scenario_tail_weight * vmax * exp_gpd if len(scen) > 1 else 0.0
        reward = exp_gpd - self.risk_weight * exp_viol * exp_gpd - tail_pen
        return base_out, exp_gpd, reward, exp_viol

    def _decode_regime_hint(self, tm) -> str | None:
        """The instantaneous decode roofline regime (memory-bandwidth-bound vs compute-bound) for the
        period — used by the adaptive planner to PRUNE the roofline action options to the regime where
        each can help. A hint for the SEARCH only; the reward is unaffected (a pruned candidate would
        still score through the physics)."""
        try:
            from .roofline import ServingConfig, Workload, roofline_regime
            gpu = (max(self.fleet_state.gpu_type_mix, key=self.fleet_state.gpu_type_mix.get)
                   if getattr(self.fleet_state, "gpu_type_mix", None) else "A100")
            pt = self.planning_prompt_tokens or 512
            wl = Workload(prompt_tokens=pt, decode_tokens=max(1, int(tm.mean)) if tm else 128,
                          context_len=pt + 320)
            return roofline_regime("decode", ServingConfig(gpu=gpu, batch_size=16), wl)["roofline_regime"]
        except Exception:
            return None

    def _planner_state(self, ar, tm, tp, cv, pr, confidence):
        """Build the cheap pre-search regime snapshot the physics-guided priors read (a SOFT prior — these
        signals decide which candidates are GENERATED, never the reward). Unknown signals stay None so the
        prior degrades gracefully and the anchors still cover the space."""
        from .physics_guided_candidates import PlannerRegimeState
        # capacity pressure proxy: forecast burstiness (p90/mean − 1), a cheap causal signal (no world read).
        cap_press = 0.0
        if ar is not None and ar.mean > 1e-9:
            cap_press = max(0.0, min(1.0, (ar.p90 / ar.mean) - 1.0))
        # Batch-1 regime signals — cheap causal proxies (forecast prompt/output only; no world/future read):
        prompt = float(self.planning_prompt_tokens or 512)
        out_mean = float(tm.mean) if (tm and tm.mean) else 128.0
        context = prompt + out_mean
        # HBM-pressure proxy: long context × burstiness press the KV budget (→ KV-precision regime gate).
        hbm_pressure = min(1.0, context / 4096.0) * (0.5 + 0.5 * cap_press)
        # prefill/decode skew: prefill GPU-seconds ≈ prompt·prefill_rate; decode ≈ out·decode_rate (decode
        # per-token ~2× prefill per-token in the legacy band). prefill-heavy when prompt dominates.
        prefill_work = prompt * 1.0
        decode_work = out_mean * 2.0
        prefill_heavy = None
        if prefill_work > 1.8 * decode_work:
            prefill_heavy = True
        elif decode_work > 1.8 * prefill_work:
            prefill_heavy = False
        # disaggregation candidate only when the phase mix is clearly skewed AND there is contention to relieve
        # (else the shared pool's multiplexing wins and the KV handoff is pure overhead — no free disaggregation).
        pd_divergence = (prefill_heavy is not None) and (cap_press >= 0.30)
        # PRODUCT-BOUNDARY default mask: core orchestration (gpu_assignment, auto-noop) is always allowed; the
        # OPTIONAL serving-engine integrations (kv_cache_precision, prefill_decode) are DEFAULT-OFF and only
        # enabled by explicit operator opt-in. An explicit allowed_new_knobs (ablation) overrides this default.
        if self.allowed_new_knobs is not None:
            allowed = self.allowed_new_knobs
        else:
            allowed = {"gpu_assignment_policy"}
            if self.enable_kv_cache_precision:
                allowed.add("kv_cache_precision_policy")
            if self.enable_prefill_decode_disagg:
                allowed.add("prefill_decode_policy")
            allowed = frozenset(allowed)
        return PlannerRegimeState(
            decode_regime=self._decode_regime_hint(tm), sla_slack=None,
            queue_pressure=0.0, capacity_pressure=cap_press,
            price_percentile=self.current_price_percentile,
            output_token_mean=(tm.mean if tm else None), hbm_pressure=round(hbm_pressure, 4),
            pd_divergence=pd_divergence, prefill_heavy=prefill_heavy,
            heterogeneous_fleet=False,   # production cost path is single-dominant-GPU → NOT_APPLICABLE
            allowed_new_knobs=allowed,
            confidence=confidence, prev_bundle=self._prev_best_bundle)

    def _planner_package_decide(self, mode, score_fn, ar, tm, tp, cv, pr, confidence):
        """Drive ONE decision with a PR #123 tournament method (run_method) over the controller's rollout
        `score_fn`. Returns `(best_bundle, MethodResult)`; the MethodResult carries the per-decision report
        (candidates generated/evaluated, node expansions, anchors present/evaluated, top-K, decision margin).
        For the *_with_progressive_widening variant it also runs a widening pass and keeps the better."""
        from .planner.candidate_generators import named_anchor_keys
        from .planner.search_methods import run_method
        pstate = self._planner_state(ar, tm, tp, cv, pr, confidence)
        nk = named_anchor_keys(self._prev_best_bundle)
        mres = run_method(_MODE_TO_METHOD[mode], score_fn, budget=self.planner_budget, state=pstate,
                          named_keys=nk, allow_quality_risk=False)
        if mode == "hierarchical_search_with_progressive_widening":
            w = run_method("progressive_widening", score_fn, budget=self.planner_budget, state=pstate,
                           named_keys=nk, allow_quality_risk=False)
            if w.best_reward > mres.best_reward:
                mres = w
        # decision margin from the top-K (relative gap); planner confidence = forecast confidence.
        tk = mres.top_k
        margin = ((tk[0]["reward"] - tk[1]["reward"]) / max(abs(tk[0]["reward"]), 1e-9)
                  if len(tk) > 1 else 1.0)
        mres.extra = {**mres.extra, "planner_mode": mode, "decision_margin": round(margin, 6),
                      "planner_confidence": round(confidence, 4)}
        return (mres.best_bundle or ActionBundle()), mres

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

        scored_for_diag = []                       # (bundle, score) capture for ONLINE diagnostics (no resolves)
        def _score(b):
            if self.decision_timeout_s > 0 and (_time.monotonic() - t_start[0]) > self.decision_timeout_s:
                timed_out[0] = True
                return -1e18                       # freeze the search; keep best-so-far
            s = _eval(b)[0]
            if self.emit_diagnostics and world is not None:
                scored_for_diag.append((b, s))     # cheap: just remembers the runners-up the search scored
            return s

        search_plan = None
        if self.candidates is not None:
            method, theoretical = "explicit", len(self.candidates)
            best_cand = max(self.candidates, key=_score)
        elif self.planner_mode in PACKAGE_PLANNER_MODES and world is not None:
            # a PR #123 tournament method (e.g. hierarchical_search) driving the per-decision rollout via the
            # planner package's run_method. The package imports NOTHING from production_baselines; it is the
            # Aurelius optimizer path, kept separate from the production_scheduler baseline.
            best_cand, search_plan = self._planner_package_decide(
                self.planner_mode, _score, ar, tm, tp, cv, pr, confidence)
            method, theoretical = search_plan.method, search_plan.candidates_generated
        elif self.physics_guided and world is not None:
            # physics-guided bounded-beam planner + progressive widening over an anchor-guaranteed candidate
            # set (the known-good bundles are always searched; no clock-only / full-space fallback).
            from .physics_guided_planner import BoundedBeamPlanner
            pstate = self._planner_state(ar, tm, tp, cv, pr, confidence)
            planner = self.physics_planner_obj or BoundedBeamPlanner()
            best_cand, search_plan = planner.plan(pstate, _score, prev_best=self._prev_best_bundle)
            method, theoretical = search_plan.strategy, search_plan.raw_candidates
        elif self.use_adaptive_search and world is not None:
            # adaptive planner over the roofline-pruned connected space (beam/CE + a regret audit that
            # MEASURES what an approximate search lost vs exhaustive — never a silent cap).
            from .search_planner import AdaptiveSearchPlanner, roofline_pruned_options
            dreg = self._decode_regime_hint(tm)
            surfaces = roofline_pruned_options(decode_regime=dreg,
                                               include_simulated=self.optimize_simulated)
            planner = self.search_planner_obj or AdaptiveSearchPlanner()
            best_cand, search_plan = planner.plan(_score, surfaces=surfaces, decode_regime=dreg)
            method, theoretical = search_plan.strategy, search_plan.raw_candidate_count
        else:
            gen = self.candidate_generator or CandidateBundleGenerator(
                include_simulated=self.optimize_simulated)
            theoretical = gen.theoretical_combinations()
            best_cand, _n, method = gen.search(_score, budget=budget)
        ab, act, _rkw, _routing = _resolve(best_cand)
        if ab is not None:
            self._prev_best_bundle = ab          # continuity anchor for the next decision's physics-guided set
        score, exp_gpd, risk_gpd, routing = _eval(best_cand)
        cumulative, win_steps = rollout_cache.get(id(ab), (score, []))
        self.last_decision_diag = {
            **clock.horizon_meta(H), "gamma": self.gamma, "uncertainty_mode": self.uncertainty_mode,
            "theoretical_bundles": theoretical, "candidate_bundles_evaluated": eval_count[0],
            "world_steps_simulated": eval_count[0] * H, "search_method": method,
            "runtime_s": round(_time.monotonic() - t_start[0], 4), "timed_out": timed_out[0],
            "cumulative_return": round(cumulative, 2), "rollout": win_steps,
            "search_plan": search_plan.to_dict() if search_plan is not None else None,
            "trajectory": traj.to_dict() if traj is not None else None} if world is not None else {}
        # ONLINE Decision Diagnostics (permanent, negligible overhead): expose ONLY values the search
        # already computed — no leave-one-out / oracle / extra solves (those are OFFLINE, in the benchmark
        # script). Never changes which action is chosen.
        if self.emit_diagnostics and world is not None:
            try:
                from .decision_diagnostics import explain_decision
                expl = explain_decision(
                    self._decision_count, ab, scored_for_diag, expected_gpd=exp_gpd,
                    expected_sla=(win_steps[0]["risk_viol"] if win_steps else 0.0), expected_cost=0.0,
                    expected_reward=score, forecast_confidence=confidence, n_evaluated=eval_count[0],
                    planning_horizon=H, forecast_horizon=H, search_strategy=method,
                    planning_latency_s=(_time.monotonic() - t_start[0]),
                    forecast_snapshot={"arrival_rate": round(ar.mean, 3),
                                       "output_token_mean": round(tm.mean, 1), "electricity_price": pr.value})
                if win_steps:
                    expl.reward_decomposition = {k: win_steps[0].get(k) for k in
                        ("gp_per_dollar", "risk_viol", "warm_hold_cost", "migration_cost",
                         "cold_start_events", "peak_c", "topology_factor", "sla_slack_ms") if k in win_steps[0]}
                self.last_decision_diag["diagnostics"] = expl.to_dict()
            except Exception:                      # diagnostics must NEVER break a decision
                pass
            self._decision_count += 1
        # Phase 8 — electricity visibility: every decision records the price it saw, the clock it picked, and
        # whether it was price-aware (so attribution can see whether electricity drove the clock choice).
        electricity_diag = {"forecast_price_per_kwh": round(pr.value, 6) if pr else None,
                            # legacy dict candidates resolve to no ActionBundle (ab=None) → no clock action (base)
                            "selected_clock": (ab.clock_policy if ab is not None else "base"),
                            "price_aware": self.electricity_price_aware,
                            # N2 SLA-slack arbitrage diagnostic: how much SLA headroom the chosen clock leaves
                            # (online serving — never time-shifted). Deferrable shifting is a SEPARATE ledger.
                            "sla_slack_ms": win_steps[0].get("sla_slack_ms") if win_steps else None,
                            "serving_time_shifted": False,   # N2 invariant: online serving is never delayed
                            "deferrable_shifted": False,     # the online controller runs no deferrable work
                            "why": ("price-aware: clock chosen against the forecast price path"
                                    if self.electricity_price_aware else
                                    "not price-aware: clock chosen on roofline/SLA only (constant price)")}
        # ForecastState (opt-in, canonical planner belief): record WHAT the planner believed about the next
        # period, BEFORE it runs. Realized + error are filled later by run_period_episode (causal — no leakage).
        # Belief-only; never a reward term. No-op unless a ForecastState is attached.
        fs = getattr(self, "forecast_state", None)
        if fs is not None:
            p_now = len(history)
            fs.horizon_steps = H
            fs.record_belief(decision_index=fs.n_decisions, target_period=p_now, made_at_period=p_now,
                             horizon_index=0, confidence=confidence, provenance="FORECAST_DERIVED",
                             belief={"arrival_rate": ar.mean, "interarrival_cv": cv.mean,
                                     "output_token_mean": tm.mean, "output_token_p95": tp.value if tp else 0.0,
                                     "electricity_price": pr.value if pr else 0.0},
                             uncertainty={"arrival_rate": {"p10": ar.p10, "p90": ar.p90}})
            fs.n_decisions += 1
        return Decision(act, exp_gpd, risk_gpd, score, False, confidence,
                        forecast={"arrival_rate": ar.to_dict(), "price": pr.value,
                                  "routing_policy": routing, "electricity": electricity_diag}, bundle=ab)

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
    mean_kv_prefix_hit_rate: float = 0.0                # per-replica KV residency (PR #106)
    prefill_tokens_saved: int = 0                       # prefill skipped by prefix hits
    model_switch_events: int = 0                        # model-load cold-starts incurred (affinity)
    realized_gpu_seconds: float = 0.0                   # PR #107 phase economics
    mean_ttft_p95: float = 0.0                          # mean per-period TTFT p95 (service-only)
    prefill_tokens_remaining: int = 0                   # prefill tokens NOT saved (paid)
    # roofline action mixes + diagnostics (PR roofline-economic MPC actions)
    precision_mix: dict = field(default_factory=dict)   # precision_policy → periods chosen
    kv_cache_precision_mix: dict = field(default_factory=dict)  # kv_cache_precision_policy → periods (Batch-1)
    spec_decode_mix: dict = field(default_factory=dict)  # spec_decode_policy → periods
    clock_mix: dict = field(default_factory=dict)       # clock_policy → periods
    colocation_mix: dict = field(default_factory=dict)  # colocation_policy → periods (SIMULATED, pruned off)
    prefill_decode_mix: dict = field(default_factory=dict)  # prefill_decode_policy → periods (SIMULATED)
    decode_regime_mix: dict = field(default_factory=dict)   # roofline decode regime → periods
    mean_decode_arithmetic_intensity: float = 0.0       # mean decode FLOP/byte (roofline x-axis)
    mean_ridge_point: float = 0.0                       # mean GPU ridge point (peak_flops/mem_bw)
    mean_power_w: float = 0.0                           # mean GPU power under the clock action (diagnostic)
    total_energy_j: float = 0.0                         # serving energy under the actions (diagnostic)
    quality_sla_risk_mean: float = 0.0                  # mean precision quality-failure fraction (int4)

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def run_period_episode(name, decide_fn, real_per_period, frames, eval_indices, *,
                       fleet_state, cost_model, sla_s=10.0, tick_seconds=60.0,
                       period_seconds=60.0, kv_service_factor=1.0, cost_scenario="owned",
                       sim_seconds=None, kv_service_factor_by_routing=None, world_state=None,
                       world_state_params=None, kv_state_pool=None, kv_capacity_blocks=512,
                       kv_model_seq=None, kv_cost_mode=None, electricity_prices=None, forecast_state=None):
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
    precision_mix: dict = {}
    kv_precision_mix: dict = {}
    spec_mix: dict = {}
    clock_mix: dict = {}
    coloc_mix: dict = {}
    pd_mix: dict = {}
    rl_regime_mix: dict = {}
    rl_ai_sum = rl_ridge_sum = rl_power_sum = rl_energy_sum = rl_q_sum = rl_n = 0.0
    cold_starts = mig_cost = warm_hold = topo_sum = topo_n = 0.0
    factor_sum, factor_n = 0.0, 0
    kv_cursor = 0                                       # cumulative request index into the KV prefix pool
    kv_hit_sum = kv_saved_sum = kv_switch_sum = 0.0
    kv_realized_gpu_s = kv_ttft_sum = kv_prefill_rem = 0.0   # PR #107 phase economics
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
                       "prewarm_policy", "placement_policy", "migration_policy",
                       "precision_policy", "spec_decode_policy", "clock_policy",
                       "colocation_policy", "prefill_decode_policy"):
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
        _pc = merged.get("precision_policy", "bf16")
        _kvp = merged.get("kv_cache_precision_policy", "inherit_weight_precision")
        _sd = merged.get("spec_decode_policy", "off")
        _ck = merged.get("clock_policy", "base")
        _co = merged.get("colocation_policy", "off")
        _pd = merged.get("prefill_decode_policy", "shared")
        precision_mix[_pc] = precision_mix.get(_pc, 0) + 1
        kv_precision_mix[_kvp] = kv_precision_mix.get(_kvp, 0) + 1
        spec_mix[_sd] = spec_mix.get(_sd, 0) + 1
        clock_mix[_ck] = clock_mix.get(_ck, 0) + 1
        coloc_mix[_co] = coloc_mix.get(_co, 0) + 1
        pd_mix[_pd] = pd_mix.get(_pd, 0) + 1
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
            pol = SimpleNamespace(prewarm_policy=_pw, placement_policy=_pl, migration_policy=_mg,
                                  batching_policy=_bp, precision_policy=_pc, spec_decode_policy=_sd,
                                  clock_policy=_ck, colocation_policy=_co, prefill_decode_policy=_pd)
            t0 = recs[0][0]
            # per-replica KV/model residency (PR #106): assign this period's requests a causal slice
            # of the Mooncake-derived prefix pool (TRACE_DERIVED_REUSE_MODEL); routing over the warm
            # replicas' persistent caches then sets per-request service time. Cursor advances so reuse
            # distance is preserved across periods (the cache persists; conversations recur).
            kv_state = None
            if kv_state_pool:
                m = len(kv_state_pool)
                hseq = [kv_state_pool[(kv_cursor + i) % m] for i in range(len(recs))]
                mseq = ([kv_model_seq[(kv_cursor + i) % len(kv_model_seq)] for i in range(len(recs))]
                        if kv_model_seq else None)
                kv_cursor += len(recs)
                kv_state = {"hash_seq": hseq, "routing": routing, "model_seq": mseq,
                            "capacity_blocks": kv_capacity_blocks, "cost_mode": kv_cost_mode}
            oc = _sim_period(world_state, pol, [(r[0] - t0, int(r[1]), r[2] if len(r) > 2 else r[1])
                                                for r in recs], fcast, sla_s=sla_s,
                             tick_seconds=tick_seconds, base_service_factor=factor,
                             replay_kwargs=replay_kw, cost_model=cost_model, fleet_state=fleet_state,
                             cost_scenario=cost_scenario, best_effort_fraction=be,
                             period_hours=period_hours, dt_seconds=period_seconds,
                             kv_state=kv_state,
                             energy_price_per_kwh=(electricity_prices.get(p) if electricity_prices else None),
                             mutate=True)
            kpi = oc.kpi
            # ForecastState (opt-in): record the REALIZED outcome for period p + the forecast error. Causal —
            # this runs AFTER the period is simulated, so error is never computed from the future. No-op
            # unless a ForecastState is attached. Belief-vs-realized only; never a reward term.
            if forecast_state is not None and recs:
                # Record realized ONLY for the variables whose unit is unambiguously comparable to the belief:
                # output-token mean/p95 (both are token counts of the same period's requests). Arrival rate is
                # NOT recorded here because the backtest caps requests/period (the realized count is the capped
                # count, not the true rate) and electricity_price is arm-dependent (flat vs forecast) — recording
                # either would fabricate a misleading "forecast error". Output length is the dominant regret
                # driver (PR #118), so it is the meaningful signal anyway.
                _toks = sorted(int(r[1]) for r in recs)
                _p95 = _toks[min(len(_toks) - 1, int(len(_toks) * 0.95))] if _toks else 0.0
                forecast_state.record_realized(p, {
                    "output_token_mean": (sum(_toks) / len(_toks)) if _toks else 0.0,
                    "output_token_p95": float(_p95)}, at_period=p)
            tot_cost += oc.operator_cost
            cold_starts += oc.cold_start_events
            mig_cost += oc.migration_cost
            warm_hold += oc.wasted_prewarm_hours
            topo_sum += oc.topology_factor
            topo_n += 1
            if oc.queue_delay_p95 > 0:
                waits_p95.append(oc.queue_delay_p95)
                waits_p99.append(oc.queue_delay_p99)
            if oc.kv_diag:
                kv_hit_sum += oc.kv_diag.get("exact_prefix_hit_rate", 0.0)
                kv_saved_sum += oc.kv_diag.get("prefill_tokens_saved", 0)
                kv_switch_sum += oc.kv_diag.get("model_switch_events", 0)
                kv_realized_gpu_s += oc.kv_diag.get("realized_gpu_seconds", 0.0)
                kv_ttft_sum += oc.kv_diag.get("ttft_p95", 0.0)
                kv_prefill_rem += oc.kv_diag.get("prefill_tokens_remaining", 0)
            _q = float(getattr(oc, "quality_sla_risk", 0.0))
            tot_g += kpi.sla_safe_goodput * (1.0 - _q)        # quality failures are not sla-safe goodput
            tot_gpu_h += kpi.gpu_hours
            tot_viol += kpi.sla_violations + _q * kpi.n_total  # …and count as SLA failures (int4 risk)
            tot_n += kpi.n_total
            tot_safe += kpi.n_sla_safe
            if oc.roofline_diag:
                rd = oc.roofline_diag
                _reg = rd.get("decode_regime", "?")
                rl_regime_mix[_reg] = rl_regime_mix.get(_reg, 0) + 1
                rl_ai_sum += rd.get("decode_arithmetic_intensity", 0.0)
                rl_ridge_sum += rd.get("ridge_point", 0.0)
                rl_power_sum += oc.power_w
                rl_energy_sum += oc.energy_j
                rl_q_sum += _q
                rl_n += 1
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
        mean_topology_factor=(topo_sum / topo_n if topo_n else 1.0),
        mean_kv_prefix_hit_rate=(kv_hit_sum / topo_n if topo_n else 0.0),
        prefill_tokens_saved=int(kv_saved_sum), model_switch_events=int(kv_switch_sum),
        realized_gpu_seconds=round(kv_realized_gpu_s, 2),
        mean_ttft_p95=round(kv_ttft_sum / topo_n, 4) if topo_n else 0.0,
        prefill_tokens_remaining=int(kv_prefill_rem),
        precision_mix=precision_mix, kv_cache_precision_mix=kv_precision_mix,
        spec_decode_mix=spec_mix, clock_mix=clock_mix,
        colocation_mix=coloc_mix, prefill_decode_mix=pd_mix, decode_regime_mix=rl_regime_mix,
        mean_decode_arithmetic_intensity=round(rl_ai_sum / rl_n, 4) if rl_n else 0.0,
        mean_ridge_point=round(rl_ridge_sum / rl_n, 2) if rl_n else 0.0,
        mean_power_w=round(rl_power_sum / rl_n, 1) if rl_n else 0.0,
        total_energy_j=round(rl_energy_sum, 1),
        quality_sla_risk_mean=round(rl_q_sum / rl_n, 4) if rl_n else 0.0)


__all__ = [
    "CAPACITY", "ORDERING", "ADMISSION", "SLA_AWARE_FALLBACK", "enumerate_actions",
    "Decision", "ModelPredictiveEconomicController", "EpisodeReport", "run_period_episode",
]
