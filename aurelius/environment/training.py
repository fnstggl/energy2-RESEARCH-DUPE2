"""Train forecasters + tune the MPC controller, with a strict claim gate.

Three disjoint time splits (no leakage):
  * **train**  — forecaster models are fit here only.
  * **val**    — controller hyper-parameters (horizon / risk weight / confidence floor)
                 are selected here (the forecasters still come from train).
  * **eval**   — held-out; the final controller-vs-baselines comparison + claim gate.

The first trainable component is **controller tuning**, not deep RL. A headline claim
is allowed ONLY if the controller beats the strongest NON-weak baseline on the held-out
eval, the splits are disjoint, and no oracle/future information is used (the controller
is causal by construction). If it does not beat the baseline, that is reported honestly.
"""

from __future__ import annotations

import itertools

from .controller import (
    SLA_AWARE_FALLBACK,
    ModelPredictiveEconomicController,
    run_period_episode,
)
from .forecasting import ForecastingModel

DEFAULT_GRID = {"horizon": [1, 2, 4], "risk_weight": [0.0, 0.5, 1.0], "confidence_min": [0.1, 0.3]}
WEAK_BASELINES = frozenset({"fifo_weak"})
DEFAULT_BASELINES = {
    "fifo_weak": {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off"},
    "sla_aware": {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off"},
    "greedy_packing": {"capacity": "forecasted_mcs", "ordering": "fifo", "admission": "off"},
    "aurelius_canonical": {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "class_aware"},
    # routing baselines (CONNECTED): the same policies but WITH best (kv-aware) routing, so the
    # MPC must beat a strong routing-enabled baseline, not merely "discover" routing. Round-robin
    # is the implicit routing of the rows above (no routing_policy key → round_robin factor).
    "sla_aware_kv_routing": {"capacity": "backlog_aware", "ordering": "abs_conformal",
                             "admission": "off", "routing_policy": "kv_aware"},
    "aurelius_canonical_kv_routing": {"capacity": "backlog_aware", "ordering": "abs_conformal",
                                      "admission": "class_aware", "routing_policy": "kv_aware"},
    # next-batch action-specific baselines (CONNECTED capacity_multiplier + batching). Batching's
    # concurrency lever is a near-free throughput win, so a COMPETENT static operator simply turns
    # it on — these baselines do, raising the bar: the MPC must beat an operator who already batches
    # (and routes KV-aware), i.e. win by per-period ADAPTATION, not by "discovering" a fixed knob.
    "sla_aware_batched": {"capacity": "backlog_aware", "ordering": "abs_conformal",
                          "admission": "off", "routing_policy": "kv_aware",
                          "batching_policy": "balanced"},
    "aurelius_static_full": {"capacity": "backlog_aware", "ordering": "abs_conformal",
                             "admission": "class_aware", "routing_policy": "kv_aware",
                             "batching_policy": "balanced"},
    # a fixed OVER-PROVISIONED operator (1.5x replicas): buys SLA headroom but pays it in gp/$ —
    # documents the capacity Pareto point so the MPC's adaptive capacity is judged against a fixed one.
    "sla_aware_capacity_1p5": {"capacity": "backlog_aware", "ordering": "abs_conformal",
                               "admission": "off", "routing_policy": "kv_aware",
                               "capacity_multiplier": 1.5},
    # stateful-action baselines (only bite on the --world-state path; harmless no-ops otherwise).
    # A competent static operator already places topology-aware → the MPC must beat THAT, not merely
    # "discover" placement. always-prewarm is the fair upper bound on prewarming (and shows its cost).
    "world_static_best": {"capacity": "backlog_aware", "ordering": "abs_conformal",
                          "admission": "class_aware", "routing_policy": "kv_aware",
                          "batching_policy": "balanced", "placement_policy": "network_aware"},
    "prewarm_always": {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off",
                       "routing_policy": "kv_aware", "batching_policy": "balanced",
                       "placement_policy": "network_aware", "prewarm_policy": "aggressive"},
}


def build_mpc_inputs(*, limit: int = 8000, bin_seconds: float = 60.0,
                     processed_dir: str | None = None, hourly_stride: int = 24,
                     sim_seconds: float = 240.0, use_world_state: bool = False,
                     control_dt_seconds: float | None = None) -> dict | None:
    """Build the (frames, per-period real trace, fleet state, cost model, common) inputs
    from the canonical sources.

    When the **2024 one-week** Azure trace is present, bin it over the full week via the
    bounded-memory streaming binner. The control interval defaults to HOURLY
    (``control_dt_seconds=None`` → 3600s, ``cycle_len=24`` diurnal, 168 real periods — see
    ``research/AZURE_TRACE_COVERAGE_AUDIT.md``); pass ``control_dt_seconds`` (e.g. 60 / 300 /
    900) to re-bin the SAME week at a SUB-HOUR control interval so ``period_seconds`` is the
    real control step (``cycle_len = 86400/dt`` keeps the diurnal cycle). The per-period load
    is a deterministic 1/``hourly_stride`` sample (proportional, so the diurnal shape is
    preserved and forecast + replay share one scale); since the sample stride is global the
    arrival RATE is dt-invariant, so a finer ``dt`` changes only the control granularity, not
    the load intensity. The controller scores actions over a bounded ``sim_seconds`` window
    (clamped to the period). When only the 2023 one-hour trace / sample is present, fall back
    to the original sub-hour (``bin_seconds``) per-minute binning so CI and the 1-hour regime
    are unchanged."""
    from collections import defaultdict

    from .cost_model import CostModel
    from .fleet_plane_v2026 import V2026FleetPlane
    from .forecasting import build_frames
    from .ingestion.azure import azure_period_frames, context_tokens, ingest_azure, to_serving_raw
    from .ingestion.mooncake import ingest_mooncake
    from .kv_cache import KVModel, gpu_mem_for, routing_service_factors

    week_dt = float(control_dt_seconds) if control_dt_seconds else 3600.0
    pf = azure_period_frames(bin_seconds=week_dt, sample_stride=hourly_stride)
    coverage = None
    week_path = pf is not None and "1week" in pf["trace_version"]
    if week_path:
        per = pf["per_period"]
        period_seconds = pf["bin_seconds"]
        cycle_len = max(1, round(86400.0 / period_seconds))     # diurnal cycle in control steps
        coverage = {"trace_version": pf["trace_version"], "tier": pf["tier"],
                    "n_bins_exact": pf["n_bins"], "total_requests_exact": pf["total_requests"],
                    "sample_stride": pf["sample_stride"],
                    "granularity": ("hourly" if period_seconds >= 3600.0 else "sub_hourly"),
                    "control_dt_seconds": period_seconds, "cycle_len": cycle_len}
    else:                                          # 2023 one-hour / sample → per-minute
        reqs, _ = ingest_azure(limit=limit)
        out, inp = to_serving_raw(reqs), context_tokens(reqs)
        if not out:
            return None
        t0 = out[0][0]
        per_dd: dict = defaultdict(list)
        for (a, ot), it in zip(out, inp):
            per_dd[int((a - t0) // bin_seconds)].append((a, ot, it))
        per = dict(per_dd)
        period_seconds, cycle_len = bin_seconds, 60
        coverage = {"trace_version": "AzureLLMInferenceTrace2023/1hour-or-sample",
                    "granularity": "per_minute", "n_periods": len(per)}

    fleet = V2026FleetPlane(processed_dir=processed_dir).state_at(0)
    gpu_type = max(fleet.gpu_type_mix, key=fleet.gpu_type_mix.get) if fleet.gpu_type_mix else "H100"
    mreqs, _ = ingest_mooncake()
    mtrain = mreqs[: int(len(mreqs) * 0.7)] or mreqs
    kv = KVModel.fit(mtrain, gpu_mem_gib=gpu_mem_for(gpu_type), mem_pressure=fleet.mem_pressure)
    # routing → KV economics (CONNECTED action): replay the Mooncake reuse trace across the
    # fleet under each routing policy → routing-specific service factor. Causal; the held-out
    # validation (kv_aware reuses more prefix than round_robin) lives in tests.
    n_servers = max(1, min(int(getattr(fleet, "capacity_envelope", 4) or 4), 4))
    rmap = routing_service_factors(mtrain, n_servers=n_servers,
                                   capacity_blocks=max(64, kv.capacity_blocks // n_servers),
                                   block_tokens=kv.block_tokens,
                                   prefill_savings_frac=kv.prefill_savings_frac)
    kv_by_routing = {p: r["service_factor"] for p, r in rmap.items()}
    kv_factor = kv_by_routing.get("round_robin", float(kv.stats(1000).get("mean_ttft_factor", 1.0)))
    anchors = {"gpu_utilization": fleet.util_target, "gpu_memory_pressure": fleet.mem_pressure,
               "network_pressure": fleet.net_pressure, "kv_reuse": kv.warm_hit_rate()}
    frames = build_frames(per, period_seconds=period_seconds, cycle_len=cycle_len,
                          price_by_cycle={c: fleet.energy_price_per_kwh for c in range(cycle_len)},
                          anchors=anchors)
    common = {"sla_s": 10.0, "period_seconds": period_seconds, "tick_seconds": 10.0,
              "kv_service_factor": kv_factor, "cost_scenario": "owned",
              "sim_seconds": (min(sim_seconds, period_seconds) if week_path else None),
              "kv_service_factor_by_routing": kv_by_routing}
    if use_world_state:
        # persistent world: a TRACE_DERIVED_SAMPLE cluster sampled from the SAME v2026 processed
        # marginals the fleet is calibrated to. Fresh per arm/config (isolated timelines).
        common["world_state_params"] = {"n_servers": 24, "n_racks": 4, "seed": 0, "warm": 8,
                                        "processed_dir": processed_dir}
    return {"frames": frames, "per": per, "fleet_state": fleet, "cost_model": CostModel(),
            "common": common, "coverage": coverage, "kv_routing": rmap}


def split_cuts(n: int, train: float = 0.5, val: float = 0.25) -> tuple:
    t1 = max(4, int(n * train))
    t2 = max(t1 + 2, int(n * (train + val)))
    t2 = min(t2, n - 1)
    return t1, t2


def train_forecasters(frames: list, train_cut: int, *, train_frac: float = 0.7) -> tuple:
    """Fit the forecaster ladder on the TRAIN periods only; return (model, report)."""
    fm = ForecastingModel().fit(frames[:train_cut], train_frac=train_frac)
    return fm, fm.report()


def make_world_state(params: dict | None):
    """Build a fresh, warm-seeded persistent world state from ``params`` (or None to disable).
    A fresh state per arm/config keeps the stateful timelines isolated; the seed is fixed so the
    sampled cluster is reproducible."""
    if not params:
        return None
    from .world_simulator import initialize_world_state, warm_seed
    ws = initialize_world_state(n_servers=params.get("n_servers", 24),
                                n_racks=params.get("n_racks", 4), seed=params.get("seed", 0),
                                processed_dir=params.get("processed_dir"))
    warm_seed(ws, params.get("warm", 16))
    return ws


def _controller(fm, fleet_state, cost_model, cfg, common, world_state=None):
    return ModelPredictiveEconomicController(
        forecasters=fm, fleet_state=fleet_state, cost_model=cost_model,
        horizon=cfg["horizon"], risk_weight=cfg["risk_weight"],
        confidence_min=cfg["confidence_min"], sla_s=common["sla_s"],
        period_seconds=common["period_seconds"], tick_seconds=common["tick_seconds"],
        kv_service_factor=common.get("kv_service_factor", 1.0),
        kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
        cost_scenario=common.get("cost_scenario", "owned"),
        sim_seconds=common.get("sim_seconds"), world_state=world_state)


def tune_controller(fm, frames, per, val_idx, *, fleet_state, cost_model,
                    grid=None, common=None) -> tuple:
    """Select the controller config maximizing held-out-of-train (val) gp/$."""
    grid = grid or DEFAULT_GRID
    common = common or {}
    keys = list(grid)
    wsp = common.get("world_state_params")
    best_cfg, best_gpd, results = None, -1.0, []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = dict(zip(keys, combo))
        ws = make_world_state(wsp)                 # fresh persistent state per config (isolated)
        ctrl = _controller(fm, fleet_state, cost_model, cfg, common, world_state=ws)
        rep = run_period_episode("mpc", lambda h: ctrl.decide(h).to_dict(), per, frames, val_idx,
                                 fleet_state=fleet_state, cost_model=cost_model,
                                 world_state=ws, **common)
        results.append({"cfg": cfg, "val_gpd": round(rep.goodput_per_dollar, 2)})
        if rep.goodput_per_dollar > best_gpd:
            best_cfg, best_gpd = cfg, rep.goodput_per_dollar
    return best_cfg, results


def train_mpc_policy(frames, per, *, fleet_state, cost_model, train=0.5, val=0.25,
                     grid=None, common=None) -> tuple:
    """Fit forecasters (train) + tune the controller (val). Returns (trained, model)."""
    common = common or {"sla_s": 10.0, "period_seconds": 60.0, "tick_seconds": 10.0}
    t1, t2 = split_cuts(len(frames), train, val)
    val_idx = list(range(t1, t2))
    fm, fcast = train_forecasters(frames, t1)
    cfg, val_results = tune_controller(fm, frames, per, val_idx, fleet_state=fleet_state,
                                       cost_model=cost_model, grid=grid, common=common)
    trained = {"forecaster_report": fcast, "controller_config": cfg,
               "splits": {"train_cut": t1, "val": [t1, t2], "eval": [t2, len(frames)]},
               "val_results": val_results, "common": common}
    return trained, fm


def claim_gate(arms: dict, *, weak=WEAK_BASELINES) -> dict:
    """Honest gate: a headline requires beating the strongest NON-weak baseline on
    SLA-safe goodput/$ **without** simply trading away SLA compliance to do it.

    The Pareto clause matters: a controller can raise goodput/$ purely by under-provisioning
    (lower cost) while letting more requests miss the SLA. That is a *cheaper* policy, not a
    *better* one — so a headline also requires the candidate's SLA-violation rate to be no
    worse than the fair baseline's. (On the full Azure week the MPC controller's gp/$ edge is
    small, regime-dependent, and always bought with a higher violation rate → this clause
    keeps the gate honestly False. See research/AURELIUS_FORECASTING_AND_MPC_CONTROLLER.md.)"""
    fair = {n: r for n, r in arms.items() if n != "mpc_controller" and n not in weak}
    fair_baseline = max(fair, key=lambda n: fair[n].goodput_per_dollar) if fair else None
    mpc_arm = arms["mpc_controller"]
    base_arm = arms[fair_baseline] if fair_baseline else None
    mpc = mpc_arm.goodput_per_dollar
    base = base_arm.goodput_per_dollar if base_arm else 0.0
    delta = 100.0 * (mpc - base) / base if base else 0.0
    sla_not_worse = base_arm is not None and mpc_arm.sla_violation_rate <= base_arm.sla_violation_rate + 1e-9
    gate = {
        "fair_baseline": fair_baseline,
        "fair_baseline_not_weak": fair_baseline is not None and fair_baseline not in weak,
        "beats_fair_baseline": delta > 0,
        "pareto_sla_not_worse": sla_not_worse,
        "mpc_sla_violation_rate": round(mpc_arm.sla_violation_rate, 4),
        "fair_sla_violation_rate": round(base_arm.sla_violation_rate, 4) if base_arm else None,
        "no_oracle": True,                 # controller is causal by construction
        "splits_disjoint": True,           # train < val < eval by construction
        "candidate_vs_baseline_pct": round(delta, 3),
        "note": "SIMULATED (directional simulator evidence), not production telemetry",
    }
    gate["headline_claim_allowed"] = (gate["fair_baseline_not_weak"] and gate["beats_fair_baseline"]
                                      and gate["pareto_sla_not_worse"] and gate["no_oracle"]
                                      and gate["splits_disjoint"])
    return gate


def evaluate_mpc(trained, fm, frames, per, *, fleet_state, cost_model,
                 baselines=None, common=None) -> dict:
    """Held-out evaluation of the tuned controller vs baselines + the claim gate."""
    baselines = baselines or DEFAULT_BASELINES
    common = common or trained.get("common", {"sla_s": 10.0, "period_seconds": 60.0, "tick_seconds": 10.0})
    e0, e1 = trained["splits"]["eval"]
    eval_idx = list(range(e0, e1))
    wsp = common.get("world_state_params")
    mpc_ws = make_world_state(wsp)                  # mpc arm: controller + episode SHARE one state
    ctrl = _controller(fm, fleet_state, cost_model, trained["controller_config"], common,
                       world_state=mpc_ws)
    arms = {"mpc_controller": run_period_episode(
        "mpc_controller", lambda h: ctrl.decide(h).to_dict(), per, frames, eval_idx,
        fleet_state=fleet_state, cost_model=cost_model, world_state=mpc_ws, **common)}
    for name, action in baselines.items():
        arms[name] = run_period_episode(            # each baseline gets its own fresh state
            name, (lambda act: (lambda h: dict(act)))(action), per, frames, eval_idx,
            fleet_state=fleet_state, cost_model=cost_model, world_state=make_world_state(wsp), **common)
    gate = claim_gate(arms)
    return {"eval_periods": len(eval_idx), "controller_config": trained["controller_config"],
            "arms": {n: r.to_dict() for n, r in arms.items()}, "gate": gate,
            "splits": trained["splits"]}


__all__ = [
    "DEFAULT_GRID", "DEFAULT_BASELINES", "WEAK_BASELINES", "SLA_AWARE_FALLBACK",
    "build_mpc_inputs", "split_cuts", "train_forecasters", "tune_controller",
    "train_mpc_policy", "claim_gate", "evaluate_mpc", "make_world_state",
]
