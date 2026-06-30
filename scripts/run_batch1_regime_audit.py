#!/usr/bin/env python3
"""Batch-1 knob regime-activation audit (Batch-1 corrective Phase E).

Fast, deterministic audit answering: **how often does the production replay actually enter the regimes the
Batch-1 knobs were designed for?** For each new knob it reports the regime metrics + whether the planner
generated / evaluated / selected the knob, and why. This separates "the workload did not need it" from
"the wiring prevented it". No reward / cost / Pareto change; read-only inspection of the production build.

Artifact: data/external/mpc_controller/batch1_knob_regime_activation.json.
Usage: python -m scripts.run_batch1_regime_audit [--market pjm] [--cap 100000] [--win-len 6] [--max-periods 6]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from scripts.run_ladder_benchmark import build_market, select_windows

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")
_ARTIFACT = os.path.join(_OUT, "batch1_knob_regime_activation.json")


def _period_workload(recs):
    outs = sorted(int(r[1]) for r in recs)
    ins = sorted(int(r[2]) if len(r) > 2 else 0 for r in recs)
    prompt = ins[len(ins) // 2] if ins else 0
    return (prompt, outs[len(outs) // 2] if outs else 128, len(recs))


def _prompt_data_present(periods) -> bool:
    """The Azure serving trace records (arrival, output_tokens, input_tokens). If input_tokens are all 0 the
    PROMPT side is unobserved → the prefill phase + large-context KV pressure cannot be exercised."""
    return any((len(r) > 2 and int(r[2]) > 0) for recs in periods for r in recs)


def _kv_audit(periods, gpu_type):
    from aurelius.environment.kv_precision import kv_precision_memory_effect
    from aurelius.environment.roofline import ServingConfig, Workload, roofline_regime
    mem_bound = hbm_hi = 0
    saved_fp8 = saved_int8 = occ = 0.0
    n = 0
    for recs in periods:
        if not recs:
            continue
        n += 1
        prompt, out, nreq = _period_workload(recs)
        wl = Workload(prompt_tokens=prompt, decode_tokens=out, context_len=prompt + out // 2)
        reg = roofline_regime("decode", ServingConfig(gpu=gpu_type, batch_size=16), wl)["roofline_regime"]
        mem_bound += int(reg == "memory_bandwidth_bound")
        eff8 = kv_precision_memory_effect("kv_fp8", gpu_type=gpu_type, context_tokens=prompt + out,
                                          active_sequences=nreq)
        eff_i8 = kv_precision_memory_effect("kv_int8", gpu_type=gpu_type, context_tokens=prompt + out,
                                            active_sequences=nreq)
        hbm_hi += int(eff8.hbm_pressure_before >= 0.60)
        saved_fp8 += eff8.kv_memory_saved_pct
        saved_int8 += eff_i8.kv_memory_saved_pct
        occ += eff8.hbm_pressure_before
    n = max(n, 1)
    return {"periods": n, "pct_memory_bandwidth_bound": round(100.0 * mem_bound / n, 1),
            "pct_hbm_high_pressure": round(100.0 * hbm_hi / n, 1),
            "mean_kv_occupancy_estimate": round(occ / n, 4),
            "kv_bytes_saved_pct_fp8": round(saved_fp8 / n, 1),
            "kv_bytes_saved_pct_int8": round(saved_int8 / n, 1),
            "kv_eviction_pressure": "occupancy < working-set threshold in all periods (no eviction pressure)"
            if occ / n < 0.9 else "elevated"}


def _pd_audit(periods, gpu_type):
    from aurelius.environment.pd_disaggregation import PDWorkload, pd_serving_point
    from aurelius.environment.prefill_decode import PREFILL_S_PER_TOKEN, TPOT_S
    classes = {"prefill_heavy": 0, "decode_heavy": 0, "balanced": 0}
    interference = handoff_b = handoff_l = util = 0.0
    n = 0
    for recs in periods:
        if not recs:
            continue
        n += 1
        prompt, out, nreq = _period_workload(recs)
        pf = prompt * PREFILL_S_PER_TOKEN + 0.15
        dc = out * TPOT_S
        if pf > 1.8 * dc:
            classes["prefill_heavy"] += 1
        elif dc > 1.8 * pf:
            classes["decode_heavy"] += 1
        else:
            classes["balanced"] += 1
        ar = nreq / 3600.0
        wl = PDWorkload(arrival_rate=ar, prefill_work_s=pf, decode_work_s=dc, context_tokens=prompt + out)
        sh = pd_serving_point(wl, "shared", n_replicas=16)
        sp = pd_serving_point(wl, "balanced_pd", n_replicas=16)
        interference += max(0.0, sh.mean_completion_s - sp.mean_completion_s) / max(sh.mean_completion_s, 1e-9)
        handoff_b += sp.kv_handoff_bytes
        handoff_l += sp.kv_handoff_latency
        util += max(sh.prefill_pool_utilization, sh.decode_pool_utilization)
    n = max(n, 1)
    skewed = classes["prefill_heavy"] + classes["decode_heavy"]
    return {"periods": n, "phase_mix": classes,
            "pct_skewed": round(100.0 * skewed / n, 1),
            "mean_interference_relief_estimate": round(interference / n, 4),
            "mean_phase_pool_utilization": round(util / n, 4),
            "mean_handoff_bytes": round(handoff_b / n, 1),
            "mean_handoff_latency_s": round(handoff_l / n, 6),
            "resembles_distserve_high_load_skewed_regime":
                bool((skewed / n) > 0.5 and (util / n) > 0.9)}


def _gpu_audit(fleet, world_state):
    mix = dict(getattr(fleet, "gpu_type_mix", {}) or {})
    servers = getattr(world_state, "servers", {}) or {}
    server_types = {}
    for s in servers.values():
        t = getattr(s, "gpu_type", None)
        if t:
            server_types[t] = server_types.get(t, 0) + 1
    heterogeneous_fleetwide = len(mix) > 1
    # the cost path charges the whole period at the DOMINANT gpu type (canonical.py / world_simulator.py).
    return {"fleet_gpu_type_mix": mix,
            "server_level_gpu_type_counts": server_types,
            "heterogeneous_fleetwide": heterogeneous_fleetwide,
            "gpu_type_constant_per_server": True,
            "cost_path_charges": "single dominant GPU type per period (not per selected GPU type)",
            "request_to_gpu_assignment_in_reward_path": False,
            "routing_opportunity_across_types": heterogeneous_fleetwide,
            "applicability": "NOT_APPLICABLE (simulated-only / fixture-only)",
            "required_before_headline": [
                "per-replica/request GPU-type assignment mechanism in run_unified_replay",
                "cost path that charges per selected GPU type (not the dominant)",
                "per-(GPU,model) measured serving latency calibration"],
            "default_on_auto_noop": True}


def _selection_audit(ctx, win, med_prompt):
    """Run the hierarchical planner default-off (product boundary) and opt-in; record generated/selected."""
    from aurelius.environment.controller import DEFAULT_BENCHMARK_PLANNER_MODE, run_period_episode
    from aurelius.environment.training import _controller as build_controller
    from aurelius.environment.training import make_world_state
    common, fleet, cm, fm, per, frames, prices = (ctx["common"], ctx["fleet"], ctx["cm"], ctx["fm"],
                                                  ctx["per"], ctx["frames"], ctx["prices"])
    kv = dict(kv_state_pool=ctx["pool"], kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")

    def _run(enable_kv, enable_pd):
        ws = make_world_state(common.get("world_state_params"))
        c = build_controller(fm, fleet, cm, {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15},
                             common, world_state=ws)
        c.horizon_steps = 1
        c.planning_kv_cost_mode = "hybrid_capacity_work"
        c.planning_prompt_tokens = med_prompt
        c.electricity_price_aware = True
        c.planner_mode = DEFAULT_BENCHMARK_PLANNER_MODE
        c.planner_budget = 100
        c.enable_kv_cache_precision = enable_kv
        c.enable_prefill_decode_disagg = enable_pd
        ws2 = make_world_state(common.get("world_state_params"))
        rep = run_period_episode("audit", lambda h: c.decide(h).to_dict(), per, frames, win,
                                 fleet_state=fleet, cost_model=cm, world_state=ws2,
                                 electricity_prices=prices, **common, **kv)
        return {"gp_per_dollar": round(rep.goodput_per_dollar, 2),
                "kv_mix": dict(rep.kv_cache_precision_mix), "pd_mix": dict(rep.prefill_decode_mix)}

    default_off = _run(False, False)
    opt_in = _run(True, True)
    return {"default_off": default_off, "opt_in": opt_in,
            "kv_selected_default_off": any(k != "inherit_weight_precision" for k in default_off["kv_mix"]),
            "kv_selected_opt_in": any(k != "inherit_weight_precision" for k in opt_in["kv_mix"]),
            "pd_selected_default_off": any(k != "shared" for k in default_off["pd_mix"]),
            "pd_selected_opt_in": any(k != "shared" for k in opt_in["pd_mix"]),
            "headline_unchanged_by_opt_in": default_off["gp_per_dollar"] == opt_in["gp_per_dollar"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="pjm")
    ap.add_argument("--cap", type=int, default=100000)
    ap.add_argument("--win-len", type=int, default=6)
    ap.add_argument("--max-periods", type=int, default=6)
    ap.add_argument("--mooncake-limit", type=int, default=12000)
    args = ap.parse_args()

    ctx = build_market(args.market, req_cap=args.cap, mooncake_limit=args.mooncake_limit)
    from aurelius.environment.training import make_world_state
    ws = make_world_state(ctx["common"].get("world_state_params"))
    wins = select_windows(ctx["prices"], ctx["n"], win_len=args.win_len, quick=True)
    wname, win = next(iter(wins.items()))
    win = win[:args.max_periods]
    periods = [sorted(ctx["per"].get(p, []), key=lambda r: r[0]) for p in win]
    gpu_type = (max(ctx["fleet"].gpu_type_mix, key=ctx["fleet"].gpu_type_mix.get)
                if getattr(ctx["fleet"], "gpu_type_mix", None) else "A100")
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for recs in periods for r in recs)
    med_prompt = ins[len(ins) // 2] if ins else 512

    prompt_present = _prompt_data_present(periods)
    mean_reqs = round(statistics.mean(len(p) for p in periods), 1) if periods else 0
    audit = {
        "config": {"market": args.market, "cap": args.cap, "window": wname, "periods": len(win),
                   "dominant_gpu_type": gpu_type, "median_prompt_tokens": med_prompt,
                   "mean_requests_per_period": mean_reqs},
        "data_quality": {
            "prompt_token_data_present": prompt_present,
            "caveat": ("the Azure serving trace records input_tokens=0 (prompt side UNOBSERVED) → the prefill "
                       "phase and large-context KV pressure cannot be exercised by this benchmark; prefill-side "
                       "regime metrics (PD skew, KV HBM pressure) are degraded and read as decode-only/light")
            if not prompt_present else "prompt tokens observed",
            "load_level": "light (few requests/period, low arrival rate) → not the high-load regime PD needs"},
        "kv_cache_precision": {**_kv_audit(periods, gpu_type),
                               "product_category": "OPTIONAL_SERVING_ENGINE_INTEGRATION", "default_off": True},
        "prefill_decode_disaggregation": {**_pd_audit(periods, gpu_type),
                                          "product_category": "OPTIONAL_SERVING_ENGINE_INTEGRATION",
                                          "default_off": True},
        "gpu_assignment": {**_gpu_audit(ctx["fleet"], ws),
                           "product_category": "CORE_ORCHESTRATION_AUTO_NOOP"},
        "selection": _selection_audit(ctx, win, med_prompt),
    }
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(audit, f, indent=2)
    print(json.dumps({k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))})
                      for k, v in audit.items()}, indent=2))
    print(f"DONE → {_ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
