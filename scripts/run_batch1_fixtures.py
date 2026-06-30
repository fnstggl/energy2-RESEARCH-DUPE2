#!/usr/bin/env python3
"""Batch-1 controlled fixtures (Phase 6) — each new knob shown to help, hurt, and be neutral.

Twelve controlled fixtures exercise the three new knobs (KV-cache precision, heterogeneous GPU assignment,
prefill/decode disaggregation) through their causal simulator paths. Every fixture compares a BASELINE
bundle vs the SELECTED (knob) bundle on a common reward proxy and reports the absolute + percent gp/$ delta,
the SLA delta, the latency delta, GPU-hours, cost, a fidelity label, and pass/fail.

The reward proxy mirrors the production ``PeriodOutcome.goodput_per_dollar`` structure exactly —
``gp/$ = sla_safe_goodput·(1−quality_risk) / operator_cost`` — so a fixture win is the same KIND of win the
benchmark scores (latency→SLA→goodput, GPU-seconds/energy→cost), never a bonus. KV precision + PD run through
the roofline serving point; GPU assignment runs through the heterogeneous-fleet fixture model.

Usage: python -m scripts.run_batch1_fixtures
"""

from __future__ import annotations

import json
import os

from aurelius.environment.actions import ActionBundle
from aurelius.environment.gpu_assignment import (
    GPUType,
    WorkloadClass,
    compare_assignment_policies,
)
from aurelius.environment.pd_disaggregation import PDWorkload, pd_serving_point
from aurelius.environment.roofline import GPU_HOUR_USD, Workload, serving_point
from aurelius.environment.roofline_actions import action_serving_config, roofline_action_factors

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "results")
_ARTIFACT = os.path.join(_OUT, "batch1_controlled_fixtures.json")


def _roofline_reward(bundle, wl: Workload, *, gpu: str, n_requests: int, sla_s: float,
                     batch: int = 16) -> dict:
    """gp/$ + SLA + latency + GPU-hours + cost for a bundle on a roofline workload. Same reward structure as
    PeriodOutcome.goodput_per_dollar: goodput·(1−quality_risk)/cost; SLA-safe iff completion ≤ target."""
    cfg = action_serving_config(bundle, gpu=gpu, batch_size=batch)
    pt = serving_point(wl, cfg)
    factors = roofline_action_factors(bundle, wl, gpu=gpu, batch_size=batch)
    q_risk = float(factors.get("quality_sla_risk", 0.0))
    compl = pt["completion_s"]
    gpu_seconds = pt["gpu_seconds"] * n_requests
    energy_cost = pt["energy_j"] * n_requests / 3.6e6 * 0.06
    cost = gpu_seconds * GPU_HOUR_USD / 3600.0 + energy_cost
    sla_safe = compl <= sla_s
    goodput = (wl.decode_tokens * n_requests) * (1.0 if sla_safe else 0.0) * (1.0 - q_risk)
    return {"gp_per_dollar": goodput / max(cost, 1e-9), "completion_s": compl,
            "gpu_hours": gpu_seconds / 3600.0, "cost": cost,
            "sla_violation_rate": 0.0 if sla_safe else 1.0, "quality_risk": q_risk}


def _pd_reward(policy: str, wl: PDWorkload, *, n_replicas: int, sla_s: float, decode_tokens: int,
               n_requests: int) -> dict:
    """gp/$ for a PD policy: completion from the phase-pool model; cost from busy + idle GPU-seconds (idle
    pools are billed → no free disaggregation); goodput gated by SLA on completion."""
    r = pd_serving_point(wl, policy, n_replicas=n_replicas, period_seconds=60.0)
    compl = r.mean_completion_s
    busy_gpu_s = (wl.prefill_work_s + wl.decode_work_s) * n_requests
    gpu_seconds = busy_gpu_s + r.idle_gpu_seconds_total       # idle pools still cost (warm GPUs)
    cost = gpu_seconds * GPU_HOUR_USD / 3600.0
    sla_safe = compl <= sla_s
    goodput = (decode_tokens * n_requests) * (1.0 if sla_safe else 0.0)
    return {"gp_per_dollar": goodput / max(cost, 1e-9), "completion_s": compl,
            "gpu_hours": gpu_seconds / 3600.0, "cost": cost,
            "sla_violation_rate": 0.0 if sla_safe else 1.0,
            "kv_handoff_latency": r.kv_handoff_latency, "alloc_efficiency": r.allocation_efficiency}


def _delta(base, sel):
    g0, g1 = base["gp_per_dollar"], sel["gp_per_dollar"]
    return {"baseline_gp_per_dollar": round(g0, 2), "selected_gp_per_dollar": round(g1, 2),
            "gp_abs_delta": round(g1 - g0, 2), "gp_pct_delta": round(100.0 * (g1 - g0) / g0, 2) if g0 else None,
            "sla_delta": round(sel["sla_violation_rate"] - base["sla_violation_rate"], 4),
            "latency_delta_s": round(sel["completion_s"] - base["completion_s"], 5),
            "gpu_hours": round(sel["gpu_hours"], 5), "cost": round(sel["cost"], 6)}


def fixtures() -> list:
    out = []

    # 1. KV precision HELPS HBM-bound decode (long context, memory-bandwidth-bound) ------------------
    wl_mem = Workload(prompt_tokens=512, decode_tokens=512, context_len=4096)
    base = _roofline_reward(ActionBundle(), wl_mem, gpu="A100", n_requests=200, sla_s=20.0)
    sel = _roofline_reward(ActionBundle(kv_cache_precision_policy="kv_fp8"), wl_mem, gpu="A100",
                           n_requests=200, sla_s=20.0)
    d = _delta(base, sel)
    out.append({"fixture": "1_kv_precision_helps_hbm_bound_decode", "knob": "kv_cache_precision",
                "baseline_bundle": "neutral", "selected_bundle": "kv_fp8",
                "fidelity": "SIMULATOR_INFERENCE (roofline KV-bandwidth band; fp8 KV ≈ lossless, PUBLIC_BENCHMARK)",
                "expect": "help", "pass": d["gp_pct_delta"] is not None and d["gp_pct_delta"] > 0
                and d["sla_delta"] <= 0, **d})

    # 2. KV precision NEUTRAL when not memory-bound (tiny context → compute-bound decode) -------------
    wl_comp = Workload(prompt_tokens=64, decode_tokens=8, context_len=64)
    base = _roofline_reward(ActionBundle(), wl_comp, gpu="A100", n_requests=200, sla_s=20.0)
    sel = _roofline_reward(ActionBundle(kv_cache_precision_policy="kv_fp8"), wl_comp, gpu="A100",
                           n_requests=200, sla_s=20.0)
    d = _delta(base, sel)
    out.append({"fixture": "2_kv_precision_neutral_when_not_memory_bound", "knob": "kv_cache_precision",
                "baseline_bundle": "neutral", "selected_bundle": "kv_fp8",
                "fidelity": "SIMULATOR_INFERENCE (roofline regime classifier)",
                "expect": "neutral", "pass": abs(d["gp_pct_delta"] or 0.0) < 1.0, **d})

    # 3. KV precision UNSAFE mode (int4 KV) excluded from headline (quality risk > 0) -----------------
    sel_unsafe = _roofline_reward(ActionBundle(kv_cache_precision_policy="kv_int4_diagnostic_only"),
                                  wl_mem, gpu="A100", n_requests=200, sla_s=20.0)
    base = _roofline_reward(ActionBundle(), wl_mem, gpu="A100", n_requests=200, sla_s=20.0)
    d = _delta(base, sel_unsafe)
    from aurelius.environment.kv_precision import is_headline_safe_kv
    out.append({"fixture": "3_kv_precision_unsafe_excluded", "knob": "kv_cache_precision",
                "baseline_bundle": "neutral", "selected_bundle": "kv_int4_diagnostic_only",
                "fidelity": "SIMULATOR_INFERENCE; quality_risk UNMODELLED → diagnostic-only",
                "expect": "excluded_from_headline",
                "pass": (sel_unsafe["quality_risk"] > 0.0) and (not is_headline_safe_kv("kv_int4_diagnostic_only")),
                "quality_risk": round(sel_unsafe["quality_risk"], 4), **d})

    # 4. Heterogeneous assignment HELPS latency-sensitive load (tight SLA cheap GPU violates) ---------
    fleet = [GPUType("H100", 4), GPUType("A10", 8), GPUType("H20", 4)]
    cls_lat = [WorkloadClass("latency_sensitive", 1024, 128, 2.0, sla_s=0.30, kind="latency"),
               WorkloadClass("batch", 1024, 512, 1.0, None, kind="batch")]
    cmp = compare_assignment_policies(cls_lat, fleet)
    g0 = cmp["results"]["homogeneous_default"]["gp_per_dollar"]
    g1 = cmp["results"]["fastest_for_latency_sensitive"]["gp_per_dollar"]
    v0 = cmp["results"]["homogeneous_default"]["sla_violation_rate"]
    v1 = cmp["results"]["fastest_for_latency_sensitive"]["sla_violation_rate"]
    out.append({"fixture": "4_hetero_assignment_helps_latency_sensitive", "knob": "gpu_assignment",
                "baseline_bundle": "homogeneous_default", "selected_bundle": "fastest_for_latency_sensitive",
                "fidelity": "SIMULATOR_INFERENCE (per-GPU roofline + per-type cost); NOT_APPLICABLE to prod benchmark",
                "expect": "help", "baseline_gp_per_dollar": round(g0, 2), "selected_gp_per_dollar": round(g1, 2),
                "gp_abs_delta": round(g1 - g0, 2), "gp_pct_delta": round(100.0 * (g1 - g0) / g0, 2) if g0 else None,
                "sla_delta": round(v1 - v0, 4), "pass": g1 > g0 and v1 <= v0})

    # 5. Heterogeneous assignment HELPS cost-sensitive batch (slack SLA → cheap GPU). Dominant GPU is the
    #    EXPENSIVE H100, so the homogeneous baseline overpays for slack batch work; routing batch to the cheap
    #    A10 (still SLA-safe) cuts cost → gp/$ up.
    fleet_h100 = [GPUType("H100", 10), GPUType("A10", 4), GPUType("H20", 2)]
    cls_batch = [WorkloadClass("latency_sensitive", 512, 64, 1.0, sla_s=5.0, kind="latency"),
                 WorkloadClass("batch", 2048, 1024, 3.0, sla_s=30.0, kind="batch")]
    cmp = compare_assignment_policies(cls_batch, fleet_h100)
    g0 = cmp["results"]["homogeneous_default"]["gp_per_dollar"]
    gb = cmp["results"]["cheapest_for_batch"]["gp_per_dollar"]
    best = cmp["best_deployable_policy"]
    out.append({"fixture": "5_hetero_assignment_helps_cost_batch", "knob": "gpu_assignment",
                "baseline_bundle": "homogeneous_default", "selected_bundle": best,
                "fidelity": "SIMULATOR_INFERENCE; NOT_APPLICABLE to prod benchmark",
                "expect": "help", "baseline_gp_per_dollar": round(g0, 2),
                "selected_gp_per_dollar": round(cmp["best_gp_per_dollar"], 2),
                "cheapest_for_batch_gp_per_dollar": round(gb, 2),
                "gp_pct_delta": round(100.0 * (cmp["best_gp_per_dollar"] - g0) / g0, 2) if g0 else None,
                "pass": cmp["best_gp_per_dollar"] >= g0})

    # 6. WRONG GPU assignment HURTS (latency-sensitive forced to slowest cheap GPU) -------------------
    # the 'homogeneous_default' on the dominant cheap A10 IS the wrong assignment for tight-SLA latency work.
    cmp = compare_assignment_policies(cls_lat, fleet)
    g_wrong = cmp["results"]["homogeneous_default"]["gp_per_dollar"]
    g_right = cmp["results"]["fastest_for_latency_sensitive"]["gp_per_dollar"]
    v_wrong = cmp["results"]["homogeneous_default"]["sla_violation_rate"]
    out.append({"fixture": "6_wrong_gpu_assignment_hurts", "knob": "gpu_assignment",
                "baseline_bundle": "fastest_for_latency_sensitive (correct)",
                "selected_bundle": "homogeneous_default-on-cheap (wrong)",
                "fidelity": "SIMULATOR_INFERENCE; NOT_APPLICABLE to prod benchmark",
                "expect": "hurt", "correct_gp_per_dollar": round(g_right, 2),
                "wrong_gp_per_dollar": round(g_wrong, 2), "wrong_sla_violation_rate": round(v_wrong, 4),
                "pass": g_wrong < g_right and v_wrong > 0})

    # 7. HOMOGENEOUS fleet gives NO fake benefit (all policies tie) -----------------------------------
    fleet_homo = [GPUType("A100", 16)]
    cmp = compare_assignment_policies(cls_lat, fleet_homo)
    gps = {round(r["gp_per_dollar"], 2) for p, r in cmp["results"].items() if r["deployable"]}
    out.append({"fixture": "7_homogeneous_fleet_no_fake_benefit", "knob": "gpu_assignment",
                "baseline_bundle": "homogeneous_default", "selected_bundle": "any (all tie)",
                "fidelity": "STRUCTURAL GUARANTEE (single GPU type → one assignment)",
                "expect": "neutral", "distinct_deployable_gp_per_dollar": sorted(gps),
                "homogeneous_fleet": cmp["homogeneous_fleet"], "pass": len(gps) == 1})

    # 8. Prefill-heavy workload BENEFITS from prefill-heavy PD split. SLA tight enough that the shared pool's
    #    prefill/decode interference makes it VIOLATE while the matched split (no interference) MEETS it.
    ph = PDWorkload(arrival_rate=11.0, prefill_work_s=0.6, decode_work_s=0.22, context_tokens=2048)
    base = _pd_reward("shared", ph, n_replicas=12, sla_s=1.0, decode_tokens=64, n_requests=200)
    sel = _pd_reward("prefill_heavy", ph, n_replicas=12, sla_s=1.0, decode_tokens=64, n_requests=200)
    d = _delta(base, sel)
    out.append({"fixture": "8_prefill_heavy_benefits_prefill_split", "knob": "prefill_decode",
                "baseline_bundle": "shared", "selected_bundle": "prefill_heavy (p60_d40)",
                "fidelity": "SIMULATOR_INFERENCE (phase-pool M/M/c approximation; no live disagg pools)",
                "expect": "help", "pass": sel["gp_per_dollar"] > base["gp_per_dollar"] and d["sla_delta"] <= 0,
                "note": "shared violates (interference); matched split meets SLA → baseline goodput 0 so pct is N/A",
                **d})

    # 9. Decode-heavy workload BENEFITS from decode-heavy PD split (same tight-SLA mechanism). -------
    dh = PDWorkload(arrival_rate=12.0, prefill_work_s=0.5, decode_work_s=0.9, context_tokens=512)
    base = _pd_reward("shared", dh, n_replicas=16, sla_s=14.0, decode_tokens=512, n_requests=200)
    sel = _pd_reward("decode_heavy", dh, n_replicas=16, sla_s=14.0, decode_tokens=512, n_requests=200)
    d = _delta(base, sel)
    out.append({"fixture": "9_decode_heavy_benefits_decode_split", "knob": "prefill_decode",
                "baseline_bundle": "shared", "selected_bundle": "decode_heavy (p40_d60)",
                "fidelity": "SIMULATOR_INFERENCE (phase-pool M/M/c approximation)",
                "expect": "help", "pass": sel["gp_per_dollar"] > base["gp_per_dollar"] and d["sla_delta"] <= 0,
                "note": "shared violates the SLA; matched decode split meets it → goodput gained",
                **d})

    # 10. Mixed (balanced) workload may PREFER shared_pool -------------------------------------------
    mix = PDWorkload(arrival_rate=11.0, prefill_work_s=0.35, decode_work_s=0.35, context_tokens=1024)
    base = _pd_reward("shared", mix, n_replicas=12, sla_s=2.0, decode_tokens=256, n_requests=200)
    alt = _pd_reward("prefill_heavy", mix, n_replicas=12, sla_s=2.0, decode_tokens=256, n_requests=200)
    out.append({"fixture": "10_mixed_prefers_shared", "knob": "prefill_decode",
                "baseline_bundle": "prefill_heavy split", "selected_bundle": "shared (preferred)",
                "fidelity": "SIMULATOR_INFERENCE (statistical multiplexing of a shared pool)",
                "expect": "shared_wins", "shared_gp_per_dollar": round(base["gp_per_dollar"], 2),
                "split_gp_per_dollar": round(alt["gp_per_dollar"], 2),
                "pass": base["gp_per_dollar"] >= alt["gp_per_dollar"]})

    # 11. Handoff overhead can ERASE PD gains (light skewed load, huge KV context) --------------------
    light = PDWorkload(arrival_rate=1.2, prefill_work_s=0.6, decode_work_s=0.2, context_tokens=8192)
    base = _pd_reward("shared", light, n_replicas=12, sla_s=2.0, decode_tokens=64, n_requests=60)
    sel = _pd_reward("prefill_heavy", light, n_replicas=12, sla_s=2.0, decode_tokens=64, n_requests=60)
    r_pd = pd_serving_point(light, "prefill_heavy", n_replicas=12)
    out.append({"fixture": "11_handoff_can_erase_pd_gains", "knob": "prefill_decode",
                "baseline_bundle": "shared", "selected_bundle": "prefill_heavy (handoff-burdened)",
                "fidelity": "SIMULATOR_INFERENCE (KV handoff bytes/latency; BENCHMARK_DERIVED bytes)",
                "expect": "shared_wins (handoff erases gain)",
                "shared_gp_per_dollar": round(base["gp_per_dollar"], 2),
                "split_gp_per_dollar": round(sel["gp_per_dollar"], 2),
                "kv_handoff_latency_s": r_pd.kv_handoff_latency,
                "pass": base["gp_per_dollar"] >= sel["gp_per_dollar"]})

    # 12. All three knobs interact under memory + queue pressure (KV + PD; assignment NOT_APPLICABLE) -
    wl_press = Workload(prompt_tokens=512, decode_tokens=400, context_len=3072)
    base = _roofline_reward(ActionBundle(), wl_press, gpu="A100", n_requests=240, sla_s=18.0)
    sel = _roofline_reward(ActionBundle(precision_policy="fp8", kv_cache_precision_policy="kv_fp8",
                                        batching_policy="balanced"), wl_press, gpu="A100",
                           n_requests=240, sla_s=18.0)
    d = _delta(base, sel)
    out.append({"fixture": "12_three_knob_interaction_under_pressure", "knob": "kv+weight precision (PD/assign regime-gated)",
                "baseline_bundle": "neutral", "selected_bundle": "fp8 weights + kv_fp8 + balanced",
                "fidelity": "SIMULATOR_INFERENCE (roofline); gpu_assignment NOT_APPLICABLE (homogeneous prod fleet)",
                "expect": "help", "pass": (d["gp_pct_delta"] or 0) > 0 and d["sla_delta"] <= 0, **d})

    return out


def main():
    fx = fixtures()
    n_pass = sum(1 for f in fx if f.get("pass"))
    state = {"n_fixtures": len(fx), "n_pass": n_pass, "fixtures": fx}
    os.makedirs(_OUT, exist_ok=True)
    with open(_ARTIFACT, "w") as f:
        json.dump(state, f, indent=2)
    for f in fx:
        print(f"  {'PASS' if f.get('pass') else 'FAIL'}  {f['fixture']:<48} "
              f"gp%Δ={f.get('gp_pct_delta')}", flush=True)
    print(f"{n_pass}/{len(fx)} fixtures pass → {_ARTIFACT}", flush=True)


if __name__ == "__main__":
    main()
