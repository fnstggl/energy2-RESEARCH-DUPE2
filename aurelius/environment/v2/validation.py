"""V2 validation suite (Phase 9) — PASS / WARN / FAIL over the V2 serving physics.

Kept in the V2 subpackage so V1's `world_validation.py` is untouched (hard rule #1: don't cannibalise V1).
Every check is deterministic, no network, no external deps. SKIPPED is used only where a conservative
approximation would be misleading (none here). Run via ``scripts/run_dt60_full_serving_physics.py`` or
directly: ``python -c "from aurelius.environment.v2.validation import run_v2_validation; ..."``.
"""

from __future__ import annotations

from .candidate_generator import generate_candidates
from .mpc_search import AdaptiveMPCSearchV2
from .roofline_serving import RooflineServingModelV2
from .tiered_kv import TieredKVStateV2
from .world_simulator import WorldSimulatorV2
from .world_state import build_fleet_v2, clone_state_v2


def _reqs(n=200, prompt=64, out=256, seed=0):
    return ([(i * 0.01, out, prompt) for i in range(n)],
            [[(i % 6) * 10 + j for j in range(2)] for i in range(n)])


def run_v2_validation() -> dict:
    """Return {"results": [(name, status, detail)], "n_pass", "n_warn", "n_fail"}."""
    out = []

    def check(name, cond, detail="", warn=False):
        out.append((name, ("PASS" if cond else ("WARN" if warn else "FAIL")), detail))

    tm = RooflineServingModelV2(gpu_type="H100")
    t = tm.estimate(prompt_tokens=512, output_tokens=128, context_tokens=2048, batch=1)
    check("roofline_timing_positive", t.decode_time_s > 0 and t.prefill_time_s > 0)
    check("ridge_point_positive", t.ridge_point > 0)
    check("decode_memory_bound_batch1", t.roofline_regime == "memory")
    # GPU monotonicity
    h = tm.estimate(prompt_tokens=512, output_tokens=128, context_tokens=2048).extra["decode_per_tok"]
    sl = RooflineServingModelV2(gpu_type="L40S").estimate(prompt_tokens=512, output_tokens=128,
                                                          context_tokens=2048).extra["decode_per_tok"]
    check("gpu_timing_monotonic_h100_faster_than_l40s", h < sl, f"{h:.6f} < {sl:.6f}")
    # model monotonicity
    big = RooflineServingModelV2(arch_name="llama-70b-gqa").estimate(prompt_tokens=512, output_tokens=128,
                                                                     context_tokens=2048).extra["decode_per_tok"]
    check("model_timing_monotonic_70b_slower", big > h)
    # precision / spec / clock regimes
    check("precision_helps_memory_decode",
          tm.estimate(prompt_tokens=512, output_tokens=128, context_tokens=2048,
                      precision="fp8").extra["decode_per_tok"] < h)
    check("spec_helps_memory_decode",
          tm.estimate(prompt_tokens=512, output_tokens=128, context_tokens=2048,
                      spec_decode="aggressive").extra["decode_per_tok"] < h)
    base_pf = tm.estimate(prompt_tokens=4096, output_tokens=8, context_tokens=4096)
    low_pf = tm.estimate(prompt_tokens=4096, output_tokens=8, context_tokens=4096, clock="low")
    check("clock_low_hurts_compute_prefill", low_pf.prefill_time_s > base_pf.prefill_time_s)
    check("clock_low_drops_power", tm.power_scale("low") < tm.power_scale("base"))
    # KV tier conservation + ordering + transfer sanity
    k = TieredKVStateV2(cap_hbm=4, cap_cpu=4, cap_remote=8, cap_ssd=16)
    for i in range(200):
        k.serve([i % 25, i % 25 + 100], prefill_s_per_token=0.0001)
    s = k.summary()
    check("kv_tier_conservation",
          s["n"] == s["HBM_hits"] + s["CPU_DRAM_hits"] + s["REMOTE_KV_hits"] + s["SSD_hits"] + s["recompute_count"])
    costs = [k._transfer_cost_s(tt, 4)[0] for tt in ("GPU_HBM", "CPU_DRAM", "REMOTE_KV", "SSD_NVME")]
    check("kv_tier_cost_ordering", costs == sorted(costs))
    # pool conservation
    st = build_fleet_v2(n_replicas=8, gpu_type="H100")
    sim = WorldSimulatorV2()
    reqs, hs = _reqs()
    o = sim.evaluate(st, reqs, hs, {}, sla_s=10, mutate=False)
    check("serving_n_conserved", o.serving["n"] == len(reqs))
    # batching saturation reachable
    from .prefill_decode_scheduler import PrefillDecodeSchedulerV2, SchedRequest
    sat = PrefillDecodeSchedulerV2(saturation_seqs=8).simulate(
        [SchedRequest(arrival_s=i * 0.001, prompt_tokens=64, output_tokens=2048) for i in range(2000)],
        timing_model=tm, n_replicas=2, sla_s=60)
    check("batching_saturation_reachable", sat.batching_regime == "saturated")
    # co-location guard: pruned without background work
    cg = generate_candidates("mixed", has_background_work=False)
    check("coloc_guard_no_background", {c["colocation_mode"] for c in cg.candidates} == {"off"})
    # adaptive search regret bounded + coupled optimum found
    space = {"a": ["a0", "a1"], "b": ["b0", "b1"]}

    def rew(c):
        if c["a"] == "a1" and c["b"] == "b1":
            return 2.0
        return 0.5 if ("a1" in c.values() or "b1" in c.values()) else 1.0

    se = AdaptiveMPCSearchV2(space)
    bm = se.search(rew, strategy="beam_search", audit=True)
    cd = se.search(rew, strategy="coordinate_descent", audit=True)
    check("beam_finds_coupled_optimum", bm.search_regret == 0.0)
    check("coordinate_descent_regret_reported", cd.search_regret is not None and cd.search_regret > 0,
          f"regret={cd.search_regret}")
    check("no_silent_candidate_cap", bm.raw_candidate_count == 4)
    # legacy parity: scalar mode reproduces the constants
    leg = RooflineServingModelV2(mode="legacy_scalar").estimate(prompt_tokens=512, output_tokens=256)
    check("legacy_scalar_parity", abs(leg.decode_time_s - 256 * 0.020) < 1e-9)
    # determinism + clone isolation
    a = sim.evaluate(st, reqs, hs, {"precision": "fp8"}, sla_s=10, mutate=False)
    b = sim.evaluate(st, reqs, hs, {"precision": "fp8"}, sla_s=10, mutate=False)
    check("deterministic_replay", a.reward == b.reward)
    c2 = clone_state_v2(st)
    c2.replicas[0].kv.admit([1, 2, 3])
    check("clone_isolation", st.replicas[0].kv.summary()["n"] == 0)
    # no direct reward bonus (structural): reward is exactly goodput/cost (relative tol — metrics are rounded)
    _ratio = (o.metrics["sla_safe_goodput_tokens"] / o.metrics["cost_usd"]) if o.metrics["cost_usd"] > 0 else 0.0
    check("no_direct_reward_bonus",
          o.reward == 0.0 or abs(o.reward - _ratio) / o.reward < 1e-2,
          f"reward={o.reward:.1f} ~ goodput/cost={_ratio:.1f}")
    # no future leakage: a request's tier hit depends only on earlier admissions (causal by construction)
    kk = TieredKVStateV2(cap_hbm=100)
    first = kk.decide([1, 2, 3], prefill_s_per_token=0.001)
    kk.admit([1, 2, 3])
    second = kk.decide([1, 2, 3], prefill_s_per_token=0.001)
    check("no_future_leakage_causal_cache", first.tier == "RECOMPUTE" and second.tier == "GPU_HBM")

    n_pass = sum(1 for _, s, _ in out if s == "PASS")
    n_warn = sum(1 for _, s, _ in out if s == "WARN")
    n_fail = sum(1 for _, s, _ in out if s == "FAIL")
    return {"results": out, "n_pass": n_pass, "n_warn": n_warn, "n_fail": n_fail}


__all__ = ["run_v2_validation"]
