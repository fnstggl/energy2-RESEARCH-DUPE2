"""Controlled fixtures for the V2 serving world model (PR: full serving-physics integration).

Proves every V2 mechanism is causal (affects TTFT/latency/queue/GPU-seconds/SLA/cost — never reward
directly), directionally correct, deterministic, and clone-safe. Maps to the Phase-10 fixture list. No
network, no external deps."""

from __future__ import annotations

import random

from aurelius.environment.v2.candidate_generator import generate_candidates
from aurelius.environment.v2.mpc_search import AdaptiveMPCSearchV2
from aurelius.environment.v2.prefill_decode_scheduler import PrefillDecodeSchedulerV2, SchedRequest
from aurelius.environment.v2.roofline_serving import RooflineServingModelV2
from aurelius.environment.v2.tiered_kv import TieredKVStateV2
from aurelius.environment.v2.world_simulator import WorldSimulatorV2
from aurelius.environment.v2.world_state import build_fleet_v2, clone_state_v2


# ---- helpers ---------------------------------------------------------------
def _decode_heavy(n=600, seed=0):
    r = random.Random(seed)
    reqs = [(i * 0.005, r.choice([512, 1024, 2048]), 64) for i in range(n)]
    hs = [[(i % 5) * 10 + j for j in range(2)] for i in range(n)]
    return reqs, hs


def _per_tok(tm, **kw):
    return tm.estimate(prompt_tokens=512, output_tokens=256, context_tokens=2048, **kw).extra["decode_per_tok"]


# ---- Phase 1: roofline timing -------------------------------------------------
def test_roofline_timing_changes_by_gpu_type():
    h100 = _per_tok(RooflineServingModelV2(gpu_type="H100"))
    l40s = _per_tok(RooflineServingModelV2(gpu_type="L40S"))
    assert h100 < l40s          # H100 HBM bandwidth >> L40S


def test_roofline_timing_changes_by_model_size():
    small = _per_tok(RooflineServingModelV2(arch_name="llama-8b-gqa"))
    big = _per_tok(RooflineServingModelV2(arch_name="llama-70b-gqa"))
    assert big > small


def test_legacy_scalar_timing_preserved():
    leg = RooflineServingModelV2(mode="legacy_scalar").estimate(prompt_tokens=512, output_tokens=256)
    assert abs(leg.decode_time_s - 256 * 0.020) < 1e-9
    assert leg.timing_model_used == "legacy_scalar"


# ---- Phase 5: precision / spec / clock physics (timing-level, exact) ----------
def test_precision_helps_memory_bound_decode():
    tm = RooflineServingModelV2(gpu_type="H100")     # decode memory-bound at batch=1
    bf16 = _per_tok(tm, precision="bf16")
    fp8 = _per_tok(tm, precision="fp8")
    assert fp8 < bf16            # fewer bytes moved on the memory leg


def test_spec_decode_helps_memory_bound_decode():
    tm = RooflineServingModelV2(gpu_type="H100")
    off = _per_tok(tm, spec_decode="off")
    aggr = _per_tok(tm, spec_decode="aggressive")
    assert aggr < off            # fewer serial weight reloads (memory leg / speedup)


def test_spec_decode_hurts_compute_bound_decode():
    tm = RooflineServingModelV2(gpu_type="H100")
    # large batch -> decode becomes compute-bound; spec multiplies the compute leg
    base = tm.estimate(prompt_tokens=64, output_tokens=256, context_tokens=256, batch=256, spec_decode="off")
    spec = tm.estimate(prompt_tokens=64, output_tokens=256, context_tokens=256, batch=256, spec_decode="aggressive")
    assert base.roofline_regime == "compute"
    assert spec.extra["decode_per_tok"] >= base.extra["decode_per_tok"]


def test_low_clock_hurts_compute_bound_prefill():
    tm = RooflineServingModelV2(gpu_type="H100")
    base = tm.estimate(prompt_tokens=4096, output_tokens=8, context_tokens=4096, clock="base")
    low = tm.estimate(prompt_tokens=4096, output_tokens=8, context_tokens=4096, clock="low")
    assert low.prefill_time_s > base.prefill_time_s      # prefill compute-bound -> down-clock slower


def test_low_clock_neutral_on_memory_bound_decode():
    tm = RooflineServingModelV2(gpu_type="H100")
    base = _per_tok(tm, clock="base")
    low = _per_tok(tm, clock="low")
    assert abs(base - low) < 1e-12                        # memory-bound: compute leg not binding
    assert tm.power_scale("low") < tm.power_scale("base")  # but power (energy) drops


def test_int4_carries_quality_risk():
    tm = RooflineServingModelV2()
    assert tm.estimate(prompt_tokens=128, output_tokens=64, precision="int4").quality_risk > 0
    assert tm.estimate(prompt_tokens=128, output_tokens=64, precision="bf16").quality_risk == 0


def test_long_context_increases_hbm_pressure():
    tm = RooflineServingModelV2()
    short = tm.estimate(prompt_tokens=128, output_tokens=64, context_tokens=512, active_sequences=32)
    long = tm.estimate(prompt_tokens=128, output_tokens=64, context_tokens=8192, active_sequences=32)
    assert long.hbm_pressure > short.hbm_pressure


# ---- Phase 3: tiered KV --------------------------------------------------------
def test_hbm_hit_beats_cpu_hit():
    k = TieredKVStateV2(cap_hbm=100, cap_cpu=100)
    k.admit([1, 2, 3])
    assert k.decide([1, 2, 3], prefill_s_per_token=0.0001).tier == "GPU_HBM"


def test_remote_beats_recompute_low_pressure_recompute_wins_high():
    from aurelius.environment.roofline_external import ARCHS
    bb = ARCHS["llama-8b-gqa"].kv_bytes_per_token * 16
    pf = 0.00002

    def long_prefix_in_remote(net):
        k = TieredKVStateV2(block_bytes=bb, cap_hbm=2, cap_cpu=2, cap_remote=1000, network_pressure=net)
        lp = list(range(40))
        k.admit(lp)
        for j in range(60):
            k.admit([1000 + j])
        return k, lp

    k_lo, lp = long_prefix_in_remote(0.0)
    k_hi, _ = long_prefix_in_remote(0.95)
    assert k_lo.decide(lp, prefill_s_per_token=pf).tier == "REMOTE_KV"
    assert k_hi.decide(lp, prefill_s_per_token=pf).tier == "RECOMPUTE"


def test_tier_capacity_changes_hit_rate():
    def run(cap):
        k = TieredKVStateV2(cap_hbm=cap, cap_cpu=cap)
        for i in range(200):
            k.serve([i % 30, (i % 30) + 100], prefill_s_per_token=0.0001)
        return k.summary()["tier_hit_rate"]
    assert run(64) >= run(4)        # more capacity -> fewer evictions -> higher hit rate


def test_tenant_safe_no_cross_domain_reuse():
    k = TieredKVStateV2(cap_hbm=100, domain="tenantA")
    k.admit([1, 2, 3])
    k.domain = "tenantB"            # a different domain must not see tenantA's blocks
    assert k.decide([1, 2, 3], prefill_s_per_token=0.001).tier == "RECOMPUTE"


# ---- Phase 2 / 4: pools + batching --------------------------------------------
def test_wrong_disaggregation_allocation_hurts():
    tm = RooflineServingModelV2(gpu_type="H100")
    sch = PrefillDecodeSchedulerV2()
    reqs = [SchedRequest(arrival_s=i * 0.01, prompt_tokens=64, output_tokens=512) for i in range(300)]
    good = sch.simulate(reqs, timing_model=tm, n_replicas=8, serving_mode="disaggregated_static",
                        prefill_frac=0.25, sla_s=20)
    bad = sch.simulate(reqs, timing_model=tm, n_replicas=8, serving_mode="disaggregated_static",
                       prefill_frac=0.85, sla_s=20)
    assert bad.summary()["completion_p95"] > good.summary()["completion_p95"]   # starved decode pool


def test_disaggregation_has_handoff_cost():
    tm = RooflineServingModelV2(gpu_type="H100")
    sch = PrefillDecodeSchedulerV2()
    reqs = [SchedRequest(arrival_s=i * 0.01, prompt_tokens=256, output_tokens=128) for i in range(100)]
    d = sch.simulate(reqs, timing_model=tm, n_replicas=8, serving_mode="disaggregated_static",
                     prefill_frac=0.5, sla_s=20)
    assert d.kv_handoff_bytes > 0 and d.kv_handoff_latency_s > 0      # no free disaggregation


def test_batching_saturation_regime():
    tm = RooflineServingModelV2(gpu_type="H100")
    sch = PrefillDecodeSchedulerV2(saturation_seqs=8)
    reqs = [SchedRequest(arrival_s=i * 0.001, prompt_tokens=64, output_tokens=2048) for i in range(2000)]
    r = sch.simulate(reqs, timing_model=tm, n_replicas=2, serving_mode="shared_pool", sla_s=60)
    assert r.batching_regime == "saturated"


# ---- Phase 5/X: simulator monetization + co-location guard ---------------------
def test_precision_reduces_realized_work_and_latency():
    # HONEST channel: operator cost is provisioned-dominated (faster service does NOT cut cost), so the
    # causal effect of fp8 on a memory-bound decode workload is lower realized GPU-seconds + lower latency,
    # NOT a manufactured gp/$ win. (This reproduces the V1 PR #106/#107 economic finding.)
    reqs, hs = _decode_heavy()
    st = build_fleet_v2(n_replicas=4, gpu_type="H100")
    sim = WorldSimulatorV2()
    bf16 = sim.evaluate(st, reqs, hs, {}, sla_s=3.0, mutate=False)
    fp8 = sim.evaluate(st, reqs, hs, {"precision": "fp8"}, sla_s=3.0, mutate=False)
    assert fp8.serving["decode_gpu_seconds"] < bf16.serving["decode_gpu_seconds"]   # fewer bytes moved
    assert fp8.serving["completion_p95"] <= bf16.serving["completion_p95"]          # faster memory-bound decode


def test_coupled_precision_spec_beats_either_alone_on_decode_time():
    # coupled physical win at the timing level (robust, regime-pinned): fp8+spec cut the memory leg more than
    # either alone -> lower decode_per_tok. This is the coupling beam search finds and coordinate descent misses.
    tm = RooflineServingModelV2(gpu_type="H100")
    fp8 = _per_tok(tm, precision="fp8")
    spec = _per_tok(tm, spec_decode="aggressive")
    both = _per_tok(tm, precision="fp8", spec_decode="aggressive")
    base = _per_tok(tm)
    assert fp8 < base and spec < base
    assert both < fp8 and both < spec


def test_colocation_inert_without_background_work():
    reqs, hs = _decode_heavy()
    st = build_fleet_v2(n_replicas=4, gpu_type="H100", background_work_gpu_seconds=0.0)
    sim = WorldSimulatorV2()
    off = sim.evaluate(st, reqs, hs, {"colocation_mode": "off"}, sla_s=3.0, mutate=False).reward
    aggr = sim.evaluate(st, reqs, hs, {"colocation_mode": "aggressive"}, sla_s=3.0, mutate=False).reward
    assert aggr <= off          # no background work -> no reclaim, only contention risk -> never better


def test_colocation_helps_with_real_background_work():
    # light load -> idle exists; real background work reclaims it -> billed GPU-seconds fall -> gp/$ rises.
    # This is the ONE legitimate gp/$ channel (the background jobs pay for otherwise-idle capacity).
    reqs = [(i * 0.05, 256, 64) for i in range(150)]
    hs = [[(i % 8) * 10 + j for j in range(2)] for i in range(150)]
    st = build_fleet_v2(n_replicas=8, gpu_type="H100", background_work_gpu_seconds=200.0)
    sim = WorldSimulatorV2()
    off = sim.evaluate(st, reqs, hs, {"colocation_mode": "off"}, sla_s=10.0, mutate=False)
    cons = sim.evaluate(st, reqs, hs, {"colocation_mode": "conservative"}, sla_s=10.0, mutate=False)
    assert cons.metrics["coloc_reclaim_gpu_seconds"] > 0
    assert cons.reward >= off.reward        # idle reclaimed by real background work -> cheaper


# ---- determinism + clone isolation --------------------------------------------
def test_determinism_and_clone_isolation():
    reqs, hs = _decode_heavy()
    st = build_fleet_v2(n_replicas=4)
    sim = WorldSimulatorV2()
    a = sim.evaluate(st, reqs, hs, {"precision": "fp8"}, sla_s=3.0, mutate=False)
    b = sim.evaluate(st, reqs, hs, {"precision": "fp8"}, sla_s=3.0, mutate=False)
    assert a.reward == b.reward
    # candidate eval on a clone must not mutate the real state
    before = sum(r.kv.summary()["n"] for r in st.replicas)
    sim.evaluate(st, reqs, hs, {}, sla_s=3.0, mutate=False)
    after = sum(r.kv.summary()["n"] for r in st.replicas)
    assert before == after


def test_clone_state_independent():
    st = build_fleet_v2(n_replicas=2)
    c = clone_state_v2(st)
    c.replicas[0].kv.admit([1, 2, 3])
    assert st.replicas[0].kv.summary()["n"] == 0      # original untouched


# ---- Phase 6: adaptive search --------------------------------------------------
def test_beam_finds_coupled_optimum_coordinate_descent_misses():
    space = {"a": ["a0", "a1"], "b": ["b0", "b1"], "c": ["c0", "c1"]}

    def reward(c):
        base = 2.0 if (c["a"] == "a1" and c["b"] == "b1") else (0.5 if (c["a"] == "a1" or c["b"] == "b1") else 1.0)
        return base + (0.01 if c["c"] == "c1" else 0.0)

    s = AdaptiveMPCSearchV2(space)
    cd = s.search(reward, strategy="coordinate_descent", audit=True)
    bm = s.search(reward, strategy="beam_search", audit=True)
    ex = s.search(reward, strategy="exhaustive_cartesian")
    assert cd.search_regret > 0.4 and cd.warning           # coordinate descent stuck, warns
    assert bm.selected_reward == ex.selected_reward         # beam matches exhaustive optimum
    assert bm.search_regret == 0.0


def test_search_reports_counts_no_silent_cap():
    space = {"x": list(range(10)), "y": list(range(10))}
    s = AdaptiveMPCSearchV2(space, exhaustive_threshold=50)
    r = s.search(lambda c: c["x"] + c["y"], strategy="auto")
    assert r.raw_candidate_count == 100
    assert r.strategy == "beam_search"      # 100 > 50 threshold -> not silently exhausted
    assert r.evaluated_candidate_count < 100


# ---- Phase 7: candidate generation --------------------------------------------
def test_compute_bound_prunes_spec_and_coloc():
    cs = generate_candidates("compute", hbm_pressure=0.2, has_background_work=False)
    assert {c["spec_decode"] for c in cs.candidates} == {"off"}
    assert {c["colocation_mode"] for c in cs.candidates} == {"off"}


def test_memory_bound_keeps_precision_and_spec_options():
    cs = generate_candidates("memory", hbm_pressure=0.2, has_background_work=False)
    assert {"fp8", "int4"} <= {c["precision"] for c in cs.candidates}
    assert "aggressive" in {c["spec_decode"] for c in cs.candidates}


def test_coloc_pruned_without_background_kept_with():
    no_bg = generate_candidates("mixed", has_background_work=False)
    with_bg = generate_candidates("mixed", has_background_work=True)
    assert {c["colocation_mode"] for c in no_bg.candidates} == {"off"}
    assert {"conservative", "aggressive"} & {c["colocation_mode"] for c in with_bg.candidates}
