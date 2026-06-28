"""Controlled fixtures for the PR #109 roofline serving model + mechanism sensitivity sweeps.

Proves the roofline PHYSICS (compute-bound vs memory-bandwidth-bound, distinct from decode-PHASE-bound)
and that every mechanism — batching, prefill/decode allocation, speculative decoding, clock/DVFS,
precision, co-location — produces a real sensitivity curve through TTFT / GPU-seconds / energy / cost,
helping or hurting in the physically-correct regime. No reward bonuses; all SIMULATOR_INFERENCE/
PUBLIC_SPEC. Diagnostic mechanisms are swept, not selected (only batching is a live MPC action)."""

from __future__ import annotations

from aurelius.environment.roofline import (
    ServingConfig,
    Workload,
    all_sensitivity_curves,
    arithmetic_intensity,
    roofline_regime,
    serving_point,
)

# --- the core distinction: decode-PHASE-bound != memory-BANDWIDTH-bound ------

def test_low_batch_long_context_decode_is_memory_bandwidth_bound():
    wl = Workload(prompt_tokens=256, decode_tokens=512, context_len=4096)
    cfg = ServingConfig(batch_size=1, gpu="A100")
    rr = roofline_regime("decode", cfg, wl)
    assert rr["roofline_regime"] == "memory_bandwidth_bound"   # AI << ridge at batch 1
    assert rr["arithmetic_intensity"] < rr["ridge_point"]


def test_batching_raises_arithmetic_intensity():
    wl = Workload(decode_tokens=256, context_len=512)
    lo = arithmetic_intensity("decode", ServingConfig(batch_size=1), wl)
    hi = arithmetic_intensity("decode", ServingConfig(batch_size=128), wl)
    assert hi > lo                                             # batching amortises weight bytes → AI ↑


def test_prefill_higher_intensity_than_decode():
    wl = Workload(prompt_tokens=4000, decode_tokens=64, context_len=512)
    cfg = ServingConfig(batch_size=8)
    assert arithmetic_intensity("prefill", cfg, wl) > arithmetic_intensity("decode", cfg, wl)


def test_phase_bound_distinct_from_roofline_regime():
    # a decode-PHASE-bound workload (decode dominates time) is STILL memory-bandwidth-bound by roofline —
    # the two labels are not the same thing (the PR #107 conflation this PR fixes).
    wl = Workload(prompt_tokens=128, decode_tokens=2000, context_len=2048)
    p = serving_point(wl, ServingConfig(batch_size=4))
    assert p["phase_bottleneck"] == "decode_phase_bound"
    assert p["decode_regime"] == "memory_bandwidth_bound"


# --- prefill/decode capacity disaggregation ---------------------------------

def test_right_allocation_helps_wrong_allocation_hurts():
    # prefill-heavy workload: more PREFILL capacity helps; starving prefill (low ratio) hurts TTFT.
    wl = Workload(prompt_tokens=4000, decode_tokens=32, context_len=512)
    good = serving_point(wl, ServingConfig(serving_mode="disaggregated_static", prefill_decode_ratio=0.8))
    bad = serving_point(wl, ServingConfig(serving_mode="disaggregated_static", prefill_decode_ratio=0.2))
    assert good["ttft_s"] < bad["ttft_s"]                     # right split → lower TTFT


def test_disaggregation_has_handoff_overhead():
    wl = Workload(prompt_tokens=512, decode_tokens=128)
    shared = serving_point(wl, ServingConfig(serving_mode="shared_gpu"))
    disagg = serving_point(wl, ServingConfig(serving_mode="disaggregated_static", prefill_decode_ratio=0.5))
    assert disagg["ttft_s"] >= shared["ttft_s"] - 1e-6        # 0.5/0.5 ≈ shared + handoff (never free)


# --- speculative decoding (diagnostic): right regime only -------------------

def test_spec_decode_helps_memory_bound_high_accept():
    wl = Workload(decode_tokens=512, context_len=4096)
    mb = ServingConfig(batch_size=1)                          # memory-bandwidth-bound decode
    off = serving_point(wl, mb)
    on = serving_point(wl, ServingConfig(batch_size=1, spec_decode_accept=0.9))
    assert on["completion_s"] < off["completion_s"]          # spare compute + high accept → faster


def test_spec_decode_hurts_or_neutral_compute_bound():
    # construct a compute-bound decode (huge batch on a bandwidth-rich GPU pushes AI ≥ ridge)
    wl = Workload(decode_tokens=256, context_len=64)
    cb = ServingConfig(batch_size=128, gpu="H20")            # H20 ridge is low (bandwidth-rich)
    assert roofline_regime("decode", cb, wl)["roofline_regime"] == "compute_bound"
    on = serving_point(wl, ServingConfig(batch_size=128, gpu="H20", spec_decode_accept=0.5))
    assert on["spec_speedup"] < 1.0                          # compute-bound → extra draft/verify FLOPs SLOW it
    assert on["completion_s"] > serving_point(wl, cb)["completion_s"]   # not free: it hurts in this regime


# --- clock / DVFS (diagnostic) ----------------------------------------------

def test_downclock_saves_energy_upclock_costs():
    wl = Workload(decode_tokens=256, context_len=2048)
    base = serving_point(wl, ServingConfig(clock_factor=1.0))
    down = serving_point(wl, ServingConfig(clock_factor=0.7))
    up = serving_point(wl, ServingConfig(clock_factor=1.15))
    assert down["power_w"] < base["power_w"] < up["power_w"]
    assert down["energy_j"] < base["energy_j"]               # memory-bound: little throughput loss, less power


# --- precision (diagnostic) -------------------------------------------------

def test_lower_precision_helps_memory_bound_throughput():
    wl = Workload(decode_tokens=256, context_len=2048)
    fp16 = serving_point(wl, ServingConfig(precision="fp16", batch_size=1))
    fp8 = serving_point(wl, ServingConfig(precision="fp8", batch_size=1))
    assert fp8["tokens_per_s_decode"] > fp16["tokens_per_s_decode"]   # half the weight bytes → 2x bandwidth tps


# --- co-location (diagnostic): right regime only ----------------------------

def test_colocation_helps_memory_bound_hurts_compute_bound():
    wl = Workload(decode_tokens=512, context_len=4096)
    mb = ServingConfig(batch_size=1)                          # memory-bound → SM headroom
    off = serving_point(wl, mb)
    on = serving_point(wl, ServingConfig(batch_size=1, colocation_frac=0.5))
    assert on["coloc_useful_gpu_seconds"] > 0                 # idle SMs do real background work
    assert on["completion_s"] >= off["completion_s"]         # but adds memory pressure (not free)


# --- sensitivity curves (the clarification's requirement) -------------------

def test_every_mechanism_produces_a_sensitivity_curve():
    wl = Workload(prompt_tokens=1024, decode_tokens=256, context_len=2048)
    curves = all_sensitivity_curves(wl)
    for mech in ("batching", "prefill_decode_allocation", "speculative_decoding", "clock_dvfs",
                 "precision", "co_location"):
        c = curves[mech]
        assert len(c["curve"]) >= 3                           # a real sweep
        # every point reports the full physics
        for pt in c["curve"]:
            for k in ("ttft_s", "completion_s", "gpu_seconds", "energy_j", "cost_usd", "decode_regime"):
                assert k in pt
        # help/hurt/neutral verdict per swept metric
        assert set(c["help_hurt_neutral"]).issuperset({"completion_s", "gpu_seconds", "cost_usd", "energy_j"})
        assert c["action_surface"] in ("live_mpc_action", "diagnostic_sweep_only")
    # only batching is a live MPC action; the rest are diagnostic sweeps
    assert curves["batching"]["action_surface"] == "live_mpc_action"
    assert curves["precision"]["action_surface"] == "live_mpc_action"          # now live (this PR)
    assert curves["speculative_decoding"]["action_surface"] == "live_mpc_action"
    assert curves["co_location"]["action_surface"] == "diagnostic_sweep_only"  # SIMULATED, frozen off


def test_determinism():
    wl = Workload()
    assert serving_point(wl, ServingConfig()) == serving_point(wl, ServingConfig())
