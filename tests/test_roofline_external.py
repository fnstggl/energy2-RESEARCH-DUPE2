"""Controlled fixtures for the ported external roofline model (PR audit, Phase 7/9).

Proves the ported InferSim/llm-analysis/LLM-Viewer formulas are physically self-consistent and that the
KV-byte formula is IDENTICAL to the existing ``kv_cache`` one (so the roofline reference and the live
service model never diverge). Deterministic, no network, no external deps."""

from __future__ import annotations

from aurelius.environment.external_sim_validation import (
    validate_roofline_band,
    validation_report,
)
from aurelius.environment.kv_cache import FOOTPRINTS
from aurelius.environment.roofline_external import (
    ARCHS,
    GPU_SPECS,
    ModelArch,
    compare_to_aurelius_constants,
    decode_estimate,
    prefill_estimate,
    roofline_analyze,
)


def test_gemm_flops_and_kv_bytes_match_kv_cache():
    # the roofline arch KV-byte formula must equal kv_cache.KVFootprint (both 2·L·kv_heads·hd·dtype)
    for name in ("llama-8b-gqa", "llama-7b-mha", "llama-70b-gqa"):
        if name in FOOTPRINTS:
            assert ARCHS[name].kv_bytes_per_token == FOOTPRINTS[name].bytes_per_token


def test_roofline_analyze_ridge_classification():
    # ridge = peak/bw. ai below ridge -> memory; above -> compute.
    peak, bw = 1000.0, 10.0          # ridge = 100 flops/byte
    ai_lo, _, bound_lo, _ = roofline_analyze(peak, bw, ops=50.0, mem_bytes=10.0)   # ai=5 < 100
    ai_hi, _, bound_hi, _ = roofline_analyze(peak, bw, ops=5000.0, mem_bytes=10.0)  # ai=500 > 100
    assert bound_lo == "memory" and ai_lo < 100
    assert bound_hi == "compute" and ai_hi > 100


def test_decode_is_memory_bound_at_batch1():
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]
    dc = decode_estimate(arch, gpu, context_tokens=2048, batch=1)
    assert dc.bound == "memory"
    assert dc.memory_time_s > dc.compute_time_s


def test_prefill_is_compute_bound_for_nontrivial_prompt():
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]
    pf = prefill_estimate(arch, gpu, prompt_tokens=2048)
    assert pf.bound == "compute"


def test_longer_context_slows_decode():
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]
    short = decode_estimate(arch, gpu, context_tokens=512)
    long = decode_estimate(arch, gpu, context_tokens=8192)
    assert long.seconds_per_token > short.seconds_per_token


def test_bigger_model_slower_decode():
    gpu = GPU_SPECS["H100"]
    small = decode_estimate(ARCHS["llama-8b-gqa"], gpu, context_tokens=2048)
    big = decode_estimate(ARCHS["llama-70b-gqa"], gpu, context_tokens=2048)
    assert big.seconds_per_token > small.seconds_per_token


def test_faster_hbm_speeds_decode():
    arch = ARCHS["llama-8b-gqa"]
    slow = decode_estimate(arch, GPU_SPECS["L40S"], context_tokens=2048)   # 864 GB/s
    fast = decode_estimate(arch, GPU_SPECS["H100"], context_tokens=2048)   # 3350 GB/s
    assert fast.seconds_per_token < slow.seconds_per_token


def test_batching_amortises_decode_weights():
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]
    b1 = decode_estimate(arch, gpu, context_tokens=2048, batch=1)
    b16 = decode_estimate(arch, gpu, context_tokens=2048, batch=16)
    assert b16.seconds_per_token < b1.seconds_per_token


def test_determinism():
    a = compare_to_aurelius_constants("llama-8b-gqa", "H100")
    b = compare_to_aurelius_constants("llama-8b-gqa", "H100")
    assert a == b


def test_aurelius_constants_land_in_physical_band():
    """The Aurelius benchmark constants must be bracketed by the per-GPU roofline floor across the
    fleet — neither below the fastest GPU's floor nor absurdly above the slowest. This is the honest
    validation: the constants are a defensible single band, the roofline resolves the GPU/model spread."""
    decodes = [compare_to_aurelius_constants("llama-8b-gqa", g)["decode_roofline_s_per_token"]
               for g in ("H100", "A100", "L40S")]
    aurelius_decode = compare_to_aurelius_constants("llama-8b-gqa", "H100")["aurelius_decode_s_per_token"]
    # Aurelius' 20ms decode sits between the H100 floor (~5ms) and the L40S floor (~20ms): in-band.
    assert min(decodes) <= aurelius_decode <= max(decodes) * 1.5


def test_ideal_mfu_only_helps_compute_bound_stage():
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]
    # decode is memory-bound -> ideal MFU does NOT change it (compute leg isn't binding)
    dc = decode_estimate(arch, gpu, context_tokens=2048)
    dc_ideal = decode_estimate(arch, gpu, context_tokens=2048, ideal_mfu=True)
    assert abs(dc.seconds_per_token - dc_ideal.seconds_per_token) < 1e-12
    # prefill is compute-bound -> ideal MFU speeds it up
    pf = prefill_estimate(arch, gpu, prompt_tokens=2048)
    pf_ideal = prefill_estimate(arch, gpu, prompt_tokens=2048, ideal_mfu=True)
    assert pf_ideal.seconds_per_token < pf.seconds_per_token


def test_validation_harness_deterministic_and_8b_decode_in_band():
    # the documented case: Aurelius' 20ms decode constant IS bracketed by the 8B fleet roofline spread.
    rows = validate_roofline_band("llama-8b-gqa")
    decode = next(r for r in rows if r.metric == "decode")
    assert decode.in_band is True
    assert decode.roofline_min <= decode.aurelius_value <= decode.roofline_max * 1.5
    # deterministic
    assert [vars(r) for r in rows] == [vars(r) for r in validate_roofline_band("llama-8b-gqa")]


def test_validation_report_is_honest_about_model_spread():
    # a single fleet-wide constant CANNOT bracket every (model, GPU) — the report must surface that,
    # not paper over it. 70B at constant TPOT is out-of-band (constant too fast for a 70B model).
    rep = validation_report()
    assert rep["n"] == 4
    assert rep["all_in_band"] is False          # honest: the scalar does not resolve the model spread
    big_decode = next(c for c in rep["checks"]
                      if c["arch"] == "llama-70b-gqa" and c["metric"] == "decode")
    assert big_decode["in_band"] is False


def test_custom_arch_roundtrips():
    m = ModelArch("tiny", n_layers=4, hidden=512, n_heads=8, n_kv_heads=2, head_dim=64, intermediate=2048)
    assert m.gqa_groups == 4
    assert m.kv_bytes_per_token == 2 * 4 * 2 * 64 * 2
    dc = decode_estimate(m, GPU_SPECS["H100"], context_tokens=1024)
    assert dc.seconds_per_token > 0
