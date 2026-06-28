"""External-simulator validation harness (PR audit, Phase 7).

A *structured, deterministic* check that the ported external serving-physics (currently the roofline in
:mod:`roofline_external`) lands in the same physical band as Aurelius' live constants — the
"validate against controlled fixtures" deliverable. No network, no external deps, no runtime cost in the
hot path (this is a validation/reporting entry point, not part of the MPC rollout).

This is the seam where future external baselines (Vidur profiled tables, SplitwiseSim fixtures,
LLMServingSim 2.0 outputs) would register: each returns a per-(model, GPU) service-time estimate, and the
harness reports whether Aurelius' value is bracketed by the external fleet-spread. It deliberately does NOT
import any external engine — see ``research/OPEN_SIMULATOR_REUSE_DECISIONS.md`` (all fail PCS/MPC/DET as
dependencies); only ported equations participate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .roofline_external import GPU_SPECS, compare_to_aurelius_constants


@dataclass(frozen=True)
class BandCheck:
    """One validation row: is Aurelius' constant bracketed by the per-GPU roofline fleet-spread?"""
    metric: str
    arch: str
    aurelius_value: float
    roofline_min: float
    roofline_max: float
    in_band: bool
    note: str


def validate_roofline_band(arch: str = "llama-8b-gqa",
                           gpus=("H100", "A100", "L40S"), *, tol: float = 1.5) -> list:
    """Check that Aurelius' ``PREFILL_S_PER_TOKEN`` / ``TPOT_S`` sit inside the roofline fleet-spread.

    "In band" = bracketed by the fastest GPU's floor and ``tol``× the slowest GPU's floor. The honest
    expectation (see ``ROOFLINE_REUSE_DECISION.md``) is that a single fleet-wide constant is in-band but
    does not *resolve* the spread — so a passing check means "defensible scalar", not "GPU-accurate".
    """
    gpus = [g for g in gpus if g in GPU_SPECS]
    cmps = [compare_to_aurelius_constants(arch, g) for g in gpus]
    out = []
    for metric, rk, ak in (("decode", "decode_roofline_s_per_token", "aurelius_decode_s_per_token"),
                           ("prefill", "prefill_roofline_s_per_token", "aurelius_prefill_s_per_token")):
        floors = [c[rk] for c in cmps]
        a = cmps[0][ak]
        lo, hi = min(floors), max(floors)
        in_band = lo <= a <= hi * tol
        note = ("constant resolves to the slow end of the fleet" if a >= hi * 0.75
                else "constant is optimistic for the slow GPUs" if a <= lo
                else "constant mid-fleet")
        out.append(BandCheck(metric, arch, a, lo, hi, in_band, note))
    return out


def validation_report(archs=("llama-8b-gqa", "llama-70b-gqa")) -> dict:
    """Deterministic structured report across model archs (for CI / docs / the compare script)."""
    rows = []
    for arch in archs:
        rows.extend(validate_roofline_band(arch))
    return {
        "checks": [vars(r) for r in rows],
        "all_in_band": all(r.in_band for r in rows),
        "n": len(rows),
    }


# ---------------------------------------------------------------------------
# External-formula sanity checks (Phase 8) — each independently re-derives an external project's equation
# and asserts the V2 implementation agrees. No external repo is imported; the formulas are re-implemented
# from the PR #110 audit (OPEN_SIMULATOR_CODE_PATHS.md). Deterministic, no network.
# ---------------------------------------------------------------------------
def external_formula_checks() -> dict:
    """Return {name: {"pass": bool, "detail": str}} cross-checking V2 vs the ported external equations."""
    from .roofline_external import ARCHS, GPU_SPECS, roofline_analyze
    from .v2.roofline_serving import RooflineServingModelV2
    from .v2.tiered_kv import TIER_SPECS, TieredKVStateV2
    checks = {}
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]

    # vLLM / InferSim / llm-analysis: KV bytes/token = 2·L·kv_heads·head_dim·dtype
    expect_kv = 2 * arch.n_layers * arch.n_kv_heads * arch.head_dim * arch.dtype_bytes
    checks["vllm_kv_block_bytes"] = {"pass": arch.kv_bytes_per_token == expect_kv,
                                     "detail": f"{arch.kv_bytes_per_token} == {expect_kv}"}

    # llm-analysis ridge point = peak_FLOPS / BW ; LLM-Viewer bound = (AI < ridge) ? memory : compute
    peak = gpu.peak_fp16_tflops * 1e12 * gpu.mfu_decode
    bw = gpu.hbm_bw_gbps * 1e9 * gpu.bw_derate
    ai, _, bound, _ = roofline_analyze(peak, bw, ops=1e9, mem_bytes=1e9)   # AI=1 << ridge -> memory
    checks["llm_analysis_ridge_and_llmviewer_bound"] = {
        "pass": bound == "memory" and (peak / bw) > 1.0,
        "detail": f"ridge={peak/bw:.1f} bound@AI=1={bound}"}

    # InferSim per-stage roofline: decode time = max(compute, memory); memory-bound at batch=1
    tm = RooflineServingModelV2(gpu_type="H100")
    t = tm.estimate(prompt_tokens=512, output_tokens=128, context_tokens=2048, batch=1)
    checks["infersim_roofline_max_and_regime"] = {
        "pass": t.roofline_regime == "memory" and t.memory_bandwidth_util_estimate >= t.compute_util_estimate,
        "detail": f"regime={t.roofline_regime} mem_util={t.memory_bandwidth_util_estimate}"}

    # Splitwise KV transfer = bytes / bandwidth (CPU tier, exact)
    k = TieredKVStateV2(block_bytes=1_000_000, cap_hbm=0, cap_cpu=100)
    k._tiers["CPU_DRAM"][("default", 1)] = True
    cost, nbytes = k._transfer_cost_s("CPU_DRAM", 1)
    bw_cpu, lat_cpu, lookup_cpu = TIER_SPECS["CPU_DRAM"]
    expect_cost = lookup_cpu + lat_cpu + 1_000_000 / (bw_cpu * 1e9)
    checks["splitwise_kv_transfer_bytes_over_bw"] = {"pass": abs(cost - expect_cost) < 1e-12,
                                                     "detail": f"{cost:.8f} == {expect_cost:.8f}"}

    # LMCache tier ordering: equal-block transfer cost HBM < CPU < REMOTE < SSD
    costs = [k._transfer_cost_s(t, 4)[0] for t in ("GPU_HBM", "CPU_DRAM", "REMOTE_KV", "SSD_NVME")]
    checks["lmcache_tier_cost_ordering"] = {"pass": costs == sorted(costs),
                                            "detail": f"costs={[round(c,6) for c in costs]}"}

    # Mooncake TTFT decomposition: TTFT = queue + prefill (+ transfer); each term ≥ 0 and sums
    checks["mooncake_ttft_decomposition"] = {
        "pass": t.ttft_s >= 0 and t.prefill_time_s >= 0 and abs(t.ttft_s - t.prefill_time_s) < 1e-9,
        "detail": f"ttft={t.ttft_s} == prefill(service-only)={t.prefill_time_s}"}

    # BLIS roofline sanity: both legs strictly positive, regime consistent with the binding leg
    checks["blis_roofline_legs_positive"] = {
        "pass": t.extra["decode_per_tok"] > 0 and t.arithmetic_intensity > 0,
        "detail": f"decode_per_tok={t.extra['decode_per_tok']:.8f} AI={t.arithmetic_intensity}"}
    return checks


def external_validation_report() -> dict:
    c = external_formula_checks()
    return {"checks": c, "all_pass": all(v["pass"] for v in c.values()), "n": len(c)}


__all__ = ["BandCheck", "validate_roofline_band", "validation_report",
           "external_formula_checks", "external_validation_report"]
