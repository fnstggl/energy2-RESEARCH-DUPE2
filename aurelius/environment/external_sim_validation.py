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


__all__ = ["BandCheck", "validate_roofline_band", "validation_report"]
