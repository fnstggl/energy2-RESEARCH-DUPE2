#!/usr/bin/env python3
"""Compare the ported external roofline model against Aurelius' benchmark constants.

Proves whether the externally-derived prefill/decode physics (Alibaba InferSim / llm-analysis /
LLM-Viewer formulas, re-implemented in ``aurelius/environment/roofline_external.py``) lands in the
same band as the existing ``PREFILL_S_PER_TOKEN`` / ``TPOT_S`` constants — the Phase-7 validation
deliverable. Deterministic, no network, no external deps.

Usage:  python scripts/compare_external_roofline.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.environment.roofline_external import (  # noqa: E402
    ARCHS,
    GPU_SPECS,
    compare_to_aurelius_constants,
    decode_estimate,
    prefill_estimate,
)


def main() -> int:
    print("=== External roofline vs Aurelius constants (per-token service floor) ===\n")
    for arch in ("llama-8b-gqa", "llama-70b-gqa"):
        for gpu in ("H100", "A100", "L40S"):
            c = compare_to_aurelius_constants(arch, gpu)
            print(f"[{arch} on {gpu}]")
            print(f"  prefill: roofline={c['prefill_roofline_s_per_token']*1e3:.4f} ms/tok "
                  f"(ideal {c['prefill_roofline_ideal_s_per_token']*1e3:.4f}) "
                  f"bound={c['prefill_bound']:7s}  aurelius={c['aurelius_prefill_s_per_token']*1e3:.4f} ms/tok")
            print(f"  decode:  roofline={c['decode_roofline_s_per_token']*1e3:.4f} ms/tok "
                  f"(ideal {c['decode_roofline_ideal_s_per_token']*1e3:.4f}) "
                  f"bound={c['decode_bound']:7s}  aurelius={c['aurelius_decode_s_per_token']*1e3:.4f} ms/tok")
            print()

    print("=== Sanity invariants ===")
    arch, gpu = ARCHS["llama-8b-gqa"], GPU_SPECS["H100"]
    dc1 = decode_estimate(arch, gpu, context_tokens=512)
    dc2 = decode_estimate(arch, gpu, context_tokens=8192)
    print(f"  decode memory-bound at batch=1: {dc1.bound == 'memory'}")
    print(f"  longer context -> slower decode: {dc2.seconds_per_token >= dc1.seconds_per_token}")
    pf = prefill_estimate(arch, gpu, prompt_tokens=2048)
    print(f"  prefill compute-bound at 2k prompt: {pf.bound == 'compute'}")
    dcb1 = decode_estimate(arch, gpu, context_tokens=2048, batch=1)
    dcb16 = decode_estimate(arch, gpu, context_tokens=2048, batch=16)
    print(f"  batching amortises decode weights (b16 < b1): "
          f"{dcb16.seconds_per_token < dcb1.seconds_per_token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
