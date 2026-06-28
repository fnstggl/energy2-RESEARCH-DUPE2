#!/usr/bin/env python3
"""Controlled disaggregation fixture: decode-heavy prefers more decode pool, prefill-heavy prefers more
prefill pool, wrong allocation hurts, handoff is never free. Deterministic.
Usage: python scripts/compare_disaggregation_fixture.py"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.environment.v2.prefill_decode_scheduler import (  # noqa: E402
    PrefillDecodeSchedulerV2,
    SchedRequest,
)
from aurelius.environment.v2.roofline_serving import RooflineServingModelV2  # noqa: E402


def _run(reqs, frac, mode="disaggregated_static"):
    tm = RooflineServingModelV2(gpu_type="H100")
    r = PrefillDecodeSchedulerV2().simulate(reqs, timing_model=tm, n_replicas=8, serving_mode=mode,
                                            prefill_frac=frac, sla_s=20)
    s = r.summary()
    s["kv_handoff_latency_s"] = round(r.kv_handoff_latency_s, 5)
    return s


def main() -> int:
    dh = [SchedRequest(arrival_s=i * 0.01, prompt_tokens=64, output_tokens=512) for i in range(300)]
    ph = [SchedRequest(arrival_s=i * 0.01, prompt_tokens=2048, output_tokens=32) for i in range(300)]
    out = {"decode_heavy": {f"p={f}": _run(dh, f)["completion_p95"] for f in (0.25, 0.5, 0.75)},
           "prefill_heavy": {f"p={f}": _run(ph, f)["completion_p95"] for f in (0.25, 0.5, 0.75)},
           "shared_decode_heavy": _run(dh, None, "shared_pool")["completion_p95"],
           "handoff_latency_s_decode_heavy_p0.5": _run(dh, 0.5)["kv_handoff_latency_s"]}
    out["decode_heavy_prefers_low_prefill_frac"] = (
        out["decode_heavy"]["p=0.25"] < out["decode_heavy"]["p=0.75"])
    out["prefill_heavy_prefers_high_prefill_frac"] = (
        out["prefill_heavy"]["p=0.75"] < out["prefill_heavy"]["p=0.25"])
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
