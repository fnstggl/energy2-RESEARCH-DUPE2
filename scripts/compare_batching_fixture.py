#!/usr/bin/env python3
"""Controlled batching fixture: effective batch + decode work change with max_active_sequences; past the
saturation point the regime flips. Deterministic. Usage: python scripts/compare_batching_fixture.py"""

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


def main() -> int:
    tm = RooflineServingModelV2(gpu_type="H100")
    reqs = [SchedRequest(arrival_s=i * 0.002, prompt_tokens=128, output_tokens=512) for i in range(1500)]
    out = {"by_token_budget": {}}
    for budget in (512, 2048, 8192, 32768):
        r = PrefillDecodeSchedulerV2(max_num_batched_tokens=budget, max_active_sequences=256,
                                     saturation_seqs=32).simulate(
            reqs, timing_model=tm, n_replicas=4, sla_s=60)
        s = r.summary()
        out["by_token_budget"][budget] = {
            "completion_p95": s["completion_p95"], "effective_batch_size": s["effective_batch_size"],
            "regime": s["batching_regime"], "decode_gpu_seconds": s["decode_gpu_seconds"]}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
