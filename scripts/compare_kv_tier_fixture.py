#!/usr/bin/env python3
"""Controlled KV-tier fixture: HBM<CPU<REMOTE<SSD transfer-cost ordering + remote-vs-recompute under
network pressure. Deterministic, no deps. Usage: python scripts/compare_kv_tier_fixture.py"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.environment.roofline_external import ARCHS  # noqa: E402
from aurelius.environment.v2.tiered_kv import TieredKVStateV2  # noqa: E402


def main() -> int:
    bb = ARCHS["llama-8b-gqa"].kv_bytes_per_token * 16
    k = TieredKVStateV2(block_bytes=bb)
    costs = {t: round(k._transfer_cost_s(t, 8)[0], 6)
             for t in ("GPU_HBM", "CPU_DRAM", "REMOTE_KV", "SSD_NVME")}
    out = {"block_bytes": bb, "transfer_cost_8_blocks": costs,
           "ordering_ok": list(costs.values()) == sorted(costs.values()), "cases": []}
    for net in (0.0, 0.5, 0.9, 0.97):
        kk = TieredKVStateV2(block_bytes=bb, cap_hbm=2, cap_cpu=2, cap_remote=1000, network_pressure=net)
        lp = list(range(40))
        kk.admit(lp)
        for j in range(60):
            kk.admit([1000 + j])
        d = kk.decide(lp, prefill_s_per_token=0.00002)
        out["cases"].append({"network_pressure": net, "tier": d.tier,
                             "net_benefit_s": round(d.net_benefit_s, 6)})
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
