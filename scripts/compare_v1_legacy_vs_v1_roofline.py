#!/usr/bin/env python3
"""Compare V1 legacy-scalar timing vs V1 roofline timing on the same workload/config (V2→V1 promotion).

Runs the SAME deterministic Azure-like window + Mooncake-style prefix stream + v2026 economics through the
V1 simulator under ``timing_model="legacy_scalar"`` and ``timing_model="roofline"`` across a few GPU types,
and (optionally) the V2 roofline as a cross-reference when the V2 package is present. Reports SLA violation
rate, p95 TTFT, p95 completion, realized GPU-seconds, energy, cost, and goodput/$. Explicitly flags cases
where the legacy scalar produces PHANTOM SLA failures on faster GPUs that the roofline removes.

Deterministic, no network, no GPU, no external deps. Usage:
    python scripts/compare_v1_legacy_vs_v1_roofline.py [out.json]
"""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.environment.cost_model import CostModel  # noqa: E402
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane  # noqa: E402
from aurelius.environment.world_simulator import (  # noqa: E402
    initialize_world_state,
    simulate_period,
    warm_seed,
)

GPUS = ("H100", "A100", "L40S")
SLA_S = 8.0


def _run(timing_model, gpu_type):
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware", batching_policy="balanced")
    prefixes = [tuple(f"p{p}_{b}" for b in range(8)) for p in range(8)]
    # decode-bearing requests: small prompt, real output -> decode matters (where the scalar is most wrong)
    recs = [(float(i) * 0.5, 384, 256) for i in range(120)]
    hs = [prefixes[i % 8] for i in range(120)]
    ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(ws, 8)
    kv = {"hash_seq": hs, "routing": "kv_aware", "cost_mode": "hybrid_capacity_work",
          "timing_model": timing_model, "gpu_type": gpu_type}
    out = simulate_period(ws, pol, recs, {"arrival_rate": 2.0, "arrival_p90": 3.0, "mean_service_s": 1.0},
                          sla_s=SLA_S, tick_seconds=10.0, cost_model=cm, fleet_state=fleet,
                          base_service_factor=0.95, period_hours=0.0167, dt_seconds=60.0,
                          kv_state=kv, mutate=False)
    d = out.kv_diag or {}
    n = max(1, out.kpi.n_total)
    return {"timing_model": timing_model, "gpu_type": gpu_type,
            "sla_violation_rate": round(out.kpi.sla_violations / n, 4),
            "ttft_p95_s": d.get("ttft_p95"), "completion_p95_s": d.get("completion_p95"),
            "realized_gpu_seconds": d.get("realized_gpu_seconds"),
            "energy_j": round(out.energy_j, 2), "operator_cost_usd": round(out.operator_cost, 5),
            "goodput_per_dollar": round(out.goodput_per_dollar, 2)}


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    rows, flags = [], []
    for gpu in GPUS:
        legacy = _run("legacy_scalar", gpu)
        roof = _run("roofline", gpu)
        rows += [legacy, roof]
        phantom = legacy["sla_violation_rate"] > roof["sla_violation_rate"]
        flags.append({"gpu_type": gpu, "legacy_sla_viol": legacy["sla_violation_rate"],
                      "roofline_sla_viol": roof["sla_violation_rate"],
                      "legacy_completion_p95_s": legacy["completion_p95_s"],
                      "roofline_completion_p95_s": roof["completion_p95_s"],
                      "phantom_sla_on_legacy": phantom})
    report = {"meta": {"sla_s": SLA_S, "gpus": list(GPUS)}, "rows": rows, "phantom_sla_flags": flags}
    js = json.dumps(report, indent=2)
    print(js)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
