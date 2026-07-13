#!/usr/bin/env python3
"""Bounded dt=60 diagnostic: V1 production-physics default (roofline) vs legacy on the world_simulator path.

Runs N one-minute periods on a persistent ClusterState under each timing mode (production=roofline,
legacy=legacy_scalar), holding the workload/actions fixed, and reports the aggregate SLA / latency / cost /
goodput-per-$ deltas so the production-physics correction is visible end-to-end. Deterministic, no network,
no GPU, no external deps. Usage:
    python scripts/diagnose_production_physics_dt60.py [n_periods] [gpu_type] [out.json]
"""

from __future__ import annotations

import json
import os
import statistics
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

SLA_S = 8.0


def _gen_periods(n):
    prefixes = [tuple(f"p{p}_{b}" for b in range(8)) for p in range(8)]
    periods = []
    for k in range(n):
        rate = 80 + (k % 5) * 20
        recs = [(float(i) * (60.0 / rate), 384, 256) for i in range(rate)]
        hs = [prefixes[(k + i) % 8] for i in range(rate)]
        periods.append((recs, hs))
    return periods


def _run_mode(timing_model, gpu_type, periods):
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware", batching_policy="balanced")
    ws = initialize_world_state(n_servers=16, n_racks=4, seed=0)
    warm_seed(ws, 8)
    viols, comp95, gpd, realized = [], [], [], []
    for recs, hs in periods:
        kv = {"hash_seq": hs, "routing": "kv_aware", "cost_mode": "hybrid_capacity_work",
              "timing_model": timing_model, "gpu_type": gpu_type}
        out = simulate_period(ws, pol, recs, {"arrival_rate": 2.0, "arrival_p90": 3.0, "mean_service_s": 1.0},
                              sla_s=SLA_S, tick_seconds=10.0, cost_model=cm, fleet_state=fleet,
                              base_service_factor=0.95, period_hours=0.0167, dt_seconds=60.0,
                              kv_state=kv, mutate=True)
        d = out.kv_diag or {}
        n = max(1, out.kpi.n_total)
        viols.append(out.kpi.sla_violations / n)
        comp95.append(d.get("completion_p95") or 0.0)
        gpd.append(out.goodput_per_dollar)
        realized.append(d.get("realized_gpu_seconds") or 0.0)
    return {"timing_model": timing_model, "gpu_type": gpu_type, "n_periods": len(periods),
            "sla_violation_rate_mean": round(statistics.mean(viols), 4),
            "completion_p95_mean_s": round(statistics.mean(comp95), 4),
            "goodput_per_dollar_mean": round(statistics.mean(gpd), 1),
            "realized_gpu_seconds_total": round(sum(realized), 1)}


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    gpu = sys.argv[2] if len(sys.argv) > 2 else "H100"
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    periods = _gen_periods(n)
    report = {"meta": {"n_periods": n, "dt_seconds": 60, "gpu_type": gpu, "sla_s": SLA_S,
                       "production_default": "roofline", "legacy_baseline": "legacy_scalar"},
              "production_roofline": _run_mode("roofline", gpu, periods),
              "legacy_scalar": _run_mode("legacy_scalar", gpu, periods)}
    js = json.dumps(report, indent=2)
    print(js)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
