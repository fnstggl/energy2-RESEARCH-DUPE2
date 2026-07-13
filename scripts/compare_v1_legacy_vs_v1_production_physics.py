#!/usr/bin/env python3
"""Compare V1 legacy-scalar timing vs the new V1 production-physics default (roofline), and run the
all-MPC-knobs vs strongest-SLA-aware-baseline benchmark — under the canonical defaults.

  production_default = roofline   (GPU/model-aware base timing — now the canonical default)
  legacy_baseline    = legacy_scalar (L40S-class fleet-wide constant — explicit regression mode)

Two comparisons:
  1. World-simulator phase-model path (where the roofline default applies): legacy vs production across GPU
     types, flagging phantom SLA failures the legacy scalar invents on fast GPUs.
  2. All-MPC-knobs candidate vs the strongest SLA-aware fair baseline through the canonical env
     (`fair_backtest`), with the Pareto headline gate.

Deterministic, no network, no GPU, no external deps. Usage:
    python scripts/compare_v1_legacy_vs_v1_production_physics.py [out.json]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
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
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MOONCAKE = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_sample.csv")


def _world_path_run(timing_model, gpu_type):
    fleet, cm = V2026FleetPlane().state_at(0), CostModel()
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind",
                          migration_policy="off", routing_policy="kv_aware", batching_policy="balanced")
    prefixes = [tuple(f"p{p}_{b}" for b in range(8)) for p in range(8)]
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
    return {"sla_violation_rate": round(out.kpi.sla_violations / n, 4),
            "completion_p95_s": d.get("completion_p95"), "realized_gpu_seconds": d.get("realized_gpu_seconds"),
            "goodput_per_dollar": round(out.goodput_per_dollar, 2), "timing_model": timing_model}


def _world_path_comparison():
    rows, flags = [], []
    for gpu in GPUS:
        legacy = _world_path_run("legacy_scalar", gpu)
        prod = _world_path_run("roofline", gpu)
        rows += [{"gpu_type": gpu, "arm": "legacy_baseline", **legacy},
                 {"gpu_type": gpu, "arm": "production_default", **prod}]
        flags.append({"gpu_type": gpu, "legacy_sla_viol": legacy["sla_violation_rate"],
                      "production_sla_viol": prod["sla_violation_rate"],
                      "phantom_sla_on_legacy": legacy["sla_violation_rate"] > prod["sla_violation_rate"]})
    return {"rows": rows, "phantom_sla_flags": flags}


def _mpc_vs_sla_baseline(limit=2500):
    """All-MPC-knobs candidate vs the strongest SLA-aware fair baseline (canonical env, Pareto gate)."""
    try:
        from aurelius.environment.ingestion.azure import ingest_azure, to_serving_raw
        from aurelius.environment.optimizer_adapter import fair_backtest
    except Exception as e:  # pragma: no cover - env-dependent
        return {"error": f"fair_backtest unavailable: {e}"}
    reqs, _ = ingest_azure(limit=limit)
    raw = to_serving_raw(reqs)
    if not raw:
        return {"error": "no azure fixture"}
    t0 = raw[0][0]
    hourly = defaultdict(list)
    for arr, tok in raw:
        hourly[int((arr - t0) // 3600)].append((arr, tok))
    rep = fair_backtest(dict(hourly), env_kwargs={"mooncake_path": _MOONCAKE, "sla_s": 10.0}).to_dict()
    return {"ranking": rep["ranking"], "fair_baseline": rep["fair_baseline"], "candidate": rep["candidate"],
            "candidate_vs_baseline_pct": rep["candidate_vs_baseline_pct"],
            "headline_claim_allowed": rep["headline_claim_allowed"], "gate": rep["gate"]}


def main() -> int:
    out_path = sys.argv[1] if len(sys.argv) > 1 else None
    report = {"meta": {"production_default": "roofline", "legacy_baseline": "legacy_scalar", "sla_s": SLA_S},
              "world_simulator_path": _world_path_comparison(),
              "mpc_vs_strongest_sla_baseline": _mpc_vs_sla_baseline()}
    js = json.dumps(report, indent=2)
    print(js)
    if out_path:
        with open(out_path, "w") as fh:
            fh.write(js)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
