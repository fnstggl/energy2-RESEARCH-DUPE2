#!/usr/bin/env python3
"""dt=60 held-out diagnostic for the per-replica KV/model residency channel (PR #106, Phase 11B).

Runs STATIC policies over a bounded 6-hour Azure window with Mooncake-derived prefix signatures
(TRACE_DERIVED_REUSE_MODEL), through ``run_period_episode`` with the residency channel active, so a
cache-affinity routing / prewarm / migration decision changes per-request service time → SLA-safe
goodput/$. Compares against the no-residency world (PR #105) and the fair baseline, and applies the
**unchanged** Pareto claim gate. Static policies (not the MPC search) isolate the *physics*: does the
residency channel pay on Azure+Mooncake? (Wiring the channel into the MPC rollout's synthetic-traffic
scoring is the named next step.)

Usage: python -m scripts.diagnose_cache_locality_dt60 --eval-periods 360
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.controller import run_period_episode
from aurelius.environment.training import build_mpc_inputs, claim_gate, make_world_state

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _static(action):
    return (lambda a: (lambda h: dict(a)))(action)


# static configs: (name, action-dict, use_residency)
def _configs():
    base = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off",
            "batching_policy": "balanced"}
    rr = {**base, "routing_policy": "round_robin"}
    kv = {**base, "routing_policy": "kv_aware"}
    return [
        ("fair_round_robin_no_residency", rr, False),
        ("fair_kv_routing_no_residency", kv, False),
        ("kv_routing_residency", kv, True),
        ("kv_routing_residency_prewarm", {**kv, "prewarm_policy": "conservative"}, True),
        ("kv_routing_residency_placement", {**kv, "placement_policy": "network_aware"}, True),
        ("kv_routing_residency_migration",
         {**kv, "placement_policy": "network_aware", "migration_policy": "conservative"}, True),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-periods", type=int, default=360)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--capacity-blocks", type=int, default=256)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=180.0, use_world_state=True,
                           control_dt_seconds=60.0)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    n = len(frames)
    ev = list(range(max(8, n - args.eval_periods), n))
    pool = _mooncake_pool(args.mooncake_limit)
    if not pool:
        raise SystemExit("no Mooncake prefix pool available")

    rows = {}
    for name, action, use_res in _configs():
        ws = make_world_state(common.get("world_state_params"))
        kw = dict(kv_state_pool=pool, kv_capacity_blocks=args.capacity_blocks) if use_res else {}
        rep = run_period_episode(name, _static(action), inp["per"], frames, ev, fleet_state=fleet,
                                 cost_model=cm, world_state=ws, **common, **kw)
        rows[name] = {"goodput_per_dollar": round(rep.goodput_per_dollar, 1),
                      "sla_violation_rate": round(rep.sla_violation_rate, 4),
                      "queue_delay_p95": round(rep.queue_delay_p95, 3),
                      "queue_delay_p99": round(rep.queue_delay_p99, 3),
                      "gpu_hours": round(rep.gpu_hours, 2),
                      "mean_kv_prefix_hit_rate": round(rep.mean_kv_prefix_hit_rate, 4),
                      "prefill_tokens_saved": rep.prefill_tokens_saved,
                      "model_switch_events": rep.model_switch_events,
                      "routing_mix": rep.routing_mix, "_rep": rep}

    # claim gate: residency arm vs the strongest non-residency fair baseline.
    fair_name = max(("fair_round_robin_no_residency", "fair_kv_routing_no_residency"),
                    key=lambda k: rows[k]["goodput_per_dollar"])
    gates = {}
    for name in ("kv_routing_residency", "kv_routing_residency_prewarm",
                 "kv_routing_residency_placement", "kv_routing_residency_migration"):
        g = claim_gate({"mpc_controller": rows[name]["_rep"], "fair": rows[fair_name]["_rep"]})
        gates[name] = {"beats_fair": g["beats_fair_baseline"], "pareto_sla_not_worse": g["pareto_sla_not_worse"],
                       "headline_allowed": g["headline_claim_allowed"], "vs": fair_name,
                       "delta_pct": g["candidate_vs_baseline_pct"]}
    for r in rows.values():
        r.pop("_rep", None)

    out = {"eval_periods": len(ev), "control_dt_seconds": 60.0, "capacity_blocks": args.capacity_blocks,
           "mooncake_pool_size": len(pool), "fair_baseline": fair_name, "rows": rows, "gates": gates}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "cache_locality_dt60.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"eval={len(ev)} periods @ dt=60s, mooncake_pool={len(pool)}, fair={fair_name}")
    print(f"{'config':>42} {'gp/$':>9} {'sla':>7} {'qp95':>6} {'gpu_h':>7} {'kv_hit':>7} {'gate':>14}")
    for name, r in rows.items():
        g = gates.get(name)
        gate = (f"{g['beats_fair']}/{g['pareto_sla_not_worse']}/{g['headline_allowed']}" if g else "-")
        print(f"{name:>42} {r['goodput_per_dollar']:>9.0f} {r['sla_violation_rate']:>7.4f} "
              f"{r['queue_delay_p95']:>6.2f} {r['gpu_hours']:>7.1f} {r['mean_kv_prefix_hit_rate']:>7.3f} {gate:>14}")


if __name__ == "__main__":
    main()
