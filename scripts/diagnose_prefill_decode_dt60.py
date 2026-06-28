#!/usr/bin/env python3
"""dt=60 diagnostic for prefill/decode + service-time-sensitive economics (PR #107, Phase 9).

Runs the PR #106 per-replica residency path with the PR #107 phase model on the 6-hour Azure window
(Mooncake-derived prefixes) and compares the three COST MODES explicitly, against honest baselines:

- fair_realistic_no_cache     — round_robin, cold requests pay full prefill (the fair comparator)
- legacy_kv_scalar_optimistic — offline fleet scalar (credits reuse to cold requests; reference only)
- residency_provisioned       — per-replica KV + phase model, provisioned cost (reproduces PR #106)
- residency_hybrid            — + hybrid cost (provisioned floor; realized work earns a bounded discount)
- residency_realized          — + realized-work cost (UPPER-BOUND counterfactual, not a production claim)

A headline is allowed only if the Pareto gate passes vs the FAIR baseline under a DEFENSIBLE cost mode
(provisioned or hybrid). Realized-work wins are labelled upper-bound; legacy-scalar wins are unsafe.

Usage: python -m scripts.diagnose_prefill_decode_dt60 --eval-periods 360
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
    # a UNIQUE-prefix pool → the phase model runs (realistic prompt-driven prefill) but NO reuse is
    # possible (each request a distinct prefix) — the FAIR no-cache baseline that also pays realistic
    # prefill, so the residency comparison is apples-to-apples (not vs the old constant-prefill model).
    uniq_pool = [tuple(f"u{i}_{b}" for b in range(8)) for i in range(len(ev) * 64 + 64)]
    base = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off",
            "batching_policy": "balanced"}
    rr, kv = {**base, "routing_policy": "round_robin"}, {**base, "routing_policy": "kv_aware"}

    # (name, action, pool, cost_mode, label)
    configs = [
        ("fair_oldmodel_no_cache", rr, None, None, "old_constant_prefill_reference"),
        ("fair_phase_no_cache", rr, uniq_pool, "hybrid_capacity_work", "fair_realistic_prefill"),
        ("legacy_kv_scalar_optimistic", kv, None, None, "optimistic_reference"),
        ("residency_provisioned", kv, pool, "provisioned_capacity", "reproduces_#106"),
        ("residency_hybrid", kv, pool, "hybrid_capacity_work", "defensible"),
        ("residency_realized", kv, pool, "realized_serving_work", "upper_bound_counterfactual"),
    ]
    rows = {}
    for name, action, use_pool, mode, label in configs:
        ws = make_world_state(common.get("world_state_params"))
        kw = dict(kv_state_pool=use_pool, kv_capacity_blocks=args.capacity_blocks, kv_cost_mode=mode) if use_pool else {}
        rep = run_period_episode(name, _static(action), inp["per"], frames, ev, fleet_state=fleet,
                                 cost_model=cm, world_state=ws, **common, **kw)
        rows[name] = {"label": label, "cost_mode": mode,
                      "goodput_per_dollar": round(rep.goodput_per_dollar, 1),
                      "sla_violation_rate": round(rep.sla_violation_rate, 4),
                      "queue_delay_p95": round(rep.queue_delay_p95, 3),
                      "gpu_hours": round(rep.gpu_hours, 2),
                      "realized_gpu_seconds": round(rep.realized_gpu_seconds, 1),
                      "mean_ttft_p95": round(rep.mean_ttft_p95, 4),
                      "kv_hit_rate": round(rep.mean_kv_prefix_hit_rate, 4),
                      "prefill_tokens_saved": rep.prefill_tokens_saved,
                      "prefill_tokens_remaining": rep.prefill_tokens_remaining, "_rep": rep}

    fair = rows["fair_phase_no_cache"]["_rep"]            # apples-to-apples: realistic prefill, no reuse
    gates = {}
    for name in ("residency_provisioned", "residency_hybrid", "residency_realized"):
        g = claim_gate({"mpc_controller": rows[name]["_rep"], "fair": fair})
        defensible = rows[name]["cost_mode"] in ("provisioned_capacity", "hybrid_capacity_work")
        gates[name] = {"beats_fair": g["beats_fair_baseline"], "pareto_sla_not_worse": g["pareto_sla_not_worse"],
                       "headline_allowed": g["headline_claim_allowed"] and defensible,
                       "delta_pct": g["candidate_vs_baseline_pct"],
                       "claim_safety": ("defensible" if defensible else "upper_bound_counterfactual")}
    for r in rows.values():
        r.pop("_rep", None)

    out = {"eval_periods": len(ev), "control_dt_seconds": 60.0, "mooncake_pool_size": len(pool),
           "fair_baseline": "fair_phase_no_cache", "rows": rows, "gates": gates}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "prefill_decode_dt60.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"eval={len(ev)} @ dt=60s, pool={len(pool)}, fair=fair_phase_no_cache")
    print(f"{'config':>30} {'mode':>22} {'gp/$':>9} {'sla':>7} {'ttft95':>7} {'realGPUs':>9} {'kvhit':>6} {'gate':>16}")
    for name, r in rows.items():
        g = gates.get(name)
        gate = (f"{g['beats_fair']}/{g['pareto_sla_not_worse']}/{g['headline_allowed']}" if g else "-")
        print(f"{name:>30} {str(r['cost_mode']):>22} {r['goodput_per_dollar']:>9.0f} {r['sla_violation_rate']:>7.4f} "
              f"{r['mean_ttft_p95']:>7.3f} {r['realized_gpu_seconds']:>9.0f} {r['kv_hit_rate']:>6.3f} {gate:>16}")


if __name__ == "__main__":
    main()
