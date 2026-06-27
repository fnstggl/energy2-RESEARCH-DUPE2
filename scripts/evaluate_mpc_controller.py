#!/usr/bin/env python3
"""Evaluate the tuned MPC controller vs strong baselines on the HELD-OUT split.

Trains forecasters (train) + tunes the controller (val) deterministically, then runs
the controller and baselines on the disjoint held-out periods and applies the claim
gate: a headline is allowed ONLY if the controller beats the strongest non-weak
baseline with disjoint splits and no oracle.

Usage:
  python -m scripts.evaluate_mpc_controller --heldout-only \
      --baselines sla_aware,aurelius_canonical,greedy_packing
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.training import (
    DEFAULT_BASELINES,
    build_mpc_inputs,
    evaluate_mpc,
    train_mpc_policy,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heldout-only", action="store_true", default=True)
    ap.add_argument("--baselines", default="fifo_weak,sla_aware,greedy_packing,aurelius_canonical,"
                    "sla_aware_kv_routing,aurelius_canonical_kv_routing,"
                    # next-batch action-specific fair baselines (batching / over-provision)
                    "sla_aware_batched,aurelius_static_full,sla_aware_capacity_1p5")
    ap.add_argument("--limit", type=int, default=28185)      # per-minute fallback (1-hour/sample)
    ap.add_argument("--bin-seconds", type=float, default=60.0)
    ap.add_argument("--hourly-stride", type=int, default=24, help="1/N per-hour sample of the 1-week trace")
    ap.add_argument("--sim-seconds", type=float, default=240.0, help="bounded controller decision window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processed-dir", default=os.environ.get("V2026_PROCESSED_DIR"))
    ap.add_argument("--out-dir", default=_OUT)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(limit=args.limit, bin_seconds=args.bin_seconds,
                           processed_dir=args.processed_dir, hourly_stride=args.hourly_stride,
                           sim_seconds=args.sim_seconds)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    baselines = {n: DEFAULT_BASELINES[n] for n in names if n in DEFAULT_BASELINES}
    trained, fm = train_mpc_policy(inp["frames"], inp["per"], fleet_state=inp["fleet_state"],
                                   cost_model=inp["cost_model"], common=inp["common"])
    rep = evaluate_mpc(trained, fm, inp["frames"], inp["per"], fleet_state=inp["fleet_state"],
                       cost_model=inp["cost_model"], baselines=baselines, common=inp["common"])
    rep["coverage"] = inp.get("coverage")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "evaluation_report.json"), "w") as f:
        json.dump(rep, f, indent=2)

    if args.json:
        print(json.dumps(rep, indent=2))
        return
    print(f"held-out eval ({rep['eval_periods']} periods); config {rep['controller_config']}")
    ranked = sorted(rep["arms"].items(), key=lambda kv: -kv[1]["goodput_per_dollar"])
    for n, a in ranked:
        print(f"  {n:20} gp/$={a['goodput_per_dollar']:>11.1f}  sla_viol={a['sla_violation_rate']:.3f}  "
              f"q_p95={a['queue_delay_p95']:.2f}s")
    g = rep["gate"]
    print(f"\nfair baseline: {g['fair_baseline']} | controller {g['candidate_vs_baseline_pct']:+.2f}% gp/$")
    print(f"  SLA-violation: mpc={g['mpc_sla_violation_rate']} vs fair={g['fair_sla_violation_rate']} "
          f"→ pareto_not_worse={g['pareto_sla_not_worse']}")
    print(f"headline claim allowed: {g['headline_claim_allowed']}  ({g['note']})")


if __name__ == "__main__":
    main()
