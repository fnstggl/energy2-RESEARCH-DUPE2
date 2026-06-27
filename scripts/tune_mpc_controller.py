#!/usr/bin/env python3
"""Tune the MPC controller (forecasters on train, hyper-params on a disjoint val split).

Writes the selected controller config + the validation grid results.

Usage:  python -m scripts.tune_mpc_controller --horizon 4 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.training import DEFAULT_GRID, build_mpc_inputs, train_mpc_policy

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=None, help="pin a single horizon (else grid 1,2,4)")
    ap.add_argument("--limit", type=int, default=28185)      # per-minute fallback (1-hour/sample)
    ap.add_argument("--bin-seconds", type=float, default=60.0)
    ap.add_argument("--hourly-stride", type=int, default=24, help="1/N per-hour sample of the 1-week trace")
    ap.add_argument("--sim-seconds", type=float, default=240.0, help="bounded controller decision window")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processed-dir", default=os.environ.get("V2026_PROCESSED_DIR"))
    ap.add_argument("--out-dir", default=_OUT)
    args = ap.parse_args()

    inp = build_mpc_inputs(limit=args.limit, bin_seconds=args.bin_seconds,
                           processed_dir=args.processed_dir, hourly_stride=args.hourly_stride,
                           sim_seconds=args.sim_seconds)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    grid = dict(DEFAULT_GRID)
    if args.horizon is not None:
        grid["horizon"] = [args.horizon]
    trained, _ = train_mpc_policy(inp["frames"], inp["per"], fleet_state=inp["fleet_state"],
                                  cost_model=inp["cost_model"], grid=grid, common=inp["common"])
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "trained_controller_config.json"), "w") as f:
        json.dump({"controller_config": trained["controller_config"], "splits": trained["splits"],
                   "coverage": inp.get("coverage"), "val_results": trained["val_results"],
                   "common": trained["common"]}, f, indent=2)
    print(f"tuned controller → {args.out_dir}/trained_controller_config.json")
    print("selected config:", trained["controller_config"], "| splits:", trained["splits"])


if __name__ == "__main__":
    main()
