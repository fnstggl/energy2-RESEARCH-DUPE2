#!/usr/bin/env python3
"""Train the forecasting ladder on the canonical environment (train split only).

Writes the per-target selection report (which model won vs naive, held-out metric,
calibration). A learned model is kept only if it beats naive on held-out data.

Usage:  python -m scripts.train_forecasters --train-frac 0.6 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.training import build_mpc_inputs, split_cuts, train_forecasters

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "data", "external", "mpc_controller")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=28185)      # per-minute fallback (1-hour/sample)
    ap.add_argument("--bin-seconds", type=float, default=60.0)
    ap.add_argument("--hourly-stride", type=int, default=24, help="1/N per-hour sample of the 1-week trace")
    ap.add_argument("--sim-seconds", type=float, default=240.0, help="bounded controller decision window")
    ap.add_argument("--seed", type=int, default=0)            # determinism is structural
    ap.add_argument("--processed-dir", default=os.environ.get("V2026_PROCESSED_DIR"))
    ap.add_argument("--out-dir", default=_OUT)
    args = ap.parse_args()

    inp = build_mpc_inputs(limit=args.limit, bin_seconds=args.bin_seconds,
                           processed_dir=args.processed_dir, hourly_stride=args.hourly_stride,
                           sim_seconds=args.sim_seconds)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    frames = inp["frames"]
    t1, _ = split_cuts(len(frames), train=args.train_frac, val=0.2)
    _, report = train_forecasters(frames, t1)
    os.makedirs(args.out_dir, exist_ok=True)
    out = {"n_frames": len(frames), "train_cut": t1, "coverage": inp.get("coverage"), "report": report}
    with open(os.path.join(args.out_dir, "trained_forecasters.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"forecasters trained on {t1}/{len(frames)} periods ({(inp.get('coverage') or {}).get('granularity')}) "
          f"→ {args.out_dir}/trained_forecasters.json")
    for tgt, r in report.items():
        print(f"  {tgt:18} {r['model_used']:16} {r['metric']}={r['holdout_metric']:.4f} "
              f"naive={r['naive_metric']:.4f} [{'WIN' if r['beats_naive'] else 'naive-kept'}]")


if __name__ == "__main__":
    main()
