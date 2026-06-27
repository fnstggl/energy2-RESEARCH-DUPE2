#!/usr/bin/env python3
"""Fair backtest of policies through the canonical environment (Phase 4).

Runs the weak FIFO reference + SLA-aware + greedy/packing + current AureliusOptimizer
canonical config + a candidate policy through a fresh CanonicalMultiPlaneEnvironment,
scores every arm via the optimizer's ObjectiveLayer (SLA-safe goodput/$), picks the
fair (non-weak) baseline, and reports the per-arm metrics + whether a headline claim
is allowed (fair baseline + held-out validation + no oracle).

Usage:
  python -m scripts.run_fair_backtest
  python -m scripts.run_fair_backtest --json --limit 4000
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from aurelius.environment.ingestion.azure import ingest_azure, to_serving_raw
from aurelius.environment.optimizer_adapter import fair_backtest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MOONCAKE = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_sample.csv")
_DEFAULT_PROCESSED = os.environ.get(
    "V2026_PROCESSED_DIR", os.path.join(_REPO, "data", "external", "alibaba_gpu_v2026", "processed"))


def _azure_hourly(limit: int) -> dict:
    reqs, _ = ingest_azure(limit=limit)
    raw = to_serving_raw(reqs)
    if not raw:
        return {}
    t0 = raw[0][0]
    hourly: dict = defaultdict(list)
    for arr, tok in raw:
        hourly[int((arr - t0) // 3600)].append((arr, tok))
    return dict(hourly)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default=_DEFAULT_PROCESSED)
    ap.add_argument("--mooncake", default=_DEFAULT_MOONCAKE)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--sla-s", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    azure_hourly = _azure_hourly(args.limit)
    report = fair_backtest(
        azure_hourly,
        env_kwargs={"mooncake_path": args.mooncake, "processed_dir": args.processed_dir,
                    "sla_s": args.sla_s}).to_dict()

    if args.json:
        print(json.dumps(report, indent=2))
        return

    print("ranking (SLA-safe goodput/$, via AureliusOptimizer.objective):")
    for name, score in report["ranking"]:
        print(f"  {name:28} {score:>14.1f}")
    print(f"\nfair baseline: {report['fair_baseline']} (strongest non-weak)")
    print(f"candidate: {report['candidate']}  →  {report['candidate_vs_baseline_pct']:+.2f}% vs fair baseline")
    print(f"headline claim allowed: {report['headline_claim_allowed']}")
    print(f"gate: {report['gate']}")
    print(f"env validation: {report['env_validation']}")
    print("\nper-arm metrics:")
    for name, a in report["arms"].items():
        print(f"  {name:28} gp/$={a['goodput_per_dollar']:>12.1f}  sla_viol={a['sla_violation_rate']:.3f}  "
              f"q_p95={a['queue_delay_p95']:.2f}s  kv={a['kv_hit_rate']:.2f}  "
              f"cost/req={a['cost_per_sla_safe_request']:.5f}")


if __name__ == "__main__":
    main()
