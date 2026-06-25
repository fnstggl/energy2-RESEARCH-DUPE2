#!/usr/bin/env python3
"""Joint optimization — combination search on the public LLM-serving traces.

Runs the deployable serving levers (capacity / ordering / admission) TOGETHER on
the Azure LLM 2024 + BurstGPT public traces, on the pure on-demand denominator,
and reports the full lever lattice + whether the best combination COMPOUNDS or is
SUBSTITUTIVE — the empirical answer to "does combining optimizers raise goodput/$?"

Run: python -m scripts.run_joint_combination_search
"""

from __future__ import annotations

import json

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    calibrate_time_warp,
    load_burstgpt_serving_requests,
    load_serving_requests,
)
from aurelius.optimizer import AureliusOptimizer

JOB_LIMIT = 5880
SERVERS = 4
TARGET_RHO = 0.85
TICK_S = 60.0
SLA_S = 10.0
SEED = 42

TRACES = [
    ("azure_llm_2024", DEFAULT_AZURE_FIXTURE, load_serving_requests),
    ("burstgpt", DEFAULT_BURSTGPT_FIXTURE, load_burstgpt_serving_requests),
]


def main() -> None:
    opt = AureliusOptimizer()
    print("=" * 78)
    print("JOINT OPTIMIZATION — combination search, ON-DEMAND denominator, public traces")
    print("  levers: C=forecasted_mcs capacity · O=abs-conformal SRPT · A=peak-shave admission")
    print(f"  target_rho={TARGET_RHO}  tick={TICK_S}s  sla={SLA_S}s  GPU=$2/hr  seed={SEED}")
    print("=" * 78)

    out_all = {}
    for trace_id, path, loader in TRACES:
        raw = loader(path, limit=JOB_LIMIT)
        warp = calibrate_time_warp(raw, servers=SERVERS, target_rho=TARGET_RHO)
        res = opt.optimize_joint(
            raw, tick_seconds=TICK_S, warp=warp, sla_s=SLA_S,
            seed=SEED, trace_id=trace_id,
        )
        out_all[trace_id] = res.to_dict()

        print(f"\n### {trace_id}  ({len(raw):,} reqs · warp={warp:.4f} · "
              f"trace_hash={res.trace_hash} · seed={res.seed})")
        print(f"{'levers':9s} {'gp/$':>11s} {'cost$':>9s} {'GPU-h':>7s} "
              f"{'SLA-safe':>9s} {'viol':>5s}  {'vs base':>8s}")
        for c in sorted(res.cells, key=lambda c: -c.goodput_per_dollar):
            vs_base = (c.goodput_per_dollar - res.base_gpd) / res.base_gpd * 100.0
            print(f"{c.label:9s} {c.goodput_per_dollar:11.1f} {c.cost_usd:9.2f} "
                  f"{c.gpu_hours:7.2f} {c.n_sla_safe:9d} {c.sla_violations:5d}  "
                  f"{vs_base:+7.2f}%")
        ov = (res.best_overall_gpd - res.base_gpd) / res.base_gpd * 100.0
        sng = (res.best_single_gpd - res.base_gpd) / res.base_gpd * 100.0
        print(f"  best single lever : {res.best_single_label:7s} ({sng:+.2f}% vs base)")
        print(f"  best combination  : {res.best_overall_label:7s} ({ov:+.2f}% vs base)")
        print(f"  INTERACTION       : {res.interaction.upper()}  "
              f"(combining {'beats' if res.compounding else 'does NOT beat'} the best single lever)")

    out = "research/results/joint_combination_search_public_traces.json"
    with open(out, "w") as f:
        json.dump(out_all, f, indent=2)
    print(f"\n[artifact] {out}  (seed + trace_hash + full lattice serialized)")


if __name__ == "__main__":
    main()
