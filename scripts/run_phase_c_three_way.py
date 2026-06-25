#!/usr/bin/env python3
"""Phase C — fair three-way benchmark on the public LLM-serving traces.

Runs the deployable replica-scaling policies on the Azure LLM 2024 + BurstGPT
public-trace fixtures, on a PURE ON-DEMAND denominator (no spot discount), in a
three-way Current-Main vs Best-Aurelius vs Candidate comparison, with seed +
trace-content hash serialized into the artifact.

Reuses the frozen evaluator (forecasted_mcs.evaluate_c_schedule) and the
canonical ObjectiveLayer ranking. Directional simulator evidence only — NOT
production savings (docs/RESULTS.md §8).

Run: python -m scripts.run_phase_c_three_way
"""

from __future__ import annotations

import json

from aurelius.benchmarks.forecasted_mcs import GPU_HOUR_USD, evaluate_c_schedule
from aurelius.benchmarks.phase_c import (
    ROLE_BEST_AURELIUS,
    ROLE_CANDIDATE,
    ROLE_CURRENT_MAIN,
    run_three_way,
    standard_replica_scaling_arms,
)
from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_FIXTURE,
    calibrate_time_warp,
    load_burstgpt_serving_requests,
    load_serving_requests,
)

# Canonical serving params (identical to scripts/run_forecasted_mcs_backtest.py).
JOB_LIMIT = 5880
SERVERS = 4
TARGET_RHO = 0.85
TICK_S = 60.0
SLA_S = 10.0
MCS_GATE = 9.5
SEED = 42

TRACES = [
    ("azure_llm_2024", DEFAULT_AZURE_FIXTURE, load_serving_requests),
    ("burstgpt", DEFAULT_BURSTGPT_FIXTURE, load_burstgpt_serving_requests),
]


def _fixed_c_reference(raw, warp, c):
    """Fixed-c static autoscale — the WEAK baseline the prior +54% claim used."""
    n_ticks = max(1, int((raw[-1][0] / warp) / TICK_S) + 1)
    return evaluate_c_schedule(
        raw, [c] * n_ticks, TICK_S, warp, SLA_S,
        policy=f"fixed_c{c}", uses_future_info=False, deployable=True,
        classification="deployable_static_baseline",
    )


def main() -> None:
    print("=" * 78)
    print("PHASE C — fair three-way, ON-DEMAND denominator (no spot), public traces")
    print(f"  servers(calib)={SERVERS}  target_rho={TARGET_RHO}  tick={TICK_S}s  "
          f"sla={SLA_S}s  gate={MCS_GATE}%  GPU=${GPU_HOUR_USD}/hr  seed={SEED}")
    print("=" * 78)

    all_results = {}
    for trace_id, path, loader in TRACES:
        raw = loader(path, limit=JOB_LIMIT)
        warp = calibrate_time_warp(raw, servers=SERVERS, target_rho=TARGET_RHO)

        arms = standard_replica_scaling_arms(raw, TICK_S, warp, SLA_S, mcs_gate=MCS_GATE)
        res = run_three_way(
            raw, arms, tick_seconds=TICK_S, warp=warp, sla_s=SLA_S,
            seed=SEED, trace_id=trace_id,
            notes=("on-demand denominator (no spot)", f"mcs_gate={MCS_GATE}%"),
        )
        all_results[trace_id] = res.to_dict()

        print(f"\n### {trace_id}  ({len(raw):,} reqs · warp={warp:.4f} · "
              f"trace_hash={res.trace_hash} · seed={res.seed})")
        print(f"{'role':14s} {'policy':24s} {'gp/$':>11s} {'cost$':>9s} "
              f"{'GPU-h':>7s} {'SLA-safe':>9s} {'viol':>5s}")
        for a in res.arms:
            print(f"{a.role:14s} {a.name:24s} {a.goodput_per_dollar:11.2f} "
                  f"{a.cost_usd:9.2f} {a.gpu_hours:7.2f} {a.n_sla_safe:9d} "
                  f"{a.sla_violations:5d}")

        # Reference: fixed-c=4 static autoscale (the WEAK baseline prior claims used).
        fixed = _fixed_c_reference(raw, warp, SERVERS)
        fixed_gpd = fixed.goodput_per_dollar
        print(f"{'reference':14s} {'fixed_c4 (weak)':24s} {fixed_gpd:11.2f} "
              f"{fixed.cost_usd:9.2f} {fixed.gpu_hours:7.2f} {fixed.n_sla_safe:9d} "
              f"{fixed.sla_violations:5d}")

        by_role = {a.role: a for a in res.arms}
        cm = by_role[ROLE_CURRENT_MAIN]
        ba = by_role[ROLE_BEST_AURELIUS]
        cand = by_role[ROLE_CANDIDATE]
        best_dep = max(res.arms, key=lambda a: a.goodput_per_dollar)
        def pct(new_gpd, base_gpd):
            return (new_gpd - base_gpd) / base_gpd * 100.0 if base_gpd else 0.0
        print(f"  best deployable ({best_dep.name}) vs current_main (reactive_lag1, FAIR): "
              f"{pct(best_dep.goodput_per_dollar, cm.goodput_per_dollar):+.2f}% goodput/$  "
              f"(GPU-h {best_dep.gpu_hours:.2f} vs {cm.gpu_hours:.2f})")
        print(f"  best deployable vs fixed_c4 (WEAK baseline): "
              f"{pct(best_dep.goodput_per_dollar, fixed_gpd):+.2f}% goodput/$")
        print(f"  candidate vs current_main: "
              f"{pct(cand.goodput_per_dollar, cm.goodput_per_dollar):+.2f}% goodput/$")
        print(f"  deployable winner: {res.deployable_winner}")
        all_results[trace_id]["fixed_c4_reference"] = fixed.to_dict()

    out = "research/results/phase_c_three_way_public_traces.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[artifact] {out}  (seed + trace_hash serialized per trace)")


if __name__ == "__main__":
    main()
