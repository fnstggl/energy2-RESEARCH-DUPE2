#!/usr/bin/env python3
"""Unified replay engine (Phase 1b-A) — the compounding A/B on real public data.

Runs the FULL closed-loop lever lattice (capacity × ordering × admission) through
ONE discrete-event loop on ONE evolving cluster state, twice:

  A) SINGLE-CLASS — the raw Azure LLM 2024 trace (every request latency-critical,
     exactly what every public serving trace gives you today), and
  B) MULTI-CLASS  — the same real Azure spine + a documented best-effort batch
     overlay (the one structural signal the public traces strip out).

The optimizer code is byte-identical across A and B; the ONLY thing that changes
is the data's workload-class structure. So if combining the levers is
SUBSTITUTIVE in A but COMPOUNDS in B, the no-compounding result was a DATA
problem, not an optimizer problem — which is the question under test.

On-demand denominator (no spot, no oracle). Deterministic + reproducible
(jobs_hash). Directional simulator only — not production savings
(``docs/RESULTS.md`` §8).

Run: python -m scripts.run_unified_replay_ab
"""

from __future__ import annotations

import json

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    calibrate_time_warp,
    load_serving_requests,
)
from aurelius.datasets.canonical import (
    _causal_predicted,
    augment_with_best_effort,
    to_jobs,
)
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.unified_replay import CLASS_LATENCY

JOB_LIMIT = 5880
SERVERS = 4
TARGET_RHO = 0.85
TICK_S = 60.0
SLA_S = 10.0
BE_FRACTION = 0.4
BE_TOKEN_MULT = 1.5


def _print_lattice(res) -> None:
    print(f"  n_jobs={res.n_jobs} (lc={res.n_latency_critical} be={res.n_best_effort}) "
          f"jobs_hash={res.jobs_hash}")
    print(f"  {'levers':9s} {'gp/$':>11s} {'cost$':>9s} {'c_mean':>7s} "
          f"{'SLA-safe':>9s} {'defer':>6s}  {'vs base':>8s}")
    for c in sorted(res.cells, key=lambda c: -c.goodput_per_dollar):
        vs = (c.goodput_per_dollar - res.base_gpd) / res.base_gpd * 100.0
        print(f"  {c.label:9s} {c.goodput_per_dollar:11.1f} {c.cost_usd:9.2f} "
              f"{c.c_mean:7.2f} {c.n_sla_safe:9d} {c.n_deferred:6d}  {vs:+7.2f}%")
    bs = (res.best_single_gpd - res.base_gpd) / res.base_gpd * 100.0
    bm = (res.best_multi_gpd - res.base_gpd) / res.base_gpd * 100.0
    print(f"  best single : {res.best_single_label:7s} ({bs:+.2f}% vs base)")
    print(f"  best multi  : {res.best_multi_label:7s} ({bm:+.2f}% vs base)")
    print(f"  INTERACTION : {res.interaction.upper()}  "
          f"(combining {'BEATS' if res.compounding else 'does NOT beat'} the best single lever)")


def main() -> None:
    opt = AureliusOptimizer()
    raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=JOB_LIMIT)
    warp = calibrate_time_warp(raw, servers=SERVERS, target_rho=TARGET_RHO)
    raw_sorted = sorted(raw, key=lambda r: r[0])

    print("=" * 80)
    print("UNIFIED REPLAY ENGINE (Phase 1b-A) — closed-loop compounding A/B")
    print(f"  Azure LLM 2024 · {len(raw)} reqs · warp={warp:.4f} · tick={TICK_S}s · "
          f"sla={SLA_S}s · GPU=$2/hr · on-demand")
    print("  levers: C=backlog-aware capacity · O=abs-conformal SRPT · A=class-aware admission")
    print("=" * 80)

    # A) single-class (raw Azure, every request latency-critical)
    pred = _causal_predicted(raw_sorted)
    single = to_jobs(raw_sorted, warp=warp, cls=CLASS_LATENCY, predicted_tokens=pred)
    res_single = opt.optimize_joint_closed_loop(
        single, tick_seconds=TICK_S, sla_s=SLA_S, trace_id="azure_single_class",
        notes=("raw Azure — single-class control arm",))
    print("\n### A) SINGLE-CLASS  (raw Azure — what public traces give you)")
    _print_lattice(res_single)

    # B) multi-class (Azure spine + documented best-effort overlay)
    multi, manifest = augment_with_best_effort(
        raw, warp=warp, fraction=BE_FRACTION, token_multiplier=BE_TOKEN_MULT)
    res_multi = opt.optimize_joint_closed_loop(
        multi, tick_seconds=TICK_S, sla_s=SLA_S, trace_id="azure_multi_class",
        notes=("Azure spine + synthetic best-effort overlay",))
    print("\n### B) MULTI-CLASS  (Azure spine + best-effort batch overlay)")
    _print_lattice(res_multi)

    # Verdict.
    print("\n" + "=" * 80)
    print("VERDICT — same optimizer code, only the data's class structure differs:")
    print(f"  single-class : {res_single.interaction.upper()}")
    print(f"  multi-class  : {res_multi.interaction.upper()}")
    if res_multi.compounding and not res_single.compounding:
        print("  => Compounding appears ONLY when the data carries workload-class")
        print("     structure. The no-compounding result is a DATA issue, not an")
        print("     optimizer issue. The closed loop was necessary; the data is binding.")
    print("=" * 80)

    out = {
        "config": {
            "trace": "azure_llm_2024", "job_limit": JOB_LIMIT, "servers": SERVERS,
            "target_rho": TARGET_RHO, "tick_s": TICK_S, "sla_s": SLA_S, "warp": warp,
            "be_fraction": BE_FRACTION, "be_token_mult": BE_TOKEN_MULT,
        },
        "single_class": res_single.to_dict(),
        "multi_class": res_multi.to_dict(),
        "manifest": manifest.to_dict(),
    }
    path = "research/results/unified_replay_compounding_ab.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[artifact] {path}  (full lattice + jobs_hash + manifest serialized)")


if __name__ == "__main__":
    main()
