#!/usr/bin/env python3
"""Canonical dataset — Alibaba-calibrated class mix vs the arbitrary overlay.

Grounds the best-effort fraction in the REAL Alibaba cluster-trace-gpu-v2023 QoS
distribution (LS/BE/Burstable) instead of an arbitrary 40% overlay, and re-runs
the closed-loop compounding A/B at the production-grounded ratio — the honest
test of whether the earlier +9.00% compounding survives a realistic class mix.

Run: python -m scripts.run_canonical_calibration_ab
"""

from __future__ import annotations

import json

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    calibrate_time_warp,
    load_serving_requests,
)
from aurelius.datasets.calibration import default_alibaba_class_mix
from aurelius.datasets.canonical import assemble_calibrated, augment_with_best_effort
from aurelius.optimizer.unified_replay import run_unified_combination

JOB_LIMIT = 5880
TICK_S = 60.0
SLA_S = 10.0


def _ab(jobs, tag):
    res = run_unified_combination(jobs, tick_seconds=TICK_S, sla_s=SLA_S, trace_id=tag)
    bs = (res.best_single_gpd - res.base_gpd) / res.base_gpd * 100.0
    bm = (res.best_multi_gpd - res.base_gpd) / res.base_gpd * 100.0
    print(f"  {tag:42s} best_single={bs:+6.2f}%  best_multi={bm:+6.2f}%  "
          f"{res.interaction.upper()}")
    return {"tag": tag, "best_single_pct": round(bs, 3), "best_multi_pct": round(bm, 3),
            "interaction": res.interaction, "result": res.to_dict()}


def main() -> None:
    raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=JOB_LIMIT)
    warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
    mix = default_alibaba_class_mix()

    print("=" * 80)
    print("CANONICAL DATASET — Alibaba-CALIBRATED class mix vs arbitrary overlay")
    print(f"  Azure LLM 2024 · {len(raw)} reqs · warp={warp:.4f} · on-demand")
    print(f"  Alibaba qos mix: BE-by-count={mix.best_effort_fraction_by_count:.3f}  "
          f"BE-by-gpu-work={mix.best_effort_fraction_by_gpu_work:.4f}  "
          f"(source={mix.source}, tier={mix.tier})")
    print("=" * 80)

    out = {"class_mix": mix.to_dict(), "runs": []}

    print("\n### Does the +9.00% compounding survive a PRODUCTION-GROUNDED class mix?")
    j40, _ = augment_with_best_effort(raw, warp=warp, fraction=0.40, token_multiplier=1.5)
    out["runs"].append(_ab(j40, "overlay 40% (the earlier headline)"))
    jc, _, _ = assemble_calibrated(raw, warp=warp, weight="count", token_multiplier=1.5)
    out["runs"].append(_ab(jc, f"Alibaba-calibrated by COUNT ({mix.best_effort_fraction_by_count:.3f})"))
    jw, _, _ = assemble_calibrated(raw, warp=warp, weight="gpu_work", token_multiplier=1.5)
    out["runs"].append(_ab(jw, f"Alibaba-calibrated by GPU-WORK ({mix.best_effort_fraction_by_gpu_work:.4f})"))

    print("\n### Sensitivity — best-effort fraction sweep (magnitude is fraction-bound)")
    print(f"  {'fraction':>10} {'best_single':>12} {'best_multi':>11} {'verdict':>14}")
    sweep = []
    for frac in (0.05, 0.10, 0.175, 0.20, 0.30, 0.40):
        jj, _ = augment_with_best_effort(raw, warp=warp, fraction=frac, token_multiplier=1.5)
        res = run_unified_combination(jj, tick_seconds=TICK_S, sla_s=SLA_S, trace_id="sweep")
        bs = (res.best_single_gpd - res.base_gpd) / res.base_gpd * 100.0
        bm = (res.best_multi_gpd - res.base_gpd) / res.base_gpd * 100.0
        print(f"  {frac:10.3f} {bs:+11.2f}% {bm:+10.2f}% {res.interaction:>14}")
        sweep.append({"fraction": frac, "best_single_pct": round(bs, 3),
                      "best_multi_pct": round(bm, 3), "interaction": res.interaction})
    out["sensitivity_sweep"] = sweep

    print("\n" + "=" * 80)
    print("VERDICT: the +9.00% required a best-effort tier ~2x the real production")
    print("ratio. At the Alibaba-grounded mix the serving-lever compounding is")
    print("neutral-to-negative. The MECHANISM is real; the MAGNITUDE needs real")
    print("best-effort SERVING economics (pilot) or a different real lever (KV/energy).")
    print("=" * 80)

    path = "research/results/canonical_dataset_alibaba_calibration.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[artifact] {path}")


if __name__ == "__main__":
    main()
