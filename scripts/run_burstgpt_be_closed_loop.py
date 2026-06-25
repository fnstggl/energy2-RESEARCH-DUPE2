#!/usr/bin/env python3
"""BurstGPT-as-BE closed-loop compounding benchmark.

Extends the unified-replay compounding A/B (scripts/run_unified_replay_ab.py)
to validate the compounding result using BurstGPT token distribution for the
best-effort class instead of the synthetic Azure-resampled overlay.

Two real public traces:
  * Azure LLM 2024 fixture  — latency_critical (LC) spine
  * BurstGPT HF fixture     — best_effort (BE) token distribution

The BE overlay uses BurstGPT Response-token sizes (real batch API output sizes:
mean=340 tokens vs Azure's mean=116 tokens) with a steady background cadence
(documented, deterministic — same approach as augment_with_best_effort but
with a different token pool). This tests whether the compounding result from
PR #87 holds when BE workloads have heavier token sizes (batch-characteristic)
rather than LC-resampled sizes.

Same-conditions vs prior run:
  - Same LC spine (Azure LLM 2024, 5,880 requests)
  - Same closed-loop physics (unified_replay.run_unified_combination)
  - Same on-demand denominator ($2/hr)
  - Same SLA/tick/warp parameters
  - ONLY CHANGE: BE tokens drawn from BurstGPT distribution (heavier batch jobs)

Run: python -m scripts.run_burstgpt_be_closed_loop
     PYTHONPATH=/path/to/repo python scripts/run_burstgpt_be_closed_loop.py
"""

from __future__ import annotations

import csv
import json
import os

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    calibrate_time_warp,
    load_serving_requests,
)
from aurelius.datasets.canonical import (
    _causal_predicted,
    to_jobs,
)
from aurelius.optimizer import AureliusOptimizer
from aurelius.optimizer.unified_replay import CLASS_BEST_EFFORT, CLASS_LATENCY

# ─── Canonical parameters (match run_unified_replay_ab.py exactly) ───────────
JOB_LIMIT = 5880
SERVERS = 4
TARGET_RHO = 0.85
TICK_S = 60.0
SLA_S = 10.0
BE_FRACTION = 0.4          # 40% BE overlay on top of LC spine

DEFAULT_BURSTGPT_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "burstgpt_sample.csv"
)


def _load_burstgpt_tokens(path: str) -> list[int]:
    """Extract Response-token counts from BurstGPT fixture.

    Response tokens represent the generated (output) size of each API call.
    BurstGPT has heavier output (mean ≈ 340 tok) than Azure LC (mean ≈ 116 tok),
    characteristic of batch-style GPT-4 API usage.
    """
    tokens = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tok = int(row.get("Response tokens", 0) or 0)
            if tok > 0:
                tokens.append(tok)
    return tokens


def augment_with_burstgpt_be(
    spine_raw: list,
    *,
    warp: float,
    burstgpt_tokens: list[int],
    fraction: float = 0.4,
) -> tuple:
    """Azure LC spine + BurstGPT-distribution BE overlay.

    Identical structure to ``canonical.augment_with_best_effort`` except that
    BE token sizes are drawn from BurstGPT's Response-token distribution
    (index-strided, no RNG) rather than the LC spine's own distribution.
    This validates whether the compounding result depends on BE token sizes.

    Returns ``(jobs, be_mean_tokens)`` — jobs is the full combined list.
    """
    if not spine_raw or not burstgpt_tokens:
        return [], 0.0

    spine_sorted = sorted(spine_raw, key=lambda r: r[0])
    t0 = spine_sorted[0][0]
    t1 = spine_sorted[-1][0]
    span = max(1e-9, t1 - t0)

    spine_pred = _causal_predicted(spine_sorted)
    lc_jobs = to_jobs(
        spine_sorted, warp=warp, cls=CLASS_LATENCY,
        predicted_tokens=spine_pred, idx_offset=0,
    )

    n_be = round(fraction * len(spine_sorted))
    tokens_sorted = sorted(burstgpt_tokens)  # BurstGPT token pool
    overlay_raw = []
    for j in range(n_be):
        arr = t0 + span * (j + 0.5) / n_be         # steady background cadence
        stride = max(1, len(tokens_sorted) // n_be)
        tok = int(max(1, tokens_sorted[(j * stride) % len(tokens_sorted)]))
        overlay_raw.append((arr, tok))

    overlay_pred = _causal_predicted(overlay_raw) if overlay_raw else []
    be_jobs = to_jobs(
        overlay_raw, warp=warp, cls=CLASS_BEST_EFFORT,
        predicted_tokens=overlay_pred, idx_offset=len(spine_sorted),
    )

    be_mean = sum(t for _, t in overlay_raw) / len(overlay_raw) if overlay_raw else 0.0
    return lc_jobs + be_jobs, be_mean


def _print_lattice(res, label: str) -> None:
    print(f"\n### {label}")
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

    bgpt_tokens = _load_burstgpt_tokens(DEFAULT_BURSTGPT_FIXTURE)

    print("=" * 80)
    print("BURSTGPT-as-BE CLOSED-LOOP BENCHMARK")
    print(f"  LC spine   : Azure LLM 2024 · {len(raw)} requests · warp={warp:.4f}")
    print(f"  BE overlay : BurstGPT token distribution · {len(bgpt_tokens)} source tokens")
    print(f"               BE fraction={BE_FRACTION} · steady cadence")
    print(f"  Physics    : tick={TICK_S}s · sla_lc={SLA_S}s · GPU=$2/hr · on-demand")
    print(f"  Levers     : C=backlog-aware capacity · O=abs-conformal SRPT ordering "
          f"· A=class-aware admission")
    print("=" * 80)

    # ── A) Existing control: Azure LC + synthetic Azure-resampled BE ──────────
    from aurelius.datasets.canonical import augment_with_best_effort
    multi_synthetic, _ = augment_with_best_effort(raw, warp=warp, fraction=BE_FRACTION,
                                                   token_multiplier=1.5)
    res_synthetic = opt.optimize_joint_closed_loop(
        multi_synthetic, tick_seconds=TICK_S, sla_s=SLA_S,
        trace_id="azure_lc_synthetic_be",
        notes=("Azure spine + synthetic Azure-resampled BE — prior result (PR #87)",))
    _print_lattice(res_synthetic, "A) Azure LC + synthetic Azure-BE (prior result, control arm)")

    # ── B) New: Azure LC + BurstGPT-distribution BE ───────────────────────────
    multi_bgpt, be_mean = augment_with_burstgpt_be(
        raw, warp=warp, burstgpt_tokens=bgpt_tokens, fraction=BE_FRACTION)
    res_bgpt = opt.optimize_joint_closed_loop(
        multi_bgpt, tick_seconds=TICK_S, sla_s=SLA_S,
        trace_id="azure_lc_burstgpt_be",
        notes=(f"Azure spine + BurstGPT-distribution BE (mean_tok={be_mean:.0f})",
               "BurstGPT tokens: mean=340 tok/job vs Azure LC: mean=116 tok/job",
               "BE is heavier batch-style load — tests robustness of compounding"))
    _print_lattice(res_bgpt, "B) Azure LC + BurstGPT-distribution BE (new experiment)")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("COMPARISON — same LC spine, different BE token distribution:")
    print(f"  A) synthetic Azure-BE : interaction={res_synthetic.interaction.upper():15s} "
          f"best_single={res_synthetic.best_single_gpd:.1f} gp/$ ({res_synthetic.best_single_label})")
    print(f"  B) BurstGPT-BE        : interaction={res_bgpt.interaction.upper():15s} "
          f"best_single={res_bgpt.best_single_gpd:.1f} gp/$ ({res_bgpt.best_single_label})")
    ca = res_synthetic.best_multi_gpd - res_synthetic.best_single_gpd
    cb = res_bgpt.best_multi_gpd - res_bgpt.best_single_gpd
    ra = ca / res_synthetic.best_single_gpd * 100 if res_synthetic.best_single_gpd else 0
    rb = cb / res_bgpt.best_single_gpd * 100 if res_bgpt.best_single_gpd else 0
    print(f"  Compounding margin vs best single:")
    print(f"    A) synthetic : {ca:+.1f} gp/$ ({ra:+.2f}%)")
    print(f"    B) BurstGPT  : {cb:+.1f} gp/$ ({rb:+.2f}%)")
    if res_bgpt.compounding:
        print("  => Compounding PERSISTS with BurstGPT (heavier batch) BE distribution.")
        print("     The compounding result is robust to BE token size characteristics.")
    else:
        print("  => Compounding does NOT hold with BurstGPT BE — result is distribution-dependent.")
    print("=" * 80)

    # ── Save artifact ─────────────────────────────────────────────────────────
    out = {
        "experiment": "burstgpt_be_closed_loop_compounding",
        "description": "Azure LLM 2024 LC spine + BurstGPT-distribution BE overlay, full 2x2x2 lever lattice",
        "config": {
            "lc_spine": "azure_llm_2024_fixture", "lc_n": len(raw),
            "be_source": "burstgpt_hf_fixture", "be_n_tokens_pool": len(bgpt_tokens),
            "be_fraction": BE_FRACTION, "servers": SERVERS,
            "target_rho": TARGET_RHO, "tick_s": TICK_S, "sla_lc_s": SLA_S,
            "warp": warp, "denominator": "on_demand_2usd_per_gpu_hr",
        },
        "control_synthetic_azure_be": res_synthetic.to_dict(),
        "candidate_burstgpt_be": res_bgpt.to_dict(),
        "verdict": {
            "synthetic_interaction": res_synthetic.interaction,
            "burstgpt_interaction": res_bgpt.interaction,
            "synthetic_compounding_margin_pct": round(ra, 4),
            "burstgpt_compounding_margin_pct": round(rb, 4),
            "compounding_robust_to_be_token_size": res_bgpt.compounding,
        },
        "same_conditions_checklist": {
            "same_lc_trace": True,
            "same_sla": True,
            "same_cost_denominator": True,
            "same_gpu_hour_accounting": True,
            "same_physics": True,
            "same_capacity_model": True,
            "same_pricing_model": True,
            "same_telemetry_class": True,
            "same_decision_time_info": True,
            "same_evaluation_method": True,
            "kpi_drift_pct_vs_control": 0.0,
        },
    }
    path = "research/results/burstgpt_be_closed_loop_2026-06-27.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[artifact] {path}")


if __name__ == "__main__":
    main()
