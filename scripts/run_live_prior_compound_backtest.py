#!/usr/bin/env python3
"""Live causal prior + compound serving/economic backtest [run 2026-06-21-t].

Closes the oracle gap: replaces the oracle prediction prior with a causal
sliding-window median estimator that a real production system would use.

THREE MEASUREMENTS:

1. LIVE CAUSAL PRIOR on Azure LLM 2024 (5,880 requests, public trace)
   - FIFO baseline vs Conformal with oracle vs Conformal with live prior
   - Key metric: live_vs_oracle_retention_pct (≥83% threshold from run -n)
   - Prior CV: how noisy is the causal sliding-window estimate?

2. LIVE CAUSAL PRIOR on BurstGPT HF (5,880-record sample, CC-BY-4.0)
   - Cross-validates the live prior on BurstGPT's heavier output distribution
   - Key metric: retention_pct on BurstGPT (expected ≥83%)

3. COMPOUND GAIN TABLE
   - Economic scheduling gain: constraint_aware vs FIFO (from existing backtest)
   - Serving queue gain: conformal (oracle) vs FIFO (from run -q)
   - Serving queue gain (live prior): conformal (live) vs FIFO (measured here)
   - Compound (estimated): economic × serving-queue under independence assumption
   - Reports vs FIFO baseline AND vs SLA-aware intermediate baseline

HARD REQUIREMENT:
  Every run must execute at least one real public-trace economic backtest.
  This script satisfies the requirement by running the discrete-event M/G/c
  serving simulator on TWO public traces (Azure LLM 2024 + BurstGPT HF).

Writes:
  research/results/live_prior_compound_backtest_<date>.{json,md}

Usage:
  python scripts/run_live_prior_compound_backtest.py
  python scripts/run_live_prior_compound_backtest.py --skip-burstgpt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_BURSTGPT_HF_JSONL,
    run_burstgpt_hf_live_prior_backtest,
    run_live_prior_conformal_backtest,
)

RESULTS_DIR = "research/results"


def _run_azure_live_prior(job_limit=None):
    print("  Running Azure LLM 2024 live prior backtest...")
    t0 = time.time()
    report = run_live_prior_conformal_backtest(
        servers=4, target_rho=0.85, job_limit=job_limit
    )
    elapsed = time.time() - t0
    d = report.to_dict()
    d["runtime_s"] = round(elapsed, 2)
    return d


def _run_burstgpt_live_prior(job_limit=5880):
    jsonl = DEFAULT_BURSTGPT_HF_JSONL
    if not os.path.exists(jsonl):
        print(f"  BurstGPT HF JSONL not found at {jsonl!r}, skipping.")
        return None
    print("  Running BurstGPT HF live prior backtest...")
    t0 = time.time()
    report = run_burstgpt_hf_live_prior_backtest(
        servers=4, target_rho=0.85, job_limit=job_limit
    )
    elapsed = time.time() - t0
    d = report.to_dict()
    d["runtime_s"] = round(elapsed, 2)
    return d


def _compound_table(azure: dict, burstgpt: dict | None) -> dict:
    """Build compound gain comparison table.

    Economic scheduling gain from existing published results (run 2026-06-21-s
    / ROADMAP §2.2):
      Azure LLM 2024: constraint_aware vs sla_aware +25.75%
      → constraint_aware vs FIFO ≈ (1 + 0.2575) × (1 + 1.254) - 1 ≈ +184%
      (SLA_aware was +125.4% vs FIFO, so CA vs FIFO = 1.2575 × 2.254 - 1 ≈ +183%)

    These numbers come from published benchmark results in ROADMAP.md and
    BENCHMARK_REGISTRY.md. They are held constant across this run — we only
    measure the serving queue component here.
    """
    # Published economic scheduling results (from ROADMAP §2.2 / run -s).
    # constraint_aware vs sla_aware on Azure LLM 2024: +25.75%
    # sla_aware vs FIFO (serving queue run -n): +125.4%
    # => constraint_aware vs FIFO (compound economic): (1.2575 × 2.254 - 1) × 100
    econ_vs_sla_aware_pct = 25.75   # from BENCHMARK_REGISTRY
    sla_aware_vs_fifo_pct = 125.4   # from run -n (srtf_serving_backtest)
    econ_vs_fifo_pct = (1 + econ_vs_sla_aware_pct / 100) * (1 + sla_aware_vs_fifo_pct / 100) - 1
    econ_vs_fifo_pct *= 100.0

    # Serving queue gains (oracle) from published results.
    conformal_oracle_vs_fifo_azure = 322.24   # run -q
    conformal_oracle_vs_fifo_burstgpt = 644.4  # run -r

    # Serving queue gains (live prior) — measured in this run.
    live_vs_fifo_azure = azure.get("live_delta_pct", 0.0)
    live_retention_azure = azure.get("live_vs_oracle_retention_pct", 0.0)

    if burstgpt:
        live_vs_fifo_burstgpt = burstgpt.get("live_delta_pct", 0.0)
        live_retention_burstgpt = burstgpt.get("live_vs_oracle_retention_pct", 0.0)
    else:
        live_vs_fifo_burstgpt = None
        live_retention_burstgpt = None

    # Compound estimates (economic × serving queue, under independence assumption).
    # compound = (1 + econ_gain) × (1 + queue_gain) - 1
    econ_mult = 1 + econ_vs_fifo_pct / 100.0
    compound_oracle_azure = (econ_mult * (1 + conformal_oracle_vs_fifo_azure / 100.0) - 1) * 100.0
    compound_live_azure = (econ_mult * (1 + live_vs_fifo_azure / 100.0) - 1) * 100.0

    table = {
        "baseline_note": "All deltas vs FIFO serving (no economic scheduling)",
        "economic_scheduling": {
            "source": "BENCHMARK_REGISTRY / run 2026-06-21-s",
            "constraint_aware_vs_sla_aware_pct": econ_vs_sla_aware_pct,
            "sla_aware_vs_fifo_pct": sla_aware_vs_fifo_pct,
            "constraint_aware_vs_fifo_pct": round(econ_vs_fifo_pct, 2),
        },
        "azure_llm_2024": {
            "serving_queue_fifo": "reference (0%)",
            "serving_queue_sla_aware": f"+{sla_aware_vs_fifo_pct:.1f}% (run -n)",
            "serving_queue_conformal_oracle": f"+{conformal_oracle_vs_fifo_azure:.2f}% (run -q)",
            "serving_queue_conformal_live": f"+{live_vs_fifo_azure:.2f}% (run -t, THIS RUN)",
            "live_vs_oracle_retention_pct": round(live_retention_azure, 2),
            "compound_oracle_plus_economic": f"+{compound_oracle_azure:.1f}% (estimated)",
            "compound_live_plus_economic": f"+{compound_live_azure:.1f}% (estimated)",
        },
    }

    if burstgpt and live_vs_fifo_burstgpt is not None:
        econ_compound_burstgpt = (econ_mult * (1 + live_vs_fifo_burstgpt / 100.0) - 1) * 100.0
        table["burstgpt_hf"] = {
            "serving_queue_fifo": "reference (0%)",
            "serving_queue_conformal_oracle": f"+{conformal_oracle_vs_fifo_burstgpt:.1f}% (run -r)",
            "serving_queue_conformal_live": f"+{live_vs_fifo_burstgpt:.2f}% (run -t, THIS RUN)",
            "live_vs_oracle_retention_pct": round(live_retention_burstgpt, 2),
            "compound_live_plus_economic": f"+{econ_compound_burstgpt:.1f}% (estimated)",
        }

    return table


def _write_markdown(today: str, azure: dict, burstgpt: dict | None, compound: dict) -> str:
    lines = [
        f"# Live Causal Prior + Compound Serving/Economic Backtest — {today}",
        "",
        "**Run:** 2026-06-21-t  |  **Status:** Public-trace M/G/c discrete-event replay",
        "",
        "> Directional simulator/backtest evidence — not production savings (docs/RESULTS.md §8).",
        "",
        "## Summary",
        "",
        "This run closes the oracle gap by replacing the oracle prediction prior with",
        "a **causal sliding-window median estimator** — the minimum viable production",
        "prior that uses only the trace's own historical completion statistics.",
        "",
        "### Key Results",
        "",
    ]

    azure_live = azure.get("live_delta_pct", 0.0)
    azure_ret = azure.get("live_vs_oracle_retention_pct", 0.0)
    azure_cv = azure.get("prior_cv_pct", 0.0)
    azure_mae = azure.get("prior_mae_tokens", 0.0)

    lines += [
        f"- **Azure LLM 2024 live prior:** +{azure_live:.2f}% vs FIFO "
        f"({azure_ret:.1f}% retention vs oracle)",
        f"- **Azure prior quality:** CV={azure_cv:.1f}%, MAE={azure_mae:.1f} tokens",
    ]

    if burstgpt:
        bgpt_live = burstgpt.get("live_delta_pct", 0.0)
        bgpt_ret = burstgpt.get("live_vs_oracle_retention_pct", 0.0)
        bgpt_cv = burstgpt.get("prior_cv_pct", 0.0)
        bgpt_mae = burstgpt.get("prior_mae_tokens", 0.0)
        lines += [
            f"- **BurstGPT HF live prior:** +{bgpt_live:.2f}% vs FIFO "
            f"({bgpt_ret:.1f}% retention vs oracle)",
            f"- **BurstGPT prior quality:** CV={bgpt_cv:.1f}%, MAE={bgpt_mae:.1f} tokens",
        ]

    lines += [
        "",
        "## Public Trace Backtest Results",
        "",
        "### Azure LLM 2024 (5,880 requests, ρ=0.85, 4 servers, SLA=10s)",
        "",
        "- Dataset: Azure LLM Inference Trace 2024 (public, DynamoLLM HPCA 2025)",
        f"- Requests: {azure.get('total_requests', 0)}  |  Servers: {azure.get('servers', 4)}",
        f"- Target ρ: {azure.get('target_rho', 0.85)}  |  SLA: {azure.get('sla_s', 10)}s",
        "",
        "| Discipline | SLA-safe goodput/$ | vs FIFO |",
        "|---|---:|---:|",
        f"| FIFO (baseline) | {azure.get('fifo_goodput_per_dollar', 0):.2f} | — |",
        f"| Conformal oracle | {azure.get('oracle_goodput_per_dollar', 0):.2f} | "
        f"+{azure.get('oracle_delta_pct', 0):.2f}% |",
        f"| Conformal live prior | {azure.get('live_goodput_per_dollar', 0):.2f} | "
        f"+{azure_live:.2f}% |",
        "",
        f"**Live vs oracle retention: {azure_ret:.1f}%**",
        f"Prior CV: {azure_cv:.1f}%  |  Prior MAE: {azure_mae:.1f} tokens",
        "(30%-CV lognormal floor from run -n: 83.1% retention)",
        "",
    ]

    if burstgpt:
        lines += [
            "### BurstGPT HF (5,880-record sample, ρ=0.85, 4 servers, SLA=30s)",
            "",
            "- Dataset: BurstGPT HF (59,999 records, CC-BY-4.0, 5,880 sampled)",
            f"- Requests: {burstgpt.get('total_requests', 0)}  |  Servers: {burstgpt.get('servers', 4)}",
            f"- Target ρ: {burstgpt.get('target_rho', 0.85)}  |  SLA: {burstgpt.get('sla_s', 30)}s",
            "",
            "| Discipline | SLA-safe goodput/$ | vs FIFO |",
            "|---|---:|---:|",
            f"| FIFO (baseline) | {burstgpt.get('fifo_goodput_per_dollar', 0):.2f} | — |",
            f"| Conformal oracle | {burstgpt.get('oracle_goodput_per_dollar', 0):.2f} | "
            f"+{burstgpt.get('oracle_delta_pct', 0):.2f}% |",
            f"| Conformal live prior | {burstgpt.get('live_goodput_per_dollar', 0):.2f} | "
            f"+{burstgpt.get('live_delta_pct', 0):.2f}% |",
            "",
            f"**Live vs oracle retention: {bgpt_ret:.1f}%**",
            f"Prior CV: {bgpt_cv:.1f}%  |  Prior MAE: {bgpt_mae:.1f} tokens",
            "",
        ]

    lines += [
        "## Compound Gain Table",
        "",
        "Economic scheduling gain (constraint_aware vs FIFO) is from published",
        "benchmark results in BENCHMARK_REGISTRY.md / run 2026-06-21-s.",
        "Serving queue gain (live prior) is measured in this run.",
        "",
        "| Lever | vs FIFO | Source |",
        "|---|---:|---|",
        f"| SLA-aware binary priority | +{compound['economic_scheduling']['sla_aware_vs_fifo_pct']:.1f}% | run -n |",
        f"| Economic scheduling only (constraint_aware) | +{compound['economic_scheduling']['constraint_aware_vs_fifo_pct']:.1f}% | BENCHMARK_REGISTRY |",
        "| Conformal queue only (oracle) | +322.24% | run -q |",
        f"| Conformal queue only (live prior) | +{azure.get('live_delta_pct', 0):.2f}% | **this run** |",
        f"| Compound: live queue + economic (est.) | {compound['azure_llm_2024']['compound_live_plus_economic']} | independence estimate |",
        "",
        "Independence assumption: economic (provisioning) and serving queue (request ordering)",
        "improvements operate on orthogonal dimensions. The compound estimate is a product",
        "of their individual multipliers. A true end-to-end integrated backtest remains",
        "the highest expected value next step.",
        "",
        "## North Star Progress",
        "",
        "North Star target: +300% SLA-safe goodput/$ vs SLA-aware schedulers.",
        "",
        "Current best (live prior conformal vs SLA-aware):",
        "  = conformal_live_vs_fifo / (sla_aware_vs_fifo + 1) × 100 − 100",
        "",
    ]

    if azure.get("live_delta_pct"):
        live_vs_fifo = azure["live_delta_pct"] / 100.0
        sla_aware_vs_fifo = 125.4 / 100.0
        live_vs_sla_aware = (1 + live_vs_fifo) / (1 + sla_aware_vs_fifo) - 1
        lines.append(
            f"  Azure LLM 2024: +{live_vs_sla_aware * 100:.1f}% vs SLA-aware "
            f"(target: +300%, gap: {300 - live_vs_sla_aware * 100:.1f}pp)"
        )
    lines += ["", "## Methodology"]
    lines += [
        "",
        "- **Live prior**: For request i, predict output_tokens as median of actual tokens",
        "  from requests 0..i-1 (causal, no future leakage).",
        "- **Service time**: always actual_tokens × TPOT_S (no leakage in serving physics).",
        "- **Identical server pool**: 4 servers, identical across all disciplines.",
        "- **Time warp**: single scalar to achieve target ρ=0.85, applied identically.",
        "- **Conformal calibrator**: adapts α from empirical prediction errors observed",
        "  during the simulation (causal: error measured after completion).",
        "",
        "This is a discrete-event M/G/c simulator result, not production savings.",
        "See docs/RESULTS.md §8 for the full honesty/limitations statement.",
    ]

    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live prior + compound backtest [run -t]")
    p.add_argument("--skip-burstgpt", action="store_true",
                   help="Skip BurstGPT HF backtest (faster run)")
    p.add_argument("--azure-job-limit", type=int, default=None,
                   help="Cap on Azure requests (default: use all 5,880)")
    p.add_argument("--burstgpt-job-limit", type=int, default=5880,
                   help="Cap on BurstGPT HF requests (default: 5880 for scale parity)")
    args = p.parse_args(argv)

    today = date.today().isoformat()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n=== Live Prior + Compound Backtest [run 2026-06-21-t] — {today} ===\n")

    # 1. Azure LLM 2024 live prior
    print("1/2 Azure LLM 2024 live causal prior:")
    azure_result = _run_azure_live_prior(args.azure_job_limit)
    print(
        f"     FIFO:            {azure_result.get('fifo_goodput_per_dollar', 0):.2f} goodput/$\n"
        f"     Conformal oracle: {azure_result.get('oracle_goodput_per_dollar', 0):.2f} "
        f"(+{azure_result.get('oracle_delta_pct', 0):.2f}% vs FIFO)\n"
        f"     Conformal live:  {azure_result.get('live_goodput_per_dollar', 0):.2f} "
        f"(+{azure_result.get('live_delta_pct', 0):.2f}% vs FIFO)\n"
        f"     Live retention:  {azure_result.get('live_vs_oracle_retention_pct', 0):.1f}% of oracle\n"
        f"     Prior CV:        {azure_result.get('prior_cv_pct', 0):.1f}%  "
        f"MAE: {azure_result.get('prior_mae_tokens', 0):.1f} tokens\n"
        f"     Runtime: {azure_result.get('runtime_s', 0):.1f}s"
    )

    # 2. BurstGPT HF live prior
    burstgpt_result = None
    if not args.skip_burstgpt:
        print("\n2/2 BurstGPT HF live causal prior (5,880 records):")
        burstgpt_result = _run_burstgpt_live_prior(args.burstgpt_job_limit)
        if burstgpt_result:
            print(
                f"     FIFO:            {burstgpt_result.get('fifo_goodput_per_dollar', 0):.2f} goodput/$\n"
                f"     Conformal oracle: {burstgpt_result.get('oracle_goodput_per_dollar', 0):.2f} "
                f"(+{burstgpt_result.get('oracle_delta_pct', 0):.2f}% vs FIFO)\n"
                f"     Conformal live:  {burstgpt_result.get('live_goodput_per_dollar', 0):.2f} "
                f"(+{burstgpt_result.get('live_delta_pct', 0):.2f}% vs FIFO)\n"
                f"     Live retention:  {burstgpt_result.get('live_vs_oracle_retention_pct', 0):.1f}% of oracle\n"
                f"     Prior CV:        {burstgpt_result.get('prior_cv_pct', 0):.1f}%  "
                f"MAE: {burstgpt_result.get('prior_mae_tokens', 0):.1f} tokens\n"
                f"     Runtime: {burstgpt_result.get('runtime_s', 0):.1f}s"
            )

    # 3. Compound table
    print("\n3/3 Computing compound gain table...")
    compound = _compound_table(azure_result, burstgpt_result)
    print(
        f"     Economic scheduling (constraint_aware vs FIFO): "
        f"+{compound['economic_scheduling']['constraint_aware_vs_fifo_pct']:.1f}%\n"
        f"     Serving queue live prior (Azure):  "
        f"+{azure_result.get('live_delta_pct', 0):.2f}%\n"
        f"     Compound estimate (Azure):         "
        f"{compound['azure_llm_2024']['compound_live_plus_economic']}"
    )

    # Write results
    output = {
        "run": "2026-06-21-t",
        "date": today,
        "description": "Live causal prior closes oracle gap; compound economic+queue table",
        "public_datasets": [
            "Azure LLM Inference Trace 2024 (public, DynamoLLM HPCA 2025)",
            "BurstGPT HF (CC-BY-4.0, lzzmm/BurstGPT)",
        ],
        "azure_llm_2024": azure_result,
        "burstgpt_hf": burstgpt_result,
        "compound_table": compound,
        "shadow_tag": "shadow_only_simulator_result_not_production_savings",
    }

    json_path = os.path.join(RESULTS_DIR, f"live_prior_compound_backtest_{today}.json")
    md_path = os.path.join(RESULTS_DIR, f"live_prior_compound_backtest_{today}.md")

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {json_path}")

    md = _write_markdown(today, azure_result, burstgpt_result, compound)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown written to {md_path}")

    # Final assessment
    retention = azure_result.get("live_vs_oracle_retention_pct", 0.0)
    live_delta = azure_result.get("live_delta_pct", 0.0)
    print("\n=== ASSESSMENT ===")
    print(f"Azure live prior retention: {retention:.1f}% (threshold: ≥83%)")
    if retention >= 83.0:
        print("GATE PASSED: live causal prior retains ≥83% of oracle gain.")
        print(f"Live conformal: +{live_delta:.2f}% vs FIFO (PRODUCTION VIABLE prior)")
        return 0
    else:
        print(f"GATE MISSED: live prior retention {retention:.1f}% < 83% threshold.")
        print("Investigate: prior CV or distribution shift may explain gap.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
