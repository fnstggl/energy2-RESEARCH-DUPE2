"""Run AMCSG (Adaptive MCS Gate Sweep) public backtest — run 2026-06-27.

Sweeps mcs_gate ∈ {9.5, 11.0, 12.5, 15.0, 17.5, 20.0}% on both
Azure LLM 2024 and BurstGPT HF at fixed_c=4, target_rho=0.85,
spot_fraction=0.95 (all-spot every tick). Identifies whether the
Erlang-C (M/M/c) conservatism in _joint_mcs_c_schedule can be safely
exploited to reduce c_mean and close the 1.35% gap to the +500%
north-star on Azure.

Usage:
    python scripts/run_amcsg_backtest.py

Outputs:
    research/results/amcsg_backtest_2026-06-27.json
    research/results/amcsg_backtest_2026-06-27.md
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.benchmarks.srtf_serving_backtest import (
    run_amcsg_azure_backtest,
    run_amcsg_burstgpt_backtest,
)


def main() -> None:
    print("=" * 72)
    print("AMCSG (Adaptive MCS Gate Sweep) — run 2026-06-27")
    print("=" * 72)
    print()
    print("Bottleneck: Azure goodput/$ = 149,235 (GSF f=0.95), 1.35% below")
    print("  +500% north-star (151,248 = 6× SLA-oracle of 25,208).")
    print("Hypothesis: M/M/c Erlang-C over-estimates queue wait for non-")
    print("  exponential service times. Raising gate reduces c_mean & cost.")
    print()

    print("Running Azure LLM 2024 gate sweep …")
    azure = run_amcsg_azure_backtest()
    print("Running BurstGPT HF gate sweep …")
    burstgpt = run_amcsg_burstgpt_backtest()

    # ── Print detailed sweep tables ──────────────────────────────────────
    for report, label in [(azure, "Azure LLM 2024"), (burstgpt, "BurstGPT HF")]:
        print()
        print(f"── {label} ──────────────────────────────────────────────")
        print(
            f"{'Gate%':>6}  {'c_mean':>6}  {'cost($)':>8}  "
            f"{'gp/$':>10}  {'Δbaseline':>10}  {'Δoracle':>10}  "
            f"{'NS500':>6}  {'p99s':>6}  {'ok':>4}"
        )
        print("-" * 80)
        for e in report.gate_results:
            ok = "✓" if e.completion_rate >= report.gate_results[0].completion_rate - 0.001 else "✗"
            ns = "✓" if e.north_star_500_achieved else "-"
            print(
                f"{e.gate_pct:>6.1f}  {e.c_schedule_mean:>6.3f}  {e.cost:>8.4f}  "
                f"{e.goodput_per_dollar:>10.0f}  "
                f"{e.goodput_vs_baseline_pct:>+9.2f}%  "
                f"{e.goodput_vs_sla_oracle_pct:>+9.1f}%  "
                f"{ns:>6}  {e.p99_s:>6.3f}  {ok:>4}"
            )
        print()
        print(f"  Baseline gate:           {report.baseline_gate}%")
        print(f"  Baseline goodput/$:      {report.baseline_goodput_per_dollar:,.0f}")
        print(f"  Best gate:               {report.best_gate}%")
        print(f"  Best goodput/$:          {report.best_goodput_per_dollar:,.0f}")
        print(f"  Best vs baseline:        {report.best_vs_baseline_pct:+.2f}%")
        print(f"  Best vs SLA-oracle:      {report.best_vs_sla_oracle_pct:+.1f}%")
        print(f"  North-star +500% hit:    {report.best_north_star_500_achieved}")
        print(f"  Max safe gate:           {report.max_safe_gate}%")
        print(f"  Erlang-C safety margin:  +{report.erlang_c_margin_pct:.1f}% above 9.5%")

    # ── Primary result ───────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("PRIMARY RESULT")
    print("=" * 72)
    azure_improvement = azure.best_vs_baseline_pct
    burstgpt_improvement = burstgpt.best_vs_baseline_pct
    azure_ns500 = azure.best_north_star_500_achieved
    burstgpt_ns500 = burstgpt.best_north_star_500_achieved

    print(f"Azure LLM 2024:  GSF(9.5%) {azure.baseline_goodput_per_dollar:,.0f} → "
          f"AMCSG({azure.best_gate}%) {azure.best_goodput_per_dollar:,.0f} "
          f"({azure_improvement:+.2f}%) | +500% NS: {azure_ns500}")
    print(f"BurstGPT HF:     GSF(9.5%) {burstgpt.baseline_goodput_per_dollar:,.0f} → "
          f"AMCSG({burstgpt.best_gate}%) {burstgpt.best_goodput_per_dollar:,.0f} "
          f"({burstgpt_improvement:+.2f}%) | +500% NS: {burstgpt_ns500}")

    # ── Save results ─────────────────────────────────────────────────────
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "research", "results",
    )
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "amcsg_backtest_2026-06-27.json")
    md_path = os.path.join(results_dir, "amcsg_backtest_2026-06-27.md")

    output = {
        "run_date": "2026-06-27",
        "experiment": "AMCSG — Adaptive MCS Gate Sweep",
        "policy": "AMCSG",
        "description": (
            "Sweeps mcs_gate ∈ {9.5, 11.0, 12.5, 15.0, 17.5, 20.0}% at "
            "fixed_c=4, target_rho=0.85, spot_fraction=0.95. Tests whether "
            "Erlang-C (M/M/c) conservatism in MCS can be safely relaxed to "
            "reduce c_mean, lower cost, and close the Azure +500% gap."
        ),
        "azure": azure.to_dict(),
        "burstgpt": burstgpt.to_dict(),
        "summary": {
            "azure_baseline_goodput_per_dollar": azure.baseline_goodput_per_dollar,
            "azure_best_gate": azure.best_gate,
            "azure_best_goodput_per_dollar": azure.best_goodput_per_dollar,
            "azure_improvement_vs_baseline_pct": round(azure_improvement, 4),
            "azure_vs_oracle_pct": round(azure.best_vs_sla_oracle_pct, 2),
            "azure_north_star_500_achieved": azure_ns500,
            "azure_erlang_c_margin_pct": azure.erlang_c_margin_pct,
            "burstgpt_baseline_goodput_per_dollar": burstgpt.baseline_goodput_per_dollar,
            "burstgpt_best_gate": burstgpt.best_gate,
            "burstgpt_best_goodput_per_dollar": burstgpt.best_goodput_per_dollar,
            "burstgpt_improvement_vs_baseline_pct": round(burstgpt_improvement, 4),
            "burstgpt_vs_oracle_pct": round(burstgpt.best_vs_sla_oracle_pct, 2),
            "burstgpt_north_star_500_achieved": burstgpt_ns500,
        },
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Markdown report
    azure_status = "FRONTIER IMPROVEMENT" if azure_improvement > 0.5 else (
        "NULL RESULT" if azure_improvement <= 0 else "MARGINAL"
    )
    with open(md_path, "w") as f:
        f.write("# AMCSG Gate Sweep Backtest — 2026-06-27\n\n")
        f.write("**Policy:** AMCSG (Adaptive MCS Gate Sweep)  \n")
        f.write(f"**Status:** {azure_status}  \n\n")
        f.write("## Primary Result\n\n")
        f.write(f"| Trace | Baseline (gate=9.5%) | Best (gate={azure.best_gate}%) | "
                f"vs Baseline | vs SLA-oracle | NS +500% |\n")
        f.write("|-------|---------------------|--------------------------------|"
                "------------|---------------|----------|\n")
        f.write(f"| Azure LLM 2024 | {azure.baseline_goodput_per_dollar:,.0f} "
                f"(${azure.baseline_cost:.2f}) | **{azure.best_goodput_per_dollar:,.0f}** "
                f"| **{azure_improvement:+.2f}%** | **{azure.best_vs_sla_oracle_pct:+.1f}%** "
                f"| {'✓' if azure_ns500 else '✗'} |\n")
        f.write(f"| BurstGPT HF | {burstgpt.baseline_goodput_per_dollar:,.0f} "
                f"(${burstgpt.baseline_cost:.2f}) | **{burstgpt.best_goodput_per_dollar:,.0f}** "
                f"| **{burstgpt_improvement:+.2f}%** | **{burstgpt.best_vs_sla_oracle_pct:+.1f}%** "
                f"| {'✓' if burstgpt_ns500 else '✗'} |\n\n")
        f.write("## Gate Sweep Detail — Azure LLM 2024\n\n")
        f.write("| Gate% | c_mean | cost($) | goodput/$ | Δbaseline | p99(s) | NS-500 |\n")
        f.write("|-------|--------|---------|-----------|-----------|--------|--------|\n")
        for e in azure.gate_results:
            f.write(
                f"| {e.gate_pct:.1f}% | {e.c_schedule_mean:.3f} | ${e.cost:.4f} "
                f"| {e.goodput_per_dollar:,.0f} | {e.goodput_vs_baseline_pct:+.2f}% "
                f"| {e.p99_s:.3f} | {'✓' if e.north_star_500_achieved else '✗'} |\n"
            )
        f.write("\n## Gate Sweep Detail — BurstGPT HF\n\n")
        f.write("| Gate% | c_mean | cost($) | goodput/$ | Δbaseline | p99(s) | NS-500 |\n")
        f.write("|-------|--------|---------|-----------|-----------|--------|--------|\n")
        for e in burstgpt.gate_results:
            f.write(
                f"| {e.gate_pct:.1f}% | {e.c_schedule_mean:.3f} | ${e.cost:.4f} "
                f"| {e.goodput_per_dollar:,.0f} | {e.goodput_vs_baseline_pct:+.2f}% "
                f"| {e.p99_s:.3f} | {'✓' if e.north_star_500_achieved else '✗'} |\n"
            )
        f.write("\n## Erlang-C Conservatism Finding\n\n")
        f.write(f"- Azure max safe gate: {azure.max_safe_gate}% "
                f"(margin: +{azure.erlang_c_margin_pct:.1f}% above 9.5%)\n")
        f.write(f"- BurstGPT max safe gate: {burstgpt.max_safe_gate}% "
                f"(margin: +{burstgpt.erlang_c_margin_pct:.1f}% above 9.5%)\n")

    print(f"Report saved:   {md_path}")


if __name__ == "__main__":
    main()
