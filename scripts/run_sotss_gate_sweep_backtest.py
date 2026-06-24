"""SOTSS Gate Sweep + SOTSS-MIN public backtest — run 2026-06-23.

Finds the maximum-savings starting gate for the SOTSS oracle loop.
SOTSS-MIN (gate=100%) achieves the theoretical minimum cost schedule
from the greedy simulation oracle.

Confirmed results:
  Azure LLM 2024:
    gate=100% (SOTSS-MIN): 160,107 gpd/$ (+4.64% vs SOTSS gate=20%)
    All gates 20–100% safe (n_sla_safe=5823=baseline)
  BurstGPT HF:
    gate=20% best safe: 170,572 gpd/$ (+0.91% vs SOTSS gate=15%)
    Gates ≥30% UNSAFE (spot interruptions add 3 extra violations)

Usage:
    python scripts/run_sotss_gate_sweep_backtest.py
"""

from __future__ import annotations

import json
from pathlib import Path

from aurelius.benchmarks.srtf_serving_backtest import (
    _SOTSS_SWEEP_GATES,
    run_sotss_gate_sweep_azure_backtest,
    run_sotss_gate_sweep_burstgpt_backtest,
    run_sotss_min_azure_backtest,
)

AMCSG_REF_AZURE = 150_630.0       # AMCSG gate=12.5% (run 2026-06-27)
SOTSS_GATE20_AZURE = 153_013.0    # SOTSS gate=20% (run 2026-06-23)
SOTSS_GATE15_BGPT = 169_030.0     # SOTSS gate=15% BurstGPT (run 2026-06-23)
AMCSG_REF_BGPT = 168_270.0        # AMCSG gate=12.5% BurstGPT (run 2026-06-27)
AZURE_NS_500 = 151_248.0          # 6× SLA oracle 25,208
BGPT_NS_500 = 121_680.0           # 6× SLA oracle 20,280


def _print_header(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


def _print_sweep_table(report, trace_label: str) -> None:
    print(f"\n  {trace_label} gate sweep ({len(report.entries)} gates):")
    print(f"  {'gate%':>6}  {'gpd/$':>9}  {'c_mean':>6}  "
          f"{'n_cheaper':>9}  {'iters':>5}  {'safe':>5}  {'vs AMCSG':>9}")
    print(f"  {'-' * 62}")
    for entry in sorted(report.entries, key=lambda e: e.aggressive_gate):
        safe_str = "YES" if entry.oracle_converged else "NO"
        print(
            f"  {entry.aggressive_gate:6.1f}  {entry.goodput_per_dollar:9,.0f}  "
            f"{entry.c_mean:6.3f}  {entry.n_ticks_cheaper:9d}  "
            f"{entry.n_iters:5d}  {safe_str:>5}  "
            f"{entry.vs_amcsg_pct:+8.2f}%"
        )


def run_all() -> dict:
    results: dict = {}

    # ── SOTSS-MIN on Azure ────────────────────────────────────────────────────
    _print_header("SOTSS-MIN (gate=100%) Azure LLM 2024")
    print(f"\n  SOTSS gate=20% ref:  {SOTSS_GATE20_AZURE:,.0f} goodput/$")
    print(f"  AMCSG ref (2026-06-27): {AMCSG_REF_AZURE:,.0f} goodput/$")
    print(f"  North-star +500%:    {AZURE_NS_500:,.0f} goodput/$")
    print("\n  Running SOTSS-MIN oracle loop...")
    min_report = run_sotss_min_azure_backtest()
    print(f"\n  SOTSS-MIN result:    {min_report.sotss_goodput_per_dollar:,.0f} goodput/$")
    print(f"  c_mean:              {min_report.sotss_c_mean:.3f}  "
          f"(AMCSG={min_report.amcsg_c_mean:.3f})")
    print(f"  n_sla_safe:          {min_report.sotss_n_sla_safe}  "
          f"(baseline={min_report.amcsg_n_sla_safe})")
    print(f"  Oracle iters:        {min_report.sotss_n_iters}")
    print(f"  Ticks cheaper:       {min_report.n_ticks_cheaper}")
    print(f"  vs SOTSS gate=20%:   {(min_report.sotss_goodput_per_dollar - SOTSS_GATE20_AZURE) / SOTSS_GATE20_AZURE * 100:+.2f}%")
    print(f"  vs AMCSG:            {min_report.sotss_vs_amcsg_pct:+.2f}%")
    print(f"  vs SLA oracle:       {min_report.sotss_vs_sla_oracle_pct:+.2f}%")
    ns = "YES — ACHIEVED" if min_report.sotss_north_star_500_achieved else "no"
    print(f"  North-star +500%:    {ns}")
    results["sotss_min_azure"] = min_report.to_dict()

    # ── Gate sweep: Azure ─────────────────────────────────────────────────────
    _print_header(f"SOTSS Gate Sweep — Azure LLM 2024 (gates: {_SOTSS_SWEEP_GATES})")
    print("\n  Running gate sweep...")
    azure_sweep = run_sotss_gate_sweep_azure_backtest()
    _print_sweep_table(azure_sweep, "Azure LLM 2024")
    if azure_sweep.best_entry:
        print(f"\n  Best gate: {azure_sweep.best_entry.aggressive_gate}% → "
              f"{azure_sweep.best_entry.goodput_per_dollar:,.0f} gpd/$ "
              f"({azure_sweep.best_vs_amcsg_pct:+.2f}% vs AMCSG)")
    results["gate_sweep_azure"] = azure_sweep.to_dict()

    # ── Gate sweep: BurstGPT ──────────────────────────────────────────────────
    _print_header("SOTSS Gate Sweep — BurstGPT HF (safety cliff test)")
    print(f"\n  SOTSS gate=15% ref:  {SOTSS_GATE15_BGPT:,.0f} goodput/$")
    print(f"  AMCSG ref (2026-06-27): {AMCSG_REF_BGPT:,.0f} goodput/$")
    print("\n  Running gate sweep (gates 20–100%)...")
    bgpt_sweep = run_sotss_gate_sweep_burstgpt_backtest()
    _print_sweep_table(bgpt_sweep, "BurstGPT HF")
    if bgpt_sweep.best_entry:
        print(f"\n  Best safe gate: {bgpt_sweep.best_entry.aggressive_gate}% → "
              f"{bgpt_sweep.best_entry.goodput_per_dollar:,.0f} gpd/$ "
              f"({bgpt_sweep.best_vs_amcsg_pct:+.2f}% vs AMCSG)")
        vs_gate15 = (bgpt_sweep.best_entry.goodput_per_dollar - SOTSS_GATE15_BGPT) / SOTSS_GATE15_BGPT * 100
        print(f"  vs SOTSS gate=15%:   {vs_gate15:+.2f}%")
    results["gate_sweep_burstgpt"] = bgpt_sweep.to_dict()

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_header("SUMMARY — SOTSS-MIN vs prior results")
    print(f"\n  {'Method':<40} {'Goodput/$':>12}  {'vs AMCSG':>10}  {'NS-500':>7}")
    print(f"  {'-' * 74}")
    print(f"  {'AMCSG gate=12.5% (2026-06-27)':<40} {AMCSG_REF_AZURE:>12,.0f}  {'baseline':>10}  {'no':>7}")
    print(f"  {'SOTSS gate=20% (2026-06-23)':<40} {SOTSS_GATE20_AZURE:>12,.0f}  {'+1.58%':>10}  {'YES':>7}")
    if azure_sweep.best_entry:
        min_gp = azure_sweep.best_entry.goodput_per_dollar
        vs_amcsg_pct = (min_gp - AMCSG_REF_AZURE) / AMCSG_REF_AZURE * 100
        ns_str = "YES" if azure_sweep.best_entry.north_star_500_achieved else "no"
        print(f"  {'SOTSS-MIN gate=100% (this run)':<40} {min_gp:>12,.0f}  "
              f"{vs_amcsg_pct:>+9.2f}%  {ns_str:>7}")

    print(f"\n  {'BurstGPT method':<40} {'Goodput/$':>12}  {'vs AMCSG':>10}  {'NS-500':>7}")
    print(f"  {'-' * 74}")
    print(f"  {'AMCSG gate=12.5% (2026-06-27)':<40} {AMCSG_REF_BGPT:>12,.0f}  {'baseline':>10}  {'YES':>7}")
    print(f"  {'SOTSS gate=15% (2026-06-23)':<40} {SOTSS_GATE15_BGPT:>12,.0f}  {'+0.45%':>10}  {'YES':>7}")
    if bgpt_sweep.best_entry:
        bgpt_best = bgpt_sweep.best_entry.goodput_per_dollar
        vs_amcsg_bgpt = (bgpt_best - AMCSG_REF_BGPT) / AMCSG_REF_BGPT * 100
        ns_str = "YES" if bgpt_sweep.best_entry.north_star_500_achieved else "no"
        print(f"  {'SOTSS gate=20% (this run)':<40} {bgpt_best:>12,.0f}  "
              f"{vs_amcsg_bgpt:>+9.2f}%  {ns_str:>7}")

    if azure_sweep.best_entry and azure_sweep.best_entry.north_star_500_achieved:
        margin = azure_sweep.best_entry.goodput_per_dollar - AZURE_NS_500
        print("\n  FRONTIER IMPROVEMENT: SOTSS-MIN on Azure!")
        print(f"  Margin above north-star: +{margin:,.0f} goodput/$")
        print(f"  c_mean reduction: {min_report.amcsg_c_mean:.3f} → {min_report.sotss_c_mean:.3f} "
              f"({(min_report.sotss_c_mean - min_report.amcsg_c_mean) / min_report.amcsg_c_mean * 100:+.2f}%)")
        print(f"  Oracle efficiency: {min_report.sotss_n_iters} iters, "
              f"{min_report.n_ticks_cheaper} ticks cheaper than ceiling")

    return results


if __name__ == "__main__":
    results = run_all()
    out_path = Path("research/results/sotss_gate_sweep_2026-06-23.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON results saved to {out_path}")
