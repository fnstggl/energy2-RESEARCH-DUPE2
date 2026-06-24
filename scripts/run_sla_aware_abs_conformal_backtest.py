"""Standalone runner: SLA-aware vs Abs-Conformal head-to-head [run 2026-06-22-y].

Answers the north-star question: does abs-conformal (live prior) achieve
+300% vs SLA-aware schedulers?

Usage:
    python scripts/run_sla_aware_abs_conformal_backtest.py

Runs the six-discipline head-to-head on:
  1. Azure LLM 2024 public trace
  2. BurstGPT HF public trace (CC-BY-4.0)

Disciplines compared:
  FIFO / SLA-aware (oracle) / SLA-aware (live) / Rel-conformal / Abs-conformal / Oracle

Primary finding:
  abs_vs_sla_aware_oracle_delta_pct — does live abs-conformal beat oracle SLA-aware?
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from aurelius.benchmarks.srtf_serving_backtest import (
    run_sla_aware_abs_conformal_azure_backtest,
    run_sla_aware_abs_conformal_burstgpt_backtest,
)


def _fmt(val: float, prec: int = 2) -> str:
    return f"{val:,.{prec}f}"


def _print_report(label: str, rpt) -> None:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Trace: {rpt.trace}")
    print(f"  n={rpt.total_requests}, servers={rpt.servers}, ρ={rpt.target_rho}, SLA={rpt.sla_s}s")
    print()
    print(f"  {'Discipline':<28} {'Goodput/$':>12}  {'vs FIFO':>10}  {'Oracle ret':>10}")
    print(f"  {'-'*28}  {'-'*12}  {'-'*10}  {'-'*10}")
    print(f"  {'FIFO':<28} {_fmt(rpt.fifo_goodput_per_dollar):>12}  {'—':>10}  {'—':>10}")
    print(f"  {'SLA-aware (oracle)':<28} {_fmt(rpt.sla_aware_oracle_goodput_per_dollar):>12}"
          f"  {_fmt(rpt.sla_aware_oracle_delta_pct):>9}%  {_fmt(rpt.sla_aware_oracle_retention_pct):>9}%")
    print(f"  {'SLA-aware (live prior)':<28} {_fmt(rpt.sla_aware_live_goodput_per_dollar):>12}"
          f"  {_fmt(rpt.sla_aware_live_delta_pct):>9}%  {_fmt(rpt.sla_aware_live_retention_pct):>9}%")
    print(f"  {'Rel-conformal (live)':<28} {_fmt(rpt.rel_conformal_goodput_per_dollar):>12}"
          f"  {_fmt(rpt.rel_conformal_delta_pct):>9}%  {_fmt(rpt.rel_vs_oracle_retention_pct):>9}%")
    print(f"  {'Abs-conformal (live) [FRONTIER]':<28} {_fmt(rpt.abs_conformal_goodput_per_dollar):>12}"
          f"  {_fmt(rpt.abs_conformal_delta_pct):>9}%  {_fmt(rpt.abs_vs_oracle_retention_pct):>9}%")
    print(f"  {'Oracle conformal [ceiling]':<28} {_fmt(rpt.oracle_goodput_per_dollar):>12}"
          f"  {_fmt(rpt.oracle_delta_pct):>9}%  {'100.0':>9}%")
    print()
    print("  HEAD-TO-HEAD (PRIMARY FINDING):")
    print(f"  Abs-conformal vs SLA-aware (oracle): {rpt.abs_vs_sla_aware_oracle_delta_pct:>+.2f}%")
    print(f"  Abs-conformal vs SLA-aware (live):   {rpt.abs_vs_sla_aware_live_delta_pct:>+.2f}%")
    print(f"  Abs-conformal vs Rel-conformal:      {rpt.abs_vs_rel_delta_pct:>+.2f}%")
    print()
    print("  Calibrator diagnostics:")
    print(f"    abs_mean_alpha={rpt.abs_mean_alpha:.6f}  rel_mean_alpha={rpt.rel_mean_alpha:.6f}")
    print(f"    abs_p90_abs_err_tokens={rpt.abs_p90_abs_err_tokens:.1f}")


def main() -> None:
    print("Run 2026-06-22-y: SLA-aware vs Abs-Conformal Head-to-Head")
    print("North-star question: does abs-conformal achieve +300% vs SLA-aware?")

    print("\nRunning Azure LLM 2024 backtest...")
    azure = run_sla_aware_abs_conformal_azure_backtest()
    _print_report("Azure LLM 2024 (SLA=10s, 5,880 requests, ρ=0.85)", azure)

    print("\nRunning BurstGPT HF backtest...")
    burstgpt = run_sla_aware_abs_conformal_burstgpt_backtest()
    _print_report("BurstGPT HF (SLA=30s, 5,880 requests, ρ=0.85)", burstgpt)

    print("\n" + "="*65)
    print("  NORTH-STAR ASSESSMENT")
    print("="*65)
    print("  Target: +300% vs SLA-aware")
    print(f"  Azure:    abs-conformal is {azure.abs_vs_sla_aware_oracle_delta_pct:+.2f}% vs oracle SLA-aware")
    print(f"  BurstGPT: abs-conformal is {burstgpt.abs_vs_sla_aware_oracle_delta_pct:+.2f}% vs oracle SLA-aware")
    gap_met = (
        azure.abs_vs_sla_aware_oracle_delta_pct >= 300
        and burstgpt.abs_vs_sla_aware_oracle_delta_pct >= 300
    )
    if gap_met:
        print("  STATUS: NORTH-STAR ACHIEVED on both traces")
    else:
        print("  STATUS: NORTH-STAR NOT YET ACHIEVED — queue alone is +83-112%")
        print("  NEXT: compound economic × queue scheduling needed for +300% target")

    # Save JSON output
    out_path = Path(__file__).parent.parent / "research" / "results" / "sla_aware_abs_conformal_backtest_2026-06-22.json"
    with open(out_path, "w") as f:
        json.dump({
            "run": "2026-06-22-y",
            "experiment": "SLA-aware vs Abs-Conformal Head-to-Head",
            "azure_llm_2024": azure.to_dict(),
            "burstgpt_hf": burstgpt.to_dict(),
        }, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
