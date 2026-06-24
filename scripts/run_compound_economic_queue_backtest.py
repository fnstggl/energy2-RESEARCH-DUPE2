"""Standalone runner: Compound Economic × Queue Scheduling [run 2026-06-22-z].

Answers the north-star question after the run-y head-to-head:
  Does compound economic + queue scheduling achieve +300% vs SLA-aware?

Architecture:
  Queue layer: abs-conformal SRPT (run 2026-06-22-x/y)
    Azure:    +83.27% vs oracle SLA-aware
    BurstGPT: +111.55% vs oracle SLA-aware
  Provisioning layer: economic scheduling (BENCHMARK_REGISTRY §1.1)
    Azure LLM 2024: +25.75% vs SLA-aware = 1.2575× cost efficiency
    (time-of-day, spot pricing, regional routing, -21.2% GPU-hours)

Independence verification:
  Queue ordering (per-request) ⊥ provisioning optimization (fleet-level)
  Compound = queue_goodput/$ × economic_cost_factor

Usage:
    python scripts/run_compound_economic_queue_backtest.py

Saves JSON output to:
    research/results/compound_economic_queue_backtest_2026-06-22.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from aurelius.benchmarks.srtf_serving_backtest import (
    ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY,
    CompoundEconomicQueueReport,
    run_compound_economic_queue_azure_backtest,
    run_compound_economic_queue_burstgpt_backtest,
)


def _fmt(val: float, prec: int = 2) -> str:
    return f"{val:,.{prec}f}"


def _print_report(label: str, rpt: CompoundEconomicQueueReport) -> None:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Trace: {rpt.trace}")
    print(f"  n={rpt.total_requests}, servers={rpt.servers}, ρ={rpt.target_rho}, SLA={rpt.sla_s}s")
    print()
    print(f"  {'Layer':<32} {'Goodput/$':>12}  {'vs FIFO':>10}  {'vs SLA-aware oracle':>20}")
    print(f"  {'-'*32}  {'-'*12}  {'-'*10}  {'-'*20}")
    print(f"  {'FIFO (baseline)':<32} {_fmt(rpt.fifo_goodput_per_dollar):>12}  {'—':>10}  {'—':>20}")
    print(f"  {'Oracle SLA-aware':<32} {_fmt(rpt.sla_aware_oracle_goodput_per_dollar):>12}"
          f"  {'—':>10}  {'0.00%':>20}")
    print(f"  {'Abs-conformal (queue only)':<32} {_fmt(rpt.abs_conformal_goodput_per_dollar):>12}"
          f"  {rpt.abs_vs_fifo_delta_pct:>+9.2f}%  {rpt.queue_vs_sla_aware_oracle_delta_pct:>+19.2f}%")
    print(f"  {'Compound (queue + economic)':<32} {_fmt(rpt.compound_goodput_per_dollar):>12}"
          f"  {rpt.compound_vs_fifo_delta_pct:>+9.2f}%  {rpt.compound_vs_sla_aware_oracle_delta_pct:>+19.2f}%")
    print()
    print(f"  Economic cost factor: {rpt.economic_cost_factor}×  ({rpt.economic_cost_factor_source})")
    print()
    print("  NORTH-STAR ASSESSMENT:")
    print("    Target:   +300% vs oracle SLA-aware")
    print(f"    Compound: {rpt.compound_vs_sla_aware_oracle_delta_pct:>+.2f}% vs oracle SLA-aware")
    status = "ACHIEVED" if rpt.north_star_achieved else "NOT ACHIEVED"
    print(f"    Status:   {status}")
    print()
    print("  RUN-T OVER-ESTIMATE CORRECTION:")
    print(f"    run-t estimate vs FIFO:    {rpt.run_t_compound_estimate_vs_fifo_pct:>+.2f}%")
    print(f"    Corrected compound vs FIFO: {rpt.corrected_compound_vs_fifo_pct:>+.2f}%")
    print(f"    Over-estimate factor: {rpt.over_estimate_factor:.3f}×")
    print("    (run-t double-counted the SLA-aware component)")
    print()
    print("  PATH TO +300%:")
    print(f"    Economic factor needed:  {rpt.economic_factor_needed_for_north_star:.4f}×")
    print(f"    Current factor:          {rpt.economic_cost_factor:.4f}×")
    print(f"    Additional factor needed: +{rpt.economic_factor_needed_delta_vs_current:.4f}×")
    gpu_saving_needed = (1.0 - 1.0 / rpt.economic_factor_needed_for_north_star) * 100.0
    print(f"    Equivalent GPU-hour saving: -{gpu_saving_needed:.1f}% (vs current -21.2%)")


def main() -> None:
    print("Run 2026-06-22-z: Compound Economic × Queue Scheduling")
    print("North-star question: does compound system achieve +300% vs SLA-aware?")
    print(f"Economic cost factor: {ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY}× "
          f"(+{(ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY - 1)*100:.2f}% from BENCHMARK_REGISTRY §1.1)")

    print("\nRunning Azure LLM 2024 compound backtest...")
    azure = run_compound_economic_queue_azure_backtest()
    _print_report("Azure LLM 2024 (SLA=10s, 5,880 requests, ρ=0.85)", azure)

    print("\nRunning BurstGPT HF compound backtest...")
    burstgpt = run_compound_economic_queue_burstgpt_backtest()
    _print_report("BurstGPT HF (SLA=30s, 5,880 requests, ρ=0.85)", burstgpt)

    print("\n" + "=" * 70)
    print("  COMPOUND NORTH-STAR SUMMARY")
    print("=" * 70)
    print("  Queue-only vs oracle SLA-aware:")
    print(f"    Azure:    {azure.queue_vs_sla_aware_oracle_delta_pct:>+.2f}%  (run -y)")
    print(f"    BurstGPT: {burstgpt.queue_vs_sla_aware_oracle_delta_pct:>+.2f}%  (run -y)")
    print(f"  Compound (queue + economic {ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY}×) vs oracle SLA-aware:")
    print(f"    Azure:    {azure.compound_vs_sla_aware_oracle_delta_pct:>+.2f}%")
    print(f"    BurstGPT: {burstgpt.compound_vs_sla_aware_oracle_delta_pct:>+.2f}%")
    print("  Target: +300% vs oracle SLA-aware")
    both_achieved = azure.north_star_achieved and burstgpt.north_star_achieved
    if both_achieved:
        print("  STATUS: NORTH-STAR ACHIEVED on both traces")
    else:
        print("  STATUS: NORTH-STAR NOT YET ACHIEVED")
        print("  Compound reaches +130-166% vs oracle SLA-aware (not +300%)")
        print(f"  Path: economic factor must increase from {ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY:.4f}× to:")
        print(f"    Azure:    {azure.economic_factor_needed_for_north_star:.4f}× (need +{azure.economic_factor_needed_delta_vs_current:.4f}× more)")
        print(f"    BurstGPT: {burstgpt.economic_factor_needed_for_north_star:.4f}× (need +{burstgpt.economic_factor_needed_delta_vs_current:.4f}× more)")
        azure_gpu_needed = (1.0 - 1.0 / azure.economic_factor_needed_for_north_star) * 100.0
        print(f"  Equivalent: need ~{azure_gpu_needed:.1f}% GPU-hour savings (vs current -21.2%)")
        print()
        print("  CORRECTION OF RUN-T OVER-ESTIMATE:")
        print("    run-t estimated +876% vs FIFO (rel-conformal) — used multiplicative")
        print("    queue_vs_fifo × economic_vs_fifo, double-counting SLA-aware component.")
        print(f"    Corrected (abs-conformal): +{azure.corrected_compound_vs_fifo_pct:.2f}% vs FIFO ({azure.over_estimate_factor:.3f}× over-estimated)")

    # Save JSON output
    out_path = (
        Path(__file__).parent.parent
        / "research" / "results"
        / "compound_economic_queue_backtest_2026-06-22.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "run": "2026-06-22-z",
                "experiment": "Compound Economic × Queue Scheduling",
                "economic_cost_factor": ECONOMIC_COST_FACTOR_BENCHMARK_REGISTRY,
                "economic_cost_factor_source": "BENCHMARK_REGISTRY §1.1 Azure LLM 2024",
                "azure_llm_2024": azure.to_dict(),
                "burstgpt_hf": burstgpt.to_dict(),
            },
            f,
            indent=2,
        )
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
