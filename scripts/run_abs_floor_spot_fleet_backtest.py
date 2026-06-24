#!/usr/bin/env python3
"""Absolute-Floor Max-Spot (AFMS) Policy Backtest — Run 2026-06-24.

Evaluates AFMS vs static 70% spot on Azure LLM 2024 + BurstGPT HF.

AFMS formula: c_spot = max(round(0.70 * c), c - 1)
  - For c ≤ 5: identical to static 70% (no regression)
  - For c ≥ 6: uses 1 on-demand absolute floor (1-2 more spot → lower cost)

Research basis:
  GFS (arXiv:2509.11134, ASPLOS '26) — Dynamic Spot Quota Allocation.
  SkyServe/SpotHedge (arXiv:2411.01438) — absolute on-demand safety floor.
  AI-Driven Multi-Region Provisioning (arXiv:2605.22778) — fleet optimization.

Usage:
    python scripts/run_abs_floor_spot_fleet_backtest.py

Outputs benchmark results to stdout and JSON to research/results/.
"""

from __future__ import annotations

import json
import os
import sys

# Repo root on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    from aurelius.benchmarks.srtf_serving_backtest import (
        run_abs_floor_spot_fleet_mcs_azure_backtest,
        run_abs_floor_spot_fleet_mcs_burstgpt_backtest,
        run_spot_fleet_mcs_azure_backtest,
        run_spot_fleet_mcs_burstgpt_backtest,
    )

    print("=" * 72)
    print("ABSOLUTE-FLOOR MAX-SPOT (AFMS) BACKTEST — Run 2026-06-24")
    print("=" * 72)
    print()
    print("Policy: c_spot = max(round(0.70 * c), c - 1)")
    print("  c ≤ 5: identical to static 70% (no regression)")
    print("  c ≥ 6: 1 on-demand absolute floor → 1 more spot → lower cost")
    print()

    results = {}

    # ── Azure LLM 2024 ────────────────────────────────────────────────────
    print("▶ Azure LLM 2024 (5880 requests, ρ=0.85, SLA=10s)")
    print("  Running static 70% baseline...")
    azure_static = run_spot_fleet_mcs_azure_backtest(
        fixed_c=4, target_rho=0.85, job_limit=5880,
        spot_fraction=0.70, spot_price_usd_hr=0.80,
        p_interrupt_hourly=0.10, seed=42,
    )
    print("  Running AFMS...")
    azure_afms = run_abs_floor_spot_fleet_mcs_azure_backtest(
        fixed_c=4, target_rho=0.85, job_limit=5880,
        spot_price_usd_hr=0.80, p_interrupt_hourly=0.10, seed=42,
    )

    print()
    print(f"  c_schedule: mean={azure_afms.c_schedule_mean:.1f}, "
          f"min={azure_afms.c_schedule_min}, max={azure_afms.c_schedule_max}, "
          f"n_ticks={azure_afms.n_ticks}, n_ticks_c≥6={azure_afms.n_ticks_c_ge_6}")
    print()
    print(f"  {'Condition':<30} {'Goodput/$':>12} {'Cost':>8} {'vs SLA-oracle':>14} {'North-star':>11}")
    print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*14} {'-'*11}")
    print(f"  {'FIFO+MCS static 70%':<30} "
          f"{azure_static.fifo_spot_fleet_goodput_per_dollar:>12,.0f} "
          f"${azure_static.cost_spot_fleet:>6.2f} "
          f"{azure_static.spot_fleet_vs_sla_oracle_pct:>13.1f}% "
          f"{'YES' if azure_static.north_star_achieved else 'NO':>11}")
    print(f"  {'FIFO+MCS AFMS':<30} "
          f"{azure_afms.afms_goodput_per_dollar:>12,.0f} "
          f"${azure_afms.cost_afms:>6.2f} "
          f"{azure_afms.afms_vs_sla_oracle_pct:>13.1f}% "
          f"{'YES' if azure_afms.north_star_achieved else 'NO':>11}")
    print()
    print(f"  AFMS vs static: cost reduction = {azure_afms.afms_vs_static_cost_reduction_pct:.3f}%, "
          f"goodput/$ improvement = {azure_afms.afms_vs_static_improvement_pct:.3f}%")
    print("  SLA-oracle baseline: 25,208 | North-star threshold: 100,832")
    results["azure"] = azure_afms.to_dict()

    # ── BurstGPT HF ──────────────────────────────────────────────────────
    burstgpt_jsonl = os.path.join(
        os.path.dirname(__file__), "..", "data", "external", "hf",
        "lzzmm__BurstGPT", "burstgpt_1_full", "processed", "normalized_sample.jsonl"
    )
    if os.path.exists(burstgpt_jsonl):
        print()
        print("▶ BurstGPT HF (5880 requests, ρ=0.85, SLA=30s)")
        print("  Running static 70% baseline...")
        burst_static = run_spot_fleet_mcs_burstgpt_backtest(
            fixed_c=4, target_rho=0.85, job_limit=5880,
            spot_fraction=0.70, spot_price_usd_hr=0.80,
            p_interrupt_hourly=0.10, seed=42,
        )
        print("  Running AFMS...")
        burst_afms = run_abs_floor_spot_fleet_mcs_burstgpt_backtest(
            fixed_c=4, target_rho=0.85, job_limit=5880,
            spot_price_usd_hr=0.80, p_interrupt_hourly=0.10, seed=42,
        )

        print()
        print(f"  c_schedule: mean={burst_afms.c_schedule_mean:.1f}, "
              f"min={burst_afms.c_schedule_min}, max={burst_afms.c_schedule_max}, "
              f"n_ticks={burst_afms.n_ticks}, n_ticks_c≥6={burst_afms.n_ticks_c_ge_6}")
        print()
        print(f"  {'Condition':<30} {'Goodput/$':>12} {'Cost':>8} {'vs SLA-oracle':>14} {'North-star':>11}")
        print(f"  {'-'*30} {'-'*12} {'-'*8} {'-'*14} {'-'*11}")
        print(f"  {'FIFO+MCS static 70%':<30} "
              f"{burst_static.fifo_spot_fleet_goodput_per_dollar:>12,.0f} "
              f"${burst_static.cost_spot_fleet:>6.2f} "
              f"{burst_static.spot_fleet_vs_sla_oracle_pct:>13.1f}% "
              f"{'YES' if burst_static.north_star_achieved else 'NO':>11}")
        print(f"  {'FIFO+MCS AFMS':<30} "
              f"{burst_afms.afms_goodput_per_dollar:>12,.0f} "
              f"${burst_afms.cost_afms:>6.2f} "
              f"{burst_afms.afms_vs_sla_oracle_pct:>13.1f}% "
              f"{'YES' if burst_afms.north_star_achieved else 'NO':>11}")
        print()
        print(f"  AFMS vs static: cost reduction = {burst_afms.afms_vs_static_cost_reduction_pct:.3f}%, "
              f"goodput/$ improvement = {burst_afms.afms_vs_static_improvement_pct:.3f}%")
        print("  SLA-oracle baseline: 20,280 | North-star threshold: 81,120")
        results["burstgpt"] = burst_afms.to_dict()
    else:
        print()
        print(f"  BurstGPT HF JSONL not found at: {burstgpt_jsonl}")
        print("  Skipping BurstGPT backtest.")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print()
    print("  AFMS policy: c_spot = max(round(0.70*c), c-1)")
    print("  Improvement mechanism: eliminate 2-on-demand waste at c=6,7,8")
    print()
    if "azure" in results:
        r = results["azure"]
        print("  Azure LLM 2024:")
        print(f"    Static 70%:   {r['static_goodput_per_dollar']:,.0f} goodput/$ "
              f"({r['static_vs_sla_oracle_pct']:.1f}% vs SLA-oracle)")
        print(f"    AFMS:         {r['afms_goodput_per_dollar']:,.0f} goodput/$ "
              f"({r['afms_vs_sla_oracle_pct']:.1f}% vs SLA-oracle)")
        print(f"    Improvement:  {r['afms_vs_static_improvement_pct']:.3f}% goodput/$ "
              f"({r['afms_vs_static_cost_reduction_pct']:.3f}% cost reduction)")
        print(f"    North-star:   {'ACHIEVED' if r['north_star_achieved'] else 'NOT ACHIEVED'}")
    if "burstgpt" in results:
        r = results["burstgpt"]
        print("  BurstGPT HF:")
        print(f"    Static 70%:   {r['static_goodput_per_dollar']:,.0f} goodput/$ "
              f"({r['static_vs_sla_oracle_pct']:.1f}% vs SLA-oracle)")
        print(f"    AFMS:         {r['afms_goodput_per_dollar']:,.0f} goodput/$ "
              f"({r['afms_vs_sla_oracle_pct']:.1f}% vs SLA-oracle)")
        print(f"    Improvement:  {r['afms_vs_static_improvement_pct']:.3f}% goodput/$ "
              f"({r['afms_vs_static_cost_reduction_pct']:.3f}% cost reduction)")
        print(f"    North-star:   {'ACHIEVED' if r['north_star_achieved'] else 'NOT ACHIEVED'}")

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "research", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "abs_floor_spot_fleet_backtest_2026-06-24.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
