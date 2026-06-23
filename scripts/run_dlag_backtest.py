"""DLAG (Dynamic Load-Aware Gate) Backtest — run 2026-06-23.

Sweeps max_gate ∈ {15.0, 17.5, 20.0, 25.0, 30.0}% using per-tick gate:

    gate_k = base_gate + (max_gate - base_gate) × max(0, 1 − ρ_k / target_ρ)

where ρ_k = λ_k × E[S_k] is the per-server utilization for tick k.

High-load ticks (ρ_k ≈ target_ρ) retain conservative base_gate (9.5%).
Idle ticks (ρ_k ≪ target_ρ) receive aggressive max_gate.

Hypothesis: Dynamic gating can capture the cost savings of high flat gates
at idle ticks without triggering SLA violations during high-load ticks.

Traces:
  - Azure LLM 2024 (5,880 req, ρ=0.85, SLA=10s) — north-star target
  - BurstGPT HF (5,880 req, ρ=0.85, SLA=30s) — cross-validation

Usage:
    python scripts/run_dlag_backtest.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurelius.benchmarks.srtf_serving_backtest import (
    _DLAG_MAX_GATES,
    run_dlag_azure_backtest,
    run_dlag_burstgpt_backtest,
)

AZURE_SLA_ORACLE = 25_208.0
AZURE_NS_500 = 6.0 * AZURE_SLA_ORACLE   # 151,248
AMCSG_REF = 150_630.0                    # AMCSG run 2026-06-27 gate=12.5% best


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _print_table(report) -> None:
    print(f"\n{'max_gate%':>10}  {'c_mean':>7}  {'cost($)':>8}  {'gpd/$':>10}  "
          f"{'vs AMCSG':>9}  {'p99(s)':>8}  {'n_safe':>6}  {'NS-500':>6}")
    for e in report.max_gate_results:
        ns = "YES" if e.north_star_500_achieved else "no"
        print(
            f"{e.max_gate_pct:>10.1f}  {e.c_schedule_mean:>7.3f}  "
            f"{e.cost:>8.4f}  {e.goodput_per_dollar:>10.0f}  "
            f"{e.goodput_vs_amcsg_pct:>+9.2f}%  {e.p99_s:>8.3f}  "
            f"{e.n_sla_safe:>6}  {ns:>6}"
        )


def run_all() -> dict:
    results = {}

    # ── Azure ─────────────────────────────────────────────────────────────────
    _print_header("DLAG Azure — base_gate=9.5%, max_gate sweep")
    print(f"  Max gates: {_DLAG_MAX_GATES}")
    azure = run_dlag_azure_backtest()
    _print_table(azure)

    best_azure = azure.best_goodput_per_dollar
    gap_pct = (AZURE_NS_500 - best_azure) / AZURE_NS_500 * 100.0
    print(f"\n  Best max_gate   : {azure.best_max_gate}%")
    print(f"  Best goodput/$  : {best_azure:,.0f}")
    print(f"  AMCSG ref       : {AMCSG_REF:,.0f}")
    print(f"  NS-500 gap      : {gap_pct:.2f}%")
    if azure.best_north_star_500_achieved:
        print("  +500% NS        : YES — ACHIEVED")
    else:
        print(f"  +500% NS        : not achieved (gap: {gap_pct:.2f}%)")
    results["dlag_azure"] = azure.to_dict()

    # ── BurstGPT ──────────────────────────────────────────────────────────────
    _print_header("DLAG BurstGPT — base_gate=9.5%, max_gate sweep")
    bgpt = run_dlag_burstgpt_backtest()
    _print_table(bgpt)
    print(f"\n  Best goodput/$  : {bgpt.best_goodput_per_dollar:,.0f}")
    results["dlag_burstgpt"] = bgpt.to_dict()

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_header("SUMMARY")
    print(f"\n  AMCSG ref (2026-06-27):  {AMCSG_REF:>10,.0f}")
    print(f"  DLAG Azure best:         {best_azure:>10,.0f}  ({(best_azure - AZURE_NS_500) / AZURE_NS_500 * 100:+.2f}% to NS-500)")
    print(f"  NS-500 threshold:        {AZURE_NS_500:>10,.0f}")
    if azure.best_north_star_500_achieved:
        print("\n  ✓ NORTH-STAR +500% ACHIEVED on Azure!")
    else:
        print(f"\n  ✗ North-star not achieved. Remaining gap: {AZURE_NS_500 - best_azure:,.0f} goodput/$")

    return results


if __name__ == "__main__":
    results = run_all()
    out_path = Path("research/results/dlag_backtest_2026-06-23.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON results saved to {out_path}")
