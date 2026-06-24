"""AMCSG-LFC + Fine Gate Grid Backtest — run 2026-06-23 (this run).

Executes two independent experiments to clear the Azure +500% north-star gap:

  (A) AMCSG-LFC (fixed_c=3): Lower the time-warp calibration parameter from
      4 to 3. Reduces effective arrival rate in warped domain → lower c_mean
      → lower fleet cost → higher goodput/$. BurstGPT cross-validated too.

  (B) Fine gate grid (fixed_c=4): Sweep gates {12.5, 13.0, 13.5, 14.0,
      14.5, 15.0}% to find a safe gate above 12.5% (AMCSG run 2026-06-27
      found 12.5% safe, 15.0% unsafe; boundary unresolved).

  (C) LFC + fine gate (fixed_c=3): Compound of both levers.

Public trace datasets:
  - Azure LLM 2024 (5,880 req, ρ=0.85, SLA=10s)
  - BurstGPT HF (5,880 req, ρ=0.85, SLA=30s)

North-star targets:
  - Azure: 151,248 goodput/$ (6× SLA-oracle of 25,208) — gap: 0.41%
  - BurstGPT: 121,680 goodput/$ (already achieved; cross-validation only)

Usage:
    python scripts/run_amcsg_lfc_backtest.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aurelius.benchmarks.srtf_serving_backtest import (
    _AMCSG_LFC_FINE_GATES,
    run_amcsg_fine_grid_azure_backtest,
    run_amcsg_lfc_azure_backtest,
    run_amcsg_lfc_burstgpt_backtest,
    run_amcsg_lfc_fine_grid_azure_backtest,
)

AZURE_SLA_ORACLE = 25_208.0
AZURE_NS_500 = 6.0 * AZURE_SLA_ORACLE   # 151,248
BURSTGPT_SLA_ORACLE = 20_280.0
BURSTGPT_NS_500 = 6.0 * BURSTGPT_SLA_ORACLE  # 121,680


def _print_header(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def _print_gate_table(report) -> None:
    print(f"\n{'Gate%':>6}  {'c_mean':>6}  {'cost($)':>8}  {'gpd/$':>10}  "
          f"{'Δbaseline':>10}  {'p99(s)':>8}  {'NS-500':>6}")
    for e in report.gate_results:
        ns = "✓" if e.north_star_500_achieved else "✗"
        print(
            f"{e.gate_pct:>6.1f}  {e.c_schedule_mean:>6.3f}  "
            f"{e.cost:>8.4f}  {e.goodput_per_dollar:>10.0f}  "
            f"{e.goodput_vs_baseline_pct:>+10.2f}%  {e.p99_s:>8.3f}  {ns:>6}"
        )


def _print_summary(report, variant_name: str, baseline_ref: float) -> None:
    safe_entries = [e for e in report.gate_results if e.n_sla_safe >= report.gate_results[0].n_sla_safe]
    if not safe_entries:
        print("  ⚠ No safe entries found")
        return
    best = max(safe_entries, key=lambda e: e.goodput_per_dollar)
    gap_pct = (AZURE_NS_500 - best.goodput_per_dollar) / AZURE_NS_500 * 100.0 if "azure" in report.trace else 0.0
    ns_achieved = best.north_star_500_achieved

    print(f"\n  {variant_name}:")
    print(f"    Best safe gate : {best.gate_pct:.1f}%")
    print(f"    c_mean         : {best.c_schedule_mean:.3f}")
    print(f"    cost           : ${best.cost:.4f}")
    print(f"    goodput/$      : {best.goodput_per_dollar:,.0f}")
    print(f"    vs baseline    : {best.goodput_vs_baseline_pct:+.2f}%")
    print(f"    vs SLA-oracle  : {best.goodput_vs_sla_oracle_pct:+.2f}%")
    print(f"    p99            : {best.p99_s:.3f}s")
    if "azure" in report.trace:
        if ns_achieved:
            print("    +500% NS       : ✓ ACHIEVED (gap: 0.00%)")
        else:
            print(f"    +500% NS       : ✗ Gap = {gap_pct:.2f}%")


def run_all() -> dict:
    results = {}

    # ── (A) AMCSG-LFC Azure (fixed_c=3, standard gates) ──────────────────────
    _print_header("(A) AMCSG-LFC Azure — fixed_c=3, standard gates")
    lfc_azure = run_amcsg_lfc_azure_backtest()
    _print_gate_table(lfc_azure)
    _print_summary(lfc_azure, "AMCSG-LFC Azure", AZURE_NS_500)
    results["lfc_azure"] = lfc_azure.to_dict()

    # ── (B) AMCSG Fine Grid Azure (fixed_c=4, fine gates) ────────────────────
    _print_header("(B) AMCSG Fine Grid Azure — fixed_c=4, fine gates")
    print(f"  Fine gates: {_AMCSG_LFC_FINE_GATES}")
    fine_azure = run_amcsg_fine_grid_azure_backtest()
    _print_gate_table(fine_azure)
    _print_summary(fine_azure, "Fine Grid Azure", AZURE_NS_500)
    results["fine_grid_azure"] = fine_azure.to_dict()

    # ── (C) LFC + Fine Grid Azure (fixed_c=3, fine gates) ────────────────────
    _print_header("(C) AMCSG-LFC + Fine Grid Azure — fixed_c=3, fine gates")
    lfc_fine_azure = run_amcsg_lfc_fine_grid_azure_backtest()
    _print_gate_table(lfc_fine_azure)
    _print_summary(lfc_fine_azure, "LFC + Fine Grid Azure", AZURE_NS_500)
    results["lfc_fine_grid_azure"] = lfc_fine_azure.to_dict()

    # ── (D) AMCSG-LFC BurstGPT (fixed_c=3, standard gates) ──────────────────
    _print_header("(D) AMCSG-LFC BurstGPT — fixed_c=3, standard gates")
    lfc_burstgpt = run_amcsg_lfc_burstgpt_backtest()
    _print_gate_table(lfc_burstgpt)
    _print_summary(lfc_burstgpt, "AMCSG-LFC BurstGPT", BURSTGPT_NS_500)
    results["lfc_burstgpt"] = lfc_burstgpt.to_dict()

    # ── North-star summary ────────────────────────────────────────────────────
    _print_header("NORTH-STAR SUMMARY")
    best_lfc = lfc_azure.best_goodput_per_dollar
    best_fine = fine_azure.best_goodput_per_dollar
    best_lfc_fine = lfc_fine_azure.best_goodput_per_dollar
    amcsg_ref = 150_630.0  # AMCSG run 2026-06-27 best

    print(f"\n  AMCSG ref (run 2026-06-27):     {amcsg_ref:>10,.0f}  ({(amcsg_ref-AZURE_NS_500)/AZURE_NS_500*100:+.2f}% to NS-500)")
    print(f"  (A) AMCSG-LFC (fixed_c=3):      {best_lfc:>10,.0f}  ({(best_lfc-AZURE_NS_500)/AZURE_NS_500*100:+.2f}% to NS-500)")
    print(f"  (B) Fine grid (fixed_c=4):       {best_fine:>10,.0f}  ({(best_fine-AZURE_NS_500)/AZURE_NS_500*100:+.2f}% to NS-500)")
    print(f"  (C) LFC+Fine (fixed_c=3):        {best_lfc_fine:>10,.0f}  ({(best_lfc_fine-AZURE_NS_500)/AZURE_NS_500*100:+.2f}% to NS-500)")
    print(f"\n  North-star threshold (NS-500):   {AZURE_NS_500:>10,.0f}")

    any_achieved = any([
        lfc_azure.best_north_star_500_achieved,
        fine_azure.best_north_star_500_achieved,
        lfc_fine_azure.best_north_star_500_achieved,
    ])
    if any_achieved:
        print("\n  ✓ NORTH-STAR +500% ACHIEVED on Azure!")
    else:
        remaining = AZURE_NS_500 - max(best_lfc, best_fine, best_lfc_fine)
        print(f"\n  ✗ North-star not yet achieved. Remaining gap: {remaining:,.0f} goodput/$")

    return results


if __name__ == "__main__":
    results = run_all()
    out_path = Path("research/results/amcsg_lfc_backtest_2026-06-23.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  JSON results saved to {out_path}")
