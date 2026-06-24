"""Public-trace economic backtest for Absolute-Error Conformal Calibration [run 2026-06-22-x].

Compares four disciplines on both public LLM traces:
  1. FIFO                     — baseline
  2. Conformal oracle         — upper bound (predicted == actual)
  3. Rel-conformal live prior — current best: relative-error calibrator, running-median prior
  4. Abs-conformal live prior — NEW: absolute-error calibrator, running-median prior

Primary question: does replacing relative error with absolute error in the conformal
calibrator break the 0.002 cap and improve SLA-safe goodput/$ on both public traces?

Public datasets used:
  - Azure LLM 2024 (azure_llm_2024_sample.csv)
  - BurstGPT HF (lzzmm__BurstGPT normalized_sample.jsonl, CC-BY-4.0)

Usage:
  python scripts/run_abs_conformal_backtest.py

Expected output (if hypothesis is correct):
  abs_mean_alpha < rel_mean_alpha (lower alpha = less capped)
  abs_conformal_delta_pct > rel_conformal_delta_pct (frontier improvement)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

# Ensure repo root on path.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from aurelius.benchmarks.srtf_serving_backtest import (
    CONFORMAL_ABS_TARGET_P90_TOKENS,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    run_abs_conformal_azure_backtest,
    run_abs_conformal_burstgpt_backtest,
)


def main() -> None:
    print("=" * 72)
    print("Absolute-Error Conformal Calibration — Public Trace Backtest")
    print(f"[run 2026-06-22-x]  target_p90_abs_tokens={CONFORMAL_ABS_TARGET_P90_TOKENS}")
    print("=" * 72)
    print()

    results: dict = {
        "run_id": "2026-06-22-x",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Absolute-error conformal calibration: replaces p90 relative prediction "
            "error with p90 absolute error in the conformal alpha calibrator. "
            "Hypothesis: shorter over-predictions (short ChatGPT, rel_err>1) no longer "
            "cap the calibrator, giving alpha closer to Pareto-optimal 0.001."
        ),
        "traces": {},
    }

    # ── Azure LLM 2024 ────────────────────────────────────────────────────────
    print("Running Azure LLM 2024 (5,880 requests, ρ=0.85, SLA=10s) ...")
    azure_rpt = run_abs_conformal_azure_backtest(job_limit=5880, sla_s=DEFAULT_SLA_S)
    print_report("Azure LLM 2024", azure_rpt)
    results["traces"]["azure_llm_2024"] = azure_rpt.to_dict()

    # ── BurstGPT HF ───────────────────────────────────────────────────────────
    if os.path.exists(DEFAULT_BURSTGPT_HF_JSONL):
        print()
        print("Running BurstGPT HF (5,880 requests, ρ=0.85, SLA=30s) ...")
        bgpt_rpt = run_abs_conformal_burstgpt_backtest(
            job_limit=5880, sla_s=DEFAULT_BURSTGPT_SLA_S
        )
        print_report("BurstGPT HF", bgpt_rpt)
        results["traces"]["burstgpt_hf"] = bgpt_rpt.to_dict()
    else:
        print(f"\n[SKIP] BurstGPT HF not found at {DEFAULT_BURSTGPT_HF_JSONL}")

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = os.path.join(_REPO, "research", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "abs_conformal_backtest_2026-06-22.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nResults saved → {out_json}")

    # ── Summary verdict ───────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("VERDICT SUMMARY")
    print("=" * 72)
    azure = results["traces"].get("azure_llm_2024", {})
    bgpt  = results["traces"].get("burstgpt_hf", {})
    if azure:
        _print_verdict("Azure LLM 2024", azure)
    if bgpt:
        _print_verdict("BurstGPT HF   ", bgpt)


def print_report(trace_name: str, rpt) -> None:
    print(f"\n{trace_name}:")
    print(f"  total_requests : {rpt.total_requests}")
    print(f"  SLA            : {rpt.sla_s}s")
    print()
    print(f"  {'Discipline':<30} {'Goodput/$':>14} {'vs FIFO':>10}  α / retention")
    print(f"  {'-'*30} {'-'*14} {'-'*10}  {'-'*24}")
    gp_fifo   = rpt.fifo_goodput_per_dollar
    gp_oracle = rpt.oracle_goodput_per_dollar
    gp_rel    = rpt.rel_conformal_goodput_per_dollar
    gp_abs    = rpt.abs_conformal_goodput_per_dollar
    _row("FIFO (baseline)",         gp_fifo,   0.0,                    "",          "")
    _row("Conformal oracle",        gp_oracle, rpt.oracle_delta_pct,   "α→0",       "100% retention")
    _row("Rel-conformal (live)",    gp_rel,    rpt.rel_conformal_delta_pct,
         f"α={rpt.rel_mean_alpha:.5f}",
         f"{rpt.rel_vs_oracle_retention_pct:.1f}% retention")
    _row("Abs-conformal (live) NEW",gp_abs,    rpt.abs_conformal_delta_pct,
         f"α={rpt.abs_mean_alpha:.5f}",
         f"{rpt.abs_vs_oracle_retention_pct:.1f}% retention")
    print()
    print(f"  ► p90 abs_err tokens (abs calibrator) : {rpt.abs_p90_abs_err_tokens:.0f} tokens")
    print(f"  ► abs_vs_rel improvement              : {rpt.abs_vs_rel_delta_pct:+.2f}%")
    frontier = "FRONTIER IMPROVEMENT" if rpt.abs_vs_rel_delta_pct > 0 else "NEUTRAL / NEGATIVE"
    print(f"  ► Frontier status                     : {frontier}")


def _row(label, gp, delta, alpha, retention):
    print(f"  {label:<30} {gp:>14.2f} {delta:>+9.2f}%  {alpha} {retention}")


def _print_verdict(label: str, d: dict) -> None:
    rel_delta = d.get("rel_conformal_delta_pct", 0.0)
    abs_delta = d.get("abs_conformal_delta_pct", 0.0)
    abs_vs_rel = d.get("abs_vs_rel_delta_pct", 0.0)
    abs_alpha = d.get("abs_mean_alpha", 0.0)
    rel_alpha = d.get("rel_mean_alpha", 0.0)
    print(f"\n  {label}:")
    print(f"    Rel-conformal vs FIFO : {rel_delta:+.2f}%")
    print(f"    Abs-conformal vs FIFO : {abs_delta:+.2f}%")
    print(f"    Abs vs Rel improvement: {abs_vs_rel:+.2f}%")
    print(f"    α: rel={rel_alpha:.5f}  abs={abs_alpha:.5f}")
    if abs_vs_rel > 0:
        print("    → FRONTIER IMPROVEMENT: abs-conformal BEATS rel-conformal")
    else:
        print("    → NEUTRAL/NEGATIVE: hypothesis NOT confirmed for this trace")


if __name__ == "__main__":
    main()
