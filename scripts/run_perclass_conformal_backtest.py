#!/usr/bin/env python3
"""Standalone runner: Per-Class Conformal α Calibration — BurstGPT HF backtest.

Run 2026-06-22-w. Tests whether per-class (per-model_id) conformal calibrators
break the binding constraint from run -v: the global calibrator saturates at
α=0.002 for ALL requests because ChatGPT's bimodal within-class variance
dominates the p90 error, masking GPT-4's lower prediction uncertainty.

Root cause (run -v):
  Global calibrator p90 error ≥ 0.80 due to ChatGPT surprise-long requests
  (~10% of ChatGPT, rel_err≈0.99) → α = 0.002 for ALL, even accurate GPT-4.

Per-class fix:
  Separate ConformalAlphaCalibrator per model_id (ChatGPT vs GPT-4).
  GPT-4 calibrator sees ONLY GPT-4 errors under GPT-4-specific prior.
  GPT-4 running median ≈ 235 tokens → lower errors → α_GPT4 → 0 → pure SRPT
  for GPT-4 dispatch, while ChatGPT retains safe α_ChatGPT = 0.002.

Conditions compared (BurstGPT HF, CC-BY-4.0):
  1. FIFO (baseline)
  2. Oracle + global mono calibrator (+644.4% vs FIFO [run -r])
  3. Global prior + global mono calibrator (+420.83% vs FIFO [run -t])
  4. Stratified prior + global mono calibrator (run -u)
  5. Stratified prior + per-class calibrator [NEW]

Usage:
    python3 scripts/run_perclass_conformal_backtest.py [--limit N] [--servers S] [--rho R]

Writes results to stdout as JSON and exits with:
  0 — run completed (result may be positive or negative)
  1 — dataset not available / import error
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-class conformal α BurstGPT HF backtest")
    parser.add_argument("--limit", type=int, default=5880,
                        help="Request limit (default 5880 for fast iteration; use 0 for all)")
    parser.add_argument("--servers", type=int, default=4, help="Server count (default 4)")
    parser.add_argument("--rho", type=float, default=0.85, help="Target utilization (default 0.85)")
    parser.add_argument("--min-stratum-history", type=int, default=20,
                        help="Min stratum completions before stratum-level prior fires (default 20)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional file path to write JSON result (also written to stdout)")
    args = parser.parse_args()

    limit = args.limit if args.limit > 0 else None

    try:
        from aurelius.benchmarks.srtf_serving_backtest import (
            DEFAULT_BURSTGPT_HF_JSONL,
            DEFAULT_BURSTGPT_SLA_S,
            LIVE_PRIOR_WINDOW,
            run_burstgpt_hf_perclass_conformal_backtest,
        )
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    import os
    if not os.path.exists(DEFAULT_BURSTGPT_HF_JSONL):
        print(
            f"BurstGPT HF dataset not found at {DEFAULT_BURSTGPT_HF_JSONL}. "
            "Download from https://huggingface.co/datasets/lzzmm/BurstGPT (CC-BY-4.0).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Running per-class conformal backtest: servers={args.servers} rho={args.rho} "
        f"limit={limit} min_stratum_history={args.min_stratum_history}",
        file=sys.stderr,
    )

    report = run_burstgpt_hf_perclass_conformal_backtest(
        servers=args.servers,
        target_rho=args.rho,
        job_limit=limit,
        sla_s=DEFAULT_BURSTGPT_SLA_S,
        prior_window=LIVE_PRIOR_WINDOW,
        min_stratum_history=args.min_stratum_history,
    )

    result = report.to_dict()
    json_str = json.dumps(result, indent=2)
    print(json_str)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json_str)
        print(f"\nResult written to {args.out}", file=sys.stderr)

    # Human-readable summary on stderr
    print("\n=== Per-Class Conformal α Backtest Results (run 2026-06-22-w) ===", file=sys.stderr)
    print(f"  Trace:                        {result['trace']}", file=sys.stderr)
    print(f"  Requests:                     {result['total_requests']}", file=sys.stderr)
    print(f"  Servers:                      {result['servers']} (ρ={result['target_rho']})", file=sys.stderr)
    print("", file=sys.stderr)
    print("  ── Goodput/$ ─────────────────────────────────────────────────", file=sys.stderr)
    print(f"  FIFO baseline:                {result['fifo_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Oracle (upper bound):         {result['oracle_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Global prior mono:            {result['global_mono_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Stratified prior mono:        {result['stratified_mono_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Stratified prior per-class:   {result['stratified_perclass_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  ── Delta vs FIFO ─────────────────────────────────────────────", file=sys.stderr)
    print(f"  Oracle delta:                 {result['oracle_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Global mono delta:            {result['global_mono_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Stratified mono delta:        {result['stratified_mono_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Stratified per-class delta:   {result['stratified_perclass_delta_pct']:>+.2f}%", file=sys.stderr)
    print("", file=sys.stderr)
    print("  ── Oracle Retention ──────────────────────────────────────────", file=sys.stderr)
    print(f"  Global mono retention:        {result['global_vs_oracle_retention_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Stratified mono retention:    {result['stratified_mono_vs_oracle_retention_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Stratified per-class ret:     {result['stratified_perclass_vs_oracle_retention_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Per-class vs mono gain:       {result['perclass_vs_mono_improvement_pct']:>+.2f}%", file=sys.stderr)
    print("", file=sys.stderr)
    print("  ── Prior Quality ─────────────────────────────────────────────", file=sys.stderr)
    print(f"  Global prior MAE (tokens):    {result['global_prior_mae_tokens']:>+.1f}", file=sys.stderr)
    print(f"  Stratified prior MAE (tokens):{result['stratified_prior_mae_tokens']:>+.1f}", file=sys.stderr)
    print(f"  Stratified stratum usage:     {result['stratified_stratum_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Stratified model usage:       {result['stratified_model_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Stratified fallback usage:    {result['stratified_fallback_pct']:>+.1f}%", file=sys.stderr)
    print("", file=sys.stderr)
    print("  ── Per-Class Calibrator Diagnostics ──────────────────────────", file=sys.stderr)
    for cls, info in result.get("perclass_diagnostics", {}).items():
        if isinstance(info, dict):
            n_comp = info.get("n_completed", "?")
            n_disp = info.get("n_dispatched", "?")
            mean_a = info.get("mean_dispatch_alpha", "?")
            has_wu = info.get("has_warmup", "?")
            print(
                f"  [{cls}] completed={n_comp} dispatched={n_disp} "
                f"mean_α={mean_a} warmup={has_wu}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
