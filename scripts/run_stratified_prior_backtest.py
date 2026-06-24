#!/usr/bin/env python3
"""Standalone runner: Stratified Feature-Aware Causal Prior — BurstGPT HF backtest.

Run 2026-06-22-u. Tests whether per-(model_id, input_bin) stratified running-median
prior improves on the global running-median prior from run -t.

Usage:
    python3 scripts/run_stratified_prior_backtest.py [--limit N] [--servers S] [--rho R]

Writes results to stdout as JSON and exits with:
  0 — run completed (result may be positive or negative)
  1 — dataset not available / import error
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified prior BurstGPT HF backtest")
    parser.add_argument("--limit", type=int, default=5880, help="Request limit (default 5880)")
    parser.add_argument("--servers", type=int, default=4, help="Server count (default 4)")
    parser.add_argument("--rho", type=float, default=0.85, help="Target utilization (default 0.85)")
    parser.add_argument("--min-stratum-history", type=int, default=20,
                        help="Min stratum completions before stratum-level prior fires (default 20)")
    args = parser.parse_args()

    try:
        from aurelius.benchmarks.srtf_serving_backtest import (
            DEFAULT_BURSTGPT_SLA_S,
            LIVE_PRIOR_WINDOW,
            run_burstgpt_hf_stratified_prior_backtest,
        )
    except ImportError as exc:
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    import os

    from aurelius.benchmarks.srtf_serving_backtest import DEFAULT_BURSTGPT_HF_JSONL

    if not os.path.exists(DEFAULT_BURSTGPT_HF_JSONL):
        print(
            f"BurstGPT HF dataset not found at {DEFAULT_BURSTGPT_HF_JSONL}. "
            "Download from https://huggingface.co/datasets/lzzmm/BurstGPT (CC-BY-4.0).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Running stratified prior backtest: servers={args.servers} rho={args.rho} "
        f"limit={args.limit} min_stratum_history={args.min_stratum_history}",
        file=sys.stderr,
    )

    report = run_burstgpt_hf_stratified_prior_backtest(
        servers=args.servers,
        target_rho=args.rho,
        job_limit=args.limit,
        sla_s=DEFAULT_BURSTGPT_SLA_S,
        prior_window=LIVE_PRIOR_WINDOW,
        min_stratum_history=args.min_stratum_history,
    )

    result = report.to_dict()
    print(json.dumps(result, indent=2))

    # Print human-readable summary to stderr
    print("\n=== Stratified Prior Backtest Results ===", file=sys.stderr)
    print(f"  FIFO goodput/$:              {result['fifo_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Oracle goodput/$:            {result['oracle_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Global prior goodput/$:      {result['global_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Stratified prior goodput/$:  {result['stratified_goodput_per_dollar']:>12,.2f}", file=sys.stderr)
    print(f"  Oracle delta vs FIFO:        {result['oracle_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Global delta vs FIFO:        {result['global_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Stratified delta vs FIFO:    {result['stratified_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Global retention:            {result['global_vs_oracle_retention_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Stratified retention:        {result['stratified_vs_oracle_retention_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Stratified vs global:        {result['stratified_vs_global_improvement_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Global prior MAE (tokens):   {result['global_prior_mae_tokens']:>+.1f}", file=sys.stderr)
    print(f"  Stratified MAE (tokens):     {result['stratified_prior_mae_tokens']:>+.1f}", file=sys.stderr)
    print(f"  Stratum-level usage:         {result['stratified_stratum_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Model-level usage:           {result['stratified_model_pct']:>+.1f}%", file=sys.stderr)
    print(f"  Global fallback usage:       {result['stratified_fallback_pct']:>+.1f}%", file=sys.stderr)


if __name__ == "__main__":
    main()
