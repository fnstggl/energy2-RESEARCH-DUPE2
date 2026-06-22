#!/usr/bin/env python3
"""Standalone runner: Absolute-Error Conformal Calibration — BurstGPT HF backtest.

Run 2026-06-22-x.  Tests whether switching the conformal calibrator from relative
error (|pred−actual|/actual) to absolute error (|pred−actual| in tokens) breaks the
α=0.002 ceiling confirmed in runs -u through -w.

Root cause: the relative-error formula is dominated by BurstGPT's 8.4% surprise-long
ChatGPT requests (actual ≈ 1 500 tokens, pred ≈ 70 tokens → rel_err ≈ 20), pushing
p90_rel above 0.40 even within the ChatGPT class alone.

Absolute error fix: p90_abs for ChatGPT lands in the short-request bucket (91.6% of
ChatGPT has |pred−actual| < 50 tokens), so p90_abs ≈ 30–50 tokens and α drops from
0.002 to ≈ 0.001, enabling more SRPT-like dispatch for 84% of traffic.

Literature basis:
  - Romano et al. NeurIPS 2019 (CQR): absolute residuals avoid constant-width
    interval inflation for heteroscedastic workloads.
  - Dewolf et al. 2023 (arXiv:2309.08313): formally proves relative-error CP
    fails conditional validity under heteroscedasticity.
  - TIE (arXiv:2604.00499, ICML 2026): distributional approach for LLM scheduling.

Usage:
    python3 scripts/run_abs_err_conformal_backtest.py [--limit N] [--servers S]
                                                       [--rho R] [--target-abs T]

Exits with:
  0 — run completed (result may be positive or negative; check stdout JSON)
  1 — dataset not available / import error
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Absolute-error conformal calibration BurstGPT HF backtest [run 2026-06-22-x]"
    )
    parser.add_argument(
        "--limit", type=int, default=5880,
        help="Request limit (default 5880 = canonical HF sample)"
    )
    parser.add_argument("--servers", type=int, default=4, help="Server count (default 4)")
    parser.add_argument("--rho", type=float, default=0.85, help="Target utilization (default 0.85)")
    parser.add_argument(
        "--target-abs", type=float, default=50.0,
        help="Target p90 absolute token error for α=alpha_max mapping (default 50)"
    )
    parser.add_argument("--ml-warmup", type=int, default=300, help="ML-HGB warmup (default 300)")
    args = parser.parse_args()

    try:
        from aurelius.benchmarks.srtf_serving_backtest import (
            DEFAULT_BURSTGPT_HF_JSONL,
            DEFAULT_BURSTGPT_SLA_S,
            run_burstgpt_hf_abs_err_conformal_backtest,
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
        f"Running abs-err conformal backtest: servers={args.servers} rho={args.rho} "
        f"limit={args.limit} target_abs={args.target_abs} ml_warmup={args.ml_warmup}",
        file=sys.stderr,
    )

    report = run_burstgpt_hf_abs_err_conformal_backtest(
        servers=args.servers,
        target_rho=args.rho,
        job_limit=args.limit,
        sla_s=DEFAULT_BURSTGPT_SLA_S,
        ml_warmup_n=args.ml_warmup,
        target_abs_tokens=args.target_abs,
    )

    result = report.to_dict()
    print(json.dumps(result, indent=2))

    # Human-readable summary to stderr
    print("\n=== Absolute-Error Conformal Calibration Results [run 2026-06-22-x] ===",
          file=sys.stderr)
    print(f"  target_abs_tokens:                  {result['target_abs_tokens']:>8.1f} tok",
          file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  FIFO goodput/$:                     {result['fifo_goodput_per_dollar']:>12,.2f}",
          file=sys.stderr)
    print(f"  Oracle goodput/$:                   {result['oracle_goodput_per_dollar']:>12,.2f}",
          file=sys.stderr)
    print(f"  Global relative goodput/$:          {result['global_rel_goodput_per_dollar']:>12,.2f}",
          file=sys.stderr)
    print(f"  Global absolute goodput/$:          {result['global_abs_goodput_per_dollar']:>12,.2f}",
          file=sys.stderr)
    print(f"  Per-class absolute goodput/$:       {result['per_class_abs_goodput_per_dollar']:>12,.2f}",
          file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  Oracle delta vs FIFO:               {result['oracle_delta_pct']:>+.2f}%", file=sys.stderr)
    print(f"  Global relative delta vs FIFO:      {result['global_rel_delta_pct']:>+.2f}%",
          file=sys.stderr)
    print(f"  Global absolute delta vs FIFO:      {result['global_abs_delta_pct']:>+.2f}%",
          file=sys.stderr)
    print(f"  Per-class absolute delta vs FIFO:   {result['per_class_abs_delta_pct']:>+.2f}%",
          file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  Global rel oracle retention:        {result['global_rel_vs_oracle_retention_pct']:>+.1f}%",
          file=sys.stderr)
    print(f"  Global abs oracle retention:        {result['global_abs_vs_oracle_retention_pct']:>+.1f}%",
          file=sys.stderr)
    print(f"  Per-class abs oracle retention:     {result['per_class_abs_vs_oracle_retention_pct']:>+.1f}%",
          file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  Global abs vs rel improvement:      {result['global_abs_vs_rel_pct']:>+.2f}%",
          file=sys.stderr)
    print(f"  Per-class abs vs rel improvement:   {result['per_class_abs_vs_rel_pct']:>+.2f}%",
          file=sys.stderr)
    print("", file=sys.stderr)
    print(f"  Global rel mean alpha:              {result['global_rel_mean_alpha']:>.6f}",
          file=sys.stderr)
    print(f"  Global abs mean alpha:              {result['global_abs_mean_alpha']:>.6f}",
          file=sys.stderr)
    if result.get("per_class_abs_mean_alpha"):
        for cls, a in result["per_class_abs_mean_alpha"].items():
            print(f"  Per-class abs mean alpha ({cls[:12]:12s}): {a:.6f}", file=sys.stderr)
    if result.get("per_class_abs_class_counts"):
        for cls, cnt in result["per_class_abs_class_counts"].items():
            print(f"  Class completions ({cls[:12]:12s}):           {cnt}", file=sys.stderr)


if __name__ == "__main__":
    main()
