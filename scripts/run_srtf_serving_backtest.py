#!/usr/bin/env python3
"""CANONICAL_SRTF_SERVING_BACKTEST_V1 — public benchmark script for the
request-level SRTF/conformal serving queue evaluation.

This script makes the SRTF/conformal serving queue results (previously only
accessible via the pytest test suite) available as a standalone, reproducible
public benchmark [ROADMAP priority #1: "Wire conformal+decoupled into serving
runtime"].

What this measures
------------------
Per-request queue discipline effects on SLA-safe goodput per dollar, evaluated
on two public real-world LLM serving traces using a discrete-event M/G/c queue
simulator.  Three disciplines are compared on each trace:

  fifo             — serve in arrival order (no prediction, baseline)
  conformal_oracle — SRPT with oracle token counts (upper bound, cheats)
  conformal_live   — SRPT with causal live prior (production-viable, causal)

The causal live prior is a sliding-window median over recent completions:
for request i, the predicted token count is the median of the last 200
completed requests (i-1, i-2, ...) — zero external model, strictly causal,
no future leakage.

Public traces used
------------------
1. **Azure LLM 2024** (DynamoLLM HPCA 2025, Microsoft Azure)
   - 5,880-record sample from the 44.1M request, 9-day production trace
   - Fixture: tests/fixtures/azure_llm_2024_sample.csv
   - Headline [run 2026-06-21-t]: +244.42% vs FIFO (81.6% oracle retention)

2. **BurstGPT HF** (CC-BY-4.0, arXiv:2401.17644)
   - 5,880-record sample from 59,999 normalized HF records
   - Source: data/external/hf/lzzmm__BurstGPT/burstgpt_1_full/processed/
   - Headline [run 2026-06-21-t]: +420.83% vs FIFO (70% oracle retention)

KPIs reported
-------------
  sla_safe_goodput_per_dollar  — primary metric (tokens × SLA-safety / infra-$)
  oracle_delta_pct             — oracle vs FIFO (maximum theoretical gain)
  live_delta_pct               — live prior vs FIFO (production-viable gain)
  live_vs_oracle_retention_pct — % of oracle gain retained by live prior

Outputs
-------
  * docs/SRTF_SERVING_BACKTEST_RESULTS.md
  * data/external/srtf_serving/processed/srtf_serving_backtest_summary.json
  (or stdout only if --no-write is passed)

Honesty / non-goals (docs/RESULTS.md §8)
-----------------------------------------
Simulator / public-trace directional result — **not** production savings.
Service time uses actual token counts (not predicted); the only variable
across disciplines is the queue ordering key.  The infra-dollar denominator
(GPU busy-time × GPU_HOUR_USD) is identical across disciplines because every
discipline processes the same request set on the same servers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    LIVE_PRIOR_WINDOW,
    LivePriorReport,
    run_burstgpt_hf_live_prior_backtest,
    run_live_prior_conformal_backtest,
)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(
    _REPO_ROOT, "data", "external", "srtf_serving", "processed",
    "srtf_serving_backtest_summary.json"
)
OUT_MD = os.path.join(_REPO_ROOT, "docs", "SRTF_SERVING_BACKTEST_RESULTS.md")


def _fmt_delta(pct: float) -> str:
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.2f}%"


def _run_azure(servers: int, target_rho: float, job_limit: int | None) -> LivePriorReport:
    print(f"  Running Azure LLM 2024 (servers={servers}, ρ={target_rho}, "
          f"limit={job_limit or 'all'})...")
    return run_live_prior_conformal_backtest(
        servers=servers,
        target_rho=target_rho,
        job_limit=job_limit,
        sla_s=DEFAULT_SLA_S,
        prior_window=LIVE_PRIOR_WINDOW,
        azure_fixture=DEFAULT_AZURE_FIXTURE,
    )


def _run_burstgpt(servers: int, target_rho: float, job_limit: int | None) -> LivePriorReport:
    if not os.path.isfile(DEFAULT_BURSTGPT_HF_JSONL):
        print(f"  [SKIP] BurstGPT HF JSONL not found at {DEFAULT_BURSTGPT_HF_JSONL}")
        return None  # type: ignore[return-value]
    print(f"  Running BurstGPT HF (servers={servers}, ρ={target_rho}, "
          f"limit={job_limit or 'all'})...")
    return run_burstgpt_hf_live_prior_backtest(
        servers=servers,
        target_rho=target_rho,
        job_limit=job_limit,
        sla_s=DEFAULT_BURSTGPT_SLA_S,
        prior_window=LIVE_PRIOR_WINDOW,
        jsonl_path=DEFAULT_BURSTGPT_HF_JSONL,
    )


def _format_report(r: LivePriorReport, trace_label: str) -> str:
    lines: list[str] = []
    lines.append(f"\n### {trace_label}")
    lines.append(f"- Requests: {r.total_requests:,} | Servers: {r.servers} | "
                 f"ρ={r.target_rho} | SLA={r.sla_s}s | Window={r.prior_window}")
    lines.append(
        f"- Prior quality: CV={r.prior_cv_pct:.1f}% | "
        f"MAE={r.prior_mae_tokens:.1f} tok | "
        f"RelMAE={r.prior_rel_mae_pct:.1f}%"
    )
    lines.append("")
    lines.append("| Discipline            | SLA-safe goodput/$ | vs FIFO      |")
    lines.append("|----------------------|-------------------:|-------------:|")
    lines.append(
        f"| FIFO (baseline)      | {r.fifo_goodput_per_dollar:>18,.0f} | —            |"
    )
    lines.append(
        f"| Conformal oracle     | {r.oracle_goodput_per_dollar:>18,.0f} | "
        f"{_fmt_delta(r.oracle_delta_pct):>12} |"
    )
    lines.append(
        f"| Conformal live prior | {r.live_goodput_per_dollar:>18,.0f} | "
        f"{_fmt_delta(r.live_delta_pct):>12} |"
    )
    lines.append("")
    lines.append(
        f"**Oracle retention: {r.live_vs_oracle_retention_pct:.1f}%** "
        f"(live prior retains {r.live_vs_oracle_retention_pct:.1f}% of oracle gain; "
        f"production-viable threshold ≥83%)"
    )
    lines.append("")
    lines.append(
        f"> Shadow tag: `{r.shadow_tag}`"
    )
    return "\n".join(lines)


def _results_table(az_r: LivePriorReport, bg_r: LivePriorReport | None) -> str:
    rows = [
        "| KPI | Azure LLM 2024 | BurstGPT HF | Unit |",
        "|-----|---------------:|------------:|------|",
    ]

    def _cell(r: LivePriorReport | None, attr: str, fmt: str) -> str:
        if r is None:
            return "—"
        v = getattr(r, attr)
        return format(v, fmt)

    kpis = [
        ("oracle_delta_pct",                  ".2f", "% vs FIFO"),
        ("live_delta_pct",                    ".2f", "% vs FIFO"),
        ("live_vs_oracle_retention_pct",      ".1f", "% of oracle"),
        ("fifo_goodput_per_dollar",           ",.0f", "tokens/$"),
        ("oracle_goodput_per_dollar",         ",.0f", "tokens/$"),
        ("live_goodput_per_dollar",           ",.0f", "tokens/$"),
        ("prior_cv_pct",                      ".1f", "%"),
        ("prior_mae_tokens",                  ".1f", "tokens"),
    ]
    labels = {
        "oracle_delta_pct": "Oracle vs FIFO",
        "live_delta_pct": "Live prior vs FIFO",
        "live_vs_oracle_retention_pct": "Oracle retention",
        "fifo_goodput_per_dollar": "FIFO goodput/$",
        "oracle_goodput_per_dollar": "Oracle goodput/$",
        "live_goodput_per_dollar": "Live goodput/$",
        "prior_cv_pct": "Prior CV",
        "prior_mae_tokens": "Prior MAE",
    }

    for attr, fmt, unit in kpis:
        az_val = _cell(az_r, attr, fmt)
        bg_val = _cell(bg_r, attr, fmt)
        rows.append(f"| {labels[attr]} | {az_val} | {bg_val} | {unit} |")

    return "\n".join(rows)


def _write_markdown(az_r: LivePriorReport, bg_r: LivePriorReport | None,
                    timestamp: str) -> str:
    lines = [
        "# SRTF Serving Queue Backtest Results",
        "",
        f"Generated: {timestamp}",
        "",
        "## Summary",
        "",
        "Request-level SRTF/conformal queue discipline evaluation on two public "
        "LLM serving traces.",
        "Service physics: M/G/c discrete-event queue, identical across disciplines.",
        "All differences in goodput/$ come purely from queue ordering.",
        "",
        "**Disciplines:**",
        "- `fifo` — arrival order (no prediction)",
        "- `conformal_oracle` — decoupled hybrid SRPT with oracle token counts (upper bound)",
        "- `conformal_live` — decoupled hybrid SRPT with causal sliding-window median prior",
        "",
        "## Results Summary",
        "",
        _results_table(az_r, bg_r),
        "",
        "## Detailed Results",
        "",
        _format_report(az_r, "Azure LLM 2024"),
    ]
    if bg_r is not None:
        lines.append(_format_report(bg_r, "BurstGPT HF"))

    lines += [
        "",
        "## Prior Benchmarks (from ROADMAP)",
        "",
        "| Run | Trace | Discipline | Result vs FIFO | Oracle Retention |",
        "|-----|-------|------------|---------------:|-----------------|",
        "| 2026-06-21-t | Azure LLM 2024 | Conformal live prior | +244.42% | 81.6% |",
        "| 2026-06-21-t | BurstGPT 5,880 | Conformal live prior | +420.83% | 70.0% |",
        "| 2026-06-21-q | Azure LLM 2024 | Conformal oracle     | +322.24% | 100%  |",
        "| 2026-06-21-r | BurstGPT 5,880 | Conformal oracle     | +644.4%  | 100%  |",
        "",
        "## Methodology",
        "",
        "- **SLA-safe goodput/$**: `Σ actual_tokens[i where e2e_latency ≤ sla_s] / "
        "(Σ service_s / 3600 × GPU_HOUR_USD)`",
        "- **Infra-dollar denominator**: `GPU_HOUR_USD = $2.00/replica-hour` × "
        "total service seconds — identical across disciplines",
        "- **Service time**: `TTFT_BASE_S (0.15s) + output_tokens × TPOT_S (0.02s)`",
        "- **Time warp**: arrivals rescaled to `target_rho=0.85` cluster utilization, "
        "applied identically to all disciplines",
        "- **Causal prior**: prediction[i] = median of actual_tokens[0..i-1] "
        "(last 200 completions), no future leakage",
        "",
        "Directional simulator evidence — **not** production savings "
        "(docs/RESULTS.md §8).",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="SRTF serving queue public benchmark")
    parser.add_argument(
        "--job-limit", type=int, default=5880,
        help="Max requests per trace (default: 5880, matching prior runs)"
    )
    parser.add_argument(
        "--full-scale", action="store_true",
        help="Override --job-limit: run on full HF dataset (all records)"
    )
    parser.add_argument(
        "--servers", type=int, default=4,
        help="Replica pool size (default: 4)"
    )
    parser.add_argument(
        "--rho", type=float, default=0.85,
        help="Target cluster utilization (default: 0.85)"
    )
    parser.add_argument(
        "--azure-only", action="store_true",
        help="Run Azure trace only (skip BurstGPT)"
    )
    parser.add_argument(
        "--burstgpt-only", action="store_true",
        help="Run BurstGPT trace only (skip Azure)"
    )
    parser.add_argument(
        "--no-write", action="store_true",
        help="Print results to stdout only, do not write files"
    )
    args = parser.parse_args()

    job_limit: int | None = None if args.full_scale else args.job_limit
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    print("=" * 70)
    print("SRTF SERVING QUEUE BACKTEST — PUBLIC BENCHMARK")
    print(f"Timestamp: {timestamp}")
    print(f"Servers: {args.servers} | ρ={args.rho} | "
          f"Job limit: {job_limit or 'full scale'}")
    print("=" * 70)

    az_r: LivePriorReport | None = None
    bg_r: LivePriorReport | None = None

    if not args.burstgpt_only:
        print("\n[1/2] Azure LLM 2024")
        az_r = _run_azure(args.servers, args.rho, job_limit)
        print(f"  → FIFO: {az_r.fifo_goodput_per_dollar:,.0f} goodput/$")
        print(f"  → Oracle: +{az_r.oracle_delta_pct:.2f}% vs FIFO")
        print(f"  → Live prior: +{az_r.live_delta_pct:.2f}% vs FIFO "
              f"({az_r.live_vs_oracle_retention_pct:.1f}% oracle retention)")

    if not args.azure_only:
        print("\n[2/2] BurstGPT HF")
        bg_r = _run_burstgpt(args.servers, args.rho, job_limit)
        if bg_r is not None:
            print(f"  → FIFO: {bg_r.fifo_goodput_per_dollar:,.0f} goodput/$")
            print(f"  → Oracle: +{bg_r.oracle_delta_pct:.2f}% vs FIFO")
            print(f"  → Live prior: +{bg_r.live_delta_pct:.2f}% vs FIFO "
                  f"({bg_r.live_vs_oracle_retention_pct:.1f}% oracle retention)")

    if az_r is None and bg_r is None:
        print("\nNo traces completed.")
        return

    # Composite results
    if az_r is not None:
        print("\n" + "─" * 70)
        print("AZURE SUMMARY")
        print(f"  FIFO goodput/$:   {az_r.fifo_goodput_per_dollar:>12,.0f}")
        print(f"  Oracle goodput/$: {az_r.oracle_goodput_per_dollar:>12,.0f}  "
              f"({_fmt_delta(az_r.oracle_delta_pct)} vs FIFO)")
        print(f"  Live goodput/$:   {az_r.live_goodput_per_dollar:>12,.0f}  "
              f"({_fmt_delta(az_r.live_delta_pct)} vs FIFO)")
        print(f"  Oracle retention: {az_r.live_vs_oracle_retention_pct:.1f}%")

    if bg_r is not None:
        print("\n" + "─" * 70)
        print("BURSTGPT SUMMARY")
        print(f"  FIFO goodput/$:   {bg_r.fifo_goodput_per_dollar:>12,.0f}")
        print(f"  Oracle goodput/$: {bg_r.oracle_goodput_per_dollar:>12,.0f}  "
              f"({_fmt_delta(bg_r.oracle_delta_pct)} vs FIFO)")
        print(f"  Live goodput/$:   {bg_r.live_goodput_per_dollar:>12,.0f}  "
              f"({_fmt_delta(bg_r.live_delta_pct)} vs FIFO)")
        print(f"  Oracle retention: {bg_r.live_vs_oracle_retention_pct:.1f}%")

    if args.no_write:
        return

    # Write outputs
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "config": {
            "servers": args.servers,
            "target_rho": args.rho,
            "job_limit": job_limit,
        },
        "results": {},
    }
    if az_r is not None:
        summary["results"]["azure_llm_2024"] = az_r.to_dict()
    if bg_r is not None:
        summary["results"]["burstgpt_hf"] = bg_r.to_dict()

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON results written to: {OUT_JSON}")

    if az_r is not None:
        md = _write_markdown(az_r, bg_r, timestamp)
        with open(OUT_MD, "w") as f:
            f.write(md)
        print(f"Markdown results written to: {OUT_MD}")

    print("\nShadow tag: shadow_only_simulator_result_not_production_savings")


if __name__ == "__main__":
    main()
