"""Phase 4 benchmark: causal adaptive frontier rho vs fixed constraint_aware.

Runs the canonical BurstGPT and Azure LLM 2024 backtests comparing:
  - constraint_aware        (fixed rho=0.65, baseline)
  - constraint_aware_adaptive (causal rolling-window rho from frontier estimator)
  - sla_aware               (headline baseline)
  - safe_high_utilization   (fixed rho=0.75, upper reference)
  - fifo                    (FIFO do-nothing sanity check)

Primary KPI: SLA-safe goodput per infrastructure dollar.
Same-conditions rule: identical trace, SLA, cost denominator, physics.

Five-Failure-Rule compliance: this is "integrate existing module" work --
  compute_frontier_rho_schedule calls the existing estimate_frontier from
  aurelius/frontier/estimator.py with causal past-window telemetry.
  No new module, no oracle information, no new optimizer path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is importable regardless of cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import aurelius.traces.azure_llm as azure_llm
import aurelius.traces.burstgpt as burstgpt
from aurelius.traces.backtest import (
    HEADLINE_BASELINE,
    run_backtest,
)

# Canonical fixtures
_BURSTGPT_FIXTURE = _ROOT / "tests/fixtures/burstgpt_sample.csv"
_AZURE_FIXTURE = _ROOT / "tests/fixtures/azure_llm_2024_sample.csv"

# Policies to run — NOT the full ALL_POLICIES to keep runtime manageable.
# constraint_aware_adaptive is the Phase 4 candidate.
_POLICIES = (
    "fifo",
    "sla_aware",
    "constraint_aware",
    "constraint_aware_adaptive",
    "safe_high_utilization",
    "min_cost_safe",
)


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def run_trace(name: str, requests, tick_seconds: float = 60.0) -> dict:
    result = run_backtest(requests, tick_seconds=tick_seconds, policies=_POLICIES)

    rows = []
    ca = result.policy_results.get("constraint_aware")
    caa = result.policy_results.get("constraint_aware_adaptive")
    sla = result.policy_results.get(HEADLINE_BASELINE)

    ca_kpi = ca.kpi.sla_safe_goodput_per_infra_dollar if ca else None
    caa_kpi = caa.kpi.sla_safe_goodput_per_infra_dollar if caa else None
    sla_kpi = sla.kpi.sla_safe_goodput_per_infra_dollar if sla else None

    # Compute margins vs sla_aware headline
    def _margin(a, b):
        if a is None or b is None or b == 0:
            return None
        return (a - b) / b * 100.0

    ca_vs_sla = _margin(ca_kpi, sla_kpi)
    caa_vs_sla = _margin(caa_kpi, sla_kpi)
    caa_vs_ca = _margin(caa_kpi, ca_kpi)

    print(f"\n{'='*70}")
    print(f"Trace: {name}  |  {result.n_requests} requests  |  {result.n_ticks} ticks")
    print(f"{'='*70}")
    print(f"{'Policy':<30} {'goodput/$':>12} {'timeout%':>10} {'vs sla_aware':>14}")
    print(f"{'-'*70}")
    for pol, pr in result.policy_results.items():
        kpi = pr.kpi.sla_safe_goodput_per_infra_dollar
        trate = pr.timeout_rate_pct_mean
        margin = _margin(kpi, sla_kpi)
        marker = ""
        if pol == "constraint_aware_adaptive":
            marker = " ◄ Phase 4 candidate"
        elif pol == "constraint_aware":
            marker = " ◄ baseline"
        margin_str = f"{margin:+.2f}%" if margin is not None else "  N/A  "
        print(f"  {pol:<28} {_fmt(kpi):>12} {trate:>9.3f}% {margin_str:>14}{marker}")
    print(f"{'-'*70}")
    if caa_vs_ca is not None:
        print(f"  constraint_aware_adaptive vs constraint_aware: {caa_vs_ca:+.2f}%")
    print()

    summary = {
        "trace": name,
        "n_requests": result.n_requests,
        "n_ticks": result.n_ticks,
        "tick_seconds": tick_seconds,
        "policies": {
            pol: {
                "sla_safe_goodput_per_infra_dollar": pr.kpi.sla_safe_goodput_per_infra_dollar,
                "sla_compliant_goodput": pr.kpi.sla_compliant_goodput,
                "total_infrastructure_cost": pr.kpi.total_infrastructure_cost,
                "active_gpu_hours": pr.kpi.active_gpu_hours,
                "timeout_rate_pct_mean": pr.timeout_rate_pct_mean,
                "queue_p99_ms": pr.queue_p99_ms,
                "latency_p99_ms": pr.latency_p99_ms,
                "scale_events": pr.scale_events,
            }
            for pol, pr in result.policy_results.items()
        },
        "margins": {
            "constraint_aware_vs_sla_aware_pct": ca_vs_sla,
            "constraint_aware_adaptive_vs_sla_aware_pct": caa_vs_sla,
            "constraint_aware_adaptive_vs_constraint_aware_pct": caa_vs_ca,
        },
    }
    return summary


def main():
    results = {}
    errors = []

    # --- BurstGPT ---
    if _BURSTGPT_FIXTURE.exists():
        print(f"\nLoading BurstGPT fixture: {_BURSTGPT_FIXTURE}")
        try:
            reqs = burstgpt.load_csv(str(_BURSTGPT_FIXTURE))
            print(f"  {len(reqs)} requests loaded")
            results["burstgpt"] = run_trace("BurstGPT (fixture)", reqs)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(f"BurstGPT: {e}")
    else:
        print(f"BurstGPT fixture not found: {_BURSTGPT_FIXTURE}")
        errors.append("BurstGPT fixture missing")

    # --- Azure LLM 2024 ---
    if _AZURE_FIXTURE.exists():
        print(f"\nLoading Azure LLM 2024 fixture: {_AZURE_FIXTURE}")
        try:
            reqs = azure_llm.load_csv(str(_AZURE_FIXTURE))
            print(f"  {len(reqs)} requests loaded")
            results["azure_llm_2024"] = run_trace("Azure LLM 2024 (fixture)", reqs)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(f"Azure LLM 2024: {e}")
    else:
        print(f"Azure LLM 2024 fixture not found: {_AZURE_FIXTURE}")
        errors.append("Azure LLM 2024 fixture missing")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY: constraint_aware_adaptive vs constraint_aware")
    print("=" * 70)
    for trace_name, s in results.items():
        margins = s["margins"]
        caa_vs_ca = margins.get("constraint_aware_adaptive_vs_constraint_aware_pct")
        caa_vs_sla = margins.get("constraint_aware_adaptive_vs_sla_aware_pct")
        ca_timeout = s["policies"].get("constraint_aware", {}).get("timeout_rate_pct_mean")
        caa_timeout = s["policies"].get("constraint_aware_adaptive", {}).get("timeout_rate_pct_mean")
        print(f"\n  {trace_name}:")
        print(f"    adaptive vs CA baseline:     {caa_vs_ca:+.2f}% goodput/$" if caa_vs_ca is not None else "    N/A")
        print(f"    adaptive vs sla_aware:       {caa_vs_sla:+.2f}% goodput/$" if caa_vs_sla is not None else "    N/A")
        print(f"    CA timeout_pct_mean:         {ca_timeout:.3f}%" if ca_timeout is not None else "    N/A")
        print(f"    adaptive timeout_pct_mean:   {caa_timeout:.3f}%" if caa_timeout is not None else "    N/A")

    if errors:
        print(f"\n  Errors: {errors}")

    # Save to research/results/
    out_dir = _ROOT / "research/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase4_frontier_rho_results.json"
    with open(out_path, "w") as fh:
        json.dump({"phase": 4, "description": "causal frontier rho adaptation", "results": results, "errors": errors}, fh, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return results, errors


if __name__ == "__main__":
    results, errors = main()
    if errors and not results:
        sys.exit(1)
