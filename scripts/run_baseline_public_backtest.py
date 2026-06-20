#!/usr/bin/env python3
"""Phase 3 — current-main public backtest baseline (NO module integration).

Snapshots the current-main public economic KPIs across the three public paths,
before any research module is wired in:

  * BurstGPT serving replay (real trace)      -> constraint_aware vs baselines
  * Azure LLM 2024 serving replay (sample)    -> constraint_aware vs baselines
  * Canonical energy backtest (JobScheduler)  -> goodput/$, misses, migrations
  * GPU placement routing (scorer DISABLED)   -> baseline routing economics

Writes research/results/baseline_public_backtest_<date>.{json,md}.
Directional simulator/backtest evidence only — NOT production savings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import burstgpt  # noqa: E402
from aurelius.traces import azure_llm  # noqa: E402
from aurelius.traces.backtest import run_backtest  # noqa: E402
from aurelius.traces.schema import time_rescale  # noqa: E402

BURSTGPT_RAW = "data/external/burstgpt/raw/BurstGPT_1.csv"
BURSTGPT_FIXTURE = "tests/fixtures/burstgpt_sample.csv"
AZURE_FIXTURE = "tests/fixtures/azure_llm_2024_sample.csv"


def _serving_baseline(requests, scales, *, tick_seconds=60.0):
    rows = {}
    for scale in scales:
        rs = requests if scale == 1.0 else time_rescale(requests, scale)
        res = run_backtest(rs, tick_seconds=tick_seconds,
                           policies=("fifo", "sla_aware", "constraint_aware"))
        ca = res.policy_results["constraint_aware"]
        sa = res.policy_results["sla_aware"]
        k = ca.kpi
        rows[str(scale)] = {
            "n_ticks": res.n_ticks,
            "sla_safe_goodput_per_infra_dollar": k.sla_safe_goodput_per_infra_dollar,
            "gpu_hours": round(k.active_gpu_hours, 4),
            "total_cost": round(k.total_infrastructure_cost, 4),
            "sla_violation_timeout_pct": round(ca.timeout_rate_pct_mean, 4),
            "queue_p99_ms": round(ca.queue_p99_ms, 3),
            "migration_scale_events": ca.scale_events,
            "ca_vs_sla_aware_margin_pct": round(res.outcome.margin_pct, 3),
        }
    return rows


def _canonical_baseline():
    from aurelius.benchmarks.canonical_backtests import run_canonical_backtest
    t0 = time.time()
    rep = run_canonical_backtest()
    runtime = time.time() - t0
    policies = {}
    for name, pm in rep.policies.items():
        policies[name] = pm.to_dict() if hasattr(pm, "to_dict") else dict(pm)
    return {
        "runtime_s": round(runtime, 3),
        "job_count": rep.job_count,
        "policies": policies,
        "standalone_vs_wrapped_delta": rep.standalone_vs_wrapped_delta,
    }


def _gpu_routing_baseline():
    try:
        from aurelius.benchmarks.gpu_routing_backtest import run_gpu_routing_backtest
        rep = run_gpu_routing_backtest()
        d = rep.to_dict() if hasattr(rep, "to_dict") else dict(rep)
        return {
            "ok": True,
            "baseline_goodput_per_dollar": d.get("baseline_goodput_per_dollar"),
            "baseline_lc_goodput_per_dollar": d.get("baseline_lc_goodput_per_dollar"),
            "baseline_realized_energy_cost_usd": d.get("baseline_realized_energy_cost_usd"),
            "baseline_pct_on_best_gpu": d.get("baseline_pct_on_best_gpu"),
        }
    except Exception as e:  # pragma: no cover
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 3 baseline public backtest")
    p.add_argument("--sample-size", type=int, default=100000)
    p.add_argument("--burstgpt-scales", default="1,300")
    p.add_argument("--azure-scales", default="1,50")
    args = p.parse_args(argv)

    today = date.today().isoformat()
    prefix = f"research/results/baseline_public_backtest_{today}"

    bpath = BURSTGPT_RAW if os.path.exists(BURSTGPT_RAW) else BURSTGPT_FIXTURE
    bgpt = burstgpt.load_csv(bpath, sample_size=args.sample_size, seed=0)
    azure = azure_llm.load_csv(AZURE_FIXTURE)
    print(f"[baseline] BurstGPT {len(bgpt):,} reqs ({bpath}); Azure {len(azure):,} reqs")

    bgpt_scales = [float(s) for s in args.burstgpt_scales.split(",")]
    azure_scales = [float(s) for s in args.azure_scales.split(",")]

    print("[baseline] BurstGPT serving baseline ...")
    bgpt_rows = _serving_baseline(bgpt, bgpt_scales)
    print("[baseline] Azure-2024 serving baseline ...")
    azure_rows = _serving_baseline(azure, azure_scales)
    print("[baseline] canonical energy backtest ...")
    canon = _canonical_baseline()
    print("[baseline] GPU routing baseline (scorer disabled) ...")
    gpu = _gpu_routing_baseline()

    payload = {
        "generated": today,
        "directional_only_not_production_savings": True,
        "description": "Current-main public backtest baseline, no module integration.",
        "burstgpt": {"source": bpath, "n_requests": len(bgpt), "by_scale": bgpt_rows},
        "azure_llm_2024": {"source": AZURE_FIXTURE, "n_requests": len(azure),
                           "by_scale": azure_rows},
        "canonical_energy": canon,
        "gpu_routing_baseline": gpu,
    }
    os.makedirs("research/results", exist_ok=True)
    with open(prefix + ".json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(prefix + ".md", payload)
    print(f"[baseline] JSON -> {prefix}.json")
    print(f"[baseline] MD   -> {prefix}.md")
    return 0


def _g(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def _canon_summary(canon):
    """Extract the canonical constraint_aware KPI row + key baselines."""
    pols = canon.get("policies", {})
    keep = ("constraint_aware_with_energy_adapter", "current_price_only", "fifo")
    rows = {}
    for key in keep:
        if key in pols:
            p = pols[key]
            rows[key] = {
                "sla_safe_goodput_per_infra_dollar": p.get("sla_safe_goodput_per_infra_dollar"),
                "total_infra_cost_usd": p.get("total_infra_cost_usd"),
                "gpu_infra_cost_usd": p.get("gpu_infra_cost_usd"),
                "realized_energy_cost_usd": p.get("realized_energy_cost_usd"),
                "deadline_misses": p.get("deadline_misses"),
                "migrations": p.get("migrations"),
                "migration_cost_usd": p.get("network_cost_usd"),
                "sla_compliant_goodput": p.get("sla_compliant_goodput"),
            }
    return rows


def _write_md(path, payload):
    L = []
    A = L.append
    A("# Baseline Public Backtest — current main (no module integration)")
    A("")
    A("> **Directional simulator/backtest evidence only — NOT production "
      "savings** (`docs/RESULTS.md` §8). Phase-3 snapshot of current-main "
      "KPIs before any research module is wired in.")
    A("")
    A(f"- Generated: {payload['generated']}")
    A("")
    A("## Serving traces (aggregate autoscaling replay)")
    A("")
    A("`constraint_aware` is the Aurelius policy; KPIs are its values. Migration "
      "count = autoscaler scale events. Deadline-miss / migration-cost / "
      "optimizer-runtime do not apply to the trivial autoscaler (serving path).")
    A("")
    A("| dataset | load | SLA-safe goodput/$ | GPU-hours | cost | SLA viol (timeout %) | queue p99 (ms) | migration (scale ev) | CA vs sla_aware |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds_key, ds in (("burstgpt", payload["burstgpt"]),
                       ("azure_llm_2024", payload["azure_llm_2024"])):
        for scale, r in ds["by_scale"].items():
            A(f"| {ds_key} ({ds['n_requests']:,}) | {scale}× | "
              f"{_g(r['sla_safe_goodput_per_infra_dollar'])} | {_g(r['gpu_hours'])} | "
              f"{_g(r['total_cost'])} | {_g(r['sla_violation_timeout_pct'],3)} | "
              f"{_g(r['queue_p99_ms'])} | {r['migration_scale_events']} | "
              f"{r['ca_vs_sla_aware_margin_pct']:+.2f}% |")
    A("")
    A("## Canonical energy backtest (JobScheduler path)")
    A("")
    cs = _canon_summary(payload["canonical_energy"])
    A(f"- Solve runtime: {_g(payload['canonical_energy'].get('runtime_s'),3)} s · "
      f"jobs: {payload['canonical_energy'].get('job_count')}")
    A("")
    A("| policy | SLA-safe goodput/$ | total cost $ | energy $ | deadline misses | migrations | migration cost $ |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for name, r in cs.items():
        A(f"| {name} | {_g(r['sla_safe_goodput_per_infra_dollar'],5)} | "
          f"{_g(r['total_infra_cost_usd'])} | {_g(r['realized_energy_cost_usd'])} | "
          f"{r['deadline_misses']} | {r['migrations']} | {_g(r['migration_cost_usd'])} |")
    A("")
    A("## GPU routing baseline (GpuPlacementScorer DISABLED)")
    A("")
    g = payload["gpu_routing_baseline"]
    if g.get("ok"):
        A(f"- baseline goodput/$: {_g(g['baseline_goodput_per_dollar'],6)}")
        A(f"- baseline latency_critical goodput/$: {_g(g['baseline_lc_goodput_per_dollar'],6)}")
        A(f"- baseline realized energy $: {_g(g['baseline_realized_energy_cost_usd'])}")
        A(f"- baseline % latency_critical on best GPU: {_g(g['baseline_pct_on_best_gpu'],3)}")
    else:
        A(f"- not run: {g.get('error')}")
    A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
