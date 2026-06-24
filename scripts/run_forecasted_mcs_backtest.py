#!/usr/bin/env python3
"""Forecasted MCS backtest — deployable vs oracle capacity provisioning.

Apples-to-apples comparison under ONE physics model (Erlang-C gate +
sequential-decode service time), ONE cost denominator (provisioned GPU-hours
over the fixed trace window × $2/hr), ONE SLA definition, ONE trace, ONE
discrete-event simulator family.

Policy families (all share the cost denominator and SLA definition):

  Fixed-capacity baselines (deployable; NO MCS), swept over c to find the
  strongest operating point:
    * FIFO + fixed c
    * SLA-aware + fixed c            (binary SLA-class priority; the documented
                                      "SLA-oracle" baseline is this at c=4)
  Variable-capacity MCS (FIFO discipline):
    * reactive lag-1 MCS             deployable, naive "last tick" forecast
    * forecast MCS (EWMA)            deployable, causal point forecast
    * forecast MCS (quantile p90+1σ) deployable, causal robust forecast
    * oracle MCS                     UPPER BOUND — peeks at tick-t actuals
  Compound (variable capacity + best Aurelius queue discipline):
    * forecast MCS (EWMA) + abs-conformal SRTF    deployable
    * oracle MCS + abs-conformal SRTF             upper bound

North-star gate (Phase 7): forecasted MCS must beat the strongest *deployable*
SLA-aware baseline by +300% (4×) to claim north-star. Oracle MCS is an upper
bound only.

Directional simulator evidence only — NOT production savings
(``docs/RESULTS.md`` §8).

Writes:
  * research/results/forecasted_mcs_backtest_<date>.json
  * research/results/forecasted_mcs_backtest_<date>.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.benchmarks.forecasted_mcs import (  # noqa: E402
    bucketize,
    evaluate_c_schedule,
    forecast_mcs_c_schedule,
    reactive_lag1_c_schedule,
)
from aurelius.benchmarks.srtf_serving_backtest import (  # noqa: E402
    DEFAULT_AZURE_FIXTURE,
    DEFAULT_BURSTGPT_HF_JSONL,
    DEFAULT_BURSTGPT_SLA_S,
    DEFAULT_SLA_S,
    _joint_mcs_c_schedule,
    calibrate_time_warp,
    load_burstgpt_serving_requests_jsonl,
    load_serving_requests,
    make_live_prior_predictions,
)

JOB_LIMIT = 5880
FIXED_C = 4
TARGET_RHO = 0.85
TICK_S = 60.0
MCS_GATE = 9.5
FIXED_C_SWEEP = (4, 5, 6, 7, 8, 10, 12)


def _delta_pct(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return (a - b) / b * 100.0


def _run_trace(name: str, raw: list, sla_s: float) -> dict:
    warp = calibrate_time_warp(raw, servers=FIXED_C, target_rho=TARGET_RHO)
    _counts, _toks, n_ticks = bucketize(raw, TICK_S, warp)

    results: list[dict] = []

    # --- Fixed-capacity baselines: sweep c for FIFO and SLA-aware ----------
    for c in FIXED_C_SWEEP:
        for disc, label in (("fifo_fixed", "fifo"), ("sla_aware", "sla_aware")):
            t0 = time.time()
            kpi = evaluate_c_schedule(
                raw, [c] * n_ticks, TICK_S, warp, sla_s,
                policy=f"{label}_fixed_c{c}", uses_future_info=False, deployable=True,
                classification="Deployable (fixed, no MCS)", discipline=disc,
                runtime_s=time.time() - t0,
            )
            results.append(kpi.to_dict())

    # --- Variable-capacity MCS (FIFO discipline) ---------------------------
    t0 = time.time()
    c_lag1 = reactive_lag1_c_schedule(raw, TICK_S, warp, mcs_gate=MCS_GATE, sla_s=sla_s)
    results.append(evaluate_c_schedule(
        raw, c_lag1, TICK_S, warp, sla_s,
        policy="reactive_lag1_mcs", uses_future_info=False, deployable=True,
        classification="Deployable (forecast MCS)", discipline="fifo",
        runtime_s=time.time() - t0,
    ).to_dict())

    t0 = time.time()
    c_ewma, diag_ewma = forecast_mcs_c_schedule(
        raw, TICK_S, warp, method="ewma", mcs_gate=MCS_GATE, sla_s=sla_s,
        ewma_alpha=0.5, warmup_c=FIXED_C, warmup_ticks=1,
    )
    results.append(evaluate_c_schedule(
        raw, c_ewma, TICK_S, warp, sla_s,
        policy="forecast_mcs_ewma", uses_future_info=False, deployable=True,
        classification="Deployable (forecast MCS)", discipline="fifo",
        forecast=diag_ewma.to_dict(), runtime_s=time.time() - t0,
    ).to_dict())

    t0 = time.time()
    c_q, diag_q = forecast_mcs_c_schedule(
        raw, TICK_S, warp, method="quantile", mcs_gate=MCS_GATE, sla_s=sla_s,
        count_window=8, quantile=0.90, safety_k=1.0, warmup_c=FIXED_C, warmup_ticks=1,
    )
    results.append(evaluate_c_schedule(
        raw, c_q, TICK_S, warp, sla_s,
        policy="forecast_mcs_quantile_p90", uses_future_info=False, deployable=True,
        classification="Deployable (forecast MCS)", discipline="fifo",
        forecast=diag_q.to_dict(), runtime_s=time.time() - t0,
    ).to_dict())

    t0 = time.time()
    c_oracle = list(_joint_mcs_c_schedule(raw, TICK_S, warp, mcs_gate=MCS_GATE, sla_s=sla_s))
    results.append(evaluate_c_schedule(
        raw, c_oracle, TICK_S, warp, sla_s,
        policy="oracle_mcs", uses_future_info=True, deployable=False,
        classification="Oracle upper bound", discipline="fifo",
        runtime_s=time.time() - t0,
    ).to_dict())

    # --- Compound: variable capacity + abs-conformal SRTF ------------------
    live_preds, prior_stats = make_live_prior_predictions(raw, window=200)
    t0 = time.time()
    d = evaluate_c_schedule(
        raw, c_ewma, TICK_S, warp, sla_s,
        policy="forecast_mcs_ewma+abs_conformal", uses_future_info=False, deployable=True,
        classification="Deployable (forecast MCS + SRTF)", discipline="abs_conformal",
        predicted_tokens=live_preds, forecast=diag_ewma.to_dict(),
        runtime_s=time.time() - t0,
    ).to_dict()
    d["live_prior_stats"] = prior_stats
    results.append(d)

    t0 = time.time()
    results.append(evaluate_c_schedule(
        raw, c_oracle, TICK_S, warp, sla_s,
        policy="oracle_mcs+abs_conformal", uses_future_info=True, deployable=False,
        classification="Oracle upper bound (+SRTF)", discipline="abs_conformal",
        predicted_tokens=live_preds, runtime_s=time.time() - t0,
    ).to_dict())

    # --- baseline selection + deltas ---------------------------------------
    by_policy = {r["policy"]: r for r in results}
    # Strongest deployable SLA-aware FIXED (no-MCS) baseline = best goodput/$
    # among all swept fixed-c sla_aware policies.
    fixed_sla = [r for r in results if r["policy"].startswith("sla_aware_fixed_c")]
    strongest_fixed = max(fixed_sla, key=lambda r: r["goodput_per_dollar"])
    # Documented "SLA-oracle" baseline (sla_aware at c=4).
    documented = by_policy.get("sla_aware_fixed_c4", strongest_fixed)
    # Strongest deployable MCS-enabled baseline (best non-oracle MCS variant
    # other than the candidate itself) — used for the strict north-star gate.
    mcs_deployable = [by_policy[p] for p in (
        "reactive_lag1_mcs", "forecast_mcs_ewma", "forecast_mcs_quantile_p90",
        "forecast_mcs_ewma+abs_conformal") if p in by_policy]
    strongest_mcs_deployable = max(mcs_deployable, key=lambda r: r["goodput_per_dollar"])
    # "Forecasted MCS baseline" = the simplest causal forecast (naive lag-1).
    fc_baseline = by_policy["reactive_lag1_mcs"]

    for r in results:
        r["delta_vs_strongest_fixed_sla_pct"] = round(
            _delta_pct(r["goodput_per_dollar"], strongest_fixed["goodput_per_dollar"]), 2)
        r["delta_vs_documented_baseline_pct"] = round(
            _delta_pct(r["goodput_per_dollar"], documented["goodput_per_dollar"]), 2)
        r["delta_vs_forecast_baseline_pct"] = round(
            _delta_pct(r["goodput_per_dollar"], fc_baseline["goodput_per_dollar"]), 2)

    return {
        "trace": name,
        "n_requests": len(raw),
        "n_ticks": n_ticks,
        "warp": round(warp, 4),
        "sla_s": sla_s,
        "tick_seconds": TICK_S,
        "mcs_gate": MCS_GATE,
        "strongest_fixed_sla_baseline": strongest_fixed["policy"],
        "strongest_fixed_sla_goodput_per_dollar": strongest_fixed["goodput_per_dollar"],
        "documented_baseline_policy": documented["policy"],
        "documented_baseline_goodput_per_dollar": documented["goodput_per_dollar"],
        "strongest_deployable_mcs_baseline": strongest_mcs_deployable["policy"],
        "strongest_deployable_mcs_goodput_per_dollar": strongest_mcs_deployable["goodput_per_dollar"],
        "policies": results,
    }


def main(argv=None) -> int:
    today = date.today().isoformat()
    prefix = f"research/results/forecasted_mcs_backtest_{today}"

    azure_raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=JOB_LIMIT)
    print(f"[fmcs] Azure {len(azure_raw):,} reqs ({DEFAULT_AZURE_FIXTURE})")
    traces = [("azure_llm_2024", azure_raw, DEFAULT_SLA_S)]

    if os.path.exists(DEFAULT_BURSTGPT_HF_JSONL):
        bgpt_raw = load_burstgpt_serving_requests_jsonl(DEFAULT_BURSTGPT_HF_JSONL, limit=JOB_LIMIT)
        print(f"[fmcs] BurstGPT HF {len(bgpt_raw):,} reqs ({DEFAULT_BURSTGPT_HF_JSONL})")
        traces.append(("burstgpt_hf", bgpt_raw, DEFAULT_BURSTGPT_SLA_S))
    else:
        print("[fmcs] BurstGPT HF JSONL not found; Azure-only run")

    payload = {
        "benchmark": "forecasted_mcs_backtest",
        "generated": today,
        "directional_only_not_production_savings": True,
        "physics": {
            "service_time": "TTFT_BASE_S(0.150) + output_tokens*TPOT_S(0.020)",
            "gate": "Erlang-C M/M/c SLA-timeout < mcs_gate%",
            "cost_denominator": "provisioned GPU-hours over fixed trace window = sum(c)*tick_hr * $2.00/hr",
            "goodput": "SLA-safe output tokens (response <= sla_s)",
        },
        "config": {
            "job_limit": JOB_LIMIT, "fixed_c_sweep": list(FIXED_C_SWEEP),
            "target_rho": TARGET_RHO, "tick_seconds": TICK_S, "mcs_gate": MCS_GATE,
        },
        "traces": [],
    }
    for name, raw, sla_s in traces:
        print(f"[fmcs] running {name} (sla={sla_s}s) ...")
        payload["traces"].append(_run_trace(name, raw, sla_s))

    os.makedirs("research/results", exist_ok=True)
    with open(prefix + ".json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(prefix + ".md", payload)
    print(f"[fmcs] JSON -> {prefix}.json")
    print(f"[fmcs] MD   -> {prefix}.md")

    for tr in payload["traces"]:
        best_fc = max(
            (r for r in tr["policies"] if r["classification"].startswith("Deployable (forecast")),
            key=lambda r: r["goodput_per_dollar"])
        oracle = next(r for r in tr["policies"] if r["policy"] == "oracle_mcs")
        print(f"[fmcs] {tr['trace']}: best deployable forecast MCS = {best_fc['policy']} "
              f"{best_fc['goodput_per_dollar']:,.0f} gp/$ "
              f"(Δ vs strongest fixed SLA {best_fc['delta_vs_strongest_fixed_sla_pct']:+.1f}%; "
              f"Δ vs documented c=4 {best_fc['delta_vs_documented_baseline_pct']:+.1f}%); "
              f"oracle={oracle['goodput_per_dollar']:,.0f}; "
              f"north_star_+300%_vs_strongest_fixed="
              f"{'YES' if best_fc['delta_vs_strongest_fixed_sla_pct'] >= 300.0 else 'NO'}")
    return 0


def _fmt(v, nd=0):
    if isinstance(v, (int, float)):
        return f"{v:,.{nd}f}"
    return str(v)


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Forecasted MCS Backtest — deployable vs oracle capacity provisioning")
    A("")
    A("> **Directional simulator evidence only — NOT production savings** "
      "(`docs/RESULTS.md` §8).")
    A("")
    A(f"- Generated: {payload['generated']}")
    A(f"- Physics: {payload['physics']['service_time']}; gate = {payload['physics']['gate']}")
    A(f"- Cost denominator: {payload['physics']['cost_denominator']}")
    A(f"- Goodput: {payload['physics']['goodput']}")
    A(f"- Config: {json.dumps(payload['config'])}")
    A("")
    A("All policies share one physics model, one cost denominator (provisioned "
      "GPU-hours over the **fixed** trace window — a backed-up queue does not "
      "extend the billing window), one SLA definition, one trace, and one "
      "discrete-event simulator family. Only the per-tick capacity schedule and "
      "the queue discipline differ.")
    A("")

    for tr in payload["traces"]:
        A(f"## {tr['trace']} — {tr['n_requests']:,} req, {tr['n_ticks']} ticks, "
          f"SLA {tr['sla_s']}s, warp {tr['warp']}")
        A("")
        A(f"- Strongest deployable **fixed** SLA-aware baseline (no MCS): "
          f"**{tr['strongest_fixed_sla_baseline']}** = "
          f"{_fmt(tr['strongest_fixed_sla_goodput_per_dollar'])} goodput/$")
        A(f"- Documented leaderboard baseline (sla_aware @ c=4): "
          f"**{tr['documented_baseline_policy']}** = "
          f"{_fmt(tr['documented_baseline_goodput_per_dollar'])} goodput/$ "
          f"(under-provisioned — see SLA violations below)")
        A(f"- Strongest deployable **MCS** baseline: "
          f"**{tr['strongest_deployable_mcs_baseline']}** = "
          f"{_fmt(tr['strongest_deployable_mcs_goodput_per_dollar'])} goodput/$")
        A("")
        A("### KPI table (Phase 5 required format)")
        A("")
        A("`Δ vs SLA-aware fixed` is vs the strongest swept fixed-c SLA-aware "
          "baseline; `Δ vs forecasted-MCS baseline` is vs the naive lag-1 forecast.")
        A("")
        A("| Policy | Goodput/$ | Δ vs SLA-aware fixed | Δ vs forecasted-MCS baseline | "
          "GPU-hours | Cost $ | SLA violations | p99 queue |")
        A("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in tr["policies"]:
            A(f"| {r['policy']} | {_fmt(r['goodput_per_dollar'])} | "
              f"{r['delta_vs_strongest_fixed_sla_pct']:+.1f}% | "
              f"{r['delta_vs_forecast_baseline_pct']:+.1f}% | "
              f"{_fmt(r['gpu_hours'], 3)} | {_fmt(r['cost_usd'], 2)} | "
              f"{r['sla_violations']} | {_fmt(r['p99_wait_s'], 2)}s |")
        A("")
        A(f"_Documented leaderboard baseline reference: {tr['documented_baseline_policy']} "
          f"= {_fmt(tr['documented_baseline_goodput_per_dollar'])} goodput/$ "
          f"(sla_aware @ c=4, under-provisioned)._")
        A("")
        A("### Classification")
        A("")
        A("| Policy | Forecast type | Uses future info? | Deployable? | Classification |")
        A("|---|---|---|---|---|")
        ftype = {
            "reactive_lag1_mcs": "naive lag-1",
            "forecast_mcs_ewma": "EWMA point",
            "forecast_mcs_quantile_p90": "rolling p90 + 1σ",
            "oracle_mcs": "clairvoyant",
            "forecast_mcs_ewma+abs_conformal": "EWMA point",
            "oracle_mcs+abs_conformal": "clairvoyant",
        }
        for r in tr["policies"]:
            if r["policy"].startswith("fifo_fixed_c") or r["policy"].startswith("sla_aware_fixed_c"):
                ft = "none (static, swept)"
            else:
                ft = ftype.get(r["policy"], "—")
            A(f"| {r['policy']} | {ft} | "
              f"{'YES' if r['uses_future_info'] else 'no'} | "
              f"{'YES' if r['deployable'] else 'NO'} | {r['classification']} |")
        A("")
        A("### Forecast error (causal, vs realised ticks)")
        A("")
        A("| Policy | arrival MAE | arrival rel-MAE | arrival bias | service MAE (s) |")
        A("|---|---:|---:|---:|---:|")
        for r in tr["policies"]:
            f = r.get("forecast")
            if f:
                A(f"| {r['policy']} | {_fmt(f['arr_mae'], 2)} | {f['arr_rel_mae_pct']:.1f}% | "
                  f"{f['arr_bias']:+.2f} | {_fmt(f['svc_mae_s'], 3)} |")
        A("")
        # decomposition
        oracle = next(r for r in tr["policies"] if r["policy"] == "oracle_mcs")
        fc = max((r for r in tr["policies"]
                  if r["classification"].startswith("Deployable (forecast")
                  and "conformal" not in r["policy"]),
                 key=lambda r: r["goodput_per_dollar"])
        sfx = next(r for r in tr["policies"]
                   if r["policy"] == tr["strongest_fixed_sla_baseline"])
        A("### Decomposition")
        A("")
        A(f"**Best deployable forecast MCS ({fc['policy']}) vs strongest fixed SLA-aware "
          f"({sfx['policy']}):**")
        A(f"- Goodput/$: {_fmt(sfx['goodput_per_dollar'])} → {_fmt(fc['goodput_per_dollar'])} "
          f"({_delta_pct(fc['goodput_per_dollar'], sfx['goodput_per_dollar']):+.1f}%)")
        A(f"- GPU-hours: {sfx['gpu_hours']:.3f} → {fc['gpu_hours']:.3f} "
          f"({_delta_pct(fc['gpu_hours'], sfx['gpu_hours']):+.1f}% — "
          f"{'MORE capacity bought' if fc['gpu_hours'] > sfx['gpu_hours'] else 'less capacity'})")
        A(f"- SLA violations: {sfx['sla_violations']} → {fc['sla_violations']} "
          f"(Δ {fc['sla_violations'] - sfx['sla_violations']:+d})")
        A("")
        A(f"**Oracle MCS → best deployable forecast MCS ({fc['policy']}):**")
        A(f"- Goodput/$: {_fmt(oracle['goodput_per_dollar'])} → {_fmt(fc['goodput_per_dollar'])} "
          f"({_delta_pct(fc['goodput_per_dollar'], oracle['goodput_per_dollar']):+.1f}% — forecast retains "
          f"{100.0 * fc['goodput_per_dollar'] / max(1e-9, oracle['goodput_per_dollar']):.1f}% of oracle)")
        A(f"- GPU-hours: oracle {oracle['gpu_hours']:.3f} → forecast {fc['gpu_hours']:.3f} "
          f"({_delta_pct(fc['gpu_hours'], oracle['gpu_hours']):+.1f}%)")
        A(f"- SLA violations: oracle {oracle['sla_violations']} → forecast {fc['sla_violations']} "
          f"(Δ {fc['sla_violations'] - oracle['sla_violations']:+d})")
        A("")
        ns = fc["delta_vs_strongest_fixed_sla_pct"] >= 300.0
        A(f"**North-star (+300% vs strongest fixed SLA-aware): "
          f"{'ACHIEVED' if ns else 'NOT ACHIEVED'}** "
          f"(best deployable forecast MCS is {fc['delta_vs_strongest_fixed_sla_pct']:+.1f}%).")
        A("")

    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
