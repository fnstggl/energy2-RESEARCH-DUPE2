#!/usr/bin/env python3
"""Forecasted-MCS component ablation.

Measures each optional Aurelius component composed with the deployable
forecasted-MCS capacity policy, under ONE physics model / ONE SLA / ONE cost
denominator (provisioned GPU-hours over the fixed trace window). Keeps only
components that improve SLA-safe goodput/$ vs the forecasted-MCS baseline.

Conditions (per trace):
  C1  no MCS (best fixed c, swept)            deployable, no capacity planning
  C2  oracle MCS                              UPPER BOUND (tick-t actuals)
  ref OSOTSS (online_sotss)                   arrival-oracle (causal tokens only)
  C3  forecasted MCS (EWMA)                   deployable BASELINE
  C4  forecasted MCS + queue policy           + abs-conformal SRTF ordering
  C5  forecasted MCS + energy routing         + real CAISO/PJM/ERCOT energy term
  C6  forecasted MCS + placement              + real heterogeneous GPU menu

Cost levers (energy, placement) are also applied to C1 so the interaction (does
the component help MCS MORE than the fixed baseline?) is isolated.

Directional simulator evidence only — NOT production savings (docs/RESULTS.md §8).
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
)
from aurelius.benchmarks.mcs_ablation import (  # noqa: E402
    GPU_MENU,
    evaluate_with_energy,
    evaluate_with_placement,
    placement_schedule,
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
from aurelius.optimizer.policies.replica_scaling import compute_online_sotss_schedule  # noqa: E402

JOB_LIMIT = 5880
TARGET_RHO = 0.85
TICK_S = 60.0
MCS_GATE = 9.5
FIXED_C_SWEEP = (4, 5, 6, 7, 8, 10, 12)


def _d(a, b):
    return (a - b) / b * 100.0 if b else 0.0


def _run_trace(name, raw, sla_s):
    warp = calibrate_time_warp(raw, servers=4, target_rho=TARGET_RHO)
    _c, _t, n_ticks = bucketize(raw, TICK_S, warp)
    live_preds, _ = make_live_prior_predictions(raw, window=200)

    def base_eval(c_sched, policy, discipline="fifo", future=False, deployable=True,
                  predicted=None, klass="Deployable"):
        return evaluate_c_schedule(
            raw, c_sched, TICK_S, warp, sla_s, policy=policy,
            uses_future_info=future, deployable=deployable, classification=klass,
            discipline=discipline, predicted_tokens=predicted,
        ).to_dict()

    # C1 — no MCS: best fixed c (FIFO), swept.
    fixed = []
    for c in FIXED_C_SWEEP:
        fixed.append(base_eval([c] * n_ticks, f"fixed_c{c}", klass="Deployable (no MCS)"))
    no_mcs = dict(max(fixed, key=lambda r: r["goodput_per_dollar"]))
    no_mcs["policy"] = f"no_mcs_best_fixed({no_mcs['policy']})"

    # C2 — oracle MCS (upper bound).
    c_oracle = list(_joint_mcs_c_schedule(raw, TICK_S, warp, mcs_gate=MCS_GATE, sla_s=sla_s))
    oracle = base_eval(c_oracle, "oracle_mcs", future=True, deployable=False,
                       klass="Oracle upper bound")

    # ref — OSOTSS (arrival-oracle: causal tokens, actual arrival counts).
    c_osotss, *_ = compute_online_sotss_schedule(raw, TICK_S, warp, sla_s)
    osotss = base_eval(c_osotss, "osotss_arrival_oracle", future=True, deployable=False,
                       klass="Arrival-oracle (tokens causal)")

    # C3 — forecasted MCS (EWMA), the deployable BASELINE.
    c_fc, _diag = forecast_mcs_c_schedule(
        raw, TICK_S, warp, method="ewma", mcs_gate=MCS_GATE, sla_s=sla_s,
        ewma_alpha=0.5, warmup_c=4, warmup_ticks=1)
    forecast = base_eval(c_fc, "forecasted_mcs", klass="Deployable (forecast MCS) [BASELINE]")
    base_gpd = forecast["goodput_per_dollar"]

    # C4 — + queue policy (abs-conformal SRTF) on the SAME forecast capacity.
    queue = base_eval(c_fc, "forecasted_mcs+queue", discipline="abs_conformal",
                      predicted=live_preds, klass="Deployable (forecast MCS + SRTF)")

    # C5 — + energy routing: real energy term, flat region vs cheapest-region routed.
    energy_flat = evaluate_with_energy(raw, c_fc, TICK_S, warp, sla_s, route=False)
    energy_routed = evaluate_with_energy(raw, c_fc, TICK_S, warp, sla_s, route=True)
    # interaction: same routing on the no-MCS best fixed schedule.
    best_fixed_c = int(no_mcs["policy"].split("fixed_c")[1].rstrip(")"))
    nm_energy_flat = evaluate_with_energy(raw, [best_fixed_c] * n_ticks, TICK_S, warp, sla_s, route=False)
    nm_energy_routed = evaluate_with_energy(raw, [best_fixed_c] * n_ticks, TICK_S, warp, sla_s, route=True)

    # C6 — + placement: per-tick heterogeneous GPU menu (joint capacity+hardware).
    c_place, gpu_place = placement_schedule(raw, TICK_S, warp, sla_s, gate=MCS_GATE)
    placement = evaluate_with_placement(raw, c_place, gpu_place, TICK_S, warp, sla_s)
    # interaction: placement on the no-MCS fixed load (single best GPU per whole trace
    # is just the A10 reference at c=best; per-tick placement on a flat-demand baseline).
    c_place_nm, gpu_place_nm = placement_schedule(raw, TICK_S, warp, sla_s, gate=MCS_GATE)

    # ---- assemble + deltas vs forecasted-MCS baseline ----
    rows = [no_mcs, oracle, osotss, forecast, queue]
    for r in rows:
        r["delta_vs_forecast_mcs_pct"] = round(_d(r["goodput_per_dollar"], base_gpd), 2)

    energy_row = {
        "policy": "forecasted_mcs+energy_routing",
        "classification": "Deployable (forecast MCS + energy)",
        "goodput_per_dollar": energy_routed["goodput_per_dollar"],
        "gpu_hours": energy_routed["gpu_hours"],
        "cost_usd": energy_routed["cost_usd"],
        "sla_violations": energy_routed["sla_violations"],
        "p99_wait_s": energy_routed["p99_wait_s"],
        "energy_cost_usd": energy_routed["energy_cost_usd"],
        "energy_pct_of_cost": round(100.0 * energy_routed["energy_cost_usd"] / energy_routed["cost_usd"], 3),
        "goodput_per_dollar_energy_flat": energy_flat["goodput_per_dollar"],
        "routing_gain_pct": round(_d(energy_routed["goodput_per_dollar"], energy_flat["goodput_per_dollar"]), 3),
        "routing_gain_pct_no_mcs": round(_d(nm_energy_routed["goodput_per_dollar"], nm_energy_flat["goodput_per_dollar"]), 3),
        "delta_vs_forecast_mcs_pct": round(_d(energy_routed["goodput_per_dollar"], base_gpd), 2),
    }
    place_row = {
        "policy": "forecasted_mcs+placement",
        "classification": "Deployable (forecast MCS + placement)",
        "goodput_per_dollar": placement["goodput_per_dollar"],
        "gpu_hours": placement["gpu_hours"],
        "cost_usd": placement["cost_usd"],
        "sla_violations": placement["sla_violations"],
        "p99_wait_s": placement["p99_wait_s"],
        "gpu_mix": placement["gpu_mix"],
        "delta_vs_forecast_mcs_pct": round(_d(placement["goodput_per_dollar"], base_gpd), 2),
    }
    rows.extend([energy_row, place_row])

    # ---- component keep/drop verdicts (vs forecasted-MCS baseline) ----
    base_viol = forecast["sla_violations"]

    def verdict(gpd, viol, *, extra_ok=True):
        improves = gpd > base_gpd * 1.005          # >0.5% to beat noise
        sla_ok = viol <= base_viol * 1.10 + 5      # no material SLA regression
        return "KEEP" if (improves and sla_ok and extra_ok) else "DROP"

    components = [
        {"component": "queue policy (abs-conformal SRTF)",
         "delta_vs_forecast_mcs_pct": queue["delta_vs_forecast_mcs_pct"],
         "sla_violations": queue["sla_violations"],
         "verdict": verdict(queue["goodput_per_dollar"], queue["sla_violations"])},
        {"component": "energy routing (CAISO/PJM/ERCOT)",
         "delta_vs_forecast_mcs_pct": energy_row["delta_vs_forecast_mcs_pct"],
         "routing_gain_pct": energy_row["routing_gain_pct"],
         "routing_gain_pct_no_mcs": energy_row["routing_gain_pct_no_mcs"],
         "energy_pct_of_cost": energy_row["energy_pct_of_cost"],
         "sla_violations": energy_row["sla_violations"],
         # keep only if routing helps MCS more than baseline AND is material (>0.5%)
         "verdict": "KEEP" if (energy_row["routing_gain_pct"] > 0.5 and
                               energy_row["routing_gain_pct"] > energy_row["routing_gain_pct_no_mcs"] + 0.5)
                    else "DROP"},
        {"component": "placement (heterogeneous GPU menu)",
         "delta_vs_forecast_mcs_pct": place_row["delta_vs_forecast_mcs_pct"],
         "sla_violations": place_row["sla_violations"],
         "gpu_mix": place_row["gpu_mix"],
         "verdict": verdict(place_row["goodput_per_dollar"], place_row["sla_violations"])},
    ]

    return {
        "trace": name, "n_requests": len(raw), "n_ticks": n_ticks,
        "warp": round(warp, 4), "sla_s": sla_s, "mcs_gate": MCS_GATE,
        "forecast_mcs_baseline_goodput_per_dollar": round(base_gpd, 2),
        "conditions": rows,
        "components": components,
    }


def main(argv=None) -> int:
    today = date.today().isoformat()
    prefix = f"research/results/mcs_ablation_backtest_{today}"

    azure = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=JOB_LIMIT)
    print(f"[abl] Azure {len(azure):,} reqs")
    traces = [("azure_llm_2024", azure, DEFAULT_SLA_S)]
    if os.path.exists(DEFAULT_BURSTGPT_HF_JSONL):
        bgpt = load_burstgpt_serving_requests_jsonl(DEFAULT_BURSTGPT_HF_JSONL, limit=JOB_LIMIT)
        print(f"[abl] BurstGPT HF {len(bgpt):,} reqs")
        traces.append(("burstgpt_hf", bgpt, DEFAULT_BURSTGPT_SLA_S))

    payload = {
        "benchmark": "mcs_ablation_backtest", "generated": today,
        "directional_only_not_production_savings": True,
        "gpu_menu": [g.__dict__ for g in GPU_MENU],
        "traces": [],
    }
    for nm, raw, sla in traces:
        print(f"[abl] {nm} ...")
        t0 = time.time()
        payload["traces"].append(_run_trace(nm, raw, sla))
        print(f"[abl] {nm} done in {time.time()-t0:.1f}s")

    os.makedirs("research/results", exist_ok=True)
    with open(prefix + ".json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(prefix + ".md", payload)
    print(f"[abl] JSON -> {prefix}.json")
    print(f"[abl] MD   -> {prefix}.md")
    for tr in payload["traces"]:
        keep = [c["component"] for c in tr["components"] if c["verdict"] == "KEEP"]
        print(f"[abl] {tr['trace']}: KEEP {keep or 'none'} vs forecasted-MCS baseline "
              f"{tr['forecast_mcs_baseline_goodput_per_dollar']:,.0f} gp/$")
    return 0


def _fmt(v, nd=0):
    return f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v)


def _write_md(path, payload):
    L = []
    A = L.append
    A("# Forecasted-MCS Component Ablation")
    A("")
    A("> **Directional simulator evidence only — NOT production savings** (`docs/RESULTS.md` §8).")
    A("")
    A(f"- Generated: {payload['generated']}")
    A("- One physics model, one SLA, one cost denominator (provisioned GPU-hours "
      "over the fixed trace window). Only the listed component varies.")
    A("- **Keep rule:** a component is kept only if it improves SLA-safe goodput/$ "
      "vs the forecasted-MCS baseline by >0.5% **without** material SLA regression, "
      "and (for cost levers) helps MCS more than it helps the fixed baseline.")
    A("- GPU menu (real median on-demand $/gpu-hr × documented decode throughput): "
      + ", ".join(f"{g['name']} ${g['price_usd_hr']:.2f}/hr" for g in payload["gpu_menu"]))
    A("")
    for tr in payload["traces"]:
        A(f"## {tr['trace']} — {tr['n_requests']:,} req, {tr['n_ticks']} ticks, SLA {tr['sla_s']}s")
        A("")
        A(f"Forecasted-MCS baseline: **{_fmt(tr['forecast_mcs_baseline_goodput_per_dollar'])} goodput/$**")
        A("")
        A("| Condition | Goodput/$ | Δ vs forecasted MCS | GPU-h | Cost $ | SLA viol | p99 queue |")
        A("|---|---:|---:|---:|---:|---:|---:|")
        for r in tr["conditions"]:
            A(f"| {r['policy']} | {_fmt(r['goodput_per_dollar'])} | "
              f"{r.get('delta_vs_forecast_mcs_pct', 0):+.1f}% | {_fmt(r['gpu_hours'],3)} | "
              f"{_fmt(r['cost_usd'],2)} | {r['sla_violations']} | {_fmt(r['p99_wait_s'],2)}s |")
        A("")
        A("### Component verdicts (vs forecasted-MCS baseline)")
        A("")
        A("| Component | Δ goodput/$ | SLA viol | Notes | Verdict |")
        A("|---|---:|---:|---|---|")
        for c in tr["components"]:
            notes = ""
            if "routing_gain_pct" in c:
                notes = (f"energy={c['energy_pct_of_cost']}% of cost; routing gain "
                         f"{c['routing_gain_pct']:+.2f}% (MCS) vs {c['routing_gain_pct_no_mcs']:+.2f}% (fixed)")
            elif "gpu_mix" in c:
                notes = "GPU mix: " + ", ".join(f"{k}×{v}" for k, v in c["gpu_mix"].items())
            A(f"| {c['component']} | {c['delta_vs_forecast_mcs_pct']:+.1f}% | "
              f"{c['sla_violations']} | {notes} | **{c['verdict']}** |")
        A("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
