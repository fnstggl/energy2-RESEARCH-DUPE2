#!/usr/bin/env python3
"""min_cost_safe policy — public backtest.

Validates the ``min_cost_safe`` policy (per-tick minimum-replica oracle with
9.5% timeout gate) against ``safe_high_utilization`` (rho=0.75 EWMA-anticipatory,
the current headline policy), ``constraint_aware`` (rho=0.65, Aurelius canonical),
and ``sla_aware`` (headline baseline) on the two primary public traces.

Theory: ``min_cost_safe`` finds the smallest replica count per tick where
per-tick timeout_rate_pct < 9.5%, applying cache prefill savings identically
to constraint_aware. Because each per-tick value is strictly below 9.5%,
aggregate timeout < 9.5% < 10% is guaranteed by construction — no separate
gate check required.

Tradeoff vs ``safe_high_utilization``: SHU anticipates load via EWMA (max of
current + smoothed peak), preventing under-provisioning during surge ramp-up.
MCS is purely reactive and may under-provision during sudden load spikes, but
never over-provisions during quiet periods. Expected regime: MCS matches or
beats SHU on workloads with gradual, predictable load; SHU is safer on bursty
workloads where EWMA anticipation matters.

Directional simulator / public-trace evidence only — NOT production savings
(``docs/RESULTS.md`` §8).

Writes:
  * research/results/min_cost_safe_backtest_<date>.json
  * research/results/min_cost_safe_backtest_<date>.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.traces import azure_llm, burstgpt  # noqa: E402
from aurelius.traces.backtest import run_backtest, _MCS_TIMEOUT_GATE, _SHU_TARGET_RHO  # noqa: E402
from aurelius.traces.schema import time_rescale  # noqa: E402

BURSTGPT_RAW = "data/external/burstgpt/raw/BurstGPT_1.csv"
BURSTGPT_FIXTURE = "tests/fixtures/burstgpt_sample.csv"
AZURE_FIXTURE = "tests/fixtures/azure_llm_2024_sample.csv"
BURSTGPT_HF_JSONL = "data/external/hf/lzzmm__BurstGPT/burstgpt_1_full/processed/normalized_sample.jsonl"

COMPARE_POLICIES = ("fifo", "sla_aware", "constraint_aware", "safe_high_utilization",
                    "min_cost_safe")
BURSTGPT_SCALES = (1, 300)
AZURE_SCALES = (1, 50, 500)
# HF JSONL scales: 100x → ~12 rps max, 500x → ~31 rps max — both within safe MCS region
BURSTGPT_HF_SCALES = (100, 500)


def _run_scale(requests, scale, *, tick_seconds=60.0):
    rs = requests if scale == 1 else time_rescale(requests, scale)
    t0 = time.time()
    res = run_backtest(rs, tick_seconds=tick_seconds, policies=COMPARE_POLICIES)
    elapsed = time.time() - t0
    out = {
        "scale": scale,
        "n_ticks": res.n_ticks,
        "n_requests": len(rs),
        "runtime_s": round(elapsed, 2),
        "policies": {p: r.summary() for p, r in res.policy_results.items()},
        "ca_vs_sla_pct": round(res.outcome.margin_pct, 3),
    }
    mcs = res.policy_results.get("min_cost_safe")
    shu = res.policy_results.get("safe_high_utilization")
    ca = res.policy_results.get("constraint_aware")
    sla = res.policy_results.get("sla_aware")
    if mcs and ca and sla and shu:
        def gpd(r):
            return r.kpi.sla_safe_goodput_per_infra_dollar or 0.0
        out["mcs_vs_ca_pct"] = round((gpd(mcs) - gpd(ca)) / gpd(ca) * 100.0, 3) if gpd(ca) else None
        out["mcs_vs_sla_pct"] = round((gpd(mcs) - gpd(sla)) / gpd(sla) * 100.0, 3) if gpd(sla) else None
        out["mcs_vs_shu_pct"] = round((gpd(mcs) - gpd(shu)) / gpd(shu) * 100.0, 3) if gpd(shu) else None
        out["shu_vs_ca_pct"] = round((gpd(shu) - gpd(ca)) / gpd(ca) * 100.0, 3) if gpd(ca) else None
        out["mcs_timeout_pct"] = round(mcs.timeout_rate_pct_mean, 4)
        out["shu_timeout_pct"] = round(shu.timeout_rate_pct_mean, 4)
        out["ca_timeout_pct"] = round(ca.timeout_rate_pct_mean, 4)
        out["mcs_gpu_hours"] = round(mcs.kpi.active_gpu_hours, 2)
        out["shu_gpu_hours"] = round(shu.kpi.active_gpu_hours, 2)
        out["ca_gpu_hours"] = round(ca.kpi.active_gpu_hours, 2)
    return out


def _classify(mcs_vs_ca_pct, mcs_timeout_pct):
    if mcs_timeout_pct > 10.0:
        return "UNSAFE"
    if mcs_vs_ca_pct is None:
        return "UNKNOWN"
    if mcs_vs_ca_pct > 1.0:
        return "ALPHA_WIN"
    if abs(mcs_vs_ca_pct) <= 1.0:
        return "TIE"
    return "LOSS"


def _classify_vs_shu(mcs_vs_shu_pct, mcs_timeout_pct):
    if mcs_timeout_pct > 10.0:
        return "UNSAFE"
    if mcs_vs_shu_pct is None:
        return "UNKNOWN"
    if mcs_vs_shu_pct > 1.0:
        return "ALPHA_WIN"
    if abs(mcs_vs_shu_pct) <= 1.0:
        return "TIE"
    return "LOSS"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--burstgpt-scales", default=",".join(str(s) for s in BURSTGPT_SCALES))
    p.add_argument("--azure-scales", default=",".join(str(s) for s in AZURE_SCALES))
    p.add_argument("--sample-size", type=int, default=100000)
    args = p.parse_args(argv)

    today = date.today().isoformat()
    prefix = f"research/results/min_cost_safe_backtest_{today}"

    bpath = BURSTGPT_RAW if os.path.exists(BURSTGPT_RAW) else BURSTGPT_FIXTURE
    bgpt = burstgpt.load_csv(bpath, sample_size=args.sample_size, seed=0)
    azure = azure_llm.load_csv(AZURE_FIXTURE)
    print(f"[mcs] BurstGPT {len(bgpt):,} reqs ({bpath}); Azure {len(azure):,} reqs")

    hf_reqs = None
    hf_path = None
    if os.path.exists(BURSTGPT_HF_JSONL):
        hf_reqs = burstgpt.load_hf_jsonl(BURSTGPT_HF_JSONL, limit=20000, seed=0)
        hf_path = BURSTGPT_HF_JSONL
        print(f"[mcs] BurstGPT HF JSONL {len(hf_reqs):,} reqs ({hf_path})")
    else:
        print(f"[mcs] BurstGPT HF JSONL not found; skipping HF section")

    bgpt_scales = [int(s) for s in args.burstgpt_scales.split(",")]
    azure_scales = [int(s) for s in args.azure_scales.split(",")]

    bgpt_rows = []
    for sc in bgpt_scales:
        print(f"[mcs] BurstGPT scale {sc}x ...")
        bgpt_rows.append(_run_scale(bgpt, sc))

    azure_rows = []
    for sc in azure_scales:
        print(f"[mcs] Azure scale {sc}x ...")
        azure_rows.append(_run_scale(azure, sc))

    hf_rows = []
    if hf_reqs is not None:
        for sc in BURSTGPT_HF_SCALES:
            print(f"[mcs] BurstGPT HF scale {sc}x ...")
            hf_rows.append(_run_scale(hf_reqs, sc))

    all_rows = bgpt_rows + azure_rows + hf_rows

    # Primary verdict: MCS vs SHU (primary comparison, stronger baseline)
    vs_shu_verdicts = [_classify_vs_shu(r.get("mcs_vs_shu_pct"), r.get("mcs_timeout_pct", 0.0))
                       for r in all_rows]
    # Secondary verdict: MCS vs CA
    vs_ca_verdicts = [_classify(r.get("mcs_vs_ca_pct"), r.get("mcs_timeout_pct", 0.0))
                      for r in all_rows]

    def overall(verdicts):
        if all(v == "ALPHA_WIN" for v in verdicts):
            return "ALPHA_WIN"
        if "UNSAFE" in verdicts:
            return "UNSAFE"
        if all(v == "TIE" for v in verdicts):
            return "TIE"
        return "MIXED_" + "_".join(sorted(set(verdicts)))

    payload = {
        "benchmark": "min_cost_safe_policy_backtest",
        "generated": today,
        "directional_only_not_production_savings": True,
        "mcs_timeout_gate": _MCS_TIMEOUT_GATE,
        "shu_target_rho": _SHU_TARGET_RHO,
        "policy_description": (
            "min_cost_safe: per-tick minimum replicas where timeout_rate_pct < 9.5% gate. "
            "Pure reactive (no EWMA). Cache prefill savings applied. "
            "Aggregate timeout < 9.5% < 10% guaranteed by per-tick construction."
        ),
        "policies_compared": list(COMPARE_POLICIES),
        "burstgpt": {
            "source": bpath,
            "n_requests": len(bgpt),
            "scales": bgpt_rows,
        },
        "azure_llm_2024": {
            "source": AZURE_FIXTURE,
            "n_requests": len(azure),
            "scales": azure_rows,
        },
        "verdicts_vs_shu": vs_shu_verdicts,
        "verdicts_vs_ca": vs_ca_verdicts,
        "overall_vs_shu": overall(vs_shu_verdicts),
        "overall_vs_ca": overall(vs_ca_verdicts),
    }
    if hf_reqs is not None:
        payload["burstgpt_hf_jsonl"] = {
            "source": hf_path,
            "n_requests": len(hf_reqs),
            "scales": hf_rows,
        }

    os.makedirs("research/results", exist_ok=True)
    with open(prefix + ".json", "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(prefix + ".md", payload)
    print(f"[mcs] JSON -> {prefix}.json")
    print(f"[mcs] MD   -> {prefix}.md")
    print(f"[mcs] overall vs SHU: {payload['overall_vs_shu']}")
    print(f"[mcs] overall vs CA:  {payload['overall_vs_ca']}")
    return 0


def _fmt(v, nd=2):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def _write_md(path, payload):
    L = []
    A = L.append
    A("# min_cost_safe Policy Backtest")
    A("")
    A("> **Directional simulator evidence only — NOT production savings** "
      "(`docs/RESULTS.md` §8).")
    A("")
    A(f"- Generated: {payload['generated']}")
    A(f"- MCS timeout gate: {payload['mcs_timeout_gate']}% per-tick "
      f"(aggregate guaranteed < 10%)")
    A(f"- SHU reference target rho: {payload['shu_target_rho']} (anticipatory EWMA)")
    A(f"- **Overall vs SHU: `{payload['overall_vs_shu']}`**")
    A(f"- **Overall vs CA: `{payload['overall_vs_ca']}`**")
    A("")
    A("## Results")
    A("")
    A("| dataset | scale | MCS gpd/$ | SHU gpd/$ | MCS vs SHU % | MCS vs CA % | "
      "MCS timeout % | MCS GPU-h | SHU GPU-h | verdict vs SHU |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    datasets = [("burstgpt", payload["burstgpt"]), ("azure_2024", payload["azure_llm_2024"])]
    if "burstgpt_hf_jsonl" in payload:
        datasets.append(("burstgpt_hf", payload["burstgpt_hf_jsonl"]))
    for ds_key, ds in datasets:
        for row in ds["scales"]:
            mcs = row.get("policies", {}).get("min_cost_safe", {})
            shu = row.get("policies", {}).get("safe_high_utilization", {})
            verdict = _classify_vs_shu(row.get("mcs_vs_shu_pct"), row.get("mcs_timeout_pct", 0.0))
            A(f"| {ds_key} | {row['scale']}× | "
              f"{_fmt(mcs.get('sla_safe_goodput_per_infra_dollar'))} | "
              f"{_fmt(shu.get('sla_safe_goodput_per_infra_dollar'))} | "
              f"{_fmt(row.get('mcs_vs_shu_pct'))}% | "
              f"{_fmt(row.get('mcs_vs_ca_pct'))}% | "
              f"{_fmt(row.get('mcs_timeout_pct'))}% | "
              f"{_fmt(row.get('mcs_gpu_hours'))} | "
              f"{_fmt(row.get('shu_gpu_hours'))} | "
              f"`{verdict}` |")
    A("")
    A("## Interpretation")
    A("")
    A("- `min_cost_safe` searches from MIN_REPLICAS upward for the smallest fleet "
      "where per-tick timeout_rate_pct < 9.5% gate (with cache prefill savings).")
    A("- Because each per-tick value is strictly below 9.5%, aggregate timeout < 9.5% "
      "< 10% is guaranteed by construction — stronger than the 10% aggregate gate alone.")
    A("- No EWMA anticipation: purely reactive. Advantage over SHU during gradual-load "
      "ramp-down (never over-provisions). Disadvantage during sudden burst ramp-up.")
    A("- `safe_high_utilization` (SHU, rho=0.75, EWMA-anticipatory) is the primary "
      "comparison baseline and the current Aurelius headline policy.")
    A("- Simulator results only — not production savings.")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
