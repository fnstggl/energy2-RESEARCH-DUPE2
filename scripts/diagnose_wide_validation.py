#!/usr/bin/env python3
"""Track A — WIDE validation of the current Aurelius MPC (no improvements; current main behaviour).

Validates the PR #114 headline (+82.1% SLA-safe gp/$ vs the strongest SLA-aware baseline) across MULTIPLE
windows and load regimes BEFORE any forecaster/clock change — so we only promote the headline if it holds.

Same harness/decider that produced the headline (full action layer, adaptive beam+local search, planning/eval
parity settings from PR #112) — replayed over regime-classified windows instead of just the trace tail:
  low_load · bursty · long_prompt · long_output · high_sla_pressure · mixed · tail · long (24-period).

Per arm (current MPC / sla_aware / greedy / fifo) and per window we report the wide metric set:
gp/$, GPU-hours, GPU-seconds, cost, energy(J), SLA-violation rate, TTFT p95, completion p95/p99, and the
selected precision / spec-decode / clock / batching / routing mixes. Pareto gate (`claim_gate`) per window.

Diagnostic only — NO controller / forecaster / simulator change. Usage:
    python -m scripts.diagnose_wide_validation --mpc-periods 6 --window-len 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

from aurelius.environment.controller import SLA_AWARE_FALLBACK, run_period_episode
from aurelius.environment.training import _controller as build_controller
from aurelius.environment.training import (
    build_mpc_inputs,
    claim_gate,
    make_world_state,
    train_forecasters,
)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def _mooncake_pool(limit):
    from aurelius.environment.ingestion.mooncake import ingest_mooncake
    reqs, _ = ingest_mooncake()
    pool = [tuple(r.hash_ids) for r in reqs if getattr(r, "hash_ids", None)]
    return pool[:limit] if limit else pool


def _period_stats(per, p):
    """(arrival_count, output_median, prompt_median, interarrival_cv) for one real period."""
    recs = sorted(per.get(p, []), key=lambda r: r[0])
    if not recs:
        return (0, 0.0, 0.0, 0.0)
    outs = sorted(int(r[1]) for r in recs)
    ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for r in recs)
    gaps = [recs[i + 1][0] - recs[i][0] for i in range(len(recs) - 1)]
    cv = 0.0
    if len(gaps) >= 2:
        m = statistics.mean(gaps)
        cv = statistics.pstdev(gaps) / m if m > 0 else 0.0
    return (len(recs), outs[len(outs) // 2], ins[len(ins) // 2], cv)


def _window_stats(per, win):
    rows = [_period_stats(per, p) for p in win]
    rows = [r for r in rows if r[0] > 0]
    if not rows:
        return None
    return {"arrival": statistics.mean(r[0] for r in rows), "out": statistics.mean(r[1] for r in rows),
            "prompt": statistics.mean(r[2] for r in rows), "cv": statistics.mean(r[3] for r in rows)}


def _classify_windows(per, n, win_len, mpc_periods):
    """Pick one representative window per load regime by scanning contiguous spans and ranking on stats.

    Returns {regime: window(list of period indices)}. Regimes with no usable span are omitted (documented).
    """
    starts = list(range(8, n - win_len + 1))
    cand = []
    for s in starts:
        w = list(range(s, s + win_len))
        st = _window_stats(per, w)
        if st:
            cand.append((w, st))
    if not cand:
        return {}
    regimes = {}

    def _pick(key, reverse):
        ranked = sorted(cand, key=lambda c: c[1][key], reverse=reverse)
        return ranked[0][0] if ranked else None

    regimes["low_load"] = _pick("arrival", False)
    regimes["bursty"] = _pick("cv", True)
    regimes["long_prompt"] = _pick("prompt", True)
    regimes["long_output"] = _pick("out", True)
    # high SLA pressure ~ high arrival × output (most decode work per unit time)
    pressure = sorted(cand, key=lambda c: c[1]["arrival"] * c[1]["out"], reverse=True)
    regimes["high_sla_pressure"] = pressure[0][0] if pressure else None
    # mixed = closest to the median window on every dimension (z-score sum minimised)
    med = {k: statistics.median(c[1][k] for c in cand) for k in ("arrival", "out", "prompt", "cv")}
    sd = {k: (statistics.pstdev([c[1][k] for c in cand]) or 1.0) for k in med}
    mixed = min(cand, key=lambda c: sum(abs(c[1][k] - med[k]) / sd[k] for k in med))
    regimes["mixed"] = mixed[0]
    regimes["tail"] = list(range(max(8, n - mpc_periods), n))
    # one longer window (≈24 periods) if tractable
    long_len = min(24, n - 8)
    if long_len >= win_len:
        regimes["long_24p"] = list(range(n - long_len, n))
    return {k: v for k, v in regimes.items() if v}


def _row(rep):
    """Extract the wide metric set from an EpisodeReport (only fields it actually exposes)."""
    return {
        "gp_per_dollar": round(rep.goodput_per_dollar, 1),
        "sla_violation_rate": round(rep.sla_violation_rate, 5),
        "gpu_hours": round(rep.gpu_hours, 3),
        "gpu_seconds": round(rep.realized_gpu_seconds, 1),
        "operator_cost": round(rep.total_operator_cost, 2),
        "energy_j": round(rep.total_energy_j, 1),
        "ttft_p95_s": round(rep.mean_ttft_p95, 4),
        "completion_p95_s": round(rep.queue_delay_p95, 4),
        "completion_p99_s": round(rep.queue_delay_p99, 4),
        "precision_mix": rep.precision_mix, "spec_decode_mix": rep.spec_decode_mix,
        "clock_mix": rep.clock_mix, "batching_mix": rep.batching_mix,
        "routing_mix": rep.routing_mix, "capacity_mix": {str(k): v for k, v in rep.capacity_multiplier_mix.items()},
        "decode_regime_mix": rep.decode_regime_mix,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpc-periods", type=int, default=6)      # periods scored by the MPC arm per window
    ap.add_argument("--window-len", type=int, default=8)       # frames per regime window
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--mooncake-limit", type=int, default=20000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    inp = build_mpc_inputs(hourly_stride=args.stride, sim_seconds=180.0, use_world_state=True,
                           control_dt_seconds=60.0)
    if inp is None:
        raise SystemExit("no Azure serving data available")
    common, fleet, cm, frames, per = (inp["common"], inp["fleet_state"], inp["cost_model"],
                                      inp["frames"], inp["per"])
    n = len(frames)
    pool = _mooncake_pool(args.mooncake_limit)
    if not pool:
        raise SystemExit("no Mooncake prefix pool available")
    kv = dict(kv_state_pool=pool, kv_capacity_blocks=256, kv_cost_mode="hybrid_capacity_work")
    cfg = {"horizon": 4, "risk_weight": 0.5, "confidence_min": 0.15}
    windows = _classify_windows(per, n, args.window_len, args.mpc_periods)
    if not windows:
        raise SystemExit("no usable windows")

    def _med_prompt(win):
        ins = sorted(int(r[2]) if len(r) > 2 else int(r[1]) for p in win for r in per.get(p, []))
        return ins[len(ins) // 2] if ins else 512

    def _ep(name, decide_fn, win, **kw):
        ws = make_world_state(common.get("world_state_params"))
        return run_period_episode(name, decide_fn, per, frames, win, fleet_state=fleet,
                                  cost_model=cm, world_state=ws, **common, **kv, **kw)

    def _mpc_decider(win):
        ws = make_world_state(common.get("world_state_params"))
        ctrl = build_controller(train_fm, fleet, cm, cfg, common, world_state=ws)
        ctrl.horizon_steps = 1
        ctrl.planning_kv_cost_mode = "hybrid_capacity_work"
        ctrl.planning_prompt_tokens = _med_prompt(win)
        cur = {"i": 0}

        def _dec(h):
            cur["i"] += 1
            return ctrl.decide(h).to_dict()
        return _dec

    def fifo(h):
        return {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off"}

    def greedy(h):
        return {"capacity": "backlog_aware", "ordering": "fifo", "admission": "off",
                "routing_policy": "kv_aware", "batching_policy": "aggressive"}

    def sla_aware(h):
        return dict(SLA_AWARE_FALLBACK)

    # train forecasters on history strictly before the earliest evaluated period (no leakage)
    earliest = min(min(w) for w in windows.values())
    train_fm, _ = train_forecasters(frames, max(8, earliest - 8))

    results = {}
    deltas = []
    for regime, win in windows.items():
        arms = {}
        for nm, fn in [("fifo", fifo), ("greedy", greedy), ("sla_aware", sla_aware)]:
            arms[nm] = _ep(nm, fn, win)
        arms["mpc_controller"] = _ep("current_mpc", _mpc_decider(win), win)
        gate = claim_gate(arms)
        fair = max(("fifo", "greedy", "sla_aware"), key=lambda k: arms[k].goodput_per_dollar)
        base_gpd = arms[fair].goodput_per_dollar
        mpc_gpd = arms["mpc_controller"].goodput_per_dollar
        delta = round(100.0 * (mpc_gpd - base_gpd) / base_gpd, 2) if base_gpd else None
        if delta is not None:
            deltas.append(delta)
        results[regime] = {
            "periods": [int(min(win)), int(max(win))], "n_periods": len(win),
            "window_stats": _window_stats(per, win),
            "fair_baseline": fair, "delta_pct": delta,
            "pareto_claim_allowed": bool(gate.get("headline_claim_allowed")),
            "pareto_sla_not_worse": bool(gate.get("pareto_sla_not_worse")),
            "arms": {nm: _row(rep) for nm, rep in arms.items()},
        }
        print(f"  [{regime:>17}] periods {min(win)}-{max(win)}  MPC {mpc_gpd:.0f} vs {fair} "
              f"{base_gpd:.0f}  Δ={delta}%  pareto={results[regime]['pareto_claim_allowed']}")

    summary = {"n_windows": len(results), "deltas_pct": deltas,
               "delta_min": min(deltas) if deltas else None,
               "delta_median": round(statistics.median(deltas), 2) if deltas else None,
               "delta_max": max(deltas) if deltas else None,
               "all_pareto_safe": all(r["pareto_sla_not_worse"] for r in results.values()),
               "all_beat_baseline": all((r["delta_pct"] or 0) > 0 for r in results.values()),
               "headline_holds": (all((r["delta_pct"] or 0) > 0 for r in results.values())
                                  and all(r["pareto_sla_not_worse"] for r in results.values()))}
    out = {"stride": args.stride, "n_frames": n, "mpc_periods": args.mpc_periods,
           "window_len": args.window_len, "windows": results, "summary": summary}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "wide_validation.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"SUMMARY: {len(results)} windows | Δ min/median/max = {summary['delta_min']}/"
          f"{summary['delta_median']}/{summary['delta_max']}% | headline_holds={summary['headline_holds']}")
    if args.json:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
