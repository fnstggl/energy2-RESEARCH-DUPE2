#!/usr/bin/env python3
"""Control-interval × horizon ablation for the receding-horizon MPC.

Two axes, on the SAME one-week Azure trace:
- ``--dt-seconds``: the CONTROL INTERVAL. The week is re-binned at this resolution
  (``period_seconds = dt``); the sample stride is global so the arrival RATE is dt-invariant
  — a finer dt changes only how often the controller acts, not the load intensity. This is
  what tests whether sub-hour control unlocks the deferred-benefit actions (prewarm /
  migration / placement): a warmed pool survives across steps only when ``dt`` is shorter
  than the calibrated ~300s idle timeout (``research/SUBHOUR_MPC_ACTION_VALUE_TEST.md``).
- ``--horizons``: the planning horizon H (in SIM STEPS). The real lookahead is ``H × dt``.

For each (dt, H) it trains forecasters on the pre-eval week, runs the world-state MPC on the
held-out tail (committing only the first action each interval), and reports gp/$, SLA, queue
p95/p99, GPU-hours, the chosen-action mixes (capacity / batching / routing / prewarm /
placement / migration), per-decision runtime, world-steps simulated, and the Pareto claim
gate vs a fair baseline. ``dt=3600`` reproduces the hourly ablation; ``H=1`` is the
single-period controller.

Usage: python -m scripts.sweep_mpc_horizon --dt-seconds 3600,900,300,60 --horizons 1,2,4,8,12,24
"""

from __future__ import annotations

import argparse
import json
import os
import time

from aurelius.environment.controller import ModelPredictiveEconomicController, run_period_episode
from aurelius.environment.training import (
    DEFAULT_BASELINES,
    build_mpc_inputs,
    claim_gate,
    make_world_state,
    train_forecasters,
)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def _run_one_dt(dt, horizons, *, stride, sim_seconds, risk_weight, eval_periods, eval_span_hours,
                max_eval_periods=0):
    """Re-bin the week at control interval ``dt`` and run the horizon sweep on the held-out tail.

    ``eval_span_hours`` (when set) fixes the held-out window in REAL TIME, so every dt is scored
    on the SAME diurnal slice (same load, ramps included) and the only thing that varies is how
    often the controller acts — the apples-to-apples control-frequency comparison. The eval period
    COUNT is then ``span/dt`` (24 at hourly, 1440 at 60s). ``--max-eval-periods`` caps that count
    so the FINEST dt stays tractable: at the cap the window is the most-recent ``cap × dt`` seconds
    (still spanning a diurnal ramp), reported as ``eval_span_hours`` so the truncation is explicit.
    ``--eval-periods`` overrides with a fixed count (for quick smoke runs)."""
    inp = build_mpc_inputs(hourly_stride=stride, sim_seconds=sim_seconds, use_world_state=True,
                           control_dt_seconds=dt)
    if inp is None:
        return None
    common, fleet, cm, frames = inp["common"], inp["fleet_state"], inp["cost_model"], inp["frames"]
    n = len(frames)
    ev = eval_periods if eval_periods else max(8, round(eval_span_hours * 3600.0 / dt))
    if max_eval_periods and ev > max_eval_periods:
        ev = max_eval_periods                                  # tractability cap at the finest dt
    # held-out tail of `ev` decisions; forecasters fit on the whole pre-eval week.
    t_cut = max(8, n - ev)
    eval_idx = list(range(t_cut, n))
    dt_real = common["period_seconds"]
    fm, _ = train_forecasters(frames, t_cut)

    fair = run_period_episode(
        "fair", (lambda a: (lambda h: dict(a)))(DEFAULT_BASELINES["aurelius_canonical_kv_routing"]),
        inp["per"], frames, eval_idx, fleet_state=fleet, cost_model=cm,
        world_state=make_world_state(common.get("world_state_params")), **common)

    rows = []
    for H in horizons:
        ws = make_world_state(common.get("world_state_params"))
        ctrl = ModelPredictiveEconomicController(
            forecasters=fm, fleet_state=fleet, cost_model=cm, risk_weight=risk_weight,
            confidence_min=0.1, sla_s=common["sla_s"], period_seconds=common["period_seconds"],
            tick_seconds=common["tick_seconds"], kv_service_factor=common.get("kv_service_factor", 1.0),
            kv_service_factor_by_routing=common.get("kv_service_factor_by_routing"),
            cost_scenario=common.get("cost_scenario", "owned"), sim_seconds=common.get("sim_seconds"),
            world_state=ws, horizon_steps=H)
        t0 = time.monotonic()
        rep = run_period_episode("mpc", lambda h: ctrl.decide(h).to_dict(), inp["per"], frames,
                                 eval_idx, fleet_state=fleet, cost_model=cm, world_state=ws, **common)
        wall = time.monotonic() - t0
        gate = claim_gate({"mpc_controller": rep, "fair": fair})
        diag = ctrl.last_decision_diag
        rows.append({"horizon_steps": H, "lookahead_minutes": diag.get("lookahead_minutes"),
                     "lookahead_hours": diag.get("lookahead_hours"),
                     "goodput_per_dollar": round(rep.goodput_per_dollar, 1),
                     "sla_violation_rate": round(rep.sla_violation_rate, 4),
                     "queue_delay_p95": round(rep.queue_delay_p95, 3),
                     "queue_delay_p99": round(rep.queue_delay_p99, 3), "gpu_hours": round(rep.gpu_hours, 2),
                     "capacity_multiplier_mix": {str(k): v for k, v in rep.capacity_multiplier_mix.items()},
                     "batching_mix": rep.batching_mix, "routing_mix": rep.routing_mix,
                     "prewarm_mix": rep.prewarm_mix, "placement_mix": rep.placement_mix,
                     "migration_mix": rep.migration_mix, "wall_seconds_total": round(wall, 1),
                     "runtime_s_per_decision": round(wall / max(1, len(eval_idx)), 3),
                     "world_steps_last_decision": diag.get("world_steps_simulated"),
                     "beats_fair": gate["beats_fair_baseline"],
                     "pareto_sla_not_worse": gate["pareto_sla_not_worse"],
                     "headline_allowed": gate["headline_claim_allowed"]})
    return {"control_dt_seconds": dt_real, "n_periods_total": n, "eval_index_range": [t_cut, n],
            "eval_periods": len(eval_idx), "eval_span_hours": round(len(eval_idx) * dt_real / 3600.0, 3),
            "cycle_len": inp["coverage"].get("cycle_len"), "sim_seconds": common.get("sim_seconds"),
            "fair_gp_per_dollar": round(fair.goodput_per_dollar, 1),
            "fair_sla": round(fair.sla_violation_rate, 4),
            "fair_queue_p95": round(fair.queue_delay_p95, 3), "fair_gpu_hours": round(fair.gpu_hours, 2),
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-seconds", default="3600", help="comma list of control intervals (s)")
    ap.add_argument("--horizons", default="1,2,4,8,12,24")
    ap.add_argument("--eval-span-hours", type=float, default=24.0,
                    help="held-out window in REAL hours (same diurnal slice at every dt)")
    ap.add_argument("--eval-periods", type=int, default=0,
                    help="fixed eval period count, overrides --eval-span-hours (for smoke runs)")
    ap.add_argument("--max-eval-periods", type=int, default=360,
                    help="cap on eval decisions per dt (keeps the finest dt tractable)")
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--sim-seconds", type=float, default=180.0)
    ap.add_argument("--risk-weight", type=float, default=0.3)
    ap.add_argument("--out", default="mpc_subhour_action_value.json")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any checkpoint and recompute every dt from scratch")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    dts = [float(x) for x in args.dt_seconds.split(",")]
    horizons = [int(x) for x in args.horizons.split(",")]
    os.makedirs(_OUT, exist_ok=True)
    out_path = os.path.join(_OUT, args.out)

    # checkpoint/resume: each dt block is flushed as soon as it finishes, and an existing
    # checkpoint with the SAME config (stride / span / risk / horizons) is reused so an
    # interrupted sweep only recomputes the dt cells it never reached.
    cfg = {"stride": args.stride, "eval_span_hours": args.eval_span_hours,
           "eval_periods_override": args.eval_periods, "max_eval_periods": args.max_eval_periods,
           "risk_weight": args.risk_weight, "sim_seconds": args.sim_seconds, "horizons": horizons}
    out = {**cfg, "by_dt": {}}
    if not args.no_resume and os.path.exists(out_path):
        try:
            prev = json.load(open(out_path))
            if {k: prev.get(k) for k in cfg} == cfg:          # same config → safe to resume
                out["by_dt"] = prev.get("by_dt", {})
        except (ValueError, OSError):
            pass

    for dt in dts:
        key = str(int(dt))
        if key in out["by_dt"]:
            print(f"[resume] dt={key}s already in checkpoint — skipping", flush=True)
            continue
        print(f"[run] dt={key}s horizons={horizons} ...", flush=True)
        res = _run_one_dt(dt, horizons, stride=args.stride, sim_seconds=args.sim_seconds,
                          risk_weight=args.risk_weight, eval_periods=args.eval_periods,
                          eval_span_hours=args.eval_span_hours, max_eval_periods=args.max_eval_periods)
        if res is None:
            raise SystemExit("no Azure serving data available")
        out["by_dt"][key] = res
        with open(out_path, "w") as f:                        # flush checkpoint after each dt
            json.dump(out, f, indent=2)
        print(f"[done] dt={key}s written to {args.out}", flush=True)

    by_dt = out["by_dt"]
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    for dt in [str(int(x)) for x in dts]:
        res = by_dt.get(dt)
        if res is None:
            continue
        print(f"\n=== dt={dt}s  ({res['eval_periods']} eval periods = {res['eval_span_hours']}h, "
              f"cycle_len={res['cycle_len']})  fair gp/$={res['fair_gp_per_dollar']} "
              f"sla={res['fair_sla']} ===")
        print(f"{'H':>2} {'look(min)':>9} {'gp/$':>10} {'sla':>7} {'qp95':>6} {'qp99':>6} "
              f"{'gpu_h':>7} {'rt/dec':>7} {'prewarm':>20} {'migr':>14} | gate")
        for r in res["rows"]:
            pw = ",".join(f"{k}:{v}" for k, v in sorted(r["prewarm_mix"].items()))
            mg = ",".join(f"{k}:{v}" for k, v in sorted(r["migration_mix"].items()))
            print(f"{r['horizon_steps']:>2} {str(r['lookahead_minutes']):>9} "
                  f"{r['goodput_per_dollar']:>10.0f} {r['sla_violation_rate']:>7.4f} "
                  f"{r['queue_delay_p95']:>6.2f} {r['queue_delay_p99']:>6.2f} {r['gpu_hours']:>7.1f} "
                  f"{r['runtime_s_per_decision']:>7.3f} {pw:>20} {mg:>14} | "
                  f"{r['beats_fair']}/{r['pareto_sla_not_worse']}/{r['headline_allowed']}")


if __name__ == "__main__":
    main()
