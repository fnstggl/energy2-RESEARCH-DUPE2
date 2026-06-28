#!/usr/bin/env python3
"""dt=60 serving-roofline closure diagnostic (PR #108, Phase 11).

Numerically PROVES the Azure+Mooncake bottleneck — is it decode-PHASE-bound? memory-BANDWIDTH-bound?
compute-bound? — on the real prompt/output distribution, then runs every mechanism's sensitivity sweep
(batching, prefill/decode allocation, speculative decoding, clock/DVFS, precision, co-location) through
the same roofline physics and reports the help/hurt/neutral region for each. Fully simulated; only
batching is a live MPC action, the rest are diagnostic sweeps (fully simulated, not controller-selected).

Bounded: samples the 6-hour Azure window; the roofline analysis is analytical (no MPC loop needed to
classify the bottleneck). Usage: python -m scripts.diagnose_serving_roofline_dt60
"""

from __future__ import annotations

import argparse
import json
import os

from aurelius.environment.roofline import (
    GPU_SPECS,
    ServingConfig,
    Workload,
    all_sensitivity_curves,
    roofline_regime,
    serving_point,
)
from aurelius.environment.training import build_mpc_inputs

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "external", "mpc_controller")


def _azure_tokens(eval_periods):
    inp = build_mpc_inputs(hourly_stride=96, use_world_state=True, control_dt_seconds=60.0)
    if inp is None:
        return None
    per, frames = inp["per"], inp["frames"]
    n = len(frames)
    ev = range(max(0, n - eval_periods), n)
    outs, ins = [], []
    for p in ev:
        for r in per.get(p, []):
            outs.append(int(r[1]))
            ins.append(int(r[2]) if len(r) > 2 else int(r[1]))
    gpu_mix = getattr(inp["fleet_state"], "gpu_type_mix", {}) or {"A100": 1.0}
    return outs, ins, gpu_mix


def _pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))] if s else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-periods", type=int, default=360)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    tk = _azure_tokens(args.eval_periods)
    if tk is None:
        raise SystemExit("no Azure serving data")
    outs, ins, gpu_mix = tk
    prompt_med, prompt_p95 = _pct(ins, 0.5), _pct(ins, 0.95)
    out_med, out_p95 = _pct(outs, 0.5), _pct(outs, 0.95)
    gpu = max(gpu_mix, key=gpu_mix.get) if gpu_mix else "A100"
    gpu = gpu if gpu in GPU_SPECS else "A100"

    # representative Azure workload (median prompt/output; context ≈ prompt+half the output)
    wl = Workload(prompt_tokens=max(1, prompt_med), decode_tokens=max(1, out_med),
                  context_len=max(1, prompt_med + out_med // 2))
    cfg = ServingConfig(gpu=gpu, batch_size=args.batch)
    pt = serving_point(wl, cfg)
    dr = roofline_regime("decode", cfg, wl)
    pr = roofline_regime("prefill", cfg, wl)

    bottleneck = {
        "gpu": gpu, "prompt_median": prompt_med, "prompt_p95": prompt_p95,
        "output_median": out_med, "output_p95": out_p95,
        "decode_gpu_sec_share": pt["decode_gpu_sec_share"], "phase_bottleneck": pt["phase_bottleneck"],
        "decode_arithmetic_intensity": dr["arithmetic_intensity"], "ridge_point": dr["ridge_point"],
        "decode_roofline_regime": dr["roofline_regime"], "prefill_roofline_regime": pr["roofline_regime"],
        "ttft_s": pt["ttft_s"], "completion_s": pt["completion_s"],
        "verdict": (f"Azure is {pt['phase_bottleneck']} AND decode is {dr['roofline_regime']} "
                    f"(AI {dr['arithmetic_intensity']} vs ridge {dr['ridge_point']})")}

    # every mechanism's sensitivity curve on the representative Azure workload
    curves = all_sensitivity_curves(wl, cfg)
    sens = {}
    for mech, c in curves.items():
        # the region where it helps completion (or saves cost/energy) on THIS workload
        comp = c["help_hurt_neutral"]["completion_s"]
        cost = c["help_hurt_neutral"]["cost_usd"]
        energy = c["help_hurt_neutral"]["energy_j"]
        helps = [c["settings"][i] for i, v in enumerate(comp) if v == "help"]
        cost_helps = [c["settings"][i] for i, v in enumerate(cost) if v == "help"]
        energy_helps = [c["settings"][i] for i, v in enumerate(energy) if v == "help"]
        sens[mech] = {"action_surface": c["action_surface"], "settings": c["settings"],
                      "completion_help_at": helps, "cost_help_at": cost_helps,
                      "energy_help_at": energy_helps,
                      "decode_regime_per_setting": [pt2["decode_regime"] for pt2 in c["curve"]]}

    out = {"eval_periods": args.eval_periods, "n_requests": len(outs), "bottleneck_proof": bottleneck,
           "sensitivity": sens,
           "claim_safety": {"bottleneck": "trace_derived_workload + public_spec_roofline",
                            "mechanism_curves": "simulator_inferred", "live_action": "batching only"}}
    os.makedirs(_OUT, exist_ok=True)
    with open(os.path.join(_OUT, "serving_roofline_dt60.json"), "w") as f:
        json.dump(out, f, indent=2)
    if args.json:
        print(json.dumps(out, indent=2))
        return
    print(f"Azure dt=60 ({len(outs)} reqs, gpu={gpu}): prompt med={prompt_med}/p95={prompt_p95} "
          f"output med={out_med}/p95={out_p95}")
    print("BOTTLENECK PROOF:", bottleneck["verdict"])
    print(f"  decode_gpu_sec_share={bottleneck['decode_gpu_sec_share']} "
          f"phase={bottleneck['phase_bottleneck']} | decode AI={dr['arithmetic_intensity']} "
          f"ridge={dr['ridge_point']} regime={dr['roofline_regime']}")
    print("MECHANISM SENSITIVITY (help region on this workload):")
    for mech, s in sens.items():
        print(f"  {mech:>26} [{s['action_surface']:>20}]: completion helps at {s['completion_help_at']} "
              f"| cost helps at {s['cost_help_at']}")


if __name__ == "__main__":
    main()
