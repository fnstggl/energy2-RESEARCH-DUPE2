#!/usr/bin/env python3
"""Azure LLM 2024 — Dynamic Safe Frontier Calibration + Shadow Evaluation.

Drives the dynamic frontier estimator through an offline streaming
replay of the Azure 2024 trace, pairs each prediction with the realized
outcome from the next decision window, updates per-workload confidence,
and aggregates calibration metrics across the run. Multi-pass: between
passes the harness applies bounded parameter updates and reports whether
oracle-alpha capture improved.

Hard rules (asserted by tests + docs):

- **No future leakage.** Each per-tick decision sees only the rolling
  telemetry window (t' ≤ t).
- **No production-savings claims.** Simulator / shadow-mode evidence
  only.
- **No new datasets.** Uses the same Azure 2024 trace that powers the
  committed dynamic frontier benchmark.
- **No safety-gate weakening.** Bounded parameter ranges only; if the
  false-safe rate creeps up, the harness tightens — never loosens.
- **No oracle leakage into the decision loop.** Oracle/baseline goodput
  per window is consumed only by the calibration record builder, never
  the estimator.
- **The committed Azure 2024 dynamic frontier JSON / MD is read-only.**
  This script writes to a *new* path
  (``azure_2024_dynamic_frontier_calibration_summary.json``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aurelius.frontier import (  # noqa: E402
    CalibrationReplayConfig,
    ConfidenceUpdateConfig,
    DynamicControllerConfig,
    DynamicEstimatorConfig,
    MultiPassCalibrationConfig,
    OracleSeriesPoint,
    RiskConfig,
    SafetyConfig,
    ServingTelemetryTick,
    WorkloadFrontierProfile,
    run_multi_pass_calibration,
    telemetry_tick_from_arrival_tick,
)
from aurelius.traces import azure_llm as az  # noqa: E402
from aurelius.traces import backtest as bt  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "azure_llm_2024", "processed",
    "azure_2024_dynamic_frontier_calibration_summary.json")
OUT_MD = os.path.join(
    REPO_ROOT, "docs",
    "AZURE_2024_DYNAMIC_FRONTIER_CALIBRATION_RESULTS.md")
RAW_DIR = os.path.join(REPO_ROOT, "data", "external", "azure_llm_2024", "raw")
FIXTURE = os.path.join(REPO_ROOT, "tests", "fixtures",
                       "azure_llm_2024_sample.csv")

TICK_S = 60.0
PRIMARY_SCALE_FULL = 10.0
PRIMARY_SCALE_FIXTURE = 100.0
CANDIDATE_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
STATIC_CA_RHO = 0.65
WINDOW_MINUTES = 60  # default; matches the dynamic frontier benchmark.


def _resolve_paths(raw_dir):
    return {v: os.path.join(raw_dir, f) for v, f in
            (("code", "AzureLLMInferenceTrace_code_1week.csv"),
             ("conv", "AzureLLMInferenceTrace_conv_1week.csv"))
            if os.path.exists(os.path.join(raw_dir, f))}


def _scale_ticks(ticks, f: float):
    if f == 1.0:
        return list(ticks)
    from dataclasses import replace
    return [replace(t, request_count=int(round(t.request_count * f)),
                    arrival_rate_rps=t.arrival_rate_rps * f,
                    total_prompt_tokens=int(round(t.total_prompt_tokens * f)),
                    total_output_tokens=int(round(t.total_output_tokens * f)),
                    model_mix={k: int(round(v * f))
                               for k, v in t.model_mix.items()})
            for t in ticks]


def _eval_tick(tick, target_rho: float, *, tick_hours: float) -> dict:
    """Apply a target rho to one arrival tick via the unchanged engine
    physics. Returns realized per-tick metrics."""
    r = bt._size_for_target(tick.arrival_rate_rps,
                            max(1.0, tick.output_tokens_mean),
                            bt._tick_throughput_tokps(tick), target_rho)
    ev = bt.evaluate_tick(tick, r, prefill_savings=0.0,
                          tick_hours=tick_hours)
    return {
        "tick_index": tick.tick_index, "start_s": tick.start_s,
        "rho_target": target_rho,
        "replicas": ev.replicas, "rho": ev.rho,
        "timeout_pct": ev.timeout_rate_pct,
        "queue_p95_ms": ev.queue_wait_p95_ms,
        "queue_p99_ms": ev.queue_wait_p99_ms,
        "latency_p99_ms": ev.latency_p99_ms,
        "gpu_hours": sum(ev.gpu_hours_by_type.values()),
        "energy_cost": ev.energy_cost,
        "tokens_offered": ev.tokens_offered,
        # Per-tick goodput/$ proxy (token-level, same shape as KPI numerator).
        "goodput_per_dollar": _tick_goodput_per_dollar(ev),
    }


def _tick_goodput_per_dollar(ev) -> Optional[float]:
    """Per-tick goodput/$ — SLA-compliant tokens divided by per-tick infra
    cost. Mirrors ``InfrastructureCostConfig`` defaults; safe-floor
    matches the committed Azure 2024 audit (50% timeout cliff)."""
    timeout_pct = ev.timeout_rate_pct or 0.0
    if timeout_pct >= 50.0:
        sla_tokens = 0
    else:
        sla_tokens = int(ev.tokens_offered
                          * max(0.0, 1.0 - timeout_pct / 100.0))
    # Pre-registered cost basis: $2.50/GPU-hour ≈ engine default.
    gpu_hours = sum(ev.gpu_hours_by_type.values())
    infra_cost = gpu_hours * 2.50 + (ev.energy_cost or 0.0)
    if infra_cost <= 0:
        return None
    return sla_tokens / infra_cost


def _telemetry_tick(arrival_tick, eval_result) -> ServingTelemetryTick:
    return telemetry_tick_from_arrival_tick(
        arrival_tick,
        active_replicas=eval_result["replicas"],
        queue_p95_ms=eval_result["queue_p95_ms"],
        queue_p99_ms=eval_result["queue_p99_ms"],
        latency_p99_ms=eval_result["latency_p99_ms"],
        timeout_pct=eval_result["timeout_pct"],
        sla_violation_pct=(eval_result["timeout_pct"]
                           if eval_result["timeout_pct"] > 0 else 0.0),
        mean_utilization=eval_result["rho"],
        telemetry_confidence="medium",
        source="azure_2024_streaming_replay_calibration")


def _build_oracle_series(ticks, *, tick_hours: float,
                          candidate_rhos=CANDIDATE_RHOS
                          ) -> list[OracleSeriesPoint]:
    """Per-tick analysis-only oracle.

    For each tick: evaluate every candidate rho and pick the highest
    safe one (timeout <= safe, queue p99 <= safe). Baseline is the
    static constraint_aware rho (0.65). Both are realized post-hoc.
    """
    out: list[OracleSeriesPoint] = []
    for t in ticks:
        best_rho = None
        best_gpd = float("-inf")
        baseline_gpd = None
        for rho in candidate_rhos:
            ev = _eval_tick(t, rho, tick_hours=tick_hours)
            safe = (ev["timeout_pct"] <= SAFE_TIMEOUT_PCT
                    and ev["queue_p99_ms"] <= SAFE_QUEUE_P99_MS)
            gpd = ev["goodput_per_dollar"]
            if abs(rho - STATIC_CA_RHO) < 1e-9:
                baseline_gpd = gpd
            if safe and gpd is not None and gpd > best_gpd:
                best_gpd = gpd
                best_rho = rho
        if best_gpd == float("-inf"):
            best_gpd = None
        out.append(OracleSeriesPoint(
            timestamp_s=float(t.start_s),
            workload_id="azure_llm_2024_week",
            best_safe_rho=best_rho,
            oracle_goodput_per_dollar=best_gpd,
            baseline_goodput_per_dollar=baseline_gpd,
        ))
    return out


def _f(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.{nd}f}" if abs(v) >= 1 else f"{v:.{nd + 2}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _write_md(path: str, payload: dict) -> None:
    L: list[str] = []
    A = L.append
    A("# Azure LLM 2024 — Dynamic Frontier Calibration + Shadow Evaluation\n")
    A("> **Simulator / shadow-mode benchmark. Directional only — NOT a "
      "production-savings claim** (`docs/RESULTS.md` §8). Streaming replay of "
      "the Azure 2024 trace; each per-tick decision sees only the "
      "telemetry from t' ≤ t (no future leakage). The robust energy "
      "engine is **unchanged**; the static frontier controller and "
      "committed Azure 2024 artifacts (including the existing dynamic "
      "frontier JSON / MD) are **read-only**. Real-cluster execution is "
      "disabled by default. The oracle row is **analysis-only**.\n")
    A("- **Read first:** `docs/RESULTS.md`, "
      "`docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`, "
      "`docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`, "
      "`docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`, "
      "`docs/PILOT_TELEMETRY_CONTRACT.md`.\n")

    cfg = payload["config"]
    A("## 1. Configuration\n")
    A(f"- **Tick seconds:** {cfg['tick_seconds']}")
    A(f"- **Primary scale:** {cfg['primary_scale']}×")
    A(f"- **Candidate rho grid:** `{cfg['candidate_rhos']}`")
    A(f"- **Window (minutes):** {cfg['window_minutes']}")
    A(f"- **Safety thresholds (pre-registered):** timeout ≤ "
      f"{cfg['safety_timeout_pct']}% AND queue p99 ≤ "
      f"{cfg['safety_queue_p99_ms']} ms")
    A(f"- **Passes:** {cfg['passes']}")
    A(f"- **Target oracle-alpha capture:** "
      f"{cfg['target_oracle_alpha_capture']:.2f} "
      "(aspiration, not a forced pass condition)")
    A(f"- **Max false-safe rate (safety floor):** "
      f"{cfg['max_false_safe_rate']:.4f}")
    A("- **No future leakage:** each decision sees t' ≤ t only.\n")

    src = payload["source"]
    A("## 2. Source\n")
    A(f"- **Trace:** `{src['path']}` ({src['n_ticks']:,} ticks @ "
      f"{src['tick_seconds']:.0f}s; "
      f"{src['time_span_seconds']:,.0f} s total)\n")

    A("## 3. Pass-by-pass results\n")
    A("| pass | n records | capture (overall) | capture (mean per-window) | "
      "safety correct % | false safe % | false unsafe % | "
      "conservative miss % | MAE timeout (pp) | MAE queue p99 (ms) | "
      "avg conf before | avg conf after | "
      "unsafe risk thr | deadband |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in payload["passes"]:
        s = p["summary"]
        est_snap = p.get("estimator_config_snapshot", {})
        ctrl_snap = p.get("controller_config_snapshot", {})
        cap_o = s.get("oracle_alpha_capture_pct_overall")
        cap_m = s.get("oracle_alpha_capture_pct_mean")
        sc_rate = s.get("safety_correct_rate")
        A(f"| {p['pass_index']} | {s.get('n_records', 0):,} | "
          f"{_f((cap_o * 100.0) if cap_o is not None else None, nd=2)}"
          f"{' %' if cap_o is not None else ''} | "
          f"{_f((cap_m * 100.0) if cap_m is not None else None, nd=2)}"
          f"{' %' if cap_m is not None else ''} | "
          f"{_f((sc_rate * 100.0) if sc_rate is not None else None, nd=2)}"
          f"{' %' if sc_rate is not None else ''} | "
          f"{_f((s.get('false_safe_rate', 0.0) or 0.0) * 100.0, nd=3)} % | "
          f"{_f((s.get('false_unsafe_rate', 0.0) or 0.0) * 100.0, nd=3)} % | "
          f"{_f((s.get('conservative_miss_rate', 0.0) or 0.0) * 100.0, nd=3)}"
          f" % | "
          f"{_f(s.get('mae_timeout_pct'), nd=3)} | "
          f"{_f(s.get('mae_queue_p99_ms'), nd=1)} | "
          f"{_f(s.get('average_confidence_before'), nd=3)} | "
          f"{_f(s.get('average_confidence_after'), nd=3)} | "
          f"{_f(est_snap.get('unsafe_risk_threshold'), nd=3)} | "
          f"{_f(ctrl_snap.get('deadband_rho'), nd=3)} |")
    A("")

    # Per-pass action / rho distribution.
    A("## 4. Recommendation distribution by pass\n")
    A("| pass | RAISE | KEEP | LOWER | INSUFFICIENT | avg rec rho | "
      "rho distribution |")
    A("|---|---|---|---|---|---|---|")
    for p in payload["passes"]:
        s = p["summary"]
        ad = s.get("action_distribution", {})
        rd = s.get("rho_distribution", {})
        rd_s = ", ".join(f"{k}:{v}" for k, v in rd.items())
        A(f"| {p['pass_index']} | {ad.get('RAISE_RHO', 0)} | "
          f"{ad.get('KEEP_RHO', 0)} | {ad.get('LOWER_RHO', 0)} | "
          f"{ad.get('INSUFFICIENT_TELEMETRY', 0)} | "
          f"{_f(s.get('average_recommended_rho'), nd=3)} | "
          f"{rd_s} |")
    A("")

    res = payload["multi_pass"]
    A("## 5. Multi-pass outcome\n")
    A(f"- **Initial oracle-alpha capture (pass 0):** "
      f"{_f((res['initial_oracle_alpha_capture'] * 100.0) if res['initial_oracle_alpha_capture'] is not None else None, nd=2)}"
      f"{' %' if res['initial_oracle_alpha_capture'] is not None else ''}")
    A(f"- **Final oracle-alpha capture:** "
      f"{_f((res['final_oracle_alpha_capture'] * 100.0) if res['final_oracle_alpha_capture'] is not None else None, nd=2)}"
      f"{' %' if res['final_oracle_alpha_capture'] is not None else ''}")
    A(f"- **Target (aspirational, not forced):** "
      f"{res['target_oracle_alpha_capture'] * 100.0:.0f} %")
    A(f"- **Target reached:** "
      f"{'YES' if res['reached_target'] else 'NO'}")
    A(f"- **Safety floor held:** "
      f"{'YES' if res['safety_floor_held'] else 'NO'}")
    A(f"- **Stopped reason:** `{res['stopped_reason']}`\n")
    if not res["reached_target"]:
        A("### Why the target was not reached\n")
        A(payload.get("why_not", "—"))
        A("")

    A("### Overfit / generalization caveats\n")
    for n in res.get("overfit_risk_notes", []):
        A(f"- {n}")
    A("")

    A("## 6. Honesty / scope\n")
    A("- The Dynamic Safe Frontier Calibration harness is **opt-in** and "
      "**disabled by default**. The static frontier controller remains "
      "the committed default; this run is a measurement.")
    A("- **No production mutation.** Decisions are recommendation-only "
      "(`executable_in_real_cluster=False`).")
    A("- **No future leakage.** Each per-tick decision sees only the "
      "telemetry from t' ≤ t in its rolling window. The oracle and "
      "baseline are computed **offline, post-hoc** and are visible to "
      "the calibration-record builder only — never to the dynamic "
      "estimator.")
    A("- **Bounded parameter updates.** Between passes the harness may "
      "tighten safety knobs OR relax conservatism within pre-registered "
      "bounded ranges. Safety vetoes, oracle labels, and the engine "
      "physics are NOT modified.")
    A("- **No ML training.** Confidence updates are deterministic and "
      "categorical (false-safe / false-unsafe / conservative-miss / "
      "accurate-safe / large-error penalty).")
    A("- The 95 % oracle-alpha target is **aspirational**. If we do "
      "not reach it without weakening safety or leaking the oracle into "
      "the decision loop, we report the gap honestly.")
    A("- This is **directional simulator / shadow-mode evidence** — NOT a "
      "production-savings claim. Pilot telemetry is required to calibrate "
      "the safe rho per workload before any production claim.")
    A("- The robust energy engine, the static frontier controller, the "
      "committed Azure 2024 audit / backtest / controller / integration "
      "/ full-trace / dynamic frontier JSON are **NOT modified** by this "
      "benchmark.\n")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def _why_not(res_dict: dict) -> str:
    """Generate an honest 'why we did not reach the target' note."""
    parts = []
    fp = res_dict.get("passes", [])
    if not fp:
        return "no passes completed"
    last = fp[-1]["summary"]
    cap = last.get("oracle_alpha_capture_pct_overall")
    fs = last.get("false_safe_rate") or 0.0
    cm = last.get("conservative_miss_rate") or 0.0
    fu = last.get("false_unsafe_rate") or 0.0
    if cap is None:
        parts.append(
            "Oracle-alpha capture is undefined for this trace "
            "(oracle ≤ baseline within the safety band, so there is no "
            "alpha to capture).")
    else:
        parts.append(
            f"Final oracle-alpha capture {cap * 100.0:.2f} % is below "
            f"the aspirational 95 % target. The dynamic estimator's "
            "Erlang-C tail calibration is workload-specific and noisy on "
            "short windows; closing the gap requires per-workload pilot "
            "data, not constant tuning.")
    if fs > 0:
        parts.append(
            f"False-safe rate {fs * 100.0:.4f} % is non-zero — we will "
            "not relax the unsafe-risk threshold to chase capture "
            "because doing so would push false-safe further up.")
    if cm > 0:
        parts.append(
            f"Conservative-miss rate {cm * 100.0:.4f} % suggests the "
            "controller stays a notch below the oracle's realized best "
            "safe rho. This is by design (conservative margin + "
            "deadband + hysteresis) and is the right trade-off for an "
            "estimator that does not yet have pilot calibration.")
    if fu > 0:
        parts.append(
            f"False-unsafe rate {fu * 100.0:.4f} % comes from the "
            "low-confidence telemetry windows where the estimator "
            "correctly falls back to LOWER_RHO rather than guess.")
    parts.append(
        "Closing the remaining gap is a pilot-telemetry calibration "
        "task, not a tuning task. See "
        "`docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md` §5.")
    return " ".join(parts)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--fixture", default=FIXTURE)
    p.add_argument("--out-json", default=OUT_JSON)
    p.add_argument("--out-md", default=OUT_MD)
    p.add_argument("--tick-seconds", type=float, default=TICK_S)
    p.add_argument("--primary-scale", type=float, default=None)
    p.add_argument("--window-minutes", type=int, default=WINDOW_MINUTES,
                   help="rolling telemetry window size in minutes")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--target-oracle-alpha-capture", type=float, default=0.95)
    p.add_argument("--min-safety-rate", type=float, default=0.99)
    p.add_argument("--max-false-safe-rate", type=float, default=0.01)
    p.add_argument("--max-ticks", type=int, default=None)
    args = p.parse_args(argv)

    paths = _resolve_paths(args.raw_dir)
    if paths:
        agg = az.stream_week_aggregate(paths, tick_seconds=args.tick_seconds)
        ticks = agg["arrival_ticks"]
        src_path = f"{args.raw_dir}/AzureLLMInferenceTrace_*_1week.csv"
        scale_default = PRIMARY_SCALE_FULL
    else:
        agg = az.stream_week_aggregate({"conv": args.fixture},
                                        tick_seconds=args.tick_seconds)
        ticks = agg["arrival_ticks"]
        src_path = args.fixture
        scale_default = PRIMARY_SCALE_FIXTURE

    primary_scale = (args.primary_scale if args.primary_scale is not None
                     else scale_default)
    if args.max_ticks:
        ticks = ticks[: args.max_ticks]
    ticks = _scale_ticks(ticks, primary_scale)
    tick_hours = args.tick_seconds / 3600.0

    profile = WorkloadFrontierProfile(
        workload_id="azure_llm_2024_week",
        workload_type="inference_standard",
        telemetry_confidence="medium",
        priority_class="standard",
        candidate_rhos=tuple(CANDIDATE_RHOS),
        source="azure_2024_streaming_replay_calibration")

    safety_cfg = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                               max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                               min_telemetry_confidence="low")
    risk_cfg = RiskConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                           max_queue_p99_ms=SAFE_QUEUE_P99_MS,
                           min_telemetry_confidence="low")

    multi_cfg = MultiPassCalibrationConfig(
        passes=args.passes,
        target_oracle_alpha_capture=args.target_oracle_alpha_capture,
        min_safety_rate=args.min_safety_rate,
        max_false_safe_rate=args.max_false_safe_rate)

    window_ticks = int(args.window_minutes * 60.0 / args.tick_seconds)
    replay_cfg = CalibrationReplayConfig(
        window_ticks=window_ticks, decision_interval_ticks=1,
        candidate_rhos=tuple(CANDIDATE_RHOS),
        bootstrap_rho=STATIC_CA_RHO,
        safety_timeout_pct=SAFE_TIMEOUT_PCT,
        safety_queue_p99_ms=SAFE_QUEUE_P99_MS,
        initial_confidence=0.5,
        source="azure_2024_streaming_replay_calibration")

    print(f"[calibration] building oracle series ({len(ticks)} ticks)…",
          flush=True)
    oracle = _build_oracle_series(ticks, tick_hours=tick_hours)
    print(f"[calibration] oracle ready", flush=True)

    def eval_fn(target_rho, idx):
        return _eval_tick(ticks[idx], target_rho, tick_hours=tick_hours)

    def telemetry_fn(arrival_tick, ev):
        return _telemetry_tick(arrival_tick, ev)

    confidence_cfg = ConfidenceUpdateConfig()
    result = run_multi_pass_calibration(
        workload_profile=profile, ticks=ticks,
        eval_fn=eval_fn, telemetry_fn=telemetry_fn,
        oracle_series=oracle,
        config=replay_cfg,
        multi_pass_config=multi_cfg,
        safety_cfg=safety_cfg, risk_cfg=risk_cfg,
        confidence_cfg=confidence_cfg)

    res_dict = result.to_dict()
    payload = {
        "config": {
            "tick_seconds": args.tick_seconds,
            "primary_scale": primary_scale,
            "candidate_rhos": list(CANDIDATE_RHOS),
            "safety_timeout_pct": SAFE_TIMEOUT_PCT,
            "safety_queue_p99_ms": SAFE_QUEUE_P99_MS,
            "window_minutes": args.window_minutes,
            "passes": args.passes,
            "target_oracle_alpha_capture": args.target_oracle_alpha_capture,
            "min_safety_rate": args.min_safety_rate,
            "max_false_safe_rate": args.max_false_safe_rate,
            "no_future_leakage": True,
            "execution_mode_default": "shadow",
            "real_execution_disabled_by_default": True,
        },
        "source": {
            "path": src_path,
            "n_ticks": len(ticks),
            "tick_seconds": args.tick_seconds,
            "time_span_seconds": (ticks[-1].start_s - ticks[0].start_s
                                   if ticks else 0.0),
            "scale": primary_scale,
        },
        "passes": res_dict["passes"],
        "multi_pass": {
            "stopped_reason": res_dict["stopped_reason"],
            "reached_target": res_dict["reached_target"],
            "safety_floor_held": res_dict["safety_floor_held"],
            "target_oracle_alpha_capture":
                res_dict["target_oracle_alpha_capture"],
            "initial_oracle_alpha_capture":
                res_dict["initial_oracle_alpha_capture"],
            "final_oracle_alpha_capture":
                res_dict["final_oracle_alpha_capture"],
            "overfit_risk_notes": res_dict["overfit_risk_notes"],
        },
        "why_not": _why_not(res_dict),
        # Required by the smoke test — both first and last pass are
        # surfaced for quick inspection.
        "summary_first_pass": res_dict["passes"][0]["summary"]
            if res_dict["passes"] else {},
        "summary_last_pass": res_dict["passes"][-1]["summary"]
            if res_dict["passes"] else {},
        "reached_target": res_dict["reached_target"],
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    _write_md(args.out_md, payload)

    print(f"[calibration] passes={len(res_dict['passes'])} "
          f"initial_capture={res_dict['initial_oracle_alpha_capture']} "
          f"final_capture={res_dict['final_oracle_alpha_capture']} "
          f"reached_target={res_dict['reached_target']} "
          f"stopped_reason={res_dict['stopped_reason']}", flush=True)
    print(f"[calibration] JSON -> {args.out_json}")
    print(f"[calibration] MD   -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
