#!/usr/bin/env python3
"""Batch Inference Frontier v1 — Incremental Alpha Audit (audit-only).

Audits whether the Batch Inference Frontier (static v1 + dynamic v1)
produces genuinely incremental alpha beyond the existing Dynamic
Serving Frontier and constraint-aware scheduler on the SAME Azure
LLM 2024 ticks under the SAME canonical KPI + safety constraints.

Compared baselines (all run on the same arrival ticks, same per-tick
serving physics from ``aurelius/traces/backtest.py``):

  - ``sla_aware`` (fixed rho 0.50; reactive sizer)
  - ``current_price_only`` (fixed rho 0.65 = engine default,
    energy-cost-only — no SLA gate; used as the no-SLA-info baseline)
  - ``constraint_aware_static`` (fixed rho 0.65 = engine default)
  - ``static_serving_frontier_controller`` (fixed rho 0.75 = committed
    Azure 2024 frontier winner)
  - ``dynamic_serving_frontier`` (the committed dynamic estimator;
    streaming replay with rolling window)
  - ``static_batch_inference_frontier`` (best safe (rho, slack) point
    chosen offline)
  - ``dynamic_batch_inference_frontier`` (NEW — rolling-window dynamic
    estimator from ``dynamic_batch_inference_estimator.py``)

Alpha decomposition (per ``docs/RESULTS.md`` §6):

  - ``duplicated_serving_frontier_alpha_pct`` =
        (dynamic_serving − constraint_aware_static) / constraint_aware_static × 100
        (the existing rho controller's alpha; the batch frontier should
         NOT claim this as its own).
  - ``deadline_flex_scenario_alpha_pct`` =
        (static_batch − constraint_aware_static) / constraint_aware_static × 100
        (the batch-scenario relaxation alpha at a static knob).
  - ``true_incremental_alpha_vs_dynamic_serving_pct`` =
        (dynamic_batch − dynamic_serving) / dynamic_serving × 100
        (the only number that justifies scheduler integration).

Acceptance gate: ``true_incremental_alpha_vs_dynamic_serving_pct > 2.0``
under conservative assumptions AND ``constraint_aware_safe`` (no SLA /
queue-p99 regression vs constraint_aware_static). On accept, the script
emits a "PROPOSE_INTEGRATION" verdict. Otherwise: "SHADOW_DIAGNOSTIC".

This is **research / audit** code — NOT a constraint_aware integration.
The robust energy engine, the serving rho controller, the constraint_aware
default rho (0.65), and every committed serving / training / dynamic /
calibration artifact are **unchanged**. Real cluster execution is
disabled by default everywhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from aurelius.benchmarks.economics import (  # noqa: E402
    InfrastructureCostConfig, compute_economic_kpi,
)
from aurelius.frontier import (  # noqa: E402
    DynamicControllerConfig, DynamicEstimatorConfig, RiskConfig, SafetyConfig,
    WorkloadFrontierProfile, choose_dynamic_rho,
    estimate_dynamic_frontier, telemetry_tick_from_arrival_tick,
)
from aurelius.frontier.batch_inference_estimator import (  # noqa: E402
    BatchInferenceEstimatorConfig, estimate_batch_inference_frontier,
)
from aurelius.frontier.batch_inference_models import (  # noqa: E402
    BatchInferenceFrontierCandidate, BatchInferenceWorkloadProfile,
)
from aurelius.frontier.batch_inference_safety import (  # noqa: E402
    BatchInferenceSafetyConfig,
)
from aurelius.frontier.dynamic_batch_inference_estimator import (  # noqa: E402
    BatchArrivalTelemetryTick, DynamicBatchControllerConfig,
    DynamicBatchEstimatorConfig,
    choose_dynamic_batch_decision, estimate_dynamic_batch_frontier,
    telemetry_tick_from_arrival_tick as batch_telemetry_from_arrival,
)
from aurelius.traces import azure_llm  # noqa: E402
from aurelius.traces import backtest as bt  # noqa: E402
from aurelius.traces.replay import requests_to_arrival_ticks  # noqa: E402
from aurelius.traces.schema import time_rescale  # noqa: E402

OUT_JSON = os.path.join(
    REPO_ROOT, "data", "external", "frontier",
    "batch_inference_frontier_incremental_alpha_audit_summary.json")
AZURE_FIXTURE = os.path.join(
    REPO_ROOT, "tests", "fixtures", "azure_llm_2024_sample.csv")

TICK_S = 60.0
PRIMARY_SCALE = 100.0
CANDIDATE_RHOS = (0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
STATIC_CA_RHO = 0.65
STATIC_FC_RHO = 0.75
SLA_AWARE_RHO = 0.50
SAFE_TIMEOUT_PCT = 10.0
SAFE_QUEUE_P99_MS = 2000.0
INCREMENTAL_ALPHA_GATE_PCT = 2.0


# ---------------------------------------------------------------------------
# Per-tick + KPI helpers — same shape as the committed dynamic frontier
# audit (scripts/run_azure_2024_dynamic_frontier.py); replicated here so the
# audit module does not modify the committed audit.
# ---------------------------------------------------------------------------

def _eval_tick(tick, target_rho: float, *, tick_hours: float) -> dict:
    r = bt._size_for_target(tick.arrival_rate_rps,
                            max(1.0, tick.output_tokens_mean),
                            bt._tick_throughput_tokps(tick), target_rho)
    ev = bt.evaluate_tick(tick, r, prefill_savings=0.0,
                          tick_hours=tick_hours)
    return {
        "tick_index": tick.tick_index, "start_s": tick.start_s,
        "request_count": tick.request_count, "rho_target": target_rho,
        "replicas": ev.replicas, "rho": ev.rho,
        "timeout_pct": ev.timeout_rate_pct,
        "queue_p95_ms": ev.queue_wait_p95_ms,
        "queue_p99_ms": ev.queue_wait_p99_ms,
        "latency_p99_ms": ev.latency_p99_ms,
        "gpu_hours": sum(ev.gpu_hours_by_type.values()),
        "energy_cost": ev.energy_cost,
        "tokens_offered": ev.tokens_offered,
        "sla_ms": ev.sla_ms,
    }


def _kpi_from_evals(evals: list, *, deadline_slack_ms: Optional[float] = None
                    ) -> dict:
    cfg = InfrastructureCostConfig()
    tokens_per_tick = [e["tokens_offered"] for e in evals]
    timeout_pct_per_tick = [e["timeout_pct"] for e in evals]
    energy_per_tick = [e["energy_cost"] for e in evals]
    gpu_hours_per_tick = [{"a100-40gb": e["gpu_hours"]} for e in evals]
    kpi = compute_economic_kpi(
        tokens_per_tick=tokens_per_tick,
        timeout_rate_pct_per_tick=timeout_pct_per_tick,
        energy_cost_per_tick=energy_per_tick,
        active_gpu_hours_by_type_per_tick=gpu_hours_per_tick,
        migration_count=0, config=cfg)
    active = [e for e in evals if e["request_count"] > 0]
    aw = sum(e["request_count"] for e in active) or 1
    queue_p99_w = sum(e["queue_p99_ms"] * e["request_count"]
                       for e in active) / aw
    timeout_w = sum(e["timeout_pct"] * e["request_count"]
                     for e in active) / aw
    rho_w = sum(e["rho"] * e["request_count"] for e in active) / aw
    deadline_miss_pct = None
    if deadline_slack_ms is not None:
        misses_w = 0
        total_w = 0
        for e in active:
            budget = e["sla_ms"] + deadline_slack_ms
            if e["latency_p99_ms"] > budget:
                misses_w += e["request_count"]
            total_w += e["request_count"]
        if total_w > 0:
            deadline_miss_pct = 100.0 * misses_w / total_w
    return {
        "goodput_per_dollar": float(
            kpi.sla_safe_goodput_per_infra_dollar or 0.0),
        "sla_compliant_goodput": int(kpi.sla_compliant_goodput),
        "gpu_hours": float(kpi.active_gpu_hours),
        "infra_cost": float(kpi.total_infrastructure_cost),
        "timeout_pct_mean": float(timeout_w),
        "queue_p99_ms": float(queue_p99_w),
        "mean_utilization_rho": float(rho_w),
        "n_active_ticks": len(active),
        "deadline_miss_pct": deadline_miss_pct,
    }


# ---------------------------------------------------------------------------
# Baseline policies (all share _eval_tick + _kpi_from_evals).
# ---------------------------------------------------------------------------

def _run_static_policy(ticks, *, tick_hours: float, target_rho: float,
                       deadline_slack_s: Optional[float] = None) -> dict:
    evals = [_eval_tick(t, target_rho, tick_hours=tick_hours) for t in ticks]
    slack_ms = (1000.0 * deadline_slack_s
                if deadline_slack_s is not None else None)
    return {"kpi": _kpi_from_evals(evals, deadline_slack_ms=slack_ms),
            "evals": evals,
            "rho_history": [target_rho] * len(ticks)}


def _run_current_price_only_policy(ticks, *, tick_hours: float) -> dict:
    """Energy-cost-only baseline — same fixed rho but no SLA gate.

    For the v1 audit we mirror the docs/RESULTS.md §3 'energy / flexible
    batch / arbitrage' headline candidate: the policy picks the lowest
    rho replica count that still serves any arrival (i.e. MIN_REPLICAS=1
    when load permits). On the Azure 2024 ticks this collapses to
    rho=0.95 (cheapest static), because we have no carbon / DA-price
    signal to drive a real cheap-window shift.
    """
    return _run_static_policy(
        ticks, tick_hours=tick_hours, target_rho=0.95)


def _run_dynamic_serving_replay(ticks, *, tick_hours: float,
                                window_ticks: int = 30) -> dict:
    """Mirrors scripts/run_azure_2024_dynamic_frontier.py::_run_dynamic_replay
    without modifying that file. Same telemetry + same controller +
    same risk config."""
    profile = WorkloadFrontierProfile(
        workload_id="azure_2024_audit",
        workload_type="inference_standard",
        telemetry_confidence="medium",
        priority_class="standard",
        candidate_rhos=tuple(CANDIDATE_RHOS),
        source="incremental_alpha_audit")
    est_cfg = DynamicEstimatorConfig(min_window_ticks=8)
    ctl_cfg = DynamicControllerConfig()
    safety_cfg = SafetyConfig(max_timeout_pct=SAFE_TIMEOUT_PCT,
                              max_queue_p99_ms=SAFE_QUEUE_P99_MS)
    risk_cfg = RiskConfig()

    history = []
    current_rho = STATIC_CA_RHO
    prev_action = None
    evals = []
    rho_history = []
    decisions = []

    for i, t in enumerate(ticks):
        if len(history) >= est_cfg.min_window_ticks:
            est = estimate_dynamic_frontier(
                workload_profile=profile,
                telemetry_window=history[-window_ticks:],
                current_rho=current_rho,
                candidate_rhos=CANDIDATE_RHOS,
                config=est_cfg, safety_config=safety_cfg, risk_config=risk_cfg)
            dec = choose_dynamic_rho(est, current_rho=current_rho,
                                     config=ctl_cfg,
                                     previous_action=prev_action)
            new_rho = (dec.recommended_rho
                       if dec.recommended_rho is not None else current_rho)
            current_rho = new_rho
            prev_action = dec.action
            decisions.append({"tick": i, "action": dec.action,
                              "rho": current_rho})
        ev = _eval_tick(t, current_rho, tick_hours=tick_hours)
        evals.append(ev)
        rho_history.append(current_rho)
        history.append(telemetry_tick_from_arrival_tick(t))
    return {"kpi": _kpi_from_evals(evals), "evals": evals,
            "rho_history": rho_history,
            "decisions": decisions}


def _run_static_batch_policy(ticks, *, tick_hours: float,
                             deadline_slack_s: float) -> dict:
    """The static Batch Inference Frontier picks the highest-safe-goodput
    (rho, slack) point offline. We then evaluate every tick at that rho."""
    profile = BatchInferenceWorkloadProfile(
        workload_id="azure_2024_audit_static_batch",
        trace_source="azure_llm_2024",
        synthetic_scenario_label="azure_llm_2024_batch_flex_scenario_v1",
        deadline_miss_rate_sla_pct=2.0,
        queue_wait_sla_p99_ms=SAFE_QUEUE_P99_MS,
        telemetry_confidence="medium")
    cands = [BatchInferenceFrontierCandidate(
                target_rho=R, deadline_slack_seconds=deadline_slack_s)
             for R in CANDIDATE_RHOS]
    pts = estimate_batch_inference_frontier(
        profile, ticks, cands,
        estimator_config=BatchInferenceEstimatorConfig(
            tick_seconds=TICK_S),
        safety_config=BatchInferenceSafetyConfig(
            max_deadline_miss_rate_pct=2.0,
            max_timeout_pct=SAFE_TIMEOUT_PCT,
            max_queue_p99_ms=SAFE_QUEUE_P99_MS))
    safe = [p for p in pts if p.is_safe]
    if not safe:
        chosen_rho = STATIC_CA_RHO
    else:
        chosen_rho = max(
            safe, key=lambda p: (p.predicted_goodput_per_dollar or 0.0)
        ).candidate.target_rho
    res = _run_static_policy(ticks, tick_hours=tick_hours,
                             target_rho=chosen_rho,
                             deadline_slack_s=deadline_slack_s)
    res["chosen_rho"] = chosen_rho
    res["deadline_slack_seconds"] = deadline_slack_s
    return res


def _run_dynamic_batch_replay(ticks, *, tick_hours: float,
                              window_ticks: int = 30,
                              deadline_slack_s: float = 300.0) -> dict:
    """Streaming-replay dynamic batch frontier with no future leakage."""
    profile = BatchInferenceWorkloadProfile(
        workload_id="azure_2024_audit_dynamic_batch",
        trace_source="azure_llm_2024",
        synthetic_scenario_label="azure_llm_2024_batch_flex_scenario_v1",
        deadline_miss_rate_sla_pct=2.0,
        queue_wait_sla_p99_ms=SAFE_QUEUE_P99_MS,
        telemetry_confidence="medium")
    est_cfg = DynamicBatchEstimatorConfig(
        min_window_ticks=8,
        candidate_rhos=CANDIDATE_RHOS,
        candidate_deadline_slack_seconds=(deadline_slack_s,),
        candidate_deferral_seconds=(0.0, 60.0, 300.0),
        candidate_batch_concurrency=(1,))
    ctl_cfg = DynamicBatchControllerConfig()
    safety_cfg = BatchInferenceSafetyConfig(
        max_deadline_miss_rate_pct=2.0,
        max_timeout_pct=SAFE_TIMEOUT_PCT,
        max_queue_p99_ms=SAFE_QUEUE_P99_MS)

    current_candidate = BatchInferenceFrontierCandidate(
        target_rho=STATIC_CA_RHO,
        deadline_slack_seconds=deadline_slack_s,
        deferral_window_seconds=0.0,
        batch_concurrency=1)
    history: list[BatchArrivalTelemetryTick] = []
    evals = []
    rho_history = []
    decisions = []
    defer_seconds_history = []
    prev_action = None

    for i, t in enumerate(ticks):
        if len(history) >= est_cfg.min_window_ticks:
            est = estimate_dynamic_batch_frontier(
                profile, history[-window_ticks:],
                current_candidate=current_candidate,
                estimator_config=est_cfg,
                safety_config=safety_cfg,
                tick_seconds=TICK_S)
            dec = choose_dynamic_batch_decision(
                est, current_candidate=current_candidate,
                config=ctl_cfg, previous_action=prev_action,
                confidence="medium")
            if dec.recommended_candidate is not None:
                current_candidate = dec.recommended_candidate
            prev_action = dec.action
            decisions.append({"tick": i, "action": dec.action,
                              "rho": current_candidate.target_rho,
                              "defer_s": current_candidate.deferral_window_seconds})
        ev = _eval_tick(t, current_candidate.target_rho,
                        tick_hours=tick_hours)
        evals.append(ev)
        rho_history.append(current_candidate.target_rho)
        defer_seconds_history.append(
            current_candidate.deferral_window_seconds or 0.0)
        history.append(batch_telemetry_from_arrival(
            t, timeout_pct=ev["timeout_pct"],
            queue_p99_ms=ev["queue_p99_ms"],
            latency_p99_ms=ev["latency_p99_ms"],
            observed_rho=ev["rho"], active_replicas=ev["replicas"]))
    return {"kpi": _kpi_from_evals(
                evals, deadline_slack_ms=1000.0 * deadline_slack_s),
            "evals": evals, "rho_history": rho_history,
            "decisions": decisions,
            "defer_seconds_history": defer_seconds_history,
            "deadline_slack_seconds": deadline_slack_s,
            "n_actions": _count_actions(decisions)}


def _count_actions(decisions: list) -> dict:
    out: dict = {}
    for d in decisions:
        out[d["action"]] = out.get(d["action"], 0) + 1
    return out


# ---------------------------------------------------------------------------
# Audit driver.
# ---------------------------------------------------------------------------

def _safety_regression(label: str, result: dict, ref: dict) -> dict:
    """Returns dict of regression flags vs constraint_aware_static."""
    return {
        "policy": label,
        "timeout_regression_vs_ca_static": (
            result["kpi"]["timeout_pct_mean"]
            > ref["kpi"]["timeout_pct_mean"] + 1e-6),
        "queue_p99_regression_vs_ca_static": (
            result["kpi"]["queue_p99_ms"]
            > ref["kpi"]["queue_p99_ms"] + 1e-6),
    }


def _pct_delta(a, b):
    if b is None or b == 0:
        return None
    return (a - b) / b * 100.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=AZURE_FIXTURE)
    p.add_argument("--scale-rps", type=float, default=PRIMARY_SCALE)
    p.add_argument("--tick-seconds", type=float, default=TICK_S)
    p.add_argument("--window-ticks", type=int, default=30)
    p.add_argument("--deadline-slack-s", type=float, default=300.0)
    p.add_argument("--output", default=OUT_JSON)
    args = p.parse_args(argv)

    reqs = azure_llm.load_csv(args.source)
    if args.scale_rps != 1.0:
        reqs = time_rescale(reqs, factor=args.scale_rps)
    ticks = requests_to_arrival_ticks(reqs, tick_seconds=args.tick_seconds)
    tick_hours = args.tick_seconds / 3600.0
    print(f"[audit] requests={len(reqs):,} ticks={len(ticks):,} "
          f"active={sum(1 for t in ticks if t.request_count > 0)}")

    results: dict = {}
    print("[audit] running baselines...")
    results["sla_aware"] = _run_static_policy(
        ticks, tick_hours=tick_hours, target_rho=SLA_AWARE_RHO,
        deadline_slack_s=args.deadline_slack_s)
    results["current_price_only"] = _run_current_price_only_policy(
        ticks, tick_hours=tick_hours)
    results["constraint_aware_static"] = _run_static_policy(
        ticks, tick_hours=tick_hours, target_rho=STATIC_CA_RHO,
        deadline_slack_s=args.deadline_slack_s)
    results["static_serving_frontier_controller"] = _run_static_policy(
        ticks, tick_hours=tick_hours, target_rho=STATIC_FC_RHO,
        deadline_slack_s=args.deadline_slack_s)
    print("[audit] running dynamic serving frontier...")
    results["dynamic_serving_frontier"] = _run_dynamic_serving_replay(
        ticks, tick_hours=tick_hours, window_ticks=args.window_ticks)
    print("[audit] running static batch frontier...")
    results["static_batch_inference_frontier"] = _run_static_batch_policy(
        ticks, tick_hours=tick_hours,
        deadline_slack_s=args.deadline_slack_s)
    print("[audit] running dynamic batch frontier...")
    results["dynamic_batch_inference_frontier"] = _run_dynamic_batch_replay(
        ticks, tick_hours=tick_hours, window_ticks=args.window_ticks,
        deadline_slack_s=args.deadline_slack_s)

    # --- Alpha decomposition + safety regressions ---
    ca_static = results["constraint_aware_static"]["kpi"]
    dyn_serv = results["dynamic_serving_frontier"]["kpi"]
    static_batch = results["static_batch_inference_frontier"]["kpi"]
    dyn_batch = results["dynamic_batch_inference_frontier"]["kpi"]

    duplicated_alpha = _pct_delta(dyn_serv["goodput_per_dollar"],
                                  ca_static["goodput_per_dollar"])
    deadline_flex_alpha = _pct_delta(static_batch["goodput_per_dollar"],
                                     ca_static["goodput_per_dollar"])
    incremental_alpha_vs_dynamic_serv = _pct_delta(
        dyn_batch["goodput_per_dollar"], dyn_serv["goodput_per_dollar"])
    incremental_alpha_vs_static_batch = _pct_delta(
        dyn_batch["goodput_per_dollar"], static_batch["goodput_per_dollar"])

    safety_flags = {
        k: _safety_regression(k, v, results["constraint_aware_static"])
        for k, v in results.items()
    }

    # Acceptance gate — both conditions must hold.
    no_safety_regression = not (
        safety_flags["dynamic_batch_inference_frontier"]["timeout_regression_vs_ca_static"]
        or safety_flags["dynamic_batch_inference_frontier"]["queue_p99_regression_vs_ca_static"]
    )
    alpha_gate_passed = (
        incremental_alpha_vs_dynamic_serv is not None
        and incremental_alpha_vs_dynamic_serv > INCREMENTAL_ALPHA_GATE_PCT)
    verdict = ("PROPOSE_INTEGRATION"
               if (alpha_gate_passed and no_safety_regression)
               else "SHADOW_DIAGNOSTIC")

    # --- Compact per-policy comparison table ---
    table = []
    for label, r in results.items():
        kpi = r["kpi"]
        table.append({
            "policy": label,
            "goodput_per_dollar": kpi["goodput_per_dollar"],
            "sla_compliant_goodput": kpi["sla_compliant_goodput"],
            "gpu_hours": kpi["gpu_hours"],
            "infra_cost": kpi["infra_cost"],
            "timeout_pct_mean": kpi["timeout_pct_mean"],
            "queue_p99_ms": kpi["queue_p99_ms"],
            "mean_utilization_rho": kpi["mean_utilization_rho"],
            "deadline_miss_pct": kpi.get("deadline_miss_pct"),
        })

    dyn_batch_actions = (results["dynamic_batch_inference_frontier"]
                         .get("n_actions") or {})

    payload = {
        "doc_version": "batch_inference_frontier_incremental_alpha_audit_v1",
        "production_claim": False,
        "ml_training": False,
        "modifies_serving_rho_controller": False,
        "modifies_constraint_aware_default": False,
        "uses_oracle_as_headline": False,
        "executable_in_real_cluster": False,
        "source": {
            "trace": "azure_llm_2024", "path": args.source,
            "scale_rps": args.scale_rps,
            "tick_seconds": args.tick_seconds,
            "request_count": len(reqs), "tick_count": len(ticks),
        },
        "config": {
            "window_ticks": args.window_ticks,
            "deadline_slack_seconds": args.deadline_slack_s,
            "candidate_rhos": list(CANDIDATE_RHOS),
            "safe_timeout_pct": SAFE_TIMEOUT_PCT,
            "safe_queue_p99_ms": SAFE_QUEUE_P99_MS,
            "incremental_alpha_gate_pct": INCREMENTAL_ALPHA_GATE_PCT,
        },
        "policy_kpi_table": table,
        "alpha_decomposition_pct": {
            "duplicated_serving_frontier_alpha_pct": duplicated_alpha,
            "deadline_flex_scenario_alpha_pct": deadline_flex_alpha,
            "true_incremental_alpha_vs_dynamic_serving_pct":
                incremental_alpha_vs_dynamic_serv,
            "incremental_alpha_vs_static_batch_pct":
                incremental_alpha_vs_static_batch,
        },
        "safety_regression_flags": safety_flags,
        "dynamic_batch_action_distribution": dyn_batch_actions,
        "acceptance_gate": {
            "incremental_alpha_gate_pct": INCREMENTAL_ALPHA_GATE_PCT,
            "incremental_alpha_value_pct": incremental_alpha_vs_dynamic_serv,
            "alpha_gate_passed": alpha_gate_passed,
            "no_safety_regression": no_safety_regression,
            "verdict": verdict,
        },
        "honesty_notes": [
            "simulator / public-trace evidence only — NOT production savings",
            "Azure LLM 2024 is a SERVING trace re-used as a synthetic "
            "batch-flex scenario; the deadline-slack is a scenario knob",
            "no oracle / clairvoyant baseline used as headline",
            "no serving rho controller default changed",
            "no constraint_aware default changed",
            "executable_in_real_cluster is False at construction",
        ],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)

    # Pretty-print compact summary to stdout.
    print("\n=== Incremental Alpha Audit — Batch Inference Frontier v1 ===")
    print(f"trace: {os.path.basename(args.source)} "
          f"scale={args.scale_rps}× ticks={len(ticks):,}")
    print(f"{'policy':<40} {'goodput/$':>15} {'timeout%':>10} "
          f"{'queueP99':>10} {'mean_rho':>10}")
    for row in table:
        print(f"{row['policy']:<40} {row['goodput_per_dollar']:>15,.2f} "
              f"{row['timeout_pct_mean']:>10.4f} "
              f"{row['queue_p99_ms']:>10.2f} "
              f"{row['mean_utilization_rho']:>10.4f}")
    print("\n=== Alpha decomposition ===")
    print(f"  duplicated_serving_frontier_alpha_pct      = "
          f"{duplicated_alpha:+.4f} %"
          if duplicated_alpha is not None else
          "  duplicated_serving_frontier_alpha_pct      = n/a")
    print(f"  deadline_flex_scenario_alpha_pct           = "
          f"{deadline_flex_alpha:+.4f} %"
          if deadline_flex_alpha is not None else
          "  deadline_flex_scenario_alpha_pct           = n/a")
    print(f"  true_incremental_alpha_vs_dynamic_serv_pct = "
          f"{incremental_alpha_vs_dynamic_serv:+.4f} %  "
          f"(gate: > +{INCREMENTAL_ALPHA_GATE_PCT}%)"
          if incremental_alpha_vs_dynamic_serv is not None else
          f"  true_incremental_alpha_vs_dynamic_serv_pct = n/a  "
          f"(gate: > +{INCREMENTAL_ALPHA_GATE_PCT}%)")
    print(f"\n  no_safety_regression = {no_safety_regression}")
    print(f"\n  VERDICT: {verdict}")
    print(f"\n  summary -> {args.output}")
    return 0 if verdict == "PROPOSE_INTEGRATION" else 0


if __name__ == "__main__":
    raise SystemExit(main())
