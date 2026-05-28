#!/usr/bin/env python3
"""Generate the Simulator Realism + Benchmark Validation report (task spec §9).

Runs every scenario under all policies (paired same-seed A/B), runs the simulator
realism audit, computes the packing frontier where applicable, and writes a
markdown report with:
  - the full per-scenario comparison table
  - mean/median energy-cost delta + engine net-savings vs each baseline
  - the realism audit verdicts
  - honest notes on where constraint_aware wins / loses

Usage:
    python scripts/generate_realism_report.py [--steps 24] [--seed 42] \
        [--out docs/REALISM_BENCHMARK_VALIDATION.md]

All outputs are [SANDBOX] / simulator-only. Not production savings.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aurelius.benchmarks import (  # noqa: E402
    ConstraintBenchmarkRunner,
    CrossScenarioReport,
    run_realism_audit,
)
from aurelius.simulation.cluster.scenarios import list_scenarios  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="docs/REALISM_BENCHMARK_VALIDATION.md")
    args = ap.parse_args()

    runner = ConstraintBenchmarkRunner()
    scenarios = list_scenarios()
    print(
        f"Running {len(scenarios)} scenarios × 5 policies "
        f"(seed={args.seed}, steps={args.steps})"
    )
    all_results: dict = {}
    for scn in scenarios:
        try:
            all_results[scn] = runner.run_scenario(
                scn, seed=args.seed, steps=args.steps,
            )
        except Exception as exc:
            print(f"  scenario {scn} failed: {exc}")

    audit = run_realism_audit(seed=args.seed)
    cross = CrossScenarioReport.from_results(all_results)

    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines.append("# Simulator Realism + Benchmark Validation Report")
    lines.append("")
    lines.append(
        f"_Generated {ts} · seed={args.seed} · steps={args.steps} · **[SANDBOX]**_"
    )
    lines.append("")
    lines.append("> All numbers are **simulator-only, uncalibrated** directional "
                 "results. Not production savings. See the realism audit "
                 "verdict below.")
    lines.append("")
    lines.append("> **Primary KPI:** `sla_safe_goodput_per_infrastructure_dollar`. "
                 "Per-workload comparison uses the *workload-relevant strong "
                 "baseline*, not FIFO. FIFO is the sanity-only baseline. "
                 "Telemetry-failsafe scenarios are scored on KEEP-correctness, "
                 "not alpha.")
    lines.append("")
    lines.append("> ML forecasting is a later phase, after the optimizer has the "
                 "right objective and workload-aware decision rules. Simulator "
                 "results remain not production savings claims.")
    lines.append("")
    lines.append(f"**Realism audit overall verdict: `{audit.overall_verdict}`**")
    lines.append("")
    lines.append("## 1. Realism audit (per-subsystem)")
    lines.append("")
    lines.append("| Subsystem | Verdict | Calibration |")
    lines.append("|---|---|---|")
    for sv in audit.subsystems:
        lines.append(
            f"| {sv.subsystem} | `{sv.verdict}` | "
            f"{sv.calibration_confidence or '—'} |"
        )
    lines.append("")
    lines.append("Headline findings:")
    for f in audit.headline_findings:
        lines.append(f"- {f}")
    lines.append("")

    # Sections A–D: per-workload-type baseline reporting tables.
    lines.append(cross.to_markdown())

    # Telemetry-confidence inline panel: how the engine perceived data quality.
    lines.append("## E. Telemetry confidence (constraint_aware engine)")
    lines.append("")
    lines.append("Telemetry truth signal from the engine assessments. Telemetry-"
                 "failsafe scenarios are expected to show partial-confidence and "
                 "force-KEEP behavior.")
    lines.append("")
    lines.append("| scenario | mean confidence | partial |")
    lines.append("|---|---|---|")
    for scn, res in all_results.items():
        pr = res.policy_results.get("constraint_aware")
        if pr is None:
            continue
        confs = [
            er.assessment.confidence
            for er in (pr.engine_results or [])
            if er is not None
        ]
        mean_conf = statistics.mean(confs) if confs else None
        is_partial = bool(pr.final_state.is_partial) if pr.final_state else False
        conf_str = f"{mean_conf:.2f}" if mean_conf is not None else "—"
        lines.append(f"| {scn} | {conf_str} | {'yes' if is_partial else 'no'} |")
    lines.append("")

    lines.append("## F. What remains simulator-only / needs real telemetry")
    lines.append("")
    lines.append("- Every calibration parameter is an uncalibrated prior (none "
                 "measured on real hardware). All KPI numbers are directional.")
    lines.append("- Telemetry truth (Mission 1, FIXED): the canonical `ClusterState` "
                 "derives provenance confidence + `is_partial` from the simulator's "
                 "per-subsystem tiers, so degraded-telemetry scenarios report "
                 "low/partial confidence and the engine force-KEEPs (telemetry "
                 "subsystem verdict graduated to REALISTIC_ENOUGH_FOR_DEV). The "
                 "tiers themselves remain uncalibrated heuristics.")
    lines.append("- ML forecasting is a later phase. The current optimizer relies "
                 "on the engine's workload-aware decision rules; once those land, "
                 "a calibrated forecaster will get layered on top. Simulator "
                 "results remain not production savings claims.")
    lines.append("- Next calibration step: run a read-only shadow pilot against "
                 "real Prometheus/DCGM/K8s telemetry to calibrate the priors and "
                 "the confidence model (C = R·F·K·S·N) against measured "
                 "staleness/coverage/noise.")
    lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out} ({len(cross.rows)} scenario rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
