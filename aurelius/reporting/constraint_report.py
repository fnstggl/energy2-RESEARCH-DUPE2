"""Formatting helpers for constraint-aware assessment/recommendation reports.

Supports text (terminal) and JSON output.
No secrets are included in any report output.
Sandbox outputs are labeled to prevent use in economic claims.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

from ..constraints.engine import EngineResult
from ..simulation.cluster.engine import TickMetrics
from ..state.models import ConstraintAssessment, ConstraintType, TopologyLinkType

# Constraint display order and label
_CONSTRAINT_LABELS: dict[str, str] = {
    "energy": "Energy-Bound",
    "thermal": "Thermal-Bound",
    "queue": "Queue-Bound",
    "latency": "Latency-Bound",
    "communication": "Communication-Bound",
    "memory": "Memory-Bound (indirect)",
    "topology": "Topology-Bound",
    "utilization": "Utilization-Bound",
    "none": "No Binding Constraint",
}

_BAR_WIDTH = 30  # characters for score bar


def _bar(score: float, width: int = _BAR_WIDTH) -> str:
    filled = max(0, min(width, int(score * width)))
    return "█" * filled + "░" * (width - filled)


def _confidence_label(conf: float) -> str:
    if conf >= 0.75:
        return "HIGH"
    if conf >= 0.40:
        return "MEDIUM"
    if conf >= 0.15:
        return "LOW"
    return "VERY LOW"


def format_assessment_text(assessment: ConstraintAssessment) -> str:
    """Format a ConstraintAssessment as human-readable terminal text."""
    lines: list[str] = []

    sandbox_note = "  [SANDBOX — not for savings claims]" if assessment.provenance.is_sandbox else ""
    ts_str = assessment.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append("=" * 60)
    lines.append(f"CONSTRAINT ASSESSMENT{sandbox_note}")
    lines.append(f"Timestamp : {ts_str}")
    if assessment.region:
        lines.append(f"Region    : {assessment.region}")
    conf_lbl = _confidence_label(assessment.confidence)
    lines.append(f"Confidence: {assessment.confidence:.2f} ({conf_lbl})")
    lines.append("")

    bc = assessment.binding_constraint
    bc_label = _CONSTRAINT_LABELS.get(bc.value if bc else "none", "Unknown")
    if bc is None:
        lines.append("BINDING CONSTRAINT: None detected")
        lines.append(f"  Reason: {assessment.rationale}")
    else:
        lines.append(f"BINDING CONSTRAINT: {bc_label} ({bc.value})")
        lines.append(f"  {assessment.rationale}")
    lines.append("")

    if assessment.scores:
        lines.append("CONSTRAINT SCORES:")
        sorted_scores = sorted(assessment.scores.items(), key=lambda x: -x[1])
        for ct, score in sorted_scores:
            lbl = _CONSTRAINT_LABELS.get(ct.value, ct.value)
            bar = _bar(score)
            marker = " ◀ BINDING" if ct == bc else ""
            lines.append(f"  {lbl:<30} {bar} {score:.3f}{marker}")
        lines.append("")

    if assessment.missing_signals:
        lines.append("MISSING TELEMETRY SIGNALS:")
        for sig in sorted(assessment.missing_signals):
            lines.append(f"  - {sig}")
        lines.append("")

    if assessment.safe_action_types:
        lines.append("SAFE ACTIONS:")
        for a in assessment.safe_action_types:
            lines.append(f"  + {a}")
        lines.append("")

    if assessment.disallowed_action_types:
        lines.append("DISALLOWED ACTIONS (current constraint):")
        for a in assessment.disallowed_action_types:
            lines.append(f"  ✗ {a}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_recommendations_text(result: EngineResult) -> str:
    """Format an EngineResult (recommendations) as terminal text."""
    lines: list[str] = []
    sandbox_note = "  [SANDBOX]" if result.assessment.provenance.is_sandbox else ""
    ts_str = result.assessment.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append("=" * 60)
    lines.append(f"RECOMMENDATIONS{sandbox_note}")
    lines.append(f"Timestamp  : {ts_str}")
    lines.append("Mode       : recommendation_only (dry-run)")
    lines.append(
        f"Total      : {len(result.recommendations)} recommendations "
        f"({result.actionable_count} actionable, {result.noop_count} KEEP)"
    )
    lines.append(f"Elapsed    : {result.elapsed_ms:.1f}ms")
    lines.append("")

    for i, rec in enumerate(result.recommendations, 1):
        noop_marker = " [KEEP]" if rec.is_noop else ""
        sla_marker = f" [SLA:{rec.sla_status.upper()}]" if rec.sla_status != "unknown" else ""
        bc_str = rec.binding_constraint.value if rec.binding_constraint else "none"
        lines.append(
            f"  {i}. {rec.action_type}{noop_marker}{sla_marker}"
            f" — workload={rec.workload_id} constraint={bc_str}"
            f" conf={rec.confidence:.2f}"
        )
        if rec.net_benefit is not None:
            lines.append(f"     Net benefit: {rec.net_benefit:+.4f}")
        if rec.migration_penalty is not None:
            lines.append(f"     Migration penalty: {rec.migration_penalty:.4f}")
        if rec.rationale:
            lines.append(f"     Rationale: {rec.rationale[:120]}")

    lines.append("")
    if result.rejected:
        lines.append(f"REJECTED ACTIONS: {len(result.rejected)}")
        for rej in result.rejected[:5]:
            reason = rej.get("reason", "unknown")
            action = rej.get("action", "?")
            lines.append(f"  - {action}: {reason}")
        if len(result.rejected) > 5:
            lines.append(f"  ... and {len(result.rejected) - 5} more")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_engine_result_json(result: EngineResult) -> str:
    """Serialize EngineResult to JSON. No secrets included."""
    return json.dumps(result.to_dict(), indent=2, default=str)


def format_telemetry_check_text(assessment: ConstraintAssessment) -> str:
    """Format telemetry coverage report from a ConstraintAssessment."""
    lines: list[str] = []
    sandbox_note = "  [SANDBOX]" if assessment.provenance.is_sandbox else ""

    lines.append("=" * 60)
    lines.append(f"TELEMETRY COVERAGE CHECK{sandbox_note}")
    lines.append("")

    # Which constraints have data vs not
    scored = set(assessment.scores.keys())
    all_constraints = list(ConstraintType)

    lines.append("CONSTRAINT DETECTION CAPABILITY:")
    for ct in all_constraints:
        if ct == ConstraintType.NONE:
            continue
        status = "DETECTABLE  " if ct in scored else "MISSING DATA"
        score_str = f"(score={assessment.scores[ct]:.3f})" if ct in scored else ""
        lines.append(f"  {ct.value:<22} {status} {score_str}")
    lines.append("")

    lines.append(f"MISSING SIGNALS ({len(assessment.missing_signals)} total):")
    if assessment.missing_signals:
        for sig in sorted(assessment.missing_signals):
            lines.append(f"  - {sig}")
    else:
        lines.append("  None — all required signals present")
    lines.append("")

    detectable_count = sum(1 for ct in scored if ct != ConstraintType.NONE)
    total_count = sum(1 for ct in all_constraints if ct != ConstraintType.NONE)
    coverage_pct = (detectable_count / total_count * 100) if total_count else 0.0
    lines.append("COVERAGE SUMMARY:")
    lines.append(f"  Detectable constraints: {detectable_count}/{total_count} ({coverage_pct:.0f}%)")
    lines.append(f"  Overall confidence    : {assessment.confidence:.2f} ({_confidence_label(assessment.confidence)})")

    if assessment.confidence < 0.4:
        lines.append("")
        lines.append("WARNING: Low confidence — classifier may default to KEEP/no-op.")
        lines.append("         Add more telemetry sources to improve constraint detection.")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_topology_report_text(cluster_state: Any) -> str:
    """Format a topology summary report from a ClusterState."""
    from ..state.models import ClusterState
    state: ClusterState = cluster_state

    lines: list[str] = []
    sandbox_note = "  [SANDBOX]" if state.provenance.is_sandbox else ""
    ts_str = state.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines.append("=" * 60)
    lines.append(f"TOPOLOGY REPORT{sandbox_note}")
    lines.append(f"Timestamp : {ts_str}")
    lines.append("")

    # Aggregate counts from all regions (nodes are inside RegionState)
    all_nodes = {nid: n for r in state.regions.values() for nid, n in r.nodes.items()}
    total_gpus = sum(len(n.gpus) for n in all_nodes.values())

    lines.append("CLUSTER TOPOLOGY:")
    lines.append(f"  Regions   : {len(state.regions)}")
    lines.append(f"  Nodes     : {len(all_nodes)}")
    lines.append(f"  Total GPUs: {total_gpus}")
    lines.append("")

    # Node inventory per region
    for region_id, region in sorted(state.regions.items()):
        if not region.nodes:
            continue
        lines.append(f"REGION: {region_id}")
        for nid, node in sorted(region.nodes.items()):
            zone_str = f" zone={node.zone}" if node.zone else ""
            rack_str = f" rack={node.rack_id}" if node.rack_id else ""
            gpu_count = len(node.gpus)
            lines.append(f"  {nid:<20} GPUs={gpu_count}{zone_str}{rack_str}")

        # Topology per region (stored in RegionState.topology)
        if region.topology is not None:
            topo = region.topology
            pair_count = len(topo.pair_levels)
            lines.append(f"  Topology: {len(topo.gpu_uuids)} GPUs, {pair_count} link pairs")
            if topo.pair_levels:
                all_links = list(topo.pair_levels.values())
                # Bandwidth quality: lower penalty = better (NVSWITCH=0.0, REGION=1.0)
                _PENALTY: dict[TopologyLinkType, float] = {
                    TopologyLinkType.NVSWITCH: 0.00, TopologyLinkType.NV4: 0.05,
                    TopologyLinkType.NV3: 0.10,      TopologyLinkType.NV2: 0.15,
                    TopologyLinkType.NV1: 0.20,      TopologyLinkType.PIX: 0.35,
                    TopologyLinkType.PXB: 0.45,      TopologyLinkType.PHB: 0.55,
                    TopologyLinkType.NODE: 0.70,     TopologyLinkType.SYS: 0.85,
                    TopologyLinkType.RACK: 0.92,     TopologyLinkType.REGION: 1.00,
                }
                best = min(all_links, key=lambda lnk: _PENALTY.get(lnk, 0.5))
                worst = max(all_links, key=lambda lnk: _PENALTY.get(lnk, 0.5))
                lines.append(
                    f"    Best link : {best.value} quality={1.0 - _PENALTY.get(best, 0.5):.2f}"
                )
                lines.append(
                    f"    Worst link: {worst.value} quality={1.0 - _PENALTY.get(worst, 0.5):.2f}"
                )
        else:
            lines.append("  Topology: not available (nvidia-smi topo not collected)")
        lines.append("")

    # Service placement
    services = state.all_services
    if services:
        lines.append("SERVICE PLACEMENT:")
        for sid, svc in sorted(services.items()):
            node_str = f"node={svc.node_id}" if svc.node_id else "node=UNKNOWN"
            region_str = f"region={svc.region}" if svc.region else ""
            lines.append(f"  {sid:<30} {node_str} {region_str}")
        lines.append("")

    lines.append("NOTE: Topology-aware placement recommendations are in constraint-report.")
    lines.append("=" * 60)
    return "\n".join(lines)


def format_scenario_comparison_table(
    scenario_name: str,
    tick_metrics: list[TickMetrics],
    engine_results: list[EngineResult],
) -> str:
    """Format a baseline vs Aurelius comparison table for a simulated scenario."""
    lines: list[str] = []

    lines.append("=" * 80)
    lines.append(f"SCENARIO: {scenario_name}")
    lines.append("COMPARISON: No-Op Baseline vs Constraint-Aware Aurelius")
    lines.append("[SANDBOX — is_sandbox=True — not for external savings claims]")
    lines.append("")

    if not tick_metrics:
        lines.append("No ticks run.")
        lines.append("=" * 80)
        return "\n".join(lines)

    # Aggregate metrics
    total_cost = sum(m.total_energy_cost for m in tick_metrics)
    total_tokens = sum(m.total_tokens for m in tick_metrics)
    total_sla_violations = sum(m.sla_violations for m in tick_metrics)
    total_migrations = sum(m.migration_count for m in tick_metrics)
    total_throttled = sum(m.thermal_throttle_gpu_count for m in tick_metrics)

    p99_values = [m.p99_latency_ms for m in tick_metrics if m.p99_latency_ms is not None]
    q95_values = [m.queue_wait_p95_ms for m in tick_metrics if m.queue_wait_p95_ms is not None]
    util_values = [m.mean_gpu_util_pct for m in tick_metrics]

    avg_p99 = sum(p99_values) / len(p99_values) if p99_values else None
    avg_q95 = sum(q95_values) / len(q95_values) if q95_values else None
    avg_util = sum(util_values) / len(util_values) if util_values else None

    # Engine summary
    total_actionable = sum(r.actionable_count for r in engine_results)
    total_noop = sum(r.noop_count for r in engine_results)
    total_recs = len(engine_results)

    # Constraint distribution
    constraint_counts: dict[str, int] = {}
    for er in engine_results:
        bc = er.assessment.binding_constraint
        key = bc.value if bc else "none"
        constraint_counts[key] = constraint_counts.get(key, 0) + 1

    lines.append("AGGREGATE METRICS (simulator baseline — no optimizer interventions):")
    lines.append(f"  Total ticks          : {len(tick_metrics)}")
    lines.append(f"  Total energy cost    : ${total_cost:.4f}")
    if total_tokens > 0:
        cpt = total_cost / total_tokens
        lines.append(f"  Cost per token       : ${cpt:.8f}")
    lines.append(f"  Total tokens served  : {total_tokens:,}")
    if avg_util is not None:
        lines.append(f"  Avg GPU utilization  : {avg_util:.1f}%")
    if avg_p99 is not None:
        lines.append(f"  Avg p99 latency      : {avg_p99:.1f}ms")
    if avg_q95 is not None:
        lines.append(f"  Avg queue wait p95   : {avg_q95:.1f}ms")
    lines.append(f"  SLA violations       : {total_sla_violations}")
    lines.append(f"  Migrations           : {total_migrations}")
    lines.append(f"  Thermal throttle ticks: {total_throttled}")
    lines.append("")

    lines.append("AURELIUS RECOMMENDATIONS (recommendation_only — not executed):")
    lines.append(f"  Ticks assessed       : {total_recs}")
    lines.append(f"  Actionable recs      : {total_actionable}")
    lines.append(f"  KEEP (no-op) recs    : {total_noop}")
    lines.append("")

    lines.append("BINDING CONSTRAINT DISTRIBUTION:")
    for ct_val, count in sorted(constraint_counts.items(), key=lambda x: -x[1]):
        label = _CONSTRAINT_LABELS.get(ct_val, ct_val)
        pct = count / total_recs * 100 if total_recs else 0.0
        lines.append(f"  {label:<35} {count:3d} ticks ({pct:.0f}%)")
    lines.append("")

    # Per-tick table header
    lines.append("PER-TICK SUMMARY:")
    hdr = f"{'Tick':>4}  {'Cost $':>8}  {'Tokens':>8}  {'Util%':>5}  "
    hdr += f"{'p99ms':>6}  {'SLA_V':>5}  {'Migr':>4}  Constraint"
    lines.append(hdr)
    lines.append("-" * 80)

    for tick_m, eng_r in zip(tick_metrics, engine_results):
        bc = eng_r.assessment.binding_constraint
        bc_str = (bc.value if bc else "none")[:14]
        p99_str = f"{tick_m.p99_latency_ms:.0f}" if tick_m.p99_latency_ms is not None else "  N/A"
        row = (
            f"{tick_m.tick:>4}  {tick_m.total_energy_cost:>8.4f}  "
            f"{tick_m.total_tokens:>8}  {tick_m.mean_gpu_util_pct:>5.1f}  "
            f"{p99_str:>6}  {tick_m.sla_violations:>5}  "
            f"{tick_m.migration_count:>4}  {bc_str}"
        )
        lines.append(row)

    lines.append("")
    lines.append("NOTE: Recommendations are dry-run only. No cluster mutations occurred.")
    lines.append("      Savings estimates come from engine net_benefit fields (heuristic).")
    lines.append("=" * 80)
    return "\n".join(lines)


def format_validate_connectors_report(results: list[dict[str, Any]]) -> str:
    """Format connector validation results."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CONNECTOR VALIDATION REPORT")
    lines.append("(Fake/sandbox connectors — same code paths as real integration)")
    lines.append("")

    all_passed = all(r.get("passed", False) for r in results)
    overall = "ALL PASSED" if all_passed else "FAILURES DETECTED"
    lines.append(f"OVERALL: {overall}")
    lines.append("")

    for r in results:
        name = r.get("name", "unknown")
        passed = r.get("passed", False)
        status = "PASS" if passed else "FAIL"
        detail = r.get("detail", "")
        lines.append(f"  [{status}] {name}")
        if detail:
            for line in textwrap.wrap(detail, 56):
                lines.append(f"       {line}")
        if not passed:
            err = r.get("error", "")
            if err:
                lines.append(f"       ERROR: {err[:100]}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
