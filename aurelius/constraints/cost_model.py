"""Migration cost/risk model for Aurelius constraint-aware orchestration.

Phase 8: before any recommendation is emitted, estimate the total cost impact
including hidden operational penalties. Conservative heuristics are used when
no trained predictor is available.

Design rules:
- All penalty estimates are HEURISTIC and labeled as such.
- A recommendation with net_expected_savings <= 0 must produce a KEEP (no-op).
- Critical workloads (latency_sensitive=True, priority_tier="critical") always
  receive a larger risk buffer than batch/flexible workloads.
- The model never touches KV cache internals, NCCL, or memory allocators.
- MigrationGovernor prevents trigger-happy optimization through cooldown,
  per-workload history, and cluster-level migration-rate limits.

This module is read-only over ClusterState. It produces cost estimates and
decisions but does NOT execute migrations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from ..sla.actions import MIGRATION_ACTIONS, ActionType
from ..state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    Provenance,
    Recommendation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model configuration
# ---------------------------------------------------------------------------

@dataclass
class CostModelConfig:
    """Configurable parameters for the migration cost/risk estimator.

    All values are HEURISTIC — derived from engineering judgment, not from
    calibrated production telemetry. Tune against real outcomes before
    using in production savings claims.
    """

    # Cold-start penalty (ms added to p99 during warmup period)
    cold_start_p99_penalty_ms: float = 2000.0     # HEURISTIC
    # Topology degradation: fraction of NVLink bandwidth lost on cross-node placement
    topology_degradation_fraction: float = 0.30   # HEURISTIC
    # Cache warmup: fraction of prefix cache hit rate lost during warmup ticks
    cache_warmup_hit_rate_loss: float = 0.40       # HEURISTIC
    # Queue instability: extra p95 queue wait during migration (ms)
    queue_instability_penalty_ms: float = 300.0   # HEURISTIC
    # SLA risk buffer multiplier for critical/latency-sensitive workloads
    critical_workload_risk_multiplier: float = 2.5  # HEURISTIC
    # Batch workload risk multiplier (lower = more permissive migration)
    batch_workload_risk_multiplier: float = 0.4   # HEURISTIC
    # Minimum net savings ($) required to justify a migration
    min_net_savings_threshold: float = 0.0        # HEURISTIC: 0 = break-even; tune upward for conservatism
    # Minimum savings improvement fraction over current cost to justify migrate
    # e.g. 0.05 means at least 5% savings after penalties
    min_savings_improvement_fraction: float = 0.0  # HEURISTIC

    # Hysteresis: minimum seconds between migrations for the same workload
    min_migration_interval_s: float = 300.0       # HEURISTIC: 5 min
    # Max migrations per workload per hour
    max_migrations_per_workload_per_hour: int = 2  # HEURISTIC
    # Max cluster-wide migrations per minute
    max_cluster_migrations_per_minute: int = 3    # HEURISTIC
    # Cooldown after SLA violation (seconds)
    sla_violation_cooldown_s: float = 600.0       # HEURISTIC: 10 min


# ---------------------------------------------------------------------------
# Per-action cost/risk estimate
# ---------------------------------------------------------------------------

@dataclass
class MigrationCostEstimate:
    """Per-candidate action cost/risk breakdown.

    All monetary fields are in the same units as EnergyState.price_per_mwh
    (dollars per MWh). Penalty fields are additive cost deltas.

    net_expected_savings > 0 means the action is estimated to save money/resources
    after all penalties. net_expected_savings <= 0 means the action is a no-op
    candidate or should be blocked.
    """
    workload_id: str
    action_type: str
    timestamp: datetime

    # Raw savings before penalties (from energy/cost model)
    gross_energy_savings: Optional[float] = None       # $/MWh equivalent
    gross_compute_savings: Optional[float] = None      # cost units

    # Latency penalties
    cold_start_penalty_ms: float = 0.0
    cache_warmup_penalty_ms: float = 0.0

    # Queue/throughput penalties
    queue_instability_penalty_ms: float = 0.0
    lost_batching_efficiency_pct: float = 0.0

    # Infrastructure penalties
    network_transfer_cost: float = 0.0
    topology_degradation_score: float = 0.0    # [0, 1]; higher = worse

    # SLA and thermal risk penalties
    sla_risk_penalty: float = 0.0
    thermal_penalty: float = 0.0
    failure_retry_penalty: float = 0.0

    # Composite
    total_penalty: float = 0.0
    net_expected_savings: Optional[float] = None
    confidence: float = 0.0

    # Governance
    blocked_by_cooldown: bool = False
    blocked_reason: Optional[str] = None

    explanation: str = ""

    def is_viable(self) -> bool:
        """True if the action passes the cost/benefit threshold."""
        if self.blocked_by_cooldown:
            return False
        if self.net_expected_savings is None:
            return False
        return self.net_expected_savings > 0

    def to_dict(self) -> dict:
        return {
            "workload_id": self.workload_id,
            "action_type": self.action_type,
            "timestamp": self.timestamp.isoformat(),
            "gross_energy_savings": self.gross_energy_savings,
            "gross_compute_savings": self.gross_compute_savings,
            "cold_start_penalty_ms": self.cold_start_penalty_ms,
            "cache_warmup_penalty_ms": self.cache_warmup_penalty_ms,
            "queue_instability_penalty_ms": self.queue_instability_penalty_ms,
            "lost_batching_efficiency_pct": self.lost_batching_efficiency_pct,
            "network_transfer_cost": self.network_transfer_cost,
            "topology_degradation_score": self.topology_degradation_score,
            "sla_risk_penalty": self.sla_risk_penalty,
            "thermal_penalty": self.thermal_penalty,
            "failure_retry_penalty": self.failure_retry_penalty,
            "total_penalty": self.total_penalty,
            "net_expected_savings": self.net_expected_savings,
            "confidence": self.confidence,
            "blocked_by_cooldown": self.blocked_by_cooldown,
            "blocked_reason": self.blocked_reason,
            "explanation": self.explanation,
        }


# ---------------------------------------------------------------------------
# Migration governor (hysteresis / rate limiting)
# ---------------------------------------------------------------------------

@dataclass
class MigrationGovernor:
    """Prevents trigger-happy migration optimization.

    Tracks per-workload migration history and cluster-wide migration rate.
    All governance checks are in-memory; production deployments should
    persist migration history to the Postgres store (Phase 11/12).
    """

    config: CostModelConfig = field(default_factory=CostModelConfig)

    # workload_id → list of migration timestamps (most recent last)
    _workload_history: dict[str, list[datetime]] = field(default_factory=dict, repr=False)
    # cluster-wide migration timestamps
    _cluster_history: list[datetime] = field(default_factory=list, repr=False)
    # workload_id → timestamp of last SLA violation
    _sla_violation_times: dict[str, datetime] = field(default_factory=dict, repr=False)

    def record_migration(self, workload_id: str, ts: datetime) -> None:
        """Record that a migration was executed for the given workload."""
        if workload_id not in self._workload_history:
            self._workload_history[workload_id] = []
        self._workload_history[workload_id].append(ts)
        self._cluster_history.append(ts)

    def record_sla_violation(self, workload_id: str, ts: datetime) -> None:
        """Record a SLA violation so the governor enforces cooldown."""
        self._sla_violation_times[workload_id] = ts

    def check_allowed(self, workload_id: str, now: datetime) -> tuple[bool, Optional[str]]:
        """Return (allowed, reason_if_blocked).

        Checks:
        1. Per-workload minimum interval since last migration
        2. Per-workload rate limit (max N migrations per hour)
        3. Cluster-wide migration rate limit
        4. SLA violation cooldown for this workload
        """
        cfg = self.config

        # SLA violation cooldown
        if workload_id in self._sla_violation_times:
            last_violation = self._sla_violation_times[workload_id]
            age_s = (now - last_violation).total_seconds()
            if age_s < cfg.sla_violation_cooldown_s:
                remaining = cfg.sla_violation_cooldown_s - age_s
                return False, f"SLA violation cooldown: {remaining:.0f}s remaining"

        # Per-workload minimum interval
        wl_history = self._workload_history.get(workload_id, [])
        if wl_history:
            last_migration = wl_history[-1]
            age_s = (now - last_migration).total_seconds()
            if age_s < cfg.min_migration_interval_s:
                remaining = cfg.min_migration_interval_s - age_s
                return False, f"Minimum migration interval: {remaining:.0f}s remaining"

        # Per-workload hourly rate
        one_hour_ago = now - timedelta(hours=1)
        recent_wl = [t for t in wl_history if t > one_hour_ago]
        if len(recent_wl) >= cfg.max_migrations_per_workload_per_hour:
            return False, (
                f"Workload migration rate limit: {len(recent_wl)} migrations in last hour "
                f"(max {cfg.max_migrations_per_workload_per_hour})"
            )

        # Cluster-wide rate limit (per minute)
        one_minute_ago = now - timedelta(minutes=1)
        recent_cluster = [t for t in self._cluster_history if t > one_minute_ago]
        if len(recent_cluster) >= cfg.max_cluster_migrations_per_minute:
            return False, (
                f"Cluster migration rate limit: {len(recent_cluster)} migrations in last minute "
                f"(max {cfg.max_cluster_migrations_per_minute})"
            )

        return True, None

    def reset(self) -> None:
        """Clear all history (e.g. for a new simulation run)."""
        self._workload_history.clear()
        self._cluster_history.clear()
        self._sla_violation_times.clear()


# ---------------------------------------------------------------------------
# Migration cost model
# ---------------------------------------------------------------------------

class MigrationCostModel:
    """Estimates total cost impact of a candidate action including operational penalties.

    Conservative heuristics are used when no trained predictor is available.
    The model never trains on or returns real customer data.

    Usage:
        model = MigrationCostModel(config=CostModelConfig())
        estimate = model.estimate(
            workload_id="llm-service-1",
            action_type=ActionType.MIGRATE,
            assessment=constraint_assessment,
            state=cluster_state,
            gross_savings=5.0,
            is_latency_sensitive=True,
            priority_tier="critical",
            current_topology_score=0.9,
            target_topology_score=0.5,
        )
        if estimate.is_viable():
            # emit recommendation
    """

    def __init__(
        self,
        config: Optional[CostModelConfig] = None,
        governor: Optional[MigrationGovernor] = None,
    ) -> None:
        self.config = config or CostModelConfig()
        self.governor = governor or MigrationGovernor(config=self.config)

    def estimate(
        self,
        workload_id: str,
        action_type: str,
        assessment: ConstraintAssessment,
        state: ClusterState,
        gross_savings: Optional[float] = None,
        is_latency_sensitive: bool = False,
        priority_tier: str = "standard",
        current_topology_score: Optional[float] = None,
        target_topology_score: Optional[float] = None,
        now: Optional[datetime] = None,
    ) -> MigrationCostEstimate:
        """Produce a conservative cost/risk estimate for a candidate action.

        Parameters
        ----------
        workload_id: Workload being considered for the action.
        action_type: The action type string (ActionType.value).
        assessment: Current ConstraintAssessment from the classifier.
        state: Current ClusterState snapshot.
        gross_savings: Gross expected savings before penalties (cost units).
            None means savings are unknown; estimate will have low confidence.
        is_latency_sensitive: True for latency-sensitive/critical inference workloads.
        priority_tier: "critical" | "latency_sensitive" | "standard" | "batch" | "flexible".
        current_topology_score: Topology score for current placement (0-1, higher=better).
        target_topology_score: Topology score for proposed placement (0-1).
        now: Override current time (useful for testing).
        """
        cfg = self.config
        ts = state.timestamp
        if now is None:
            now = datetime.now(tz=timezone.utc)

        is_migration = action_type in {a.value for a in MIGRATION_ACTIONS}
        is_critical = is_latency_sensitive or priority_tier in ("critical", "latency_sensitive")
        is_batch = priority_tier in ("batch", "flexible")

        # 1. Governance check (cooldown, rate limits)
        if is_migration:
            allowed, block_reason = self.governor.check_allowed(workload_id, now)
            if not allowed:
                return MigrationCostEstimate(
                    workload_id=workload_id,
                    action_type=action_type,
                    timestamp=ts,
                    gross_energy_savings=gross_savings,
                    net_expected_savings=None,
                    confidence=0.0,
                    blocked_by_cooldown=True,
                    blocked_reason=block_reason,
                    explanation=f"Blocked by migration governor: {block_reason}",
                )

        # 2. Determine workload risk multiplier
        if is_critical:
            risk_mult = cfg.critical_workload_risk_multiplier
        elif is_batch:
            risk_mult = cfg.batch_workload_risk_multiplier
        else:
            risk_mult = 1.0

        # 3. Cold-start + cache warmup penalties
        cold_start_ms = 0.0
        cache_warmup_ms = 0.0
        if is_migration:
            cold_start_ms = cfg.cold_start_p99_penalty_ms * risk_mult
            # Cache warmup: fraction of hit rate lost = extra latency
            cache_warmup_ms = cfg.cache_warmup_hit_rate_loss * cfg.cold_start_p99_penalty_ms * risk_mult

        # 4. Queue instability during migration
        queue_penalty_ms = 0.0
        if is_migration:
            queue_penalty_ms = cfg.queue_instability_penalty_ms * risk_mult

        # 5. Topology degradation
        topo_score = 0.0
        if current_topology_score is not None and target_topology_score is not None:
            delta = current_topology_score - target_topology_score
            if delta > 0:
                # Moving to worse topology: score proportional to degradation
                topo_score = delta  # [0, 1]

        # 6. Batching efficiency loss from migration disruption
        lost_batching_pct = 0.0
        if is_migration:
            lost_batching_pct = 5.0 * risk_mult  # HEURISTIC: 5-12.5% efficiency loss during migration

        # 7. SLA risk penalty
        # Latency-sensitive workloads during latency-bound or queue-bound constraint = extra risk
        sla_risk = 0.0
        binding = assessment.binding_constraint
        if is_critical and binding in (ConstraintType.LATENCY, ConstraintType.QUEUE, ConstraintType.MEMORY):
            # Migrating during an active latency/queue/memory constraint is especially risky
            sla_risk = cold_start_ms * 0.5  # HEURISTIC: 50% of cold-start as SLA risk cost
        elif is_critical and binding is not None:
            sla_risk = cold_start_ms * 0.2  # HEURISTIC: any active constraint adds some risk

        # 8. Thermal penalty
        thermal_penalty = 0.0
        if binding == ConstraintType.THERMAL and action_type == ActionType.CONSOLIDATE.value:
            # Consolidating during thermal constraint is especially harmful
            thermal_penalty = cold_start_ms * 0.3  # HEURISTIC

        # 9. Failure/retry penalty
        failure_retry_penalty = cold_start_ms * 0.1  # HEURISTIC: 10% additional for retry risk

        # 10. Network transfer cost (rough proportional estimate)
        network_cost = 0.0
        if is_migration and topo_score > 0:
            network_cost = topo_score * 0.5  # HEURISTIC: topology fraction of gross savings

        # 11. Aggregate total penalty
        total_penalty = (
            cold_start_ms * 0.001     # convert ms to abstract cost units (HEURISTIC scaling)
            + cache_warmup_ms * 0.001
            + queue_penalty_ms * 0.001
            + topo_score * (gross_savings or 0.0) * cfg.topology_degradation_fraction
            + sla_risk * 0.001
            + thermal_penalty * 0.001
            + failure_retry_penalty * 0.001
            + network_cost
        )

        # 12. Net expected savings
        gross = gross_savings if gross_savings is not None else 0.0
        net = gross - total_penalty
        if gross_savings is None:
            net_value = None
            conf = 0.2  # HEURISTIC: low confidence when gross savings unknown
        else:
            net_value = net
            # Confidence: driven by assessment confidence + whether we have all key signals
            conf = assessment.confidence * (0.9 if current_topology_score is not None else 0.7)

        # 13. Build explanation
        parts = []
        parts.append(f"Action: {action_type} for workload {workload_id}.")
        if gross_savings is not None:
            parts.append(f"Gross savings: {gross:.3f}. Total penalty: {total_penalty:.3f}. Net: {net:.3f}.")
        if is_critical:
            parts.append(f"Critical workload — risk multiplier {risk_mult:.1f}×.")
        if cold_start_ms > 0:
            parts.append(f"Cold-start p99 penalty: {cold_start_ms:.0f}ms.")
        if binding is not None:
            parts.append(f"Active binding constraint: {binding.value}.")
        if net_value is not None and net_value <= 0:
            parts.append("Net savings non-positive — KEEP recommended.")

        explanation = " ".join(parts)

        return MigrationCostEstimate(
            workload_id=workload_id,
            action_type=action_type,
            timestamp=ts,
            gross_energy_savings=gross_savings,
            gross_compute_savings=None,
            cold_start_penalty_ms=cold_start_ms,
            cache_warmup_penalty_ms=cache_warmup_ms,
            queue_instability_penalty_ms=queue_penalty_ms,
            lost_batching_efficiency_pct=lost_batching_pct,
            network_transfer_cost=network_cost,
            topology_degradation_score=topo_score,
            sla_risk_penalty=sla_risk,
            thermal_penalty=thermal_penalty,
            failure_retry_penalty=failure_retry_penalty,
            total_penalty=total_penalty,
            net_expected_savings=net_value,
            confidence=min(1.0, max(0.0, conf)),
            blocked_by_cooldown=False,
            blocked_reason=None,
            explanation=explanation,
        )

    def should_keep(
        self,
        estimate: MigrationCostEstimate,
        cfg: Optional[CostModelConfig] = None,
    ) -> tuple[bool, str]:
        """Return (should_keep, reason) — True when the action should be a no-op.

        A KEEP is recommended when:
        - blocked_by_cooldown
        - net_expected_savings is None (unknown gross savings)
        - net_expected_savings <= min_net_savings_threshold
        - confidence is very low
        """
        if cfg is None:
            cfg = self.config

        if estimate.blocked_by_cooldown:
            return True, estimate.blocked_reason or "migration governor blocked"

        if estimate.net_expected_savings is None:
            return True, "gross savings unknown; cannot estimate net benefit"

        if estimate.net_expected_savings <= cfg.min_net_savings_threshold:
            return True, (
                f"net savings {estimate.net_expected_savings:.3f} ≤ threshold "
                f"{cfg.min_net_savings_threshold:.3f}"
            )

        if estimate.confidence < 0.15:  # HEURISTIC
            return True, f"confidence {estimate.confidence:.3f} too low to recommend action"

        return False, "action passes cost/benefit threshold"

    def make_recommendation(
        self,
        workload_id: str,
        action_type: str,
        estimate: MigrationCostEstimate,
        assessment: ConstraintAssessment,
        provenance: Optional[Provenance] = None,
        recommendation_id: Optional[str] = None,
    ) -> Recommendation:
        """Convert a cost estimate into a Recommendation.

        Always emits in recommendation_only mode (Phase 9 wires execution).
        """
        import uuid
        keep, keep_reason = self.should_keep(estimate)
        if keep:
            final_action = ActionType.KEEP.value
            sla_status = "unknown"
            is_noop = True
            rationale = f"KEEP: {keep_reason}. {estimate.explanation}"
        else:
            final_action = action_type
            sla_status = "satisfied"
            is_noop = False
            rationale = estimate.explanation

        prov = provenance or Provenance(
            source="migration-cost-model",
            fetched_at=estimate.timestamp,
            confidence=(
                "high" if estimate.confidence >= 0.7
                else "medium" if estimate.confidence >= 0.4
                else "low"
            ),
            is_sandbox=assessment.provenance.is_sandbox,
        )

        return Recommendation(
            recommendation_id=recommendation_id or str(uuid.uuid4()),
            workload_id=workload_id,
            action_type=final_action,
            timestamp=estimate.timestamp,
            provenance=prov,
            binding_constraint=assessment.binding_constraint,
            expected_effect={
                "gross_savings": estimate.gross_energy_savings or 0.0,
                "total_penalty": estimate.total_penalty,
            },
            confidence=estimate.confidence,
            sla_status=sla_status,
            migration_penalty=estimate.total_penalty,
            net_benefit=estimate.net_expected_savings,
            rationale=rationale,
            is_noop=is_noop,
            implementation_mode="recommendation_only",
        )
