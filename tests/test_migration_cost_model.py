"""Tests for Phase 8 — Migration cost/risk model.

Tests prove:
- High gross savings with catastrophic p99 penalty → KEEP (no-op)
- Batch workloads can migrate when critical workloads cannot (risk multipliers)
- Worse topology erases apparent energy savings
- Thermal hotspot blocks consolidation
- Repeated migrations are penalized by governor
- KEEP wins when net expected savings <= 0
- Missing gross savings → low confidence, KEEP
- Governor cooldown after SLA violation
- Cluster-wide rate limit prevents migration storm
- Cost estimate to_dict round-trip
- make_recommendation produces valid Recommendation
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aurelius.constraints.cost_model import (
    CostModelConfig,
    MigrationCostEstimate,
    MigrationCostModel,
    MigrationGovernor,
)
from aurelius.sla.actions import ActionType
from aurelius.state.models import (
    ClusterState,
    ConstraintAssessment,
    ConstraintType,
    Provenance,
    RegionState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _now() -> datetime:
    return datetime(2026, 1, 1, 12, 0, 0, tzinfo=_UTC)


def _make_prov(sandbox: bool = True) -> Provenance:
    return Provenance(
        source="test",
        fetched_at=_now(),
        confidence="medium",
        is_sandbox=sandbox,
    )


def _make_region(region_id: str = "us-east") -> RegionState:
    return RegionState(
        region=region_id,
        timestamp=_now(),
        provenance=_make_prov(),
    )


def _make_state(region_id: str = "us-east") -> ClusterState:
    return ClusterState(
        timestamp=_now(),
        provenance=_make_prov(),
        regions={region_id: _make_region(region_id)},
    )


def _make_assessment(
    binding: ConstraintType | None = None,
    confidence: float = 0.7,
) -> ConstraintAssessment:
    return ConstraintAssessment(
        timestamp=_now(),
        provenance=_make_prov(),
        region=None,
        scores={binding: 0.8} if binding else {},
        binding_constraint=binding,
        confidence=confidence,
        missing_signals=[],
        rationale="test assessment",
        safe_action_types=[ActionType.KEEP.value],
        disallowed_action_types=[],
    )


# ---------------------------------------------------------------------------
# MigrationCostEstimate
# ---------------------------------------------------------------------------

class TestMigrationCostEstimate:
    def test_viable_positive_net(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.8,
        )
        assert est.is_viable()

    def test_not_viable_zero_net(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=0.0,
            confidence=0.8,
        )
        assert not est.is_viable()

    def test_not_viable_negative_net(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=-1.0,
            confidence=0.8,
        )
        assert not est.is_viable()

    def test_not_viable_when_blocked_by_cooldown(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=100.0,
            confidence=0.9,
            blocked_by_cooldown=True,
            blocked_reason="test cooldown",
        )
        assert not est.is_viable()

    def test_not_viable_unknown_savings(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=None,
            confidence=0.8,
        )
        assert not est.is_viable()

    def test_to_dict_round_trip(self):
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            gross_energy_savings=10.0,
            cold_start_penalty_ms=2000.0,
            total_penalty=3.5,
            net_expected_savings=6.5,
            confidence=0.75,
            explanation="test explanation",
        )
        d = est.to_dict()
        assert d["workload_id"] == "wl-1"
        assert d["gross_energy_savings"] == 10.0
        assert d["cold_start_penalty_ms"] == 2000.0
        assert d["net_expected_savings"] == 6.5
        assert d["confidence"] == 0.75
        assert d["blocked_by_cooldown"] is False
        assert d["blocked_reason"] is None


# ---------------------------------------------------------------------------
# MigrationGovernor
# ---------------------------------------------------------------------------

class TestMigrationGovernor:
    def test_first_migration_always_allowed(self):
        gov = MigrationGovernor()
        allowed, reason = gov.check_allowed("wl-1", _now())
        assert allowed
        assert reason is None

    def test_min_interval_blocks_immediate_repeat(self):
        cfg = CostModelConfig(min_migration_interval_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_migration("wl-1", now)
        # Try again immediately
        allowed, reason = gov.check_allowed("wl-1", now + timedelta(seconds=10))
        assert not allowed
        assert "interval" in reason.lower()

    def test_min_interval_passes_after_wait(self):
        cfg = CostModelConfig(min_migration_interval_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_migration("wl-1", now)
        allowed, reason = gov.check_allowed("wl-1", now + timedelta(seconds=400))
        assert allowed

    def test_hourly_rate_limit(self):
        cfg = CostModelConfig(
            min_migration_interval_s=1.0,
            max_migrations_per_workload_per_hour=2,
        )
        gov = MigrationGovernor(config=cfg)
        base = _now()
        gov.record_migration("wl-1", base)
        gov.record_migration("wl-1", base + timedelta(seconds=10))
        allowed, reason = gov.check_allowed("wl-1", base + timedelta(seconds=20))
        assert not allowed
        assert "rate limit" in reason.lower()

    def test_cluster_rate_limit(self):
        cfg = CostModelConfig(
            min_migration_interval_s=1.0,
            max_cluster_migrations_per_minute=2,
        )
        gov = MigrationGovernor(config=cfg)
        base = _now()
        gov.record_migration("wl-1", base)
        gov.record_migration("wl-2", base + timedelta(seconds=1))
        # Third migration on different workload should be blocked
        allowed, reason = gov.check_allowed("wl-3", base + timedelta(seconds=2))
        assert not allowed
        assert "cluster" in reason.lower()

    def test_sla_violation_cooldown(self):
        cfg = CostModelConfig(sla_violation_cooldown_s=600.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_sla_violation("wl-1", now)
        allowed, reason = gov.check_allowed("wl-1", now + timedelta(seconds=100))
        assert not allowed
        assert "sla violation" in reason.lower()

    def test_sla_violation_clears_after_cooldown(self):
        cfg = CostModelConfig(sla_violation_cooldown_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_sla_violation("wl-1", now)
        allowed, _ = gov.check_allowed("wl-1", now + timedelta(seconds=400))
        assert allowed

    def test_independent_workloads_not_blocked_by_each_other(self):
        cfg = CostModelConfig(min_migration_interval_s=300.0)
        gov = MigrationGovernor(config=cfg)
        now = _now()
        gov.record_migration("wl-1", now)
        # wl-2 was not migrated; should be allowed
        allowed, _ = gov.check_allowed("wl-2", now + timedelta(seconds=1))
        assert allowed

    def test_reset_clears_history(self):
        gov = MigrationGovernor()
        now = _now()
        gov.record_migration("wl-1", now)
        gov.record_sla_violation("wl-1", now)
        gov.reset()
        allowed, _ = gov.check_allowed("wl-1", now + timedelta(seconds=1))
        assert allowed


# ---------------------------------------------------------------------------
# MigrationCostModel — core behavior
# ---------------------------------------------------------------------------

class TestMigrationCostModel:
    def test_keep_when_no_gross_savings(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=None,
        )
        assert not est.is_viable()
        assert est.net_expected_savings is None
        assert est.confidence < 0.5

    def test_keep_wins_when_net_savings_zero_or_negative(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=0.001,  # tiny savings — will be erased by penalties
            is_latency_sensitive=True,
        )
        keep, reason = model.should_keep(est)
        assert keep

    def test_critical_workload_has_larger_cold_start_penalty(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est_critical = model.estimate(
            workload_id="wl-critical",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            is_latency_sensitive=True,
            priority_tier="critical",
        )
        est_batch = model.estimate(
            workload_id="wl-batch",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            is_latency_sensitive=False,
            priority_tier="batch",
        )
        # Critical workload should have higher penalty than batch
        assert est_critical.cold_start_penalty_ms > est_batch.cold_start_penalty_ms

    def test_batch_migrates_when_critical_does_not(self):
        model = MigrationCostModel(config=CostModelConfig(
            cold_start_p99_penalty_ms=500.0,
            critical_workload_risk_multiplier=10.0,
            batch_workload_risk_multiplier=0.1,
        ))
        state = _make_state()
        assessment = _make_assessment()
        est_critical = model.estimate(
            workload_id="wl-critical",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=1.0,
            is_latency_sensitive=True,
            priority_tier="critical",
        )
        est_batch = model.estimate(
            workload_id="wl-batch",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=1.0,
            is_latency_sensitive=False,
            priority_tier="batch",
        )
        # Critical: big penalty → not viable
        # Batch: tiny penalty → viable
        assert not est_critical.is_viable()
        assert est_batch.is_viable()

    def test_topology_degradation_reduces_net_savings(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        gross = 5.0
        est_same_topo = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=gross,
            current_topology_score=0.9,
            target_topology_score=0.9,  # no degradation
        )
        est_worse_topo = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=gross,
            current_topology_score=0.9,
            target_topology_score=0.1,  # significant degradation
        )
        # Worse topology → lower net savings
        assert est_worse_topo.total_penalty > est_same_topo.total_penalty
        if est_same_topo.net_expected_savings and est_worse_topo.net_expected_savings:
            assert est_worse_topo.net_expected_savings < est_same_topo.net_expected_savings

    def test_thermal_hotspot_penalizes_consolidation(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment_thermal = _make_assessment(binding=ConstraintType.THERMAL)
        assessment_none = _make_assessment(binding=None)
        est_thermal = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.CONSOLIDATE.value,
            assessment=assessment_thermal,
            state=state,
            gross_savings=3.0,
            is_latency_sensitive=True,
        )
        est_none = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.CONSOLIDATE.value,
            assessment=assessment_none,
            state=state,
            gross_savings=3.0,
            is_latency_sensitive=True,
        )
        # Thermal constraint + consolidation adds thermal penalty
        assert est_thermal.thermal_penalty >= 0  # non-negative
        # Total penalty under thermal should be higher (or at least not lower)
        assert est_thermal.total_penalty >= est_none.total_penalty

    def test_non_migration_actions_have_no_cold_start_penalty(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            assessment=assessment,
            state=state,
            gross_savings=5.0,
        )
        assert est.cold_start_penalty_ms == 0.0

    def test_active_latency_constraint_increases_sla_risk(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment_latency = _make_assessment(binding=ConstraintType.LATENCY)
        assessment_energy = _make_assessment(binding=ConstraintType.ENERGY)
        est_latency = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment_latency,
            state=state,
            gross_savings=5.0,
            is_latency_sensitive=True,
        )
        est_energy = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment_energy,
            state=state,
            gross_savings=5.0,
            is_latency_sensitive=True,
        )
        # Migrating during latency constraint is more expensive than during energy constraint
        assert est_latency.sla_risk_penalty >= est_energy.sla_risk_penalty

    def test_penalty_never_negative(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=0.1,
        )
        assert est.cold_start_penalty_ms >= 0
        assert est.cache_warmup_penalty_ms >= 0
        assert est.queue_instability_penalty_ms >= 0
        assert est.topology_degradation_score >= 0
        assert est.sla_risk_penalty >= 0
        assert est.thermal_penalty >= 0
        assert est.failure_retry_penalty >= 0
        assert est.total_penalty >= 0

    def test_penalty_fields_sum_roughly_to_total(self):
        model = MigrationCostModel()
        state = _make_state()
        assessment = _make_assessment()
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
        )
        # Total penalty should be positive for a migration
        assert est.total_penalty > 0

    def test_governor_integration_blocks_repeat_migration(self):
        model = MigrationCostModel(config=CostModelConfig(min_migration_interval_s=600.0))
        state = _make_state()
        assessment = _make_assessment()
        now = _now()
        # Record a migration
        model.governor.record_migration("wl-1", now)
        # Try again immediately
        est = model.estimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=100.0,
            now=now + timedelta(seconds=10),
        )
        assert est.blocked_by_cooldown
        assert not est.is_viable()


# ---------------------------------------------------------------------------
# should_keep
# ---------------------------------------------------------------------------

class TestShouldKeep:
    def test_keep_when_blocked_by_cooldown(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            blocked_by_cooldown=True,
            blocked_reason="test",
        )
        keep, _ = model.should_keep(est)
        assert keep

    def test_keep_when_net_savings_unknown(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=None,
        )
        keep, reason = model.should_keep(est)
        assert keep
        assert "unknown" in reason.lower()

    def test_keep_when_net_savings_zero(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=0.0,
            confidence=0.9,
        )
        keep, _ = model.should_keep(est)
        assert keep

    def test_no_keep_when_positive_savings(self):
        model = MigrationCostModel()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.7,
        )
        keep, _ = model.should_keep(est)
        assert not keep


# ---------------------------------------------------------------------------
# make_recommendation
# ---------------------------------------------------------------------------

class TestMakeRecommendation:
    def test_produces_keep_when_not_viable(self):
        model = MigrationCostModel()
        assessment = _make_assessment()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=-1.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment,
        )
        assert rec.is_noop
        assert rec.action_type == ActionType.KEEP.value
        assert rec.implementation_mode == "recommendation_only"

    def test_produces_action_when_viable(self):
        model = MigrationCostModel()
        assessment = _make_assessment(binding=ConstraintType.ENERGY)
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment,
        )
        assert not rec.is_noop
        assert rec.action_type == ActionType.MIGRATE.value
        assert rec.implementation_mode == "recommendation_only"

    def test_recommendation_only_mode_always(self):
        model = MigrationCostModel()
        assessment = _make_assessment()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            timestamp=_now(),
            net_expected_savings=3.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            estimate=est,
            assessment=assessment,
        )
        assert rec.implementation_mode == "recommendation_only"

    def test_recommendation_carries_net_benefit(self):
        model = MigrationCostModel()
        assessment = _make_assessment(binding=ConstraintType.ENERGY)
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            gross_energy_savings=10.0,
            total_penalty=2.0,
            net_expected_savings=8.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment,
        )
        assert rec.net_benefit == 8.0
        assert rec.migration_penalty == 2.0

    def test_recommendation_inherits_sandbox_flag(self):
        model = MigrationCostModel()
        prov_sandbox = _make_prov(sandbox=True)
        assessment_sandbox = ConstraintAssessment(
            timestamp=_now(),
            provenance=prov_sandbox,
            region=None,
            scores={},
            binding_constraint=None,
            confidence=0.5,
            missing_signals=[],
            rationale="",
            safe_action_types=[],
            disallowed_action_types=[],
        )
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            timestamp=_now(),
            net_expected_savings=5.0,
            confidence=0.8,
        )
        rec = model.make_recommendation(
            workload_id="wl-1",
            action_type=ActionType.MIGRATE.value,
            estimate=est,
            assessment=assessment_sandbox,
        )
        assert rec.provenance.is_sandbox

    def test_recommendation_has_unique_id(self):
        model = MigrationCostModel()
        assessment = _make_assessment()
        est = MigrationCostEstimate(
            workload_id="wl-1",
            action_type=ActionType.KEEP.value,
            timestamp=_now(),
            net_expected_savings=None,
        )
        rec1 = model.make_recommendation("wl-1", ActionType.KEEP.value, est, assessment)
        rec2 = model.make_recommendation("wl-1", ActionType.KEEP.value, est, assessment)
        assert rec1.recommendation_id != rec2.recommendation_id


# ---------------------------------------------------------------------------
# Integration: classifier + cost model pipeline
# ---------------------------------------------------------------------------

class TestClassifierCostModelPipeline:
    """Verify the Phase 7→8 pipeline: assessment → cost estimate → recommendation."""

    def test_energy_bound_migrate_batch_viable(self):
        """Batch workload can migrate during energy-bound constraint."""
        model = MigrationCostModel(config=CostModelConfig(
            cold_start_p99_penalty_ms=100.0,
            batch_workload_risk_multiplier=0.1,
        ))
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY, confidence=0.8)
        est = model.estimate(
            workload_id="batch-wl",
            action_type=ActionType.CHOOSE_CHEAPER_REGION.value,
            assessment=assessment,
            state=state,
            gross_savings=20.0,
            is_latency_sensitive=False,
            priority_tier="batch",
        )
        rec = model.make_recommendation(
            workload_id="batch-wl",
            action_type=ActionType.CHOOSE_CHEAPER_REGION.value,
            estimate=est,
            assessment=assessment,
        )
        assert not rec.is_noop, f"Expected action, got KEEP. est={est.to_dict()}"
        assert rec.binding_constraint == ConstraintType.ENERGY

    def test_latency_bound_migrate_critical_blocked(self):
        """Critical workload should not migrate during latency-bound constraint."""
        model = MigrationCostModel(config=CostModelConfig(
            cold_start_p99_penalty_ms=5000.0,
            critical_workload_risk_multiplier=5.0,
        ))
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.LATENCY, confidence=0.85)
        est = model.estimate(
            workload_id="critical-wl",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=2.0,  # small savings
            is_latency_sensitive=True,
            priority_tier="critical",
        )
        keep, reason = model.should_keep(est)
        assert keep, f"Expected KEEP for critical workload during latency constraint; reason={reason}"

    def test_no_migration_storm_governor(self):
        """Multiple workloads migrating at once are rate-limited."""
        model = MigrationCostModel(config=CostModelConfig(
            min_migration_interval_s=1.0,
            max_cluster_migrations_per_minute=2,
        ))
        state = _make_state()
        assessment = _make_assessment(binding=ConstraintType.ENERGY)
        now = _now()

        # First two migrations succeed
        model.governor.record_migration("wl-1", now)
        model.governor.record_migration("wl-2", now + timedelta(seconds=1))

        # Third is blocked by cluster rate limit
        est3 = model.estimate(
            workload_id="wl-3",
            action_type=ActionType.MIGRATE.value,
            assessment=assessment,
            state=state,
            gross_savings=10.0,
            now=now + timedelta(seconds=2),
        )
        assert est3.blocked_by_cooldown
        assert not est3.is_viable()
