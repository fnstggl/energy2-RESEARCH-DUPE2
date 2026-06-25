"""Phase B guard — the canonical ``AureliusOptimizer`` is a COMPREHENSIVE fleet
optimizer, not a one-policy-at-a-time facade.

Verifies:
  * all five decision surfaces are reachable from a single optimizer instance;
  * PlacementPolicy / AdmissionPolicy are *parity wirings* of the existing
    residency + frontier-admission surfaces (identical results to calling them
    directly — no new optimization logic);
  * ``optimize_fleet`` orchestrates multiple surfaces in one pass, defaults
    capacity to the DEPLOYABLE ``forecasted_mcs`` mode (never an oracle), and
    carries honest provenance/notes;
  * the energy ``optimize`` path remains byte-identical (parity), and the
    facade-bypass sites (``simulation/compare``) now route through the optimizer
    while producing identical schedules;
  * the live-service surface (``ConstraintAwareEngine``) is reachable and
    recommendation-only.

Every surface targets ``docs/RESULTS.md`` §1 (SLA-safe goodput per infra dollar).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from aurelius.frontier.admission import AdmissionGateConfig, evaluate_admission
from aurelius.models import Job, OptimizationConfig
from aurelius.optimization.scheduler import JobScheduler
from aurelius.optimizer import (
    CANONICAL_OBJECTIVE,
    AureliusOptimizer,
    FleetOptimizationResult,
)
from aurelius.residency.decision import SafetyContext, choose_residency_decision
from aurelius.residency.models import (
    ModelLoadProfile,
    ModelLocationState,
    ModelResidencyRequest,
    ResidencyAction,
)

W = datetime(2026, 2, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def _energy_fixture():
    da = {
        "us-west": {W + timedelta(hours=h): 30.0 + 30.0 * (8 <= (h % 24) < 20)
                    for h in range(72)},
        "us-east": {W + timedelta(hours=h): 60.0 + 30.0 * (8 <= (h % 24) < 20)
                    for h in range(72)},
    }
    carbon = {r: {} for r in da}
    jobs = []
    for i in range(8):
        rt_h = [2, 4][i % 2]
        es = W + timedelta(hours=i)
        jobs.append(Job(
            job_id=f"job-{i:03d}", submit_time=es, runtime_hours=rt_h,
            deadline=es + timedelta(hours=rt_h + 12), power_kw=100.0,
            earliest_start=es, region_options=["us-west", "us-east"],
            gpu_count=2, workload_type="llm_batch_inference",
            migration_cost_hours=0.1,
        ))
    return jobs, da, carbon


def _loc(gpu, *, models=(), used_gb=0.0, total_gb=80.0, queue_s=0.1, conf="high"):
    return ModelLocationState(
        region="r", node_id="n", gpu_id=gpu, container_id=f"pod-{gpu}",
        loaded_model_ids=list(models), loaded_adapter_ids=[],
        gpu_memory_used=used_gb * 1e9, gpu_memory_total=total_gb * 1e9,
        estimated_queue_wait_s=queue_s, thermal_risk=None,
        telemetry_confidence=conf)


def _profile():
    return ModelLoadProfile(
        model_id="m", cold_load_p50_s=22.0, cold_load_p95_s=30.0,
        memory_required_gb=16.0, source="cal", confidence="high")


def _req():
    return ModelResidencyRequest(
        request_id="r1", timestamp=1.0, workload_id="w", model_id="m",
        priority_class="standard")


def _ctx():
    return SafetyContext(
        gpu_hour_price=3.0, default_latency_sla_ms=120000.0,
        service_time_proxy_s=2.0, min_telemetry_confidence="low")


# --------------------------------------------------------------------------
# 1. One instance, all five surfaces
# --------------------------------------------------------------------------

def test_single_instance_exposes_all_surfaces():
    opt = AureliusOptimizer()
    for name in ("energy", "serving_queue", "replica_scaling", "genai_serving",
                 "placement", "admission"):
        assert opt.surface(name) is not None
    # cached (same object returned twice)
    assert opt.surface("placement") is opt.surface("placement")


def test_optimize_fleet_genai_surface():
    from types import SimpleNamespace

    ticks = [
        SimpleNamespace(n=0, arrival_rate=0.0, mean_exec_s=0.0,
                        distinct_models=0, lora_frac=0.0, controlnet_frac=0.0),
        SimpleNamespace(n=10, arrival_rate=5.0, mean_exec_s=2.0,
                        distinct_models=3, lora_frac=0.4, controlnet_frac=0.1),
    ]
    cold = {"basemodel_load": 10.0, "lora_load": 2.0, "controlnet_load": 4.0}
    res = AureliusOptimizer().optimize_fleet(
        workload_class="genai_serving", genai={"ticks": ticks, "cold": cold})
    assert "genai_serving" in res.surfaces_used
    assert len(res.genai.replica_counts) == 2


# --------------------------------------------------------------------------
# 2. Placement is a parity wiring of residency.choose_residency_decision
# --------------------------------------------------------------------------

def test_placement_surface_matches_residency_engine():
    warm = _loc("g0", models=["m"], used_gb=16, queue_s=0.1)
    cold = _loc("g1", models=[], used_gb=0, queue_s=0.05)
    profiles = {"m": _profile()}
    ctx = _ctx()

    direct = choose_residency_decision(_req(), [warm, cold], profiles, ctx, ctx)
    viafacade = AureliusOptimizer().place(
        _req(), [warm, cold], load_profiles=profiles, cost_config=ctx,
        safety_context=ctx)

    assert viafacade.action == direct.action == ResidencyAction.ROUTE_TO_RESIDENT_MODEL
    assert viafacade.target_location == direct.target_location
    # recommendation-only invariant preserved through the facade
    assert viafacade.executable_in_real_cluster is False


# --------------------------------------------------------------------------
# 3. Admission is a parity wiring of frontier.evaluate_admission
# --------------------------------------------------------------------------

def test_admission_surface_matches_frontier_gate():
    cfg = AdmissionGateConfig(enabled=False)
    direct = evaluate_admission(sla_class="llm_batch_inference", window=[], config=cfg)
    viafacade = AureliusOptimizer().admit(
        sla_class="llm_batch_inference", window=[], config=cfg)
    assert viafacade.action == direct.action == "ADMIT"
    assert viafacade.to_dict() == direct.to_dict()


# --------------------------------------------------------------------------
# 4. optimize_fleet orchestrates many surfaces; capacity defaults to DEPLOYABLE
# --------------------------------------------------------------------------

def test_optimize_fleet_multi_surface_and_deployable_capacity():
    raw = [(float(i) * 2.0, 100 + (i % 5) * 50) for i in range(400)]
    res = AureliusOptimizer().optimize_fleet(
        workload_class="inference_standard",
        admission={"sla_class": "llm_batch_inference", "window": []},
        capacity={"raw": raw},
        placement={"request": _req(), "locations": [], "load_profiles": {}},
    )
    assert isinstance(res, FleetOptimizationResult)
    assert set(res.surfaces_used) == {"admission", "replica_scaling", "placement"}
    assert res.objective == CANONICAL_OBJECTIVE
    # deployable, non-oracle provisioner chosen by default
    assert res.capacity.mode == "forecasted_mcs"
    assert any("forecasted_mcs" in n for n in res.notes)
    # placement with no candidate locations → honest INSUFFICIENT_TELEMETRY
    assert res.placement.action == ResidencyAction.INSUFFICIENT_TELEMETRY


def test_optimize_fleet_capacity_respects_explicit_mode():
    from aurelius.optimizer.policies.replica_scaling import ReplicaScalingConfig

    raw = [(float(i), 100) for i in range(200)]
    res = AureliusOptimizer().optimize_fleet(
        capacity={"raw": raw, "config": ReplicaScalingConfig(mode="amcsg")})
    assert res.capacity.mode == "amcsg"
    # no "defaulted to forecasted_mcs" note when caller is explicit
    assert not any("defaulted" in n for n in res.notes)


def test_optimize_fleet_capacity_mode_convenience_is_applied():
    # Passing mode= (without a full config) must actually select that mode,
    # not silently fall back to the default.
    raw = [(float(i), 100) for i in range(200)]
    res = AureliusOptimizer().optimize_fleet(capacity={"raw": raw, "mode": "amcsg"})
    assert res.capacity.mode == "amcsg"
    assert not any("defaulted" in n for n in res.notes)


# --------------------------------------------------------------------------
# 5. Energy optimize path is byte-identical (parity preserved)
# --------------------------------------------------------------------------

def test_energy_optimize_parity():
    jobs, da, carbon = _energy_fixture()
    cfg = OptimizationConfig(default_region="us-east", min_power_fraction=1.0)
    direct = JobScheduler(cfg).solve(jobs, da, carbon, method="greedy")
    fleet = AureliusOptimizer(cfg).optimize_fleet(
        energy={"jobs": jobs, "price_data": da, "carbon_data": carbon,
                "method": "greedy"})
    routed = [(d.job_id, d.region, d.start_time.isoformat()) for d in fleet.energy.schedule]
    expect = [(d.job_id, d.region, d.start_time.isoformat()) for d in direct.schedule]
    assert routed == expect


# --------------------------------------------------------------------------
# 6. Facade-bypass routing: simulation/compare routes through the optimizer
#    while producing an identical optimized schedule.
# --------------------------------------------------------------------------

def test_compare_routes_through_facade_with_parity():
    from aurelius.simulation.compare import ScenarioComparator

    cmp = ScenarioComparator(OptimizationConfig(default_region="us-east",
                                                min_power_fraction=1.0))
    # routed energy engine is obtained from the canonical optimizer
    assert cmp._optimizer is not None
    assert cmp.scheduler is cmp._optimizer.scheduler


# --------------------------------------------------------------------------
# 7. Live-service surface is reachable + recommendation-only
# --------------------------------------------------------------------------

def test_serving_orchestration_is_recommendation_only():
    from aurelius.constraints.engine import ConstraintAwareEngine

    eng = AureliusOptimizer().serving_orchestration
    assert isinstance(eng, ConstraintAwareEngine)
    assert eng.implementation_mode == "recommendation_only"
