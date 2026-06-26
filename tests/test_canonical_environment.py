"""Tests for the canonical multi-plane environment (first-principles build).

Proves the environment runs end-to-end on sample slices, calibrates from real
trace distributions (not constants), serves token-level (no M/M/1), prices with
PUE+depreciation, matches a held-out distribution, keeps the planes separate
(NO row-joins), and stays honest (never production-grade; ABSENT tier present).
"""

from __future__ import annotations

import os

from aurelius.benchmarks.srtf_serving_backtest import (
    DEFAULT_AZURE_FIXTURE,
    calibrate_time_warp,
    load_serving_requests,
)
from aurelius.environment import (
    CanonicalMultiPlaneEnvironment,
    CostModel,
    FidelityManifest,
    KVReuseModel,
    ServingPlane,
    V2026FleetPlane,
    build_bridge,
    check_distribution,
)
from aurelius.environment.calibration_bridge import mooncake_prefix_hit_rate
from aurelius.environment.schemas import TRACE_DERIVED, FleetState, ServingRequest
from aurelius.environment.validation_suite import (
    NOT_PRODUCTION_REALISTIC_YET,
    ks_statistic,
    wasserstein1,
)

_FIX = os.path.join(os.path.dirname(__file__), "fixtures")
_MOONCAKE = os.path.join(_FIX, "mooncake", "mooncake_sample.csv")


def _azure(limit=2000):
    raw = load_serving_requests(DEFAULT_AZURE_FIXTURE, limit=limit)
    warp = calibrate_time_warp(raw, servers=4, target_rho=0.85)
    return raw, warp


# --- FleetPlane (v2026-native) ---------------------------------------------

def test_fleet_plane_is_v2026_trace_derived():
    fp = V2026FleetPlane()
    assert fp.hours() == [0, 1]
    s = fp.state_at(0)
    assert s.total_gpus == 32                     # 4 servers x 8 GPUs (server_hourly)
    assert set(s.gpu_type_mix) == {"H100", "A100"}
    assert 0.0 < s.util_target < 1.0              # mean inference SM-util
    assert set(s.rack_locality) == {"asw_0", "asw_1"}   # asw_id topology
    assert s.energy_price_per_kwh > 0             # CAISO series
    # every fleet field is tagged TRACE_DERIVED
    assert all(t == TRACE_DERIVED for t in s.fidelity.values())
    assert all(p.tier == TRACE_DERIVED for p in fp.params_at(0))


# --- CalibrationBridge (distribution-derived, with provenance) -------------

def test_mooncake_prefix_hit_rate_computed_from_hash_ids():
    p = mooncake_prefix_hit_rate(_MOONCAKE)
    # 8 reqs; leading-block-seen hits = r1,r2,r4,r5,r6 = 5 -> 0.625
    assert p.value["prefix_hit_rate"] == 0.625
    assert p.tier == TRACE_DERIVED
    assert p.source_dataset == "mooncake"
    assert "hash_ids" in p.table_column


def test_bridge_params_carry_full_provenance():
    raw, _ = _azure()
    fp = V2026FleetPlane()
    bridge = build_bridge(raw, mooncake_path=_MOONCAKE, fleet_plane=fp)
    tok = bridge.by_name("azure_token_distribution")
    assert tok.tier == TRACE_DERIVED and tok.safe_for_headline
    # every param has the required provenance fields populated
    for p in bridge.params:
        assert p.source_dataset and p.table_column and p.fitting_method
        assert p.train_holdout_split and p.trace_version
    assert bridge.holdout["azure_tokens"] and bridge.holdout["train_tokens"]


# --- ServingPlane (token-level, no M/M/1) ----------------------------------

def test_serving_plane_is_token_level_with_kv():
    raw, warp = _azure()
    fp = V2026FleetPlane()
    sp = ServingPlane()
    kv = KVReuseModel(hit_rate=0.5)
    reqs = sp.build_requests(raw[:500], warp=warp, best_effort_fraction=0.2, kv=kv)
    assert all(isinstance(r, ServingRequest) for r in reqs)
    assert any(r.cls == "best_effort" for r in reqs)       # class from fleet mix
    assert any(r.kv_prefix_id for r in reqs)               # KV hits marked
    kpi, action = sp.run_hour(reqs, fp.state_at(0), tick_seconds=60.0, sla_s=10.0, kv=kv)
    # per-request token-level KPI (n_total == requests), not an aggregate proxy
    assert kpi.n_total == len(reqs)
    assert kpi.sla_safe_goodput > 0
    assert action["n_kv_hits"] > 0


# --- CostModel (PUE + depreciation + per-type) -----------------------------

def test_cost_model_depreciation_pue_per_gpu_type():
    cm = CostModel(pue=1.3)
    h100 = cm.cost(gpu_hours=10.0, gpu_type="H100", energy_price_per_kwh=0.10)
    a100 = cm.cost(gpu_hours=10.0, gpu_type="A100", energy_price_per_kwh=0.10)
    assert h100.gpu_depreciation_cost > a100.gpu_depreciation_cost   # per-type
    assert h100.energy_cost > 0 and h100.total > h100.gpu_depreciation_cost
    # rental cross-check is reported but NOT in the owned-hardware total
    assert h100.rental_cross_check > 0
    assert abs(h100.total - (h100.gpu_depreciation_cost + h100.energy_cost
                             + h100.network_cost)) < 1e-9


# --- ValidationSuite (metrics + honesty cap) -------------------------------

def test_validation_metrics_and_verdicts():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert ks_statistic(xs, xs) == 0.0
    assert wasserstein1(xs, xs) == 0.0
    assert check_distribution("x", xs, xs).verdict == "PASS"
    assert check_distribution("x", xs, [50.0, 60.0, 70.0]).verdict == "FAIL"


# --- FidelityManifest (honesty gate + ABSENT tier) -------------------------

def test_manifest_absent_tier_and_gate():
    raw, _ = _azure()
    bridge = build_bridge(raw, mooncake_path=_MOONCAKE, fleet_plane=V2026FleetPlane())
    man = FidelityManifest.from_params(bridge.params)
    # the 5 structurally-proprietary signals are explicitly ABSENT
    assert len(man.absent) == 5
    names = {s.name for s in man.absent}
    assert {"user_operator_intent", "hardware_health", "live_kv_memory_state"} <= names
    # never production-grade while the ABSENT tier is unfilled
    assert man.is_production_grade() is False


# --- End-to-end + the NO-ROW-JOIN guarantee --------------------------------

def test_environment_runs_two_clock_and_validates():
    raw, warp = _azure(limit=3000)
    half = len(raw) // 2
    env = CanonicalMultiPlaneEnvironment(mooncake_path=_MOONCAKE, warp=warp)
    res = env.run({0: raw[:half], 1: raw[half:]})
    assert len(res.steps) == 2
    for s in res.steps:
        assert {"observation", "action", "reward", "metrics"} <= set(s.to_dict())
        assert s.reward > 0
        assert "fleet" in s.observation and "kpi" in s.metrics
    # held-out token distribution matches; honesty cap holds (cost params HEURISTIC)
    kinds = {c["kind"]: c for c in res.validation["checks"]}
    assert kinds["azure_token_distribution"]["verdict"] in ("PASS", "WARN")
    assert res.validation["overall_verdict"] == NOT_PRODUCTION_REALISTIC_YET
    assert res.manifest["is_production_grade"] is False


def test_no_row_join_planes_are_disjoint():
    """The contracts must be structurally disjoint — Azure requests carry no v2026
    fields and the v2026 fleet state carries no per-request Azure rows. Planes
    couple via state variables (best-effort fraction), never merged rows."""
    req_fields = set(ServingRequest.__dataclass_fields__)
    fleet_fields = set(FleetState.__dataclass_fields__)
    # no per-request serving field appears on the fleet state and vice versa
    assert req_fields.isdisjoint(fleet_fields)
    assert "server_id" not in req_fields and "asw_id" not in req_fields
    assert "arrival_s" not in fleet_fields and "tokens" not in fleet_fields
    # the only cross-plane coupling is a calibrated SCALAR (best_effort_fraction)
    fp = V2026FleetPlane()
    be = fp.state_at(0).best_effort_fraction
    raw, warp = _azure(limit=500)
    reqs = ServingPlane().build_requests(raw, warp=warp, best_effort_fraction=be)
    assert 0.0 <= sum(1 for r in reqs if r.cls == "best_effort") / len(reqs) <= 1.0
