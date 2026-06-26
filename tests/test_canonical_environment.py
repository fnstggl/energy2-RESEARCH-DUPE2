"""Tests for the CanonicalMultiPlaneEnvironment scaffold (PR-1).

Verifies the two-clock orchestration runs, the cost model adds PUE + depreciation,
the fidelity manifest is honest (never production-grade while fields are below
MEASURED, and records the three integration seams), and the validation seed
discriminates matching vs non-matching distributions.
"""

from __future__ import annotations

from aurelius.datasets.canonical import augment_with_best_effort
from aurelius.environment import (
    CanonicalMultiPlaneEnvironment,
    CostModel,
    FleetPlane,
    match_distribution,
)
from aurelius.environment.canonical import TIER_MEASURED


def _hourly_jobs(hours=2, n=600):
    out = {}
    raw = [(float(i) * 1.2, 100 + (i % 6) * 40) for i in range(n)]
    for h in range(hours):
        shifted = [(t + 0.2 * h, tok) for t, tok in raw]
        jobs, _ = augment_with_best_effort(shifted, warp=1.0, fraction=0.2)
        out[h] = jobs
    return out


def test_environment_runs_two_clock_loop():
    env = CanonicalMultiPlaneEnvironment()
    res = env.run(_hourly_jobs(hours=3), tick_seconds=60.0, sla_s=10.0)
    assert len(res.hours) == 3
    assert res.total_goodput > 0 and res.total_cost > 0
    assert res.goodput_per_dollar > 0
    # each hour carries fleet state + serving kpi + cost
    for h in res.hours:
        assert {"hour", "fleet", "kpi", "cost"} <= set(h)


def test_cost_model_adds_pue_and_depreciation():
    cm = CostModel(pue=1.3, gpu_depreciation_per_gpu_hour=0.5, gpu_kw=0.7)
    c = cm.cost(gpu_hours=10.0, gpu_type="H100", energy_price_per_kwh=0.10)
    # depreciation = gpu_hours * per_hour
    assert c.depreciation_cost == 0.5 * 10.0
    # energy = gpu_hours * gpu_kw * pue * price
    assert abs(c.energy_cost - 10.0 * 0.7 * 1.3 * 0.10) < 1e-9
    assert c.total > c.gpu_infra_cost  # PUE+depreciation strictly add cost
    # the added knobs are flagged as modeled assumptions, not measured
    a = cm.assumptions()
    assert a["pue"]["tier"] != TIER_MEASURED
    assert a["gpu_depreciation_per_gpu_hour"]["tier"] != TIER_MEASURED


def test_manifest_is_honest_and_records_seams():
    env = CanonicalMultiPlaneEnvironment()
    man = env.manifest()
    # honesty gate: not production-grade while any field is below MEASURED
    assert man.is_production_grade() is False
    # the three integration seams are recorded explicitly (not papered over)
    assert len(man.seams) == 3
    assert any("calibration" in s for s in man.seams)
    assert any("time-model" in s for s in man.seams)
    # framing preserved
    assert "NOT real production telemetry" in man.framing


def test_only_class_mix_is_trace_derived_today():
    """PR-1 honesty: only the best-effort/priority mix is trace-derived; every
    other fleet field is a documented HEURISTIC default until PR-2 hooks land."""
    fleet = FleetPlane().state_at(0)
    assert fleet.fidelity["best_effort_fraction"] in ("PROXY", "MEASURED_REAL")
    assert fleet.fidelity["util_target"] == "HEURISTIC"
    assert fleet.fidelity["gpu_type_mix"] == "HEURISTIC"


def test_real_energy_price_series_is_tagged_measured():
    fp = FleetPlane(energy_price_series=[0.05, 0.09, 0.12])
    s1 = fp.state_at(1)
    assert s1.energy_price_per_kwh == 0.09
    assert s1.fidelity["energy_price_per_kwh"] == TIER_MEASURED


def test_validation_seed_discriminates():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert match_distribution("x", xs, xs).verdict == "PASS"
    assert match_distribution("x", xs, [100.0, 200.0, 300.0]).verdict == "FAIL"
    # a small shift should WARN or PASS, not FAIL
    near = [v + 0.2 for v in xs]
    assert match_distribution("x", xs, near).verdict in ("PASS", "WARN")
