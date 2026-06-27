"""Tests for the operator-side CostModel (Phase 3): owned + leased + sensitivity.

Proves the model is OPERATOR-side (depreciation of CapEx or a lease contract, not
cloud-tenant arbitrage), that every component is present and fidelity-tagged, and
that the sensitivity sweep makes the heuristic assumptions visible (low ≤ base ≤
high across PUE / CapEx / electricity / service-life / power / utilization).
"""

from __future__ import annotations

from aurelius.environment.cost_model import CostModel, GPUEconomics
from aurelius.environment.schemas import (
    EXTERNAL_OBSERVED,
    INFERRED,
    TRACE_DERIVED,
)


def test_legacy_cost_crosscheck_unchanged():
    cm = CostModel()
    h = cm.cost(gpu_hours=10.0, gpu_type="H100", energy_price_per_kwh=0.10)
    a = cm.cost(gpu_hours=10.0, gpu_type="A100", energy_price_per_kwh=0.10)
    assert h.gpu_depreciation_cost > a.gpu_depreciation_cost
    assert h.energy_cost > 0 and h.rental_cross_check > 0
    assert abs(h.total - (h.gpu_depreciation_cost + h.energy_cost + h.network_cost)) < 1e-9


def test_gpu_economics_depreciation_and_power():
    e = GPUEconomics("H100", acquisition_usd=35040.0, active_power_kw=0.7, idle_power_kw=0.1,
                     service_life_years=4.0)
    assert abs(e.depreciation_per_gpu_hour() - 35040.0 / (4 * 8760)) < 1e-9   # = $1.00/hr
    assert e.power_kw(0.0) == 0.1 and e.power_kw(1.0) == 0.7                   # idle..active
    assert e.power_kw(0.5) == 0.4                                             # utilization-adjusted


def test_operator_cost_owned_vs_leased():
    cm = CostModel()
    owned = cm.operator_cost(gpu_hours=10.0, gpu_type="H100", energy_price_per_kwh=0.10,
                             utilization=0.8, scenario="owned")
    leased = cm.operator_cost(gpu_hours=10.0, gpu_type="H100", energy_price_per_kwh=0.10,
                              utilization=0.8, scenario="leased")
    assert owned.basis == "owned_depreciation" and owned.depreciation_cost > 0 and owned.lease_cost == 0
    assert leased.basis == "leased_contract" and leased.lease_cost > 0 and leased.depreciation_cost == 0
    assert owned.energy_cost > 0 and leased.energy_cost > 0
    # per-useful-unit outputs
    d = owned.to_dict(n_sla_safe=1000, sla_safe_tokens=50000.0, sla_safe_goodput=50000.0)
    for k in ("energy_cost", "depreciation_cost", "network_cost", "queue_delay_cost",
              "sla_penalty_cost", "total_operator_cost", "cost_per_sla_safe_request",
              "cost_per_sla_safe_token", "goodput_per_dollar"):
        assert k in d
    assert d["cost_per_sla_safe_request"] > 0 and d["goodput_per_dollar"] > 0


def test_sensitivity_bands_visible_and_ordered():
    cm = CostModel()
    s = cm.sensitivity(gpu_hours=10.0, gpu_type="H100", energy_price_per_kwh=0.10, utilization=0.8)
    assert s["low_total"] <= s["base_total"] <= s["high_total"]
    for factor in ("pue", "acquisition", "electricity", "service_life", "power_draw", "utilization"):
        assert factor in s["per_factor"]
    # electricity has a real effect on the total (2x price → higher cost branch)
    assert s["per_factor"]["electricity"]["high"] > s["per_factor"]["electricity"]["low"]


def test_params_fidelity_tiers():
    cm = CostModel()
    owned = {p.name: p.tier for p in cm.params(scenario="owned")}
    assert owned["energy_price_per_kwh"] == TRACE_DERIVED
    assert owned["pue"] == INFERRED and owned["gpu_acquisition_usd"] == INFERRED
    assert "leased_usd_per_gpu_hour" not in owned
    leased = {p.name: p.tier for p in cm.params(scenario="leased")}
    assert leased["leased_usd_per_gpu_hour"] == EXTERNAL_OBSERVED


def test_no_tenant_side_arbitrage_surface():
    # the operator cost model exposes only owned/leased bases — no spot/reserved/
    # on-demand-arbitrage method or scenario exists.
    cm = CostModel()
    assert not any(hasattr(cm, m) for m in ("spot_cost", "reserved_cost", "arbitrage"))
    bases = {cm.operator_cost(gpu_hours=1, gpu_type="H100", energy_price_per_kwh=0.1,
                              scenario=s).basis for s in ("owned", "leased")}
    assert bases == {"owned_depreciation", "leased_contract"}


def test_env_reports_cost_breakdown_and_sensitivity():
    import os

    from aurelius.environment.canonical import CanonicalMultiPlaneEnvironment
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mooncake = os.path.join(repo, "tests", "fixtures", "mooncake", "mooncake_sample.csv")
    res = CanonicalMultiPlaneEnvironment(mooncake_path=mooncake).run(
        {0: [(float(i) * 0.5, 100 + (i * 7) % 300) for i in range(40)]})
    cost = res.steps[0].metrics["cost"]
    assert cost["basis"] == "owned_depreciation" and cost["total_operator_cost"] > 0
    assert "cost_per_sla_safe_token" in cost
    sens = res.cost_sensitivity
    assert sens["low_total"] <= sens["base_total"] <= sens["high_total"]
    # cost validation checks present + passing
    by = {c["kind"]: c for c in res.validation["checks"]}
    assert by["cost_operator_side_only"]["verdict"] == "PASS"
    assert by["cost_sensitivity_bands"]["verdict"] == "PASS"
