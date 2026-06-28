"""Price-aware clock/power tests (focused — causal cost path, regime-dependent latency, Pareto gate)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aurelius.environment.cost_model import CostModel
from aurelius.environment.price_series import MARKETS, load_price_series, price_percentiles
from aurelius.environment.roofline import Workload
from aurelius.environment.roofline_actions import roofline_action_factors
from aurelius.environment.training import claim_gate


def _bundle(**kw):
    d = dict(precision_policy="bf16", spec_decode_policy="off", clock_policy="base",
             colocation_policy="off", prefill_decode_policy="shared")
    d.update(kw)
    return SimpleNamespace(**d)


# --- electricity price enters cost ONLY through energy × price ---------------
def test_electricity_price_affects_cost_only_through_energy():
    cm = CostModel()
    b1 = cm.operator_cost(gpu_hours=10, gpu_type="H100", energy_price_per_kwh=0.05, power_scale=1.0)
    b2 = cm.operator_cost(gpu_hours=10, gpu_type="H100", energy_price_per_kwh=0.10, power_scale=1.0)
    # depreciation (and all non-energy terms) are price-independent
    assert b2.depreciation_cost == b1.depreciation_cost
    # the entire cost change is the energy change — price touches nothing else
    assert abs((b2.total_operator_cost - b1.total_operator_cost) - (b2.energy_cost - b1.energy_cost)) < 1e-9
    # energy is LINEAR in price (2× price → 2× energy cost) — no hidden price coupling
    assert abs(b2.energy_cost - 2.0 * b1.energy_cost) < 1e-9


# --- downclock cuts power/energy, and the saving scales with price -----------
def test_downclock_cuts_energy_and_saving_scales_with_price():
    wl = Workload(prompt_tokens=64, decode_tokens=512, context_len=384)
    pf_low = roofline_action_factors(_bundle(clock_policy="low"), wl, gpu="H100")["power_factor"]
    pf_high = roofline_action_factors(_bundle(clock_policy="high"), wl, gpu="H100")["power_factor"]
    assert pf_low < 1.0 < pf_high                          # DVFS: down cuts power, up raises it
    cm = CostModel()

    def _save(price):
        base = cm.operator_cost(gpu_hours=10, gpu_type="H100", energy_price_per_kwh=price, power_scale=1.0)
        low = cm.operator_cost(gpu_hours=10, gpu_type="H100", energy_price_per_kwh=price, power_scale=pf_low)
        return base.energy_cost - low.energy_cost
    assert _save(0.30) > _save(0.03) > 0                    # downclock saves more when power is expensive


# --- the downclock latency penalty is regime-dependent (why it's not free everywhere) ----
def test_downclock_latency_penalty_is_regime_dependent():
    wl = Workload(prompt_tokens=64, decode_tokens=512, context_len=384)
    low = roofline_action_factors(_bundle(clock_policy="low"), wl, gpu="H100", batch_size=16)
    # decode is memory-bandwidth-bound here → SM clock has NO effect on decode time (free to downclock)
    assert low["decode_regime"] == "memory_bandwidth_bound"
    assert low["decode_factor"] == pytest.approx(1.0, abs=1e-6)
    # but prefill carries a compute component → downclock DOES slow prefill (a latency cost can exist,
    # so downclock is not unconditionally free — under tight SLA on prefill-heavy work it can hurt)
    assert low["prefill_factor"] > 1.0


# --- the Pareto gate still blocks SLA-shedding (a cheaper-but-worse policy) --
def test_pareto_gate_blocks_sla_shedding():
    # an arm that wins gp/$ purely by letting MORE requests miss SLA must NOT pass the Pareto clause
    shedding = {"mpc_controller": SimpleNamespace(goodput_per_dollar=200.0, sla_violation_rate=0.08),
                "sla_aware": SimpleNamespace(goodput_per_dollar=100.0, sla_violation_rate=0.02)}
    g = claim_gate(shedding)
    assert g["beats_fair_baseline"] and not g["pareto_sla_not_worse"]    # higher gp/$ but SLA worse → blocked
    # a true Pareto win (gp/$ up, SLA not worse) passes the clause
    clean = {"mpc_controller": SimpleNamespace(goodput_per_dollar=200.0, sla_violation_rate=0.01),
             "sla_aware": SimpleNamespace(goodput_per_dollar=100.0, sla_violation_rate=0.02)}
    assert claim_gate(clean)["pareto_sla_not_worse"]


# --- price series is real + EIA is honestly absent --------------------------
def test_price_series_real_and_absent_documented():
    s = load_price_series("pjm")
    assert len(s) > 500
    pct = price_percentiles(s)
    assert 0.0 < pct["p10"] < pct["p90"] < 5.0             # sane $/kWh range
    assert set(MARKETS) == {"pjm", "ercot", "caiso"}
    with pytest.raises(ValueError):                        # EIA is not wired → must refuse, never fabricate
        load_price_series("eia")
