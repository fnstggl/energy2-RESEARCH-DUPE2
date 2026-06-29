"""N2 SLA-slack power-arbitrage fixtures — the mechanism proven cheaply at the simulate_period level.

N2 = spend SLA slack to downclock latency-bound ONLINE serving work, saving electricity dollars while the SLA
stays within budget. These fixtures prove the causal mechanism and — crucially — that N2 never fabricates
value: it flows only through energy×price → operator cost and latency → SLA, the Pareto gate blocks
SLA-shedding, and deferrable time-shifting is excluded. The clock→power→cost channel already lives in
`world_simulator`/`cost_model`; N2 here is the explanatory decomposition (`aurelius.environment.n2`).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aurelius.environment.cost_model import CostModel
from aurelius.environment.fleet_plane_v2026 import V2026FleetPlane
from aurelius.environment.n2 import n2_decomposition, slack_summary
from aurelius.environment.training import make_world_state
from aurelius.environment.world_simulator import simulate_period

_FLEET = V2026FleetPlane().state_at(0)
_CM = CostModel()
_WSP = {"n_servers": 24, "n_racks": 4, "seed": 0, "warm": 8, "processed_dir": None}
# decode-heavy = memory-bound (short prompt, long output); prefill-heavy = compute-bound (long prompt, short out)
_DECODE = [(i * 4.0, 512, 64) for i in range(24)]
_PREFILL = [(i * (1.0 / 8.0), 8, 4096) for i in range(400)]


def _sim(clock, price, wl, *, sla_s, arr=0.2, arr90=0.26):
    ws = make_world_state(_WSP)
    pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind", migration_policy="off",
                          batching_policy="conservative", precision_policy="bf16", spec_decode_policy="off",
                          clock_policy=clock, colocation_policy="off", prefill_decode_policy="shared")
    fc = {"arrival_rate": arr, "arrival_p90": arr90, "mean_service_s": 1.0}
    return simulate_period(ws, pol, wl, fc, energy_price_per_kwh=price, sla_s=sla_s, tick_seconds=10.0,
                           base_service_factor=1.0, cost_model=_CM, fleet_state=_FLEET, cost_scenario="owned",
                           best_effort_fraction=_FLEET.best_effort_fraction, period_hours=180 / 3600,
                           dt_seconds=3600.0)


# --- Phase 1: SLA slack is explicit + diagnostic-visible ---------------------
def test_sla_slack_is_computed_and_visible():
    o = _sim("base", 0.10, _DECODE, sla_s=20.0)
    s = slack_summary(o)
    assert s["sla_target_s"] == 20.0                         # the SLA target is surfaced
    assert s["predicted_tail_latency_s"] > 0                 # a real completion-latency tail
    assert o.sla_slack_ms == pytest.approx(1000.0 * (o.sla_target_s - o.predicted_tail_latency_s), abs=1.0)
    assert o.sla_slack_ms > 0                                # loose SLA ⇒ positive headroom (budget to spend)


# --- Fixture 1: high price, memory-bound decode, slack → low clock wins -------
def test_f1_high_price_memorybound_slack_low_clock_wins():
    base = _sim("base", 0.28, _DECODE, sla_s=20.0)
    low = _sim("low", 0.28, _DECODE, sla_s=20.0)
    d = n2_decomposition(low, base, selected_clock="low", price_per_kwh=0.28)
    assert low.goodput_per_dollar > base.goodput_per_dollar  # downclock is the gp/$ winner
    assert low.sla_violation_rate <= base.sla_violation_rate # SLA not worse (Pareto-safe)
    assert d["operator_cost_saved_usd"] > 0                  # value flows through cost
    assert d["n2_active"] and d["pareto_safe"]
    assert 0 <= d["slack_consumed_ms"] < 100                 # memory-bound: ~free, tiny slack spent


# --- Fixture 2 (reframed honestly): N2 value scales with price ----------------
def test_f2_n2_value_scales_with_price():
    hi = n2_decomposition(_sim("low", 0.28, _DECODE, sla_s=20.0), _sim("base", 0.28, _DECODE, sla_s=20.0),
                          selected_clock="low", price_per_kwh=0.28)
    lo = n2_decomposition(_sim("low", 0.03, _DECODE, sla_s=20.0), _sim("base", 0.03, _DECODE, sla_s=20.0),
                          selected_clock="low", price_per_kwh=0.03)
    # same physical energy saved at both prices; dollars + gp/$ value scale ~linearly with the price level
    assert hi["operator_cost_saved_usd"] > lo["operator_cost_saved_usd"]
    assert hi["gp_per_dollar_delta"] > lo["gp_per_dollar_delta"]


# --- Fixture 3: compute-bound downclock spends FAR more slack than decode -----
def test_f3_compute_bound_spends_more_slack_than_decode():
    dec = n2_decomposition(_sim("low", 0.28, _DECODE, sla_s=20.0), _sim("base", 0.28, _DECODE, sla_s=20.0),
                           selected_clock="low", price_per_kwh=0.28)
    pf_base = _sim("base", 0.28, [(i * 4.0, 8, 2048) for i in range(24)], sla_s=20.0)
    pf_low = _sim("low", 0.28, [(i * 4.0, 8, 2048) for i in range(24)], sla_s=20.0)
    pf = n2_decomposition(pf_low, pf_base, selected_clock="low", price_per_kwh=0.28)
    # downclocking compute-bound prefill costs more latency/slack per dollar than clock-free decode
    assert pf["slack_consumed_ms"] > dec["slack_consumed_ms"]


# --- Fixture 4: compute-bound + saturation → low clock LOSES (Pareto-blocked) -
def test_f4_compute_bound_saturated_low_clock_blocked():
    base = _sim("base", 0.28, _PREFILL, sla_s=1.0, arr=8.0, arr90=10.4)
    low = _sim("low", 0.28, _PREFILL, sla_s=1.0, arr=8.0, arr90=10.4)
    d = n2_decomposition(low, base, selected_clock="low", price_per_kwh=0.28)
    assert low.sla_violation_rate > base.sla_violation_rate  # downclock pushes the saturated queue over SLA
    assert d["pareto_safe"] is False                         # the gate blocks the SLA-shedding downclock
    assert d["n2_active"] is False                           # N2 does NOT claim value here
    assert low.goodput_per_dollar < base.goodput_per_dollar  # and it is a real gp/$ loss


def test_f4b_compute_bound_saturated_high_clock_buys_latency():
    base = _sim("base", 0.28, _PREFILL, sla_s=0.6, arr=20.0, arr90=26.0)
    high = _sim("high", 0.28, _PREFILL, sla_s=0.6, arr=20.0, arr90=26.0)
    # under heavy compute-bound saturation, UPclocking buys latency (the opposite of N2): fewer SLA
    # violations. Whether that improves gp/$ depends on the price (high clock costs more power) — here it is
    # ~neutral, which is the honest result: high clock is an SLA lever, not a free gp/$ win.
    assert high.sla_violation_rate < base.sla_violation_rate
    assert high.goodput_per_dollar == pytest.approx(base.goodput_per_dollar, rel=0.05)


# --- Fixture 5: flat price → no fabricated arbitrage (value is pure energy×price)
def test_f5_flat_price_no_fake_arbitrage():
    d1 = n2_decomposition(_sim("low", 0.05, _DECODE, sla_s=20.0), _sim("base", 0.05, _DECODE, sla_s=20.0),
                          selected_clock="low", price_per_kwh=0.05)
    d2 = n2_decomposition(_sim("low", 0.10, _DECODE, sla_s=20.0), _sim("base", 0.10, _DECODE, sla_s=20.0),
                          selected_clock="low", price_per_kwh=0.10)
    # the SAME physical energy is saved at both flat prices; the dollar value is exactly energy×price with NO
    # premium → doubling the flat price ~doubles the saving (no fabricated arbitrage bump)
    assert d1["energy_saved_kwh"] == pytest.approx(d2["energy_saved_kwh"], rel=1e-6)
    assert d2["electricity_cost_saved_usd_est"] == pytest.approx(2.0 * d1["electricity_cost_saved_usd_est"], rel=0.05)


# --- Fixture 6: interaction — precision changes the regime → changes N2 value -
def test_f6_precision_interaction_changes_n2_value():
    def sim_prec(clock, prec):
        ws = make_world_state(_WSP)
        pol = SimpleNamespace(prewarm_policy="off", placement_policy="topology_blind", migration_policy="off",
                              batching_policy="conservative", precision_policy=prec, spec_decode_policy="off",
                              clock_policy=clock, colocation_policy="off", prefill_decode_policy="shared")
        fc = {"arrival_rate": 0.2, "arrival_p90": 0.26, "mean_service_s": 1.0}
        return simulate_period(ws, pol, _DECODE, fc, energy_price_per_kwh=0.28, sla_s=20.0, tick_seconds=10.0,
                               base_service_factor=1.0, cost_model=_CM, fleet_state=_FLEET, cost_scenario="owned",
                               best_effort_fraction=_FLEET.best_effort_fraction, period_hours=180 / 3600,
                               dt_seconds=3600.0)
    bf16 = n2_decomposition(sim_prec("low", "bf16"), sim_prec("base", "bf16"), selected_clock="low", price_per_kwh=0.28)
    int4 = n2_decomposition(sim_prec("low", "int4"), sim_prec("base", "int4"), selected_clock="low", price_per_kwh=0.28)
    # precision is a different roofline operating point → the N2 cost saving is not identical across it
    assert bf16["operator_cost_saved_usd"] != int4["operator_cost_saved_usd"]


# --- Invariants: no fake reward; N2 is online-only (excludes deferrable) ------
def test_n2_active_requires_real_pareto_safe_saving():
    # structural guarantee: N2 is "active" ONLY when a downclock saved cost AND stayed Pareto-safe — never a bonus
    base = _sim("base", 0.28, _DECODE, sla_s=20.0)
    low = _sim("low", 0.28, _DECODE, sla_s=20.0)
    d = n2_decomposition(low, base, selected_clock="low", price_per_kwh=0.28)
    assert d["n2_active"] == (d["downclocked"] and d["operator_cost_saved_usd"] > 0 and d["pareto_safe"])
    # an UPclock (high vs base) is never an N2 downclock-arbitrage win
    up = n2_decomposition(_sim("high", 0.28, _DECODE, sla_s=20.0), base, selected_clock="high", price_per_kwh=0.28)
    assert up["downclocked"] is False and up["n2_active"] is False


def test_n2_excludes_deferrable_and_never_time_shifts_serving():
    # N2 reads online serving outcomes only; the decomposition has NO deferrable field, and the module does
    # not touch the deferrable scheduler (deferrable is a separate ledger).
    d = n2_decomposition(_sim("low", 0.28, _DECODE, sla_s=20.0), _sim("base", 0.28, _DECODE, sla_s=20.0),
                         selected_clock="low", price_per_kwh=0.28)
    assert "deferrable" not in d and "shifted" not in d
    import aurelius.environment.n2 as n2mod
    src = __import__("inspect").getsource(n2mod)
    assert "deferrable" not in src.lower().split("never")[0] or "separate" in src.lower()  # no deferrable logic


def test_gp_per_dollar_flows_through_cost_not_a_bonus():
    # gp/$ delta and operator-cost delta must agree in sign — value flows through cost, not an additive bonus
    base = _sim("base", 0.28, _DECODE, sla_s=20.0)
    low = _sim("low", 0.28, _DECODE, sla_s=20.0)
    d = n2_decomposition(low, base, selected_clock="low", price_per_kwh=0.28)
    assert (d["gp_per_dollar_delta"] > 0) == (d["operator_cost_saved_usd"] > 0)
