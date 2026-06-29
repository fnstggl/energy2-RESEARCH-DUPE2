"""N2 — SLA-slack power arbitrage: an EXPLANATORY decomposition of the existing reward, never a bonus.

N2 is the behaviour where the MPC spends remaining **SLA slack** (ms of headroom before the latency target)
to run latency-bound *online* serving work at a lower clock, saving watts → joules → electricity dollars, while
the SLA stays within budget. The clock→power→cost channel is ALREADY in the reward (`world_simulator` applies
`power_scale` to the operator cost; the planner already downclocks at high price — PR #117). So N2 here is
purely a **read-out** of two already-simulated `PeriodOutcome`s — the selected clock vs. a base-clock
counterfactual on the SAME online serving period — answering: how much SLA slack did the downclock spend, and
how many electricity dollars / gp/$ did that buy, and was it Pareto-safe?

Hard rules (mirror the honesty contract):
  * No reward bonus — value flows only through energy×price → operator cost and latency → SLA slack.
  * N2 is ONLINE serving only — deferrable time-shifting is NOT N2 (different module, different ledger).
  * A downclock that worsens the SLA-violation rate is NOT Pareto-safe and must be reported as such.
  * Flat price ⇒ the selected and base clocks face the same cost signal ⇒ N2 value is incidental, not arbitrage.
"""

from __future__ import annotations

_J_PER_KWH = 3.6e6
_CLOCK_RANK = {"low": 0, "base": 1, "high": 2}     # for "did we downclock vs base" semantics


def slack_summary(outcome) -> dict:
    """The SLA-slack diagnostic for one period outcome (latency-class completion tail vs the SLA target)."""
    return {
        "sla_target_s": round(outcome.sla_target_s, 4),
        "predicted_tail_latency_s": round(outcome.predicted_tail_latency_s, 4),
        "sla_slack_ms": outcome.sla_slack_ms,
        "sla_slack_pct": round(outcome.sla_slack_pct, 4),
        "sla_violation_rate": round(outcome.sla_violation_rate, 5),
    }


def n2_decomposition(selected, base, *, selected_clock: str, price_per_kwh: float, base_clock: str = "base") -> dict:
    """Decompose the N2 value of `selected` (the chosen clock) vs `base` (a base-clock counterfactual) on the
    SAME online serving period. Both are `PeriodOutcome`s already produced by `simulate_period`.

    Returns a pure read-out. Positive `electricity_cost_saved_usd` ⇒ the selected clock cost fewer operator
    dollars (dominated by the power_scale energy channel); positive `slack_consumed_ms` ⇒ the selected clock
    spent SLA headroom. `pareto_safe` is False if the selected clock raised the SLA-violation rate.
    """
    # the actual reward channel: operator cost + gp/$ the selected clock bought vs base (drives the gate)
    cost_saved = round(base.operator_cost - selected.operator_cost, 6)
    gpd_delta = round(selected.goodput_per_dollar - base.goodput_per_dollar, 2)
    # the "why" channel: energy (diagnostic energy_j) × price — labelled, can differ from cost_saved for a
    # mixed prefill/decode load (see world_simulator energy_j vs the cost-model energy term)
    energy_saved_kwh = round((base.energy_j - selected.energy_j) / _J_PER_KWH, 5)
    energy_cost_saved_est = round(energy_saved_kwh * float(price_per_kwh), 6)
    slack_consumed_ms = round(base.sla_slack_ms - selected.sla_slack_ms, 2)
    pareto_safe = bool(selected.sla_violation_rate <= base.sla_violation_rate + 1e-9)
    downclocked = _CLOCK_RANK.get(selected_clock, 1) < _CLOCK_RANK.get(base_clock, 1)
    # N2 is "active" only when a downclock actually spent slack to save cost while staying Pareto-safe
    n2_active = bool(downclocked and cost_saved > 0 and pareto_safe)
    return {
        "selected_clock": selected_clock,
        "base_clock": base_clock,
        "downclocked": downclocked,
        "sla_slack_ms_selected": selected.sla_slack_ms,
        "sla_slack_ms_base": base.sla_slack_ms,
        "slack_consumed_ms": slack_consumed_ms,
        "energy_saved_kwh": energy_saved_kwh,
        "electricity_cost_saved_usd_est": energy_cost_saved_est,   # diagnostic energy channel (labelled)
        "operator_cost_saved_usd": cost_saved,                     # the channel that actually drives gp/$
        "gp_per_dollar_delta": gpd_delta,
        "sla_violation_rate_selected": round(selected.sla_violation_rate, 5),
        "sla_violation_rate_base": round(base.sla_violation_rate, 5),
        "pareto_safe": pareto_safe,
        "n2_active": n2_active,
        "price_per_kwh": round(float(price_per_kwh), 6),
    }
