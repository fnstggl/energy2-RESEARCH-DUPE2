# N2 Decision Diagnostics (Phase 9)

What every MPC decision now exposes for N2 (SLA-slack power arbitrage), where it comes from, and the honest
switching-threshold story. All of it is **read-out of values the search already computed** — no extra solves
online, no reward change.

## Per-decision fields

On `Decision.forecast["electricity"]` (`controller.py`), every decision records:

| field | meaning | source |
|--|--|--|
| `selected_clock` | the clock the MPC chose (`base` / `low` / `high`) | the selected `ActionBundle` |
| `forecast_price_per_kwh` | the electricity price the planner priced against | forecast trajectory (real or constant) |
| `price_aware` | whether the rollout priced each horizon step (N2 on) | `controller.electricity_price_aware` |
| `sla_slack_ms` | SLA headroom left by the chosen clock (target − completion tail) | rollout `PeriodOutcome.sla_slack_ms` |
| `serving_time_shifted` | **always `False`** — N2 never delays online serving work | invariant |
| `deferrable_shifted` | **always `False`** online — deferrable is a separate offline ledger | invariant |

The rollout step record and the Decision Diagnostics Engine (`emit_diagnostics`) also carry `sla_slack_ms` in
the reward decomposition, so the slack the planner traded is visible alongside gp/$, SLA risk, warm-hold, etc.

## Offline N2 decomposition (`aurelius/environment/n2.py`)

For attribution (backtest, fixtures — where a base-clock counterfactual solve is allowed),
`n2_decomposition(selected, base)` returns, per period: `slack_consumed_ms`, `energy_saved_kwh`,
`electricity_cost_saved_usd_est` (diagnostic energy channel), `operator_cost_saved_usd` (the channel that
drives gp/$), `gp_per_dollar_delta`, `pareto_safe`, and `n2_active`. `n2_active` is **structurally** True only
when a downclock saved cost **and** stayed Pareto-safe — it can never be a standalone bonus.

## Switching thresholds — the honest story

The task asks for "the price threshold where the clock would change" and "the slack threshold where the clock
would change." Measured by fixture sweeps (`test_n2_power_arbitrage` + a price/load sweep), the answer is **the
clock is governed by a slack/load threshold, not a price threshold**:

- **In the slack regime (ample SLA headroom), `low` clock is the gp/$ winner at *every* price** — from $0.01 to
  $0.30. Price does **not flip** the clock; it **amplifies the downclock advantage** (memory-bound decode:
  `low − base` gp/$ grows from +194 at $0.01 to +4195 at $0.30; compute-bound prefill: +8 → +176). So the N2
  "price threshold" in the slack regime is effectively **$0** — downclocking is preferred whenever slack
  exists, and the price sets the size of the win.
- **The clock flips on the LOAD / slack axis.** Under saturation (no slack), downclocking pushes the queue
  over the SLA → the gp/$-optimal clock flips `low → base → high`: at moderate compute-bound saturation `low`
  loses (SLA-violation 0.21 vs base 0.16, Pareto-blocked); at heavy saturation `high` wins (buys latency,
  violation 0.41 vs base 0.44). This is the real switching threshold — **the SLA-slack budget running out**,
  exactly the N2 framing.

So N2's price-awareness is best read as: *given* there is slack, spend more of it (downclock harder) the more
expensive power is; *when* slack runs out (saturation), stop downclocking and, if needed, upclock to protect
the SLA. The Pareto gate enforces the second half — it blocks any downclock that sheds SLA.

## Honest limit

In this simulator memory-bandwidth-bound decode is treated as **clock-independent in latency**, so downclocking
decode is ~free and wins even when measured slack is negative (an overloaded window) — the N2 value then flows
through the (free) decode power saving, not through spending positive slack. This is the documented upper-bound
assumption (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`); real decode has some clock sensitivity, which would make the
slack budget bind for decode too and **shrink** the N2 value. The slack-budget framing is literally binding for
compute-bound work today.
