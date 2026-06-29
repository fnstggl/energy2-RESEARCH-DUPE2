# Electricity-Only Value Results (Phase 8)

The electricity value, split into its two **separate** mechanisms with **separate ledgers**. **Hard rule
(enforced):** deferrable energy-cost savings are NEVER blended into serving gp/$ — the economic objective does
not value deferrable work, so its saving stays in its own ledger. Source: the N2 backtest
(`checkpointed_n2_backtest.json`) + the #117 dedicated deferrable validation.

## 1. N2 — online serving power-shaping (the new mechanism)

`all_knobs_real_price_n2_dvfs` − `all_knobs_real_price_no_n2` (only `electricity_price_aware` differs; same
real price, same workload, online serving only — no work is time-shifted):

| market·window | price $/kWh | **N2 gp/$ value** | clock (N2) | SLA n2 / no_n2 | downclock frac |
|--|--|--|--|--|--|
| pjm·volatile | 0.076–0.178 | **+5784.4** | `low×3` (downclock) | 0.388 / 0.383 | 1.0 |
| pjm·expensive | 0.092–0.178 | **+4859.2** | `low×3` (downclock) | 0.529 / 0.525 | 1.0 |
| caiso·expensive | 0.029–0.031 | +1539.4 | `high×2` (upclock — cheap power) | 0.479 / 0.483 | 0.0 |
| ercot·expensive | 0.045–0.059 | +1180.3 | `base×3` (drop the upclock) | 0.483 / 0.483 | 0.0 |
| ercot·volatile | 0.032–0.059 | 0.0 | unchanged | 0.483 / 0.483 | 0.0 |
| caiso·volatile | 0.028–0.031 | 0.0 | unchanged | 0.446 / 0.446 | 0.0 |

- **N2 value: mean +2227 gp/$, median ≈ +1360, best +5784 (pjm·volatile) — all DIRECTIONAL_ONLY.** It is
  largest where the price is highest (high-price PJM, downclock) and ~0 where the price is flat-and-low.
- **SLA impact: ≈ neutral (±0.004 vs. no_n2)** — N2 buys gp/$ through the energy channel, not by shedding SLA
  *within the MPC family*. (The HEADLINE Pareto gate vs. the SLA-aware baseline still fails — the whole MPC
  family sits above the baseline's violation rate; that is the base-MPC property, not an N2 regression.)
- **Energy / electricity cost saved:** the gp/$ lift flows through the operator-cost energy term
  (`power_scale` × billed energy × price). On the expensive PJM windows N2 cut operator cost ≈ 1.5%.
- **Slack consumed:** `mean_sla_slack_ms` is **negative** in every window (the backtest windows are overloaded),
  so N2's value here flows through the **clock-free memory-bound-decode** assumption, not through spending
  positive slack — an honest upper-bound (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`). The fixture proof
  (`test_n2_power_arbitrage`) shows the positive-slack case directly: at high price, downclocking memory-bound
  decode spends ~6 ms of slack for a real Pareto-safe gp/$ gain that scales with price.
- **Pareto-safe fraction (vs baseline): 0/6** → reported DIRECTIONAL_ONLY, never a headline.

## 2. Deferrable — energy time-shifting (separate ledger, unchanged from #117)

`all_knobs_real_price_deferrable_only` − `all_knobs_real_price_no_n2` = **0.0 serving gp/$ in all 6 cells** —
the scheduler runs only on spare GPU-seconds and **never touches serving**. Its value is a separate
energy-cost ledger:

- In-backtest (3-period windows): shifting saving is $0 with deadlines respected, or non-zero only when
  deadlines are missed (the cramped-window artefact from #117) — **no valid in-backtest saving**.
- **Dedicated 24 h validation (#117, `deferrable_shifting_validation.json`):** valid deadline-respecting saving
  **4.7% PJM / 19.4% ERCOT / 5.4% CAISO**, 0 missed, $0 under a flat price — the honest deferrable number.
- **SIMULATOR_INFERENCE workload**; never folded into serving gp/$.

## 3. Combined electricity value

- `n2_plus_deferrable − no_n2` = the **N2 serving lift** (deferrable adds 0 to serving gp/$).
- **interaction = 0 in all 6 cells** — N2 (power-shaping) and deferrable (time-shifting) are orthogonal in
  serving gp/$; their joint value is the N2 serving lift **plus** the deferrable energy saving (added, not
  multiplied; different ledgers).
- **Pareto gate / classification:** N2 serving lift = **DIRECTIONAL_ONLY** (real causal, fails the
  vs-baseline SLA clause); deferrable saving = **FIXTURE_ONLY** (separate ledger). **No HEADLINE_SAFE
  electricity gp/$ saving is claimed.**

## Honest bottom line

- **N2 is a real, causal, price-directional online power-shaping lever** worth **+4859/+5784 gp/$ (≈+1.5%) in
  high-price PJM windows**, ~0 at low price, **SLA-neutral within the MPC family** — but **not headline-safe**
  (the MPC family sits above the SLA-aware baseline's violation rate).
- **Deferrable is a real energy-cost saving (4.7–19.4%) in a separate ledger**, contributing **0** to serving
  gp/$ by construction.
- The two are **orthogonal** (interaction 0). Nothing is blended; nothing is tuned; the Pareto gate is unchanged.
