# Token-Shape Forecasting + Price-Aware Clock/Power Shaping — Summary

Diagnostic-first PR. **No reward shaping, no Pareto-gate weakening, no parameter tuning to force a benchmark.**
Every effect flows through the existing causal path (planning workload → MPC; clock → power → energy ×
electricity_price → cost; clock → latency → SLA). Magnitudes are bounded-window simulator-inferred; the robust
findings are directions. Tracks A–E with per-track docs:
`WIDE_VALIDATION_CURRENT_MPC.md`, `TOKEN_SHAPE_FORECAST_GAP_RESULTS.md`,
`PRICE_AWARE_CLOCK_SHAPING_DIAGNOSTIC.md`, `TOKEN_SHAPE_AND_PRICE_SHAPING_RESULTS.md`.

## The 10 questions, answered

1. **Did +82.1% validate across wider windows?** **No — and the figure itself was misattributed.** PR #114's
   "current 183 152" was its harness's `oracle_var=None` arm, i.e. **oracle planning with the exact future**
   (`current_mpc_full == oracle_gp_per_dollar` *exactly* in that artifact). The *deployable* MPC (no foresight)
   beats the strongest baseline by **median +60% (range +25%…+144%)** across 8 regimes, **Pareto-safe in 7/8**
   (one int4 window sheds SLA). So the direction is robust; +82.1% is not a uniform, deployable headline.

2. **Did output-length attribution decrease?** **No.** Its raw leave-one-out planner-value is unchanged
   (37 550 gp/$ before and after): a recent-window quantile ≈ the global median (output ~stationary at median
   45), and output's value lives in its **distribution/tail**, which collapsing to any point loses equally. The
   normalized share *rising* 62.8→69.0% is a renormalisation artifact, not a real worsening.

3. **Did prompt-length attribution decrease?** **Yes** — raw planner-value down **36%** (14 758 → 9 400); the
   recent prompt median (982) is a better point than the global one (828).

4. **Did the improved forecaster close the gap to oracle?** **Partially (58.5%)** — better than single-median
   (0%) but **below the already-shipped PR #113 scenario ensemble (96.7%)**, and in the combined run it was
   **−1.68% vs current main and Pareto-unsafe** (it picked int4). **Not a robust win.**

5. **Did burstiness modelling matter?** **Marginally.** interarrival_cv is ~12% of forecast value and was held
   constant in the attribution; the forecaster emits burstiness scenarios but they were not the decisive lever
   on this trace.

6. **Did price-aware clock shaping produce Pareto-safe savings?** **Yes.** Downclocking memory-bound decode is
   Pareto-safe in **all** tested regimes (`decode_factor = 1.000` → ~0 latency cost, ~19% power cut), and the
   dollar saving **scales with price (≈11× larger at PJM p90 than p10)** through the real
   `energy × electricity_price` term. But small in absolute terms (~3% of cost at p90; depreciation dominates).

7. **Did electricity price become decision-relevant once clock/power shaping was modelled?** **Yes, in
   mechanism.** Feeding the real PJM p90 price shifts the live MPC's **downclock fraction 0.0 → 0.5**. It reads
   0.0% in the default MPC only because the Azure window feeds a **constant cheap** price and the saving is
   below the action-reorder threshold.

8. **New best gp/$ vs the strongest SLA-aware baseline?** The **current main MPC**: **+57–60%** (Track A
   median +60%, Pareto-safe in 7/8; Track E window +57.5%). The token-shape forecaster did **not** raise this.

9. **What is still simulator-inferred?** All magnitudes (bounded windows). Specifically: the +82.1% was
   oracle-planning; the roofline model treats memory-bound decode as **exactly** clock-independent
   (`decode_factor = 1.0` → downclock latency cost = 0, an **upper bound** on its attractiveness); energy /
   SLA / int4 quality-risk are simulator physics. Real serving telemetry is needed to confirm.

10. **What should be built next?** (a) **Wire a real diurnal price series into the planner's period frames** —
    converts the Pareto-safe-but-dormant clock lever into a selected action (highest-value, lowest-risk).
    (b) A **per-request output-length distributional predictor** — the dominant 62.8% lever, *not* closable by
    recency quantiles. (c) **Real serving telemetry** to attribute world-model fidelity and validate the
    clock-independence assumption. (d) Keep the **token-shape forecaster opt-in / off by default** (the
    `scenario_builder` hook); prefer the PR #113 scenario ensemble.

## What shipped

- **`token_shape_forecaster.py`** — recent-empirical token-shape scenarios (4 families) + a planner projection
  that drops into the controller's `scenario_builder` seam (opt-in; default off → behaviour unchanged).
- **`price_series.py`** — real PJM/ERCOT/CAISO day-ahead prices in $/kWh + percentile/diurnal helpers; EIA /
  ENTSO-E / 5-min real-time honestly marked **ABSENT** (never fabricated).
- **Diagnostics** (no controller/forecaster/simulator change beyond the opt-in hook): `diagnose_wide_validation`,
  `diagnose_token_shape_gap`, `diagnose_price_aware_clock`, `diagnose_combined`.
- **Tests:** `test_token_shape_forecaster.py` (7), `test_price_aware_clock.py` (5) — no leakage, deterministic
  quantiles, burstiness, price-only-through-energy, regime-dependent latency, Pareto gate blocks SLA-shedding.
