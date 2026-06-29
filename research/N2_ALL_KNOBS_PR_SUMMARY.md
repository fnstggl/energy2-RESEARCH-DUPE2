# PR — N2 SLA-Slack Power Arbitrage + All-Knobs MPC Backtest + Regret Audit — Summary

Makes **SLA slack an explicit, diagnostic-visible power-arbitrage budget** and proves N2 (spend slack to
downclock online serving work, saving electricity dollars while SLA stays in budget) — as a **decomposition of
the existing reward, never a bonus**. The clock→power→cost channel already drove the objective (PR #117), so
the reward, cost model, and Pareto gate are **byte-identical**. Built on `main` with PR #117 merged.

## 16 questions, answered honestly

**1. Did N2 become a real MPC behaviour?** Yes. With `electricity_price_aware=True` the planner makes
**price-appropriate clock choices in both directions** — downclocks at high price (pjm·expensive → `low×3`),
drops the upclock at medium price (ercot → `base×3`), upclocks at low price (caiso → `high×2`, cheap energy for
throughput). All raise gp/$ vs. the price-unaware arm.

**2. Did the controller treat SLA slack as a power-arbitrage budget?** Yes, explicitly: SLA slack
(`sla_target − completion-latency tail`) is now computed on the KPI/PeriodOutcome and recorded per decision;
the fixture sweep shows the clock is governed by the **slack/load budget** — downclock while slack exists (more
the higher the price), stop/upclock when saturation exhausts it. The Pareto gate enforces the stop.

**3. Did N2 work on online latency-bound serving without delaying it?** Yes. `serving_time_shifted = False` is
an invariant; N2 only changes the *power of work that runs now*. Deferrable time-shifting is a separate ledger
and is excluded (`deferrable_shifted = False` online).

**4. gp/$ value from N2?** `n2_dvfs − no_n2`: **+4859 / +5784 gp/$ (≈+1.5%) in high-price PJM windows**,
+1180 ERCOT, +1539 CAISO, **0** in low-price volatile windows; **mean +2227, median ≈ +1360, best +5784**.
**DIRECTIONAL_ONLY** — SLA-neutral within the MPC family, but not headline-safe.

**5. Value from deferrable shifting?** **0 to serving gp/$** (separate ledger; never steals serving). Valid
energy-cost saving (#117 dedicated 24 h validation): **4.7% PJM / 19.4% ERCOT / 5.4% CAISO**, 0 missed, $0 flat
price. SIMULATOR_INFERENCE; never blended into serving gp/$.

**6. Total all-knobs gp/$ vs. the strongest SLA-aware baseline?** Clock-focused real-price+N2 arms reach
**333–398k gp/$ but sit BELOW the SLA-aware baseline on the Pareto frontier in every window** → no headline.
The full **adaptive** all-knobs search is **heavy at hourly cadence** (one cell ran >2.5 min without
completing — the #117 intractability) → the all-knobs total is not tractable in a bounded run and is deferred.

**7. Median result?** N2 lift median ≈ **+1360 gp/$**; worst window **0**.

**8. Best-window result?** **+5784 gp/$ (pjm·volatile)** — labelled best-window, **not headline-safe**.

**9. What is headline-safe?** **Nothing.** 0/6 cells pass `pareto_sla_not_worse` (the MPC family sits above the
baseline's SLA-violation rate). The gate keeps the headline honestly False.

**10. What is directional-only?** The N2 serving lift (+4859/+5784 etc.) — a real causal within-MPC isolation
that fails the vs-baseline SLA clause. The deferrable saving is FIXTURE_ONLY.

**11. Remaining gap to oracle?** `oracle − n2`: **median ≈ +1580 gp/$, up to +7520 (ercot·expensive)**;
near-zero where price is unambiguous (pjm·expensive +102). Scenario planning is **not** a robust win (median
≈ −600, one −8952 loss) — kept off.

**12. Forecast variables explaining regret?** Primarily **arrival_rate** and **output_length** (load + token
shape → queue/SLA + roofline regime); electricity price contributes little (published day-ahead). `kv_reuse`,
`queue_pressure`, `sla_pressure` are not consumed forecasts.

**13. Simulator components robust enough?** Price path, cost path (energy×price), Pareto gate, work
conservation, SLA-slack computation, "N2-is-a-decomposition" — **ROBUST_ENOUGH_FOR_CURRENT_CLAIM**.

**14. Components that remain directional?** DVFS power curve, **memory-bandwidth-bound-decode = clock-independent
latency** (makes N2 an **upper bound**), completion-tail model, deferrable workload, spec-decode/int4 magnitudes
— **DIRECTIONAL_SIMULATOR_INFERENCE**.

**15. Production telemetry needed?** Real GPU **power telemetry**, real **per-request output length**, real
**cache-hit** rates, **thermal/power-cap** behaviour, **true demand charges** — **NEEDS_PRODUCTION_TELEMETRY**.

**16. What should the next PR improve?** **Real GPU power telemetry** — it closes the single dominant magnitude
assumption (the DVFS curve + clock-independent-decode upper bound) behind every electricity dollar; then a
coarser-cadence adaptive all-knobs sweep to measure the all-knobs total.

## Verification

- `ruff` clean; **N2 fixtures (11) + 78 regression tests** pass; reward/cost/Pareto byte-identical (defaults
  flat-price-identical; the 12 `test_electricity_controller` pass).
- N2 backtest: **48/48 cells COMPLETED, 0 TIMEOUT, 0 FAILED** (the runner cannot jam).
- Artifacts: `checkpointed_n2_backtest.json`; docs: audit, robustness, decision-diagnostics, all-knobs results,
  regret/attribution, electricity-only value.
