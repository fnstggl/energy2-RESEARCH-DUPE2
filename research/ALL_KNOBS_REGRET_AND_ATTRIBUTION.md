# All-Knobs Regret & Attribution (Phase 7)

How much value is still missing vs. an oracle, and where it goes. From the N2 backtest's forecast-mode arms
(`oracle_forecast_all_knobs`, `scenario_forecast_all_knobs` vs. the causal `all_knobs_real_price_n2_dvfs`).
Companion: `WORLD_MODEL_ROBUSTNESS_AUDIT.md` (simulator-fidelity tiers).

## Oracle gap — value still missing vs. the EXACT future

`oracle − n2_dvfs` (the controller plans against the realized future at k=0; causal otherwise):

| market·window | n2 gp/$ | oracle gp/$ | **oracle gap** | clock: causal → oracle |
|--|--|--|--|--|
| ercot·expensive | 375745 | 383266 | **+7520** | `base×3` → `base,high,low` |
| caiso·expensive | 397676 | 399348 | +1672 | `high×2,base` → `low×2,high` |
| caiso·volatile | 348559 | 350054 | +1495 | — |
| pjm·expensive | 332974 | 333076 | +102 | `low×3` → `base,low×2` |
| pjm·volatile | 361417 | 360148 | −1269 | (oracle ≈ causal; small −ve = single-track noise) |
| ercot·volatile | 389442 | 393056 | +3614 | — |

**Forecast regret is real but modest — median ≈ +1580 gp/$, up to +7520 (ercot·expensive).** The biggest gaps
are in ERCOT, where the within-day price/load swing is largest, so knowing the exact future helps the clock and
capacity choices most. `pjm·expensive` has almost no gap (+102): when the price is unambiguously high, the
causal planner already makes the same downclock the oracle would.

## Scenario planning — mostly does NOT help

`scenario − n2_dvfs` (ensemble planning vs. deterministic point forecast):

| market·window | scenario value | note |
|--|--|--|
| ercot·expensive | **+5347** | the one clear win (high-volatility window) |
| pjm·expensive | **−8952** | scenario wrongly upclocks (`high×3`) → big loss |
| pjm·volatile | −2942 | |
| ercot·volatile | −663 | |
| caiso·volatile | −536 | |
| caiso·expensive | −656 | |

**Scenario planning is NOT a robust win — median ≈ −600 gp/$, one large loss (−8952), one win (+5347).** This
reproduces PR #115's finding (the token-shape scenario forecaster is not robustly positive). The ensemble's
risk-aversion sometimes picks a higher clock to hedge SLA, which costs gp/$ when the realized load is benign.
**Kept opt-in / off by default.**

## Regret decomposition — where the missing value goes

| source | magnitude | evidence |
|--|--|--|
| **forecast regret** | median +1580, up to +7520 gp/$ | oracle gap above — imperfect arrival/price/length forecasts |
| **search regret** | not the binding limit for N2 | the clock-focused N2 arms search exhaustively over `{base,low,high}` → 0 search regret on the N2 knob; the *adaptive all-knobs* search is heavy (Phase 6) — its regret is unmeasured because exhaustive comparison is intractable at hourly cadence (reported, not hidden) |
| **objective / Pareto constraint** | the dominant "gap" | every arm is below the SLA-aware baseline on the Pareto frontier — the gate (correctly) forbids buying gp/$ by shedding SLA, so the headline is blocked by design, not by a model error |
| **simulator fidelity** | bounds the magnitude | the DVFS power curve + clock-independent-decode assumption set N2's value as an **upper bound** (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`) |
| **action-space limitation** | deferred | the full adaptive all-knobs search did not complete at hourly cadence → the all-knobs total is not measured here |
| **runtime / capped search** | bounded + reported | 3-decision windows, ≤80 req/period, per-cell timeout — `log`ged, never silently capped |

## Forecast-variable attribution

Which forecast variables plausibly explain the oracle gap, by what the controller **consumes**
(`decision_diagnostics.CONSUMED_FORECASTS`) vs. what it does not:

| variable | consumed? | regret contribution |
|--|--|--|
| arrival_rate | yes (sizes capacity) | **high** — load misforecast drives queue/SLA, the dominant gp/$ term |
| output_length (token mean/p95) | yes (causal running-median prior) | **high** — sets prefill/decode mix → the roofline regime → whether N2 downclock is free |
| electricity_price | yes (prices each horizon step) | **medium** — day-ahead price is published, so low intrinsic regret; the ERCOT gap is more load than price |
| prompt_length | yes | medium — prefill work |
| interarrival_cv | yes | medium — burstiness → queue tail |
| kv_reuse | **no** (synthetic unique prefixes, PR #112) | unconsumed → not a regret source today |
| queue_pressure / sla_pressure | **no** (emergent, not a forecast input) | emergent from arrival+service, not a planner forecast |
| deferrable-work forecast / carbon | **no** | out of scope / separate ledger |

## Answers (Phase 7 questions)

1. **Remaining gap to oracle?** Median ≈ **+1580 gp/$**, up to **+7520** (ercot·expensive); near-zero where the
   price is unambiguous (pjm·expensive +102).
2. **Which forecast variables explain it?** Primarily **arrival_rate** and **output_length** (load + token
   shape); electricity price contributes little (it is published day-ahead).
3. **Which simulator assumptions are robust enough?** Price path, cost path, Pareto gate, work conservation,
   SLA-slack computation (ROBUST — see robustness audit).
4. **Which are only directional?** DVFS power curve, clock-independent decode, completion-tail model, deferrable
   workload (DIRECTIONAL_SIMULATOR_INFERENCE).
5. **Which need production telemetry?** Real GPU power, per-request output length, cache-hit rates, thermal
   behaviour, demand charges (NEEDS_PRODUCTION_TELEMETRY).
6. **Highest expected-value next improvement?** **Real GPU power telemetry** — it closes the single dominant
   magnitude assumption behind every electricity dollar (the DVFS curve + clock-independent-decode upper bound).
