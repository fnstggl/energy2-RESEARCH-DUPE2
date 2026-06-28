# Electricity Economic Controller — PR Summary

Lands the **electricity world-model infrastructure** + a **bounded causal validation**, built on stable `main`
(the regressed PR #115 token-shape forecaster is NOT present / NOT the planner). All opt-in, flat-price-identical
by default. The full historical backtest is **deferred (TOO_HEAVY)** to a checkpointed follow-up. **No headline
electricity goodput/$ saving is claimed.**

## The 16 questions

1. **Electricity data found/used?** Real day-ahead ISO prices in `data/` (`$/MWh ÷1000 = $/kWh`), loaded by
   `price_series.py`; the canonical `region_registry.py` maps region→market. EIA is **absent** for price (it
   serves demand, not LMP); 5-minute RT is absent. Carbon data exists but is **not** wired (kept ABSENT).
2. **PJM available?** **Yes** — `data/pjm_us_east_dam.csv` (us-east).
3. **ERCOT available?** **Yes** — `data/ercot_us_south_dam.csv` (us-south).
4. **CAISO available?** **Yes** — `data/caiso_us_west_dam.csv` (us-west).
5. **Price→region mapping?** `region_registry.py`: us-east→PJM, us-south→ERCOT, us-west→CAISO (+ us-central→SPP).
6. **How is power modeled?** `PowerState` (electricity.py): `clock → power_w = TDP·(0.4+0.6·clock^2.4) →
   energy_J → kWh → × price → $`, cumulative ledger. The DVFS curve is `SIMULATOR_INFERENCE`; lever =
   clock-locking (NOT power-cap, which wouldn't engage on memory-bound decode → would book phantom savings).
7. **How is clock/DVFS modeled?** A live CONNECTED action (`clock_policy ∈ {base, low, high}`, since PR #111),
   now price-responsive: with `electricity_price_aware=True` the rollout prices each horizon step at the
   forecast price path.
8. **Is clock a production MPC action?** **Yes** — already connected; this PR makes it respond to real diurnal
   prices (opt-in; flat-price-identical default).
9. **Is DeferrableWorkState persistent?** **Yes** — clones/advances with `CanonicalWorldState`; jobs persist
   across periods; work is conserved.
10. **Is energy shifting a production MPC action?** Implemented as a persistent pool + price-aware look-ahead
    scheduler (run at the cheapest remaining period before the deadline). Validated in the bounded smoke; the
    workload is a conservative `SIMULATOR_INFERENCE` generator (no real deferrable trace exists).
11. **Did electricity become decision-relevant?** The **causal path is proven** (bounded smoke P3: the
    price-aware planner downclocks ≥ the not-price-aware one; P2: high price costs more). The full attribution
    re-run that would *quantify* "above 0%" is **deferred** with the backtest — **no numeric claim is made**.
12. **Did goodput/$ improve?** **No headline claimed** (the all-arm gp/$ comparison is the deferred backtest).
13. **Pareto-safe?** The Pareto gate is **unchanged**; flat-price reproduces the baseline exactly; deferrable
    is serving-dominated (can't steal serving capacity). No headline gp/$ exists to gate yet.
14. **Did attribution change after actions?** **Deferred (TOO_HEAVY)** — the controller now *consumes* a
    varying price and records it per decision, so the re-run will be meaningful, but it is not run here.
15. **What remains simulator-inferred?** The DVFS power curve, the entire deferrable workload, all magnitudes,
    and the full historical backtest.
16. **What production telemetry would improve this?** Per-GPU power telemetry (to replace the DVFS curve and
    validate the memory-bound `decode_factor=1.0`), a real deferrable-job trace, nodal datacenter→price-node
    basis, and live SLO telemetry.

## What shipped (all opt-in, flat-price-identical default)

`electricity.py` (ElectricityState, PowerState, PriceProfile), `deferrable.py` (DeferrableWorkState +
price-aware scheduler), real diurnal price frames + per-period price override in cost
(`training.build_mpc_inputs(electricity_market=…)`, `simulate_period(energy_price_per_kwh=…)`,
`run_period_episode(electricity_prices=…)`), price-aware clock planning (`controller.electricity_price_aware`),
decision-diagnostics electricity fields, world-state ElectricityState/PowerState/DeferrableWorkState.

Docs: `ELECTRICITY_WORLD_MODEL_GAP_AUDIT`, `ELECTRICITY_PRODUCTION_REALISM_AUDIT`,
`ELECTRICITY_WORLD_MODEL_ARCHITECTURE`, `ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS` (+ the bounded-smoke block and
deferred-backtest plan), and the pointer result docs. Tests: `tests/test_electricity_controller.py` (12).

## Honest bottom line

Infrastructure complete and causal; the causal path is proven in fixtures + a bounded smoke; the full
historical backtest and the electricity-attribution re-run are **deferred (too heavy)** to a separate
checkpointed PR; **no electricity savings headline is claimed**.
