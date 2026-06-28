# Electricity World-Model Architecture

How electricity became a first-class economic control domain in the Aurelius canonical world model — not a
passive cost label. Built on stable `main` (the regressed PR #115 token-shape forecaster is **not** present and
is **not** the planner). Every mechanism is opt-in and **flat-price-identical by default**, so production
behaviour is unchanged unless the electricity actions are switched on and honestly validated. Audits:
`ELECTRICITY_WORLD_MODEL_GAP_AUDIT.md`, `ELECTRICITY_PRODUCTION_REALISM_AUDIT.md`.

## Components

```
region (FleetState.region) ──► region_registry ──► market (PJM/ERCOT/CAISO)
                                                       │
price_series.py  (real day-ahead CSV, $/MWh ÷1000 = $/kWh) ─► diurnal profile (hour-of-day mean)
                                                       │
                          ElectricityState  (per-period: price, percentile, spike, volatility, forecast/realized,
                          (electricity.py)              forecast_error, provenance)
                                                       │
build_mpc_inputs(electricity_market=…) ─► price_by_cycle ─► PeriodFrame.electricity_price ─► ForecastingModel
                                                       │                                     forecasts a varying
                                                       ▼                                     price PATH
              MPC horizon rollout (controller, electricity_price_aware) ─► clock/DVFS chosen vs the price path
                                                       │
PowerState (electricity.py): clock → power_w = TDP·(0.4+0.6·clock^2.4) → energy_j → kWh → × price → $
                                                       │
        cost_model.operator_cost(energy_price_per_kwh=per-period price, power_scale=clock power_factor)
                                                       │
              DeferrableWorkState (deferrable.py): shiftable pool + price-aware look-ahead scheduler
```

### 1. ElectricityState (`electricity.py`)
Per-period electricity view that clones with `CanonicalWorldState` (deepcopy): `market`, `region`,
`current_price`, `forecast_price`, `price_percentile`, `volatility`, `spike`, `forecast_error`, `provenance`.
Built from `PriceProfile` (the real diurnal price path + market percentiles). `TRACE_DERIVED` for real markets,
`SIMULATOR_INFERENCE` for the flat fallback.

### 2. PowerState (`electricity.py`)
Formalises the power/energy accounting the world simulator already computes (`PeriodOutcome.power_w / energy_j`
from the DVFS roofline action): `clock_state → power_w → energy_kWh → × price → $`, cumulative ledger. The DVFS
power curve is `SIMULATOR_INFERENCE`. **Lever = clock-locking** (which shapes memory-bound decode energy) — NOT
power-capping (a cap below decode draw never engages → would book phantom savings; see realism audit).

### 3. Clock/DVFS (already a live CONNECTED MPC action — `actions.py` `clock_policy ∈ {base, low, high}`)
Wired since PR #111. When `electricity_price_aware=True`, the horizon rollout prices each step at the
**forecast price path** (`traj.point("electricity_price", k)`), so the controller chooses clock against real
diurnal prices. Memory-bound decode is clock-independent (`decode_factor = 1.0`) so downclocking it is ~free
in latency but cuts power — the lever electricity exploits.

### 4. DeferrableWorkState + energy shifting (`deferrable.py`)
Persistent shiftable-work pool (batch/embeddings/fine-tune/maintenance) — a **conservative
SIMULATOR_INFERENCE** generator (no real trace). The `price_aware` scheduler runs a job at the cheapest
remaining period before its deadline (causal look-ahead over known day-ahead prices). Invariants enforced in
code + tests: work is **conserved** (delayed jobs persist, never vanish), missed deadlines are **penalised**
(no free dodging), **serving dominates** (0 spare ⇒ defer), and **flat price ⇒ price_aware == asap cost** (no
fake shifting value).

### 5. Economic objective
Electricity cost flows ONLY through `energy_kWh × price` in `cost_model.operator_cost`, now with a **per-period
price** (`simulate_period(energy_price_per_kwh=…)`, `run_period_episode(electricity_prices=…)`). Deferrable
energy cost + missed-deadline penalty are a **separate ledger** (not folded into serving gp/$, which would
confound). `$/MWh → $/kWh` is a ÷1000 conversion (tested).

## Price data sources & provenance

| market | region | data | status |
|--|--|--|--|
| PJM | us-east | `data/pjm_us_east_dam.csv` | TRACE_DERIVED (day-ahead, hourly) |
| ERCOT | us-south | `data/ercot_us_south_dam.csv` | TRACE_DERIVED |
| CAISO | us-west | `data/caiso_us_west_dam.csv` | TRACE_DERIVED |
| EIA | — | adapter present | **ABSENT** (EIA serves demand, not price) |
| 5-min real-time | — | — | **ABSENT** (only hourly DAM committed) |

Real prices are aligned to the trace by **hour-of-day** (the diurnal profile), because the Azure serving trace
carries no wall-clock timestamp to calendar-align against. Live APIs are an interchangeable provider of the
same `price_series` interface; offline we fall back to the committed CSVs and record the provenance.

## Limitations / unsupported (skipped rather than faked)

Demand charges (gameable $/kW-month peak), GPU power-caps (don't engage on memory-bound decode), dynamic
cooling/COP (kept flat PUE 1.3×), carbon-as-reward (data is one-region + unwired → ABSENT), region-shifting of
deferrable work (no multi-region fleet), 5-minute RT prices. What real production telemetry would improve this:
per-GPU power telemetry (to replace the DVFS power curve and validate `decode_factor=1.0`), a real deferrable
job trace (to replace the synthetic pool), nodal datacenter→price-node basis, and live SLO telemetry.

## Honesty contract

Opt-in, flat-price-identical defaults; cost changes only through energy × price; SLA stays Pareto-safe; missed
deadlines penalised; no reward bonuses; provenance-labelled. Tests: `tests/test_electricity_controller.py`.
Results: `REAL_PRICE_DVFS_VALIDATION.md`, `ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md`,
`ELECTRICITY_ATTRIBUTION_AFTER_ACTIONS.md`.
