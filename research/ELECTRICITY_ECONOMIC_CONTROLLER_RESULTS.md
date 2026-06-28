# Electricity Economic Controller — Results

**Scope of this PR: land the electricity world-model *infrastructure* + a *bounded* causal validation. The full
historical PJM/ERCOT/CAISO backtest is explicitly DEFERRED (too heavy — see below). No headline electricity
goodput/$ saving is claimed yet.**

## Implementation — complete

| component | status | module |
|--|--|--|
| ElectricityState (per-period price/percentile/spike/forecast/provenance) | ✅ | `electricity.py` |
| PowerState (clock→watts→J→kWh→$ ledger) | ✅ | `electricity.py` |
| Real diurnal price frames (opt-in, flat-identical default) | ✅ | `training.build_mpc_inputs(electricity_market=…)` |
| Per-period electricity-price override in cost | ✅ | `simulate_period(energy_price_per_kwh=…)`, `run_period_episode(electricity_prices=…)` |
| Price-aware clock/DVFS planning path | ✅ | `controller.electricity_price_aware` (rollout prices each horizon step) |
| DeferrableWorkState + price-aware scheduler | ✅ | `deferrable.py` |
| Decision-diagnostics electricity fields | ✅ | `controller` Decision.forecast["electricity"] |
| Region→market registry | ✅ (pre-existing) | `region_registry.py` |

All opt-in and **flat-price-identical by default** → production behaviour unchanged unless switched on.
Persistent state clones/advances with `CanonicalWorldState`. Tests: `tests/test_electricity_controller.py` (12).

## Full historical backtest — SKIPPED (TOO_HEAVY), deferred to a follow-up

The 5-arm full-week **hourly** sweep (`scripts/diagnose_electricity_controller.py`) did **not** complete and is
marked **SKIPPED / TOO_HEAVY**. Root cause: electricity needs an **hourly** cadence for the diurnal price to
vary period-to-period, but at hourly periods the eval replays a *full hour* of real requests each (~60× the
request volume of the 60s-period diagnostics the harness was tuned for), and the world-state MPC runs a full
adaptive search per decision. A single market ran >20 min without emitting. Capping per-hour request volume +
a single input build helped but the all-arm sweep remains impractical in one shot.

**Follow-up (separate PR):** checkpointed backtest — persist after each (market, arm, period) cell; smaller
cells (a few hours per cell); a hard per-cell runtime cap; resume-from-checkpoint; run markets/arms
incrementally rather than one long job. The script + per-period price plumbing are already in place; only the
runner needs checkpointing.

## Bounded smoke validation — the causal path, proven cheaply

`scripts/smoke_electricity_validation.py` proves the **mechanisms** at the **fixture level** (direct
`simulate_period` calls + the deferrable scheduler — no `build_mpc_inputs`, no MPC search, runs in seconds), so
it is not blocked by the hourly-cadence planning cost. Artifact:
`data/external/mpc_controller/electricity_smoke_pjm.json`. **PJM, all six PASS:**

| property | result (PJM, p10 $0.026 / p90 $0.281) | pass |
|--|--|--|
| **P1** real diurnal price path varies | 24 distinct hourly prices | ✅ |
| **P2** high price increases cost (energy × price) | cost $0.099 (p10) → $0.116 (p90) | ✅ |
| **P3** price-aware clock favours downclock at high price | low-clock gp/$ gain **+3281 at p90** vs **+403 at p10**, SLA unchanged (Pareto-safe) | ✅ |
| **P4** deferrable shifts to cheap hours under slack | price-aware **$0.115** vs asap **$0.655** (avg paid $0.02), 0 missed | ✅ |
| **P5** flat price → no fake shifting value | price-aware == asap ($0.575) | ✅ |
| **P6** serving SLA not violated for free | no spare → 0 completed / 8 missed (deferred, never steals capacity) | ✅ |

P3 is a fixture-level proof that the gp/$-optimal clock is **price-dependent** (downclock is a Pareto-safe gp/$
win whose advantage scales with price), so a gp/$-maximising price-aware planner downclocks more when power is
expensive. The independent full-MPC confirmation is PR #115 Track D/E (downclock fraction 0.0→0.5 at PJM p90).

## Honest bottom line

- **Infrastructure is complete and causal:** real prices flow into planner frames and per-period cost; the
  clock/DVFS planner can price each horizon step; deferrable work shifts to cheap hours under deadline slack.
- **The causal path is proven in fixtures + the bounded smoke**, not via a full backtest.
- **No headline electricity saving is claimed** — quantifying the gp/$ value (and whether electricity
  attribution rises above PR #114's 0%) requires the deferred checkpointed backtest.
- Magnitudes that *are* shown are simulator-inferred; the DVFS power curve and the entire deferrable workload
  are `SIMULATOR_INFERENCE` (see `ELECTRICITY_PRODUCTION_REALISM_AUDIT.md`).
