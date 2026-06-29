# Electricity Backtest Plan (Phase 0)

Finishes what PR #116 deferred: a **checkpointed** historical electricity backtest that quantifies the gp/$
value of the electricity actions — without jamming (the PR #116 lesson: hourly MPC decisions are heavy, so the
runner must be cell-isolated, time-capped, and resumable). Built on `main` with PR #116 merged.

## Safety (confirmed before any run)

- **PR #116 is merged** (`electricity.py` / `deferrable.py` on main; merge `b5ae46e`).
- **PR #115 token-shape forecaster is absent** → not default behaviour.
- **Flat-price mode reproduces previous behaviour** (`build_price_profile(None)` → a single constant price;
  `electricity_price_aware` defaults `False`; `electricity_prices` defaults `None`).
- **Pareto gate unchanged** (`training.claim_gate` untouched).

## What is backtested

Seven arms on the SAME serving workload + Pareto gate, hourly cadence (so the diurnal price varies):

| arm | electricity_price_aware | realized price | deferrable |
|--|--|--|--|
| 1 `baseline_sla_aware` | — (fixed policy) | real | — |
| 2 `current_main_mpc_flat_price` | off | constant fleet | — |
| 3 `current_main_mpc_real_price` | off | real | — |
| 4 `real_price_dvfs_only` | **on** | real | — |
| 5 `real_price_deferrable_only` | off | real | **price-aware** |
| 6 `real_price_dvfs_plus_deferrable` | **on** | real | **price-aware** |
| 7 `all_knobs_current_aurelius` | on | real | price-aware (all live action surfaces) |

Non-electric knobs are held at the controller defaults across arms 2–7 so the electricity effect is isolated
(arm 7 keeps them live for the total-Aurelius number).

## Which actions are live

Clock/DVFS is a **true MPC action** (selected in `ActionBundle`, priced against the real price path).
Deferrable energy-shifting is a **planner-visible scheduler** (not an `ActionBundle` knob) — its value is a
separate energy-cost ledger, reported separately (see `ELECTRICITY_MPC_ACTION_SURFACES.md`). Region shifting is
**skipped** (no multi-region fleet).

## What is checkpointed

A **cell** = (market, window, arm). After **every** cell the runner writes
`data/external/mpc_controller/checkpointed_electricity_backtest.json` with the cell's status
(`COMPLETED` / `TIMEOUT` / `FAILED` / `SKIPPED`), runtime, and metrics. `--resume` skips completed cells;
`--force` reruns. Each cell runs in an **isolated subprocess with a hard timeout** — a slow cell is killed and
marked `TIMEOUT`, never jamming the run. Inputs are built once per market and shared (fork COW), separate from
per-cell evaluation.

## Headline-safe criteria

A number is **HEADLINE_SAFE** only if: the cell `COMPLETED` (not TIMEOUT/partial), the **Pareto gate passed**
(beats the fair baseline on gp/$ AND SLA not worse), the market+window+baseline are named, and it is reported as
that specific cell (never an unlabelled "up to"). Otherwise it is `DIRECTIONAL_ONLY`, `FIXTURE_ONLY`,
`SKIPPED`, or `FAILED`. Deferrable value is the energy-cost saving on a `SIMULATOR_INFERENCE` workload — never a
serving-gp/$ headline.

## Runtime safeguards

`--quick` (1 market, 1 short window, 2–3 arms, ≤3 decisions, minutes) vs `--full` (all markets, multiple
windows, checkpointed/resumable). Flags: `--markets --windows --arms --max-decisions
--max-requests-per-period --cell-timeout-seconds --resume --force`. Quick mode is run FIRST to prove the runner;
full mode is bounded and resumable. No tuning; no Pareto-gate weakening; no headline unless a cell completes and
the gate passes.
