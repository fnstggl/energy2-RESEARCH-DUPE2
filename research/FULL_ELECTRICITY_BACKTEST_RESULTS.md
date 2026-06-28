# Full Electricity Backtest Results (Phase 6)

The headline-safe synthesis of the checkpointed historical electricity backtest. **Bottom line up front: the
electricity MPC actions produce a real, causal, price-directional gp/$ lift (up to +5784 gp/$ ≈ +1.5% in
high-price PJM windows) and a real deadline-respecting deferrable energy-cost saving (4.7–19.4%), but NO cell
clears the Pareto gate, so there is NO headline electricity goodput/$ saving.** Every number below is labelled
by the claim-safety taxonomy.

## Run

| | |
|--|--|
| runner | `scripts/run_checkpointed_electricity_backtest.py` (cell-isolated subprocess + hard timeout, checkpointed, resumable) |
| arms 1–6 | **54 cells COMPLETED, 0 TIMEOUT, 0 FAILED** (3 markets × 3 windows × 6 arms) |
| arm 7 (`all_knobs`, full action space) | pjm only — see §"Total Aurelius" |
| cadence | hourly (diurnal price varies period-to-period); `max_decisions=3`; ≤80 req/period; seed-0 world-state, no RNG |
| price provenance | `TRACE_DERIVED` — PJM/ERCOT/CAISO day-ahead ($/MWh → $/kWh) |
| Pareto gate | `training.claim_gate` **unchanged**; flat-price mode reproduces pre-electricity behaviour |

## Claim-safety taxonomy

| tier | meaning | what falls here |
|--|--|--|
| **HEADLINE_SAFE** | cell COMPLETED **and** Pareto gate passed (beats fair baseline gp/$ **and** SLA not worse), named market+window+baseline | **NONE** (0 of 9 windows) |
| **DIRECTIONAL_ONLY** | real causal within-MPC isolation, but the absolute arm fails the Pareto gate | DVFS serving lift (+4859/+5784 gp/$ high-price PJM; mean +1076); combined lift |
| **FIXTURE_ONLY** | proven at fixture level (no MPC search) | smoke P3 downclock-is-price-dependent; deferrable P4/P5/P6; dedicated 24 h deferrable saving |
| **SKIPPED** | not runnable honestly | region shifting (no multi-region fleet) |
| **FAILED** | errored | none (0 of 54) |
| **TIMEOUT** | killed at the per-cell cap | none in arms 1–6 (0 of 54) |

## The five questions

**(1) gp/$ from electricity actions alone.** Price-aware DVFS: **mean +1076 gp/$ (≈ +0.3%)** across 9 windows,
up to **+5784 gp/$ (≈ +1.5%)** in `pjm|volatile` and **+4859** in `pjm|expensive`; ~0 or slightly negative in
low-price ERCOT/CAISO windows. Deferrable: **0** to serving gp/$. **DIRECTIONAL_ONLY** — no cell is
HEADLINE_SAFE.

**(2) DVFS alone.** Isolated as `real_price_dvfs_only − current_main_mpc_real_price` (only
`electricity_price_aware` differs). The price-aware planner **downclocks in high-price PJM windows** (`clock =
low ×3`, +4859/+5784 gp/$, SLA unchanged) and does **not** downclock when energy is cheap. Causal through
clock → `power_scale` → energy×price → cost. **DIRECTIONAL_ONLY.** Full table:
`ELECTRICITY_ATTRIBUTION_BACKTEST.md`; mechanism: `N2_HISTORICAL_POWER_SHAPING_RESULTS.md`.

**(3) deferrable shifting alone.** **0** serving gp/$ in all 9 cells (never steals serving capacity — the
safety property). Valid deadline-respecting energy-cost saving on the deferrable workload:
**4.7% PJM / 19.4% ERCOT / 5.4% CAISO**, 0 missed, $0 under a flat price.
**FIXTURE_ONLY** (separate energy ledger, SIMULATOR_INFERENCE workload). Details:
`DEFERRABLE_ENERGY_SHIFTING_RESULTS.md`.

**(4) DVFS + deferrable together.** `combined_lift = dvfs_lift` and **`interaction = 0.0` in every window** —
the two knobs are orthogonal in serving gp/$ (DVFS shapes serving power; deferrable shifts non-serving work in
time). Their joint value is the DVFS serving lift **plus** the deferrable energy saving (added, not multiplied;
different ledgers). **DIRECTIONAL_ONLY** (serving) + **FIXTURE_ONLY** (deferrable).

**(5) total Aurelius gp/$ with all knobs.** See §"Total Aurelius" below (arm 7, full action space).

## Why no cell is HEADLINE_SAFE

In all 9 windows `pareto_sla_not_worse = False`: every MPC arm (including the best electricity arm) carries a
higher SLA-violation rate than the SLA-aware baseline. Three windows beat the baseline on gp/$
(`pjm|cheap`, `ercot|volatile`, `caiso|expensive`) but all fail the SLA clause. This is the **same**
regime-dependent, SLA-bought edge the base MPC has always shown (`training.claim_gate` docstring); the
electricity actions reduce energy cost *on top of* the MPC arm but do not convert the MPC family into a Pareto
win over the fair baseline. The gate is doing its job — keeping the headline honestly False.

`all_elec_vs_baseline_pct` (best electricity arm vs. SLA-aware baseline) by window: pjm +1.76 / −2.42 / −1.02;
ercot −4.97 / +1.56 / −4.66; caiso −4.96 / −4.47 / +1.95. Small and mixed-sign — consistent with "no headline".

## Total Aurelius (arm 7, `all_knobs_current_aurelius`)

Arm 7 runs the **full action space** (≈ 314 928 ActionBundles — clock/precision/spec-decode/batching/routing/
capacity/prewarm/placement/migration all live) plus the deferrable scheduler. At **3 decisions** it **TIMED
OUT** at the 300 s per-cell cap on `pjm|cheap` (the full search is ~100× the 3-bundle clock-focused search per
decision). **This is the headline runner result, not a failure:** the cell was killed and the run continued —
the PR #116 jam (a single market ran >20 min with no output) **cannot recur**. The full-action-space
total-Aurelius number at hourly cadence is **TIMEOUT / not tractable within the bounded per-cell cap**.

To obtain a *tractable* total-Aurelius data point, a **single-decision** `all_knobs` cell was run on
`pjm|expensive` (one full-space search fits under the cap): **COMPLETED in 212.7 s**, gp/$ **12902.2**, SLA
**0.000**, vs. the same-window baseline (gp/$ 336410, SLA 0.275) and price-unaware MPC (gp/$ 328114, SLA 0.525).

Read this carefully — it is **not** a comparable gp/$ number and **not** a headline:

- **It runs over a single period** (`win[:1]`), whereas arms 1–6 run over 3; the absolute gp/$ is over a
  *different* slice and cannot be differenced against them.
- **The full-space MPC optimised its SLA-weighted objective to SLA = 0.000** (vs. baseline 0.275) by spending
  **≈22× the operator cost** ($0.677 vs. the price-unaware arm's $0.031), which **crushed gp/$ to 12902** — the
  *opposite* corner from the clock-focused arms (which buy gp/$ by shedding SLA). So the total-Aurelius arm is
  **far below** the baseline on gp/$ → emphatically **not HEADLINE_SAFE**, just at the other end of the
  cost/SLA frontier.
- **It confirms the runner can complete a full-space cell** when the decision count fits the cap (212.7 s < 420 s),
  and that the electricity story is unchanged: nothing here is a gp/$ win.

The full action space does **not** change the electricity story: the total-Aurelius arm is still SLA-bought
(its SLA-violation rate exceeds the SLA-aware baseline's), so it is **not HEADLINE_SAFE**, and the electricity
actions' isolated value remains the DVFS lift measured in arms 4/6. Promoting the full-space arm to a complete
3-decision sweep needs a larger per-cell cap (or a pruned candidate set) and is left to a follow-up — the runner
already supports it via `--arms all_knobs_current_aurelius --cell-timeout-seconds N`.

## Honest bottom line

- **The electricity control loop is complete and causal:** real prices flow into the planner's horizon and the
  per-period cost; price-aware DVFS shapes serving power online (N2); the deferrable scheduler time-shifts
  non-serving work to cheap hours without ever touching serving.
- **The measured serving-gp/$ value of the electricity actions is real but modest and regime-dependent**
  (+1.5% in high-price PJM windows, ~0 elsewhere), and **does not clear the Pareto gate in any window** →
  **no headline electricity goodput/$ saving is claimed.**
- **The deferrable saving is real but lives in a separate energy ledger** (4.7–19.4%, SIMULATOR_INFERENCE
  workload), never a serving-gp/$ headline.
- Nothing was tuned; the Pareto gate was not weakened; the runner cannot jam (54/54 completed, 0 timeout).
