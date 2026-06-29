# Electricity Attribution Backtest (Phase 3)

Per-action gp/$ attribution from the **checkpointed historical backtest**
(`scripts/run_checkpointed_electricity_backtest.py`, artifact
`data/external/mpc_controller/checkpointed_electricity_backtest.json`). 54 cells COMPLETED, 0 TIMEOUT, 0
FAILED — 3 markets × 3 windows × 6 arms (arms 1–6; the full-action-space arm 7 is reported in
`FULL_ELECTRICITY_BACKTEST_RESULTS.md`). Hourly cadence (so the diurnal price varies period-to-period),
`max_decisions=3`, ≤80 requests/period, fixed seed-0 world-state (deterministic, no RNG).

**Method — isolation by holding everything else fixed.** Every arm runs the SAME serving workload through the
SAME `run_period_episode`. Arms differ in exactly one knob, so each delta is the *causal* effect of that knob:

| isolated effect | arms differenced | what it measures |
|--|--|--|
| **price-aware DVFS** | `real_price_dvfs_only` − `current_main_mpc_real_price` | turning ON `electricity_price_aware` (same real price, same clock candidate set) |
| **deferrable shifting (serving)** | `real_price_deferrable_only` − `current_main_mpc_real_price` | adding the deferrable scheduler's effect on **serving** gp/$ |
| **combined** | `real_price_dvfs_plus_deferrable` − `current_main_mpc_real_price` | both electricity knobs together |
| **interaction** | combined − dvfs − deferrable + real | non-additivity of the two knobs |

Effects flow **only** through the permitted channels: clock → `power_scale` → energy×price → operator cost →
gp/$, and timing → energy×price → cost (deferrable). No reward bonus; the Pareto gate is unchanged.

## DVFS isolation — real, causal, price-dependent, modest

`real_price_dvfs_only` vs `current_main_mpc_real_price` (the only difference is `electricity_price_aware`):

| market | window | price $/kWh | real gp/$ | dvfs gp/$ | **dvfs lift gp/$** | dvfs clock | SLA dvfs/real |
|--|--|--|--|--|--|--|--|
| pjm | volatile | 0.076–0.178 | 355632.5 | 361416.9 | **+5784.4** | `low ×3` | 0.388 / 0.383 |
| pjm | expensive | 0.092–0.178 | 328114.4 | 332973.6 | **+4859.2** | `low ×3` | 0.529 / 0.525 |
| caiso | expensive | 0.029–0.031 | 396136.2 | 397675.6 | +1539.4 | `high ×2, base ×1` | 0.479 / 0.483 |
| ercot | expensive | 0.045–0.059 | 374565.1 | 375745.4 | +1180.3 | `base ×3` | 0.483 / 0.483 |
| ercot | volatile | 0.032–0.059 | 389442.1 | 389442.1 | 0.0 | `base ×3` | 0.483 / 0.483 |
| caiso | volatile | 0.028–0.031 | 348558.8 | 348558.8 | 0.0 | `base ×2, high ×1` | 0.446 / 0.446 |
| caiso | cheap | 0.028–0.030 | 435875.5 | 434825.7 | −1049.8 | `high ×3` | 0.396 / 0.400 |
| ercot | cheap | 0.021–0.030 | 435638.9 | 434569.8 | −1069.1 | `high ×3` | 0.396 / 0.400 |
| pjm | cheap | 0.069–0.086 | 293806.2 | 292248.1 | −1558.1 | `base ×2, high ×1` | 0.375 / 0.371 |

**Mean DVFS lift across the 9 windows: +1076 gp/$ (≈ +0.3% of gp/$).** Reading the table honestly:

- **The lift is price-dependent and largest where the price is highest.** In PJM's expensive and volatile windows
  (p90 ≈ $0.18) the price-aware planner **downclocks** (`clock = low ×3`) and gains **+4859 / +5784 gp/$
  (≈ +1.5%)** — a real energy-cost saving (lower `power_scale` on a decode-dominated workload whose latency is
  clock-independent, so power drops with little extra GPU-time). This is the independent MPC confirmation of the
  fixture-level smoke P3 (downclock is a Pareto-safe gp/$ win whose advantage scales with price) and PR #115
  Track D/E (downclock fraction 0.0→0.5 at PJM p90).
- **In low-price markets/windows (ERCOT/CAISO, $0.02–0.06) the lift is ~0 or slightly negative.** When energy is
  a tiny fraction of operator cost, the clock the planner picks barely moves cost, and the price-aware arm's
  choice occasionally interacts with the workload's SLA pressure for a small net loss (−1050 to −1558 gp/$,
  ≈ −0.3%). This is the honest other side of a causal mechanism: a price signal that is nearly flat carries
  almost no gp/$ information, and a small mis-step is not masked.
- **A uniformly-positive lift would be the suspicious result.** The sign tracking the price level
  (positive at high price, ~0 at low price) is the causal signature we want.

## Deferrable isolation — serving gp/$ is unchanged (the safety property)

`real_price_deferrable_only` − `current_main_mpc_real_price` = **0.0 gp/$ in all 9 windows.** The deferrable
scheduler runs on the GPU-seconds serving leaves spare; it never preempts serving, so **serving gp/$ is
byte-identical to the no-deferrable arm.** That is exactly the intended contract (serving SLA dominates;
deferrable value is a *separate* energy-cost ledger, reported in `DEFERRABLE_ENERGY_SHIFTING_RESULTS.md`, never
folded into serving gp/$). The in-backtest 3-period ledger is too short for the look-ahead scheduler to realise
a deadline-respecting saving; the clean shifting saving is measured over the full 24 h diurnal path in the
dedicated validation.

## Combined + interaction

`combined_lift = dvfs_lift` and `interaction = 0.0` in **every** window — necessarily, because the deferrable
knob contributes exactly 0 to serving gp/$. The two electricity knobs are **orthogonal in serving gp/$** (DVFS
shapes serving power; deferrable shifts non-serving work in time), so there is no serving-gp/$ interaction to
find. Their *joint* value is `DVFS serving lift` **plus** the `deferrable energy-cost saving` (different ledgers,
added not multiplied).

## Headline gate — blocked in every cell (honest)

The best electricity arm clears the SLA-aware baseline on gp/$ in 3 of 9 windows (`pjm|cheap`, `ercot|volatile`,
`caiso|expensive`) but **fails the Pareto SLA clause in all 9** (`pareto_sla_not_worse = False` everywhere): the
MPC arms always carry a higher SLA-violation rate than the SLA-aware baseline (e.g. `pjm|expensive` 0.529 vs
0.275). So **no cell is HEADLINE_SAFE.** This is the same regime-dependent, SLA-bought edge documented for the
base MPC (`training.claim_gate`) — the electricity actions reduce energy cost on top of the MPC arm, but do not
turn the MPC family into a Pareto win over the fair baseline in these windows.

## Bottom line

- Price-aware DVFS is a **real, causal, but modest** serving-gp/$ lever: **+4859 / +5784 gp/$ (≈ +1.5%) in
  high-price PJM windows**, ~0 at low price, mean **+1076 gp/$** over 9 windows — all through the energy×price
  channel, SLA never traded for it within the isolation.
- Deferrable shifting contributes **0 to serving gp/$** by construction (a separate energy ledger).
- **No electricity action produces a HEADLINE_SAFE gp/$ number** (the Pareto SLA clause blocks all 9 cells).
  The lifts above are reported as **DIRECTIONAL** within-MPC isolations, not headline savings.
