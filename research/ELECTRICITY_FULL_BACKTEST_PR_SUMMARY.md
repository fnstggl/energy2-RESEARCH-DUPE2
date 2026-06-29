# PR — Full Electricity MPC Knobs + Checkpointed Historical Backtest — Summary

Finishes the electricity control loop and runs a **checkpointed historical backtest** that quantifies the gp/$
value of the electricity actions — without jamming (the PR #116 lesson: hourly MPC decisions are heavy). Built
on `main` with PR #116 merged. No tuning; the Pareto gate is unchanged; no headline is claimed unless a cell
completed **and** the gate passed.

## 15 questions, answered honestly

**1. What does this PR add?** A cell-isolated, timeout-protected, resumable backtest runner
(`scripts/run_checkpointed_electricity_backtest.py`), a dedicated deferrable-shifting validation
(`scripts/validate_deferrable_shifting.py`), focused tests, and six research docs (plan, action-surfaces,
attribution, N2 power-shaping, deferrable, full results). No production-path behaviour change.

**2. Did the backtest complete without jamming?** **Yes.** Arms 1–6: **54/54 cells COMPLETED, 0 TIMEOUT, 0
FAILED.** Each cell = (market, window, arm) runs in a forked subprocess with a hard per-cell timeout; the
artifact is checkpointed after every cell; `--resume` skips completed cells. The PR #116 jam (a single market
ran >20 min) cannot recur — a slow cell is killed and marked TIMEOUT, never blocking the run.

**3. gp/$ from electricity actions alone?** Price-aware DVFS: **mean +1076 gp/$ (≈ +0.3%)** across 9 windows,
up to **+5784 gp/$ (≈ +1.5%)** in high-price PJM windows; ~0/slightly-negative at low price. Deferrable: **0**
to serving gp/$ (separate energy ledger). **DIRECTIONAL_ONLY** — no cell is HEADLINE_SAFE.

**4. DVFS alone?** Isolated by toggling only `electricity_price_aware`. The planner **downclocks in high-price
PJM windows** (`clock = low ×3`: `pjm|volatile` +5784.4, `pjm|expensive` +4859.2 gp/$, SLA unchanged) and does
**not** downclock when energy is cheap (the lift is then ~0 or slightly negative). Price-directional, causal
through clock → `power_scale` → energy×price → cost.

**5. Deferrable alone?** **0** serving gp/$ in all 9 cells (never steals serving capacity). Valid
deadline-respecting energy saving on the deferrable workload: **4.7% PJM / 19.4% ERCOT / 5.4% CAISO**, 0 missed,
$0 under a flat price. SIMULATOR_INFERENCE workload; separate ledger.

**6. DVFS + deferrable together?** `combined = dvfs` and **`interaction = 0.0` in every window** — orthogonal
in serving gp/$ (power-shaping vs. time-shifting). Joint value = DVFS serving lift **+** deferrable energy
saving (different ledgers).

**7. Total Aurelius gp/$ (all knobs)?** The full-action-space arm (≈314 928 bundles) at the comparable
**3-decision** horizon **TIMED OUT** at the 300 s per-cell cap (~100× the clock-focused search/decision) — the
runner killed it and continued (anti-jam works; the PR #116 >20-min jam cannot recur). A **single-decision**
probe **completed** (212.7 s) but lands at an SLA-saturating corner (SLA **0.000** vs baseline 0.275, gp/$
**12902** over one period) — **not comparable** to the 3-decision arms and **far below** baseline gp/$. So
**there is no total-Aurelius gp/$ headline**; the electricity value remains the DVFS isolation (Q4). A full
3-decision sweep needs a larger cap or a pruned candidate set (left to follow-up; the runner supports it).

**8. Is any number headline-safe?** **No.** All 9 windows fail `pareto_sla_not_worse` (the MPC arms carry a
higher SLA-violation rate than the SLA-aware baseline). 3 windows beat baseline gp/$ but all fail the SLA
clause. The gate keeps the headline honestly False — exactly as designed.

**9. Is price-aware DVFS causal — evidence?** Yes. (a) The only input difference between the DVFS and the
price-unaware arm is `electricity_price_aware`; (b) the selected clock changes with price (downclock at high
price, not at low); (c) the gp/$ delta flows through the cost model's energy term
(`billable_gpu_hours × power_kw × power_scale × pue × price`), and `world_validation` asserts
`power_factor(low) < 1 < power_factor(high)`; (d) the sign tracks the price level (a uniformly-positive lift
would be the suspicious result). Independent fixture confirmation: smoke P3 + PR #115 Track D/E.

**10. Is the deferrable saving real or fake?** Real **and** bounded. The dedicated 24 h validation shows a
strictly-positive saving with **0 missed deadlines** (work completed, just moved to cheaper hours) and **$0
saving under a flat price** (`no_fake_shifting = True`). Work is conserved and missed deadlines are penalised,
so the policy cannot "save" by dropping work.

**11. Fidelity labels?** Price path: **TRACE_DERIVED** (PJM/ERCOT/CAISO day-ahead). DVFS power curve + roofline
regime + the entire deferrable workload: **SIMULATOR_INFERENCE**. MPC backtest gp/$: simulator evidence
(directional), not production telemetry. Fixture proofs (smoke P3–P6, dedicated deferrable): **FIXTURE_ONLY**.

**12. Did you tune anything or weaken the Pareto gate?** No. `training.claim_gate` is untouched; flat-price
mode reproduces pre-electricity behaviour exactly; arms differ in exactly one knob; the world-state is fixed
seed-0 (deterministic, no RNG). The "no headline" conclusion is the gate working, not a number we chose.

**13. What is skipped, and why?** **Region shifting** — there is no multi-region fleet model (single sampled
cluster), so a cross-region move would be a free, consequence-free saving. `region_shiftable` is forced False
and tested. Promoting deferrable into a true MPC-selected `ActionBundle` knob is **deferred** (needs joint
serving/deferrable co-scheduling). See `ELECTRICITY_MPC_ACTION_SURFACES.md`.

**14. Is default behaviour unchanged?** Yes. `electricity_price_aware` defaults False; `electricity_prices`
defaults None; `build_price_profile(None)` → a single constant price. All electricity behaviour is opt-in and
flat-price-identical by default. The 12 existing `tests/test_electricity_controller.py` pass.

**15. What is the runner's safety envelope?** Flags: `--markets --windows --arms --max-decisions
--max-requests-per-period --cell-timeout-seconds --win-len --resume --force --quick --full`. `--quick` proves
the runner in minutes (1 market, 1 window, 3 arms); `--full` is bounded + resumable. Timeouts mark TIMEOUT and
continue; failures mark FAILED and continue; the artifact is written after every cell.

## Artifacts

- `data/external/mpc_controller/checkpointed_electricity_backtest.json` — 54 (+arm 7) cells, statuses, metrics, summary
- `data/external/mpc_controller/deferrable_shifting_validation.json` — dedicated 24 h deferrable validation
- docs: `ELECTRICITY_BACKTEST_PLAN.md`, `ELECTRICITY_MPC_ACTION_SURFACES.md`,
  `ELECTRICITY_ATTRIBUTION_BACKTEST.md`, `N2_HISTORICAL_POWER_SHAPING_RESULTS.md`,
  `DEFERRABLE_ENERGY_SHIFTING_RESULTS.md`, `FULL_ELECTRICITY_BACKTEST_RESULTS.md`
- tests: `tests/test_electricity_backtest.py` (8), `tests/test_electricity_controller.py` (12, regression)
