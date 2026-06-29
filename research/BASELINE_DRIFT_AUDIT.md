# Baseline Drift Audit (Diagnostic)

Is the recent "all-knobs got worse" result (PR #121) a real regression, or an artifact of comparing two
**different setups**? This audit traces the two results to their exact scripts/configs and asks: are they
apples-to-apples? **They are not.** The baseline policy is identical; everything around it changed.

## The two results

| | **Result A — "+82.1%"** | **Result B — PR #121 "all-knobs"** |
|--|--|--|
| script | `scripts/diagnose_mpc_attribution.py` | `scripts/run_checkpointed_all_knobs_backtest.py` |
| artifact | `mpc_attribution.json` | `checkpointed_all_knobs_backtest.json` |
| MPC gp/$ | **183,152** (full action layer) | ~344,726 (pjm·expensive, deployable) |
| SLA-aware baseline gp/$ | **100,555** | ~352,581 |
| headline | **MPC +82.1%** vs baseline | **−2.2%** (median −3.1%; 0/4 Pareto-safe) |

## Config comparison (file:line)

| dimension | A (+82.1%) | B (PR #121) | same? |
|--|--|--|--|
| **(a) window** | Azure 1-week, 120 eval / 6 MPC periods, median prompt 828 / output 45 | hourly PJM/ERCOT/CAISO electricity windows, 3-decision cells | **NO** |
| **(b) dt / period** | `control_dt_seconds=60`, `sim_seconds=180` (`diagnose_mpc_attribution.py:79`) | `control_dt_seconds=3600` (hourly), `sim_seconds=60` (`run_checkpointed_electricity_backtest.py:57`) | **NO (60× dt)** |
| **(c) timing model** | `legacy_scalar` (default) | `legacy_scalar` (default; one arm sets `AURELIUS_TIMING_MODEL=roofline`) | yes (deployable) |
| **(d) search / action surfaces** | **full adaptive beam+local search over the full connected space** (`use_adaptive_search=True`, all knobs) | **clock-focused** (`--search clock` → `c.candidates=[ActionBundle(clock_policy=c) for c in base/low/high]`, all other knobs at default) | **NO — the decisive difference** |
| **(e) baseline policy** | `SLA_AWARE_FALLBACK` (`controller.py:45`) | `SLA_AWARE_FALLBACK` (identical) | **YES** |
| **(f) price mode** | flat (no `electricity_market`) | real diurnal (most arms); flat for the flat arm | NO |
| **(g) roofline mode** | legacy | legacy (deployable); roofline only for the roofline arm | yes (deployable) |
| **(h) request cap** | 20,000 (Mooncake) | **80 / period** (`per[:req_cap]`) | **NO (250× fewer)** |
| **(i) Pareto / claim gate** | `training.claim_gate` | `training.claim_gate` (identical) | **YES** |
| **(j) KV cost mode** | `hybrid_capacity_work` | `hybrid_capacity_work` | YES |

## Findings

1. **The baseline POLICY did not change.** `SLA_AWARE_FALLBACK = {capacity: backlog_aware, ordering:
   abs_conformal, admission: off}` is byte-identical in both (`controller.py:45`; called in both scripts). So
   the baseline gp/$ moving from **100,555 → ~352,581** is **not a stronger baseline policy** — it is the
   **window + dt + request-cap + price** changing the absolute gp/$ scale. **gp/$ is not comparable across the
   two windows** (different dt, different request density, different cost base). *Not apples-to-apples.*

2. **The MPC's search shrank from "all knobs" to "clock only."** Result A ran the **full adaptive search over
   the connected action space** (routing/batching/capacity/placement/migration/prewarm/precision/spec/clock).
   Result B's bounded "deployable_all_knobs" ran with **`--search clock`**, i.e. only `{base, low, high}` clock
   with every other surface frozen at its `ActionBundle` default. **The PR #121 bounded "all-knobs" arm is
   mislabelled — it is clock-only.** It cannot use the capacity/routing/batching levers that produced +82.1%
   (see `ACTION_SUBSET_CONTAINMENT.md`).

3. **Two compounding causes, neither a real MPC regression:**
   - **baseline drift** (the window/dt/cap changed the absolute gp/$ scale — the ~100k→~350k jump), and
   - **search containment** (the bounded all-knobs arm searched only the clock, so it left the winning
     capacity/routing/batching subset unexplored — see the empirical regret in
     `PLANNER_SEARCH_REGRET_DIAGNOSTIC.md`).

## What would be apples-to-apples

To compare the MPC's all-knobs capability against the +82.1% result you must hold the window/dt/cap FIXED and
run the **same full search** (not clock-only). PR #118 found the full adaptive search is heavy at hourly
cadence (it times out / is SKIPPED_TOO_HEAVY) — which is *why* the bounded runs fell back to clock-only. So the
honest statement is: **"all-knobs got worse" is unproven; the bounded run measured clock-only on a different
window, not all-knobs on the same window.** The next-step recommendation (a tractable full search — beam /
progressive widening) is in `PLANNER_SEARCH_REGRET_DIAGNOSTIC.md`.

## Honesty

No tuning; no gate change; no simulator change. This audit is config-tracing only. The `+82.1%` itself remains
a **bounded, simulator-inferred** directional result (its own doc says so) — this audit does not re-validate it,
only establishes that it and the PR #121 number are measured under different setups and cannot be differenced.
