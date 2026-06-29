# Canonical All-Knobs Backtest Results (Phase 9)

The bounded run of the tractable all-knobs runner (`scripts/run_checkpointed_all_knobs_backtest.py`, artifact
`data/external/mpc_controller/checkpointed_all_knobs_backtest.json`). **The runner does NOT jam:** 2 markets ×
2 windows × 7 arms → **24 cells COMPLETED, 4 SKIPPED_TOO_HEAVY (the v2_reference arm), 0 TIMEOUT, 0 FAILED.**
Clock-focused search, `max_decisions=3`, ≤80 req/period, seed-0. All KPIs report baseline → new with absolute +
relative deltas (the reporting standard); a change < 0.1% is flagged *approximately unchanged*.

## Headline: deployable all-knobs vs the strongest SLA-aware baseline

**No window is HEADLINE_SAFE (0 of 4).** The deployable all-knobs MPC sits at or below the SLA-aware baseline
on the Pareto frontier — it raises the SLA-violation rate in every window, so the gate (`pareto_sla_not_worse`)
blocks any headline. This is the same SLA-bought property the base MPC has always shown (#117/#118), now
measured with the new state diagnostics.

`pjm|expensive` (deployable_all_knobs vs strongest_sla_aware_baseline), full reporting standard:

```
Goodput/$        (higher is better)
Baseline:          352,581.1
New:               344,725.8
Absolute change:   -7,855.3 gp/$
Relative change:   -2.23%

SLA violation rate (lower is better)
Baseline:          0.2750
New:               0.5250
Absolute change:   +0.2500
Relative change:   +90.9%   ← the gate-blocking term: SLA is worse

GPU-hours         (lower is better)
Baseline:          0.0333
New:               0.0333
Absolute change:   +0.0000
Relative change:   approximately unchanged (0.0%)

Operator cost     (lower is better)
Baseline:          $0.02936
New:               $0.02943
Absolute change:   +$0.00007
Relative change:   +0.24%

Energy             (lower is better)
Baseline:          0.0000 kWh   (the fixed baseline does not exercise the DVFS energy diagnostic)
New:               0.0472 kWh
Absolute change:   +0.0472 kWh
Relative change:   N/A (baseline 0)
```

gp/$ across the 4 windows (deployable − baseline):

| market·window | baseline gp/$ | new gp/$ | abs | rel | SLA base → new | headline-safe |
|--|--|--|--|--|--|--|
| ercot·volatile | 386,978 | 393,006 | **+6,028** | **+1.56%** | 0.225 → 0.483 | **No** (SLA worse) |
| pjm·expensive | 352,581 | 344,726 | −7,855 | −2.23% | 0.275 → 0.525 | No |
| pjm·volatile | 392,972 | 377,340 | −15,632 | −3.98% | 0.175 → 0.383 | No |
| ercot·expensive | 396,377 | 377,075 | −19,302 | −4.87% | 0.204 → 0.483 | No |

- **Median gp/$ change: −3.10%** (deployable below the fair baseline).
- **Best-window: ercot·volatile +6,028 gp/$ (+1.56%)** — but **not headline-safe** (SLA 0.225 → 0.483).
- **Worst-window: ercot·expensive −19,302 gp/$ (−4.87%)**.
- **Pareto-safe fraction: 0 / 4.** **No headline all-knobs gp/$ saving is claimed.**

## Oracle gap (forecast regret)

`oracle_forecast_all_knobs − all_knobs_n2`, per window: ercot·expensive **+7,520.7**, ercot·volatile +3,613.6,
pjm·expensive +102.1, pjm·volatile −1,268.9 (≈ noise). **Median ≈ +1,580 gp/$, up to +7,520** — consistent with
#118; the regret is forecast-driven and largest in ERCOT (widest within-day swing).

## Forecast error (the new ForecastState, captured per cell)

Output-length forecast error (the dominant regret driver, now measured causally):

| window | output_token_mean MAPE | output_token_p95 MAPE |
|--|--|--|
| pjm·expensive | **7.8%** | ~ |
| pjm·volatile | 10.9% | ~ |
| ercot·expensive | 9.4% | ~ |
| ercot·volatile | 18.9% | ~ |

(Arrival-rate / electricity-price errors are deliberately NOT reported — the backtest caps requests/period and
the deployable arm uses a flat price, so those realized values are not unit-comparable to the belief; reporting
them would fabricate a misleading error. Output length is the meaningful, clean signal.)

## RequestState (the new canonical lifecycle, captured per cell)

`request_conserved = True` in **all 4 windows** — `arrived = running + completed + dropped` holds for every
cell (e.g. pjm·expensive: 240 arrived, ~114 completed, ~126 missed-SLA, 0 lost). The queue summary
(backlog/class-mix/completion-rate) is consolidated from RequestState, resolving the `world_state.QueueState`
placeholder.

## Roofline-timing arm (#119) — an honest, non-flattering result

`all_knobs_roofline_timing` vs `deployable_all_knobs` (pjm·expensive):

```
Goodput/$   Baseline (legacy timing): 344,726   New (roofline timing): 298,773
            Absolute: -45,953 gp/$    Relative: -13.3%   (lower)
SLA         Baseline: 0.5250          New: 0.5708         (worse)
```

The roofline-resolved timing is **more realistic, not more flattering**: it exposes *more* SLA violations on
this window than the optimistic GPU-blind scalar, lowering gp/$. That is the honest direction — better physics
can reveal worse economics. It stays behind a flag (default legacy), as #119 shipped it.

## Tractability (the PR #118 gap, closed)

| outcome | count |
|--|--|
| COMPLETED | 24 |
| SKIPPED_TOO_HEAVY | 4 (the v2_reference arm — no tractable V2 serving path on this branch) |
| TIMEOUT | 0 |
| FAILED | 0 |

The runner checkpoints after every cell, resumes, caps per-cell runtime, and marks heavy/absent arms
SKIPPED_TOO_HEAVY rather than stalling. Full *adaptive* all-knobs at hourly cadence remains heavy (#118) — the
runner supports `--search adaptive` with the same timeout protection, and would mark such cells TIMEOUT.

## Bottom line

- The all-knobs backtest is now **tractable + resumable + non-jamming**, with the new canonical states
  (ForecastState error, RequestState conservation) captured per cell.
- **Deployable all-knobs is ~3% below the SLA-aware baseline (median), best-window +1.56%, 0/4 headline-safe**
  → **no headline gp/$ saving is claimed** (the gate blocks it, correctly).
- The biggest remaining lever is forecast fidelity (oracle gap up to +7,520; output-length MAPE 7.8–18.9%),
  which ForecastState now measures — the next-roadmap #1.
