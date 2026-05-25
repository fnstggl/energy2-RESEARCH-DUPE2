# Weather-Aware Optimization: Production-Readiness Validation

Branch: `claude/weather-production-validation` · Date: 2026-05-24

Validates the four open caveats from the weather PR (#54) with runnable
bootstrap evidence. Harness: `benchmarks/weather_validation.py`
(`--mode rt_vs_dam|gating|sensitivity`) and `benchmarks/weather_concentration.py`.
All runs use the real `BacktestEngine` + optimizer over the single available
price window (2025-09-01 → 2026-03-05), day-ahead Open-Meteo forecast weather,
bootstrap 95% CIs over independent job-mix seeds.

## TL;DR verdict

Weather-aware forecasting is **conditionally production-useful** and the alpha
**survives RT settlement**, but it is **flexibility-gated, not region/season-gated**,
and its dollar value is **concentrated in winter and unverifiable across years**.

| Question | Answer |
|---|---|
| Enable in production? | **Yes, narrowly** — for flexible/deferrable fleets, full weather. |
| Where/when exactly? | **All regions, all seasons** (NOT a subset). Long, deadline-slack jobs only. |
| Survives RT settlement? | **Yes.** +9.24pp [+6.42, +11.51], all 6 seeds positive. |
| Broad or one-event? | **Broad in sign** (6/6 folds positive) but **dollar-concentrated** (~49% one cold-snap fold); multi-year **unverified**. |

---

## Caveat 1 — RT settlement (does alpha survive DA→RT basis?)

Optimizer plans on day-ahead price + day-ahead forecast weather; scored on
real-time settlement. DA→RT basis is large/volatile (us-east RT−DA sd ≈ $70/MWh).

| scoring | mean OFF | mean ON | delta | 95% CI |
|---|---|---|---|---|
| DAM | 6.22% | 15.05% | **+8.83pp** | [+5.78, +11.67] |
| **RT (realized)** | 5.12% | 14.36% | **+9.24pp** | [+6.42, +11.51] |

**RT erodes absolute savings for both arms (basis risk), but the weather
*advantage* persists** — all 6 RT seeds positive. **Weather improves, not
worsens, realized RT cost.** ✅ Survives.

## Caveat 2 — Gating (safest production rule), RT-scored

| gating rule | delta | 95% CI | verdict |
|---|---|---|---|
| **full (all regions, all seasons)** | **+9.24pp** | [+6.42, +11.51] | **significant (+)** |
| PJM/us-east only | −2.77pp | [−8.02, +1.60] | n.s. |
| winter months only | +1.37pp | [−0.62, +3.03] | n.s. |
| PJM + winter only | −3.88pp | [−8.23, −0.18] | **significant (−) — HURTS** |

**This reverses PR #54's gating recommendation.** Restricting weather to the
region/season where it looked best at the *forecast* level **destroys the
savings alpha** — PJM+winter-only is significantly negative. The joint
forecaster needs **symmetric weather across all regions** to preserve the
cross-region price ranking the optimizer routes on; partial coverage corrupts
that ranking. **Safest rule: enable weather everywhere or not at all. Do NOT gate.**

## Caveat 3 — Multi-year robustness & concentration

**Multi-year is untestable:** every price file in the repo (DAM and RT) spans a
single window, 2025-06-01 → 2026-03-15 — **one winter**. Open-Meteo history goes
back years, but price data does not, so cross-year generalization cannot be
measured. Stated as a hard limitation, not assumed away.

Concentration within the window (RT-scored, per fold): **6/6 folds positive**,
but the top fold (the Jan-26 cold snap) = **49% of net extra savings**, top-2 =
**72%**. So the alpha is **broad in sign but dollar-concentrated in winter
events**. The headline +9pp should be read as "positive every fold, but ~half
the dollars come from one cold snap we happen to have data for."

## Caveat 4 — Scheduler realism & sensitivity (RT-scored, weather=full)

**No realistic workload trace exists** in the repo — only a 12-job demo fixture
(`data/fixtures/sample_customer_workload_trace.csv`) submitted after the price
window ends. So results use synthetic jobs with an explicit sensitivity sweep.

| variant | delta | 95% CI | verdict |
|---|---|---|---|
| base (50 jobs, training) | +10.51pp | [+8.42, +12.05] | sig (+) |
| 25 jobs | +11.63pp | [+9.56, +13.93] | sig (+) |
| 100 jobs | +9.28pp | [+7.30, +11.07] | sig (+) |
| **short duration** | **+2.61pp** | [−5.83, +12.81] | **n.s.** |
| **tight deadline (12h slack)** | **−0.20pp** | [−0.53, +0.32] | **n.s.** |
| loose deadline (14d slack) | +13.75pp | [+11.35, +16.16] | sig (+) |
| no migration | +10.51pp | [+8.42, +12.05] | sig (+) (≡ base) |

**The alpha is flexibility-gated.** It is robust to job count and independent of
migration (migration never changes these schedules), but **collapses to ~zero
for short jobs and tight deadlines** and **grows with deadline slack**. Mechanism:
weather improves the price forecast; value is realized only when the optimizer
has freedom to *shift* jobs to the cheaper hours the better forecast reveals.
(`mixed_workload` variant did not complete — container reclaim during idle; non-essential.)

> Note: the deadline lever mutates `job.deadline` (the optimizer reads
> `deadline`/`latest_start`, not `max_delay_hours` — the prior PR's inert
> override was corrected here).

---

## Honest claim we are allowed to make

> "For **flexible, deferrable** GPU workloads (training/batch with deadline
> slack) scheduled across CAISO/PJM/ERCOT, enabling weather-aware day-ahead
> price forecasting **for all regions** improved realized **RT-settled** savings
> by ~**9 pp** vs no-weather across all job-mix seeds in the single winter
> (2025-26) for which we have price data. The benefit **survives DA→RT basis**,
> is **robust to job count and migration**, **requires scheduling flexibility**
> (≈0 for short or tight-deadline jobs), and is **dollar-concentrated in winter
> cold-snap periods**. Multi-year generalization is **not yet verified**."

What we may **not** claim: a year-round uniform edge; benefit for
realtime/tight-SLA fleets; region- or season-gated deployment; or that the
magnitude holds outside this one winter.

## Minimum remaining work before an unconditional claim
1. Acquire ≥2 more winters of DAM+RT price data and re-run (the only way to kill
   the one-event concentration risk).
2. Validate on a real customer workload trace (none exists today).
3. Ship a production guard: enable weather only when the fleet's deferrable-hours
   ratio clears a threshold (alpha is ~0 without flexibility).

## Reproduce
```
python benchmarks/weather_validation.py --mode rt_vs_dam   --seeds 6
python benchmarks/weather_validation.py --mode gating      --seeds 6
python benchmarks/weather_validation.py --mode sensitivity --seeds 4
python benchmarks/weather_concentration.py
```
Artifacts: `benchmarks/results/weather_validation_{rt_vs_dam,gating,sensitivity}.json`.
