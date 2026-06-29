# Physics-Guided Planner Results

Bounded, checkpointed backtest of the physics-guided planner against the PR #121 clock-only artifact, the
fixed 24-grid, and the exhaustive ground truth, on the **same window** the PR #121 diagnostic used
(`scripts/run_physics_guided_planner_backtest.py`, artifact `physics_guided_planner_backtest.json`). 9 arms,
deterministic (seed-0 fixed world; no RNG). Every KPI is reported absolute + percent. **No simulator / reward
/ Pareto-gate change; no tuning. int4 is opt-in and never a headline (quality-risked).**

## Headline (pjm·expensive, 2 decisions, baseline = strongest SLA-aware)

**The physics-guided bounded beam recovers a Pareto-dominant +161.31% gp/$ — more goodput at LOWER cost and
BETTER SLA — and lands EXACTLY on the safe exhaustive ground truth (0 search regret) at ~40 evaluations,
where clock-only forfeits 516,650 gp/$.**

| KPI | baseline (SLA-aware) | physics_guided_beam | abs Δ | rel Δ | direction |
|--|--|--|--|--|--|
| **goodput/$** | 311,659.1 | **814,382.7** | **+502,723.6** | **+161.31%** | higher ✓ |
| **SLA violation rate** | 0.3375 | **0.04375** | **−0.2938** | **−87.04%** | lower ✓ |
| GPU-hours | 0.0222 | 0.0167 | −0.0055 | −24.78% | lower ✓ |
| operator cost ($) | 0.02032 | 0.01547 | −0.00485 | −23.87% | lower ✓ |
| serving energy (kWh) | 0.0* | 0.0646 | +0.0646 | n/a* | diagnostic |
| candidates evaluated | — | 40.5 | — | — | bounded |
| search regret vs safe-exhaustive | — | **0.0** | — | **0.00%** | none ✓ |

`*` the SLA-aware baseline arm runs the legacy-timing path (no roofline energy attribution), so its energy is
0.0 and the relative delta is undefined — the absolute kWh is reported (the physics arm exposes roofline
energy). **Pareto gate: PASS** (beats the fair baseline AND SLA strictly better) — headline-safe.

## Full arm table (pjm·expensive) — reporting standard

| arm | gp/$ | abs Δ vs base | rel Δ | SLA | SLA Δ | headline-safe | gen | eval | margin | widen | regret vs safe-exh (abs / %) |
|--|--|--|--|--|--|--|--|--|--|--|--|
| strongest_sla_aware_baseline | 311,659.1 | — | — | 0.3375 | — | — (is baseline) | — | — | — | — | 502,724 / 161.3% |
| **clock_only** (PR #121 artifact) | 297,733.2 | −13,925.9 | **−4.47%** | 0.5625 | +0.225 | **No** (SLA worse) | 3 | 4 | — | — | **516,650 / 173.5%** |
| fixed_24_grid | 624,798.6 | +313,139.5 | +100.48% | 0.0375 | −0.300 | Yes | 24 | 25 | — | — | 189,584 / 30.3% |
| physics_guided_candidates | 624,798.6 | +313,139.5 | +100.48% | 0.0375 | −0.300 | Yes | 18 | 18 | 0.015 | 0 | 189,584 / 30.3% |
| **physics_guided_beam** | **814,382.7** | **+502,723.6** | **+161.31%** | 0.04375 | −0.294 | **Yes** | 18.5 | 40.5 | 0.118 | 0 | **0 / 0.0%** |
| physics_guided_widening | 814,382.7 | +502,723.6 | +161.31% | 0.04375 | −0.294 | Yes | 24.5 | 46 | 0.116 | 1 | 0 / 0.0% |
| oracle_forecast (diagnostic) | 852,522.5 | +540,863.4 | +173.54% | 0.0625 | −0.275 | Yes | 18.5 | 41.5 | 0.135 | 0 | −38,140 / −4.5% |
| exhaustive_default4 (SAFE ground truth) | 814,382.7 | +502,723.6 | +161.31% | 0.04375 | −0.294 | Yes | 54 | 55 | — | — | 0 / 0.0% |
| exhaustive_int4_diagnostic (**UNSAFE**) | 1,018,757.3 | +707,098.2 | +226.88% | 0.05625 | −0.281 | **diagnostic only** | 81 | 82 | — | — | −204,375 / −20.1% |

## Multi-market confirmation (pjm / ercot / caiso · expensive)

The direction is robust across all three electricity markets. **clock-only is never headline-safe (SLA always
worse); the physics-guided beam is always Pareto-dominant and always at 0 search regret vs the safe
exhaustive.**

| market | baseline gp/$ | clock_only (rel Δ, SLA) | physics_guided_beam (rel Δ, SLA) | beam search regret | beam headline-safe |
|--|--|--|--|--|--|
| pjm·expensive | 311,659.1 | −4.47%, SLA 0.5625 (worse) | **+161.31%**, SLA 0.04375 (better) | **0** | **Yes** |
| ercot·expensive | 373,537.9 | −10.31%, SLA 0.475 (worse) | **+191.41%**, SLA 0.00625 (better) | **0** | **Yes** |
| caiso·expensive | 406,766.5 | +0.50%, SLA 0.5125 (worse) | **+147.85%**, SLA 0.01875 (better) | **0** | **Yes** |

- **clock_only never passes the Pareto gate** in any market (SLA always far worse than baseline; gp/$ between
  −10.31% and +0.50%). This is the PR #121 artifact, reproduced three times.
- **physics_guided_beam is Pareto-dominant in every market** (+147.85% to +191.41% gp/$ with SLA *better* than
  baseline) and **matches the safe exhaustive ground truth exactly (0 search regret) in all three**.
- **int4 is not a free win.** Its diagnostic ceiling exceeds the safe optimum on pjm (+226.88%) and caiso
  (+166.86%), but on **ercot the safe fp8 beam (+191.41%) actually beats the int4 exhaustive (+187.22%)** —
  int4's quality/SLA penalty can outweigh its cost saving. This is exactly why int4 is opt-in, not a headline.

## What the numbers say

1. **Containment alone deletes the PR #121 regression.** `physics_guided_candidates` (just the generated set,
   argmax, **18 evaluations**) scores **+100.48%** — identical to the fixed 24-grid and to the prior
   diagnostic. The "all-knobs got worse" result was *entirely* the clock-only candidate set; the moment the
   fp8+aggressive bundle is in the set, the simulator picks it. (clock_only here reproduces −4.47%, SLA worse.)
2. **The beam adds coupled value beyond the 24-grid.** `physics_guided_beam` lifts +100.48% → **+161.31%**
   (624,799 → 814,383 gp/$, +30.3% more) by reaching balanced-batching / capacity / clock combinations the
   24-grid does not contain — and it does so with **0 search regret vs the 54-bundle safe exhaustive**
   (814,382.7 == 814,382.7). The bounded beam (40.5 evals) finds the exact exhaustive-safe optimum.
3. **Search regret, measured against the exhaustive ground truth:** clock_only **516,650 gp/$ (173.5%)** →
   physics_guided_beam **0**. The new planner forfeits *nothing* the exhaustive search would have found, while
   clock-only forfeits essentially everything.
4. **It's a true Pareto win, not SLA-bought.** +161.31% gp/$ comes with **−24.78% GPU-hours, −23.87% cost,
   and −87% SLA violations** — fp8 + aggressive batching pack more goodput per GPU-hour at lower precision
   pressure. Every physics arm is headline-safe; only clock-only fails the gate (SLA worse).
5. **Progressive widening did no harm and found no more (here).** On this window the beam already reached the
   safe optimum, so widening explored one extra round on the closer decision and returned the same point
   (814,383). Correct behaviour: widen when close, keep the optimum, never regress.
6. **The forecast ceiling is now the residual.** `oracle_forecast` (planning against the exact future) reaches
   **+173.54%**, i.e. **38,140 gp/$ (~4.5%) above** the forecast-driven beam — the gap is *forecast quality*,
   not search. With search solved, output-length / arrival forecasting is the next highest-ROI lever.
7. **int4 is the unsafe ceiling, explicitly excluded from the headline.** `exhaustive_int4_diagnostic` reaches
   +226.88%, but its extra 204,375 gp/$ over the safe optimum comes from int4 — which carries an **unmodelled
   quality/SLA risk** (`PRECISION_QUALITY_RISK`, no quality model). Per the honesty contract int4 is opt-in and
   **never a headline**; it is reported here only as a labelled diagnostic ceiling. The headline-safe number is
   the fp8 result (+161.31%).

## Honesty

- **Reward, cost model, and Pareto gate are byte-identical**; the only change is *which candidates the planner
  evaluates*. Any gp/$ change is attributable to search/candidate containment, proven by the 0 regret against
  the unchanged exhaustive ground truth (point 3).
- **Bounded, simulator-inferred, one primary window / 2 decisions.** The *direction* (the bounded beam recovers
  the Pareto-dominant multi-knob win clock-only forfeits, at 0 search regret) is the robust finding; absolute
  magnitudes are SIMULATOR_INFERENCE (the roofline fp8/batching bands; `WORLD_MODEL_ROBUSTNESS_AUDIT.md`).
- **No headline is claimed on int4** (quality-risked) or on the oracle (non-deployable). The deployable,
  headline-safe claim is **physics_guided_beam: +161.31% gp/$, SLA −87%, GPU-hours −24.8%, cost −23.9%,
  Pareto-PASS, 0 search regret, ~40 evaluations.**
- Reproducible: `data/external/mpc_controller/physics_guided_planner_backtest.json`;
  `python -m scripts.run_physics_guided_planner_backtest --quick`.
