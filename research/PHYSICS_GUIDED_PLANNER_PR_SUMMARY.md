# PR Summary — Physics-Guided Planner

Next-generation planner that fixes the candidate-generation/search bottleneck PR #121 exposed. Candidate
generation is now a **first-class, anchor-guaranteed planner layer**; a **bounded beam MPC + progressive
widening** searches it; a **search-regret auditor** measures what each strategy leaves on the table. The
simulator physics, reward, and Pareto gate are **byte-identical**; the planner is **opt-in**
(`controller.physics_guided`, default off). int4 is opt-in and never a headline.

## Pipeline

```
Forecast distributions → World model → Physics-guided candidate generator
   → Bounded beam MPC → Progressive widening → Decision → Search-regret auditor
```

New modules: `physics_guided_candidates.py`, `physics_guided_planner.py`, `search_regret_auditor.py`;
runner `scripts/run_physics_guided_planner_backtest.py`; tests `tests/test_physics_guided_planner.py`;
docs (audit, architecture, results, search-regret, this summary). One-line robustness fix in `controller.py`
(legacy dict candidates no longer crash the electricity diagnostic).

## The 11 questions

1. **Did we replace the clock-only fallback?** **Yes.** The physics planner never uses clock-only; the
   generator always emits a multi-knob, anchor-guaranteed set. `clock_only` survives in the backtest **only**
   as the labelled PR #121 artifact comparator. Test `test_never_falls_back_to_clock_only` enforces it.
2. **Does the planner contain the previous-best action subset?** **Yes — by construction.** Neutral, the
   SLA-aware safe bundle, the **previous best**, the `fp8+aggressive` family, the +82% family, capacity-adjusted
   and clock low/base/high are **guaranteed anchors**, never dropped by the cap. Tests assert each.
3. **Did gp/$ improve vs clock-only?** **Yes, dramatically.** pjm·expensive: physics beam **+161.31%** vs
   clock-only **−4.47%**. Clock-only forfeits **63.4%** of the achievable gp/$ (516,650 gp/$).
4. **Did gp/$ improve vs the SLA-aware baseline?** **Yes, in all three markets:** pjm **+161.31%**, ercot
   **+191.41%**, caiso **+147.85%** — each with **better** SLA.
5. **Was it Pareto-safe?** **Yes.** Every physics arm beats the fair baseline AND has strictly lower SLA
   violations → **gate PASS**. (clock-only fails the gate in every market — SLA worse.)
6. **How many candidates generated / evaluated?** ~18–25 generated, **~40–46 evaluated** (beam) — bounded by
   `max_evaluated=120`. Compare: 314,928 (full connected), 81 (int4 exhaustive), 55 (safe exhaustive).
7. **Runtime?** ~30–90 s per cell (2 decisions) — tractable at hourly cadence, where the full adaptive search
   did not complete in > 3 min (the PR #118 result that forced clock-only).
8. **Search regret?** **0** — the bounded beam lands on the safe exhaustive optimum in all three markets.
   clock-only regret: 516,650 (pjm) / 753,499 (ercot) / 599,383 (caiso) gp/$.
9. **Did progressive widening help?** On these windows the beam already reached the safe optimum, so widening
   explored one extra round on the closer decision and returned the same point — **correct behaviour** (widen
   when close, never regress). It is available for harder/closer decisions; tests prove it expands on a small
   margin and stops on a large one.
10. **What remains intractable?** The full 314,928-bundle adaptive search at hourly cadence — *per-evaluation*
    world-rollout cost, not candidate count. The bounded beam sidesteps it (≤ ~46 rollouts).
11. **Next planner improvement?** **Forecast fidelity** — the oracle arm (exact future) sits ~4.5% above the
    forecast-driven beam (+173.54% vs +161.31% on pjm); output-length/arrival forecasting is the next
    highest-ROI lever. Then cross-period action coupling, a calibrated vLLM baseline, and learned candidate
    proposal *only if* multi-knob grids become intractable.

## Headline-safe result (deployable, no int4, no oracle)

| KPI | baseline (SLA-aware) | physics_guided_beam | rel Δ |
|--|--|--|--|
| goodput/$ (pjm) | 311,659 | 814,383 | **+161.31%** |
| SLA violation rate | 0.3375 | 0.04375 | **−87.0%** |
| GPU-hours | 0.0222 | 0.0167 | −24.78% |
| operator cost | 0.02032 | 0.01547 | −23.87% |
| search regret vs exhaustive | — | 0 | — |
| candidates evaluated | — | 40.5 | bounded |

Pareto gate **PASS**. ercot **+191.41%** / caiso **+147.85%** confirm the direction (both Pareto-dominant, 0
regret).

## Honesty

- **No simulator / reward / cost / Pareto-gate change; no tuning.** Improvement is attributable to
  search/candidate containment, proven by 0 regret against the unchanged exhaustive ground truth.
- **int4 is opt-in and never a headline** (unmodelled quality risk); its ceiling is reported as a labelled
  diagnostic. On ercot the safe fp8 beam even **beats** the int4 exhaustive — int4 is not a free win.
- **No headline on the oracle** (non-deployable). Bounded, simulator-inferred, primary window pjm·expensive
  (2 decisions) + ercot/caiso confirmation; the **direction and 0 regret** are the robust findings.
- Reproducible: `data/external/mpc_controller/physics_guided_planner_backtest.json`;
  `python -m scripts.run_physics_guided_planner_backtest --quick`.
