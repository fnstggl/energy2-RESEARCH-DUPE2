# Planner Search-Regret Diagnostic (Diagnostic)

On a tiny controlled window, with everything except the **candidate space** held fixed, how much gp/$ does each
search leave on the table (search regret), and can adding knobs make the planner *worse*? **Evidence: adding
knobs makes it dramatically better; the clock-only search (PR #121) carries enormous regret; the full adaptive
search is the runtime bottleneck.** `scripts/diagnose_search_containment.py`, artifact
`search_containment_diagnostic.json` (pjm·expensive, periods 81–82, baseline `sla_aware` = 311,659 gp/$).

## Search comparison (same window, vary only the candidate space)

| search | raw candidates | evaluated | chosen bundle | gp/$ | vs baseline | SLA | search regret gp/$ |
|--|--|--|--|--|--|--|--|
| **clock_only** (PR #121 bounded) | 3 | 4 | clock=high, bf16, conservative | 297,733 | −4.47% | 0.5625 | **327,065** |
| **grid_multi_knob** (exhaustive 24) | 24 | 25 | clock=high, **fp8, aggressive** | **624,799** | **+100.5%** | **0.0375** | **0** |
| **adaptive** (full connected space) | ≈ 314,928 | — | — | — | — | — | **INTRACTABLE** (did not complete in > 3 min at hourly cadence; killed — the PR #118 finding) |

`search regret = best_known_gp/$ (624,799) − chosen gp/$`. The best-known point is the grid winner (fp8 +
aggressive batching + high clock); the 24-bundle grid is exhaustively evaluated, so its own regret is **0**.

## Reporting standard (the key deltas)

```
Goodput/$                Baseline 311,659
  clock_only  297,733    abs  -13,926   rel  -4.47%
  grid_multi  624,799    abs +313,140   rel +100.48%
  clock→grid             abs +327,066   rel +109.85%   (the value clock-only forfeits)

SLA violation rate       Baseline 0.3375   (lower is better)
  clock_only  0.5625     abs +0.2250
  grid_multi  0.0375     abs -0.3000

Evaluated candidates     clock_only 4   →   grid_multi 25   (+21, +525%)  — still seconds of runtime
```

## Findings

1. **Search regret of clock-only is enormous: 327,065 gp/$ (it forfeits +109.85% gp/$).** The clock-only space
   (PR #121's bounded "all-knobs") cannot reach the fp8 + aggressive-batching winner, so it lands −4.47% below
   the baseline while a 24-bundle grid lands +100.5% above it — a Pareto-dominant, headline-safe point.
2. **Adding knobs does NOT make the planner worse.** The opposite: 3 → 24 candidates moved gp/$ from −4.47% to
   +100.5% and SLA from worse to better. A planner can only get worse from *more* knobs if its **search** fails
   to evaluate the good region — and here the small grid evaluates it exhaustively (regret 0). PR #121 got worse
   because it had *fewer* knobs (clock-only), not more.
3. **The bottleneck is the FULL adaptive search at hourly cadence, not the action space.** The full connected
   space (~314,928 bundles) did not complete (intractable — the PR #118 result that *forced* the clock-only
   fallback). But a **curated 24-bundle multi-knob grid is tractable in seconds and recovers > +100%.** So the
   value is reachable cheaply; the planner just needs the *right* bounded candidate set, not the full space and
   not clock-only.

## Recommended next planner PR (evidence-based)

The evidence supports a **bounded, structured multi-knob search**, not the two extremes:

| option | verdict | evidence |
|--|--|--|
| clock-only (status quo bounded arm) | **reject** — forfeits +109.85% gp/$ | clock_only regret 327,065 |
| full adaptive over ~314,928 bundles | **reject for hourly cadence** — intractable | did not complete > 3 min (#118) |
| **curated multi-knob grid / progressive widening / beam over {precision, batching, capacity, clock}** | **adopt** | 24 bundles, evaluated in seconds, regret 0, +100.5% Pareto-dominant |

**Recommendation:** the next planner PR should make the bounded all-knobs runner default to a **curated
multi-knob candidate set** (at minimum `precision × batching × capacity × clock`, ~24–60 bundles) — or
equivalently a **progressive-widening / beam search** seeded from that grid — instead of `--search clock`. This
is small, tractable at hourly cadence, and recovers the action-combination value the +82.1% full search found.
**Do NOT** jump straight to a full hierarchical MPC; the cheap multi-knob grid already closes most of the gap and
should be measured first. (Hierarchical MPC / action decomposition is justified only if a broad multi-knob grid
itself becomes intractable as the surface count grows — not yet demonstrated.)

## Honesty

Diagnostic only — no MPC/simulator/gate change, no tuning. One window, 2 decisions, simulator-inferred
magnitudes; the **direction** (clock-only forfeits a large, Pareto-dominant multi-knob win) is the robust
finding. The grid winner uses fp8 (no quality penalty in the simulator; a real fp8 quality cost is a separate
fidelity question — `WORLD_MODEL_ROBUSTNESS_AUDIT.md`).
