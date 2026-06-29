# Physics-Guided Search Regret

How much gp/$ does each search strategy leave on the table versus the **exhaustive ground truth**, and does
the bounded physics-guided planner ever lose to a bundle it actually contained? Measured two ways: (1) with
the **real world rollout** across the backtest arms (`physics_guided_planner_backtest.json`), and (2) offline
on a controlled surface space with `aurelius/environment/search_regret_auditor.py`. **No simulator / reward /
gate change.**

## Real search regret (pjm·expensive, world rollout, vs the 54-bundle SAFE exhaustive = 814,382.7 gp/$)

`regret = gp/$(exhaustive_safe) − gp/$(strategy)`; "% of optimum forfeited" = `regret / exhaustive_safe`.

| strategy | evaluated | gp/$ | search regret (abs) | % of optimum forfeited |
|--|--|--|--|--|
| **clock_only** (PR #121 artifact) | 4 | 297,733.2 | **516,649.5** | **63.4%** |
| fixed_24_grid | 25 | 624,798.6 | 189,584.1 | 23.3% |
| physics_guided_candidates (containment only) | 18 | 624,798.6 | 189,584.1 | 23.3% |
| **physics_guided_beam** | 40.5 | **814,382.7** | **0.0** | **0.0%** |
| physics_guided_widening | 46 | 814,382.7 | 0.0 | 0.0% |
| exhaustive_default4 (SAFE ground truth) | 55 | 814,382.7 | 0.0 | 0.0% |

- **clock-only forfeits 63.4% of the achievable gp/$** (516,650 gp/$). This is the PR #121 "regression":
  pure search containment, nothing about the MPC.
- **Containment alone** (physics_guided_candidates) cuts regret 63.4% → **23.3%** (it reaches the 24-grid
  winner but not the balanced-batching / capacity / clock couplings beyond it).
- **The bounded beam closes regret to 0** — it lands on the exact 55-bundle exhaustive-safe optimum with
  **40.5 evaluations**, never enumerating the 314,928-bundle full space or the 81-bundle int4 space.

## Missed-bundle provenance

| ground truth | exhaustive winner | physics planner: generated? | pruned? | evaluated? | verdict |
|--|--|--|--|--|--|
| SAFE (54, bf16/fp8) | `{fp8, balanced/aggressive, capacity, clock}` (gp/$ 814,383) | **yes** | no | **yes** | **found — 0 regret** |
| +int4 (81) | int4 bundle (gp/$ 1,018,757) | no | **yes (quality risk, no model)** | no | **honest prune, not a miss** |

The only bundle the safe planner does not reach is the **int4** optimum — and that is an **intentional,
recorded prune** (`int4_excluded: quality/SLA risk with no quality model → headline-unsafe`), not a search
failure. int4 is opt-in (`allow_quality_risk`) and never a headline; its 204,375 gp/$ edge over the safe
optimum is the quality-risked ceiling, reported as a labelled diagnostic only (`PHYSICS_GUIDED_PLANNER_RESULTS.md`).

## The hard rule — lose to a CONTAINED old-best ⇒ FAIL

The auditor's pass/fail gate is narrow and strict: if the bounded planner ever chooses a bundle that scores
**lower than a known-good bundle it actually evaluated** (the fp8+aggressive family, the previous best, or a
supplied old-best), the diagnostic **FAILS** (`lost_to_contained_old_best=True`). This catches a true search
bug — picking worse than something you had in hand — while *not* penalising an honest prune (int4). On every
window measured here the planner **passes**: it never loses to a contained old-best, because the beam selects
the argmax over everything it evaluates and the known-strong family is always evaluated.

## Offline auditor (controlled fixture, no world rollout)

`audit_search_regret(score_fn, state)` reproduces the comparison on the small default-4 space where exhaustive
is tractable. On a fixture where `fp8 + aggressive` is the coupled win (the PR #121 shape), with int4 modelled
as quality-risked:

```
exhaustive (safe ground truth)  regret 0
physics_guided (beam)           regret 0       ← matches exhaustive
fixed_24_grid                   regret 0
adaptive (existing planner)     regret 0
clock_only                      regret > 0     ← leaves the coupled win on the table
```

The auditor also powers the regression test `test_auditor_detects_missed_contained_optimum`: a deliberately
broken planner that evaluates the winner but returns neutral is correctly flagged FAILED — proving the gate
has teeth.

## Honesty

Search regret is measured against the **unchanged** exhaustive ground truth — so a regret of 0 means the
bounded planner found exactly what brute force would, with ~40 rollouts instead of 55 (safe) / 81 (int4) /
314,928 (full). The result is bounded and simulator-inferred (one primary window, 2 decisions); the robust
finding is the **direction and the 0 regret** (the beam forfeits nothing the exhaustive search finds, while
clock-only forfeits 63%). The int4 gap is a recorded prune, not a search miss, and never a headline.
