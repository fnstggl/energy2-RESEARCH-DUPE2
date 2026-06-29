# Action-Subset Containment (Diagnostic)

Is the action subset that won the **+82.1%** result still reachable in the candidate space the PR #121 bounded
"all-knobs" arm actually searched? **No.** PR #121's bounded run used `--search clock`, whose candidate space is
**only** `{clock ∈ base/low/high}` with every other surface frozen at its `ActionBundle` default — so the
winning subset (which uses non-default **precision** and **batching**) is **not contained**, and it is not
removed by a pruning *rule* but by an explicit candidate-list replacement.

## What the +82.1% winner used vs. what clock-only can reach

The +82.1% run (`diagnose_mpc_attribution.py`) searched the **full connected space** (routing, batching,
capacity, placement, migration, prewarm, precision, spec-decode, clock). Its advantage came from *combining*
those surfaces. The PR #121 bounded arm's candidate space:

```python
# run_checkpointed_all_knobs_backtest.py  (search == "clock")
def _clock_candidates():
    return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]   # 3 bundles
c.candidates = _clock_candidates()        # ← REPLACES the generator; all other surfaces at default
```

So `precision_policy=bf16`, `batching_policy=conservative`, `capacity_multiplier=1.0`, `routing=…default`, etc.
are **frozen**. **Any winning bundle that needs fp8 / aggressive batching / non-default capacity is unreachable.**

## Containment test — controlled, same window, vary ONLY the candidate space

`scripts/diagnose_search_containment.py` holds window / dt / cost / cap / baseline FIXED (pjm·expensive, periods
81–82, baseline `sla_aware` = 311,659 gp/$) and compares the clock-only candidate space against a small
**exhaustive grid** over `clock × precision × capacity × batching` (24 bundles). Artifact:
`search_containment_diagnostic.json`.

| candidate space | raw / evaluated | chosen bundle | gp/$ | vs baseline | SLA | SLA Δ vs base |
|--|--|--|--|--|--|--|
| **clock_only** (PR #121) | 3 / 4 | clock=high, **bf16, conservative** | 297,733 | **−4.47%** | 0.5625 | **+0.225 (worse)** |
| **grid_multi_knob** | 24 / 25 | clock=high, **fp8, aggressive** | **624,799** | **+100.5%** | 0.0375 | **−0.30 (better)** |

```
Goodput/$        Baseline (sla_aware): 311,659
  clock_only:    297,733   →  abs -13,926   rel -4.47%   (loses to baseline)
  grid_multi:    624,799   →  abs +313,140  rel +100.48% (Pareto-DOMINANT)

SLA violation    Baseline: 0.3375   (lower better)
  clock_only:    0.5625   →  abs +0.225   (worse)
  grid_multi:    0.0375   →  abs -0.300   (better)
```

## Findings

1. **The winning subset is NOT contained in clock-only.** The grid winner is `{clock=high, precision=fp8,
   batching=aggressive}`. Two of its three non-default levers (fp8, aggressive) **cannot** be expressed in the
   3-bundle clock-only space. The subset was removed **not by a pruning rule** but by the explicit
   `c.candidates = _clock_candidates()` replacement in the bounded runner (search=="clock").
2. **Present-but-not-selected? No — it was ABSENT.** This is not a "search missed it" case; the bundle is not in
   the enumerated space at all. (The fast 24-bundle grid is exhaustively evaluated — `evaluated=25` — so within
   the grid there is **zero** search regret; the winner is found.)
3. **Adding knobs makes the planner BETTER, not worse.** Going from 3 (clock) to 24 (clock×precision×capacity×
   batching) candidates moved the result from −4.47% to **+100.5%**, and from SLA-worse to **SLA-better** — a
   Pareto-dominant, headline-safe point on this window.

## Conclusion

PR #121's "all-knobs got worse" is a **candidate-containment artifact**: the bounded arm searched clock-only, so
the fp8 + aggressive-batching subset that doubles gp/$ (while improving SLA) was unreachable. The fix is not in
the MPC or the gate — it is to **search a broader candidate set**. A small exhaustive multi-knob grid (24
bundles, evaluated in seconds) already recovers and exceeds the advantage (see
`PLANNER_SEARCH_REGRET_DIAGNOSTIC.md` for the search-regret framing and the recommended planner direction).
