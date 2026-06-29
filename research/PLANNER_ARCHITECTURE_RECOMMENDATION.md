# Planner Architecture Recommendation

The evidence-based answer to "which MPC search architecture maximises SLA-safe goodput/$ under a bounded
budget", from the tournament (`research/MPC_SEARCH_METHOD_TOURNAMENT.md`,
`data/external/mpc_controller/search_method_tournament.json`). **Diagnostic only — this PR does not change the
default planner.** The recommendation is a measurement, not a hypothesis we set out to confirm; it would have
been an acceptable outcome for the current planner to remain best.

## The 12 questions

**1. Is search currently the bottleneck?** **Yes, decisively.** `clock_only` (the PR #121 artifact) is the
worst method on all 7 windows and the only one that fails the Pareto gate (SLA worse than baseline on every
market). Switching from clock-only to a multi-knob search recovers **+34% to +190%** (synthetic) and **+530%
to +1273%** (markets). Candidate generation / search — not the simulator, reward, or gate — is the bottleneck.

**2. Did the fixed multi-knob grid beat clock-only again?** **Yes, on every window** — reproducing PR #121.
On pjm·expensive (the #121 window): clock_only +15.0% (SLA 0.66, **fails** Pareto) vs fixed_grid **+569.6%**
(SLA 0.29, passes). The 24/54-bundle multi-knob grid recovers the win clock-only forfeits.

**3. Did physics-guided generation beat the fixed grid?** **No — they tie** (both 1,165,080 avg). With the
required-anchor contract forcing the full core grid into every candidate set, `physics_guided_grid` and
`fixed_grid` search the same SET; the regime prior only changes the ORDER (which matters at tiny budgets) and
seeds the beam. Physics-guided generation is **not** worse, but on these windows it is not better than the
fixed grid either.

**4. Did beam search beat the fixed grid?** **No — they tie** (1,165,080 avg). Beam over the core surfaces
finds the same core-grid optimum as the grid; on markets both leave **~52% regret** because neither explores
the connected surfaces (placement/migration/routing/capacity-policy). Beam's coupling advantage (proven in
the #122 fixtures) does not manifest when the optimum is in the core grid.

**5. Did progressive widening reduce runtime without losing reward?** **Yes — it adds reward at modest cost.**
On markets it widens into the connected surfaces and reaches **+919–952%** (vs the grids' +570%) at **54
evaluations** — fewer than hierarchical's 80 — and it stops early when the decision is clear (synthetic:
0 widening rounds). It is the second-best family overall (1,367,529 avg).

**6. Did hierarchical search help or hurt?** **It helped, decisively.** Searching by control timescale
(slow: capacity/placement/migration/prewarm · medium: precision/batching/spec/clock · fast:
routing/admission/ordering), with cross-group coupling and a final polish, it **wins all 3 markets** by ~2×
(pjm 1,454,636 vs the grids' 709,283; **+1273%** vs baseline; **regret 0**; SLA *better* than baseline). On
synthetic (optimum in the core grid) it ties the best at a few more evaluations — no reward harm. It never
hurt.

**7. Which method gives the best gp/$ under runtime caps?** **`hierarchical_search`** — it leads at **every**
budget (10/25/50/100) on the markets and the gap to the grids widens with budget; it is on the
runtime-vs-gp/$ Pareto frontier.

**8. Which gives the best Pareto-safe result?** **`hierarchical_search`** — highest average Pareto-safe gp/$
(1,524,356) and Pareto-safe on all 7 windows. (clock_only is the only method that fails the gate — on 4/7.)

**9. Which has the lowest search regret?** **`hierarchical_search`** — regret **0** on every window (the only
method at 0 on the markets, where the core-grid methods sit at ~52%).

**10. Which method should become the default?** `hierarchical_search` **meets every condition of the
default-change rule** on the windows tested — it beats the current default (the #122 physics-guided beam =
709,283 on pjm) on gp/$ (1,454,636), does not worsen SLA (0.288 vs 0.292 — better), has acceptable runtime
(~80 evaluations), contains the required anchors, and has lower regret (0 vs 51%). **However**, because the
result is bounded and simulator-inferred (budgets ≤100, one decision/window, 7 windows), the recommendation
is to **adopt it OPT-IN first** (a selectable planner), validate on broader windows, and only then flip the
global default — and this PR deliberately does **not** change the default. The strongest single architecture
to adopt is a **hybrid**: physics-guided generation (anchors + regime ordering) → **hierarchical
connected-space search** (the part that wins markets) → beam coupling within groups → progressive widening →
search-regret audit.

**11. Which methods should remain diagnostic-only?** `clock_only` (the degenerate artifact — kept only as
the comparator), `exhaustive_small` (tractable only on tiny fixtures), `random_grid` (the physics-prior
ablation control), and the oracle (non-deployable — not run here). `coordinate_descent` stays a comparator
(it ties the grids but offers no advantage).

**12. Recommended long-term planner architecture.**
```
Forecast distributions → World model
  → Physics-guided candidate generator (anchors + regime priors; invariant-enforced)
    → HIERARCHICAL connected-space search (slow / medium / fast timescale groups, cross-group coupled)
      → Beam coupling within each group + coordinate polish
        → Progressive widening (expand only when the decision is close)
          → Decision
            → Search-regret auditor (FAIL on loss to a contained old-best)
```
The decisive lesson is **reach**: the #122 physics-guided beam searches the core grid well but cannot see the
connected surfaces (placement/migration/routing/capacity-policy) that more than double gp/$ on real markets.
The next planner's job is to search those surfaces **tractably** — which hierarchical decomposition does at
~80 evaluations with 0 regret — without falling back to the intractable full-space adaptive search.

## Default-change decision (this PR)

| criterion | hierarchical_search vs current default (physics-guided beam) | met? |
|--|--|--|
| beats current default on gp/$ | 1,454,636 vs 709,283 (pjm); +31% avg across windows | ✓ |
| does not worsen SLA vs the Pareto gate | 0.288 vs 0.292 (better); Pareto-safe on all 7 | ✓ |
| acceptable runtime | ~80 evaluations / bounded; 0 timeouts | ✓ |
| contains previous-best anchor candidates | anchors_evaluated = True on every window | ✓ |
| search regret lower than current default | 0% vs ~52% on markets | ✓ |

All five criteria are met **on the windows tested** → `hierarchical_search` is the recommended next default.
**Per the task's caution ("do not merge into production behaviour automatically unless results justify it"),
this PR keeps it opt-in / diagnostic** and recommends a broader-window validation before flipping the global
default. The honest, conservative path: wire hierarchical (or the hybrid) as a selectable planner, re-run the
tournament on more windows/markets, then change the default if the result holds.

## What remains unresolved

- **Magnitudes are simulator-inferred.** The +1273% market figure is a bounded directional result; the robust
  claim is "connected-space search beats core-grid search on real markets," not the exact number.
- **Higher budgets (250/500/1000)** were not run (the connected-space rollout union balloons there) — a
  follow-up at a higher per-window timeout.
- **The connected-surface win needs fidelity scrutiny:** hierarchical's gain leans on placement / migration /
  routing levers whose magnitudes are SIMULATOR_INFERENCE (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`); validate before
  productionising.
- **Forecast fidelity** (the oracle gap from #122) remains the residual once search is solved.
