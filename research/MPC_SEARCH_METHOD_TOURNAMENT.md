# MPC Search-Method Tournament

Given the **same** world model, workload window, action space, baselines, seed, and bounded budgets, which
search method finds the best SLA-safe goodput/$? This compares 14 methods at equal **evaluation** budgets
(the portable, hardware-independent currency) and equal wall-clock, across 4 synthetic bottleneck fixtures
(exhaustive-able → true regret) and 3 real electricity-market planning decisions (the PR #121 containment
window + ERCOT/CAISO). **Diagnostic only — no simulator / reward / Pareto-gate / cost-model / baseline /
action-semantics change; no tuning; the default planner is NOT changed by this PR.** The purpose is to
**measure**, not to validate a hypothesis. Artifact: `data/external/mpc_controller/search_method_tournament.json`
(`scripts/run_search_method_tournament.py`, seed-0, 0 timeouts).

## Method roster (literature-informed)

Discrete combinatorial bundles, no gradient, expensive black-box rollouts, budgets ≤ a few hundred. The
listed minimum (clock-only, fixed/expanded multi-knob grids, physics-guided generation, beam, progressive
widening, hierarchical, exhaustive-small) plus **justified additions** for this regime: **cross-entropy
method (CEM)** and **simulated annealing** (the standard strong global searches for discrete black-box
budgets), **random-restart** hill-climbing, **coordinate descent** (the local-search comparator), and a
**hybrid** (physics generation → beam → polish → widening). MCTS and Bayesian optimisation were considered
and **rejected** for this budget: their per-node bookkeeping / surrogate fitting is not worth it at ≤100
evaluations over a ≤6-surface decision (a documented choice, re-openable if the surface count grows).

## Core invariant (enforced; FAIL if violated)

Every candidate set always contains the required anchors — `ActionBundle()` (neutral), the production
SLA-aware baseline bundle, the previous-best MPC bundle, the **ACTION_SUBSET_CONTAINMENT diagnostic winner**
(`precision=fp8, batching=aggressive, clock=high`), and the entire bounded core grid (`precision × batching ×
capacity × clock`). Physics-guided generation may ADD candidates and may reorder; it may not REMOVE an
anchor. `clock_only` is the deliberate degenerate artifact (anchors excluded — that is the finding). Across
the run every non-`clock_only` method evaluated the named anchors (`anchors_evaluated=True`); 0 anchor-contract
violations.

## Headline (per-window winner at budget 100, reporting standard)

| window | regimes | baseline gp/$ | winner | winner gp/$ (Δ%) | winner SLA | winner regret | clock_only (Δ%, SLA, safe) |
|--|--|--|--|--|--|--|--|
| memory_bound_decode | memory, SLA-slack | 1,138,191 | fixed_grid* | 1,963,836 (**+72.5%**) | 0.000 | 0% (TRUE) | −6.5%, 0.229, **No** |
| compute_bound_prefill | compute/SLA-tight | 140,919 | fixed_grid* | 189,110 (**+34.2%**) | 0.000 | 0% (TRUE) | +0.6%, 0.000, Yes |
| sla_tight | SLA-tight | 557,578 | progressive_widening | 1,393,778 (**+150.0%**) | 0.000 | 0% (TRUE) | +0.6%, 0.450, Yes |
| queue_bound | queue/capacity | 794,024 | fixed_grid* | 2,305,689 (**+190.4%**) | 0.000 | 0% (TRUE) | +0.6%, 0.542, Yes |
| **pjm·expensive** (PR #121) | memory | 105,924 | **hierarchical_search** | 1,454,636 (**+1273%**) | 0.288 | **0%** | +15.0%, 0.664, **No** |
| **ercot·expensive** | memory | 129,810 | **hierarchical_search** | 1,703,036 (**+1212%**) | 0.232 | **0%** | −0.5%, 0.664, **No** |
| **caiso·expensive** | memory | 121,866 | **hierarchical_search** | 1,660,408 (**+1262%**) | 0.245 | **0%** | +4.6%, 0.664, **No** |

`*` on the synthetic windows the optimum lies **inside the core grid**, so every multi-knob method ties the
winner (regret ≈0); `fixed_grid` is named winner because it reaches it at the **fewest evaluations** (55).
`(TRUE)` = true-exhaustive regret; markets use the best-known reference (`NOT_TRUE_EXHAUSTIVE`).

## The two regimes the data splits into

1. **Optimum inside the core grid (the 4 synthetic windows).** Every multi-knob method — `fixed_grid`,
   `physics_guided_grid`, `beam_search`, `hybrid`, `cross_entropy`, `simulated_annealing`, `exhaustive_small`
   — ties at the same gp/$ (regret ≈0). They differ only in **evaluation efficiency**: `fixed_grid` 55,
   `beam_search` 43–50, `cross_entropy` ~30, `simulated_annealing` 100. When the winner is in the core grid,
   **brute-forcing the core grid is as good as anything and cheaper than the adaptive methods.**
2. **Optimum in the CONNECTED space (all 3 real markets).** The core-grid methods (`fixed_grid`,
   `physics_guided_grid`, `beam_search`, `hybrid`, `exhaustive_small`) all tie at **+530–570%** vs baseline
   but leave **~52% regret** — they cannot reach the placement / migration / routing / capacity-policy levers.
   **`hierarchical_search` reaches them and wins by ~2×** (pjm 1,454,636 vs the grids' 709,283; **+1273%** vs
   baseline, regret 0, SLA *better* than baseline). `progressive_widening` / `cross_entropy` /
   `simulated_annealing` / `random_restart` land in between (**+919–952%**, ~22% regret).

## Budget scaling (where each planner dominates) — pjm·expensive gp/$ by budget

```
budget →            10        25        50       100
clock_only       121,864   121,864   121,864   121,864     (flat — only 3 candidates)
fixed_grid       542,988   542,988   701,446   709,283     (plateaus at the core-grid optimum)
beam_search      542,988   709,283   709,283   709,283
cross_entropy    542,988 1,114,798 1,114,798 1,114,798     (connected optimum found by budget 25)
simulated_anneal 661,597   661,597 1,114,798 1,114,798
progressive_wide 542,988   709,283 1,114,798 1,114,798
hierarchical_search 752,127 955,253 1,448,727 1,454,636   (dominates at EVERY budget; gap widens)
```

`hierarchical_search` leads at **every** budget and the gap to the grids **widens** with budget — the
connected-space advantage is not a high-budget artifact. The grids plateau at the core-grid ceiling (709k).

## Compute budget, not just runtime (the portable comparison)

At budget 100 on pjm, **gp/$ per evaluation** rewards the cheap searches (`cross_entropy` 38,441/eval at 29
evals; `clock_only` 40,621/eval at 3 evals but a terrible absolute) while `hierarchical_search` spends more
(80 evals, 18,183/eval) for the highest absolute gp/$. **Pareto-efficient planners (evaluations vs gp/$):**
`clock_only` (cheapest, worst), `hierarchical_search` (best reward), `coordinate_descent`, `cross_entropy`,
`random_restart` (good reward at fewer evaluations). The same frontier holds for runtime-vs-gp/$ and
evaluations-vs-regret. Memory (working-set bundles) tracks evaluations and never exceeded ~100 distinct
bundles per method — all bounded.

## Aggregate across all 7 windows (avg Pareto-safe gp/$ — the recommendation metric)

The recommended planner maximises the **average** Pareto-safe gp/$, not a single best number.

| rank | method | avg | median | worst | best | Pareto-safe fraction |
|--|--|--|--|--|--|--|
| 1 | **hierarchical_search** | **1,524,356** | 1,660,408 | 189,110 | 2,305,689 | 1.0 |
| 2 | progressive_widening | 1,367,529 | 1,322,974 | 189,110 | 2,305,689 | 1.0 |
| 2 | random_restart | 1,367,529 | 1,322,974 | 189,110 | 2,305,689 | 1.0 |
| 2 | simulated_annealing | 1,367,529 | 1,322,974 | 189,110 | 2,305,689 | 1.0 |
| 5 | cross_entropy | 1,364,125 | 1,322,974 | 189,110 | 2,290,835 | 1.0 |
| 6 | fixed_grid / physics_guided_grid / beam_search / hybrid / exhaustive_small / coordinate_descent | 1,165,080 | 818,987 | 189,110 | 2,305,689 | 1.0 |
| 13 | random_grid | 1,128,056 | 715,633 | 189,110 | 2,290,835 | 1.0 |
| 14 | **clock_only** | 420,629 | 141,833 | 121,864 | 1,063,744 | **0.43** |

**`hierarchical_search` is the overall winner** (+31% over the tied core-grid family, +262% over clock-only)
and the only method that is both top-ranked AND Pareto-safe on every window. **clock_only is the worst and
the only method that fails the Pareto gate** (safe on 3/7, fails all 3 markets — SLA worse than baseline).

## Ablation (memory_bound_decode, budget 100 — the component decomposition)

On a window whose optimum is **in the core grid**, every variant reaches it (gp/$ 1,963,836) — so the
ablation here isolates **evaluation cost and the anchor contract**, not reward: `beam_no_anchors` 34 evals
(anchors **absent** — invariant violated, diagnostic-only), `beam_search` 47, `physics_guided_grid` /
`beam_physics_seed` 56, `progressive_widening` 63, `random_grid` 100. So the anchor floor costs ~13 extra
evaluations for the containment guarantee; physics-seeding adds ~9. **The components' REWARD value appears
only when the optimum is outside the core grid** (the markets), where connected-space search (hierarchical /
widening / CEM) is what separates +570% from +1273% — the grids and the physics/beam variants cannot.

## Honesty

- **No simulator / reward / Pareto-gate / baseline / cost change; no tuning.** Every method scored by the
  identical memoized world rollout; the only variable is which bundles each method probes.
- **Bounded + simulator-inferred.** Budgets ≤100, one decision per window, 4 synthetic + 3 markets. The
  **direction** (clock-only worst everywhere; multi-knob recovers the win; connected-space search beats
  core-grid search on real markets; no single method wins all windows) is the robust finding; the absolute
  gp/$ magnitudes are SIMULATOR_INFERENCE.
- **No headline claimed beyond the Pareto gate.** clock_only fails it on every market (reported, not hidden).
  The 250/500/1000 budgets are left for a follow-up (the connected-space exploration balloons the per-window
  rollout union there) — noted, never silently dropped. 0 timeouts.
- The recommendation is in `research/PLANNER_ARCHITECTURE_RECOMMENDATION.md`.
