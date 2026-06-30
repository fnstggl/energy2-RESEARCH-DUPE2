# Hierarchical Planner Default Audit (Phase A)

Should `hierarchical_search` (the PR #123 tournament winner) become the default Aurelius benchmark planner?
This audit answers the prerequisite questions **before** any default is flipped. The hard rule: do not blindly
flip the default before this is written and before the default-change gate (Phase C) passes on the validation
windows.

## 1. What is the current default Aurelius planner?

The controller's dataclass default is **`use_adaptive_search=True`** (`controller.py`), i.e. the
`AdaptiveSearchPlanner` (beam / CE / coordinate over the roofline-pruned connected space) when a world state is
attached; `physics_guided=False` (the PR #122 `BoundedBeamPlanner` is opt-in). **However**, PR #118/#121 found
the bare adaptive planner over the full connected space is **intractable at hourly cadence** (did not complete
in > 3 min), which is exactly why the backtests override it (clock-only, then the physics-guided beam). So the
*deployable* default in practice is the **physics-guided core-grid beam** (PR #122); the *dataclass* default
(`AdaptiveSearchPlanner`) is the intractable one. The default-change comparison uses the physics-guided beam as
the realistic current default and reports the adaptive planner's intractability honestly.

## 2. What did PR #123 compare?

14 search methods on the SAME world / workload / action space / baselines / seed, at equal **evaluation**
budgets (10/25/50/100) and wall-clock, across 4 synthetic bottleneck fixtures (exhaustive-able → true regret)
and 3 real electricity-market decisions (pjm/ercot/caiso expensive). Methods: clock_only, fixed/expanded grid,
physics_guided_grid, random_grid, beam_search, progressive_widening, **hierarchical_search**, coordinate
descent, cross_entropy, random_restart, simulated_annealing, hybrid, exhaustive_small.

## 3. Which method won on average Pareto-safe gp/$?

**`hierarchical_search`** — average Pareto-safe gp/$ **1,524,356**, ahead of progressive_widening /
random_restart / simulated_annealing (1,367,529), cross_entropy (1,364,125), and the entire core-grid family
(fixed_grid / physics_guided_grid / beam_search / hybrid / exhaustive_small, all tied at **1,165,080**).
clock_only was last (420,629) and the only method to fail the Pareto gate (on the markets).

## 4. Which method had the lowest search regret?

**`hierarchical_search`** — regret **0** on every window (the only method at 0 on the real markets, where the
core-grid methods sit at ~52% because they cannot reach the connected surfaces).

## 5. Which method timed out least?

**None timed out** — the tournament reported `timeout_rate = 0.0` across all windows. `hierarchical_search`
ran at ~80 evaluations per decision, well within the per-cell cap.

## 6. Which method contained the required anchor bundles?

Every non-`clock_only` method evaluated the named anchors (`anchors_evaluated=True`), including
`hierarchical_search` (it seeds the anchors before the group search). `clock_only` deliberately excludes them
(the artifact). 0 anchor-contract violations.

## 7. Which method found connected-surface wins?

**`hierarchical_search`** (and, partially, progressive_widening / CEM / annealing). On the real markets the
optimum lies in the **connected** space (placement / migration / routing / capacity-policy). hierarchical's
slow/medium/fast timescale decomposition reaches it and won by ~2× (pjm 1,454,636 vs the core-grid methods'
709,283). The core-grid methods (fixed_grid, physics-guided beam, hybrid) **cannot** see those surfaces and
all tied at +570% with ~52% regret. This is the central PR #123 finding the default change rests on.

## 8. What are the risks of making `hierarchical_search` default?

1. **Simulator-inferred magnitudes.** The +1273% market figure is bounded and SIMULATOR_INFERENCE; the
   *direction* (connected-space search beats core-grid search) is robust, the exact number is not.
2. **Connected-surface fidelity.** hierarchical's win leans on placement / migration / routing levers whose
   magnitudes are SIMULATOR_INFERENCE (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`); a weakly-calibrated surface could
   inflate the win — Phase G attributes this explicitly.
3. **Runtime.** It costs more evaluations than the grids (~80 vs ~55); acceptable here, but it must stay
   bounded and timeout-protected.
4. **int4 / quality risk.** Must remain excluded from any headline unless a quality model exists.
5. **Window coverage.** PR #123 used one decision per window on 7 windows; a default change should hold across
   the broader benchmark ladder.

## 9. What validation is required before changing the default?

The **default-change gate** (Phase C), across the validation windows: gp/$ higher than the current default,
SLA not worse, the `production_scheduler` AND `sla_aware` Pareto gates pass (or failures documented), required
anchors always contained, search regret lower than the current default where measurable, runtime bounded,
timeout rate acceptable, **no oracle data**, and **no int4 / quality-risked action in the headline**. All
conditions must pass; any failure keeps it opt-in with the failed condition documented.

## 10. Should `hierarchical_search` become default for benchmark runs, production simulation runs, or both?

**Benchmark runs first.** If the gate passes on the validation ladder, this PR makes `hierarchical_search` the
default planner for **benchmark / standard-MPC reporting** (the task's authorisation: passing the gate is
sufficient to flip the benchmark default — no separate PR needed). **Production simulation runs should remain
opt-in** until the connected-surface fidelity (Phase G) and broader-window validation are confirmed — the
honest, conservative split. This audit does not itself flip anything; the gate decides.
