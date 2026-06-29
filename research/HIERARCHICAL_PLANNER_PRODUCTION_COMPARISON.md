# Hierarchical Planner — Production Comparison & Default-Change Decision (Phase B/C/E/H)

Should `hierarchical_search` (the PR #123 tournament winner) become the **default benchmark planner**, and does
it survive a production-grade comparison? This answers the 15 questions, applies the default-change gate to the
real ladder, and records the decision. Evidence: `data/external/mpc_controller/ladder_benchmark.json` (Phase E),
`…/hierarchical_regret_validation.json` (Phase F), `…/default_change_gate_verdict.json` (Phase C applied).
Magnitudes are **SIMULATED**; directions robust. No tuning; the Pareto gate is unchanged.

## 1. What planner modes are now selectable?

The controller exposes a `planner_mode` (default `None` → behaviour-preserving). Package modes (driven through
the PR #123 planner `run_method`): `fixed_multi_knob_grid`, `physics_guided_grid`, **`hierarchical_search`**,
`hierarchical_search_with_progressive_widening`, `exhaustive_small_diagnostic`. The existing controller branches
remain for `clock_only_diagnostic` (explicit candidates), `physics_guided_beam` (the prior default), the
adaptive search, and `oracle_diagnostic` (`planning_oracle_records`). The planner package imports **nothing**
from `production_baselines` (AST-enforced) — it is the Aurelius optimiser path, separate from the baseline.

## 2. What was the prior default, and what is the new default?

**Prior:** the PR #122 physics-guided bounded-beam (the *deployable* default; the dataclass `use_adaptive_search`
is intractable at hourly cadence per #118, so the beam was the realistic default). **New (this PR):**
`hierarchical_search`, via the constant `controller.DEFAULT_BENCHMARK_PLANNER_MODE = "hierarchical_search"`,
because the default-change gate **passed**. The flip is declared + wired (the constant; `training._controller`
takes a `planner_mode` hook; the ladder's headline arm uses the constant), **not** merely left opt-in.

## 3. What is the default-change gate, and what did it return?

A pure 10-condition contract (`aurelius/environment/planner/default_change_gate.py`,
`tests/test_default_change_gate.py`). Applied to the ladder + regret aggregates
(`scripts/apply_default_change_gate.py`): **`flip_benchmark_default` — all 10 conditions PASS.**

## 4. The 10 conditions, with the measured values

| # | condition | result |
|--|--|--|
| 1 | gp/$ higher than prior default | ✅ 772,465 vs 692,067 (**+80,398 / +11.6%** avg) |
| 2 | SLA not worse than prior default | ✅ 0.000 vs 0.0139 |
| 3 | production_scheduler Pareto pass | ✅ +148% avg, SLA not worse |
| 4 | sla_aware Pareto pass (or documented) | ✅ +164% avg, SLA not worse |
| 5 | required anchors always contained | ✅ anchors evaluated on every fixture |
| 6 | search regret ≤ prior default | ✅ 0.0% vs 0.64% (Phase F, exact) |
| 7 | runtime bounded | ✅ ≤75 evals/decision ≤ 120 budget |
| 8 | timeout rate acceptable | ✅ 0/24 cells timed out |
| 9 | no oracle data | ✅ causal; oracle is a separate arm |
| 10 | no int4 / quality-risked headline lever | ✅ `quality_sla_risk_mean=0.0` (fp8, not int4) |

## 5. hierarchical_search vs the prior default — abs + pct

| market | hierarchical gp/$ | prior default gp/$ | abs Δ | pct Δ | SLA (hier vs prior) |
|--|--|--|--|--|--|
| pjm | 783,862 | 685,113 | +98,749 | **+14.4%** | 0.000 vs 0.018 |
| ercot | 785,952 | 747,856 | +38,095 | **+5.1%** | 0.000 vs 0.006 |
| caiso | 747,580 | 643,230 | +104,351 | **+16.2%** | 0.000 vs 0.018 |
| **avg** | **772,465** | **692,067** | **+80,398** | **+11.6%** | 0.000 vs 0.014 |

It beats the prior default on gp/$ on every market **and** has a strictly lower SLA-violation rate.

## 6. Did hierarchical have lower search regret? (Phase F, exact)

**Yes.** On the exhaustive-able synthetic fixtures (memory-bound / compute-bound / SLA-tight / queue-bound),
hierarchical's true regret is **0.0% on all four** vs the prior default's max **0.64%**. On `sla_tight` the
prior default (and exhaustive-over-core-surfaces) leave 0.64% on the table because the optimum is a
**connected-surface** bundle they cannot reach — hierarchical reaches and evaluates it (regret 0). That is the
"reach" advantage demonstrated with **exact** regret, not just a bounded reference.

## 7. Were the required anchors always contained?

**Yes** on every fixture (`anchors_evaluated=True`). The anchor contract — the named known-good bundles are
always in the evaluated set — held everywhere (Phase F `anchors_held_everywhere=True`). Note: the regret
reference (true optimum) sometimes lies **outside** the static core-grid "reachable" set (that is precisely the
connected-surface bundle hierarchical reaches), so the `true_opt_contained` flag reading False is the *reach*
signal, not an anchor violation.

## 8. Was runtime bounded / any timeouts?

**Bounded, zero timeouts.** hierarchical ran **72–75 evaluations/decision** (≤ the 120 cap), ~40–55s/decision
on the markets; 0/24 ladder cells timed out. It costs more than the beam (~50–55 evals) but stays well within
the per-cell wall-clock cap.

## 9. How close is hierarchical to the oracle?

Within **0.9–2.4%** (pjm 783,862 / 803,030; ercot 785,952 / 797,483; caiso 747,580 / 754,189). The small
oracle gap means the causal forecast + bounded search leave very little on the table in these windows — the
residual is forecast fidelity, not search.

## 10. Does any of the win depend on a quality-risked (int4) lever?

**No.** `allow_quality_risk=False` is enforced in both the planner package and the physics-guided generator;
the realized `quality_sla_risk_mean=0.0` confirms the winning bundle uses **fp8** (lossless-safe), never int4.
The headline is quality-safe.

## 11. Where does hierarchical's edge over the prior default come from?

The prior default (physics-guided beam) searches the **core grid** well but cannot vary the **connected**
surfaces. hierarchical reaches them: on the markets the winner adds `capacity_policy=forecasted_mcs`,
`placement=network_aware`, `admission=class_aware`, `ordering=abs_conformal` beyond the core grid — the PR #123
"reach" finding, here worth +5–16% over the beam (and ~2× over a pure core-grid method). Full attribution +
fidelity: `research/CONNECTED_SURFACE_VALUE_ATTRIBUTION.md`.

## 12. Is the flip scoped honestly (benchmark vs production)?

**Yes.** The gate authorises the **benchmark** default flip (the audit Q10 split). hierarchical_search is now
the default **benchmark** planner (constant + wired). **Production-simulation runs stay on the physics-guided
beam** until the broader-window validation (cheap/volatile windows, more markets, higher budgets) confirms the
connected-surface fidelity — the conservative split in `HIERARCHICAL_PLANNER_DEFAULT_AUDIT.md`. The flip is
deliberately *not* forced into the dataclass default, so raw construction stays behaviour-preserving and the
specialized isolation backtests keep their explicit planners.

## 13. What are the residual risks?

1. **Magnitude is SIMULATOR_INFERENCE.** The +11.6% (vs prior default) / +148% (vs production_scheduler) rests
   on the simulator's roofline economics; direction robust, exact size not production-validated.
2. **Window coverage.** Bounded to the expensive window on 3 markets, 3 decisions each. The cheap/volatile
   windows and higher budgets are not yet run.
3. **Runtime.** ~1.4× the beam's evaluations; fine here, must stay timeout-protected at scale.
4. **Connected-surface fidelity.** placement/capacity-policy magnitudes are SIMULATOR_INFERENCE
   (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`); they drive the reach win and need pilot validation before production.

## 14. What would keep it opt-in / revert the flip?

Any gate condition failing on broader windows: a window where hierarchical does not beat the prior default on
gp/$, or worsens SLA, or fails the production_scheduler/sla_aware Pareto clause, or regret exceeds the prior
default, or a timeout/runtime blow-up, or a quality-risked lever entering the headline. The gate is a pure
function re-runnable on any new ladder; if it returns `keep_opt_in`, the flip reverts and the failed condition
is documented.

## 15. Bottom line

`hierarchical_search` **passed the default-change gate on the validation ladder** — it beats the prior default
on gp/$ at no SLA cost (+11.6% avg), Pareto-dominates both the realistic `production_scheduler` (+148% avg) and
`sla_aware` (+164% avg), has **0% search regret** (≤ the prior default's 0.64%), keeps the anchors, stays
bounded (≤75 evals, 0 timeouts), uses no oracle and no quality-risked lever, and sits within ~1–2% of the
oracle. So it is now the **default benchmark planner**; the physics-guided beam is retained as the labelled
prior default and the production-simulation default pending broader-window validation. The result is
**SIMULATED** and bounded; the direction is robust, the magnitude is not yet production-validated, and we say so.
