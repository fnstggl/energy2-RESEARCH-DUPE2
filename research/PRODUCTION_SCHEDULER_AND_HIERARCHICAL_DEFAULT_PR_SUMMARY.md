# PR Summary — Production-Scheduler Baseline + Hierarchical Planner as Default (19 questions)

Two deliverables in one PR: (1) a single canonical, realistic `production_scheduler` benchmark baseline — the
honest production bar future headlines compare against; and (2) productionising the PR #123 tournament winner
`hierarchical_search` as a selectable planner with a **default-change gate** that, having **passed**, flips the
default benchmark planner. All evidence is SIMULATED directional; nothing was tuned to the benchmark; the Pareto
gate is unchanged; the two deliverables are kept strictly separate (the user's clarification).

## 1. What does this PR deliver?

A new `production_scheduler` baseline arm (+ the missing `vllm_only` / `topology_aware` rungs); the
`hierarchical_search` planner wired as a selectable mode; a pure 10-condition **default-change gate**; an 8-arm
benchmark ladder runner; a regret/containment validation; a connected-surface attribution; the default flip
(gate passed); 5 research docs; 4 test files (36 tests); 3 result artifacts.

## 2. What is `production_scheduler`, and where does it live?

A deterministic, causal heuristic `decide_fn(history)→action` in the **evaluation layer**
(`aurelius/environment/production_baselines.py`). It runs through the unchanged reward path like `fifo`/
`sla_aware`. It is **not** a planner mode, **not** routed through the MPC path, and shares **no** MPC-search /
economic / oracle / hierarchical code — enforced by an AST import scan (`test_production_scheduler_baseline`).

## 3. Did you keep production_scheduler separate from the Aurelius MPC path?

**Yes — by construction and by test.** production_scheduler imports only the standard library. The planner
package imports nothing from production_baselines (reverse AST test). The ladder keeps them as separate arms:
`production_scheduler` (baseline) vs `aurelius_mpc_*` (optimiser) vs `oracle_diagnostic` (upper bound). No
Aurelius decision is ever routed through production_scheduler.

## 4. What levers does production_scheduler use — and deliberately not?

**Uses:** SLA-aware ordering, backlog autoscaling + 1.25× headroom under pressure, class admission under
pressure, **continuous batching always on** (balanced→aggressive), KV-aware routing, rack-local placement, warm
pool via the autoscaler idle timeout. **Forgoes** (Aurelius's edge): precision arbitrage (bf16), DVFS/clock
(base), migration (off), spec decode (off), future prices, oracle, global economic objective, search.

## 5. How does production_scheduler rank against the other baselines?

It has the **best SLA of every baseline** on all 3 markets and **Pareto-dominates `sla_aware`** (higher gp/$
AND lower violations): pjm +12.0%, ercot +3.5%, caiso +4.5% gp/$ at roughly half the SLA-violation rate. Honest
caveat: `vllm_only` posts higher raw gp/$ at a worse SLA — a different point on the cost/SLA curve, not
Pareto-better. Detail: `research/PRODUCTION_SCHEDULER_BASELINE_RESULTS.md`.

## 6. The headline: does Aurelius MPC beat production_scheduler?

**Yes, Pareto-dominantly, every market** (default planner `hierarchical_search` vs `production_scheduler`,
abs + pct, SLA never worse):

| market | hierarchical | production | abs Δ | pct Δ | SLA |
|--|--|--|--|--|--|
| pjm | 783,862 | 330,711 | +453,151 | **+137.0%** | 0.000 vs 0.065 ✓ |
| ercot | 785,952 | 303,982 | +481,969 | **+158.6%** | 0.000 vs 0.071 ✓ |
| caiso | 747,580 | 301,228 | +446,352 | **+148.2%** | 0.000 vs 0.065 ✓ |

Aurelius also beats `vllm_only` on gp/$ AND SLA, so it Pareto-dominates the **entire** ladder.

## 7. Every gp/$ comparison has both absolute and percent deltas?

**Yes** — the ladder `summarize` emits `abs_delta` + `pct_delta` for each comparison (vs production_scheduler,
vs sla_aware, oracle gap), pinned by `test_ladder_benchmark_reporting`. The docs report both throughout.

## 8. Where does Aurelius's edge over production_scheduler come from?

The **economic arbitrage the baseline forgoes**: the winner ran `fp8` (lossless-safe), `clock=high`,
`spec=aggressive`, `batching=aggressive`, and **capacity consolidation** (0.75 vs prod's 1.25), plus a slightly
better connected placement (`network_aware` vs `rack_local`). Routing was `kv_aware` for **both** — no edge
there. Full attribution + fidelity labels: `research/CONNECTED_SURFACE_VALUE_ATTRIBUTION.md`.

## 9. Did you chase a positive result?

**No — the opposite.** Integration surfacing showed production_scheduler at gp/$ ≈ 10k (an obvious strawman)
from two unrealistic choices; both were hunted down and fixed *against* an easy Aurelius win: continuous
batching is always on (no batch-shrinking under burst), and no eager prewarm pool (the autoscaler idle timeout
is the warm pool; an eager pool's warm-hold dwarfed the served work with zero cold starts avoided at backtest
scale). Both fixes make the bar **stronger**. Neither was tuned to the benchmark.

## 10. What is `hierarchical_search`, and how is it selectable?

The PR #123 tournament winner: search by control timescale (slow capacity/placement/migration · medium
precision/batching/spec/clock · fast routing/admission/ordering) with cross-group coupling + a polish. It is a
`planner_mode` on the controller (default `None` → behaviour-preserving), driven through the planner package's
`run_method`. Modes: `fixed_multi_knob_grid`, `physics_guided_grid`, `hierarchical_search`,
`hierarchical_search_with_progressive_widening`, `exhaustive_small_diagnostic` (+ existing branches for
clock-only / physics-guided / adaptive / oracle).

## 11. What is the default-change gate, and what did it return?

A **pure** 10-condition contract (`planner/default_change_gate.py`). Applied to the real ladder + regret
aggregates (`scripts/apply_default_change_gate.py`): **`flip_benchmark_default` — all 10 conditions PASS**
(`data/external/mpc_controller/default_change_gate_verdict.json`).

## 12. The 10 conditions and their measured values?

(1) gp/$ +11.6% over prior default ✓; (2) SLA 0.000 ≤ 0.014 ✓; (3) production Pareto +148% ✓; (4) sla_aware
Pareto +164% ✓; (5) anchors always contained ✓; (6) regret 0.0% ≤ prior 0.64% ✓; (7) ≤75 evals ≤ 120 cap ✓;
(8) 0/24 timeouts ✓; (9) no oracle ✓; (10) no int4/quality-risked lever (`quality_sla_risk_mean=0.0`) ✓.

## 13. hierarchical_search vs the prior default (physics-guided beam)?

Beats it on gp/$ at strictly lower SLA on every market: pjm +14.4%, ercot +5.1%, caiso +16.2% (avg **+11.6%**,
+80,398 gp/$), SLA 0.000 vs ~0.014. Detail: `research/HIERARCHICAL_PLANNER_PRODUCTION_COMPARISON.md`.

## 14. Did you validate search regret and anchor containment?

**Yes (Phase F, exact regret).** On the exhaustive-able synthetic fixtures hierarchical has **0.0% regret on
all four** vs the prior default's max **0.64%**; on `sla_tight` it reaches a connected-surface optimum the
core-grid methods cannot (regret 0). Anchors held on every fixture. `scripts/run_hierarchical_regret_validation.py`.

## 15. Is the default actually flipped — not merely left opt-in?

**Yes.** The gate passed, so `controller.DEFAULT_BENCHMARK_PLANNER_MODE = "hierarchical_search"` is declared,
`training._controller` takes a `planner_mode` hook, and the ladder's headline arm uses the constant. It is
deliberately **not** forced into the dataclass default (that would silently change the specialized isolation
backtests via branch precedence and break behaviour-preservation) — the flip is scoped to **benchmark
reporting**, declared + wired + evidenced.

## 16. Benchmark vs production split?

The gate authorises the **benchmark** default flip. **Production-simulation runs stay on the physics-guided
beam** until broader-window validation (cheap/volatile windows, more markets, higher budgets) confirms the
connected-surface fidelity — the conservative split from `HIERARCHICAL_PLANNER_DEFAULT_AUDIT.md`.

## 17. What is the oracle arm, and is it deployable?

`oracle_diagnostic` plans the strongest search against the **exact future** — a **non-deployable upper bound**,
a separate arm, never a headline. Aurelius's default is within **0.9–2.4%** of it, so the forecast+search leave
little on the table. (Its sub-second runtime is expected: a single known-future scenario is far cheaper than
the forecast ensemble.)

## 18. What is the fidelity, and what is excluded from the headline?

**SIMULATED / SIMULATOR_INFERENCE.** The dominant edge levers (fp8/clock/spec roofline economics) are
SIMULATOR_INFERENCE in magnitude (robust in direction — `WORLD_MODEL_ROBUSTNESS_AUDIT.md`); the best-calibrated
lever (kv-routing, TRACE_DERIVED) is *shared* with the baseline. **int4 is excluded** from every headline arm
(`allow_quality_risk=False`; realized `quality_sla_risk_mean=0.0`). The run is bounded: 3 markets, expensive
window, 3 decisions/window, req_cap 56.

## 19. Tests, lint, and what would overturn this?

**36 new tests pass** (production_scheduler registry/determinism/no-economic-or-oracle/separation; gate
blocks-unsafe/allows-safe/documented-exception; hierarchical selectable/anchor-contained/bounded/reports-counts/
reaches-connected-surfaces/default-flip-wired; ladder reporting abs+pct/headline=production_scheduler/oracle-
diagnostic-only) + 57 affected existing tests pass (no regressions); **ruff clean**. **What would overturn:** a
broader-window run where any gate condition fails (no gp/$ win, worse SLA, a failed Pareto clause, higher
regret, a timeout, or a quality-risked lever) → the pure gate returns `keep_opt_in`, the flip reverts, and the
failed condition is documented. We report the bounded result honestly and do not extrapolate the magnitude.
