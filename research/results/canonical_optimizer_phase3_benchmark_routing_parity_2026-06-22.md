# Canonical Optimizer — Phase 3 Benchmark-Routing Parity (2026-06-22)

> **Phase 3 of `research/OPTIMIZER_UNIFICATION_PLAN.md`.** Makes public benchmark
> entry points call the canonical `AureliusOptimizer` interface instead of
> constructing the underlying engines directly. **Architecture-unification parity
> step, not a new optimization run:** no new optimizer, no new priors, no
> benchmark-definition/assumption changes, no FIFO-only claim, 0% KPI drift.
> Directional simulator evidence only (`docs/RESULTS.md §8` unchanged/unmet).

## Entry-point classification

| Entry point | Engine before | Phase 3 action | Status |
|---|---|---|---|
| `benchmarks/canonical_backtests.py` (canonical energy benchmark) | `JobScheduler(cfg)` direct | route → `AureliusOptimizer(config=cfg)` | **ROUTED** |
| `benchmarks/gpu_routing_backtest.py` | `JobScheduler(cfg, gpu_placement_scorer=…)` | route → `AureliusOptimizer(config=cfg, gpu_placement_scorer=…)` | **ROUTED** |
| `benchmarks/srtf_backtest.py` | `JobScheduler(cfg)` ×2 | route → `AureliusOptimizer(config=cfg)` ×2 | **ROUTED** |
| `benchmarks/srtf_contention_backtest.py` | `JobScheduler(cfg)` | route → `AureliusOptimizer(config=cfg)` | **ROUTED** |
| `benchmarks/srtf_serving_backtest.py` (serving queue benchmark) | shim → extracted fn (Phase 2) | route shim → `AureliusOptimizer(policy="serving_queue")` | **ROUTED** |
| `traces/backtest.py` Azure LLM 2024 replay | replica-provisioning autoscaler (Erlang-C), **no `JobScheduler`** | — | **CANNOT ROUTE YET** (API mismatch: needs a `ReplicaScalingPolicy`, Phase ≥4) |
| `traces/backtest.py` BurstGPT replay | same autoscaler | — | **CANNOT ROUTE YET** (same) |
| public-trace rollup (`docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md`) | manual aggregation of per-trace JSONs (no runner) | — | **N/A** (no code path; components are the routed/replay runners) |
| `backtesting/engine.py` (`BacktestEngine`, used by `benchmarks/run_benchmark.py`) | `JobScheduler(self.config)` | — | **CAN ROUTE — DEFERRED** (shared core engine also used by simulation/shadow; wider blast radius → separate step) |

**Five entry points routed; energy + serving worlds now call the canonical
facade.** The Azure/BurstGPT replays are a different decision type (per-tick
replica provisioning), so they map to a future `ReplicaScalingPolicy`, not the
energy/serving policies — routing them now would be a behavior change, not a
parity refactor, so they are correctly left until that policy exists.

## How parity is preserved
- Energy: `AureliusOptimizer(policy="energy")` is the Phase-1 verbatim delegate to
  `JobScheduler`; routing only changes the *construction call site*. GPU routing
  is passed through as a `scheduler_kwarg` (`gpu_placement_scorer`,
  `region_gpu_types`), unchanged.
- Serving: the benchmark's Phase-2 shim now calls a module-level
  `AureliusOptimizer(policy="serving_queue")` which dispatches to the same
  extracted `simulate_decoupled_hybrid_abs_conformal` with the same injected
  `_summarize`. Every call site is unchanged.
- No objective formula, SLA definition, queueing model, batching assumption,
  price trace, workload trace, scenario hash, or `_summarize`/`economics.py` math
  was touched.

## Parity evidence — 0% KPI drift
- **Energy canonical benchmark**: `python scripts/run_canonical_backtests.py --json`
  byte-identical before vs. after; and the existing golden-snapshot test
  `tests/test_canonical_energy_backtest.py` (**17 passed**) now exercises the
  routed code and reproduces `aurelius/benchmarks/golden/canonical_energy_backtest.json`
  exactly (`sla_safe_goodput_per_infra_dollar=0.337299`).
- **Serving abs-conformal benchmark**: `python scripts/run_abs_conformal_backtest.py`
  result JSON KPIs byte-identical (only `timestamp` differs); Azure **+313.14%**,
  BurstGPT **+557.12%** (unchanged).
- **Routed == direct**: `test_energy_routing_matches_direct_scheduler`
  (`AureliusOptimizer(cfg).optimize(...)` schedule + objective == `JobScheduler(cfg).solve(...)`
  on the full 1000-job canonical trace); `test_serving_shim_routing_matches_extracted_impl`.
- **Tests**: 11 new Phase-3 routing guards + 139 routed-benchmark/parity tests +
  17 canonical-golden tests, all green; ruff clean on changed files.

## Required guarantees → proof
| Requirement | Proof |
|---|---|
| Benchmark entry points use AureliusOptimizer | `test_energy_benchmark_constructs_aurelius_optimizer` (4 modules; no `JobScheduler(` construction remains), `test_serving_benchmark_routes_through_optimizer` |
| Energy benchmark 0% KPI drift | golden test (17) + byte-identical JSON + `test_energy_routing_matches_direct_scheduler` |
| Serving benchmark 0% KPI drift | byte-identical JSON + `test_serving_shim_routing_matches_extracted_impl` |
| No benchmark definitions changed | `test_no_benchmark_definitions_changed` (constants pinned); no runner/scenario/`_summarize` edits |
| No new calculated priors added | `test_routing_added_no_priors` (decision module prior-free); routing is construction-only |
| No FIFO-only claim promoted | `test_no_fifo_only_claim_promoted` (oracle + rel deltas + `shadow_tag`) |
| No actual-output-token leakage at decision time | `test_no_actual_token_leakage_static` (`.actual_tokens` only in `calibrator.update`) |

## Scope discipline (what was NOT done)
- No new optimizer; energy core (`scheduler.py`) and the serving discipline
  untouched.
- Azure/BurstGPT replicas + `BacktestEngine` intentionally **not** routed
  (different decision type / shared core engine).
- 3 remaining `ruff` findings (`oracle_gp`, `baseline_sched_by_id`,
  `TTFTShadowPrior` annotation) are **pre-existing on main** in functions this PR
  did not modify; left as-is.

## Baseline-governance note (per BASELINE GOVERNANCE directive)
Phase 3 makes **no optimization claim**: the routed configuration is identical to
Current Main and to Best Aurelius (0% drift). There is therefore no Current-Main
vs Best-Aurelius vs Candidate comparison to report — by construction they are the
same configuration. Future *optimization* phases (e.g. policy combinations) must
report that three-way comparison and only count a candidate as a frontier
improvement if it beats the best validated AureliusOptimizer configuration.

## Phase 3 merge gate
| Gate | Status |
|---|---|
| All changed benchmark entry points route through AureliusOptimizer | ✅ 5 routed |
| Energy benchmark 0% KPI drift | ✅ golden + byte-identical |
| Serving benchmark 0% KPI drift | ✅ byte-identical |
| Public rollup 0% drift where touched | ✅ N/A (not touched; components unchanged) |
| Benchmark definitions unchanged | ✅ |
| Tests pass | ✅ 11 Phase-3 + 139 + 17 golden |
| main verified after merge | _pending merge_ |
