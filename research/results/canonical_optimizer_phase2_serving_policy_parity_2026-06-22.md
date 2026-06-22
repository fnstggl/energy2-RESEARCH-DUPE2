# Canonical Optimizer — Phase 2 Serving-Policy Extraction Parity (2026-06-22)

> **Phase 2 of `research/OPTIMIZER_UNIFICATION_PLAN.md`.** Moves the strongest
> validated serving-queue discipline out of the benchmark monolith and behind the
> canonical `AureliusOptimizer` policy interface. **Parity extraction, not a
> research change:** no new optimizer, no new priors, no benchmark-definition or
> benchmark-assumption changes, no FIFO-only claim promoted. Directional
> simulator evidence only (`docs/RESULTS.md §8` gate unchanged and unmet).

## What was extracted
- **Discipline (strongest validated, run 2026-06-22-x):** Decoupled Hybrid SRPT
  with **absolute-error conformal adaptive α** — +313.14% (Azure LLM 2024) /
  +557.12% (BurstGPT HF) SLA-safe goodput/$ vs FIFO (directional).
- **Moved verbatim** from `aurelius/benchmarks/srtf_serving_backtest.py` to
  **`aurelius/optimizer/policies/serving_queue.py`**:
  - `AbsoluteErrorConformalCalibrator` (was benchmark lines 979–1066)
  - `simulate_decoupled_hybrid_abs_conformal` (was `_simulate_decoupled_hybrid_abs_conformal`, lines 5813–5954)
  - the 4 conformal constants it needs (identical values).
- **`ServingQueuePolicy`** (decision-layer policy) wraps the discipline and is
  reachable via `AureliusOptimizer(policy="serving_queue")`.

## How parity is preserved (extraction mechanics)
- The discipline body is byte-identical to the original except three intentional
  edits: public name; a `*, summarize` callback replacing the hard call to the
  benchmark's `_summarize`; and a dropped inner type annotation. Built
  programmatically from the exact source lines to guarantee fidelity.
- The benchmark **imports the moved symbols back** and keeps a thin shim
  `_simulate_decoupled_hybrid_abs_conformal(...)` that injects its own
  `_summarize`. Every existing call site and
  `from aurelius.benchmarks.srtf_serving_backtest import AbsoluteErrorConformalCalibrator`
  keeps working unchanged. Dependency direction is one-way (benchmark → optimizer);
  no circular import. The benchmark no longer **owns** this discipline's logic.
- `_summarize` (evaluation/KPI math) stays in the benchmark — the decision layer
  does not absorb the evaluation layer.

## Parity evidence — 0% KPI drift
**Serving benchmark** (`python scripts/run_abs_conformal_backtest.py`), before vs.
after extraction: result JSON KPIs **byte-identical** (only the `timestamp` field
differs); stdout identical. Headline (unchanged):

| Trace | Discipline | Goodput/$ | vs FIFO | α | retention |
|---|---|---:|---:|---:|---:|
| Azure LLM 2024 | abs-conformal (live) | 55097.06 | +313.14% | 0.00022 | 97.8% |
| BurstGPT HF | abs-conformal (live) | 42901.59 | +557.12% | 0.00056 | 88.3% |

**Energy benchmark** (`python scripts/run_canonical_backtests.py --json`):
**byte-identical** before vs. after (`sla_safe_goodput_per_infra_dollar=0.337299`)
→ **0% energy KPI drift** (energy path untouched).

**Tests:** `142 passed` across the serving + parity + energy-core suites,
including the **17** `tests/test_abs_conformal_backtest.py` tests that exercise
the extracted calibrator + discipline via the benchmark re-export; plus **9** new
Phase-2 guard tests (`tests/test_canonical_serving_policy_phase2.py`). Phase-1
parity tests updated (serving_queue is now implemented) and green. ruff clean on
all new/changed optimizer + test files.

## Required guarantees → how proven
| Requirement | Proof |
|---|---|
| Extracted policy matches old benchmark-local behavior | serving benchmark byte-identical; `test_policy_path_matches_benchmark_shim`; the 17 re-export tests |
| No benchmark assumptions changed | `test_benchmark_constants_unchanged` (TTFT/TPOT/α/warmup/window/target all pinned; extracted == benchmark) |
| No FIFO-only claim promoted | `test_no_fifo_only_claim_promoted` — result still reports oracle + rel-conformal deltas + `shadow_tag`; policy module computes no goodput/claim |
| No actual-output-token leakage at decision time | `test_no_actual_token_leakage_static` (`.actual_tokens` only inside `calibrator.update(...)`) + `test_no_actual_token_leakage_behavioral` (sub-warmup: scrambling actuals leaves the schedule identical) |
| No new calculated priors introduced | `test_no_new_priors_introduced` (no sklearn/lightgbm/numpy import; reads `predicted_tokens`, never assigns) |
| Accessible through AureliusOptimizer / policy interface | `test_serving_policy_accessible_via_optimizer`; `IMPLEMENTED_POLICIES = {energy, serving_queue}` |

## Scope discipline (what was NOT done)
- No optimizer invented; the discipline is the existing one, moved.
- No serving physics, KPI math (`_summarize`/`economics.py`), scenario hashes, or
  benchmark runners changed.
- Energy core (`aurelius/optimization/scheduler.py`) untouched.
- One remaining `ruff` finding (`F841 oracle_gp`) is **pre-existing on main**
  (run-z compound-economic code, unrelated to this extraction) and left as-is to
  avoid changing benchmark code outside scope.

## Phase 2 merge gate
| Gate | Status |
|---|---|
| Serving benchmark parity-identical (within deterministic tolerance) | ✅ byte-identical KPIs |
| Energy benchmark 0% drift | ✅ byte-identical |
| Benchmark definitions unchanged | ✅ |
| Extracted policy accessible through `AureliusOptimizer` | ✅ `policy="serving_queue"` |
| Tests pass | ✅ 142 + 9 Phase-2 |
| `main` verified after merge | _pending merge_ |
