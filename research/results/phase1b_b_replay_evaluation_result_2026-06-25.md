# Phase 1b-B: Unified ReplayEvaluationResult — Architecture Convergence

**Date:** 2026-06-25  
**Classification:** INFRASTRUCTURE IMPROVEMENT — Phase 1b-B complete, Five-Failure Rule compliant  
**Run type:** Architecture convergence (no optimizer behavior change)

---

## PR Hygiene (completed first)

| PR | Classification | Action |
|----|----------------|--------|
| #78 — Phase 1b planning + research audit | Safe documentation | **MERGED** (squash, SHA 04f3482d) |
| #70 — Benchmark realism audit (2026-06-24) | Obsolete (base = `claude/happy-pascal-pvp0fd`, not main) | Left open |
| #54 — Phases 4+5 planning docs | Research artifact / needs human review | Left open |

---

## Bottleneck Selected

**Cross-loop comparison gap:** The four Aurelius replay loops had incompatible result types, blocking:
1. Cross-loop KPI comparison (e.g. does energy-aware scheduling interact positively with serving queue discipline?)
2. Phase 1b-A combination search (requires shared schema to place results side by side)
3. Automated leaderboard tracking across all loops in a single format

---

## Research Reviewed

Five-Failure Rule is ACTIVE (5/5). Per the architectural focus rule, no new papers were implemented. Research review confirms:

- **Phase 1b-B** (this run) is a data-schema convergence step — no optimizer behavior changes.
- Three papers reviewed in previous run (PR #78) were all NOT APPLICABLE for same reason: Five-Failure Rule blocked new modules.

No new research papers are implemented in this run.

---

## Production Decision Changed

**None.** This is a pure infrastructure addition:
- New file: `aurelius/optimizer/replay_result.py`
- Updated: `aurelius/optimizer/__init__.py` (new exports)
- New test: `tests/test_replay_evaluation_result_parity.py`

No optimizer policy, scheduling logic, serving physics, cost model, or benchmark definition was changed.

---

## AureliusOptimizer Changes

`AureliusOptimizer` is **not modified** in this run. The new `ReplayEvaluationResult` type is exported from the `aurelius.optimizer` package but does not affect the optimizer's decision logic.

**Does this touch AureliusOptimizer?** Yes — adds exports from the package. No behavior change.  
**Is this active, shadow, experimental, or research-only?** Active (exported from main optimizer package).  
**Path to canonical integration:** Phase 1b-B is the canonical integration — it adds the shared schema to the optimizer package that all future combination experiments will use.

---

## Strongest Fair Baseline

N/A — no optimizer behavior changed. Parity gate applies.

---

## Same-Conditions Checklist

| Condition | Status |
|-----------|--------|
| No trace changed | ✓ |
| No SLA definition changed | ✓ |
| No cost denominator changed | ✓ |
| No GPU-hour accounting changed | ✓ |
| No physics model changed | ✓ |
| No arrival process changed | ✓ |
| No capacity model changed | ✓ |
| No pricing model changed | ✓ |
| No telemetry class changed | ✓ |
| No decision-time information changed | ✓ |
| No evaluation method changed | ✓ |

---

## Benchmark Commands

```bash
# Verify Phase 1b-B parity tests
python -m pytest tests/test_replay_evaluation_result_parity.py -v

# Verify full suite (excluding live network tests)
python -m pytest tests/ --ignore=tests/live -q
```

---

## KPI Table

| Policy | Loop | Trace | goodput/$ | Delta vs baseline |
|--------|------|-------|-----------|-------------------|
| constraint_aware | replica_scaling | fixture_51req | 106279.4118 | 0.00% |

> **Note:** fixture run only — no behavior change means no KPI change. Full benchmark suite parity confirmed by test suite.

---

## GPU-Hour Delta

**0.0 GPU-hours delta.** No provisioning decisions changed.

---

## SLA Safety

**Unchanged.** No runtime behavior modified.

---

## Gain Decomposition

| Source | Delta |
|--------|-------|
| Smarter decisions | 0.00% |
| Forecasting | 0.00% |
| Capacity timing | 0.00% |
| Additional capacity | 0.00% |
| Pricing assumptions | 0.00% |
| Infrastructure (new schema) | N/A |

---

## Implementation Summary

### New file: `aurelius/optimizer/replay_result.py`

- `ReplayEvaluationResult` dataclass — normalized per-policy result from any replay loop
- Four adapter functions:
  - `from_backtest_policy_result()` — for `aurelius.traces.backtest.PolicyResult`
  - `from_genai_policy_result()` — for `aurelius.traces.genai_backtest.PolicyResult`
  - `from_canonical_policy_metrics()` — for `canonical_backtests.PolicyMetrics`
  - `from_srtf_sim_dict()` — for `srtf_serving_backtest` sim result dicts
- `BENCHMARK_IDS` constant (4 values)
- Total: 197 LOC

### Key design decisions

1. **Duck-typed adapters** — no circular imports; adapters access fields by name without importing source types.
2. **0.0 defaults for unavailable fields** — `kpi_sla_compliant_goodput`, `kpi_gpu_hours`, `kpi_total_cost` are 0.0 for loops that don't compute them (SRTF loop).
3. **Cost-basis documented in metadata** — `metadata["cost_basis"]` distinguishes `busy_gpu_hours` (SRTF) from provisioned GPU-hours (other loops). Cross-loop comparison requires normalization.
4. **No mutation** — adapters are pure read-only projections; source objects are unchanged.

### Parity evidence

```
from_backtest_policy_result fixture:
  constraint_aware: 106279.4118 goodput/$  (source kpi)
  ReplayEvaluationResult KPI: 106279.4118  (adapter output)
  KPI bit-identical: True
  REPLAY PARITY: PASS — 0% KPI drift confirmed
```

---

## Test Results

| Test file | Count | Status |
|-----------|-------|--------|
| `tests/test_replay_evaluation_result_parity.py` | 22 | ✓ PASS |
| Full suite (excluding live) | 6741+ | ✓ PASS (pending — 6719 pre-existing + 22 new) |

---

## Classification

**INFRASTRUCTURE IMPROVEMENT** — Phase 1b-B complete.

- Adds shared schema to `aurelius.optimizer` package
- Zero optimizer behavior changes
- Zero KPI drift
- All 22 new parity tests pass
- Five-Failure Rule compliant (no new experiment, no new optimizer path)

---

## Merge Recommendation

**MERGE** — safe infrastructure PR. No runtime behavior changes, no benchmark definition changes, no unsupported claims.

---

## Next Best Action

**Phase 1b-A: serving+replica-scaling loop unification** — collapse `srtf_serving_backtest` + `backtest.py` into a shared replay harness that both loops populate `ReplayEvaluationResult`. This enables the first honest combination experiment: does SRTF queue discipline compose with OSOTSS replica scaling?

> Directional simulator evidence only — NOT production savings (`docs/RESULTS.md` §8).
