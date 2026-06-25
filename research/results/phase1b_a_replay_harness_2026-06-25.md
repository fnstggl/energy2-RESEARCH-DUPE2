# Phase 1b-A: ReplayHarness Unified Entry Point — 2026-06-25

## Run Classification

**ARCHITECTURE CONVERGENCE — Five-Failure Rule Compliant (ACTIVE)**

No frontier improvement attempted. This run implements Phase 1b-A: the unified
`ReplayHarness` entry point that dispatches all four Aurelius replay loops
through a single `run()` API returning `ReplayEvaluationResult` objects.

---

## 1. PR Hygiene

| PR  | Title | Classification | Action |
|-----|-------|----------------|--------|
| #70 | Benchmark realism audit (2026-06-24) | Obsolete (base = `claude/happy-pascal-pvp0fd`, not main) | Left open |
| #54 | Phases 4+5 Canonical Frontier Discovery | Research artifact / needs human review | Left open |

No open PRs are mergeable (both blocked by classification). No new mergeable PRs
exist on this branch yet (PR will be created after this commit).

---

## 2. Repository Audit

### Architecture Status

| Phase | Status |
|-------|--------|
| Phase 1a — Canonical AureliusOptimizer interface | DONE |
| Phase 2 — ServingQueuePolicy extraction | DONE |
| Phase 3/3b/3c — AMCSG/SOTSS-MIN/OSOTSS routing | DONE |
| Phase 3d — GenAI canonical routing | DONE |
| Phase 3e — CA/SHU backtest serving routing | DONE |
| Phase 4 — Frontier rho adaptation | DONE (null on fixtures; impl retained) |
| Phase 5 — Dead code deprecation | DONE (2,873 LOC removed) |
| **Phase 1b-B — Unified ReplayEvaluationResult** | **DONE** |
| **Phase 1b-A — ReplayHarness unified dispatch** | **DONE (this run)** |
| Phase 1b-C — Energy overlay on serving traces | Not started |

### Test Suite Health

- 153 parity tests passing (22 new Phase 1b-A tests + 131 prior)
- All canonical paths parity-gated at 0% KPI drift
- No regressions detected

---

## 3. Bottleneck

**Question: What prevents another real +25% against the strongest fair baseline?**

Structural barriers confirmed unchanged from prior runs:
1. BurstGPT 15-request n_sla_safe gap — irreducible without new oracle/simulation model
2. Azure OSOTSS-to-oracle gap (0.33%) — requires oracle service time or new predictor
3. forecasted_mcs deployability gap — structural burst-onset prediction gap

**New opportunity opened by Phase 1b-A:**
- `ReplayHarness` enables unified evaluation of cross-domain policy combinations
- Phase 1b-C (energy overlay) is now the highest-priority remaining integration step
- Energy-aware replica scaling on serving traces has not been tested

---

## 4. Research Review

No new papers reviewed (architecture run, Five-Failure Rule active).

---

## 5. Implementation: Phase 1b-A

### What was implemented

**`aurelius/optimizer/replay_harness.py`** (~230 LOC):

- `ReplayHarnessError` — raised on invalid config
- `ReplayHarnessConfig` — unified config dataclass with validation
  - `benchmark_id`: one of `BENCHMARK_IDS`
  - `trace_id`: string label for output `ReplayEvaluationResult.trace_id`
  - `policies`: list of policy names to evaluate
  - `tick_seconds`: tick length in seconds (default 60.0)
  - `backend_kwargs`: forwarded verbatim to the backend's `run_backtest()` call
- `ReplayHarness` — the unified entry point
  - `run(config, data)` → `list[ReplayEvaluationResult]`
  - Dispatches to one of four private backends based on `config.benchmark_id`

### Backend dispatch table

| `benchmark_id` | `data` type | Backend called | KPI adapter |
|---|---|---|---|
| `"replica_scaling"` | `Sequence[NormalizedLLMRequest]` | `backtest.run_backtest()` | `from_backtest_policy_result()` |
| `"genai_serving"` | `Sequence[NormalizedGenAIRequest]` | `genai_backtest.run_backtest()` | `from_genai_policy_result()` |
| `"energy"` | `{}` (not used; backend builds its own data) | `canonical_backtests.run_canonical_backtest()` | `from_canonical_policy_metrics()` |
| `"serving_queue"` | `dict` with `"sim_dicts"`, `"n_requests"`, `"n_ticks"`, `"servers"` | pass-through adapter | `from_srtf_sim_dict()` |

### Design choices

- **Deferred imports**: each `_run_*` method imports its backend inside the function
  body to avoid triggering heavyweight dependency loads at import time.
- **serving_queue pass-through**: the SRTF engine has 14k LOC and dozens of
  specialised entry-points. Rather than encoding one unified signature, the harness
  accepts pre-computed sim_dicts from any SRTF entry-point and normalises the output.
- **Policy ordering preserved**: result list follows `config.policies` order.
- **Missing policies silently skipped**: policies not in the backend output are
  omitted from the result list (relevant for the pass-through backends).

### Files added/modified

| File | Change |
|------|--------|
| `aurelius/optimizer/replay_harness.py` | **New** (~230 LOC) |
| `aurelius/optimizer/__init__.py` | Add `ReplayHarness`, `ReplayHarnessConfig`, `ReplayHarnessError` exports |
| `tests/test_phase1b_a_replay_harness_parity.py` | **New** (22 parity tests) |
| `research/ROADMAP.md` | Updated |
| `research/GAP_ANALYSIS.md` | Updated |
| `research/OPTIMIZER_UNIFICATION_PLAN.md` | Phase 1b-A marked DONE |

---

## 6. Canonical Public Replays

Three canonical replays run to confirm parity (no behavior change from Phase 1b-A):

| Benchmark | Result | vs Baseline | Status |
|-----------|--------|-------------|--------|
| AMCSG Azure LLM 2024 | 150,630 gp/$ | +0.93% vs GSF(9.5%) | ✓ matches ROADMAP |
| AMCSG BurstGPT HF | 168,270 gp/$ | +0.30% vs GSF(9.5%) | ✓ matches ROADMAP |
| Energy canonical (constraint_aware) | 0.337299 gp/$ | +11.1% vs current_price_only | ✓ matches ROADMAP |

All replays confirmed deterministic (std=0). No regressions.

---

## 7. Same-Conditions Checklist

This run is an architecture parity run — no optimizer decisions changed.

- Same trace: ✓ (same BurstGPT/Azure fixtures)
- Same SLA: ✓ (unchanged)
- Same cost denominator: ✓ (unchanged)
- Same GPU-hour accounting: ✓ (unchanged)
- Same physics: ✓ (unchanged)
- Same decision-time information: ✓ (unchanged)
- Same evaluation method: ✓ (unchanged)
- KPI drift: **0.00%**

---

## 8. KPI Table

| Metric | Value |
|--------|-------|
| KPI change | **0.00%** |
| GPU-hour delta | **0** (architecture run) |
| SLA violations delta | **0** |
| New tests added | **22** |
| Total canonical parity tests | **153** |

---

## 9. Gain Decomposition

No KPI gain in this run — this is a pure architecture convergence step.

Phase 1b-A value:
- **Enablement**: `ReplayHarness` is the unified entry point that future
  combination policy tests (Phase 1b-C) will use
- **Architecture simplification**: one call site for all 4 replay loops
- **Cross-loop parity**: same `ReplayEvaluationResult` schema regardless of which
  backend ran the policy

---

## 10. Run Classification

**ARCHITECTURE CONVERGENCE — Five-Failure Rule Compliant**

- Improves AureliusOptimizer: ✓ (via `aurelius.optimizer` export)
- New module: ❌ (not a new optimizer — a routing facade)
- New optimizer path: ❌
- KPI change: 0.00%
- SLA safety: maintained
- Five-Failure counter: **UNCHANGED**

---

## 11. Merge Recommendation

**MERGE** — safe infrastructure, 0% KPI drift, 22 new parity tests, no
benchmark changes, no optimizer behavior change. Fits the Five-Failure Rule
"architecture simplification" criterion.

---

> Directional simulator evidence only — NOT production savings (`docs/RESULTS.md` §8).
