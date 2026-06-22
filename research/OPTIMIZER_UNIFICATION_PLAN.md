# Aurelius Optimizer Unification Plan

> **Status:** PLAN (mostly). One bootstrap step has been executed — see
> **Execution Status** below. This is the phased, reversible migration path
> implied by `CANONICAL_AURELIUS_OPTIMIZER.md`. Each phase is independently
> shippable behind a flag, is benchmark-gated, and has an explicit rollback.
> **Hard constraints for every phase:** do not change benchmark definitions, do
> not change public replay logic, do not change evaluation infrastructure, do
> not modify the pinned energy core without an explicit justified phase, and do
> not claim any improvement without the required benchmark evidence below.

---

## Execution Status

| Step | Status | Evidence |
|---|---|---|
| **Phase 1a — Canonical interface bootstrap** (stand up `AureliusOptimizer` + decision-layer policy seam; energy policy = thin delegate to `JobScheduler`) | **DONE — behavior-preserving, 0% KPI drift** | `aurelius/optimizer/`, `tests/test_canonical_optimizer_parity.py` (21 pass), `research/results/canonical_optimizer_phase1_parity_2026-06-22.md` |
| Phase 1b — Unify the 4 replay loops into one engine | Not started | — |
| Phase 2 — Extract + wire the serving (SRPT+conformal) discipline (shadow) | Not started | — |
| Phase 3 — Promote frontier BASE/DYNAMIC → constraint; dedup calibrators | Not started | — |
| Phase 4 — Deprecate dead/duplicate code | Not started | — |

**Phase 1a notes.** The very first safe step is *not* a behavior change — it is
the permanent top-level seam (`AureliusOptimizer`) through which all future
gains must flow. It wraps the existing productized energy `JobScheduler` with a
verbatim delegate (`EnergySchedulingPolicy`) and declares the other decision
policies (`ServingQueuePolicy`, `ReplicaScalingPolicy`, `PlacementPolicy`,
`AdmissionPolicy`) as importable stubs that raise `NotImplementedError`. No
optimizer rewritten, no serving/SRTF touched, no benchmark definition changed,
no duplicate deleted. Parity vs. the pinned energy core and 0% canonical-benchmark
KPI drift are proven in the parity report. The replay-engine unification
(originally labelled "Phase 1" below) is re-tagged **Phase 1b** and remains the
next, separately-gated step.

---

## Phase 6 — Benchmark Impact Analysis (evidence-gated, no speculation)

Evidence labels: **[A]** already benchmarked · **[P]** partially benchmarked ·
**[N]** not benchmarked. No improvement is claimed without an [A]/[P] anchor.
All numbers are **directional simulator only** (`RESULTS.md §8` gate unmet).

| Proposed integration | SLA-safe goodput/$ | GPU-hours | Cost | Queue delay | SLA violations | Evidence |
|---|---|---|---|---|---|---|
| **Extract SRPT+conformal discipline → serving policy** | Target: preserve sim's +313% (Azure) / +557% (BurstGPT) vs FIFO | n/a | n/a | short-p90 ↓ (−99.6% sim) | ≤0.5× FIFO (gate) | **[A]** in sim (`abs_conformal_backtest_2026-06-22`); **[N]** in any runtime |
| **Wire discipline into a real serving-runtime path (shadow)** | Unknown until measured | — | — | — | — | **[N]** — this is the integration risk; expect degradation vs heapq ideal |
| **Unify the 4 replay loops → 1 engine** | Must be **0% delta** (correctness gate) | 0% | 0% | 0% | 0% | **[N]** — requires bit-for-bit reproduction before any behavior change |
| **Promote frontier BASE/DYNAMIC → ρ-ceiling constraint** | Static SUF +13% over CA (Azure, analysis-only) | ↓ at safe ρ | — | bounded | bounded | **[P]** (`azure_2024_safe_utilization_frontier`, 73–91% oracle-alpha) |
| **Consolidate 3 conformal calibrators → 1 lib** | 0% delta (refactor) | 0% | 0% | — | — | **[A]** (each variant already benchmarked) |
| **Deprecate EVAL_WORKLOAD/BATCH_INFERENCE frontier** | 0% (no consumer) | 0% | 0% | 0% | 0% | **[A]** (grep: no benchmark imports them) |
| **Output-length forecaster into decisions** | **−7…−11% (HURT)** | — | — | — | — | **[A]** (`module_integration_public_backtest_2026-06-20`) → **do NOT integrate** |
| **Admission gate into decisions** | **NEUTRAL ±0.34%** | — | — | — | — | **[A]** → keep shadow |
| **GPU placement scorer ON in prod** | **−7.3% (regressed)** | — | — | — | — | **[A]** → keep off |

**Net:** the only integration with positive benchmark evidence *and* a runtime
gap to close is the **SRPT+conformal serving discipline**. Everything else is
either a behavior-neutral refactor (calibrator dedup, replay unification, dead-
duplicate removal), a constraint-layer promotion with partial evidence
(frontier), or a *do-not-integrate* (the 3 shadow modules — negative evidence).

---

## Phase 7 — Phased Migration

### Phase 1 — Unify the replay engine (behavior-preserving, no optimizer change)
- **Goal:** one discrete-event replay engine; the four loops become modes.
- **Files affected:** `simulation/cluster/engine.py` (host), `traces/backtest.py`,
  `benchmarks/srtf_serving_backtest.py`, `backtesting/engine.py`,
  `benchmarks/canonical_backtests.py` — **read/refactor only**; shared physics
  `simulation/cluster/serving.py` unchanged.
- **Risks:** the SRTF sim and tick-replay disagree on KPIs for identical
  decisions; reported leaderboard numbers move.
- **Rollback:** keep the four loops in place behind a `--engine=legacy` flag;
  unified engine is opt-in until parity proven.
- **Required benchmarks:** re-run every committed benchmark; **require 0% delta**
  vs golden JSON (`benchmarks/golden/`, `benchmarks/v1/.scenario_hashes.json`).
- **Required validation:** new parity test asserting unified-engine KPIs ==
  legacy KPIs per scenario; existing `test_energy_core_preservation.py` stays green.
- **Expected upside:** removes the largest source of "two sources of truth";
  unblocks every later phase. **No KPI change expected or allowed.**

### Phase 2 — Extract + wire the serving discipline (flag-gated, shadow-validated)
- **Goal:** lift the SRPT+conformal discipline out of the benchmark file into a
  reusable `policy` + shared `calibration` lib; expose a serving-runtime entry
  point that runs it in **shadow** (no actuation).
- **Files affected:** new `aurelius/optimization/serving_policy.py` +
  `aurelius/forecasting/conformal_calibration.py`; `srtf_serving_backtest.py`
  imports from them (benchmark unchanged in behavior);
  `traces/module_backtest.py` gains an opt-in shadow consumer.
- **Risks:** the discipline never ran outside a heapq sim — continuous-batching
  preemption, KV eviction, and real TTFT differ; conformal α may be trace-overfit
  (Azure α≈0.0002 vs BurstGPT α≈0.0006).
- **Rollback:** policy defaults `enabled=False`, `shadow_only=True`,
  `executable_in_real_cluster=False` (same guards as existing shadow modules).
- **Required benchmarks:** discipline must reproduce its sim numbers post-extraction
  (**0% delta** vs `abs_conformal_backtest_2026-06-22`); shadow run logs predicted-
  vs-realized ordering on a public trace.
- **Required validation:** calibrator unit tests carried over; leakage guard
  (order by *predicted*, physics by *actual*) preserved and tested.
- **Expected upside:** **[A]** up to +313%/+557% vs FIFO *in sim*; **[N]** runtime
  gain unknown — the point of this phase is to *measure* it safely, not claim it.

### Phase 3 — Promote frontier as a constraint + consolidate (low-risk)
- **Goal:** BASE/DYNAMIC frontier becomes a hard ρ-ceiling constraint consumed by
  the optimizer; dedup the 3 conformal calibrators into the Phase-2 lib.
- **Files affected:** `constraints/frontier_integration.py` (flip from optional
  recommender to a constraint provider, still flag-gated),
  `frontier/{controller,dynamic_controller}.py` (consumed, not changed),
  calibrator call sites.
- **Risks:** over-constraining ρ lowers goodput on traces where the frontier is
  conservative.
- **Rollback:** `frontier_integration.enabled` stays `False` by default; promotion
  is opt-in per benchmark.
- **Required benchmarks:** **[P]** SUF +13% must hold on Azure 2024; no regression
  on BurstGPT; safety gates (timeout ≤0.5× FIFO) stay closed.
- **Expected upside:** safer high-ρ operation; gives the frontier work a real
  consumer (today it has none).

### Phase 4 — Deprecate dead/duplicate code (after parity locked)
- **Goal:** remove EVAL_WORKLOAD + BATCH_INFERENCE frontier families, the unused
  second DCGM connector, and the dead `carbon_objective` config; mark
  vLLM/Triton/Ray/OTel connectors experimental.
- **Files affected:** `frontier/eval_workload_*`, `frontier/batch_inference_*`,
  their scripts/tests; `connectors/dcgm.py`; `models.py:389-393`.
- **Risks:** something undocumented imports them.
- **Rollback:** deletions land in their own revertable commit; preceded by a
  repo-wide import check proving zero non-test/non-own-script consumers.
- **Required benchmarks:** none affected (these have no benchmark consumer) —
  re-run the suite to confirm **0% delta**.
- **Expected upside:** ~2.8K LOC of dead frontier + duplicate connector removed;
  ends the "5 parallel families" maintenance tax.

---

## Migration risks (consolidated)
1. **Integration risk (highest):** the headline discipline has only ever run in
   an idealized heapq simulator; real serving runtimes will differ. Phase 2 is
   explicitly a *measurement* phase, not a claim.
2. **Pinned energy core:** `test_energy_core_preservation.py` + `ENERGY_SYSTEM_MAP §8`
   forbid silent drift. Any change touching `JobScheduler` needs an explicit phase.
3. **Benchmark-number movement:** unifying replay (Phase 1) could shift reported
   KPIs; mitigated by the **0%-delta parity gate** before any behavior change.
4. **Negative-evidence modules:** output-length / admission / GPU-placement
   already regressed or were neutral when integrated — they are *excluded* from
   integration by design.
5. **Conformal overfit:** α differs 3× across the two traces; needs out-of-sample
   validation before any runtime promotion.
6. **Naming collisions:** "constraint_aware" means three things; clarify before
   integrating to avoid leaderboard-provenance confusion.

## Benchmark risks (consolidated)
- Public replay logic (`traces/backtest.py`, `simulation/cluster/serving.py`,
  `economics.py`) and scenario hashes are **frozen**; the plan reproduces them,
  never edits them.
- No oracle as headline; FIFO is sanity-only; SLA stays a numerator filter.
- `RESULTS.md §8` production-claim gate is met by **no** result; all upside stays
  labeled directional-simulator until real-runtime evidence exists.

## Decision gate to even begin
Do not start Phase 1 until: (a) this audit PR is reviewed and the unification is
explicitly approved, and (b) a parity harness exists that can prove 0% KPI delta.
Absent both, the correct action is to leave the architecture as-is and keep the
serving discipline as the documented research-canonical, unwired.
