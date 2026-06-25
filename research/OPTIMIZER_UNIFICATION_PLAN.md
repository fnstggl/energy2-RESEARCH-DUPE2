# Aurelius Optimizer Unification Plan

> **Status:** PARTIALLY EXECUTED (updated 2026-06-25). Phases 1a / 2 / 3 / 3b / 5
> plus the **Phase B comprehensive-optimizer consolidation** are done — the
> canonical optimizer now holds all five surfaces and orchestrates them via
> `optimize_fleet()`. The replay-loop unification (Phase 1b) and the
> constraint/forecast promotions (Phase 4) remain. See **Execution Status** below.
> This is the phased, reversible migration path
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
| **Phase 2 — Extract the serving (abs-conformal SRPT) discipline behind the policy interface** | **DONE — parity extraction, 0% serving + energy KPI drift** | `aurelius/optimizer/policies/serving_queue.py`, `tests/test_canonical_serving_policy_phase2.py` (9 pass) + `test_abs_conformal_backtest.py` (17 re-export), `research/results/canonical_optimizer_phase2_serving_policy_parity_2026-06-22.md` |
| **Phase 3 — Route public benchmark entry points through AureliusOptimizer** | **DONE — 5 entry points routed, 0% energy + serving KPI drift** | `canonical_backtests` / `gpu_routing_backtest` / `srtf_backtest` / `srtf_contention_backtest` + serving shim; `tests/test_canonical_optimizer_phase3_routing.py` (11) + `test_canonical_energy_backtest.py` (17 golden); `research/results/canonical_optimizer_phase3_benchmark_routing_parity_2026-06-22.md` |
| **Phase 3b — Route AMCSG + SOTSS-MIN canonical backtest entry points through AureliusOptimizer** | **DONE — 0% KPI drift, initial_violations now propagated in sotss_min** | `_run_amcsg_backtest` + `_run_sotss_backtest` (both AMCSG baseline + oracle) routed through `_REPLICA_SCALING_OPTIMIZER.optimize()`; `ReplicaScalingPolicy.optimize(mode="sotss_min")` captures `init_viols` instead of discarding; 33 new parity tests; `research/results/amcsg_sotss_canonical_routing_parity_2026-06-24.md` |
| **Phase B — Comprehensive optimizer consolidation** (all 5 surfaces under one facade; `placement`←residency, `admission`←frontier gate; `optimize_fleet()`; `serving_orchestration`←ConstraintAwareEngine; route the 3 energy facade-bypass sites) | **DONE — 0% energy KPI drift; parity wirings, no new optimization logic** | `aurelius/optimizer/aurelius_optimizer.py`, `policies/__init__.py`; `tests/test_comprehensive_optimizer.py` (8) + `test_canonical_optimizer_parity.py` (updated) |
| Phase 1b — Unify the 4 replay loops into one engine | Not started | — |
| Phase 4 — Promote frontier BASE/DYNAMIC → constraint; dedup calibrators | Not started | — |
| **Phase 5 — Deprecate dead/duplicate code** | **DONE — 2,873 LOC removed, 0% KPI delta, 39 dead tests deleted** | Deleted `aurelius/frontier/eval_workload_{models,estimator,controller,safety}.py` + `batch_inference_{models,estimator,controller,safety}.py` (1,827 LOC); `tests/test_{eval_workload,batch_inference}_frontier.py` (692 LOC, 39 tests); `scripts/run_{eval,batch}_inference_frontier.py` (354 LOC). Zero non-test/non-script consumers confirmed by repo-wide import check. Lint/mypy pass; research docs updated. |

**Phase 3 notes.** Five public benchmark entry points now construct the canonical
`AureliusOptimizer` instead of the underlying engines: the four energy benchmarks
(`canonical_backtests`, `gpu_routing_backtest`, `srtf_backtest`,
`srtf_contention_backtest`) route through `AureliusOptimizer(policy="energy")`
(GPU placement passed through as a scheduler kwarg), and the serving benchmark's
abs-conformal shim routes through `AureliusOptimizer(policy="serving_queue")`.
Routing is construction-only: energy `routed == direct JobScheduler`, serving
`shim == extracted impl` → 0% KPI drift (energy golden snapshot reproduced;
serving JSON byte-identical). The Azure/BurstGPT trace replays are **not** routed
— they are a per-tick replica-provisioning autoscaler (Erlang-C), a different
decision type that maps to a future `ReplicaScalingPolicy`, not the energy/serving
policies; routing them now would be a behavior change, not a parity refactor.
`BacktestEngine` (a shared core engine) is left for a separate step.

**Phase 2 notes.** The strongest validated serving discipline (Decoupled Hybrid
SRPT + absolute-error conformal α, run -x) was moved **verbatim** from the 7.6k-LOC
benchmark monolith into `aurelius/optimizer/policies/serving_queue.py` and wrapped
as `ServingQueuePolicy` (reachable via `AureliusOptimizer(policy="serving_queue")`).
The benchmark imports the discipline + calibrator back and keeps a thin shim that
injects its own `_summarize` (evaluation stays in the benchmark; one-way
dependency, no circular import). Parity is exact: the serving and energy
benchmarks are byte-identical before/after, and the existing 17 abs-conformal
tests pass against the re-exported objects. No new optimizer, no new priors, no
benchmark-assumption change, no FIFO-only claim, no actual-token decision-time
leakage (all guarded by tests).

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

## Optimization Governance (binding for all future optimization phases)

These rules govern any phase that makes an **optimization claim** (Phases 1b–5
that change behavior, and any new policy). They do **not** apply to parity
refactors (Phases 1a/2/3), which by definition change no behavior.

1. **Baseline governance.** Every optimization claim must report a three-way
   comparison — **Current Main** vs **Best validated Aurelius** vs **Candidate** —
   plus a relevant industry-standard baseline. Never compare exclusively against
   FIFO, random, greedy, or other trivial baselines. A candidate is a *frontier
   improvement* only if it beats the best validated `AureliusOptimizer`
   configuration, not merely FIFO.

2. **Optimizer-first rule.** Do not build an optimization whose only purpose is a
   benchmark number. Before implementing any optimization, document: (a) the
   production decision that changes, (b) the information available at decision
   time, (c) why it should improve SLA-safe goodput/$, (d) how it operates on real
   infrastructure. If no realistic production decision exists, do not implement it.
   (Decision-time information only — no actual-output-token leakage, enforced by
   tests.)

3. **Policy combination search.** When ≥2 validated policies exist, evaluate
   combinations (A, B, …, A+B, A+C, B+C, A+B+C, …) when computationally feasible,
   and measure interaction effects — do not assume individually-beneficial
   policies compose positively. The benchmark frontier is the best-performing
   validated *combination* inside `AureliusOptimizer`.
   - **Feasibility note (today):** the two validated policies — `energy`
     (batch-job cost on price traces) and `serving_queue` (request-queue goodput/$
     on LLM traces) — operate on **disjoint workload classes**, so no shared
     benchmark defines `energy + serving_queue` yet. A meaningful combination
     search becomes feasible after **Phase 1b** (unified replay) and the
     `ReplicaScalingPolicy`/`PlacementPolicy`/`AdmissionPolicy` exist on a common
     workload. Until then there is no honest combination to measure, and none is
     fabricated.

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
