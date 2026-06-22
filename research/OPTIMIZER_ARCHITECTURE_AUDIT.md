# Aurelius Optimizer Architecture Audit

> **Status:** AUDIT ONLY. No code merged, no optimizer replaced, no benchmark or
> replay logic changed. This document is an architecture analysis produced to
> answer a single question: *is Aurelius a coherent optimization system, or a
> collection of partially-connected optimizers, schedulers, and benchmark
> policies that are not actually working together?*
>
> **Date:** 2026-06-22 · **Scope:** whole repository · **Method:** static
> import-graph tracing + decision-flow tracing + research-doc reconciliation.
> Every claim below carries a `file:line` or grep reference.

---

## 0. Executive Answer

**Aurelius is not currently a single coherent optimization system.** It is two
distinct eras of optimization work plus a large research/shadow periphery, and
the strongest, most recent results come from a path that is *by the authors'
own admission* not connected to any runtime.

Concretely, the repository contains **at least three independent decision
engines**, **four independent replay/backtest loops**, and **six engine
families spread across 42 `scripts/run_*.py` entry points** — with no shared
optimization core between the energy world and the serving world.

| # | Decision engine | Era | Objective | Wired to runtime? | Produces headline numbers? |
|---|---|---|---|---|---|
| 1 | `JobScheduler` (`optimization/scheduler.py:45`) | Era-1 energy | weighted **cost** (energy-dominant) | No (sim/backtest/shadow only) | Energy: +11–25% vs price-only |
| 2 | `ConstraintAwareEngine` (`constraints/engine.py`) | Era-1.5 | binding-constraint recommendations | No (recommendation-only) | — |
| 3 | `traces/backtest.py` inline policies | Era-1.5 serving | replica provisioning / tick | No (replay) | Public LLM leaderboard (Azure +25.75% CA) |
| 4 | `srtf_serving_backtest.py` disciplines (6,628 LOC) | **Era-2 serving** | **SLA-safe goodput/$** | **No — "integration pending"** | **All recent headlines: +313% / +557% vs FIFO** |
| 5 | `frontier/` controllers (5 families) | research | safe-utilization frontier | No (`executable_in_real_cluster=False`) | Frontier audits (analysis-only) |

The project's *stated* north-star (`ROADMAP.md` §1) is **"maximize SLA-safe
goodput per infrastructure dollar."** But the engine that optimizes that metric
best (engine 4) shares **zero code** with the engine that is productized,
CI-gated, and marked "source of truth — do not modify" (engine 1, the energy
`JobScheduler`). The two were never unified; engine 4 was built precisely
*because* engine 1 "cannot express the SRTF benefit at all"
(`srtf_serving_backtest.py:14-21`).

This is the fragmentation. The rest of this document inventories it, traces the
decision flow, and quantifies the duplication and disconnection.

---

## Phase 1 — Architecture Inventory

Required fields per component: Name · Location · Purpose · Inputs · Outputs ·
Production path? · Replay? · Benchmark-only? · Shadow-only? · Default-on? · Last
benchmarked · Measured impact · Status (Active/Shadow/Experimental/Dead/Duplicate).

### 1.1 Core optimizers / schedulers

| Component | Location | Purpose | Inputs → Outputs | Prod | Replay | Bench-only | Shadow | Default-on | Status |
|---|---|---|---|---|---|---|---|---|---|
| **`JobScheduler`** | `optimization/scheduler.py:45` | Energy batch optimizer: when (time-shift), where (region), how-fast (throttle/migrate) | jobs+price+carbon+config → `ScheduleDecision[]` | No | Yes | No | partial | **Yes** (it *is* the optimizer) | **Active (canonical-energy)** |
| `ObjectiveFunction` | `optimization/objective.py:76` | Scores a schedule's weighted monetary cost | schedule → cost breakdown | No | Yes | No | No | Yes | Active |
| `ConstraintBuilder` | `optimization/constraints.py:36` | Feasibility (deadline/region/power-cap) | job+decision → violations | No | Yes | No | No | Yes | Active |
| **`ConstraintAwareEngine`** | `constraints/engine.py` | Classify binding constraint → gated recommendations | `ClusterState` → `EngineResult` | No | Yes | No | No | on constraint path | **Active (constraint path)** |
| **serving inline policies** | `traces/backtest.py:305` (`_run_policy`) | Replica-provisioning autoscaler: fifo / sla_aware / **constraint_aware** / queue_aware / cache_affinity | arrival ticks → replicas/tick → KPI | No | Yes | No | No | Yes (this is the public LLM replay) | **Active (public-leaderboard)** |
| **serving-queue disciplines** | `benchmarks/srtf_serving_backtest.py` (`simulate_queue:1197`, `_simulate_srpt_preemptive:365`, `_simulate_decoupled_hybrid*`) | Discrete-event M/G/c SRPT/aging/decoupled/conformal disciplines | per-request (arrival, tokens) → response/wait KPIs | **No** | self-contained | **Yes** | tagged shadow | n/a (sim) | **Experimental (headline research, unwired)** |
| migration / MPC | `scheduler.py:730,801,882,1070` (`_apply_migration_optimization`, `replan_remainder`) | Multi-region migration & receding-horizon replan | schedule → migrated schedule | No | No | No | No | No | **Experimental (test-only)** |

### 1.2 Frontier controllers (`aurelius/frontier/`, 33 files, ~9.7K LOC)

A "Safe Utilization Frontier" recommender: over a ρ (utilization) grid, veto
points that breach safety gates, recommend the highest **SLA-safe goodput/$**
ρ. **All families are recommendation-only by construction**
(`frontier/models.py:217-221` raises if `executable_in_real_cluster=True`).

| Family | Files | Exported in `__init__.py`? | Only callers (outside family) | Default-on | Status |
|---|---|---|---|---|---|
| **BASE** | `controller/estimator/models/safety/shadow/risk/execution.py` | Yes | `constraints/frontier_integration.py:54` (**`enabled=False`**), 4 scripts, 4 tests | **No** | **Active (gated, disabled by default)** |
| **DYNAMIC** | `dynamic_*` ×10 | Yes | `scripts/run_azure_2024_dynamic_frontier.py` + 3 tests | No | **Shadow/Experimental** |
| **TRAINING** | `training_*` ×7 | Yes | 3 training scripts + 3 tests | No | **Experimental** |
| **EVAL_WORKLOAD** | `eval_workload_*` ×4 | **No** | 1 script + 2 tests | No | **Dead/Duplicate** (copy-paste of BASE) |
| **BATCH_INFERENCE** | `batch_inference_*` ×4 | **No** | 1 script + 2 tests | No | **Dead/Duplicate** (docstring: *"mirrored from the serving + training controllers"*) |
| `admission.py` | 459 LOC | Yes | `traces/module_backtest.py:40` + tests | **No (`enabled=False`)** | **Shadow (benchmark-only)** |
| `telemetry_provenance.py` | — | Yes | within DYNAMIC | n/a | Supporting |

**No benchmark runner imports any frontier family** (the `srtf_serving_backtest`
hits for "frontier" are the unrelated phrase *"Pareto frontier"*). The only core
reach is BASE via `frontier_integration`, which defaults `enabled=False`
(`frontier_integration.py:92`) and is passed `None` by `traces/backtest.py:310`.

### 1.3 Forecasting / calibration / ML (`forecasting/` 26 files, `ml/` 8, `learning/` 3)

**Architectural contract (honored in code):** `forecasting/__init__.py:52` —
*"Forecasting is ADVISORY DATA ONLY — does NOT influence optimizer behavior."*
**Zero forecasters feed a production decision.** The only forecast→decision
mechanism is `Job.predicted_output_tokens` (`models.py:121`), a static SRTF
tiebreaker in `scheduler.py:317-332` that is **set only by tests/benchmarks**,
never by a forecaster.

| Forecaster | Location | Predicts | Consumer | Status |
|---|---|---|---|---|
| PriceQuantileForecaster | `forecasting/price_model.py` | energy price p50/p90 | cli, backtest engine | Research/advisory |
| CarbonQuantileForecaster | `forecasting/carbon_model.py` | carbon p50/p90 | uncertainty/baseline | Research/advisory |
| RegimeDetector / SpreadRiskModel | `forecasting/regime.py`, `spread_risk.py` | price-regime / DA→RT spread | backtest engine (opt-in) | Research/advisory |
| CARA latency / queue / TTFT-tail | `forecasting/cara_*` | TTFT, E2E, queue-wait | tests + `scripts/run_cara_*` only | **Research-only** (docstring: *"Nothing here is wired into any controller or scheduler"*) |
| CARA output-length | `forecasting/cara_output_length_forecaster.py` | output tokens | `traces/module_backtest.py` (sim) | **Shadow** (never assigned to `predicted_output_tokens`) |
| Cache/prefix-reuse | `forecasting/cache_prefix_forecaster.py` | KV reuse % | tests + 1 script | Research-only |
| TTFT shadow / shadow-prior | `forecasting/ttft_shadow*.py` | TTFT p50 | gpu_placement_scorer (opt-in) | **Shadow** (`apply_to_scorer=False`) |
| **GpuPlacementScorer** | `forecasting/gpu_placement_scorer.py` | GPU placement penalty | `scheduler.py` (off by default) + `gpu_routing_backtest.py` | **Shadow** (`enabled=False`; never passed in prod) |
| ConstraintShadowScorer | `forecasting/constraint_shadow_scorer.py` | residency score refine | tests + 2 scripts | Shadow |
| economic_ml / overlay | `forecasting/economic_*.py` | derived cost targets | tests + scripts | Research-only (diagnostic) |
| 3× conformal calibrators | **inline in** `srtf_serving_backtest.py:879,959,5711` | prediction-error α | only inside that backtest | **Experimental + Duplicate** |
| ml/ trainers + learning/promotion | `ml/`, `learning/promotion.py` | bias corrections / forecaster gating | offline cron | Active-offline (governs advisory forecasters only) |

### 1.4 Supporting / runtime subsystems

| Subsystem (LOC) | Purpose | Used in | Default-on | Status |
|---|---|---|---|---|
| `shadow/` (1487) | record predicted-vs-realized savings | CLI subcmds + tests | No | Active (observational) |
| `sla/` (1986) | SLA hard-gate + soft-penalty selection | constraints/engine, scheduler `_eval_action_sla`, CLI | conditional | Active (decision gate, constraint path) |
| `constraints/` (5221) | binding-constraint classification → recs | cli_constraint, constraint_runner, traces/backtest | engine: yes; frontier shim: **No** | Active engine + Experimental shim |
| `residency/` (3051) | model placement / cold-start recommender | 7 scripts + 7 tests; own backtest/sim | No (`MUTATION_ALLOWED=False`) | **Shadow/Experimental — standalone** |
| `execution/` (5144) | AWS Batch / K8s / Slurm executors | backtest engine (dataclasses only); real exec never instantiated outside tests | No (`allow_real_execution=False`) | **Experimental/Scaffolding** |
| `connectors/` (3809) | telemetry → ClusterState | state/assemble (DCGM/K8s/Topo) | No (Fake default) | Active (3) + Scaffolding (4: vLLM/Triton/Ray/OTel) |
| `simulation/` (14931) | `replay.py` (synthetic, CLI/API) + `cluster/engine.py` (GPU sim, 3877 LOC) | replay→CLI/API; cluster→cli_constraint/benchmarks | per path | Active (**two** engines) |
| `safety/` (543) | quantile gate filtering risky decisions | shadow/runner only | within shadow runner | Active (scoped) |
| `monitoring/` (337) | forecast-drift detector | learning-loop cron + tests | No | Experimental |
| `roi/` (430) | ROI projection (sales) | CLI `roi` | No | Active (standalone tool) |
| `state/` (2213) | assemble connectors → `ClusterState` | connectors/constraints/frontier | yes (constraint path) | Active (core infra) |
| `backtesting/` (1839) | walk-forward price/carbon + shared baselines/evaluator | CLI `backtest`, learning loop, reused by canonical loop | on-demand | Active (canonical for price/carbon walk-forward) |
| `api/` (256) | FastAPI `/simulate` | deployed; wraps `simulation/replay` only | yes if served | Active but limited (synthetic only) |
| `traces/` (20 files, ~11K LOC) | trace ingest + serving replay | 50+ scripts + tests | n/a | Data/replay backbone for benchmarks |

---

## Phase 2 — Decision Flow Tracing

The audit prompt's canonical chain is
`Workload → Forecasts → Constraints → Objective → Optimizer → Decision → Replay → KPI`.
There is **no single such chain.** There are (at least) four, and they diverge
at the *Optimizer* node.

### Path A — Energy / batch (the productized, CI-gated core)
```
synthetic jobs + real CAISO/PJM/ERCOT prices+carbon
  → (advisory) Price/CarbonQuantileForecaster        [forecasting/*]
  → ConstraintBuilder feasibility                     [optimization/constraints.py]
  → ObjectiveFunction: min(α·energy+β·carbon+γ·risk+δ·SLA)  [objective.py:228]
  → JobScheduler.solve(method=greedy|local_search|milp|*_migrate)  [scheduler.py:217]
  → ScheduleDecision[]
  → BacktestEngine walk-forward / canonical loop      [backtesting/engine.py:284]
  → KPI: $ savings vs current_price_only, deadline misses
```
Entry points: `benchmarks/run_benchmark.py` (the "Standardized Benchmark
Runner"), `scripts/run_canonical_backtests.py`, CLI `aurelius backtest`.
**This is the only path that constructs `JobScheduler`.**

### Path B — Public LLM serving replay (the CA leaderboard)
```
real Azure-2024 / BurstGPT arrivals + per-request tokens
  → NO forecaster (realized values)
  → target_rho heuristic
  → inline policy {fifo, sla_aware, constraint_aware, queue_aware, cache_affinity}  [traces/backtest.py:305]
       (constraint_aware optionally calls frontier_integration.select_constraint_aware_rho — default OFF)
  → replicas-per-60s-tick
  → Erlang-C serving physics                          [simulation/cluster/serving.py]
  → KPI: SLA-safe goodput/$ via economics.py
```
Entry points: `scripts/run_burstgpt_backtest.py`, `run_azure_llm_2024_backtest.py`.
**Never constructs `JobScheduler`. `constraint_aware` here is a third, inline
implementation — not `ConstraintAwareEngine`, not `JobScheduler`.**

### Path C — Serving-queue disciplines (the Era-2 headline research)
```
real Azure-2024 / BurstGPT per-request (arrival, output_tokens)
  → INLINE lognormal-noise forecast + INLINE HGB prior + INLINE conformal calibrators  [srtf_serving_backtest.py:879,959,5248,5711]
  → INLINE SLA threshold
  → INLINE discipline: SRPT-preemptive / aging / decoupled-hybrid / conformal-α  [simulate_queue:1197]
  → per-request completion times (heapq discrete-event)
  → INLINE goodput/$ summarizer                       [_summarize:1362]
  → KPI: +313% / +557% vs FIFO
```
Entry points: `scripts/run_abs_conformal_backtest.py`,
`run_live_prior_compound_backtest.py`, `run_stratified_prior_backtest.py` (3
scripts only). **Imports nothing from `optimization/`, `frontier/`, or
`simulation/`'s replay engine. Self-contained 6,628-LOC monolith.**

### Path D — Constraint engine (cli_constraint + synthetic scenarios)
```
synthetic YAML scenarios / live ClusterState
  → connectors → state/assemble → ClusterState
  → ConstraintAwareEngine: classify binding constraint → SLA gate → cost gate  [constraints/engine.py]
  → recommendations (recommendation-only)
  → ClusterSimulator / constraint_runner loop          [simulation/cluster/engine.py]
  → KPI: SLA-safe goodput/$
```
Entry points: `aurelius/cli_constraint.py`, `benchmarks/constraint_runner.py`.

### Where the paths diverge
- **Optimizer node:** A=`JobScheduler`; B=inline tick policy; C=inline queue
  discipline; D=`ConstraintAwareEngine`. **Four different optimizers.**
- **Replay node:** A=`backtesting/engine` + `canonical_backtests` loop;
  B=`traces/backtest` tick loop; C=`srtf_serving_backtest` heapq loop;
  D=`simulation/cluster/engine`. **Four different replay loops.** Only the
  *physics primitives* in `simulation/cluster/serving.py` are partially shared
  (B, C, D import it; A does not).
- **Objective node:** A minimizes **cost**; B/C/D score **SLA-safe goodput/$**
  *after the fact*. The energy optimizer never optimizes the north-star metric.
- **Forecast node:** A uses advisory price/carbon forecasters; B uses none;
  C reimplements forecasting inline; D uses none. The 26-file `forecasting/`
  package feeds **only** path A, and only advisorily.

---

## Phase 3 — Architectural Fragmentation

### 3.1 Duplicate / near-duplicate implementations

| Duplication | Evidence | Maint. cost | Complexity cost | Bench impact | Integration difficulty |
|---|---|---|---|---|---|
| **Frontier EVAL_WORKLOAD + BATCH_INFERENCE families** | copy-paste of BASE; not exported; batch_inference docstring says *"mirrored from the serving + training controllers"* | ~1.4K LOC × 2 to maintain | High (5 parallel families) | **None** (no benchmark imports them) | Trivial to remove (only own scripts/tests) |
| **3 conformal calibrators inline** | `srtf_serving_backtest.py:879` (rel), `:959` (abs), `:5711` (per-class) + `forecasting/cara_latency_calibration.py` SplitConformalUpperBound — 4 unrelated impls | Medium | Medium | Headline-bearing but trapped in one file | Medium (extract to a shared calibration lib) |
| **2 DCGM connectors** | `connectors/dcgm.py` (newer, referenced but unused) vs `ingestion/dcgm_provider.py` (older, actually used by `backtesting/engine.py`) | Low | Medium | None | Low |
| **4 replay/backtest loops** | `simulation/replay`, `backtesting/engine`, `simulation/cluster/engine`, `srtf_serving_backtest` | **High** | **High** | High (two sources of truth for "how serving behaves") | **Hard** (the unification target) |
| **"constraint_aware" overloaded** | means `JobScheduler` (energy docs), `ConstraintAwareEngine` (cli_constraint), and an inline tick policy (`traces/backtest.py`) — three different things | Conceptual | High (naming) | Confusing leaderboard provenance | Low (rename/clarify) |
| **4 connector scaffolds** | vLLM/Triton/Ray/OTel — test-only, not exported, never in `state/assemble` | Low | Low | None | Low (mark experimental) |

### 3.2 Disconnected from runtime decisions
Everything advanced is shadow/opt-in. **No default path can mutate real
infrastructure.** Enforced in code:
- `frontier_integration.py:92` `enabled=False`; `:133-136` requires
  `shadow_only=False` AND `allow_real_execution=True` (both default False).
- `frontier/admission.py:98` `enabled=False`.
- `residency/shadow.py:39` `MUTATION_ALLOWED=False`; `:74-76` raises on execution.
- `forecasting/gpu_placement_scorer.py` `enabled=False`; `scheduler.py:478`
  fails open to penalty `0.0`; never passed a scorer in prod instantiations
  (`backtesting/engine.py:284`, `energy_adapter.py:363`, `shadow/runner.py:122`,
  `simulation/compare.py:61` all pass `config` only).
- `frontier/models.py:217-221` raises if `executable_in_real_cluster=True`.

**Disconnected modules:** the entire Era-2 serving research
(`srtf_serving_backtest`), all 5 frontier families, `residency/`, the whole
`forecasting/` stack (advisory), the 3 shadow modules, migration/MPC in
`JobScheduler`.

### 3.3 Benchmark-only modules
`srtf_serving_backtest` disciplines, `frontier/admission` (only
`module_backtest`), `gpu_routing_backtest` scorer, the 3 conformal calibrators,
`srtf_backtest`/`srtf_contention` SRTF priors.

### 3.4 Modules that affect only one benchmark / cannot influence SLA-safe goodput/$ at runtime
- The conformal-α discipline (best goodput/$ result in the repo) **cannot
  influence runtime goodput/$** — it exists only inside a heapq simulator with
  no serving-runtime caller (`grep -rln srtf_serving_backtest aurelius/ | grep -v benchmarks/` → empty).
- `GpuPlacementScorer` moved its proxy metric but **regressed real KPI**
  (−7.3% latency_critical goodput/$, `BENCHMARK_REGISTRY.md §5b`).
- `OutputLengthForecastBundle` **HURT** (−7…−11%); `WorkloadAdmissionGate`
  **NEUTRAL**. All three remain `enabled=False` ("INFRASTRUCTURE ONLY").

### 3.5 Dead config surface
`OptimizationConfig.carbon_objective` / `carbon_threshold_gco2_per_kwh` /
`"cost_with_carbon_weight"` / `"carbon_constrained"` (`models.py:389-393`) are
**never read** anywhere in `optimization/` — carbon is always a fixed β-weighted
soft term. Latent doc-vs-code discrepancy.

---

## Phase 4 — Research Review (does each belong in the canonical optimizer?)

| Research area | Where it lives | Benchmark evidence | Verdict |
|---|---|---|---|
| **SRTF / SRPT / aging / decoupled-hybrid** | `srtf_serving_backtest.py` (sim) | +274% → +322% vs FIFO; overhead-robust to 0.30s/event; cross-validated Azure+BurstGPT | **INTEGRATE** into the canonical serving optimizer — this *is* the goodput/$ lever, but must leave the sim and gain a real-runtime implementation |
| **Conformal calibration (rel/abs/per-class/adaptive-α)** | inline in same file | abs-conformal +19.95%/+26.17% over rel-conformal; 97.8%/88.3% oracle retention | **INTEGRATE as a shared calibration layer** feeding the SRPT discipline; deduplicate the 3 inline impls first |
| **Output-length forecasting** | `forecasting/cara_output_length_forecaster.py` | module-integration: **HURT −7…−11%**; running-median is near-oracle for *ordering* (Azure r=−0.022) | **RESEARCH-ONLY** until a predictor beats the running-median ceiling; do not wire |
| **Admission control** | `frontier/admission.py` | module-integration: **NEUTRAL ±0.34%** | **RESEARCH-ONLY / keep as optional shadow gate**; not a goodput/$ lever on tested traces |
| **GPU placement / routing** | `forecasting/gpu_placement_scorer.py` + `scheduler.py` hook | **regressed −7.3%** real KPI; proxy moved | **RESEARCH-ONLY**; keep the scheduler hook (off) but do not promote |
| **Conformal-confidence (frontier)** | `frontier/dynamic_confidence.py` | mis-named "calibration"; not conformal | **DEPRECATE name / keep as DYNAMIC internal** |
| **Energy arbitrage** | `JobScheduler` | +11% canonical at 0 deadline misses; 25% mean pilot | **KEEP as a sub-objective / a workload class** of the canonical optimizer |
| **Frontier safe-utilization** | `frontier/` BASE/DYNAMIC | static SUF +13% over CA (Azure); 73–91% oracle-alpha capture | **INTEGRATE BASE/DYNAMIC as a constraint layer**; **DEPRECATE EVAL_WORKLOAD + BATCH_INFERENCE** (dead duplicates) |
| **Residency / cold-start** | `residency/` | Alibaba GenAI +89% model-affinity (analysis); diagnostic only | **Keep RESEARCH/experimental**; candidate for a later placement sub-module, not now |

---

## Answers to the 10 audit questions

1. **Canonical optimizer today?** Split-identity. `JobScheduler` (energy cost)
   is the *productized/CI-gated* canonical core; the Decoupled-Hybrid SRPT +
   conformal-α serving discipline (in `srtf_serving_backtest.py`) is the
   *research* canonical that produces every headline number — **but is
   simulator-only and unwired.** There is no single coherent canonical optimizer.
2. **Modules that influence decisions?** `JobScheduler` (energy path),
   `ConstraintAwareEngine` (constraint path), `traces/backtest.py` inline
   policies (public LLM leaderboard), `srtf_serving_backtest` disciplines
   (research KPI), the SLA gate (when a registry is passed), `state/`+`connectors/`
   (constraint-path telemetry).
3. **Shadow-only?** GpuPlacementScorer, WorkloadAdmissionGate,
   OutputLengthForecastBundle, all frontier families, `residency/`, the entire
   `forecasting/` stack, `frontier_integration` (disabled), safety gate (scoped).
4. **Benchmark-only?** `srtf_serving_backtest` disciplines + 3 conformal
   calibrators, `frontier/admission`, `gpu_routing` scorer,
   `srtf_backtest`/`srtf_contention` priors.
5. **Duplicate?** Frontier EVAL_WORKLOAD/BATCH_INFERENCE; 3 conformal impls; 2
   DCGM connectors; 4 replay loops; overloaded "constraint_aware"; 4 connector
   scaffolds.
6. **Disconnected from runtime?** The Era-2 headline serving research, all
   frontier, residency, forecasting, the 3 shadow modules, migration/MPC.
   Nothing mutates real infra on any default path.
7. **Long-term architecture?** A single canonical optimizer with a clean
   forecast→constraint→objective→decision→replay→eval stack and the SRPT+
   conformal discipline promoted into a real serving-runtime path. See
   `research/CANONICAL_AURELIUS_OPTIMIZER.md`.
8. **Safest migration path?** Phased, flag-gated, shadow-validated,
   benchmark-gated; unify replay first (no behavior change), then wire the
   discipline behind a flag. See `research/OPTIMIZER_UNIFICATION_PLAN.md`.
9. **Benchmark evidence?** Strong *directional simulator* evidence for
   SRPT+conformal; NEGATIVE/NEUTRAL for the 3 shadow modules; modest for energy.
   No result meets the `RESULTS.md §8` production-claim gate.
10. **Risks?** Integration risk (the discipline never ran outside a heapq sim);
    the energy core is pinned "do not modify"; unifying replay could move
    reported numbers; the 3 shadow modules already regressed when integrated;
    conformal α may be trace-overfit; hard rule: public replay logic must not
    change. See the Risks section of the unification plan.

---

## Evidence appendix (load-bearing references)
- `aurelius/optimization/scheduler.py:45,114-115,217,317-332,465-478,730,1070`
- `aurelius/optimization/objective.py:76,228-236`; `models.py:121,377,389-393`
- `aurelius/benchmarks/srtf_serving_backtest.py:13-21,365,879,959,1197,1362,5711`
- `aurelius/benchmarks/canonical_backtests.py:476-478`;
  `srtf_contention_backtest.py:352-376`; `gpu_routing_backtest.py:387,415`
- `aurelius/traces/backtest.py:305,310,328-346,504`
- `aurelius/constraints/engine.py`; `constraints/frontier_integration.py:47-59,92,133-136`
- `aurelius/frontier/__init__.py:32-203`; `models.py:217-221`; `admission.py:98`;
  `batch_inference_controller.py:1-22`; `eval_workload_controller.py:1-9`
- `aurelius/forecasting/__init__.py:52`; `gpu_placement_scorer.py`; `cara_*`
- `aurelius/residency/shadow.py:39,74-76`; `simulation/cluster/engine.py`,
  `simulation/replay.py`, `simulation/cluster/serving.py`
- `tests/test_energy_core_preservation.py` (pins the energy core output)
- `research/BENCHMARK_REGISTRY.md §5b`; `research/ROADMAP.md §1-3,7`;
  `docs/ENERGY_SYSTEM_MAP.md §8`; `research/GAP_ANALYSIS.md` (run -x)
