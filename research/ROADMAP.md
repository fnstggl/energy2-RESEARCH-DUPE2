# Aurelius Research Roadmap

> **Persistent research memory.** Every run reads this file first and
> updates it at the end. This document is Aurelius's long-term research
> brain — tracking what has been tried, what works, what failed, and
> where the highest expected-value experiments are.
>
> **Binding rules:** No claim may be added here that is not backed by a
> committed result artifact. No production-savings claim may appear — only
> simulator / public-trace directional numbers. `docs/RESULTS.md` §8
> production-claim gate applies.

---

## 1. North Star Objective

**Primary:** Maximize SLA-safe goodput per infrastructure dollar on
public benchmarks.

**Long-term aspiration:** +300% SLA-safe goodput/$ versus SLA-aware
schedulers on the canonical public-trace rollup.

**Current headline:** Median **+9%** (mean +19%, weighted +26%) across 8
public-trace and frozen-synthetic benchmarks, 6 wins, 2 safe ties, 0
unsafe regressions. LLM-serving subset median **+23%**.

**Request-level SRTF [run 2026-06-20-g]:** On the real Azure LLM 2024 serving
queue (discrete-event M/G/c, ρ=0.85), shortest-predicted-job-first cuts
short-request p90 latency by **−99.6%** and mean response by **−62%** vs FIFO,
for **+323% SLA-safe goodput/$** — at the documented cost of a long-request
p99 regression. Robust to a 30%-CV forecast prior. Directional simulator
result; baseline is FIFO, not SLA-aware. See `docs/SRTF_SERVING_BACKTEST_RESULTS.md`.

**SRTF+aging anti-starvation — failure analysis [run 2026-06-20-i]:** Non-preemptive
aging (`effective_key=max(0,pred−alpha×wait)`, alpha=1.844 tok/s) degrades to FIFO at
ρ=0.85 due to the "age-out wave" problem.  All starved long requests simultaneously
age to effective_key=0 at t≈pred/alpha seconds, flooding high-priority slots and
blocking short requests.  Short requests then back up and also age out → FIFO.
Partial result: **+33.3% goodput/$** vs FIFO (transient SRTF phase before wave);
starvation bounded (overall p99 = 732.7s vs SRTF's 2188.7s).  Short-p90 improvement:
~0% (FIFO-like in steady state).  **Path forward: preemptive SRPT** (PecSched
arXiv:2409.15104, Equinox arXiv:2508.16646).  See `docs/SRTF_AGING_BACKTEST_RESULTS.md`.

---

## 2. Current Best Results (Benchmark Leaderboard)

See `docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md` for full table.

| trace | workload class | CA goodput/$ | strongest safe baseline | margin | safety |
|---|---|---:|---|---:|---|
| BurstGPT | llm_serving | 1,615,694 | cache_affinity_baseline | **+1.77%** | SAFE |
| Azure LLM 2023 conv | llm_serving | 2,326,157 | sla_aware | **+19.86%** | SAFE |
| Azure LLM 2024 week | llm_serving | 2,555,325 | sla_aware | **+25.75%** | SAFE |
| Alibaba GenAI 2026 | llm_serving | 9.84 | sla_aware | **+89.46%** | SAFE |
| Alibaba GPU v2023 | gpu_packing | — | best_fit | **tie** | SAFE |
| MIT Supercloud bounded | training | — | best_fit | **+16%** | SAFE |
| Philly training | training | — | best_fit | **tie** | SAFE |
| Canonical energy | energy_flex | — | current_price_only | **+11%** | SAFE |

Dynamic Frontier Estimator: **73.2%** oracle-alpha capture on Azure 2024.
Calibration aspirational target (95%) **NOT** reached (final 91.07%).

---

## 3. Current Architecture Summary

### Forecasting stack
- **Energy price:** `aurelius/forecasting/price_model.py` — seasonal naive
  (v5) + optional LightGBM quantile (disabled when LightGBM absent).
- **Carbon intensity:** `aurelius/forecasting/carbon_model.py` — regional
  lookup + proxy from energy price.
- **TTFT / E2E latency:** `aurelius/forecasting/cara_latency_forecaster.py`
  — HGB quantile regression on CARA telemetry (shadow-only; `build_now`
  status per `docs/FORECAST_LEVERAGE_AUDIT.md`).
- **Queue wait:** `aurelius/forecasting/cara_queue_forecaster.py` — HGB
  on derived CARA queue proxy (shadow-only).
- **Output token length [run 2026-06-20-b]:**
  `aurelius/forecasting/cara_output_length_forecaster.py` — calibrated
  output-token-count predictor. Two components: (a) `BiasCalibrationForecaster`
  debiases `num_predicted_output_tokens` via Huber regression; (b)
  `HGBOutputLengthForecaster` predicts `actual_output_tokens` at p50/p90/p95
  from all predict-time CARA features. Shadow-only. 39 tests passing.
  Enables semi-clairvoyant scheduling (arXiv:2604.06970).
- **Heterogeneous GPU placement scorer [run 2026-06-20-c]:**
  `aurelius/forecasting/gpu_placement_scorer.py` — wraps `TTFTShadowPrior`
  to produce per-(gpu_type, model_size, prompt_token_bin) TTFT p50 ranking
  and normalized latency-penalty scores for the scheduler. Peer-normalized:
  fastest GPU type in the candidate set gets `penalty_floor`; slowest gets
  `penalty_ceil`. `enabled=False` default; SLA-class gated (only
  `latency_critical` receives non-zero penalty). 37 tests passing. Research
  basis: arXiv:2604.07472 (Fast Heterogeneous Serving) + arXiv:2604.16682
  (KAIROS). Enables exploitation of the 9× TTFT spread in CARA GPU data.
  **NOW WIRED into scheduler [run 2026-06-20-d].**
- **Cache / prefix reuse:** `aurelius/forecasting/cache_prefix_forecaster.py`
  — HGB on SwissAI bucket-reuse + CC-traces (shadow-ready for integration
  review; single-dataset caveat applies).
- **Economic ML alpha:** `aurelius/ml/economic_ml_forecaster.py` — modular
  HGB per target; `cache_reuse_pct` and `peak_vram_gb` shadow-ready
  (single-dataset caveat).

### Optimization stack
- **Core scheduler:** `aurelius/optimization/scheduler.py` — greedy +
  local search + optional MILP (PuLP).
- **SLA-aware correction:** folded into scheduler via `sla_registry`.
- **GPU placement penalty [NEW - run 2026-06-20-d]:** `GpuPlacementScorer`
  now wired into `_find_best_slot` + `_sla_adjusted_score`. When enabled,
  TTFT-based latency penalties are computed per-region using the fitted
  `TTFTShadowPrior` and folded into the candidate ranking for
  `latency_critical` SLA jobs. Controlled via `gpu_placement_scorer` +
  `region_gpu_types` kwargs on `JobScheduler.__init__`. Shadow-only:
  `enabled=False` default; fail-open on missing/insufficient data.
- **SRTF sort key [NEW - run 2026-06-20-f]:** `predicted_output_tokens:
  Optional[float] = None` added to `Job` dataclass; `_SLA_CLASS_RANK` +
  length-prior tiebreak wired into `_solve_greedy`. Sort order:
  `(−priority, sla_class_rank, length_prior_or_inf, deadline)`.  Jobs without
  priors get `float("inf")` — fully backward-compatible (degrades to original
  `(−priority, deadline)` for homogeneous-SLA-class jobs).  37 new tests in
  `tests/test_srtf_scheduling.py`. Research basis: arXiv:2604.06970,
  arXiv:2410.01035, arXiv:2604.07931. Benchmark module: `srtf_backtest.py`.
- **Energy/carbon shifting:** primary economic lever; already sufficient.
- **Constraint scoring:** `aurelius/constraints/` — multi-constraint
  scorer with region, thermal, topology, migration veto.

### Frontier stack
- **Static Safe Utilization Frontier Controller:**
  `aurelius/frontier/controller.py` — rho grid sweep + veto gates.
- **Dynamic Frontier Estimator v1:**
  `aurelius/frontier/dynamic_estimator.py` — telemetry-driven rho
  recommendation; 73.2% oracle-alpha capture.
- **Dynamic Controller:** `aurelius/frontier/dynamic_controller.py` —
  RAISE / KEEP / LOWER decision with deadband + hysteresis.
- **Risk estimation:** `aurelius/frontier/risk.py` — deterministic SLA,
  queue-blowup, churn risk in [0,1].
- **Admission Gate v1 [NEW - run 2026-06-20]:**
  `aurelius/frontier/admission.py` — flow-rate admission control (ADMIT /
  DEFER / REJECT) based on KV-cache pressure + queue tail trend. Shadow-
  only. Research basis: arXiv:2604.11001.

### Constraint stack
- `aurelius/constraints/` — data residency, power, carbon budget, thermal,
  topology, migration cost, reliability risk, GPU availability.
- `aurelius/frontier/safety.py` — SLA / queue / latency / thermal /
  topology / memory / churn veto gates (hard safety — not weights).

---

## 4. Research Areas

### 4.1 Forecasting

#### TTFT / E2E Latency Forecasting
- **Status:** Implemented (shadow-only, CARA data)
- **Expected upside:** High (9× p99 spread across GPU types)
- **Current state:** TTFT p50 shadow_ready (2 of 3 holdouts); TTFT p95
  diagnostic_only (time-holdout subgroup undercoverage); TTFT p99
  baseline_fallback (67% fallback rate on time holdout).
- **Implemented:** HGB quantile regression on CARA train_flat (76,825
  rows). Queue-feature augmentation experiment — negative result (no
  improvement; CARA queue wait is too small to be a driver).
- **Failed:** Queue-feature augmentation for TTFT tail improvement.
- **Open questions:** Can cross-workload generalization improve tail
  coverage? Does TPOT (per-token latency) add complementary signal?
- **Bottleneck:** Needs measured queue-wait labels or pilot telemetry
  to fix time-holdout subgroup undercoverage.

#### Queue Wait Forecasting
- **Status:** Implemented (shadow-only)
- **Expected upside:** High
- **Current state:** shadow_ready for some cells; limited by CARA's
  near-zero actual queue wait (research cluster, not hot serving).
- **Bottleneck:** No real measured queue-wait signal. CARA's
  `derived_queue_wait_s` is a proxy.

#### Cache / Prefix Reuse Forecasting
- **Status:** Implemented (shadow_ready_for_integration_review)
- **Expected upside:** Medium (single-dataset SwissAI caveat)
- **Implemented:** HGB on SwissAI bucket-reuse (60k rows) + CC-traces
  3000 MiB expansion. `cache_reuse_pct` HGB +29.8% MAE vs baseline.
- **Caveat:** Single-dataset (SwissAI). Cross-dataset generalization not
  yet evidenced.
- **Next:** Integrate Mooncake FAST25 traces (Apache-2.0; KV-block-hash
  prefix reuse from real Kimi production).

#### Energy Price Forecasting
- **Status:** Already sufficient (seasonal naive + LightGBM optional)
- **Expected upside:** Low (oracle gap 34 pp on CAISO/PJM backtest,
  but seasonal naive + 24h horizon already captures the main pattern)

#### Economic ML Alpha (cache_reuse_pct, peak_vram_gb)
- **Status:** shadow_ready_for_integration_review (two targets)
- **Expected upside:** Medium for these two targets; blocked by
  missing labels for cold_start and migration costs.

### 4.2 Admission Control

#### Workload Admission Gate v1 [NEW - run 2026-06-20]
- **Status:** Implemented (shadow-only, enabled=False default)
- **Expected upside:** Medium — prevents KV-cache overflow at high load,
  reduces tail latency spikes, improves queue stability.
- **Research basis:** arXiv:2604.11001 "Flow-Controlled Scheduling for
  LLM Inference with Provable Stability Guarantees" (April 2026).
- **Implementation:** `aurelius/frontier/admission.py`
  `AdmissionGateConfig`, `AdmissionDecision`, `evaluate_admission()`,
  `evaluate_admission_batch()`.
- **Tests:** 38 passing tests in `tests/test_frontier_admission.py`.
- **Decisions covered:** ADMIT / DEFER / REJECT based on KV-cache
  utilization trend + queue p99 trend + timeout rate.
- **Safety invariants:**
  - Latency-critical SLA classes always ADMIT.
  - Missing telemetry → fail-open ADMIT.
  - REJECT only for `best_effort` / `background*` at KV saturation ≥0.99.
  - `enabled=False` default prevents accidental production use.
- **Open questions:** What defer window achieves the best SLA-safe
  goodput trade-off under the Azure LLM 2024 trace? Requires
  simulation integration to quantify.
- **Next steps:**
  1. Wire into `DynamicFrontierEstimate` / cluster simulator for backtest.
  2. Evaluate on Azure LLM 2024 week trace with load-spiked replay.
  3. Tune `kv_soft_ceiling` and `max_defer_ms` from CARA data.

### 4.3 Heterogeneous GPU Placement Scoring

#### GPU Type Routing via TTFT-Prior Penalty
- **Status:** Implemented + wired into scheduler (shadow-only, enabled=False default)
- **Expected upside:** High — CARA shows 9× TTFT p99 spread across GPU types;
  routing latency_critical requests to the appropriate GPU type could reduce
  TTFT violations by 30-50% directionally.
- **Research basis:** arXiv:2604.07472 (Fast Heterogeneous Serving, near-optimal
  allocation in <1s), arXiv:2604.16682 (KAIROS, heterogeneous scheduling with
  TTFT-aware SLO routing).
- **Implementation:** `aurelius/forecasting/gpu_placement_scorer.py` —
  `GpuPlacementConfig`, `GpuPlacementScore`, `GpuPlacementScorer`.
- **Tests:** 37 passing in `tests/test_gpu_placement_scorer.py`.
- **Safety invariants:**
  - `enabled=False` default prevents accidental production use.
  - Fail-open: missing/insufficient prior → penalty = 0.0 (neutral).
  - Only `latency_critical` SLA class receives non-zero penalty.
  - No controller / scheduler / executor imports.
- **Open questions:** What penalty_floor/ceil achieves optimal goodput/$ vs
  SLA-safe tradeoff on BurstGPT and Azure 2024? Requires scheduler integration.
- **Wired [run 2026-06-20-d]:**
  - `GpuPlacementScorer.latency_penalty` now folds into `_sla_adjusted_score`
    via `gpu_penalty` parameter for `latency_critical` placements.
  - `TTFTShadowPrior.predict()` extended: when `model_size=None`, falls through
    to `by_gpu` lookup (GPU-only prior without model-size context).
  - `TTFTShadowPrior.by_gpu_counts` added for `subgroup_n()` fallback.
  - `JobScheduler` accepts `gpu_placement_scorer` + `region_gpu_types` kwargs.
  - 28 new tests passing; 105 total passing.
- **Benchmarked [run 2026-06-20-e]:**
  - `aurelius/benchmarks/gpu_routing_backtest.py` provides the end-to-end
    GPU routing benchmark harness with CARA-calibrated synthetic prior.
  - `CANONICAL_REGION_GPU_TYPES` assigns H100/A100/T4 to the three canonical
    regions (us-east/us-west/us-south).
  - Confirmed: GPU routing routes substantially more `latency_critical` jobs
    to H100 vs baseline on flat-price synthetic data.
  - 34 new tests; zero regressions.
- **Real canonical benchmark [run 2026-06-20-f]:**
  - Baseline goodput/$: **0.300667** — GPU routing goodput/$: **0.300246**
  - Delta: **−0.000422 (−0.14%)** ← negative finding
  - Routing quality: +54.7 pp H100 placement for LC jobs, ~65% TTFT penalty
    reduction. Routing direction confirmed correct.
  - Root cause of negative goodput/$ delta: H100 GPUs are in PJM (us-east),
    the highest-cost energy region in the canonical trace. TTFT improvement does
    not count toward goodput/$ when all jobs already meet their long (26-day)
    deadlines; energy cost does count. On this trace, routing to PJM costs more
    than it gains in TTFT credit.
  - Finding: GPU routing improves TTFT quality but is energy-price-dominated
    when goodput/$ is the KPI. For measurable uplift, need a trace where TTFT
    violations are the binding SLA constraint (LLM serving, not energy batch).
- **Next steps:**
  1. Evaluate on BurstGPT / Azure LLM 2024 where latency violations are common
     and TTFT improvement directly reduces SLA miss rate.
  2. Tune `penalty_floor`/`penalty_ceil` from CARA TTFT distribution data.
  3. Consider energy-price-aware penalty attenuation (reduce GPU routing penalty
     when the preferred GPU region has materially higher energy cost).

### 4.4 Semi-Clairvoyant Scheduling

#### Output Length Prediction for Token-Magnitude Priors
- **Status:** Sort key wired into scheduler [run 2026-06-20-f]
- **Expected upside:** High — "Scheduling the Unschedulable" (arXiv:
  2604.06970) shows token magnitude priors increase P90 short-request
  performance by 32% vs FIFO, and removing magnitude increases p95 by
  5.8×. LAPS-SD (arXiv:2505.17074, IJCAI 2025) extends to speculative
  decoding — predicted length + token acceptance rate together determine
  optimal service ordering. TRAIL (arXiv:2410.01035) achieves near-SRTF
  via embedding-based SPRPT without clairvoyant access.
- **Complexity:** Low — sort key is wired; remaining work is LLM trace replay.
- **Risks:** Without good token prediction the scheduling gains erode.
- **Datasets available:** CARA carries `num_predicted_output_tokens` vs
  `actual_output_tokens` — ready for backtest.
- **Wired [run 2026-06-20-f]:**
  - `predicted_output_tokens: Optional[float] = None` added to `Job` dataclass.
  - `_SLA_CLASS_RANK` + length-prior sort key wired into `_solve_greedy`.
  - Canonical energy backtest: **0.0% delta** (expected — energy scheduling
    has no queue contention; SRTF gain applies when requests compete for
    limited GPU capacity at the same time, as in LLM serving traces).
  - `aurelius/benchmarks/srtf_backtest.py` A/B benchmark module added.
  - 37 new tests; 0 regressions.
- **Evaluated on real Azure LLM 2024 serving queue [run 2026-06-20-g]:**
  - **Negative finding (batch scheduler):** `srtf_contention_backtest.py` shows
    the merged greedy `JobScheduler` sort key is inert even at 4.6× capacity
    contention (Δ ≤ 0.05%). Root cause: the greedy batch path has NO queue-wait
    semantics — it falls back to `earliest_start` rather than making a job wait,
    so order never changes a completion time. The Erlang-C serving model is also
    aggregate (no per-request ordering). The merged sort key lives in the wrong
    layer for serving workloads.
  - **Large positive finding (request-level queue):** `srtf_serving_backtest.py`
    — a discrete-event non-preemptive M/G/c simulator over the REAL Azure LLM
    2024 trace (5,880 requests, real heavy-tailed output lengths) at ρ=0.85,
    c=4. SRTF (shortest-predicted-first) vs FIFO:
    - short-request **p90 latency: −99.6%** (696s → 3.0s)
    - **mean response: −62.2%** (344s → 130s)
    - **SLA-safe goodput/$: +323%** (discipline-invariant denominator)
    - 30%-CV forecast prior: short-p90 still −99.5% (robust to forecast error;
      no actual-length leakage — ordering uses predicted, physics uses actual)
    - **Honest cost:** long-request p99 REGRESSES (733s → 2189s) — non-preemptive
      SJF starves long jobs; mitigation = aging / SPRT preemption / hybrid bands.
    - See `docs/SRTF_SERVING_BACKTEST_RESULTS.md`. 38 new tests; 0 regressions.
- **Next steps:**
  1. Expose an SRTF/SPRPT ordering option in the SERVING path (not the batch
     scheduler) driven by `OutputLengthForecastBundle.p50`, with an aging /
     preemption guard to bound long-request starvation; re-run this backtest
     end-to-end as the value-realization step.
  2. Add an aging term so long requests cannot be starved beyond a TTL.

### 4.5 Probabilistic Demand Modeling

#### Hermes-style PDGraph for Multi-Step LLM Applications
- **Status:** Not Started
- **Expected upside:** High for agentic / tool-use workloads (Hermes
  shows >70% reduction in completion time, >80% p95 reduction).
- **Research basis:** "Efficient Serving of LLM Applications with
  Probabilistic Demand Modeling" (arXiv:2506.14851, April 2026).
- **Complexity:** High — needs DAG of LLM calls + tool calls per
  application.
- **Datasets available:** CC-traces weka (agentic; 136k rows), LMCache
  agentic traces (4,976 rows), AgentPerfBench.
- **Next steps:** Audit CC-traces for multi-step application structure.

### 4.6 Carbon-Aware Joint Optimization

#### Carbon-Aware Compute-Power MILP (Prosumer Datacenter)
- **Status:** Not Started
- **Expected upside:** Medium — carbon-power MILP (arXiv:2605.03751)
  shows substantial improvement vs compute-only or energy-only baselines,
  with inference routing flexibility as the major value driver.
- **Complexity:** High — needs battery/generation dispatch model.
- **Risks:** Realistic microgrid parameters hard to obtain for simulator.
- **Note:** Energy shifting already implemented and already sufficient per
  `FORECAST_LEVERAGE_AUDIT.md`. This is the next frontier.

### 4.7 Data Expansion

#### Mooncake FAST25 Traces
- **Status:** Not Started
- **Expected upside:** Low-Medium (additive to KV/prefix-reuse)
- **Source:** https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
- **License:** Apache-2.0
- **Signals:** KV-block-hash list, input/output token counts, timestamps.
- **Verdict from scavenger audit:** Bounded ingest feasible. Adds
  real Kimi production prefix-reuse signal for cross-dataset validation
  of cache_prefix_forecaster.

---

## 5. Bottlenecks

1. **Pilot telemetry (Tier 1):** Every forecaster beyond seasonal_naive
   and the deterministic risk estimator needs measured labels — queue
   wait, SLA outcomes, cold-start latency, migration cost. All of these
   are `blocked_by_missing_labels` without pilot integration.

2. **TTFT p99 tail calibration:** Currently at `baseline_fallback` (67%
   fallback rate on time holdout). Queue features don't help (negative
   result, PR #128). Needs pilot telemetry with measured queue-wait.

3. **Dynamic Frontier Estimator oracle-alpha gap:** 91.07% vs 95%
   aspirational target. The remaining 8.93% likely requires pilot
   calibration data — the simulator's synthetic window is too smooth.

4. **Output length prediction (calibration gate):** Infrastructure is now
   built (`cara_output_length_forecaster.py`, 39 tests). Next gate:
   integrate with CARA data and run backtest on scheduling-prior benefit.
   Wiring into the scheduler requires evaluating p50 as a SRTF-like prior.

5. **Admission gate simulation integration:** The `WorkloadAdmissionGate`
   (v1) is implemented but not wired into the cluster simulator or any
   backtest. Quantifying its goodput/$ impact requires trace-replay.

---

## 6. Highest Expected Value Opportunities (Ranked)

| rank | opportunity | expected upside | complexity | status | next step |
|---|---|---|---|---|---|
| 1 | SRTF/SPRPT ordering in the SERVING path + aging guard | High (Azure 2024 queue: −99.6% short-p90, +323% goodput/$ vs FIFO [run -g]) | Medium | **Quantified [run -g]** — value proven in serving simulator; not yet wired into serving runtime | Expose ordering option in serving path driven by OutputLengthForecastBundle.p50 + aging/preemption to bound long-tail starvation |
| 2 | GPU routing on LLM serving trace (TTFT violation reduction) | Medium | Low | **Benchmarked [run -f]** — energy trace −0.14% (price-dominated, expected) | Evaluate on BurstGPT where TTFT is the binding constraint |
| 3 | Admission gate simulation integration | Medium (prevents KV overflow spikes) | Medium | Implemented (unconnected) | Wire into cluster simulator + Azure 2024 replay |
| 4 | Wire OutputLengthForecastBundle.p50 as SRTF shadow prior | High (replaces runtime_hours proxy with calibrated token estimate) | Low | Infrastructure built (shadow) | Replace SRTF_TOKENS_PER_HOUR proxy with p50 from OutputLengthForecastBundle |
| 5 | BOute-style MOBO routing (arXiv:2602.10729, MLSys 2026) | High (2.57× improvement / 15-61% cost) | High | Not Started | Model deployment × routing co-optimisation via Bayesian BO |
| 6 | Mooncake trace ingestion (KV prefix reuse cross-validation) | Low-Medium | Low | Not Started | Bounded ingest (Apache-2.0) |
| 7 | TTFT p50 shadow integration (already shadow_ready) | Medium | Low-Medium | shadow_ready | Wire into routing decision |
| 8 | Hermes PDGraph for agentic workloads | High (for agentic) | High | Not Started | CC-traces agentic structure audit |
| 9 | Carbon-power MILP joint optimization | Medium | High | Not Started | Microgrid model design |
| 10 | TPOT forecasting after CARA train.jsonl expansion | Medium | Low | build_after_data_expansion | Expand CARA to train.jsonl (392 MB) |

---

## 7. Experiment History

### Run 2026-06-20-h — MODULE INTEGRATION + ECONOMIC VALIDATION

**Goal:** Stop building shadow modules. Wire the three existing research modules
(`WorkloadAdmissionGate`, `OutputLengthForecastBundle`, `GpuPlacementScorer`)
into the actual public replay path and measure whether they improve real public
benchmark KPIs. No new papers, no new modules, no synthetic-only main evidence.
(Ran in parallel with runs -f/-g, which wired SRTF into the batch scheduler and
proved a per-request serving-queue SRTF win — see below; the two are
complementary, see the cross-reference at the end.)

- **Phase 1 (audit):** Confirmed all three modules were shadow/dead code in the
  default replay path. `GpuPlacementScorer` is wired into `JobScheduler` but
  `JobScheduler` is only used by the canonical *energy* backtest — the public
  LLM traces (Azure 2024 / BurstGPT) run a *different* aggregate per-tick
  autoscaling replay (`aurelius/traces/backtest.py`) that never constructs a
  `JobScheduler`. So GPU placement never touched the public LLM replay.
- **Data:** Downloaded the real BurstGPT trace (1,429,738 requests, CC-BY-4.0).
  Azure-2024 full week is SAS-gated (HTTP 401) → used the committed 5,880-request
  sample (as the canonical Azure runner itself does). Real CAISO/PJM/ERCOT price
  CSVs present.
- **Phase 3 (baseline):** `research/results/baseline_public_backtest_2026-06-20.*`.
- **Phase 4 (integration):** Added `aurelius/traces/module_backtest.py` — reuses
  the LOCKED `backtest.py`/`serving.py`/`economics.py` verbatim, adds additive
  provisioning variants. A disabled gate is byte-identical to the locked
  constraint_aware baseline (`tests/test_module_backtest.py`). 153 tests pass.
- **Phase 6/7 (results, BurstGPT 100/300/600× = robust evidence):**
  - **WorkloadAdmissionGate → NEUTRAL** (goodput/$ Δ +0.19 / −0.34 / −0.29%).
    The baseline already provisions to a safe rho, so the gate rarely fires.
  - **OutputLengthForecastBundle → HURT** (−7.1 / −11.3 / −11.2%). The autoscaler
    already reads the realized per-tick mean; a forecast can only mis-size. Its
    SRTF ordering lever is *absent* from the aggregate replay physics.
  - **GpuPlacementScorer → proxy moved, real KPI regressed.** Routing proxy
    +54.7pp, but real goodput/$ −0.0004 overall and **−7.3% on the
    latency_critical subset** (routes to pricier H100, no monetized TTFT benefit).
- **Phase 8 (decision):** No module improves SLA-safe goodput/$ on the robust
  aggregate public replay. **Do NOT enable any of the three in runtime.** Keep
  shadow-only. Merge backtest infrastructure + report only.
- **Cross-reference to runs -f/-g:** This run's "output length forecasting has no
  lever in the *aggregate autoscaling* benchmark" finding is **consistent** with
  run -g's per-request serving-queue SRTF win: the value lives in per-request
  ordering, not aggregate sizing. This run independently confirms the aggregate
  path is the wrong surface — the exact reason run -g moved to a per-request
  discrete-event queue. The two results do not contradict.
- **Final status: INFRASTRUCTURE ONLY.** No runtime decision path changed by this
  run; no benchmark/SLA/price/workload definition changed; the three modules stay
  `enabled=False`.

### Run 2026-06-20-g (previous run)
- **Phase 1 (audit):** Run -f wired the SRTF sort key into the batch
  `JobScheduler` and showed it neutral on the energy trace, hypothesizing the
  gain needs queue contention. This run tests that hypothesis on the REAL
  Azure LLM 2024 serving trace and looks for actual measurable improvement.
- **Phase 2 (research):** Re-grounded on arXiv:2604.06970 (SRTF +32% p90
  short-request), arXiv:2410.01035 (TRAIL/SPRPT), arXiv:2604.07931 (ELIS robust
  length prediction). Key methodological insight: the +32% figure is a
  *request-level queue-discipline* result; it requires a discrete-event queue,
  not an aggregate Erlang-C model or a batch placement scheduler.
- **Phase 3 (benchmarks):** 38 new tests (27 serving + 11 contention); 0
  regressions across scheduler / canonical / gpu-routing / azure-ingestion
  suites (64 passed, 1 skipped on the regression slice).
- **Phase 4 (implementation):**
  - `aurelius/benchmarks/srtf_contention_backtest.py` (NEW): probes the merged
    batch scheduler under a binding power cap. **Negative finding:** Δ ≤ 0.05%
    even at 4.6× contention — the greedy batch path has no queue-wait semantics
    (falls back to `earliest_start`), so order never changes completion time.
  - `aurelius/benchmarks/srtf_serving_backtest.py` (NEW): discrete-event
    non-preemptive M/G/c simulator over the real Azure LLM 2024 request stream
    (5,880 real requests, real output-length distribution). FIFO vs
    shortest-predicted-first; perfect + 30%-CV-forecast priors; leakage guard
    (ordering=predicted, physics=actual); discipline-invariant goodput/$
    denominator (total GPU busy-seconds).
  - `tests/test_srtf_serving_backtest.py` (27), `tests/test_srtf_contention_backtest.py` (11).
  - `docs/SRTF_SERVING_BACKTEST_RESULTS.md` — full results + caveats.
- **Benchmark results (real Azure LLM 2024, ρ=0.85, c=4, SLA=10s):**
  - short-request p90 response: 696.2s → 3.03s (**−99.6%**)
  - mean response: 343.9s → 129.9s (**−62.2%**)
  - SLA-safe goodput/$: 13,336 → 56,481 (**+323.5%**)
  - 30%-CV forecast prior: short-p90 still −99.5% (robust)
  - long-request p99: 732.7s → 2188.7s (**REGRESSES** — non-preemptive SJF
    starvation; documented cost, asserted in tests)
  - holds across ρ ∈ {0.80, 0.85, 0.92}: short-p90 −99.5%+, goodput/$ +252–324%
- **Decision:** Positive (research infrastructure — does NOT change runtime
  decisions). Two artifacts: (a) negative finding locating where the merged
  sort key is inert; (b) large positive finding quantifying where SRTF actually
  pays off on a real serving trace. Honest caveats recorded: FIFO (not
  SLA-aware) baseline; regime-dependent magnitude; simulator/public-trace
  directional, not production savings; long-tail regression is the cost.
- **Next recommended direction:**
  1. Expose SRTF/SPRPT ordering in the SERVING runtime path (not the batch
     scheduler) driven by `OutputLengthForecastBundle.p50`, with an
     aging/preemption guard bounding long-request starvation; re-run this
     backtest end-to-end to realize the value.
  2. Add SRPT (preemptive) variant to the simulator and quantify the long-tail
     recovery vs the non-preemptive starvation cost.

### Run 2026-06-20-f (previous run)
- **Phase 1 (audit):** Repository audit. Run -e built the GPU routing benchmark
  harness and stated that canonical CSV files were "not present." This run
  discovered those CSV files ARE present (`data/caiso_us_west_dam.csv` etc.),
  ran both the GPU routing and the SRTF benchmarks with real CAISO/PJM/ERCOT
  price data, and wired `predicted_output_tokens` into the greedy scheduler
  sort key as an SRTF prior.
- **Phase 2 (research):** 3 new papers reviewed:
  1. "TRAIL: Embedding-Based Scheduling for LLMs" (arXiv:2410.01035) — SPRPT
     (Shortest Predicted Remaining Processing Time) via intermediate-layer
     embeddings achieves near-SRTF performance without clairvoyant token
     length access; validates the SRTF direction for production use.
  2. "Robust Length Prediction for Efficient LLM Serving" [ELIS]
     (arXiv:2604.07931) — iterative SRTF with encoder-based length predictor
     shows strong latency improvement on multi-tenant LLM serving clusters;
     supports the `OutputLengthForecastBundle` → SRTF prior pipeline.
  3. "EnergyLens: Energy-Aware LLM Inference Serving" (arXiv:2605.14249) —
     energy-aware batching and scheduling; confirms that energy × SLA joint
     optimization is the frontier direction for Aurelius's combined KPI.
- **Phase 3 (benchmarks):** 37 new SRTF tests + existing scheduler tests; all
  passing. Zero regressions. Two canonical benchmarks run with real
  CAISO/PJM/ERCOT price data for the first time.
- **Phase 4 (implementation):**
  - `aurelius/models.py`: added `predicted_output_tokens: Optional[float] = None`
    to `Job` dataclass — SRTF scheduling prior; `None` = no prior available;
    fully backward-compatible.
  - `aurelius/optimization/scheduler.py`: added `_SLA_CLASS_RANK` class variable
    (`latency_critical=0, deadline=1, best_effort=2`); modified `_solve_greedy`
    sort key to `(−priority, sla_class_rank, length_prior_or_inf, deadline)`.
    Backward-compatible: `predicted_output_tokens=None` → `float("inf")` →
    same order as original `(−priority, deadline)` for all-same-sla_class jobs.
  - `aurelius/benchmarks/srtf_backtest.py` (NEW): A/B benchmark module.
    `SRTF_TOKENS_PER_HOUR = 500_000.0`; `SRTF_ELIGIBLE_WORKLOAD_TYPES` =
    `realtime_inference` + `llm_batch_inference`; `augment_jobs_with_srtf_priors()`;
    `run_srtf_backtest()` (baseline vs SRTF); `SRTFBacktestReport` dataclass.
  - `tests/test_srtf_scheduling.py` (NEW): 37 tests (6 classes):
    `TestJobPredictedOutputTokens` (5), `TestSLAClassRank` (4),
    `TestGreedySortKey` (9), `TestAugmentJobsWithSRTFPriors` (10),
    `TestSRTFBacktestReport` (4), `TestSRTFEndToEnd` (5).
- **Benchmark results (real CAISO/PJM/ERCOT canonical trace, seed=20260201,
  1000 jobs, 26-day window):**
  - **GPU routing:** baseline=0.300667, gpu_routing=0.300246, Δ=−0.14%.
    Routing quality: +54.7 pp H100 placement for LC jobs (confirmed correct).
    Root cause of negative Δ: H100 GPUs are in PJM (us-east), the highest
    energy-cost region. On a 26-day energy-shifting window all jobs already
    meet deadlines without routing; energy cost dominates over TTFT gain.
    On LLM serving traces with binding TTFT SLAs the expected direction flips.
  - **SRTF scheduling:** baseline=0.352783, srtf=0.352783, Δ=0.0%.
    Expected neutral: energy scheduling has no queue contention (each job
    selects its optimal time slot independently over 26 days). SRTF gain
    materializes under request-queue pressure (LLM serving traces).
    Short-first ordering: ~100% of eligible jobs sorted correctly.
- **Decision:** Positive (infrastructure). Both canonical benchmarks confirm
  correct implementation and expected neutral/negative results on the energy
  trace (no queue contention). SRTF sort key wired and backward-compatible.
  Zero regressions. Infrastructure ready for LLM serving trace evaluation.
- **Next recommended direction:**
  1. Evaluate SRTF on BurstGPT and Azure LLM 2024 (queue contention present)
     — expected +15–32% p90 short-request gain from arXiv:2604.06970.
  2. Evaluate GPU routing on a trace where TTFT violations are the binding SLA
     constraint (not deadline miss on a 26-day batch-scheduling window).
  3. Wire `WorkloadAdmissionGate` into cluster simulator for Azure 2024 replay.

### Run 2026-06-20-e
- **Phase 1 (audit):** Repository audit completed. Run -d wired `GpuPlacementScorer`
  into the scheduler but left GPU routing unvalidated on any price-data trace because
  the canonical benchmark had no `region_gpu_types` metadata. This run targets the
  #1 EV gap: adding synthetic `region_gpu_types` to the canonical benchmark and
  building a full GPU routing evaluation harness.
- **Phase 2 (research):** No new papers added (implementation-only run).
- **Phase 3 (benchmarks):** 5132 tests passing (5098 pre-existing + 34 new).
  Pre-existing failures unchanged (PyYAML, LightGBM, live API, benchmark harness
  file structure). 0 new failures introduced.
- **Phase 4 (implementation):**
  - `aurelius/benchmarks/gpu_routing_backtest.py` — NEW benchmark module:
    - `CANONICAL_REGION_GPU_TYPES`: us-west→a100, us-east→h100, us-south→t4
      (based on CARA dataset fleet composition; H100 in PJM zone, A100 in CA,
      T4/lower-tier in Texas ERCOT zone)
    - `SYNTHETIC_GPU_TTFT_P50_S`: CARA-calibrated p50 medians
      (H100≈0.12 s, A100≈0.28 s, T4≈0.95 s; 8× spread; CARA cites 9× p99 spread)
    - `build_synthetic_prior()`: 200 rows per GPU type, Gaussian noise 10% CV;
      `by_gpu_counts ≥ 50` (passes min_subgroup_rows threshold)
    - `augment_jobs_with_sla_class()`: stamps `sla_class` from
      `WORKLOAD_DEFAULT_SLA_CLASS` (realtime_inference → latency_critical)
    - `GpuRoutingReport`: dataclass + `to_dict()` with routing quality KPIs
      (pct_latency_critical_on_best_gpu, mean_gpu_penalty, energy delta,
       goodput/$ delta for all jobs and latency_critical subset)
    - `_compute_ttft_penalty()`: per-schedule TTFT penalty accounting
    - `run_gpu_routing_backtest()`: end-to-end benchmark comparing baseline
      (no GPU routing) vs gpu_routing (scorer enabled, region_gpu_types wired)
  - `tests/test_gpu_routing_backtest.py` — 34 new tests:
    - 8 × prior builder (by_gpu, by_gpu_counts, TTFT ordering, row counts)
    - 6 × job augmentation (all workload types, non-mutation invariant)
    - 4 × GpuRoutingReport structure / serialization
    - 6 × penalty computation (floor/ceil, mixed, empty, disabled)
    - 6 × integration with mocked price data (routing improvement verified)
    - 4 × regression invariants (TTFT ordering, 5× spread, region coverage)
- **Benchmark results (synthetic flat-price evaluation):**
  - GPU routing routes more latency_critical jobs to H100 (us-east) vs baseline.
  - Mean TTFT penalty for lc jobs drops from ~0.27 (baseline mix of A100/T4) to
    ~0.05 (floor; H100 routing confirmed).
  - Routing improvement is measurable and directionally positive on synthetic data.
  - Full quantitative delta on real canonical price data requires data files
    (caiso_us_west_dam.csv etc.) to be present; benchmark infra is complete.
- **Decision:** Positive. Benchmark infrastructure complete. 34 new tests pass.
  No regressions. The GPU routing evaluation harness is now in place; running
  `run_gpu_routing_backtest()` with real price data yields the full KPI table.
- **Next recommended direction:**
  1. Wire `OutputLengthForecastBundle` p50 into greedy scheduler sort key as
     SRTF prior (next highest EV: arXiv:2604.06970 shows +32% p90 short-request).
  2. Run `run_gpu_routing_backtest()` with real canonical price data present
     to produce the quantitative goodput/$ delta table.
  3. Wire `WorkloadAdmissionGate` into cluster simulator for Azure 2024 replay.

### Run 2026-06-20-d (previous run)
- **Phase 1 (audit):** Repository audit completed. Three prior runs have implemented
  WorkloadAdmissionGate (run -a), OutputLengthForecastBundle (run -b), and
  GpuPlacementScorer (run -c). This run targets the #1 EV gap: wiring the
  GpuPlacementScorer into the scheduler for latency_critical placements.
- **Phase 2 (research):** 3 new papers reviewed:
  1. "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
     Large Language Model Serving" (arXiv:2401.09670, OSDI'24) — separates
     prefill (latency-bound) from decode (throughput-bound) onto dedicated GPU
     pools to optimize TTFT independently of TPOT. Maps directly to the
     placement scorer rationale: routing latency_critical prefills to fast
     GPU types is equivalent to DistServe's disaggregated prefill pool.
  2. "Splitwise: Efficient Generative LLM Inference Using Phase Splitting"
     (arXiv:2311.18677, ISCA'24) — extends the disaggregation idea to KV-cache
     migration between prefill and decode nodes. Validates that per-request
     TTFT can be significantly reduced via hardware-tier routing. Relevance:
     the 9× TTFT spread in CARA across GPU types (H100 vs T4) is the
     heterogeneous-cluster analogue of Splitwise's phase splitting.
  3. "Efficient LLM Scheduling by Learning to Rank" (arXiv:2408.15792) —
     formally cited in gpu_placement_scorer.py but not yet documented in
     experiment history. Proposes SRTF-like ranking of requests by predicted
     service time; when combined with the GPU placement scorer (rank-by-TTFT),
     the two approaches become complementary: GPU routing reduces absolute TTFT,
     and SRTF ordering reduces queueing delay for short requests.
- **Phase 3 (benchmarks):** 105 tests passing (77 pre-existing + 28 new).
  Pre-existing failures: 5 LightGBM, 4 PyYAML/SLA, 4 benchmark harness file
  structure, 5 live API (all pre-existing, unchanged). 0 new failures.
- **Phase 4 (implementation):**
  - `aurelius/forecasting/ttft_shadow_prior.py`:
    - `by_gpu_counts` field added (total rows per GPU type for subgroup_n fallback)
    - `predict()` extended: `model_size=None` now falls through to `by_gpu` lookup
      enabling GPU-type peer comparison without model-size context
    - `subgroup_n()` uses `by_gpu_counts` when `model_size=None`
    - `to_dict()` / `load_prior()` updated (backward-compatible)
  - `aurelius/optimization/scheduler.py`:
    - `__init__()`: `gpu_placement_scorer` + `region_gpu_types` kwargs
    - `_sla_adjusted_score()`: `gpu_penalty: float = 0.0` parameter (additive)
    - `_find_best_slot()`: pre-computes peer GPU TTFT p50s per job; folds
      `latency_penalty` from scorer into candidate score via `_sla_adjusted_score`
    - Full fail-open: disabled/missing scorer → gpu_penalty=0.0 for all candidates
  - `tests/test_scheduler_gpu_placement.py` — 28 new tests:
    - 9 × TTFTShadowPrior GPU fallback behavior
    - 5 × `_sla_adjusted_score` with gpu_penalty parameter
    - 14 × end-to-end scheduler routing integration
- **Benchmark results (directional, synthetic prior):**
  - With equal prices: latency_critical jobs routed to H100 vs T4 (confirmed).
  - With T4 20% cheaper: TTFT penalty (0.50 × |obj|) exceeds price advantage
    → latency_critical jobs still go to H100 (confirmed).
  - best_effort jobs: unaffected, route to cheapest region as before (confirmed).
  - Three-GPU ranking: h100 → a100 → t4 preference order verified end-to-end.
  - Quantitative SLA-safe goodput/$ delta on BurstGPT/Azure 2024 traces
    requires adding synthetic `region_gpu_types` to benchmark replay (next run).
- **Decision:** Positive. Integration complete. 28 new tests pass. No regression.
  Architecture closes the gap from "scorer built but unconnected" (run -c) to
  "scorer active in scheduler for latency_critical SLA class" (run -d).
- **Next recommended direction:**
  1. Add synthetic `region_gpu_types` metadata to BurstGPT + Azure 2024 canonical
     backtest — assign regions to GPU types matching CARA fleet (H100 / A100 / T4)
     and measure SLA-safe goodput/$ delta with GPU routing enabled vs disabled.
  2. Wire `OutputLengthForecastBundle` p50 into greedy scheduler sort key as SRTF
     prior — use `num_predicted_output_tokens` as a secondary sort dimension after
     SLA class, with `actual_output_tokens` reserved as label-only.
  3. Wire `WorkloadAdmissionGate` into cluster simulator for Azure 2024 trace replay.

### Run 2026-06-20-c (previous run)
- **Phase 1 (audit):** Repository audit completed. Two prior runs have implemented
  WorkloadAdmissionGate v1 (run -a) and OutputLengthForecastBundle v1 (run -b).
  This run targets the #3 EV opportunity: Heterogeneous GPU Placement Scorer.
- **Phase 2 (research):** 3 new papers reviewed:
  1. "KAIROS: Stateful, Context-Aware Power-Efficient Agentic Inference Serving"
     (arXiv:2604.16682, April 2026) — hardware-aware placement, heterogeneous
     scheduling, and power-aware cluster orchestration. Defines SLOs via TTFT +
     TBT metrics with per-GPU-type awareness. Maps directly to placement scorer.
  2. "Semi-Clairvoyant Scheduling of Speculative Decoding Requests" [LAPS-SD]
     (arXiv:2505.17074, IJCAI 2025) — Least-Attained/Perceived-Service for SD
     adaptively schedules based on predicted output length + token acceptance rate.
     Extends arXiv:2604.06970 to speculative decoding regime; relevant to the
     SRTF scheduling direction.
  3. "LLM Serving Needs Mathematical Optimization and Algorithmic Foundations,
     Not Just Heuristics" (arXiv:2605.01280, May 2026) — advocates for rigorous
     MILP/LP formulation for LLM scheduling; supports Aurelius's existing MILP
     path and validates the TTFT-aware placement direction.
  Also reviewed: Hetis (arXiv:2509.08309, fine-grained parallelism for
  heterogeneous GPU clusters), TokenFlow (arXiv:2510.02758, burst-resilient
  preemptive scheduling), AccelGen (arXiv:2503.13737, SLO-guaranteed multi-app
  heterogeneous inference).
- **Phase 3 (benchmarks):** 4964 tests passing (subset; excludes backtesting/ml/
  live/html-reporting dirs which need lightgbm/matplotlib). 5 pre-existing
  LightGBM failures unchanged. 37 new GPU placement scorer tests all pass.
- **Phase 4 (implementation):** Implemented `GpuPlacementScorer v1`:
  - `aurelius/forecasting/gpu_placement_scorer.py` — 280 LOC, pure stdlib +
    numpy, shadow-only.
  - Components: `GpuPlacementConfig` (configures enabled/sla_classes/thresholds),
    `GpuPlacementScore` (per-candidate score with ttft_p50_s, relative_rank,
    latency_penalty, status), `GpuPlacementScorer` (rank_gpu_types + score).
  - Integration point: wraps `TTFTShadowPrior` (already fitted from CARA);
    adds peer-normalized penalty in [0, penalty_floor..penalty_ceil].
  - Safety: enabled=False default; fail-open for missing/insufficient prior;
    no penalty for non-latency-critical SLA classes; no controller imports.
  - `tests/test_gpu_placement_scorer.py` — 37 tests, all passing.
  - Exported from `aurelius/forecasting/__init__.py`.
- **Decision:** Positive (closes the #3 EV gap). Scorer is shadow-only with full
  safety tagging. 37 tests pass. Enables scheduler to optionally exploit the 9×
  TTFT spread across GPU types seen in CARA data.
- **Next recommended direction:**
  1. Wire `GpuPlacementScorer.latency_penalty` into scheduler objective for
     `latency_critical` placements — fold as additive penalty on `obj.total`.
  2. Evaluate on BurstGPT + Azure LLM 2024 with synthetic GPU-type labels to
     quantify goodput/$ delta from TTFT-aware routing.
  3. Wire `OutputLengthForecastBundle` p50 into SRTF scheduler ordering.
  4. Wire admission gate into cluster simulator for Azure 2024 trace replay.

### Run 2026-06-20-b (previous run)
- **Phase 1 (audit):** Repository audit completed. Previous run (2026-06-20)
  implemented WorkloadAdmissionGate v1. This run builds on that foundation.
- **Phase 2 (research):** 3 new papers reviewed:
  1. "Predicting LLM Output Length via Entropy-Guided Representations"
     (arXiv:2602.11812, ICLR 2026) — EGTP + PLP achieves -29.16% MAE vs
     baselines for output length prediction; enables semi-clairvoyant scheduling.
  2. "Fast Heterogeneous Serving: Scalable Mixed-Scale LLM Allocation for
     SLO-Constrained Inference" (arXiv:2604.07472) — AGH achieves near-optimal
     SLO-compliant allocation in 3 seconds; applicable to heterogeneous GPU
     placement scorer.
  3. "BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via
     Multi-Objective Bayesian Optimization" (arXiv:2602.10729, MLSys 2026) —
     2.57× improvement or 15-61% cost reduction via MOBO routing + deployment
     co-optimisation across heterogeneous GPU/model combinations.
  Also reviewed: SLAI scheduler (arXiv:2508.01002, 53% median TTFT reduction,
  26% capacity increase), Fluid-Guided Online Scheduling (arXiv:2504.11320).
- **Phase 3 (benchmarks):** 5123 tests passing, 10 pre-existing failures
  (jinja2 / HTML reporting dependency not installed locally), 18 skipped.
  Key modules confirmed: canonical energy backtest 17/17, frontier admission
  38/38, CARA latency forecaster all passing.
- **Phase 4 (implementation):** Implemented `OutputLengthForecastBundle v1`:
  - `aurelius/forecasting/cara_output_length_forecaster.py` — 320 LOC.
    Components: `BiasCalibrationForecaster` (Huber regression debiasing),
    `HGBOutputLengthForecaster` (HGB quantile at p50/p90/p95),
    `OutputLengthForecastBundle` (combines both with batch API),
    `compute_bias_stats`, `compute_percentile_stats` (pure audit helpers).
  - `tests/test_cara_output_length_forecaster.py` — 39 tests, all passing.
  - Exported from `aurelius/forecasting/__init__.py`.
  - Key design: `actual_output_tokens` is leakage — only used as label.
    `num_predicted_output_tokens` is predict-time; calibration corrects
    systematic engine bias. p90 ≥ p50 invariant enforced in all paths.
- **Decision:** Positive (enables #1 ranked research opportunity). Module
  is shadow-only with full safety tagging. 39 tests pass. 5123 total pass.
- **Next recommended direction:**
  1. Wire `OutputLengthForecastBundle` into CARA latency backtest to
     measure calibration MAE reduction on CARA train/test split.
  2. Implement SRTF-like scheduling prior using p50 predictions.
  3. Evaluate on Azure LLM 2024 trace with simulated output-length priors.
  4. Build heterogeneous GPU placement scorer (rank 3, next highest EV).

### Run 2026-06-20 (previous run)
- **Phase 1 (audit):** Repository audit completed. No `research/` directory
  existed; created with ROADMAP, BENCHMARK_REGISTRY, GAP_ANALYSIS.
- **Phase 2 (research):** 3 new papers reviewed:
  1. "Scheduling the Unschedulable" (arXiv:2604.06970) — semi-clairvoyant
     scheduling, token magnitude priors, Adaptive DRR + cost-ladder admit.
  2. "Efficient Serving with Probabilistic Demand Modeling" [Hermes]
     (arXiv:2506.14851) — Gittins policy + PDGraph + backend warming.
  3. "Flow-Controlled Scheduling for LLM Inference" (arXiv:2604.11001) —
     provably-stable flow-rate admission control for KV-cache overflow.
  Also reviewed: DynamoLLM (HPCA'25, 52% energy reduction, 38% carbon,
  61% cost savings at SLA parity), CarbonFlex (2505.18357),
  Carbon-Aware MILP (arXiv:2605.03751).
- **Phase 3 (benchmarks):** 5018 tests passing (pre-existing: 12 failures
  from missing LightGBM + FastAPI dependencies); canonical energy backtest
  17/17 passing.
- **Phase 4 (implementation):** Implemented `WorkloadAdmissionGate v1`:
  - `aurelius/frontier/admission.py` — 340 LOC, pure stdlib, shadow-only.
  - `tests/test_frontier_admission.py` — 38 tests, all passing.
  - Exported from `aurelius/frontier/__init__.py`.
- **Decision:** Neutral-to-positive (safety infrastructure). Gate is
  shadow-only and adds no benchmark risk. 38 tests pass. Merged.
- **Next recommended direction:** Audit CARA output token prediction
  (actual vs predicted). Then wire admission gate into cluster simulator
  for Azure 2024 trace replay to quantify goodput/$ delta.
