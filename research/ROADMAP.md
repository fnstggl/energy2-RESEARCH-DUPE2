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
- **Next steps:**
  1. Run `run_gpu_routing_backtest()` with real canonical price data to get
     the quantitative goodput/$ delta table.
  2. Tune `penalty_floor`/`penalty_ceil` from CARA TTFT distribution data.
  3. Wire into BurstGPT / Azure 2024 trace replay with per-region GPU labels.

### 4.4 Semi-Clairvoyant Scheduling

#### Output Length Prediction for Token-Magnitude Priors
- **Status:** Infrastructure built (shadow-only)
- **Expected upside:** High — "Scheduling the Unschedulable" (arXiv:
  2604.06970) shows token magnitude priors increase P90 short-request
  performance by 32% vs FIFO, and removing magnitude increases p95 by
  5.8×. LAPS-SD (arXiv:2505.17074, IJCAI 2025) extends to speculative
  decoding — predicted length + token acceptance rate together determine
  optimal service ordering.
- **Complexity:** Medium — `OutputLengthForecastBundle` is built; remaining
  work is wiring p50 into scheduler request ordering.
- **Risks:** Without good token prediction the scheduling gains erode.
- **Datasets available:** CARA carries `num_predicted_output_tokens` vs
  `actual_output_tokens` — ready for backtest.
- **Next steps:**
  1. Wire `OutputLengthForecastBundle` p50 into greedy scheduler sort key.
  2. Evaluate on BurstGPT and Azure 2024 with simulated output-length priors.

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
| 1 | Output length prediction → semi-clairvoyant scheduling | High (+32% p90 short-request per arXiv:2604.06970) | Medium | Infrastructure built (shadow) | Wire calibration + HGB into CARA backtest; integrate p50 as SRTF prior |
| 2 | GPU routing benchmark on real price data | High (TTFT 9× spread) | Low | **Benchmark infra complete [run -e]** | Run run_gpu_routing_backtest() with canonical CSV files present |
| 3 | Admission gate simulation integration | Medium (prevents KV overflow spikes) | Medium | Implemented (unconnected) | Wire into cluster simulator + Azure 2024 replay |
| 4 | BOute-style MOBO routing (arXiv:2602.10729, MLSys 2026) | High (2.57× improvement / 15-61% cost) | High | Not Started | Model deployment × routing co-optimisation via Bayesian BO |
| 5 | Mooncake trace ingestion (KV prefix reuse cross-validation) | Low-Medium | Low | Not Started | Bounded ingest (Apache-2.0) |
| 6 | TTFT p50 shadow integration (already shadow_ready) | Medium | Low-Medium | shadow_ready | Wire into routing decision |
| 7 | Hermes PDGraph for agentic workloads | High (for agentic) | High | Not Started | CC-traces agentic structure audit |
| 8 | Carbon-power MILP joint optimization | Medium | High | Not Started | Microgrid model design |
| 9 | TPOT forecasting after CARA train.jsonl expansion | Medium | Low | build_after_data_expansion | Expand CARA to train.jsonl (392 MB) |

---

## 7. Experiment History

### Run 2026-06-20-e (this run)
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
