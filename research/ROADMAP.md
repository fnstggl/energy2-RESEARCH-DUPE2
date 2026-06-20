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
- **Output token length [NEW - run 2026-06-20-b]:**
  `aurelius/forecasting/cara_output_length_forecaster.py` — calibrated
  output-token-count predictor. Two components: (a) `BiasCalibrationForecaster`
  debiases `num_predicted_output_tokens` via Huber regression; (b)
  `HGBOutputLengthForecaster` predicts `actual_output_tokens` at p50/p90/p95
  from all predict-time CARA features. Shadow-only. 39 tests passing.
  Enables semi-clairvoyant scheduling (arXiv:2604.06970).
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

### 4.3 Semi-Clairvoyant Scheduling

#### Output Length Prediction for Token-Magnitude Priors
- **Status:** Not Started
- **Expected upside:** High — "Scheduling the Unschedulable" (arXiv:
  2604.06970) shows token magnitude priors increase P90 short-request
  performance by 32% vs FIFO, and removing magnitude increases p95 by
  5.8×.
- **Complexity:** Medium — needs output token prediction from request
  features (prompt type, user context, model).
- **Risks:** Without good token prediction the scheduling gains erode.
- **Datasets needed:** CARA carries `num_predicted_output_tokens` vs
  `actual_output_tokens` — ready for output-length prediction audit.
- **Next steps:** Audit CARA actual vs predicted output token counts.

### 4.4 Probabilistic Demand Modeling

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

### 4.5 Carbon-Aware Joint Optimization

#### Carbon-Aware Compute-Power MILP (Prosumer Datacenter)
- **Status:** Not Started
- **Expected upside:** Medium — carbon-power MILP (arXiv:2605.03751)
  shows substantial improvement vs compute-only or energy-only baselines,
  with inference routing flexibility as the major value driver.
- **Complexity:** High — needs battery/generation dispatch model.
- **Risks:** Realistic microgrid parameters hard to obtain for simulator.
- **Note:** Energy shifting already implemented and already sufficient per
  `FORECAST_LEVERAGE_AUDIT.md`. This is the next frontier.

### 4.6 Data Expansion

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
| 2 | Admission gate simulation integration | Medium (prevents KV overflow spikes) | Medium | Implemented (unconnected) | Wire into cluster simulator + Azure 2024 replay |
| 3 | Heterogeneous GPU placement scorer (TTFT 9× spread) | High | Medium | build_now | Build placement scorer using HGB TTFT forecasts per GPU type |
| 4 | BOute-style MOBO routing (arXiv:2602.10729, MLSys 2026) | High (2.57× improvement / 15-61% cost) | High | Not Started | Model deployment × routing co-optimisation via Bayesian BO |
| 5 | Mooncake trace ingestion (KV prefix reuse cross-validation) | Low-Medium | Low | Not Started | Bounded ingest (Apache-2.0) |
| 6 | TTFT p50 shadow integration (already shadow_ready) | Medium | Low-Medium | shadow_ready | Wire into routing decision |
| 7 | Hermes PDGraph for agentic workloads | High (for agentic) | High | Not Started | CC-traces agentic structure audit |
| 8 | Carbon-power MILP joint optimization | Medium | High | Not Started | Microgrid model design |
| 9 | TPOT forecasting after CARA train.jsonl expansion | Medium | Low | build_after_data_expansion | Expand CARA to train.jsonl (392 MB) |

---

## 7. Experiment History

### Run 2026-06-20-b (this run)
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
