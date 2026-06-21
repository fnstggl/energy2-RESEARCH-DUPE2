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

**Aging-SRTF anti-starvation [run 2026-06-20-i]:** SRTF-with-Aging
(key(r,t) = predicted_tokens / (1 + α·wait_s)) reduces long-request p99
starvation by **55%** vs pure SRTF while retaining **+22.4% goodput/$** vs FIFO
(α=0.05) or **+70.7% goodput/$** vs FIFO (α=0.01, recommended sweet spot).
Short_p90 improvement preserved at 70-78% vs FIFO (vs 99.6% for pure SRTF).
Research basis: Astraea (arXiv:2512.14142), FlowPrefill (arXiv:2602.16603),
Equinox (arXiv:2508.16646). 37 new tests. See `docs/SRTF_AGING_BACKTEST_RESULTS.md`.

**Preemptive SRPT [run 2026-06-20-j]:** Discrete-event M/G/c SRPT preemptive
simulator added. On Azure LLM 2024 (5,880 requests, ρ=0.85): **+322.2%
SLA-safe goodput/$ vs FIFO** (within 0.3 pp of SRTF perfect) with the **best
short_p90 across all disciplines** (1.89s, +99.73% vs FIFO's 696s). Anti-starvation
guarantee: long jobs make monotonic forward progress (remaining service decreases
on every quantum). Long_p99 regression (+223.4%) nearly matches SRTF (+223.5%)
— starvation bounded but not eliminated at high utilization. Research basis:
TRAIL (arXiv:2410.01035), FlowPrefill (arXiv:2602.16603), SRPT multiserver
(arXiv:1805.07686). 42 new tests. See `docs/SRPT_PREEMPTIVE_BACKTEST_RESULTS.md`.

**Hybrid Aging+Preemptive SRPT [run 2026-06-20-k]:** Hybrid discipline implemented
with preemption+dispatch key `remaining_s / (1 + α·accumulated_wait_s)` (α=0.01).
On Azure LLM 2024: **+64.2% goodput/$ vs FIFO**, long_p99 **34.7% better than pure SRPT**.
Key finding: α=0.01 aging dispatch key promotes long-waiting requests, making hybrid
behave like Aging-SRTF (+64.2%) rather than SRPT (+322%). Anti-starvation IS working
(long_p99: 1,550s vs SRPT 2,373s). Root cause identified: unified aging key for
preemption + dispatch. Next step: decouple — use remaining_s for preemption (SRPT
throughput) + remaining/(1+α·wait) for dispatch only (starvation bound). Research basis:
FastServe (NSDI '26), Chimera (arXiv:2603.22206), SEK-SMOD (arXiv:2510.25963). 43 new
tests. See `docs/HYBRID_AGING_PREEMPTIVE_BACKTEST_RESULTS.md`.

**Decoupled Hybrid SRPT [run 2026-06-21-l]:** Decoupled discipline implemented with
PREEMPTION by pure `remaining_s` (SRPT) and DISPATCH by `remaining_s / (1 + α·total_wait_s)`
(aging). On Azure LLM 2024 (5,880 requests, ρ=0.85, c=4, SLA=10s): **+184.5% SLA-safe
goodput/$ vs FIFO** — between Aging-SRTF (+70.7%) and SRPT (+322.2%). Long_p99 regression
+132.3% (better than SRPT's +223.4%, worse than Aging-SRTF's +113.8%). Short_p90 improvement
+97.9% (best after pure SRPT at +99.7%). Key finding: decoupling preemption from dispatch
recovers 185% of SRPT's goodput while moderating long-tail regression. Research basis:
FastServe (NSDI '26), Chimera (arXiv:2603.22206). 42 new tests. See
`docs/DECOUPLED_HYBRID_BACKTEST_RESULTS.md`.

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

#### Aging-SRTF Anti-Starvation Guard [NEW — run 2026-06-20-i]
- **Status:** Implemented + benchmarked on Azure LLM 2024 (5880 requests)
- **Expected upside:** Medium–High — resolves the key production barrier from
  run -g (long-request starvation). Recommended α=0.01 gives +70.7% goodput/$
  vs FIFO with 49% starvation reduction.
- **Research basis:** Astraea (arXiv:2512.14142), FlowPrefill (arXiv:2602.16603),
  Equinox (arXiv:2508.16646).
- **Implementation:** `simulate_queue(..., discipline="aging_srtf", aging_alpha=...)`
  in `aurelius/benchmarks/srtf_serving_backtest.py`. O(|ready|) per dispatch
  (re-evaluates all waiting requests at current time t). 37 tests passing.
- **Key results (Azure LLM 2024, ρ=0.85, SLA=10s):**
  - α=0.01: +70.7% gp/$, short_p90 −78.1%, long_p99 +113.8% (49% less starvation)
  - α=0.05: +22.4% gp/$, short_p90 −70.7%, long_p99 +101.7% (55% less starvation)
- **New loaders/benchmarks:** `load_burstgpt_serving_requests()`,
  `run_aging_srtf_backtest()`, `run_burstgpt_aging_backtest()`.
- **BurstGPT:** Sample too small (51 rows) for robust starvation analysis;
  full 1.4M-row dataset needed for cross-trace confirmation.
- **Next steps:**
  1. Add preemptive SRPT variant to simulator (FlowPrefill-style).
  2. Cross-validate on full BurstGPT (1.4M rows).
  3. Wire aging_srtf into the serving runtime path driven by
     OutputLengthForecastBundle.p50.

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
| 1 | **Decoupled Hybrid: SRPT preemption + aging dispatch** | **Very High** (+322% goodput + better long_p99) | **Low** | **Next [run -l]** — root cause from run -k identified | Use remaining_s for preemption; remaining_s/(1+α·wait) for dispatch. Expected: SRPT goodput + Aging-SRTF long_p99 |
| 2 | Wire SRPT preemptive or aging-SRTF into SERVING runtime | High (+322% or +70.7% gp/$ vs FIFO in sim) | Medium | **Quantified [runs -i/-j/-k]** — serving simulator; not yet in runtime | Expose SRPT preemptive in serving path driven by OutputLengthForecastBundle.p50 |
| 3 | Hybrid Aging+Preemptive SRPT (combined key) | Medium (+64.2% gp/$ vs FIFO confirmed) | Medium | **DONE [run -k]** | α=0.01 behaves like Aging-SRTF; next: decoupled design for SRPT-level goodput |
| 4 | Full BurstGPT (1.4M rows) cross-validation | Medium (confirms generalization) | Low | Not Started | Download full BurstGPT_1.csv; run run_burstgpt_srpt_preemptive_backtest() at scale |
| 5 | Wire OutputLengthForecastBundle.p50 as live SRPT prior | High (replaces oracle prior) | Low | Infrastructure built (shadow) | Replace perfect-prior in aging_srtf / hybrid with OutputLengthForecastBundle.p50 |
| 6 | GPU routing on LLM serving trace (TTFT violation reduction) | Medium | Low | **Benchmarked [run -f]** — energy trace −0.14% (price-dominated) | Evaluate on BurstGPT where TTFT is the binding constraint |
| 7 | Admission gate simulation integration | Medium (prevents KV overflow spikes) | Medium | Implemented (unconnected) | Wire into cluster simulator + Azure 2024 replay |
| 8 | BOute-style MOBO routing (arXiv:2602.10729, MLSys 2026) | High (2.57× improvement / 15-61% cost) | High | Not Started | Model deployment × routing co-optimisation via Bayesian BO |
| 9 | Mooncake trace ingestion (KV prefix reuse cross-validation) | Low-Medium | Low | Not Started | Bounded ingest (Apache-2.0) |
| 10 | Hermes PDGraph for agentic workloads | High (for agentic) | High | Not Started | CC-traces agentic structure audit |
| 11 | Carbon-power MILP joint optimization | Medium | High | Not Started | Microgrid model design |

---

## 7. Experiment History

### Run 2026-06-21-l — DECOUPLED HYBRID SRPT SERVING-QUEUE SIMULATOR

**Goal:** Fix the root cause identified in run -k: the unified aging key for both
preemption and dispatch makes the hybrid behave like Aging-SRTF (+64.2% goodput)
instead of SRPT (+322%). Solution: decouple the two decisions — PREEMPTION by pure
`remaining_s` (preserves SRPT throughput benefit), DISPATCH by `remaining_s / (1 +
α·total_wait_s)` (adds starvation bound for queue selection). Hypothesis: achieve
SRPT-level goodput (+322%) with long_p99 regression bounded closer to Aging-SRTF
(+113% vs FIFO) instead of SRPT (+223%).

- **Phase 1 (audit):** Read ROADMAP, GAP_ANALYSIS. Confirmed run -k root cause:
  unified aging key at α=0.01 has flip point at ~66.7s accumulated wait, causing
  systematic promotion of long-waiting requests over short fresh arrivals. The fix
  is to decouple preemption from dispatch.

- **Phase 3 (research — 2 papers):**
  1. **FastServe** (USENIX NSDI '26) — MLFQ with skip-join promotes requests between
     queues based on remaining tokens; starvation prevention via bounded promotion.
     Validates decoupling dispatch priority from preemption trigger.
  2. **Chimera** (arXiv:2603.22206, March 2026) — aging-based dispatch key for
     multi-agent LLM serving; explicit aging factor in dispatch but NOT preemption.
     The Chimera design is exactly the decoupled architecture this run implements.

- **Phase 4 (implementation):**
  - `aurelius/benchmarks/srtf_serving_backtest.py` extended:
    - `DECOUPLED_HYBRID_ALPHA_DEFAULT = 0.01` constant.
    - `_simulate_decoupled_hybrid(requests, servers, aging_alpha)` — full
      discrete-event M/G/c simulator. Preemption: when a new arrival has
      `service_s < server_remaining_s` (pure SRPT, no aging factor).
      Dispatch: `min_i(remaining_s_i / (1 + α·total_wait_s_i))` (aging key
      applied to queue selection only). Tracks `frozen_wait_s` across preemption
      intervals for correct accumulated-wait accounting.
    - `simulate_queue(..., discipline="decoupled_hybrid")` dispatch branch.
    - `DecoupledHybridReport` dataclass (6 disciplines + delta KPIs for all 5
      non-FIFO disciplines).
    - `_run_decoupled_hybrid_backtest_on_trace()` shared backtest helper.
    - `run_decoupled_hybrid_backtest()` — Azure LLM 2024 public API.
    - `run_burstgpt_decoupled_hybrid_backtest()` — BurstGPT cross-validation.
  - `tests/test_srtf_decoupled_hybrid_backtest.py` (NEW) — 42 tests, 9 classes,
    all passing. Key tests: verify SRPT preemption triggers independently of aging
    (frozen_wait does NOT block preemption), aging dispatch correctly promotes
    long-waiting jobs in queue selection, α=0 collapses to pure SRPT.

- **Phase 7 (benchmark results — public trace replay):**

  **Dataset:** Azure LLM 2024 (5,880 requests, real output-length distribution)
  **Command:** `python -c "from aurelius.benchmarks.srtf_serving_backtest import run_decoupled_hybrid_backtest; r = run_decoupled_hybrid_backtest(servers=4, target_rho=0.85, aging_alpha=0.01); print(r.to_dict())"`

  **Azure LLM 2024 (5,880 requests, ρ=0.85, SLA=10s, c=4):**
  | KPI | FIFO | Aging-SRTF (α=0.01) | SRPT Preemptive | Hybrid (α=0.01) | **Decoupled (α=0.01)** |
  |---|---:|---:|---:|---:|---:|
  | SLA-safe goodput/$ | 13,336 | 22,768 (+70.7%) | 56,311 (+322.2%) | 21,899 (+64.2%) | **37,945 (+184.5%)** |
  | short_p90 response (s) | 696.16 | 152.61 | 1.89 | 169.26 | **14.41** |
  | short_p90 improvement | — | +78.1% | +99.7% | +75.7% | **+97.9%** |
  | long_p99 response (s) | 733.55 | 1,568 | 2,373 | 1,550 | **1,703** |
  | long_p99 regression | — | +113.8% | +223.4% | +111.3% | **+132.3%** |

  **BurstGPT (51-request fixture, ρ=0.85, SLA=30s, c=4):**
  | KPI | FIFO | SRPT Preemptive | **Decoupled (α=0.01)** |
  |---|---:|---:|---:|
  | SLA-safe goodput/$ | 70,975 | 67,754 (−4.5%) | **67,754 (−4.5%)** |
  | short_p90 improvement | — | +67.5% | **+67.5%** |
  | long_p99 regression | — | +16.0% | **+16.1%** |

- **Key empirical finding:** Decoupled Hybrid achieves **+184.5% goodput/$ vs FIFO**
  by preserving SRPT preemption while using aging for dispatch. This is 2.63× the
  gain of Hybrid (+64.2%) and 2.6× the gain of Aging-SRTF (+70.7%), confirming the
  root cause of run -k's underperformance. However, decoupled falls short of pure
  SRPT (+322.2%) — the aging dispatch occasionally promotes long-waiting requests over
  shorter fresh jobs, reducing throughput by ~137pp vs pure SRPT.

- **Secondary finding — small-workload scaling:** On small traces (<300 requests, 2
  servers), all preemptive disciplines produce identical results since queue depth is
  low and aging dispatch rarely reorders at α=0.01. The decoupled vs SRPT gap only
  emerges at scale (5,880 requests, 4 servers).

- **Decision:** FRONTIER IMPROVEMENT (simulator). Decoupled hybrid closes the most
  important gap in the scheduler portfolio: it delivers >2.5× the goodput improvement
  of both Aging-SRTF and the naive Hybrid, while providing meaningful starvation
  protection (long_p99 +132% vs +223% for pure SRPT). Best positioning: +184.5%
  goodput/$ with +97.9% short_p90 improvement and +132.3% long_p99 regression.
  Implementation shadow-only (enabled=False), simulator result only.
  See `docs/DECOUPLED_HYBRID_BACKTEST_RESULTS.md`.

- **Next recommended direction:**
  1. **Alpha sweep:** Profile α ∈ {0.001, 0.005, 0.01, 0.05} to find the goodput/
     long_p99 Pareto front — expect α=0.001 → near-SRPT goodput (+315%+) with mild
     starvation reduction; α=0.05 → aging_srtf-like (+70%) with strong starvation
     protection.
  2. **Wire into serving runtime:** Connect decoupled_hybrid (α=0.01) to the live
     serving runtime path driven by `OutputLengthForecastBundle.p50` as the predicted-
     tokens prior. This makes the 30%-CV prediction-error robustness study the critical
     next test.
  3. **Full BurstGPT cross-validation:** Run `run_burstgpt_decoupled_hybrid_backtest()`
     on the full 1.4M-row BurstGPT dataset (the 51-request fixture is too small for
     meaningful queue dynamics).

### Run 2026-06-20-k — HYBRID AGING+PREEMPTIVE SRPT SERVING-QUEUE SIMULATOR

**Goal:** Combine aging's anti-starvation guarantee with SRPT preemption mechanics
into a single discipline: `key(r,t) = remaining_s / (1 + α·accumulated_wait_s)`.
Hypothesis: achieves near-SRPT goodput/$ (+300% vs FIFO) while capping long_p99
regression closer to Aging-SRTF levels (+113% vs FIFO), because accumulated wait
progressively reduces a request's effective key toward zero.

- **Phase 3 (research — 3 papers):**
  1. **FastServe** (USENIX NSDI '26) — iteration-level preemptive MLFQ + starvation
     prevention for LLM serving; skip-join multi-level feedback queue avoids
     head-of-line blocking without full KV-cache eviction overhead.
  2. **Chimera** (arXiv:2603.22206, March 2026) — STJF with aging-based anti-starvation
     for multi-agent LLM serving; explicit aging factor in dispatch key.
  3. **SEK-SMOD** (arXiv:2510.25963, SIGMETRICS 2026) — first policy to provably
     outperform SRPT-k at all loads via strategic large-job re-prioritization.

- **Phase 4 (implementation):**
  - `aurelius/benchmarks/srtf_serving_backtest.py` extended:
    - `HYBRID_AGING_ALPHA_DEFAULT = 0.01` constant.
    - `_simulate_hybrid_aging_preemptive(requests, servers, aging_alpha)` — full
      discrete-event M/G/c simulator tracking `frozen_wait_s` (accumulated wait
      while in queue) per request, re-evaluating effective keys at dispatch time.
    - `simulate_queue(..., discipline="hybrid_aging_preemptive")` dispatch branch.
    - `HybridAgingPreemptiveReport` dataclass (5 disciplines + delta KPIs).
    - `_run_hybrid_backtest_on_trace()` shared helper.
    - `run_hybrid_aging_preemptive_backtest()` — Azure LLM 2024 public API.
    - `run_burstgpt_hybrid_backtest()` — BurstGPT cross-validation.
  - `tests/test_srtf_hybrid_backtest.py` (NEW) — 43 tests, 9 classes, all passing.

- **Phase 7 (benchmark results — public trace replay):**

  **Azure LLM 2024 (5,880 requests, ρ=0.85, SLA=10s, c=4):**
  | KPI | FIFO | Aging-SRTF (α=0.05) | SRPT-preemptive | **Hybrid (α=0.01)** |
  |---|---:|---:|---:|---:|
  | SLA-safe goodput/$ | 13,336 | 22,768 (+70.7%) | 56,311 (+322.2%) | **21,899 (+64.2%)** |
  | short_p90 response (s) | 696.16 | 152.61 | 1.89 | **169.26** |
  | short_p90 improvement | — | +78.1% | +99.73% | **+75.7%** |
  | long_p99 response (s) | 733.55 | 1,568.16 | 2,372.56 | **1,550.23** |
  | long_p99 regression | — | +113.8% | +223.4% | **+111.3%** |
  | mean_response_s | 343.89 | 183.06 | 129.58 | **187.02** |

  **BurstGPT (51-request fixture, ρ=0.85, SLA=30s, c=4):**
  | KPI | FIFO | SRPT-preemptive | **Hybrid (α=0.01)** |
  |---|---:|---:|---:|
  | SLA-safe goodput/$ | 70,975 | 67,754 (−4.5%) | **67,754 (−4.5%)** |
  | short_p90 improvement | — | +67.5% | **+67.5%** |
  | long_p99 regression | — | +16.0% | **+16.1%** |

- **Key finding — α=0.01 makes hybrid behave like Aging-SRTF, not SRPT:**
  At α=0.01, the dispatch key `remaining_s / (1 + α·total_wait)` actively promotes
  long-waiting requests over shorter but fresher arrivals. The "flip point" where
  a waiting request with remaining_s=5s beats a fresh 3s arrival: `total_wait >
  (5/3−1)/0.01 = 66.7s`. On the Azure trace at ρ=0.85, a meaningful fraction of
  requests wait >66.7s (heavy tail), causing systematic short-request bypassing.
  This makes hybrid goodput (~21,899) nearly equal to Aging-SRTF (~22,768), not SRPT (~56,311).

- **Anti-starvation IS working:** Hybrid long_p99=1,550s vs SRPT long_p99=2,373s →
  **34.7% improvement** in long_p99. The aging preemption key correctly accumulates
  protection for long-waiting requests, reducing preemption by new arrivals.

- **Decision:** FRONTIER IMPROVEMENT (simulator) — partial success. Hybrid achieves
  meaningful long_p99 improvement vs SRPT (−35%) and reasonable goodput vs FIFO
  (+64.2%). However, the goodput/SRPT ratio is only 0.39, not the near-SRPT parity
  originally hypothesized. The root cause is the unified aging key for both
  preemption and dispatch decisions.

- **Next recommended direction (run 2026-06-20-l):**
  **Decoupled Hybrid:** use `remaining_s` for preemption (preserves SRPT goodput
  benefit) and `remaining_s / (1 + α·total_wait)` for dispatch only (anti-starvation).
  Expected: SRPT-level goodput ($+322%) with Aging-SRTF-level long_p99 (+113% vs FIFO).
  Alternative: try much smaller α=0.001 to preserve SRPT character while adding weak
  aging protection.

### Run 2026-06-20-j — PREEMPTIVE SRPT SERVING-QUEUE SIMULATOR

**Goal:** Address the theoretical starvation risk of non-preemptive SRTF by
adding a preemptive SRPT discipline to the serving-queue simulator.  In
preemptive SRPT, when a shorter job arrives it immediately reclaims the server
running the longest-remaining job.  The preempted job re-enters the waiting
queue with its current remaining service time, guaranteeing monotonic forward
progress (remaining service never increases).

- **Phase 1 (audit):** Read ROADMAP, GAP_ANALYSIS.  Confirmed #2 highest-EV
  opportunity: preemptive SRPT variant.  SRTF perfect shows +323.5% goodput/$
  vs FIFO but +223.5% long_p99 regression (starvation).  Aging-SRTF bounds it
  to +113.8% at the cost of goodput (α=0.01: +70.7% vs FIFO).  Preemptive SRPT
  should deliver near-SRTF goodput + best-possible short_p90 + bounded starvation.

- **Phase 3 (research — 3 papers):**
  1. **TRAIL** (arXiv:2410.01035, ICLR 2025) — near-SRTF performance via
     embedding-based SPRPT with limited preemptions.
  2. **FlowPrefill** (arXiv:2602.16603, Feb 2026) — operator-level preemption
     blueprint for SLO-aware LLM serving; decouples preemption granularity from
     prefill scheduling.
  3. **SRPT Multiserver** (arXiv:1805.07686, 2018) — SRPT server-selection rule
     for M/G/k (preempt the server with the longest remaining service when a
     shorter job arrives).

- **Phase 4 (implementation):**
  - `aurelius/benchmarks/srtf_serving_backtest.py` extended:
    - `_simulate_srpt_preemptive(requests, servers)` — discrete-event M/G/c
      SRPT preemptive simulator with per-server version counters (stale-event
      detection), remaining-service tracking, and heapq-based waiting queue.
    - `simulate_queue(..., discipline="srpt_preemptive")` dispatch branch.
    - `SRTFPreemptiveReport` dataclass (all 4 disciplines + delta KPIs).
    - `_run_preemptive_backtest_on_trace()` — shared backtest helper.
    - `run_srpt_preemptive_backtest()` — Azure LLM 2024 public function.
    - `run_burstgpt_srpt_preemptive_backtest()` — BurstGPT cross-validation.
  - `tests/test_srtf_preemptive_backtest.py` (NEW) — 42 tests, 9 classes, all passing.

- **Phase 7 (benchmark results — public trace replay):**

  **Azure LLM 2024 (5,880 requests, ρ=0.85, SLA=10s, c=4, time_warp=21.95):**
  | KPI | FIFO | SRTF-perfect | Aging-SRTF (α=0.01) | SRPT Preemptive |
  |---|---:|---:|---:|---:|
  | SLA-safe goodput/$ | 13,336 | 56,481 (+323.5%) | 22,768 (+70.7%) | **56,311 (+322.2%)** |
  | short_p90 response (s) | 696.16 | 3.03 | 152.61 | **1.89** |
  | short_p90 improvement | — | +99.57% | +78.08% | **+99.73%** |
  | long_p99 response (s) | 733.55 | 2,373.09 | 1,568.16 | **2,372.56** |
  | long_p99 regression | — | +223.5% | +113.8% | **+223.4%** |
  | mean_response_s | 343.89 | 129.89 | 183.06 | **129.58** |
  | p50_response_s | 342.20 | 2.71 | 58.49 | **2.09** |

  **BurstGPT (51-request fixture, ρ=0.85, SLA=30s, c=4):**
  | KPI | FIFO | SRTF-perfect | SRPT Preemptive |
  |---|---:|---:|---:|
  | SLA-safe goodput/$ | 70,975 | 67,754 (−4.5%) | **67,754 (−4.5%)** |
  | short_p90 improvement | — | +56.5% | **+67.5%** |
  | long_p99 regression | — | +10.8% | **+16.0%** |

- **Decision:** FRONTIER IMPROVEMENT (simulator).  SRPT preemptive achieves
  near-SRTF goodput (+322.2% vs +323.5%) with the best short_p90 across all
  four disciplines (+99.73% vs FIFO).  Theoretical anti-starvation guarantee
  (monotonic remaining-service decrease) confirmed in implementation; empirically,
  long_p99 regression (+223.4%) matches SRTF at ρ=0.85 because high short-job
  arrival rate keeps long jobs continuously preempted.

- **Empirical finding — goodput vs Aging-SRTF:**  SRPT preemptive (+322.2%)
  dramatically outperforms Aging-SRTF (+70.7%) on Azure LLM 2024.  The preemptive
  variant is the better choice when goodput/$ is the primary KPI; Aging-SRTF is
  preferable when long_p99 latency SLA must also be bounded.

- **Next recommended direction:**
  1. Hybrid Aging+Preemptive SRPT: use remaining_s / (1 + α·wait_s) as the
     preemption key, combining anti-starvation aging with preemptive scheduling.
  2. Cross-validate on full BurstGPT (1.4M rows) using run_burstgpt_srpt_preemptive_backtest().
  3. Wire SRPT preemptive into the serving runtime path with OutputLengthForecastBundle.p50
     as the predicted_tokens prior.

### Run 2026-06-20-i — AGING-SRTF ANTI-STARVATION GUARD + BURSTGPT CROSS-VALIDATION

**Goal:** Address the #1 production barrier from run -g: long-request starvation
under non-preemptive SRTF (p99: 733s → 2373s, +223.5% regression vs FIFO). Add
the aging-SRTF discipline (key(r,t) = predicted_tokens / (1 + α·wait_s)) to
bound starvation while preserving as much of the SRTF short-request benefit as
possible. Cross-validate on BurstGPT for generalization.

- **Phase 1 (audit):** Read ROADMAP, GAP_ANALYSIS. Confirmed #1 opportunity:
  implement aging guard for SRTF. All three shadow modules (WorkloadAdmissionGate,
  OutputLengthForecastBundle, GpuPlacementScorer) remain unconnected to the
  aggregate LLM benchmark path per run -h's finding.

- **Phase 3 (research — 3 new papers):**
  1. **Astraea** (arXiv:2512.14142, Dec 2025) — state-aware scheduling for LLM
     agents with aging-based starvation prevention: a request in the
     lowest-priority queue is promoted to highest priority when its response-ratio
     exceeds a predefined aging threshold. Maps directly to our aging key formula.
  2. **Equinox** (arXiv:2508.16646, Aug 2025) — holistic fair scheduling for LLMs
     via a dual-counter framework: User Fairness Counter (latency, weighted tokens)
     + Resource Fairness Counter (throughput, GPU utilization). MoPE predictions
     enable proactive fairness-aware scheduling with up to 1.3× throughput, 60%
     latency improvement. Validates the aging-SRTF direction.
  3. **FlowPrefill** (arXiv:2602.16603, Feb 2026) — decouples preemption from
     prefill scheduling granularity to mitigate head-of-line blocking. Operator-
     level preemption enables SLO-aware prioritization for newly arriving high-
     priority requests. Maps to the preemptive SRPT variant needed to eliminate
     (vs bound) long-tail starvation.
  Also found: arXiv:2601.22996 (Competitive Non-Clairvoyant KV-Cache Scheduling,
  Feng et al. Jan 2026 — GSA with geometric phase structure and competitive ratio
  61.92; maps to admission gate memory management).

- **Phase 4 (implementation):**
  - `aurelius/benchmarks/srtf_serving_backtest.py` extended:
    - `AGING_ALPHA_DEFAULT = 0.05`; `DEFAULT_BURSTGPT_FIXTURE`; `DEFAULT_BURSTGPT_SLA_S = 30.0`
    - `simulate_queue(discipline="aging_srtf", aging_alpha)` — O(|ready|) dispatch,
      re-evaluates effective key for all waiting requests at dispatch time t.
    - `_summarize()` extended: adds `long_p90_response_s`, `long_p99_response_s`.
    - `load_burstgpt_serving_requests()` — BurstGPT CSV loader.
    - `SRTFAgingReport` — FIFO / SRTF-perfect / aging_SRTF comparison dataclass.
    - `_run_aging_backtest_on_trace()` — internal shared helper.
    - `run_aging_srtf_backtest()` — Azure LLM 2024 multi-discipline benchmark.
    - `run_burstgpt_aging_backtest()` — BurstGPT cross-validation benchmark.
  - `tests/test_srtf_aging_backtest.py` (NEW) — 37 tests, all passing.

- **Phase 7 (benchmark results — public trace replay):**

  **Azure LLM 2024 (5880 requests, ρ=0.85, SLA=10s, c=4):**
  | KPI | FIFO | SRTF-perfect | Aging-SRTF (α=0.05) |
  |---|---:|---:|---:|
  | SLA-safe goodput/$ | 13,336 | 56,481 (+323.5%) | 16,317 (+22.4%) |
  | short_p90 response (s) | 696.16 | 3.03 (+99.6% impr.) | 204.02 (+70.7% impr.) |
  | long_p99 response (s) | 733.55 | 2,373 (+223.5% regr.) | 1,479 (+101.7% regr.) |

  **Alpha sensitivity (α=0.01 sweet spot):** +70.7% gp/$ vs FIFO, 49% starvation
  reduction (long_p99: +113.8% vs FIFO rather than +223.5%), short_p90 +78.1%.

  **BurstGPT (51-request sample):** Sample too small for starvation characterization;
  SRTF direction confirmed (+56.6% short_p90). Full 1.4M-row dataset needed.

- **Decision:** FRONTIER IMPROVEMENT (simulator). The aging-SRTF discipline
  quantifies the full fairness–efficiency trade-off curve for the first time on a
  real LLM serving trace. At α=0.01: +70.7% goodput/$ vs FIFO, 49% starvation
  reduction. Implementation is simulator-only (shadow); not wired into serving runtime.
  See `docs/SRTF_AGING_BACKTEST_RESULTS.md`.

- **Run category:** FRONTIER IMPROVEMENT (serving-queue simulator; both the
  quantification of the trade-off curve and the aging-SRTF discipline are new).

- **Next recommended direction:**
  1. Add preemptive SRPT variant: when shorter job arrives, preempt at operator
     boundary (FlowPrefill-style) — eliminates rather than bounds starvation.
  2. Cross-validate on full BurstGPT (1.4M rows) using run_burstgpt_aging_backtest().
  3. Wire aging_srtf (α=0.01) into the serving runtime path driven by
     OutputLengthForecastBundle.p50 as the predicted_tokens prior.

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
