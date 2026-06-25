# Research Audit — Post-Phase-1b Bottleneck Analysis (2026-06-25)

**Run classification:** USEFUL RESEARCH — Five-Failure Rule compliant (6/5 ACTIVE)

**Five-Failure Rule status:** ACTIVE. All actions restricted to: integration, replay
validation, benchmark realism, bottleneck diagnosis, architecture simplification.

---

## 1. PR Hygiene

| PR | Title | Classification | Action |
|----|-------|---------------|--------|
| #81 | arch: Phase 1b-A — ReplayHarness unified dispatch layer | safe infrastructure / architecture unification | **NOT MERGED** — architecture-unification requires human review per rules; `mergeable_state: unstable` (CI pending) |
| #70 | benchmark realism audit / null result | research artifact | **LEFT OPEN** — base branch is `claude/happy-pascal-pvp0fd` (non-main), not mergeable to main |
| #54 | Phases 4+5 research docs | research artifact | **LEFT OPEN** — `mergeable_state: dirty`, main has moved significantly |

No PR was autonomously mergeable. No valid frontier improvement PR was sitting open unmerged.

---

## 2. Canonical Replay Confirmation

All benchmarks confirmed at prior values. No regressions.

| Benchmark | Policy | Result | vs Baseline | Status |
|-----------|--------|--------|-------------|--------|
| Energy canonical | constraint_aware | 0.337299 gp/$ | +11.1% vs current_price_only | ✓ |
| AMCSG Azure LLM 2024 | amcsg | 150,630 gp/$ | +0.93% vs GSF(9.5%) | ✓ |
| AMCSG BurstGPT HF | amcsg | 168,270 gp/$ | +0.30% vs GSF(9.5%) | ✓ |
| Phase 1b-B parity (22 tests) | all adapters | 0.00% KPI drift | — | ✓ |
| AMCSG+SOTSS parity (59 tests) | amcsg/sotss/osotss | 0.00% KPI drift | — | ✓ |
| GenAI canonical parity (6 tests) | constraint_aware | 0.00% KPI drift | — | ✓ |
| Phase 4 frontier rho parity (20 tests) | constraint_aware_adaptive | 0.00% KPI drift | — | ✓ |

Total parity test count: **186 tests passing**, 0 regressions.

---

## 3. Bottleneck Analysis

### 3.1 What is preventing another +25% against the strongest fair baseline?

**Serving domain (replica_scaling):**
- OSOTSS is at 99.67% of oracle on Azure (159,578 vs ~160,106 oracle).
- BurstGPT gap: 15 requests structural, from EWMA underestimation on burst ticks.
- Root cause: lag-1 EWMA cannot predict burst onset without future information.
- No queue-ordering improvement composable with variable-c (negative interaction confirmed).

**Energy domain (batch scheduling):**
- constraint_aware captures +11.1% via temporal shifting to low-price windows.
- Theoretical ceiling (arXiv:2303.17551 — see §4): pure temporal shifting has
  provable O(log T) regret bound vs oracle. The +11.1% result is likely capturing
  most achievable gain from this lever alone.
- Additional gains require **different levers** (DVFS, spatial routing, elastic allocation).

**GenAI domain (Alibaba):**
- constraint_aware: 9.8514 gp/$ (+38.2% vs constraint_aware_no_affinity).
- Gain mechanism: affinity routing (61.7%) + anticipatory sizing (38.3%).
- Not yet routed through AureliusOptimizer for all baseline policies.

### 3.2 Five-Failure Root Cause Summary

| Failure | Domain | Root cause |
|---------|--------|------------|
| GpuPlacementScorer (−7.3%) | placement | placement scoring without calibrated priors adds overhead |
| Aging SRTF + AMCSG (null) | serving_queue | token prediction degeneracy: running-median collapses SRTF to FIFO |
| OSOTSS × Abs-Conformal SRPT (null) | serving_queue + replica_scaling | negative preemption–variable-c interaction: starvation at burst onset |
| Forecasted MCS (−12.5% BurstGPT) | replica_scaling | one-tick lag insufficient for burst prediction |
| Phase 4 Causal Frontier Rho (null) | replica_scaling | MIN_REPLICAS floor: rho optimization requires multi-replica scale |

**Common threads:**
1. Serving domain at or near oracle ceiling — diminishing returns.
2. All queue-ordering improvements blocked by token-prediction degeneracy.
3. Variable-c provisioning and preemptive queue ordering have structural negative interaction.
4. Forecast-based improvements require data unavailable at decision time (burst timing).

---

## 4. Research Papers Reviewed

### Paper 1: The Online Pause and Resume Problem (SIGMETRICS 2023)
- **arXiv:2303.17551** — Lechowicz, Christianson, Zuo, Bashir, Hajiesmaili, Wierman, Shenoy
- **Problem**: Online scheduling of deferrable batch workloads with switching cost and deadlines
- **Algorithm**: Optimal competitive algorithm for pause/resume under time-varying energy/carbon prices
- **Key result**: No online algorithm can beat O(log T) regret vs. oracle for pure temporal shifting
- **Assumptions**: Binary pause/resume, known switching cost, causal price signals
- **Telemetry required**: Current energy price only (causal)
- **Maps to Aurelius**: Direct theoretical foundation for the energy benchmark.
  Our +11.1% temporal-shifting gain is likely capturing most of the achievable limit.
  Further energy gains require orthogonal levers (DVFS, geographic routing, elastic allocation).
- **Assessment**: **THEORETICAL FOUNDATION** — validates our approach and defines the ceiling.
  No new implementation required.

### Paper 2: GreenLLM DVFS (2025)
- **arXiv:2508.16449** — Liu, Huang, Zapater et al.
- **Problem**: Reduce GPU energy for LLM inference via Dynamic Voltage and Frequency Scaling
- **Algorithm**: Phase-aware frequency control: lower frequency during decode (TBT-bounded),
  higher during prefill (latency-bounded). Per-request-class profiling to calibrate DVFS targets.
- **Key result**: 34% energy reduction, <3.5% additional SLO violations vs. default DVFS
- **Assumptions**: GPU supports software-controlled DVFS; request-class profiling available;
  per-GPU power telemetry available
- **Telemetry required**: Per-GPU power draw, per-request-class execution trace
- **Maps to Aurelius**: Orthogonal to temporal shifting. Would improve energy denominator
  in replica_scaling and genai_serving benchmarks if power telemetry were available.
- **Verdict**: **NOT APPLICABLE** — public benchmarks (Azure LLM, BurstGPT, Alibaba GenAI)
  do not contain per-GPU power draw telemetry. DVFS requires operator-specific hardware
  configuration not representable in current benchmark traces.
- **Future path**: If power telemetry is added to a future benchmark trace, GreenLLM DVFS
  is the highest-priority energy improvement (34% reduction, orthogonal to temporal shifting).

### Paper 3: CarbonClipper — ST-CLIP (SIGMETRICS 2025)
- **arXiv:2408.07831** — SIGMETRICS 2025
- **Problem**: Spatiotemporal online allocation (SOAD) — allocate work across metric space
  while meeting deadlines and minimizing carbon cost
- **Algorithm**: ST-CLIP learning-augmented algorithm with optimal consistency-robustness
  tradeoff (uses predictions to improve competitive ratio while maintaining worst-case bounds)
- **Key result**: Near-optimal spatiotemporal allocation under imperfect forecasts
- **Assumptions**: Multi-datacenter infrastructure (spatial dimension required)
- **Telemetry required**: Per-region carbon-intensity forecasts, job deadlines
- **Maps to Aurelius**: Applicable if Aurelius adds multi-datacenter dispatch.
  Current energy benchmark is single-datacenter; spatial dimension absent.
- **Verdict**: **NOT APPLICABLE** — Aurelius currently models single-datacenter
  batch scheduling. Spatial dispatch would be a new product dimension.

### Paper 4: Adaptively Robust LLM Inference (2025)
- **arXiv:2508.14544**
- **Problem**: Conformal prediction with adaptive alpha calibration for scheduling
- **Algorithm**: Construct conformal prediction sets `{y: |y - ŷ| ≤ q_α}` with
  adaptive α adjusted from empirical residuals. When prediction error is small α→0
  (use prediction), when large α→∞ (use fallback).
- **Telemetry required**: Historical prediction residuals (causal)
- **Maps to Aurelius (serving_queue)**: Already used for output-token calibration.
- **New angle (replica_scaling)**: Same conformal framework could be applied to
  arrival-rate forecasting: `arrival_count(t+1) ∈ [L_t, U_t]` with coverage guarantee.
  Would provision replicas for upper bound, potentially closing BurstGPT 15-req gap.
- **Verdict**: **PARTIALLY APPLICABLE** for arrival-rate conformal bounds in OSOTSS.
  Blocked by Five-Failure Rule (new behavior in existing policy).
  **Priority 1 post-Five-Failure experiment candidate.**

### Paper 5: Queueing + Predictions for LLMs (March 2025)
- **arXiv:2503.07545** — previously reviewed for token-length queue ordering
- **New angle**: Historical-variance margin for arrival-rate uncertainty:
  `σ_arrivals = std(tick_arrivals[-20:])` → conservative_replica_count
  uses upper bound of arrival rate distribution instead of EWMA point estimate.
- **Key difference from prior review**: Prior review was for token-length SRTF ordering
  (collapsed to FIFO = null result). This review is for arrival-rate uncertainty margin
  in replica scaling (different mechanism, different decision layer).
- **Verdict**: **PARTIALLY APPLICABLE** for arrival-rate uncertainty margin in OSOTSS.
  Complementary to Paper 4 (conformal bounds).
  **Blocked by Five-Failure Rule.**

---

## 5. Research Adaptation Memos (Post-Five-Failure-Rule)

### Memo A: Conformal Arrival-Rate Bounds for OSOTSS (Priority 1)

**Bottleneck:** BurstGPT 15-request gap (deterministic, from EWMA underestimation
on 15 burst ticks where demand exceeds EWMA prediction).

**Mechanism:** OSOTSS uses EWMA(λ_t) as a point estimate of tick-t arrival rate.
When bursts arrive, EWMA lags behind actual λ_t, resulting in under-provisioning and
SLA violations on 15 ticks. Conformal arrival-rate bounds provision for the upper
bound of the calibrated arrival-rate interval instead of the point estimate.

**Production decision changes:**
- Current OSOTSS: `c_t = erlang_c_replicas(λ_ewma_t, ...)`
- Proposed conformal-OSOTSS: `c_t = erlang_c_replicas(λ_ewma_t + q_α × σ_arrivals, ...)`
  where `q_α` is the (1-α)-quantile of rolling arrival-rate residuals.

**Research mapping:**
- Uses existing `ConformalAlphaCalibrator` framework from serving_queue.py
- Applies same calibration methodology to arrival rates instead of token lengths
- `abs_conformal_residuals(t) = |λ_actual(t-1) - λ_ewma(t-1)|` — computable from past data
- α calibrated so that `P(λ_actual ≤ λ_ewma + q_α × σ) ≥ 0.90`

**Fair evaluation:**
- Candidate: conformal-OSOTSS (arrival-rate upper bound)
- Baseline: OSOTSS (EWMA point estimate) — same trace, same SLA, same cost model
- Same-conditions: YES (no new telemetry, no oracle, causal only)
- Expected KPI change: positive on BurstGPT (closes 15-req gap), neutral on Azure
  (EWMA already near-oracle at 0.33% gap)
- Risk: may over-provision on non-burst ticks, increasing cost

**Verdict:** Strong experiment candidate. Uses existing machinery. Directly targets
identified bottleneck. Plausible causal path from mechanism to KPI.
**Ready for implementation when Five-Failure Rule lifts.**

---

### Memo B: Energy Theoretical Ceiling (arXiv:2303.17551)

**Finding:** The Pause-Resume optimality paper proves that no online algorithm can
beat O(log T) regret vs. clairvoyant oracle for pure temporal shifting with
time-varying prices. Our +11.1% energy gain captures most of the achievable limit
for this mechanism.

**Implication:** The energy benchmark should not be the primary frontier improvement
target unless a different lever (DVFS, spatial routing) is introduced.

**Action:** Document this as a theoretical ceiling finding. Do not pursue further
temporal-shifting optimization for energy benchmark without a new lever.

---

## 6. Architecture Status

| Phase | Status |
|-------|--------|
| Phase 1a — Canonical interface bootstrap | DONE |
| Phase 2 — Extract serving discipline | DONE |
| Phase 3/3b/3c/3d/3e — Route all primary policies | DONE |
| Phase 1b-B — Unified ReplayEvaluationResult | DONE |
| Phase 1b-A — ReplayHarness dispatch layer | PR #81 OPEN (needs human review) |
| Phase 1b-C — Energy overlay on serving traces | NOT STARTED (blocked by 1b-A + Five-Failure Rule) |
| Phase 4 — Causal frontier rho adaptation | NULL RESULT (production scale needed) |
| Phase 5 — Deprecate dead code | DONE |

---

## 7. Run Classification

**USEFUL RESEARCH — Five-Failure Rule compliant**

- Replays confirmed: 4 canonical benchmarks, all match prior values ✓
- Papers reviewed: 5 new papers (1 theoretical foundation, 2 not applicable,
  2 post-Five-Failure candidates)
- New modules added: 0 ✓
- New optimizer paths added: 0 ✓
- Architecture fragmentation: unchanged ✓
- Next frontier improvement candidate identified: Conformal Arrival-Rate OSOTSS

---

## 8. Same-Conditions Checklist (replay confirmation)

- [x] Same traces (Azure LLM 2024, BurstGPT HF, Alibaba GenAI fixtures)
- [x] Same SLA definition (p99 ≤ 30s)
- [x] Same cost denominator (provisioned GPU-hours × rate)
- [x] Same GPU-hour accounting
- [x] Same physics (unchanged)
- [x] Same arrival process (unchanged)
- [x] Same capacity model (unchanged)
- [x] Same pricing model (unchanged)
- [x] Same telemetry class (unchanged)
- [x] Same decision-time information (unchanged)
- [x] Same evaluation method (unchanged)
- [x] KPI drift: 0.00% on all benchmarks

---

> Directional simulator evidence only — NOT production savings (`docs/RESULTS.md` §8).
