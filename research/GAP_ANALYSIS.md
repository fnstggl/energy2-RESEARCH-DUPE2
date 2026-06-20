# Aurelius Gap Analysis

> **Updated every run.** This document answers the 13 standard gap-analysis
> questions for each run, then ranks future opportunities by expected value.
>
> **Binding rules:** All numbers are simulator / public-trace directional.
> No production claim. `docs/RESULTS.md` §8 production-claim gate not met.

---

## Run 2026-06-20-h — module integration + economic validation

This run pivoted from building shadow modules to **validating** the three
existing ones on real public replay (`WorkloadAdmissionGate`,
`OutputLengthForecastBundle`, `GpuPlacementScorer`). Artifacts:
`research/results/{baseline,module_integration}_public_backtest_2026-06-20.*`,
`research/PUBLIC_BACKTEST_COMMANDS.md`. Key answers:

- **Q1 (biggest limit):** The decision-surface mismatch. The public LLM-serving
  benchmark (Azure 2024 / BurstGPT) is an *aggregate per-tick autoscaling*
  replay; it exposes a provisioning decision, not the per-request placement /
  ordering / GPU-routing decisions the three modules were built for.
- **Q3 (weakest):** `OutputLengthForecastBundle` in the *aggregate* replay — the
  autoscaler already reads the realized per-tick mean (clairvoyant), so a
  forecast can only under-/over-size. Measured **−7…−11%** goodput/$ on BurstGPT.
  (Consistent with run -g: the SRTF benefit lives in a *per-request* serving
  queue, NOT the aggregate autoscaler — this run independently confirms the
  module has no lever in the aggregate path, exactly the gap run -g exploits.)
- **Q4 (suboptimal decisions):** None of the three modules improved any public
  KPI on the aggregate replay. `WorkloadAdmissionGate` neutral (baseline already
  SLA-safe); `GpuPlacementScorer` moves the routing proxy (+54.7pp) but regresses
  real latency_critical goodput/$ (−7.3%).
- **Q11 (benchmark weakness):** Azure-2024 full week is SAS-gated (401); the
  5,880-row sample yields only 11–32 ticks at saturating scales → noisy. BurstGPT
  (real 1.43M trace) is the robust evidence.
- **Q13 (next):** Do not enable the three modules in the aggregate path. The
  output-length SRTF value belongs in the *per-request serving queue* run -g
  built — pursue that, not aggregate-replay sizing.

**Decision: INFRASTRUCTURE ONLY** — backtest infra + report merged; no runtime
decision change; the three modules stay `enabled=False`.

---

## Run 2026-06-20-g

### Q1. What currently limits Aurelius most?

**The proven SRTF value lives in a layer Aurelius does not yet schedule.** Run
-g proved (on the real Azure LLM 2024 queue) that shortest-predicted-job-first
cuts short-request p90 latency by −99.6% and lifts SLA-safe goodput/$ by +323%
vs FIFO — but only in a request-level serving queue. The merged batch
`JobScheduler` sort key (run -f) is inert for this (no queue-wait semantics),
and the serving path has no per-request ordering hook yet. Wiring SRTF into the
serving runtime (with an anti-starvation guard) is the gap.

**Secondary:** long-request starvation under non-preemptive SJF (p99 733s →
2189s) needs an aging/preemption mitigation before any runtime use.

### Q2. What theoretically offers the largest gain?

**SRTF/SPRPT ordering in the serving request queue.** Quantified, not
hypothetical: +252–324% SLA-safe goodput/$ across ρ∈{0.80,0.85,0.92} on the
real trace, robust to a 30%-CV forecast prior. The remaining work is exposing
the ordering hook in the serving path + an aging guard.

### Q3. Which forecasts are weakest?

1. **Output length p50 as the live SRTF prior** — the serving backtest used a
   simulated prior; the real `OutputLengthForecastBundle.p50` must drive the
   ordering for the value to transfer. (Robustness is encouraging: 30%-CV noise
   barely dented the gain.)
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving-queue ordering is FIFO** — the single largest measured gap
   (request-level SRTF not yet in the serving path).
2. **No anti-starvation aging** — needed before SRTF can go live.
3. **GPU penalty calibration** — heuristic floor/ceil (unchanged).

### Q5. Which workloads benefit least?

**Batch / energy-shifting workloads** — confirmed again. The batch scheduler has
no queue contention to exploit; SRTF is a serving-queue phenomenon. Light-load
serving (ρ=0.10) also benefits little — the win scales with contention.

### Q6. Which research direction appears strongest?

**Serving-path SRTF + aging guard**, then SRPT (preemptive) to recover the
long-tail. The simulator is built and the value is quantified; this is now an
implementation task, not a research question.

### Q7. What is the shortest path to another +10% gain?

1. Expose an ordering hook in the serving path keyed on
   `OutputLengthForecastBundle.p50`.
2. Add an aging term (a request's effective key decreases with wait time) so no
   request waits beyond a TTL — bounds the long-tail regression.
3. Re-run `srtf_serving_backtest` end-to-end with the live prior.

### Q8. What is the shortest path to another +50% gain?

The serving-queue SRTF result already shows >+250% goodput/$ vs FIFO in
simulation; even discounting heavily for the FIFO-not-SLA-aware baseline and
regime sensitivity, realizing a fraction of it in the serving runtime is the
highest-leverage move available.

### Q9. What would need to be true to achieve +300%?

The +300% target is vs SLA-aware (not FIFO). The serving SRTF result is vs FIFO,
so it is **not** a +300%-vs-SLA-aware claim. Reaching the aspirational target
still requires the full stack: live output-length prior, serving-path SRTF with
aging, heterogeneous GPU placement on serving traces, measured queue-wait
labels, agentic PDGraph, joint carbon+placement, pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **Service-time model** `TTFT_BASE + tokens·TPOT` — a documented proxy; real
   continuous-batching throughput is load-dependent (batch size effects) and may
   compress the short/long gap.
2. **Time-warp realism** — the public sample is downsampled; warping to ρ=0.85
   preserves shape but not absolute burst micro-structure.
3. **Non-preemptive SJF is the right discipline** — SRPT (preemptive) or a
   hybrid may dominate by recovering the long-tail; not yet measured.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — weaker than SLA-aware; the headline % is vs FIFO.
2. **Single trace (Azure 2024)** — BurstGPT replay through the same simulator
   would cross-validate (BurstGPT carries real request+response tokens too).
3. **No preemption modeled** — the long-tail cost may be overstated relative to
   a preemptive implementation.

### Q12. Which public datasets should be added?

1. **BurstGPT through the serving simulator** — cross-trace validation of the
   SRTF serving result (real request/response tokens available).
2. **Vidur profiling CSVs** — load-dependent service-time calibration.
3. **ShareGPT** — output-length cross-dataset validation for the prior.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Expose SRTF/SPRPT ordering in the serving path driven by
   `OutputLengthForecastBundle.p50`, with an aging/preemption guard.
2. Add a preemptive SRPT variant to `srtf_serving_backtest` and measure the
   long-tail recovery vs the non-preemptive starvation cost.

**Short-term (2–3 runs):**
3. Cross-validate on BurstGPT through the same simulator.
4. Wire the live output-length prior and re-run end-to-end.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Serving-path SRTF/SPRPT + aging guard | High | Medium | **Value quantified [run -g]** (+323% goodput/$ vs FIFO, Azure 2024 sim); not yet in serving runtime |
| 2 | Preemptive SRPT variant + long-tail recovery measurement | High | Low | Simulator built [run -g]; add preemption |
| 3 | Wire OutputLengthForecastBundle.p50 as live SRTF prior | High | Low | Infrastructure built (shadow) |
| 4 | GPU routing on LLM serving trace (TTFT binding) | Medium | Low | Benchmarked on energy trace [run -f] |
| 5 | Admission gate → cluster simulator integration | Medium | Medium | Implemented (unconnected) |
| 6 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 7 | Mooncake trace ingestion (KV prefix reuse) | Low-Med | Low | Not started |
| 8 | Carbon-power MILP joint optimization | Medium | High effort | Not started |

---

## Run 2026-06-20-f

### Q1. What currently limits Aurelius most?

**SRTF scheduling not yet evaluated on LLM serving traces.** The sort key is
wired and backward-compatible, but the expected +32% p90 short-request gain
(arXiv:2604.06970) can only be measured on traces with queue contention (BurstGPT,
Azure LLM 2024) — not on the canonical 26-day energy-shifting trace where jobs
have no shared queue.

**Secondary:** GPU routing goodput/$ is negative on the canonical energy trace
(−0.14%) because H100 GPUs are in the highest-cost PJM energy region and the
TTFT improvement has no direct goodput/$ credit when no jobs miss deadlines.

### Q2. What theoretically offers the largest gain?

**SRTF evaluation on LLM serving traces** — sort key is wired; running BurstGPT
and Azure LLM 2024 with queue contention is the lowest-effort next step.
Expected: +15–32% p90 short-request goodput on serving traces.

**Second:** Wire `OutputLengthForecastBundle.p50` as the SRTF prior value
(replaces `runtime_hours × 500K tokens/hour` proxy with calibrated token estimate).

### Q3. Which forecasts are weakest?

1. **SRTF prior quality** — current prior uses `runtime_hours × SRTF_TOKENS_PER_HOUR`
   (rough proxy); calibrated `OutputLengthForecastBundle.p50` is built but not yet
   wired as the prior source.
2. **GPU-type-specific TTFT penalty calibration** — `penalty_floor/ceil` heuristic;
   not tuned from goodput/$ sensitivity on LLM serving traces.
3. **TTFT p99 tail** — still at baseline_fallback (67% fallback on time holdout).
4. **Queue wait** — derived proxy only; no real measured wait labels.

### Q4. Which optimizer decisions remain suboptimal?

1. **SRTF on LLM serving traces** — sort key is wired but the gain only
   materializes under queue contention; evaluation pending.
2. **Batch admission timing** — `WorkloadAdmissionGate` implemented but not
   connected to any trace replay.
3. **GPU penalty calibration** — heuristic floor/ceil; not from goodput data.

### Q5. Which workloads benefit least?

**Energy batch scheduling** — confirmed neutral for both SRTF (0%) and GPU routing
(−0.14%) on the 26-day canonical energy trace. Both features provide value only
under request-queue pressure (LLM serving workloads).

### Q6. Which research direction appears strongest?

**Evaluating SRTF on BurstGPT and Azure LLM 2024** — zero new implementation
required; the benchmark harness (`srtf_backtest.py`) is built. This is a run of
the existing code on a trace with queue contention.

### Q7. What is the shortest path to another +10% gain?

1. Run `srtf_backtest` on BurstGPT and Azure 2024 with `predicted_output_tokens`
   set from `num_predicted_output_tokens` or `runtime_hours` proxy.
2. If short requests are served first, p90 TTFT drops → more SLA-safe goodput/$.
Estimated complexity: 1 run of low scope (replay + result recording).

### Q8. What is the shortest path to another +50% gain?

1. SRTF on LLM serving traces (+15–32% directional).
2. Wire `OutputLengthForecastBundle.p50` as SRTF prior (better priors → larger gain).
3. Admission gate cluster simulator integration (+3–8% from KV overflow reduction).
Combined: +50% plausible within 2–3 runs on LLM serving traces.

### Q9. What would need to be true to achieve +300%?

Unchanged. Requires: accurate output length prediction, heterogeneous GPU
placement benchmarked on LLM serving traces (not energy trace), measured
queue-wait labels, agentic PDGraph, joint carbon + placement optimization,
pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **SRTF gain transfers from pure LLM queue to Aurelius's job model** — the
   canonical Job model uses `runtime_hours` as the service time, not token
   counts. On BurstGPT and Azure 2024 the proxy is reasonable, but the exact
   gain depends on how well `runtime_hours × SRTF_TOKENS_PER_HOUR` correlates
   with actual request service time.
2. **GPU routing direction flips on LLM trace** — the energy trace result
   (−0.14%) was driven by PJM energy prices. On BurstGPT (no energy shifting,
   synthetic prices), the TTFT improvement should dominate.
3. **No queue contention assumption on canonical energy trace** — the 26-day
   window is long enough for all jobs to find cheap slots independently. If a
   shorter window or higher job density was used, SRTF would show a delta.

### Q11. Which benchmark weaknesses exist?

1. **Canonical energy trace lacks queue contention** — SRTF and GPU routing
   benefits are hidden on this trace. LLM serving traces are the right vehicle.
2. **No per-region GPU-type labels in public LLM traces** — BurstGPT and Azure
   2024 lack GPU-type metadata. Synthetic assignment needed for GPU routing eval.
3. **SRTF prior is a proxy** — `runtime_hours × 500K` is rough; calibrated p50
   from `OutputLengthForecastBundle` would reduce proxy error.

### Q12. Which public datasets should be added?

1. **BurstGPT / Azure 2024 replay with synthetic GPU-type labels** — existing
   traces, no new data needed; synthetic assignment from CARA fleet composition.
2. **Vidur profiling CSVs** — measured kernel latency for penalty calibration.
3. **ShareGPT** — output token counts for length predictor cross-dataset validation.
4. **Mooncake FAST25 traces** — KV prefix reuse cross-validation (unchanged).

### Q13. What should be attempted next?

**Immediate (next run):**
1. Run `run_srtf_backtest()` adapted for BurstGPT or Azure LLM 2024 trace
   (where jobs share GPU time and queue contention is present).
2. Wire `OutputLengthForecastBundle.p50` as the `predicted_output_tokens` prior
   source to replace the `runtime_hours × SRTF_TOKENS_PER_HOUR` proxy.

**Short-term (2–3 runs):**
3. Wire `WorkloadAdmissionGate` into cluster simulator for Azure 2024 replay.
4. Evaluate GPU routing on BurstGPT / Azure 2024 where TTFT violations are
   the binding SLA constraint.
5. Vidur CSV ingestion for penalty calibration.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | SRTF on LLM serving traces (BurstGPT / Azure 2024) | High | Low effort | Sort key wired [run -f] — eval pending |
| 2 | Wire OutputLengthForecastBundle.p50 as SRTF prior | High | Low effort | Infrastructure built (shadow) |
| 3 | GPU routing on LLM serving trace (TTFT violation reduction) | Medium | Low effort | Wired [run -d], benchmarked [run -f] — eval on LLM trace pending |
| 4 | Admission gate → cluster simulator integration | Medium | Medium | Implemented (unconnected) |
| 5 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 6 | Vidur CSV ingestion for GPU penalty calibration | Low-Med | Low | Not started |
| 7 | Mooncake trace ingestion (KV prefix reuse) | Low-Med | Low | Not started |
| 8 | Carbon-power MILP joint optimization | Medium | High effort | Not started |
| 9 | Hermes PDGraph agentic routing | High | High effort | Not started |

---

## Run 2026-06-20-e

### Q1. What currently limits Aurelius most?

**Output length forecasting not yet driving scheduling decisions.** The
`OutputLengthForecastBundle` is implemented in shadow mode but its p50
predictor is not wired into the scheduler's greedy sort key. Without length
priors, all jobs are treated as equal priority in the request queue, losing
the SRTF-like gain of short-first ordering (+32% p90 per arXiv:2604.06970).

**Secondary:** The GPU routing benchmark (`run_gpu_routing_backtest()`) is
now fully instrumented. The canonical CSV files were believed absent (gitignored)
at run -e time; run -f discovered they ARE present (`data/caiso_us_west_dam.csv`
etc.) and ran the benchmark with real CAISO/PJM/ERCOT data — result: −0.14%
goodput/$ (energy-price-dominated; see run -f for root cause analysis).

### Q2. What theoretically offers the largest gain?

**Semi-clairvoyant SRTF scheduling via output length p50** — the next
single-run implementation that can produce a measurable delta. The
`OutputLengthForecastBundle` (run -b) is built; wiring `p50` into the
scheduler sort key is a 1–2 file change. Expected gain: +15–32% p90
short-request goodput on LLM-serving traces (arXiv:2604.06970).

**Second:** SRTF evaluation on LLM serving traces (BurstGPT / Azure 2024).
GPU routing on real canonical data was run in run -f (−0.14%, energy-price-dominated);
the LLM serving trace evaluation remains pending.

### Q3. Which forecasts are weakest?

1. **GPU-type-specific TTFT penalty calibration** — `penalty_floor=0.05` /
   `penalty_ceil=0.50` are heuristic constants not tuned from goodput/$ data.
   Vidur profiling CSVs would enable data-driven calibration.
2. **Output token length** — forecaster built; not yet driving scheduling.
3. **TTFT p99 tail** — still at baseline_fallback (67% fallback on time holdout).
4. **Queue wait** — derived proxy only; no real measured wait labels.

### Q4. Which optimizer decisions remain suboptimal?

1. **Request ordering without length priors** — greedy sort is by deadline/
   priority only; output length p50 not used as SRTF weight.
2. **Batch admission timing** — `WorkloadAdmissionGate` implemented but not
   connected to any trace replay.
3. **GPU penalty calibration** — heuristic floor/ceil; not from goodput data.

### Q5. Which workloads benefit least?

**GPU packing / training scheduling** — unchanged. The GPU placement scorer
applies only to `latency_critical` LLM-serving jobs. Training / packing
workloads are unaffected by all new infrastructure in runs -c through -e.

### Q6. Which research direction appears strongest?

**SRTF-like scheduling via output token length priors** is the highest-EV
next step. Infrastructure is complete; integration is low-complexity and
directly measurable on BurstGPT and Azure 2024 traces.

### Q7. What is the shortest path to another +10% gain?

1. Wire `OutputLengthForecastBundle.p50` as the secondary scheduler sort key
   after SLA class (actual_output_tokens reserved as label-only).
2. Run on BurstGPT and Azure LLM 2024 with simulated length priors (use
   `num_predicted_output_tokens` from CARA as the shadow prior).
3. If short requests are served first, p90 TTFT drops → more SLA-safe goodput.
Estimated complexity: 1 run of low-medium scope (sort key + benchmark replay).

### Q8. What is the shortest path to another +50% gain?

1. Output length SRTF scheduling (+15–32%).
2. GPU routing on real price data (quantified from +routing_improvement_pp).
3. Admission gate cluster simulator integration (+3–8% from KV overflow reduction).
Combined: +50% plausible within 2–3 runs.

### Q9. What would need to be true to achieve +300%?

Unchanged. Requires: accurate output length prediction, heterogeneous GPU
placement (wired + benchmarked), measured queue-wait labels, agentic PDGraph,
joint carbon + placement optimization, pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **TTFT spread from CARA generalizes to production** — CARA is a research
   cluster. H100/T4 relative TTFT under production token distributions and
   serving frameworks (vLLM, TensorRT-LLM) may differ.
2. **Synthetic region_gpu_types match fleet reality** — the assignment
   us-east→H100, us-west→A100, us-south→T4 is a reasonable approximation
   but actual cloud region GPU fleets are heterogeneous within a region.
3. **SRTF gain transfers from LLM serving to the canonical energy trace** —
   the canonical trace uses `runtime_hours` (not output token count) as the
   job length signal. SRTF gains may be smaller outside pure LLM serving.

### Q11. Which benchmark weaknesses exist?

1. **Canonical CSVs confirmed present** — `data/caiso_us_west_dam.csv` etc.
   ARE in the repo. `run_gpu_routing_backtest()` was run in run -f: −0.14%
   goodput/$ (energy-price-dominated; see run -f root cause).
2. **No per-region GPU-type labels in public traces** — BurstGPT, Azure 2024,
   Alibaba GenAI all lack GPU-type metadata. Synthetic assignment is an
   approximation.
3. **BurstGPT short duration (34 min)** — GPU routing benefit may be
   dominated by model prewarm cost in a 34-minute window.

### Q12. Which public datasets should be added?

1. **Vidur profiling CSVs** — measured kernel latency on A100/H100/A40/T4 for
   LLM model sizes; enables data-driven penalty_floor/ceil calibration.
2. **ShareGPT** — output token counts for length predictor cross-dataset validation.
3. **Mooncake FAST25 traces** — KV prefix reuse cross-validation (unchanged).

### Q13. What should be attempted next?

**Immediate (next run):**
1. Wire `OutputLengthForecastBundle` p50 into the greedy scheduler sort key
   (after SLA class) as an SRTF prior; use `num_predicted_output_tokens` from
   CARA as the shadow prior value. Reserve `actual_output_tokens` as label-only.
2. Evaluate on BurstGPT and Azure 2024 traces.

**Short-term (2–3 runs):**
3. Wire `WorkloadAdmissionGate` into cluster simulator for Azure 2024 replay.
4. Obtain or mount canonical CSV files; run `run_gpu_routing_backtest()` with
   real price data to produce the quantitative GPU routing goodput/$ table.
5. Vidur CSV ingestion for penalty calibration.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Output token p50 → SRTF scheduler sort key | High | Low effort | Infrastructure built (shadow) |
| 2 | GPU routing benchmark on real price data | High | Low effort | Benchmark infra complete [run -e] |
| 3 | Admission gate → cluster simulator integration | Medium | Medium | Implemented (unconnected) |
| 4 | Vidur CSV ingestion for GPU penalty calibration | Low-Med | Low | Not started |
| 5 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 6 | Mooncake trace ingestion (KV prefix reuse) | Low-Med | Low | Not started |
| 7 | Carbon-power MILP joint optimization | Medium | High effort | Not started |
| 8 | Hermes PDGraph agentic routing | High | High effort | Not started |

---

## Run 2026-06-20-d

### Q1. What currently limits Aurelius most?

**Benchmark evaluation gap for GPU placement routing.** The GpuPlacementScorer
is now wired into the scheduler (run -d), but its goodput/$ impact has not
yet been measured on public traces because BurstGPT and Azure LLM 2024 lack
per-region GPU-type labels. Adding synthetic `region_gpu_types` metadata to
the canonical backtest is the immediate next step.

**Secondary:** Three shadow modules remain unconnected to any trace-replay backtest:
1. `WorkloadAdmissionGate` — implemented but not wired into cluster simulator
2. `OutputLengthForecastBundle` — p50 not yet used as scheduler sort key
3. GPU routing on public traces — wired but not yet benchmarked with GPU-type labels

### Q2. What theoretically offers the largest gain?

**Quantifying the GPU placement routing gain** on BurstGPT and Azure LLM 2024
with synthetic GPU-type metadata is now the shortest path to a measurable
benchmark delta. The 9× TTFT spread across GPU types in CARA data (H100 vs T4)
suggests that routing `latency_critical` requests to faster GPU types could
raise the SLA-safe rho ceiling, enabling more goodput per dollar.

**Second:** Output length p50 as SRTF prior — infrastructure complete;
integration is one scheduler sort-key change away.

### Q3. Which forecasts are weakest?

1. **GPU-type-specific TTFT calibration** — the scorer uses heuristic
   penalty_floor/ceil values ([0.05, 0.50]). These are not tuned from
   actual SLA-safe goodput/$ sensitivity data.
2. **Output token length** — forecaster built; calibration not validated
   on real CARA data.
3. **TTFT p99 tail** — still at baseline_fallback (67% fallback on time holdout).
4. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **GPU routing without benchmark validation** — the scheduler now routes
   `latency_critical` jobs by GPU type, but the gain magnitude is unknown.
2. **Request ordering without length priors** — p50 output length not used
   as a scheduling weight.
3. **Batch admission timing** — admission gate implemented but unconnected.

### Q5. Which workloads benefit least?

**GPU packing / training scheduling** — unchanged. GPU placement scorer applies
only to `latency_critical` LLM-serving jobs (SLA class gated). Training/packing
workloads are unaffected.

### Q6. Which research direction appears strongest?

**GPU placement benchmark evaluation** is now the most concrete next step:
add synthetic `region_gpu_types` to canonical backtest replay, enable the
scorer, measure before/after SLA-safe goodput/$. The implementation is ready;
only the benchmark annotation is missing.

### Q7. What is the shortest path to another +10% gain?

1. Add `region_gpu_types` synthetic metadata to BurstGPT + Azure 2024 replay
   (assign H100 to primary region, T4 to secondary region from CARA fleet data).
2. Run canonical backtest with GPU placement scorer enabled.
3. If `latency_critical` jobs route to H100 and reduce TTFT violations, the
   safe rho ceiling rises → more goodput/$.
Estimated complexity: 1 run of low scope (annotation + backtest run, no new algo).

### Q8. What is the shortest path to another +50% gain?

1. GPU placement routing benchmark (+5-15% directional estimate from TTFT spread).
2. Output length p50 → SRTF scheduling (+15-30% on LLM-serving traces).
3. Admission gate → cluster simulator (+3-8% from KV overflow reduction).
Combined: +50% plausible within 2-3 runs.

### Q9. What would need to be true to achieve +300%?

Unchanged. Requires: accurate output length prediction, heterogeneous GPU
placement (now wired), measured queue-wait labels, agentic PDGraph, joint
carbon + placement optimization, pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **TTFT spread generalizes from CARA to production** — CARA covers a research
   cluster; H100/T4 relative performance may differ under production load profiles.
2. **penalty_floor/ceil heuristic** — [0.05, 0.50] is a design choice. If the
   goodput/$ sensitivity to TTFT is lower than assumed, the penalty may be too
   aggressive and divert latency_critical jobs from cheaper regions unnecessarily.
3. **synthetic region_gpu_types** — assigning GPU types to regions synthetically
   may not match real heterogeneous cluster topology (GPU types per region in
   practice depend on fleet age and procurement).

### Q11. Which benchmark weaknesses exist?

1. **No per-region GPU-type labels** in any public trace — BurstGPT, Azure 2024,
   Alibaba GenAI all lack GPU-type metadata. Synthetic assignment needed.
2. **BurstGPT short duration** (34 min) — GPU routing benefit may be small in
   a 34-minute window where model prewarm dominates.
3. **TTFT calibration on CARA** — p50 is from a research cluster; production
   values may differ.

### Q12. Which public datasets should be added?

1. **Vidur profiling CSVs** — now the highest priority for GPU placement scorer
   calibration. Provides measured kernel latency on A100/H100/A40/T4 for
   specific LLM model sizes; enables penalty_floor/ceil tuning from data.
2. **Mooncake FAST25 traces** — KV prefix reuse cross-validation (unchanged).
3. **ShareGPT** — output token counts for length predictor cross-dataset validation.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Add synthetic `region_gpu_types` to BurstGPT + Azure 2024 canonical backtest
   (assign H100 / A100 / T4 to the CANONICAL_REGIONS based on CARA fleet composition).
2. Run canonical backtest with GPU placement scorer enabled; record before/after
   SLA-safe goodput/$ table.

**Short-term (2-3 runs):**
3. Wire `OutputLengthForecastBundle` p50 into scheduler greedy sort key.
4. Wire `WorkloadAdmissionGate` into cluster simulator.
5. Vidur CSV ingestion for penalty calibration.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | GPU routing benchmark evaluation (BurstGPT + Azure 2024) | High | Low effort | Wired (unvalidated on trace) |
| 2 | Output token calibration → SRTF scheduling | High | Medium | Infrastructure built |
| 3 | Admission gate → simulator integration | Medium | Medium | Implemented (unconnected) |
| 4 | BOute MOBO routing co-optimisation | High | High effort | Not Started |
| 5 | Vidur CSV ingestion for GPU penalty calibration | Low-Med | Low | Not Started |
| 6 | Mooncake trace ingestion | Low-Med | High | Not Started |
| 7 | TTFT p50 shadow integration | Medium | Low | shadow_ready |
| 8 | Hermes PDGraph agentic routing | High | High effort | Not Started |
| 9 | Carbon-power MILP joint optimization | Medium | High effort | Not Started |

---

## Run 2026-06-20-c

### Q1. What currently limits Aurelius most?

**Pilot telemetry** remains the top bottleneck. The GPU placement scorer, output
length forecaster, and admission gate are all implemented in shadow mode, but
their real goodput/$ impact cannot be quantified until wired into a backtest
simulation with GPU-type-annotated traces.

**Secondary:** Three shadow modules are now implemented but not yet wired into
the scheduler or cluster simulator:
1. `GpuPlacementScorer` — penalty ready but not folded into `_sla_adjusted_score`
2. `OutputLengthForecastBundle` — p50 ready but not used as scheduler sort key
3. `WorkloadAdmissionGate` — implemented but not connected to any trace replay

### Q2. What theoretically offers the largest gain?

**Wiring GpuPlacementScorer into the scheduler** for `latency_critical` SLA class
is now the shortest path to a measurable benchmark delta. The 9× TTFT spread
across GPU types seen in CARA is the largest unexploited signal in the system.
If routing `latency_critical` requests to faster GPU types reduces TTFT violations,
the allowed rho ceiling rises → more SLA-safe goodput/$.

**Second:** Semi-clairvoyant scheduling via output length p50 — infrastructure is
complete; integration is one scheduler change away.

### Q3. Which forecasts are weakest?

1. **GPU-type-specific TTFT at predict time** — the `GpuPlacementScorer` is built
   but needs integration; its real penalty calibration (penalty_floor/ceil) is
   a heuristic, not tuned from trace data.
2. **Output token length** — forecaster built; calibration not yet validated on
   real CARA data (data is gitignored).
3. **TTFT p99 tail** — still at baseline_fallback (67% fallback on time holdout).
4. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **GPU type selection without TTFT awareness** — `GpuPlacementScorer` built but
   not yet wired into `_find_best_slot` or `_sla_adjusted_score`.
2. **Request ordering without length priors** — `OutputLengthForecastBundle` built
   but not wired into greedy sort order.
3. **Batch admission timing** — admission gate implemented but unconnected.

### Q5. Which workloads benefit least?

**GPU packing / training scheduling** — unchanged. CA is near frontier on Alibaba
GPU, MIT Supercloud, Philly. The new GPU placement scorer applies to LLM serving
traces only (latency_critical SLA class), not training workloads.

### Q6. Which research direction appears strongest?

**GPU placement scorer → scheduler integration** is now the clearest single-run
deliverable. The infrastructure is complete; the remaining work is:
1. Pass `GpuPlacementScorer.latency_penalty` into scheduler objective for
   `latency_critical` placements.
2. Evaluate on BurstGPT with synthetic GPU-type labels from CARA prior table.

Second: **LAPS-SD insight (arXiv:2505.17074)** — speculative decoding reduces
per-token cost; combining output length prediction with SD token acceptance rate
could yield a compound gain for SD-capable LLM serving clusters.

### Q7. What is the shortest path to another +10% gain?

1. Wire `GpuPlacementScorer.latency_penalty` into `scheduler._sla_adjusted_score`
   as an additive term when `sla_class == "latency_critical"` (single function,
   ~10 lines of change).
2. Add GPU-type metadata to the BurstGPT and Azure 2024 trace replay.
3. Evaluate: if `latency_critical` requests route to h100 over t4 when the TTFT
   spread is large, the SLA-safe rho ceiling rises → more goodput/$.
Estimated complexity: 1 run of low-medium scope.

### Q8. What is the shortest path to another +50% gain?

1. Wire GPU placement scorer → BurstGPT evaluation → estimated +5-15%.
2. Wire output length p50 into SRTF scheduler ordering → +15-30% on LLM traces.
3. Wire admission gate into Azure 2024 replay → +3-8% from KV overflow reduction.
Combined: +50% total from three integrations is plausible within 2-3 runs.

### Q9. What would need to be true to achieve +300%?

Unchanged from prior runs. Requires: accurate output length prediction,
heterogeneous GPU placement (now built), measured queue-wait labels, agentic
PDGraph, joint carbon + placement optimization, pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **TTFT p50 stability across time** — the `TTFTShadowPrior` is a static table
   fitted from CARA data. If GPU performance varies by cluster load or driver
   version, the static prior may over-penalize under-loaded slower GPU types.
2. **penalty_floor/ceil heuristic calibration** — the [0.05, 0.50] range is a
   design choice, not tuned from trace data. If the actual goodput/$ sensitivity
   to TTFT is lower than assumed, the penalty may introduce routing distortions.
3. **Latency-critical fraction in public traces** — BurstGPT and Azure 2024 don't
   carry explicit SLA class labels; synthetic assignment from workload_type may
   under- or over-represent `latency_critical` workloads.

### Q11. Which benchmark weaknesses exist?

1. **No GPU-type labels in Azure 2024** — the scorer can't be directly validated
   on the largest trace without synthetic GPU-type assignment.
2. **BurstGPT short duration** (34 min) — may miss the TTFT benefit for long
   sessions where GPU type choice compounds over many requests.
3. **GPU packing traces at safe frontier** — Alibaba GPU, MIT Supercloud, Philly
   unchanged; scorer does not help training workloads.

### Q12. Which public datasets should be added?

1. **Mooncake FAST25 traces** — still highest priority for KV prefix reuse.
2. **Vidur profiling CSVs** — kernel latency priors for heterogeneous placement
   scorer tuning (validates penalty calibration on A100/H100/A10G/T4).
3. **ShareGPT conversation traces** — output token counts for length predictor
   cross-dataset validation.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Wire `GpuPlacementScorer.latency_penalty` into `scheduler._sla_adjusted_score`
   for `latency_critical` workloads — ~10 lines of change, medium impact.
2. Add GPU-type metadata to benchmark trace replay for BurstGPT + Azure 2024.
3. Evaluate and record before/after SLA-safe goodput/$ delta.

**Short-term (2-3 runs):**
4. Wire output length p50 into scheduler greedy sort key.
5. Admission gate → cluster simulator integration.
6. Mooncake trace ingestion.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | GPU placement scorer → scheduler integration | High | Low effort | Built (unconnected) |
| 2 | Output token calibration → SRTF scheduling | High | Medium | Infrastructure built |
| 3 | Admission gate → simulator integration | Medium | Medium | Implemented (unconnected) |
| 4 | BOute MOBO routing co-optimisation | High | High effort | Not Started |
| 5 | Mooncake trace ingestion | Low-Med | High | Not Started |
| 6 | TTFT p50 shadow integration | Medium | Low | shadow_ready |
| 7 | Hermes PDGraph agentic routing | High | High effort | Not Started |
| 8 | Carbon-power MILP joint optimization | Medium | High effort | Not Started |
| 9 | CARA train.jsonl TPOT expansion | Medium | Low | build_after_data |

---

## Run 2026-06-20-b

### Q1. What currently limits Aurelius most?

**Pilot telemetry** remains the top bottleneck. The output length forecaster
infrastructure is now built, but verifying calibration gain requires running
on actual CARA data (currently gitignored). Two components now exist and are
unit-tested; their real-world MAE improvement needs the CARA analysis_sample
JSONL to quantify.

**Secondary:** The output length predictor and admission gate are both
implemented but not yet wired into any backtest simulation. The gap between
"component built" and "goodput/$ quantified" requires simulator integration.

### Q2. What theoretically offers the largest gain?

**Semi-clairvoyant scheduling via calibrated output length** (arXiv:2604.06970
+ arXiv:2602.11812). The infrastructure is now in place:
- `BiasCalibrationForecaster` debiases `num_predicted_output_tokens`
- `HGBOutputLengthForecaster` predicts actual output length at p50/p90/p95
- The p50 prediction can be used as a SRTF-like scheduling weight

Expected impact when wired: 32% p90 short-request improvement + tail latency
reduction from admission gate, potentially +15-30% SLA-safe goodput/$.

### Q3. Which forecasts are weakest?

1. **Output token length** — forecaster built (shadow); calibration not yet
   validated on real CARA data; bias magnitude unknown until data is loaded.
2. **TTFT p99 tail** — still at baseline_fallback (67% fallback on time holdout).
3. **Queue wait** — derived proxy only (CARA research cluster runs cool).
4. **Cold-start latency / migration cost** — blocked_by_missing_labels.

### Q4. Which optimizer decisions remain suboptimal?

1. **Request ordering without length priors** — the scheduler currently uses
   FIFO / SLA-class ordering; it does not use `num_predicted_output_tokens` or
   the new calibrated p50 estimate. Wiring the p50 as a scheduling weight would
   enable SRTF-like behaviour for short requests.
2. **Batch admission timing** — admission gate (implemented) not yet wired in.
3. **Heterogeneous GPU routing** — TTFT 9× spread across GPU types not exploited.

### Q5. Which workloads benefit least?

**GPU packing / training scheduling** — unchanged from prior run. CA is near
frontier on Alibaba GPU, MIT Supercloud, Philly. Job duration prediction
remains the missing lever here.

### Q6. Which research direction appears strongest?

**Calibrated output length → SRTF scheduling** is now the clearest path.
The infrastructure gap is closed; the remaining work is:
1. Run calibration on CARA train/test split (requires data script)
2. Wire p50 into scheduler request ordering
3. Evaluate on Azure LLM 2024 + BurstGPT with simulated prior quality

Second: **Heterogeneous GPU placement scorer** — TTFT spread across GPU types
is 9×, and the `HGBOutputLengthForecaster` pattern gives a direct blueprint.

### Q7. What is the shortest path to another +10% gain?

1. Wire the `BiasCalibrationForecaster` into the dynamic routing path.
2. Use calibrated p50 as a secondary scoring dimension (after SLA class) in
   the greedy scheduler — prefer shorter predicted outputs at equal cost.
3. Evaluate on BurstGPT (currently +1.77%) where length-aware routing is most
   likely to improve margin.
Estimated complexity: 1 run of medium scope (no new data needed).

### Q8. What is the shortest path to another +50% gain?

1. Complete CARA output length backtest to validate calibration quality.
2. Wire calibrated p50 into scheduler → expected +15-30% on LLM-serving traces.
3. Add heterogeneous GPU placement scorer → +5-15% from TTFT spread exploitation.
4. Admission gate → cluster sim integration → +3-8% from KV overflow prevention.
Combined: +50% total is plausible within 3-4 runs.

### Q9. What would need to be true to achieve +300%?

Unchanged from prior run. Requires: accurate output length prediction,
heterogeneous GPU placement, measured queue-wait labels, agentic PDGraph,
joint carbon + placement optimization, pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **CARA `num_predicted_output_tokens` bias is correctable** — the calibration
   model assumes a stable scale + offset correction. If the engine uses multiple
   prediction algorithms or model-dependent biases, a single Huber regression
   may not capture the full correction. Per-model-size variant may help.
2. **HGB output length generalisation** — trained only on CARA (5 instance types,
   Qwen 2.5 model family). Generalization to other model families is unverified.
3. **p50 as SRTF prior** — the scheduling gain depends on the ratio of
   prediction accuracy to the natural variance. If actual output token variance
   within each bin is large relative to between-bin variance, the SRTF gain
   may be smaller than the 32% figure from arXiv:2604.06970.

### Q11. Which benchmark weaknesses exist?

Unchanged from prior run. Key: Azure LLM 2024 has no output token labels;
BurstGPT has no output token labels. The calibration forecaster can only be
validated on CARA (which has both fields).

### Q12. Which public datasets should be added?

1. **Mooncake FAST25 traces** — still highest priority for KV prefix reuse.
2. **Vidur profiling CSVs** — provides kernel latency priors for heterogeneous
   GPU placement scorer (now ranked #3 opportunity).
3. **ShareGPT conversation traces** — has output token counts; could serve as
   a second validation dataset for the output length predictor.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Run CARA output length calibration backtest — compute MAE of:
   (a) raw `num_predicted_output_tokens` vs actual
   (b) `BiasCalibrationForecaster` calibrated vs actual
   (c) `HGBOutputLengthForecaster` p50 vs actual
   This is the missing validation gate for the new module.
2. Wire p50 output length into scheduler scoring and evaluate on BurstGPT.

**Short-term (2-3 runs):**
3. Heterogeneous GPU placement scorer using HGB TTFT forecasts.
4. Admission gate → cluster simulator integration.
5. Mooncake trace ingestion.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Output token calibration → SRTF scheduling | High | Medium | Infrastructure built |
| 2 | Admission gate → simulator integration | Medium | Medium | Implemented (unconnected) |
| 3 | Heterogeneous GPU placement scorer | High | Medium | build_now |
| 4 | BOute MOBO routing co-optimisation | High | High effort | Not Started |
| 5 | Mooncake trace ingestion | Low-Med | High | Not Started |
| 6 | TTFT p50 shadow integration | Medium | Low | shadow_ready |
| 7 | Hermes PDGraph agentic routing | High | High effort | Not Started |
| 8 | Carbon-power MILP joint optimization | Medium | High effort | Not Started |
| 9 | CARA train.jsonl TPOT expansion | Medium | Low | build_after_data |

---

## Run 2026-06-20

### Q1. What currently limits Aurelius most?

**Pilot telemetry.** Every high-leverage forecaster (TTFT p99, queue-wait,
cold-start, migration cost, output length prediction) needs measured labels
from real production clusters. The public corpus is at the frontier for
arrival patterns, model-affinity, and prefix reuse; the remaining forecasting
gaps cannot be closed from public data alone.

**Secondary bottleneck:** The admission gate (`admission.py`) and the Dynamic
Frontier Estimator are both implemented but not wired into the cluster
simulator. Quantifying their goodput/$ impact on the Azure 2024 trace requires
simulation integration — currently only unit-tested.

### Q2. What theoretically offers the largest gain?

**Semi-clairvoyant scheduling via output length prediction** (arXiv:2604.06970).
When token magnitude priors are available:
- 32% improvement in short-request P90 vs FIFO
- Removing magnitude priors causes 5.8× p95 increase
- Adaptive Deficit Round Robin + feasible-set scoring achieves 100%
  completion + 100% deadline satisfaction under high congestion

This is the highest-leverage theoretical gain not yet attempted in Aurelius.
CARA already carries `num_predicted_output_tokens` vs `actual_output_tokens`.

### Q3. Which forecasts are weakest?

1. **TTFT p99 tail** — `baseline_fallback` (67% fallback on time holdout).
   Queue-feature augmentation didn't help (negative result).
2. **Queue wait** — derived proxy only (CARA research cluster runs cool).
3. **Cold-start latency** — `blocked_by_missing_labels` (no server-class
   model-load telemetry in any public dataset).
4. **Migration cost** — `blocked_by_missing_labels` (no migration event
   logs in public datasets beyond Mooncake's cache-loss proxy).

### Q4. Which optimizer decisions remain suboptimal?

1. **Request routing under heterogeneous GPU types** — the TTFT 9× p99
   spread across GPU types exists but is not exploited by the scheduler
   (heterogeneous placement scorer status: `build_now` but not built).
2. **Batch admission timing** — the batch inference controller uses a
   static deadline-slack window; the flow-rate admission gate (newly
   built) provides the missing dynamic signal but isn't wired in yet.
3. **Agentic / multi-step workload routing** — Hermes PDGraph approach
   not implemented. CC-traces shows structured multi-step patterns.

### Q5. Which workloads benefit least?

**GPU packing / training scheduling** — `constraint_aware` already ties
`best_fit` / FFD / topology_aware on all three packing traces (Alibaba GPU,
MIT Supercloud, Philly). These schedulers are near the safe frontier.
Further gains here require better job duration prediction (not yet built).

### Q6. Which research direction appears strongest?

**Semi-clairvoyant scheduling** (output token length priors) is the strongest
unexplored direction with direct public-trace evidence. The CARA dataset
already contains the required labels. This is implementable without new data.

Second: **Admission gate simulation integration** — the gate is built; wiring
it into the Azure 2024 trace replay could directly improve the p99 tail on
the largest committed benchmark.

### Q7. What is the shortest path to another +10% gain?

1. Wire admission gate into cluster simulator (no new algorithm needed).
2. Evaluate on Azure LLM 2024 under a high-load burst scenario.
3. If gate prevents KV overflow spikes that currently inflate timeout_pct,
   the allowed rho ceiling rises → more goodput/$.
Estimated complexity: 1-2 runs of medium scope.

### Q8. What is the shortest path to another +50% gain?

1. Build output-token-length predictor on CARA actual vs predicted.
2. Use length priors to implement Adaptive DRR (arXiv:2604.06970) for
   multi-class request scheduling.
3. The 32% short-request p90 improvement from the paper, extrapolated to
   Aurelius's mixed workload, suggests a potential +15-30% SLA-safe goodput/$.
4. Combined with the admission gate's tail-latency improvement, 50% total
   uplift is plausible on LLM-serving traces.

### Q9. What would need to be true to achieve +300%?

The +300% target vs `sla_aware` baselines requires:
1. **Accurate output length prediction** enabling tight scheduling (semi-clairvoyant).
2. **Heterogeneous GPU placement** using TTFT forecasts (9× spread exploitation).
3. **Measured queue-wait labels** to close the TTFT p99 tail gap.
4. **Agentic workload support** via PDGraph routing (Hermes-style).
5. **Cross-region + carbon joint optimization** (currently energy-only).
6. **Real production calibration** — the +300% is an aspirational simulator
   target; it almost certainly requires pilot telemetry before being reachable.

### Q10. Which assumptions might be wrong?

1. **KV cache utilization as the primary flow-control signal** — the admission
   gate uses `mean_utilization` as a KV proxy. If the actual KV fill and the
   GPU utilization diverge significantly, the gate may fire too early or too
   late. Pilot telemetry with explicit KV fill rate would validate this.
2. **Cache affinity proxy on BurstGPT** — the cache affinity baseline uses
   model-level routing, not real KV hit rate. If real KV hit rates are lower
   than modeled, BurstGPT's +1.77% gain may be smaller in production.
3. **Stable diffusion workload for Alibaba GenAI** — the +89% is largely a
   model-affinity effect specific to stable-diffusion serving. May not
   generalize to pure LLM serving.
4. **Deterministic risk estimator calibration** — risk scores in [0,1] are
   heuristic, not trained on real SLA outcomes. Their calibration is unknown.

### Q11. Which benchmark weaknesses exist?

1. **Azure LLM 2024** has no cache/session/latency signal — cannot validate
   cache-aware routing or TTFT forecasting on the largest trace.
2. **BurstGPT** has no output token labels — cannot evaluate output length
   prediction.
3. **GPU packing traces** are at the safe frontier — incremental improvements
   here require better job duration prediction, which no public dataset provides.
4. **Canonical energy backtest** uses synthetic job mix — not customer-derived.
5. **Small-scale traces** (BurstGPT 34 min, Azure 2023 0.003 days) are too
   short for temporal forecasting holdouts.

### Q12. Which public datasets should be added?

Priority order:
1. **Mooncake FAST25 traces** (Apache-2.0) — KV prefix reuse cross-validation.
   Bounded ingest feasible. Closes the single-dataset caveat on cache forecaster.
2. **Azure Functions 2019 / 2021** — arrival shape for embedding / ETL workloads.
   Large (~1B invocations) but bounded ingest feasible.
3. **Vidur profiling CSVs** — kernel latency priors for heterogeneous placement.
4. **CARA train.jsonl** expansion — 392 MB, 359k rows; unlocks TPOT forecasting
   at `strong` strength (current moderate strength insufficient).

### Q13. What should be attempted next?

**Immediate (next run):**
1. Audit CARA actual vs predicted output token counts — zero new data, already
   committed. Enables output length prediction assessment.
2. Mooncake trace ingestion — bounded, Apache-2.0, adds KV prefix reuse
   cross-dataset validation.

**Short-term (2-3 runs):**
3. Wire admission gate into cluster simulator — quantify goodput/$ on Azure
   2024 high-load burst scenario.
4. Build heterogeneous GPU placement scorer (TTFT 9× spread → routing alpha).

**Medium-term:**
5. Output token length predictor on CARA → semi-clairvoyant scheduling.
6. CARA train.jsonl expansion → TPOT forecasting upgrade to `strong`.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Output token length prediction (CARA) | High | Medium | Not Started |
| 2 | Admission gate → simulator integration | Medium | Medium | Implemented (unconnected) |
| 3 | Mooncake trace ingestion | Low-Med | High | Not Started |
| 4 | TTFT p50 shadow integration | Medium | Low | shadow_ready |
| 5 | Heterogeneous GPU placement scorer | High | Medium | build_now |
| 6 | CARA train.jsonl TPOT expansion | Medium | Low | build_after_data |
| 7 | Hermes PDGraph agentic routing | High | High effort | Not Started |
| 8 | Carbon-power MILP joint optimization | Medium | High effort | Not Started |
