# Aurelius Gap Analysis

> **Updated every run.** This document answers the 13 standard gap-analysis
> questions for each run, then ranks future opportunities by expected value.
>
> **Binding rules:** All numbers are simulator / public-trace directional.
> No production claim. `docs/RESULTS.md` §8 production-claim gate not met.

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
