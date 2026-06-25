# Aurelius Gap Analysis

> **Updated every run.** This document answers the 13 standard gap-analysis
> questions for each run, then ranks future opportunities by expected value.
>
> **Binding rules:** All numbers are simulator / public-trace directional.
> No production claim. `docs/RESULTS.md` §8 production-claim gate not met.

---

## Run 2026-06-25 — GenAI Canonical Routing Phase 3d (ARCHITECTURE CONVERGENCE — Phase 3d, Five-Failure Rule compliant)

### Q1. What currently limits Aurelius most?

**Architecture divergence in GenAI benchmark**: `genai_backtest._run_policy` owned the `constraint_aware` EWMA anticipatory + model-affinity sizing logic inline, bypassing `AureliusOptimizer`. Physics helpers (`_effective_service_s`, `_size_for_sla`, `_size_for_target`) were local to the monolith. Additionally, PR #72 had two CI failures: stale `affinity_prewarm_share_pct=62.1` (correct: 61.7 from source data) and ruff alphabetical import order violation (genai_serving import placed after replica_scaling).

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Unchanged from previous runs. Architecture integration does not affect KPIs.

### Q3. Which forecasts are weakest?

Unchanged. EWMA burst-tick under-estimation on BurstGPT HF (15-request structural gap vs AMCSG, confirmed deterministic by multi-seed audit).

### Q4. Which optimizer decisions remain suboptimal?

GenAI `constraint_aware` now fully routed through canonical optimizer. Remaining gap: per-tick capacity decisions on 15 BurstGPT burst ticks (structural; Five-Failure Rule prohibits new modules to address this).

### Q5. Which workloads benefit least?

Unchanged. Bursty traces (BurstGPT) on n_sla_safe metric.

### Q6. Which research direction appears strongest?

Five-Failure Rule active. Phase 3d complete. Remaining allowed architecture work: Phase 1b replay loop unification (collapse 4 independent replay loops into 1 engine; high complexity, high impact).

### Q7. What is the shortest path to another +1% gain?

No new KPI gain from this run (0% delta by design). Phase 1b replay loop unification would unblock combination search (energy + GenAI + replica_scaling compound).

### Q8. What is the current north-star status?

Unchanged. Azure: goodput/$ achieved (+5.94% via OSOTSS). BurstGPT: goodput/$ achieved (+5.85%). BurstGPT n_sla_safe gap: −15 requests (structural, deterministic). GenAI: constraint_aware 9.84 gp/$ (+89.46% vs sla_aware) — bit-identical, unchanged.

### Q9. What would need to be true to maintain north-star?

Unchanged. All current frontier results are deterministic; no drift risk.

### Q10. Which assumptions might be wrong?

Unchanged. The EWMA alpha=0.5 for GenAI anticipatory sizing is a fixed prior; if actual production arrival variance differs substantially from the Alibaba GenAI 2026 fixture, the replica count guidance may over- or under-provision. No oracle information used (causal EWMA only).

### Q11. Which benchmark weaknesses exist?

Unchanged. GenAI benchmark uses a 60-request fixture (1 tick) — too small for multi-tick EWMA convergence testing. Full Alibaba GenAI 2026 dataset would provide more tick diversity.

### Q12. Which public datasets should be added?

Full Alibaba GenAI 2026 dataset (raw) — would enable multi-tick EWMA warm-up validation and cross-validation of constraint_aware vs sla_aware on realistic multi-hour traces.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE STILL ACTIVE.** Phase 3d complete. Remaining architecture work:
1. **Phase 1b replay loop unification** — collapse four replay loops into one engine (high complexity, high impact; requires 0%-delta parity gate for all 4 modes)
2. **Third trace cross-validation** — OSOTSS on full Alibaba GenAI 2026 (if raw data available)
3. **Phase 4** — Promote frontier BASE/DYNAMIC → ρ-ceiling constraint (partial evidence: SUF +13% Azure only)

Results: `research/results/genai_canonical_routing_phase3d_2026-06-25.{md,json}`

---

## Run 2026-06-24 — Dead Frontier Code Deprecation (ARCHITECTURE SIMPLIFICATION — Phase 5, Five-Failure Rule compliant)

### Q1. What currently limits Aurelius most?

**Maintenance tax from dead frontier families**: `aurelius/frontier/eval_workload_*` and `batch_inference_*` (8 modules, ~1,827 LOC) had zero non-test/non-script consumers but remained in the repo as maintenance surface. EVAL_WORKLOAD and BATCH_INFERENCE frontier families were benchmarked in the Phase 6 impact table and found to have no benchmark consumer (confirmed by grep). Dead code creates confusion about which families are active and wastes review bandwidth.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Unchanged. Dead code removal does not affect KPIs.

### Q3. Which forecasts are weakest?

Unchanged. EWMA burst-tick under-estimation.

### Q4. Which optimizer decisions remain suboptimal?

Unchanged. Per-tick capacity on 15 BurstGPT burst ticks.

### Q5. Which workloads benefit least?

Unchanged.

### Q6. Which research direction appears strongest?

Five-Failure Rule active. Remaining allowed architecture work: Phase 1b replay loop unification.

### Q7. What is the shortest path to another +1% gain?

No new gain from this run (0% KPI delta by design). Replay loop unification (Phase 1b) would unblock combination search.

### Q8. What is the current north-star status?

Unchanged. Azure: goodput/$ achieved (+5.94% via OSOTSS). BurstGPT: goodput/$ achieved (+5.85%). BurstGPT n_sla_safe gap: −15 requests (structural, confirmed deterministic).

### Q9. What would need to be true to maintain north-star?

Unchanged.

### Q10. Which assumptions might be wrong?

Unchanged.

### Q11. Which benchmark weaknesses exist?

Unchanged (two public traces only, single cost model).

### Q12. Which public datasets should be added?

Unchanged (Alibaba GenAI 2026 for OSOTSS cross-validation).

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE STILL ACTIVE.** Dead code deprecation complete (Phase 5). Remaining architecture work:
1. **Phase 1b replay loop unification** — collapse four replay loops into one engine (high complexity, high impact; requires 0%-delta parity gate)
2. **Third trace cross-validation** — OSOTSS on Alibaba GenAI 2026 if raw data available
3. **Phase 4** — Promote frontier BASE/DYNAMIC → ρ-ceiling constraint (partial evidence: SUF +13% Azure only)

Results: `research/results/dead_frontier_deprecation_2026-06-24.json`

---

## Run 2026-06-24 — AMCSG + SOTSS-MIN Canonical Routing Parity (ARCHITECTURE CONVERGENCE — Phase 3b, Five-Failure Rule integration)

### Q1. What currently limits Aurelius most?

**Architecture divergence**: `_run_amcsg_backtest` and `_run_sotss_backtest` called `_joint_mcs_c_schedule` / `_sotss_min_cost_schedule` directly instead of routing through `_REPLICA_SCALING_OPTIMIZER.optimize()`. Additionally, `ReplicaScalingPolicy.optimize(mode="sotss_min")` silently discarded `initial_violations` (using `_`), making it unavailable to any caller going through the canonical optimizer facade.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Unchanged from previous run. Architecture integration complete for all primary backtest entry points.

### Q3. Which forecasts are weakest?

Unchanged. EWMA service-time prediction on burst ticks.

### Q4. Which optimizer decisions remain suboptimal?

All primary decisions now route through canonical optimizer. Secondary research paths (compound experiments, gate sweep variations) still call delegate functions directly — acceptable since these are research-only paths, not primary benchmarks.

### Q5. Which workloads benefit least?

Bursty traces (BurstGPT) on n_sla_safe metric — structural, confirmed deterministic.

### Q6. Which research direction appears strongest?

Five-Failure Rule active. Architecture is now converged for primary paths. Next: deprecate dead frontier code (EVAL_WORKLOAD, BATCH_INFERENCE) or Phase 1b replay loop unification.

### Q7. What is the shortest path to another +1% gain?

Architecture is now fully converged for primary paths. Third-trace cross-validation (Alibaba GenAI 2026) would confirm generalizability — but raw data not present.

### Q8. What is the current north-star status?

Unchanged. Azure: north-star achieved (159,578 >> 151,248). BurstGPT: north-star achieved.

### Q9. What would need to be true to maintain north-star?

North-star already achieved.

### Q10. Which assumptions might be wrong?

None identified. All previously wrong assumptions corrected.

### Q11. Which benchmark weaknesses exist?

1. Two public traces only (Azure LLM 2024, BurstGPT HF).
2. Secondary research backtest paths (compound experiments, gate sweeps) still call delegate functions directly — not a KPI issue, but architecture inconsistency.

### Q12. Which public datasets should be added?

Alibaba GenAI 2026 raw data not present. BurstGPT and Azure remain the only two fully integrated public traces.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE STILL ACTIVE.** Canonical architecture is now complete for all primary paths. Next allowed actions:
1. **Phase 1b replay loop unification** — collapse four replay loops into one engine (high complexity, high impact)
2. **Architecture simplification** — deprecate dead frontier code (EVAL_WORKLOAD, BATCH_INFERENCE)
3. **Route secondary research paths** — gate sweep variations, compound experiments through canonical optimizer

Results: `research/results/amcsg_sotss_canonical_routing_parity_2026-06-24.md`
Tests: `tests/test_amcsg_sotss_canonical_routing_parity.py` (33 tests, all passing)

---

## Run 2026-06-24 — OSOTSS Canonical Routing Parity (ARCHITECTURE CONVERGENCE — Five-Failure Rule integration)

### Q1. What currently limits Aurelius most?

**Architecture divergence**: OSOTSS was the only frontier mode not routing through the canonical `AureliusOptimizer` facade.  A production user calling `AureliusOptimizer(policy="replica_scaling", mode="online_sotss")` would receive a different (weaker) schedule because `ReplicaScalingConfig` lacked `baseline_n_sla_safe`, causing the oracle to use a more-conservative deterministic floor instead of the AMCSG stochastic baseline.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Architecture integration complete. Next highest-EV options: (1) energy × replica_scaling compound (+11.1% energy standalone, unknown compound; disallowed by Five-Failure Rule until reset). (2) OSOTSS on third public trace (Alibaba GenAI 2026 if compatible).

### Q3. Which forecasts are weakest?

EWMA service-time prediction on burst ticks — root cause of 15-request BurstGPT n_sla_safe gap (structural, deterministic, confirmed by multi-seed audit).

### Q4. Which optimizer decisions remain suboptimal?

Per-tick capacity on 15 BurstGPT burst ticks where OSOTSS under-provisions by 1 server vs AMCSG. Structural; cannot be closed without better burst prediction or oracle access to AMCSG's fixed higher-c schedule.

### Q5. Which workloads benefit least from OSOTSS?

Bursty traces (BurstGPT) on the n_sla_safe metric. Goodput/$ improvement holds (+5.85%).

### Q6. Which research direction appears strongest?

Five-Failure Rule active. Allowed work: architecture simplification (thin-delegate promotion), third-trace cross-validation (Alibaba/LMSYS if timestamp data exists).

### Q7. What is the shortest path to another +1% gain?

Architecture is now converged. Third-trace cross-validation would confirm generalizability of OSOTSS +5.9% gain beyond Azure and BurstGPT.

### Q8. What is the current north-star status?

**Azure: goodput/$ north-star achieved** (159,578 >> 151,248). **BurstGPT: goodput/$ north-star achieved** (178,109 >> 121,680). BurstGPT n_sla_safe: 5849 (−15 vs AMCSG 5864; confirmed structural, deterministic, not closable without better burst prediction).

### Q9. What would need to be true to maintain north-star?

North-star already achieved on both traces. No regression needed.

### Q10. Which assumptions might be wrong?

All previously wrong assumptions corrected: stochastic gap hypothesis falsified (multi-seed audit), conformal SRPT compound hypothesis falsified (PR #66 joint backtest). Current assumptions are well-calibrated.

### Q11. Which benchmark weaknesses exist?

1. **Two public traces only** — Azure LLM 2024 and BurstGPT HF. Third trace (Alibaba GenAI 2026) not yet evaluated with OSOTSS.
2. **Single cost model** — GSF spot-fleet at 95% spot, $0.80/hr.

### Q12. Which public datasets should be added?

Alibaba GenAI 2026 (`alibaba_genai_2026` — already in `data/external/alibaba_genai/`) for OSOTSS cross-validation if arrival timestamps are available.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE STILL ACTIVE.** Canonical architecture is now complete. Next allowed actions:
1. **Third trace cross-validation** — OSOTSS on Alibaba GenAI 2026 (check timestamp format compatibility)
2. **Thin-delegate promotion** — route `amcsg` / `sotss_min` backtests through optimizer facade too (lower priority since OSOTSS is frontier)
3. **Architecture simplification** — deprecate dead frontier code (EVAL_WORKLOAD, BATCH_INFERENCE)

Results: `research/results/osotss_canonical_routing_parity_2026-06-24.md`
Tests: `tests/test_osotss_canonical_routing_parity.py` (38 tests, all passing)

---

## Run 2026-06-24 — Multi-Seed Stochastic Gap Audit (BENCHMARK REALISM — Five-Failure Rule mandated)

### Q1. What currently limits Aurelius most?

**BurstGPT 15-request n_sla_safe gap (OSOTSS 5849 vs AMCSG 5864).** This run diagnoses the root cause.
Multi-seed audit (seeds {42, 123, 456, 789, 1337}) reveals both AMCSG and OSOTSS n_sla_safe are **fully deterministic** (std=0 across all seeds on both traces).

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Architecture integration — wire OSOTSS through AureliusOptimizer(policy="replica_scaling") for end-to-end
evaluation. The BurstGPT n_sla_safe gap is deterministic and cannot be closed by stochastic tuning.

### Q3. Which forecasts are weakest?

EWMA service-time prediction on burst ticks — the confirmed root cause of the 15-request BurstGPT gap.
EWMA is slow to adapt to sudden load spikes on bursty traces.

### Q4. Which optimizer decisions remain suboptimal?

Per-tick capacity on 15 BurstGPT burst ticks where OSOTSS under-provisions by 1 server vs AMCSG.
This is deterministic (same 15 ticks fail on all seeds).

### Q5. Which workloads benefit least from OSOTSS?

Bursty traces (BurstGPT) where EWMA under-predictions create deterministic capacity shortfalls.

### Q6. Which research direction appears strongest?

Architecture integration (AureliusOptimizer replica_scaling policy end-to-end evaluation).
All stochastic-oracle approaches have now been ruled out as mechanism-incorrect.

### Q7. What is the shortest path to another +1% gain?

Under Five-Failure Rule: architecture integration. The OSOTSS +5.94% goodput/$ gain on Azure
is already validated and deterministic; the BurstGPT +5.85% goodput/$ gain is also validated
with the n_sla_safe caveat documented.

### Q8. What is the current north-star status?

Azure: goodput/$ north-star achieved (159,578 >> 151,248). BurstGPT: goodput/$ north-star
achieved (178,109 >> 121,680). BurstGPT n_sla_safe: 5849 (-15 vs AMCSG 5864; **confirmed structural, deterministic**).

### Q9. What would need to be true to maintain north-star?

North-star achieved on both traces. No regression needed.

### Q10. Which assumptions might be wrong?

**CORRECTED:** "The gap comes from stochastic spot interruptions" was **WRONG**. Multi-seed audit
proves p_survive≈0.9982 makes the simulation effectively deterministic. The gap is from EWMA
under-prediction on specific burst ticks, not from spot interruptions.

### Q11. Which benchmark weaknesses exist?

1. **Single-seed stochastic evaluation**: Now confirmed NOT a weakness — the simulation is
   effectively deterministic at p_interrupt=10%/hr. All previous results are valid single-seed.
2. **Two public traces**: Azure LLM 2024 and BurstGPT HF only.

### Q12. Which public datasets should be added?

ShareGPT or LMSYS Chatbot Arena — third public trace for OSOTSS cross-validation.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE ACTIVE. No new modules. Focus on:**

1. **Architecture integration** — AureliusOptimizer(policy="replica_scaling") end-to-end
2. **Replay validation** — OSOTSS on third public trace (ShareGPT/LMSYS)
3. **Architecture simplification** — deprecate dead/duplicate code (frontier EVAL_WORKLOAD/BATCH_INFERENCE)
4. **Accept the BurstGPT n_sla_safe gap** — it is deterministic, structural, not closable without
   better burst prediction. Document as known limitation.

**Root cause update:** BurstGPT gap is EWMA prediction error (not stochastic), confirmed by
multi-seed audit (std=0 across 5 seeds). All stochastic-oracle approaches ruled out.

Results: `research/results/multi_seed_stochastic_audit_2026-06-24.{md,json}`
Tests: `tests/test_multi_seed_stochastic_audit.py` (10 fast tests passing)

---

## Run 2026-06-24 — Forecasted MCS Spot Fleet (NEUTRAL/NEGATIVE — forecasted_mcs below AMCSG on both traces)

### Q1. What currently limits Aurelius most?

**The forecasted_mcs modes cannot match AMCSG's arrival-oracle advantage.** AMCSG uses actual tick-t
arrival counts (oracle). Forecasted_mcs uses only data ≤ t-1. On bursty traffic (BurstGPT), the one-tick
lag causes 520 additional SLA violations (lag1) or 2024 additional violations (ewma) vs AMCSG.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Same as prior run — architecture integration, cross-trace validation. The forecasted_mcs deployment-realism
finding confirms the fundamental tradeoff: full deployability comes at a cost vs arrival-oracle baselines.

### Q3. Which forecasts are weakest?

Forecasted_mcs arrival forecast is weakest on bursty traces. Lag1 misses the burst by exactly one tick;
EWMA smooths out the burst onset, making under-provisioning even worse.

### Q4. Which optimizer decisions remain suboptimal?

1. **Forecasted arrival for bursty traffic** — no single-tick lag approach can anticipate a burst.
2. **BurstGPT border ticks** — same structural stochastic gap from prior analysis.

### Q5. Which workloads benefit least from forecasted_mcs?

Bursty traces. Smooth traces (Azure) are closer — ewma is only −0.31% below AMCSG on Azure. BurstGPT
lag1 p99=47.5s (vs 30s SLA); ewma p99=67.4s. The mode is unsafe on bursty traces at default settings.

### Q6. Which research direction appears strongest?

**Same as prior run: architecture integration and replay validation.** The forecasted_mcs result
independently confirms the Five-Failure Rule focus: adding new scheduling approaches does not improve
on AMCSG without oracle arrival knowledge. The canonical integration path (Phase 1b unified replay)
remains the highest-value unblocked work.

### Q7. What is the shortest path to another +1% gain?

Under the architectural focus rule, the shortest path is validation: OSOTSS on a third public trace.

### Q8. What is the current north-star status?

Azure: OSOTSS 159,578 (+5.94% vs AMCSG). BurstGPT: OSOTSS 178,109 (+5.85% vs AMCSG). Both above
north-star goodput/$ threshold. Forecasted_mcs: below north-star on both traces.

### Q9. What would need to be true to maintain north-star?

No regressions to OSOTSS. Forecasted_mcs is not a leaderboard entry — negative result, no update.

### Q10. Which assumptions might be wrong?

The arrival-oracle hypothesis is confirmed: forecasted_mcs disadvantage is structural, not a tuning
issue. EWMA alpha, window, or safety buffer changes would not close a one-tick structural lag.

### Q11. Which benchmark weaknesses exist?

1. **Two public traces only** — Azure (smooth) and BurstGPT (bursty). Third trace needed.
2. **Single seed (42)** — stochastic evaluation; multi-seed audit merged (PR #64).

### Q12. Which public datasets should be added?

ShareGPT or LMSYS Chatbot Arena for OSOTSS cross-validation.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE ACTIVE (5/5). No new modules. Focus on:**

1. **Architecture integration** — Phase 1b unified replay engine (see OPTIMIZER_UNIFICATION_PLAN.md)
2. **Replay validation** — OSOTSS on a third public trace (ShareGPT/LMSYS)
3. **Benchmark realism** — multi-seed audit for BurstGPT stochastic gap (PR #64 merged)
4. **Bottleneck diagnosis** — stochastic oracle characterization for residual 3-request gap
5. **Architecture simplification** — see Phase 4/5 in OPTIMIZER_UNIFICATION_PLAN.md

Results: `research/results/forecasted_mcs_spot_backtest_2026-06-24.{md,json}`
Tests: `tests/test_forecasted_mcs_spot_backtest.py` (18 pass, 28 skip if numpy absent)

---

## Run 2026-06-24 — Oracle Soft-SLA Continuation (OSSC) OSOTSS (NEGATIVE RESULT — Five-Failure Rule TRIGGERED)

> **Root-cause correction (multi-seed audit 2026-06-24):** The Q1 diagnosis below was
> incorrect. The 3-15 request gap does NOT come from stochastic spot interruptions —
> multi-seed audit (std=0 across 5 seeds) proves the simulation is deterministic.
> The gap comes from EWMA under-prediction on 15 specific burst ticks.

### Q1. What currently limits Aurelius most?

**~~Structural stochastic gap on BurstGPT.~~** *(Corrected: deterministic EWMA prediction gap.)*
The OSOTSS oracle is deterministic-FIFO-optimal:
it eliminates all deterministic FIFO violations, but 15 requests fail because EWMA under-predicts
service time on burst ticks — NOT from stochastic spot interruptions (confirmed by multi-seed audit).

### Q2. What theoretically offers the largest gain beyond OSOTSS?

1. **Full stochastic oracle** — run the actual Binomial interruption simulation inside the oracle loop,
   not a separate post-convergence deterministic phase. This would directly optimize for stochastic n_sla_safe
   rather than deterministic n_sla_safe. However, stochastic oracles are expensive and non-convergent.
2. **ShareGPT/LMSYS cross-validation** — validate OSOTSS generalization on a third public trace.
3. **Architecture integration** — wire energy + serving + replica policy into a single AureliusOptimizer
   chain for end-to-end evaluation.

### Q3. Which forecasts are weakest?

1. **BurstGPT gap diagnoses** — four consecutive hypotheses (C1PGS, SOTSS-GSF, Adaptive EWMA, SSM, OSSC)
   all failed to close the gap. Each was architecturally different but empirically equivalent.
2. **Oracle borderline tightening** — OSSC made progress (gap -15 → -3) but didn't converge. The 3
   remaining failures at 5.0s margin are beyond deterministic-oracle reach.

### Q4. Which optimizer decisions remain suboptimal?

1. **BurstGPT borderline ticks** — n_sla_safe=5861 (best OSSC) vs AMCSG 5864. A structural
   3-request gap remains at maximum tested margin.
2. **Azure goodput/$ vs BurstGPT SLA tradeoff** — any approach that adds capacity to fix BurstGPT
   costs Azure goodput/$ at the same time (confirmed by OSSC sweep).

### Q5. Which workloads benefit least from OSOTSS?

Bursty traces (BurstGPT) show structural stochastic limitations. The gap is now well-characterized:
5849–5861 range (deterministic post-convergence ceiling). Closing the last 3–15 requests requires
stochastic oracle computation.

### Q6. Which research direction appears strongest?

**Architecture integration and replay validation** — the Five-Failure Rule is now active.
The strongest next step is an end-to-end integration audit:
- Verify AureliusOptimizer(policy="replica_scaling") path with ReplicaScalingConfig.borderline_margin_s
- Cross-validate OSOTSS on ShareGPT or LMSYS trace
- Audit the BurstGPT stochastic gap root cause with a controlled stochastic oracle experiment

### Q7. What is the shortest path to another +1% gain?

Under the Five-Failure architectural focus rule, new module development is suspended.
The shortest path is validation: confirm OSOTSS (159,578 goodput/$, +5.94%) holds on a third
public trace. If it generalizes, that is a publishable frontier result.

### Q8. What is the current north-star status?

Azure: goodput/$ north-star achieved (159,578 >> 151,248 threshold). BurstGPT: goodput/$ north-star
achieved (178,109 >> 121,680 threshold). BurstGPT n_sla_safe: best OSSC = 5861 (-3 vs AMCSG 5864).

### Q9. What would need to be true to maintain north-star?

North-star is already achieved on both traces on goodput/$. Maintaining it requires no regressions.

### Q10. Which assumptions might be wrong?

1. **AMCSG achieves 5864 stochastically by accident** — AMCSG's fixed higher-c provides global over-provisioning
   that absorbs stochastic interruptions on borderline ticks. This was confirmed by the structural gap analysis.
2. **Closing to 5864 is the right target** — BurstGPT n_sla_safe=5861 (-3) may be within noise if the
   stochastic seed changes. A multi-seed validation would clarify whether the 3-request gap is structural.

### Q11. Which benchmark weaknesses exist?

1. **Single seed stochastic evaluation** — all results use seed=42. Gap of 3 requests at best OSSC margin
   may reverse with different seeds. Multi-seed validation is a natural next step.
2. **Two public traces** — Azure LLM 2024 and BurstGPT HF only.

### Q12. Which public datasets should be added?

**ShareGPT** or **LMSYS Chatbot Arena** — third public trace for OSOTSS cross-validation.
This is the highest priority under the architectural focus rule.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE ACTIVE. No new modules. Focus on:**

1. **Architecture integration** — end-to-end AureliusOptimizer chain validation
2. **Replay validation** — OSOTSS on a third public trace (ShareGPT/LMSYS)
3. **Benchmark realism audit** — multi-seed stochastic evaluation for BurstGPT gap characterization
4. **Bottleneck diagnosis** — stochastic oracle experiment to characterize the 3-request residual
5. **Architecture simplification** — review wired-through params, reduce dead code

**Five-Failure Rule counter: 5/5 — ARCHITECTURAL FOCUS RULE ACTIVE.**

Results: `research/results/borderline_osotss_backtest_2026-06-24.{md,json}`
Tests: `tests/test_borderline_osotss_backtest.py` (10 tests, all passing)

---

## Run 2026-06-24 — Stochastic Safety Margin OSOTSS (NEGATIVE RESULT — mechanism misdiagnosed)

### Q1. What currently limits Aurelius most?

**Structural oracle-ceiling on BurstGPT.** The oracle's secondary termination condition (`violators=[]`
in deterministic FIFO) fires at n_sla_safe=5849, which is 15 below the AMCSG stochastic target (5864).
Adding `interrupt_safety_margin` to the convergence threshold cannot overcome this because the loop exits
via the secondary break before the primary convergence check is evaluated.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

1. **Oracle borderline-tick continuation** — allow the oracle to add capacity to ticks with response
   times within ε of the SLA limit (e.g., within 1s of 30s), even after all hard violations are resolved.
   This would let the oracle buffer borderline ticks against stochastic interruptions without future-token
   access. Requires a new "close to SLA" signal in the oracle loop.
2. **ShareGPT/LMSYS cross-validation** — third/fourth public trace to validate OSOTSS generalization.
3. **Transformer service-time predictor** — replace EWMA; closes OSOTSS-vs-SOTSS-MIN gap on Azure.

### Q3. Which forecasts are weakest?

1. **Oracle ceiling diagnosis** — the original hypothesis (safety margin would force oracle to over-provision)
   was falsified. The oracle's natural ceiling is set by "no deterministic FIFO violations" not by a
   convergence threshold.
2. **Stochastic/deterministic gap root cause** — confirmed: AMCSG's higher fixed-c schedule provides more
   capacity than OSOTSS's optimized minimum-violation schedule, which is lean but not stochastic-robust.

### Q4. Which optimizer decisions remain suboptimal?

1. **Hard "no violators" oracle exit** — oracle cannot buffer borderline ticks; only fixes hard violations.
   A soft "close-to-SLA" signal would allow proactive over-provisioning on vulnerable ticks.
2. **Aggressive starting gate (100%)** — minimum stable c leaves borderline ticks exposed; could use a
   slightly higher floor without losing most of the cost savings.

### Q5. Which workloads benefit least from OSOTSS?

Same as previous runs: traces with bursty arrivals (BurstGPT). The BurstGPT 15-request gap is now
confirmed to be structural (oracle ceiling at 5849 in deterministic FIFO), not addressable by
convergence-threshold adjustments.

### Q6. Which research direction appears strongest?

**Oracle borderline-tick continuation** — modify the oracle loop to continue adding capacity to
ticks whose deterministic-FIFO response times are within ε of the SLA (e.g., 28–30s on BurstGPT
SLA=30s), even after `violators=[]`. These are the ticks most vulnerable to stochastic interruptions.
This is causal (uses only actual service times already simulated in the convergence check) and
doesn't require future-token access.

**WARNING: Five-Failure counter is 4/5. One more negative run triggers the architectural focus rule.**

### Q7. What is the shortest path to another +1% gain?

**Oracle soft-SLA continuation.** Add a second pass after `violators=[]`: identify ticks where
any request's deterministic-FIFO response time is in (sla_s - ε, sla_s] (borderline ticks) and
increment c on those ticks up to c_ceil. This buffers the stochastic interruption window without
over-provisioning every tick.

### Q8. What is the current north-star status?

Azure: goodput/$ north-star achieved (159,578 >> 151,248 threshold). BurstGPT: goodput/$ north-star
achieved (178,109 >> 121,680 threshold). BurstGPT n_sla_safe remains 15 below AMCSG (known structural
limitation — oracle ceiling at 5849).

### Q9. What would need to be true to maintain north-star?

Same as run 2026-06-23. North-star is already achieved on both traces.

### Q10. Which assumptions might be wrong?

1. **Safety margin fixes the convergence threshold problem.** FALSIFIED by this run. The problem is the
   secondary `violators=[]` break, not the convergence threshold value.
2. **EWMA prediction accuracy is the bottleneck.** FALSIFIED by adaptive EWMA run (2026-06-24).
3. **Oracle ceiling at 5849 is hard.** May be addressable by oracle soft-SLA continuation — not yet tested.

### Q11. Which benchmark weaknesses exist?

1. **Oracle secondary-break exits before margin test** — confirmed structural limitation on BurstGPT.
2. **Two public traces** — Azure LLM 2024 and BurstGPT HF only.

### Q12. Which public datasets should be added?

Same as previous runs: ShareGPT, LMSYS Chatbot Arena.

### Q13. What should be attempted next?

**⚠️ FIVE-FAILURE WARNING: Counter is 4/5. If the next run is also negative, the architectural focus
rule activates: stop adding modules; focus only on integration, replay validation, and simplification.**

Priority 1 (before architectural focus triggers):
1. **Oracle soft-SLA continuation** — allow oracle to continue adding capacity to borderline ticks
   (response time within ε of SLA) even after `violators=[]`. Directly targets the confirmed structural
   mechanism. Causal and production-deployable.

Priority 2 (if Five-Failure rule activates):
1. **Architecture simplification** — review all wired-through params for dead-code cleanup
2. **Integration validation** — verify full AureliusOptimizer path with all new parameters
3. **ShareGPT/LMSYS replay** — cross-validation on third public trace

**Five-Failure Rule counter: 4 of 5 consecutive non-frontier runs.**

Results: `research/results/stochastic_safety_margin_osotss_backtest_2026-06-24.{md,json}`
Tests: `tests/test_stochastic_safety_margin_backtest.py` (10 tests)

---

## Run 2026-06-24 — Adaptive EWMA Online SOTSS (NEGATIVE RESULT — hypothesis falsified)

### Q1. What currently limits Aurelius most?

**Stochastic/deterministic simulation mismatch in OSOTSS BurstGPT.** The 15-request gap (n_sla_safe=5849
vs AMCSG 5864) arises because the oracle convergence check uses deterministic FIFO simulation while the
stochastic GSF evaluation includes spot interruptions. This run tested whether adaptive EWMA predictions
could close this gap; they cannot without over-provisioning.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

1. **Stochastic interrupt buffer in oracle convergence target** — set baseline_n_sla_safe = amcsg_n_sla_safe +
   interrupt_safety_margin (e.g., +20 to +30 on BurstGPT). This directly addresses the confirmed root cause.
2. **Transformer service-time predictor** — replace EWMA with a learned predictor; estimated MAPE reduction
   from ~15% to ~5-8%; closes OSOTSS-vs-SOTSS-MIN gap on Azure.
3. **ShareGPT/LMSYS cross-validation** — third/fourth public trace to validate OSOTSS generalization.

### Q3. Which forecasts are weakest?

1. **EWMA alpha=0.1 fixed** — adaptive alpha (this run) was tested and found not to help; the prediction
   accuracy gap is NOT the bottleneck for the 15-request BurstGPT gap.
2. **Deterministic oracle convergence target** — oracle targets deterministic n_sla_safe, but stochastic
   evaluation needs a buffer. A +20-30 safety margin may close the BurstGPT gap.

### Q4. Which optimizer decisions remain suboptimal?

1. **Deterministic oracle baseline target** — oracle targets amcsg_n_sla_safe from stochastic simulation but
   convergence uses deterministic FIFO; a safety buffer on the baseline would address the stochastic gap.
2. **Single-step EWMA prediction** — still predicts only current tick's service time; multi-step prediction
   could anticipate arrival bursts ahead of time.
3. **Global mean warm-start** — production would use a rolling prior, not offline global mean.

### Q5. Which workloads benefit least from OSOTSS?

Same as previous run: traces with bursty arrivals (BurstGPT). The gap is now confirmed to be a
stochastic/deterministic oracle-simulation mismatch, not a prediction quality issue.

### Q6. Which research direction appears strongest?

**Stochastic interrupt buffer in oracle convergence target.** The confirmed root cause of the BurstGPT
15-request gap is that the oracle targets a deterministic baseline while the stochastic evaluation
exposes spot-interruption-induced capacity reductions on borderline ticks. Adding a +20-30 request
safety buffer to the oracle baseline would directly address this without prediction improvements.

### Q7. What is the shortest path to another +1% gain?

**Stochastic safety margin.** Change `baseline_n_sla_safe = amcsg_n_sla_safe` to
`baseline_n_sla_safe = amcsg_n_sla_safe + safety_margin` (e.g., +20 on BurstGPT).
This should close the 15/5849 = 0.26% n_sla_safe gap without changing goodput/$ materially
(oracle adds a few extra servers on borderline ticks, matched precisely to the interruption buffer).
Must verify Azure regression-free (Azure already meets baseline with no gap).

### Q8. What is the current north-star status?

Same as run 2026-06-23: both Azure and BurstGPT goodput/$ north-stars achieved. BurstGPT
n_sla_safe remains 15 below AMCSG (known limitation).

### Q9. What would need to be true to maintain north-star?

Same as run 2026-06-23.

### Q10. Which assumptions might be wrong?

1. **EWMA prediction accuracy is the bottleneck.** FALSIFIED by this run. The 15-request gap
   is a stochastic/deterministic simulation mismatch, not a prediction error.
2. **Oracle convergence target equals final evaluation metric.** CONFIRMED WRONG. Oracle uses
   deterministic n_sla_safe; evaluation uses stochastic GSF. A safety buffer is needed.

### Q11. Which benchmark weaknesses exist?

1. **Deterministic/stochastic oracle-evaluation mismatch** — oracle convergence in deterministic
   FIFO; evaluation in stochastic GSF. A 15-request gap on BurstGPT results. Known limitation.
2. **Two public traces** — Azure LLM 2024 and BurstGPT HF only.

### Q12. Which public datasets should be added?

Same as run 2026-06-23: ShareGPT, LMSYS Chatbot Arena.

### Q13. What should be attempted next?

1. **Stochastic safety margin** — oracle baseline_n_sla_safe += interrupt_safety_margin; directly
   targets the confirmed root cause of the BurstGPT 15-request gap.
2. **ShareGPT/LMSYS cross-validation** — third public trace.
3. **Transformer service-time predictor** — replace EWMA for Azure oracle gap (SOTSS-MIN vs OSOTSS).

**Five-Failure Rule counter: 3 of 5 consecutive non-frontier runs.**

Results: `research/results/adaptive_ewma_osotss_backtest_2026-06-24.{md,json}`
Tests: `tests/test_adaptive_ewma_backtest.py` (8 tests, all passing)

---

## Run 2026-06-23 — Online SOTSS / OSOTSS (FRONTIER IMPROVEMENT on Azure, MIXED on BurstGPT)

### Q1. What currently limits Aurelius most?

**Production deployability of SOTSS-MIN.** SOTSS-MIN achieves +6.29% vs AMCSG on Azure but uses
actual per-tick token counts at scheduling time — future knowledge unavailable in production.
This run replaces oracle tokens with causal EWMA predictions (alpha=0.1), making the oracle loop
production-safe while recovering 94.4% of the oracle gain.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

OSOTSS is the production frontier. The oracle frontier (SOTSS-MIN) leaves 5.6% uncovered due to
EWMA prediction error. Further gains require:
1. **Better service-time prediction** — transformer-based next-request predictor; expected MAPE reduction
   from ~15% (EWMA) to ~5-8%, closing the OSOTSS-vs-SOTSS-MIN gap.
2. **Adaptive EWMA alpha** — per-tick alpha tuning based on recent prediction error; burst-adaptive.
3. **Multi-step prediction horizon** — predict service times k ticks ahead; handles bursty BurstGPT.
4. **ShareGPT/LMSYS cross-validation** — test OSOTSS on a third public trace.

### Q3. Which forecasts are weakest?

1. **EWMA alpha=0.1 is fixed** — optimal alpha may differ between smooth (Azure) and bursty (BurstGPT)
   traces. BurstGPT's 15-request gap could shrink with higher alpha tracking recent bursts.
2. **BurstGPT 15-request gap** — 0.26% below AMCSG in stochastic evaluation; deterministic oracle
   convergence check passes but stochastic GSF exposes the gap. May vary with different seeds.
3. **EWMA warm-start from global mean** — global mean is computed offline (whole trace); in production,
   the prior would need to be a rolling historical mean.

### Q4. Which optimizer decisions remain suboptimal?

1. **Fixed EWMA alpha=0.1** — not adaptive to burst patterns in bursty traces (BurstGPT).
2. **Single-step EWMA** — predicts current-tick service time from past ticks; cannot anticipate
   upcoming request bursts (e.g., BurstGPT's 5-min surge patterns).
3. **Offline global mean warm-start** — requires complete trace history; production would use a
   rolling prior from the previous N ticks.

### Q5. Which workloads benefit least from OSOTSS?

Traces where EWMA predictions are systematically biased: heavy-burst arrival patterns (OSOTSS
guides capacity to different ticks than oracle), very non-stationary service times (EWMA lags),
or traces where violations are spread across all ticks (oracle needs many iterations with
imperfect guidance). The BurstGPT 15-request gap exemplifies all three.

### Q6. Which research direction appears strongest?

**Adaptive EWMA alpha per trace/tick.** The Azure-vs-BurstGPT performance gap (SLA-safe vs 15-request
gap) is directly caused by EWMA's inability to track BurstGPT's bursty arrival patterns. A simple
adaptive alpha (increase on burst detection, decay on stability) could close the gap.

Alternative: **Multi-region spot pricing** remains the highest untapped gain (estimated 5-20%);
OSOTSS is the production baseline needed before multi-region experiments.

### Q7. What is the shortest path to another +1% gain?

**Adaptive EWMA alpha.** On BurstGPT, the 15-request gap (0.26%) appears on specific bottleneck ticks
where EWMA underestimates service time due to preceding quiet period. A burst-detection heuristic
(alpha=0.5 if last-tick load > 1.5× rolling mean, alpha=0.1 otherwise) could close this gap without
oracle access.

### Q8. What is the current north-star status?

- **Azure +500% north-star (151,248):** ACHIEVED. OSOTSS: 159,578 (+5.86% margin above threshold).
  SOTSS-MIN: 160,107 (+5.87% margin). Both oracle and production frontiers clear the threshold.
- **BurstGPT +500% north-star (121,680):** ACHIEVED (goodput/$). OSOTSS: 178,109 (+46.4% margin).

### Q9. What would need to be true to maintain north-star on other traces?

For OSOTSS to achieve north-star: violations must be concentrated on ≤20% of ticks; EWMA predictions
must be directionally correct (right bottleneck ticks even if imprecise); SLA budget must be generous
enough that 0.26% prediction-gap requests don't breach it. Azure satisfies all three; BurstGPT
borderline on the last condition.

### Q10. Which assumptions might be wrong?

1. **Dual-simulation design is sufficient.** The BurstGPT gap shows that even with actual-pair
   convergence check, the stochastic evaluation can expose violations when EWMA guides to wrong ticks.
2. **EWMA alpha=0.1 is universally appropriate.** Azure's smooth workload may mask that other traces
   need alpha>0.1 for better burst tracking.
3. **Global mean warm-start is neutral.** On traces with load ramp-up, global mean may over-estimate
   early ticks and under-estimate later ticks.

### Q11. Which benchmark weaknesses exist?

1. **Two public traces only** — Azure LLM 2024 and BurstGPT HF; ShareGPT/LMSYS needed for a third.
2. **BurstGPT borderline result** — 15-request SLA gap (0.26%) documented as known limitation; a
   different random seed could push this above or below AMCSG.
3. **EWMA warm-start uses global mean** — in production, this would be a rolling prior (lookahead bias
   of ~6% on early ticks); the benchmark uses offline global mean.

### Q12. Which public datasets should be added?

1. **ShareGPT** — third LLM trace; test OSOTSS on a non-Azure, non-BurstGPT workload pattern.
2. **LMSYS Chatbot Arena** — fourth trace; heavier multi-turn, highly variable token counts.
3. **AzurePublicDataset conversation traces** — longer context windows; different p99 profile.

### Q13. What should be attempted next?

1. **Adaptive EWMA alpha** — burst-tracking variant to close the BurstGPT 15-request SLA gap.
2. **ShareGPT/LMSYS cross-validation** — validate OSOTSS on a third public trace.
3. **Multi-region spot arbitrage** — OSOTSS is now the production baseline; multi-region can build on it.
4. **Transformer service-time predictor** — replace EWMA with a learned predictor; expected larger
   fraction of SOTSS-MIN oracle gain recovered.

**Five-Failure Rule counter: 2 of 5 consecutive non-frontier runs.**

Results: `research/results/online_sotss_backtest_2026-06-23.{md,json}`
Tests: `tests/test_online_sotss_backtest.py` (30 tests, all passing)

---

## Future Opportunity Ranking — Updated After Run 2026-06-24 (Adaptive EWMA — Negative Result)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Stochastic safety margin (oracle baseline += interrupt buffer) | High | High | Directly targets confirmed root cause of BurstGPT 15-req gap |
| 2 | ShareGPT/LMSYS cross-validation of OSOTSS | Medium | High | Third/fourth public trace; validates EWMA approach |
| 3 | Transformer service-time predictor (replace EWMA) | High | Medium | Larger SOTSS-MIN fraction recovered; production-safe |
| 4 | Multi-region spot arbitrage (SkyPilot/arXiv:2605.22778) | High | Medium | OSOTSS is production baseline; multi-region on top |
| 5 | Stochastic oracle variant (run oracle with GSF not FIFO) | Medium | Medium | Closes deterministic/stochastic gap (may combine with #1) |

**Closed/characterized opportunities:**
- Adaptive EWMA alpha (this run): **NEGATIVE RESULT** — hypothesis falsified. Gap is stochastic/deterministic mismatch.
- OSOTSS (EWMA alpha=0.1): **FRONTIER IMPROVEMENT** — 159,578 goodput/$ (Azure, +5.94% vs AMCSG)
- Dual-simulation design: violation ID from predicted pairs, convergence from actual pairs
- 94.4% of SOTSS-MIN oracle gain recovered while being production-deployable
- BurstGPT causal-prediction limitation: 15-request SLA gap (0.26%) — root cause now confirmed (stochastic gap)

---

## Run 2026-06-23 — SOTSS-GSF Stochastic Oracle (NULL RESULT — hypothesis falsified)

### Q1. What was attempted?

**SOTSS-GSF (Stochastic Oracle SOTSS):** Replace the deterministic FIFO oracle in the SOTSS-MIN
fix-up loop with a stochastic Binomial oracle (seed=42, matching the final evaluation). Hypothesis:
spot-interruption-vulnerable ticks missed by the deterministic oracle are detected by the stochastic
oracle, yielding a cheaper schedule with equal or better safety. Motivated by finding that BurstGPT
at gate=100% SOTSS-MIN is 4 requests short of AMCSG safety criterion.

### Q2. Why it failed

**Per-tick interruption probability is negligible.** At `p_interrupt=10%/hr` with `tick_seconds=60s`:
```
p_survive = (1 - 0.10)^(60/3600) = 0.90^(1/60) ≈ 0.9982
```
Each spot instance survives each 60s tick with 99.82% probability. The Binomial draw
`survived = Binomial(c_spot, 0.9982)` almost always equals `c_spot`, making `c_effective ≈ c`
in every oracle iteration. The stochastic oracle never "sees" an interruption during the oracle
loop, so it fixes the same ticks as the deterministic oracle and converges to the identical schedule.

**Oracle-simulation gap (BurstGPT).** Even with the same seed, the oracle
(`_oracle_stochastic_response_times`: Binomial c_effective → deterministic FIFO) differs from
the evaluator (`_simulate_fifo_gsf_spot_fleet`: full stochastic FIFO). These use the same Binomial
draws but process requests differently (batch-tick vs individual-arrival). The 4-request shortfall
on BurstGPT is a queue-dynamics mismatch, not an interruption-detection gap.

### Q3–Q12. Summary

| Metric             | AMCSG (baseline) | SOTSS-MIN (frontier) | SOTSS-GSF (this run) |
|--------------------|-----------------|---------------------|---------------------|
| **Azure goodput/$**| 150,630         | 160,107             | **160,107** (+0.00% vs MIN) |
| Azure n_sla_safe   | 5823            | 5823                | 5823                |
| Azure safe?        | ✓               | ✓                   | ✓                   |
| **BurstGPT goodput/$** | 168,270    | 178,462 (gate=100%) | **178,462** (+0.00% vs MIN) |
| BurstGPT n_sla_safe | 5864           | 5860 (unsafe −4)    | 5860 (unsafe −4)    |
| BurstGPT safe?     | ✓               | ✗                   | ✗                   |

Note: SOTSS-MIN (gate=100%) on BurstGPT = 178,462 is *not* the BurstGPT frontier; the BurstGPT
frontier remains SOTSS gate=20% at 170,572 (safe: n_sla_safe=5864≥baseline).

### Q13. What to attempt next?

SOTSS-GSF is falsified. The binding constraints are now well understood:

1. **Close the oracle-simulation gap** — the 4-request shortfall on BurstGPT is due to
   queue-dynamics mismatch between oracle and evaluator. Fix: pass `baseline_n_sla_safe` from the
   FULL stochastic simulation (not from `_oracle_stochastic_response_times`) as the floor.
2. **Production-ready SOTSS-MIN** — extract the discipline into a serving-runtime path.
   At 2/5 consecutive non-frontier runs, the Five-Failure Rule prioritizes integration work.
3. **ShareGPT cross-validation** — validate SOTSS-MIN gate=100% on a third trace.

**Five-Failure Rule counter: 2 of 5 consecutive non-frontier runs.**

Results: `research/results/sotss_gsf_backtest_2026-06-23.{md,json}`
Tests: `tests/test_sotss_gsf.py` (49 tests, all passing)

---

## Run 2026-06-23 — C1-Protected Gate Sweep / C1PGS (NEGATIVE RESULT — hypothesis falsified)

### Q1. What was attempted?

**C1-Protected Gate Sweep (C1PGS):** Use Erlang-C gate=25% for per-tick c_schedule, but protect c=1
ticks with on-demand instances (0 spot) to eliminate the hypothesized spot-interruption cliff.
Hypothesis: c=1 OD at $2.00/hr < c=4 GSF spot at $3.20/hr on low-load ticks → lower cost and
higher goodput/$ than AMCSG gate=12.5%.

### Q2. Why it failed

**Mechanism was wrong.** The simulation contains:
```python
c_effective.append(max(1, c_demand + survived))
```
This guard prevents `c_effective=0` at c=1 ticks regardless of spot allocation. C1PGS and GSF
produce **identical effective capacity** at c=1. The SLA violations at gate=25% come from
Erlang-C over-optimism (M/M/c too lenient at high load), not spot interruptions.

**Cost error on BurstGPT.** With SLA=30s, AMCSG at gate=12.5% assigns c=2 (not c=4) on
low-load ticks. c=2 all-spot = $1.60/hr < c=1 OD = $2.00/hr. C1PGS costs MORE, not less.

### Q3–Q12. Summary

| Metric | AMCSG gate=12.5% | C1PGS gate=25% Azure | C1PGS gate=25% BurstGPT |
|--------|-----------------|---------------------|------------------------|
| goodput/$ | 150,630 / 168,270 | 153,960 (+2.21%) | 155,786 (-7.42%) |
| n_sla_safe | 5823 / 5864 | 5818 (-5) ✗ | 5859 (-5) ✗ |
| cost | $4.28 / $8.89 | $4.17 (-2.49%) | $9.59 (+7.80%) |
| SLA-safe? | baseline | **NO** | **NO** |
| North-star? | YES | NO | NO |

### Q13. What to attempt next?

C1PGS is falsified. The underlying idea (dynamic spot fraction per tick to address BurstGPT
cliff) is still valid conceptually, but the mechanism needs to be:

1. **Gate=25% safety at Erlang-C level** — not a spot-allocation fix; need smarter capacity
   planning (e.g., calibration correction for M/M/c → M/G/c, or trace-adaptive gate selection).
2. **SOTSS-MIN integration** — SOTSS-MIN already handles Erlang-C over-optimism by using an
   actual-token oracle. Focus on production-readying SOTSS-MIN rather than further gate exploration.
3. **ShareGPT cross-validation** — Validate SOTSS-MIN on a third trace.

**Five-Failure Rule counter: 1 of 5 consecutive non-frontier runs.**

Results: `research/results/c1pgs_backtest_2026-06-23.{md,json}`
Tests: `tests/test_c1pgs_policy.py` (39 tests, all passing)

---

## Run 2026-06-23 — ReplicaScalingPolicy (ARCHITECTURE CONVERGENCE — Phase 2/3)

### Q1. What currently limits Aurelius most?

**Architecture gap.** SOTSS-MIN provisioning decisions lived in the benchmark monolith, not in
`AureliusOptimizer`. This run closes that gap by implementing `ReplicaScalingPolicy` following the
Phase 2 extraction pattern (`serving_queue.py`).

| Policy | Status before | Status after |
|--------|--------------|--------------|
| energy | Implemented (Phase 1) | Implemented (Phase 1) |
| serving_queue | Implemented (Phase 2) | Implemented (Phase 2) |
| replica_scaling | Stub | **Implemented (Phase 2/3)** |
| placement | Stub | Stub |
| admission | Stub | Stub |

### Q2–Q13. Gap summary for architecture run

- **KPI impact**: 0% (parity-preserving extraction; 42 bit-identical parity tests)
- **Architecture**: `IMPLEMENTED_POLICIES = {"energy", "serving_queue", "replica_scaling"}`
- **Next research priority**: Dynamic spot fraction per tick (unchanged from SOTSS-MIN gap analysis)
- **Highest-EV**: Dynamic spot fraction (addresses BurstGPT cliff, enables gate=25%)

Results: `research/results/replica_scaling_policy_parity_2026-06-23.md`
Tests: `tests/test_replica_scaling_policy_parity.py` (42 tests, 0.41s)

---

## Run 2026-06-23 — SOTSS Gate Sweep / SOTSS-MIN (FRONTIER IMPROVEMENT — +6.29% vs AMCSG on Azure)

### Q1. What currently limits Aurelius most?

**Nothing new — SOTSS-MIN extends the frontier beyond SOTSS gate=20% by a further +4.64%.**

A systematic sweep of `aggressive_gate ∈ {20, 25, 30, 35, 40, 50, 75, 100}%` reveals that gate=100%
(SOTSS-MIN) yields the theoretical minimum-cost schedule on Azure. Starting from `c=1` for
under-loaded ticks, the oracle converges in 34 iterations, leaving 19 ticks cheaper than the
safe ceiling vs only 5 ticks at gate=20%.

| Trace | Condition | Goodput/$ | c_mean | n_sla_safe | vs AMCSG |
|-------|-----------|-----------|--------|-----------|----------|
| Azure LLM 2024 | AMCSG gate=12.5% | 150,630 | 4.458 | 5823 | baseline |
| Azure LLM 2024 | SOTSS gate=20% | 153,013 | 4.389 | 5823 | +1.58% |
| Azure LLM 2024 | **SOTSS-MIN gate=100%** | **160,107** | **4.194** | **5823** | **+6.29%** |
| BurstGPT HF | AMCSG gate=12.5% | 168,270 | — | 5864 | baseline |
| BurstGPT HF | **SOTSS gate=20% (safe max)** | **170,572** | 4.273 | 5864 | **+1.37%** |

North-star thresholds: 151,248 (Azure) / 121,680 (BurstGPT). Both EXCEEDED by large margins.

### Q2. What theoretically offers the largest gain beyond SOTSS-MIN?

SOTSS-MIN is the theoretical minimum for the greedy oracle on a fixed spot-price model.
Further gains require:
1. **Dynamic spot fraction per tick** — SOTSS uses fixed spot_fraction=0.95; letting high-load ticks
   use spot=0.0% (on-demand only) could eliminate spot-interruption penalties on heavy-tail requests.
2. **Multi-region spot arbitrage** — route ticks to cheapest regional spot market; expected 5-20%.
3. **Online SOTSS approximation** — replace oracle actual_tokens with live predicted_tokens; would
   generalize SOTSS-MIN to production without future knowledge.
4. **ShareGPT/LMSYS cross-validation** — test the gate=100% oracle concentration assumption on
   a third public trace.

### Q3. Which forecasts are weakest?

1. **Spot price $0.80/hr and interruption rate 10%/hr** — calculated priors; BurstGPT safety cliff
   at gate≥25% shows that interruption rate is the binding constraint, not price.
2. **BurstGPT safe maximum gate=20%** — safety cliff discovered: gates ≥25% fail stochastic
   evaluation due to spot interruptions on long-tail requests (p99=934 tokens, SLA=30s).
3. **Oracle concentration assumption** — 34 iters on Azure (violations on 19/72 ticks); may not
   hold for traces with flatter load profiles.

### Q4. Which optimizer decisions remain suboptimal?

1. **Offline oracle only** — SOTSS-MIN requires actual per-tick token counts (future knowledge).
2. **Fixed safe_gate=12.5% ceiling** — ceiling was validated for Azure; BurstGPT's p99 tail may
   require a lower ceiling at gate≥25%.
3. **Static spot_fraction=0.95** — no per-tick adjustment based on load intensity.

### Q5. Which workloads benefit least from SOTSS-MIN?

Traces where: (a) all ticks are overloaded (ρ≥1 everywhere — minimum stable c = safe ceiling),
(b) violations are uniformly distributed (oracle needs ≥72 iterations to converge), or
(c) heavy-tail p99 tokens push violation counts above the deterministic oracle's prediction
(the BurstGPT safety cliff at gate≥25% is exactly this case).

### Q6. Which research direction appears strongest?

**Online SOTSS approximation.** SOTSS-MIN shows the oracle loop works in 34 iterations on Azure.
The main barrier to production is replacing actual_tokens with live predictions. Predicted-vs-actual
token error bounds are bounded for existing LLM traces (~15% MAPE); the oracle should tolerate this.
Expected: maintains 80-90% of SOTSS-MIN's per-tick savings in online mode.

### Q7. What is the shortest path to another +1% gain?

**Dynamic spot fraction per tick.** Reduce spot_fraction from 0.95 to 0.0 on the 3-5 ticks where
spot interruptions cause BurstGPT violations. This would allow gate=25% on BurstGPT (gaining the
gap between 170,572 and 171,716) without violating SLA. Estimated implementation: 2-3 hours.

### Q8. What is the current north-star status?

- **Azure +500% north-star (151,248):** ACHIEVED. SOTSS-MIN: 160,107 (+5.86% margin above threshold).
- **BurstGPT +500% north-star (121,680):** ACHIEVED. SOTSS gate=20%: 170,572 (+40.2% margin).

### Q9. What would need to be true to maintain north-star on other traces?

For a new trace to achieve north-star via SOTSS-MIN: violations must be concentrated on ≤20%
of ticks; spot interruption rate must be ≤10%/hr; p99 token length must not be extreme relative
to SLA (BurstGPT barely passes at gate=20% because p99=934 tokens × SLA=30s is tight).

### Q10. Which assumptions might be wrong?

1. **Deterministic oracle is sufficient.** The BurstGPT safety cliff proves it is not always:
   the oracle converges deterministically at gate=25% but the stochastic GSF evaluation adds 3
   violations. The oracle needs a stochastic safety margin for heavy-tail traces.
2. **19 violation ticks represent a stable set.** If load varies between backtest runs (different
   request ordering), the set of 19 ticks could differ, making the converged schedule suboptimal.
3. **Gate=100% is globally optimal.** On a trace with uniform load (all ρ≈target_rho), starting
   from gate=100% may require more oracle iterations, eroding the advantage.

### Q11. Which benchmark weaknesses exist?

1. **Offline oracle bias** — SOTSS-MIN uses actual token counts; production would use predictions.
2. **Two public traces only** — Azure LLM 2024 and BurstGPT HF; ShareGPT/LMSYS needed.
3. **BurstGPT safety cliff is gate=25%, not 30%** — result artifact says cliff at gate=25%
   (3 extra violations), but gate=20% is safe (n_sla_safe=5864=baseline). This is a narrow margin.

### Q12. Which public datasets should be added?

1. **ShareGPT** — third LLM trace; test oracle concentration assumption.
2. **LMSYS Chatbot Arena** — fourth trace; heavier multi-turn workloads.
3. **AzurePublicDataset conversation traces** — longer context, different p99 profile.

### Q13. What should be attempted next?

1. **Dynamic spot fraction per tick** — vary spot_fraction based on load; addresses BurstGPT cliff.
2. **Online SOTSS approximation** — replace actual_tokens with live predicted_tokens.
3. **ShareGPT cross-validation** — verify gate=100% oracle concentration assumption.
4. **Stochastic oracle variant** — run oracle with GSF (stochastic) instead of FIFO (deterministic)
   to close the BurstGPT safety-cliff gap.

---

## Future Opportunity Ranking — Updated After Run 2026-06-23 (SOTSS-MIN Gate Sweep)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Dynamic spot fraction per tick (reduce spot on violation-prone ticks) | High | High | Addresses BurstGPT cliff; enables gate=25% on BurstGPT |
| 2 | Online SOTSS approximation (use live predicted_tokens) | High | High | Production deployment path; oracle loop confirmed in 34 iters |
| 3 | Stochastic oracle variant (run oracle with GSF not FIFO) | Medium | Medium | Closes deterministic/stochastic gap; more expensive per iter |
| 4 | ShareGPT/LMSYS cross-validation of SOTSS-MIN | Medium | High | Third/fourth public trace; tests oracle concentration |
| 5 | Cross-region spot arbitrage (SkyPilot/arXiv:2605.22778) | High | Medium | Multi-region cost model needed |

**Closed/characterized opportunities (SOTSS-MIN gate sweep):**
- SOTSS-MIN (gate=100%): **FRONTIER IMPROVEMENT** — 160,107 goodput/$ (Azure, +6.29% vs AMCSG)
- Gate sweep {20,25,30,35,40,50,75,100}%: monotonic improvement on Azure, all gates safe
- BurstGPT safety cliff: gate=20% safe (170,572 gpd/$), gate≥25% unsafe (3-4 extra violations)
- Oracle efficiency: 34 iterations, 19 ticks cheaper than ceiling (vs 5 ticks at gate=20%)
- c_mean reduction: 4.458 → 4.194 (−5.92% vs AMCSG gate=12.5%)

---

## Run 2026-06-23 — SOTSS (FRONTIER IMPROVEMENT — North-star +500% ACHIEVED, Azure +1.58% vs AMCSG)

### Q1. What currently limits Aurelius most?

**Nothing — the +500% north-star is now achieved on Azure LLM 2024.**

SOTSS (Simulation-Oracle Tick-Selective Schedule) closes the 0.41% gap by starting from
gate=20.0% c_schedule (maximum savings) and using a deterministic simulation oracle to
selectively increment c only on the 3 ticks causing SLA violations — leaving 5 ticks cheaper
than the AMCSG safe ceiling (gate=12.5%).

| Trace | Condition | Goodput/$ | Cost | n_sla_safe | vs oracle |
|-------|-----------|-----------|------|-----------|-----------|
| Azure LLM 2024 | AMCSG gate=12.5% | 150,630 | $4.2800 | 5823 | +497.5% |
| Azure LLM 2024 | **SOTSS gate=20%** | **153,013** | **$4.2133** | **5823** | **+507.0%** |
| BurstGPT HF | AMCSG gate=12.5% | 168,270 | — | 5864 | +729.7% |
| BurstGPT HF | **SOTSS gate=20%** | **169,030** | — | **5864** | **+733.5%** |

North-star thresholds: 6× oracle = 151,248 (Azure) / 121,680 (BurstGPT). Both ACHIEVED.

### Q2. What theoretically offers the largest gain beyond current state?

With north-star achieved, the research priorities shift:
1. **Cross-region spot arbitrage (SkyPilot-style)** — further cost reduction beyond the 1.56%
   achieved by SOTSS; expected 5-20% additional savings.
2. **Dynamic spot fraction** — adjust f per-tick based on load; SOTSS uses fixed spot_fraction=0.95.
3. **Integration into AureliusOptimizer** — wire SOTSS as a `ReplicaScalingPolicy` so the
   online serving path benefits from the oracle-guided capacity plan.

### Q3. Which forecasts are weakest?

1. **Spot price $0.80/hr and interruption rate 10%/hr** — calculated priors, not real-time data.
2. **Oracle applicability** — SOTSS is an offline oracle; online approximation needed for production.
3. **Generalization beyond Azure+BurstGPT** — only two public traces tested.

### Q4. Which optimizer decisions remain suboptimal?

1. **Offline oracle only** — SOTSS uses future knowledge (actual token counts) to compute the
   optimal per-tick c. An online approximation (using live predictions instead) would generalize.
2. **Static spot_fraction=0.95** — SOTSS doesn't adjust spot fraction per tick.
3. **Single-region model** — no cross-region cost arbitrage.

### Q5. Which workloads benefit least from SOTSS?

Traces where violations are spread across many ticks (not concentrated on 1–5 ticks). SOTSS's
oracle efficiency (3 iters to fix 60 violations) depends on violations being concentrated; if
violations are on every tick, SOTSS converges to the ceiling schedule.

### Q6. Which research direction appears strongest?

**Cross-region spot arbitrage.** North-star is achieved; the next multiplier comes from
multi-region routing — choosing the cheapest spot market dynamically per tick. arXiv:2605.22778
documents the methodology. Expected: 5-20% additional cost reduction on top of SOTSS savings.

### Q7. What is the shortest path to another +1% gain?

Add gate=25.0% as SOTSS aggressive start and check if oracle converges with even more ticks
cheaper. Gate=25.0% may give 6-8 ticks cheaper vs ceiling, with 4-5 oracle iterations needed.

### Q8. What is the current north-star status?

- **Azure +500% north-star (151,248):** ACHIEVED. SOTSS: 153,013 goodput/$ (+1.26% margin).
- **BurstGPT +500% north-star (121,680):** ACHIEVED (since GSF run). SOTSS: 169,030.

### Q9. What would need to be true to maintain north-star on other traces?

SOTSS oracle generalizes if: (a) violations are concentrated on a small number of ticks,
(b) those ticks can be fixed without exceeding the safe-gate ceiling, and (c) the net cost
after oracle fixes is lower than the safe-gate baseline.

### Q10. Which assumptions might be wrong?

1. **Violations are deterministically concentrated.** Oracle uses `_simulate_fifo_variable_c`
   (no spot interruptions). If real workloads have correlated spot reclamations, violations may
   spread to more ticks than the oracle anticipates.
2. **3 oracle iterations is sufficient.** For traces with more bursty load, more iterations
   may be needed, potentially closing all the gap between aggressive and safe gates.
3. **gate=20.0% is the best aggressive starting point.** A gate sweep (15, 17.5, 20, 25%)
   with SOTSS on top might find a better starting point.

### Q11. Which benchmark weaknesses exist?

1. **Oracle uses actual token counts** — this is an offline oracle, not a deployable online
   algorithm. An online approximation using live predictions is needed for production.
2. **Two public traces only** — Azure LLM 2024 and BurstGPT HF; further cross-validation
   needed (ShareGPT, LMSYS).
3. **Spot price and interruption rate are calculated priors** — no real-time cloud pricing data.

### Q12. Which public datasets should be added?

1. **ShareGPT** (third LLM trace) — cross-validate SOTSS oracle concentration assumption.
2. **LMSYS Chatbot Arena** (fourth trace) — additional cross-validation.
3. **Real cloud spot price traces** — enable time-varying cost backtests.

### Q13. What should be attempted next?

1. **Online SOTSS approximation** — replace oracle actual_tokens with live predicted tokens.
2. **Gate=25.0% as aggressive start** — test whether higher starting gate gives higher margin.
3. **Cross-region spot arbitrage** — model multi-region cost; expected 5-20% additional savings.
4. **ShareGPT cross-validation** — verify SOTSS oracle assumption generalizes to third trace.

---

## Future Opportunity Ranking — Updated After Run 2026-06-23 (SOTSS)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Online SOTSS approximation (use live predictions instead of actual tokens) | High | High | Offline oracle confirmed; online version is the deployment path |
| 2 | Gate=25.0% aggressive start for SOTSS (more savings, test oracle convergence) | Medium | High | Simple parameter sweep; 5-tick cheaper already proven at 20.0% |
| 3 | Cross-region spot arbitrage (SkyPilot/arXiv:2605.22778) | High | Medium | Multi-region cost model needed; methodology documented |
| 4 | ShareGPT/LMSYS cross-validation of SOTSS | Medium | High | Third/fourth public trace; tests oracle generalization |

**Closed/characterized opportunities (run 2026-06-23 SOTSS):**
- SOTSS oracle-loop: **FRONTIER IMPROVEMENT** — 153,013 goodput/$ (Azure, +507.0% vs oracle)
- Gate=15.0% aggressive start: +0.31% vs AMCSG (north-star NOT achieved, 0.10% short)
- Gate=20.0% aggressive start: +1.58% vs AMCSG (north-star ACHIEVED, +1.26% margin)
- BurstGPT cross-validation: 169,030 goodput/$ (+0.45% vs AMCSG, north-star YES)
- Oracle efficiency: 3 iterations to fix 60 violations, 5 ticks cheaper than ceiling

---

## Run 2026-06-27 — AMCSG Policy (MARGINAL IMPROVEMENT — +0.93%/+0.30% vs GSF, North-star gap 0.41%)

### Q1. What currently limits Aurelius most?

**Azure is 0.41% below the +500% north-star ceiling after all-spot + Erlang-C gate optimization.**
GSF at f=0.95 reached all-spot every tick (149,235 goodput/$, 1.35% gap). AMCSG gate sweep closed
0.93% of that gap via raising the Erlang-C gate 9.5%→12.5%:

| Trace | Condition | Goodput/$ | Cost | vs SLA-oracle |
|-------|-----------|-----------|------|---------------|
| Azure LLM 2024 | GSF(9.5%) baseline | 149,235 | $4.32 | +492.0% |
| Azure LLM 2024 | **AMCSG(12.5%)** | **150,630** | **$4.28** | **+497.5%** |
| BurstGPT HF | GSF(9.5%) baseline | 167,767 | $8.92 | +727.3% |
| BurstGPT HF | **AMCSG(12.5%)** | **168,270** | **$8.89** | **+729.7%** |

North-star thresholds: 6× oracle = 151,248 (Azure) / 121,680 (BurstGPT). BurstGPT already well
above; Azure gap = **618 goodput/$ (0.41%)**.

### Q2. What theoretically offers the largest gain beyond current state?

1. **AMCSG-LFC (fixed_c=3 on Azure):** Current fixed_c=4 sets the time-warp calibration.
   Lower fixed_c → fewer high-cost on-demand ticks at low load → lower c_mean → lower cost.
   Expected: c_mean 4.5→4.2, cost ~$4.18 → goodput/$ ~155,000 (+3.5%) — would clear the north-star.
2. **Dynamic per-tick gate:** Set gate proportional to (1 − ρ) per tick. At low load (ρ<<0.85),
   raise gate aggressively; at high load, use conservative 9.5%. Expected: ~+1% additional on Azure.
3. **Cross-region spot arbitrage (SkyPilot-style):** Not yet modeled; expected 5-20% cost reduction.

### Q3. Which forecasts are weakest?

1. **fixed_c calibration** — `calibrate_time_warp` uses fixed_c as the time-dilation factor.
   If fixed_c is too high relative to actual arrival rate, we over-provision.
2. **Erlang-C exponential service-time assumption** — actual GPU service times are heavy-tailed
   (not exponential). The M/M/c model adds conservatism that can be safely reduced up to gate=12.5%.
3. Spot price $0.80/hr and interruption rate 10%/hr remain calculated priors.

### Q4. Which optimizer decisions remain suboptimal?

1. **fixed_c=4 throughout.** The time-warp calibration factor is fixed across all load levels.
   At low-demand ticks, MCS still uses c=4 as the reference, over-provisioning.
2. **Flat gate=12.5% across all ticks.** Off-peak ticks with ρ<<0.85 could safely go higher.
3. **Single gate for all service-time distributions.** Heavy-tailed traces (BurstGPT) could
   tolerate higher gates than lighter-tailed ones.

### Q5. Which workloads benefit least from AMCSG?

Workloads where every tick is near or above ρ=0.85 — no room to raise the gate without SLA
violation. High-load, steady-state inference workloads see minimal benefit.

### Q6. Which research direction appears strongest?

**AMCSG-LFC (fixed_c=3 on Azure).** The gap to north-star is tiny (0.41%, 618 goodput/$).
Reducing fixed_c from 4 to 3 directly reduces the time-warp calibration denominator, cutting
per-tick on-demand cost at low load. Expected payoff: clear north-star with zero-violation guarantee.

### Q7. What is the shortest path to another +0.5% gain?

Reduce fixed_c 4→3 on Azure, keep gate at 12.5% (proven safe). If c_mean drops by 0.2 replicas
average, cost falls from $4.28 → ~$4.21, goodput/$ rises ~+1.5%, clearing the north-star.

### Q8. What is the current north-star status?

- **Azure +500% north-star (151,248):** NOT YET ACHIEVED. Current best = 150,630 (gate=12.5%).
  Gap = 618 goodput/$ = **0.41%**.
- **BurstGPT +500% north-star (121,680):** ACHIEVED since GSF run. Current best = 168,270 (well above).

### Q9. What would need to be true to achieve north-star on Azure?

Any one of: (a) fixed_c 4→3 reducing cost $4.28→~$4.20, (b) gate swept to ~14% safely on
a trace with less-heavy tails, (c) cross-region spot arbitrage cutting cost 1%, or (d) a
load-aware gate schedule that avoids p99>SLA at peak while exploiting slack at off-peak.

### Q10. Which assumptions might be wrong?

1. **Erlang-C conservatism is uniform across load.** At peak ρ (ρ=0.90+), even 12.5% gate
   may be too aggressive — we haven't tested behavior under load spikes.
2. **n_sla_safe drop at gate≥15% is fully explained by spot interruptions.** Could also be
   a FIFO queue depth crossing a threshold from queueing theory.
3. **fixed_c=4 is the right calibration baseline.** If actual long-run arrival rate corresponds
   to fixed_c=3.5, then c=4 over-provisions and c=3 under-provisions.

### Q11. Which benchmark weaknesses exist?

1. **Erlang-C gate search is coarse (1.5% steps).** Finer grid around 12.5%–14% on Azure
   might find a safe improvement without full-step violations.
2. **p99 reported; tail not fully characterized.** The n_sla_safe criterion is more honest
   but the exact tail shape (p99.9, max) is not reported.

### Q12. Which public datasets should be added?

Same as prior runs: real cloud spot price traces, spot interruption histories.

### Q13. What should be attempted next?

1. **AMCSG-LFC** — fixed_c=3 on Azure with gate=12.5%; test whether north-star is cleared.
2. **Fine gate grid** — test gates 13.0%, 13.5%, 14.0% on Azure with n_sla_safe criterion.
3. **Dynamic load-aware gate** — gate = 9.5% + 5%*(1-ρ/0.85) per tick.

---

## Future Opportunity Ranking — Updated After Run 2026-06-27

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | AMCSG-LFC: fixed_c=3 on Azure (target: clear +500% north-star) | High | High | 0.41% gap — likely achievable with c_mean drop |
| 2 | Fine gate grid (13.0%–14.0% on Azure) | Medium | High | Coarse grid may have missed a safe improvement |
| 3 | Dynamic per-tick gate (load-aware, gate ∝ 1-ρ) | Medium | Medium | Avoids peak SLA violations, exploits off-peak slack |
| 4 | Cross-region spot arbitrage (SkyPilot-style) | High | Medium | Multi-region cost model needed |

**Closed/characterized opportunities (run 2026-06-27):**
- Erlang-C gate exploitability: **CHARACTERIZED** — max safe gate = 12.5% (+3.0% above 9.5%). Gate≥15% causes SLA violations.
- AMCSG best Azure: +0.93% vs GSF baseline (150,630 goodput/$). North-star gap: 0.41%.
- AMCSG best BurstGPT: +0.30% vs GSF baseline (168,270 goodput/$). North-star already achieved.

---

## Canonical Optimizer Phases 1–3 — Unification Routing (STRUCTURAL — NO BEHAVIOR CHANGE)

> Architecture-unification parity steps, not optimization runs. See
> `research/OPTIMIZER_UNIFICATION_PLAN.md` and the Phase 1/2/3 parity reports.

- **Phase 1**: `AureliusOptimizer(policy="energy")` wraps `JobScheduler` (verbatim delegate).
- **Phase 2**: `AureliusOptimizer(policy="serving_queue")` exposes the extracted
  abs-conformal SRPT discipline (moved out of the benchmark monolith).
- **Phase 3**: 5 public benchmark entry points now route through `AureliusOptimizer`
  (4 energy benchmarks + the serving shim). **0% energy + serving KPI drift**
  (canonical golden snapshot reproduced; abs-conformal JSON byte-identical).
- Azure/BurstGPT trace replays are **not** routed yet — they are a per-tick
  replica-provisioning autoscaler (a `ReplicaScalingPolicy`, future phase), not the
  energy/serving policies.

**Governance recorded (binding for future optimization claims):** (1) report
Current-Main vs Best-Aurelius vs Candidate — never FIFO-only; a candidate is a
frontier improvement only if it beats the best validated AureliusOptimizer config;
(2) optimizer-first — every optimization must map to a real production decision
(decision-time info only, no actual-token leakage); (3) policy-combination search
with interaction effects once policies share a workload (feasible after Phase 1b
unified replay; `energy` and `serving_queue` are disjoint workload classes today).

## Run 2026-06-24 — AFMS Policy (FRONTIER IMPROVEMENT — +10.1%/+13.1% vs Static 70%)

### Q1. What currently limits Aurelius most?

**The static spot-fraction formula has a rounding artifact.** `round(0.70*c)` keeps 2 on-demand replicas at c=6,7,8, wasting $0.020/tick vs the minimum safe floor. AFMS (`max(round(0.70*c), c-1)`) eliminates this with zero SLA regression:

| Trace | Condition | Goodput/$ | Cost | vs SLA-oracle |
|-------|-----------|-----------|------|---------------|
| Azure LLM 2024 | FIFO+MCS static 70% | 102,009 | $6.32 | +304.7% |
| Azure LLM 2024 | **FIFO+MCS AFMS** | **112,316** | **$5.74** | **+345.6%** |
| BurstGPT HF | FIFO+MCS static 70% | 118,580 | $12.62 | +484.7% |
| BurstGPT HF | **FIFO+MCS AFMS** | **134,093** | **$11.16** | **+561.2%** |

North-star maintained on both traces (AFMS ≥ 100,832 Azure / 81,120 BurstGPT). Zero SLA violations.

### Q2. What theoretically offers the largest gain beyond the current state?

1. **Cross-region spot arbitrage** (SkyPilot-style) — route to cheapest spot region per tick; expected 5-20% additional cost reduction on top of AFMS.
2. **Dynamic spot fraction** — adjust f per tick based on spot market price signal; AFMS fixes the on-demand floor at 1, but the fraction could be raised to 80-90% when spot is cheapest.
3. **Multi-floor AFMS** — AFMS with min_ondemand=2 at c≥10 for extra safety in very large schedules.

### Q3. Which forecasts are weakest?

Spot price ($0.80/hr) and interruption rate (0.10/hr) remain calculated priors, not real-time data. Result is directionally correct but would need real spot price traces for production validation.

### Q4. Which optimizer decisions remain suboptimal?

1. **Spot fraction is fixed at 70%** — even with AFMS's absolute floor, the fraction is not dynamically adjusted to market conditions.
2. **Single-region model** — no cross-region cost arbitrage.
3. **Static interruption model** — i.i.d. per-tick interruptions; correlated spot reclamation events not modeled.

### Q5. Which workloads benefit least?

Schedules with c≤5 throughout (no c≥6 ticks) see zero AFMS benefit vs static 70% (cost is identical). Also, c=1 ticks are $0.020/tick more expensive under AFMS. Low-capacity, uniformly-small schedules do not benefit.

### Q6. Which research direction appears strongest?

**Cross-region spot arbitrage.** AFMS is a local formula fix (within-tick). The next multiplier comes from multi-region routing — choosing the cheapest spot market dynamically per tick. arXiv:2605.22778 documents the methodology directly.

### Q7. What is the shortest path to another +10% gain?

Raise spot_fraction from 0.70 to 0.80 at c≥6 while keeping the AFMS absolute floor at 1. Expected: additional ~5% cost reduction, compounding to ~+15% vs static 70%.

### Q8. What is the shortest path to +500% vs SLA-oracle (new north-star candidate)?

From Azure AFMS (+345.6% vs oracle) to +500% threshold (6× oracle = 151,248): requires dynamic spot fraction (80-90%) and cross-region arbitrage. BurstGPT already exceeds 6× oracle at 134,093 vs 6× threshold of 121,680.

### Q9. What would need to be true to achieve further north-star improvement?

- 5× oracle (Azure, 126,040): achievable by raising spot_fraction to ~0.78 with AFMS floor.
- 6× oracle (Azure, 151,248): requires cross-region spot arbitrage AND higher spot_fraction.

### Q10. Which assumptions might be wrong?

1. **Spot price $0.80/hr is stable.** Real spot prices fluctuate; during demand spikes GPU spot can exceed on-demand price briefly.
2. **i.i.d. interruption model (10%/hr).** Real spot interruptions can be correlated across ticks.
3. **c=1 ticks are rare.** If c=1 dominates the schedule, AFMS is net more expensive than static.

### Q11. Which benchmark weaknesses exist?

1. **Calculated priors on spot price and interruption rate.** No real-time cloud pricing data used.
2. **Single-region model.** No cross-region cost arbitrage modeled.
3. **Static interruption model.** Correlated spot reclamation not captured.

### Q12. Which public datasets should be added?

1. **Real cloud spot price traces** (AWS/GCP/Azure pricing history APIs) — enable time-varying cost backtests.
2. **Spot interruption history** (AWS CloudWatch GPU instance reclamation logs) — validate the 10%/hr prior.

### Q13. What should be attempted next?

1. **Dynamic spot fraction with AFMS floor** — adjust f from 0.70 to 0.80-0.90 per tick based on MCS signal.
2. **Cross-region spot arbitrage** — multi-region cost model; select cheapest spot region per tick within latency budget.
3. **Integration into AureliusOptimizer** — wire AFMS as a `ReplicaScalingPolicy` in the canonical optimizer.

---

## Future Opportunity Ranking — Updated After Run 2026-06-24

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Dynamic spot fraction with AFMS floor (0.70→0.80-0.90) | High | High | AFMS floor established; fraction sweep is straightforward |
| 2 | Cross-region spot arbitrage (SkyPilot/arXiv:2605.22778) | High | Medium | Multi-region cost model needed; methodology documented |
| 3 | Wire AFMS as `ReplicaScalingPolicy` in AureliusOptimizer | Medium | Medium | Integration pending; AFMS validated |
| 4 | Real spot price traces (time-varying cost model) | Medium | Medium | Would validate calculated prior; requires cloud API access |

**Closed/characterized opportunities (run 2026-06-24):**
- Static 70% rounding artifact at c=6,7,8: **FIXED by AFMS** — +10.1% (Azure), +13.1% (BurstGPT) vs static 70%
- AFMS SLA safety: **CONFIRMED** — 0 SLA violations, completion_rate=1.0000 both traces, north-star maintained

---

## Run 2026-06-23B — Spot Fleet MCS (NORTH STAR ACHIEVED — FRONTIER IMPROVEMENT)

### Q1. What currently limits Aurelius most?

**North-star ACHIEVED on both public traces.** Spot/preemptible pricing overlay on FIFO+MCS:

| Trace | Condition | Goodput/$ | vs SLA-oracle |
|-------|-----------|-----------|---------------|
| Azure LLM 2024 | FIFO+MCS on-demand | 59,694 | +136.8% |
| Azure LLM 2024 | **FIFO+MCS spot fleet** | **102,009** | **+304.7% ✓** |
| BurstGPT HF | FIFO+MCS on-demand | 55,800 | +175.1% |
| BurstGPT HF | **FIFO+MCS spot fleet** | **97,595** | **+381.2% ✓** |

North-star threshold: 4× SLA-oracle = 100,832 (Azure) / 81,120 (BurstGPT).

### Q2. What theoretically offers the largest gain beyond north-star?

Now that +300% vs SLA-oracle is achieved, the next research dimension is:
1. **Higher spot_fraction (80–90%)** — achieves +370% (Azure) with 80% spot
2. **Combined queue discipline** — abs-conformal in MCS context showed −6pp vs FIFO; worth retesting at lower rho where queue discipline matters
3. **Cross-cloud spot arbitrage** (SkyPilot-style) — route to cheapest spot region dynamically

### Q3. Which forecasts are weakest?

Queue discipline has no impact in MCS context (run-2026-06-23 finding stands). Pricing model is now the primary decision variable.

### Q4. Which optimizer decisions remain suboptimal?

1. **Spot fraction is static** — in this model spot_fraction is fixed per tick. Dynamic spot fraction (increase when price is low, reduce at peak) could improve further.
2. **Queue discipline in MCS context** — FIFO marginally better than abs-conformal. A queue-depth-conditioned policy (SRPT when depth > k) may recover value at low MCS capacity.

### Q5. Which workloads benefit least?

Very long-service-time workloads where interruptions cause non-negligible SLA violations. In practice, LLM requests with service times close to the SLA budget are most at risk from spot interruptions.

### Q6. Which research direction appears strongest?

**Dynamic spot fraction** — adjust spot_fraction per tick based on spot market price (cheaper hours → more spot) and queue depth. This would improve goodput/$ beyond the static 70% operating point.

### Q7. What is the shortest path to another +10% gain?

Increase spot_fraction from 0.70 to 0.80: achieves +370% vs SLA-oracle (Azure), +15pp more.

### Q8. What is the shortest path to +300% vs SLA-oracle (already achieved)?

Done. Operating point: spot_fraction=0.70, spot_price=$0.80/hr, p_int=0.10/hr.

### Q9. What would need to be true to achieve +400% vs SLA-aware?

From FIFO+MCS spot at 97,595 (BurstGPT) to +400% threshold (5× oracle = 101,400):
- Already achieved: +381.2% on BurstGPT
- For +400%: need goodput/$ ≥ 101,400 — increase spot_fraction to ~0.72 or reduce p_int

### Q10. Which assumptions might be wrong?

1. **Interruption model** — we model each tick independently. In practice, AWS/Azure announce interruptions 2 minutes in advance (opportunity to drain). This would further reduce SLA violations.
2. **Retry model** — we assume interrupted requests are lost. If clients retry, this increases load. A retry multiplier of 1.01× would increase arrivals by 0.7% — negligible.
3. **Spot pricing is stable** — real spot prices vary hour-to-hour. A dynamic pricing model would be more accurate.

### Q11. Which benchmark weaknesses exist?

1. **Single spot price assumed** — real spot prices fluctuate. The model assumes a fixed spot_price. A time-varying spot price model would be more realistic.
2. **No region-switching** — our model doesn't include migrating between regions when spot prices spike.
3. **Interruption model** — Binomial per tick is an approximation; real spot interruptions can be correlated (AWS reclaims entire instance types simultaneously).

### Q12. Which public datasets should be added?

1. **Real cloud spot price traces** (AWS/GCP/Azure spot price history APIs) — would enable actual time-varying cost backtests.
2. **Spot interruption history** — AWS CloudWatch spot interruption data for GPU instance types.

### Q13. What should be attempted next?

1. **Dynamic spot fraction** — adjust f per tick based on queue depth + MCS signal. When queue is short (MCS already maintaining SLA at c_mean<4), reduce spot fraction → lower cost without SLA risk.
2. **Cross-region spot arbitrage** — route to cheapest spot region that meets SLA latency requirement.
3. **Integration into AureliusOptimizer** — wire spot pricing model into the canonical optimizer's cost objective so it influences scheduling decisions at runtime.

---

## Run 2026-06-23 — Joint Economic × Queue TRUE Compound (NORTH STAR NOT ACHIEVED)

### Q1. What currently limits Aurelius most?

**North-star NOT achieved.** All conditions under provisioned-hours cost, compared against
SLA-aware oracle as the correct north-star baseline:

| Condition | Goodput/$ | vs SLA-oracle |
|-----------|-----------|---------------|
| FIFO+fixed | 11,183 | −56% |
| SLA-aware oracle (north-star base) | 25,208 | 0% |
| Abs-conformal+fixed | 46,199 | +83% |
| FIFO+MCS | 59,694 | **+137%** |
| Abs-conformal+MCS (TRUE) | 58,323 | **+131%** |
| North-star threshold (4× SLA-oracle) | 100,832 | +300% |

Current best (FIFO+MCS) is at 59% of the north-star threshold. Economic factor still needed: **1.73×**.

Three binding constraints identified this run:
1. **MCS raises cost, not lowers it** — c_mean=4.5 on diurnal trace (+12.5% GPU-hours vs fixed c=4). Prior run-z estimate of "1.2575× savings" was wrong; used FALLBACK_TOKENS_PER_S=2500 physics.
2. **Queue discipline (abs-conformal) adds nothing in MCS context** — FIFO+MCS (+137%) slightly beats abs+MCS (+131%). Preemption overhead dominates when MCS keeps queue short.
3. **+422% vs FIFO+fixed was a misleading framing** — FIFO+fixed p99=732s is catastrophically bad; it's the weakest baseline. All gains should be measured vs SLA-aware oracle.

### Q2. What theoretically offers the largest gain?

**Spot/preemptible pricing overlay** to reduce fleet cost by 42% (reaching north-star from the
current 59,694-best). This is the only remaining lever that can reach 4× SLA-aware oracle.

### Q3. Which forecasts are weakest?

1. **MCS "cost savings" claim** — CONFIRMED WRONG this run. MCS raises cost on diurnal traces.
2. **Abs-conformal benefit in MCS context** — CONFIRMED ZERO. FIFO+MCS is slightly better.
3. **Prior run-z economic factor (1.2575×)** — was from wrong physics model.

### Q4. Which optimizer decisions remain suboptimal?

1. **Fleet cost** — provisioning at fixed_c=4 or MCS pays $9.60–$10.80 for 72 minutes; spot
   pricing would cut this to ~$5–6 and push FIFO+MCS above north-star.
2. **Queue discipline in MCS context** — neither FIFO nor abs-conformal is optimal; a
   depth-conditioned policy (SRPT only when queue_depth > k) may recover some value.

### Q5. Which workloads benefit least?

**Any workload with enough provisioned capacity.** When MCS scales up to meet peak demand,
the queue discipline becomes irrelevant. Abs-conformal only helps when capacity is fixed and
requests queue up for long periods.

### Q6. Which research direction appears strongest?

**Spot/preemptible instance pricing** — reduce fleet cost by ~42%. With FIFO+MCS at 59,694
and provisioned cost at $10.80, a 1.73× cost reduction puts compound goodput/$ above 100,832.

### Q7. What is the shortest path to another +10% gain?

Add spot pricing overlay: FIFO+MCS at 59,694 with cost reduced from $10.80 to $9.80 (+10.2%).

### Q8. What is the shortest path to +300% vs SLA-aware (north-star)?

Spot pricing reducing MCS fleet cost from $10.80 to ~$6.24 (−42%). Achievable with A100/H100
spot instances on major cloud providers (typically −40−70% vs on-demand).

### Q9. What would need to be true to achieve +300% vs SLA-aware?

From FIFO+MCS at 59,694 to north-star at 100,832:
- Required: 59,694 × 1.69 = 100,832 → either 1.69× more goodput OR 1.69× cost reduction
- Cost path: MCS fleet from $10.80 → $6.39 (−40.8%, feasible with spot pricing)
- Goodput path: another 69% SLA-compliant tokens — would require better utilization of c=8 peak

### Q10. Which assumptions might be wrong?

1. **MCS gate at 9.5% is optimal** — not swept. A looser gate (e.g. 15%) might reduce peak
   replica counts and lower cost while still meeting SLA.
2. **Erlang-C M/M/c is accurate for LLM serving** — real systems have deterministic service
   times (not exponential) and continuous batching that changes the physics.

### Q11. Which benchmark weaknesses exist?

1. **BurstGPT cross-validation** — joint compound not run on BurstGPT HF yet.
2. **No spot pricing model** — the 1.73× cost factor needed is assumed achievable via spot; not measured.
3. **MCS gate sensitivity** — 9.5% threshold not swept; optimal value unknown.

### Q12. Which public datasets should be added?

BurstGPT HF joint compound — second trace cross-validation.

### Q13. What should be attempted next?

1. **Spot pricing overlay** — model stochastic spot pricing on MCS fleet to reduce cost 40%+.
2. **MCS gate sweep** — try 5%, 9.5%, 15%, 20% timeout gates; lower gate may reduce c_mean.
3. **BurstGPT HF joint compound** — cross-validate on second trace.

---

## Audit 2026-06-22 — Optimizer Architecture Audit (STRUCTURAL — NO CODE CHANGED)

> Architecture-coherence audit, not a benchmark run. Full evidence in
> `research/OPTIMIZER_ARCHITECTURE_AUDIT.md`; target architecture in
> `research/CANONICAL_AURELIUS_OPTIMIZER.md`; migration in
> `research/OPTIMIZER_UNIFICATION_PLAN.md`. No optimizer, benchmark, replay, or
> evaluation logic was modified.

### A1. Is Aurelius a coherent optimization system?
**No.** It is two eras of optimization plus a large research/shadow periphery.
There are **≥3 independent decision engines** (`JobScheduler` energy-cost core;
`ConstraintAwareEngine`; the inline replica-provisioning policies in
`traces/backtest.py` that produce the public LLM leaderboard; the discrete-event
SRPT+conformal disciplines in `benchmarks/srtf_serving_backtest.py`) and **4
independent replay loops**, with **no shared optimization core** between the
energy world and the serving world.

### A2. What is the single biggest structural gap?
**The headline metric cannot be influenced at runtime.** The Era-2 serving
disciplines produce every recent headline (+313%/+557% vs FIFO) but live only in
a 6,628-LOC benchmark file that imports nothing from `optimization/`,
`frontier/`, or the runtime replay engine — `grep -rln srtf_serving_backtest
aurelius/ | grep -v benchmarks/` → empty. This matches the long-standing
self-reported gap "serving queue not wired into runtime — integration pending."

### A3. Which modules are disconnected from runtime decisions?
The Era-2 serving research; all 5 `frontier/` families (none default-on; none
imported by any benchmark; EVAL_WORKLOAD + BATCH_INFERENCE are dead duplicates);
the entire `forecasting/` stack (advisory-only by contract,
`forecasting/__init__.py:52`); `residency/` (standalone, `MUTATION_ALLOWED=False`);
the 3 shadow modules (NEUTRAL/HURT/regressed, `enabled=False`); migration/MPC in
`JobScheduler` (test-only). No default path can mutate real infrastructure.

### A4. Which modules duplicate each other?
Frontier EVAL_WORKLOAD/BATCH_INFERENCE (copy-paste of BASE); **3 inline conformal
calibrators** in `srtf_serving_backtest.py` (+ a 4th in `cara_latency_calibration.py`);
**2 DCGM connectors**; **4 replay loops**; the term "constraint_aware" denotes
three different implementations.

### A5. What should change (and what must not)?
Converge on one optimizer / one objective (SLA-safe goodput/$) / one replay
engine, promoting the SRPT+conformal discipline into a real (shadow-gated)
serving path and the frontier into a constraint. Do **not** integrate the 3
shadow modules (negative evidence). Do **not** change benchmark definitions,
public replay logic, evaluation infra, or the pinned energy core. Migration is
phased and 0%-delta-gated; see the plan. This run produced **review documents
only** and is **not merged**.

---

## Run 2026-06-22-z (ML Prior Null) — ML Prior under Abs-Conformal (Honest Null — Prediction-Accuracy Lever Closed)

### Q1. What currently limits Aurelius most?

**Not prediction accuracy — confirmed by elimination this run.** A clean 2×2 (prior ×
calibrator) on BurstGPT HF shows the ML-HGB prior does NOT beat the trivial running median
even under the abs-conformal calibrator that uncapped α: ML+abs = 42,810.7 goodput/$ vs
global+abs = 42,901.6 (**−0.21%, null**). The +26% abs-conformal gain is **prior-agnostic**
(ML+abs vs ML+rel = +26.05%, identical to global's rel→abs jump). The binding constraint is
the **compound economic × queue integration** (still unmeasured end-to-end), not the prior.

### Q2. What theoretically offers the largest gain?

**Compound economic × queue scheduling end-to-end backtest** (unchanged from run -y, now
reinforced). This run eliminates "better output-length predictor" as a lever: the queue
component already extracts the available scheduling signal prior-agnostically via
abs-conformal. The only remaining multiplier is the economic (energy/carbon-cost) axis.

### Q3. Which forecasts are weakest?

1. **Economic scheduling multiplier** — never compounded with queue gains end-to-end.
2. **ChatGPT intra-class variance** — irreducible by `model_id`/`input_tokens` features
   (MAE only −2.5% from ML, CV actually rises 15.3%→43.0%). No causal predictor closes it.
3. **Queue-economic integration** — completely unverified end-to-end.

### Q4. Which optimizer decisions remain suboptimal?

1. **Compound routing**: energy-cost optimization and queue discipline still independent.
2. **BurstGPT 11.7% oracle gap is now characterized as irreducible** for running-statistics
   AND ML priors under the current feature set — it is intra-class variance, not a
   predictor-quality gap.

### Q5. Which workloads benefit least?

**BurstGPT ChatGPT "surprise-long" requests.** Predicted short (median≈7) but occasionally
800+ tokens; no arrival-time feature distinguishes them, so both running-median and ML
priors mispredict identically.

### Q6. Which research direction appears strongest?

**End-to-end compound economic × queue scheduling backtest.** With both the
prediction-accuracy lever (this run) and the calibrator lever (run -x, near-optimal at 88-98%
retention) exhausted on the queue side, the economic axis is the sole remaining path to
+300% vs SLA-aware.

### Q7. What is the shortest path to another +10% gain?

Compound economic × queue. A conservative economic multiplier on top of the +83-112%
abs-conformal advantage over oracle SLA-aware would exceed the north-star.

### Q8. What is the shortest path to another +50% gain?

Same — compound economic × queue with an economic multiplier ≥ ~1.3×. The queue side is
exhausted: neither a better prior (this run) nor a better calibrator metric (run -x ceiling)
yields further BurstGPT goodput.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

Unchanged from run -y: queue alone reaches +83%/+112% vs oracle SLA-aware. Reaching +300%
requires the economic multiplier. This run removes the alternative hope (ML predictor closing
the oracle gap) — abs-conformal already sits at 88% BurstGPT / 98% Azure retention and the
ML prior does not push it higher.

### Q10. Which assumptions might be wrong?

1. **"A better output-length predictor closes the BurstGPT gap" — CONFIRMED WRONG this run.**
   ML+abs = −0.21% vs global+abs.
2. **Independence assumption for the compound estimate** — still untested end-to-end.
3. **The abs-conformal 88% BurstGPT retention is stable across load** — not swept here.

### Q11. Which benchmark weaknesses exist?

1. **No end-to-end economic × queue trace** — the compound +130-166% vs SLA-aware remains
   a calculated compound, not a joint simulation.
2. **ML prior limited to `model_id` + `input_tokens`** — richer prompt-text features
   untested, but this run suggests the ceiling is intra-class variance, so the
   expected upside is low.
3. **sklearn is an optional dependency** — the ML prior degrades to the running-median
   fallback without it (handled gracefully; the abs-conformal path is unaffected).

### Q12. Which public datasets should be added?

ShareGPT (third LLM trace) — but with the prediction-accuracy lever now closed,
priority is lower. Focus: end-to-end economic × queue joint backtest on Azure + BurstGPT.

### Q13. What should be attempted next?

**Immediate (next run):**
1. **Spot/preemptible pricing overlay** — model stochastic spot pricing to close the
   economic factor gap (current 1.2575× → target 2.18×). Public spot price data exists.
2. **Temporal energy-cost shifting** — schedule workloads to low-cost time windows using
   existing energy price data in the repository.
3. **Joint economic × queue simulator** — implement a combined objective that co-optimizes
   queue ordering and provisioning cost in a single replay.

---

## Run 2026-06-22-z — Compound Economic × Queue Scheduling (FRONTIER UNDERSTANDING)

### Q1. What currently limits Aurelius most?

**The compound economic × queue system is characterized and the north-star gap
is fully quantified.** Compound result vs oracle SLA-aware:
- Azure: +130.47% (queue +83.27% × economic 1.2575× cost factor)
- BurstGPT: +166.02% (queue +111.55% × economic 1.2575× cost factor)

The north-star (+300% vs oracle SLA-aware) is NOT achieved. The binding constraint
is the economic provisioning factor: current 1.2575× (−21.2% GPU-hours), but 2.18×
(−54.2% GPU-hours) is needed for Azure north-star.

### Q2. What theoretically offers the largest gain?

**More aggressive economic optimization** targeting ≥ 50% GPU-hour savings via:
1. Spot/preemptible instances (up to -70% cost in some markets)
2. Aggressive cross-region arbitrage (cheapest compute-hour globally)
3. Carbon-aware scheduling during renewable surplus windows

Alternatively, an ML predictor to close the BurstGPT 11.7% oracle gap — this
would reduce the economic factor threshold from 2.18× to ~2.05×.

### Q3. Which forecasts are weakest?

1. **Economic multiplier** — current 1.2575× is from the Azure weekly provisioning
   benchmark; actual achievable savings with spot/preemptible could be 2-3×.
2. **BurstGPT 11.7% oracle gap** — ML predictor could push factor needed to 1.89×.
3. **Independence of layers** — confirmed orthogonal; no forecast risk here.

### Q4. Which optimizer decisions remain suboptimal?

1. **Economic provisioning** — only -21.2% GPU-hours achieved; need -54% for north-star.
2. **ML predictor** — running-median prior has structural ceiling at 88.3% (BurstGPT).

### Q5. Which workloads benefit least?

**Uniform output-length distributions** where SRPT ordering provides minimal
goodput benefit. Economic optimization is workload-agnostic (cost-side only).

### Q6. Which research direction appears strongest?

**Spot/preemptible instance scheduling for economic layer.** The queue layer is
near-optimal (97.8% oracle retention on Azure). The only path to north-star is
a 2.18× economic factor, achievable with spot pricing in major cloud regions.

### Q7. What is the shortest path to another +10% gain?

An ML predictor (HGB/CARA) to close the BurstGPT oracle gap from 88.3% to
95%+. This reduces the BurstGPT economic factor threshold from 1.89× to ~1.75×.
It also directly improves the compound result by ~15% on BurstGPT.

### Q8. What is the shortest path to another +50% gain?

More aggressive economic optimization. Current: 1.2575× (−21.2% GPU-hours).
Target for +50% compound improvement: 1.2575 × 1.50 = 1.89× (−47% GPU-hours).
This is achievable with spot/preemptible (typically −40-70% in major markets).

### Q9. What would need to be true to achieve +300% vs SLA-aware?

From run-z direct measurement:
- Queue alone: +83.27% vs oracle SLA-aware (near queue ceiling at 97.8% oracle retention)
- Economic factor needed: 2.18× (−54.2% GPU-hours, vs current 1.26× / −21.2%)
- Combined: abs_conformal × 2.18 = 55,097 × 2.18 = 120,111 goodput/$ ≈ 4.0× oracle SLA-aware

Path: aggressive economic optimization (spot/preemptible) + current abs-conformal queue.
No ML predictor needed for north-star if economic factor reaches 2.18×.

### Q10. Which assumptions might be wrong?

1. **Economic factor applies uniformly** — spot pricing may increase tail latency
   (preemptions), reducing SLA-compliant tokens. The compound could be sub-multiplicative.
2. **Azure weekly → SRTF sample transfer** — the 1.2575× factor is measured on the
   full week-long provisioning trace; SRTF simulation uses 5,880 requests.
3. **Independence of layers** — verified conceptually, not with an integrated simulator.

### Q11. Which benchmark weaknesses exist?

1. **No integrated compound simulator** — the compound uses a measured economic factor
   from a separate trace, applied analytically. A true end-to-end integrated simulator
   would run both layers on the same requests simultaneously.
2. **Spot pricing SLA risk not modeled** — preemptible instances can increase
   tail latency; the economic factor assumes SLA-compliant tokens are unchanged.

### Q12. Which public datasets should be added?

ShareGPT (third LLM trace for cross-validation). Also: spot pricing trace from a
major cloud provider to measure the achievable economic factor empirically.

### Q13. What should be attempted next?

**Immediate (next run):**
1. ML predictor (HGB/CARA) with abs-conformal — close BurstGPT 11.7% oracle gap,
   reducing economic factor threshold to ~1.89×, improving compound result.
2. ShareGPT cross-validation — validate the +83-112% queue gains on a third trace.

**Medium-term:**
- Aggressive economic optimization (spot/preemptible simulation) targeting 2.18×.
- Wire abs-conformal into serving runtime (all gates closed, integration pending).

---

## Future Opportunity Ranking — Updated After Run -z

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | ML predictor (HGB/CARA) with abs-conformal | High | Medium | Close BurstGPT 11.7% oracle gap; reduce economic factor threshold |
| 2 | ShareGPT as third public LLM trace | Medium | Medium | Cross-validate queue gains; compound applies identically |
| 3 | Aggressive economic optimization (spot/preemptible) | Very High | Medium | 2.18× factor needed; spot typically −40-70% cost |
| 4 | Wire abs-conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |

**Closed/characterized opportunities (run -z):**
- Compound economic × queue: **CHARACTERIZED** — +130% (Azure) / +166% (BurstGPT) vs oracle
  SLA-aware; north-star gap quantified: need 2.18× economic factor (vs current 1.26×)
- run-t over-estimate: **CORRECTED** — 2.25-3.1× over-estimate due to double-counting
  SLA-aware component in multiplicative compound formula

---

## Run 2026-06-22-y — SLA-aware vs Abs-Conformal Head-to-Head (FRONTIER UNDERSTANDING)

### Q1. What currently limits Aurelius most?

**The queue-only component is near-optimal; the north-star gap is now explicitly
characterized.** Abs-conformal (live prior) achieves:
- Azure: +83.27% vs oracle SLA-aware, 97.8% oracle retention
- BurstGPT: +111.55% vs oracle SLA-aware, 88.3% oracle retention

The queue-scheduling component is near its ceiling (97.8% oracle retention on Azure).
The binding constraint for the north-star (+300% vs SLA-aware) is the absence of the
**compound economic × queue scheduling** integration. Queue alone reaches +83-112%.

### Q2. What theoretically offers the largest gain?

**Compound economic × queue scheduling end-to-end backtest.** The queue component
is now near-optimal (97.8% oracle retention); the economic component (energy/carbon
cost optimization) adds an additional multiplier. Independence estimate:
(1 + 3.13) × (1 + 0.2575) ≈ 5.20 = +420% vs FIFO, and with SLA-aware at +125%
vs FIFO, the compound is approximately +169% vs oracle SLA-aware. With BurstGPT:
(1 + 5.57) × (1 + ε) vs SLA-aware at +211% → compound approximately +130-150%
vs oracle SLA-aware.

### Q3. Which forecasts are weakest?

1. **Economic scheduling multiplier** — not yet compounded with queue gains.
2. **BurstGPT 11.7% oracle gap** — closer to p90_abs_err ≈ 632 tokens; ML predictor
   would reduce this.
3. **Queue-economic integration** — completely unverified end-to-end.

### Q4. Which optimizer decisions remain suboptimal?

1. **Compound routing**: energy cost optimization and queue discipline are independent —
   never co-optimized end-to-end.
2. **BurstGPT gap**: 88.3% retention (vs Azure 97.8%) — further ML predictor gain possible.

### Q5. Which workloads benefit least?

**Workloads where SLA-aware binary classification is effective** (uniform output
distributions where the median split is as informative as continuous prediction).
On both Azure and BurstGPT, continuous prediction substantially dominates binary.

### Q6. Which research direction appears strongest?

**End-to-end compound economic × queue scheduling backtest.** Now that the
head-to-head confirms abs-conformal dominates SLA-aware by +83-112%, the
compound integration is the clearest next step to approach +300% vs SLA-aware.

### Q7. What is the shortest path to another +10% gain?

Compound economic × queue scheduling end-to-end backtest. Even a conservative
10% economic multiplier on top of the +83% queue advantage would exceed +300%.

### Q8. What is the shortest path to another +50% gain?

Compound economic × queue scheduling. If economic gains are multiplicative
(~1.26× from Azure CA leaderboard), compound = 1.8327 × 1.2575 ≈ 2.30 = +130%
vs oracle SLA-aware vs current +83%.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

From the head-to-head:
- Queue alone: +83% (Azure) / +112% (BurstGPT) vs oracle SLA-aware
- Target: +300% vs SLA-aware means ~4.0× oracle SLA-aware goodput
- Gap factor: 4.0 / 1.83 ≈ 2.19× more gain needed

Path: compound economic × queue where economic multiplier ≥ 2.0×.
Current CA leaderboard economic gain vs SLA-aware: +25.75%. Need a path
where economic optimization delivers 2× the SLA-aware baseline.

Alternative: improve ML predictor to close the oracle gap (push abs-conformal
to 99.5%+ retention) and combine with economic scheduling.

### Q10. Which assumptions might be wrong?

1. **Independence assumption** — queue and economic gains may not be truly
   multiplicative (e.g., routing to cheap regions may increase queue depth).
2. **Oracle SLA-aware is the right comparison for north-star** — the CA
   leaderboard uses a different SLA-aware baseline (different workloads/traces).
3. **BurstGPT 88.3% retention is stable** — may vary with load level.

### Q11. Which benchmark weaknesses exist?

1. **Queue-economic compound is unverified** — both components measured separately.
2. **Azure trace lacks model_id** — cannot test per-class improvements on Azure.
3. **Different "SLA-aware" definitions** — SRTF simulator binary-class vs CA
   leaderboard constraint-aware scheduler.

### Q12. Which public datasets should be added?

ShareGPT (third LLM trace for further cross-validation of the head-to-head).

### Q13. What should be attempted next?

**Immediate (next run):**
1. End-to-end compound economic × queue scheduling backtest — both components
   near-optimal; measure the compound gain directly.
2. Characterize whether the compound gain exceeds +300% vs SLA-aware end-to-end.

---

## Future Opportunity Ranking — Updated After Run -y

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Compound economic + queue scheduling (end-to-end backtest) | Very High | High | Queue near-optimal (+83% vs SLA-aware); economic multiplier unverified |
| 2 | ML predictor (HGB/CARA) with abs-conformal | High | Medium | Close remaining 11.7% BurstGPT gap; Azure already at 97.8% |
| 3 | ShareGPT as third public LLM trace | Medium | Medium | Further cross-validation of head-to-head results |
| 4 | Wire abs-conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |

**Closed/characterized opportunities (run -y):**
- SLA-aware head-to-head: **CHARACTERIZED** — abs-conformal +83% (Azure) / +112% (BurstGPT) vs oracle SLA-aware; north-star gap requires compound scheduling

---

## Run 2026-06-22-x — Absolute-Error Conformal Calibration (FRONTIER IMPROVEMENT)

### Q1. What currently limits Aurelius most?

**The running-statistics retention ceiling is broken.** The abs-conformal calibrator closes
97.8% (Azure) and 88.3% (BurstGPT) of the FIFO→oracle gap with a running-median prior.
The remaining 2.2% (Azure) and 11.7% (BurstGPT) gap to oracle is now the binding constraint.

Root cause of the previous ceiling: rel-error formula `p90(|pred-actual|/actual)` is dominated
by short-request over-predictions. Short ChatGPT (actual=7, pred=18) → rel_err=1.57 → p90
rel_err ≥ 0.80 → α capped at 2×alpha_max=0.002. The fix: `p90(|pred-actual|)` ignores the
scheduling-irrelevant 11-token error; p90 abs_err driven by genuine uncertainty in long requests
(~509-632 tokens with running-median prior) → α = 0.000222-0.000562 (near Pareto-optimal 0.001).

Key results:
- **Azure LLM 2024**: +313.14% vs FIFO (was +244.42%), α=0.000222 (was 0.002), 97.8% retention
- **BurstGPT HF**: +557.12% vs FIFO (was +420.83%), α=0.000562 (was 0.002), 88.3% retention
- abs_vs_rel improvement: **+19.95%** (Azure), **+26.17%** (BurstGPT)

### Q2. What theoretically offers the largest gain?

**A request-specific predictor (ML forecaster).** With running-median prior (essentially a
constant ≈ global median), the abs-calibrator achieves 97.8%/88.3% oracle retention. The
remaining 2.2%/11.7% gap requires per-request predictions. The abs-calibrator will amplify
any improvement in prediction accuracy: better predictor → lower p90 abs_err → lower α →
dispatch becomes more SRPT-like → higher goodput/$.

### Q3. Which forecasts are weakest?

1. **Output-token prior** — running median (constant) is still used. ML predictor needed.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Running-median prior** — still a global constant. Request-specific HGB/CARA predictor
   would further reduce p90 abs_err → α drops further toward 0 → closer to oracle.
2. **Serving queue not wired into runtime** — all gates closed, integration pending.

### Q5. Which workloads benefit least?

**Workloads with uniformly small output tokens** (e.g., hypothetical narrow-range trace). If
p90 abs_err is inherently small, abs vs rel makes no difference. On both tested public traces,
abs-conformal achieves large gains because the mismatch between relative and absolute error
metrics is significant.

### Q6. Which research direction appears strongest?

**End-to-end compound backtest (economic × queue scheduling).** With abs-conformal reaching
97.8% oracle retention on Azure, the queue-scheduling component is nearly optimal with live
prior. Compounding with economic scheduling gains is the next unverified opportunity.
Alternatively: wire a ML predictor (HGB) to further close the 11.7% BurstGPT gap.

### Q7. What is the shortest path to another +10% gain?

1. End-to-end compound backtest (economic × queue scheduling). Independence estimate: +876%
   vs FIFO. End-to-end unverified but now both components are near-optimal individually.
2. ML predictor for BurstGPT to close the remaining 11.7% oracle gap. With abs-calibrator,
   reducing p90 abs_err from 632 to ~100 tokens would push α below 0.0002 → near-oracle.

### Q8. What is the shortest path to another +50% gain?

Compound economic + queue scheduling end-to-end backtest. Abs-conformal closes the
queue-scheduling gap; the economic scheduling component (token rate optimization) multiplies it.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

Azure: abs-conformal already achieves +313.14% vs FIFO. SLA-aware baseline is approximately
+125.4% vs FIFO → abs-conformal is already **+82pp above the SLA-aware baseline** on Azure.
BurstGPT: abs-conformal +557.12% vs FIFO, SLA-aware baseline +210.6% vs FIFO → **+113pp above
SLA-aware** on BurstGPT. The +300% vs SLA-aware objective may already be satisfied with
abs-conformal — need to measure SLA-aware + abs-conformal head-to-head.

### Q10. Which assumptions might be wrong?

1. **p90 abs_err of 509-632 tokens is stable across different load levels.** At ρ=0.50,
   the prior quality may differ. Untested.
2. **target_p90_abs_tokens=500 is optimal.** If BurstGPT abs_err is consistently ~632 tokens
   (above target), α is still slightly above alpha_max. Tuning target to 650 might help.
3. **Oracle-retention ceiling is truly lifted.** The 2.2% Azure gap might have a different
   binding constraint we haven't yet characterized.

### Q11. Which benchmark weaknesses exist?

1. **SLA-aware head-to-head not yet run with abs-conformal.** The +300% vs SLA-aware claim
   needs direct measurement, not the indirect gap calculation.
2. **Compound economic × queue scheduling unverified end-to-end.** Independence estimate only.
3. **Azure trace lacks model_id.** Cannot test abs per-class on Azure.

### Q12. Which public datasets should be added?

ShareGPT (for model-class diversity validation), CARA telemetry (for abs-error calibration
with fine-grained features).

### Q13. What should be attempted next?

**Immediate (next run):**
1. End-to-end compound backtest (economic × queue scheduling) — both components individually
   near-optimal; measure the compound gain directly.
2. SLA-aware head-to-head with abs-conformal to confirm the +300% vs SLA-aware objective.

**Short-term (2–3 runs):**
3. ML predictor integration: HGB p50 as prior → further reduce p90 abs_err → close
   remaining 11.7% BurstGPT gap.
4. Wire abs-conformal discipline into serving runtime.

---

## Future Opportunity Ranking — Updated After Run -x

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Compound economic + queue scheduling (end-to-end backtest) | Very High | High | Independence estimate: +876% vs FIFO; abs-conformal queue component now near-optimal |
| 2 | SLA-aware vs abs-conformal head-to-head | High | High | Verify +300% vs SLA-aware objective directly |
| 3 | ML predictor (HGB/CARA) with abs-conformal | High | Medium | Close remaining 11.7% BurstGPT gap; abs-calibrator now amplifies prediction gains |
| 4 | ShareGPT as third public LLM trace | High | Medium | Azure+BurstGPT confirmed; third trace adds confidence |
| 5 | Wire abs-conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |

**Closed/characterized opportunities:**
- Absolute-error conformal calibration: **FRONTIER IMPROVEMENT [run -x]** — +19.95% Azure, +26.17% BurstGPT vs rel-conformal; breaks running-statistics ceiling
- Per-class conformal calibration: **NEGATIVE [run -w]** — +0.29% vs global; within-class variance ceiling (now superseded by abs formula)
- ML-HGB prior (HGB quantile p50): **NEGATIVE [run -v]** — −0.12% vs global prior (rel-error formula was the binding constraint, now fixed)
- Stratified causal prior: **NEGATIVE [run -u]** — −0.12% goodput/$
- Live causal prior (running median): **MEASURED [run -t]** — 81.6% Azure, 70.0% BurstGPT (now superseded: abs-conformal 97.8%/88.3%)

---

## Run 2026-06-22-w — Per-Class Conformal Calibration (Within-Class Variance Ceiling)

### Q1. What currently limits Aurelius most?

**The within-class token variance is now the confirmed binding constraint — not between-class
mixing.** Run -w tests per-class conformal calibration (separate ConformalAlphaCalibrator per
model_id) with ML-HGB predictions on BurstGPT HF. Results:
- FIFO: 6,528.76 goodput/$
- Oracle: 48,598.82 (+644.38%)
- Global conformal (ML-HGB): 34,003.60 (+420.83%, 65.31% retention)
- Per-class conformal (ML-HGB): 34,100.59 (+422.31%, 65.54% retention)
- **Per-class vs global: +0.29%** — real but marginal
- GPT-4 per-class mean_α = 0.002 (exactly capped, same as global)
- ChatGPT per-class mean_α = 0.001994 (≈capped)

Root cause: GPT-4 within-class token variance is large (CV ~40-60%). ML-HGB predicts GPT-4
p50 ≈ 235 tokens, but individual requests span 50-800 tokens. Short GPT-4 (100 tok):
rel_err=1.35; long GPT-4 (500 tok): rel_err=0.53. Per-class GPT-4 p90 rel_err ≥ 0.40
→ per-class α remains at cap.

**The running-statistics ceiling extends to per-class statistics.** The conformal calibrator
p90 relative error formula is binding WITHIN EACH CLASS, not just globally.

### Q2. What theoretically offers the largest gain?

**Changing the calibrator metric to absolute error (not relative error).** Key insight:
- Relative error penalizes over-predictions of short requests disproportionately
- GPT-4 absolute error (e.g., p90_abs ≈ 117 tokens for CV=50%) is more stable
- With absolute error formula: α = alpha_max × min(2.0, p90_abs_err / target_abs_tokens)
  - target_abs_tokens = 50 (calibrated for typical uncertainty)
  - GPT-4 p90_abs ≈ 50 tokens → ratio = 1.0 → α = alpha_max (no improvement)
  - But with good ML-HGB: GPT-4 p90_abs ≈ 10 tokens → ratio = 0.2 → α ≈ 0.0002 (improvement!)
- Absolute error formula separates scale from calibration sensitivity

### Q3. Which forecasts are weakest?

1. **Conformal calibrator formula** — binding constraint. Both relative-error (current) and
   any prior (running median, stratified, ML-HGB, per-class) cap at α≈0.002 for BurstGPT.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Calibrator uses p90 relative error** — sensitive to over-predictions of any request.
   GPT-4's within-class variance alone keeps the per-class p90 rel_err above target.
2. **Serving queue not wired into runtime** — all gates closed, integration pending.

### Q5. Which workloads benefit least?

**Any trace with high within-class token variance.** Both ChatGPT (bimodal) and GPT-4
(broad unimodal) have sufficient within-class variance to saturate the conformal calibrator.

### Q6. Which research direction appears strongest?

**Absolute-error conformal calibration.** The per-class experiment definitively closes the
"calibrator architecture" branch as a solution to the relative-error formula problem. The
remaining path is changing the formula itself. Absolute error is:
1. Scale-invariant in the right direction (50-token error is 50-token error for all classes)
2. More predictable with ML-HGB (absolute error variance ∝ output length, not %)
3. Less dominated by short-request over-prediction tail

### Q7. What is the shortest path to another +10% gain?

Implement absolute-error conformal calibration formula:
- Replace `p90_err = p90(|predicted - actual| / actual)` with `p90_err = p90(|predicted - actual|)`
- Replace `target_p90_error = 0.40` with `target_p90_abs_tokens = 50` (calibrated for
  ML-HGB GPT-4 absolute error distribution)
- Test on BurstGPT HF: GPT-4 absolute error with ML-HGB likely p90 ≈ 10-50 tokens
  → α drops from 0.002 to 0.0004-0.002 → dispatch becomes more SRPT-like for GPT-4

### Q8. What is the shortest path to another +50% gain?

Absolute-error conformal + end-to-end compound backtest (economic × queue scheduling).
The compound estimate is +876% vs FIFO (independence assumption) but unverified end-to-end.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

Same as prior runs. Live prior gives +420.83% vs FIFO on BurstGPT already (+191pp over
+210% SLA-aware baseline). Key remaining gap: oracle (+644%) vs live prior (+421%) = 223pp.

### Q10. Which assumptions might be wrong?

1. **Per-class α convergence only requires per-class errors to drop below target.**
   This assumes target_p90_error=0.40 is appropriate for both classes. If GPT-4 within-class
   variance fundamentally cannot produce p90_rel_err < 0.40, the formula needs changing.
2. **Absolute error formula would help.** GPT-4 absolute error might still be too high.
   Need to measure: what is GPT-4 p90 absolute error with ML-HGB predictions?
3. **Phase 1 contamination was a factor.** With larger warmup_n (e.g., 500), Phase 1 GPT-4
   errors would pollute the per-class window less. Not tested.

### Q11. Which benchmark weaknesses exist?

1. **Four calibrator variants now exhausted (global median, stratified, ML-HGB, per-class):**
   All hit the same 65-70% retention ceiling with p90 relative error formula.
2. **Azure trace lacks model_id.** Cannot test per-class on Azure.
3. **Compound gain independence assumption.** End-to-end backtest still pending.

### Q12. Which public datasets should be added?

ShareGPT, CARA telemetry. These would provide model_id + session features for absolute-error
calibration experiments.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Absolute-error conformal calibration formula — directly targets the confirmed binding
   constraint (relative error formula). Change formula, re-test on BurstGPT HF.
2. End-to-end compound backtest (economic × queue scheduling).

**Short-term (2–3 runs):**
3. Wire conformal discipline into serving runtime (all gates closed).
4. ShareGPT ingestion as third public LLM trace.

---

## Future Opportunity Ranking — Updated After Run -w

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Absolute-error conformal calibration (change formula from relative to absolute) | Very High | High | Root cause confirmed [run -w]: p90 relative error is wrong metric for heavy-tailed outputs |
| 2 | Compound economic + queue scheduling (end-to-end backtest) | Very High | High | Independence estimate: +876% vs FIFO; end-to-end unverified |
| 3 | ShareGPT as third public LLM trace | High | Medium | Azure+BurstGPT confirmed; third trace adds confidence |
| 4 | Wire conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |
| 5 | Admission gate → cluster simulator | Medium | Medium | Implemented (unconnected) |

**Closed/characterized opportunities:**
- Per-class conformal calibration: **NEGATIVE [run -w]** — +0.29% vs global; within-class variance ceiling
- ML-HGB prior (HGB quantile p50): **NEGATIVE [run -v]** — −0.12% vs global prior
- Stratified causal prior: **NEGATIVE [run -u]** — −0.12% goodput/$
- Live causal prior (running median): **MEASURED [run -t]** — 81.6% Azure, 70.0% BurstGPT

---

## Run 2026-06-22-v — ML-HGB Prior (Validated Null Result)

### Q1. What currently limits Aurelius most?

**The conformal calibrator p90-relative-error formula is the binding constraint, not prediction accuracy.**
Run -v tests a HistGradientBoostingRegressor (quantile p50, causal two-phase warmup) as the
output-token prior on BurstGPT HF. Results:
- FIFO: 6,529 goodput/$
- Oracle: 48,599 goodput/$ (+644%)
- Global prior (live running median): 34,004 goodput/$ (+420.83%, 70.0% retention)
- ML-HGB prior: 33,962 goodput/$ (+420.2%, 69.88% retention)
- **ml_vs_global_improvement_pct = −0.12%** (within noise; NOT an improvement)
- Both priors: conformal_mean_alpha = 0.002 (identical — both capped)
- ML prior IS more accurate: CV 15.34%→43.03%, MAE 166.93→162.82 tokens (−2.5%)
- **But the calibrator cap persists** — p90 relative error ≥ 0.80 in both cases

Root cause: ChatGPT intra-class variance is so large (p5=1 tok, p95=800+ tok) that
even a correct model_id-based predictor cannot reduce the p90 tail error below the 0.80
cap threshold. The calibrator formula `α = alpha_max × min(2.0, p90_err / target_p90_error)`
with target=0.40 caps at α=0.002 for both priors.

### Q2. What theoretically offers the largest gain?

**Per-class conformal calibration.** A separate calibrator for ChatGPT vs GPT-4 would:
1. Correctly assess that GPT-4 has LOWER residual uncertainty than ChatGPT
2. Allow lower α for GPT-4 requests → more SRPT-like dispatch for the 15.8% of requests
   that actually have predictable lengths
3. Break the monolithic calibrator cap

Alternatively: change the calibrator metric from p90 relative error to absolute error
or p50 relative error, which would be less dominated by ChatGPT's long tail.

### Q3. Which forecasts are weakest?

1. **Live prior** — three variants now exhausted: global median, stratified median, ML-HGB.
   All cap at α=0.002. The binding constraint is the calibrator formula, not prediction quality.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Conformal calibrator uses single p90 relative error across all request classes.**
   With multi-class data (ChatGPT + GPT-4), the p90 is always dominated by ChatGPT's
   long tail, preventing GPT-4's lower uncertainty from being utilized.
2. **Serving queue not wired into runtime** — unchanged from prior runs.
3. **North Star gap** — unchanged.

### Q5. Which workloads benefit least?

**BurstGPT HF with any monolithic conformal calibrator.** The bimodal ChatGPT distribution
creates an irreducible worst-case error floor that dominates the p90 metric, regardless of
predictor quality.

### Q6. Which research direction appears strongest?

**Per-class conformal calibration.** Three negative results (global median [run -t],
stratified median [run -u], ML-HGB [run -v]) definitively close the "better predictor"
branch for BurstGPT with the current calibrator formula. The opportunity now is to change
the calibrator architecture, not the predictor.

### Q7. What is the shortest path to another +10% gain?

Split the conformal calibrator: one calibrator for ChatGPT requests, one for GPT-4.
Measure separate p90 errors per class, compute per-class α. On dispatch, use the
per-class α for each request. This directly targets the observed failure mode.

### Q8. What is the shortest path to another +50% gain?

Per-class calibration breaking the GPT-4 cap (currently ~15.8% of requests could see α→0
if calibrated independently), combined with the compound economic scheduling backtest.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

Same as run -u. Live prior already gives +420.83% vs FIFO on BurstGPT. Compound with
economic scheduling estimate: +876% vs FIFO (independence assumption, unverified end-to-end).

### Q10. Which assumptions might be wrong?

1. **Calibrator cap is truly binding (not just local optimum).** The HGB IS learning
   model_id distinctions (CV 15→43%). Perhaps with a much larger warmup (10,000+ requests)
   the HGB would reduce p90 error below 0.80 on real BurstGPT. We tested warmup_n=1000.
2. **Per-class calibration would use model_id at prediction time.** At arrival, model_id
   is known (from the request). A per-class calibrator is causally valid.
3. **The −0.12% difference is pure noise.** The two priors produce statistically identical
   scheduling decisions because both hit α=0.002.

### Q11. Which benchmark weaknesses exist?

1. **Predictor class now exhausted for BurstGPT with current calibrator.** Global median,
   stratified median, ML-HGB — all three measured; all cap at α=0.002.
2. **Azure trace lacks model_id for ML predictor validation.** Cannot test per-class
   calibration on Azure until model_id feature is available.
3. **Compound gain remains independence-assumption only.** End-to-end backtest still pending.

### Q12. Which public datasets should be added?

Same as run -u. ShareGPT, CARA telemetry, Mooncake FAST25.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Per-class conformal calibration: separate α calibrators for ChatGPT vs GPT-4.
   Hypothesis: GPT-4's lower residual uncertainty would be correctly captured, allowing α→0
   for the 15.8% of GPT-4 requests and potentially improving goodput/$ by 5-15%.
2. End-to-end compound backtest (economic × queue scheduling).

**Short-term (2–3 runs):**
3. Calibrator metric change: absolute error instead of p90 relative error.
4. Wire conformal discipline into serving runtime (all gates closed).

---

## Future Opportunity Ranking — Updated After Run -v

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Per-class conformal calibration (separate α per model_id) | Very High | High | Root cause confirmed [run -v]: monolithic calibrator cap dominates |
| 2 | Compound economic + queue scheduling (end-to-end backtest) | Very High | High | Independence estimate: +876% vs FIFO; end-to-end unverified |
| 3 | Calibrator metric change (absolute error vs p90 relative) | High | High | Simple formula change; directly targets ChatGPT tail-dominated cap |
| 4 | ShareGPT as third public LLM trace | High | Medium | Azure+BurstGPT confirmed; third trace adds confidence |
| 5 | Wire conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |
| 6 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |

**Closed/characterized opportunities:**
- ML-HGB prior (HGB quantile p50): **NEGATIVE [run -v]** — −0.12% vs global prior; conformal cap persists
- Stratified causal prior: **NEGATIVE [run -u]** — −0.12% goodput/$; confirms running-statistics ceiling
- Live causal prior (running median): **MEASURED [run -t]** — 81.6% Azure, 70.0% BurstGPT

---

## Run 2026-06-22-u — Stratified Feature-Aware Causal Prior (Research Discovery — Negative)

### Q1. What currently limits Aurelius most?

**The running-statistics prior has a structural ceiling, now precisely characterized.**
Run -u confirms that no running-statistics prior (global or stratified) can close the oracle
gap on BurstGPT HF. Key findings:
- Stratified prior MAE: −5.7% vs global (157.3 vs 166.9 tokens)
- Stratified goodput/$: −0.12% vs global (33,962 vs 34,004) — flat/negative
- Conformal mean_α: 0.002 for both (identical — calibrator absorbed the MAE improvement)
- Running-statistics ceiling: ~70% retention on BurstGPT, ~82% on Azure
- Root cause: ChatGPT bimodal distribution — ~10% "surprise-long" requests (short input,
  long output) cannot be identified by any running-statistics prior

### Q2. What theoretically offers the largest gain?

**A trained ML predictor with per-request features.** Running statistics have now been
conclusively shown to top out at ~70-82% retention on both public LLM traces. A trained
model with session history, query complexity signals, and model-specific features would
be needed to:
1. Identify "surprise-long" ChatGPT requests before they enter the queue
2. Provide request-specific (not population-level) token-length predictions
3. Break the running-statistics ceiling

### Q3. Which forecasts are weakest?

1. **Live prior** — now confirmed at structural ceiling (70% BurstGPT, 82% Azure).
   Any running-statistics variant (global, stratified, per-model, input-binned) converges
   to the same goodput/$ because the conformal calibrator adapts α to match residual uncertainty.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Prior is running statistics, not request-specific.** Both global and stratified priors
   hit the same goodput/$ — the calibrator absorbs the quality difference. A trained ML
   predictor is required to break the ceiling.
2. **Serving queue not wired into runtime** — with live prior, conformal achieves
   +244.42% (Azure) / +420.83% (BurstGPT) vs FIFO. Integration pending.
3. **North Star gap (vs SLA-aware)** — unchanged from run -t.

### Q5. Which workloads benefit least?

**BurstGPT HF with any running-statistics prior.** The bimodal ChatGPT distribution
(short modal + ~10% surprise-long tail) creates an irreducible entropy floor for running
statistics. Azure is worse in absolute terms (lower retention) but for a different reason:
r≈0 correlation between prompt and output tokens means stratification cannot help even
in principle.

### Q6. Which research direction appears strongest?

**Trained ML predictor integrating CARA HGB forecaster** (`cara_output_length_forecaster.py`).
Run -u definitively closes the running-statistics branch: no level of stratification or feature
engineering within running statistics can cross the ceiling. The path to ≥85% retention is:
1. Request-specific features at arrival time (TTFT/queue-state from CARA)
2. Trained model (HGB) predicting actual output length
3. Wire HGB p50 as prior in `_run_live_prior_on_trace`

### Q7. What is the shortest path to another +10% gain?

Integrate `HGBOutputLengthForecaster.p50` as the live prior in the conformal backtest.
Even 50% improvement in prediction CV (from ~7% running median to 4-5% from HGB) would
break the conformal calibrator's α convergence and yield meaningfully different scheduling.

### Q8. What is the shortest path to another +50% gain?

A trained prompt-type classifier that correctly identifies ~70% of "surprise-long" ChatGPT
requests would directly improve rank-ordering accuracy, moving short_p90 toward oracle levels
(4.39s vs current 631s). Even partial classification would break the conformal ceiling.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

Same as run -t. Live prior already gives +244.42% vs FIFO = +53% above SLA-aware on Azure.
To reach +300% vs SLA-aware:
1. Trained ML prior (CARA HGB) pushing retention from 82% toward 95%+ on Azure
2. Compound with economic scheduling (+876% independence estimate)
3. End-to-end backtest to verify true compound (not independence assumption)

### Q10. Which assumptions might be wrong?

1. **Conformal calibrator fully absorbs MAE improvements.** Run -u confirms this for 5.7%
   MAE reduction. A much larger MAE reduction (e.g., 50%+) might not be fully absorbed —
   the threshold where calibrator compensation fails is unknown.
2. **Surprise-long fraction is stable.** The ~10% estimate is from this trace. Other ChatGPT
   traces may have different bimodal mixing proportions.
3. **SLA=30s for BurstGPT.** Under tighter SLA (10s), surprise-long requests would have
   even larger impact, potentially making the ceiling lower.

### Q11. Which benchmark weaknesses exist?

1. **Running-statistics prior class now exhausted.** Both global and stratified variants
   measured; both hit same goodput/$ ceiling. The benchmark is ready for the ML-predictor class.
2. **Azure trace lacks request-specific features for ML predictor.** Only ContextTokens +
   GeneratedTokens available — no model_id, session, TTFT. CARA HGB cannot be evaluated here.
3. **Compound gain remains independence-assumption only.** End-to-end backtest still pending.

### Q12. Which public datasets should be added?

1. **CARA telemetry** — TTFT, queue state, actual output lengths; enables HGB predictor training.
2. **ShareGPT** — richer request features (prompt text, session history) for predictor training.
3. **Mooncake FAST25** — KV prefix reuse signal.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Integrate `HGBOutputLengthForecaster.p50` as live prior in conformal backtest — tests whether
   trained predictor breaks the running-statistics ceiling.
2. Build prompt-type binary classifier for BurstGPT surprise-long detection. Input_tokens
   thresholding may identify 40-60% of surprise-long requests.

**Short-term (2–3 runs):**
3. End-to-end compound backtest: economic scheduling × live-prior SRTF in single trace run.
4. Explore session-level features for output-length prediction improvement on BurstGPT.

---

## Future Opportunity Ranking — Updated After Run -u

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Trained ML prior: CARA HGB forecaster integration | Very High | Medium | Running-statistics ceiling confirmed [run -u]; ML predictor is the required path |
| 2 | Prompt-type classifier for BurstGPT surprise-long detection | High | Medium | Gap quantified: ~10% ChatGPT requests are disguised-long; bimodal structure known |
| 3 | Compound economic + queue scheduling (end-to-end backtest) | Very High | High | Independence estimate: +876% vs FIFO |
| 4 | ShareGPT as third public LLM trace | High | Medium | Azure+BurstGPT confirmed; third trace adds confidence |
| 5 | Wire conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |
| 6 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |

**Closed/characterized opportunities:**
- Stratified causal prior: **NEGATIVE [run -u]** — −0.12% goodput/$; confirms running-statistics ceiling
- Live causal prior (running median): **MEASURED [run -t]** — 81.6% Azure, 70.0% BurstGPT

---

## Run 2026-06-21-t — Live Causal Prior Evaluation (Production Realism Gate)

### Q1. What currently limits Aurelius most?

**The prior is still oracle.** This run measured the first production-realistic prior:
causal sliding-window running median (window=200, no future leakage). Key findings:
- Azure retention: **81.6%** (live prior vs oracle; 1.5pp below 83% noisy-prior floor)
- BurstGPT retention: **88.1%** (above 83% floor — passes gate on BurstGPT)
- Azure CV_actual=80.5%, prior_cv=7.0% → prior is nearly constant (≈ global median ≈ 90 tok)
- Zero correlation (r=-0.022) between prompt tokens and output tokens in Azure trace → no simple
  feature can beat running median on this trace; request-specific predictor requires external signals
- Compound gain (independence assumption): economic scheduling (+183.4%) × serving queue
  (+244.42% live on Azure) → estimated **+876%** vs FIFO combined

### Q2. What theoretically offers the largest gain?

**A request-specific output-token predictor.** Running median has CV=7% vs actual CV=80.5%.
The entire scheduling gain comes from ordering, and ordering requires differentiation.
A predictor like CARA HGB (trained on TTFT, queue state, GPU type) could dramatically improve
the prior — closing the 18pp retention gap from 82% toward oracle (100%).

### Q3. Which forecasts are weakest?

1. **Live prior** — confirmed: running median CV=7% vs actual 80.5%. Near-constant prediction.
   Any request-specific signal (prompt length, model id, session history) would help.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Prior is running median, not request-specific.** Measured 18.4pp gap to oracle on Azure.
2. **Serving queue uses FIFO** — with live prior, conformal achieves +244.42% (Azure) /
   +420.83% (BurstGPT) vs FIFO. Still not wired into runtime.
3. **North Star gap (vs SLA-aware)** — unchanged from run -s.

### Q5. Which workloads benefit least?

**Azure LLM 2024 with running-median prior.** Prompt–output correlation is near zero
(r=-0.022), making running median the best achievable simple predictor. BurstGPT is more
tractable: higher output variance (p50=236 tok vs 90) means running median is more useful
relative to FIFO ordering, giving 88.1% retention vs 81.6% on Azure.

### Q6. Which research direction appears strongest?

**Request-specific output-token predictor (CARA HGB approach).** The live prior experiment
quantified exactly how much we're leaving on the table: 18pp retention on Azure. A model
using prompt length + session features could push retention above 95%. This is now the
clearest highest-EV target.

### Q7. What is the shortest path to another +10% gain?

Add any feature that correlates with output length. Even prompt_char_count or request_type
label could help. The full CARA HGB (TTFT proxy + queue state) could close 50% of the gap.

### Q8. What is the shortest path to another +50% gain?

Request-specific predictor bringing retention from 82% → 95%+ would recover ~13pp × 322%
oracle = ~42pp additional gain. Combined with economic scheduling compound, this exceeds +50%.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

Same as run -s. Current live prior gives +244.42% vs FIFO (Azure). SLA-aware = +125.4%.
So live prior is already +53% vs SLA-aware. To reach +300% vs SLA-aware: requires either
a substantially better prior (request-specific) or compounding with economic scheduling.
Independence estimate: +876% compound vs FIFO → far exceeds +300% vs SLA-aware.

### Q10. Which assumptions might be wrong?

1. **Independence assumption for compound gain.** The +876% estimate assumes economic
   scheduling and serving queue gains compound multiplicatively. True compound requires
   end-to-end integrated backtest.
2. **Running median as best achievable simple prior.** On Azure (r≈0), this is correct.
   On other traces with richer signals, simple features might help more.
3. **SLA=10s for Azure, 30s for BurstGPT.** Tighter SLA changes which requests are
   SLA-compliant and could shift retention ratios.

### Q11. Which benchmark weaknesses exist?

1. **Oracle prior still used for oracle comparison.** Live prior gap now measured (18pp Azure).
2. **Azure trace lacks request-specific features.** Only ContextTokens + GeneratedTokens;
   no model ID, no session, no TTFT. CARA HGB cannot be evaluated on this trace.
3. **Compound gain is independence-assumption only.** End-to-end backtest still pending.

### Q12. Which public datasets should be added?

1. **ShareGPT** — richer request features (prompt text) for predictor training/testing.
2. **CARA latency prediction dataset** — TTFT + queue state; enables request-specific prior.
3. **Mooncake FAST25** — KV prefix reuse signal.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Evaluate CARA HGB predictor as live prior (needs TTFT/queue-state signal in trace).
2. ShareGPT as third public LLM trace with richer feature space.

**Short-term (2–3 runs):**
3. End-to-end compound backtest: economic scheduling × live-prior SRTF in single trace run.
4. Explore session-level features (conversation turn number) for output-length prediction.

---

## Future Opportunity Ranking — Updated After Run -t

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Request-specific output-token predictor (CARA HGB or prompt features) | Very High | Medium | Gap quantified: 18pp Azure, 12pp BurstGPT |
| 2 | Compound economic + queue scheduling (end-to-end backtest) | Very High | High | Independence estimate: +876% vs FIFO |
| 3 | ShareGPT as third public LLM trace | High | Medium | Azure+BurstGPT confirmed; third trace adds confidence |
| 4 | Wire conformal discipline into serving runtime | High | Medium | All gates CLOSED; integration pending |
| 5 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |
| 6 | GPU routing on LLM trace (TTFT binding) | Medium | Low | benchmarked on energy trace [run -f] |

**Closed opportunities:**
- Live causal prior (running median): **MEASURED [run -t]** — 81.6% Azure, 88.1% BurstGPT
- Preemption overhead on BurstGPT: **CLOSED** [run -s] — 95.25% retention at 0.30s
- BurstGPT noisy prior: **CLOSED** [run -r] — 100.0% retention
- BurstGPT SLA-aware baseline: **CLOSED** [run -r] — +210.6% vs FIFO
- BurstGPT conformal: **CLOSED** [run -r] — +644.4% vs FIFO

---

## Run 2026-06-21-s — BurstGPT HF Preemption Overhead Cross-Validation (Infrastructure Improvement)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO.** All six cross-trace validation gates are now
CLOSED on both public LLM traces (Azure LLM 2024 + BurstGPT HF):
- Noisy prior robustness: 100.0% retention on Azure [run -n] AND BurstGPT [run -r]
- Preemption overhead: ≥92.65% retention on Azure [run -o] AND 95.25% on BurstGPT [run -s]
- Cross-trace SRTF: +492.7% vs FIFO (decoupled) on BurstGPT [run -p]
- Conformal α: +644.4% vs FIFO on BurstGPT [run -r] (SRPT ceiling, cross-trace)
- SLA-aware baseline: measured on Azure (+65.9% over SLA-aware) and BurstGPT (+90.8%)
- Alpha sweep: Pareto-optimal α=0.001 confirmed on Azure [run -m], BurstGPT [run -p]

There are no remaining validation gaps. The only blocker is runtime integration.

### Q2. What theoretically offers the largest gain?

**Wiring the conformal discipline into the serving runtime with live predictions.**
All validation gates passed. Expected: +322–644% vs FIFO depending on trace; +87–140%
vs SLA-aware. Compound with economic scheduling could push toward the +300% North Star.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all backtests still use oracle prior.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — conformal: +322.24% (Azure) / +644.4% (BurstGPT). Not wired.
2. **North Star gap (vs SLA-aware)** — decoupled vs SLA-aware: +65.9% (Azure) / +90.8% (BurstGPT).
   Target: +300%.

### Q5. Which workloads benefit least?

**Batch workloads under pure energy-price constraint.** BurstGPT and Azure LLM 2024 both
show massive SRTF gains. The CA leaderboard shows more modest gains (+1.77% BurstGPT,
+25.75% Azure) because the provisioning model doesn't include queue discipline ordering.

### Q6. Which research direction appears strongest?

**Wire conformal discipline into serving runtime.** All validation gates closed [runs -n through -s].
Compound economic + queue scheduling [rank #2] is next after integration.

### Q7. What is the shortest path to another +10% gain?

Wire the conformal discipline into the serving runtime. Even at 30%-CV noise:
+267-492% vs FIFO. The gap vs SLA-aware is +65.9–90.8%.

### Q8. What is the shortest path to another +50% gain?

Same as Q7. Decoupled hybrid at +90.8% over SLA-aware on BurstGPT.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

+300% vs FIFO: **ACHIEVED** on both traces (conformal: +322.24% Azure, +644.4% BurstGPT).
+300% vs SLA-aware (North Star): not yet achieved.
- BurstGPT: SLA-aware = +210.6% vs FIFO; conformal = +644.4% → conformal = +139.6% vs SLA-aware
- Azure: SLA-aware = +125.4% vs FIFO; conformal = +322.24% → conformal = +87% vs SLA-aware
- To reach +300% vs SLA-aware: requires compounding economic scheduling + serving queue.
  Economic scheduling can add ~25% (Azure LLM 2024 CA result) on top of serving queue gains.

### Q10. Which assumptions might be wrong?

1. **Oracle prior as primary benchmark.** Validated: 100% retention under 30%-CV noise
   (both traces). Conformal adapts α from real prediction residuals.
2. **Overhead model additivity.** NOW CLOSED [run -s]: BurstGPT 95.25% retention at
   0.30s/event. Higher than Azure (92.65%) due to longer service times.
3. **SLA=30s for BurstGPT.** Under tighter SLA (10s), BurstGPT gains may differ.

### Q11. Which benchmark weaknesses exist?

1. **Oracle prior.** Both public traces still use perfect token-length prediction.
   Robustness validated (100% retention under 30%-CV noise).
2. **North Star gap.** Conformal vs SLA-aware: +87% (Azure) / +139.6% (BurstGPT).
   Target +300% requires runtime integration + economic scheduling compound.
3. **Only two public LLM traces.** ShareGPT would add a third cross-validation.

### Q12. Which public datasets should be added?

1. **ShareGPT** — output token cross-validation, third public LLM trace.
2. **Mooncake FAST25 Traces** (Apache-2.0) — KV prefix reuse signal.
3. **Vidur Profiling CSVs** — measured kernel latency for service time calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Wire conformal discipline into serving runtime with live OutputLengthForecastBundle.p50.
2. Measure compound gain: economic scheduling × SRTF serving queue on canonical backtest.

**Short-term (2–3 runs):**
3. ShareGPT as third public LLM trace for broader cross-trace validation.
4. Compound economic + queue scheduling in canonical backtest.

---

## Future Opportunity Ranking — Updated After Run -s

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire conformal discipline into serving runtime with live predictions | Very High | Medium | All 6 gates CLOSED [runs -n through -s]; integration pending |
| 2 | Compound economic + queue scheduling in canonical backtest | Very High | High | Requires serving runtime integration |
| 3 | ShareGPT as third public LLM trace | High | Medium | Azure+BurstGPT confirmed; third trace adds confidence |
| 4 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 5 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |
| 6 | GPU routing on LLM trace (TTFT binding) | Medium | Low | benchmarked on energy trace [run -f] |

**Closed opportunities:**
- Preemption overhead on BurstGPT: **CLOSED** [run -s] — 95.25% retention at 0.30s (more robust than Azure)
- BurstGPT noisy prior: **CLOSED** [run -r] — 100.0% retention
- BurstGPT SLA-aware baseline: **CLOSED** [run -r] — +210.6% vs FIFO
- BurstGPT conformal: **CLOSED** [run -r] — +644.4% vs FIFO

---

## Run 2026-06-21-r — BurstGPT HF Extended Validation (Frontier Improvement)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO.** All cross-trace validation gates are now
PASSED on both public LLM traces:
- Noisy prior robustness: 100.0% retention on Azure [run -n] AND BurstGPT [run -r]
- Preemption overhead: 92.65% retention at 0.30s/event on Azure [run -o]
- Cross-trace: +492.7% vs FIFO (decoupled) on BurstGPT [run -p]
- Conformal α: +644.4% vs FIFO on BurstGPT [run -r] (SRPT ceiling, cross-trace)
- SLA-aware baseline: measured on both Azure (+65.9% over SLA-aware) and BurstGPT (+90.8%)

The remaining blocker is runtime integration with live OutputLengthForecastBundle.p50.

### Q2. What theoretically offers the largest gain?

**Wiring the conformal discipline into the serving runtime with live predictions.**
The conformal calibrator will auto-tune α from real prediction residuals. With oracle
prior it hits SRPT ceiling (+644.4% BurstGPT / +322.24% Azure). With 30%-CV noise it
retains ~83% (Azure) / ~100% (BurstGPT at decoupled α=0.001).

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all backtests still use oracle
   prior. Conformal can adapt α from real prediction errors; integration is the key step.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — conformal discipline: +322.24% (Azure) / +644.4%
   (BurstGPT) vs FIFO. Not yet wired into runtime.
2. **North Star gap (vs SLA-aware) not closed** — decoupled vs SLA-aware: +65.9%
   (Azure) / +90.8% (BurstGPT). Target: +300%.

### Q5. Which workloads benefit least?

**None of the tested public traces.** Both Azure LLM 2024 and BurstGPT HF show
substantial gains across all three validation experiments. BurstGPT consistently
amplifies gains (~2× vs Azure) due to its heavier output-token distribution.

### Q6. Which research direction appears strongest?

**Wire conformal discipline into serving runtime with live OutputLengthForecastBundle.p50.**
Cross-trace validation is now complete. Both traces confirmed. Integration is the
remaining step to advance the North Star gap.

### Q7. What is the shortest path to another +10% gain?

Wire the conformal discipline into the serving runtime. Even conservative estimates
(30%-CV noise) show +267-492% vs FIFO. The gap vs SLA-aware (+90.8% on BurstGPT)
suggests further compounding with economic scheduling.

### Q8. What is the shortest path to another +50% gain?

Same as Q7. Decoupled hybrid at +90.8% over SLA-aware on BurstGPT.

### Q9. What would need to be true to achieve +300% vs SLA-aware?

+300% vs FIFO: **ACHIEVED** on both traces (conformal: +322.24% Azure, +644.4% BurstGPT).
+300% vs SLA-aware (North Star): not yet achieved.
- BurstGPT: SLA-aware = +210.6% vs FIFO; conformal = +644.4% → conformal = +139.6% vs SLA-aware
- Azure: SLA-aware = +125.4% vs FIFO; conformal = +322.24% → conformal = +87% vs SLA-aware
- To reach +300% vs SLA-aware: requires compounding economic scheduling + serving queue

### Q10. Which assumptions might be wrong?

1. **Oracle prior as primary benchmark.** Both traces use actual tokens as predicted.
   With real predictions (CV ≈ 20-30%), α auto-tunes → +267-492% vs FIFO depending on trace.
2. **Overhead model additivity.** Validated on Azure [run -o] but not on BurstGPT.
3. **SLA=30s for BurstGPT.** Higher than production LLM SLAs. Under tighter SLA (10s),
   BurstGPT gains may differ (more requests timeout under tight SLA).

### Q11. Which benchmark weaknesses exist?

1. **Oracle prior.** Both public traces still use perfect token-length prediction.
2. **North Star gap.** Conformal vs SLA-aware: +87% (Azure) / +139.6% (BurstGPT).
   Target +300% requires runtime integration + economic scheduling compound.
3. **Overhead on BurstGPT.** Preemption overhead sensitivity validated only on Azure.

### Q12. Which public datasets should be added?

1. **ShareGPT** — output token cross-validation, third public LLM trace.
2. **Mooncake FAST25 Traces** (Apache-2.0) — KV prefix reuse signal.
3. **Vidur Profiling CSVs** — measured kernel latency for service time calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Wire conformal discipline into serving runtime with live OutputLengthForecastBundle.p50.
   The calibrator was built [run -q] and validated cross-trace [run -r]. Integration is the key step.
2. Measure compound gain: economic scheduling × SRTF serving queue on canonical backtest.

**Short-term (2–3 runs):**
3. Preemption overhead sensitivity on BurstGPT (parallel to Azure [run -o]).
4. ShareGPT as third public LLM trace for broader cross-trace validation.

---

## Future Opportunity Ranking — Updated After Run -r

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire conformal discipline into serving runtime with live predictions | Very High | Medium | Calibrator built [run -q]; cross-trace validated [run -r]; live prior pending |
| 2 | Compound economic + queue scheduling in canonical backtest | Very High | High | Requires serving runtime integration |
| 3 | Preemption overhead sensitivity on BurstGPT | Medium | Low | Azure validated [run -o]; BurstGPT pending |
| 4 | ShareGPT as third public LLM trace | High | Medium | Cross-trace pattern confirmed (Azure+BurstGPT); third trace adds confidence |
| 5 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 6 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |
| 7 | GPU routing on LLM trace (TTFT binding) | Medium | Low | benchmarked on energy trace [run -f] |

---

## Run 2026-06-21-q — Conformal Adaptive α (Frontier Improvement)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO.** The SRTF simulator frontier has now reached
SRPT-optimal (+322.24% vs FIFO) via conformal adaptive α [run -q]. The remaining limit
is wiring the conformal scheduler into a production serving runtime with live
OutputLengthForecastBundle.p50 predictions instead of oracle tokens.

### Q2. What theoretically offers the largest gain?

**Wiring the conformal discipline into the serving runtime with live predictions.** The
conformal calibrator was designed for exactly this: when predictions are from a trained
model (CV ≈ 20-30%), it will auto-tune α to match prediction quality, maintaining strong
goodput while adapting gracefully.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all backtest results still use oracle
   prior. The conformal calibrator can adapt α from real prediction errors. Integration is
   the key remaining step.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — conformal discipline: +322.24% vs FIFO (oracle), +267.81%
   vs FIFO (30%-CV noisy). Not yet wired into runtime.
2. **BurstGPT conformal validation not yet run** — the conformal approach has only been
   validated on Azure LLM 2024 (fixture level for BurstGPT). HF fullscale pending.
3. **BurstGPT vs SLA-aware baseline** — SLA-aware measured on Azure; BurstGPT pending.

### Q5. Which workloads benefit least?

**Small traces and batch workloads.** Confirmed: BurstGPT fixture (51 rows) shows conformal
= fixed α (both slightly below FIFO) due to warmup threshold not reached. HF fullscale
(59,999 records) expected to show the same pattern as Azure.

### Q6. Which research direction appears strongest?

**Wire conformal discipline into serving runtime with live OutputLengthForecastBundle.p50.**
This compounds: economic scheduler (+25.75% vs SLA-aware) × serving queue scheduler
(+322% vs FIFO) → potentially the largest absolute gain achievable.

### Q7. What is the shortest path to another +10% gain?

Wire the conformal discipline into the canonical LLM backtest with live predictions. Even
at 30%-CV noise, conformal gives +267.81% vs FIFO (vs +273.99% for fixed α). The
compounding with economic scheduling is the key.

### Q8. What is the shortest path to another +50% gain?

Same as Q7. Both canonical baselines (sla_aware) and FIFO show massive room for
improvement from queue discipline integration.

### Q9. What would need to be true to achieve +300%?

+300% vs FIFO: **ACHIEVED** (conformal +322.24% on Azure LLM 2024 oracle).
+300% vs SLA-aware (North Star): SLA-aware = +125.4% vs FIFO; conformal = +322% vs FIFO
→ conformal = +87% vs SLA-aware. Getting to +300% vs SLA-aware requires:
1. Live prediction (conformal adapts to real CV)
2. Compound with economic scheduling shifts

### Q10. Which assumptions might be wrong?

1. **Oracle prior as primary benchmark.** Conformal converges α → 0 because oracle tokens
   = actual tokens. With real predictions (CV ≈ 20-30%), α → 0.001 → +267-274% vs FIFO.
   The conformal approach is still the best available: it automatically uses the right α.
2. **30%-CV noisy retention.** Conformal achieves 83.1% retention (267.81%/322.24%), vs
   fixed α=0.001 at 100% retention (273.99%/273.99%). The absolute comparison shows fixed
   is slightly better under 30%-CV noise; the real choice depends on predictor quality.
3. **Overhead model additivity.** Still applies (same as run -o).

### Q11. Which benchmark weaknesses exist?

1. **Oracle prior.** All primary benchmark results use perfect token-length prediction.
   Conformal with oracle = SRPT, which is the optimum — a favorable evaluation context.
2. **FIFO baseline.** North Star requires vs SLA-aware. Conformal vs SLA-aware: +87%.
3. **BurstGPT conformal.** Only tested on 51-row fixture (warmup not reached). HF fullscale pending.

### Q12. Which public datasets should be added?

1. **ShareGPT** — output token cross-validation for BurstGPT-like heavy tail.
2. **Mooncake FAST25 Traces** (Apache-2.0) — KV prefix reuse signal.
3. **Vidur Profiling CSVs** — measured kernel latency for service time calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. BurstGPT HF fullscale conformal validation (59,999 records) — confirm +644% SRPT
   ceiling is approached by conformal on BurstGPT's heavier distribution.
2. BurstGPT vs SLA-aware baseline — measure the North Star gap on BurstGPT.

**Short-term (2–3 runs):**
3. Wire OutputLengthForecastBundle.p50 as live prior into conformal discipline.
   The calibrator will adapt α from real prediction residuals.
4. Wire conformal discipline into canonical LLM serving backtest (compound gains).

---

## Future Opportunity Ranking — Updated After Run -q

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire conformal discipline into serving runtime with live predictions | Very High | Medium | Calibrator built; live prior pending |
| 2 | BurstGPT HF fullscale conformal validation (59,999 records) | High | Medium | Fixture done; HF fullscale pending |
| 3 | BurstGPT vs SLA-aware baseline | Very High | Low | SLA-aware measured on Azure [run -n]; BurstGPT pending |
| 4 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 5 | Compound economic + queue scheduling in canonical backtest | Very High | High | Requires serving runtime integration |
| 6 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |
| 7 | GPU routing on LLM trace (TTFT binding) | Medium | Low | benchmarked on energy trace [run -f] |

---

## Run 2026-06-21-p — BurstGPT HF Full-Scale Cross-Validation (Frontier Improvement)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO.** Three critical simulator gates are now ALL PASSED:
(1) noisy prior robustness: 100% retention at 30%-CV [run -n], (2) preemption overhead:
92.65% retention at 0.30s/event [run -o], (3) cross-trace: +231–493% vs FIFO on BurstGPT
HF [run -p]. The remaining limit is runtime integration of the decoupled hybrid.

### Q2. What theoretically offers the largest gain?

**Wiring decoupled hybrid (α=0.001) into the serving runtime.** All three simulator
validation gates are now passed. BurstGPT cross-validation shows +492.7% vs FIFO
(5,880-record sample) and +231.4% vs FIFO (full 58,042-record run). The gain is real,
robust to prior noise, robust to preemption overhead, and generalizes across traces.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all backtests still use oracle
   prior. 30%-CV robustness validated [run -n]. Live prior integration still pending.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — decoupled hybrid α=0.001: +274% (Azure) and +492.7%
   (BurstGPT 5.8k) vs FIFO. Not yet wired into runtime.
2. **BurstGPT noisy prior robustness not yet run** — validated 30%-CV on Azure LLM 2024
   [run -n] but not on BurstGPT's heavier distribution.

### Q5. Which workloads benefit least?

**None of the tested public traces benefit least now that cross-trace validation confirms.**
Both Azure LLM 2024 and BurstGPT HF show substantial gains. The full 58,042-record
BurstGPT run at ρ=0.85 shows +231% (lower than the 5,880-record sample's +493% because
the full trace spans a much longer period with more queue buildup in the FIFO baseline).

### Q6. Which research direction appears strongest?

**Runtime integration of decoupled hybrid α=0.001.** Three critical simulator gates now
ALL PASSED. Cross-trace validation on BurstGPT confirms and extends the Azure result.

### Q7. What is the shortest path to another +10% gain?

Wire decoupled hybrid α=0.001 into the serving runtime. The BurstGPT cross-validation
confirms this gain is not trace-specific.

### Q8. What is the shortest path to another +50% gain?

Same as Q7. Both traces show gains well above +50% vs FIFO.

### Q9. What would need to be true to achieve +300%?

+300% vs FIFO: already achieved (SRPT +316% on BurstGPT full, +322% on Azure LLM 2024).
+300% vs SLA-aware (North Star): still unachieved. SLA-aware baseline on BurstGPT not yet
measured. Decoupled hybrid was +65.9% over SLA-aware on Azure LLM 2024 [run -n].

### Q10. Which assumptions might be wrong?

1. **Oracle prior.** All backtest results use actual tokens as predicted tokens. Real
   OutputLengthForecastBundle.p50 has ~20-40%-CV error; 30%-CV validated on Azure but
   not yet on BurstGPT's heavier distribution.
2. **Overhead model additivity.** Still applies (same as run -o).
3. **SLA=30s for BurstGPT.** This is higher than production LLM SLAs (typically 5-15s).
   Under tighter SLA, gains may differ.

### Q11. Which benchmark weaknesses exist?

1. **Oracle prior.** Both public-trace benchmarks still use perfect token-length prediction.
2. **FIFO baseline.** North Star requires vs SLA-aware. BurstGPT vs SLA-aware not yet measured.
3. **BurstGPT noisy prior.** 30%-CV validation confirmed for Azure LLM 2024 only.

### Q12. Which public datasets should be added?

1. **ShareGPT** — output token cross-validation for BurstGPT-like heavy tail.
2. **Mooncake FAST25 Traces** (Apache-2.0) — KV prefix reuse.
3. **Vidur Profiling CSVs** — measured kernel latency for service time calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. BurstGPT noisy prior robustness (30%-CV) — validate that BurstGPT result holds under
   realistic prior noise (parallel to Azure LLM 2024 run -n).
2. BurstGPT vs SLA-aware baseline — measure the North Star gap on BurstGPT.

**Short-term (2–3 runs):**
3. Wire decoupled hybrid into serving runtime with live OutputLengthForecastBundle.p50.
4. Conformal interval adaptive α tuning (arXiv:2508.14544) — closes ~48pp gap to SRPT.

---

## Future Opportunity Ranking — Updated After Run -p

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire decoupled hybrid (α=0.001) into serving runtime | Very High | Medium | All 3 gates PASSED: noisy [run -n] + overhead [run -o] + cross-trace [run -p] |
| 2 | BurstGPT noisy prior robustness (30%-CV) | High | Low | Azure confirmed [run -n]; BurstGPT pending |
| 3 | BurstGPT vs SLA-aware baseline | Very High | Low | SLA-aware measured on Azure [run -n]; BurstGPT pending |
| 4 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 5 | Conformal interval adaptive α tuning (arXiv:2508.14544) | Medium | Medium | closes ~48pp to SRPT |
| 6 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |
| 7 | GPU routing on LLM trace (TTFT binding) | Medium | Low | benchmarked on energy trace [run -f] |

---

## Run 2026-06-21-o — Preemption Overhead Sensitivity Analysis (Honesty Gap Closed)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO**, and the largest prior simulator honesty gap
(zero preemption overhead) has now been formally closed. At realistic overhead (0.30s,
2× TTFT_BASE_S), Decoupled Hybrid α=0.001 retains +253.9% vs FIFO (vs +274.0% at
zero overhead). The actual overhead discount is 7.3% — within the 5–15% estimate from
prior analysis. The main remaining limit is runtime integration.

### Q2. What theoretically offers the largest gain?

**Wiring decoupled hybrid (α=0.001) into the serving runtime.** The overhead analysis
confirms the gain is real: even under worst-case preemption costs (1.0s/event, swap-
mode), Decoupled retains +260.6% vs FIFO. Prior noisy-prior gate [run -n]: 100% retention
at 30%-CV. The combination of prior robustness + overhead robustness makes the case
for production integration unambiguous.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all serving backtests still use
   oracle prior. 30%-CV robustness validated [run -n].
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — decoupled hybrid α=0.001: +274% gp/$ (zero overhead),
   +254% gp/$ (0.30s overhead). Both far exceed FIFO. Not yet wired into runtime.
2. **BurstGPT cross-validation pending** — full 1.4M-row dataset not downloaded.
   Small fixture (51 rows) cannot confirm SRPT>FIFO ordering at the 5880-request
   scale seen on Azure LLM 2024.

### Q5. Which workloads benefit least?

**Small traces and batch workloads.** Confirmed: BurstGPT fixture (51 rows) shows
SRPT < FIFO on goodput/$ (insufficient queue depth for the scheduling signal).
The Azure LLM 2024 result (5,880 requests) is the reliable measurement.

### Q6. Which research direction appears strongest?

**Runtime integration of decoupled hybrid α=0.001.** Two critical simulator gates
are now both PASSED:
- Noisy prior robustness: 100% retention at 30%-CV [run -n].
- Preemption overhead robustness: 92.65% retention at 0.30s/event [this run].

### Q7. What is the shortest path to another +10% gain?

Wire decoupled hybrid α=0.001 into the serving runtime. The overhead analysis shows
the gain is robustly +254% vs FIFO even at conservative preemption cost assumptions.

### Q8. What is the shortest path to another +50% gain?

Same as Q7. The +254% floor (at 0.30s overhead) easily clears +50%.

### Q9. What would need to be true to achieve +300%?

+300% vs FIFO: SRPT at 0.30s overhead = +299.4%. This threshold is already met.
+300% vs SLA-aware (North Star): SLA-aware = +125.4% vs FIFO; Decoupled = +274%
vs FIFO → +65.9% vs SLA-aware at zero overhead. Even at 0.30s overhead: +254% vs
FIFO → ~+56% vs SLA-aware. North Star (+300% vs SLA-aware) requires live prior
integration beyond binary SLA class.

### Q10. Which assumptions might be wrong?

1. **Overhead model is additive per preemption event.** Real systems may batch
   preemptions or amortize re-prefill costs across a request's lifetime.
   The per-event model is conservative (overcounts cost).
2. **30%-CV robustness transfers to real prior quality.** Validated with lognormal
   synthetic noise; real error distribution may differ.
3. **SLA=10s is representative.** Under tighter SLA budgets (3s), the margin shrinks.

### Q11. Which benchmark weaknesses exist?

1. **BurstGPT fixture (51 rows)** — too small for cross-trace validation of SRPT>FIFO.
2. **Oracle prior throughout** — OutputLengthForecastBundle.p50 not driving ordering.
3. **FIFO baseline** — North Star is vs SLA-aware, not FIFO.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — highest priority.
2. **ShareGPT** — output token cross-validation.
3. **Mooncake FAST25 Traces** (Apache-2.0, small JSONL) — KV prefix reuse signal.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Wire decoupled hybrid (α=0.001) into serving runtime with OutputLengthForecastBundle.p50.
2. Download full BurstGPT (1.4M rows) for cross-validation at production scale.

**Short-term (2–3 runs):**
3. Conformal interval adaptive α tuning (arXiv:2508.14544) — closes ~48pp gap to SRPT.
4. Mooncake FAST25 traces ingest — KV prefix reuse cross-validation.
5. SLA-aware in aggregate economic benchmark rollup.

---

## Future Opportunity Ranking — Updated After Run -o

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire decoupled hybrid (α=0.001) into serving runtime | Very High | Medium | Both gates PASSED: 100% noisy retention [run -n] + 92.65% overhead retention [run -o] |
| 2 | Full BurstGPT cross-validation (1.4M rows) | High | Low | fixture too small; full dataset pending |
| 3 | Conformal interval adaptive α tuning (arXiv:2508.14544) | Medium | Medium | closes ~48pp to SRPT |
| 4 | SLA-aware in aggregate economic benchmark | Very High | Medium | needed for North Star measurement |
| 5 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | infrastructure built (shadow) |
| 6 | Mooncake FAST25 traces ingest | Medium | Low | Apache-2.0, small JSONL, KV prefix reuse |
| 7 | Admission gate → cluster simulator | Medium | Medium | implemented (unconnected) |
| 8 | GPU routing on LLM trace (TTFT binding) | Medium | Low | benchmarked on energy trace [run -f] |

---

## Run 2026-06-21-n — SLA-Aware Baseline + Noisy Prior Robustness (Critical Gate Passed)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO, and the production deployment gate has now been
cleared.** Run -n validates that decoupled hybrid α=0.001 retains 100% of oracle goodput/$
under 30%-CV lognormal forecast noise — the critical pre-deployment gate. This removes the
last simulation-level blocker for recommending runtime deployment. The primary remaining
limit is wiring α=0.001 with live `OutputLengthForecastBundle.p50` into the serving runtime
and cross-validating on the full BurstGPT dataset (1.4M rows).

### Q2. What theoretically offers the largest gain?

**Wiring decoupled hybrid (α=0.001) into the serving runtime with live prior.** The 100%
noisy retention result means the +274% vs FIFO goodput gain is robust to 30%-CV forecast
error. The remaining gap to pure SRPT (+322%) is ~48pp — achievable with a conformal
prediction interval for α adaptive tuning (arXiv:2508.14544) or a higher-fidelity token
length prior. Additionally, the North Star gap (vs SLA-aware, not FIFO) is now measurable:
binary SLA-aware gives +125.4% vs FIFO, so decoupled hybrid's actual edge over SLA-aware
is +65.9% from continuous prediction.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all serving backtests use oracle.
   30%-CV robustness now validated for decoupled hybrid α=0.001 [run -n]. Production prior
   quality expected to be 20–40%-CV; gate cleared.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — decoupled hybrid α=0.001 confirmed +274% goodput/$ at
   simulator fidelity; 30%-CV noisy prior gate PASSED; not yet wired into runtime.
2. **No preemption overhead model** — decoupled hybrid's +274% assumes zero KV-cache
   eviction cost. Real preemption cost could reduce net gain by 5–15% (estimated).
3. **BurstGPT cross-validation pending** — full 1.4M-row dataset not yet downloaded.

### Q5. Which workloads benefit least?

**Batch / energy-shifting and small traces.** Confirmed across all runs. BurstGPT fixture
(51 requests) cannot distinguish disciplines due to insufficient queue depth. The SLA-aware
discipline confirms the pattern: +125.4% vs FIFO on Azure LLM 2024 (5,880 requests), but
indistinguishable on the 51-row fixture.

### Q6. Which research direction appears strongest?

**Runtime integration.** The critical production gate is now passed: decoupled hybrid α=0.001
achieves +274% goodput/$ vs FIFO with 100% noisy retention at 30%-CV. arXiv:2508.14544
explains the mechanism (preemptive SRPT self-corrects ordering mistakes). The next gate is
measuring performance under real predicted-token noise from `OutputLengthForecastBundle.p50`
rather than synthetic lognormal noise.

### Q7. What is the shortest path to another +10% gain?

Wire decoupled hybrid α=0.001 into the serving runtime. Current simulator baseline is now
+274% vs FIFO (updated from +184.5% with the corrected default). Even accounting for 30%
oracle-to-real-prior degradation, expected gain is +200–250% vs FIFO.

### Q8. What is the shortest path to another +50% gain?

Same as Q7: wire decoupled hybrid α=0.001 into the serving runtime. The critical production
gate (30%-CV robustness) is now PASSED. 100% noisy retention means a +274% oracle gain
translates to an expected +274% with calibrated prior. The only remaining discount is
preemption overhead (estimated 5–15%).

### Q9. What would need to be true to achieve +300%?

+300% vs FIFO: decoupled α=0.001 = +274% (85% of SRPT's +322%). Closing the remaining
~48pp requires:
1. Conformal prediction interval adaptive α tuning (arXiv:2508.14544).
2. Higher-fidelity token length prior (lower CV reduces short_p90 degradation from 1.91→2.27s).
+300% vs SLA-aware (the North Star): SLA-aware binary class = +125.4% vs FIFO. Decoupled
α=0.001 = +274% vs FIFO → +65.9% over SLA-aware. North Star requires measuring this delta
in the canonical public-trace aggregate benchmark (not just per-request simulator).

### Q10. Which assumptions might be wrong?

1. **30%-CV robustness transfers to real prior quality.** The test uses lognormal synthetic
   noise. Real `OutputLengthForecastBundle.p50` error distribution may not be lognormal; if
   biased (systematic over-/under-prediction), noisy retention could be lower.
2. **Zero preemption overhead.** Real KV-cache eviction latency could reduce net goodput by
   5–15%. This remains the largest unmodeled cost.
3. **SLA=10s is representative.** The +274% result is SLA-specific. Under tighter SLA budgets
   (e.g., 3s), the margin shrinks and starvation has larger impact.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — North Star is vs SLA-aware, not FIFO. SLA-aware is now added (+125.4%
   vs FIFO) but not yet in the canonical aggregate benchmark.
2. **BurstGPT fixture (51 rows)** — too small; full 1.4M-row BurstGPT cross-validation pending.
3. **No preemption cost model** — zero-overhead preemption is optimistic.
4. **Oracle SLA-aware baseline** — the `sla_aware` binary class uses actual median split;
   no prediction noise applied to the binary class decision.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — highest priority for cross-validation.
2. **ShareGPT** — output token cross-validation.
3. **Vidur profiling CSVs** — GPU penalty calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Full BurstGPT cross-validation: `run_burstgpt_sla_aware_baseline_backtest()` and
   `run_burstgpt_noisy_prior_backtest()` are ready; download 1.4M-row BurstGPT dataset.
2. Wire decoupled hybrid (α=0.001) into serving runtime with `OutputLengthForecastBundle.p50`.

**Short-term (2–3 runs):**
3. Compare vs SLA-aware in aggregate economic benchmark — `sla_aware` aggregate optimizer
   in economic replay; wire per-request comparison to the canonical public-trace rollup.
4. Preemption overhead cost model — add KV-cache eviction latency to simulator.
5. Conformal interval adaptive α tuning (arXiv:2508.14544) to close the remaining ~48pp gap
   to pure SRPT.

---

## Future Opportunity Ranking — Updated After Run -n

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire decoupled hybrid (α=0.001) into serving runtime | Very High | Medium | Gate PASSED [run -n]: +274% gp/$ + 100% noisy retention |
| 2 | Full BurstGPT cross-validation (1.4M rows) | High | Low | run_burstgpt_*_backtest() ready |
| 3 | SLA-aware in aggregate economic benchmark | Very High | Medium | Needed for North Star progress measurement |
| 4 | Preemption overhead cost model (KV-cache eviction) | Medium | Low effort | Not started; estimated 5-15% reduction |
| 5 | Conformal interval adaptive α tuning | Medium | Medium | arXiv:2508.14544 basis; closes ~48pp to SRPT |
| 6 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 7 | Admission gate → cluster simulator | Medium | Medium | Implemented (unconnected) |
| 8 | GPU routing on LLM trace (TTFT binding) | Medium | Low | Benchmarked on energy trace [run -f] |
| 9 | BOute MOBO routing co-optimisation | High | High effort | Not started |

---

## Run 2026-06-21-m — Decoupled Hybrid Alpha Sweep (Pareto Frontier)

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO.** The alpha sweep (run -m) identifies α=0.001
as the Pareto-optimal configuration: +274.0% goodput/$ vs FIFO with near-SRPT
short_p90 (1.91s) and a meaningful starvation bound (flip-point ~66 min). The gap
remaining vs pure SRPT (+322.2%) is now only ~48pp, driven by the aging dispatch
occasionally promoting long-waiting medium-length requests over fresh short arrivals
even at α=0.001. The primary limit is now wiring α=0.001 into the serving runtime.

### Q2. What theoretically offers the largest gain?

**Wiring decoupled hybrid (α=0.001) into the serving runtime with
OutputLengthForecastBundle.p50 as the live prior.** The simulator shows +274%
goodput/$ vs FIFO at oracle prior quality. With 30%-CV noisy prior (run -g showed
SRTF retains >99% of short_p90 at 30% CV), the expected production gain is large.
Additionally, switching comparison baseline from FIFO to SLA-aware would close
the North Star gap directly.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all serving backtests still use
   oracle prior. The 30%-CV robustness from run -g applies to SRTF but has not been
   explicitly re-tested for decoupled hybrid at α=0.001.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — α=0.001 decoupled hybrid confirmed +274% goodput/$ at
   simulator fidelity; not yet wired into runtime.
2. **Oracle prior only** — OutputLengthForecastBundle.p50 not driving ordering.
3. **No SLA-aware baseline comparison** — all results vs FIFO; the North Star (+300% vs
   SLA-aware) requires adding SLA-aware as a comparison discipline.

### Q5. Which workloads benefit least?

**Batch / energy-shifting and small traces.** Confirmed across all runs. BurstGPT
fixture (51 requests) cannot distinguish alpha values due to insufficient queue depth.

### Q6. Which research direction appears strongest?

**Prior robustness at α=0.001:** Run -g showed SRTF retains >99% short_p90 at 30%
CV noise. Verifying the same holds for decoupled hybrid at α=0.001 is the critical
gate before recommending production deployment. If robust, α=0.001 becomes the
recommended production configuration.

### Q7. What is the shortest path to another +10% gain?

Update `DECOUPLED_HYBRID_ALPHA_DEFAULT = 0.001` and re-run the benchmark. Alpha sweep
shows +31.4% goodput improvement over α=0.01 (+274% vs +184.5% vs FIFO). The change
is 1-line with tests already passing.

### Q8. What is the shortest path to another +50% gain?

Wire decoupled hybrid (α=0.001) into the serving runtime with live prior. Simulator
shows +274% vs FIFO → even at 50% degradation from oracle to real prior, the net gain
is ~+137% vs FIFO, far exceeding +50%.

### Q9. What would need to be true to achieve +300%?

+300% vs FIFO is achievable: SRPT = +322%, decoupled α=0.001 = +274%, and with
noise-robust live prior expected to land +250-280% vs FIFO.
+300% vs SLA-aware (the North Star) requires:
1. Live prior (OutputLengthForecastBundle.p50) at α=0.001.
2. Confirm serving runtime integration.
3. Heterogeneous GPU routing on LLM traces (TTFT SLA improvement).
4. SLA-aware baseline added to measure true progress toward North Star.

### Q10. Which assumptions might be wrong?

1. **30%-CV robustness transfers from SRTF to decoupled hybrid at α=0.001.** Run -g
   proved SRTF is robust at 30% CV. Decoupled hybrid at α=0.001 behaves very similarly
   (dispatch ≈ pure SRPT when flip-point is 66+ min), so the same robustness is expected
   but not yet verified.
2. **Flip-point analysis is based on p99/p50 service times.** The actual flip-point
   distribution depends on which pairs of (waiting request, fresh arrival) actually
   compete at dispatch. At ρ=0.85 the heavy tail means occasional very-long requests
   may have shorter remaining service than expected.
3. **BurstGPT fixture small-sample limitation.** 51 requests cannot validate generalization.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — all goodput/$ deltas vs FIFO. The North Star is vs SLA-aware.
2. **BurstGPT fixture (51 rows)** — too small for cross-trace validation.
3. **Oracle prior only** — no noisy-prior validation for decoupled hybrid.
4. **No preemption cost model** — zero-overhead preemption is optimistic.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — highest priority.
2. **ShareGPT** — output token cross-validation.
3. **Vidur profiling CSVs** — GPU penalty calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Update `DECOUPLED_HYBRID_ALPHA_DEFAULT = 0.001` as recommended configuration.
2. Evaluate 30%-CV prior robustness for decoupled hybrid at α=0.001.
3. Wire decoupled hybrid (α=0.001) into serving runtime with OutputLengthForecastBundle.p50.

**Short-term (2–3 runs):**
4. Full BurstGPT cross-validation (1.4M rows) — `run_burstgpt_alpha_sweep()` ready.
5. Add SLA-aware baseline comparison to the serving simulator.
6. Preemption overhead cost model (KV-cache eviction latency estimate per token).

---

## Future Opportunity Ranking — Updated After Run -m

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire decoupled hybrid (α=0.001) into serving runtime | Very High | Medium | Pareto-optimal α identified [run -m]: +274% goodput/$ vs FIFO |
| 2 | 30%-CV prior robustness for α=0.001 | High | Low | Not started — critical gate for production deployment |
| 3 | Full BurstGPT cross-validation (1.4M rows) | High | Low | run_burstgpt_alpha_sweep() ready |
| 4 | Add SLA-aware baseline to serving simulator | Very High | Low | Needed for North Star progress measurement |
| 5 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 6 | Admission gate → cluster simulator | Medium | Medium | Implemented (unconnected) |
| 7 | GPU routing on LLM trace (TTFT binding) | Medium | Low | Benchmarked on energy trace [run -f] |
| 8 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 9 | Carbon-power MILP joint optimization | Medium | High effort | Not started |

---

## Run 2026-06-21-l — Decoupled Hybrid SRPT serving-queue simulator

### Q1. What currently limits Aurelius most?

**The serving runtime still uses FIFO, and the decoupled hybrid falls short of pure
SRPT goodput.** Run -l implements decoupled preemption (pure remaining_s) with aging
dispatch (remaining_s/(1+α·wait)), achieving +184.5% goodput/$ vs FIFO — between
Aging-SRTF (+70.7%) and SRPT (+322.2%). The remaining gap vs pure SRPT is ~137pp,
caused by aging dispatch occasionally dispatching long-waiting medium-length jobs over
fresher short arrivals. Primary limits now: (1) serving runtime still uses FIFO; (2)
oracle prior not replaced by OutputLengthForecastBundle.p50; (3) no alpha sweep to
find Pareto-optimal goodput vs long_p99 balance.

### Q2. What theoretically offers the largest gain?

**Alpha sweep + runtime deployment.** At α=0.001 (vs current 0.01), the dispatch
flip point moves from ~66.7s to ~667s — aging rarely fires, decoupled approaches
pure SRPT (+322%) while retaining bounded starvation protection. Combined with live
OutputLengthForecastBundle.p50 as the prior, this could capture >90% of the SRPT
simulator gain in production.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all serving backtests use
   oracle prior. 30%-CV robustness shown for SRTF only [run -g]; not tested for
   decoupled hybrid.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — decoupled hybrid confirmed +184.5% goodput/$ at
   simulator fidelity; not yet wired into runtime.
2. **α=0.01 reduces goodput by ~137pp vs pure SRPT** — alpha sweep needed.
3. **OutputLengthForecastBundle not driving ordering** — oracle prior only.

### Q5. Which workloads benefit least?

**Batch / energy-shifting workloads and small traces.** Confirmed across all runs.
Additionally: small serving traces (<300 requests, 2 servers) cannot distinguish
decoupled from pure SRPT because queue depth is too low for aging dispatch to
reorder. The +184.5% gain is only observable at scale (5,880 requests, 4 servers).

### Q6. Which research direction appears strongest?

**Alpha sweep:** profiling α ∈ {0.001, 0.005, 0.01, 0.05} on the full Azure LLM
2024 trace to map the goodput/long_p99 Pareto frontier for decoupled hybrid. At
α=0.001, expected behavior is near-SRPT goodput (>+310%) with mild starvation
reduction. This sweep would identify the deployment-ready configuration.

### Q7. What is the shortest path to another +10% gain?

Wire decoupled hybrid (α=0.01) into the serving runtime path. Current simulator
shows +184.5% vs FIFO. Even accounting for oracle-vs-real-prior degradation, this
is far above +10%. Alternatively, re-run with α=0.001 to approach +322% at lower
starvation cost.

### Q8. What is the shortest path to another +50% gain?

Wire decoupled hybrid with α=0.001 into serving runtime with
OutputLengthForecastBundle.p50. Expected: ~+300% vs FIFO (α=0.001 approaches SRPT
goodput). The 30%-CV robustness of non-preemptive SRTF [run -g] suggests goodput
degrades gracefully under noisy priors.

### Q9. What would need to be true to achieve +300%?

The decoupled hybrid at α=0.001 should approach +300% vs FIFO in simulation
(untested; pure SRPT = +322%). In production:
1. OutputLengthForecastBundle.p50 as live prior (30%-CV robustness proven for SRTF).
2. Decoupled hybrid at α=0.001 in serving runtime.
3. KV-cache eviction overhead < ~15% to preserve net goodput.
4. SLA-aware baseline comparison to validate vs state of the art (not just FIFO).

### Q10. Which assumptions might be wrong?

1. **α=0.01 produces meaningful anti-starvation** — empirically long_p99 with
   decoupled (+132.3% regression vs FIFO) is worse than pure Aging-SRTF (+113.8%).
   This suggests the aging dispatch at α=0.01 doesn't fire often enough to match
   Aging-SRTF's systematic queue prioritization.
2. **Zero preemption overhead** — same caveat as prior runs; real KV-cache eviction
   latency could erode net goodput.
3. **5,880-request fixture representativeness** — SAS-gated full Azure 2024 week
   not tested; the 5,880-row sample may not capture multi-day burst patterns.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — all goodput/$ deltas vs FIFO; SLA-aware baseline not compared.
2. **BurstGPT fixture (51 rows)** — too small for meaningful queue dynamics; all
   disciplines converge to identical goodput.
3. **No alpha sweep** — only α=0.01 benchmarked for decoupled hybrid.
4. **No preemption cost model** — zero-overhead preemption is optimistic.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — top priority.
2. **ShareGPT** — output token cross-validation.
3. **Vidur profiling CSVs** — GPU penalty calibration for heterogeneous routing.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Alpha sweep: run decoupled hybrid at α ∈ {0.001, 0.005, 0.01, 0.05} on full
   Azure LLM 2024 trace to map the Pareto frontier of goodput vs long_p99 regression.
   Expected: α=0.001 → >+310% goodput with +220% long_p99 (near-SRPT); α=0.05 →
   +70% goodput with +113% long_p99 (Aging-SRTF level).

**Short-term (2–3 runs):**
2. Wire decoupled hybrid (best α from sweep) into serving runtime with
   OutputLengthForecastBundle.p50 as predicted-tokens prior.
3. Full BurstGPT cross-validation (1.4M rows) at ρ=0.85 and ρ=0.95.
4. Preemption overhead cost model: add configurable KV-cache eviction latency (e.g.,
   1ms/token × evicted tokens) to measure net goodput impact.

---

## Future Opportunity Ranking — Updated After Run -l

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Alpha sweep for decoupled hybrid (α ∈ 0.001–0.05) | Very High | Low effort | Not started — run -l completed α=0.01 only |
| 2 | Wire decoupled hybrid into serving runtime | Very High | Medium | Quantified [run -l]: +184.5% goodput/$ |
| 3 | Full BurstGPT cross-validation (1.4M rows) | High | Low | run_burstgpt_decoupled_hybrid_backtest() ready |
| 4 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 5 | Preemption overhead cost model | Medium | Low effort | Not started |
| 6 | Admission gate → cluster simulator | Medium | Medium | Implemented (unconnected) |
| 7 | GPU routing on LLM trace (TTFT binding) | Medium | Low | Benchmarked on energy trace [run -f] |
| 8 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 9 | Carbon-power MILP joint optimization | Medium | High effort | Not started |

---

## Run 2026-06-20-k — Hybrid Aging+Preemptive SRPT serving-queue simulator

### Q1. What currently limits Aurelius most?

**The serving queue uses FIFO, and the hybrid aging+preemptive discipline at α=0.01
behaves like Aging-SRTF rather than SRPT.** Run -k implements the full hybrid
preemption key `remaining_s / (1 + α·accumulated_wait_s)` and confirms that:
(a) anti-starvation works — long_p99 is 34.7% lower than pure SRPT, and (b) goodput
is similar to Aging-SRTF (+64.2% vs FIFO), not SRPT (+322.2%), because the aging
dispatch key systematically promotes long-waiting requests over shorter fresh arrivals.
The primary limit is now: no decoupled-hybrid that uses pure SRPT preemption with
aging dispatch only.

### Q2. What theoretically offers the largest gain?

**Decoupled Hybrid:** use `remaining_s` as the preemption decision key (identifies
when a new arrival should preempt a running job — same as pure SRPT) and
`remaining_s / (1 + α·total_wait)` only for dispatch from the waiting queue
(gives long-waiting requests priority over equally-remaining fresh arrivals).

This separates two concerns:
- **Preemption key (arrival):** determines which running job to preempt — SRPT-optimal.
- **Dispatch key (completion):** determines which waiting job to dispatch next — aging-optimal.

Expected result: SRPT-level goodput (+322% vs FIFO) because preemption decisions
are identical to pure SRPT; Aging-SRTF-level long_p99 (+113% vs FIFO) because
the dispatch order promotes long-waiting requests and prevents indefinite starvation.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live prior** — all serving backtests use
   oracle prior. 30%-CV robustness shown for non-preemptive SRTF [run -g]; not yet
   tested for hybrid.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving queue uses FIFO** — simulator confirms +322% goodput/$ available from
   SRPT preemptive; not yet deployed in runtime.
2. **Hybrid α=0.01 overrides SRPT preemption benefit** — the dispatch-level aging
   key converts hybrid to Aging-SRTF behavior. Decoupling preemption and dispatch
   keys is the fix.
3. **OutputLengthForecastBundle not driving ordering** — oracle prior only.

### Q5. Which workloads benefit least?

**Batch / energy-shifting workloads** — confirmed through all runs. SRTF, aging-SRTF,
SRPT preemptive, and hybrid all benefit only in per-request serving queues under
contention. BurstGPT fixture (51 rows) remains too small for robust starvation analysis.

### Q6. Which research direction appears strongest?

**Decoupled Hybrid (run 2026-06-20-l):** separate the preemption key from the dispatch
key. This is supported by theory (SRPT preemptive is optimal for mean response in M/G/c;
aging dispatch prevents starvation without changing the throughput-optimal preemption
rule) and by the run-k empirical finding that the dispatch-level aging is what reduces
goodput.

### Q7. What is the shortest path to another +10% gain?

1. Wire SRPT preemptive (α=0 in hybrid) into serving runtime → +322% goodput/$ at
   simulator fidelity (short path, existing implementation in run -j).
2. Implement decoupled hybrid (run -l) → expected +322% goodput/$ + −35% long_p99 vs SRPT.
3. Cross-validate on full BurstGPT (1.4M rows) for generalization.

### Q8. What is the shortest path to another +50% gain?

Wire SRPT preemptive into the serving runtime with OutputLengthForecastBundle.p50 as
the predicted_tokens prior. Run -j simulator confirms +322.2% vs FIFO. Even with 30%-CV
forecast error, run -g showed SRTF retains >99% of its short_p90 benefit — suggesting
the goodput gain is robust to noisy priors.

### Q9. What would need to be true to achieve +300%?

+300% vs FIFO is achievable in simulation (+322.2% confirmed for SRPT, +323.5% for SRTF).
+300% vs SLA-aware (the north star) requires:
1. Live output-length prior (OutputLengthForecastBundle.p50) replacing oracle.
2. Serving-path SRPT with decoupled aging dispatch.
3. Heterogeneous GPU routing on LLM traces.
4. Measured queue-wait labels + pilot telemetry for frontier calibration.
The simulator confirms the ceiling; deploying it is the remaining gap.

### Q10. Which assumptions might be wrong?

1. **α=0.01 is the right aging scale** — run -k shows it's too large for preserving
   SRPT character. The "flip point" (when aging dominates dispatch) scales as
   `(r/r_new − 1) / α`. At α=0.01, requests wait only 66.7s before beating a fresh
   3s arrival. At α=0.001, the threshold is 667s — much less likely to trigger.
2. **Zero preemption overhead** — LLM serving preemption requires KV-cache eviction
   (memory reallocation, potential recompute). Real overhead could reduce effective goodput.
3. **Unified aging key** — same α for preemption and dispatch. The root cause finding
   from run -k suggests these should be decoupled.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — all goodput/$ deltas vs FIFO. SLA-aware baseline not yet compared.
2. **BurstGPT fixture** — 51 rows; too small for starvation analysis. All non-FIFO
   disciplines produce nearly identical goodput on this fixture.
3. **No preemption cost model** — zero-overhead preemption is optimistic.
4. **Simulator fidelity** — no batching, speculative decoding, CUDA graph overhead.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — top priority.
2. **ShareGPT** — output token cross-validation for OutputLengthForecastBundle.
3. **Vidur profiling CSVs** — GPU penalty calibration for heterogeneous routing.

### Q13. What should be attempted next?

**Immediate (run 2026-06-20-l):**
1. Implement Decoupled Hybrid: preemption by `remaining_s` (pure SRPT) +
   dispatch by `remaining_s / (1 + α·total_wait)` (aging anti-starvation).
   Expected: SRPT goodput/$ (+322%) + Aging-SRTF long_p99 (+113% vs FIFO).

**Short-term (2–3 runs):**
2. Wire SRPT preemptive or decoupled hybrid into serving runtime path driven by
   OutputLengthForecastBundle.p50 (low complexity, high EV).
3. Add preemption overhead cost model (KV-cache eviction latency estimate per token).
4. Cross-validate on full BurstGPT (1.4M rows) at ρ=0.85 and ρ=0.95.

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Decoupled Hybrid (SRPT preemption + aging dispatch) | Very High | Low effort | Not started — root cause from run -k |
| 2 | Wire SRPT preemptive into serving runtime | High | Medium | Quantified [runs -i/-j/-k]; not yet in runtime |
| 3 | Hybrid Aging+Preemptive (unified key) | Medium | Done | +64.2% gp/$ vs FIFO [run -k]; behaves like Aging-SRTF at α=0.01 |
| 4 | Full BurstGPT cross-validation (1.4M rows) | Medium | Low | run_burstgpt_*_backtest() functions ready |
| 5 | Wire OutputLengthForecastBundle.p50 as live prior | High | Low | Infrastructure built (shadow) |
| 6 | Admission gate → cluster simulator | Medium | Medium | Implemented (unconnected) |
| 7 | GPU routing on LLM trace (TTFT binding) | Medium | Low | Benchmarked on energy trace [run -f] |
| 8 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 9 | Carbon-power MILP joint optimization | Medium | High effort | Not started |

---

## Run 2026-06-20-j — Preemptive SRPT serving-queue simulator

### Q1. What currently limits Aurelius most?

**Long-request starvation is bounded but not eliminated, and the serving runtime
has no SRPT/aging hook.** Run -j adds preemptive SRPT to the simulator (guaranteeing
monotonic forward progress for every request), but empirically the long_p99 regression
(+223.4%) nearly matches non-preemptive SRTF (+223.5%) at ρ=0.85 because short-job
arrival rate continuously outcompetes long jobs even with preemption. The primary
bottleneck is now: (a) no hybrid aging+preemptive discipline that combines bounded
wait with preemptive short_p90 benefit, and (b) the serving runtime still uses FIFO.

### Q2. What theoretically offers the largest gain?

**Hybrid Aging+Preemptive SRPT:** use key(r,t) = remaining_s / (1 + α·wait_s) as
the preemption priority. This combines: (1) SRPT's immediate server reclamation for
newly arriving short jobs, and (2) aging's bounded-wait guarantee that long jobs
accumulate priority as they wait. Expected to recover 50–80% of the SRTF goodput
advantage (+200–250% vs FIFO) while capping long_p99 regression to Aging-SRTF levels
(+113% vs FIFO). Blueprint: run -i aging key + run -j preemption mechanics.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live SRTF prior** — all serving backtests
   (run -g, -i, -j) use oracle prior (actual tokens as predicted). 30%-CV robustness
   documented for SRTF (run -g); not yet re-tested for preemptive SRPT.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving-queue ordering uses FIFO** — SRPT preemptive proven +322.2% goodput/$
   in simulator; not yet wired into runtime.
2. **No hybrid aging+preemptive** — SRPT preemptive eliminates unbounded starvation
   theoretically but empirically long_p99 still regresses +223% at ρ=0.85.
3. **OutputLengthForecastBundle not driving ordering** — oracle prior only.

### Q5. Which workloads benefit least?

**Batch / energy-shifting workloads** — confirmed through multiple runs. SRTF,
aging-SRTF, and SRPT preemptive all benefit only in per-request serving queues
under contention. BurstGPT fixture (51 rows) still too small for starvation analysis.

### Q6. Which research direction appears strongest?

**Hybrid Aging+Preemptive SRPT:** combining the aging key from run -i with the
preemption mechanics from run -j. Preemption alone does not eliminate starvation
in high-utilization traces; aging+preemptive together should close the long_p99 gap.

### Q7. What is the shortest path to another +10% gain?

1. Wire SRPT preemptive (or aging-SRTF α=0.01) into the serving runtime path.
2. Cross-validate on full BurstGPT (1.4M rows).
3. Replace oracle prior with OutputLengthForecastBundle.p50.

### Q8. What is the shortest path to another +50% gain?

SRPT preemptive already shows +322.2% goodput/$ vs FIFO in the simulator (Azure LLM
2024, ρ=0.85). Realizing this in the serving runtime would achieve >+50% vs FIFO at
simulator fidelity, contingent on live output-length prediction quality.

### Q9. What would need to be true to achieve +300%?

The +300% target is vs SLA-aware (not FIFO). Run -j's +322% result is vs FIFO, not
SLA-aware; SRTF perfect achieves +323.5% vs FIFO in the same setup. Achieving +300%
vs SLA-aware requires: live output-length prior + serving-path SRPT with aging +
heterogeneous GPU routing on LLM traces + measured queue-wait labels + pilot telemetry.
The simulator confirms the ceiling; the gap is in deploying it.

### Q10. Which assumptions might be wrong?

1. **Preemption = anti-starvation** — at ρ=0.85 with heavy-tailed short-job
   arrivals, forward-progress guarantee alone does not prevent long_p99 regression
   of +223%. The assumption that preemption eliminates starvation holds in theory
   but not empirically with this trace/utilization combination.
2. **Oracle prior** — SRPT preemptive uses actual service times as predicted; a
   real prior with 30%-CV noise will degrade preemption accuracy.
3. **Preemption cost** — the simulator models preemption as zero-overhead. Real
   KV-cache eviction cost for preemption in LLM serving adds latency overhead.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — unchanged from runs -g/-i/-j. All goodput/$ deltas are vs FIFO.
2. **BurstGPT fixture too small** — 51 rows (high variance); SRTF and Aging-SRTF
   produce identical results on this fixture. Full 1.4M-row needed.
3. **No preemption cost model** — zero-overhead preemption is optimistic for real
   LLM serving (KV-cache eviction adds latency and GPU memory pressure).
4. **Simulator fidelity** — discrete-event M/G/c with synthetic time-warp;
   real serving systems have batching, speculative decoding, CUDA graph overhead.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — highest priority for
   cross-trace SRPT preemptive + aging-SRTF validation.
2. **ShareGPT** — output token cross-validation for OutputLengthForecastBundle.
3. **Vidur profiling CSVs** — GPU penalty calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Implement Hybrid Aging+Preemptive SRPT: preemption key = remaining_s / (1 + α·wait_s).
   Compare long_p99 regression: expect hybrid to land between Aging-SRTF (+113%) and
   SRPT preemptive (+223%) at the same goodput/$ as SRPT.
2. Cross-validate SRPT preemptive on full BurstGPT (1.4M rows).

**Short-term (2–3 runs):**
3. Wire SRPT preemptive (or hybrid) into serving runtime driven by OutputLengthForecastBundle.p50.
4. Add preemption overhead cost model to the simulator (KV-cache eviction latency).

---

## Future Opportunity Ranking (Expected Value × Feasibility)

| rank | opportunity | EV | feasibility | status |
|---|---|---|---|---|
| 1 | Wire SRPT preemptive or aging-SRTF into serving runtime | High | Medium | Both quantified [runs -i/-j]; not yet in runtime |
| 2 | Hybrid Aging+Preemptive SRPT (key = rem/(1+α·wait)) | High | Low effort | Not started — combines run -i + run -j mechanics |
| 3 | Full BurstGPT cross-validation (1.4M rows) | Medium | Low | run_burstgpt_srpt_preemptive_backtest() ready |
| 4 | Wire OutputLengthForecastBundle.p50 as live SRPT prior | High | Low | Infrastructure built (shadow) |
| 5 | Admission gate → cluster simulator integration | Medium | Medium | Implemented (unconnected) |
| 6 | GPU routing on LLM serving trace (TTFT binding) | Medium | Low | Benchmarked on energy trace [run -f] |
| 7 | BOute MOBO routing co-optimisation | High | High effort | Not started |
| 8 | Mooncake trace ingestion (KV prefix reuse) | Low-Med | Low | Not started |
| 9 | Carbon-power MILP joint optimization | Medium | High effort | Not started |

---

## Run 2026-06-20-i — Aging-SRTF anti-starvation + BurstGPT cross-validation

### Q1. What currently limits Aurelius most?

**Long-request starvation under non-preemptive SRTF is quantified and partially
mitigated but not eliminated.** Run -i shows that aging-SRTF (α=0.05) cuts the
long_p99 regression by 55% while retaining +22.4% goodput/$ vs FIFO, and
α=0.01 retains +70.7% goodput/$ vs FIFO with 49% starvation reduction. The
remaining limits are: (a) the serving runtime has no aging_srtf hook yet, and
(b) non-preemptive scheduling still starves long requests under heavy short-job
streams — preemptive SRPT would eliminate rather than bound this.

### Q2. What theoretically offers the largest gain?

**Preemptive SRPT (Shortest Remaining Processing Time):** when a shorter job
arrives, preempt the current job at an operator boundary. The preempted job
resumes with remaining_service = initial − elapsed. FlowPrefill (arXiv:2602.16603)
shows this is feasible with minimal overhead. This would eliminate (not just bound)
long-request starvation while preserving the full SRTF short-request gain.

**Second:** Wire aging_srtf (α=0.01) into the serving runtime path driven by
OutputLengthForecastBundle.p50 — the live oracle prior.

### Q3. Which forecasts are weakest?

1. **OutputLengthForecastBundle.p50 as live SRTF prior** — the serving backtest
   used a perfect oracle prior. The real prior has 30%-CV forecast error. Run -g
   showed robustness at 30% CV noise; alpha sensitivity data suggests similar
   robustness for aging-SRTF.
2. **TTFT p99 tail** — unchanged, baseline_fallback.
3. **Queue wait** — derived proxy only.

### Q4. Which optimizer decisions remain suboptimal?

1. **Serving-queue ordering uses FIFO** — aging_srtf proven better in simulator;
   not yet wired into runtime.
2. **No preemption** — non-preemptive SJF still starves long jobs; aging bounds
   it but does not eliminate it. Preemptive SRPT is the next step.
3. **OutputLengthForecastBundle not driving ordering** — still uses oracle prior.

### Q5. Which workloads benefit least?

**Batch / energy-shifting workloads** — confirmed through multiple runs. SRTF
and aging benefit only in per-request serving queues under contention.

**Small-scale BurstGPT sample** — 51 requests too few to characterize starvation
or confirm goodput/$ generalization. Full 1.4M-row dataset needed.

### Q6. Which research direction appears strongest?

**Preemptive SRPT + aging** would eliminate the starvation problem entirely while
preserving the short-request benefit. FlowPrefill (arXiv:2602.16603) provides the
blueprint. This is a simulator-only change, directly measurable in run -i's framework.

### Q7. What is the shortest path to another +10% gain?

1. Wire aging_srtf (α=0.01) into the serving runtime path → retains +70.7%
   goodput/$ vs FIFO with bounded starvation.
2. Replace oracle prior with OutputLengthForecastBundle.p50 → live prior.
3. Re-run `run_aging_srtf_backtest()` end-to-end with live prior.

### Q8. What is the shortest path to another +50% gain?

The serving-queue aging-SRTF result (α=0.01) already shows +70.7% goodput/$ vs
FIFO in simulation. Realizing even a fraction of this in the serving runtime, combined
with the forecast-prior integration, would achieve +50% vs FIFO at the simulator
fidelity level.

### Q9. What would need to be true to achieve +300%?

Unchanged — the +300% target is vs SLA-aware (not FIFO). The aging-SRTF results
are vs FIFO, not SLA-aware, so they do not directly claim the target. Requires:
live output-length prior, serving-path SRTF/SRPT with aging, heterogeneous GPU
routing on LLM traces, measured queue-wait labels, agentic PDGraph, joint carbon
optimization, pilot telemetry.

### Q10. Which assumptions might be wrong?

1. **Oracle prior over-estimates real gain** — aging-SRTF uses the perfect
   prior (actual tokens as predicted). With 30%-CV noise (run -g), pure SRTF
   short_p90 was −99.5% (vs −99.6% perfect). Expected similar robustness for aging.
2. **Aging parity time** — 87-second parity for p99 requests at α=0.05 was
   calibrated analytically; real optimal α depends on actual request mix and
   service time distribution.
3. **Non-preemptive assumption** — SRPT (preemptive) changes the starvation math
   fundamentally; the aging bound holds only for non-preemptive scheduling.

### Q11. Which benchmark weaknesses exist?

1. **FIFO baseline** — goodput/$ gains are vs FIFO, not vs SLA-aware. Still the
   right comparison for understanding ordering discipline value.
2. **BurstGPT sample too small** — 51 rows, 51 non-failures; need 1.4M-row full
   dataset for meaningful cross-trace confirmation.
3. **No live forecast integration** — perfect oracle prior used throughout; 30%-CV
   robustness is documented (run -g) but not re-tested for aging.

### Q12. Which public datasets should be added?

1. **BurstGPT full dataset** (1.4M rows, CC-BY-4.0) — highest priority for
   cross-trace aging-SRTF validation. `run_burstgpt_aging_backtest()` is ready.
2. **ShareGPT** — output token cross-validation for OutputLengthForecastBundle.
3. **Vidur profiling CSVs** — GPU penalty calibration.

### Q13. What should be attempted next?

**Immediate (next run):**
1. Add preemptive SRPT variant to `simulate_queue` (discipline="srpt_preemptive"):
   remaining_service = initial_service_s − elapsed_s; preempt at server event.
2. Re-run `run_aging_srtf_backtest()` with preemptive SRPT — expected to recover
   the long_p99 regression to near-FIFO levels while preserving SRTF short_p90.

**Short-term (2–3 runs):**
3. Cross-validate on full BurstGPT (1.4M rows).
4. Wire aging_srtf (α=0.01) into serving runtime path with OutputLengthForecastBundle.p50.

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

---

## Run 2026-06-22 — Q&A Updates

### Q14. What is the `safe_high_utilization` policy and why does it help?

`safe_high_utilization` uses the same EWMA-anticipatory sizing as
`constraint_aware` but at rho=0.75 (vs CA's 0.65). This targets higher
GPU utilization, allowing more tokens served per GPU-hour, while maintaining
strict 0% per-tick timeout tolerance. The frontier audit on the full Azure 2024
trace confirmed rho=0.75 is the boundary of the safe anticipatory frontier:
- `anticipatory@0.75`: +12.97% vs CA, timeout=9.465% (SAFE < 10% gate)
- `anticipatory@0.85`: UNSAFE (11.648% > 10% gate)

Key design choices validated by frontier decomposition:
- No hysteresis (step=0.0 neutral contribution)
- EWMA anticipation over reactive (reactive@0.75 was unsafe)
- Strict 0% per-tick timeout_tol (relaxed tol would push aggregate above gate)

### Q15. Why do fixture-scale backtests show TIE vs CA?

At arrival rates below ~10 rps, `_size_for_target` ceiling arithmetic gives
the same base replica count for rho=0.65 and rho=0.75:

  base = ceil(plan_rate / (mu * rho))

When plan_rate is small, ceil(rate/(mu*0.65)) == ceil(rate/(mu*0.75)) == 1.
This is not a policy regression — it reflects that SHU improvement requires
sufficient load to push past the ceiling boundary. The mechanism is validated
at higher scales:
- BurstGPT HF JSONL scale 500×: +13.43% vs CA (SAFE, 2.49% timeout)
- Azure fixture scale 500×: +22.09% vs CA (SAFE, 4.04% timeout)
- Full Azure 2024 trace frontier: +12.97% vs CA (SAFE, 9.465% timeout)

### Q16. What is the current compound gain over sla_aware baseline?

The compound KPI trajectory:
- `sla_aware` baseline: 1× reference
- `constraint_aware` (+25.75% vs sla_aware): economic sizing + cache savings
- `safe_high_utilization` (+12.97% over CA): rho=0.75 utilization expansion
  → estimated ~+41.9% compound vs sla_aware at realistic load

The north-star remains +300% vs sla_aware. Current gap drivers:
1. Serving-queue gains (conformal SRPT: +322% in queue simulator) are not yet
   compounded with economic gains — wiring queue+economic is the next major lever.
2. Semi-clairvoyant scheduling (output length prediction) not yet integrated.
3. Heterogeneous GPU placement scorer not yet built.

### Q17. What's the safe rho expansion feasibility for other policy dimensions?

The frontier pattern (rho=0.75 safe, rho=0.85 unsafe, rho=0.65 conservative)
suggests ~0.10 rho increment per safety tier. The next natural extension:
- Adaptive rho: demand-responsive rho ∈ [0.65, 0.75] based on EWMA load trend
- Time-of-day rho: lower rho during burst hours, higher during trough
- Per-instance rho: heterogeneous sizing based on per-server KV fill rate
All three require real-time KV fill telemetry — blocked on production pilot data.

---

## Run 2026-06-22 (MCS) — min_cost_safe Policy (Per-Tick Minimum-Replica Oracle)

### Q1. What currently limits Aurelius most?

**Economic provisioning factor vs north-star.** After SHU (+12.97% vs CA, merged), the binding
economic constraint is the provisioning factor: current best is ~1.46× over FIFO, but north-star
needs 2.18×. `min_cost_safe` addresses this by finding the minimum replica count per tick (oracle
for the per-tick gate), avoiding SHU's EWMA over-provisioning during ramp-down.

### Q2. What theoretically offers the largest gain?

**Compound economic × queue scheduling** (unchanged). MCS improves the economic provisioning
factor for the LLM serving workload at high load. The queue scheduling gain (conformal SRPT)
remains the larger multiplicative lever, still unmeasured end-to-end with the economic layer.

### Q3. Which forecasts are weakest?

Same as previous run: TTFT p99 at `baseline_fallback` (67% fallback rate). MCS is pure-reactive
and makes no forecasts — it directly evaluates replica counts against the physics model per tick.

### Q4. Where is the serving physics most uncertain?

Same as SHU run. `min_cost_safe` directly searches the physics model (Erlang-C + tail multipliers)
for the minimum safe replica count — no rho-parameterization uncertainty. The main uncertainty
is the BurstGPT token-distribution representativeness.

### Q5. What is the north-star gap today?

After MCS integration at high load:
- Fixture scales (1×, 50×): TIE with SHU and CA — same ceiling arithmetic
- Azure 500×: MCS = +24.55% vs SHU, +52.07% vs CA, 7.05% aggregate timeout (SAFE)
- BurstGPT HF 500×: MCS = +2.57% vs SHU, +16.35% vs CA, 2.59% timeout (SAFE)

The compound KPI trajectory (at high load where policies differentiate):
- `sla_aware` baseline: 1× reference
- `constraint_aware`: +25.75% vs sla_aware (full Azure 2024 week)
- `safe_high_utilization`: ~+41.9% vs sla_aware (frontier-validated)
- `min_cost_safe` (fixture 500×): ~+78% vs sla_aware (extrapolated; full-trace audit pending)

North-star is +300% vs sla_aware. The compound queue + economic gain remains the
primary path; MCS improves the economic factor but does not close the north-star gap alone.

### Q6. What safety gates are proven?

`min_cost_safe` has a STRONGER safety guarantee than SHU:
- **Per-tick gate**: each tick's timeout_rate_pct < 9.5% (strict inequality)
- **Aggregate guarantee**: mean(values each < 9.5%) < 9.5% < 10% — by construction
- SHU achieves 9.465% aggregate on the full Azure 2024 trace (close to gate)
- MCS achieves 7.05% aggregate on Azure 500× fixture (1.95pp lower than SHU aggregate)

### Q7. What is the MCS vs SHU mechanism difference?

SHU: EWMA-anticipatory → sizes for `max(current_rate, ewma_rate)` → protects against
ramp-up but over-provisions during ramp-down (ewma_rate lags behind falling load).

MCS: per-tick oracle → `min r: evaluate_tick(t, r).timeout_rate_pct < 9.5%` → always
finds the true minimum safe replica count for the CURRENT tick's actual load. During
ramp-down, MCS immediately drops replicas (no EWMA lag). During sudden ramp-up,
MCS may be 1 tick slower than SHU (no anticipation).

Net effect at high load: MCS uses ~20% fewer GPU-hours than SHU at Azure 500× (0.12 vs 0.15).

### Q8. Why does MCS tie with SHU at low scales?

At low arrival rates (< ~10 rps), even 1 replica has timeout_rate_pct = 0% (well below the
9.5% gate), so `_min_cost_safe_replicas` returns MIN_REPLICAS=1 — same as SHU's ceiling
arithmetic (base=1 for both rho=0.65 and rho=0.75 at these rates). Differentiation only
appears when load is high enough that 1 replica would breach the 9.5% gate and the optimal
count is 2+ replicas, creating room for MCS to use fewer than SHU's EWMA-inflated estimate.

### Q9. What is the compute cost of MCS?

`_min_cost_safe_replicas` runs an upward search from MIN_REPLICAS, calling `evaluate_tick`
once per replica count. At high load (e.g., Azure 500×, answer=2-3 replicas), this is 2-3
physics evaluations per tick — same order as `_constraint_trim` which also calls evaluate_tick
per probe. The per-tick cost is O(r*) where r* is the optimal replica count (typically 1-10).

### Q10. What residual risk does MCS carry?

The main residual risk is burst under-provisioning: if load spikes suddenly between ticks,
SHU's EWMA would have already sized up (max of current + ewma_rate) while MCS only uses
the current tick's stats. At the fixture scale-500× simulated load pattern, this reactive
gap costs 0% (TIE at 100× where rates are moderate) to +2.57% at 500× (MCS still wins
because ramp-down savings outweigh ramp-up latency). A full-trace frontier audit would
quantify this tradeoff on real burst patterns.

### Q11. What policies should be tested next?

1. **Full-trace MCS frontier audit** on Azure 2024 7-day trace (equivalent of
   `run_azure_2024_safe_utilization_frontier.py` for SHU).
2. **Adaptive gate MCS**: per-tick gate tracks load trend (lower gate during ramp-up,
   higher gate during ramp-down) to combine SHU's anticipatory protection with MCS's
   minimum-cost oracle.
3. **Compound MCS + conformal queue**: wire min_cost_safe into the economic backtest
   alongside the conformal SRPT serving layer to measure the true compound gain.

### Q12. How does MCS relate to the architecture audit findings?

MCS lives entirely in `aurelius/traces/backtest.py` (the public LLM serving leaderboard layer),
orthogonal to `JobScheduler` (energy scheduling), `srtf_serving_backtest.py` (queue discipline),
and `frontier/` (rho controllers). This is consistent with the architecture audit's finding
that the economic provisioning layer and the serving queue layer are independent. MCS improves
the economic provisioning side; the queue-discipline improvements (+557% from conformal SRPT)
remain in the serving layer. These are multiplicatively compoundable.

### Q13. What should be attempted next?

**Highest priority:** Full-trace frontier audit for `min_cost_safe` to validate the +24.5% fixture
result holds on the full Azure 2024 week trace. This requires running `min_cost_safe` against the
raw 7-day trace (same setup as the SHU frontier audit).

**Second priority:** Wire the compound economic × queue backtest: run SLA-safe goodput/$ for the
joint (min_cost_safe economic provisioning) + (conformal SRPT queue discipline) system, replacing
the independence assumption with a measured compound result.

**Current binding constraint:** The economic provisioning factor (MCS at fixture 500×: ~1.78× vs
sla_aware, extrapolated) needs to reach 2.18× for north-star. MCS gets closer than SHU (1.46×)
but the gap remains. Spot/preemptible pricing (projected 2.88× combined) remains the highest
expected-value unlocking mechanism if the full-trace MCS audit confirms the fixture gains.

---

## Run 2026-06-25 — ZFHC Policy (FRONTIER IMPROVEMENT — +1.4%/+4.9% vs AFMS)

### Q1. What currently limits Aurelius most?

**After AFMS, the dominant remaining cost lever is the on-demand floor at high-c ticks.** AFMS keeps exactly 1 on-demand replica (the absolute floor), costing $0.020/tick extra vs all-spot. At c≥8, the stochastic interruption risk is low enough (P(any interrupt at c=8 per tick) ≈ 1.3%) that the all-spot allocation is SLA-safe. ZFHC removes this floor at c≥threshold:

| Trace | AFMS baseline | ZFHC(thr=8) | vs AFMS | vs SLA-oracle |
|-------|--------------|------------|---------|---------------|
| Azure LLM 2024 | 112,316 ($5.74) | **113,904** ($5.66) | **+1.4%** | **+351.9%** |
| BurstGPT HF | 134,093 ($11.16) | **140,647** ($10.64) | **+4.9%** | **+593.5%** |

North-star maintained on both traces. Zero SLA violations all thresholds. BurstGPT new record: +593.5% vs oracle.

### Q2. What theoretically offers the largest gain beyond the current state?

1. **Adaptive threshold** — set threshold per-tick using a preemption risk model: lower threshold during low-risk periods, higher threshold during high-risk periods. Expected gain: 1-3% additional improvement vs static threshold=8.
2. **Cross-region spot arbitrage** — route high-c ticks to cheapest available spot region (SkyPilot-style). Literature: 20-40% cost reduction on top of single-region spot pricing. Not yet explored in Aurelius.
3. **Dynamic spot price signal** — replace static $0.80/hr assumed spot price with time-varying signal. If spot price varies across the trace, a smarter threshold would lower the on-demand floor when spot is cheap and raise it when spot is expensive.
4. **Integration into AureliusOptimizer** — wire ZFHC as a `ReplicaScalingPolicy` so the optimizer can combine it with other policies (conformal queue, MCS autoscaler).

### Q3. What was the hypothesis for this run?

**H1:** Removing the on-demand floor at c≥8 (ZFHC) saves $0.020/tick per affected tick vs AFMS.  
**H2:** The stochastic interruption model (Binomial(c_spot, p_survive), seed=42) produces near-zero actual interruptions at c=8+ ticks, so SLA completions remain at 1.0000.  
**H3:** Best threshold is the lowest tested (8) because it maximizes the number of affected ticks.

### Q4. Were the hypotheses confirmed?

**H1: CONFIRMED.** $0.020/tick saving verified algebraically: (GPU_HOUR_USD − spot_price) × (60/3600) = (2.00−0.80) × 0.01667 = $0.020/tick. Total savings match: Azure 4 ticks × $0.020 = $0.08; BurstGPT 26 ticks × $0.020 = $0.52. (Actual: Azure −$0.08, BurstGPT −$0.52. Exact match.)

**H2: CONFIRMED.** Completion rate = 1.0000 on both traces at all thresholds including threshold=8. Zero SLA violations. p99 response time identical to AFMS (9.95s Azure, 22.9s BurstGPT).

**H3: CONFIRMED.** Threshold=8 is best on both traces due to maximum affected ticks.

### Q5. What new information was learned?

1. **Azure c_max=8 limits ZFHC gains.** Only 4/72 ticks are at c=8, so Azure gains are modest (+1.4%). BurstGPT's richer high-c distribution (26 ticks at c≥8) allows +4.9% improvement.
2. **The $0.020/tick correction is cost-model correct.** Initial comment said $0.033/tick (wrongly counting 1 on-demand in vacuum). The correct saving is the differential: replacing on-demand with spot saves (demand - spot) = $1.20/hr per GPU, not $2.00/hr.
3. **BurstGPT hits 593% vs oracle.** This is the highest achieved vs-oracle percentage in any run. The threshold sweep shows monotonic improvement as threshold decreases (8 > 10 > 12), suggesting even lower thresholds (e.g., 6) might yield further gains — but c=6 has higher interruption probability.

### Q6. What remains uncertain?

1. **Correlated interruptions.** The model assumes i.i.d. interruptions per spot instance. Real cloud spot interruptions can be correlated (AZ-wide events). This is the primary unmodeled risk.
2. **Spot price sensitivity.** At spot=$1.00/hr, the ZFHC saving becomes only $0.017/tick; at spot=$1.40/hr (equal to 70% of demand), the saving drops to $0.010/tick. Current results are anchored to spot=$0.80/hr.
3. **Lower-threshold safety.** Would threshold=6 maintain SLA on both traces? P(any interrupt at c=6 per tick) ≈ 1.0% — marginal risk. Not tested this run.

### Q7. What are the most plausible next improvements?

**Ranked by expected goodput/$ × implementation feasibility:**

| Rank | Opportunity | Expected Δgoodput/$ | Basis | Risk |
|------|------------|--------------------|----|------|
| 1 | Lower ZFHC threshold to 6 (c≥6 all-spot) | +1-3% BurstGPT | More affected ticks; c=6 still has low P(interrupt) | Low: check SLA safety |
| 2 | Cross-region spot arbitrage | +15-30% | Literature (SkyPilot, SpotHedge) | High: new infra |
| 3 | Dynamic threshold via preemption risk model | +1-2% | GFS adaptive quota | Medium: model needed |
| 4 | Wire into AureliusOptimizer as policy | Structural | Phase 3 routing | Low: plumbing only |
| 5 | Real spot price signal | ±5% depending on price signal | Pricing variability | Medium: data needed |

### Q8. Is the north-star still achievable?

**YES — already achieved and extended.** Both traces exceed 4× SLA-oracle:
- Azure: 113,904 / 100,832 = **1.13× over north-star** (+13%)
- BurstGPT: 140,647 / 81,120 = **1.73× over north-star** (+73%)

If the north-star is updated to 500% vs oracle: Azure (113,904 / 25,208 = +351.9%) does not yet reach 500%. BurstGPT (140,647 / 20,280 = +593.5%) already exceeds 500%.

The new research question is: can Azure reach +500% vs oracle? That would require goodput/$ ≥ 5 × 25,208 = 126,040. Current best (ZFHC-8): 113,904. Gap: +10.7%. Cross-region arbitrage or lower thresholds could close this gap.

### Q9. What are the binding constraints going forward?

1. **Azure c_max=8 ceiling.** The schedule rarely goes above c=8, limiting ZFHC gains. Higher c would require either lower cost per tick (more affordable scaling) or higher load (different operating point).
2. **Static interruption model.** The i.i.d. Binomial model is a simplification. Production hardening requires a real interrupt-rate signal.
3. **Single-region model.** No cross-region routing implemented. SpotHedge showed 43% savings from multi-region hedging — this is the biggest unmodeled lever.

### Q10. What was the most important failure mode avoided?

**Using the wrong cost saving formula.** The initial implementation comment claimed $0.033/tick saving (cost of 1 on-demand in vacuum). The correct saving is $0.020/tick (differential: on-demand replaced by spot). Test 8 caught this error before it could propagate to result reporting.

### Q11. What papers guided this run?

1. GFS (arXiv:2509.11134, ASPLOS '26) — capacity-conditioned spot quota: safety scales with c.
2. SpotServe (arXiv:2311.15566, ASPLOS 2024) — all-spot LLM fleet: 54% cost reduction; production deployments use 0 on-demand floor at sufficient c.
3. SageServe (arXiv:2502.14617) — forecast-aware autoscaling: at high c, marginal cost of on-demand floor dominates savings opportunity.

### Q12. How does ZFHC relate to the architecture?

ZFHC is a spot fleet provisioning policy, operating in the same layer as AFMS — the per-tick replica allocation in `srtf_serving_backtest.py`. It does not affect:
- The FIFO queue discipline (unchanged)
- The MCS autoscaler (c_schedule is computed by MCS, then ZFHC allocates spot/on-demand within each c)
- The conformal predictor (unchanged)
- AureliusOptimizer (not yet wired; this is the "integration" direction)

### Q13. What should be attempted next?

**Highest priority (run 2026-06-26):** Test ZFHC with threshold=6 on BurstGPT. The sweep showed monotonic: thr=8 > thr=10 > thr=12 (lower = more affected ticks = more savings). The natural next step is thr=6, which would affect all ticks with c=6,7,8,9,10,11,12,13,14 — a much larger fraction of the schedule. The risk is c=6 has P(any interrupt) ≈ 1.0%, which is safe in expectation but non-trivial.

**Second priority:** Cross-region spot arbitrage. This is the single largest unmodeled lever. Requires modeling multi-region spot prices and routing latency overhead.

**Third priority:** Integration into AureliusOptimizer as a `ReplicaScalingPolicy`. This is structural work (no KPI change expected) but required before any production claim can be made.

**Current frontier (leaderboard state after run-2026-06-25):**
- Azure LLM 2024: 113,904 goodput/$ (ZFHC thr=8, +351.9% vs oracle, north-star ✓)
- BurstGPT HF: 140,647 goodput/$ (ZFHC thr=8, +593.5% vs oracle, north-star ✓)

---

## Run 2026-06-26 — GSF Policy (FRONTIER IMPROVEMENT — +31.0%/+19.3% vs ZFHC)

### Q1. What currently limits Aurelius most?

**After ZFHC, some low-c ticks (c=2,3,4 with c<8) still keep 1 on-demand replica due to the AFMS base formula.** At c=4 with f=0.70: round(0.70×4)=3 spot replicas, 1 on-demand = $0.020/tick extra. The GSF policy sweeps the base spot fraction f over {0.70, 0.80, 0.85, 0.90, 0.95, 1.00}, progressively removing on-demand cost at low-c ticks:

| Trace | ZFHC(8) baseline | GSF(0.95) | vs ZFHC | vs SLA-oracle |
|-------|-----------------|-----------|---------|---------------|
| Azure LLM 2024 | 113,904 ($5.66) | **149,235** ($4.32) | **+31.0%** | **+492.0%** |
| BurstGPT HF | 140,647 ($10.64) | **167,767** ($8.92) | **+19.3%** | **+727.3%** |

North-star maintained on both traces. Zero SLA violations. Both new repository records.

### Q2. What theoretically offers the largest gain beyond the current state?

The GSF policy has reached its practical ceiling: at f=0.95, ALL ticks become all-spot on both traces. The remaining avenue for goodput/$ improvement requires either:
1. **Reducing provisioning cost further**: lower fixed_c, reduce MCS gate conservatism, or cross-region arbitrage.
2. **Increasing completions at lower cost**: accept slightly higher interruption probability (higher f is already maxed at 1.00).

Top opportunities post-GSF:
1. **Lower fixed_c (4→3) on Azure**: At Azure's load level (c_mean=4.5), c=3 might satisfy demand at lower base cost. Expected: +5-10% if c=3 is sufficient.
2. **Adaptive MCS gate (9.5%→8%)**: Less conservative over-provisioning; reduces c_mean from 4.5 toward 4.0.
3. **Cross-region spot arbitrage**: Route each tick to cheapest available spot region. Literature: 20-40% additional cost reduction.

### Q3. What was the hypothesis for this run?

**H1:** Raising the spot fraction f above 0.70 at low-c ticks removes the on-demand floor and reduces cost.  
**H2:** At f=0.95, banker's rounding on c=1,2,3,4 all yields c_spot=c (all-spot), making the policy equivalent to all-spot-always.  
**H3:** The step from f=0.90 to f=0.95 produces the largest jump in goodput/$ due to the discrete nature of c values and rounding.

### Q4. Were the hypotheses confirmed?

**H1: CONFIRMED.** Cost reduction at f=0.95: −23.7% (Azure, $5.66→$4.32), −16.2% (BurstGPT, $10.64→$8.92).

**H2: CONFIRMED.** At f=0.95: n_ticks_c_all_spot = 72/72 (Azure), 154/154 (BurstGPT). Every single tick is all-spot.

**H3: CONFIRMED.** The jump from f=0.90 (40/72 ticks all-spot, $4.96) to f=0.95 (72/72 ticks, $4.32) is the dominant step. Gains at f=0.70→0.80→0.85→0.90 are monotonic but smaller.

### Q5. What new information was learned?

1. **The fraction sweep reveals a step-function gain at f=0.95, not a smooth gradient.** Between f=0.90 and f=0.95: Azure jumps +$0.64/run (12.9% additional cost reduction) and goodput/$ jumps +19,256 (14.8% additional). This is not visible from just testing f=0.70 and f=1.00.
2. **f=0.95 ≡ f=1.00 in practice.** Both produce identical cost and goodput/$ because banker's rounding on c=1,2,3,4 at f=0.95 already reaches c_spot=c.
3. **Azure north-star gap is 1.7%.** 149,235 vs 151,248 (6× oracle = +500%). The policy has been exhausted for fraction optimization; crossing the gap requires other levers.
4. **BurstGPT now exceeds 7× oracle (+727%).** The BurstGPT trace benefits more from higher spot fractions due to its heavier load distribution and more sub-threshold ticks.

### Q6. What remains uncertain?

1. **Azure +500% north-star.** At 149,235 vs 151,248, the gap is only 1.7% but the policy ceiling has been reached. Need new lever (lower fixed_c or MCS gate).
2. **Correlated interruptions.** i.i.d. Binomial model remains. Real cloud spot interruptions can be AZ-wide.
3. **Spot price sensitivity.** Results anchored to spot=$0.80/hr. At spot=$1.00/hr, the all-spot savings shrink.

### Q7. What are the most plausible next improvements?

**Ranked by expected goodput/$ × implementation feasibility:**

| Rank | Opportunity | Expected Δgoodput/$ | Basis | Risk |
|------|------------|--------------------|----|------|
| 1 | Lower fixed_c 4→3 on Azure | +5-10% | c=3 may saturate demand at c_mean=4.5; sub-threshold ticks cost less | Low: parameter change |
| 2 | Adaptive MCS gate (9.5%→8%) | +3-5% | Less conservative over-provisioning; reduces c_mean | Low: parameter change |
| 3 | Cross-region spot arbitrage | +15-30% | Literature (SkyPilot, SpotHedge): 20-40% multi-region discount | High: new infra |
| 4 | Adaptive interruption model | ±2% | Replace static 10%/hr with cloud signal | Medium: data needed |
| 5 | Wire into AureliusOptimizer | Structural | Phase 3 routing; no KPI change expected | Low: plumbing |

### Q8. Is the north-star still achievable?

**YES — and BurstGPT already exceeds 7× oracle.** Azure is at +492% vs 500% target (98.4% of way there).

The +500% target on Azure requires goodput/$ ≥ 151,248 (6× oracle). Current: 149,235. Gap: 2,013 goodput/$. At same goodput (5880 completed), this means cost needs to drop from $4.32 to $4.26 — a further 1.4% cost reduction. Achievable with:
- fixed_c 4→3: expected to reduce c_mean below 4.0, cutting cost proportionally
- MCS gate 9.5%→8%: less over-provisioning at load troughs

### Q9. What are the binding constraints going forward?

1. **All-spot ceiling reached.** GSF(0.95) = 100% spot on all ticks. No further savings from fraction tuning.
2. **c_mean=4.5 (Azure) is the next lever.** The MCS schedule drives cost; reducing provisioned capacity (via lower fixed_c or MCS gate) is the only remaining lever within this simulation framework.
3. **Static interruption model.** i.i.d. Binomial. Production hardening requires real interrupt-rate signal.

### Q10. What was the most important failure mode avoided?

**Conflating f=0.95 and f=1.00 as identical before verifying.** The test suite confirms they are operationally identical (n_ticks_c_all_spot = 72/72 and 154/154 for both), preventing an incorrect claim that f=1.00 offers "additional" improvement beyond f=0.95.

### Q11. What papers guided this run?

1. GFS (arXiv:2509.11134, ASPLOS '26) — graduated spot quota: fraction sweep was directly motivated by GFS's dynamic spot quota parameter.
2. SpotServe (arXiv:2311.15566, ASPLOS 2024) — 100% spot fleet at sufficient capacity; validated our f=1.00 ceiling result.
3. SkyPilot (NSDI '23) — cross-region spot arbitrage: the top-ranked next opportunity post-GSF.

### Q12. How does GSF relate to the architecture?

GSF generalizes AFMS (f=0.70 with c-1 floor) and ZFHC (all-spot at c≥8). At f=0.95 the policies collapse to a single point: 100% spot always. GSF is implemented entirely in `srtf_serving_backtest.py` and does not touch:
- The FIFO queue discipline
- The MCS autoscaler (c_schedule computed by MCS, GSF allocates within each c)
- The conformal predictor
- AureliusOptimizer

### Q13. What should be attempted next?

**Highest priority (run 2026-06-27):** Lower fixed_c from 4 to 3 on Azure. At c_mean=4.5 with the current schedule, many ticks spend at c=3 or c=4 already. Reducing the fixed_c parameter shifts the distribution leftward, potentially reaching +500% vs oracle.

**Second priority:** Adaptive MCS gate (9.5%→8%). Combined with lower fixed_c, this could close the remaining 1.7% gap.

**Third priority:** Cross-region spot arbitrage. This is the single largest unmodeled lever.

**Current frontier (leaderboard state after run-2026-06-26):**
- Azure LLM 2024: **149,235** goodput/$ (GSF f=0.95, **+492.0%** vs oracle, north-star ✓, gap to +500%: 1.7%)
- BurstGPT HF: **167,767** goodput/$ (GSF f=0.95, **+727.3%** vs oracle, north-star ✓)

## Run 2026-06-23 — AMCSG-LFC + Fine Grid + DLAG (THREE-LEVER NULL RESULT)

### Q1. What was the north-star gap entering this run?

Azure: 150,630 goodput/$ (AMCSG gate=12.5%). Target: 151,248 (6× oracle). Gap: 618 goodput/$ (0.41%).
BurstGPT: 168,270 goodput/$ (AMCSG gate=12.5%). Already above +500% north-star (121,680). Cross-validation only.

### Q2. What hypotheses were tested?

Three independent levers:
- **(A) AMCSG-LFC (fixed_c=3):** Reduce time-warp calibration factor → lower c_mean → lower cost → higher goodput/$.
- **(B) Fine gate grid (fixed_c=4):** Resolve the 12.5%→15.0% safety boundary at 0.5% resolution.
- **(C) DLAG (Dynamic Load-Aware Gate):** Per-tick gate = base_gate at high load, max_gate at idle.

### Q3. What did each hypothesis find?

**(A) AMCSG-LFC — UNSAFE on Azure:**
- Reducing fixed_c from 4→3 reduces the time-warp multiplier by 25%.
- Azure p99 = 10.030s for ALL gates (even gate=9.5%) → SLA=10s violated.
- Root cause: Azure's heavy-tailed GPU service times require fixed_c≥4 to keep Erlang-C conservative enough.
- BurstGPT safe (SLA=30s has large headroom): LFC c_mean drops from 4.5 to ~3.7.
- **Conclusion:** fixed_c is a hard constraint for Azure. Minimum safe fixed_c=4.

**(B) Fine gate grid — NULL (boundary at 13.0%→13.5%, not 12.5%→15.0%):**
- Gates 12.5% and 13.0% produce IDENTICAL c_schedule (c_mean=4.458). Same cost, same goodput/$.
- Gate=13.5% pushes p99=10.030s > SLA=10s (unsafe).
- The Erlang-C function is integer-valued; 12.5% and 13.0% round to same c per tick.
- **Conclusion:** The safe frontier is 13.0% (not a new win — identical to 12.5%). No improvement possible via fine grid.

**(C) DLAG — NULL (collapses to base_gate on uniform loads):**
- Azure is calibrated to ρ=target_rho=0.85 throughout. Per-tick slack = max(0, 1−0.85/0.85) = 0 for every tick.
- gate_k = base_gate = 9.5% for all ticks regardless of max_gate. All max_gate values (15–30%) produce identical results.
- Azure DLAG: 149,235 goodput/$ (−0.93% vs AMCSG 150,630). n_sla_safe=5823 (57 violations at tick boundary).
- BurstGPT DLAG: 168,018 at max_gate=25% (marginal, below AMCSG 168,270).
- **Conclusion:** DLAG requires genuine load variance (bursty traces). Azure at ρ≈target_rho offers no idle slack.

### Q4. What is the new structural understanding?

The Azure +500% north-star gap has three confirmed closure mechanisms that DO NOT WORK:
1. ~~Lower fixed_c~~ (unsafe: p99 > SLA at fixed_c=3)
2. ~~Fine gate sweep~~ (identical c_schedule at 12.5% and 13.0%; boundary at 13.0→13.5% with no win)
3. ~~Dynamic per-tick gate~~ (collapses to base_gate when ρ ≈ target_rho throughout)

The gap (618 goodput/$, 0.41%) requires a fundamentally different lever. The Erlang-C gate family
has been exhausted at fixed_c=4. The provisioning model ceiling has been hit.

### Q5. What is the north-star gap after this run?

**UNCHANGED: 0.41%** (618 goodput/$). Azure: 150,630 vs target 151,248.
BurstGPT remains above north-star: 168,270 vs threshold 121,680.

### Q6. What structural levers remain unexplored?

**Highest priority:**
1. **Tick granularity (tick_seconds=30s):** Shorter ticks reduce over-provisioning at burst transitions.
   Azure's 60s ticks provision for the worst request in the tick; finer ticks adapt faster. Low risk.
2. **Per-tick SLA_eff = 0.9 × SLA_s:** Provision against 9s instead of 10s. Creates 10% headroom buffer,
   potentially allowing a safe gate between 13.0% and 13.5% without SLA breach. Pure re-parameterization.
3. **Spot fraction above 95% (f=0.97–0.99):** ZFHC already allows all-spot at c≥8. Below that, increasing
   spot fraction from 0.95 toward 1.0 reduces on-demand cost further. Interruption tail risk.
4. **Cross-tick work stealing:** Servers completing tick k early start tick k+1 requests. Reduces effective
   queue depth at burst entries, potentially reducing c_k for boundary ticks.
5. **Abs-conformal SRTF calibration on Azure (oracle baseline):** Establishes how much scheduling alone
   can contribute on top of the current provisioning floor.

**Lower priority (requires new infrastructure):**
- Cross-region spot arbitrage (SkyPilot regions)
- Batch grouping by token length (reduce service-time variance)
- Preemptive per-request timeout admission

### Q7. What are the binding constraints?

1. **fixed_c ≥ 4 for Azure:** Proven hard constraint. LFC is unsafe.
2. **Gate ceiling at 13.0%:** The Erlang-C integer-c rounding makes 12.5% and 13.0% equivalent.
   13.5% is unsafe. No gate above 13.0% is safe at fixed_c=4.
3. **DLAG degeneracy:** Dynamic gating requires load variance. Azure at ρ=0.85=target_rho gives no slack.
4. **DLAG safety:** Even base_gate=9.5% in DLAG has n_sla_safe=5823 (vs 5880 in AMCSG), meaning
   the idle-tick under-provisioning causes 57 SLA violations. DLAG is NOT safety-equivalent to AMCSG at gate=9.5%.

### Q8. What should be attempted next?

**Priority 1 (run next):** `tick_seconds=30s` sweep on Azure. Implement `run_amcsg_azure_30s_backtest()`.
Hypothesis: halving tick duration reduces stranded capacity at burst boundaries, lowering effective c_mean
by 5–10%, closing the 0.41% gap.

**Priority 2:** `_erlang_c_sla_timeout_pct(sla_s=9.0)` effective SLA tightening. Check if gate=13.5%
becomes safe when the Erlang-C is calibrated against 9s instead of 10s, while the actual DES still uses 10s.
This exploits the M/M/c conservatism without changing the external SLA contract.

**Priority 3:** DLAG on a synthetically bursty Azure re-sample (interleave high-load and idle ticks).
This isolates whether DLAG's mechanism is sound and just needs a bursty trace to prove it. Do NOT
use this for north-star claims — it's a mechanistic validator only.

---

## Run 2026-06-24 — Aging SRTF + AMCSG Compound (HONEST NULL RESULT — Five-Failure Rule integration experiment)

### Q1. What currently limits Aurelius most?

**Prediction degeneracy**: The running-median live prior (window=200) collapses per-request token predictions to near-constant (stdev=8.1 vs actual stdev=93.1, 37 unique values ≈91 tokens). Aging SRTF priority key degenerates to near-FIFO. Queue discipline cannot exceed FIFO performance without accurate per-request predictions.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Per-request token prediction accuracy (Trail/NP-SRPT style). Under current running-median prior, queue discipline is ceiling-limited at FIFO performance.

### Q3. Which forecasts are weakest?

Per-request token length prediction. Running-median window=200 is the binding constraint for queue discipline experiments.

### Q4. Which optimizer decisions remain suboptimal?

Queue dispatch order — but only improvable if prediction accuracy is substantially improved.

### Q5. Which workloads benefit least from aging SRTF?

Both traces show null result. BurstGPT shows +0.09% (noise level). Azure shows +0.00%.

### Q6. Which research direction appears strongest?

Five-Failure Rule active. Per-request token prediction (blocked by pilot telemetry) is the highest-EV research direction. Trail (ICLR 2025, arXiv:2410.01035) addresses this directly.

### Q7. What is the shortest path to another +1% gain?

Under Five-Failure Rule: none available without pilot telemetry for per-request token prediction. Architecture is converged, all queue discipline paths are prediction-limited.

### Q8. What is the current north-star status?

**Both traces north-star achieved.** Azure: 159,578 gp/$ (OSOTSS). BurstGPT: 178,109 gp/$ (OSOTSS). The aging SRTF experiment does not change these.

### Q9. What would need to be true to maintain north-star?

North-star already achieved. No regression present.

### Q10. Which assumptions might be wrong?

**CONFIRMED:** "Running-median prior produces useful predictions" was wrong. stdev=8.1 vs actual stdev=93.1. Only 37 unique predicted values. The prediction diversity needed for SRTF-class improvements is absent.

### Q11. Which benchmark weaknesses exist?

1. **Two public traces only** — Azure LLM 2024 and BurstGPT HF. Third trace blocked (Alibaba: image gen workload; ShareGPT: no timestamps; LMSYS: no processed data).
2. **Running-median prior** — Not representative of production token prediction quality.

### Q12. Which public datasets should be added?

None viable for OSOTSS/aging-SRTF replay. All three candidate traces are blocked for different structural reasons.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE ACTIVE (6/5). Allowed actions: integration, validation, diagnosis, architecture simplification.**

1. **Accept prediction-degeneracy as binding constraint** — Document that aging SRTF / SRPT improvements require per-request token prediction accuracy that the running-median prior cannot provide.
2. **Architecture simplification** — Deprecate dead frontier code (EVAL_WORKLOAD, BATCH_INFERENCE).
3. **Thin-delegate promotion** — Route remaining non-OSOTSS backtests through AureliusOptimizer facade.
4. **Do NOT attempt new queue discipline variants** — All are prediction-limited; outcomes are predetermined.

**Prediction degeneracy diagnosis:**
- `LIVE_PRIOR_WINDOW=200`: running-median stdev=8.1 (actual stdev=93.1)
- 37 unique predicted values, mode≈91 tokens
- Aging key collapses to near-constant → degenerate to near-FIFO
- BLOCKED by pilot telemetry — no path to per-request accuracy without production deployment

Results: `research/results/aging_srtf_amcsg_compound_2026-06-24.{md,json}`
Tests: `tests/test_aging_srtf_amcsg_compound.py` (24 tests: 23 passed, 1 skipped)

---

## Run 2026-06-24 — Alibaba GenAI Third-Trace Cross-Validation (Benchmark Realism)

**Five-Failure Rule counter: 6/5 (ACTIVE)**

### Q1. What currently limits Aurelius most?

**Five-Failure Rule ACTIVE (6/5).** The binding constraint is prediction degeneracy: the running-median
prior collapses per-request predictions to near-constant (stdev=8.1 vs actual 93.1). All queue-discipline
experiments are ceiling-limited at FIFO. The second constraint is the absence of a third production-grade
trace class for LLM serving.

### Q2. What theoretically offers the largest gain beyond OSOTSS?

Per-request token prediction (Trail/ICLR 2025, arXiv:2410.01035) is the highest-EV path, blocked by
pilot telemetry requirements. Energy-denominator reduction (GreenLLM arXiv:2508.16449: 45% energy via
SLO-aware DVFS) is a second candidate that doesn't require prediction accuracy.

### Q3. Which forecasts are weakest?

Per-request token length prediction. Running-median window=200 is near-constant and cannot support
queue-discipline improvements. Cold-start duration is well-calibrated when affinity metadata is available
(2.79s), but degrades to 22.85s without it.

### Q4. Which optimizer decisions remain suboptimal?

Queue dispatch order (prediction-limited, blocked). Cold-start / adapter preloading policy (already
exploited by affinity routing). Energy denominator (GreenLLM DVFS not implemented).

### Q5. Which workloads benefit least from current constraint_aware?

Text-LLM serving (Azure, BurstGPT) shows +5.94–5.85% OSOTSS gains but requires token-prediction accuracy
for further gains. Both traces are relatively homogeneous (single-model-class). The Alibaba GenAI trace
(multi-model-class LoRA) shows much larger gains (+38.2%) because the affinity signal is stronger when
adapter cold-starts are costly.

### Q6. Which research direction appears strongest?

1. Energy-denominator reduction (GreenLLM DVFS, arXiv:2508.16449) — orthogonal to prediction accuracy, no pilot requirement.
2. Canonical integration of genai_backtest.py through AureliusOptimizer (architecture convergence).
3. Per-request token prediction via Trail (blocked, requires pilot telemetry).

### Q7. What is the shortest path to another +1% gain?

Under Five-Failure Rule: energy-denominator reduction via DVFS scheduling (if CAISO/energy data available
in serving benchmark). Alternative: canonical integration of GenAI policy through AureliusOptimizer.
No new LLM-serving queue-discipline experiments — all prediction-limited.

### Q8. What is the current north-star status?

**Both LLM traces north-star achieved.** Azure: 159,578 gp/$ (OSOTSS). BurstGPT: 178,109 gp/$ (OSOTSS).
The GenAI ablation adds a third data point (9.8514 gp/$ constraint_aware) on a separate workload type.
Not directly comparable (different units/scale: LLM uses per-job tokens, GenAI uses exec_time seconds).

### Q9. What would need to be true to maintain north-star?

North-star already achieved on both LLM traces. No regression present. GenAI workload is a separate
benchmark track — north-star for GenAI is constraint_aware (9.8514 gp/$), not yet routed through
AureliusOptimizer.

### Q10. Which assumptions might be wrong?

1. **"Affinity effect is universal"** — VERIFIED across two workload classes (LLM + LoRA image gen).
   Confidence increased.
2. **"Cold-start is the dominant cost driver"** — CONFIRMED on Alibaba GenAI: 22.85s without affinity
   vs 2.79s with affinity. 61.7% of the gain is affinity/cold-start.
3. **"sla_aware is a fair baseline"** — CONFIRMED WRONG. sla_aware fails SLA (6.214% timeout on GenAI
   trace) and is excluded. Strongest SLA-safe baseline is constraint_aware_no_affinity.

### Q11. Which benchmark weaknesses exist?

1. **Alibaba GenAI cold-start calibration absent:** `cold_start_calibration_s: {}` because pipeline
   CSV files are not publicly available (empty downloads). Default cold-start values used.
2. **No per-request latency SLA:** GenAI SLA is `2.0 × exec_time + 30.0s` — a generous proxy.
3. **genai_backtest.py not canonical:** not routed through AureliusOptimizer; separate simulation.

### Q12. Which public datasets should be added?

No new datasets recommended under Five-Failure Rule. The three committed traces (Azure LLM 2024,
BurstGPT, Alibaba GenAI 2026) cover the main serving workload classes. Next priority is canonical
integration of the existing GenAI benchmark, not new data.

### Q13. What should be attempted next?

**⛔ FIVE-FAILURE RULE ACTIVE (6/5). Allowed actions: integration, validation, diagnosis, architecture simplification.**

1. **Canonical GenAI integration:** Route genai_backtest.py through `AureliusOptimizer(policy="replica_scaling")`
   or a new `GenAIServingPolicy`. Requires 0% KPI drift parity gate. Updates OPTIMIZER_UNIFICATION_PLAN.md.
2. **Energy denominator experiment:** GreenLLM-style DVFS scheduling. Orthogonal to prediction accuracy.
   Does not require new queue discipline.
3. **OSOTSS parity test suite:** 38 tests committed, all passing — confirms Phase 3 canonical routing closure.
4. **Do NOT attempt new queue discipline variants** — all are prediction-limited.

Results: `research/results/alibaba_genai_third_trace_2026-06-24.{md,json}`
Tests: `tests/test_osotss_canonical_routing_parity.py` (38 tests: 38 passed)
