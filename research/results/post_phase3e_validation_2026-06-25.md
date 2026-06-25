# Run: Post-Phase-3e Validation + Research Review (2026-06-25)

**Classification:** BENCHMARK REALISM AUDIT  
**Five-Failure Rule:** ACTIVE (5/5) — compliant (validation only, no new modules)  
**Date:** 2026-06-25  
**Branch:** claude/tender-einstein-4e0gsh

---

## 1. PR Hygiene

| PR | Title | Classification | Action |
|---|---|---|---|
| #74 | Phase 3e: Route constraint_aware/SHU through AureliusOptimizer (0% KPI drift) | SAFE_INFRASTRUCTURE | **MERGED** → SHA 49877c7 |
| #70 | benchmark realism audit + research review — null result | RESEARCH_ARTIFACT | LEFT OPEN (base = non-main branch, cannot merge to main) |
| #54 | Phases 4 + 5 — Canonical Frontier Discovery & Integration Master Plan | RESEARCH_ARTIFACT | LEFT OPEN (docs-only, says "recommend leaving open for review") |

**PR #74 merged first.** 103 tests pass post-merge.

---

## 2. Repository Audit (post Phase 3e)

### Architecture state

| Policy | Path | Status |
|---|---|---|
| energy | `AureliusOptimizer(policy="energy")` → `EnergySchedulingPolicy` → `JobScheduler` | ACTIVE |
| serving_queue | `AureliusOptimizer(policy="serving_queue")` → `ServingQueuePolicy` (conformal SRPT) | ACTIVE |
| replica_scaling | `AureliusOptimizer(policy="replica_scaling")` → `ReplicaScalingPolicy.optimize()` (amcsg/sotss_min/online_sotss/forecasted_mcs) | ACTIVE |
| replica_scaling (serving) | `AureliusOptimizer(policy="replica_scaling")` → `ReplicaScalingPolicy.optimize_from_ticks()` (constraint_aware/SHU) | ACTIVE — Phase 3e |
| genai_serving | `AureliusOptimizer(policy="genai_serving")` → `GenAIServingPolicy` | ACTIVE — Phase 3d |
| placement | stub | SHADOW |
| admission | stub | SHADOW |

**Architecture convergence: COMPLETE.** All 6 primary production policies flow through AureliusOptimizer.

### Phase execution status

| Phase | Status |
|---|---|
| 1a — Canonical interface | DONE |
| 2 — Extract serving discipline | DONE |
| 3/3b/3c/3d/3e — Route entry points | DONE (all sub-phases) |
| 5 — Deprecate dead code | DONE |
| **1b — Unify replay loops** | **NOT STARTED** |
| **4 — Frontier promotion** | **NOT STARTED** |

---

## 3. Bottleneck Identification

**Question: What is currently preventing Aurelius from achieving another +25% against the strongest fair baseline?**

| Domain | Current best | vs strongest fair baseline | Oracle ceiling | Gap to oracle |
|---|---|---|---|---|
| Azure LLM serving | OSOTSS 159,578 gp/$ | +5.94% vs AMCSG | SOTSS-MIN 160,107 (+6.29%) | 0.35% (94.4% captured) |
| BurstGPT serving | OSOTSS 178,109 gp/$ | +5.85% vs AMCSG (goodput/$) | n_sla_safe gap −15 (structural) | Irreducible |
| Energy canonical | 0.337299 gp/$ | +11.1% vs current_price_only | Unknown | Unknown |
| GenAI Alibaba | 9.84 gp/$ | +38.2% vs no_affinity | Unknown | Unknown |

**Root cause of replica-scaling ceiling:** The BurstGPT 15-request SLA gap is a stochastic/deterministic mismatch at minimum-capacity ticks. Confirmed deterministic (std=0 across 5 seeds). All five historical optimization attempts (C1PGS, SOTSS-GSF, Adaptive EWMA, SSM, OSSC) targeted this gap and failed because the gap is irreducible: Binomial(c_spot, 0.9982) ≈ c_spot makes the stochastic oracle identical to the deterministic oracle.

**Conclusion:** The next +25% cannot come from replica-scaling improvements. It must come from: (a) energy policy improvements, (b) GenAI policy improvements, or (c) Phase 1b combination search.

---

## 4. Canonical Benchmark Validation

All replays run post Phase 3e merge (SHA 49877c7). Results match historical values.

### 4.1 AMCSG Gate Sweep (run_amcsg_backtest.py)

| Trace | Gate | Goodput/$ | vs Oracle | NS-500 |
|---|---|---|---|---|
| Azure LLM 2024 | 12.5% | **150,630** | +497.5% | ✗ |
| BurstGPT HF | 11.0% | **168,270** | +729.7% | ✓ |

Match vs historical: **EXACT** ✓

### 4.2 SOTSS-MIN (run_sotss_backtest.py, gate=20%)

| Trace | Goodput/$ | vs AMCSG | NS-500 |
|---|---|---|---|
| Azure LLM 2024 | **153,013** | +1.58% | ✓ |
| BurstGPT HF | **169,030** | +0.45% | ✓ |

Match vs historical: **EXACT** ✓

### 4.3 Canonical Energy (run_canonical_backtests.py)

| Policy | Goodput/$ | Deadline misses |
|---|---|---|
| current_price_only | 0.303676 | 0 |
| constraint_aware wrapped | **0.337299** (+11.1%) | 0 |

Match vs historical: **EXACT** ✓

### 4.4 BurstGPT Fixture (run_burstgpt_backtest.py)

| Policy | Goodput/$ |
|---|---|
| constraint_aware | 8691.77 |
| min_cost_safe | 8691.77 |

Status: **STABLE** ✓

---

## 5. Research Review (3 papers)

### Five-Failure Rule compliance

ACTIVE (5/5). No new modules. Focus on integration, benchmark realism, bottleneck diagnosis.

### Papers reviewed

#### Paper 1: DynamoLLM (HPCA 2025)

- **Source:** Microsoft Research / HPCA 2025 (provider of Azure LLM trace)
- **Problem:** Dynamic resource allocation for LLM inference
- **Objective:** Minimize GPU cost while meeting per-request latency SLAs
- **Algorithm:** (1) Offline profile GPU efficiency curve (throughput vs configuration); (2) Online greedy allocation that selects GPU count from efficiency curve given current traffic
- **Assumptions:** Efficiency curve stable across traffic regimes; traffic approximated by Poisson
- **Telemetry required:** Actual per-request latencies, GPU utilization, profiling data
- **Mapping to AureliusOptimizer:** Conceptually similar to OSOTSS — both optimally allocate GPU capacity per arrival rate estimate. DynamoLLM uses a profiled curve; OSOTSS uses Erlang-C theory.
- **Why not implement:** The profiled efficiency curve is a trace-specific prior (requires profiling the specific GPU model and workload class). Erlang-C is already assumption-free at the theoretical level. Adding a profiled curve would be a calculated prior.
- **Verdict:** **NOT APPLICABLE** — Erlang-C already theoretically sound; profiled curve introduces trace-specific priors.

#### Paper 2: Llumnix (OSDI 2024)

- **Source:** Alibaba / OSDI 2024
- **Problem:** Live migration of LLM requests across serving instances
- **Objective:** Reduce queue imbalances and tail latency
- **Algorithm:** Monitors per-replica queue lengths; preemptively migrates running/queued requests to less-loaded replicas
- **Assumptions:** Migration is cheap (< 100ms at prefill boundary); requests can be interrupted and resumed
- **Telemetry required:** Per-replica queue lengths, migration latency, KV cache state
- **Mapping to AureliusOptimizer:** Would require a new migration/routing policy
- **Why not implement:** Five-Failure Rule prohibits new modules. Migration is a new decision dimension not in any existing policy.
- **Verdict:** **NOT APPLICABLE** — Five-Failure Rule: no new modules.

#### Paper 3: Preble (NSDI 2024)

- **Source:** Stanford / NSDI 2024
- **Problem:** Prefix caching for LLM serving
- **Objective:** Maximize KV cache reuse to reduce TTFT for repeated prefixes
- **Algorithm:** Tracks per-request prefix hashes; routes to replicas with matching cached prefixes; evicts low-popularity prefixes
- **Assumptions:** Users repeat prompts or share system prompts; prefix hash matching is cheap
- **Telemetry required:** Per-request prefix text or hash, per-replica KV cache state
- **Mapping to AureliusOptimizer:** BurstGPT cache_affinity_baseline already approximates this; GenAI model-affinity does model-level routing (analogous to prefix routing at adapter level)
- **Why not implement:** BurstGPT has no session IDs (prefix locality = model-level proxy only, already captured). GenAI model-affinity already provides 61.7% of the GenAI gain. Full Preble integration would need session-level data not present in BurstGPT.
- **Verdict:** **NOT APPLICABLE** — already approximated; no additional gain without session data.

### Summary

All three papers address problems already covered by existing Aurelius policies or require new modules. No new implementation warranted this run.

---

## 6. Same-Conditions Checklist

Not applicable for this run (no optimization candidate implemented). All replays used identical parameters to historical runs.

---

## 7. Gain Decomposition

N/A — no KPI delta (validation run).

---

## 8. Classification

**BENCHMARK REALISM AUDIT** — Five-Failure Rule compliant.

- Merged PR #74 (valid infrastructure improvement)
- All 4 canonical public replays confirmed against historical values
- 3 research papers reviewed, all NOT APPLICABLE
- No new modules added
- Architecture convergence complete (6/6 policies in AureliusOptimizer)

---

## 9. Identified Next Research Directions

### Highest priority: Phase 1b — Replay Loop Unification

**Bottleneck addressed:** Cannot test energy+serving combination. The two domains are on disjoint workloads (batch training vs LLM serving) and cannot be combined without a unified replay engine.

**Mechanism:** Unify 4 separate replay loops (`canonical_backtests.py`, `backtest.py`, `genai_backtest.py`, `gpu_scheduling.py`) into one engine. Expected KPI delta: 0% (parity gate). Enables future combination search.

**Risk:** HIGH complexity. Physics models differ across workload types (M/M/c Erlang-C vs energy objective vs LoRA serving vs GPU packing). Would require a common input/output schema and careful parity testing for all 4 modes.

**Decision changed:** None directly — this is an architecture enabler.

**Strongest fair baseline:** Each workload's existing strongest fair baseline (unchanged).

**Path to AureliusOptimizer:** Phase 1b output is a unified `AureliusOptimizerReplay` entry point.

### Second priority: GenAI EWMA Alpha Configurability

**Bottleneck addressed:** The EWMA alpha=0.5 in GenAIServingPolicy is a fixed prior. A principled alpha selection rule (based on traffic CV) could improve the GenAI KPI without trace-specific tuning.

**Mechanism:** Make `GENAI_EWMA_ALPHA` configurable in `GenAIServingPolicy`. Sweep alpha ∈ {0.1, 0.2, 0.3, 0.5, 0.7, 0.9} on the Alibaba GenAI trace. Establish a principled selection rule based on arrival rate coefficient of variation (CV): higher CV → higher alpha (faster response to changes).

**Risk:** MEDIUM. If alpha selection is trace-specific (only improves on Alibaba), it violates the anti-gaming rule. Need to validate that the selection rule generalizes across all traces.

**Decision changed:** EWMA alpha for GenAI serving (per-tick arrival rate prediction).

**Operator relevance:** A GPU fleet operator running multi-model GenAI serving would want optimal EWMA responsiveness for their traffic patterns.

### Third priority: Phase 4 — Frontier Promotion

**Bottleneck addressed:** Frontier recommendations (safe ρ bounds) are currently advisory-only. Promoting to hard constraints could improve efficiency.

**Evidence:** Static SUF +13% over CA on Azure (analysis-only, partially benchmarked).

**Path to AureliusOptimizer:** Frontier ρ-ceiling becomes a hard constraint in `ReplicaScalingPolicy`.

**Risk:** MEDIUM. Frontier evidence only exists for Azure; BurstGPT result unknown. May conflict with OSOTSS (which already optimizes below the frontier bound).

---

## 10. Merge Recommendation

**This run produces documentation/research artifacts only.** No runtime optimizer changes.

Merge criteria: safe infrastructure, documentation updates, benchmark validation artifacts.

**Recommendation: MERGE.**

---

## 11. Run Artifacts

- `research/results/post_phase3e_validation_2026-06-25.md` — this file
- `research/results/post_phase3e_validation_2026-06-25.json` — structured results
- `research/ROADMAP.md` — updated with this run
- `research/GAP_ANALYSIS.md` — updated with this run
