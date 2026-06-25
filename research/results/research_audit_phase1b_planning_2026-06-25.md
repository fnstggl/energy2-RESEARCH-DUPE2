# Research Audit + Phase 1b Planning — 2026-06-25

## Run Classification

**REPORT ONLY — Five-Failure Rule Compliant (5/5 active)**

No frontier improvement attempted. This run confirms parity, reviews recent research, and produces the Phase 1b implementation plan.

---

## 1. PR Hygiene

| PR | Title | Classification | Action |
|----|-------|----------------|--------|
| #77 | Phase 4 extended validation — prefill-savings integration fix + 20 parity tests | Safe infrastructure | **MERGED** (all CI pass, 0% KPI drift, 20 new parity tests) |
| #70 | Benchmark realism audit (2026-06-24) | Obsolete (base = `claude/happy-pascal-pvp0fd`, not main) | Left open |
| #54 | Phases 4+5 Canonical Frontier Discovery planning docs | Research artifact / needs human review | Left open per PR comment |

Merge commit: `a0f60c6` on main.

---

## 2. Repository Audit

### Architecture Status

| Component | Status |
|-----------|--------|
| Phase 1a — Canonical AureliusOptimizer interface | DONE |
| Phase 2 — ServingQueuePolicy extraction | DONE |
| Phase 3/3b/3c — AMCSG/SOTSS-MIN/OSOTSS routing | DONE |
| Phase 3d — GenAI canonical routing | DONE |
| Phase 3e — CA/SHU backtest serving routing | DONE |
| Phase 4 — Frontier rho adaptation | DONE (null result on fixtures, impl retained) |
| Phase 5 — Dead code deprecation | DONE (2,873 LOC removed) |
| Phase 1b — Replay loop unification | NOT STARTED |

### Test Suite Health

- 175 parity tests passing across all canonical paths
- test_canonical_optimizer_parity: 21 pass
- test_amcsg_sotss_canonical_routing_parity: 33 pass
- test_osotss_canonical_routing_parity: 38 pass
- test_phase3e_serving_canonical_routing_parity: 11 pass
- test_genai_canonical_routing_parity: 6 pass
- test_phase4_frontier_rho_parity: 20 pass (Phase 4 extended validation, just merged in #77)
- test_amcsg_lfc: 43 pass
- test_canonical_serving_policy_phase2: 9 pass

### Current Frontier

| Trace | Policy | Goodput/$ | vs Strongest Baseline | Status |
|-------|--------|-----------|----------------------|--------|
| Azure LLM 2024 | OSOTSS | 159,578 | +5.94% vs AMCSG | FRONTIER |
| Azure LLM 2024 | SOTSS-MIN | 160,107 | +6.29% vs AMCSG | Oracle (not deployable) |
| BurstGPT HF | OSOTSS | 178,109 | +5.85% vs AMCSG | FRONTIER (+n_sla_safe gap) |
| Canonical energy | constraint_aware | +11.1% | vs current_price_only | FRONTIER |

---

## 3. Bottleneck Analysis

### Q: What prevents another real +25% against the strongest fair baseline?

**Structural barriers identified:**

1. **BurstGPT 15-request n_sla_safe gap (confirmed irreducible):**
   - Root cause: stochastic/deterministic simulation mismatch
   - AMCSG achieves 5,864 SLA-safe requests; OSOTSS achieves 5,849
   - At p_interrupt=10%/hr, 60s ticks: p_survive ≈ 0.9982 → Binomial(c_spot, 0.9982) ≈ c_spot
   - Oracle loop uses deterministic FIFO, evaluation uses Binomial → 15-request structural gap
   - All 5 fix attempts (C1PGS, SOTSS-GSF, Adaptive EWMA, SSM, OSSC) confirmed gap is irreducible
   - NOT addressable without new oracle mechanism or simulation model change

2. **Azure OSOTSS-to-oracle gap (0.33%, 1-tick mismatch):**
   - OSOTSS misses 1 of SOTSS-MIN's 19 cheaper ticks on Azure (finds 18/19)
   - Root cause: EWMA prediction slightly over-estimates service time on that 1 tick → identifies it as non-reducible when it actually is
   - Closing this 0.33% requires either: oracle service time access (not deployable) or better quantile/distribution-aware service time estimator (new module)

3. **Forecasted_mcs deployability gap (large):**
   - AMCSG: 150,630 goodput/$ (Azure); 168,270 (BurstGPT)
   - forecasted_mcs ewma (alpha=0.5): 150,162 (-0.31% Azure); 103,192 (-38.7% BurstGPT)
   - Structural: lag1 (best causal approach) still -12.5% on BurstGPT because burst onset is not predictable from lag-1 arrival count
   - No parameter sweep (quantile, safety_k) can close a structural burst-onset prediction gap

4. **serving_queue × replica_scaling negative interaction (confirmed):**
   - Conformal SRPT + any variable-capacity provisioner = negative compound
   - Both over-provisioning (MCS) and under-provisioning (OSOTSS) create negative conformal interaction
   - Mechanism: preemption overhead × capacity-drop → starvation during scale events
   - Phase 1b would enable combination search, but the best combination is already known: FIFO+OSOTSS

---

## 4. Research Review

### Papers Reviewed (3)

#### 4.1 SageServe (arXiv:2502.14617, Feb 2025)
- **Problem**: Serving autoscaling with forecast-aware replica decisions
- **Approach**: Time-series forecast (Chronos/TimesFM) of request arrivals 1-5 ticks ahead
- **Decision**: Dynamic replica target pre-tick
- **Key assumption**: Forecasting model trained on historical arrival data
- **Aurelius applicability**: NOT APPLICABLE — requires training a new time-series forecasting model (new module, blocked by Five-Failure Rule)
- **Key insight**: Even if implemented, would need deployment-time training data that doesn't exist for the Azure/BurstGPT fixtures

#### 4.2 OServe (arXiv:2602.12151, 2025)
- **Problem**: Spatial-temporal workload orchestration for LLM serving
- **Approach**: Forecast-based multi-region load balancing with spatial placement
- **Decision**: Request routing across nodes + per-node replica scaling
- **Key assumption**: Multi-node deployment with router telemetry
- **Aurelius applicability**: NOT APPLICABLE — requires multi-node routing infrastructure (new module), also violates same-conditions rule (single-node benchmark)

#### 4.3 PecSched (arXiv:2409.15104, 2024)
- **Problem**: Preemptive efficient cluster scheduling for LLM inference
- **Approach**: SLA-safe admission with per-request output-length prediction
- **Decision**: Admit/reject based on predicted queue completion time
- **Key assumption**: Per-request output token prediction model
- **Aurelius applicability**: NOT APPLICABLE — requires output-length prediction model (new module, blocked), and the serving_queue×replica_scaling negative interaction makes admission-on-top-of-variable-capacity risky

### Conclusion

No reviewed paper is applicable to AureliusOptimizer improvements without adding new modules. The Five-Failure Rule correctly blocks all of them.

---

## 5. Phase 1b Implementation Plan

### What Phase 1b IS

Phase 1b = "Unify 4 replay loops into one engine"

The 4 loops are:
1. **Energy loop**: `canonical_backtests.py` → `AureliusOptimizer(policy="energy")` → `JobScheduler`
2. **Serving/SRTF loop**: `srtf_serving_backtest.py` → `AureliusOptimizer(policy="serving_queue")` + `AureliusOptimizer(policy="replica_scaling")`
3. **BurstGPT/Azure replica-scaling loop**: `backtest.py` → `AureliusOptimizer(policy="replica_scaling")`
4. **GenAI loop**: `genai_backtest.py` → `AureliusOptimizer(policy="genai_serving")`

### Why Phase 1b Matters

Phase 1b is an ENABLING STEP, not a KPI improvement by itself:
- It would enable combination search: energy × replica_scaling on the same trace
- It would enable a single evaluation harness for all policy types
- It would reduce architectural fragmentation (loops 2+3 share most logic)
- It would allow testing energy-aware replica scaling on serving traces

### Concrete Phase 1b Steps

**Step 1b-A (easiest): Unify loops 2+3 (serving+replica-scaling)**

`srtf_serving_backtest.py` and `backtest.py` both:
- Use BurstGPT/Azure LLM traces
- Use Erlang-C provisioning (MCS)
- Evaluate via `economics.py`
- Route through `AureliusOptimizer`

Unification: create a `ServingBacktestRunner` that accepts a `policy_name` and dispatches to the appropriate optimizer. Both loops share the same cost model and trace format.

**Blocker**: The physics differ: `backtest.py` is per-TICK aggregate, `srtf_serving_backtest.py` is per-REQUEST SRTF. A unified runner would need to handle both physics at an abstract interface level.

**Step 1b-B (medium): Unified economics evaluation layer**

All 4 loops compute goodput/$ differently:
- Loop 1 (energy): ROI based on energy cost savings vs baseline
- Loops 2+3 (serving): `compute_economic_kpi()` from `economics.py`
- Loop 4 (GenAI): custom per-model cost accounting

A unified `ReplayEvaluationResult` dataclass could standardize the output format without changing any evaluation logic. This is 50-100 LOC.

**Step 1b-C (hardest): Energy-overlay on serving traces**

This is the only Phase 1b variant that could enable a KPI improvement:
- Take Azure LLM 2024 timestamps (real 2024 dates)
- Overlay hourly ERCOT energy prices for those dates
- Modify the cost denominator to include energy cost
- Test whether energy-aware replica scaling (delay non-urgent scale-up to cheap-energy periods) improves goodput/$

Complexity: requires ingesting real energy price data for the Azure trace timestamps, normalizing to tick granularity, modifying the cost model. 

**Risk**: Changes the benchmark assumption (adds energy cost component), which requires careful governance. This would be a NEW BENCHMARK, not a change to existing benchmarks. Existing results are preserved.

### Recommended Sequence

1. This run → document the plan (done)
2. Next run → implement Step 1b-B (unified evaluation result type)
3. Next run → implement Step 1b-A (serving+replica-scaling unification)
4. Future run → Step 1b-C (energy overlay) if energy-aware gains are hypothesized

---

## 6. Same-Conditions Checklist (this run)

- Same trace: N/A (no benchmark run)
- Same SLA: N/A
- Same cost denominator: N/A
- Same GPU-hour accounting: N/A
- Canonical parity tests: 175 PASS ✓

---

## 7. Classification

**REPORT ONLY — Research Audit (Five-Failure Rule compliant)**

- No frontier improvement
- No new modules
- No new optimizer paths
- Architecture: Phase 1b planning documented
- Next highest-priority task: Phase 1b-B (unified evaluation result type)

---

## 8. Run Artifacts

- `research/ROADMAP.md`: updated with this run
- `research/GAP_ANALYSIS.md`: updated with this run
- `research/results/research_audit_phase1b_planning_2026-06-25.md`: this file
