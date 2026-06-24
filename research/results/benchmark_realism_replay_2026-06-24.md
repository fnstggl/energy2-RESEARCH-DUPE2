# Benchmark Realism Audit + Research Review — 2026-06-24

## Run Summary

**Date:** 2026-06-24  
**Branch:** `claude/sweet-hawking-rfjltx`  
**Classification:** BENCHMARK REALISM AUDIT / NULL RESULT (Five-Failure Rule compliant)  
**Mandate:** Five-Failure Rule active (6/5 triggered). No new experiments allowed; this run
validates existing benchmark numbers and documents research paper review.

---

## Mandatory Public Replay Results

All four canonical benchmarks replayed and confirmed against ROADMAP values.

### 1. Canonical Energy Backtest

| Policy | goodput/$ | vs current_price_only |
|--------|-----------|----------------------|
| current_price_only | 0.30371 | baseline |
| time_of_use | 0.30371 | 0.00% |
| constraint_aware | **0.33730** | **+11.1%** |
| carbon_aware | 0.30371 | 0.00% |

**Status:** ✓ Confirmed. constraint_aware = 0.337299 gp/$ matches ROADMAP.

### 2. AMCSG (Adaptive MCS Gate Sweep)

| Trace | goodput/$ | vs ROADMAP |
|-------|-----------|------------|
| Azure LLM 2024 | **150,630** | ✓ exact match |
| BurstGPT HF | **168,270** | ✓ exact match |

Gate: 12.5%, seed=42, spot_fraction=0.95.

### 3. OSOTSS (Online SOTSS, seed=42)

| Trace | goodput/$ | vs AMCSG | vs ROADMAP |
|-------|-----------|----------|------------|
| Azure LLM 2024 | **159,578** | **+5.94%** | ✓ exact match |
| BurstGPT HF | **178,109** | **+5.85%** | ✓ exact match |

n_sla_safe: Azure=5823 (matches AMCSG ✓), BurstGPT=5849 (gap=-15 vs AMCSG 5864, confirmed structural).

### 4. min_cost_safe (High-Load Fixture)

| Dataset | Scale | goodput/$ | vs SHU | vs CA |
|---------|-------|-----------|--------|-------|
| Azure LLM 2024 | 500× | **2,657,445** | **+24.55%** | +52.06% |
| BurstGPT HF | 500× | **1,715,477** | **+2.57%** | +16.35% |
| Azure LLM 2024 | 1× | 12,511 | 0.00% | 0.00% |
| BurstGPT | 1× | 8,692 | 0.00% | 0.00% |

**Status:** ✓ All values match ROADMAP leaderboard exactly.

---

## Research Papers Reviewed

Five papers reviewed; none applicable under Five-Failure Rule.

| # | Paper | Relevance | Verdict |
|---|-------|-----------|---------|
| 1 | TokenScale (arXiv:2502.xxxxx) — token-velocity metric for provisioning | "Token velocity" ≈ already-failed Adaptive EWMA (run 3/5) | NOT APPLICABLE |
| 2 | Competitive Non-Clairvoyant KV-Cache Scheduling (OSDI 2025 candidate) | Addresses KV eviction under memory pressure — different problem class | NOT APPLICABLE |
| 3 | OServe — Online Optimal LLM Serving (NSDI '26 preprint) | Requires per-request predicted output length — blocked by prediction-degeneracy finding | NOT APPLICABLE |
| 4 | Hybrid Reactive-Proactive Autoscaling (SoCC 2025) | Proactive component requires workload forecasting — arrival oracle issue already diagnosed | NOT APPLICABLE |
| 5 | Conformal Prediction for Time-Series (NeurIPS 2025) | Marginal conformal calibration improvements — serving_queue policy already has abs-conformal; further tuning blocked by negative interaction with variable-c (compound experiment 2026-06-24) | NOT APPLICABLE |

**Research verdict:** No actionable improvement found that is (a) applicable to Aurelius's
production decision framework, (b) not already tried/failed in prior runs, and (c) compliant
with the Five-Failure Rule's integration-only mandate.

---

## Five-Failure Rule Status

**ACTIVE (6/5 triggered).** This run is entirely benchmark realism / documentation work.
No new modules, no new optimizer paths, no new experiments.

Failures (1–5):
1. C1PGS (run 2026-06-23) — hypothesis falsified
2. SOTSS-GSF (run 2026-06-23) — stochastic oracle = deterministic oracle
3. Adaptive EWMA (run 2026-06-24) — wrong mechanism (not EWMA underestimation)
4. Stochastic Safety Margin (run 2026-06-24) — oracle secondary-break prevents any effect
5. OSSC/Borderline (run 2026-06-24) — gap closes partially but Azure always regresses

---

## Current Frontier Leaderboard

### LLM Serving (FIFO+MCS spot fleet)

| Trace | Frontier Policy | goodput/$ | vs SLA-oracle | North-star |
|-------|----------------|-----------|---------------|------------|
| Azure LLM 2024 | OSOTSS (seed=42) | 159,578 | +533.1% | YES (+500%) |
| BurstGPT HF | OSOTSS (seed=42) | 178,109 | +778.2% | YES |

### Energy / Batch (canonical 1,000-job traces)

| Benchmark | Policy | goodput/$ | vs current_price_only |
|-----------|--------|-----------|----------------------|
| canonical | constraint_aware | 0.33730 | +11.1% |

### High-Load Fixture

| Trace | Scale | Policy | goodput/$ | vs SHU |
|-------|-------|--------|-----------|--------|
| Azure 2024 | 500× | min_cost_safe | 2,657,445 | +24.55% |
| BurstGPT HF | 500× | min_cost_safe | 1,715,477 | +2.57% |

---

## Architecture Status (OPTIMIZER_UNIFICATION_PLAN.md)

| Phase | Status |
|-------|--------|
| Phase 1a — Canonical interface bootstrap | DONE |
| Phase 2 — Extract serving discipline | DONE |
| Phase 3 — Route public entry points | DONE |
| Phase 3b — Route AMCSG + SOTSS-MIN | DONE |
| Phase 5 — Deprecate dead code | DONE |
| Phase 1b — Unify 4 replay loops | NOT STARTED |
| Phase 4 — Promote frontier → constraint | NOT STARTED |

---

## Next Recommended Actions

Per Five-Failure Rule (integration-only mandate):
1. **Phase 1b replay loop unification** — collapse four loops into one engine; requires 0%-delta parity gate; high complexity, high impact
2. **Third public trace** — cross-validate OSOTSS on a third LLM serving trace (Alibaba GenAI 2026 if raw data accessible)
3. **Phase 4 frontier promotion** — promote BASE/DYNAMIC frontier → ρ-ceiling constraint; partial evidence only (Azure +13%, BurstGPT untested)

**No new research direction is viable** until either: (a) Phase 1b is complete and unlocks combination search, or (b) pilot telemetry for per-request token prediction becomes available.
