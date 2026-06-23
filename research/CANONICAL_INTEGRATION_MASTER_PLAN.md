# Canonical Optimizer Integration Master Plan (Phase 5 — Planning Only)

> **Planning/architecture run. No optimizer logic added, no policy created/
> extracted, no benchmark/replay/eval/objective/dataset changed, nothing merged.**
> Replaces phase-by-phase guessing with a complete, dependency-ordered roadmap for
> what should enter `AureliusOptimizer`, in what order, with what evidence and
> parity gates. Directional simulator only (`docs/RESULTS.md §8` gate unmet).
>
> Companion docs: `NON_CANONICAL_SYSTEM_INVENTORY.md` (what exists),
> `OPTIMIZER_INTEGRATION_DEPENDENCY_GRAPH.md` (ordering), `CANONICAL_FRONTIER.md`
> + `POLICY_ABLATION_REPORT.md` + `POLICY_INTERACTION_ANALYSIS.md` (evidence),
> `NEXT_PHASE_IMPLEMENTATION_PROMPT.md` (the exact next prompt).

## Where we are (main `d35b94e`)
Implemented policies: **energy, serving_queue, replica_scaling** (MCS/SOTSS-MIN +
GSF/C1PGS spot-schedule computation). Stubs: **placement** (shadow scorer
**harmful** −7.3% lc), **admission** (shadow gate **neutral**). The current
frontier driver — **spot-fleet pricing (GSF: +492%/+727% vs SLA-oracle)** — is
**benchmark-local** (`srtf_serving_backtest.py`), not a canonical objective/policy.

## Governance (binding on every phase below)
- Report **Current Main vs Best Aurelius vs Candidate**; FIFO only as sanity.
- **Parity extractions** (moving existing logic): merge iff **0% KPI drift**.
- **Measurement phases** (combination search): no behavior change; report honestly.
- **Optimizer-first**: every integrated decision must map to a real production
  decision; **no actual-output-token leakage** at decision time.
- No new optimizer/objective/SLA/pricing/trace/dataset assumptions; no tuning.

---

## Phase roadmap (next 8 phases, dependency-ordered)

### Phase 5.1 — ObjectiveLayer: canonical spot/preemptible cost interface  ★ NEXT
- **Goal:** extract the spot-fleet **cost model** (`_spot_fleet_cost`,
  `_zfhc_spot_fleet_cost`, `_abs_floor_spot_fleet_cost`) into a canonical
  cost-objective interface the optimizer owns; benchmark delegates back.
- **Why next:** the cost denominator is the **largest measured lever** (Phase 4)
  and is currently benchmark-local; it is the prerequisite for treating any spot
  policy as canonical (dependency graph critical path). Pure parity extraction →
  safe.
- **Systems:** spot-fleet cost model. **Files:** new
  `aurelius/optimizer/objective/spot_cost.py` (or `optimizer/cost/`); thin
  delegates in `srtf_serving_backtest.py`; `optimizer/__init__.py` export.
- **Expected value:** High (enabling) · **Risk:** Med.
- **Parity tests:** new `tests/test_canonical_spot_cost_parity.py` — extracted
  cost == benchmark cost, bit-identical on fixtures; existing spot backtests
  byte-identical.
- **Benchmark gates:** GSF/ZFHC/AFMS spot backtests 0% KPI drift; energy + serving
  benchmarks 0% drift.
- **Merge rule:** merge iff 0% drift on all touched benchmarks + tests pass +
  main verified. **Block rule:** any drift → PR, no merge.
- **Rollback:** delegates revert to inlined cost fns (single revert commit).

### Phase 5.2 — Canonicalize the GSF spot policy into ReplicaScalingPolicy (spot mode)
- **Goal:** route the spot-fraction decision (GSF, the current record) through
  `ReplicaScalingPolicy` (it already owns `compute_sotss_gsf_schedule`); make the
  spot-fleet benchmark dispatch via `AureliusOptimizer(policy="replica_scaling")`.
- **Why next:** completes the cost lever (5.1 prices spot; 5.2 decides spot
  fraction). Depends on 5.1's cost interface.
- **Systems:** GSF/ZFHC/AFMS spot policies. **Files:** `replica_scaling.py`
  (spot-mode entry), `srtf_serving_backtest.py` (delegate spot-fleet sims).
- **Value:** High · **Risk:** Med · **Evidence:** Strong (GSF records).
- **Parity tests:** spot-fleet sims == policy path, bit-identical; no actual-token
  leakage. **Gates:** GSF Azure 149,235 / BurstGPT 167,767 reproduced exactly.
- **Merge rule:** 0% drift. **Block:** drift → no merge. **Rollback:** revert delegate.

### Phase 5.3 — Route BacktestEngine (energy walk-forward) through `energy` policy
- **Goal:** finish Phase-3 routing for the one remaining energy entry point.
- **Why next:** independent, low-risk; closes the energy world fully into AO.
- **Systems:** `BacktestEngine`. **Files:** `backtesting/engine.py`,
  `benchmarks/run_benchmark.py` (no behavior change).
- **Value:** Neutral (parity) · **Risk:** Low · **Evidence:** Strong (Phase-1 parity).
- **Parity tests:** energy golden snapshot reproduced; `BacktestEngine` output
  byte-identical. **Gates:** canonical energy 0% drift; full suite no new failures.
- **Merge rule:** 0% drift. **Block:** drift. **Rollback:** revert one commit.

### Phase 5.4 — Consolidate duplicate trace-replay provisioning
- **Goal:** route `traces/backtest.py:_min_cost_safe_replicas` (Azure/BurstGPT
  replay) through the canonical `ReplicaScalingPolicy` (remove the duplicate).
- **Why next:** removes a second provisioning implementation; depends on
  replica_scaling existing (done) and ideally 5.1/5.2 (shared cost/spot).
- **Systems:** trace-replay provisioning. **Files:** `traces/backtest.py`.
- **Value:** Med (de-dup) · **Risk:** Med · **Evidence:** Partial.
- **Parity tests:** Azure/BurstGPT replay KPIs byte-identical before/after.
  **Gates:** public-trace replay 0% drift. **Merge:** 0% drift. **Rollback:** revert.

### Phase 5.5 — Deprecate dead frontier families + formalize ShadowResearchLayer
- **Goal:** remove `frontier` EVAL_WORKLOAD + BATCH_INFERENCE (dead copy-paste);
  document GpuPlacementScorer (harmful) + admission (neutral) + CARA/output-length
  + residency as ShadowResearchLayer, **off by default**.
- **Why next:** independent; shrinks surface; no runtime path uses these.
- **Systems:** dead frontier, shadow modules. **Files:** delete
  `frontier/eval_workload_*`, `frontier/batch_inference_*` (+ their scripts/tests);
  doc-only for the shadow modules.
- **Value:** Low (maintenance) · **Risk:** Low · **Evidence:** Strong (no importers).
- **Parity tests:** repo-wide import check proves zero non-test consumers; full
  suite 0% delta. **Merge:** suite green. **Rollback:** revert deletion commit.

### Phase 5.6 — Unified ReplayLayer  (enabling, hard)
- **Goal:** collapse the 4 replay loops into one discrete-event engine that
  consumes a `Decision` stream from any policy; legacy loops become modes behind a
  flag until parity is proven.
- **Why next:** the hard prerequisite for honest multi-policy composition (Phase 4
  showed combinations are undefined without a shared replay).
- **Systems:** `simulation/cluster/engine`, `traces/backtest`, `srtf_serving_backtest`,
  `backtesting/engine`. **Files:** new unified engine module; legacy loops gain a
  `--engine=legacy` switch.
- **Value:** High (enabling) · **Risk:** **High** · **Evidence:** N/A (refactor).
- **Parity tests:** unified engine reproduces every committed benchmark KPI
  **bit-for-bit** (parity harness). **Gates:** **0% delta on all benchmarks**
  before any default switch. **Merge rule:** unified engine opt-in until 100%
  parity, then flip default in a separate commit. **Block:** any delta → stay
  opt-in. **Rollback:** `--engine=legacy` is the default until parity locked.

### Phase 5.7 — Honest policy-combination search
- **Goal:** on the unified replay, measure energy ⊕ serving ⊕ replica ⊕ cost(spot)
  combinations and interaction effects; identify the best validated AO combination.
- **Why next:** the actual frontier question; only feasible after 5.6.
- **Systems:** all implemented policies. **Files:** new
  `benchmarks/policy_combination_matrix.py` (read-only harness); docs.
- **Value:** High (frontier discovery) · **Risk:** Med · **Evidence:** generates it.
- **Gates:** report Current Main vs Best Aurelius vs each combination vs strongest
  baseline; a combination is "frontier" only if it beats the best validated AO
  config. **Merge rule:** harness + docs only (no runtime change) — may merge if
  it adds no behavior. **Rollback:** revert harness.

### Phase 5.8 — ConstraintLayer (frontier safe-ρ + SLA gate as hard constraints)
- **Goal:** promote `frontier` BASE/DYNAMIC safe-utilization to a ρ-ceiling
  constraint and the SLA gate to a hard constraint the optimizer respects.
- **Why next:** after composition exists, constrain it safely; lowest urgency.
- **Systems:** `frontier` BASE/DYNAMIC, `sla/`, `constraints/`. **Files:**
  `constraints/frontier_integration.py` (flag-gated), `optimizer/` constraint hook.
- **Value:** Med (safety) · **Risk:** Med · **Evidence:** Partial (SUF +13% analysis).
- **Gates:** no regression vs current; safety gates (timeout ≤ 0.5× FIFO) closed.
  **Merge:** opt-in, no regression. **Rollback:** keep `enabled=False` default.

---

## Final synthesis (answers the Phase-5 final-output prompt)

- **Top canonical integration candidates:** (1) spot/preemptible **cost objective**
  (5.1), (2) **GSF spot policy** into ReplicaScaling (5.2), (3) **unified replay**
  (5.6, enabling).
- **Systems to integrate:** spot cost model + GSF policy; BacktestEngine routing;
  trace-replay provisioning (consolidate); unified replay; then composition +
  ConstraintLayer.
- **Systems to keep shadow:** GpuPlacementScorer (harmful), WorkloadAdmissionGate
  (neutral), CARA/output-length/cache forecasters, residency, training frontier,
  C1PGS / SOTSS-GSF (validated null/negative — keep, off).
- **Systems to deprecate:** `frontier` EVAL_WORKLOAD + BATCH_INFERENCE (dead).
- **Systems not ready for integration:** PlacementPolicy (harmful until a
  non-regressing scorer exists), AdmissionPolicy (neutral), any multi-policy
  combination (blocked on unified replay).
- **Recommended phase sequence:** 5.1 → 5.2 → (5.3, 5.5 in parallel) → 5.4 → 5.6 →
  5.7 → 5.8.
- **Highest-value next phase:** **Phase 5.1 — ObjectiveLayer spot/preemptible cost
  interface** (parity extraction).
- **Why it comes next:** it canonicalizes the **largest measured lever** (cost
  denominator) as a parity extraction (safe, 0% drift), and it is the **critical-
  path prerequisite** for every spot/provisioning composition that follows.
