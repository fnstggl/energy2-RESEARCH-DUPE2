# Next Phase Implementation Prompt (recommended)

> Copy-paste this as the next Claude Code prompt. It is **Phase 5.1 — ObjectiveLayer
> spot/preemptible cost interface**: a parity extraction of the spot-fleet cost
> model into the canonical optimizer. Highest-value *unblocked* integration (the
> dominant measured lever, the cost denominator), executed as a behavior-preserving
> extraction like Phases 2/3.

---

PHASE 5.1 — CANONICAL SPOT/PREEMPTIBLE COST OBJECTIVE (PARITY EXTRACTION)

You are Claude Code operating inside the Aurelius repository.

This is an architecture-unification **parity extraction**, NOT an optimization
run, NOT a benchmark-improvement run, NOT a new-policy run.

GOAL
Extract the spot/preemptible **cost model** out of the serving benchmark monolith
into a canonical `AureliusOptimizer` ObjectiveLayer cost interface, so the cost
denominator (the largest measured goodput/$ lever) is owned by the optimizer
rather than the benchmark — with **0% KPI drift**.

STRICT PROHIBITIONS
- Do NOT create a new optimizer, a new policy, or a new benchmark.
- Do NOT change benchmark assumptions, datasets, replay logic, evaluation logic,
  SLA definitions, objective weights, traces, or pricing **values**.
- Do NOT tune anything for gains. Do NOT promote any FIFO-only claim.
- Do NOT introduce actual-output-token leakage at decision time.
- Do NOT merge unless every parity gate passes.

READ FIRST
- research/CANONICAL_INTEGRATION_MASTER_PLAN.md (Phase 5.1)
- research/NON_CANONICAL_SYSTEM_INVENTORY.md (section A)
- research/OPTIMIZER_INTEGRATION_DEPENDENCY_GRAPH.md
- aurelius/optimizer/aurelius_optimizer.py and aurelius/optimizer/policies/
- aurelius/optimizer/policies/replica_scaling.py (existing extraction pattern)
- aurelius/benchmarks/srtf_serving_backtest.py — the spot-fleet cost functions:
  `_spot_fleet_cost` (~:8038), `_zfhc_spot_fleet_cost` (~:8961),
  `_abs_floor_spot_fleet_cost` (~:8538), and `_zfhc_expected_interruptions`.

SCOPE (precise)
1. Create `aurelius/optimizer/objective/spot_cost.py` (new package
   `aurelius/optimizer/objective/` with `__init__.py`). Move the spot-fleet
   **cost** functions VERBATIM (same formulas, same constants: spot_price,
   spot_fraction, p_interrupt, on-demand $/hr). Keep them pure cost functions
   (no queue/KPI logic — evaluation stays in the benchmark).
2. In `srtf_serving_backtest.py`, replace the moved cost functions with thin
   imports/shims that delegate to the new module (keep names re-exported so
   existing call sites and tests resolve unchanged).
3. Expose the cost interface through `aurelius/optimizer/__init__.py`
   (e.g. `from .objective.spot_cost import ...`). Do NOT wire it into any policy's
   decision in this phase — interface extraction only.

FILES TO CREATE/MODIFY
- create: aurelius/optimizer/objective/__init__.py, aurelius/optimizer/objective/spot_cost.py
- create: tests/test_canonical_spot_cost_parity.py
- create: research/results/canonical_optimizer_phase5_1_spot_cost_parity_<date>.md
- modify: aurelius/benchmarks/srtf_serving_backtest.py (delegate the cost fns)
- modify: aurelius/optimizer/__init__.py (export); research/OPTIMIZER_UNIFICATION_PLAN.md (status)

PARITY GATES (all required)
- Unit: extracted cost functions return bit-identical values to the originals on
  fixtures (new test file).
- Spot benchmarks byte-identical (ignore timestamps): run, before vs after, for
  GSF, ZFHC, and abs-floor spot fleet (Azure + BurstGPT). GSF must reproduce
  Azure 149,235 / BurstGPT 167,767 goodput/$ exactly.
- Energy + serving benchmarks 0% drift (canonical golden snapshot reproduced;
  abs-conformal JSON byte-identical).
- ruff clean on new/changed files; full non-live suite shows no NEW failures
  (pre-existing lightgbm/fastapi env failures excepted).

BENCHMARK GATES
- 0% KPI drift on every touched benchmark. No benchmark definition changed
  (assert constants/pricing values unchanged in a test).

PR & MERGE RULES
- Develop on the designated feature branch; open a PR (ready for review).
- Merge ONLY if: cost interface is canonical + 0% drift on all touched benchmarks
  + parity tests pass + benchmark definitions unchanged + main verified after merge.
- If ANY drift exists: create the PR, DO NOT merge.

REQUIRED REPORTING (governance)
- In the parity report, state Current Main vs Best Aurelius vs Candidate are
  identical by construction (parity extraction, 0% drift); FIFO shown only as
  sanity. No optimization/frontier claim.

FINAL STATUS must be exactly one of:
- PHASE 5.1 SUCCESS — SPOT COST INTERFACE CANONICAL, PR MERGED, VERIFIED ON MAIN.
- PHASE 5.1 BLOCKED — PR CREATED, NOT MERGED.
