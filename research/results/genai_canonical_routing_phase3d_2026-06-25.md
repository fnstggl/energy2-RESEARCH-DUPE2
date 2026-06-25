# GenAI Canonical Routing Phase 3d — Run 2026-06-25

## Classification

**Architecture Convergence / Infrastructure Improvement**

Five-Failure Rule compliant integration work: extracts the `constraint_aware` GenAI
replica-sizing decision (EWMA anticipatory + model-affinity cold-start routing) from the
`genai_backtest._run_policy` monolith into a canonical `GenAIServingPolicy` class,
routed through `AureliusOptimizer(policy="genai_serving")`.

Follows the Phase 2/3 extraction pattern exactly:
- Phase 2: `ServingQueuePolicy` extracted from srtf_serving_backtest monolith
- Phase 3b: `ReplicaScalingPolicy` routing through canonical optimizer for AMCSG+SOTSS-MIN
- Phase 3c: OSOTSS canonical routing (closed last primary replica-scaling gap)
- **Phase 3d (this run): `GenAIServingPolicy` — closes the GenAI benchmark gap**

## Changes

### `aurelius/optimizer/policies/genai_serving.py` (NEW)

Canonical physics owner for multi-model GenAI serving. Extracted verbatim from
`genai_backtest.py`:

- Constants: `GENAI_MIN_REPLICAS=1`, `GENAI_SLA_LATENCY_MULT=2.0`,
  `GENAI_SLA_LATENCY_ABS_S=30.0`, `GENAI_TARGET_RHO_SLA=0.65`,
  `GENAI_TARGET_RHO_UTIL=0.85`, `GENAI_EWMA_ALPHA=0.5`
- `genai_effective_service_s` — per-request mean service time with cold-start
- `genai_eval_tick_timeout` — Erlang-C p99 timeout rate for one tick
- `genai_size_for_sla` — minimum SLA-safe replica count (binary search, GENAI_MIN_REPLICAS..4096)
- `genai_size_for_target` — target-rho replica count for one tick
- `GenAIServingResult` dataclass: `replica_counts`, `affinity=True`, `mode="constraint_aware"`
- `GenAIServingPolicy(OptimizationPolicy)` with `name="genai_serving"`:
  - EWMA anticipatory sizing: causal alpha=0.5, initialised from first non-zero tick
  - `smoothed_rate = max(t.arrival_rate, ewma)` (anticipatory: never under-estimates)
  - SLA-safe sizing via `genai_size_for_sla(..., affinity=True)`
  - Falls back to `GENAI_MIN_REPLICAS` on zero-arrival ticks

### `aurelius/optimizer/policies/__init__.py`

- Added Phase 3d to docstring (GenAIServingPolicy row)
- Added `from .genai_serving import (...)` — placed **before** `from .replica_scaling import (...)`
  (alphabetical: g < r — fixes the ruff import-order lint failure from PR #72's branch)
- Added `GenAIServingPolicy` to `POLICY_REGISTRY`
- Updated `IMPLEMENTED_POLICIES`: now 4 of 6 (`energy`, `serving_queue`, `replica_scaling`, `genai_serving`)
- Updated `__all__` with all genai_ symbols

### `aurelius/traces/genai_backtest.py`

Physics helpers now delegate to canonical policy module (benchmark → policy, one direction):
- `_effective_service_s(tick, cold, affinity)` → `genai_effective_service_s(...)`
- `_size_for_sla(tick, cold, affinity)` → `genai_size_for_sla(...)`
- `_size_for_target(tick, cold, affinity, target_rho)` → `genai_size_for_target(...)`

`constraint_aware` path in `_run_policy` now routes through canonical optimizer:
```python
_ca_counts: list[int] = []
if policy == "constraint_aware":
    _ca_counts = _GENAI_OPTIMIZER.optimize(ticks, cold).replica_counts
# ... in tick loop:
r = _ca_counts[i]
```

Module-level optimizer: `_GENAI_OPTIMIZER = AureliusOptimizer(policy="genai_serving")`

### `data/external/alibaba_genai/processed/model_residency_audit_summary.json`

Updated `affinity_prewarm_share_pct` from `62.1` → `61.7` to fix
`test_summary_is_reproducible`: the audit script regenerates `61.7` from
`alibaba_genai_ablation_summary.json` (which has `affinity_share_pct=61.7`);
the committed JSON had a stale value from an earlier run. Now regenerated value
matches committed value.

### `tests/test_genai_canonical_routing_parity.py` (NEW)

6 parity tests using committed fixture at `tests/fixtures/alibaba_genai_sample/`:

| Test | What it checks |
|------|---------------|
| `test_replica_counts_bit_identical` | `GenAIServingPolicy.optimize()` replica counts are bit-identical to self-contained EWMA reference loop |
| `test_all_policies_run_via_backtest` | All 5 policies (`fifo`, `sla_aware`, `queue_aware`, `utilization_aware`, `constraint_aware`) complete without error |
| `test_constraint_aware_beats_sla_aware` | `constraint_aware` gp/$ ≥ `sla_aware`; timeout_rate=0% |
| `test_constraint_aware_zero_timeout` | SLA-sizing loop keeps timeout at 0% on fixture |
| `test_outcome_not_a_loss` | outcome ≠ "LOSS" vs sla_aware (fixture may TIE) |
| `test_optimizer_facade_policy_name` | `"genai_serving"` in both `POLICY_REGISTRY` and `IMPLEMENTED_POLICIES` |

All 6 pass. ✓

### `research/OPTIMIZER_UNIFICATION_PLAN.md`

Phase 3d row marked DONE in execution status table.

## Parity Results

**Bit-identical to pre-Phase-3d inline constraint_aware logic:**

| Metric | Canonical (Phase 3d) | Reference (inline EWMA) |
|--------|----------------------|-------------------------|
| Replica counts | [per fixture] | [per fixture] |
| Ticks differing | **0** | — |
| affinity | True | True |
| mode | constraint_aware | constraint_aware |
| timeout_rate_pct | 0.0% | 0.0% |
| outcome vs sla_aware | TIE | TIE |

**KPI change: 0.00%** — pure architecture convergence, no optimizer behavior drift.

## Same-Conditions Checklist

- [x] Same trace (Alibaba GenAI 2026 fixture: 60 requests, 1 tick)
- [x] Same SLA (`GENAI_SLA_LATENCY_ABS_S=30s` + `GENAI_SLA_LATENCY_MULT=2.0×`)
- [x] Same cost denominator (replica-hours × infra_dollar_per_replica_hour)
- [x] Same physics (Erlang-C: `erlang_c_wait_s`, `saturation_amplifier`, `tail_multipliers`)
- [x] Same arrival process (tick-aggregated, same `_aggregate_ticks` call)
- [x] Same cold-start priors (same `calibrate_cold_start` result)
- [x] Same EWMA parameters (alpha=0.5, initialised from first non-zero tick)
- [x] Same affinity routing (always True for constraint_aware)
- [x] Same decision-time information (causal EWMA: no future-arrival access)
- [x] Same evaluation method (`run_backtest` → `_run_policy` → `_eval_policy`)

## AureliusOptimizer Integration Status (Post This Run)

| Policy | Canonical Path | Status |
|--------|---------------|--------|
| `energy` | `EnergySchedulingPolicy` → `JobScheduler.solve()` | IMPLEMENTED (Phase 1) |
| `serving_queue` | `ServingQueuePolicy` → abs-conformal SRPT | IMPLEMENTED (Phase 2) |
| `replica_scaling` | `ReplicaScalingPolicy` → AMCSG/SOTSS-MIN/OSOTSS | IMPLEMENTED (Phase 3b/3c) |
| `genai_serving` | **`GenAIServingPolicy` → EWMA + affinity sizing** | **IMPLEMENTED (Phase 3d)** |
| `placement` | `PlacementPolicy` (stub — NotImplementedError) | NOT IMPLEMENTED |
| `admission` | `AdmissionPolicy` (stub — NotImplementedError) | NOT IMPLEMENTED |

4 of 6 declared policies now implemented. GenAI benchmark is fully integrated.

## PR #72 Context

PR #72 (on a branch derived from `b3d47c5`) implemented Phase 3d but had two CI failures:
1. `test_summary_is_reproducible`: stale `affinity_prewarm_share_pct=62.1` (correct=61.7)
2. Ruff lint: `from .genai_serving` placed after `from .replica_scaling` (wrong alphabetical order)

Both issues are fixed on this branch. PR #72 is superseded by this work on the correct
development branch `claude/tender-einstein-s3lqym`.

## Tests

**83 tests passing** (6 new + 77 pre-existing):
- `tests/test_genai_canonical_routing_parity.py` — **6 pass** (new)
- `tests/test_model_residency_genai_audit.py` — **10 pass** (fixed by 62.1→61.7)
- Pre-existing test suite — 67 pass, 8 skip (pre-existing numpy/pandas deps absent)
- Ruff lint: **0 errors**

## Frontier Status

**Unchanged.** OSOTSS remains the replica-scaling frontier:
- Azure LLM 2024: 159,578 gp/$ (+5.94% vs AMCSG)
- BurstGPT HF: 178,109 gp/$ (+5.85% vs AMCSG)

GenAI leaderboard (Alibaba GenAI 2026):
- `constraint_aware`: 9.84 gp/$ (+89.46% vs sla_aware) — unchanged (bit-identical)

## Next Steps

1. **Phase 1b replay loop unification** — collapse four independent replay loops into one
   engine (highest remaining architecture value; 0%-delta parity gate required)
2. **Third trace cross-validation** — OSOTSS on Alibaba GenAI 2026 full dataset (if raw
   data becomes available)
3. **Phase 4 (frontier promotion)** — promote BASE/DYNAMIC frontier controller to
   ρ-ceiling constraint in canonical optimizer (partial evidence: SUF +13% Azure only)
