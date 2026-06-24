# GenAI Canonical Routing Parity — Phase 3d — Run 2026-06-24

## Classification

**Architecture Convergence / Infrastructure Improvement**

Five-Failure Rule compliant integration work (6/5 active): routes the GenAI
`constraint_aware` multi-model serving policy through the canonical
`AureliusOptimizer(policy="genai_serving")` facade, and moves the physics
ownership from inline functions in `genai_backtest.py` to a dedicated
`GenAIServingPolicy` module — the canonical physics owner.

0% KPI drift. No new optimizer logic. No new priors. No benchmark definitions
changed.

## Changes

### `aurelius/optimizer/policies/genai_serving.py` (NEW)

Canonical physics owner for GenAI multi-model serving. Extracted verbatim
from `genai_backtest.py` inline functions.

- Constants (canonical owner): `GENAI_MIN_REPLICAS=1`, `GENAI_SLA_LATENCY_MULT=2.0`,
  `GENAI_SLA_LATENCY_ABS_S=30.0`, `GENAI_TARGET_RHO_SLA=0.65`,
  `GENAI_TARGET_RHO_UTIL=0.85`, `GENAI_EWMA_ALPHA=0.5`.
- `genai_effective_service_s(mean_exec_s, n, distinct_models, lora_frac, controlnet_frac, cold, affinity)` —
  per-request mean service time with model cold-start; affinity routes to warm replicas
  (amortises base-model reload over distinct_models arrivals); non-affinity treats every
  request as a potential cold-start.
- `genai_eval_tick_timeout(...)` — Erlang-C p99 timeout rate [0, 50] for one tick.
- `genai_size_for_sla(...)` — minimum SLA-safe replica count (binary search 1..4096).
- `genai_size_for_target(...)` — target-rho replica count.
- `GenAIServingResult` dataclass: `replica_counts: list`, `affinity: bool = True`,
  `mode: str = "constraint_aware"`.
- `GenAIServingPolicy(OptimizationPolicy)`: `name = "genai_serving"`,
  `optimize(ticks, cold, tick_hours=1.0) → GenAIServingResult`. Implements causal
  EWMA (alpha=0.5) anticipatory sizing + model-affinity cold-start routing — verbatim
  extraction of the `constraint_aware` branch in `_run_policy`.

### `aurelius/optimizer/policies/__init__.py`

- Added import of `GenAIServingPolicy`, `GenAIServingResult`, and all six
  physics functions/constants from `.genai_serving`.
- Added `GenAIServingPolicy.name: "genai_serving"` to `POLICY_REGISTRY`.
- Added `"genai_serving"` to `IMPLEMENTED_POLICIES`.
- Updated module docstring to document Phase 3d.
- Updated `__all__` with all new exports.

### `aurelius/traces/genai_backtest.py`

- Added import: `AureliusOptimizer` and three physics functions from
  `aurelius.optimizer.policies.genai_serving` (`genai_effective_service_s`,
  `genai_size_for_sla`, `genai_size_for_target`).
- Module-level singleton: `_GENAI_OPTIMIZER = AureliusOptimizer(policy="genai_serving")`.
- `_effective_service_s(tick, cold, affinity)` → thin wrapper delegating to
  `genai_effective_service_s(tick.mean_exec_s, tick.n, ...)` (canonical physics owner).
- `_size_for_sla(tick, cold, affinity)` → thin wrapper delegating to
  `genai_size_for_sla(tick.n, tick.arrival_rate, ...)` (canonical physics owner).
- `_size_for_target(tick, cold, affinity, target_rho)` → thin wrapper delegating to
  `genai_size_for_target(tick.n, tick.arrival_rate, ...)` (canonical physics owner).
- `_run_policy(...)`: pre-computes `_ca_counts = _GENAI_OPTIMIZER.optimize(ticks, cold).replica_counts`
  when `policy == "constraint_aware"`, then uses `_ca_counts[i]` in the per-tick loop
  (replaces the inline EWMA + `_size_for_sla` call). Removes the now-unused `ewma`
  tracking variable. The evaluation path (`_eval_tick`) is unchanged.

### `tests/test_genai_canonical_routing_parity.py` (NEW)

6 new parity tests:

1. `test_replica_counts_bit_identical` — `GenAIServingPolicy.optimize(ticks, cold).replica_counts`
   must be bit-identical to a self-contained reference implementation of the same EWMA
   anticipatory-sizing + model-affinity loop (the pre-Phase-3d inline logic). **All ticks match.**
2. `test_all_policies_run_via_backtest` — `run_backtest` completes for all 5 policies
   (fifo, sla_aware, queue_aware, utilization_aware, constraint_aware) without error.
3. `test_constraint_aware_beats_sla_aware` — constraint_aware gp/$ ≥ sla_aware on fixture.
4. `test_constraint_aware_zero_timeout` — SLA-sizing loop keeps constraint_aware timeout=0%.
5. `test_outcome_not_a_loss` — backtest outcome ≠ LOSS vs sla_aware (fixture may TIE due to
   small sample; full trace gives ALPHA_WIN +38.2%).
6. `test_optimizer_facade_policy_name` — `"genai_serving"` in `POLICY_REGISTRY` and
   `IMPLEMENTED_POLICIES`.

## Parity Results

**0% KPI drift — bit-identical replica counts on the committed fixture.**

| Test | Result |
|------|--------|
| Replica counts (all ticks, fixture) | Bit-identical ✓ |
| constraint_aware timeout_rate_pct | 0.000% ✓ |
| constraint_aware gp/$ ≥ sla_aware gp/$ | ✓ |
| All 5 policies run without error | ✓ |
| Pre-existing ablation tests (7/7) | Pass ✓ |
| New parity tests (6/6) | Pass ✓ |

Known-good KPIs on the Alibaba GenAI 2026 fixture (60 requests):

| Policy | gp/$ | timeout% |
|--------|------|----------|
| constraint_aware | 3.3333 | 0.000% |
| sla_aware | 2.4583 | 1.206% |
| queue_aware | 3.2222 | 2.373% |

Full-trace result (Alibaba GenAI 2026, PR #71, unchanged by this PR):
- constraint_aware vs sla_aware: **+38.2%** gp/$
- constraint_aware vs constraint_aware_no_affinity: **+38.2%** gp/$

## Same-Conditions Checklist

- [x] Same trace (Alibaba GenAI 2026, `tests/fixtures/alibaba_genai_sample/`)
- [x] Same SLA (ABS=30s, MULT=2.0)
- [x] Same cost denominator ($3.0/GPU-hr, same across all policies)
- [x] Same GPU-hour accounting (replica_hours = sum(r * tick_hours))
- [x] Same physics (Erlang-C M/M/c, saturation_amplifier, tail_multipliers)
- [x] Same cold-start calibration (medians from pipeline layer)
- [x] Same EWMA alpha (0.5, causal, initialised from first non-zero tick)
- [x] Same affinity decision (always True for constraint_aware, False for others)
- [x] Same evaluation method (genai_backtest.run_backtest, unchanged)
- [x] Circular-import-free (benchmark → policy, one direction only)

## AureliusOptimizer Integration Status (Post This Run)

| Workload | Policy | Canonical Path | Status |
|----------|--------|---------------|--------|
| Energy/batch | `energy` | `JobScheduler.solve` delegate | Phase 1a ✓ |
| LLM serving (abs-conformal SRPT) | `serving_queue` | extracted `ServingQueuePolicy` | Phase 2 ✓ |
| Replica scaling (AMCSG/SOTSS-MIN/OSOTSS) | `replica_scaling` | extracted `ReplicaScalingPolicy` | Phase 3b/3c ✓ |
| GenAI multi-model serving (constraint_aware) | `genai_serving` | **extracted `GenAIServingPolicy`** | **Phase 3d ✓** |
| Placement/routing | `placement` | stub (NotImplementedError) | Future |
| Admission control | `admission` | stub (NotImplementedError) | Future |

## Five-Failure Rule Compliance

ACTIVE (6/5). This phase is explicitly allowed:
- "Integration of existing modules" — routes existing `constraint_aware` logic (unchanged behavior)
  through the canonical facade. No new optimizer paths, no new models, no new priors.
- `genai_effective_service_s`, `genai_size_for_sla`, `genai_size_for_target` are **moved**
  from inline definitions in the benchmark to a dedicated canonical owner module — not new algorithms.
- Zero benchmark definitions changed.
- Zero evaluation infrastructure changed.
