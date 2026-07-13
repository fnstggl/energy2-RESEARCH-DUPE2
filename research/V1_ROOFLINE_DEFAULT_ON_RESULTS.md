# V1 Roofline Default-On Results

PR #119 promoted roofline-resolved GPU/model-aware timing into V1 but left it opt-in with `legacy_scalar` as
the default. **This PR makes `roofline` the canonical default** and keeps `legacy_scalar` only as an explicit
regression-reproduction mode.

## Why the default changed

V1's scalar `TPOT_S=0.020` is GPU-**blind** and behaves like an **L40S-class** decode constant
(`ROOFLINE_REUSE_DECISION.md`). On a fast fleet it overstates decode ~4× and fabricates SLA violations the
hardware does not have. The roofline path prices each GPU/model correctly, is BENCHMARK_DERIVED (public specs +
arch), deterministic, clone-safe, and fixture-validated — so per the production-default policy it becomes the
canonical behaviour. Keeping the less-realistic scalar as the default purely to preserve historical numbers is
explicitly disallowed.

## What changed (code)

- `prefill_decode.DEFAULT_TIMING_MODEL = "roofline"` (was `legacy_scalar`); `LEGACY_TIMING_MODEL` added.
- `env_timing_model()` default → `roofline`; `AURELIUS_TIMING_MODEL=legacy_scalar` restores the old path.
- `compute_phase_serving(...)` default `timing_model="roofline"`; `world_simulator.simulate_period` resolves
  roofline by default (fleet dominant GPU auto-resolved).
- Docstrings/comments updated to state roofline is the default.

## What old behaviour is still available

Bit-for-bit legacy reproduction via **either**:
- `timing_model="legacy_scalar"` (function/`kv_state` argument), or
- `AURELIUS_TIMING_MODEL=legacy_scalar` (environment).

Verified identical to pre-flip output: `test_explicit_legacy_is_bit_for_bit_scalar` (decode = `256·TPOT_S·0.92`
exactly).

## Before/after numbers (world_simulator phase-model path, `compare_v1_legacy_vs_v1_roofline.py`, SLA 8 s)

| GPU | metric | legacy_scalar (old default) | **roofline (new default)** |
|--|--|--|--|
| H100 | SLA violation rate | 0.700 | **0.092** |
| H100 | completion p95 (s) | 7.254 | **2.003** |
| A100 | SLA violation rate | 0.700 | **0.417** |
| A100 | completion p95 (s) | 7.254 | **3.202** |
| L40S | SLA violation rate | 0.700 | 0.725 |
| L40S | completion p95 (s) | 7.254 | 7.335 |

The legacy scalar is GPU-blind (identical across GPUs) and is L40S-class (on L40S roofline ≈ legacy). On
H100/A100 the new default removes phantom SLA violations the scalar invented. On L40S the new default is
(correctly) marginally *worse* — confirming it is a physics correction, not a one-way ratchet.

## Does this change public benchmark semantics?

**Yes, intentionally — for the world_simulator / persistent-state phase-model path** (active when
`kv_state["cost_mode"]` is set: the PR #99–#107 MPC-over-ClusterState path). Numbers there change as documented
above (a production-physics correction).

**No change to the canonical two-clock env (`CanonicalMultiPlaneEnvironment` / `fair_backtest`)**: that path
prices serving with a *separate* `ServingPlane._service_time_s` (token-based), which this PR does not touch.
Verified: `fair_backtest` is byte-identical under `AURELIUS_TIMING_MODEL=legacy_scalar` vs `roofline`
(candidate-vs-baseline −10.71% in both). Promoting roofline into the serving plane is the documented next step
(see `FULL_V1_PRODUCTION_PHYSICS_PROMOTION_RESULTS.md`).

## Risks of default-on roofline

- **Default GPU/model resolution**: when a caller doesn't pin a GPU, the world simulator uses the fleet's
  *dominant* replica GPU; the bare function defaults to H100/llama-8b. A heterogeneous fleet is therefore
  priced by its dominant type, not per-replica — a coarse (but far better than GPU-blind) approximation.
- **Unknown v2026 GPU types** (`XPU-*`) resolve to the roofline default GPU — labelled, conservative.
- **Constant MFU**: the roofline floor is optimistic on small batches (per-kernel MFU/tile-quant absent).
- **Downstream consumers** of absolute service times that were implicitly tuned to the L40S-class scalar may
  shift; the legacy mode is provided for any that need the old numbers.

## Remaining calibration gaps

Per-(GPU,model) measured MFU; per-replica GPU resolution (vs dominant-GPU heuristic); roofline timing in the
canonical serving plane; profiled-table calibration (Vidur-style) as a validation baseline.

## Test changes (all intentional, documented)

- `test_default_is_roofline_now`, `test_simulate_period_default_is_roofline_legacy_is_explicit`,
  `test_env_timing_model_opt_in_only` (env default now roofline) — assert the new default.
- `test_explicit_legacy_is_bit_for_bit_scalar` — legacy still exact.
- `test_prefill_decode_economics::{test_prefix_hit_reduces_prefill_only_decode_unchanged,
  test_prefill_heavy_benefits_more_from_reuse}` — pinned to explicit `legacy_scalar` (they are PR #107 scalar
  regression fixtures). No assertion was weakened; legacy is requested explicitly.
