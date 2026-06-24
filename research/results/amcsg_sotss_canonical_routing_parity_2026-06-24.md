# AMCSG + SOTSS-MIN Canonical Routing Parity — Phase 3b Architecture Integration

**Date:** 2026-06-24  
**Run classification:** ARCHITECTURE CONVERGENCE — Five-Failure Rule compliant integration  
**KPI change:** 0.00% (parity refactor by design)

---

## Summary

Phase 3b routes the remaining two primary replica-scaling backtest entry points
through the canonical `AureliusOptimizer(policy="replica_scaling")` facade:

1. **`_run_amcsg_backtest`**: Each gate in the gate sweep (`{9.5, 11.0, 12.5, 15.0, 17.5, 20.0}%`) now calls `_REPLICA_SCALING_OPTIMIZER.optimize(config=ReplicaScalingConfig(mode="amcsg", safe_gate_pct=gate, ...))` instead of `_joint_mcs_c_schedule()` directly.

2. **`_run_sotss_backtest`**: Both the AMCSG baseline computation and the SOTSS-MIN oracle loop route through `_REPLICA_SCALING_OPTIMIZER.optimize(mode="amcsg")` and `_REPLICA_SCALING_OPTIMIZER.optimize(mode="sotss_min")` respectively.

Additionally, `ReplicaScalingPolicy.optimize(mode="sotss_min")` previously discarded `initial_violations` from `compute_sotss_min_schedule()` (using `_`). This is now captured as `init_viols` and returned in `ReplicaScalingResult.initial_violations`.

---

## Architecture Change

**Before (Phase 3):**
```
_run_amcsg_backtest → _joint_mcs_c_schedule → compute_mcs_c_schedule
_run_sotss_backtest → _joint_mcs_c_schedule + _sotss_min_cost_schedule → compute_* directly
_run_online_sotss_backtest → _REPLICA_SCALING_OPTIMIZER.optimize(mode="online_sotss") ✓
```

**After (Phase 3b):**
```
_run_amcsg_backtest → _REPLICA_SCALING_OPTIMIZER.optimize(mode="amcsg") ✓
_run_sotss_backtest → _REPLICA_SCALING_OPTIMIZER.optimize(mode="amcsg") + .optimize(mode="sotss_min") ✓
_run_online_sotss_backtest → _REPLICA_SCALING_OPTIMIZER.optimize(mode="online_sotss") ✓
```

All primary replica-scaling backtest entry points now route through `AureliusOptimizer`.

---

## Parity Verification

| Metric | Before (direct call) | After (via optimizer) | Delta |
|---|---|---|---|
| AMCSG Azure best_goodput/$ | 150,630 | 150,629.9 | 0.00% |
| AMCSG BurstGPT best_goodput/$ | 168,270 | 168,270.0 | 0.00% |
| SOTSS-MIN Azure goodput/$ | 160,107 | 160,106.6 | 0.00% |
| SOTSS-MIN vs AMCSG | +6.29% | +6.29% | 0.00pp |
| SOTSS-MIN initial_violations | 117 (dropped) | 117 (propagated) | **FIXED** |
| SOTSS-MIN c_mean | 4.194 | 4.194 | 0.00 |
| AMCSG c_mean at gate=12.5% | 4.458 | 4.458 | 0.00 |

---

## Bug Fix: initial_violations now propagated in sotss_min mode

`ReplicaScalingPolicy.optimize(mode="sotss_min")` previously used `_` to discard the third return value from `compute_sotss_min_schedule()`:

```python
# Before:
c_sched, n_iters, _, n_ticks_cheaper, baseline_n_sla_safe = compute_sotss_min_schedule(...)
# ReplicaScalingResult(...) — initial_violations defaulted to 0
```

```python
# After:
c_sched, n_iters, init_viols, n_ticks_cheaper, baseline_n_sla_safe = compute_sotss_min_schedule(...)
# ReplicaScalingResult(..., initial_violations=init_viols)
```

For the full Azure 5,880-request trace: `initial_violations=117` (117 requests violated SLA before any oracle iterations). This value was already used in `SOTSSReport.sotss_initial_violations` via the direct `_sotss_min_cost_schedule()` call; it is now consistently propagated through the canonical optimizer path.

---

## Same-Conditions Checklist

| Condition | Status |
|---|---|
| Same trace | ✓ (Azure LLM 2024, BurstGPT HF) |
| Same SLA definition | ✓ (10s / 30s) |
| Same cost denominator | ✓ (GSF spot-fleet, $0.80/hr, 95% spot) |
| Same GPU-hour accounting | ✓ |
| Same physics model | ✓ (Binomial interruption, Erlang-C) |
| Same arrival process | ✓ |
| Same capacity model | ✓ |
| Same pricing model | ✓ |
| Same decision-time information | ✓ (parity refactor — no logic change) |
| Same evaluation method | ✓ |

---

## Tests

- New: `tests/test_amcsg_sotss_canonical_routing_parity.py` — **33 tests, all passing**
  - 8 AMCSG canonical routing parity tests
  - 10 SOTSS-MIN canonical routing parity tests (including `initial_violations` propagation)
  - 8 end-to-end KPI parity tests
  - 7 other structural/field validation tests
- Existing: 179 previously passing tests — **0 regressions**
- Total: **212 passing**

---

## Classification

**ARCHITECTURE CONVERGENCE — Five-Failure Rule integration**

Not a frontier improvement (0% KPI change by design). Closes the last gap in
canonical AureliusOptimizer routing for primary replica-scaling entry points.

---

## Merge Recommendation

**MERGE** — safe infrastructure improvement:
- No runtime behavior change (0.00% KPI drift)
- No benchmark weakening
- Fixes a silent bug (`initial_violations` propagation)
- 33 new parity tests pass
- 0 regressions in 212 total tests
