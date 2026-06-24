# ReplicaScalingPolicy — Phase 2/3 Architecture Convergence Run 2026-06-23

**Run type**: Architecture Convergence (Phase 2/3 extraction)
**Status**: ARCHITECTURE IMPROVEMENT — ReplicaScalingPolicy implemented, all provisioning decisions now govern through AureliusOptimizer
**KPI impact**: 0% (parity-preserving extraction, identical algorithm/constants/tie-breaks)

## Summary

Implements `ReplicaScalingPolicy` in `aurelius/optimizer/policies/replica_scaling.py`,
following the same extraction pattern as Phase 2 (`serving_queue.py`). The per-tick
replica-count provisioning logic (AMCSG MCS gate sweep and SOTSS-MIN oracle loop) is
extracted verbatim from `srtf_serving_backtest.py` into the canonical AureliusOptimizer
policy seam. The benchmark functions become thin delegates; the policy module is the
canonical owner.

**Parity verified**: 42 tests pass (0.41s), all asserting bit-identical results between
policy functions and benchmark originals across 6 test classes.

## Architecture State After This Run

| Policy             | Status          | Phase |
|-------------------|-----------------|-------|
| EnergySchedulingPolicy | Implemented | Phase 1 |
| ServingQueuePolicy | Implemented    | Phase 2 |
| **ReplicaScalingPolicy** | **Implemented** | **Phase 2/3** |
| PlacementPolicy    | Stub (raises)   | Phase 3 |
| AdmissionPolicy    | Stub (raises)   | Phase 3 |

`IMPLEMENTED_POLICIES = frozenset({"energy", "serving_queue", "replica_scaling"})`

## What Changed

### New file: `aurelius/optimizer/policies/replica_scaling.py`

Canonical owner of all per-tick provisioning logic:

| Symbol | Extracted from |
|--------|---------------|
| `REPLICA_TTFT_BASE_S = 0.150` | `TTFT_BASE_S` |
| `REPLICA_TPOT_S = 0.020` | `TPOT_S` |
| `REPLICA_SAFE_GATE = 12.5` | `_SOTSS_SAFE_GATE` |
| `REPLICA_AGGRESSIVE_GATE = 100.0` | `_SOTSS_MIN_GATE` |
| `REPLICA_MAX_ORACLE_ITERS = 500` | `_SOTSS_MIN_MAX_ITERS` |
| `_replica_service_time_s(tok)` | `_service_time_s` |
| `_replica_calibrate_warp(raw, servers, rho)` | `calibrate_time_warp` |
| `_replica_erlang_c_sla_timeout_pct(lam, mu, c, thresh)` | `_erlang_c_sla_timeout_pct` |
| `compute_mcs_c_schedule(raw, tick_s, warp, gate, sla_s)` | `_joint_mcs_c_schedule` |
| `_oracle_fifo_response_times(pairs, c_sched, tick_s)` | `_simulate_fifo_variable_c` (oracle-use only) |
| `compute_sotss_min_schedule(raw, tick_s, warp, sla_s, ...)` | `_sotss_min_cost_schedule` |
| `ReplicaScalingConfig` | new dataclass |
| `ReplicaScalingResult` | new dataclass |
| `ReplicaScalingPolicy` | new policy class |

### Updated: `aurelius/optimizer/policies/__init__.py`

- Added imports from `replica_scaling`
- `ReplicaScalingPolicy` is now the real implementation (not a stub)
- `IMPLEMENTED_POLICIES` extended to include `"replica_scaling"`

### Updated: `aurelius/benchmarks/srtf_serving_backtest.py`

- Imports `compute_mcs_c_schedule` and `compute_sotss_min_schedule` from policy
- `_joint_mcs_c_schedule` → thin delegate to `compute_mcs_c_schedule`
- `_sotss_min_cost_schedule` → thin delegate to `compute_sotss_min_schedule`
- `_REPLICA_SCALING_OPTIMIZER = AureliusOptimizer(policy="replica_scaling")` wired up

### New: `tests/test_replica_scaling_policy_parity.py`

42 parity tests across 6 classes:

1. **TestErlangCParity** (8 tests): Erlang-C formula, service time, constants
2. **TestMCSScheduleParity** (6 tests): `compute_mcs_c_schedule` vs `_joint_mcs_c_schedule`
3. **TestOracleFIFOParity** (5 tests): `_oracle_fifo_response_times` vs `_simulate_fifo_variable_c`
4. **TestSOTSSScheduleParity** (7 tests): `compute_sotss_min_schedule` vs `_sotss_min_cost_schedule`
5. **TestReplicaScalingPolicy** (10 tests): policy contract, AureliusOptimizer integration
6. **TestEdgeCases** (6 tests): empty input, single request, single tick

## No KPI Impact by Design

This is a parity-preserving extraction:
- Algorithm: identical (same decision logic)
- Constants: identical (same numeric values)
- Tie-breaks: identical (same event ordering)
- 42 bit-identical parity tests confirm 0% KPI drift

The next research run that calls `run_sotss_min_azure_backtest()` or `run_amcsg_azure_backtest()`
will still produce 160,107 / 150,630 goodput/$ respectively — decisions now flow through
`AureliusOptimizer(policy="replica_scaling")`.

## Classification

**ARCHITECTURE IMPROVEMENT** — not a frontier KPI improvement.
- Zero SLA regressions (parity-preserving)
- Architecture convergence goal advanced: 3 of 5 decision-layer policies implemented
- Next: BurstGPT dynamic spot fraction (highest-EV research target per GAP_ANALYSIS)
