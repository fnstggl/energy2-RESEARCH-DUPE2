# OSOTSS Canonical Routing Parity — Run 2026-06-24

## Classification

**Architecture Convergence / Infrastructure Improvement**

Five-Failure Rule compliant integration work: routes OSOTSS through the canonical
`AureliusOptimizer(policy="replica_scaling")` facade, closing the last production-
decision gap where OSOTSS called the policy function directly rather than through
the optimizer contract.

## Changes

### `aurelius/optimizer/policies/replica_scaling.py`

- `ReplicaScalingResult`: added `initial_violations: int = 0` — the FIFO-violation
  count before the oracle started iterating, captured from `online_sotss` mode and
  returned via the canonical result object.
- `ReplicaScalingConfig`: added `baseline_n_sla_safe: Optional[int] = None` — allows
  callers to pass the AMCSG stochastic GSF safety floor (computed from the stochastic
  GSF simulation) so the oracle targets the correct SLA-safe count rather than
  computing a more-conservative deterministic floor internally.
- `ReplicaScalingPolicy.optimize(mode="online_sotss")`: passes `baseline_n_sla_safe`
  and captures `initial_violations` (previously discarded as `_`).

### `aurelius/benchmarks/srtf_serving_backtest.py`

- `_run_online_sotss_backtest`: routes through `_REPLICA_SCALING_OPTIMIZER.optimize(
  config=ReplicaScalingConfig(mode="online_sotss", baseline_n_sla_safe=amcsg_n_sla_safe,
  ...))` instead of calling `_online_sotss_cost_schedule` directly. This makes
  `AureliusOptimizer(policy="replica_scaling")` the canonical owner of all four
  provisioning modes: `amcsg`, `sotss_min`, `online_sotss`, `forecasted_mcs`.

### `tests/test_osotss_canonical_routing_parity.py` (NEW)

38 new tests across 5 test classes:
1. `TestOSOTSSCanonicalScheduleParity` — c_schedule parity across 5 seeds ×2 (with/without baseline override), burst trace, smooth trace
2. `TestOSOTSSCanonicalDiagnosticParity` — oracle_iters, initial_violations, n_ticks_cheaper, baseline_n_sla_safe parity
3. `TestBaselineNSlaSafeConfig` — default/None, explicit-zero, high-baseline capacity ordering, result field contracts
4. `TestAureliusOptimizerOSOTSSFacade` — mode tag, valid c_schedule, empty-raw handling, cheaper-than-amcsg property
5. `TestBacktestLevelKPIParity` — backtest-level KPI fixture regression (OSOTSS > AMCSG, initial_viol ≥ 0, etc.)

## Parity Results

**Bit-identical to previously validated OSOTSS result:**

| Metric | Azure (canonical) | Azure (validated) | BurstGPT (canonical) | BurstGPT (validated) |
|--------|-------------------|-------------------|----------------------|----------------------|
| OSOTSS gp/$ | 159,578 | 159,578 | 178,109 | 178,109 |
| AMCSG gp/$ | 150,630 | 150,630 | 168,270 | 168,270 |
| vs AMCSG | +5.94% | +5.94% | +5.85% | +5.85% |
| n_sla_safe OSOTSS | 5823 | 5823 | 5849 | 5849 |
| n_sla_safe AMCSG | 5823 | 5823 | 5864 | 5864 |
| oracle_iters | 35 | 35 | 11 | 11 |
| initial_violations | 117 (new) | n/a | 36 (new) | n/a |
| n_ticks_cheaper | 18 | 18 | 40 | 40 |

**KPI change: 0.00%** — pure architecture convergence, no optimizer behavior drift.

## Same-Conditions Checklist

- [x] Same trace (Azure LLM 2024, BurstGPT HF)
- [x] Same SLA (10s Azure, 30s BurstGPT)
- [x] Same cost denominator (GSF spot-fleet GPU-hours × GPU_HOUR_USD)
- [x] Same GPU-hour accounting
- [x] Same physics (GSF Binomial interruption model, seed=42)
- [x] Same arrival process (actual tick-t arrivals, arrival-oracle class)
- [x] Same capacity model (AMCSG gate=12.5% ceiling, OSOTSS oracle loop)
- [x] Same pricing model ($0.80/hr spot, 10%/hr interruption, 95% spot)
- [x] Same telemetry class (arrival-oracle for capacity; causal EWMA for service-time prediction)
- [x] Same decision-time information
- [x] Same evaluation method (GSF stochastic spot-fleet simulation)

## AureliusOptimizer Integration Status (Post This Run)

| Mode | Canonical Path | Notes |
|------|---------------|-------|
| `amcsg` | via `_joint_mcs_c_schedule` → `compute_mcs_c_schedule` | thin delegate, correct |
| `sotss_min` | via `_sotss_min_cost_schedule` → `compute_sotss_min_schedule` | thin delegate, correct |
| `online_sotss` | **NOW via `_REPLICA_SCALING_OPTIMIZER.optimize()`** | Phase 3 COMPLETE |
| `forecasted_mcs` | via `_REPLICA_SCALING_OPTIMIZER.optimize()` | already canonical |

All four production-relevant modes now have a clear canonical path through
`AureliusOptimizer(policy="replica_scaling")`.

## Tests

**143 tests passing** across all replica-scaling test files after this change:
- `tests/test_osotss_canonical_routing_parity.py` — 38 pass (new)
- `tests/test_replica_scaling_policy_parity.py` — 42 pass (no regression)
- `tests/test_online_sotss_backtest.py` — 9 pass, 21 skip (data-size skips unchanged)
- `tests/test_replica_scaling_forecasted_mcs.py` — pass (no regression)
- `tests/test_adaptive_ewma_backtest.py` — pass (no regression)
- `tests/test_stochastic_safety_margin_backtest.py` — pass (no regression)
- `tests/test_borderline_osotss_backtest.py` — pass (no regression)

## Frontier Status

**Unchanged.** OSOTSS remains the frontier algorithm:
- Azure: 159,578 gp/$ (+5.94% vs AMCSG, +533.1% vs SLA-oracle)
- BurstGPT: 178,109 gp/$ (+5.85% vs AMCSG, +778.2% vs SLA-oracle)

## Next Steps

1. **AMCSG thin-delegate → canonical** (optional): `_joint_mcs_c_schedule` and
   `_sotss_min_cost_schedule` still call the policy function via thin delegates
   rather than the optimizer facade. Lower priority since OSOTSS is the frontier.
2. **Third public trace** — OSOTSS cross-validation on Alibaba GenAI 2026 or LMSYS
   (if timestamp data is available).
3. **Energy × replica_scaling compound** — the Phase 4 ablation found energy policy
   +11.1% standalone; test whether energy + OSOTSS compounds positively.
