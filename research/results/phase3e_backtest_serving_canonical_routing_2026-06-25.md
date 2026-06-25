# Phase 3e: Backtest Serving Canonical Routing

> **Architecture Convergence — NOT a frontier improvement.**
> **Five-Failure Rule compliant.** KPI drift: **0.00%**.
> **Directional simulator evidence only — NOT production savings** (`docs/RESULTS.md` §8).

- Generated: 2026-06-25
- Classification: ARCHITECTURE_CONVERGENCE
- Phase: 3e (last unrouted production policies → canonical optimizer)

## What Changed

Routes `constraint_aware` and `safe_high_utilization` policies from
`aurelius/traces/backtest.py` through `AureliusOptimizer(policy="replica_scaling")`
via a new `ReplicaScalingPolicy.optimize_from_ticks()` method.

**Physics extracted to `aurelius/optimizer/policies/replica_scaling.py`:**
- `compute_constraint_aware_schedule(ticks, tick_hours, *, ca_target_rho=0.65)` — EWMA anticipatory sizing + hysteresis trim at ρ=0.65
- `compute_shu_schedule(ticks, tick_hours)` — EWMA anticipatory sizing at ρ=0.75, no hysteresis
- `_bt_timeout_rate_pct(...)` — timeout rate computation (lazy serving import)
- `_bt_constraint_trim(...)` — cache-aware trim loop using `_bt_timeout_rate_pct`
- `_bt_size_for_target(...)` — pure-math target-rho replica sizing

**`backtest.py` refactoring:**
- `_SERVING_OPTIMIZER = AureliusOptimizer(policy="replica_scaling")` at module level
- `_run_policy()` pre-computes CA/SHU schedule via `_SERVING_OPTIMIZER.policy.optimize_from_ticks()` before the evaluation loop
- Evaluation loop (`evaluate_tick`) runs unchanged — only replica count decision is routed

**Architectural completeness achieved:**
After Phase 3e, ALL primary production policies are routed through `AureliusOptimizer`:
- `energy` → `EnergySchedulingPolicy` (Phase 1a)
- `serving_queue` → `ServingQueuePolicy` (Phase 2)
- `replica_scaling` (amcsg/sotss_min/online_sotss) → `ReplicaScalingPolicy` (Phase 3b/3c)
- `genai_serving` → `GenAIServingPolicy` (Phase 3d)
- `constraint_aware` / `safe_high_utilization` → `ReplicaScalingPolicy.optimize_from_ticks()` (**Phase 3e**)

## Parity Verification

| Layer | Result |
|---|---|
| Physics: `_bt_timeout_rate_pct` == `evaluate_tick().timeout_rate_pct` | **PASS** — bit-identical (140 tick-replica pairs) |
| Schedule: `compute_constraint_aware_schedule` == original `_run_policy` loop | **PASS** — bit-identical (Azure 50x 32 ticks, BurstGPT 51 ticks) |
| Schedule: `compute_shu_schedule` == original `_run_policy` loop | **PASS** — bit-identical (both datasets) |
| Optimizer interface: `optimize_from_ticks` routes correctly | **PASS** |
| MCS anchor: Azure 500x = 2,657,445 unchanged | **PASS** |

## Results — KPI Ordering Confirmation

| dataset | scale | CA gpd/$ | SHU gpd/$ | SHU vs CA % |
|---|---:|---:|---:|---:|
| azure_llm_2024_sample | 1× | 12,511 | 12,511 | 0.0% |
| azure_llm_2024_sample | 50× | 604,601 | 604,601 | 0.0% |
| azure_llm_2024_sample | 200× | 1,601,103 | 1,738,187 | +8.6% |
| azure_llm_2024_sample | 500× | 1,747,578 | 2,133,670 | +22.1% |
| burstgpt_sample | 1× | 8,692 | 8,692 | 0.0% |
| burstgpt_sample | 50× | 214,755 | 214,755 | 0.0% |

At low load (1×/50×) constraint_trim makes both policies converge (SLA slack absorbed).
At high load (200×/500×) SHU (ρ=0.75) outperforms CA (ρ=0.65) as expected.

## Architecture State After Phase 3e

```
AureliusOptimizer
└── ReplicaScalingPolicy
    ├── optimize(raw, warp, config)         — amcsg / sotss_min / online_sotss / forecasted_mcs
    └── optimize_from_ticks(ticks, ...)     — constraint_aware / safe_high_utilization  ← NEW
```

`backtest.py` `_run_policy()` now calls the canonical optimizer for CA/SHU — zero inline
sizing logic remains for production policies. Only `min_cost_safe` (oracle, not production)
and `sla_aware`/`queue_aware`/`fifo` (simple baselines) remain inline.

## Tests

`tests/test_phase3e_serving_canonical_routing_parity.py` — 9 tests:
- 2 physics-layer parity tests (Azure + BurstGPT)
- 4 schedule-layer parity tests (CA + SHU × Azure + BurstGPT)
- 2 optimizer-interface tests
- 1 MCS anchor canary
- 1 KPI ordering check (Azure 200x)
