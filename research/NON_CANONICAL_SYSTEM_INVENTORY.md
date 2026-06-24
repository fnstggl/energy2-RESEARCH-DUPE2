# Non-Canonical System Inventory (Phase 5 — Planning Only)

> **Planning/architecture run. No code, benchmark, replay, eval, objective,
> dataset, or tuning changes. Nothing merged.** Catalogs every optimization-related
> system not yet fully represented inside `AureliusOptimizer`, ranked by
> benchmark evidence. Directional simulator only (`docs/RESULTS.md §8` gate unmet).
>
> **Current optimizer state (main `d35b94e`):** `IMPLEMENTED_POLICIES =
> {energy, serving_queue, replica_scaling}`. `placement` and `admission` are
> `NotImplementedError` stubs. (Note: `ReplicaScalingPolicy` was implemented after
> Phase 4 — MCS gate sweep + SOTSS-MIN oracle + GSF/C1PGS spot-schedule
> computation, parity-extracted from `srtf_serving_backtest.py`, 42 parity tests.)

## Steps 1–3 — Inventory × value/risk × architecture fit

Legend — Value: High/Med/Low/Neutral/Harmful/Unknown · Risk: Low/Med/High ·
Evidence: Strong/Partial/Weak/None · Parity: Easy/Moderate/Hard/NotYet ·
In AO?: yes/partial/no.

### A. Provisioning & cost (the dominant measured lever)

| System | Location | Decision / objective | In AO? | Value | Risk | Evidence | Parity | Architecture fit |
|---|---|---|---|---|---|---|---|---|
| **Spot-fleet cost model** (`_spot_fleet_cost`, `_*_spot_fleet_cost`) | `srtf_serving_backtest.py:8038+` | spot/preemptible pricing → **cost denominator** | **no** | **High** | Med | **Strong** (GSF: Azure 149,235/+492%, BurstGPT 167,767/+727% vs SLA-oracle) | Moderate | **ObjectiveLayer** (cost interface) |
| **GSF spot policy** (`_gsf_spot_replicas`, `compute_sotss_gsf_schedule`) | `srtf_serving_backtest.py:9419`, `replica_scaling.py:414` | spot fraction per tick | **partial** (schedule fn canonical; fleet sim/cost in benchmark) | **High** | Med | **Strong** (current record, +31%/+19% vs ZFHC) | Moderate | **ReplicaScalingPolicy** (spot mode) + ObjectiveLayer |
| ZFHC spot policy (`_zfhc_spot_replicas`) | `srtf_serving_backtest.py:8944+` | spot fraction (zero-floor high-c) | no | Med (superseded by GSF) | Low | Strong (+4.9% vs AFMS) | Moderate | ReplicaScalingPolicy (spot mode) |
| AFMS / abs-floor spot (`_abs_floor_spot_replicas`) | `srtf_serving_backtest.py:8517+` | spot w/ absolute on-demand floor | no | Med (superseded) | Low | Strong (+10.1%/+13.1%) | Moderate | ReplicaScalingPolicy (spot mode) |
| C1PGS spot replicas | `replica_scaling.py:710` | per-GPU spot scaling | partial (canonical fn) | **Neutral** | Low | Strong (**negative result**) | Easy | ReplicaScalingPolicy (kept, off) |
| **MCS / AMCSG gate sweep** | `replica_scaling.py:117` (`compute_mcs_c_schedule`) | per-tick min-cost-safe replicas | **yes** | High | — | Strong (AMCSG 150,630 gpd/$) | done | ReplicaScalingPolicy ✅ |
| **SOTSS-MIN oracle loop** | `replica_scaling.py:326` (`compute_sotss_min_schedule`) | cheapest safe c-schedule | **yes** | High | — | Strong (160,107 gpd/$, +6.3% vs AMCSG) | done | ReplicaScalingPolicy ✅ |
| OSOTSS (online SOTSS, causal EWMA) | `replica_scaling.py` (EWMA α=0.1) / benchmark | online causal provisioning | partial | High | Med | Partial (production-deployable variant) | Moderate | ReplicaScalingPolicy (online mode) |
| SOTSS-GSF stochastic oracle | `replica_scaling.py:414` | spot-aware oracle | partial | Neutral | — | Strong (**null result** 2/5) | Easy | ReplicaScalingPolicy (kept, off) |
| **SHU / min_cost_safe (trace-replay copy)** | `traces/backtest.py:419-467` (`_min_cost_safe_replicas`) | per-tick replicas for Azure/BurstGPT replay | **no (duplicate)** | Med | Med | Partial (SHU = "current headline" replay policy) | Moderate | **ReplicaScalingPolicy** (consolidate duplicate) |

### B. Serving queue & energy (already canonical)

| System | Location | In AO? | Value | Architecture fit |
|---|---|---|---|---|
| `JobScheduler` energy arbitrage | `optimization/scheduler.py` via `energy` policy | **yes** | High (+11.1% vs safe baseline) | EnergySchedulingPolicy ✅ |
| Abs-conformal SRPT discipline | `optimizer/policies/serving_queue.py` | **yes** | High@fixed-c / Neutral@provisioned | ServingQueuePolicy ✅ |
| `BacktestEngine` energy walk-forward | `backtesting/engine.py` (`run_benchmark.py`) | **no** (constructs `JobScheduler` directly) | Neutral (parity) | EnergySchedulingPolicy (route; deferred from Phase 3) |

### C. Serving replay policies (Azure/BurstGPT trace replay — un-routed)

| System | Location | In AO? | Value | Architecture fit |
|---|---|---|---|---|
| `ALL_POLICIES` = fifo / sla_aware / **constraint_aware** / queue_aware / cache_affinity | `traces/backtest.py:559`, `_run_policy:318` | **no** | Med (constraint_aware is the public-LLM leaderboard policy) | **ReplayLayer** + ConstraintLayer; some are baselines (keep) |
| `ConstraintAwareEngine` | `constraints/engine.py` | no | Med | ConstraintLayer |

### D. Forecasting / calibration (advisory; ForecastLayer)

| System | Location | In AO? | Value | Evidence | Fit |
|---|---|---|---|---|---|
| Price / carbon quantile forecasters | `forecasting/price_model.py`, `carbon_model.py` | no (advisory) | Med (feeds energy) | Partial | ForecastLayer |
| Conformal calibrators (rel/abs/per-class) | `serving_queue.py` (abs) + `srtf_serving_backtest.py` (rel/per-class) | partial | Med (abs canonical) | Strong | ForecastLayer (dedup rel/per-class) |
| CARA latency/queue/TTFT, cache-prefix, output-length | `forecasting/cara_*`, `cache_prefix_*`, `cara_output_length_*` | no (shadow) | Low/Neutral (output-length **HURT** −7…−11%) | Partial/Strong-neg | ShadowResearchLayer / ForecastLayer (gated) |

### E. Placement / admission / frontier / residency (keep shadow or deprecate)

| System | Location | In AO? | Value | Evidence | Fit |
|---|---|---|---|---|---|
| `GpuPlacementScorer` | `forecasting/gpu_placement_scorer.py` | stub (off) | **Harmful** (−7.3% lc gpd/$) | Strong-neg | PlacementPolicy → **keep ShadowResearchLayer, do not promote** |
| `WorkloadAdmissionGate` | `frontier/admission.py` | stub (off) | **Neutral** (±0.34%) | Strong | AdmissionPolicy → keep ShadowResearchLayer |
| `frontier/` BASE + DYNAMIC (safe-utilization) | `frontier/controller.py`, `dynamic_*` | no (default off) | Med (SUF +13% analysis) | Partial | ConstraintLayer (ρ-ceiling) |
| `frontier/` TRAINING | `frontier/training_*` | no | Unknown | Partial | ShadowResearchLayer |
| `frontier/` EVAL_WORKLOAD, BATCH_INFERENCE | `frontier/eval_workload_*`, `batch_inference_*` | no | Neutral (dead copy-paste) | None | **Deprecated / dead code** |
| `residency/` model placement/cold-start | `residency/` | no (MUTATION_ALLOWED=False) | Unknown (Alibaba +89% analysis) | Partial | ShadowResearchLayer |

### F. Replay & evaluation infrastructure

| System | Location | In AO? | Value | Fit |
|---|---|---|---|---|
| 4 replay loops (`simulation/replay`, `backtesting/engine`, `simulation/cluster/engine`, `srtf_serving_backtest`) | various | no | Enabling (no direct KPI) | **ReplayLayer** (unify — hard) |
| `economics.py` KPI math | `benchmarks/economics.py` | no | n/a (must stay frozen) | **EvaluationLayer** (keep) |

## Inventory verdicts
- **Integrate next (High value, Strong evidence, not-yet canonical):** the
  **spot-fleet cost model + best spot policy (GSF)** — this is the current
  frontier driver and the largest un-routed lever (Phase 4's "cost denominator").
- **Consolidate (duplicate):** `traces/backtest.py` `_min_cost_safe_replicas` →
  the canonical `replica_scaling` provisioning.
- **Route (low-risk parity):** `BacktestEngine` → `energy` policy.
- **Keep ShadowResearchLayer (off):** GpuPlacementScorer (harmful), admission
  gate (neutral), CARA/output-length forecasters, residency, training frontier.
- **Deprecate:** `frontier` EVAL_WORKLOAD + BATCH_INFERENCE (dead copy-paste).
- **Enabling prerequisite:** unified **ReplayLayer** (needed before honest
  energy+serving+replica composition).
