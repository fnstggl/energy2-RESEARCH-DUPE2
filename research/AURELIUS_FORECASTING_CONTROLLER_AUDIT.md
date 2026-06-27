# Phase 0 — Forecasting & Action-Surface Audit

Pre-build audit of every prediction and action surface in the merged canonical
environment (post #94/#95), so the forecasting layer and the model-predictive
controller are built on **real** surfaces only — no fake predictors, no fake actions.
File-level evidence cited.

## A. Existing predictors

| predictor | classification | where | notes |
|---|---|---|---|
| output-token predictor | **RUNNING_STATISTIC** | `serving_plane._causal_predicted_tokens` | causal running-median prior; no model |
| capacity `forecasted_mcs` | **HEURISTIC** | `optimizer/layers.py ForecastLayer.causal_capacity_forecast` (EWMA, mcs_gate) | EWMA on past arrivals → replica count; the one decision-feeding forecast |
| capacity `reactive_lag1` | **RUNNING_STATISTIC** | `optimizer/unified_replay.py` | lag-1 backlog reaction |
| capacity `backlog_aware` | **RUNNING_STATISTIC** | `optimizer/unified_replay.py` | reacts to live backlog (not a forecast) |
| price/carbon quantile | **HEURISTIC** (ADVISORY) | `ForecastLayer.ADVISORY` | <0.3% of KPI per the layer's own taxonomy |
| KV reuse rate | **RUNNING_STATISTIC** | `environment/kv_cache.py KVModel.fit` | exact-prefix hit-rate from a Mooncake cache replay; stable train≈holdout |
| arrival-rate predictor | **ABSENT** | — | env reads actual arrivals; no forecaster |
| queue / SLA-risk predictor | **ABSENT** | — | `backlog_aware` reacts; nothing predicts SLA risk |
| GPU util / mem / network predictor | **ABSENT** (anchored constant) | `fleet_plane_v2026` | v2026 has **no per-hour series**; marginals are constant full-trace values |
| job-runtime predictor | **ABSENT** | — | job_execution duration not exposed per hour |
| `cara_latency` / `output_length` / `cache_prefix` ML forecasters | **PLACEHOLDER** (RESEARCH_ONLY) | `ForecastLayer.RESEARCH_ONLY` | the layer marks these neutral/negative, not decision-feeding |
| any REAL_ML forecaster | **ABSENT** | — | no trained model consumes EnvStep; `grep` finds no torch/sklearn-fit in the env path |
| oracle predictors | **none** | — | no surface reads future arrivals/tokens |

**Takeaway:** every prediction today is a running statistic or heuristic; there is
**no trained ML forecaster** and **no arrival / SLA-risk forecaster** — exactly the
gap this PR fills. The fleet/KV signals are anchored constants (no per-hour series to
forecast), which the forecasting layer must report honestly rather than fake.

## B. Existing connected actions

| action | classification | where | notes |
|---|---|---|---|
| ordering / scheduling | **CONNECTED_TO_ENV** | `serving_plane.run_hour(ordering=…)` → `unified_replay` | `fifo` / `abs_conformal` |
| admission / defer | **CONNECTED_TO_ENV** | `run_hour(admission=…)` | `off` / `class_aware` |
| replica / capacity adjustment | **CONNECTED_TO_ENV** | `run_hour(capacity=…)` | `reactive_lag1` / `backlog_aware` / `forecasted_mcs` |
| KV-aware routing | **SIMULATED_ONLY** | `kv_cache.KVAwareRouter` | exists + validated, but **not wired into serving** (`grep` shows it's never used in `canonical.py`/`serving_plane.py`) |
| KV enable/disable | construction flag (not a per-step action) | `canonical.py __init__ kv_enabled` | not policy-controlled per hour |
| cost scenario (owned/leased) | construction flag / **evaluation only** | `canonical.py __init__ cost_scenario` | an accounting choice, not a product action |
| placement / packing | **NOT_IMPLEMENTED** | — | no placement action in the env |
| migration | **NOT_IMPLEMENTED** | — | — |
| prewarming / pre-position | **NOT_IMPLEMENTED** | — | `warmup_c` is derived from capacity, not an action |
| clock / DVFS | **NOT_IMPLEMENTED** | — | service time is fixed; no clock knob |
| precision / model routing | **NOT_IMPLEMENTED** | — | no precision action; **no quality data** |
| speculative decoding | **NOT_IMPLEMENTED** | — | — |
| energy / price-aware shifting | **NOT_IMPLEMENTED** | — | electricity price feeds **cost only**, not scheduling |
| per-chip perf/watt routing | **REQUIRES_PILOT_TELEMETRY** | — | per-chip DCGM/Zeus telemetry ABSENT |

**Connected action set for this PR's controller: `{capacity, ordering, admission}`** —
the only three actions that actually execute in the environment. KV routing is
SIMULATED_ONLY (a roadmap item, not a controller action this PR). Everything else is
NOT_IMPLEMENTED or pilot-gated and is recorded in the Phase 4 roadmap, **not** offered
as a fake action.

## C. Forecast targets — forecastability triage

| target | forecastable? | plan |
|---|---|---|
| arrival_rate | **YES** (temporal/diurnal) | full ladder; primary controller input |
| input_tokens, output_tokens | **YES** | full ladder (mean + p95 quantile) |
| queue_delay (serving) | **YES** (derived from load vs capacity) | full ladder on realized serving wait |
| SLA_risk | **YES** (derived) | quantile/classification ladder |
| electricity_price | **YES** (diurnal, near-deterministic) | seasonal baseline dominates; ladder confirms |
| KV_reuse | weak (stable) | naive (fitted rate) — stable train≈holdout; report, don't overfit |
| GPU_utilization, GPU_memory_pressure, network_pressure | **anchored-constant** | naive marginal; **report no per-hour series** (RUNNING_STATISTIC), not a fake model |
| job_runtime | **ABSENT** | SKIPPED with the required artifact |

Hard rule enforced downstream: a learned model is used **only if it beats the naive
baseline on held-out data**; otherwise the naive baseline is kept and the result is
reported honestly.
