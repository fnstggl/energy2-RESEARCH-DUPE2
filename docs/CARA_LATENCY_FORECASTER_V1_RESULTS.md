# CARA Latency Forecaster v1 — Results

> **Research / backtest / shadow-only PR.** No ML model from this work
> is wired into any controller. No scheduler defaults are changed. No
> external savings number is quoted. Live customer telemetry remains
> the only Tier 1 calibration source per `docs/RESULTS.md` §8.
>
> **Read first:**
> - `docs/RESULTS.md`
> - `docs/FORECAST_LEVERAGE_AUDIT.md`
> - `docs/HF_CARA_SWISSAI_TELEMETRY_AUDIT.md`
> - `docs/HF_DATASET_REGISTRY.md`
> - `docs/BENCHMARK_BASELINE_AUDIT.md`
> - `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`

## 1. Mission + scope (binding)

Build TTFT + E2E latency forecasters at p50 / p95 / p99 for
heterogeneous GPU placement and request routing, using the CARA
analysis-tier ingest from PR #124 (76,825 rows of train_flat + 38,509
rows of train_queue_details — `strong` strength).

The forecaster's success metric is **incremental alpha vs the
strongest realistic baseline** (`per_instance_type_p{q}`) measured by
pinball loss + tail-underprediction safety, evaluated on **three
holdout strategies** (random / by_instance_type / time).

## 2. Data + leakage rules

- **Source:** `data/external/hf/asdwb__cara_latency_prediction/train_flat/processed/analysis_sample.jsonl` (gitignored; regenerable via `scripts/audit_cara_swissai_telemetry.py --target-set analysis_tier`).
- **Rows used:** 76,825 (train_flat).
- **Targets:** `actual_ttft_s`, `actual_e2e_latency_s`.
- **Leakage fields blocked from every feature pipeline:**
  `actual_e2e_latency_s`, `actual_ttft_s`, `actual_tpot_s`,
  `actual_output_tokens`, `completion_timestamp_s` (legacy aliases too).
  Enforced by `aurelius/forecasting/cara_latency_features.LeakageError`.
- **Predict-time features (24 numeric + 8 categorical):** request shape
  (`num_prompt_tokens`, `num_predicted_output_tokens`), scheduler state
  at decision time (`num_running`, `num_waiting`,
  `num_active_decode_seqs`, `pending_*_tokens`, `decode_ctx_*`,
  `token_budget_per_iter`, `prefill_chunk_size`, `max_num_seqs`,
  `num_preempted`, `ema_*`), KV-cache pressure (`kv_cache_utilization`,
  `kv_free_blocks`, `kv_evictions_per_s`), derived
  (`model_size`, `gpu_type`, `prompt_token_bin`, `queue_depth_bin`,
  `kv_util_bin`, `hour_of_day`). Full schema audit at
  `data/external/forecasting/cara_latency_forecaster_v1/schema_audit.json`.
- **`actual_output_tokens` mode:** the v1 driver uses
  `output_tokens_mode="predicted_only"` — `num_predicted_output_tokens`
  enters features, `actual_output_tokens` does not. An `oracle_shape`
  variant is supported by the feature pipeline but every metric it emits
  must be labelled `analysis_only`.

## 3. Baselines (the bars the ML must clear)

Per the mission PHASE 1 spec:

1. `global_constant_p{q}` — predicts the train-set p{q} for every row.
2. `per_instance_type_p{q}` — predicts p{q} of the matching CARA
   instance_type (5 unique). The **headline baseline** for the
   incremental-alpha gate.
3. `per_model_gpu_p{q}` — predicts p{q} of the matching
   (`model_size`, `gpu_type`). Equivalent to (2) because CARA's
   `instance_type` is a unique combination of `model_size` + `gpu_type`.
4. `queue_depth_bin_p{q}` — predicts p{q} of the matching
   `queue_depth_bin`.
5. `simple_rule_placement_score` — `per_instance_type_p95` + queue-depth
   penalty.

## 4. Models attempted

| Family | Implementation | Quantile | Role |
|---|---|---|---|
| Baselines (group quantile) | `GroupConstantQuantileBaseline` | p50 / p95 / p99 | strongest-realistic baseline |
| Simple-rule placement score | `SimpleRulePlacementScoreBaseline` | p95 | routing-rule baseline |
| Gradient boosting quantile | `sklearn.ensemble.HistGradientBoostingRegressor(loss="quantile")` | p50 / p95 / p99 | v1 ML default |
| Random forest median | `sklearn.ensemble.RandomForestRegressor` | mean / p50 | robustness candidate |
| HGB p95 × 1.10 (conservative-multiplier calibration) | `ConservativeMultiplierCalibration` | p95 | safety variant |
| Fallback-to-baseline wrapper | `FallbackToBaseline` | composite | OOD safety guard (analysis-only) |

## 5. Holdout strategies (3, not just random)

1. `random_holdout` — deterministic 80/20 split, seed 1773889.
2. `holdout_by_instance_type` — every `qwen2.5-7b_a30` request held
   out (~21k rows). Tests the OOD-instance generalisation that the
   constraint-aware engine actually cares about.
3. `time_holdout` — last 20% chronologically by
   `prediction_timestamp_s`. Tests temporal drift between collection
   sweeps.

## 6. Per-quantile incremental alpha (pinball-loss improvement vs `per_instance_type_p{q}`)

> The gate is **apples-to-apples**: ML p{q} pinball vs baseline p{q}
> pinball. MAE-against-actuals comparisons are *not* the headline
> because per-(instance_type) p95 is a tail predictor, while HGB p50 is
> a median predictor — comparing them on MAE would be misleading.

### TTFT (`actual_ttft_s`)

| Holdout | p50 alpha | p95 alpha | p99 alpha |
|---|---:|---:|---:|
| random | **+51.1%** | **+46.7%** | +22.6% |
| by_instance_type | **+37.1%** | **+76.7%** | **+79.4%** |
| time | **+42.0%** | +7.1% | **-16.8%** (regression) |

**Consolidated gate:**

- TTFT **p50 → `candidate_for_shadow_integration`** ✅ (clears 5% gate on 3/3 holdouts, no safety regression).
- TTFT p95 → `diagnostic_only` (loses safety check on time_holdout despite +7% alpha).
- TTFT p99 → `promising_needs_validation` (strong on 2 holdouts, regresses on time_holdout).

### E2E (`actual_e2e_latency_s`)

| Holdout | p50 alpha | p95 alpha | p99 alpha |
|---|---:|---:|---:|
| random | +2.9% | +1.7% | +0.2% |
| by_instance_type | **-5.1%** (regression) | **+58.5%** | **+64.0%** |
| time | +2.0% | +1.2% | -1.3% |

**Consolidated gate:**

- E2E p50 → `diagnostic_only` (parity or regression).
- E2E p95 → `diagnostic_only` (strong OOD signal swamped by parity on random + time).
- E2E p99 → `promising_needs_validation` (strong OOD evidence, near-parity elsewhere).

### What this means

The forecaster captures **per-(instance_type, queue_state) variation in
TTFT** clearly. E2E latency is dominated by what model size you picked
plus the actual output-token count, and we deliberately withhold
`actual_output_tokens` to avoid leakage. Without that feature the v1
model is at parity with `per_instance_type_p{q}` on E2E — an honest
ceiling.

The time_holdout regression for TTFT p99 (-17%) is a genuine warning:
the last 20% of CARA's chronological data may carry distribution shift
not present in the first 80%. **Shadow integration must respect this
non-stationarity** — a forecaster running on a 1-hour-old window is not
the same as a forecaster trained on chronological data 24 hours stale.

## 7. Subgroup metrics (HGB p95, random holdout)

Per-(instance_type) and per-(prompt_token_bin / queue_depth_bin /
kv_util_bin) subgroup metrics are written into
`model_comparison.json` under
`per_target.<target>.per_holdout.<holdout>.subgroup_metrics_for_hgb_p95`.
Subgroups below 100 holdout rows are flagged `INSUFFICIENT_SAMPLE`. No
subgroup at the random_holdout level fell below the threshold for any
of the 5 CARA instance_types; deeper bin combinations (e.g.
`queue_depth_bin = [100, 1000000)`) trigger the flag because CARA's
sweep 2 holds num_waiting near zero per the README.

## 8. Routing / placement backtest (counterfactual proxy)

`scripts/run_cara_latency_forecaster_v1_backtest.py` ranks every
holdout request across 5 candidate `instance_type`s and reports the
realised-latency distribution under 4 policies.

> **Honesty caveat (binding):** CARA only carries the realised latency
> at the instance_type each request **actually went to**. Counterfactual
> realised latencies are estimated as the bucket-mean of historical
> requests at the alternative instance with the same
> `(prompt_token_bin, queue_depth_bin)`. This is an honest proxy, NOT a
> measurement. Every result carries `result_quality =
> counterfactual_bucket_mean_proxy`.

### Per-policy realised-latency (15,365 holdout rows, target = TTFT)

| Policy | p50 (s) | p95 (s) | p99 (s) | Picks |
|---|---:|---:|---:|---|
| `round_robin` | 0.059 | 0.198 | 0.479 | 3,073 per instance |
| `per_instance_type_p95` | 0.029 | 0.050 | 0.050 | all → `qwen2.5-3b_a30` |
| `per_instance_type_p95_with_queue` | 0.029 | 0.050 | 0.050 | all → `qwen2.5-3b_a30` |
| **`ml_hgb_p95`** | 0.043 | 0.082 | 0.103 | mixed (54% to 7b_a30) |

### Per-policy realised-latency (target = E2E)

| Policy | p50 (s) | p95 (s) | p99 (s) | Picks |
|---|---:|---:|---:|---|
| `round_robin` | 6.12 | 23.01 | 36.66 | round-robin |
| `per_instance_type_p95` | 2.24 | 3.02 | 6.02 | all → `qwen2.5-3b_a30` |
| `per_instance_type_p95_with_queue` | 2.24 | 3.02 | 6.02 | all → `qwen2.5-3b_a30` |
| **`ml_hgb_p95`** | 3.13 | 4.49 | 8.74 | mostly `14b_v100` + `3b_a30` |

**Routing promotion classification (both targets):** `diagnostic_only`
with `safety_regression = 1`.

### Why the ML routing loses

CARA carries 5 `instance_type`s with no model-required-quality
constraint and no capacity constraint. Under "minimise realised
latency" alone, the trivially-best policy is **"send every request to
the smallest fastest model"** (`qwen2.5-3b_a30`). The baseline
`per_instance_type_p95_with_queue` discovers this immediately. The ML
model picks a mix of instance_types based on per-request features,
which is *worse* on latency-only because it occasionally routes to
larger / slower instances.

**This is an honest negative result for unconstrained routing.** The
forecaster's value is in *per-row latency prediction* (where TTFT p50
clears the gate cleanly), **not** in solving a routing problem where
the trivial baseline already routes optimally because no constraint
binds it.

In production:

- Capacity bounds make the trivial baseline infeasible.
- Required-model-quality constraints rule out the smallest model for
  many requests.
- Cost differentiation makes the smallest-fastest model not always the
  best $/token.

CARA doesn't carry any of those signals. The routing backtest's
negative result reflects the *dataset's missing constraints*, not the
forecaster's quality.

## 9. Incremental alpha gate decisions

Per mission PHASE 6 (binding):

| Target | quantile | Per-holdout alpha (3 strategies) | Consolidated gate |
|---|---|---|---|
| TTFT | p50 | +51.1 / +37.1 / +42.0 | **`candidate_for_shadow_integration`** ✅ |
| TTFT | p95 | +46.7 / +76.7 / +7.1 | `diagnostic_only` (safety regression on time) |
| TTFT | p99 | +22.6 / +79.4 / -16.8 | `promising_needs_validation` |
| E2E | p50 | +2.9 / -5.1 / +2.0 | `diagnostic_only` |
| E2E | p95 | +1.7 / +58.5 / +1.2 | `diagnostic_only` (random + time at parity) |
| E2E | p99 | +0.2 / +64.0 / -1.3 | `promising_needs_validation` |

The ML routing policy is `diagnostic_only` (safety regression) under
the unconstrained-latency formulation; see §8 for why.

## 10. Safety / honesty invariants enforced in code

- No scheduler / robust energy engine / controller is touched. Verified
  by the no-executor-reference grep test.
- No production-savings phrase in code, docs, or commit messages.
  Verified by the banned-phrase grep test.
- Raw + analysis_sample.jsonl gitignored (CARA train.jsonl + train
  queue_details). Verified by `git check-ignore`.
- Leakage fields (`actual_*`, `completion_timestamp_s`,
  `actual_output_tokens`) raise `LeakageError` if any feature pipeline
  would emit them in `predicted_only` mode.
- The `oracle_shape` variant — which can use `actual_output_tokens` —
  is supported but labelled `analysis_only`; the v1 driver does NOT use
  it for any reported metric.
- Holdout splits are deterministic (seeded).
- Per-quantile gate comparisons are apples-to-apples (pinball loss),
  not MAE-against-actuals (which would unfairly favour the median).
- Counterfactual routing latencies are labelled
  `counterfactual_bucket_mean_proxy`. No oracle baseline is used as a
  headline.

## 11. What advances + what doesn't

**Advances (this PR):**

- TTFT p50 forecasting at `candidate_for_shadow_integration` quality
  on all 3 holdouts. Direct input to the constraint-aware engine's
  per-request latency prior.
- A reusable, leakage-checked feature pipeline + per-quantile gate
  classifier that future forecasters can re-use.
- An honest counterfactual proxy methodology that doesn't pretend
  CARA carries ground-truth counterfactuals.

**Does NOT advance (yet):**

- E2E latency forecasting beyond parity on random + time holdouts —
  blocked by the deliberate exclusion of `actual_output_tokens` as a
  feature.
- Routing decisions under unconstrained latency — the trivial baseline
  wins by routing all traffic to the smallest model. Real routing
  needs capacity + quality + cost constraints CARA doesn't carry.
- TTFT p99 — time_holdout regression suggests temporal non-stationarity.
- Production integration — no.

## 12. Next actions (documented for the next run)

1. **Production-feasible routing backtest** — repeat the routing
   simulation with explicit `(model_size, max_required_quality)`
   constraints + per-(GPU type) cost mapping. The simplest realistic
   constraint: each request specifies a minimum model size. Without
   that signal, only per-row latency forecasting is provable.
2. **Time-window training** — train on rolling 12-hour windows to
   measure time_holdout p99 regression as a function of staleness.
3. **Production telemetry calibration** — once a pilot deployment lands
   measured `replica_count` and `SLA_label`, re-run with those signals
   so the SLA / autoscaling forecasts (currently `priors_only`) can be
   promoted.
4. **Shadow integration of TTFT p50** — wire the v1 TTFT p50 model as
   a *recommendation-only signal* into
   `aurelius/frontier/dynamic_estimator.py`'s prior path. Keep the
   deterministic Erlang-C tail as the binding safety floor.
