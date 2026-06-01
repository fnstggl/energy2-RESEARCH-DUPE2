# Economic Overlay Layer v1

> **Overlay / shadow PR.** No production scheduler / residency / scorer /
> robust-energy behaviour is changed. No real execution is enabled. No
> production savings are claimed. The existing
> `ConstraintShadowScorer` (PR #139) remains the safety floor; this PR
> ADDS an overlay that supplies missing economic terms from public data.
> HF / public-trace / public-list-price data is NEVER labelled pilot
> telemetry or operator truth.
>
> **Read first:**
> - `docs/RESULTS.md`
> - `docs/HF_DATASET_REGISTRY.md`
> - `docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md` (which $-coefficients are
>   operator-policy-only)
> - `docs/HF_ECONOMIC_SIGNAL_DISCOVERY_AUDIT.md` (PR #140 — which
>   public datasets exist)
> - `docs/CACHE_PREFIX_REUSE_FORECASTER_V1.md`,
>   `docs/CARA_LATENCY_FORECASTER_V1_CALIBRATION.md`
> - `data/external/economic_overlay/source_coverage_matrix.json`
>   (machine-readable Phase-1 audit)
> - `data/external/economic_overlay/economic_overlay_summary.json`
>   (Phase-3 build rollup)
> - `data/external/economic_overlay/economic_overlay_eval.json`
>   (Phase-6 A-H evaluation)

## 1. Goal

Build a public-data-calibrated economic overlay that joins Aurelius
operational traces to real / derived economic signals so SLA-safe
goodput per dollar can be evaluated without invented constants.

The constraint-scorer-upgrade audit (PR #139) closed with
`final_status = diagnostic_only` and explicitly named the missing
slots:

> *Pass-1 (priors only) does not improve SLA-safe goodput/$ — Optimum/
> CARA service-time priors widen the cost denominator without a
> compensating $-savings term. Cache savings can't translate to a
> dollar improvement until the operator supplies a per-GPU price
> spread.*

This PR builds the overlay that supplies that price spread from
public data **without inventing constants**, while making the
operator-policy-only coefficients (`energy_price_per_kwh_usd`,
`carbon_price_per_kg_usd`, fleet-actual `$/hr`) explicitly missing.

## 2. Binding rules (CORE PRINCIPLE)

Every numeric overlay value carries a `value_quality` label drawn
from a frozen vocabulary:

- `measured` — Level 1, direct observation (dataset / API / operator
  policy).
- `derived` — Level 2, transparent formula over Level 1 / Level 3
  inputs. The exact formula is recorded in
  `formula_by_field[field]`.
- `prior` — Level 3, public benchmark or market prior (research use).
- `scenario_prior` — Level 3 with explicit *scenario* tag (region /
  market assumed, never operator-truth).
- `missing` — Level 4, not computable; **must not** be filled with
  any constant.

Forbidden in this module (regex-asserted by
`tests/test_economic_overlay_formulas.py::test_no_invented_constants_in_module`):

- hardcoded `GPU_HOUR_PRICE = …`, `DEFAULT_GPU_HOUR = …`,
  `CACHE_VALUE_WEIGHT = …`, `CACHE_WEIGHT = …`,
  `MIGRATION_PENALTY = …`, `UTILITY_SCORE = …`,
  `COMPOSITE_WEIGHT = …`.
- any utility-weighted composite (no `0.4·latency + 0.3·cache + …`
  form anywhere).
- treating PJM / ERCOT / CAISO LMP as operator utility contract
  (it's `measured` only as a *market price*, never as the
  operator's tariff).
- treating `afhubbard/gpu-prices` as operator invoice (it's a public
  list price; the limitation block records this explicitly).

## 3. Sources (Phase 1)

Full per-source matrix in
`data/external/economic_overlay/source_coverage_matrix.json`. Headline:

### 3.1 Operational traces (already ingested)

| Source | Role | What it supplies |
|---|---|---|
| `asdwb/cara_latency_prediction` | Tier-2 telemetry | TTFT / TPOT / e2e / queue state / KV utilization |
| `Qinghao/AcmeTrace` | Tier-2 telemetry + Tier-3 cluster | DCGM gpu_util + IPMI gpu_power_w + queue_wait + state |
| `optimum-benchmark/llm-perf-leaderboard` | Tier-4 benchmark | TTFT/TPOT p50/p90/p95/p99 + per-request CodeCarbon kWh + peak VRAM |
| `eth-easl/swissai-serving-trace` | Tier-4 cache residency | reuse_percentage + bucket_count |
| `ejhusom/llm-inference-energy-consumption` | Tier-4 energy | Real Ollama timing + CodeCarbon kWh + `load_duration_ns` (cold-start signal) |

### 3.2 Economic overlays (added in this PR)

| Source | Role | What it supplies | License | Live in this PR? |
|---|---|---|---|---|
| `afhubbard/gpu-prices` | Public-list GPU rental pricing across 12+ clouds | gpu_price_usd_per_hour by (provider, gpu_type, region, is_spot) | CC-BY-4.0 | YES — 1 most-recent snapshot, 50,613 raw rows aggregated to 2,090 (provider × gpu × region × spot) medians |
| PJM Data Miner — DA LMP | US-East energy market | $/MWh hourly, last 7 days | API | YES — live via `aurelius.ingestion.grid_apis.pjm` with `PJM_API_KEY` |
| ERCOT — scenario | US-South energy market scenario | scalar midpoint $/kWh | n/a | NO — credentials incomplete (ERCOT_PASSWORD missing); scenario midpoint only |
| CAISO — scenario | US-West energy market scenario | scalar midpoint $/kWh | n/a | NO — no CAISO creds; scenario midpoint only |
| WattTime — scenario | Carbon intensity | scalar g CO2 / kWh | n/a | NO — WattTime auth currently failing in env; scenario midpoint only |
| `no_operator_policy_overlay` | Baseline | empty | n/a | n/a |

All non-PJM live fetches degrade to `scenario_prior` tables defined
in `aurelius.forecasting.economic_overlay::SCENARIO_OVERLAYS`. Each
scenario value is fixed in source so the scenario-vs-measured
distinction is auditable in code review.

## 4. Schema (Phase 3)

`aurelius.forecasting.economic_overlay.EconomicOverlayRecord` — one
per operational trace row. Numeric fields default to `None`
(NOT 0.0) and carry per-field `value_quality_by_field[field]`
labels and per-derived-field `formula_by_field[field]` strings.

Per-record fields (subset; see source for full list):

- Operational inputs (copied from the trace): `gpu_type`, `gpu_count`,
  `prompt_tokens`, `output_tokens`, `ttft_s`, `tpot_s`,
  `e2e_latency_s`, `throughput_tok_s`, `cache_reuse_pct`,
  `kv_utilization`, `peak_vram_gb`, `gpu_power_w`, `energy_kwh`.
- Joined economic inputs: `electricity_price_usd_per_kwh`,
  `carbon_intensity_g_per_kwh`, `gpu_price_usd_per_hour`.
- Derived seconds: `estimated_gpu_seconds`, `estimated_prefill_seconds`,
  `estimated_decode_seconds`.
- Derived costs: `estimated_gpu_cost_usd`, `estimated_energy_cost_usd`,
  `estimated_carbon_kg`, `estimated_carbon_cost_usd` (operator only),
  `estimated_prefill_cost_usd`, `estimated_decode_cost_usd`,
  `estimated_cache_value_usd`, `estimated_migration_cost_usd`,
  `estimated_cold_start_cost_usd`.
- SLA: `sla_s`, `sla_met`, `sla_safe_goodput`,
  `sla_safe_goodput_per_dollar`.
- Auditability: `value_quality_by_field`, `formula_by_field`,
  `limitations`, `overlay_class`.

## 5. Join logic + formulas (Phase 4)

```
gpu_seconds_estimate
  = e2e_latency_s                     [measured]
  | ttft_s + tpot_s * output_tokens   [derived]
  | missing

energy_kwh
  = trace.energy_kwh                          [measured  — Optimum / ejhusom]
  | gpu_power_w * gpu_seconds / 3_600_000     [derived_from_power_prior — AcmeTrace]
  | missing

electricity_price_usd_per_kwh
  = operator_policy.energy_price_per_kwh_usd  [measured]
  | nearest PJM DA LMP / 1000                 [measured]
  | scenario_overlay.price_per_kwh_usd_p50    [scenario_prior]
  | missing

carbon_intensity_g_per_kwh
  = watttime scenario midpoint (us-east)      [scenario_prior]
  | missing

gpu_price_usd_per_hour
  = operator_policy.gpu_hour_price_per_type[gpu_type]   [measured]
  | afhubbard exact (family + provider + region + spot) [prior]
  | afhubbard family + region on-demand median          [prior]
  | afhubbard family-only on-demand median              [prior]
  | afhubbard nearest-capability-family median          [prior_fuzzy_match]
  | missing

# Costs
estimated_gpu_cost_usd
  = gpu_price_usd_per_hour * gpu_count * gpu_seconds / 3600

estimated_prefill_cost_usd
  = ttft_s * (gpu_price_usd_per_hour / 3600) * gpu_count

estimated_decode_cost_usd
  = (tpot_s * output_tokens) * (gpu_price_usd_per_hour / 3600) * gpu_count

estimated_energy_cost_usd
  = energy_kwh * electricity_price_usd_per_kwh

estimated_carbon_kg
  = energy_kwh * carbon_intensity_g_per_kwh / 1000

estimated_carbon_cost_usd
  = estimated_carbon_kg * operator_policy.carbon_price_per_kg_usd
  | missing                                          # operator-policy-only

estimated_cache_value_usd
  = cache_reuse_pct * ttft_s * (gpu_price_usd_per_hour / 3600) * gpu_count

estimated_migration_cost_usd
  = cache_loss_pct * ttft_s * (gpu_price_usd_per_hour / 3600) * gpu_count
  | missing if cache_loss_pct absent

estimated_cold_start_cost_usd
  = model_load_duration_s * (gpu_price_usd_per_hour / 3600) * gpu_count
  | missing if no model_load_duration_s (proxy when source !=
    "measured", derived when source == "measured")

# Headline
sla_met
  = (e2e_latency_s <= sla_s)
sla_safe_goodput
  = output_tokens if sla_met else 0
cost_denom
  = gpu_cost + energy_cost + migration_cost + cold_start_cost
    - cache_value
sla_safe_goodput_per_dollar
  = sla_safe_goodput / cost_denom   when cost_denom > 0
```

**Carbon cost is NOT in the headline cost denominator.** Carbon kg
*is* tracked as a physical quantity (g CO2 → kg). A $-carbon term
only appears when `OperatorPricingPolicy.carbon_price_per_kg_usd`
is supplied; the field stays `missing` otherwise. This matches the
mission spec.

## 6. Scenario overlays (Phase 5)

Defined in `SCENARIO_OVERLAYS` and mirrored in the source matrix:

- `pjm_energy_overlay` — US-East midpoint $0.045/kWh (replace with
  live PJM in production; we DO use the live feed in this PR when
  `PJM_API_KEY` is set).
- `ercot_energy_overlay` — US-South midpoint $0.038/kWh
  (`scenario_prior` only in this PR).
- `caiso_energy_overlay` — US-West midpoint $0.072/kWh
  (`scenario_prior` only in this PR).
- `watttime_carbon_overlay` — US-East midpoint 410 g CO2 / kWh
  (`scenario_prior` only in this PR).
- `market_price_public_gpu_overlay` — `afhubbard/gpu-prices`
  (Phase 2 normalised sample; CC-BY-4.0).
- `no_operator_policy_overlay` — baseline; every economic term
  comes back missing.

Tests force every scenario entry to be labelled `scenario_prior`
(or `missing` for `no_operator_policy_overlay`).

## 7. Three result classes (mission §6, binding)

Records are classified by the inputs that flow into the headline
`sla_safe_goodput_per_dollar`:

- `measured_same_record` — gpu_price is `measured` (operator policy)
  AND energy_kwh is `measured` (trace).
- `cross_dataset_joined` — at least one input from public overlays
  (gpu price prior + PJM measured energy price) and no
  `scenario_prior` input.
- `scenario_prior` — at least one input is `scenario_prior`.

Per-class headline is reported separately in
`economic_overlay_eval.json::variants[*].metrics.headline_sla_safe_goodput_per_dollar_per_class`.

**They are NEVER combined into a single headline.**

## 8. Phase-2 ingest summary

| Source | Bytes raw (gitignored) | Bytes committed | Rows committed | License |
|---|---:|---:|---:|---|
| `afhubbard/gpu-prices` snapshot 2026-06-01 | 340,597 (parquet) | 747,990 (jsonl) | 2,090 aggregated medians | CC-BY-4.0 |
| PJM DA LMP us-east, 2026-05-23..30 | n/a (live API) | 32,938 | 169 hourly | API (live fetch) |
| ERCOT scenario | n/a | n/a (in code) | n/a | scenario_prior |
| CAISO scenario | n/a | n/a (in code) | n/a | scenario_prior |
| WattTime scenario | n/a | n/a (in code) | n/a | scenario_prior |

Operational sources reuse the existing committed JSONL fixtures
under `tests/fixtures/hf/` and the per-config samples under
`data/external/hf/.../processed/normalized_sample.jsonl` from the
prior ingest PRs (#129, #133, #135, #138). **No new operational
ingest is performed by this PR** — Phase 2 expansion was scoped to
the overlay-side tables.

Total new committed bytes under `data/external/economic_overlay/`:
~1.4 MB (well under the 100-MB-per-file and 300-MB-per-PR caps).

## 9. Build + eval rollup

Driver: `scripts/build_economic_overlay_v1.py`. Eval:
`scripts/run_economic_overlay_eval_v1.py`.

```bash
HF_TOKEN=… PJM_API_KEY=… python3 scripts/build_economic_overlay_v1.py
python3 scripts/run_economic_overlay_eval_v1.py
```

35 operational rows joined across 5 sources (CARA test_flat +
Optimum 1×A100/A10/T4 + AcmeTrace seren_ipmi + SwissAI Qwen3-32B +
ejhusom alpaca/gemma 7b workstation).

### 9.1 Field-quality breakdown (full overlay variant E)

| Term | n total | n derived | n scenario_prior | n missing |
|---|---:|---:|---:|---:|
| estimated_gpu_cost_usd | 35 | 35 | 0 | 0 |
| estimated_energy_cost_usd | 35 | 25 | 0 | 10 |
| estimated_carbon_kg | 35 | 0 | 25 | 10 |
| estimated_carbon_cost_usd | 35 | 0 | 0 | 35 *(operator-policy-only)* |
| estimated_cache_value_usd | 35 | 5 | 0 | 30 |
| estimated_migration_cost_usd | 35 | 0 | 0 | 35 |
| estimated_cold_start_cost_usd | 35 | 5 | 0 | 30 |
| estimated_prefill_cost_usd | 35 | 30 | 0 | 5 |
| estimated_decode_cost_usd | 35 | 30 | 0 | 5 |
| sla_safe_goodput_per_dollar | 35 | 35 | 0 | 0 |

### 9.2 A through H variant headline (Phase 6)

Primary KPI: SLA-safe goodput per dollar.

| Variant | n records | n goodput/$ | n gpu_cost | n energy_cost | n cache_value |
|---|---:|---:|---:|---:|---:|
| A — existing scorer baseline (NO overlay) | 35 | **0** | 0 | 0 | 0 |
| B — + public GPU price overlay | 35 | 35 | 35 | 0 | 5 |
| C — + energy/carbon overlay only | 35 | 25 | 0 | 25 | 0 |
| D — + cache value overlay | 35 | 35 | 35 | 0 | 5 |
| E — + full economic overlay | 35 | 35 | 35 | 25 | 5 |
| F — full + TTFT p50 prior | 35 | 35 | 35 | 25 | 5 |
| G — full + cache/prefix prior | 35 | 35 | 35 | 25 | 30 |
| H — full + both priors | 35 | 35 | 35 | 25 | 35 |

Per-class headline (E_existing_plus_full_overlay):

- `measured_same_record`: 0 records — no operator policy supplied by
  default, so no record satisfies "both inputs measured".
- `cross_dataset_joined`: 35 — every record has gpu price (prior) +
  PJM measured energy price.
- `scenario_prior`: 0 — PJM live, no scenario inputs.

Add `OperatorPricingPolicy(gpu_hour_price_per_type=…,
energy_price_per_kwh_usd=…)` to lift records into
`measured_same_record`. Drop PJM_API_KEY and switch to
`ercot_energy_overlay` to lift into `scenario_prior`.

Ranking / top-1 change rate vs baseline: **0.0** for every variant.
The overlay is **additive** to the existing scorer — it does not
re-rank candidates, it fills in dollar terms that the production
scorer reports as uncalibrated.

## 10. Promotion (Phase 9)

```
final_status = economic_overlay_ready
reason       = Baseline (no overlay) computes no economic goodput/$;
               full overlay computes it on every record where SLA
               fields are present. The overlay supplies the missing
               inputs deterministically from public data, but the
               result depends on public GPU list price and
               scenario_prior carbon — NOT production-ready.
carbon_cost_held_missing_under_default_policy = True
```

Why not `shadow_ready_for_integration_review`: the overlay does not
change the scorer's ranking (it's additive). The mission's
`>5% goodput/$ improvement` rule does not apply because the baseline
goodput/$ is `0/35` — *every* improvement is from an undefined to a
defined number. That's a calibration unlock, not a ranking
improvement.

Why not `diagnostic_only`: the overlay reliably computes the
$-denominated terms that PR #139 reported as
`not_computable_without_operator_policy`. That is a measurable,
deterministic improvement in the *computability* of the headline KPI.

## 11. What stays pilot-only (binding)

Unchanged from `docs/CONSTRAINT_SCORER_UPGRADE_AUDIT.md::§10`:

- Real measured energy per request from the production cluster
  (Optimum/ejhusom are cross-hardware priors, not pilot data).
- Real per-GPU power draw on the production cluster (AcmeTrace is
  a cross-cluster prior).
- Real measured cache_hit per request.
- Cold-start latency per (model, GPU, cluster) — public datasets
  only give a proxy or a single-machine measurement.
- Operator-supplied `energy_price_per_kwh_usd` from the actual
  utility bill or live spot feed (PJM LMP is a market price, NOT
  the operator's contracted tariff).
- Operator-supplied `carbon_price_per_kg_usd` (shadow price).
- Operator-supplied per-GPU `$/hr` from the cloud invoice or
  internal chargeback (afhubbard/gpu-prices is the public LIST
  price; this is recorded as `prior`, never `measured`).
- Operator memory-pressure pricing policy.

## 12. What can / cannot be claimed externally

**CAN be claimed:**

- "Aurelius now computes per-record `sla_safe_goodput_per_dollar`
  from public-data overlays" — true; 35/35 derivable in the eval.
- "The overlay supplies real GPU price priors across 12+ public
  clouds via `afhubbard/gpu-prices`."
- "The overlay supplies real PJM hourly DA LMP energy price when
  `PJM_API_KEY` is configured."
- "Every overlay value carries an explicit `value_quality` label and
  per-derived-field formula string."
- "The overlay produces enough signal to train economic ML targets
  *for research/simulation*, with the explicit caveat that all
  values are public-data priors or scenarios, not operator truth."

**CANNOT be claimed:**

- Production cost savings (`production_claim=False` is pinned).
- Operator chargeback or invoice truth (the public list price is
  not an operator invoice).
- Carbon cost from public data (carbon cost stays `missing` until
  the operator supplies `carbon_price_per_kg_usd`).
- Generalisation beyond PJM us-east for the live energy lookup.
- Cross-region routing decisions calibrated on the overlay — the
  overlay does not score candidates.

## 13. Tests

`tests/test_economic_overlay_sources.py` — Phase 1 source matrix
schema, scenario labelling, no secret leak, raw downloads gitignored,
no oracle/FIFO headline, production scorer files untouched.

`tests/test_economic_overlay_formulas.py` — Phase 3 + Phase 4
formula correctness, missing-when-inputs-missing for every term,
carbon-cost-requires-operator-policy, value_quality vocabulary
respected, no invented constants in the module.

`tests/test_economic_overlay_joining.py` — Phase 4 + Phase 5 join
logic (afhubbard exact + fuzzy fallback), operator policy
precedence, PJM measured vs scenario, WattTime carbon is physical
intensity not price.

`tests/test_economic_overlay_eval.py` — Phase 6 + Phase 9 eval
rollup integrity, all 8 variants present, baseline-A has 0
computable goodput/$, full-overlay-E lifts that to ≥30, promotion
is a known state with a reason, per-class headline reported
separately.

All 4 files run in milliseconds (no external API calls in tests).

## 14. Reproducibility

```bash
# Phase 1 + 2 + 3 (build):
HF_TOKEN=… PJM_API_KEY=… python3 scripts/build_economic_overlay_v1.py
# Phase 6 (eval A through H):
python3 scripts/run_economic_overlay_eval_v1.py
# Tests:
pytest tests/test_economic_overlay_*.py -q
```

Without `PJM_API_KEY`, the build falls back to the
`pjm_energy_overlay` scenario midpoint and every record's
`overlay_class` flips from `cross_dataset_joined` to
`scenario_prior`. The tests note the missing live data and skip
the relevant assertions.
