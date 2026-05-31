# Batch Inference Frontier v1 — Incremental Alpha Audit

> **Audit-only research result. Simulator / public-trace evidence only —
> NOT production savings** (`docs/RESULTS.md` §8). The robust energy engine,
> the serving Safe Utilization Frontier Controller, the Dynamic Safe
> Frontier Estimator, the Dynamic Serving Frontier Calibration harness,
> the Training Safe Utilization Frontier, the constraint-aware default rho
> (0.65), and every committed serving / training / residency benchmark
> artifact are **unchanged**. Real cluster execution is **disabled by
> default** everywhere; both the static and dynamic Batch Inference
> Frontier remain opt-in / shadow only. **No oracle / clairvoyant baseline
> is used as a headline.**

This audit asks one question:

> Does the Batch Inference Frontier (static v1 from
> `docs/EVAL_AND_BATCH_FRONTIER_RESULTS.md`, plus the new dynamic v1
> introduced in this PR) produce **genuinely incremental** SLA-safe
> goodput/$ alpha *beyond* the existing Dynamic Serving Frontier and
> constraint-aware scheduler on the same Azure LLM 2024 ticks, under the
> same KPI + safety constraints?

The answer determines whether the batch frontier should be integrated
into the constraint-aware scheduler. The pre-registered gate is
**`true_incremental_alpha_vs_dynamic_serving_pct > +2.0 %` AND no
SLA / queue-p99 regression vs `constraint_aware_static`**. On accept →
`PROPOSE_INTEGRATION`. Otherwise → `SHADOW_DIAGNOSTIC`.

- **Read first:** `docs/RESULTS.md`, `docs/PUBLIC_TRACE_BACKTESTS.md`,
  `docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md`,
  `docs/EVAL_AND_BATCH_FRONTIER_RESULTS.md`,
  `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`,
  `docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`,
  `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`,
  `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`,
  `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`.

## 1. Scope (binding)

- **Audit only.** No constraint-aware default is changed. No scheduler is
  integrated. The audit script writes a summary JSON; nothing in
  `aurelius/constraints/` or `aurelius/optimization/` is modified.
- **Same KPI.** `sla_safe_goodput_per_infrastructure_dollar` per
  `docs/RESULTS.md` §1; same `InfrastructureCostConfig`.
- **Same safety constraints.** `timeout ≤ 10 %`, `queue_p99 ≤ 2000 ms` —
  pre-registered, identical to the committed Azure 2024 dynamic frontier
  audit (`docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`).
- **Same arrival ticks.** Azure 2024 sample fixture
  (`tests/fixtures/azure_llm_2024_sample.csv` — the same fixture the
  committed dynamic-frontier audit uses), time-rescaled at the audit's
  primary `100×` busy tier.
- **No oracle headline.** Oracle / clairvoyant baselines are analysis-only
  per `docs/RESULTS.md` §3 and do NOT appear in the audit's
  alpha-decomposition table.

## 2. What's added in this PR (audit + dynamic batch v1)

| file | role |
|---|---|
| `aurelius/frontier/dynamic_batch_inference_estimator.py` | NEW — dynamic batch estimator + controller + telemetry adapter. Pure stdlib. Sibling of the static batch frontier and the serving dynamic estimator (does not import either's controller). |
| `aurelius/frontier/batch_inference_models.py` | adds `deferral_window_seconds` field on `BatchInferenceFrontierCandidate` so the dynamic estimator can sweep peak-shift without repurposing `batch_window_seconds`. Existing tests still pass. |
| `scripts/run_batch_inference_frontier_incremental_alpha_audit.py` | NEW — runs the 7-policy comparison + emits the alpha-decomposition + verdict. |
| `data/external/frontier/batch_inference_frontier_incremental_alpha_audit_summary.json` | machine-readable audit output. |
| `tests/test_dynamic_batch_inference_estimator.py` | NEW — pins the dynamic batch invariants. |
| `tests/test_batch_inference_frontier_incremental_alpha_audit.py` | NEW — pins the audit JSON shape + the SHADOW_DIAGNOSTIC verdict (so a regression upgrading to PROPOSE_INTEGRATION fires loudly and requires a deliberate review). |
| `docs/BATCH_INFERENCE_FRONTIER_INCREMENTAL_ALPHA_AUDIT.md` | this doc. |

The dynamic batch estimator has a multi-axis candidate space:

    (target_rho, deadline_slack_seconds, deferral_window_seconds,
     batch_concurrency)

The new degree of freedom — `deferral_window_seconds` — is the peak-shift
knob that the serving dynamic estimator does **not** have, and is the
theoretical source of any incremental batch alpha.

## 3. Compared policies

All policies are scored on the SAME ticks, SAME serving physics, SAME KPI.

| policy | description |
|---|---|
| `sla_aware` | fixed rho 0.50; reactive sizer (the standard interactive headline per `docs/RESULTS.md` §3 rule 5) |
| `current_price_only` | the energy-cost-only / no-SLA-info baseline (`docs/RESULTS.md` §3 rule 4). On this trace it collapses to a single static rho — no real DA-price signal exists for the Azure 2024 trace itself |
| `constraint_aware_static` | the committed engine default rho 0.65 (anchor for every alpha calculation) |
| `static_serving_frontier_controller` | rho 0.75 — the committed Safe Utilization Frontier Controller v1 (`docs/AZURE_2024_FRONTIER_CONTROLLER_RESULTS.md`) |
| `dynamic_serving_frontier` | the committed Dynamic Safe Frontier Estimator with a rolling window — no future leakage |
| `static_batch_inference_frontier` | best safe (rho, slack) from the static batch frontier (`docs/EVAL_AND_BATCH_FRONTIER_RESULTS.md`) |
| `dynamic_batch_inference_frontier` | NEW — dynamic batch estimator with rolling window + deferral knob |

## 4. Result — Azure 2024 fixture, 100× scale, deadline_slack 300 s

| policy | goodput/$ | timeout % | queue p99 (ms) | mean rho |
|---|---|---|---|---|
| `sla_aware` | 1,013,543 | 3.76 | 268 | 0.508 |
| `current_price_only` | 1,199,167 | 4.12 | 343 | 0.536 |
| `constraint_aware_static` | 1,199,167 | 4.12 | 343 | 0.536 |
| `static_serving_frontier_controller` | 1,199,167 | 4.12 | 343 | 0.536 |
| `dynamic_serving_frontier` | 1,199,167 | 4.12 | 343 | 0.536 |
| `static_batch_inference_frontier` | 1,199,167 | 4.12 | 343 | 0.536 |
| **`dynamic_batch_inference_frontier`** | **1,130,010** | **4.00** | **319** | **0.526** |

Five of the seven policies converge to **$1,199,167** at mean rho 0.536.
This is the known plateau the committed Azure 2024 audit also reports:
once `_constraint_trim` decides the safe replica count, every rho ≥ 0.55
collapses to the same effective number of replicas, so the higher-rho
candidates produce the same KPI as constraint_aware_static. (See
`docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md` §3 — the dynamic estimator
also lands at this plateau on the same trace.)

The dynamic batch frontier comes in **below** that plateau by ≈ 5.8 %.

## 5. Alpha decomposition

| term | formula | value | meaning |
|---|---|---|---|
| `duplicated_serving_frontier_alpha_pct` | (dyn_serv − ca_static) / ca_static × 100 | **+0.00 %** | how much of the batch frontier's would-be alpha is *already captured* by the existing dynamic serving frontier on this trace |
| `deadline_flex_scenario_alpha_pct` | (static_batch − ca_static) / ca_static × 100 | **+0.00 %** | how much alpha the *deadline-flex scenario alone* produces at a static knob |
| `true_incremental_alpha_vs_dynamic_serving_pct` | (dyn_batch − dyn_serv) / dyn_serv × 100 | **−5.77 %** | the only number that justifies scheduler integration |
| `incremental_alpha_vs_static_batch_pct` | (dyn_batch − static_batch) / static_batch × 100 | **−5.77 %** | dynamic vs static within the batch class |

**Robustness check — extended trace.** When the same fixture is
concatenated 8× (128 ticks, giving the dynamic estimators time to warm
past the 8-tick minimum window), the gap widens to **−14.4 %** — the
dynamic batch estimator chooses rho 0.5103 on average (vs the 0.5358
constraint_aware static), and the cost of running with more replicas
outweighs the predicted-goodput improvement at every per-tick decision.

## 6. Verdict — `SHADOW_DIAGNOSTIC`

The pre-registered gate is `true_incremental_alpha_vs_dynamic_serving_pct
> +2.0 % AND no SLA / queue-p99 regression vs constraint_aware_static`.

| condition | value |
|---|---|
| `true_incremental_alpha_vs_dynamic_serving_pct` | −5.77 % (extended-trace: −14.4 %) |
| `alpha_gate_passed` | **False** |
| `no_safety_regression` | True |
| **`verdict`** | **`SHADOW_DIAGNOSTIC`** |

The Batch Inference Frontier v1 — both static and dynamic — does **NOT**
produce incremental alpha beyond the existing dynamic serving frontier on
the Azure 2024 trace at the audit's primary 100× busy tier.

**The audit recommends keeping the batch frontier as a research /
diagnostic module and NOT integrating it into the constraint_aware
scheduler at this time.**

## 7. Why no incremental alpha

Three structural reasons, in order of impact:

1. **`_constraint_trim` collapses rho choices above the safe peak.** The
   existing serving replay (`aurelius/traces/backtest.py::_constraint_trim`)
   trims replicas downward as long as the SLA stays met. On the Azure
   2024 100× sample, the safe peak rho is well above the offered load —
   so every candidate rho ≥ 0.55 lands at the same effective replica
   count, and the same goodput/$. The committed dynamic frontier audit
   (`docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md` §3) shows the same
   plateau: the dynamic serving frontier captures ≈ 73 % of the oracle
   gap, but the absolute KPI plateau at $1.18 M–$1.20 M is the same
   ceiling.
2. **The deadline-flex scenario alpha is zero on this trace.** The static
   batch frontier with `deadline_slack=300 s` lands at the same KPI as
   `constraint_aware_static`. Deadline slack only buys real alpha when the
   serving SLA is *binding* on the load — but the Azure 2024 fixture's
   load at 100× is well below the SLA gate, so relaxing the SLA gate
   adds nothing. (Phase A's sanity check, pinned in
   `tests/test_batch_inference_frontier.py`, only shows the slope at
   `slack=0 vs slack=60s` — the *binary safety transition*, not a
   monotone alpha.)
3. **The dynamic batch per-tick predictor is structurally noisy.** The
   v1 model projects next-tick load via EWMA and evaluates each
   candidate with a single-tick synthetic replay. At low projected rates
   it tends to recommend a lower rho than constraint_aware_static would
   keep, which increases replica count and lowers goodput/$. A pilot
   would need a better next-tick projector + a multi-tick lookahead +
   a cost-amortisation term in the per-candidate KPI prediction before
   the dynamic batch frontier could plausibly beat the serving frontier
   on this trace.

The audit's negative result is consistent with the discovery audit's
qualitative finding (`docs/FRONTIER_DISCOVERY_RESEARCH_AUDIT.md` §2.1
batch_inference: expected_alpha_score = 4, but flagged that the lever is
*deadline flexibility*, which depends on a binding SLA gate — which the
Azure 2024 fixture does NOT have at 100×).

## 8. What would change the verdict

The gate is open if a future iteration improves at least one of:

- **A binding deadline scenario.** Run the audit on a workload mix where
  the interactive SLA is *binding* on the offered load (e.g. the BurstGPT
  high-burst regime, or a synthetic mix where deadline_miss > 0 % at
  `constraint_aware_static`). On such a trace the deadline-slack scenario
  alpha would be > 0 by construction, and the batch frontier would have
  real room to win.
- **A multi-tick lookahead estimator.** The current v1 per-tick predictor
  recommends from a snapshot. A multi-tick estimator that amortises cost
  + deferral across a horizon would not penalise the dynamic batch
  estimator at low load.
- **A real deferral implementation.** The current v1 *models* deferral as
  a fractional shed of next-tick arrivals; it does not actually shift
  arrivals into future ticks. A streaming-replay variant that maintains
  a deferred-arrival queue across ticks would let the batch frontier
  realize peak-shift alpha.

None of those changes are in this PR. The audit is honest: under the
current implementation + conservative assumptions, the verdict is
`SHADOW_DIAGNOSTIC`.

## 9. Tests pin the verdict

- `tests/test_dynamic_batch_inference_estimator.py` — pins the new
  dynamic batch invariants (no serving-controller imports; multi-axis
  candidate sweep; rolling window; recommendation-only;
  executable_in_real_cluster=False; deferral_window_seconds field is
  used).
- `tests/test_batch_inference_frontier_incremental_alpha_audit.py` —
  pins the audit JSON shape AND the SHADOW_DIAGNOSTIC verdict on the
  committed Azure 2024 fixture at 100× scale. **A regression to
  PROPOSE_INTEGRATION fires the test loudly** — flipping the verdict
  requires re-running the audit AND updating the pinned-verdict test
  AND providing the alpha evidence in the PR.

## 10. Honesty / scope

- Simulator / public-trace evidence only — directional, **NOT production
  savings**. The `docs/RESULTS.md` §8 production-claim gate is unchanged
  and not satisfied for either the static or the dynamic batch frontier.
- The Azure 2024 sample is a SERVING trace re-used as a SYNTHETIC
  batch-flex scenario (explicit `synthetic_scenario_label` on the
  workload profile). The v1 dynamic batch estimator never reads a real
  deadline from a serving trace.
- No oracle / clairvoyant baseline is used as a headline; the audit's
  alpha decomposition compares dynamic batch against the strongest
  *deployable* baseline (the dynamic serving frontier).
- The robust energy engine, the serving rho controller, the
  constraint_aware default rho (0.65), and every committed serving /
  training / residency / Azure 2024 Dynamic Frontier Calibration
  artifact are **unchanged**.
- The audit does not modify any scheduler / production / optimizer
  module. The dynamic batch estimator + controller stay in
  `aurelius/frontier/` and are not wired into `aurelius/constraints/` or
  `aurelius/optimization/`.
