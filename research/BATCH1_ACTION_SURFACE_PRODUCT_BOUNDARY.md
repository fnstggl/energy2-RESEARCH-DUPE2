# Batch-1 action-surface product boundary (corrective PR)

**Product boundary.** Aurelius is primarily a **GPU fleet orchestrator**. It may optimize fleet-level
decisions by default. It must **not** silently take control of serving-engine internals unless the serving
stack explicitly exposes that capability. Every action surface is classified into exactly one category
(`actions.product_category`): `CORE_ORCHESTRATION_DEFAULT`, `CORE_ORCHESTRATION_AUTO_NOOP`,
`OPTIONAL_SERVING_ENGINE_INTEGRATION`, `DIAGNOSTIC_ONLY`, `PLANNED_ONLY`.

## Classification of the Batch-1 knobs

| knob | category | default | reaches default headline? | requires opt-in? |
|--|--|--|--|--|
| `gpu_assignment_policy` | **CORE_ORCHESTRATION_AUTO_NOOP** | ON / AUTO | no (auto-noop: single-dominant-GPU cost path → NOT_APPLICABLE) | no |
| `kv_cache_precision_policy` (fp8/int8) | **OPTIONAL_SERVING_ENGINE_INTEGRATION** | **OFF** | no (not generated unless enabled) | **yes** (`enable_kv_cache_precision`) |
| `kv_cache_precision_policy` = `kv_int4_diagnostic_only` | **DIAGNOSTIC_ONLY** | OFF | no | yes + `allow_quality_risk` |
| `prefill_decode_policy` | **OPTIONAL_SERVING_ENGINE_INTEGRATION** | **OFF** | no (not generated unless enabled) | **yes** (`enable_prefill_decode_disagg`) |
| `precision_policy` = `int4` (weights) | **DIAGNOSTIC_ONLY** | bf16 (no-op) | no | `allow_quality_risk` |
| existing fleet knobs (capacity/ordering/admission/routing/batching/precision fp8/clock/prewarm/placement/migration) | CORE_ORCHESTRATION_DEFAULT | various no-ops | yes (unchanged) | no |

Enforcement is structural, not advisory: the controller's default `allowed_new_knobs` includes
`gpu_assignment_policy` (core, auto-noop) but **excludes** the optional serving-engine surfaces unless their
`enable_*` flag is set. A disabled optional knob is frozen at its no-op by both the candidate generator and
the hierarchical group search, with the recorded reason *"optional serving-engine integration: default-off
(requires explicit opt-in)"*.

## The ten questions

1. **Which knobs are core fleet orchestration?** `gpu_assignment_policy` (CORE_ORCHESTRATION_AUTO_NOOP) plus
   all pre-existing fleet knobs (capacity, ordering, admission, routing, batching, weight precision fp8,
   clock, prewarm, placement, migration → CORE_ORCHESTRATION_DEFAULT).
2. **Which are optional serving-engine integrations?** `kv_cache_precision_policy` and
   `prefill_decode_policy` (both OPTIONAL_SERVING_ENGINE_INTEGRATION, default-off).
3. **Which are diagnostic-only?** `kv_cache_precision_policy = kv_int4_diagnostic_only` and
   `precision_policy = int4` (no quality model → DIAGNOSTIC_ONLY values, excluded from the headline planner).
4. **Which are default-on?** Core orchestration: `gpu_assignment_policy` (auto-noop) + the pre-existing fleet
   knobs.
5. **Which are default-off?** `kv_cache_precision_policy`, `prefill_decode_policy`, and every diagnostic-only
   value.
6. **Which can affect the default headline?** Only core orchestration knobs that are causally represented in
   the reward path. `gpu_assignment_policy` is core but **auto-noops** (the cost path charges a single
   dominant GPU type), so it cannot move the headline today. The optional integrations are default-off → they
   cannot affect the default headline.
7. **Which require explicit operator opt-in?** `kv_cache_precision_policy` (`enable_kv_cache_precision`) and
   `prefill_decode_policy` (`enable_prefill_decode_disagg`); int4 additionally needs `allow_quality_risk`.
8. **Does Aurelius inspect prompts or alter model outputs?** **No.** No knob reads prompt *contents* or
   changes model outputs. KV precision and PD operate on serving-engine resource layout; GPU assignment routes
   by workload *class*, not prompt text. (KV-int4/int4 carry a *quality-risk channel* as an SLA penalty — a
   model of the RISK of a degraded output, not an inspection or alteration of outputs.)
9. **Does any Batch-1 knob violate the product boundary?** **No (after this corrective PR).** The two
   serving-engine integrations are now default-off and require explicit opt-in; the core knob (GPU assignment)
   is auto-noop where it cannot be causally represented. Before this PR, KV/PD were enabled-by-default
   (regime-gated but on) — that crossed the boundary and is the defect this PR corrects.
10. **What should be merged now?** The product-boundary classification, the default-off gating of the two
    serving-engine integrations, the auto-noop GPU-assignment behaviour, the DistServe-grounded PD model + its
    fixtures, the cap=100,000 correction, and the regime-activation audit. See `BATCH1_MERGE_RECOMMENDATION.md`.
