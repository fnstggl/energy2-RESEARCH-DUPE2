# Batch-1 merge recommendation (corrective PR)

## Recommendation: **MERGE** (after the corrective edits in this PR, which are now applied)

PR #125 should **NOT** merge as originally written, and **SHOULD** merge with the corrective edits in this PR
applied. With those edits, it satisfies every decision rule below.

## Decision-rule check

| rule | verdict |
|--|--|
| Defaults preserved AND risky serving-engine knobs default-off? | ✅ KV precision + PD are now `OPTIONAL_SERVING_ENGINE_INTEGRATION`, **default-off** (controller `enable_*` flags, default False). The default benchmark headline is **unchanged** (551,149.66, SLA 0.0). |
| Heterogeneous GPU assignment default-on but auto-noop when homogeneous/not applicable? | ✅ `CORE_ORCHESTRATION_AUTO_NOOP`; deterministic no-op on the single-dominant-GPU cost path (NOT_APPLICABLE); homogeneous fleet provably ties. |
| KV precision / PD can affect the default headline without explicit opt-in? | ✅ **Fixed.** They are default-off (not generated unless `enable_*` is set). Audit confirms the headline is identical default-off vs opt-in. |
| Any knob affects the headline without a realistic causal world-model path? | ✅ None. GPU assignment is auto-noop (no fake fleet); KV/PD reach reward only through roofline service-time/GPU-seconds and are default-off anyway. |
| Any knob wired but impossible to select due to a bug? | ✅ None. The opt-in audit run confirms KV/PD candidates are generated/evaluated when enabled — they are simply not selected because this window's regime does not favour them. |
| Production benchmark simply does not enter the target regime? | ✅ Documented. The Azure trace has **no prompt-token data** (no prefill / large-context KV) and **light load** → it structurally cannot exercise the KV-binding or PD-high-load regimes. Non-selection is a benchmark limitation, not evidence the knobs are low-value. |
| `cap=120` recommended anywhere? | ✅ **Fixed.** The recommended Benchmark v1 cap is **100,000 (uncapped)** — the highest stable cap under the V1 timeout. The cap=120 recommendation is withdrawn/marked obsolete in every doc and script. |

## What changed in this corrective PR (on top of PR #125)

1. **Product-boundary classification** (`actions.product_category`): every surface tagged
   CORE_ORCHESTRATION_DEFAULT / CORE_ORCHESTRATION_AUTO_NOOP / OPTIONAL_SERVING_ENGINE_INTEGRATION /
   DIAGNOSTIC_ONLY / PLANNED_ONLY.
2. **Default-off serving-engine integrations**: `enable_kv_cache_precision` / `enable_prefill_decode_disagg`
   (default False); the controller's default `allowed_new_knobs` excludes them; the generator + hierarchical
   search freeze them at no-op with a recorded reason. GPU assignment stays core / auto-noop.
3. **DistServe-grounded PD model**: KV-bandwidth-sufficiency guard + decode-TPOT-blocking-by-prefill term
   (chunked-prefill residual) + SLO-attainment goodput metric. New DistServe-shaped fixtures (correct split
   helps up to ~12× goodput; wrong split hurts; insufficient KV bandwidth hurts; light/balanced prefers
   shared). The model reproduces a **DistServe-order** win in the genuine regime → **not underpowered**.
4. **Benchmark cap correction**: 120 → **100,000 (uncapped)** everywhere.
5. **Regime-activation audit**: `scripts/run_batch1_regime_audit.py` +
   `data/external/mpc_controller/batch1_knob_regime_activation.json` + the two audit docs.

## Honest bottom line

The Batch-1 knobs are **correct and causally grounded**, but the **production benchmark cannot test them**
(no prompt data, light load, single-dominant-GPU cost path). The right product posture — implemented here — is:
**GPU assignment on/auto-noop; KV precision and PD disaggregation off by default, opt-in for operators whose
serving stack exposes them.** This keeps the fleet orchestrator's defaults honest and unchanged while making
the serving-engine integrations available without ever silently taking control of serving internals.
**MERGE.**
