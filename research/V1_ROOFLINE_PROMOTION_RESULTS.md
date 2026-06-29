# V1 Roofline Promotion — Results

Selectively promoted the **roofline-resolved GPU/model-aware base timing** from PR #110's V2 world model into
the canonical V1 model, behind a `timing_model` flag (default `legacy_scalar` — bit-for-bit unchanged). See
`V2_TO_V1_PROMOTION_PLAN.md` for the full mechanism-by-mechanism classification.

## What was promoted (PROMOTE_NOW)

- A `timing_model` selector on `prefill_decode.compute_phase_serving` — `legacy_scalar` (default) keeps the
  fleet-wide scalar `PREFILL_S_PER_TOKEN`/`TPOT_S`; `roofline` resolves the per-request **base** prefill/decode
  rate from `roofline_external` (already on `main`) per (GPU type, model arch, prompt, context).
- `resolve_serving_rates(gpu_type, model, prompt, out)` — the conservative GPU/model resolver with explicit
  provenance (BENCHMARK_DERIVED public specs + arch; unknown GPU/model → documented default, never a guess).
- `env_timing_model()` + a `kv_state["timing_model"]`/`["gpu_type"]` config pass-through in
  `world_simulator.simulate_period`, with the fleet's **dominant replica GPU** auto-resolved
  (`_dominant_gpu_type`) as the conservative default. Default stays `legacy_scalar`.
- Provenance labels (`TIMING_PROVENANCE`) and a `timing_model` field surfaced in `PhaseResult.summary()`.

## What was intentionally NOT promoted

- **precision / speculative decoding / clock** — already present in V1 on `main` as *relative* factors
  (`roofline_actions.py`); the promoted base rate composes with them unchanged. Nothing to add.
- **tiered KV cache, remote-vs-recompute, prefill/decode pools, continuous-batching token budget, chunked
  prefill, co-location, adaptive MPC search, V2 candidate generator, full CanonicalWorldStateV2/
  WorldSimulatorV2** — KEEP_V2_ONLY: invasive, or duplicative of existing V1 mechanisms, or Pareto-neutral on
  the dt60 window (see PR #110 diagnostic). Promoting them now would risk V1 stability for little gain.
- **V2 validation suite, diagnostic scripts, external cross-checks** — VALIDATION_ONLY (reference baselines).

## Why roofline timing belongs in V1

V1's scalar `TPOT_S=0.020` is GPU-**blind** and behaves like an **L40S-class** decode constant
(`ROOFLINE_REUSE_DECISION.md`). Applied to a fast fleet it overstates decode time ~4× and fabricates SLA
violations that the hardware does not have. The promoted roofline prices each GPU correctly while leaving the
default untouched.

## Before/after numbers (`scripts/compare_v1_legacy_vs_v1_roofline.py`, SLA 8 s, same workload)

| GPU | timing | SLA viol | completion p95 (s) | realized GPU-s | gp/$ |
|--|--|--|--|--|--|
| H100 | legacy_scalar | 0.700 | 7.254 | 868.3 | 65,061 |
| H100 | **roofline** | **0.092** | **2.003** | **240.1** | **712,521** |
| A100 | legacy_scalar | 0.700 | 7.254 | 868.3 | 65,061 |
| A100 | **roofline** | **0.417** | **3.202** | **383.4** | **289,081** |
| L40S | legacy_scalar | 0.700 | 7.254 | 868.3 | 65,061 |
| L40S | **roofline** | 0.725 | 7.335 | 879.0 | 59,525 |

Three things this proves:
1. **Legacy is GPU-blind** — identical numbers across H100/A100/L40S (a fleet-wide constant).
2. **The scalar is L40S-class** — on L40S the roofline ≈ legacy (p95 7.33 vs 7.25; `phantom_sla_on_legacy =
   False`). The constant was calibrated to an L40S-class GPU.
3. **Phantom SLA on fast GPUs** — on H100 the legacy scalar reports 0.700 violations where the roofline (which
   prices H100 decode correctly) reports 0.092; on A100, 0.700 → 0.417. The legacy constant was inventing SLA
   failures on hardware fast enough to meet the SLA.

The gp/$ swing on H100/A100 is a **realism correction**, not a manufactured win: it comes entirely from
removing phantom SLA violations (more requests are correctly counted as SLA-safe) and from realized GPU-seconds
falling to their true level — both through the existing cost/goodput channels. There is **no** direct reward
bonus; on L40S the roofline is (correctly) slightly *worse* than the rounded scalar, so the change is not a
one-way ratchet.

## Should V1's default switch to roofline now?

**No — keep `legacy_scalar` as the default in this PR.** Reasons: (a) public benchmark semantics must not
change silently (hard requirement); (b) the roofline needs a per-period fleet GPU/model resolution wired through
the trace-ingestion path before it can be the headline default (right now it uses the dominant-replica
heuristic); (c) the correct next step is to validate the roofline default against the full Azure+Mooncake+v2026
pipeline. The flag makes roofline opt-in (config `kv_state["timing_model"]="roofline"` or
`AURELIUS_TIMING_MODEL=roofline`) so it can be exercised and compared without disturbing existing results.

## Risks

- **Conservative GPU fallback**: unknown v2026 GPU types (e.g. `XPU-*`) resolve to the roofline default GPU
  inside `resolve_serving_rates`. This only matters in roofline mode (opt-in) and is labelled; a per-type
  resolver is a follow-up.
- **Model assumption**: the resolver defaults to `llama-8b-gqa` unless a model is supplied; real fleets are
  multi-model. Again opt-in and labelled.
- **Per-request resolver cost**: `resolve_serving_rates` is called per request in roofline mode (cheap
  arithmetic); for very large periods a per-(GPU,model,bucket) cache is an easy optimization if needed.

## Remaining gaps / recommended next promotion PR

1. Wire a per-period **fleet GPU/model mix** (not just the dominant type) into the roofline base so a
   heterogeneous fleet is priced per-replica, then re-evaluate switching the default to roofline.
2. Add a profiled-MFU calibration (Vidur-style) as a validation baseline to tighten the roofline floor.
3. Only after (1)+(2): consider promoting the V2 continuous-batching token-budget model behind its own flag.

## Hygiene

`legacy_scalar` is bit-for-bit unchanged (verified: `test_v1_roofline_timing.py::
test_legacy_scalar_is_default_and_bit_for_bit`, `..._benchmark_stability`, and all 38 prior
prefill/decode/world tests pass). ruff clean; 13 new V1 tests pass; deterministic; no network; no GPU; no new
runtime dependency (reuses on-repo `roofline_external`). V2 is untouched and still on its own branch.
