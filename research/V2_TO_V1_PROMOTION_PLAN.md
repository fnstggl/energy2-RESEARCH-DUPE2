# V2 → V1 Promotion Plan

Selectively promote the **validated** serving mechanisms from PR #110's V2 world model into the canonical V1
model on `main`, without replacing V1, deleting V2, renaming V1→V2, or changing public benchmark semantics
unless explicitly guarded. Objective: **make V1 less wrong while preserving V1's stability.**

## What `main` already has (inspected, not assumed)

`main` (post PR #117) already contains a roofline *action* layer independent of PR #110:
- `aurelius/environment/roofline_external.py` — the ported FLOP/bandwidth roofline (GPUSpec/ModelArch tables,
  `prefill_estimate`/`decode_estimate`/`roofline_analyze`). **Present and reusable.**
- `aurelius/environment/roofline.py` + `roofline_actions.py` — precision/spec/clock/co-location/PD-allocation
  as **relative ratio factors** (`roofline_action_factors`) that modulate `compute_phase_serving`. The doc
  `ROOFLINE_MPC_ACTION_CALIBRATION.md` states the live model **keeps its scalar absolute level
  (`TPOT_S`/`PREFILL_S_PER_TOKEN`)**; roofline supplies only the *delta*.
- `compute_phase_serving(...)` already accepts `roofline_factors` (precision/spec/clock), bit-for-bit at neutral.

`main` does **not** contain `aurelius/environment/v2/` — **PR #110 is not merged to main.** So "promotion" here
means bringing the *validated mechanism* (roofline-resolved base timing, proven in PR #110's
`RooflineServingModelV2` + `CANONICAL_WORLD_STATE_V2_COMPARISON.md`) into V1's actual code, reusing the
`roofline_external` module that is already on main.

## The precise gap

V1's **absolute base** prefill/decode rate is still the scalar `PREFILL_S_PER_TOKEN=0.00015` /
`TPOT_S=0.020`. Per `ROOFLINE_REUSE_DECISION.md`, `TPOT_S=0.020` is an **L40S-class** decode constant; applied
to an **H100** fleet it overstates decode ~4× and can produce **phantom SLA violations** (PR #110
`CANONICAL_WORLD_STATE_V2_COMPARISON.md`: legacy scalar → 99.4 % violations; roofline → 0 %). The roofline
*factors* on main do not fix this — they are relative deltas around the wrong absolute level. **The base rate
must become GPU/model-aware.** That is the one high-value, low-risk promotion.

## Classification of every V2 mechanism

| mechanism | decision | rationale |
|--|--|--|
| **roofline timing by GPU/model** | **PROMOTE_NOW** | the core realism correction; reuses `roofline_external` already on main; deterministic, no deps |
| **legacy scalar fallback** | **PROMOTE_NOW** | default, must reproduce V1 bit-for-bit (benchmark compatibility) |
| **GPU/model-aware timing floor** | **PROMOTE_NOW** | resolved per (GPU, model, prompt, context) from public specs (BENCHMARK_DERIVED) |
| **provenance labels for timing params** | **PROMOTE_NOW** | every promoted rate carries a fidelity tier |
| **validation that H100 ≠ L40S-priced** | **PROMOTE_NOW** | a focused test + the comparison script flag phantom SLA on fast GPUs |
| **precision mode** | **PROMOTE_BEHIND_FLAG (already in V1)** | already present on main as a relative factor (`roofline_actions`); this PR adds the GPU/model **base** it modulates — composes, no change to the factor path |
| **speculative decoding mode** | **PROMOTE_BEHIND_FLAG (already in V1)** | same — already a relative factor on main; unchanged |
| **clock / power state** | **PROMOTE_BEHIND_FLAG (already in V1)** | same — already a relative factor on main; unchanged |
| **continuous batching token budget** | **KEEP_V2_ONLY** | V1 already has `BATCH_DECODE_FACTOR` + saturation tail; the token-budget model is a different mechanism — not promoting now (avoid changing batching semantics) |
| **chunked prefill behaviour** | **KEEP_V2_ONLY** | V1 has no chunked-prefill path; invasive; defer |
| **tiered KV cache (HBM/CPU/REMOTE/SSD)** | **KEEP_V2_ONLY** | V1 has a single per-replica LRU; the 4-tier cascade is invasive and Pareto-neutral on the dt60 window — defer |
| **remote-vs-recompute KV decision** | **KEEP_V2_ONLY** | depends on the tiered cache; defer with it |
| **prefill/decode pool scheduler** | **KEEP_V2_ONLY** | the live V1 replay has **no disaggregated pools** (main's roofline doc marks PD-allocation SIMULATED_ONLY/frozen); promoting would require new cluster state — defer |
| **co-location** | **KEEP_V2_ONLY** | no real background-work trace; main already keeps co-location frozen/SIMULATED_ONLY; no imaginary goodput |
| **adaptive MPC search** | **KEEP_V2_ONLY** | main's planner already measures search regret (`search_regret_is_measured`); the beam/exhaustive planner is not needed for this timing promotion — defer |
| **V2 candidate generator** | **KEEP_V2_ONLY** | main already has regime-aware pruning; defer |
| **V2 validation suite** | **VALIDATION_ONLY** | lives in PR #110's branch; referenced as a baseline, not promoted to V1 runtime |
| **V2-vs-V1 diagnostic scripts** | **VALIDATION_ONLY** | this PR adds `scripts/compare_v1_legacy_vs_v1_roofline.py` (the V1-internal analogue) |
| **external simulator cross-checks** | **VALIDATION_ONLY** | reference only; no runtime dependency |
| **full CanonicalWorldStateV2 / WorldSimulatorV2** | **KEEP_V2_ONLY** | wholesale world-model swap is out of scope (hard rule: don't replace V1) |

## What this PR implements (smallest safe change, biggest realism gain)

1. `compute_phase_serving(..., timing_model="legacy_scalar", gpu_type=..., model=...)` — a `timing_model`
   selector. `legacy_scalar` (default) reproduces V1 **bit-for-bit**; `roofline` resolves the per-request
   **base** `prefill_s_per_token` and `tpot_s` from `roofline_external` (GPU/model/prompt/context-aware). The
   existing `roofline_factors` (precision/spec/clock) compose unchanged on top.
2. `resolve_serving_rates(gpu_type, model, prompt, out)` — the conservative GPU/model resolver with explicit
   provenance (BENCHMARK_DERIVED public specs; lazy import avoids the `prefill_decode ↔ roofline_external` cycle).
3. `env_timing_model()` + a `kv_state["timing_model"]` config pass-through in `world_simulator.simulate_period`
   (default `legacy_scalar`; the world's dominant replica `gpu_type` auto-resolves the conservative default) —
   the **config/env/argument** selection path. Benchmarks that don't opt in are untouched.
4. Tests (`tests/test_v1_roofline_timing.py`), the comparison script, and results doc
   (`V1_ROOFLINE_PROMOTION_RESULTS.md`).

## Guard rails honored

- V1 not replaced; V2 not deleted/renamed; default stays `legacy_scalar` (no public-benchmark change unless a
  caller explicitly opts in).
- No reward bonus / action scalar / roofline bonus — the roofline base rate changes only TTFT / completion /
  queue / GPU-seconds / SLA / cost, exactly like the scalar it replaces.
- Deterministic, clone-safe, no network, no GPU, no heavy/external runtime dependency (reuses on-repo
  `roofline_external`).
