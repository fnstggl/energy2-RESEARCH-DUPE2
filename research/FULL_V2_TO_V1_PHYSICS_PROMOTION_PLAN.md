# Full V2→V1 Physics Promotion Plan

Plan for promoting the remaining validated production-serving physics from PR #110 / V2 into the canonical V1
world model, making the production-realistic path the default where validation supports it. Classification per
mechanism; the executed decisions + justifications live in `PRODUCTION_DEFAULT_DECISIONS.md` (the authoritative
table). This doc records the *plan* and the inspection that produced it.

## Inspection of current `main` (post PR #117/#119)

`main` already contains far more than the V2 audit branch had when PR #110 was written:
- **Roofline timing** (`roofline_external.py`) + the `timing_model` flag (PR #119, default was legacy).
- **Roofline actions** (`roofline_actions.py`, `roofline.py`): precision/spec/clock as CONNECTED relative
  factors; co-location + prefill/decode allocation as SIMULATED_ONLY/frozen with recorded reasons.
- **Adaptive MPC search** (`search_planner.py` `AdaptiveSearchPlanner`): beam/exhaustive + **regret audit**;
  `candidate_search.py` `CandidateBundleGenerator`. Regime-aware pruning. Already wired in `controller.py`.
- **Persistent ClusterState** (`world_state.py`) with routing/placement/migration/prewarm (PR #99–#107).
- **Single-tier KV** (`kv_cache.StatefulKVCache` GPU-HBM LRU) + per-replica residency (`world_serving.py`).
- **Validation harness** (`world_validation.py`): 45 PASS / 3 SKIPPED, incl. `_roofline_action_checks`.

PR #110's `aurelius/environment/v2/` is **not** on main. So "promotion" = bringing validated V2 *mechanisms*
into V1's canonical architecture as deterministic state transitions / causal timing — not pasting V2 code.

## Classification

| Mechanism | Decision | Notes |
|--|--|--|
| roofline GPU/model base timing | **PROMOTE_NOW_DEFAULT_ON** | flipped default in this PR (was opt-in) |
| GPU/model-aware prefill & decode timing | **PROMOTE_NOW_DEFAULT_ON** | same roofline path |
| arithmetic-intensity regime classification | **PROMOTE_NOW_DEFAULT_ON** | already on; drives candidate pruning |
| timing provenance labels | **PROMOTE_NOW_DEFAULT_ON** | surfaced in metrics |
| legacy scalar timing | **KEEP_AS_LEGACY_ONLY** | explicit regression mode only |
| precision / spec / clock | **PROMOTE_NOW_DEFAULT_ON** (already CONNECTED) | available actions, neutral at default policy |
| roofline-aware batching factor | **PROMOTE_NOW_DEFAULT_ON** (already on) | V1's calibrated factor + saturation |
| continuous-batching token budget | **PROMOTE_NOW_BEHIND_CONFIG** | additive scheduler; V1 has no token-budget loop → follow-up |
| chunked prefill | **PROMOTE_NOW_BEHIND_CONFIG** | needs a per-iteration scheduler in V1 → follow-up |
| prefill/decode pool allocation | **PROMOTE_NOW_BEHIND_CONFIG** (frozen) | live replay has no pools; default-on would model phantom capacity |
| tiered KV cache (HBM/CPU/remote/SSD) | **REJECT (separate migration PR)** | V1 is single-tier; needs new world-state tier model + residency telemetry |
| remote-vs-recompute decision | **REJECT (separate migration PR)** | depends on tiered KV |
| cache lookup / eviction pressure | **PROMOTE_NOW_DEFAULT_ON** (already on) | `world_serving` + `StatefulKVCache` |
| cache transfer overhead | **PROMOTE_NOW_BEHIND_CONFIG** | cross-tier transfer SKIPPED until tiered KV |
| co-location | **PROMOTE_NOW_BEHIND_CONFIG** (frozen) | no background-work trace; off unless `background_work_gpu_seconds` supplied |
| production candidate generation | **PROMOTE_NOW_DEFAULT_ON** (already on) | regime-aware, audited |
| adaptive MPC search + regret | **PROMOTE_NOW_DEFAULT_ON** (already on) | beam/exhaustive + estimated regret |
| production diagnostics + provenance | **PROMOTE_NOW_DEFAULT_ON** | timing_model, regime, queue, KV, regret |

## Why the flag/REJECT decisions are *not* "benchmark preservation"

- **Continuous batching / chunked prefill**: V1 has no per-iteration scheduler. These replace the calibrated
  `BATCH_DECODE_FACTOR` model with a different mechanism — an additive scheduler change requiring its own
  validation, not a flag flip. (Architecture, not benchmark preservation.)
- **Prefill/decode pools**: the live cluster replay has no disaggregated pools (a structural fact). Default-on
  would model a capacity split that does not exist. (Architecture.)
- **Tiered KV / remote-vs-recompute**: V1's `StatefulKVCache` is a single GPU-HBM LRU tier. Multi-tier is new
  persistent state + needs per-replica residency + cross-node bandwidth telemetry that is proprietary in public
  traces. (Architecture + telemetry.)
- **Co-location**: no background-work stream exists in the public traces; default-on credits zero useful work
  and only interference → misleading. (Telemetry.)

## This PR's executed scope

1. **Make roofline the canonical default timing** (the one fully-validated, ready, BENCHMARK_DERIVED mechanism
   not yet default). Keep legacy as explicit regression mode. Update tests/docs.
2. Confirm precision/spec/clock/routing/placement/migration/prewarm/candidate-search/regret are already
   default-on (no change needed) and record them in the decisions table.
3. Keep pools/co-location/tiered-KV/continuous-batching non-default with **specific** technical blockers.
4. Run the all-MPC-knobs vs strongest-SLA-aware-baseline benchmark + the legacy-vs-production comparison.
5. Document results, risks, telemetry gaps, and the recommended next PR (serving-plane roofline + tiered KV).
