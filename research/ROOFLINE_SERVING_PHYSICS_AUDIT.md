# Roofline / Serving Physics Audit (PR #109, Phase 1)

Public serving + roofline mechanisms, each classified and mapped to the conservative model in
`roofline.py`. **The key distinction this PR enforces:** *decode-phase-bound* (decode dominates
time/work) ≠ *memory-bandwidth-bound* (arithmetic intensity below the GPU ridge point). They are
computed separately and never conflated. No UNKNOWN.

## Mechanisms + fidelity

| mechanism | model | fidelity |
|--|--|--|
| GPU peak FLOPs / HBM bandwidth (A100 312 TF/2.0 TB/s; H100 989 TF/3.35 TB/s; …) | `GPU_SPECS` | PUBLIC_SPEC (vendor sheets) |
| roofline: tokens/s = min(compute, bandwidth) | `_tokens_per_s` | PUBLIC_PAPER (Williams roofline) |
| ridge point = peak_flops / mem_bw | `_ridge_point` | PUBLIC_SPEC |
| arithmetic intensity (prefill vs decode, batch, context, precision) | `arithmetic_intensity` | PUBLIC_PAPER (decode is bandwidth-bound: vLLM/DistServe) |
| model FLOPs/token, KV bytes/token | `MODEL_SPECS` (llama-8b-gqa arch) | BENCHMARK_DERIVED |
| continuous batching → AI ↑, occupancy | batch in AI + `_tokens_per_s` | PUBLIC_PAPER (Orca/vLLM) |
| prefill/decode disaggregation | `serving_mode`, `prefill_decode_ratio`, handoff | PUBLIC_PAPER (DistServe/Splitwise) |
| chunked prefill (Sarathi) | — | ABSENT (not modelled; noted) |
| speculative decoding (draft/verify FLOPs, acceptance, serial reduction) | `spec_decode_*` | PUBLIC_PAPER + SIMULATOR_INFERENCE (regime logic) |
| clock / DVFS (power ~ clock^2.4; compute scales, bandwidth flat) | `clock_factor`, `_power_w` | PUBLIC_PAPER (DVFS) + SIMULATOR_INFERENCE |
| precision (fp16/fp8/int4 weight & KV bytes) | `PRECISION_BYTES` | BENCHMARK_DERIVED |
| co-location (idle-SM compute work in memory-bound regime, interference) | `colocation_frac` | SIMULATOR_INFERENCE |
| Azure prompt/output distribution | the diagnostic's workload | TRACE_DERIVED |
| Mooncake prefix reuse → prefill_hit_frac | `Workload.prefix_hit_frac` (PR #106) | TRACE_DERIVED_REUSE_MODEL |
| Alibaba v2026 GPU-type mix | the diagnostic's `gpu` | TRACE_DERIVED |
| MLPerf throughput sanity | the `_tokens_per_s` bands | PUBLIC_SPEC (sanity only) |

## The decode-phase-bound vs memory-bandwidth-bound distinction (numeric)

- **decode-phase-bound:** `decode_gpu_seconds / (prefill+decode) > 0.66` — a *time/work* statement.
- **memory-bandwidth-bound:** `arithmetic_intensity(decode) < ridge_point` — a *roofline* statement.
- A workload can be (and Azure decode usually is) **both**: decode dominates time **and** decode AI ≪
  ridge (≈1 vs ≈153 on A100 at batch 1). The `roofline.py` model reports both labels per period; the
  validation suite asserts they are computed independently (`phase_bound_distinct_from_roofline_regime`).

## Action surfaces (the hard rule)

Existing connected actions: routing, capacity_multiplier, **batching**, prewarm, placement, migration,
admission, ordering. **Only batching maps to a roofline mechanism with a live action surface.** Per the
hard rule and the user's clarification, the other mechanisms (prefill/decode allocation, speculative
decoding, clock/DVFS, precision, co-location) are **fully simulated** in `serving_point` and **swept
diagnostically** — "diagnostic" means fully-simulated-but-not-controller-selected, **not** incomplete.
The MPC may only select a mechanism whose action surface already evaluates it end-to-end. Adding a fake
controller knob for the others would violate the contract, so they remain counterfactual sweeps with
explicit claim-safety labels.

## Not modelled (ABSENT, with reason)
- chunked prefill (Sarathi): a prefill-scheduling refinement; omitting it is conservative (we model
  prefill as bulk work). Telemetry to unblock: per-chunk prefill latency.
- exact draft-model architecture for speculative decoding: we use a FLOPs-fraction band, not a specific
  draft model — labelled SIMULATOR_INFERENCE; a specific draft model would need its own benchmark.
