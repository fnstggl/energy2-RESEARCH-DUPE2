# dt=60 Serving Roofline Closure Diagnostic (PR #108, Phase 11)

Numerically proves the Azure+Mooncake bottleneck on the real prompt/output distribution and reports
every mechanism's sensitivity curve. `data/external/mpc_controller/serving_roofline_dt60.json`
(`scripts/diagnose_serving_roofline_dt60.py`). Fully simulated; only batching is a live MPC action.

## Numeric bottleneck proof (the headline)

Azure dt=60, 6 427 requests, GPU=H100: **prompt median 857 / p95 4 625**, **output median 43 / p95 437**.

| quantity | value | meaning |
|--|--|--|
| `decode_gpu_sec_share` | **0.51** | → **mixed_phase_bound** (NOT decode-phase-bound) |
| decode arithmetic intensity | **14.35** | FLOP/byte at batch 16 |
| ridge point (H100) | **295.22** | peak_flops / mem_bw |
| decode roofline regime | **memory_bandwidth_bound** | AI 14.4 ≪ ridge 295 |

**Verdict: Azure is `mixed_phase_bound` AND its decode is `memory_bandwidth_bound`.** This *corrects*
PR #107's loose "decode-bound": because Azure prompts are long (median 857) and outputs short (median
43), **prefill is ~half the GPU-seconds** — the workload is phase-MIXED, not decode-phase-dominated. The
roofline fact that matters is that decode is **bandwidth-bound** (AI ≈ 14 ≪ ridge 295), so the levers
that help are the ones that raise effective bandwidth-per-token (batching, lower precision) — not raw
compute.

## Mechanism sensitivity curves (help region on Azure's workload)

| mechanism | action surface | completion helps at | cost helps at | physically correct? |
|--|--|--|--|--|
| **batching** | **live MPC action** | 2…128 | 2…128 | ✓ memory-bound → amortise weight loads → faster + cheaper |
| prefill/decode allocation | diagnostic sweep | (none) | (none) | ✓ mixed load: 0.5 split ≈ shared, handoff overhead → no win |
| speculative decoding | diagnostic sweep | accept ≥ 0.3 | (none) | ✓ memory-bound spare compute → faster latency, but MORE FLOPs → not a cost win |
| clock / DVFS | diagnostic sweep | 1.15 (upclock) | 1.15 | small (mixed prefill has some compute sensitivity) |
| precision | diagnostic sweep | fp8, int4 | fp8, int4 | ✓ memory-bound → fewer weight/KV bytes → faster + cheaper |
| co-location | diagnostic sweep | (none) | (none) | ✓ helps utilisation/throughput, not foreground completion latency |

Each curve is the same `serving_point` physics swept across the operating range; the help/hurt/neutral
regions are computed vs the neutral baseline. All are SIMULATOR_INFERENCE except the bottleneck (trace +
PUBLIC_SPEC).

## Required interpretation

1. **Was Azure decode-phase-bound?** **No** — `mixed_phase_bound` (decode share 0.51). #107's claim was
   imprecise; the long prompts make prefill ~half the work.
2. **Memory-bound, compute-bound, or mixed?** Decode is **memory-bandwidth-bound** (AI 14.4 ≪ ridge 295);
   prefill at batch 16 is also below ridge → bandwidth-bound. Compute-bound only at very high batch.
3. **Did disaggregation improve TTFT?** Not on the mixed median workload (0.5 split ≈ shared + handoff);
   it helps **prefill-heavy** workloads (fixtures: right allocation → lower TTFT).
4. **Did disaggregation improve completion?** No (same reason).
5. **Did disaggregation improve gp/$?** No on this workload; the sweep shows no help region.
6. **Did roofline-aware batching change the optimum?** Yes — batching is the live lever and helps both
   completion and cost across the range (memory-bound amortisation), which is the actionable result.
7. **Spec decode help only in the expected regime?** Yes — helps latency in memory-bound decode (Azure),
   hurts in compute-bound (fixture); never a cost win (extra FLOPs).
8. **Downclock only in the expected regime?** Diagnostic: downclock saves energy in memory-bound decode
   (fixture); on Azure's mixed load a slight upclock helps completion (prefill has compute sensitivity).
9. **Co-location only in the expected regime?** Yes — useful only with SM headroom (memory-bound), and it
   adds memory pressure (never improves foreground completion) — correctly shows no help here.
10. **Pareto-safe?** The only **live** lever is batching, already in the MPC's bundle; its roofline
    behaviour (help in memory-bound) is now physically grounded. The diagnostic mechanisms are
    counterfactual sweeps, not live gp/$ claims.
11. **Production-safe vs simulator-inferred?** **Production-safe:** the *bottleneck classification*
    (trace-derived workload + PUBLIC_SPEC roofline) — Azure decode is memory-bandwidth-bound, mixed-phase.
    **Simulator-inferred:** the mechanism sensitivity magnitudes (spec-decode / clock / precision /
    co-location deltas). **None is a production gp/$ claim.**

## Why this closes the serving-physics domain

The serving model now (a) separates prefill/decode **work** (PR #107) and **capacity** (this PR);
(b) classifies **compute-bound vs memory-bandwidth-bound from arithmetic intensity vs the ridge**, no
longer conflated with decode-phase-bound; (c) gives every infra mechanism a fully-simulated sensitivity
curve in the correct regime; (d) keeps the MPC restricted to existing action surfaces (batching). The
numeric proof that Azure decode is memory-bandwidth-bound explains #107's marginal monetisation
physically and tells the operator the actionable lever is **batching + precision**, not compute.
