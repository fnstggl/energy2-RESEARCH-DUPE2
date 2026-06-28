# Serving Roofline Closure Gap Audit (PR #108, Phase 0)

State of the serving model after PR #107, before adding the roofline/disaggregation closure.

## Answers

1. **Separate prefill/decode accounting?** Yes (PR #107): `prefill_decode.compute_phase_serving` returns
   `prefill_gpu_seconds`, `decode_gpu_seconds`, `prefill_work_s`, `decode_work_s` per request.
2. **Separate prefill/decode CAPACITY pools?** **No.** The two work terms are summed into one `service_s`
   that feeds **one** shared cluster queue (`run_unified_replay`). Prefill and decode share the replica
   pool; there is no prefill-pool / decode-pool split, no KV handoff, no per-pool queue. → Phase 3 gap.
3. **Does "decode-bound" mean decode-PHASE-bound or memory-bandwidth-bound?** **Decode-phase-bound only.**
   `PhaseResult.summary` labels `decode_bound` when `decode_gpu_seconds > 2·prefill_gpu_seconds` — a
   **phase-TIME** comparison. It says nothing about arithmetic intensity / the roofline ridge point. The
   two are conflated by the name. → Phase 2 rename + Phase 4 roofline.
4. **compute-bound vs memory-bound measured?** **Absent.** No FLOPs, no memory bandwidth, no arithmetic
   intensity anywhere. → Phase 4 (`roofline.py`).
5. **Does batching change arithmetic intensity?** **No.** `BATCH_DECODE_FACTOR` is a fixed per-policy
   decode multiplier (0.82/0.92/1.0) + a saturation tail — not an AI-vs-ridge computation. → Phase 5.
6. **Does KV reuse reduce only prefill?** **Yes** (PR #107 fix): `prefill_tokens_remaining = prompt −
   prefix_hit`; decode is `out·TPOT`, KV-insensitive.
7. **Does decode occupancy dominate Azure+Mooncake numerically?** The #107 dt=60 run showed realized
   GPU-seconds ≈ 14 773, overwhelmingly decode (prefill saved 107k tokens cut only 16 GPU-s). So Azure is
   decode-**phase**-bound by GPU-seconds. **Whether it is also memory-bandwidth-bound is unmeasured** —
   this PR must measure it (the central question).
8. **power / clock / speculative decoding / co-location modeled?** **Absent** (ABSENT tier). → Phases
   6/7/8 add **diagnostic-only** models (no live action surface evaluates them causally).
9. **Action surfaces that already exist** (the connected bundle): routing, capacity_multiplier, batching,
   prewarm, placement, migration, admission, ordering. **No** prefill/decode-allocation, spec-decode,
   clock, or co-location action.
10. **Which stay DIAGNOSTIC (not controller actions)?** prefill/decode allocation, speculative decoding,
    clock/DVFS, co-location — none has a connected action surface the simulator can evaluate end-to-end,
    so per the hard rule they are **diagnostic regime models + sweeps**, never live reward channels. A
    disaggregation **static sweep** is a diagnostic, not a learned win.

## What this PR builds

| phase | gap | this PR |
|--|--|--|
| 2 | rename decode_bound → decode_phase_bound; add roofline regime | **implemented** |
| 4 | `roofline.py`: arithmetic intensity, ridge point, compute/memory-bound, tokens/s | **implemented** |
| 3 | prefill/decode capacity disaggregation (shared / disaggregated_static / sweep) | **implemented** (diagnostic; no new live knob) |
| 5 | roofline-aware batching | **implemented** |
| 6/7/8 | speculative decoding / clock / co-location | **implemented diagnostic-only** with regime logic + fixtures |
| 9 | unified roofline-economic recommender | **implemented** (diagnostic, claim-safety-labelled) |
| 11 | dt=60 bottleneck proof + disaggregation sweep | **run** |

**Hard rule honored:** the new infra mechanisms affect nothing in the live reward path (no action
surface); they are numeric classifiers + sweeps with claim-safety labels. The only live change is the
phase-bottleneck rename + roofline regime added to existing diagnostics. No reward bonuses anywhere.
