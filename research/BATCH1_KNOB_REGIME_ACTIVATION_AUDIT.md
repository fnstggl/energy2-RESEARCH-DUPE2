# Batch-1 knob regime-activation audit (corrective PR)

**Question:** how often does the production replay actually enter the regimes the Batch-1 knobs were designed
for? Runner: `scripts/run_batch1_regime_audit.py`; artifact:
`data/external/mpc_controller/batch1_knob_regime_activation.json`. Read-only; deterministic; no reward/cost/
Pareto change. Scope: `pjm` / price-`expensive` window / 6 periods, cap=100,000 (uncapped), ~60 requests/period.

## Headline data-quality finding (load-bearing)

**The production Azure serving trace records `input_tokens = 0` — the PROMPT side is unobserved.** It carries
arrival times + output-token counts only. Consequences:
- the **prefill phase is invisible** → every period classifies as "decode-heavy" purely because prefill work
  ≈ the fixed TTFT floor (an artifact, not a real skew);
- **large-context KV pressure cannot arise** → KV occupancy ≈ 1.2 % on an 80 GB H100, no HBM pressure;
- load is **light** (~60 req/period, low arrival) → never the high-load regime PD disaggregation needs.

So the production benchmark **structurally cannot enter** the KV-binding or PD-high-load regimes. This is a
benchmark/trace limitation, not a planner bug.

## Per-knob activation

### 1. KV-cache precision
- % periods memory-bandwidth-bound: **100 %** (decode is bandwidth-bound, as expected).
- % periods HBM / high-KV pressure: **0 %**; estimated KV occupancy **1.2 %**.
- KV bytes saved by kv_fp8 / kv_int8: **50 % / 50 %** (real, but there is no HBM pressure to relieve).
- KV eviction pressure: none (occupancy below the working-set threshold every period).
- candidates generated / evaluated / selected: with the optional integration **enabled (opt-in)**, KV
  candidates ARE generated (memory-bound regime) and evaluated, but **not selected** — fp8 *weights* +
  aggressive batching already capture the gp/$, and KV bytes don't bind (no HBM pressure). With the default
  (**default-off**) they are **not generated**.
- why: workload does not need it (no large-context / HBM pressure) — not a wiring failure.
- default-off optional integration: **yes**.

### 2. Prefill/decode disaggregation
- phase mix: prefill_heavy 0 / decode_heavy 6 / balanced 0 — but this "skew" is an **artifact of the missing
  prompt data** (prefill ≈ TTFT floor), not a real prefill/decode imbalance.
- estimated interference relief: ~0; mean phase-pool utilization **0.002** (essentially idle).
- handoff bytes ≈ 12.3 MB; handoff latency ≈ 0.12 ms (NVLink-class) — small, but irrelevant at this load.
- resembles DistServe's high-load skewed regime: **No** (light load, no observed prefill).
- candidates generated / evaluated / selected: opt-in → generated only when divergence+contention is
  detected (rare here); **not selected**. Default-off → not generated.
- why: workload is light and the prefill side is unobserved — not the regime PD targets.
- default-off optional integration: **yes**.

### 3. Heterogeneous GPU assignment
- benchmark fleet heterogeneous at server/replica level? Fleet-wide **yes** (H100/A100 mix), but `gpu_type` is
  **constant per server** — there is no per-replica/request assignment decision.
- cost model charges per selected GPU type or dominant? **Dominant** GPU type per period.
- request→GPU assignment represented in the reward path? **No.**
- applicable / simulated-only / impossible? **NOT_APPLICABLE** (simulated-only / fixture-only).
- routing opportunity across H100/A100/L40S? Exists fleet-wide, but unreachable without an assignment
  mechanism + per-type cost in the reward path.
- default-on auto-noop: **yes** (deterministic no-op on the single-dominant-GPU cost path).

## The fifteen questions

1. **Did each knob activate in the production benchmark?** No (none selected).
2. **Because the workload did not need it?** **Yes** — KV: no HBM pressure; PD: light load + no observed
   prefill; GPU assignment: no per-type cost/assignment in the reward path.
3. **Or because wiring prevented it?** No wiring bug: the opt-in run confirms KV/PD candidates ARE
   generated/evaluated when enabled; they are simply not selected. GPU assignment is intentionally auto-noop.
4. **Were candidates generated?** KV/PD: yes under opt-in (in their regime); no under the default-off product
   boundary. GPU assignment: no (frozen, NOT_APPLICABLE).
5. **Were they evaluated?** Yes (opt-in), through the unchanged reward rollout.
6. **Were they selected?** No.
7. **Did any knob fail to be selected despite improving the causal rollout?** No — in this window KV/PD do not
   improve the rollout (the regime that makes them help is absent). They DO improve it in their target regimes
   (controlled fixtures + the DistServe-shaped fixtures), which this benchmark does not enter.
8. **Did any knob get pruned incorrectly?** No. Every freeze has a recorded, correct reason (compute-bound /
   no-divergence / NOT_APPLICABLE / default-off).
9. **Are controlled fixtures enough evidence the knob works?** Yes for **direction + mechanism**: KV precision
   helps memory-bound/HBM-pressed decode; PD reproduces a DistServe-ORDER goodput win (up to ~12×) in the
   high-load skewed tight-TTFT+TPOT regime and correctly hurts on wrong-split / low-bandwidth and prefers
   shared on light/balanced load. Magnitudes remain SIMULATOR_INFERENCE.
10. **Is the production benchmark missing the regimes where the knob matters?** **Yes — decisively.** It has
    no prompt-token data (no prefill, no large-context KV) and light load. This is the central finding: the
    benchmark cannot test these knobs, so non-selection is **not** evidence they are low-value.
11. **Are any models too conservative?** The PD phase-queue model is conservative in the *moderate* regime
    (interference relief ≈ multiplexing+handoff cost → ~1×); it only shows the large DistServe win under
    genuine high-load skew. That is defensible, not a deficiency.
12. **Are any models too optimistic?** The PD SLO-attainment ratio is bimodal and can read high (~12×) when
    the shared pool is tipped over a tight SLO — at/above DistServe's reported 4.48–12.6× band. We treat the
    magnitude as a SIMULATOR_INFERENCE sanity check, never a reward bonus.
13. **Which knobs are safe to merge now?** All three, with the product-boundary defaults: GPU assignment
    (core, auto-noop), KV precision + PD (optional, default-off).
14. **Which should remain optional / default-off?** `kv_cache_precision_policy` and `prefill_decode_policy`
    (serving-engine internals); int4 values diagnostic-only.
15. **Should PR #125 merge as-is, after edits, or not merge?** **MERGE AFTER (these) FIXES** — applied in this
    corrective PR. See `BATCH1_MERGE_RECOMMENDATION.md`.
