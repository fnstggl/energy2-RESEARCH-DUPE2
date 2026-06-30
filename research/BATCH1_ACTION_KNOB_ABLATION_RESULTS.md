# Batch-1 action-knob ablation ladder (Phase 7)

Runs the **frozen Benchmark v1 cap = 120** (Phase 0) on `pjm` / the price-`expensive` window / 3 decisions
through the **unchanged reward path** with `hierarchical_search`, under eight ablation masks plus the
comparison arms. Runner: `scripts/run_batch1_ablation.py`; artifact:
`research/results/batch1_action_knob_ablation.json`. The mask is threaded through the planner via
`controller.allowed_new_knobs` → `PlannerRegimeState.allowed_new_knobs` (a disabled knob is frozen at its
no-op with a recorded reason, in both the candidate generator and the hierarchical group search).

## Ablation sequence (vs the baseline ablation arm AND vs production_scheduler)

| arm | new knobs enabled | gp/$ | Δ vs baseline (abs / %) | Δ vs production (abs / %) | SLA viol | Pareto vs prod | new knobs **selected** |
|--|--|--:|--|--|:--:|:--:|--|
| 1 baseline | — | 551,149.66 | — | +370,212.7 / **+204.6 %** | 0.0 | ✅ | none |
| 2 + KV precision | kv | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none (kept `inherit`) |
| 3 + GPU assignment | gpu | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none (NOT_APPLICABLE, frozen) |
| 4 + PD disaggregation | pd | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none (kept `shared`) |
| 5 + KV + GPU | kv,gpu | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none |
| 6 + KV + PD | kv,pd | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none |
| 7 + GPU + PD | gpu,pd | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none |
| 8 + all three | kv,pd,gpu | 551,149.66 | +0 / 0.0 % | +370,212.7 / +204.6 % | 0.0 | ✅ | none |

## Comparators (same window, same reward path)

| arm | gp/$ | note |
|--|--:|--|
| production_scheduler (HEADLINE bar) | 180,936.94 | reactive serving-stack heuristic, no economic arbitrage |
| sla_aware (honest bar) | 167,357.66 | SRPT-conformal + backlog autoscale |
| aurelius_mpc_current_default (prior Aurelius default) | 397,033.16 | physics-guided bounded beam |
| oracle_diagnostic (NON-deployable ceiling) | 602,876.88 | strongest search + exact future workload |

## What this honestly shows

**On the production Azure benchmark, the three new knobs are NEUTRAL — every ablation arm is identical
(551,149.66) and none of the new knobs is selected** (KV stays `inherit_weight_precision`, PD stays `shared`,
GPU assignment is frozen NOT_APPLICABLE). This is the correct, non-result-chasing outcome:

1. **They are regime-gated.** This window's representative Azure workload is not in the memory-bandwidth-bound
   / HBM-pressed regime that would make KV precision worth selecting, nor in a clearly prefill/decode-skewed +
   contended regime that would make a PD split worth its handoff. The candidate generator freezes them with
   recorded reasons (`kv_precision_frozen`, `prefill_decode_frozen`, `gpu_assignment_frozen`).
2. **They are Pareto-evaluated.** Even where KV precision *is* generated, the planner only keeps it if it
   raises gp/$ without worsening SLA; on this window the existing levers (fp8 **weights** + aggressive
   batching + high clock) already capture the available gp/$, so KV *cache* precision adds nothing on top.
3. **GPU assignment cannot fake a gain** — the production cost path is single-dominant-GPU, so the knob is
   frozen off (NOT_APPLICABLE) and contributes exactly 0.

**Per-knob contribution on the benchmark: KV precision = 0, PD disaggregation = 0, GPU assignment = 0
(NOT_APPLICABLE).** The combined arm is therefore **not** claimed as a benchmark improvement over the baseline
ablation arm — there is nothing to attribute. The **+204.6 %** headline vs production_scheduler is carried
entirely by the *pre-existing* knobs and is unchanged by this PR (defaults preserved bit-for-bit).

The new knobs' demonstrated value is in their **target regimes**, measured in the controlled fixtures
(`BATCH1_CONTROLLED_FIXTURES.md`): KV precision +19.5 % on memory-bound decode; PD splits recover goodput on
skewed contended loads; heterogeneous assignment +24–62 % on a real GPU mix. The Azure benchmark window simply
does not enter those regimes — so we report the knobs as **directional / regime-conditional**, enabled but
not headline-driving on Benchmark v1.

## Search diagnostics

The baseline arm generates ~80 candidates and evaluates ~81 distinct bundles per decision (the new knobs add
candidates only in their regimes; here they are frozen, so width is unchanged). No arm timed out; each ran in
~1–3 s. Determinism: re-running yields identical gp/$.
