# dt=60 Roofline MPC Action Diagnostic (Phase 11)

Bounded dt=60 diagnostic on the deterministic 6-hour Azure window with the Mooncake-derived prefix stream
(`scripts/diagnose_roofline_mpc_actions_dt60.py` → `data/external/mpc_controller/roofline_mpc_actions_dt60.json`).
Two halves: a **static action ladder** (force each roofline action on top of the previous to isolate its
causal effect) and a **live MPC-selection arm** (the adaptive planner chooses). Same hybrid cost mode, same
Pareto gate, fair baseline = the pre-roofline bundle (`A`). All effects flow through the roofline physics —
no bonuses. Azure decode is **memory-bandwidth-bound** (the regime where precision/spec help).

## Static roofline ladder (120 eval periods, hybrid cost)

| stack | gp/$ | Δ vs A | SLA viol | TTFT p95 | realized GPU-s | energy (J) | gate (beats/pareto/headline) |
|--|--|--|--|--|--|--|--|
| A pre-roofline (fair baseline) | 103 584 | — | 0.0380 | 0.892 | 4 461 | 0 | — |
| **B + precision fp8** | 143 137 | **+38.2%** | 0.0037 | 0.892 | 3 359 | 2.35e6 | **True / True / True** |
| C + roofline batching | 142 344 | +37.4% | 0.0047 | 0.892 | 3 313 | 2.32e6 | True / True / True |
| **D + spec decode** | **149 646** | **+44.5%** | **0.0000** | 0.892 | 4 084 | 2.86e6 | **True / True / True** |
| E + clock low | 145 110 | +40.1% | 0.0000 | 1.023 | 4 707 | 2.66e6 | True / True / True |
| F + co-location | 143 311 | +38.4% | 0.0000 | 1.061 | 4 884 | 2.76e6 | True / True / True |

**Reading:** precision fp8 alone is a **+38% Pareto-safe** gp/$ gain (SLA *improves* 0.038→0.004, GPU-seconds
*fall* 4461→3359 — faster AND cheaper, the memory-bandwidth-bound signature). Adding speculative decoding
gives the best gp/$ (**+44%**, SLA→0) — a **latency** win (it drives violations to zero) that raises realized
GPU-seconds (4084 > 3313: the draft+verify FLOP tax, so it is not a pure cost win). Low clock and co-location
are **net-negative** versus D on this mixed workload — low clock slows the compute-bound prefill (TTFT
0.892→1.023) and co-location adds pure interference (TTFT→1.061, GPU-s→4884) because there is no
background-work trace. Both stay above A only because precision+spec dominate; the **actionable static lever
is fp8 + spec (stack D)**.

## Live MPC selection (adaptive planner)

The controller (adaptive planner, beam + regret) selected **precision bf16 / spec aggressive / clock low** on
every period: gp/$ **138 387**, SLA 0.0056. Per-decision search: `strategy=beam_search`,
`raw_candidate_count=209 952`, `candidates_evaluated≈183` (no silent cap — the count is reported).

- **vs pre-roofline (A):** beats by **+33.6%**, SLA not worse → **Pareto-safe adaptive win**.
- **vs best static stack (D):** **−7.5%**, headline **not** allowed.

The controller **under-selected precision** (chose bf16, not fp8). Cause — an honest fidelity gap: the MPC
planning rollout (`_rollout_world`) scores candidates **without** the phase/cost model (no `kv_state`), so it
sees precision only through latency (`completion_factor`), not through its cost benefit
(`gpu_seconds_factor`); with the SLA already met in planning, fp8 ties bf16 and the tie-break keeps bf16. The
eval path (with the phase + hybrid cost model) is where fp8's cost win materialises — which the static ladder
makes explicit.

## Required interpretation

1. **Did precision become valuable?** **Yes** — fp8 is the single largest lever (+38% Pareto-safe, static).
   Live selection under-uses it (planning gap, above).
2. **Did roofline batching improve?** Marginally on this workload (C≈B); its real role is the **interaction**
   that shifts the precision/spec optimum (fixtures 11–12) and the reason the planner uses beam search.
3. **Did spec decode become valuable?** **Yes for latency** (D drives SLA→0, best gp/$); it pays a compute tax
   (higher GPU-seconds) → a latency lever, not a clean cost win.
4. **Did clock/power become valuable?** **Net-negative** here — low clock slows the compute-bound prefill;
   its honest value is energy (reported as a diagnostic), not gp/$ on this mixed load.
5. **Did co-location become valuable?** **No** — interference only (no background-work trace); it is frozen
   off in the live planner with a recorded reason. Forced on (F) it correctly hurts.
6. **Were gains Pareto-safe?** Every static stack B–F and the MPC arm beat A with SLA **not worse** (it
   improves). The headline lever is fp8+spec.
7. **Real physics or SLA-shedding?** **Real** — SLA *improves* (0.038→0.000), the opposite of shedding; the
   Pareto gate enforces it (and blocks int4, which trades SLA for cost — never stacked).
8. **Which regimes selected which actions?** Azure decode is memory-bandwidth-bound → the planner pruned to
   the mem-bound options (precision fp8/int4, spec up to aggressive, clock base/low) and selected spec+clock.
9. **Did unified selection beat single-action?** The unified MPC arm beat the pre-roofline baseline but **not**
   the best single+stack (fp8+spec) — because of the precision under-selection. So on this workload the
   strong result is the static fp8+spec stack, not the adaptive controller.
10. **What remains unrealistic?** (a) the planning/eval fidelity gap (wire the phase+cost model into
    `_rollout_world` so the MPC sees precision's cost benefit) — the named **next gap**; (b) no int4 quality
    model; (c) no background-work trace for co-location; (d) no disaggregated prefill/decode pools.

## Claim safety

- **Production-safe:** the **regime classification** (Azure decode memory-bandwidth-bound, trace +
  PUBLIC_SPEC) and the **direction** of every lever (precision/spec help memory-bound; clock trades
  latency↔energy; co-location needs background work).
- **Simulator-inferred:** the **magnitudes** (the +38%/+44% deltas, spec acceptance, DVFS exponent,
  co-location interference). The dt=60 numbers are directional simulator evidence, not production telemetry.
- **Headline:** allowed only because the Pareto gate passes vs the fair baseline under the **defensible
  hybrid** cost mode. The adaptive controller's gp/$ does **not** beat the best static stack — reported
  honestly, with the precision-selection gap as the cause and the next fix.

## Why this advances the action layer

Precision, speculative decoding, and clock are now **live, causal MPC actions** (not diagnostic sweeps),
each two-sided in the correct roofline regime, with co-location and prefill/decode honestly frozen as
modelled-but-not-live. The adaptive planner replaces the fixed 256-cap and **measures** search regret. The
remaining work is fidelity (phase/cost model in planning), not new physics.
