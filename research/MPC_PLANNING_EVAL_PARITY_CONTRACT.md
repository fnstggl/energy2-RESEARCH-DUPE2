# MPC Planning/Eval Parity Contract (Phase 1)

What "planning agrees with evaluation" means, as testable metrics with thresholds. Planning need not be
bit-identical to evaluation (it may be faster), but it must **preserve action ranking** well enough that the
search regret is bounded and **measured** — never hidden. UNKNOWN is forbidden: every gap is reported with a
number.

## Direction agreement (the core contract)

For any candidate bundle, planning and full evaluation must agree on the **sign** of the change (vs the
neutral bundle) for each quantity below. Direction agreement — not magnitude — is the contract, because the
MPC only needs the *ranking* right:

TTFT · completion latency · prefill GPU-seconds · decode GPU-seconds · realized GPU-seconds · provisioned
GPU-seconds · energy · cost · SLA risk · quality risk · gp/$ ranking on the same horizon/window.

## Parity metrics (computed in `MPC_PLANNING_EVAL_PARITY_RESULTS.md`)

Over a candidate set scored by both planning and full evaluation:

| metric | definition | threshold |
|--|--|--|
| **rank correlation** | Spearman ρ(planning score, eval score) | ≥ 0.7 on sampled real windows |
| **top-1 agreement** | planning's argmax == eval's argmax | required on controlled fixtures; reported on real windows |
| **top-k containment** | planning top-k ⊇ eval top-1 (k=3) | ≥ 0.8 on real windows |
| **absolute reward error** | mean \|plan_score − eval_score\| | reported (no hard threshold; informational) |
| **relative gp/$ error** | mean \|Δgp/$\| / eval gp/$ | reported |
| **SLA classification agreement** | sign(plan SLA change) == sign(eval SLA change) | required on fixtures |
| **Pareto gate agreement** | plan and eval agree on headline_allowed for the chosen bundle | required |
| **search regret** | `exhaustive_best_reward − chosen_reward` (full-eval scored) | near-zero on fixtures; bounded + reported on windows |

## Thresholds by setting

- **Controlled fixtures** (constructed memory-bound / compute-bound / coupled): **near-zero regret** and
  **top-1 agreement** expected — planning must rank FP8 above BF16 in the memory-bound fixture (the PR #111
  miss). A fixture that fails is a bug, not a tuning opportunity.
- **Small real windows** (dt=60, exhaustive feasible on a reduced surface set): regret **reported and
  bounded**; rank correlation ≥ 0.7.
- **Full dt=60**: regret **sampled** on representative decisions (low/high load, memory/mixed, SLA
  tight/loose, high/low reuse) — exhaustive on the full 12-surface space is infeasible (≈3·10⁵), so the
  audit samples a reduced surface set per decision and reports the sample.

## What parity does NOT promise (the measured approximations)

1. **Prefix reuse** — planning uses unique prefixes; the routing/cache-reuse benefit is the
   `kv_service_factor` channel, not the residency model. Reuse-dependent ranking (routing actions) is
   eval-only; the contract measures the resulting routing-rank error and reports it.
2. **Prompt length** — planning uses a `planning_prompt_tokens` hint (eval-window median); without it,
   output tokens proxy the prompt. The regime/phase split error from this is reported.
3. **Magnitude** — planning reward magnitude may differ from eval; only the **ranking** is contracted.

## Failure handling

If an approximate planning model has material regret on a setting, the contract requires **reporting** it
(strategy, regret %, which bundle was missed, whether a headline was missed) — never silencing it by
changing the fixture, weakening the gate, or tuning the simulator. If planning cannot be made faithful
enough, the honest outcome is "the gap is X" (physical / search-regret / cost-mode), documented in
`DT60_MPC_SEARCH_ARCHITECTURE_RESULTS.md`.
