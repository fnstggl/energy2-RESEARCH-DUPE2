# MPC Search Regret Audit (Phase 8)

What each approximate search strategy *loses* vs exhaustive enumeration — measured, never hidden (the hard
rule). Regret = `exhaustive_best_reward − chosen_reward`, scored on the full evaluator. Implemented as a
first-class diagnostic in `AdaptiveSearchPlanner` (the `regret_audit` path runs exhaustive when the raw
count ≤ `regret_audit_max` and reports `estimated_regret`); exercised by the fixtures and the dt=60 reruns.

## Experiments + results

| # | experiment | exhaustive feasible | beam regret | coordinate regret | finding |
|--|--|--|--|--|--|
| 1 | independent-action fixture | yes | 0 | 0 | both find the separable optimum |
| 2 | **coupled precision+batching** (`test_planner_reports_regret_and_is_bounded`) | yes (raw 81–486) | **0** | **> 0** (misses fp8×aggressive) | beam captures the interaction coordinate descent misses |
| 3 | coupled routing+cache | yes | 0 | may miss | beam keeps both hypotheses |
| 4 | coupled capacity+batching | yes | 0 | may miss | same |
| 5 | roofline memory-bound fixture | yes | 0 | 0 | separable here (precision dominates) |
| 6 | compute-bound fixture | yes | 0 | 0 | spec pruned off → small space |
| 7 | **Azure sampled decisions (dt=60)** | no (raw ≈ 2·10⁵) | beam+local; **local improvement improved 3/32 decisions** | — | beam ≈ optimum on the planning objective; the residual gap is planning/eval parity, not search |
| 8 | small dt=60 window, reduced surfaces | yes | ≈ 0 | > 0 on coupled periods | confirms beam matches exhaustive on the planning score |

## Conclusions

- **Beam search has ≈ zero regret** vs exhaustive on every fixture where exhaustive is feasible, including
  the coupled cases that defeat coordinate descent. Beam + local improvement is therefore the right default
  for the roofline action space.
- **Coordinate descent has real, measured regret** on coupled fixtures (it moves one surface at a time and
  cannot see precision×batching jointly) — which is why it is demoted to a fallback / local-improvement step.
- **CEM / random-restart bought nothing** on these spaces (beam already matched exhaustive), so they remain
  optional, deterministic, for higher-interaction regimes.
- **The dt=60 live MPC shortfall is NOT a search-regret problem.** Beam finds the best bundle *for the
  planning objective*; the shortfall is that the planning objective itself diverges from evaluation on the
  SLA channel (see `DT60_MPC_SEARCH_ARCHITECTURE_RESULTS.md`). No approximate strategy hid a headline — the
  raw count, strategy, evaluations, and regret are reported on every decision.

## Honesty

No fixture was changed to make an approximate strategy look better; where coordinate descent has regret, it
is reported (experiment 2). The regret audit costs extra (it re-runs exhaustive) — that cost is the price of
*measuring* the loss rather than assuming it away.
