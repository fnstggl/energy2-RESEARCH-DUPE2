# MPC Search Strategy Decision (Phase 12)

The default optimizer, chosen by measured regret / runtime / determinism / stability / Pareto safety —
**not** by sophistication. Evidence from the controlled fixtures
(`tests/test_roofline_mpc_actions.py`, `tests/test_mpc_planning_fidelity.py`) and the dt=60 reruns
(`DT60_MPC_SEARCH_ARCHITECTURE_RESULTS.md`, `MPC_SEARCH_REGRET_AUDIT.md`).

## Adopted defaults

| setting | strategy | why |
|--|--|--|
| raw candidate count ≤ `exhaustive_max` (4096) or fixtures | **exhaustive_cartesian** | tractable → zero regret by construction; deterministic |
| roofline action space (≈3·10⁵ raw) | **beam_search + local improvement** | captures cross-surface interactions (precision×batching, precision×spec) coordinate descent misses; regret ≈ 0 vs exhaustive on the reduced-surface fixtures; ~200 evals/decision (bounded) |
| strict runtime budget | **coordinate_descent fallback** + regret sampling | cheap; only when beam exceeds the latency budget — never the sole optimizer |
| high-interaction / high-recent-regret | **beam + CEM / random-restart** | optional; deterministic (seeded); used only if a regret sample shows beam is leaving value |

The strategy is selected by `AdaptiveSearchPlanner.plan` from the raw count (`exhaustive_max`) with
`large_strategy="beam_search"` and `beam_local_improve=True`; every decision **reports** the chosen
strategy, raw count, candidates evaluated, and (when the raw count ≤ `regret_audit_max`) the measured regret.

## Why beam + local, not the others

- **Coordinate descent alone is demoted** — it misses coupled optima (`test_planner_reports_regret_and_is_bounded`,
  the fp8×aggressive-batching interaction). It survives only as a fallback and as the local-improvement step
  on the beam winner (which helped 3/32 real decisions in the dt=60 rerun).
- **CEM / random-restart are optional** — on these action spaces beam already matched exhaustive (regret ≈ 0),
  so the extra samples bought nothing; they remain available, deterministic, for higher-interaction regimes.
- **Exhaustive is the default only when tractable** — the full roofline space is too large to enumerate per
  decision; the planner uses it for fixtures and the regret audit (reduced surfaces), where it is the ground
  truth.

## Planning parity: OFF by default (the honest call)

The planning/eval cost-channel parity fix (`planning_kv_cost_mode`) is **opt-in, default off**. It correctly
makes the planner select FP8 (the PR #111 miss), but the dt=60 rerun shows enabling it **net-regresses**
gp/$ (−4.5%, worse SLA) because the synthetic planning workload under-represents per-period SLA pressure, so
the now-cost-aware planner under-values speculative decoding's SLA protection. We do **not** ship a
regression. The default controller is unchanged; PR #111's +33.6% live result stands. The parity fix is
documented, tested, and available behind the flag for the follow-up that closes the SLA-representation gap.

## Determinism + safety

All strategies are deterministic (exhaustive ordering fixed; beam tie-broken by the sorted bundle key;
CEM/random-restart seeded per decision). Clone isolation, the Pareto gate, and `ActionBundle()`-default
reproduction are unchanged. No reward bonuses; no simulator tuning.
