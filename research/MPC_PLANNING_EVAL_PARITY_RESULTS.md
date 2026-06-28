# MPC Planning/Eval Parity Results (Phase 10)

Measured against the Phase-1 parity contract: does the planning rollout now agree with full evaluation on
the **direction** of each candidate's effect, and does it rank FP8 above BF16 (the PR #111 miss)? Evidence
from `tests/test_mpc_planning_fidelity.py` (3 pass) + the dt=60 reruns.

## Controlled fixtures (parity ON) — direction agreement

| candidate vs neutral | eval direction | planning direction (parity ON) | agree? |
|--|--|--|--|
| fp8 (memory-bound) — cost | lower realized GPU-s → lower cost | lower `operator_cost` (`test_…sees_fp8_cost_benefit`) | ✅ |
| fp8 — gp/$ ranking | fp8 > bf16 | fp8 > bf16 → **planner selects fp8** (`test_planning_off_picks_bf16_planning_on_picks_fp8`) | ✅ |
| bf16 default | unchanged | unchanged (parity OFF default; `test_planning_parity_off_by_default`) | ✅ |

**Top-1 agreement on the memory-bound decision: fixed.** With parity OFF the planner's argmax is bf16
(eval says fp8) — a top-1 disagreement. With parity ON the planner's argmax is fp8 — matching eval. This is
the core contract item the fix targets, and it passes.

## Azure sampled decisions (dt=60, 32 decisions) — where parity still diverges

The aggregate (`DT60_MPC_SEARCH_ARCHITECTURE_RESULTS.md`): parity ON selects fp8 in **9/32** decisions (vs
0/32 off) — the **cost channel** now agrees in direction. But planning and eval **disagree on speculative
decoding's net value**: planning (parity on) sees spec's compute tax and turns it off in 12/32 decisions to
cut cost; eval punishes that with more SLA violations (0.0225 vs 0.0041). So:

| parity metric | result |
|--|--|
| top-1 agreement (precision, single memory-bound decision) | ✅ flips bf16→fp8 |
| direction agreement (cost channel) | ✅ fp8 cheaper in both |
| direction agreement (spec **net** value: cost tax vs SLA protection) | ❌ planning over-weights the cost tax (SLA under-represented) |
| aggregate gp/$ ranking (parity-on bundle vs parity-off bundle, on eval) | ❌ parity-on bundle scores lower on eval |
| Pareto gate agreement | ✅ both arms Pareto-safe vs A |

## Diagnosis (per the contract's failure-handling rule)

The **cost-channel** parity is achieved; the residual divergence is the **SLA channel**: the synthetic,
forecast-derived planning workload under-represents per-period SLA pressure, so spec's SLA-protection value
is invisible in planning and its compute tax dominates → the planner mis-trades it. This is a
**planning-workload representativeness** gap (causal: the planner cannot see the realized period's records),
**not** the cost model and **not** search regret (`MPC_SEARCH_REGRET_AUDIT.md`: beam ≈ exhaustive).

Per the contract, this is **reported, not silenced**: the parity fix is kept opt-in (default off) so it does
not ship a regression, and the SLA-representation gap is named as the next fix (a forecast that carries
prompt length + an SLA-pressure-preserving risk path for planning).
