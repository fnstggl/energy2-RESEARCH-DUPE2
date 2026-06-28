# dt=60 MPC Search-Architecture Results (Phase 11)

Did the planning/eval fidelity fix let the live MPC recover the static FP8+spec value? Bounded dt=60 on the
6-hour Azure window + Mooncake prefixes + hybrid cost + the same Pareto gate
(`scripts/diagnose_mpc_search_architecture_dt60.py` →
`data/external/mpc_controller/mpc_search_architecture_dt60.json`; eval 120 periods, MPC 32 decisions,
median prompt 828). **Reported, not forced** — and the answer is a precise, honest *partial*.

## Result

| arm | gp/$ | %of D | SLA | precision selected | spec selected |
|--|--|--|--|--|--|
| A pre-roofline (fair) | 103 584 | 69.2% | 0.0380 | bf16 | off |
| C static fp8 | 143 137 | 95.7% | 0.0037 | fp8 | off |
| **D static fp8+spec** | **149 646** | **100%** | 0.0000 | fp8 | medium |
| **MPC, planning OFF** (= PR #111) | 138 497 | 92.5% | 0.0041 | **bf16 ×32** | aggressive ×32 |
| **MPC, planning ON** (this PR) | 132 273 | 88.4% | 0.0225 | **fp8 ×9**, bf16 ×23 | off ×12, aggr ×16, med ×4 |

Both MPC arms are Pareto-safe vs A. **Planning ON makes the planner select FP8** (9/32 decisions, vs 0 with
parity off — the mechanism is fixed, confirmed at the single-decision level too). But aggregate gp/$ **went
down −4.5%** and SLA got **worse** (0.0225 vs 0.0041).

## Why — the residual gap is now SLA-representation, not the cost channel

Planning-OFF always picks `{bf16, spec aggressive, clock low, batching aggressive}` — a uniform aggressive
bundle whose **speculative decoding drives SLA→0** (its only visible channel in latency-only planning).
Planning-ON additionally sees spec's **compute tax** (extra GPU-seconds) and precision's **cost benefit**,
so it sometimes turns **spec off** and picks **fp8** to cut cost (realized GPU-seconds fall 1430→1216). On
the *real* eval workload that trade is wrong: spec's SLA protection is worth more than its compute tax, so
turning it off raises violations more than fp8 saves cost.

The diagnosis (per the Phase-1 parity contract): the **cost-channel** planning/eval parity is **fixed**, but
it exposed a **second** gap — the synthetic, forecast-derived planning workload **under-represents per-period
SLA pressure**, so the now-cost-aware planner under-values spec's SLA-protection benefit. The blocker is
**planning-workload representativeness**, *not* search regret (beam already finds the planning optimum; +
local improvement helped 3/32) and *not* the cost model.

## Decision (Phase 12)

Keep planning parity **OFF by default** (`planning_kv_cost_mode=None`) — it is opt-in. The cost-channel
parity is proven and available, but enabling it net-regresses gp/$ on this workload until the
SLA-representation gap is closed, and we do **not** ship a regression or tune it away. The default controller
is therefore **unchanged** (no regression); PR #111's +33.6% live result stands as the shipped behaviour.

## Required interpretation

1. **Planning/eval gap that existed?** Planning scored candidates without the phase/hybrid-cost/roofline
   model, so precision's value (mostly cost) was invisible → bf16 picked.
2. **Fixed?** The cost channel — yes (synthetic unique-prefix kv_state + hybrid cost in `_rollout_world`).
3–5. **Planning now sees fp8 cost / spec latency+tax / clock energy?** Yes — proven by the flip to fp8 and
   the spec-off trade.
6–7. **Strategies / default?** exhaustive (small) / beam+local (default) / coordinate fallback / CEM
   optional; default stays beam+local, planning-parity off.
8–10. **Regret?** beam ≈ exhaustive on fixtures (regret ≈ 0); local improvement helped 3/32 real decisions;
   CEM not needed.
11. **Recover static FP8+spec?** **No** — 88.4% with parity on (a regression vs 92.5% off). The planner now
    selects fp8 but mis-trades spec under the SLA-representation gap.
12. **New dt=60 gp/$?** 132 273 (parity on) vs 138 497 (off) vs 149 646 (static D).
13. **Pareto-safe?** Yes (both arms beat A with SLA not worse-than-A; parity-on SLA is worse than parity-off
    but still ≤ A).
14. **Runtime?** ~206–208 candidate evaluations/decision (beam+local); bounded, reported per decision.
15. **What still blocks optimality?** The **synthetic planning workload's SLA representation** — a causal
    forecast/representativeness gap (the planner cannot use the realized period's records). The next fix is
    a more faithful planning workload (e.g. a forecast that carries prompt length + an SLA-pressure-preserving
    risk path), not more serving physics and not a different search algorithm.

## Claim safety

Production-safe: the planning/eval *gap diagnosis* and the *mechanism* (planner now selects fp8). The
magnitudes (−4.5%, 88.4%) are simulator-inferred on a bounded window. No headline is claimed for the parity
arm — it is reported as a measured non-improvement with the blocker named.
