# MPC Validation Report

The complete Aurelius MPC — every implemented action surface live (routing, batching, capacity, placement,
migration, prewarm, precision, speculative decoding, GPU clock, admission, ordering) — measured exactly as it
exists today against the strongest baselines on the bounded Azure+Mooncake window (hybrid cost, Pareto gate).
Artifact: `data/external/mpc_controller/mpc_attribution.json`.

## Headline

| arm | SLA-safe goodput/$ |
|--|--|
| **Current Aurelius MPC (full action layer)** | **183 152** |
| best SLA-aware baseline (`sla_aware`) | 100 555 |
| greedy (backlog+kv_aware+aggressive) | see artifact |
| FIFO | see artifact |

**Current Aurelius MPC vs best SLA-aware baseline: +82.1% SLA-safe goodput/$.**

## Method + honesty

No knobs disabled; the controller runs its adaptive beam+local search over the full connected action space
with online diagnostics on. Bounded window → the magnitude is simulator-inferred (the **direction** — the MPC
substantially beats the SLA-aware baseline with all surfaces live — is the robust finding). Baselines are
deployable fixed policies (no oracle, no future info). Per-window breakdowns are in the artifact; this is
diagnostic-only (no controller/forecaster/simulator change). The remaining planner regret is attributed in
`FORECAST_ATTRIBUTION_RESULTS.md` / `PLANNER_REGRET_BREAKDOWN.md`.
