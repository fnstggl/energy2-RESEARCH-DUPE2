# Decision Diagnostics Engine (permanent planner observability)

The MPC controller now **never produces an action without an explanation**. `decision_diagnostics.py` is a
permanent, first-class controller output, split into a strictly-separated online tier and offline tier so the
live planner stays fast.

```
Historical telemetry → Forecasts → Persistent ClusterState → World Simulation → MPC Search
                                                                                    │
                                                                                    ▼
                                                                       Decision Diagnostics Engine
                                                                                    │
                                                                                    ▼
                                                                            Chosen ActionBundle (+ explanation)
```

## ONLINE — emitted after every solve, negligible overhead

Wired into `controller.decide` (`emit_diagnostics=True` by default). It exposes **only values the search
already computed** — no leave-one-out, no Shapley, no oracle reruns, no extra MPC solves. Measured cost on a
real Azure decision: **0.48 s end-to-end** (the capture is a list-append per scored candidate plus one sort).
`controller.last_decision_diag["diagnostics"]` carries a `DecisionExplanation`:

- **decision summary** — chosen bundle, expected reward / gp$ / SLA / cost, planning & forecast horizon,
  planner confidence;
- **reward decomposition** — the chosen bundle's components straight off its `PeriodOutcome`
  (goodput, operator/warm-hold/migration cost, energy, queue delay, SLA / quality risk);
- **decision margin** — winner − runner-up reward, and `decision_margin_pct`; a small margin ⇒ a *fragile*
  decision (e.g. the validated decision came out `margin_pct ≈ 0` → `stable=False`, several bundles tied);
- **planner confidence** — a normalised blend of the forecast-spread confidence and the search decisiveness
  (winner→runner gap + score dispersion across the beam) — all already computed;
- **local switching thresholds** — for the surfaces distinguishing the winner from the runner-up, the
  decision margin as the cheap robustness proxy (precise per-variable thresholds are an *offline* analysis);
- **competing candidates** + **why-won** (the surfaces where the winner differs from the runner-up);
- search strategy, candidates evaluated, planning latency.

## OFFLINE — validation / benchmarking only (never called during live planning)

`scripts/diagnose_mpc_attribution.py` runs the expensive analyses that re-plan under perturbed inputs:

- **forecast leave-one-out attribution** (`LeaveOneOutAttributor`) — start from the ORACLE planner, degrade
  one forecast variable, measure the gp/$ drop;
- **planner-regret decomposition** (`regret_decomposition`) — Current → Scenario → Oracle;
- **automatic roadmap** (`generate_roadmap`) — ranked directly from the attribution.

Attribution is **pluggable**: `LeaveOneOutAttributor` today; the `ForecastAttributor` interface lets Shapley /
integrated-gradients drop in later WITHOUT changing the diagnostics interface, and every result records the
`attribution_method` that produced it.

## Honesty by construction

Attribution is reported only over the forecast variables the planner ACTUALLY consumes
(`CONSUMED_FORECASTS` = arrival_rate, output_length, prompt_length, interarrival_cv, electricity_price). A
variable the planner does **not** consume (`ABSENT_FORECASTS`: KV-reuse — planning uses unique prefixes since
PR #112; queue/SLA-pressure — emergent, not inputs; carbon / weather / congestion — not wired) is reported
**ABSENT (0 by construction)**, never fabricated. World-model fidelity is flagged **UNMEASURABLE in pure
simulation** (the planner and evaluator share the simulator — there is no higher-fidelity reference; isolating
it needs real serving telemetry). This PR changes **no controller decision behaviour** — it only adds the
explanation alongside the action. Tests: `tests/test_decision_diagnostics.py`.
