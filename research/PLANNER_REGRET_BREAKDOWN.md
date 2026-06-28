# Planner Regret Breakdown

Where does the remaining gap between today's MPC and a perfect planner come from — *forecast* error, *search*
error, *objective* error, or *world-model* (simulator) error? This is the offline tier of the Decision
Diagnostics Engine (`regret_decomposition`, `scripts/diagnose_mpc_attribution.py`). Artifacts:
`data/external/mpc_controller/mpc_attribution.json` and `mpc_scenario_forecaster_dt60.json` (PR #113).

## The four-point chain (Static optimum → Current → Scenario → Oracle)

Measured by varying **only the planning workload** while holding the controller, simulator and objective fixed
(so the Current→Oracle gap is forecast quality *by construction*), on the bounded dt=60 Azure window:

| arm | SLA-safe gp/$ | step | Δ gp/$ |
|--|--|--|--|
| Static optimum (best fixed bundle, no MPC) | 149 646 | — | — |
| **Current MPC** (single synthetic median planning workload) | **149 164** | vs static | **−482** |
| **Scenario MPC** (trace-derived scenario ensemble, PR #113) | **171 485** | Current → Scenario | **+22 321** |
| **Oracle MPC** (every forecast = realised future) | **174 062** | Scenario → Oracle | **+2 576** |
| | | **Current → Oracle (total planner regret)** | **+24 897** |

## Decomposition

```
REGRET: forecast 100.0% | search 0.0% | objective 0.0% | world-model UNMEASURABLE_IN_SIMULATION
```

- **Forecast quality — 100% of the measurable regret.** Current→Oracle is +24 897 gp/$ and is, by
  construction, entirely the planning-workload (forecast) gap. The planner today is *starved of a good
  workload forecast*, not of search or actions.
- **Search — 0%.** The regret audit (PR #112, `MPC_SEARCH_REGRET_AUDIT.md`) re-runs the exhaustive cartesian
  and finds the adaptive beam+local search already returns the optimum on this space → no measurable search
  regret. The gap is not "the search missed a better bundle."
- **Objective — 0%.** Same objective (SLA-safe gp/$ under the Pareto gate) across all arms; it contributes no
  differential regret here.
- **World-model fidelity — UNMEASURABLE in pure simulation.** Planner and evaluator share the simulator, so
  there is no higher-fidelity reference to attribute simulator error against. Not fabricated; see
  `WORLD_MODEL_ATTRIBUTION_RESULTS.md`. Closing it needs real serving telemetry (the standing roadmap item).

## The most important sub-finding — most of the forecast gap is a *workload-model* gap, not an oracle gap

Splitting the forecast regret:

| within-forecast component | Δ gp/$ | share of forecast regret |
|--|--|--|
| **workload-model gain** (single median → scenario ensemble), Current → Scenario | **+22 321** | **89.6%** |
| residual forecast gap (scenario → perfect oracle), Scenario → Oracle | +2 576 | 10.4% |

The residual to a *perfect* oracle is only **11.5% of the scenario gain** — i.e. ~90% of the achievable
forecast value is unlocked by replacing the single synthetic median planning workload with a trace-derived
**scenario ensemble** (already shipped opt-in in PR #113), *without* needing a perfect forecaster. Both adaptive
arms (Scenario and Oracle) **exceed the static optimum**, so MPC's value over a fixed best-bundle is real once
it plans against a realistic workload.

## Which forecast variable — see the attribution

The 100%-forecast regret is attributed across the consumed variables by leave-one-out
(`FORECAST_ATTRIBUTION_RESULTS.md`): **output_length 62.8% ≫ prompt_length 24.7% > interarrival_cv 12.3% >
arrival_rate 0.3%**; KV-reuse / queue / SLA-pressure / carbon / weather / congestion are **ABSENT (0 by
construction)** — the planner does not consume them. The roadmap (`AUTOMATIC_IMPROVEMENT_ROADMAP.md`) is ranked
directly from this.

## Honesty

Bounded window → magnitudes are simulator-inferred; the **direction and ordering** (forecast ≫ search≈0;
workload-model ≫ residual oracle gap) is the robust finding. The Current/Scenario/Oracle numbers are sourced
from the PR #113 scenario diagnostic; the +82.1% validation headline (`MPC_VALIDATION_REPORT.md`) is a separate
full-action-layer run — they agree in direction, not in absolute scale (different harness windows), and that is
stated rather than blurred. Diagnostic only: no controller / forecaster / simulator / objective change.
