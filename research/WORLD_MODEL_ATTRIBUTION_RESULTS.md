# World-Model Attribution Results

Forecast error is one source of planner regret; world-model (simulator) error is the other. The honest finding:
**within a fully-simulated environment, world-model fidelity is NOT cleanly isolable**, because the planner and
the evaluator share the *same* simulator — there is no higher-fidelity reference to substitute. Attributing
simulator error requires a ground truth the planner is wrong against, i.e. **real serving telemetry (absent)**.

| world-model component | attributable regret | why |
|--|--|--|
| queue evolution · KV-cache evolution · roofline · batching · migration · placement · replica lifecycle · power · network | **UNMEASURABLE in pure simulation** | planner & evaluator share the model → no reference; needs real telemetry |

The one measurable planner/world boundary is the **planning approximation** the controller deliberately makes
(synthetic unique-prefix workload vs the eval's real prefix stream, PR #112) — and that is a *forecast/workload*
approximation, already captured in the forecast attribution, not a simulator-fidelity error.

**We do not fabricate a percentage.** Rather than invent "queue 37% / KV 28%", we report that world-model
attribution is blocked on real telemetry, and the roadmap's standing high-effort item is exactly that: collect
real serving telemetry so simulator error becomes attributable (`AUTOMATIC_IMPROVEMENT_ROADMAP.md`).
