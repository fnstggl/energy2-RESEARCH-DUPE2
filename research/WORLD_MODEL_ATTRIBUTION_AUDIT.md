# World-Model Attribution Audit

> **Canonical results: [`WORLD_MODEL_ATTRIBUTION_RESULTS.md`](WORLD_MODEL_ATTRIBUTION_RESULTS.md).**
> This pointer exists because the PR deliverables name both an "audit" and a "results" title.

**Audit question:** how much of the planner's regret is *world-model* (simulator) error — queue evolution,
KV-cache evolution, roofline, batching, migration, placement, replica lifecycle, power, network?

**Audit verdict — the honest one:** in a fully-simulated environment this is **UNMEASURABLE**. The planner and
the evaluator share the *same* simulator, so there is no higher-fidelity reference to attribute simulator error
against. We therefore **do not fabricate** a per-component percentage (no invented "queue 37% / KV 28%").

```
world-model fidelity = UNMEASURABLE_IN_SIMULATION (needs real serving telemetry)
```

The one measurable planner/world boundary is the deliberate **planning approximation** (synthetic unique-prefix
workload vs the eval's real prefix stream, PR #112) — and that is a *workload/forecast* approximation, already
captured in the forecast attribution, not simulator-fidelity error. Isolating true world-model fidelity is the
standing high-effort roadmap item (collect real serving telemetry → `AUTOMATIC_IMPROVEMENT_ROADMAP.md`). Full
component table and reasoning in `WORLD_MODEL_ATTRIBUTION_RESULTS.md`.
