# Planner Regret Decomposition

> **Canonical document: [`PLANNER_REGRET_BREAKDOWN.md`](PLANNER_REGRET_BREAKDOWN.md).**
> This pointer exists because the PR deliverables name both titles; the full measured decomposition (the
> Static optimum → Current → Scenario → Oracle chain, the four-way forecast/search/objective/world-model split,
> and the workload-model-vs-residual-oracle sub-finding) lives there to avoid duplicating numbers that must
> stay in one place.

Headline result (bounded dt=60 Azure window, `data/external/mpc_controller/mpc_attribution.json`):

```
REGRET: forecast 100.0% | search 0.0% | objective 0.0% | world-model UNMEASURABLE_IN_SIMULATION
total planner regret (Current → Oracle) = +24 897 gp/$
  └─ workload-model gain (Current → Scenario) = +22 321  (89.6%)
  └─ residual forecast gap (Scenario → Oracle) = +2 576  (10.4%)
```

The remaining planner gap is forecast quality; search regret is ~0; world-model fidelity is not isolable in
pure simulation. See `PLANNER_REGRET_BREAKDOWN.md` for the full table, method, and honesty notes.
