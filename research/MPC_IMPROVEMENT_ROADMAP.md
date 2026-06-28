# MPC Improvement Roadmap

> **Canonical document: [`AUTOMATIC_IMPROVEMENT_ROADMAP.md`](AUTOMATIC_IMPROVEMENT_ROADMAP.md).**
> This pointer exists because the PR deliverables name both titles. The roadmap is *generated* from the
> measured attribution (`generate_roadmap`), so it lives in one place to stay in sync with the evidence.

Auto-ranked next work (from `data/external/mpc_controller/mpc_attribution.json`):

```
1. output_length forecaster        62.8%  (low effort)      ← build first
2. prompt_length forecaster        24.7%  (medium)
3. interarrival_cv forecaster      12.3%  (medium)
4. arrival_rate forecaster          0.3%  (low)             ← do NOT prioritise
5. real serving telemetry           n/a   (high)            ← unblocks world-model attribution
```

Because the remaining planner regret is 100% forecast quality (`PLANNER_REGRET_BREAKDOWN.md`), every ranked
item is a forecaster, ordered by its share of measured forecast value; the standing high-effort item is real
serving telemetry, the only way to make simulator (world-model) error attributable. Note that ~90% of the
forecast regret is *already* recovered by the opt-in scenario-ensemble workload model (PR #113) — the cheapest
win is to validate and default that on. See `AUTOMATIC_IMPROVEMENT_ROADMAP.md` for effort/confidence/honesty.
