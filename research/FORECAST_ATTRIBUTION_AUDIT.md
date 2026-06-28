# Forecast Attribution Audit

> **Canonical results: [`FORECAST_ATTRIBUTION_RESULTS.md`](FORECAST_ATTRIBUTION_RESULTS.md).**
> This pointer exists because the PR deliverables name both an "audit" and a "results" title; the measured
> leave-one-out attribution table lives in the results doc to keep the numbers in one place.

**Audit question:** which forecast variables does the MPC planner actually consume, and which of those create
planner value? **Method:** leave-one-out from the oracle (`LeaveOneOutAttributor`,
`scripts/diagnose_mpc_attribution.py`) — start from a perfect-forecast planner, degrade one variable back to
its model value, re-plan, measure the gp/$ drop. Audited over `CONSUMED_FORECASTS` only; everything else is
reported **ABSENT (0 by construction)**, never fabricated.

Result (`data/external/mpc_controller/mpc_attribution.json`):

```
output_length 62.8%  ≫  prompt_length 24.7%  >  interarrival_cv 12.3%  >  arrival_rate 0.3%
electricity_price 0.0% (≈constant)   |   kv_reuse / queue / sla_pressure / carbon / weather / congestion = ABSENT
```

**Audit verdict:** the attribution is honest — reported only over variables the planner consumes, with absent
variables explicitly listed *with reasons* and scored 0, not invented. The intuition that KV-reuse or
queue-pressure forecasts dominate is **contradicted by the evidence**: the planner does not consume them.
Full table, reading, and method notes in `FORECAST_ATTRIBUTION_RESULTS.md`; the ranking drives
`AUTOMATIC_IMPROVEMENT_ROADMAP.md`.
