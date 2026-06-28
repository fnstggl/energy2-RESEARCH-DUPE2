# Forecast Attribution Results

Which forecast variable actually creates planner value? Measured by **leave-one-out from the oracle**: start
from the ORACLE planner (every forecast = the realised future), degrade ONE variable back to its model value,
re-plan, and measure the gp/$ drop on the true workload — that drop is the variable's planner-value
(`scripts/diagnose_mpc_attribution.py`, `LeaveOneOutAttributor`, normalised to 100%).

| forecast variable | contribution to forecast planner-value | status |
|--|--|--|
| **output_length** | **62.8%** | consumed |
| prompt_length | 24.7% | consumed |
| interarrival_cv | 12.3% | consumed |
| arrival_rate | 0.3% | consumed |
| electricity_price | 0.0% (≈constant over the window) | consumed (objective) |
| kv_reuse | **0.0%** | **ABSENT** — planning uses unique prefixes (PR #112) |
| queue_pressure / sla_pressure | **0.0%** | **ABSENT** — emergent, not planner inputs |
| carbon / weather / congestion | **0.0%** | **ABSENT** — not wired |

## Reading

**Output-length prediction dominates (63%)**, then prompt-length (25%) and inter-arrival CV (12%); arrival
volume and electricity barely move the decision. This *contradicts* the intuition that KV-reuse or
queue-pressure forecasts matter most — they do **not**, because the planner does not consume them (KV-reuse is
unique-prefix in planning; queue/SLA-pressure are emergent from arrival+service). The honest attribution is
reported only over consumed variables; absent variables are 0 by construction, never fabricated.

**Method:** leave-one-out (records the `attribution_method`); it ignores interaction terms (the residual vs
the full oracle→current gap captures them). The `ForecastAttributor` interface allows Shapley /
integrated-gradients later without changing callers. Bounded window → magnitudes simulator-inferred; the
**ranking** (output ≫ prompt ≫ cv ≫ arrival) is the robust result.
