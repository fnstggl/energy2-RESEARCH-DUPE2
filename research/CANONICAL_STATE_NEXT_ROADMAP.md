# Canonical State — Next Roadmap (Phase 11)

Ranked next state/model improvements, by **evidence** (the merged backtests + this PR's audit), not guesswork.
Each item: the evidence, the expected effect, and whether it can be done offline or needs pilot telemetry.

| rank | improvement | evidence it matters | expected effect | offline or telemetry |
|--|--|--|--|--|
| **1** | **output-length / arrival forecast fidelity** (consume ForecastState error) | PR #118 oracle gap +1580…+7520 gp/$ is dominated by arrival_rate + output_length error; ForecastState now measures that error per variable | shrinks the measured oracle gap; the highest-leverage gp/$ lever | **offline** (better forecaster on existing traces) |
| **2** | **queue/SLA-pressure forecasting** (forecast the QueueState summary) | every cell is non-headline-safe because the MPC sits above baseline SLA; queue pressure (the SLA driver) is emergent, not a forecast input (`ABSENT_FORECASTS`); RequestState/QueueState now expose it | a path to Pareto-safety (the only way the gate flips) | **offline** (now that QueueState is canonical) |
| **3** | **real GPU power telemetry** (close the DVFS curve) | N2 / electricity value magnitude is a DIRECTIONAL upper bound gated by `power_w=TDP·(0.4+0.6·clock^2.4)` and clock-independent decode (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`) | tightens every electricity-dollar claim from directional to calibrated | **telemetry** (DCGM/nvml) |
| **4** | **real per-request KV-reuse forecast** | planning uses synthetic unique prefixes (PR #112) → KV reuse is unconsumed; RequestState can carry reuse keys | better prefill economics → service time → SLA | **telemetry** (real cache-hit traces) |
| **5** | **real quality model** (replace the int4 quality-risk constant) | QualityState's `quality_sla_risk` is a conservative constant, not a measured accuracy model | honest precision/accuracy trade instead of a prior | **telemetry** (eval harness) |
| **6** | **thermal state / true power caps** | no thermal model; real clocks are thermally constrained | bounds the achievable DVFS range | **telemetry** |
| **7** | **demand charges** | only day-ahead energy price is modeled; real bills include $/kW demand | changes the electricity objective shape | **telemetry / contract data** |
| **8** | **adaptive all-knobs runtime** (coarser cadence or pruned beam) | the full adaptive search is heavy at hourly cadence (this PR's runner marks heavy cells TIMEOUT/SKIPPED_TOO_HEAVY) | makes the all-knobs total measurable | **offline** (engineering) |

## The single highest-ROI next step

**(1) output-length / arrival forecast fidelity**, instrumented by the new ForecastState error summary. It
directly targets the largest measured regret (the oracle gap), is doable entirely offline on existing traces,
and every subsequent state (queue-pressure forecasting, KV-reuse) builds on a forecaster the planner can trust.
It is the first item because the evidence (oracle gap attribution) points squarely at it — not because it is
the most novel.
