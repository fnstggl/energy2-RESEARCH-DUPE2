# Automatic Improvement Roadmap

This roadmap is **not hand-written** — it is generated directly from the measured forecast attribution and
regret decomposition (`generate_roadmap` in `aurelius/environment/decision_diagnostics.py`, run by
`scripts/diagnose_mpc_attribution.py`). Each item's rank and estimated impact come straight from the
leave-one-out attribution, so the roadmap updates itself whenever the attribution is re-run. Artifact:
`data/external/mpc_controller/mpc_attribution.json`.

## What the measurement says to build, in order

The regret decomposition found the remaining planner gap is **100% forecast quality** (search ≈ 0, world-model
unmeasurable in simulation — see `PLANNER_REGRET_BREAKDOWN.md`). So every actionable item is a *forecaster*,
auto-ranked by its share of the consumed-forecast attribution:

| rank | improvement | est. impact (% of forecast regret) | effort | confidence |
|--|--|--|--|--|
| 1 | **output_length forecaster** | **62.8%** | low | medium (leave-one-out, bounded window) |
| 2 | prompt_length forecaster | 24.7% | medium | medium |
| 3 | interarrival_cv forecaster | 12.3% | medium | medium |
| 4 | arrival_rate forecaster | 0.3% | low | medium |
| 5 | **real serving telemetry** (to attribute world-model fidelity) | n/a — *unblocks* the unmeasurable axis | high | high |

```
ROADMAP: output_length forecaster > prompt_length forecaster > interarrival_cv forecaster > arrival_rate forecaster
```

## Reading

- **Build the output-length forecaster first.** It carries 62.8% of the measured forecast planner-value —
  more than the other three consumed variables combined. This is the single highest-leverage next piece of
  work, and it is *low* effort (the forecasting scaffold already exists; this is a better per-request
  output-length model, not new infrastructure).
- **Then prompt_length (24.7%) and interarrival_cv (12.3%).** Together with output_length they cover ~99.8% of
  the consumed-forecast value.
- **arrival_rate is nearly irrelevant (0.3%)** on this window — *do not* spend effort there despite the
  intuition that load forecasting matters most; the evidence says output-shape forecasting dominates.
- **The standing high-effort item is real serving telemetry.** World-model (simulator) fidelity is
  UNMEASURABLE in pure simulation (`WORLD_MODEL_ATTRIBUTION_RESULTS.md`); collecting real serving telemetry is
  the *only* way to make simulator error attributable, so it is always present as the last item — it doesn't
  rank on forecast-regret share because it unblocks a different axis entirely.

## Cross-check against the regret chain

`PLANNER_REGRET_BREAKDOWN.md` shows ~90% of the forecast regret is recovered by the **scenario-ensemble
workload model** (PR #113), already shipped opt-in — i.e. the cheapest win (turn the scenario forecaster on by
default after validation) is *already built*. The forecaster items above sharpen the remaining gap; the
residual to a perfect oracle is only ~11.5% of the scenario gain, so these are diminishing-returns refinements
beyond the scenario ensemble, ranked so the highest-value one is done first.

## Honesty

Ranks and impacts are simulator-inferred on a bounded window (the **ordering** is the robust result); every
item records its `confidence`. Impacts are shares of the *forecast* regret, not absolute gp/$ guarantees. This
document is regenerated from the attribution artifact — if the attribution changes, re-run the script and this
ranking changes with it. No controller / forecaster / simulator change was made to produce it.
