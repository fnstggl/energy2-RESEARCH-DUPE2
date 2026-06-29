# State Regret Priority (Phase 1)

Which missing/partial states most plausibly explain current planner **regret** (gap to oracle) and the
**non-headline-safe** results, ranked from evidence in the merged PRs (#117 electricity, #118 N2, #119 roofline
promotion) and the decision-diagnostics attribution. Evidence first; ranking second.

## Evidence base

- **Oracle gap (forecast regret), PR #118 N2 backtest:** `oracle − causal` = **median ≈ +1580 gp/$, up to
  +7520 (ercot·expensive)**; near-zero where the price is unambiguous. Attribution: dominated by **arrival_rate**
  and **output_length** forecasts; electricity price contributes little (published day-ahead).
- **Scenario planning is not a robust win (PR #118):** median ≈ −600 gp/$, one −8952 loss — risk-averse
  ensemble hedging picks the wrong clock when realized load is benign. So the lever is *forecast fidelity*,
  not *more hedging*.
- **No cell is headline-safe (PR #117/#118):** every MPC arm sits **above** the SLA-aware baseline's
  SLA-violation rate (`pareto_sla_not_worse=False`). The binding constraint is **SLA**, which is driven by
  **queue pressure** — and queue pressure is *emergent*, not a planner forecast input (`ABSENT_FORECASTS`).
- **PR #119 roofline promotion:** V1's GPU-blind scalar TPOT fabricates **phantom SLA violations** on a fast
  fleet → timing fidelity (RooflineState) directly inflates the measured SLA-violation rate that blocks the gate.

## Ranking

| rank | state | why it explains regret / non-headline-safety | expected impact | risk | scope |
|--|--|--|--|--|--|
| **1** | **ForecastState** | the oracle gap (+1580…+7520 gp/$) is *forecast* regret; today there is no persistent belief-vs-realized record to attribute or reduce it. Making belief + error + oracle/regret first-class is the instrument for every future forecast improvement | **High** — directly targets the measured oracle gap | Low (additive belief record, no reward change) | BUILD_NOW |
| **2** | **QueueState** (consolidate) | SLA (the gate-blocking term) is driven by queue pressure, which is *emergent, not a forecast input*. A canonical, populated QueueState (backlog, percentiles, SLA-slack, class mix) is the prerequisite for ever forecasting queue pressure → the path to Pareto-safety | **High (indirect)** — unblocks SLA-pressure forecasting | Low (populate existing placeholder from the realised replay) | EXTEND_NOW |
| **3** | **RequestState** | persistent per-request lifecycle enables true SLA attribution (which requests miss, where) and is the substrate QueueState/PlacementState consolidation needs; without it, SLA misses are only aggregate counts | Medium — better attribution, enables 1–2 | Low–Med (promote the Job lifecycle; conservation invariant) | BUILD_NOW |
| **4** | **RooflineState** (persist) | PR #119 showed timing fidelity changes the *measured* SLA-violation rate that blocks the gate; persisting the regime/timing record makes the timing-model choice and its SLA effect auditable per period | Medium — auditability of the SLA-inflation source | Low (snapshot existing `roofline_diag`) | EXTEND_NOW |
| **5** | **PlacementState** (per-request) | locality affects service time → SLA, but the replica-level placement is already canonical; per-request placement is a refinement that needs RequestState first | Low–Med | Low | EXTEND_NOW (after RequestState) |
| **6** | QualityState / DecodeState / NetworkState | already adequate (QualityState single-source; DecodeState two transient classifiers fold into RooflineState; NetworkState macro static) — no evidence they bind regret today | Low | — | KEEP_AS_IS / fold |

## What to build now (this PR)

**ForecastState (1)** and the **QueueState consolidation (2)** are the highest-leverage; **RequestState (3)**
is the enabling substrate; **RooflineState (4)** is a cheap promotion. PlacementState per-request and the rest
are EXTEND-after or KEEP_AS_IS. This matches the brief's priority (*ForecastState is the highest-priority missing
canonical state*) and the normalization rule (promote/consolidate; build only the genuine gaps).

**Honest caveat:** none of these states is expected to *flip* the Pareto gate by itself — the gate is blocked by
the base-MPC SLA property, not a single missing state. Their value is **regret attribution and the path to
SLA-pressure forecasting**, not a guaranteed headline. No headline gp/$ will be claimed unless a cell completes
and the gate passes.
