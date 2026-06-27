# Aurelius Forecasting + Model-Predictive Economic Controller

This PR turns the canonical environment from a fixed-policy backtest substrate into a
debuggable **forecast → simulate → choose** stack: a forecasting layer (with
uncertainty) and a model-predictive economic controller over the *connected* actions,
trained/tuned with strict no-leakage splits and an honest claim gate. It is **not** a
deep-RL system — that is deliberate (lookahead search over a small connected action set,
fully inspectable).

## Architecture — three separable layers

1. **Environment** (`canonical.py`, merged): the two-clock multi-plane substrate
   (Azure serving · v2026 fleet · Mooncake KV · ISO electricity). Produces the real
   per-period serving load + the anchored fleet state + the operator cost.
2. **Forecasting** (`forecasting.py`): fits causal predictors of the next period's load
   from past periods, with **uncertainty on every target**.
3. **Controller** (`controller.py`): each period, forecasts the next load, simulates
   each candidate connected action on that forecast, and chooses the best risk-adjusted
   SLA-safe goodput/$ — falling back to the SLA-aware action when confidence is low.
4. **Training** (`training.py`): fits forecasters (train), tunes the controller (val),
   evaluates vs baselines (held-out), and gates the claim.

## What is LEARNED vs SIMULATED

- **Learned (from train data only):** the forecaster ladder — naive (last/median/EWMA/
  seasonal) → linear (Ridge/ElasticNet) → boosted trees (LightGBM, else sklearn
  HistGradientBoosting; quantile-capable). A learned model is kept **only if it beats the
  best naive baseline on the held-out split**; else the naive baseline is kept. The
  controller's hyper-parameters (horizon, risk weight, confidence floor) are tuned on a
  disjoint val split.
- **Simulated:** the controller scores candidate actions by simulating the **forecasted**
  load through `unified_replay` + the operator cost model. These are SIMULATED expected
  outcomes, not measured production behaviour.

## Connected vs not-yet-implemented actions

- **Connected (the controller chooses among these):** `capacity` (reactive_lag1 /
  backlog_aware / forecasted_mcs), `ordering` (fifo / abs_conformal), `admission`
  (off / class_aware).
- **Not offered (recorded in the roadmap, never faked):** KV-aware routing
  (SIMULATED_ONLY — exists but not wired into serving), placement/packing, migration,
  prewarming, DVFS/clock, precision/model routing, speculative decoding, energy/price-aware
  shifting. See the Phase-0 audit (`AURELIUS_FORECASTING_CONTROLLER_AUDIT.md`).

## Do forecasts beat naive baselines? (held-out, full 2024 one-week Azure trace, HOURLY)

This now runs on the **2024 one-week** Azure trace (168 hourly periods, 27,303,999 requests
— see `research/AZURE_TRACE_COVERAGE_AUDIT.md`), not the 2023 one fleet hour. Held-out
hourly metrics (forecasters fit on the first 100 h, evaluated on the held-out tail;
`data/external/mpc_controller/trained_forecasters.json`):

| target | model kept | held-out metric | naive | verdict |
|---|---|---|---|---|
| arrival_rate | **Ridge** | MAE 0.144 | 0.183 | **beats naive (−21%)** |
| output_token_mean | **Ridge** | MAE 1.91 | 2.48 | **beats naive (−23%)** |
| input_token_mean | **Ridge** | MAE 27.1 | 38.7 | **beats naive (−30%)** |
| interarrival_cv | **Ridge** | MAE 0.0075 | 0.0078 | beats naive (−3%) |
| output_token_p95 | naive:seasonal | pinball 1.41 | 1.41 | naive kept (honest) |
| electricity_price | naive:last | MAE 0.0 | 0.0 | naive kept (anchored → constant) |

Every forecast carries calibrated uncertainty (mean + p10/p50/p90/p99 from held-out
residual quantiles; coverage error reported). **4 of 6 targets now have a learned model
that genuinely beats naive** — the real **diurnal** structure (a 24-hour cycle that simply
did not exist in the 1-hour trace) is what the learner exploits; the rest keep naive and
say so. The seasonal features auto-detect the cycle (`cycle_len 24` hourly / `60`
per-minute), so this is the same honest ladder, now on a week of real data.

## Does the controller earn a headline? (held-out, 42 hourly periods)

| arm | SLA-safe goodput/$ | SLA-violation rate | queue p95 |
|---|---|---|---|
| **mpc_controller** | **184,092** | **0.0385** | 58.8 s |
| fifo_weak (weak ref) | 180,930 | 0.1068 | 167.4 s |
| aurelius_canonical (fair baseline) | 180,877 | 0.0178 | 21.9 s |
| sla_aware | 177,819 | 0.0160 | 8.3 s |
| greedy_packing | 174,501 | 0.1397 | 257.2 s |

**No — and the gate now says so for the right reason.** On the full week the controller's
SLA-safe goodput/$ is **+1.78% vs the strongest fair baseline (`aurelius_canonical`)** — but
it gets there by running leaner and **letting 2.2× more requests miss the SLA** (0.0385 vs
0.0178) with far worse tail latency (58.8 s vs 21.9 s). It is *cheaper, not safer*, and is
**never Pareto-better** than the SLA-aware/canonical policies.

The claim gate therefore now includes a **Pareto clause**: a goodput/$ edge bought with a
higher SLA-violation rate is not a headline. → `pareto_sla_not_worse = False` →
`headline_claim_allowed = False` (`data/external/mpc_controller/evaluation_report.json`).

The edge is also **not robust** — it is an artefact of the simulated load operating point
and the decision-sim window, and **flips sign** across reasonable choices:

| operating point | mpc vs fair baseline (gp/$) | headline |
|---|---|---|
| stride 18 (heavier load, ~2.5 req/s) | +1.37% | — (fails Pareto) |
| stride 24 (≈4-GPU capacity, ~1.9 req/s), sim 240 s | +1.78% | — (fails Pareto) |
| stride 24, sim 600 s (more faithful decision sim) | **+0.55%** | — (fails Pareto) |
| stride 36 (lighter load, ~1.3 req/s) | **−0.70%** | False (loses outright) |

The margin shrinks as the decision sim becomes more faithful (1.78% → 0.55%) and reverses
at lighter load — so even ignoring SLA, there is no robust per-period win over the three
connected levers. **The forecasts are now genuinely good (4/6 beat naive on real diurnal
data); the binding constraint remains the connected action space.** The gains live in the
not-yet-connected first-principles levers below.

## Safe vs unsafe claims

**Safe:**
- "Aurelius trains forecasters and runs a model-predictive economic controller over the
  connected infrastructure actions on the full **one-week** Azure trace (168 hourly
  held-out periods)."
- "Forecasting beats naive baselines on the **diurnal** arrival rate, output-token mean,
  input-token mean and burstiness (held-out hourly, 4/6 targets); the rest honestly keep
  naive — a result only possible once the week of real data was wired."
- "Every forecast carries calibrated uncertainty; the controller is causal (no oracle)."
- "The claim gate is Pareto-aware: a goodput/$ edge bought with more SLA violations does
  not count — and on this trace it is correctly blocked."
- "KV held-out validation is CI-reproducible (committed VALIDATION_FIXTURE)."

**Unsafe (gated off):**
- Any headline **savings %** — the controller's +1.78% goodput/$ comes with a higher
  SLA-violation rate (fails the Pareto clause) and is **not robust** (flips sign across
  operating points); `headline_claim_allowed = False`. Directional simulator evidence only.
- "The controller is better than the SLA-aware baseline" — it is *cheaper, not safer*;
  never Pareto-better.
- "Production-grade" / use of DVFS/precision/placement/migration/fleet-KV-routing — those
  actions are NOT connected (roadmap only).

## Next action surfaces (first-principles roadmap)

Each is a controller-ready action; this PR builds the controller architecture that can
accept them. Implement one per PR, shadow-sim first, behind the same claim gate.

| lever | required state | required action | simulator change | validation | fair baseline | kill criterion |
|---|---|---|---|---|---|---|
| **N1** roofline-economic | arithmetic-intensity / memory-vs-compute-bound indicator; tokens/joule; $/joule; $/GPU-s | batch-composition + the existing levers under a roofline-aware law | derive bottleneck from tokens×batch×GPU BW/FLOPs; expose tokens/joule | tokens/joule vs measured ceiling | sla_aware + best fixed | no gp/$ gain over sla_aware on held-out |
| **N2** power-shaping | SLA slack; electricity price; per-clock energy/perf curve | `clock_factor` (DVFS) per period/phase | service-time ↑ and power ↓ (convex) with clock; decode-phase lock | energy vs latency band; price-sensitivity | fixed max-clock | no $/token reduction at equal SLA |
| **N3** anti-correlation | per-phase (prefill/decode) power profiles; rack power envelope; oversubscription headroom | co-schedule anti-correlated phases | rack peak-power constraint + phase power model | manufactured headroom vs peak cap | no-oversubscription | violates power envelope or no headroom gain |
| **N4** fleet-global KV | per-server cache state (exists) | content-addressed cross-server routing (wire `KVAwareRouter` into serving) | one `StatefulKVCache` per server in the serving loop | fleet hit-rate vs per-node baseline | per-node cache | fleet hit-rate ≤ per-node |
| **N5** adaptive fidelity | difficulty/quality proxy; quality constraint | precision / model-route action | service/quality model per precision | quality within constraint; price-biased | static precision / RouteLLM | quality breach or no $/quality gain |
| **N7** speculative scheduling | arrival + prefix-reuse forecast (forecasters exist) | prewarm / pre-position action | warmup state + pre-position cost | vs running-median forecast baseline | reactive (no prewarm) | prewarm cost > avoided cold-start |

(N6 grid-revenue and N8 per-chip perf/watt routing remain REQUIRES_PILOT_TELEMETRY.)

## Readiness statement

Per the success criteria, the honest statement Aurelius can make today is **#2**:

> "The forecasting/controller stack is implemented and trains/tunes on the **full one-week**
> Azure trace (168 held-out hourly periods) with an honest, Pareto-aware claim gate.
> Forecasting now beats naive on the real diurnal load signals (4/6 targets), but the
> controller does not earn a headline — its goodput/$ edge is small, not robust across
> operating points, and bought with a higher SLA-violation rate (cheaper, not safer), so
> the gate stays False. The binding constraint is the connected action space, not the
> forecasts; the next step is to connect a first-principles action surface (N1/N4 first)
> under the same gate."

_Updated for the 1-week-Azure / hourly rerun: see `research/AZURE_TRACE_COVERAGE_AUDIT.md`
for the coverage audit that motivated wiring the full week._
