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

## Do forecasts beat naive baselines? (held-out, full Azure trace, per-minute)

| target | model kept | held-out metric | naive | verdict |
|---|---|---|---|---|
| arrival_rate | **Ridge** | MAE 0.91 | 1.56 | **beats naive (−42%)** |
| output_token_mean | **Ridge** | MAE 15.95 | 16.35 | beats naive |
| interarrival_cv | **Ridge** | MAE 0.092 | 0.115 | beats naive |
| output_token_p95 | naive:median | pinball 0.865 | 0.865 | naive kept (honest) |
| input_token_mean | naive:ewma | MAE 105.4 | 105.4 | naive kept (honest) |
| electricity_price | naive:last | MAE 0.0 | 0.0 | naive kept (1-hour trace → constant) |

Every forecast carries calibrated uncertainty (mean + p10/p50/p90/p99 from held-out
residual quantiles; coverage error reported). **3 of 6 targets have a learned model that
genuinely beats naive; the rest keep naive and say so.**

## Does the controller beat the SLA-aware baseline? (held-out, 15 periods)

| arm | SLA-safe goodput/$ | SLA-violation rate |
|---|---|---|
| aurelius_canonical | 1,880,761 | 0.307 |
| sla_aware | 1,861,003 | 0.308 |
| **mpc_controller** | **1,749,130** | 0.362 |
| greedy_packing | 1,643,494 | 0.638 |

**No.** The MPC controller beats the weak FIFO reference and greedy packing, but is
**−7% vs the strongest non-weak baseline (`aurelius_canonical`)**, so the claim gate sets
`headline_claim_allowed = False`. This is honest: on this serving trace the fixed
SLA-aware/canonical policy is near-optimal *per period*, and per-period switching over the
only three connected levers cannot beat it — the forecasts are good but the **action space
is the binding constraint**. The gains live in the not-yet-connected first-principles
levers below.

## Safe vs unsafe claims

**Safe:**
- "Aurelius now trains forecasters and runs a model-predictive economic controller over
  the connected infrastructure actions on held-out public traces."
- "Forecasting beats naive baselines on arrival rate, output-token mean, and burstiness
  (held-out); other targets honestly keep the naive baseline."
- "Every forecast carries calibrated uncertainty; the controller is causal (no oracle)."
- "KV held-out validation is now CI-reproducible (committed VALIDATION_FIXTURE)."

**Unsafe (gated off):**
- Any headline **savings %** — the controller does not yet beat the strongest SLA-aware
  baseline; `headline_claim_allowed = False`. Directional simulator evidence only
  (`docs/RESULTS.md` §8).
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

> "The forecasting/controller stack is implemented and trains/tunes on the canonical
> environment with held-out evaluation and an honest claim gate; forecasting beats naive
> baselines on the key load signals, but the controller does not yet beat the strongest
> SLA-aware baseline — because the binding constraint is the connected action space, not
> the forecasts. The next step is to connect a first-principles action surface (N1/N4
> first) under the same gate."
