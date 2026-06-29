# World-Model Robustness Audit (Phase 7)

Classifies every simulator component the N2 / all-knobs claims depend on into three honesty tiers, so no number
is read as more real than its weakest input. **No component is UNKNOWN.** Tiers:

- **ROBUST_ENOUGH_FOR_CURRENT_CLAIM** — tested, causal, and faithful enough to support a *directional simulated*
  claim (never a production guarantee).
- **DIRECTIONAL_SIMULATOR_INFERENCE** — a modeled relationship with the right sign/shape but an uncalibrated
  magnitude; fine for "which way / roughly how much," not for a headline number.
- **NEEDS_PRODUCTION_TELEMETRY** — cannot be validated offline at all; requires real measurement.

## ROBUST_ENOUGH_FOR_CURRENT_CLAIM

| component | why robust | evidence |
|--|--|--|
| Electricity **price path** (PJM/ERCOT/CAISO day-ahead) | real published $/MWh → $/kWh; deterministic load + unit conversion | `price_series.load_price_series`; tested in `test_electricity_controller` |
| **Cost flows only through energy = gpu_hours·power_kw·power_scale·pue·price** | single audited cost path; flat price reproduces pre-electricity behaviour byte-for-byte | `cost_model.py:217-218`, `world_simulator.py:439-443` |
| **Pareto / claim gate** (SLA-not-worse) | unchanged; blocks any SLA-shedding headline | `training.claim_gate`; every N2 cell reports it |
| **Work conservation + no future leakage** | causal rollout (`frames[:p]`); deferrable jobs conserved + missed-deadline-penalised | `controller.run_period_episode`, `deferrable.py` |
| **SLA-slack computation** (latency-class completion tail vs target) | derived from the same per-job replay latencies the SLA-violation count uses | `unified_replay.py` (this PR); `test_n2_power_arbitrage` |
| **N2 is a decomposition, not a bonus** | `n2_active` requires a downclock that saved cost AND stayed Pareto-safe; reward byte-identical | `n2.py`; `test_n2_power_arbitrage` |

## DIRECTIONAL_SIMULATOR_INFERENCE

| component | what's modeled | the caveat |
|--|--|--|
| **DVFS power curve** `power_w = TDP·(0.4 + 0.6·clock^2.4)` | clock → power (the engine of N2's dollar saving) | the exponent/offset are a conservative public-roofline prior, not measured silicon |
| **Memory-bandwidth-bound decode = clock-independent latency** | downclocking decode is ~free (the N2 upper bound) | real decode has *some* clock sensitivity → this **over-states** N2's free lunch; N2 value is an **upper bound** |
| **Completion-latency tail model** | TTFT + decode work from the roofline `serving_point` | a queueing+roofline approximation, not measured per-request latency |
| **Clock factors** `{base:1.0, low:0.85, high:1.15}` | the DVFS operating points | a modest, fixed grid — real DVFS is continuous and GPU-specific |
| **Deferrable workload** | synthetic conservative pool + price-aware scheduler | no real deferrable trace exists (separate ledger; never a serving headline) |
| **Spec-decode acceptance / int4 quality risk / batching service factor** | roofline action factors | uncalibrated magnitudes; signs are physically grounded |

## NEEDS_PRODUCTION_TELEMETRY

| component | why it cannot be validated offline |
|--|--|
| Real GPU **power telemetry** (DCGM) under each clock | the DVFS curve is the single biggest magnitude assumption behind every electricity dollar; only nvml/DCGM closes it |
| Real **per-request output-length** | drives prefill/decode mix → the roofline regime → whether N2 downclock is free; planning uses a causal running-median prior, not truth |
| Real **cache-hit / KV-reuse** rates | planning uses synthetic unique prefixes (PR #112) → KV reuse is not a consumed forecast |
| Real **thermal / power-cap** behaviour | the model has no thermal throttling; a real cluster's clock is also thermally constrained |
| **True demand charges / contracted tariffs** | only day-ahead energy price is modeled; real bills include demand ($/kW) and contract terms |

## What this means for the claims

- N2's **direction** (downclock memory-bound decode at high price to cut electricity cost, Pareto-safe) is
  **ROBUST** — it rests on the audited cost path + the gate + work conservation.
- N2's **magnitude** (the gp/$ and the dollars saved) is **DIRECTIONAL** and an **upper bound**, gated by the
  DVFS power curve and the clock-independent-decode assumption — both `NEEDS_PRODUCTION_TELEMETRY` to tighten.
- Therefore: **no headline N2 goodput/$ saving is claimed**; the value is reported as a directional simulated
  decomposition, and the highest-value next step is real GPU power telemetry (closes the dominant assumption).
