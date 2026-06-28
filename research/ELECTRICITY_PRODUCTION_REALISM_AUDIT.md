# Electricity Production-Realism Audit (Phase 0B)

What would a *real* production electricity controller need, and what can Aurelius model **honestly** today vs
what must be **labelled** or **skipped rather than faked**? This gates implementation (Phase 0 = the gap audit;
this = the realism/honesty audit). Hard rule: **no simplified electricity mechanism that creates fake savings —
model conservatively or label the limitation.**

## Per-component verdict

| component | exists today? | public data | missing prod telemetry | dangerous simplification | build honestly now | label simulator-inferred | skip/defer |
|--|--|--|--|--|--|--|--|
| **ElectricityState** | price loader (#115), region registry | PJM/ERCOT/CAISO DAM ($/MWh, hrly) | live LMP feed, nodal vs hub basis | treating one hub price as fleet-wide | ✅ price path, percentile, spike, forecast/realized, provenance | forecast path (FORECAST_DERIVED) | 5-min RT (absent) |
| **PowerState** | `power_w`/`energy_j` diagnostic on PeriodOutcome | GPU TDP specs | per-GPU power telemetry | claiming exact watts | ✅ formalise idle/active/clock→watts→J→kWh→$ | the DVFS power curve (`0.4+0.6·clock^2.4`) | per-rack thermal |
| **Clock/DVFS action** | **CONNECTED live action** (`clock_policy`) | DVFS/LLM-energy papers | real freq→latency on prod GPUs | "downclock is free" everywhere | ✅ validate price-responsiveness over real prices | memory-bound `decode_factor=1.0` (upper bound) | per-kernel freq |
| **DeferrableWorkState** | **absent** (best-effort ≠ deferrable) | none (no batch/training trace) | real deferrable job mix & deadlines | inventing high-value flexible work | ✅ persistent state + conservative generator | **whole workload (SIMULATOR_INFERENCE)** | real batch trace |
| **Energy shifting action** | absent | — | real shiftability/SLAs | "free" shifting, deleting work | ✅ run/delay/run-when-cheap, deadline-penalised | value scales with the synthetic mix | region shift |
| **Economic objective** | electricity in `operator_cost` | ISO prices | real $/GPU-hr blend | hiding energy share | ✅ per-period price; energy = kWh×$/kWh | depreciation/PUE constants | demand charges |
| **Region mapping** | `region_registry.py` (PJM/ERCOT/CAISO/SPP) | registry + price CSVs | datacenter→node basis | nodal as hub | ✅ region→market→price | — | multi-region fleet |
| **Carbon / emissions** | data ingested, **not wired** | `watttime_carbon_q12026.csv` (us-west only) | full-region MOER | carbon as a free reward | ⚠️ keep **ABSENT** (1 region, unwired) | — | ✅ defer (out of scope) |
| **Cooling / PUE** | **flat 1.3×** constant | ASHRAE/PUE refs | inlet temp, chiller COP | dynamic COP from weather = fake precision | ❌ keep flat 1.3× (conservative) | the 1.3× constant | ✅ dynamic cooling |
| **Power caps** | **absent** (clock-lock is the lever) | GPU TDP | real power-limit behaviour | power-cap "savings" on memory-bound decode | ❌ — clock-locking models the decode-energy lever | — | ✅ power-cap action |
| **Demand charges** | **absent** | — | utility tariff, 15-min peak | $/kW-month peak savings (huge, unverifiable) | ❌ — | — | ✅ skip (would be fake) |
| **Data residency** | absent | — | per-job residency constraints | free cross-region shift | ❌ — | region-eligibility flag only (no shift) | ✅ region shift |
| **Serving SLA interaction** | Pareto gate + SLA replay | Azure trace | prod SLO telemetry | letting deferrable steal serving capacity | ✅ serving SLA dominates; gate unchanged | — | — |

## The four questions the hard rule forces

**What would be dangerous to simplify (and is therefore NOT done)?**
- **Demand charges** — a $/kW-month peak term would dwarf energy cost and is trivially gamed; modelling it from a
  bounded sim would manufacture the headline. **Skipped.**
- **Dynamic cooling COP from weather** — would add a precise-looking but unvalidated multiplier. **Kept flat 1.3×.**
- **Power-caps on memory-bound decode** — a cap that never engages (decode draw < cap) would book phantom savings.
  The honest lever for decode energy is **clock-locking**, which is what we model.
- **High-value deferrable work** — inventing lucrative flexible jobs would create savings out of thin air. The
  generator is **conservative** (modest energy, real deadlines, penalised misses) and **SIMULATOR_INFERENCE**.

**What can be implemented honestly now?** ElectricityState; real diurnal price frames + horizon price path
(opt-in, flat-price-identical); PowerState (formalising existing accounting); price-responsive DVFS (validate the
existing action); DeferrableWorkState + energy shifting (conservative, labelled); per-period price in the economic
objective; PJM/ERCOT/CAISO backtests; electricity diagnostics + attribution re-run.

**What must be labelled simulator-inferred?** The DVFS power curve, the memory-bound `decode_factor = 1.0`
clock-independence (an **upper bound** on downclock attractiveness — real GPUs cost some latency), the **entire
deferrable workload** (no real trace), and the PUE/depreciation constants.

**What is skipped rather than faked?** Demand charges, GPU power-caps, dynamic cooling, carbon-as-reward (data is
one-region and unwired), region-shifting of deferrable work (no multi-region fleet), and 5-minute RT prices.

## Provenance discipline

`TRACE_DERIVED` — historical PJM/ERCOT/CAISO prices. `FORECAST_DERIVED` — the price forecast path. `SIMULATOR_INFERENCE`
— DVFS power curve, deferrable workload, fallback/default prices. Every savings claim must trace to **energy ×
price** with **SLA Pareto-safe** and **no missed-deadline cheating**; flat prices must yield **no** shifting value.
