# Electricity World-Model Gap Audit (Phase 0)

What is the current electricity path in Aurelius, and why does electricity contribute ~0% to planner
attribution (PR #114)? This audit precedes any implementation. Evidence is file:line from the live code on
`main` (this PR branches from stable `main`, not the regressed #115 token-shape branch).

## The 12 questions

**1. Which electricity datasets are already available?**
Real ISO **day-ahead** wholesale prices committed in `data/`: `pjm_us_east_dam.csv` (1753 hrs), `ercot_us_south_dam.csv`
(1728 hrs), `caiso_us_west_dam.csv` (1752 hrs) — schema `timestamp,region,price_per_mwh,…`, plus
`data/combined_2025_2026/` (DAM + RT, 2025-2026). Carbon: `data/watttime_carbon_q12026.csv` (gCO₂/kWh, us-west).
Loader ported from #115: `aurelius/environment/price_series.py` (→ $/kWh, percentile + diurnal helpers).

**2. Is PJM available?** Yes — `data/pjm_us_east_dam.csv` (us-east). p10 $0.026 / p50 $0.041 / p90 $0.281 / max $2.31 /kWh (spiky).

**3. Is ERCOT available?** Yes — `data/ercot_us_south_dam.csv` (us-south). p10 $0.010 / p50 $0.021 / p90 $0.045 / p95 $0.093 (spiky tail).

**4. Is CAISO available?** Yes — `data/caiso_us_west_dam.csv` (us-west). p10 $0.009 / p50 $0.032 / p90 $0.041 / p95 $0.048 (smooth).

**5. DA, RT, or both?** **Day-ahead (DAM)** is what `price_series.load_price_series` consumes. RT files exist in
`combined_2025_2026/` but are not loaded today. 5-minute real-time is **ABSENT** (`price_series.ABSENT_MARKETS`).

**6. How are regions mapped to markets?** A canonical registry already exists:
`aurelius/ingestion/region_registry.py` — `us-east→PJM`, `us-south→ERCOT`, `us-west→CAISO`, `us-central→SPP`
(`RegionMapping` carries iso, source_region, timezone, confidence). `FleetState.region` (schemas.py) selects it.

**7. Is electricity price currently used in cost?** **Yes, causally — but fed a constant.** `cost_model.operator_cost`
(cost_model.py:217-218): `energy = gpu_hours · power_kw(util) · power_scale · pue · energy_price_per_kwh`. The
price is wired, but `training.py:141` builds `price_by_cycle = {c: fleet.energy_price_per_kwh for c in range(cycle_len)}`
— a **single constant** across the whole horizon and window. That constant is the root cause of "0%".

**8. Is power modeled per GPU / request / phase?** Partially, as a **diagnostic** (not persistent state).
`PeriodOutcome.power_w / energy_j` (world_simulator.py:447-459) come from the roofline action modulation
(`power_w = TDP·(0.4 + 0.6·clock^2.4)`); per-phase prefill/decode service times are roofline-derived
(`roofline.py`). These populate **only under non-neutral roofline actions** (neutral base clock → 0). There is no
first-class `PowerState`.

**9. Is clock/DVFS currently a real MPC action?** **Yes — CONNECTED, live.** `actions.py` `clock_policy ∈
{base, low, high}`, `reward_channel="roofline_serving"`, mapped via `CLOCK_TO_ROOFLINE = {base:1.0, low:0.85,
high:1.15}`. It already flows clock → power → energy → cost AND clock → decode/prefill latency → SLA. (The old
ROADMAP "N2 DVFS planned/null" note is **stale** — PR #111 wired it; PR #115 Track D/E confirmed the controller
selects `low` clock and responds to price.)

**10. Is deferrable work represented?** **No.** `unified_replay` has `CLASS_LATENCY` / `CLASS_BEST_EFFORT`, but
best-effort is a **5-minute-SLA serving class**, not deadline-driven shiftable work. The `Job` dataclass carries
only `cls` — no deadline, shift-window, or batch identity. There is **no** training/embeddings/batch/maintenance
workload. → a conservative **synthetic, SIMULATOR_INFERENCE-labeled** deferrable generator must be built.

**11. Why did electricity attribution show ~0%?** Three compounding reasons, all measured:
- **Constant price in planner frames** (training.py:141) → no temporal price signal to exploit.
- **Energy is a small share of owned-GPU cost** at $0.04-0.06/kWh (depreciation ≈ $0.68/GPU-hr dominates), so the
  clock's ~19% power cut moves total cost only ~0.3-3% — below what reorders the action ranking vs precision/spec.
- **No deferrable work**, so there is nothing to time-shift toward cheap hours.
PR #115 Track E proved the pathway exists: injecting the real PJM p90 price shifted the live MPC's downclock
fraction **0.0 → 0.5**. Electricity is latent, not unimportant.

**12. Which missing actions prevent electricity from becoming decision-relevant?**
1. **Real time-varying prices in the normal planner frames + horizon rollout** — today the rollout reads price
   only at step 0 (`controller.py` `pr = bundle.at("electricity_price", 0)`) and every cost path uses the constant
   `fleet_state.energy_price_per_kwh`. This is the single highest-value gap (Phase A).
2. **DeferrableWorkState + energy-shifting actions** — no way to move flexible work to cheap hours.
3. **Per-period realized price in `run_period_episode` / `simulate_period`** — so realized cost (not just planning)
   reflects the diurnal price.

## What this audit authorizes (honest scope)

Buildable now, causally and Pareto-safely: **ElectricityState** (real prices + region registry), **real diurnal
price frames + horizon price-path consumption** (opt-in; flat price reproduces today exactly), **PowerState**
(formalise the existing power/energy accounting), **price-responsive DVFS validation** (the action already exists),
**DeferrableWorkState + energy shifting** (conservative synthetic, clearly labelled), **electricity-aware
economic objective** (already in cost; formalise + per-period price), **historical PJM/ERCOT/CAISO backtests**.

Out of honest scope this PR (documented in `ELECTRICITY_PRODUCTION_REALISM_AUDIT.md`, not faked): dynamic
cooling/COP, utility **demand charges**, GPU **power-caps** (clock-locking is the modelled lever), carbon as a
reward term (data present but unwired — kept ABSENT), region-shifting of deferrable work (no multi-region fleet
model), and 5-minute real-time prices (absent). Branched from stable `main`; the token-shape forecaster is **not**
ported and is **not** the default planner.
