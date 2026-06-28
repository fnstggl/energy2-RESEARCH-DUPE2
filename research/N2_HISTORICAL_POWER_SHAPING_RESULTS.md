# N2 — Same-Window Online Serving Power-Shaping (Phase 4)

**N2 = the serving controller shaping GPU power *in place*, within the same window, in response to the realized
electricity price.** This is the price-aware DVFS path (`real_price_dvfs_only`): the MPC picks `clock_policy`
per decision and pays the realized energy×price, with serving running on the *same* periods. It is **NOT**
deferrable shifting — no work moves in time, nothing is delayed to a cheaper hour; the controller changes the
*power draw of work that runs now*. (Deferrable time-shifting is N-separate; see
`DEFERRABLE_ENERGY_SHIFTING_RESULTS.md`.)

This distinction matters for honesty: a "we saved energy cost" claim must say *how*. N2 is **power-shaping**
(clock → watts → energy on the current serving load); deferrable is **time-shifting** (move batch work to a
cheap hour). They are different mechanisms with different ledgers and are validated separately.

## What N2 changes vs. the price-unaware MPC

Source: the 54-cell backtest (`checkpointed_electricity_backtest.json`). `real_price_dvfs_only` and
`current_main_mpc_real_price` see the **same** real price path and the **same** clock candidate set
`{base, low, high}`; the only difference is `controller.electricity_price_aware`. So any change in the selected
clock is the controller responding to price **online, in the same window**.

| market | window | price $/kWh | price-unaware clock | **price-aware (N2) clock** | gp/$ Δ | SLA (N2 / unaware) |
|--|--|--|--|--|--|--|
| pjm | expensive | 0.092–0.178 | `base ×2, high ×1` | **`low ×3`** (downclock) | **+4859.2** | 0.529 / 0.525 |
| pjm | volatile | 0.076–0.178 | `base ×3` | **`low ×3`** (downclock) | **+5784.4** | 0.388 / 0.383 |
| ercot | expensive | 0.045–0.059 | `base ×2, high ×1` | **`base ×3`** (drop the upclock) | +1180.3 | 0.483 / 0.483 |
| caiso | expensive | 0.029–0.031 | `base ×3` | `high ×2, base ×1` (upclock — energy is cheap) | +1539.4 | 0.479 / 0.483 |
| ercot/caiso | volatile | 0.028–0.059 | `base`-ish | unchanged | 0.0 | equal |
| pjm/ercot/caiso | cheap | 0.021–0.086 | `base/high` | `high`-leaning (energy cheap) | −1.0k…−1.6k | ≈ equal |

**The N2 signal is real and price-directional:**

- **High price → downclock.** In PJM's expensive and volatile windows (p90 ≈ $0.18) N2 moves the whole window
  to `low` clock and gains **+4859 / +5784 gp/$ (≈ +1.5%)** with SLA essentially unchanged (0.529 vs 0.525;
  0.388 vs 0.383). The saving is the DVFS `power_scale` reduction on a decode-dominated load whose latency is
  clock-independent — power drops, GPU-time barely rises, energy×price falls.
- **High-ish price → drop the upclock.** In ERCOT's expensive window N2 removes the price-unaware arm's one
  `high` period (`→ base ×3`), shaving the most expensive watts (+1180 gp/$).
- **Cheap price → it does *not* downclock** (often leans `high`): when energy is a tiny cost fraction, N2
  spends it for throughput/SLA rather than shaving it. The gp/$ effect is then ~0 or slightly negative — the
  honest cost of a price signal that carries little information when prices are flat-and-low.

## Why this is N2 (power-shaping) and not deferrable

- **Same periods, same requests.** N2 replays the identical serving workload over the identical window; no
  request is delayed. The change is `clock_policy` on the periods that run — `PowerState.accumulate` records
  the per-period watts/energy under the chosen clock on the real timeline.
- **Deferrable is untouched here.** `real_price_dvfs_only` does **not** invoke the deferrable scheduler
  (`_DEFERRABLE_ARMS` excludes it). Its lift is entirely serving power-shaping.
- **Different ledger.** N2's effect is in `operator_cost`/`gp_per_dollar` (serving). Deferrable's effect is in
  a separate `electricity_cost` ledger over non-serving work. They never share a number.

## Honest limits

- **Magnitude is SIMULATOR_INFERENCE.** The DVFS power curve `power_w = TDP·(0.4 + 0.6·clock^2.4)` and the
  roofline regime (which work is clock-independent decode vs. clock-sensitive prefill) are modeled, not
  measured (see `ELECTRICITY_PRODUCTION_REALISM_AUDIT.md`). The *price path* is `TRACE_DERIVED` (PJM/ERCOT/CAISO
  day-ahead).
- **Not headline-safe.** N2's biggest win (+5784 gp/$ in `pjm|volatile`) still does not clear the Pareto gate:
  the MPC arm's SLA-violation rate exceeds the SLA-aware baseline's in every window (`pareto_sla_not_worse =
  False`). N2 is reported as a **DIRECTIONAL** within-MPC power-shaping lift, not a headline saving.
- **The `energy_kwh` diagnostic ≠ the billed energy.** The artifact's `energy_kwh`
  (`reference_power_w × realized_gpu_seconds`) is a coarse proxy on a different GPU-time basis than the cost
  model's energy term (`billable_gpu_hours × power_kw × power_scale × pue × price`); for a mixed prefill/decode
  load the two can diverge in sign. The gp/$ numbers above flow through the **cost-model** energy term, which
  is the one that drives the gate — `energy_kwh` is a label-only diagnostic.
