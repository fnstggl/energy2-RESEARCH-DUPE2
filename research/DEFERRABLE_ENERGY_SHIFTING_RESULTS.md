# Deferrable Energy-Shifting Results (Phase 5)

Deferrable (batch / offline / training / maintenance) work can move **in time** to run when electricity is
cheap, as long as deadlines are met. This is **time-shifting**, distinct from the same-window power-shaping of
N2 (`N2_HISTORICAL_POWER_SHAPING_RESULTS.md`). Aurelius has **no real deferrable-work trace**
(`ELECTRICITY_PRODUCTION_REALISM_AUDIT.md`), so the workload is `SIMULATOR_INFERENCE`: a conservative
deterministic pool (`deferrable.generate_deferrable_pool`) scheduled by the price-aware look-ahead policy
(`deferrable.schedule_deferrable`). The price path itself is `TRACE_DERIVED` (real day-ahead PJM/ERCOT/CAISO).

**Honest by construction** (`deferrable.py`): work is **conserved** (a delayed job waits, never deleted); a
**missed deadline is penalised** (shifting cannot dodge work for free); **serving SLA dominates** (deferrable
runs only on spare GPU-seconds and adds its energy×price to cost); with a **flat price**, price-aware and
run-asap produce the **same** cost (no fake shifting value).

## Result 1 — serving is never touched (from the 54-cell backtest)

In the historical backtest the deferrable arms (`real_price_deferrable_only`,
`real_price_dvfs_plus_deferrable`) produce **serving gp/$ identical to the no-deferrable arm in all 9
(market, window) cells** (`deferrable_serving_gp$_delta = 0.0` everywhere). The scheduler consumes only the
GPU-seconds serving leaves spare, so it **cannot** raise serving gp/$ or steal serving capacity. This is the
key safety property: deferrable value is a **separate energy-cost ledger**, never folded into serving gp/$.

(The in-backtest ledger runs over only the 3-period serving-decision window — too short for the look-ahead to
find a strictly-cheaper period inside each job's deadline. It therefore either finds none → $0 saving with
deadlines met, or would shift into a deadline miss. The *magnitude* of valid shifting is measured over a proper
horizon below.)

## Result 2 — valid, deadline-respecting shifting saving (dedicated 24 h validation)

`scripts/validate_deferrable_shifting.py` runs the **same** scheduler over the **full 24-hour real diurnal
price path** with ample spare capacity (so the only difference between `asap` and `price_aware` is *when* a job
runs — isolating the timing→price effect). Artifact:
`data/external/mpc_controller/deferrable_shifting_validation.json`.

| market | price p10 / p90 $/kWh | asap cost | price-aware cost | **saving** | shifted | missed | deadlines |
|--|--|--|--|--|--|--|--|
| ercot | 0.0097 / 0.0453 | $0.79901 | $0.64403 | **$0.15498 (19.4%)** | 18 | **0** | respected ✅ |
| caiso | 0.0090 / 0.0409 | $0.54253 | $0.51313 | **$0.02940 (5.42%)** | 24 | **0** | respected ✅ |
| pjm | 0.0259 / 0.2812 | $1.76264 | $1.68012 | **$0.08252 (4.68%)** | 18 | **0** | respected ✅ |

- **All three markets realise a strictly-positive saving with 0 missed deadlines** — work is fully completed,
  just moved to cheaper hours within each job's slack.
- **The saving scales with each market's *diurnal spread*, not its price level.** ERCOT has the widest
  within-day swing (cheap overnight vs. expensive evening) → the **largest % saving (19.4%)**, even though its
  absolute prices are low. PJM's high absolute prices give the largest *absolute* swing within a job's deadline
  window but a more modest profile-averaged %.
- **Flat-price control (no fake shifting):** with a constant price equal to each market's mean, `price_aware`
  cost **exactly equals** `asap` cost in all three markets (`no_fake_shifting = True`) — the saving above is
  genuinely the diurnal shape, not an artefact of the policy.

## Result 3 — serving protected when there is no spare (from PR #116 smoke)

The fixture smoke (`electricity_smoke_{pjm,ercot,caiso}.json`, P6) confirms the dominance rule directly: with
**no** spare capacity, deferrable jobs are **not** run (0 completed, all 8 missed-and-penalised) rather than
stealing serving capacity. Deferrable never wins by degrading serving.

## Honest bottom line

- **Deferrable energy-shifting is a real, deadline-respecting saving** on the deferrable workload:
  **4.7% (PJM), 19.4% (ERCOT), 5.4% (CAISO)** of the run-asap energy bill, 0 deadlines missed, and **$0 saving
  under a flat price** (no fabrication).
- It contributes **0 to serving gp/$** by construction and is reported as a **SIMULATOR_INFERENCE energy-cost
  ledger** (the workload is synthetic; only the price path is trace-derived). It is **never** a serving-gp/$
  headline.
- Promoting deferrable into a true MPC-selected `ActionBundle` knob (so the planner co-schedules serving vs.
  deferrable in one objective) remains **deferred** — see `ELECTRICITY_MPC_ACTION_SURFACES.md` §B.
