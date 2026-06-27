# Multi-Period (Receding-Horizon) MPC Architecture

This upgrades Aurelius's controller from a one-step optimizer into a genuine **finite-horizon,
receding-horizon Model Predictive Controller** over the persistent `CanonicalWorldState`. The goal
(per the milestone) is not to add knobs — it is to make the controller *physically correct over a
finite planning horizon*, so deferred-benefit actions (migration, prewarming, placement, and future
energy/DVFS/spec-decode) are scored through their *simulated future consequences* rather than
heuristic bonuses, and become worthwhile only when the rollout shows they pay off.

## The control loop

```
history → ForecastTrajectory[t+1 : t+H]            (causal; built from history only)
        → for each candidate FIRST action:
              clone ClusterState
              for step k in 0..H-1:
                  consume trajectory[k]
                  apply (candidate at k=0, base continuation at k>0)
                  advance the cloned world one step
                  accumulate γ^k · risk-adjusted reward
        → pick the first action with the best cumulative return
        → execute ONLY the first action on the real world; advance one step
        → re-plan from scratch next control interval
```

The real `ClusterState` is **never** mutated during candidate evaluation (every rollout runs on a
`clone`); only the chosen first action is committed. Future actions in the rollout are planning
artifacts, discarded and re-optimized each interval — standard receding horizon.

## Simulation clock — horizon is in STEPS, not hours (`simulation_clock.py`)

`H` is a number of **simulation steps**. The real lookahead is `H × dt_seconds`: `H=4` is 20 minutes
at `dt=300s`, 4 hours at `dt=3600s`. The controller reports `dt_seconds`, `horizon_steps`,
`lookahead_{seconds,minutes,hours}` for every decision. Supported control intervals: 60 / 300 / 900 /
3600 s. Per-plane native resolution is carried (serving sub-second, fleet/electricity hourly) so a
coarse plane and a fine plane are reconciled explicitly, never silently assumed equal.

## Forecast trajectory (`forecast_trajectory.py`)

The rollout consumes a forecast for **every** step, not just the next one. The existing
`ForecastingModel` already predicts `H` steps ahead recursively from history (feeding its own
predictions forward — causal, no future truth); `ForecastTrajectory` wraps it per-step, aligned to
the clock. Uncertainty is honest: `deterministic` (mean/p50 path) is the only mode when quantiles are
absent; `pessimistic`/`optimistic` use the forecaster's p90/p10 where they exist; **ensemble mode is
ABSENT** (no probabilistic sampler in the repo) — wrapped as deterministic, never fabricated.

## Reward, search, runtime

- **Reward**: cumulative discounted (`γ`, default 1.0) per-step risk-adjusted SLA-safe goodput/$ —
  every term explicit (serving cost, warm-hold, migration, topology), no hidden heuristics. The claim
  gate (`training.py`) is unchanged.
- **Search**: the connected bundle space (8748) is searched by coordinate descent (the same
  production search as before) over the FIRST action; each candidate is scored by its H-step rollout.
  Runtime budget: `max_candidate_bundles`, `max_horizon_steps`, `decision_timeout_s` (with a
  keep-best-so-far fallback). Diagnostics report theoretical vs evaluated bundles, world-steps
  simulated, runtime, and per-step credit assignment — no connected knob is silently excluded.
- **Complexity**: O(candidates × H) world-steps per decision. Coordinate descent keeps `candidates`
  ≈ 50–80; world-steps scale linearly in H.

## State cloning + backward compatibility

`clone_world_state_for_candidate` deep-copies the world; mutating a clone never touches the real
timeline. **`H=1` reproduces the single-period score exactly** (tested) — the multi-period path is a
strict generalization, and all existing stateless / persistent-world evaluations and the claim gate
are unchanged.

## What makes deferred actions discoverable — and the honest limits

Two world-state physics changes give deferred actions a *real* benefit channel (no heuristic bonus):

1. **Placement is driven by where warm replicas SIT.** The topology service-time factor reflects the
   macro network pressure of the racks the *warm* (servable) replicas occupy. `network_aware`
   activates the lowest-pressure warm replicas; if every warm replica is on a high-pressure rack there
   is no relief (no free lunch). This is the channel migration needs — physically MOVING replicas to a
   low-pressure rack creates low-pressure home racks for future placement to exploit.
2. **Warm persistence is TIME-based.** A replica stays warm until the calibrated ~300s idle timeout,
   measured in real time via `dt`. At a coarse interval (`dt ≥ timeout`, e.g. hourly) the pool cools
   every step (the PR-#102 behaviour, preserved); at a FINE interval (`dt < timeout`, e.g. 5-min) a
   warmed pool SURVIVES across steps. **This is the core reason deferred actions are multi-period
   decisions**: they only span periods when the control interval is shorter than the action's
   persistence timescale.

**Honest limitation (reported, not hidden).** On the current calibration the multi-period rollout
*correctly propagates* an action's future consequences — the gp/$ gap between e.g. prewarm and the
reactive baseline narrows monotonically as `H` grows (the machinery works) — but the deferred actions
do **not yet flip the decision** to "worthwhile" on the Azure trace, for two scoped reasons:
- **prewarm sizing is myopic** (it warms to the *current* step's forecast, so it does not pre-warm for
  a forecasted *future* ramp). Making prewarm size to the forecast trajectory's near-future peak is
  the next step.
- **migration vs cold-start-elsewhere** is not yet distinguished by warm KV-state identity (migration
  should preserve a replica's warm cache where a fresh cold-start would not). Tracking warm-replica
  identity through a move is the next step.

These are precisely scoped follow-ups, not heuristics to paper over. The controller itself is a
correct receding-horizon MPC; future infrastructure actions (energy shifting, DVFS, precision,
spec-decode, replica scaling) plug in by mutating `ClusterState` and inherit the rollout with no
controller redesign.

## Supported horizons & limitations

`H` is unbounded by design (capped only by `max_horizon_steps` for runtime safety). Practical horizons
1–24 are analyzed in `research/MPC_HORIZON_ANALYSIS.md` /
`research/MPC_CONTROL_INTERVAL_AND_HORIZON_ANALYSIS.md`. Uncertainty is deterministic-only (ensembles
ABSENT); the continuation policy is a fixed base policy (not a full inner optimization) — a deliberate
tractable choice for the rollout's future steps.
