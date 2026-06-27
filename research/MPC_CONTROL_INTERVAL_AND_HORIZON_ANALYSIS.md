# MPC Control Interval & Horizon Analysis

The MPC horizon `H` is a number of **simulation steps**, and the real lookahead is `H × dt_seconds`.
This document records what `dt_seconds` (the control interval) and `H` (the step horizon) mean, and
the held-out horizon ablation.

## Control interval (`dt_seconds`) vs horizon (`H`)

| dt_seconds | H=1 | H=4 | H=8 | H=24 |
|---|---|---|---|---|
| 60 | 1 min | 4 min | 8 min | 24 min |
| 300 | 5 min | 20 min | 40 min | 2 h |
| 900 | 15 min | 1 h | 2 h | 6 h |
| 3600 | 1 h | 4 h | 8 h | 24 h |

`H=4` is **not** "4 hours" — it is four control steps. The serving/control loop supports 60 / 300 /
900 / 3600 s; the fleet/electricity planes are hourly. The controller reports `dt_seconds`,
`horizon_steps`, `lookahead_seconds/minutes/hours`, candidate bundles evaluated, world-steps
simulated, and runtime per decision (`controller.last_decision_diag`).

**Why the control interval matters most.** Deferred-benefit actions (prewarm, migration) only span
periods when `dt_seconds < ` the action's persistence timescale (the ~300s warm idle timeout). At
hourly `dt` a warmed pool cools before the next step, so these actions cannot span periods regardless
of `H`; at minute-scale `dt` a warmed pool survives, so the horizon can capture the payoff. The clock
makes this explicit rather than assuming one hour.

## Held-out horizon ablation (Azure 2024 week, persistent world)

`scripts/sweep_mpc_horizon.py` — train once, run the world-state MPC on the held-out periods at each
`H`, commit only the first action each interval. `H=1` is the single-period anchor.

| H | lookahead | reward (gp/$) | SLA viol | GPU-h | queue p95 | runtime/decision | gate (beats/pareto/headline) |
|--:|--:|--:|--:|--:|--:|--:|---|
| 1 | 60 min | 111,079 | 0.0220 | 56.2 | — | 0.132s | True / False / **False** |
| 2 | 120 min | 111,791 | 0.0210 | 56.1 | — | 0.202s | True / False / **False** |
| 4 | 240 min | 111,665 | 0.0208 | 56.2 | — | 0.341s | True / False / **False** |

Fair baseline (aurelius_canonical_kv_routing, 1.0× capacity): gp/$ 107,152, SLA 0.0143. Capacity
mix is 0.75× ×42 at every H (lean). Larger H slightly improves SLA (0.0220→0.0208 — the rollout
anticipates future risk) at ~linear runtime; gp/$ is ~flat (diminishing returns by H=4); the gate
stays blocked (lean capacity's SLA stays above the 1.0× fair baseline). Stride-96 sample of the week.

Forecast alignment: the trajectory is built from the fitted forecaster, which predicts `H` steps
ahead recursively from history (causal). The Azure serving trace is hourly-binned here; finer serving
resolution (per-request arrivals) is available and is the path to minute-scale control (future work).
No future truth: the step-`k` forecast never uses period `t+k`'s real arrivals.

## Diminishing returns & runtime

Runtime scales ~linearly with `H` (world-steps = candidates × H). The held-out gp/$ / SLA are
reported per `H` above; the best horizon by Pareto-safe gp/$ and the runtime/quality trade are the
decision criteria. See `research/MPC_HORIZON_ANALYSIS.md` for the action-frequency view and
`research/MULTI_PERIOD_MPC_ARCHITECTURE.md` for the design + honest limitations.
