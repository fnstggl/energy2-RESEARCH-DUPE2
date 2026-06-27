# MPC Horizon Analysis

Effect of the receding-horizon length `H` (in simulation steps) on the world-state MPC's held-out
behaviour. Companion to `research/MPC_CONTROL_INTERVAL_AND_HORIZON_ANALYSIS.md` (clock / lookahead)
and `research/MULTI_PERIOD_MPC_ARCHITECTURE.md` (design + honest limitations). Artifact:
`data/external/mpc_controller/mpc_horizon_ablation.json`.

## Held-out ablation (Azure 2024 week, persistent world)

`H=1` is the single-period anchor; each row commits only the first action per control interval.

| H | lookahead | reward (gp/$) | SLA viol | GPU-h | queue p95 | runtime/decision | gate (beats/pareto/headline) |
|--:|--:|--:|--:|--:|--:|--:|---|
| 1 | 60 min | 111,079 | 0.0220 | 56.2 | — | 0.132s | True / False / **False** |
| 2 | 120 min | 111,791 | 0.0210 | 56.1 | — | 0.202s | True / False / **False** |
| 4 | 240 min | 111,665 | 0.0208 | 56.2 | — | 0.341s | True / False / **False** |

Fair baseline (aurelius_canonical_kv_routing, 1.0× capacity): gp/$ 107,152, SLA 0.0143. Capacity
mix is 0.75× ×42 at every H (lean). Larger H slightly improves SLA (0.0220→0.0208 — the rollout
anticipates future risk) at ~linear runtime; gp/$ is ~flat (diminishing returns by H=4); the gate
stays blocked (lean capacity's SLA stays above the 1.0× fair baseline). Stride-96 sample of the week.

Columns: reward (SLA-safe gp/$), SLA violation rate, GPU-hours, queue p95, prewarm / migration /
placement frequency, controller runtime per decision, and the Pareto gate. Best horizon = the one
that maximizes Pareto-safe gp/$ at acceptable runtime.

## Decision stability & action frequency

The chosen-action mixes (`capacity_multiplier_mix`, `prewarm_mix`, `placement_mix`, `migration_mix`)
are reported per `H` in the artifact. On the Azure trace at hourly control:

- **capacity** stays lean (0.75×) across horizons — the calibrated capacity economics (PR #102) hold;
- **placement** is the actively-selected stateful lever (topology-aware), as in PR #101/#102;
- **prewarm / migration** remain at no-op across horizons — at hourly `dt` their persistence window
  (~300s) is shorter than the step, so a longer step-horizon cannot capture their payoff (this is the
  control-interval point, not a horizon failure). The multi-period rollout *does* correctly propagate
  their future consequences — the reward gap to the reactive baseline narrows monotonically with `H`
  in controlled fixtures — but it does not flip the decision under current calibration.

## Diminishing returns

Runtime grows ~linearly with `H` (world-steps simulated = candidates × H, reported per decision).
Reward changes with `H` are small at hourly control (because the dominant levers are within-period);
the horizon's value is expected to appear at **minute-scale control** where deferred actions span
steps — the recommended next experiment once the serving plane is driven at finer `dt`.

## Honest verdict

The controller is a correct receding-horizon MPC (H=1 parity, clone isolation, first-action-only,
causal trajectory, deterministic — all tested). On the Azure hourly trace, larger `H` does not yet
produce a Pareto-safe headline beyond the single-period result, because the deferred-benefit actions
do not span hourly periods. This is reported, not forced. The scoped next steps (finer control
interval, non-myopic prewarm sizing, migration warm-identity) are in the architecture doc.
