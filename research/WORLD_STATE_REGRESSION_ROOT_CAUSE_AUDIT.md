# World-State MPC Regression — Root-Cause Audit

PR #101 built the persistent world state and connected prewarm/placement/migration. Its held-out
evaluation regressed: the world-path MPC posted **−42% gp/$** vs the fair baseline and chose
**capacity_multiplier=1.5 in all 42 periods**. This audit finds the cause with per-period evidence
(`scripts/diagnose_world_state_regression.py` →
`data/external/mpc_controller/world_state_regression_diagnostics.json`).

## The five candidate causes, decided by evidence

| Hypothesis | Verdict | Evidence |
|---|---|---|
| 1. Cold-start physics too frequent/expensive | **Contributing, secondary** | cold_start magnitude (30s) is evidence-supported (below); the *ramp shape* moves SLA only ~0.06 in isolation |
| 2. Migration economics / horizon mismatch | **Real but unrelated to the 1.5× regression** | migration was never selected (correctly — deferred benefit, single-period horizon); does not touch capacity |
| 3. MPC objective / risk_weight | **NOT the cause** | per-period `risk_viol = 0.00` for *every* capacity at *every* period → `risk_weight=1.0` contributes ZERO penalty; 1.5× is chosen on raw predicted gp/$, not risk |
| 4. Forecast / state integration bug | **NOT the cause** | forecasts are causal; clone isolation + mutate semantics verified in tests; warm pool is sticky (no spurious cold-starts under steady load) |
| 5. **Genuine economics** vs **simulator bug** | **SIMULATOR BUG (cost model)** | the predicted gp/$ *rises* with capacity even when `point_sla=0` (goodput identical) → cost must *fall* as capacity rises, which is backwards |

## The smoking gun: warm-hold inverts the capacity economics

Per-period capacity-option scoring (point scenario, `point_sla=0.000` at every level — i.e. goodput
is identical, nothing misses SLA), yet the predicted `exp_gpd` **increases** with capacity:

```
p126   cap=0.75 → exp_gpd 41065   cap=1.0 → 47005   cap=1.5 → 67521   (risk_viol 0.00 / 0.00 / 0.00)
p130   cap=0.75 → 44322           cap=1.0 → 61670    cap=1.5 → 135367
```

A controlled reproduction (light load, all served, warm pool 14 carried from a prior peak) shows why:

```
cap=0.75  peak_c=6   idle_warm=8   warm_hold=$16.00   serving_gpu_h=0.153 ($0.31)   total=$16.13   gp/$=124
cap=1.0   peak_c=8   idle_warm=6   warm_hold=$12.00   serving_gpu_h=0.206 ($0.41)   total=$12.18   gp/$=2627
cap=1.5   peak_c=12  idle_warm=2   warm_hold=$ 4.00   serving_gpu_h=0.306 ($0.61)   total=$ 4.27   gp/$=8433
```

The serving cost rises with capacity (0.153→0.306 GPU-h), exactly as it should. But the **warm-hold
cost falls** ($16→$4) and **dominates** the total (it is ~50× the serving cost), so the total cost
*falls* as capacity rises and the planner is rewarded for over-provisioning.

### Why warm-hold behaves this way (the mis-model)

`world_simulator` charged every warm-but-idle replica a **full period (1 hour) of GPU time**
(`WARM_HOLD_GPU_FRACTION=1.0 × period_hours=1.0`). `idle_warm = warm_capacity − peak_c`. A *low*
capacity has a *small* `peak_c`, so more of the carried warm pool is left idle → a *large* warm-hold
bill; a *high* capacity's `peak_c` exceeds the pool → ~zero idle → ~zero warm-hold. So the warm pool
the operator carried from a *previous* period is charged against the *current* capacity choice,
inverting its economics.

This is unrealistic. Public autoscaler practice cools idle replicas after a **~300-second idle
timeout** (default scale-down delay; configurable to 1 hour), not a full hour. Charging a full hour
overstates idle warm-hold by ~12× for hourly periods — enough to swamp the serving cost and flip the
capacity decision.

## Answers to the required questions

- **Why capacity 1.5× every period?** Because the broken warm-hold term made higher capacity *score
  cheaper* (it absorbs the carried warm pool, dodging the per-period idle bill). The planner was
  rational under a wrong cost signal.
- **Which reward component favored 1.5×?** The warm-hold term in `operator_cost` (NOT the SLA/risk
  term — `risk_viol=0` everywhere).
- **Extra GPU-hour cost it added?** Realized fleet GPU-hours 188.7 vs the 1.0× baseline's 121.8
  (+55%); the regression is this over-provisioning.
- **SLA improvement it bought?** Negligible/none — mpc 0.0143 vs fair 0.0162 (the baselines already
  meet SLA at 1.0×); the over-provisioning bought no real SLA headroom worth its cost.
- **Rational under the configured objective?** Yes — given the broken warm-hold cost the choice was
  score-maximising. The objective (`risk_weight=1.0`) was inert here (`risk_viol=0`).
- **Caused by risk_weight / cold-start / warm decay / horizon / search?** **None of those.** Caused by
  the **warm-hold cost model** (full-period instead of idle-timeout), a simulator cost-model bug.

## Classification

- **Simulator issue: YES (primary)** — warm-hold full-period idle charge inverts capacity economics.
- Optimizer issue: only downstream (the MPC correctly optimised a wrong signal).
- Objective issue: NO (`risk_weight` inert; `risk_viol=0`).
- Data/forecast issue: NO (forecasts causal; integration verified).
- Genuine economics: NO — the apparent "over-provisioning is cheaper" was an artifact.

## Fix direction (evidence-driven, calibrated next)

1. **Warm-hold reflects the ~300s idle timeout**, not a full period — idle replicas cool. Prewarmed
   pools intentionally held longer still pay accordingly. (Primary fix.)
2. cold_start_s stays **30s** (evidence-supported; see calibration doc) — **not** tuned down.
3. Cold-start ramp made progressive (replicas come online staggered) — a secondary realism
   improvement, small effect.
4. Re-run the objective sweep to confirm a calibrated `risk_weight` still respects SLA (we do NOT
   simply set it to 0).

Calibration + before/after results: `research/WORLD_STATE_CALIBRATION_AND_REGRESSION_DIAGNOSIS.md`.
