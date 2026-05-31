# Dynamic Serving Frontier — Calibration + Shadow Evaluation (v1)

> **Opt-in. Disabled by default. Shadow / simulator only.** Simulator
> evidence is **NOT a production-savings claim** (`docs/RESULTS.md` §8). The
> static frontier controller, the dynamic estimator v1, and the
> `constraint_aware` engine default rho (0.65) are **unchanged** by this
> module. Real pilot telemetry is required to calibrate the safe rho per
> workload before any production claim.

This document specifies the **Dynamic Serving Frontier Calibration
harness** — a closed-loop shadow-evaluation layer that scores the
Dynamic Safe Frontier Estimator v1 against the realized outcome at the
next decision window, updates per-workload confidence, and aggregates
prediction-quality metrics over a streaming replay.

- **Read first:** `docs/RESULTS.md`,
  `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`,
  `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md`,
  `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`,
  `docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`,
  `docs/PILOT_TELEMETRY_CONTRACT.md`.

## 1. What "calibration" means here

For every recommendation the dynamic estimator emits we want to answer
the same questions a pilot telemetry stream would answer in production:

1. What did Aurelius predict (goodput/$, timeout %, queue p99, SLA
   risk, recommended rho)?
2. What actually happened (the realized signals at the *next* decision
   window)?
3. Was the realized outcome inside the configured safety thresholds?
4. Did the estimator over- or under-estimate risk?
5. How confident should the controller be in the next recommendation?
6. How much of the post-hoc oracle's available alpha did we capture?

The calibration harness builds one `DynamicFrontierCalibrationRecord`
per decision and aggregates them into a calibration summary.

## 2. Prediction vs observed outcome

| model | source |
|---|---|
| `DynamicFrontierPrediction` (`aurelius/frontier/dynamic_evaluation.py`) | emitted by the dynamic estimator (`aurelius/frontier/dynamic_estimator.py`) + controller (`aurelius/frontier/dynamic_controller.py`) — recommendation-only |
| `DynamicFrontierObservedOutcome` (`aurelius/frontier/dynamic_evaluation.py`) | realized at the next decision window via the unchanged engine physics in `aurelius/traces/backtest.py` |
| `DynamicFrontierCalibrationRecord` (`aurelius/frontier/dynamic_evaluation.py`) | the (prediction, outcome) pair plus per-metric errors, categorical safety verdict, oracle-alpha bookkeeping, and the per-step confidence update |

Every numeric field is `Optional[float]` — missing telemetry stays
`None` (the pilot-telemetry contract in
`docs/PILOT_TELEMETRY_CONTRACT.md` §1). Records round-trip to JSON via
`records_to_json` / `records_from_json`.

## 3. Categorical safety verdicts

The calibration harness reports four categorical labels per record. None
of them are folded into the headline KPI (`docs/RESULTS.md` §1–§2).

| label | meaning |
|---|---|
| `safety_correct` | predicted-safe label == realized-safe label |
| `false_safe` | predicted safe (`RAISE_RHO` / `KEEP_RHO`, risk < 0.75), realized **unsafe** (timeout > 10 % or queue p99 > 2000 ms) — the most expensive error |
| `false_unsafe` | predicted unsafe (`LOWER_RHO` or risk ≥ 0.75), realized **safe** at the recommended rho |
| `conservative_miss` | controller kept rho strictly below the oracle's realized best safe rho for the window (cost: foregone alpha; not a safety violation) |

`false_safe` is the only verdict the harness will tighten safety for
(see §5).

## 4. Oracle-alpha capture

For each decision window we report the analysis-only oracle
(`OracleSeriesPoint`):

- `best_safe_rho` — the realized highest safe rho on the candidate grid.
- `oracle_goodput_per_dollar` — its realized goodput/$.
- `baseline_goodput_per_dollar` — the realized goodput/$ that the static
  `constraint_aware` baseline (rho = 0.65) would have produced for the
  same window.

```
oracle_alpha_available = oracle_goodput - baseline_goodput
oracle_alpha_captured  = actual_goodput - baseline_goodput
oracle_alpha_capture_pct = captured / available     # only when available > 0
```

Honesty rules:

- **The oracle is never visible to the estimator at decision time.** It
  is computed offline and consumed only by the calibration-record
  builder.
- **Zero / non-positive denominator** (`available <= 0` — oracle no
  better than baseline) returns `pct = None`. We do **not** silently
  treat 0/0 as 1.0.
- **Negative capture** (actual worse than baseline) is preserved as a
  negative number — never clipped to 0.
- The summary reports both `oracle_alpha_capture_pct_overall` (sum of
  captured / sum of available) and `oracle_alpha_capture_pct_mean`
  (mean of per-window pcts) so a heavy-tail bias is visible.

## 5. Confidence update logic — `aurelius/frontier/dynamic_confidence.py`

`update_confidence(record, config)` is deterministic and categorical.
It returns `(new_confidence, reason_code_string)`.

Rules:

| trigger | direction | default magnitude |
|---|---|---|
| `false_safe` | ↓ | 0.15 |
| `false_unsafe` | ↓ | 0.05 |
| `conservative_miss` | ↓ | 0.03 |
| accurate + safe + low-error | ↑ | 0.02 |
| large goodput/$ error (≥ 20 % rel) | ↓ | 0.02 |
| large timeout error (≥ 5 pp) | ↓ | 0.02 |
| large queue p99 error (≥ 500 ms) | ↓ | 0.02 |

Hard guarantees (asserted by tests):

- Confidence is clamped to `[min_confidence, max_confidence]` (default
  `[0, 1]`).
- Per-step movement is clamped to `max_update_per_step` (default 0.20).
- Confidence **MUST NOT** rise when safety was wrong (`false_safe` or
  `safety_correct=False`). A defensive `blocked_rise_unsafe` reason code
  is appended if the bound would have allowed it.
- Reason codes are comma-joined categorical labels so downstream
  summaries can audit them.

## 6. Rolling calibration replay — `aurelius/frontier/dynamic_calibration.py`

`run_dynamic_frontier_calibration_replay(...)` does one offline
streaming-replay pass:

1. Walk the trace tick-by-tick.
2. At each decision step, build the rolling telemetry window from
   *past* observations only (no future leakage, asserted by a test).
3. Call `estimate_dynamic_frontier` → `choose_dynamic_rho` to emit a
   prediction.
4. Apply the recommended rho to the current tick via the unchanged
   engine physics → realized outcome.
5. Build the calibration record (prediction × outcome × oracle window).
6. Apply the deterministic confidence update.
7. Append the tick to the rolling history for the next decision.

Default settings:

| setting | default |
|---|---|
| `window_ticks` | 60 (= 60 min @ 60 s / tick) |
| `decision_interval_ticks` | 1 |
| `candidate_rhos` | `(0.45, 0.55, 0.65, 0.75, 0.85, 0.95)` |
| `bootstrap_rho` | 0.65 (= the static `constraint_aware` default) |
| `safety_timeout_pct` | 10.0 |
| `safety_queue_p99_ms` | 2000.0 |
| `initial_confidence` | 0.5 |

## 7. Multi-pass calibration — bounded, honest, stops early

`run_multi_pass_calibration(...)` runs up to `passes` calibration
replays. Between passes the harness may apply **bounded** parameter
updates:

| knob | start | min | max | when |
|---|---|---|---|---|
| `unsafe_risk_threshold` | 0.75 | 0.55 | 0.85 | tighten if false-safe rate > floor; relax (within bounds) only if safety holds AND capture < target |
| `deadband_rho` | 0.05 | 0.02 | 0.10 | tighten on safety breach; loosen on conservative miss |
| `hysteresis_multiplier` | 2.0 | 1.0 | 3.0 | reduce slightly on conservative miss only |

Allowed changes:

- Confidence calibration (per-step magnitudes within
  `ConfidenceUpdateConfig`).
- Conservative-margin enable/disable.
- Risk threshold / deadband / hysteresis within the bounded ranges
  above.

Not allowed:

- ❌ Using future labels (oracle / baseline) inside a pass.
- ❌ Tuning to the test window without reporting overfit risk.
- ❌ Disabling safety vetoes.
- ❌ Changing oracle labels.
- ❌ Hiding unsafe points.
- ❌ Looping indefinitely. The harness stops on:
  1. `passes` exhausted,
  2. target capture reached **AND** safety floor held,
  3. proposed update would not improve safety or capture
     (`no_useful_update_proposed_below_target`).

## 8. Why 95 % is a *target*, not a forced requirement

> Goal: capture ≥ 95 % of the oracle alpha **on average** without
> weakening safety, without leaking the future into the decision loop,
> and without overfitting bounded tuning to one trace.

The harness reports whether we reached the target. If we did not, the
JSON / MD report explains why:

- The estimator's Erlang-C tail calibration is **workload-specific** —
  short windows are noisy.
- Closing the residual gap requires **per-workload pilot data**, not
  constant tuning.
- A false-safe rate above `max_false_safe_rate` triggers a tightening,
  never a relaxation. If tightening hurts capture, that loss is
  reported honestly.

There is no path inside the harness that reaches 95 % by hiding a
false-safe, by raising the safety threshold past the pre-registered
limit, or by peeking at the realized future during the decision step.
If we hit 95 % with safety held, we report it as a calibration-window
result and warn about overfit risk (`overfit_risk_notes`). Pilot
telemetry calibration remains the only path to a production claim.

## 9. Why no production-savings claims

Every artifact in this module — JSON, MD, dataclass — is shadow /
simulator only. Real-cluster execution stays disabled by default
(`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` §"Real-mode execution
boundary"). The static frontier controller is **not** demoted, the
`constraint_aware` engine default rho (0.65) is **not** changed, and the
robust energy engine is **not** modified. A successful calibration run
is *evidence the estimator is improving in shadow*, not a savings
number.

## 10. Why real pilot telemetry is still required

The calibration harness uses the engine's serving physics to produce
the "realized" outcome each tick. That is the right approximation for
**directional** shadow evidence, but pilot data is still required for:

- per-tenant SLA thresholds (`max_timeout_pct`, `max_queue_p99_ms`),
- per-workload Erlang-C tail constants (`C_queue`, `C_timeout`),
- per-tenant deadband / hysteresis defaults,
- the boundary-conditions of the real executor.

`docs/PILOT_TELEMETRY_CONTRACT.md` enumerates the fields a pilot stream
must supply. Until that data lands, the dynamic estimator stays opt-in
and recommendation-only.

## 11. Hard rules / non-goals

- ❌ Calibrated dynamic estimator is **not** default.
- ❌ Static frontier controller is **not** removed.
- ❌ `constraint_aware` engine default rho (≈ 0.65) is **not** changed.
- ❌ No new datasets ingested.
- ❌ Robust energy engine is **not** modified.
- ❌ Serving physics is **not** modified.
- ❌ No production mutation.
- ❌ No future leakage.
- ❌ No production-savings claim.
- ❌ No safety-gate weakening.
- ❌ No constants tuned per trace to force benchmark wins.
- ❌ No looping until the target is hit by overfitting.

## 12. Remaining gaps before pilot telemetry

- **Real telemetry stream.** The calibration harness is run against
  the same simulator physics that produced the prediction; pilot data
  closes the loop.
- **Workload-specific risk calibration.** A trained risk model (e.g.
  a logistic / quantile regression on pilot telemetry) is out of scope
  for v1; any future learned component must preserve the existing
  `[0, 1]` probability interface and reason codes so the controller
  stays unchanged.
- **Multi-workload arbitration.** v1 calibrates one workload at a time;
  cross-workload trade-offs (priority, co-residency) remain a future
  task.
- **Real-mode execution.** Same boundary as the dynamic estimator and
  the static controller — the executor stub stays a stub until the
  binding-boundary work in
  `docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` lands.
