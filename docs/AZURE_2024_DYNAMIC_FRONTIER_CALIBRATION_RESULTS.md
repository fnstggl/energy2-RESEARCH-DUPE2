# Azure LLM 2024 — Dynamic Frontier Calibration + Shadow Evaluation

> **Simulator / shadow-mode benchmark. Directional only — NOT a production-savings claim** (`docs/RESULTS.md` §8). Streaming replay of the Azure 2024 trace; each per-tick decision sees only the telemetry from t' ≤ t (no future leakage). The robust energy engine is **unchanged**; the static frontier controller and committed Azure 2024 artifacts (including the existing dynamic frontier JSON / MD) are **read-only**. Real-cluster execution is disabled by default. The oracle row is **analysis-only**.

- **Read first:** `docs/RESULTS.md`, `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`, `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`, `docs/AZURE_2024_DYNAMIC_FRONTIER_RESULTS.md`, `docs/PILOT_TELEMETRY_CONTRACT.md`.

## 1. Configuration

- **Tick seconds:** 60.0
- **Primary scale:** 100.0×
- **Candidate rho grid:** `[0.45, 0.55, 0.65, 0.75, 0.85, 0.95]`
- **Window (minutes):** 60
- **Safety thresholds (pre-registered):** timeout ≤ 10.0% AND queue p99 ≤ 2000.0 ms
- **Passes:** 3
- **Target oracle-alpha capture:** 0.95 (aspiration, not a forced pass condition)
- **Max false-safe rate (safety floor):** 0.0100
- **No future leakage:** each decision sees t' ≤ t only.

## 2. Source

- **Trace:** `/home/user/energy2/tests/fixtures/azure_llm_2024_sample.csv` (1,560 ticks @ 60s; 93,540 s total)

## 3. Pass-by-pass results

| pass | n records | capture (overall) | capture (mean per-window) | safety correct % | false safe % | false unsafe % | conservative miss % | MAE timeout (pp) | MAE queue p99 (ms) | avg conf before | avg conf after | unsafe risk thr | deadband |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1,552 | 91.07 % | 88.73 % | 86.53 % | 0.45103 % | 13.015 % | 13.660 % | 3.636 | 362.8 | 0.00376 | 0.00344 | 0.75000 | 0.05000 |
| 1 | 1,552 | 91.07 % | 88.73 % | 85.44 % | 0.45103 % | 14.111 % | 14.755 % | 3.630 | 362.3 | 0.00367 | 0.00335 | 0.77000 | 0.04500 |
| 2 | 1,552 | 91.07 % | 88.73 % | 85.44 % | 0.45103 % | 14.111 % | 14.755 % | 3.630 | 362.3 | 0.00367 | 0.00335 | 0.79000 | 0.04000 |

## 4. Recommendation distribution by pass

| pass | RAISE | KEEP | LOWER | INSUFFICIENT | avg rec rho | rho distribution |
|---|---|---|---|---|---|---|
| 0 | 284 | 1064 | 204 | 0 | 0.75786 | 0.45:5, 0.55:9, 0.65:29, 0.70:224, 0.75:869, 0.80:276, 0.85:122, 0.95:18 |
| 1 | 293 | 1038 | 221 | 0 | 0.75609 | 0.45:5, 0.55:9, 0.65:33, 0.70:220, 0.75:880, 0.80:307, 0.85:79, 0.95:19 |
| 2 | 293 | 1038 | 221 | 0 | 0.75609 | 0.45:5, 0.55:9, 0.65:33, 0.70:220, 0.75:880, 0.80:307, 0.85:79, 0.95:19 |

## 5. Multi-pass outcome

- **Initial oracle-alpha capture (pass 0):** 91.07 %
- **Final oracle-alpha capture:** 91.07 %
- **Target (aspirational, not forced):** 95 %
- **Target reached:** NO
- **Safety floor held:** YES
- **Stopped reason:** `passes_exhausted:3, final_capture=0.910728142250231`

### Why the target was not reached

Final oracle-alpha capture 91.07 % is below the aspirational 95 % target. The dynamic estimator's Erlang-C tail calibration is workload-specific and noisy on short windows; closing the gap requires per-workload pilot data, not constant tuning. False-safe rate 0.4510 % is non-zero — we will not relax the unsafe-risk threshold to chase capture because doing so would push false-safe further up. Conservative-miss rate 14.7552 % suggests the controller stays a notch below the oracle's realized best safe rho. This is by design (conservative margin + deadband + hysteresis) and is the right trade-off for an estimator that does not yet have pilot calibration. False-unsafe rate 14.1108 % comes from the low-confidence telemetry windows where the estimator correctly falls back to LOWER_RHO rather than guess. Closing the remaining gap is a pilot-telemetry calibration task, not a tuning task. See `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md` §5.

### Overfit / generalization caveats

- calibration_window_is_replay_window:any tuning that helps here may not generalize; pilot telemetry remains required before any production claim.

## 6. Honesty / scope

- The Dynamic Safe Frontier Calibration harness is **opt-in** and **disabled by default**. The static frontier controller remains the committed default; this run is a measurement.
- **No production mutation.** Decisions are recommendation-only (`executable_in_real_cluster=False`).
- **No future leakage.** Each per-tick decision sees only the telemetry from t' ≤ t in its rolling window. The oracle and baseline are computed **offline, post-hoc** and are visible to the calibration-record builder only — never to the dynamic estimator.
- **Bounded parameter updates.** Between passes the harness may tighten safety knobs OR relax conservatism within pre-registered bounded ranges. Safety vetoes, oracle labels, and the engine physics are NOT modified.
- **No ML training.** Confidence updates are deterministic and categorical (false-safe / false-unsafe / conservative-miss / accurate-safe / large-error penalty).
- The 95 % oracle-alpha target is **aspirational**. If we do not reach it without weakening safety or leaking the oracle into the decision loop, we report the gap honestly.
- This is **directional simulator / shadow-mode evidence** — NOT a production-savings claim. Pilot telemetry is required to calibrate the safe rho per workload before any production claim.
- The robust energy engine, the static frontier controller, the committed Azure 2024 audit / backtest / controller / integration / full-trace / dynamic frontier JSON are **NOT modified** by this benchmark.

