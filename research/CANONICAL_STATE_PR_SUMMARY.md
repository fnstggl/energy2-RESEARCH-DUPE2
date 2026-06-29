# PR — Canonical State Coverage + All-Knobs Tractable Backtest — Summary

Normalization-first: audit what already exists, **promote/consolidate** rather than duplicate, build only the
genuine gaps (ForecastState, RequestState), and make the all-knobs backtest tractable. The reward, cost model,
and Pareto gate are **byte-identical**; the new states are opt-in observers, never reward terms. Built on `main`
with #117/#118/#119 merged.

## 16 questions, answered

**1. Which states are now FULL?** The 11 already-FULL canonical states (Replica/Server/Rack/KV/Electricity/
Power/Deferrable/Migration/Warm/Network/Quality — KEEP_AS_IS) **plus** the two new ones: **ForecastState**
(belief + realized + error) and **RequestState** (lifecycle + conservation). **QueueState** is now FULL-as-a-
consolidated-summary (derived from RequestState, no longer a placeholder).

**2. Which states remain PARTIAL?** **PlacementState** per-request mapping (recorded on RequestState but not yet
driven by a canonical per-request router); **RooflineState** is a persisted *record* (snapshot of the existing
`roofline_diag`) not a live mutator; **CostState** placeholder still superseded by `PeriodOutcome` (consolidate
in a follow-up — low risk, no reward effect).

**3. Which states remain ABSENT?** thermal state, true power caps, demand charges — all NEEDS_PRODUCTION_TELEMETRY
(labelled, never fabricated).

**4. What was ported from V2 / open simulators?** Design patterns only (no heavy-sim dependency): request
lifecycle + queue-discipline shape from LLMServingSim/BLIS/vLLM/Orca; roofline arithmetic-intensity/ridge from
InferSim/llm-analysis/LLM-Viewer (already on-repo via #110/#119). "V2" on this branch is the #119 roofline-timing
promotion + the audit docs — there is no `aurelius/environment/v2/` package, so the runner marks a `v2_reference`
arm SKIPPED_TOO_HEAVY rather than inventing one.

**5. Did RequestState become persistent?** **Yes** — `request_state.RequestLifecycleState` promotes the ephemeral
replay `Job` lifecycle into a clone-safe canonical pool with the conservation invariant `arrived = running +
completed + dropped` (validated; `request_conserved = True` in all backtest cells). Identity TRACE_DERIVED;
fine lifecycle timestamps SIMULATOR_INFERENCE.

**6. Did QueueState become persistent?** **Yes (consolidated)** — `RequestLifecycleState.queue_summary()` is the
single authoritative queue view (backlog, class mix, completion rate), resolving the `world_state.QueueState`
placeholder duplication. The live heap stays in the replay (not duplicated).

**7. Did ForecastState become first-class?** **Yes — the top-priority deliverable.** `forecast_state.ForecastState`
persists per-decision belief + provenance + confidence, then realized + per-variable forecast error (MAE/MAPE),
referencing existing forecaster outputs (no new model). Causal/leak-free: belief before the period, error only
after. Wired opt-in into `decide` (belief) and `run_period_episode` (realized).

**8. Did PlacementState become persistent?** Partially — replica-level placement was already FULL canonical; the
per-request placement field now lives on RequestState (recorded), but a canonical per-request router is EXTEND-
after (honest).

**9. Did RooflineState become persistent?** As a **record** — `RooflineRecord.from_diag` snapshots the per-period
roofline regime (folds DecodeState's phase classification). Diagnostic + planning state, not a reward term.

**10. Did the all-knobs backtest become tractable?** **Yes** — `run_checkpointed_all_knobs_backtest.py`:
**24 COMPLETED, 4 SKIPPED_TOO_HEAVY, 0 TIMEOUT, 0 FAILED**. Checkpoints after every cell, resumes, hard per-cell
caps, COMPLETED/TIMEOUT/FAILED/SKIPPED_TOO_HEAVY. It does not jam.

**11. What is the deployable all-knobs gp/$ number?** Per the reporting standard (deployable vs SLA-aware
baseline): **median −3.10%**; by window **+1.56% (ercot·volatile, best), −2.23%, −3.98%, −4.87%**. Absolute:
e.g. pjm·expensive **352,581 → 344,726 gp/$ (−7,855, −2.23%)**.

**12. Was it Pareto-safe?** **No — 0 of 4 windows.** Every deployable cell raises the SLA-violation rate vs the
baseline (e.g. pjm·expensive 0.275 → 0.525), so `pareto_sla_not_worse = False` and **no headline is claimed**.

**13. What is the oracle gap?** `oracle − N2`: **median ≈ +1,580 gp/$, up to +7,520 (ercot·expensive)** — forecast
regret, consistent with #118.

**14. What is the search/runtime bottleneck?** The full **adaptive** all-knobs search at hourly cadence (the #118
finding) — clock-focused is the tractable default; adaptive cells are timeout-protected / SKIPPED_TOO_HEAVY. The
biggest *value* bottleneck is **forecast fidelity** (the oracle gap), now instrumented by ForecastState.

**15. Next highest-ROI state/model improvement?** **Output-length / arrival forecast fidelity**, measured by
ForecastState (output-length MAPE 7.8–18.9% here; the oracle gap is forecast-driven). Doable offline on existing
traces; then queue/SLA-pressure forecasting (the path to Pareto-safety). See `CANONICAL_STATE_NEXT_ROADMAP.md`.

**16. What remains simulator-inferred or needs pilot telemetry?** SIMULATOR_INFERENCE: request fine-grained
lifecycle timestamps, RooflineState power/regime, DVFS curve, deferrable workload. NEEDS_PRODUCTION_TELEMETRY:
real GPU power, real per-request output length, real cache-hit rates, thermal/power-caps, demand charges.

## Verification

- `ruff` clean; **canonical-state fixtures (11) + 91 regression tests** pass; reward/cost/Pareto byte-identical
  (the new states are opt-in; defaults unchanged).
- Reproducible artifacts: `checkpointed_all_knobs_backtest.json` (24 COMPLETED + 4 SKIPPED_TOO_HEAVY).
- Docs: coverage audit, regret priority, architecture, controlled fixtures, next roadmap, all-knobs results.
