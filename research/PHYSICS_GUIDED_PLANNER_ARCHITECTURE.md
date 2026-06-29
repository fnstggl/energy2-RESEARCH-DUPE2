# Physics-Guided Planner Architecture

The next-generation planner that fixes the candidate-generation/search bottleneck PR #121 exposed. It makes
**candidate generation a first-class planner layer** instead of an afterthought, and replaces both failure
modes — clock-only (too narrow) and full-space adaptive (intractable per-eval) — with a bounded,
physics-guided, anchor-guaranteed search. No simulator/reward/gate change; the new path is opt-in
(`controller.physics_guided`), default off.

```
Forecast distributions  →  World model  →  Physics-guided candidate generator
   → Bounded beam MPC  →  Progressive widening  →  Decision  →  Search-regret auditor
```

| stage | module | role |
|--|--|--|
| forecast distributions | `forecasting.py`, `forecast_trajectory.py` | causal H-step belief (point + p90 risk) — unchanged |
| world model | `world_simulator.py`, `world_state.py` | the rollout that *scores* a candidate (`simulate_period`) — unchanged |
| **physics-guided candidate generator** | **`physics_guided_candidates.py`** | regime priors + guaranteed anchors → a bounded set |
| **bounded beam MPC** | **`physics_guided_planner.py`** | beam over the set, captures cross-surface coupling |
| **progressive widening** | **`physics_guided_planner.py`** | expand surfaces only when the decision is close |
| decision | `controller.py` (`decide`, opt-in branch) | applies the chosen first action; records the plan |
| **search-regret auditor** | **`search_regret_auditor.py`** | offline: measures what each strategy left on the table |

## Why candidate generation is now a first-class layer

PR #121's "all-knobs got worse" was **not** an MPC, reward, or gate failure. It was a *candidate-containment*
failure: the bounded runner replaced the search with three clock-only bundles
(`run_checkpointed_all_knobs_backtest.py`, `--search clock`), so the `{precision=fp8, batching=aggressive,
clock=high}` bundle that doubled gp/$ while improving SLA **was never in the enumerated set**
(`PHYSICS_GUIDED_PLANNER_AUDIT.md`, Q2/Q3). When the candidate set is implicit and hand-supplied, a single
config flag silently determines whether the planner can even *see* the winning action. Making generation an
explicit, audited layer — with structural guarantees about what it always contains — is what prevents that
class of regression. **The planner can only be as good as its candidate set; so the candidate set gets first-class
treatment, guarantees, and its own regret audit.**

## Why brute force is too expensive

The full connected action space is **314,928 bundles** (12 CONNECTED surfaces; `PHYSICS_GUIDED_PLANNER_AUDIT.md`
Q6). Enumeration is hopeless, but even the existing adaptive **beam over all 12 surfaces** is intractable —
**not** because the candidate count is huge (the beam keeps it ~300–400) but because **each evaluation is a full
world rollout** (`controller._rollout_world` → clone + two `simulate_period` calls). At hourly cadence that
beam *did not complete in > 3 min and was killed* (the PR #118 result that forced the clock-only fallback,
Q7). The lever that matters is therefore the **rollout count**, and it must be bounded to ~tens, not hundreds.

## Why clock-only caused a fake regression

Clock-only is a 3-bundle set varying only `clock ∈ {base, low, high}`, every other surface frozen at its
default (bf16 / conservative / 1.0×). On the PR #121 window it scored **−4.47%** vs the SLA-aware baseline
(SLA *worse*), while a 24-bundle multi-knob grid scored **+100.48%** (SLA *better*, Pareto-dominant) — a
**search regret of 327,065 gp/$** for clock-only (`PLANNER_SEARCH_REGRET_DIAGNOSTIC.md`). The "regression"
was the difference between *what the planner searched* and *what it could have searched*, nothing about the
MPC itself. Deleting the clock-only fallback is the single most important fix; the new planner never uses it.

## How physics priors generate candidates

`PhysicsGuidedCandidateGenerator` reads a cheap pre-search `PlannerRegimeState` (roofline regime, SLA slack,
queue/capacity/price/token/HBM pressure, confidence, previous bundle) and emits a bounded set over the four
high-value CONNECTED surfaces `precision × batching × capacity × clock`:

- **Regime priors (soft, Phase 2)** shape *which options* enter the grid: memory-bandwidth-bound → fp8 + low
  clock + (SLA-permitting) aggressive batching; compute-bound → base/high clock, conservative batching, no
  aggressive spec; queue/SLA pressure → higher capacity + safer batching; price-high → low clock when slack;
  HBM pressure → lower precision + conservative batching. The priors **only decide what is generated; the
  reward is unaffected** — a generated candidate still scores through the real physics, so a wrong prior cannot
  manufacture a win, only a wasted evaluation.
- **Guaranteed anchors (structural)** are always in the set and **never dropped by the cap**: `ActionBundle()`
  (neutral), the production-safe SLA-aware bundle, the previous best, the `fp8 + aggressive` known-strong
  family + the +82% combined-lever family, capacity-adjusted (1.5× / 0.75×) bundles, and clock low/base/high.
  This is the containment guarantee — the bundles PR #121 proved recover the win are *always searched*.
- **Honest exclusions (recorded):** co-location is generated only with `background_work=True` (no
  background-work trace exists); SIMULATED_ONLY / PLANNED surfaces are never default-generated; `int4` is
  proposed only when memory-bandwidth-bound and is **never** a known-strong anchor (quality/SLA risk). Every
  exclusion lands in `pruned_reasons`. Target 30–100 candidates; configurable hard cap drops *prior-grid*
  candidates first, never anchors.

## How beam search works

`BoundedBeamPlanner` runs a deterministic beam over the generator's surface options, **seeded with the
neutral bundle and every anchor** (so the known-strong winner is always in contention). It adds one surface
at a time, keeping the top-`beam_width` partial bundles — which lets it construct a **coupled optimum**
(`{fp8, aggressive}` is a win only *together*; neither lever alone improves on neutral) that a single-dimension
**coordinate descent gets stuck before** (it commits no single move, so it never reaches the pair). A bounded
**coordinate polish** over the full safe capacity×clock range then re-couples cheap headline levers
(`capacity=1.5`, `clock=high`) onto the beam winner, so a Pareto improvement is never missed when the regime
prior gated those values out of the grid. The plan reports raw / generated / evaluated counts, the selected
bundle, the top-K, the decision margin, runtime, whether the anchors / previous-best / known-strong were
contained, and the beam width.

## How progressive widening works

The planner starts on the 4 default surfaces. It widens to the optional CONNECTED surfaces
(`routing → prewarm → spec_decode → migration`, one per round) **only when the decision is close**: small
decision margin, low forecast confidence, a tight SLA, or a recent regret-audit failure. If the margin is
large it **stops early** (the winner is clear; more search is wasted rollouts). Each round records the
surfaces added, the trigger reason, the candidates added, and the runtime — and the whole loop is bounded by
`max_evaluated` so the rollout count can never blow up.

## How search regret is measured

`search_regret_auditor.py` compares, on the small **default-4 space where exhaustive is tractable** (81
bundles), the physics-guided beam against clock-only, the fixed 24-grid, exhaustive (the ground truth), and
the existing adaptive planner. It reports each strategy's **search regret** (best-exhaustive − best-found,
absolute and %), the **missed bundle** (exhaustive argmax), and — for the physics planner — whether that
bundle was **generated / pruned / evaluated**. The backtest additionally measures regret *with the real world
rollout* by differencing each arm against the exhaustive arm on the same window. **Hard rule:** if the bounded
planner ever *loses to a contained old-best bundle*, the audit FAILS — that is a true search bug, distinct
from an honest prune (e.g. int4 outside the memory-bound regime), which is reported as a prune, not a failure.

## How this differs from a simulator improvement

Nothing here touches the world model, the reward, the cost model, or the Pareto gate — all are byte-identical.
The only thing that changed is **which candidates the planner evaluates and in what order**. Any gp/$ change is
attributable to *search and candidate containment*, not to a richer or re-tuned physics. That is the whole
point: the +100% on the PR #121 window was always *reachable* by the simulator; the planner simply could not
see it. This PR makes it visible by construction, and proves the improvement comes from search by measuring
regret against the unchanged exhaustive ground truth.

## What remains for the long-term planner architecture

- **Forecast fidelity** is now the dominant residual (the oracle arm's gp/$ ceiling above the deployable arm).
  Output-length / arrival forecasting is the next highest-ROI lever — measured, offline-improvable.
- **Multi-step coupling across the horizon** (the beam couples surfaces *within* a decision; cross-period
  action coupling under the receding-horizon rollout is still greedy on the first action).
- **A calibrated production baseline** (vLLM-like continuous-batching arm) between FIFO and the SLA-aware
  heuristic, so the production-relevant margin is anchored (`PRODUCTION_BASELINE_LADDER.md`).
- **Hierarchical / learned candidate proposal** is justified *only if* a broad multi-knob grid itself becomes
  intractable as surfaces grow — not demonstrated yet; the bounded beam already closes the PR #121 gap.
- **Real telemetry** for the SIMULATED_ONLY / PLANNED surfaces (co-location background work, disaggregated
  prefill/decode pools, per-link topology) would let progressive widening reach them honestly.
