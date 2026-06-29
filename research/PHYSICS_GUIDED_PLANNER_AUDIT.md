# Physics-Guided Planner Audit (Phase 0)

What does the planner/search path look like **today**, and exactly where does it fail the way PR #121
exposed? This audit reads the live search code and answers the 8 questions the next-gen planner must fix.
It is **read-only** — no simulator/reward/gate change. Evidence is `file:line`.

Files read: `controller.py`, `candidate_search.py`, `search_planner.py`, `action_registry.py`,
`actions.py`, `roofline_actions.py`, `run_checkpointed_all_knobs_backtest.py`, `diagnose_search_containment.py`,
`ACTION_SUBSET_CONTAINMENT.md`, `PLANNER_SEARCH_REGRET_DIAGNOSTIC.md`, `BASELINE_DRIFT_AUDIT.md`,
`CANONICAL_ALL_KNOBS_BACKTEST_RESULTS.md`.

---

## 1. What candidate spaces exist today?

There are **three** search entry points, chosen in `ModelPredictiveEconomicController.decide`
(`controller.py:400-418`), each scoring every candidate with a full receding-horizon `_rollout_world`
(clone + `simulate_period`, `controller.py:188-259`):

| # | path | when | candidate space | generator |
|--|--|--|--|--|
| A | **explicit list** | `self.candidates is not None` | a hand-supplied `list[ActionBundle]`; argmax over it (`controller.py:401-403`) | caller |
| B | **adaptive planner** | `use_adaptive_search and world` (the default) | the **roofline-pruned connected space** (`controller.py:404-413`) | `AdaptiveSearchPlanner.plan` (`search_planner.py:235`) over `roofline_pruned_options` (`search_planner.py:48`) |
| C | **generator search** | else | the connected (+opt. simulated) space, exhaustive≤256 else coordinate descent | `CandidateBundleGenerator.search` (`candidate_search.py:88`) |

Two real generators back these:

- **`CandidateBundleGenerator`** (`candidate_search.py:31`): exhaustive (≤`EXHAUSTIVE_BUDGET=256`),
  `latin_hypercube`, or `coordinate` neighbours; honours `frozen`/`frozen_reasons`; never silently drops a
  connected surface.
- **`AdaptiveSearchPlanner`** (`search_planner.py:126`): strategy by raw count — `exhaustive_cartesian`
  (≤`exhaustive_max=4096`), `beam_search` (`_beam`, `search_planner.py:152`), `coordinate_descent`,
  `cross_entropy`, `random_restart`; plus a **regret audit** (`search_planner.py:269`) that runs the exhaustive
  comparison when raw ≤ `regret_audit_max=20000`. Roofline surfaces are regime-pruned and co-location /
  prefill-decode are `FROZEN_OFF` with a recorded reason (`search_planner.py:40-45`).

**So the machinery already exists** (beam, regime-pruning, regret audit). What is missing is a *bounded,
anchor-guaranteed candidate set*: every entry point either searches the full 314,928-raw space (path B/C,
intractable per-eval) or a hand-supplied list (path A) with **no guarantee** the known-good bundles are in it.

## 2. Where does clock-only replace the full candidate set?

Path A, via the explicit-list override. The PR #121 bounded runner sets it directly
(`run_checkpointed_all_knobs_backtest.py:88-89`):

```python
if search == "clock":
    c.candidates = _clock_candidates()          # [ActionBundle(clock_policy=c) for c in (base, low, high)]
```

and `decide` then does `best_cand = max(self.candidates, key=_score)` (`controller.py:401-403`). This is a
**total replacement** of the candidate set with 3 clock-only bundles — every other surface frozen at its
`ActionBundle` default (bf16 / conservative / 1.0×). It is **not a pruning rule** (which would still let the
physics score the pruned point); the bundles simply do not exist in the enumerated set. The same shape is the
`clock_only` arm in `diagnose_search_containment.py:40-42`. **This is the artifact** the new planner must
delete: there must be no clock-only fallback (the runner default `--search clock`, `:220`, is the live source).

## 3. Which old winning bundles are not contained?

| winner | surfaces | reachable in clock-only? |
|--|--|--|
| **PR #121 24-grid winner** (`search_containment_diagnostic.json`) | `{clock=high, precision=fp8, capacity_multiplier=1.0, batching=aggressive}` | **NO** — needs fp8 + aggressive (frozen at bf16/conservative) |
| **+82.1% full-search family** (`mpc_attribution.json`, `BASELINE_DRIFT_AUDIT.md`) | combined `capacity / routing / batching / precision / clock` levers from the full connected search | **NO** — needs ≥4 non-clock surfaces |

Both are absent from clock-only because two (24-grid) or four+ (the +82% family) of their levers are non-clock.
The 24-grid winner scored **624,799 gp/$ (+100.48% vs the SLA-aware baseline, SLA 0.0375 vs 0.3375 — Pareto
dominant)** while clock-only scored **297,733 (−4.47%, SLA worse)** on the identical window
(`PLANNER_SEARCH_REGRET_DIAGNOSTIC.md`). **Search regret of clock-only = 327,065 gp/$ (forfeits +109.85%).**
The new planner must contain **both** families by construction (as guaranteed anchors).

## 4. Which knobs are connected and safe to search today?

The **12 CONNECTED surfaces** (`actions.py:80-171`, `CONNECTED_SURFACES`) — each reaches the scored reward
through a real channel, no bonus:

| surface | options | reward channel |
|--|--|--|
| `capacity_policy` | reactive_lag1 / backlog_aware / forecasted_mcs | run_unified_replay |
| `ordering_policy` | fifo / abs_conformal | run_unified_replay |
| `admission_policy` | off / class_aware | run_unified_replay |
| `routing_policy` | round_robin / shortest_queue / kv_aware | kv_service_factor |
| `capacity_multiplier` | 1.0 / 0.75 / 1.5 | run_unified_replay |
| `batching_policy` | conservative / balanced / aggressive | run_unified_replay |
| `prewarm_policy` | off / conservative / aggressive | world_simulator |
| `placement_policy` | topology_blind / rack_local / network_aware | world_simulator |
| `migration_policy` | off / conservative / aggressive | world_simulator |
| `precision_policy` | bf16 / fp8 / int4 | roofline_serving |
| `spec_decode_policy` | off / shallow / medium / aggressive | roofline_serving |
| `clock_policy` | base / low / high | roofline_serving |

**Default search surfaces for the new planner** (highest value-density, all CONNECTED): `precision × batching ×
capacity_multiplier × clock` — the 4 that produced the +100.48% win. **Optional, only when relevant/supported:**
`routing`, `prewarm`, `migration`, `spec_decode` (CONNECTED, but each can hurt and adds cost; add via
progressive widening, not by default).

## 5. Which knobs should stay out unless real background work / telemetry exists?

| surface | status | why out by default |
|--|--|--|
| `colocation_policy` | SIMULATED_ONLY | **No background-work trace** (Azure is all latency-critical; `ReplicaState.workload_class` unused). Co-location credits **no** goodput and can only hurt foreground SLA (`actions.py:174-181`, `roofline_actions.py:112-117`). **Generate only if `background_work=True`.** |
| `prefill_decode_policy` | SIMULATED_ONLY | live replay has **no disaggregated prefill/decode capacity pools**; only roofline models the split analytically (`actions.py:182-188`, `search_planner.py:43-44`). |
| `kv_routing_policy`, `topology_policy` | SIMULATED_ONLY | fleet effect already CONNECTED via `routing_policy`; no per-request prefix ids / no network model in reward (`actions.py:189-199`). |
| `kv_placement_policy`, `energy_policy` | PLANNED | no actuator the simulator honours (`actions.py:202-210`). Never optimized. |
| `precision_policy = int4` | CONNECTED but **quality-risked** | `PRECISION_QUALITY_RISK = 0.05` (`roofline_actions.py:53`); int4 wins are labelled unsafe/diagnostic. Propose int4 **only when memory-bandwidth-bound** (`search_planner.py:62-65`), never as a headline anchor. |

The new generator **must** carry these guards forward: co-location gated on `background_work`; SIMULATED/PLANNED
never default-generated; int4 regime-gated and never a "known-strong" anchor.

## 6. What is the raw Cartesian candidate count?

Measured from `ACTION_SPECS`:

```
CONNECTED only (12 surfaces):  3·2·2·3·3·3·3·3·3·3·4·3  =  314,928
CONNECTED + SIMULATED (16):                            =  11,337,408
Default-4 surface space (prec·batch·cap·clock):  3·3·3·3 =      81
PR #121 diagnostic grid (clock·{bf16,fp8}·{1.0,1.5}·{cons,aggr}):  =  24
clock-only:                                                       =   3
```

So the live default (path B) builds a search whose **raw count is 314,928** (`controller.py:409` →
`roofline_pruned_options` keeps all 12 connected surfaces, regime-pruning only precision/spec/clock). The
beam keeps the *evaluated* count bounded (~300–400) but every evaluation is a full world rollout.

## 7. Which parts are actually intractable?

**Not the candidate count — the per-evaluation cost.** Each `_score(b)` runs `_rollout_world` =
`clone_world_state_for_candidate` + a point `simulate_period` + a risk `simulate_period` (`controller.py:244-249`).
At hourly cadence the full-space beam's ~300–400 rollouts did **not complete in > 3 min and were killed**
(`PLANNER_SEARCH_REGRET_DIAGNOSTIC.md`, the PR #118 result that *forced* the clock-only fallback). The
**exhaustive 314,928** is hopeless; even **beam over all 12 surfaces** is too heavy because the rollout count
× rollout cost is large. **Tractable:** a bounded candidate set (≤ ~120 rollouts) — the 24-grid completed in
seconds and recovered +100.48%.

## 8. Which small candidate sets already recover large value?

| set | size | gp/$ | vs SLA-aware baseline | SLA | source |
|--|--|--|--|--|--|
| clock-only | 3 | 297,733 | −13,926 abs / **−4.47%** | 0.5625 (worse) | `search_containment_diagnostic.json` |
| **24-grid (prec·cap·batch·clock)** | **24** | **624,799** | **+313,140 abs / +100.48%** | **0.0375 (better)** | same |

```
Goodput/$ (baseline sla_aware 311,659):  clock-only 297,733 (abs −13,926, −4.47%)
                                         24-grid    624,799 (abs +313,140, +100.48%)  ← Pareto-dominant
SLA violation (baseline 0.3375, lower better):  clock-only 0.5625 (+0.225)   24-grid 0.0375 (−0.300)
```

**A 24-bundle multi-knob grid — evaluated in seconds — already recovers a Pareto-dominant +100.48%.** The
new planner needs the *right bounded set*, not the full space and not clock-only.

---

## Conclusion → what the next planner must do

1. **Delete the clock-only fallback** (Q2). Default to a bounded, physics-guided, anchor-guaranteed set.
2. **Contain both old winners by construction** (Q3): the 24-grid `{fp8, aggressive, high}` and the +82% family
   are **guaranteed anchors**, never droppable.
3. **Search the 4 high-value CONNECTED surfaces by default** (Q4), widening to routing/prewarm/migration/spec
   only when the decision is close (progressive widening).
4. **Keep co-location / SIMULATED / PLANNED / int4-headline out** (Q5) with the existing recorded-reason guards.
5. **Bound the rollout count** (~30–100, hard-capped), never the full 314,928 (Q6/Q7), reusing the beam + regret
   machinery that already exists (`search_planner.py`).
6. **Measure regret offline** against exhaustive on the small default-4 space (81 ≤ tractable) so any miss is
   visible, and **fail** if the planner loses to a contained old-best (Q8).

This audit changes no code, reward, or gate. The build follows in `physics_guided_candidates.py` +
`physics_guided_planner.py` + `search_regret_auditor.py`.
