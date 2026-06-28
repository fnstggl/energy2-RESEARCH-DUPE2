# MPC Search & Planning/Eval Fidelity Audit (Phase 0)

Why the live MPC under-selected FP8 in PR #111 (it reached +33.6% gp/$ vs the static FP8+spec +44.5%). The
physics is correct; the **planner is partially blind** — it scores candidates with a rollout that omits the
phase/cost/roofline economics the evaluation path uses. Every claim is grounded in `file:line`.

## Where planning vs evaluation is computed

1. **Planning rollout** — `controller._rollout_world` (`controller.py:168-214`). Clones the world
   (`clone_world_state_for_candidate`), builds **synthetic** jobs from the forecast (`_synth_jobs`,
   `controller.py:198-203`), and calls `simulate_period(..., mutate=…)` **without `kv_state`**
   (`common0`, `controller.py:182-185`, has no `kv_state`). Reward = `out.goodput_per_dollar`
   (`controller.py:208`).
2. **Evaluation rollout** — `run_period_episode` (`controller.py`), world path, calls
   `simulate_period(..., kv_state=kv_state, mutate=True)` with the Mooncake prefix pool **and**
   `kv_cost_mode` (`controller.py:493-499` region) → the phase model + hybrid cost + roofline modulation
   run.

## Which physics are present in EVAL but MISSING in planning

`simulate_period` gates the phase/cost/roofline economics on `kv_state` + `cost_mode`
(`world_simulator.py`): `phase = compute_phase_serving(...)` runs only `if cost_mode and per_req_factor is
not None`, and `per_req_factor` requires `kv_state`. Since planning passes **no `kv_state`**:

| physics | eval | planning (today) |
|--|--|--|
| prefill/decode phase split (PR #107) | ✅ | ❌ (lumped `_service_time_s`) |
| realized GPU-seconds | ✅ | ❌ (capacity integral only) |
| hybrid cost mode (PR #107) | ✅ | ❌ (provisioned only) |
| **precision cost benefit** (`gpu_seconds_factor`) | ✅ | ❌ — only the latency factor `rl_svc` (`completion_factor`) is applied (`world_simulator.py` `_svc` non-phase branch) |
| spec decode latency + compute tax | latency ✅ cost ✅ | latency ✅ (via `rl_svc`); **cost tax ❌** |
| clock/power energy (CostModel `power_scale`) | ✅ | ❌ |
| int4 quality/SLA risk (`PeriodOutcome.quality_sla_risk`) | ✅ | partial — `goodput_per_dollar` property folds it, but no phase context |
| per-replica KV residency (PR #106) | ✅ | ❌ |
| representative workload prompt length | real Azure (median 857) | **output tokens as a proxy** (`_synth_jobs` sets `in_tok = out_tok`, `controller.py:198`) |

**Answers (1-15):** (1) `_rollout_world`. (2) `run_period_episode` world path. (3-9) the table above —
planning sees precision/spec **latency** but not the **cost/GPU-seconds** side, no hybrid cost, no clock
energy, no phase split. (10) planning uses `goodput_per_dollar` but the **provisioned** cost basis, not the
eval's **hybrid** mode → different cost economics. (11) the budget enters `decide` via the adaptive planner
(`search_planner.AdaptiveSearchPlanner`, `controller.py` `use_adaptive_search` branch) — no fixed 256 cap
remains (PR #111). (12) strategies in `search_planner.py`: `_exhaustive`, `_beam`, `_coordinate`,
`_random_restart`, `_cross_entropy` + `search_regret_audit`. (13) PR #111 used `beam_search` (raw 209 952 >
`exhaustive_max`; no regret audit at that size). (14) **evidence the planner missed FP8:**
`DT60_ROOFLINE_MPC_ACTION_DIAGNOSTIC.md` — the MPC selected `precision=bf16` while the static fp8 stack was
+38% and the eval scored the MPC arm −7.5% vs the best static stack; fp8 was *available* and pruned in by
the regime hint, so the miss was a **scoring** miss, not a candidate miss. (15) the required change is
**planning/eval parity**: run the phase/cost/roofline model in `_rollout_world`.

## Root cause (the one-line diagnosis)

In planning, a candidate's only roofline channel is the **latency** factor `rl_svc` applied to a lumped
service time (`world_simulator.py` `_svc` non-phase branch). FP8's value in eval is mostly **cost**
(`gpu_seconds_factor` → realized GPU-seconds → hybrid cost), which the planning path never computes. With
the SLA already met by the synthetic workload, FP8's latency benefit is ~0, so the planner is indifferent
bf16↔fp8 and the tie-break keeps bf16.

## The fix (Phase 2) — and its bounded approximations

Thread a **synthetic unique-prefix `kv_state` + the eval `cost_mode`** into the two `simulate_period` calls
in `_rollout_world`, so planning runs the SAME phase + hybrid-cost + roofline-modulation path as eval. Two
honest approximations, both measured by the parity contract (Phase 1):

1. **No invented prefix reuse** — planning uses unique prefixes (no residency hits), so it does not model
   the routing/cache-reuse benefit (that benefit stays in the existing `kv_service_factor` channel). This is
   conservative (never credits reuse the eval would not).
2. **Prompt-length proxy** — the synthetic workload lacks real prompt tokens; we add a
   `planning_prompt_tokens` hint (the eval window's median) so the regime/phase split is representative.
   Without it, planning uses output tokens as the prompt — a known bias the parity contract measures.

Neither tunes the simulator nor weakens the gate; both preserve `ActionBundle()` defaults reproducing
today's behaviour (the fix is opt-in via `planning_kv_cost_mode`, default `None`).
