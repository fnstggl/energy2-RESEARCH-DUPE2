# Roofline MPC Action Gap Audit (Phase 0)

Audit of the current MPC action space before turning roofline mechanisms (precision, batching,
speculative decoding, clock/power, co-location, prefill/decode allocation) into first-class controller
action surfaces. Every claim is grounded in `file:line`. **No implementation precedes this audit.**

## 1. Which actions already exist?

`ActionBundle` (`aurelius/environment/actions.py:189-212`) is a frozen dataclass with **16 fields**, each
defaulting to its no-op / reward-path value (`ActionBundle()` == today's behaviour). The canonical
metadata is `ACTION_SPECS` (`actions.py:80-180`), one `ActionSpec` per surface carrying
`status / options / sim_param / fidelity / reward_channel`.

| field | status today | options | reward channel |
|--|--|--|--|
| capacity_policy | CONNECTED | reactive_lag1, backlog_aware, forecasted_mcs | run_unified_replay |
| ordering_policy | CONNECTED | fifo, abs_conformal | run_unified_replay |
| admission_policy | CONNECTED | off, class_aware | run_unified_replay |
| routing_policy | CONNECTED | round_robin, shortest_queue, kv_aware | kv_service_factor |
| capacity_multiplier | CONNECTED | 1.0, 0.75, 1.5 | run_unified_replay |
| batching_policy | CONNECTED | conservative, balanced, aggressive | run_unified_replay + compute_phase_serving |
| prewarm_policy | CONNECTED | off, conservative, aggressive | world_simulator |
| placement_policy | CONNECTED | topology_blind, rack_local, network_aware | world_simulator |
| migration_policy | CONNECTED | off, conservative, aggressive | world_simulator |
| kv_routing_policy | SIMULATED_ONLY | off, prefix_affinity | (fleet effect already via routing) |
| topology_policy | SIMULATED_ONLY | off, net_aware | (unused in reward) |
| kv_placement_policy | PLANNED | lru, reuse_aware | — |
| **clock_policy** | **PLANNED** | nominal, low, high | — (this PR) |
| **precision_policy** | **PLANNED** | full, fp8, int8 | — (this PR) |
| **spec_decode_policy** | **PLANNED** | off, on | — (this PR) |
| energy_policy | PLANNED | off, defer_to_cheap | — (out of scope) |

**Key finding:** `clock_policy`, `precision_policy`, `spec_decode_policy` already exist as bundle fields
but are **PLANNED** — represented, never optimized, never consumed by physics (`actions.py:163-174`).
There is **no** `colocation` or `prefill_decode_allocation` field yet.

## 2. Which actions are currently optimized by MPC?

The 9 **CONNECTED** surfaces (`CONNECTED_SURFACES`, `actions.py:183`). The planner searches whole
bundles (`ModelPredictiveEconomicController.decide`, `controller.py:211-308`) via
`CandidateBundleGenerator.search` (`candidate_search.py:88-115`). With 9 connected surfaces the space is
`3·2·2·3·3·3·3·3·3 = 8 748` combinations > `EXHAUSTIVE_BUDGET=256`, so the controller **already runs
coordinate descent** (`candidate_search.py:95-115`), cost ≈ `surfaces·options·passes`, **not** a Cartesian
product. Adding surfaces therefore adds *bounded* search cost.

## 3. Which roofline mechanisms are simulated but not selected?

`roofline.py` (`serving_point`, `ServingConfig`, `sweep_mechanism`) fully simulates **all six** mechanisms
(precision, batching, prefill/decode split, spec decode, clock, co-location) — but it is called **only**
from `scripts/diagnose_serving_roofline_dt60.py:75,76,90`, **never** from `controller.py`,
`world_simulator.py`, or `training.py`. Confirmed by `roofline.py:246` self-labelling
`live = mechanism == "batching"`. So: batching is the only mechanism with a live action surface; the rest
are diagnostic sweeps. **This PR closes that gap for precision/spec/clock** and keeps
prefill/decode-allocation + co-location modelled-but-not-live (see Q10, Q4).

## 4. Where should each new action live?

The live serving physics is **not** in `roofline.py`; it is in `world_simulator.simulate_period`
(`world_simulator.py:276-447`) → `prefill_decode.compute_phase_serving` (`prefill_decode.py:86-124`,
per-request `prefill_work_s`/`decode_work_s`/`service_s`/`realized_gpu_seconds`) + `run_unified_replay`
(cluster queue). roofline.py is a *standalone* analytical model.

**Design: a single physics law, applied as a no-op-anchored modulation.** A new module
`roofline_actions.py` maps the bundle's action policies → a `roofline.ServingConfig`, calls
`roofline.serving_point` at the **action** config and the **neutral** config for the period's
representative workload, and returns the **ratios** (`decode_factor`, `prefill_factor`,
`gpu_seconds_factor`, `power_factor`, `ttft_factor`, plus the decode regime / AI / ridge and a precision
quality-risk). At neutral defaults the two configs are identical → every factor is exactly `1.0` →
`simulate_period` is bit-for-bit unchanged (satisfies "old defaults reproduce previous results"). The
ratios are applied inside `compute_phase_serving` (prefill × `prefill_factor`, decode × `decode_factor`)
and to `realized_gpu_seconds`/energy. **The live model keeps its calibrated absolute levels
(`TPOT_S`, `PREFILL_S_PER_TOKEN`); roofline supplies only the relative mechanism delta.** No reward bonus:
every effect flows through service time → queue/SLA/goodput and GPU-seconds → cost.

## 5. How are candidate bundles cloned/evaluated?

`decide` (`controller.py:211`) scores each candidate via `_eval` (`controller.py:258-273`) →
`_rollout_world` (`controller.py:161-209`), which **clones** the world
(`clone_world_state_for_candidate` = `world_state.clone()` = `copy.deepcopy`, `world_state.py:197`) and
rolls the first action `H` steps on the clone (`mutate=True` on the clone only; the real world is touched
only when the chosen action is committed in `run_period_episode`, `controller.py:493-499`). A
`mutate=False` scoring call is a pure read (`world_simulator.py:295-297`). Reward is `gp/$` derived from
`sla_safe_goodput / operator_cost` (`controller.py:157`, `_gpd`) minus a risk penalty
(`controller.py:202` `reward = exp_gpd - risk_weight·risk_viol·exp_gpd`) — **never a scalar bonus**.

## 6. How are diagnostics reported?

`EpisodeReport` (`controller.py:362-392`) carries per-period mixes (`routing_mix`, `batching_mix`,
`capacity_multiplier_mix`, `prewarm/placement/migration_mix`) + `realized_gpu_seconds`, `mean_ttft_p95`,
`queue_delay_p95/p99`, etc., accumulated in `run_period_episode` (`controller.py:439-566`). **This PR adds**
`precision_mix`, `spec_decode_mix`, `clock_mix`, `colocation_mix`, `prefill_decode_mix`, plus roofline
diagnostics (decode regime mix, mean decode AI / ridge, power_w, energy_j). The per-decision search
diagnostics live in `last_decision_diag` (`controller.py:299-305`) and `SearchReport`
(`candidate_search.py:118-137`).

## 7. Which existing tests guard fake wins?

- `training.py:231-264` `claim_gate` — headline requires `beats_fair_baseline AND pareto_sla_not_worse AND
  no_oracle AND splits_disjoint` (`controller`-independent gate).
- `tests/test_mpc_training.py`: `test_claim_gate_blocks_when_not_beating_fair_baseline`,
  `test_claim_gate_allows_only_when_beats_and_clean`,
  `test_claim_gate_blocks_when_win_comes_from_more_sla_violations` (the SLA-shedding guard).
- `tests/test_action_surface.py`: `test_status_counts_match_audit` (asserts the CONNECTED count) and the
  enumerate-bundle count test — **both must be updated** when precision/spec/clock become CONNECTED.
- `validate_action_bundle` (`action_registry.py:54-67`) rejects a PLANNED surface set away from its no-op
  (prevents actuating a fake knob). After promotion, precision/spec/clock become legally actuatable;
  co-location + prefill/decode stay SIMULATED_ONLY (actuatable only on explicit opt-in).

## 8. What state must be added for precision/spec/clock/co-location?

None of these need *persistent cross-period* state — they are per-period serving settings, like
`batching_policy`. They are read from the bundle in `simulate_period` and fed to the modulation. The plumbing
that must change:
- `run_period_episode` merge list (`controller.py:447-448`) — add the 5 new fields so the chosen values reach the world path.
- the `pol = SimpleNamespace(...)` (`controller.py:478`) — currently carries only prewarm/placement/migration; it must also carry `batching_policy` (a latent bug: `compute_phase_serving` reads `getattr(bundle,"batching_policy","balanced")` at `world_simulator.py:355`, so today the world path *always* sees "balanced") **and** the 5 new fields.
- `roofline_actions` needs the period's representative workload (median prompt/decode from `recs`) + GPU type (`fleet_state.gpu_type_mix`) + batch (from `batching_policy`). All already available in `simulate_period`.

## 9. What baselines must be updated?

To keep wins honest, the fair claim-gate baseline must be a **competent** operator who already uses the
best *static* roofline setting — otherwise "MPC discovers fp8" is an unfair win over a strawman who never
quantizes. The Phase-11 diagnostic adds strong static baselines (e.g. a fixed-fp8 operator, a fixed-batch
operator) so the headline requires the **adaptive, regime-aware** controller to beat the best **static**
roofline choice — not merely to beat bf16/conservative.

## 10. What could make this PR misleading?

1. **A precision "win" with no quality cost.** fp8/int4 are not free: lower precision risks output
   quality/SLA. We attach a conservative quality-risk to fp8/int4 and label int4 diagnostic-grade unless a
   quality model exists.
2. **Co-location inventing background goodput.** No background-work trace exists
   (`ReplicaState.workload_class`, `world_state.py:65`, is defined but **never read/written**; the Azure
   trace is all latency-critical). Co-location is implemented (idle-SM credit + interference) but **pruned
   off by default with a documented reason**; it can only *hurt* foreground SLA here, never manufacture
   useful work.
3. **Prefill/decode allocation claimed as structurally live.** The live cluster replay has no disaggregated
   prefill/decode capacity pools — only `roofline.serving_point` models the split analytically. So
   prefill/decode allocation stays **SIMULATED_ONLY** (diagnostic), not a structurally-live action.
4. **Clock "energy savings" booked as GPU-hour savings.** Clock's honest live effect is via GPU-seconds in
   the *compute-bound* regime; its energy effect is reported as a diagnostic (`power_w`, `energy_j`) and only
   reaches cost through the legitimate energy term — never by pretending energy = GPU-hours.
5. **Searching a fake knob.** Anything not causally wired stays non-CONNECTED; `validate_action_bundle`
   blocks actuating it. The candidate generator records *why* each pruned surface is frozen.

## Verdict — what becomes live vs modelled

- **New live CONNECTED actions:** `precision_policy` (bf16/fp8/int4), `spec_decode_policy`
  (off/shallow/medium/aggressive), `clock_policy` (low/base/high); `batching_policy` made roofline-aware.
  Each is two-sided (helps in one regime, hurts in another), causal through TTFT/service/GPU-seconds/energy,
  no-op at default.
- **Modelled, not live (SIMULATED_ONLY, documented):** `prefill_decode_policy` (no disaggregated capacity
  pools in the live replay) and `colocation_policy` (no background-work trace) — fully simulated in the same
  `serving_point` physics, swept diagnostically, pruned off by the generator with a recorded reason.
