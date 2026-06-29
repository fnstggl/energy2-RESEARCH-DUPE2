# Full V1 Production-Physics Promotion Results

Results of promoting the canonical production-realistic serving physics into V1 and making it the default where
validation supports it. Companion to `FULL_V2_TO_V1_PHYSICS_PROMOTION_PLAN.md` and the authoritative
`PRODUCTION_DEFAULT_DECISIONS.md` table.

## What was promoted / made default

- **Roofline GPU/model-aware base timing → canonical default** (was opt-in in PR #119). Legacy scalar retained
  only as `timing_model="legacy_scalar"` / `AURELIUS_TIMING_MODEL=legacy_scalar`. See
  `V1_ROOFLINE_DEFAULT_ON_RESULTS.md`.
- **Already default-on (confirmed, recorded in the decisions table):** precision/spec/clock connected actions,
  roofline-aware batching factor, cache lookup + eviction pressure, kv-aware routing, placement/migration/
  prewarm, regime-aware candidate generation, adaptive MPC search with **measured regret**, production
  diagnostics + provenance.

## What remains non-default (each with a specific technical blocker)

| Mechanism | Status | Blocker |
|--|--|--|
| Continuous-batching token budget / chunked prefill | config / follow-up | V1 has no per-iteration scheduler (additive architecture) |
| Prefill/decode pools | config (frozen) | live replay has no disaggregated pools |
| Tiered KV / remote-vs-recompute / cross-tier transfer | separate migration PR | V1 is single GPU-HBM LRU tier; needs new tier state + residency telemetry |
| Co-location | config (frozen) | no background-work trace in public data |
| Legacy scalar timing | legacy-only | the old, less-realistic model — kept for regression only |

No row is gated merely to preserve benchmark numbers (policy requirement).

## Did the simulator become more production-like?

**Yes, for the world_simulator / persistent-state MPC path.** The roofline default removes the L40S-class
scalar's phantom SLA violations on fast GPUs (`compare_v1_legacy_vs_v1_production_physics.py`, SLA 8 s):

| GPU | legacy SLA viol | production (roofline) SLA viol | phantom on legacy? |
|--|--|--|--|
| H100 | 0.700 | **0.092** | yes |
| A100 | 0.700 | **0.417** | yes |
| L40S | 0.700 | 0.725 | no (scalar is L40S-class) |

## Benchmark: all-MPC-knobs vs the strongest SLA-aware baseline (the requested test)

Ran the full optimizer (all connected knobs: capacity/ordering/admission/routing/batching/prewarm/placement/
migration/precision/spec/clock) through the canonical two-clock environment on the public Azure+Mooncake+v2026
fixtures via `fair_backtest`, scored by SLA-safe goodput/$ with the Pareto headline gate:

| arm | role | result |
|--|--|--|
| `greedy_packing` | **strongest SLA-aware fair baseline** | reference |
| `aurelius_state_conditioned` (all MPC knobs) | candidate | **−9.5 % to −10.7 %** vs fair baseline |
| **headline claim allowed** | Pareto gate | **False** |

**Honest result:** the all-knobs MPC does **not** beat the strongest SLA-aware baseline (greedy_packing) on
this window — it is ~10 % behind on SLA-safe goodput/$, so no headline is claimed. This is **identical under
legacy and roofline** timing, because the canonical two-clock env prices serving with a *separate*
`ServingPlane._service_time_s`, which this PR does not change. The MPC's value on this public window is not a
goodput/$ win; the gate correctly blocks an unsupported claim. (The `greedy_packing` baseline is a strong
capacity-packing heuristic; the MPC's stateful actions — prewarm/migration/placement — do not pay off on this
short, low-pressure window, consistent with the PR #104–#108 sub-hour diagnostics.)

## Are gp/$ changes realism corrections or optimization wins?

- The roofline default's effect (world_simulator path) is a **realism correction** — removing phantom SLA
  violations + pricing realized GPU-seconds correctly, through the existing cost/goodput channels. No direct
  reward bonus (`no_direct_reward_bonus` validation check).
- The MPC-vs-baseline benchmark shows **no optimization win** on this window (−10 %); the gate blocks a
  headline. No fake gp/$ was manufactured.

## V1 production vs V2

V2 (`aurelius/environment/v2/`, PR #110) is not on main and is not imported. Its validated *mechanism*
(roofline base timing) is what was promoted; its full world model (tiered KV, pools, continuous batching) is
not promoted (architectural — see the decisions table). The V2 controlled fixtures remain the reference for the
deferred mechanisms.

## Remaining production-telemetry gaps (proprietary, pilot-only)

Per-(GPU,model) measured MFU; real per-replica KV residency + cross-node transfer bandwidth (tiered KV);
measured spec-decode acceptance; per-GPU power-vs-clock curves; a background-work stream (co-location);
disaggregated prefill/decode pool deployment. All labelled; none claimed as measured.

## Recommended next PR

1. **Promote roofline timing into the canonical `ServingPlane._service_time_s`** so the two-clock env /
   `fair_backtest` benchmark is also GPU/model-aware (the one place the default flip does not yet reach).
2. **Per-replica GPU resolution** (vs the dominant-GPU heuristic) for heterogeneous fleets.
3. **Tiered KV migration** (new world-state tier model + remote-vs-recompute) as its own PR.
4. Profiled-MFU (Vidur-style) calibration as a validation baseline.
