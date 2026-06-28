# dt=60 World-Model Realism Diagnostic (PR #105)

**Question.** After the PR #105 realism work (replica/migration **identity**, **cold-start
decomposition**, **migration KV-preservation correction**), does sub-(warm-timeout) control at
**dt = 60 s** make prewarm / migration / placement valuable?

**Verdict.** **No — still not.** Over a deterministic 6-hour Azure window, 360 control decisions × H ∈
{1, 4, 12, 24} (1,440 decisions), the MPC selects **prewarm 0×, migration 0×, placement `topology_blind`
360×** at every horizon. The gp/$ edge over the fair operator (+5.4 … +6.1 %) is again bought by
**under-provisioning** (`capacity_multiplier = 0.75`, which *lowers* GPU-hours 9.15 vs 9.94 and *raises*
SLA violations 0.0156 vs 0.0112) → **claim gate `False` at every dt/H**. No Pareto-safe improvement.

This is the honest, expected outcome of landing the *scaffolding* (identity + cold-start components +
the migration surcharge fix) **without** the per-replica KV-residency benefit channel (Gap 2), and it
identifies that channel as the **next** missing mechanism.

> SIMULATED directional evidence on the calibrated world model, not production telemetry. No calibration
> tuned to a result; the Pareto gate is unchanged.

## Run

`scripts/sweep_mpc_horizon.py --dt-seconds 60 --horizons 1,4,12,24 --eval-span-hours 6
--max-eval-periods 360 --risk-weight 0.3` → `data/external/mpc_controller/mpc_dt60_world_realism.json`.
World validation: **15 PASS / 0 FAIL / 5 SKIPPED** (`run_world_validation()`).

## Results — dt = 60 s (fair: gp/$ 71 133, SLA 0.0112, q_p95 9.83 s, GPU-h 9.94)

| H | look | gp/$ | Δ% | SLA | q_p99 | GPU-h | **prewarm** | **placement** | **migration** | cap | batch | gate |
|--|--|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 1 m | 74 992 | +5.4 | 0.0162 | 4.64 | 9.19 | off×360 | blind×360 | off×360 | 0.75 | conservative | T/F/F |
| 4 | 4 m | 75 320 | +5.9 | 0.0159 | 4.60 | 9.16 | off×360 | blind×360 | off×360 | 0.75 | conservative | T/F/F |
| 12 | 12 m | 75 416 | +6.0 | 0.0157 | 4.57 | 9.15 | off×360 | blind×360 | off×360 | 0.75 | conservative | T/F/F |
| 24 | 24 m | 75 470 | +6.1 | 0.0156 | 4.55 | 9.15 | off×360 | blind×360 | off×360 | 0.75 | conservative | T/F/F |

Routing shifts toward `kv_aware` as H grows (23→85 of 360) — the controller does adapt the *connected*
KV-routing lever — but the *stateful* actions stay at their no-op.

## Comparison vs the previous (PR #104, pre-realism) dt = 300 s

| | dt=300 (PR #104, pre-realism) | dt=60 (PR #105, post-realism) |
|--|--|--|
| prewarm selected | 0 / 1440 | **0 / 1440** |
| migration selected | 0 / 1440 | **0 / 1440** |
| placement non-blind | 0 / 1440 | **0 / 1440** |
| any headline gate True | No | **No** |
| gp/$ lever | capacity 0.75 + kv routing | capacity 0.75 + kv routing |

(Absolute gp/$ is not comparable across the two — different real-time windows/scale; the **mix and gate**
are the comparable signals, and both are unchanged.)

## What changed because of the richer world model — and what didn't

- **Migration surcharge corrected (behavioral):** the old flat `cache_factor = 1.04` became ≈ 1.004
  (conservative), so migration is no longer *strictly dominated by a surcharge*. But its only remaining
  **benefit** is the small macro topology discount (≤ 0.08), which still does not exceed the $0.40 move
  cost + capacity loss on this load → migration stays off. The correction was necessary but not
  sufficient.
- **Cold-start decomposition (fidelity, not behavioral):** decomposing 30 s into engine/model/kv/ready
  does **not** change the *total* a warm or prewarmed replica avoids, so prewarm economics are unchanged;
  on a warm-seeded cluster with this load the reactive baseline rarely cold-starts, so prewarm has little
  to avoid and its warm-hold cost dominates → prewarm stays off.
- **Replica identity (scaffolding):** `weights_loaded` / `kv_warm_frac` are now conserved across moves
  and cooldowns (validated), but `kv_warm_frac` is **not yet consumed** by the serving service-time, so
  it cannot yet pay off — that is exactly Gap 2.

## Is the gp/$ edge real or SLA-shedding? (unchanged answer)

SLA-shedding. At every H the MPC beats fair gp/$ only by `capacity_multiplier = 0.75`, which lowers
GPU-hours (9.15 vs 9.94) and **raises** SLA violations (0.0156 vs 0.0112). The Pareto clause rejects it
(`headline_allowed = False`). No gain comes from state reuse, because the state-reuse channel is not yet
wired into serving economics.

## Next missing production-reality mechanism (the honest pointer)

**Gap 2 — per-replica Mooncake prefix residency consumed by the serving service-time.** Today
`kv_cache.py` models per-server KV residency and locality routing, but only as an **offline fleet-scalar
`service_factor`**; the persistent replicas carry `kv_warm_frac` but the serving replay does not read it.
Until a routed/placed/migrated replica's *resident prefixes* actually reduce its serving time:

- **migration** has no benefit beyond the ≤ 0.08 topology discount (so it can't beat its move cost);
- **placement** has no per-replica locality to exploit (so `topology_blind` is optimal);
- **prewarm**'s value stays capped at avoiding occasional cold-starts the reactive pool already covers.

Wiring Gap 2 (assign Mooncake-derived prefix signatures to Azure requests; route to the replica holding
the most resident leading prefix; migration moves residency, cold-start loses it) is the smallest change
that gives migration and placement a *real* benefit channel — and is the recommended next increment. The
calibration plan and sources are in `research/WORLD_MODEL_REALISM_GAP_AUDIT.md` (Gap 2); the validation
hooks are already stubbed `SKIPPED` in `world_validation.py`.

## Safe vs unsafe claims

**Safe:** the world model now conserves replica/migration identity and decomposes cold-start with cited
bands (15/0 validation); migration is no longer dominated by an unrealistic flat KV surcharge; at dt=60
the MPC still does not find prewarm/migration/placement worthwhile, and its gp/$ edge is SLA-shedding,
not state-reuse. **Unsafe (not claimed):** that the richer world model makes any deferred action
valuable; that prewarm/migration/placement are Pareto-safe at any dt/H tested; any per-replica KV-reuse
serving benefit (Gap 2 deferred).
