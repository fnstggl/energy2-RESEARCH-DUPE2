# Prefill/Decode + Economics Gap Audit (PR #107, Phase 0)

Audit of the current serving/economic path **before** changing it, to explain why PR #106 saved 108k
prefill tokens but moved no goodput/$, and to scope the bridge.

## Where each quantity is computed today

1. **Service time** — `aurelius/benchmarks/srtf_serving_backtest.py:312`:
   `_service_time_s(out) = TTFT_BASE_S(0.150) + out·TPOT_S(0.020)`. A **constant** 0.15 s prefill/TTFT
   term + an output-token-driven decode term (50 tok/s/seq). Applied in `world_simulator.simulate_period`
   as `service_s = _service_time_s(out) · service_factor`.
2. **KV hits reduce service time** — `world_serving.simulate_residency_serving` (PR #106) returns a
   per-request `service_factor` that **multiplies the whole `_service_time_s`** (prefill *and* decode).
3. **TTFT** — not separately computed; approximated by the constant `TTFT_BASE_S = 0.15 s`. It does **not**
   depend on prompt length or on the KV hit.
4. **Queue delay** — `run_unified_replay` (cluster replay, c interchangeable servers); `wait = start_s −
   arrival_s`. One blended queue, no prefill/decode phases.
5. **GPU-hours** — `unified_replay.py:535`: `gpu_hours = Σ_tick c_per_tick · tick_seconds / 3600`, the
   integral of **active server count** over the period. `c_per_tick` adapts to backlog but floors at
   `warmup_c` (≈4). At light load it sits **at the floor** regardless of service speed.
6. **Cost → gp/$** — `cost_model.operator_cost(gpu_hours, …)`; `gp/$ = sla_safe_goodput / cost`.

## Why PR #106 saved prefill but did not move gp/$ — two compounding causes

**Cause A — there was no prompt-driven prefill to reduce, and the residency scaled the wrong thing.**
Prefill in the live model is the *constant* `TTFT_BASE_S`; it does not depend on prompt tokens. So a KV
hit (which physically skips re-prefilling matched prompt blocks) had **no prompt-prefill term to cut**.
Worse, PR #106's per-request `service_factor` multiplied the **entire** `_service_time_s` — so a KV hit
sped up **decode too**, which is **physically wrong** (KV reuse skips prefill, never decode). The headline
"108k prefill tokens saved" reduced a service factor that was acting on the decode-dominated total, an
overstatement of where the saving lands.

**Cause B — cost is a capacity-integral pinned to the floor.** `gpu_hours = Σ c_per_tick·tick`. At the
diagnostic's light load, `c_per_tick` ≈ `warmup_c` (the floor) every tick, so `gpu_hours ≈ floor ·
period_time` **independent of how fast each request is served**. Faster service finishes requests sooner
but does not drop the active-server floor → cost is flat → gp/$ cannot move via cost. It could only move
via SLA-safe goodput, and the SLA was already met.

## Answers to the audit questions

| # | question | answer |
|--|--|--|
| 1 | where is service time computed | `_service_time_s` (constant prefill + decode) × residency `service_factor` in `simulate_period` |
| 2 | where do KV hits reduce service | `world_serving` per-request factor — but it scales the **whole** service (bug: should be prefill-only) |
| 3 | where is TTFT | nowhere explicit; ≈ the constant `TTFT_BASE_S`, prompt-/KV-insensitive |
| 4 | where is queue delay | `run_unified_replay`, one blended queue (no phases) |
| 5 | where are GPU-hours | `Σ c_per_tick·tick/3600` — capacity integral, floor-bound at light load |
| 6 | where does cost enter gp/$ | `cost_model.operator_cost(gpu_hours)`; gp/$ = goodput/cost |
| 7 | why #106 saved prefill but no gp/$ | Cause A (no prompt-prefill term + whole-service scaling) + Cause B (floor-bound cost) |
| 8 | optimistic scalar baselines | `kv_service_factor_by_routing` (fleet scalar, credits reuse to cold requests) — `legacy_kv_scalar_optimistic` |
| 9 | actions that already affect serving | routing, capacity_multiplier, batching, prewarm, placement, migration (all connected) |
| 10 | what must become phase-specific | **prefill** (prompt-driven, KV-reducible), **decode** (output-driven, KV-insensitive), **TTFT** vs completion, **GPU-seconds** (realized vs provisioned), queueing, SLA |

## What this PR builds (scope)

1. **Prompt-driven prefill, decode untouched.** `prefill_work_s = TTFT_BASE_S + prefill_tokens_remaining ·
   PREFILL_S_PER_TOKEN (+ model_cold)`; `decode_work_s = out · TPOT_S`. `prefill_tokens_remaining =
   prompt − prefix_hit` (from PR #106 residency). KV reuse cuts **prefill only**; decode is unchanged —
   fixing Cause A.
2. **Realized GPU-seconds + cost modes.** `realized_gpu_seconds = Σ(prefill_work+decode_work)`. Three
   explicit `cost_mode`s — `provisioned_capacity` (reproduces #106, floor-bound), `realized_serving_work`
   (upper-bound counterfactual: cost follows realized work), `hybrid_capacity_work` (provisioned floor +
   service can reduce active-replica-seconds, bounded) — fixing Cause B *honestly* (no mode is free).
3. **Phase diagnostics + TTFT vs completion**, baseline cleanup (relabel the optimistic scalar; add a
   realistic cold-cache fair baseline), validation, fixtures, a bounded dt=60 across cost modes.

**Honest expectation (stated up front, not after the fact):** because Azure outputs make serving
**decode-bound** (decode ≈ out·0.02 s ≫ prefill ≈ 0.15 s + prompt·~0.00015 s), KV reuse will improve
**TTFT** strongly but **completion latency / realized GPU-seconds only marginally** — so even with
realized-work cost, the Azure gp/$ gain should be small, and prefill-heavy fixtures should benefit far
more. The bridge is built and measured honestly; the held-out monetization is bounded by the workload's
decode-bound nature, which the diagnostic will quantify.
