# Aurelius Connected Actions + Results

This PR connects the first **simulatable** infrastructure action beyond capacity/ordering/
admission — **KV-aware routing** — into the canonical reward path, makes the MPC optimize whole
`ActionBundle`s via a `CandidateBundleGenerator`, and re-evaluates honestly against routing-
enabled baselines with the Pareto-aware claim gate. Audit: `AURELIUS_SIMULATABLE_ACTION_
CONNECTION_AUDIT.md`. Architecture: `AURELIUS_ACTION_SURFACE_AND_MPC_ARCHITECTURE.md`.

## What was connected (and how it reaches the reward)

| action | status before → after | simulator effect | fidelity / provenance |
|---|---|---|---|
| **routing_policy** (round_robin / shortest_queue / kv_aware) | SIMULATED_ONLY → **CONNECTED** | `fleet_kv_routing` replays the Mooncake prefix trace across N server caches; the fleet prefix-reuse **depth** sets a service-time discount (`kv_service_factor`) → goodput/$ | reuse = TRACE_DERIVED (Mooncake FULL_TRACE); applying it to Azure serving = SIMULATED (fleet channel, no row-join) |
| capacity_policy | already CONNECTED | `CapacityController` sizes replicas → GPU-hours/queue | — |
| ordering_policy, admission_policy | already CONNECTED | dispatch + best-effort deferral | — |

The connected bundle space is now **capacity(3) × ordering(2) × admission(2) × routing(3) =
36 bundles**, searched exhaustively by the `CandidateBundleGenerator` (which reports
dimensions, combinations, candidates evaluated, method, best bundle, and a per-surface
ablation — no connected knob is silently excluded).

## Validation — KV-aware routing is a real lever (causal, Mooncake held-out)

On the committed Mooncake fixture (12,031 real prefix requests), per routing policy:

| policy | mean prefix reuse depth | prefill tokens saved | service factor |
|---|---|---|---|
| round_robin | 0.187 | 337,152 | 0.832 |
| shortest_queue | 0.190 | 340,272 | 0.829 |
| **kv_aware** | **0.271** | **703,136** | **0.757** |

`kv_aware` co-locates shared prefixes → ~**45% more reuse depth** and ~**2× the prefill tokens
saved** of round-robin → a strictly smaller service factor (more discount). The binary hit
rate is saturated (~1.0) and does not discriminate — reuse **depth** is the honest metric. The
router is causal (scores only blocks admitted by earlier requests; no future-request oracle) —
`tests/test_action_connection.py`.

## Held-out evaluation (full 2024 one-week Azure trace, 42 hourly eval periods)

Baselines include routing-enabled ones (`*_kv_routing`) so the MPC must beat a strong baseline
that *already* uses the best routing — it cannot win merely by discovering routing.

| arm | SLA-safe goodput/$ | SLA-violation | routing chosen |
|---|---|---|---|
| **mpc_controller** | **198,300** | 0.0433 | kv_aware (42/42) |
| aurelius_canonical_kv_routing (fair baseline) | 195,413 | 0.0138 | kv_aware |
| sla_aware_kv_routing | 191,649 | 0.0124 | kv_aware |
| aurelius_canonical | 174,963 | 0.0200 | round_robin |
| fifo_weak (weak ref) | 174,537 | 0.1106 | round_robin |
| sla_aware | 172,288 | 0.0176 | round_robin |
| greedy_packing | 168,180 | 0.1425 | round_robin |

**The headline finding — connecting one real action moved goodput/$ by ~12%.** Every arm that
adopts kv-aware routing (≈191–198k gp/$) beats every round-robin arm (≈168–175k): e.g.
`aurelius_canonical` **174,963 → 195,413 (+11.7%)** purely by routing kv-aware. That is far
more than per-period switching over capacity/ordering/admission ever delivered — **the binding
constraint really was the connected action space**, exactly the motivation for this PR. The
MPC correctly selects `kv_aware` in all 42 periods.

**The MPC's *joint* edge is small and not a headline.** Against the strongest routing-enabled
fair baseline (`aurelius_canonical_kv_routing`), the MPC is **+1.48%** gp/$ — but at a higher
SLA-violation rate (0.0433 vs 0.0139). The Pareto clause therefore blocks the claim:
`pareto_sla_not_worse = false → headline_claim_allowed = false`
(`data/external/mpc_controller/evaluation_report.json`). Cheaper, not safer — reported
honestly, not forced. The value delivered by this PR is the **+12% from connecting routing**
(captured by any policy that adopts it), not a controller headline.

## Safe vs unsafe claims

**Safe:**
- "Aurelius now optimizes over a larger connected action bundle in the canonical simulator —
  capacity, ordering, admission, **and KV-aware routing** — searched as whole bundles by a
  candidate generator with ablation, scored on SLA-safe goodput/$ against strong SLA-aware and
  routing-enabled baselines with a Pareto-aware claim gate."
- "KV-aware routing is connected through a causal, Mooncake-validated fleet-reuse channel; it
  measurably changes the service factor and therefore goodput/$."

**Unsafe:**
- "Aurelius optimizes every GPU-fleet knob." (It optimizes four connected surfaces; batching,
  placement, migration, prewarming, energy-shifting, clock/DVFS, precision and speculative
  decoding remain PLANNED/SIMULATED_ONLY with documented missing pieces.)
- Any headline savings the gate does not allow.

## Next actions to connect (unchanged priority)

Prewarming first (the forecast already beats naive; it needs only warm-pool state + a
cold-start tax), then clock/DVFS and batching once their physical models exist. See the
per-action plan in `AURELIUS_ACTION_SURFACE_AND_MPC_ARCHITECTURE.md`.
