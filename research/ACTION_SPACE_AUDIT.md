# Aurelius Action-Space Audit (current optimizer decision variables)

What Aurelius's MPC optimizer can decide **today**, by category, with fidelity and interactions. Source of
truth: `aurelius/environment/actions.py` (`ACTION_SPECS`, the canonical per-surface audit-as-data). This is a
**read-only audit** — no code/sim/reward/action change. It is the baseline for `NEXT_ACTION_SPACE_ROADMAP.md`
(what to add next). 18 surfaces total: **12 CONNECTED** (optimized by default, reach reward), **4
SIMULATED_ONLY** (modelled but opt-in/diagnostic), **2 PLANNED** (represented, never actuated).

**Decision frequency (all surfaces):** one value per **MPC control period** — hourly (dt=3600 s) in the
electricity backtests, configurable via `SimulationClock`. The controller chooses the full bundle each period
from the causal forecast; `hierarchical_search` searches the connected surfaces by control timescale.

**Reward channels (how a connected surface moves goodput/$):** `run_unified_replay` (capacity/ordering/
admission/batching/capacity_multiplier), `kv_service_factor` (routing over the Mooncake prefix trace),
`world_simulator` (prewarm/placement/migration — persistent cluster state), `roofline_serving`
(precision/spec/clock — per-request prefill/decode service-time + GPU-seconds + power). A default bundle is the
no-op → reproduces today's behaviour; each connected surface can also **hurt** (no surface can fake a win).

## CONNECTED surfaces (optimized by default)

| # | surface (field) | category | options (default first) | implementation / reward channel | fidelity | sim-derived? |
|--|--|--|--|--|--|--|
| 1 | **capacity_policy** | capacity / autoscaling | reactive_lag1 · backlog_aware · forecasted_mcs | Erlang-C + live-backlog `CapacityController` → `run_unified_replay` | calibrated control law | partly (replay) |
| 2 | **capacity_multiplier** | capacity | 1.0 · 0.75 · 1.5 | scales sized replica count → `run_unified_replay` | direct multiplier | partly |
| 3 | **ordering_policy** | scheduling | fifo · abs_conformal | class-priority + SRPT + conformal guard → `run_unified_replay` | research-grade latency sched | partly |
| 4 | **admission_policy** | admission | off · class_aware | best-effort deferral under load → `run_unified_replay` | realistic | partly |
| 5 | **batching_policy** | batching / compute | conservative · balanced · aggressive | per-replica continuous-batch concurrency + service inflation (`BATCHING_MODELS` (1,1)/(2,1.15)/(4,1.5)) | **INFERRED** (public prior, not trace-calibrated) | ✅ |
| 6 | **routing_policy** | routing | round_robin · shortest_queue · kv_aware | `fleet_kv_routing` over the Mooncake trace → routing-specific service factor (`kv_service_factor`) | **TRACE_DERIVED** (Mooncake prefix reuse) — best-calibrated | partly (real trace) |
| 7 | **prewarm_policy** | autoscaling / warm pool | off · conservative · aggressive | warm-pool state + cold-start ramp → `world_simulator` (warm replicas avoid cold-start gap; pays warm-hold) | **BENCHMARK_DERIVED** cold-start (vLLM/TGI load regime) | partly |
| 8 | **placement_policy** | placement / topology | topology_blind · rack_local · network_aware | macro rack-locality + v2026 network-pressure service-time discount → `world_simulator` | **MACRO ONLY** (no per-link/NVLink/NVSwitch/PFC-ECN) | ✅ |
| 9 | **migration_policy** | migration | off · conservative · aggressive | live move: cost+capacity-loss+KV-invalidation now, locality benefit next period → `world_simulator` | **BENCHMARK_DERIVED** move cost/duration | ✅ |
| 10 | **precision_policy** | precision / compute | bf16 · fp8 · int4 | roofline: lower precision cuts weight+KV bytes → higher tokens/s in mem-BW-bound regime → `roofline_serving` | **SIMULATOR_INFERENCE**; int4 quality risk INFERRED (no quality model) → int4 diagnostic-only | ✅ |
| 11 | **spec_decode_policy** | speculative execution | off · shallow · medium · aggressive | roofline: draft proposes k / target verifies; helps latency only when decode mem-BW-bound + high acceptance; never a cost win | **SIMULATOR_INFERENCE** acceptance/overhead bands | ✅ |
| 12 | **clock_policy** | energy / compute (DVFS) | base · low · high | roofline DVFS: throughput ~clock, power ~clock^2.4; low clock saves energy in mem-BW-bound regime | **SIMULATOR_INFERENCE** conservative DVFS band; energy is diagnostic (not booked as GPU-h) | ✅ |

## SIMULATED_ONLY surfaces (modelled in the same physics; opt-in / diagnostic, frozen off by default)

| # | surface | category | why not connected (limitation) | roadmap |
|--|--|--|--|--|
| 13 | **colocation_policy** | compute (idle-SM co-location) | NO background-work trace (Azure is all latency-critical) → can only HURT foreground SLA here; generator prunes off | N3 |
| 14 | **prefill_decode_policy** | scheduling (PD disaggregation) | live replay has NO disaggregated prefill/decode capacity pools — only roofline models the split analytically | N4 |
| 15 | **kv_routing_policy** | routing (per-request KV prefix) | Azure trace has no per-request prefix ids; the FLEET KV effect is already CONNECTED via `routing_policy` | N4 |
| 16 | **topology_policy** | networking | no network model in `run_unified_replay`; `net_penalty` unused in reward (macro effect is in `placement_policy`) | N4 |

## PLANNED surfaces (represented, never actuated)

| # | surface | category | what is missing | roadmap |
|--|--|--|--|--|
| 17 | **kv_placement_policy** | storage / cache | `StatefulKVCache` LRU is simulated STATE, not an action; needs an eviction/placement lever + counterfactual sim | — |
| 18 | **energy_policy** | energy | price IS in the objective (`CostModel`), but there is no temporal-shift / power-shape ACTION the simulator honours | N2 |

## Coverage by category (what is already covered vs thin)

| category | covered today | thin / absent |
|--|--|--|
| compute | precision, clock, spec, batching (roofline) | parallelism degree (TP/PP), KV-cache precision (separate from weights) |
| scheduling | ordering (SRPT/conformal), admission | PD disaggregation (SIM only), queue partitioning per SLA class |
| placement | placement (macro topology), migration | per-link/NVLink/NVSwitch, heterogeneous GPU-type assignment |
| routing | routing (fleet KV-aware, TRACE_DERIVED) | per-request prefix routing (SIM), congestion/network-path (SIM) |
| batching | batching (3 presets) | explicit token budget / max_num_seqs / chunked-prefill budget |
| admission | admission (class_aware) | per-class queue partitioning, request hedging/replication |
| precision | bf16/fp8/int4 (weights) | KV-cache quant, fp4/mxfp4, per-layer mixed precision |
| capacity | capacity_policy + multiplier (autoscale + warm pool) | GPU-memory oversubscription / KV-util target, reservation-vs-spot sourcing |
| migration | migration (consolidation) | (covered; magnitude BENCHMARK_DERIVED) |
| autoscaling | capacity (backlog) + prewarm | predictive scale-to-zero, cross-cluster spillover |
| speculative execution | spec_decode | request-level speculation/hedging, draft-model selection |
| topology | placement (macro) | network-path / congestion-aware (SIM only) |
| energy | clock (DVFS, diagnostic) | price-aware temporal shifting ACTION (PLANNED), power cap, carbon objective |
| networking | placement network-pressure (macro) | per-link / collective scheduling (ABSENT) |
| storage | — | KV/cache tiering (PLANNED), checkpoint/weight placement (ABSENT) |

## Interaction map (which knobs couple — matters for search decomposition)

- **roofline cluster** {precision, clock, spec, batching}: all set the *same* per-request prefill/decode
  service-time + GPU-seconds via `roofline_serving`; strongly coupled (e.g. fp8 frees bandwidth that changes
  whether `clock=low` or `spec` pays). `hierarchical_search` groups these in the **medium** timescale.
- **capacity cluster** {capacity_policy, capacity_multiplier, prewarm}: compose multiplicatively on replica
  count / warm-hold; over-provisioning one makes the others redundant. **slow** timescale.
- **service-factor cluster** {routing, placement}: both discount per-request service time (KV reuse / topology)
  — partially substitutable. routing = **fast**, placement = **slow**.
- **load-shedding vs scaling** {admission, capacity}: admission defers best-effort while capacity scales up;
  substitutes under pressure.
- **migration ↔ placement**: migration only pays if it moves toward a better `placement` target.

## Honest status

- The **best-calibrated** lever is `routing_policy` (TRACE_DERIVED from Mooncake). The **highest-impact but
  least-calibrated** levers are the roofline cluster (precision/clock/spec, all SIMULATOR_INFERENCE) — they
  drove the PR #124 headline and are the priority for pilot telemetry (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`).
- Search regret over this space is **~0** (PR #123 tournament + PR #124 Phase F): the search now finds the best
  bundle *within* this action space. So the next marginal gain is more likely from **expanding the action
  space** than from changing the search — the premise of `NEXT_ACTION_SPACE_ROADMAP.md`.
