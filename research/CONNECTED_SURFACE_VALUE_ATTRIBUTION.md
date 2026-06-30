# Connected-Surface Value Attribution (Phase G)

Where does `aurelius_mpc_hierarchical_search`'s gp/$ advantage actually come from — and how much of it leans on
**simulator-inferred** connected surfaces whose magnitudes are not production-calibrated? This attributes the
win **honestly**, separating the two distinct comparisons it wins, and labelling each surface's fidelity. The
headline is genuine and Pareto-safe; this document is the part that says *which levers* produce it and *which of
those we trust how much*.

Evidence: `data/external/mpc_controller/ladder_benchmark.json` (Phase E, 3 markets × expensive window) and
`data/external/mpc_controller/search_method_tournament.json` (PR #123). Magnitudes are **SIMULATOR_INFERENCE**
unless noted; the *directions* are robust.

## The two comparisons win for DIFFERENT reasons

**This is the central honesty point.** "Aurelius beats X" attributes differently depending on X:

### 1. hierarchical_search vs the core-grid search (PR #123) → the CONNECTED surfaces

The core-grid planner can only vary `{capacity_multiplier, batching, precision, clock}`. Hierarchical reaches
the **connected** surfaces the grid cannot. The selected bundles on the real markets (identical across
pjm/ercot/caiso):

| surface | core-grid winner | hierarchical winner | connected? |
|--|--|--|--|
| capacity_policy | (default reactive) | **forecasted_mcs** | ✅ CONNECTED |
| placement_policy | (default topology_blind) | **network_aware** | ✅ CONNECTED |
| admission_policy | (default off) | **class_aware** | ✅ CONNECTED |
| routing_policy | (default round_robin) | **kv_aware** | ✅ CONNECTED |
| ordering_policy | (default fifo) | **abs_conformal** | ✅ CONNECTED |
| capacity_multiplier / batching / precision / clock | 0.75 / aggressive / fp8 / high | (same) | core grid |

Reaching those connected surfaces **roughly doubles** gp/$ over the core-grid optimum: pjm 709,283 →
1,454,636 (**+105%**), ercot 818,987 → 1,703,036 (+108%), caiso 783,859 → 1,660,408 (+112%). **This is the
PR #123 "reach" finding** — the search-architecture value, and the reason hierarchical wins the tournament.

### 2. hierarchical_search vs `production_scheduler` (this ladder) → the ECONOMIC arbitrage

`production_scheduler` is **not** a core grid — it is a realistic scheduler that *already uses the connected
serving-stack surfaces* (kv-aware routing, rack-local placement, backlog autoscaling, class admission,
continuous batching). So against it, the connected surfaces are **largely shared**, and Aurelius's marginal
win comes from the levers production_scheduler **deliberately forgoes** (it runs the deployed model as-is). The
per-decision mixes (every market, all 3 decisions):

| surface | production_scheduler | hierarchical (Aurelius) | this is… |
|--|--|--|--|
| routing_policy | kv_aware | kv_aware | **same** (no edge here) |
| placement_policy | rack_local | network_aware | connected (small: topo factor 0.987) |
| migration_policy | off | off | **same** |
| prewarm_policy | off | off | **same** |
| capacity_multiplier | 1.25 (SLA headroom) | **0.75** (consolidate) | economic (cost ↓) |
| batching_policy | balanced | **aggressive** | economic (throughput ↑) |
| precision_policy | bf16 | **fp8** | economic arbitrage (lossless-safe) |
| clock_policy | base | **high** | economic arbitrage (roofline/DVFS) |
| spec_decode_policy | off | **aggressive** | economic (throughput ↑) |

So vs production_scheduler the **+137%/+159%/+148%** edge is **mostly roofline / economic arbitrage**
(fp8 + high clock + aggressive spec/batching) **plus capacity consolidation** (0.75 vs 1.25) — exactly the
edge the production baseline is defined NOT to have. The connected-surface contribution here is small
(`network_aware` over `rack_local` is a ~1.3% topology service-time factor; routing is identical). This is the
honest decomposition: **Aurelius's whole point — economic optimisation of the deployed stack — is what beats a
strong production scheduler, not a connected-surface trick the baseline lacks.**

## Fidelity of each lever (what to trust how much)

| lever | direction | magnitude fidelity | note |
|--|--|--|--|
| `precision=fp8` | robust | **SIMULATOR_INFERENCE** (roofline) | lossless-safe; `quality_sla_risk_mean=0.0` (NOT int4) → headline-safe |
| `clock=high` | robust | SIMULATOR_INFERENCE (power/roofline) | the N2 mechanism; gated by SLA slack |
| `spec_decode=aggressive` | robust | SIMULATOR_INFERENCE | throughput model, not pilot-measured |
| `batching=aggressive` | robust | SIMULATOR_INFERENCE | `BATCHING_MODELS` (4.0, 1.5) continuous-batch point |
| `capacity_multiplier=0.75` | robust | TRACE-anchored cost, inferred SLA effect | consolidation; SLA stayed 0.0, so safe here |
| `routing=kv_aware` | robust | **TRACE_DERIVED** (Mooncake prefix reuse) | `mean_kv_prefix_hit_rate=0.952`; the best-calibrated lever |
| `placement=network_aware` | robust | SIMULATOR_INFERENCE (`TOPOLOGY_MAX_DISCOUNT=0.08`) | small effect (topo factor 0.987) |
| `capacity_policy=forecasted_mcs` | robust | SIMULATOR_INFERENCE | drives the #123 core-grid gap, not the production-scheduler gap |

**int4 is excluded** from every headline arm (`allow_quality_risk=False`); `quality_sla_risk_mean=0.0`
confirms no quality-risked lever entered the winning bundle. The one **best-calibrated** lever (kv-aware
routing, TRACE_DERIVED) is *shared* with production_scheduler, so it is not where the edge comes from — the
edge is in the SIMULATOR_INFERENCE roofline/economic levers. **That is the honest caveat: the headline
magnitude rests on the simulator's roofline economics (fp8/clock/spec), whose direction is robust but whose
exact size is not production-validated** (`research/WORLD_MODEL_ROBUSTNESS_AUDIT.md`).

## What would change the attribution

- **A quality model for int4** would add a lever (currently excluded) — could raise the ceiling, but only if
  quality is provably protected.
- **Pilot telemetry on fp8 / clock / spec throughput** would convert the dominant levers from
  SIMULATOR_INFERENCE to measured — the single biggest fidelity upgrade for this headline.
- **A larger-scale workload** (req_cap ≫ 56) would re-price the warm pool and capacity levers (the backtest
  scale currently makes an eager warm pool look pathological — see `PRODUCTION_SCHEDULER_BASELINE_RESULTS.md`),
  potentially making `prewarm` worthwhile for both arms.

## One-line attribution

> vs the **core-grid search**, hierarchical wins by **reaching connected surfaces** (placement / capacity-policy
> / admission / routing) — the search-architecture "reach" value (SIMULATOR_INFERENCE magnitude). vs the
> **production_scheduler**, it wins by **economic arbitrage of the deployed stack** (fp8 + high clock +
> aggressive spec/batching + capacity consolidation) — the edge the production baseline is defined not to have,
> also SIMULATOR_INFERENCE in magnitude, robust in direction, and **quality-safe** (no int4).
