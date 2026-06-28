# Roofline MPC Action Controlled Fixtures (Phase 10)

Each required fixture, proven in `tests/test_roofline_mpc_actions.py` (21 tests) +
`world_validation._roofline_action_checks` (12 checks). Every action helps / hurts / is neutral in the
**physically-correct roofline regime**, through the SAME `roofline.serving_point` physics the live path and
diagnostics use — no bonus. Regimes are *constructed, not tuned*: `MEM` = memory-bandwidth-bound decode
(H100, batch 8, long context); `COMP` = compute-bound decode (H20, batch 128) — both asserted in
`test_regimes_are_what_we_claim`.

| # | user fixture | proven by |
|--|--|--|
| 1 | memory-bound decode where precision helps | `test_fp8_helps_memory_bound_both_latency_and_cost` (decode ×<1 AND cheaper) |
| 2 | compute-bound prefill where precision helps less | `test_precision_helps_less_when_prefill_compute_bound` (prefill_factor≈1) |
| 3 | memory-bound decode where spec helps latency | `test_spec_helps_memory_bound_latency_but_pays_a_compute_tax` |
| 4 | compute-bound workload where spec hurts | `test_spec_hurts_or_neutral_compute_bound` (decode_factor ≥ 1) |
| 5 | memory-bound where low clock saves energy | `test_low_clock_saves_power_memory_bound` (power<1, decode ≈1) |
| 6 | compute-bound where low clock hurts | `test_low_clock_hurts_latency_compute_bound` (decode>1) |
| 7 | compute-bound where high clock helps but costs more | `test_high_clock_helps_compute_bound_latency_at_higher_power` |
| 8 | co-location helps with background work + SM headroom | `test_colocation_useful_only_with_background_and_sm_headroom` |
| 9 | co-location hurts under memory pressure / compute-bound | `test_colocation_hurts_more_when_compute_bound` |
| 10 | no-background-work → co-location off/pruned | `test_colocation_credits_no_background_goodput_without_a_trace` + `…frozen_off_with_reasons` |
| 11 | batching optimum changes with precision | `test_precision_benefit_depends_on_batch` |
| 12 | batching optimum changes with spec decode | `test_spec_benefit_depends_on_regime_set_by_batch` |
| 13 | unified controller selects different bundles by regime | `test_planner_prunes_differently_by_regime` |
| 14 | Pareto gate blocks SLA-shedding | `world_validation: pareto_gate_blocks_sla_shedding` + `test_int4_carries_quality_risk` (int4 SLA-unsafe) |
| 15 | candidate pruning keeps runtime bounded | `test_planner_reports_regret_and_is_bounded` (beam < exhaustive, regret measured) |

Plus the honesty fixtures: `test_neutral_bundle_is_exactly_no_op` (defaults reproduce),
`test_determinism`, `test_serving_point_is_the_single_physics_source` (the factor IS a serving_point ratio —
no separate magic path), `test_high_clock_does_not_magically_help_memory_bound_throughput` (clock cannot fake
bandwidth), `test_int4_carries_quality_risk_unlike_fp8`.

## Each fixture reports (per the spec)

The factors a fixture asserts map directly to the operator-facing quantities: `decode_factor`/`prefill_factor`
→ TTFT + completion latency; `gpu_seconds_factor` → GPU-seconds → cost; `power_factor` → energy;
`quality_sla_risk` → SLA; `interference_factor` → co-location foreground penalty; the planner's `SearchPlan`
→ raw candidate count, strategy, candidates evaluated, estimated regret, selected bundle, runtime. The dt=60
diagnostic (`DT60_ROOFLINE_MPC_ACTION_DIAGNOSTIC.md`) reports the same quantities on the real Azure workload
with the Pareto gate.

## Why each action helped or hurt (the regime law)

- **precision** helps whenever decode is **memory-bandwidth-bound** (fewer bytes → more bandwidth-bound
  tokens/s) and is ~neutral on compute-bound prefill; `int4` adds a quality/SLA risk → not a clean win.
- **speculative decoding** is a **latency** lever in the memory-bound regime (spare compute verifies the
  draft) but pays a **compute tax** (`gpu_seconds_factor > decode_factor`) → not automatically a cost win;
  it hurts when compute-bound (FLOP-limited).
- **clock** trades latency for energy: low clock saves power but slows a compute-bound phase; high clock buys
  compute-bound latency at higher energy; it cannot move memory-bandwidth-bound throughput.
- **co-location** credits **no** background goodput without a trace (only interference) and hurts more when
  compute-bound (no idle SMs) — so the live planner freezes it off with a recorded reason.
- **batching** interacts with precision/spec by setting the arithmetic intensity (regime), so the optimum
  shifts with the other actions — the reason the planner uses **beam search** (captures interactions) and a
  **regret audit** (measures what an approximate search lost).
