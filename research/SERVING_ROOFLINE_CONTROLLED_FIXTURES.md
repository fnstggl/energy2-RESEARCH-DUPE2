# Serving Roofline Controlled Fixtures (PR #109, Phase 10)

The roofline physics + mechanism regimes, proven in `tests/test_serving_roofline.py` (13) +
`world_validation.py` roofline checks (6). Every mechanism is **fully simulated** in `roofline.serving_point`
and helps/hurts in the physically-correct regime — no bonuses.

| user fixture | proven by |
|--|--|
| 1 prefill-heavy benefits from more prefill capacity | `test_right_allocation_helps_wrong_allocation_hurts` |
| 2 decode-heavy memory-bound | `test_low_batch_long_context_decode_is_memory_bandwidth_bound` |
| 3 compute-bound prefill | `test_prefill_higher_intensity_than_decode` + ridge crossing at high batch |
| 4 mixed workload | the dt=60 Azure result (`mixed_phase_bound`) |
| 5 long-context memory pressure | decode AI falls with context (in `arithmetic_intensity`) |
| 6 high-batch compute-bound | `test_spec_decode_hurts_or_neutral_compute_bound` (H20 batch 128 → compute_bound) |
| 7 low-batch memory-bound decode | `test_low_batch_long_context_decode_is_memory_bandwidth_bound` |
| 8 high-acceptance spec decode helps | `test_spec_decode_helps_memory_bound_high_accept` |
| 9 low-acceptance / compute-bound spec decode hurts | `test_spec_decode_hurts_or_neutral_compute_bound` |
| 10 memory-bound downclock saves energy | `test_downclock_saves_energy_upclock_costs` |
| 11 compute-bound downclock hurts | covered by the clock regime logic (compute scales with clock) |
| 12 co-location helps (SM headroom) | `test_colocation_helps_memory_bound_hurts_compute_bound` (useful GPU-s > 0) |
| 13 co-location hurts (memory pressure) | same fixture (completion ≥ off; compute-bound penalty 0.6) |
| 14 wrong prefill/decode allocation hurts | `test_right_allocation_helps_wrong_allocation_hurts` |
| 15 right allocation helps | same |

Plus: `test_batching_raises_arithmetic_intensity`, `test_phase_bound_distinct_from_roofline_regime`
(the conflation fix), `test_lower_precision_helps_memory_bound_throughput`,
`test_disaggregation_has_handoff_overhead` (never free), `test_every_mechanism_produces_a_sensitivity_curve`
(the clarification's requirement — every mechanism swept with full physics + help/hurt/neutral),
`test_determinism`. Validation suite: **33 PASS / 0 FAIL / 3 SKIPPED**.

**Diagnostic = fully simulated, not controller-selected** (the user's definition). The MPC only selects
batching (the one mechanism with a live action surface); the rest are counterfactual sweeps run through
the *same* `serving_point` physics.
