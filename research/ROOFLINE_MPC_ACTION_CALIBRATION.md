# Roofline MPC Action Calibration (Phase 9)

How each roofline action reaches the reward, with the equation, source, confidence, validation metric, and
the production telemetry that would upgrade it. **One physics law** (`roofline.serving_point`) supplies every
mechanism delta as a **ratio vs the neutral config**, applied to the live calibrated service times +
realized GPU-seconds + power (`roofline_actions.py` → `prefill_decode.compute_phase_serving` /
`world_simulator.simulate_period`). At default policies every ratio is exactly `1.0`, so the live path is
bit-for-bit unchanged (validated: `roofline_action/neutral_defaults_reproduce`). **No effect touches reward
directly** — only through TTFT / service time / queueing / GPU-seconds / energy / SLA / cost.

## The modulation (shared by all actions)

```
neutral = serving_point(workload, cfg(all defaults))
action  = serving_point(workload, cfg(bundle policies))
decode_factor = action.decode_gpu_seconds / neutral.decode_gpu_seconds      # decode wall-clock (latency)
prefill_factor = action.prefill_gpu_seconds / neutral.prefill_gpu_seconds   # prefill wall-clock (latency)
gpu_seconds_factor = action.serving_gpu_seconds / neutral.serving_gpu_seconds  # COMPUTE cost (billing)
power_factor = action.power_w / neutral.power_w                             # DVFS energy (→ cost via power_scale)
interference_factor = action.coloc_penalty / neutral.coloc_penalty         # co-location foreground penalty
```
`compute_phase_serving` applies `prefill_factor`/`decode_factor`×`interference_factor` to per-request service
times (→ queue/SLA/goodput) and `gpu_seconds_factor`×`interference_factor` to realized GPU-seconds (→ cost);
`simulate_period` passes `power_factor` to `CostModel.operator_cost(power_scale=…)` (energy term only).

| action | status | equation (in `roofline.py`) | source | confidence | validation metric | limitation / telemetry to upgrade |
|--|--|--|--|--|--|--|
| **precision** (`bf16`/`fp8`/`int4`) | CONNECTED | weight & KV bytes ∝ `PRECISION_BYTES`; lower bytes → higher bandwidth-bound tokens/s | PUBLIC_SPEC (dtype sizes) + PUBLIC_PAPER (roofline) | **high** (bytes are exact; the roofline reduction is standard) | `precision_helps_memory_bound`; fixture `test_fp8_helps_memory_bound…` | no quality model → `int4` quality risk is INFERRED (`PRECISION_QUALITY_RISK`); telemetry: per-task accuracy vs precision |
| **speculative decoding** (`off`/`shallow`/`medium`/`aggressive`) | CONNECTED | accepted tokens cut serial steps (`spec_speedup`); draft+verify add FLOPs (`gpu_seconds`×(1+overhead·0.5)) | PUBLIC_PAPER (Leviathan/Medusa) + SIMULATOR_INFERENCE (acceptance bands) | **medium** (acceptance is workload- and draft-model-specific) | `spec_latency_win_pays_compute_tax`, `spec_hurts_compute_bound` | acceptance bands are a prior, not a specific draft model; telemetry: measured acceptance per draft/target pair |
| **clock / DVFS** (`base`/`low`/`high`) | CONNECTED | compute ∝ clock; bandwidth ≈ flat; power ≈ `tdp·(0.4+0.6·clock^2.4)` | PUBLIC_PAPER (DVFS power) + SIMULATOR_INFERENCE (exponent) | **medium** | `clock_changes_power_not_memory_bw`; fixtures 5–7 | energy booked only via the `power_scale` energy term (NOT as GPU-hour savings); telemetry: measured power-vs-clock curve per GPU |
| **roofline-aware batching** (`conservative`/`balanced`/`aggressive`) | CONNECTED | existing `BATCH_DECODE_FACTOR` + saturation; roofline batch sets the precision/spec interaction | PUBLIC_PAPER (Orca/vLLM) + INFERRED magnitudes | **medium** | `test_precision_benefit_depends_on_batch` | own magnitudes are the existing band; telemetry: continuous-batching throughput/latency curve |
| **co-location** (`off`/`conservative`/`aggressive`) | SIMULATED_ONLY (frozen) | idle-SM background work only when memory-bandwidth-bound; interference `coloc_penalty` always | SIMULATOR_INFERENCE | **low** (no background trace) | `no_imaginary_background_goodput` | **no background-work trace** → credits ZERO useful work, only interference; telemetry: a batch/flex job stream with class + SLA |
| **prefill/decode allocation** (`shared`/`p40_d60`/`p60_d40`) | SIMULATED_ONLY (frozen) | disaggregated capacity split + handoff (`serving_mode`) | PUBLIC_PAPER (DistServe/Splitwise) | **low** (no live pools) | covered by `roofline.sweep` | the **live cluster replay has no disaggregated pools** → analytical only; telemetry: a real prefill/decode-pool deployment |

## Honesty invariants (locked in `world_validation._roofline_action_checks`, 12 PASS)

- `neutral_defaults_reproduce` — defaults change nothing.
- `no_direct_reward_bonus` — `affects_reward ⇔ CONNECTED`; every effect is physics.
- `int4_carries_quality_risk` — lower precision is never free.
- `no_imaginary_background_goodput` — co-location credits nothing without a trace.
- `connected_live_simulated_frozen_split` — precision/spec/clock live; co-location + prefill/decode frozen
  with recorded reasons.
- `candidate_pruning_is_regime_aware` + `search_regret_is_measured` — the planner proposes int4/spec only in
  the regime that can use them, and **measures** the loss of approximate search vs exhaustive.
- `pareto_gate_blocks_sla_shedding` — a cheaper-but-less-safe arm (e.g. int4 quality risk) is not a headline.

## Fidelity summary

Production-safe: the **bytes** (precision), the **roofline regime** (AI vs ridge, trace + PUBLIC_SPEC), and
the **direction** of every effect. Simulator-inferred: the **magnitudes** (spec acceptance, DVFS exponent,
co-location interference). Absent (labelled): a quality model for int4, a background-work trace for
co-location, disaggregated capacity pools for prefill/decode. Nothing is UNKNOWN.
