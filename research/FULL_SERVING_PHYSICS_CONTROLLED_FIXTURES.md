# Full Serving-Physics Controlled Fixtures (Phase 10)

Every V2 mechanism proven causal, directional, deterministic, and clone-safe — affecting reward ONLY through
TTFT / completion latency / queueing / GPU-seconds / energy / power / memory & bandwidth pressure / SLA / cost.
Fixtures live in `tests/test_v2_serving_physics.py` (28 PASS), the V2 validation suite
`aurelius/environment/v2/validation.py` (22 PASS / 0 WARN / 0 FAIL), the external-formula checks
`aurelius/environment/external_sim_validation.py::external_formula_checks` (7 PASS), and the runnable
`scripts/compare_*_fixture.py`. Numbers below are reproduced deterministically.

| # | fixture | expected physics | observed | where |
|--|--|--|--|--|
| 1 | roofline timing by GPU | H100 < L40S decode/tok | 5.31 ms < 20.58 ms | `test_roofline_timing_changes_by_gpu_type` |
| 2 | roofline timing by model | 70B > 8B decode/tok | 51.3 ms > 5.31 ms | `test_roofline_timing_changes_by_model_size` |
| 3 | precision helps memory-bound decode | fp8 < bf16 decode/tok; lower realized work | 2.65 < 5.31 ms; decode GPU-s ↓ | `test_precision_helps_memory_bound_decode`, `test_precision_reduces_realized_work_and_latency` |
| 4 | precision neutral/harmful when quality dominates | int4 carries quality risk; gp/$ neutral under slack | quality_risk 0.06; dt60 §2 | `test_int4_carries_quality_risk`, DT60 |
| 5 | spec helps memory-bound high-accept decode | aggressive < off decode/tok | 2.65 < 5.31 ms | `test_spec_decode_helps_memory_bound_decode` |
| 6 | spec hurts compute-bound decode | large-batch decode compute-bound → spec ≥ base | regime=compute, spec ≥ base | `test_spec_decode_hurts_compute_bound_decode` |
| 7 | low clock helps memory-bound energy | down-clock neutral on decode time, power ↓ | Δdecode<1e-12; power 1.0→0.7 | `test_low_clock_neutral_on_memory_bound_decode` |
| 8 | low clock hurts compute-bound latency | prefill (compute-bound) slower at low clock | 277 ms > 245 ms | `test_low_clock_hurts_compute_bound_prefill` |
| 9 | co-location helps with real background work | idle reclaimed → cheaper | reclaim>0, gp/$ ↑ (G: 16→11.5 GPU-h) | `test_colocation_helps_with_real_background_work`, DT60 §5 |
| 10 | co-location pruned without background | aggressive ≤ off; candidates pruned to `off` | aggr ≤ off; coloc∈{off} | `test_colocation_inert_without_background_work`, `test_coloc_pruned_without_background_kept_with` |
| 11 | prefill-heavy benefits from prefill pool | high prefill_frac best | p=0.75 (0.451 s) < p=0.25 (0.476 s) | `compare_disaggregation_fixture.py` |
| 12 | decode-heavy benefits from decode pool | low prefill_frac best | p=0.25 (5.1 s) ≪ p=0.75 (3285 s) | `compare_disaggregation_fixture.py` |
| 13 | wrong disaggregation hurts | starved pool blows up | p=0.85 ≫ p=0.25 completion | `test_wrong_disaggregation_allocation_hurts` |
| 14 | HBM cache beats CPU cache | resident HBM chosen; cost order HBM<CPU | tier=GPU_HBM; 0.0005<0.00226 s | `test_hbm_hit_beats_cpu_hit`, `compare_kv_tier_fixture.py` |
| 15 | remote KV beats recompute (long prefix, low load) | remote chosen at low pressure | tier=REMOTE_KV, net +0.0061 s | `test_remote_beats_recompute_...`, `compare_kv_tier_fixture.py` |
| 16 | recompute beats remote under pressure | recompute chosen at high pressure | tier=RECOMPUTE at 0.9/0.97 | same |
| 17 | tiered cache changes value | capacity → hit rate; HBM-only vs full | cap64 ≥ cap4 hit rate; DT60 D | `test_tier_capacity_changes_hit_rate`, DT60 §7 |
| 18 | continuous batching ↑ throughput to saturation | larger token budget → larger batch, lower latency/GPU-s | budget 512→32768: p95 359623→2.88 s, decode GPU-s 4015→93 | `compare_batching_fixture.py` |
| 19 | aggressive batching hurts under tight SLA | saturation tail inflates decode queue | regime=saturated tail penalty | `test_batching_saturation_regime` |
| 20 | beam finds coupled optimum coord-descent misses | beam=optimum, coord stuck | beam 2.01 (regret 0) vs coord 1.01 (regret 0.50, WARN) | `test_beam_finds_coupled_optimum_...`, `compare_mpc_search_strategies.py` |
| 21 | exhaustive confirms optimizer regret | exhaustive = beam optimum | exhaustive 2.01, regret reported | `compare_mpc_search_strategies.py` |
| 22 | legacy scalar preserved as baseline | scalar reproduces TPOT/PREFILL constants | decode = 256·0.020 exactly | `test_legacy_scalar_timing_preserved` |

## Each fixture's economic chain (no direct reward)

Reward is exactly `sla_safe_goodput_tokens / cost_usd` (validated structurally, `no_direct_reward_bonus`).
Every mechanism reaches it through a physical quantity:

- precision/spec/clock → roofline service seconds (compute/memory leg) → TTFT/completion + realized GPU-s + power.
- tiered KV → prefill tokens saved vs transfer/recompute seconds → service time → TTFT.
- pools/handoff → phase queue waits + handoff latency → TTFT/completion + idle GPU-s.
- batching → effective batch → decode amortisation + saturation tail → completion + decode GPU-s.
- co-location → idle GPU-s reclaimed by REAL background work → billed GPU-h → cost (contention → SLA).

## Determinism & isolation

`test_determinism_and_clone_isolation`, `test_clone_state_independent`, and the validation suite's
`deterministic_replay` / `clone_isolation` / `no_future_leakage_causal_cache` prove byte-identical replay,
that candidate evaluation never mutates the real state, and that a request's cache outcome depends only on
earlier admissions. External-formula cross-checks (`external_formula_checks`, 7 PASS) confirm the ported KV
bytes (vLLM/InferSim/llm-analysis), ridge point (llm-analysis/LLM-Viewer), per-stage roofline (InferSim),
KV transfer = bytes/bw (Splitwise), tier ordering (LMCache), and TTFT decomposition (Mooncake).
