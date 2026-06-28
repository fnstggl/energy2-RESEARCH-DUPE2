# Prefill/Decode Serving Physics Audit (PR #107, Phase 1)

Public serving mechanisms and the honest approximation we implement for each, with a fidelity tier
(`TRACE_DERIVED` · `TRACE_DERIVED_REUSE_MODEL` · `BENCHMARK_DERIVED` · `PUBLIC_PAPER` ·
`SIMULATOR_INFERENCE` · `ABSENT`). No proprietary behavior; numbers are public spec bands.

## A. Prefill phase
| mechanism | fidelity | this PR |
|--|--|--|
| prompt tokens processed before first token | TRACE_DERIVED (Azure `in_tok`) | **implemented** (`prefill_tokens = prompt`) |
| prefill tokens remaining after KV hit | TRACE_DERIVED_REUSE_MODEL (Mooncake) | **implemented** (`prompt − prefix_hit`) |
| prefill throughput (tokens/s) | BENCHMARK_DERIVED | **implemented** (`PREFILL_S_PER_TOKEN`; vLLM/Sarathi prefill ≈ 5–15k tok/s on A100/H100 → ~0.00015 s/tok base) |
| TTFT = prefill queue + prefill work | PUBLIC_PAPER (DistServe/Splitwise) | **implemented** (TTFT computed separately) |
| compute-bound prefill | PUBLIC_PAPER (roofline) | approximate (per-token rate; full roofline deferred) |
| chunked prefill (Sarathi) | PUBLIC_PAPER | **deferred** (audited; not material to this diagnostic) |

## B. Decode phase
| mechanism | fidelity | this PR |
|--|--|--|
| output tokens generated iteratively | TRACE_DERIVED (Azure `out_tok`) | **implemented** (`decode = out·TPOT_S`) |
| per-token decode (memory-bound, ~50 tok/s/seq) | BENCHMARK_DERIVED (`TPOT_S=0.02`) | **implemented** (existing constant, KV-insensitive — the fix) |
| active sequence occupancy | PUBLIC_PAPER (vLLM continuous batching) | **implemented** (active-decode count tracked) |
| decode memory-bandwidth pressure | PUBLIC_PAPER | approximate (batching factor; full HBM-BW roofline deferred) |
| max concurrent sequences | BENCHMARK_DERIVED (KV mem) | **implemented** (concurrency cap from cache capacity) |

## C. Prefill/decode interference
| mechanism | fidelity | this PR |
|--|--|--|
| prefill can block decode on shared GPUs | PUBLIC_PAPER (Sarathi/Orca) | approximate (combined service feeds the shared cluster queue) |
| disaggregation isolates phases | PUBLIC_PAPER (DistServe/Splitwise) | **implemented as a model**: prefill vs decode work are separate terms; a `disaggregated` flag is **deferred** |
| long decode bottlenecks even when prefill saved | PUBLIC_PAPER | **implemented** (decode-bound completion despite prefill savings — the key honest effect) |

## D. Batching / concurrency
| mechanism | fidelity | this PR |
|--|--|--|
| batch size ↔ throughput/latency | PUBLIC_PAPER (Orca/vLLM) | **implemented** (batching factor on decode + concurrency) |
| continuous batching → decode occupancy | PUBLIC_PAPER | approximate (active-sequence occupancy) |
| memory/KV pressure limits concurrency | BENCHMARK_DERIVED | **implemented** (cache capacity → concurrency cap) |
| chunked prefill TTFT/throughput tradeoff | PUBLIC_PAPER (Sarathi) | **deferred** |

## E. Cost / economics
| mechanism | fidelity | this PR |
|--|--|--|
| period-capacity GPU-hour accounting | SIMULATOR_INFERENCE (existing) | **implemented** (`provisioned_capacity` mode — reproduces #106) |
| realized GPU-second accounting | SIMULATOR_INFERENCE | **implemented** (`realized_serving_work` mode — upper bound) |
| occupancy-weighted / hybrid cost | SIMULATOR_INFERENCE | **implemented** (`hybrid_capacity_work` — provisioned floor + service reduces active-replica-seconds) |
| SLA-safe goodput | existing | **implemented** (unchanged gate) |
| TTFT-SLA vs completion-SLA | PUBLIC_PAPER | partial (TTFT reported; SLA stays on completion as today) |

## F. Baseline fairness
| baseline | fidelity | label |
|--|--|--|
| round_robin / shortest_queue, no cache | SIMULATED | fair (no-cache references) |
| legacy fleet KV scalar | SIMULATOR_INFERENCE | **optimistic** (credits reuse to cold requests) — reference only, not a fair headline comparator |
| realistic cache-aware, no persistent identity | SIMULATED | fair (causal fleet cache, cold requests pay full prefill) |
| persistent per-replica KV (PR #106) | TRACE_DERIVED_REUSE_MODEL | the candidate |

## Calibration bands used (public)
- `PREFILL_S_PER_TOKEN`: 0.00007 / **0.00015** / 0.0004 s/token (≈ 14k / 6.7k / 2.5k prompt tok/s) —
  BENCHMARK_DERIVED (vLLM/Sarathi prefill throughput on A100/H100-class GPUs).
- `TPOT_S` = 0.020 s/token (existing, 50 tok/s/seq) — BENCHMARK_DERIVED.
- `TTFT_BASE_S` = 0.150 s (existing minimum prefill/first-token overhead) — BENCHMARK_DERIVED.
- batching decode factor: conservative 1.0 / balanced 0.9 / aggressive 0.8 until a saturation point,
  then a tail penalty — SIMULATOR_INFERENCE bounded by Orca/vLLM qualitative curves.

## Explicitly NOT modelled here (ABSENT / deferred, with reason)
- speculative decoding, DVFS — out of scope (not requested).
- remote KV tiers, staged dirty-block migration, full roofline, chunked prefill, multi-model/adapter
  precision residency, tenant boundaries, fragmentation, cancellation, autoscaling lifecycle — **deferred**
  to the closure claim table (`PREFILL_DECODE_ECONOMICS_CALIBRATION.md`), each with why a conservative
  approximation is or isn't safe and the telemetry that would unblock it. None is UNKNOWN.
