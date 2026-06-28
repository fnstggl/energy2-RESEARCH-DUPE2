# Prior Model-Affinity / Prewarm Gain — Audit (Phase 0)

Before building the cache/locality domain, I traced the earlier result that motivated this work
(+89.5 % SLA-safe goodput/$, −21.7 % GPU-hours, ~62 % of gain from model-affinity/prewarm) to its
actual mechanism, to port the *real* physics — not a heuristic — into the canonical ClusterState.

## Where the result lives

- `research/results/alibaba_genai_third_trace_2026-06-24.md` — honest headline **+38.2 %** vs
  `constraint_aware_no_affinity` (the +89.46 % is vs `sla_aware`, which the repo itself flags as
  **under-baselined**: no baseline modelled model-affinity despite 87 distinct base models).
- `docs/ALIBABA_GENAI_ABLATION_RESULTS.md` — "primarily a model-affinity/prewarming effect (~62 % of gain)".
- `data/external/alibaba_genai/processed/alibaba_genai_ablation_summary.json` — Shapley: **affinity 61.7 %**,
  anticipatory sizing 38.3 %, interaction ~0 %.
- `data/external/benchmark_rollup/baseline_capability_matrix.json` — explicitly: +89 % is under-baselined;
  a fair baseline with affinity collapses the margin to ~+34 % (the sizing Shapley term).

## The mechanism (real, not a bonus)

`aurelius/optimizer/policies/genai_serving.py:48-77`, `genai_effective_service_s(...)`:

```
affinity:      switch_rate = min(1.0, distinct_models / n)
no-affinity:   switch_rate = 1.0 if distinct_models > 1 else 0.0
cold_s = switch_rate·basemodel_load + lora_frac·(…)·lora_load + controlnet_frac·(…)·controlnet_load
service_s = mean_exec_s + cold_s
```

- **What it is.** Affinity routing co-locates same-base-model requests on model-warm replicas, so the
  base-model reload is **amortised over `distinct_models` arrivals** (`switch_rate = distinct_models/n`)
  instead of risking a cold load per request. Cold-start drops 22.85 s → 2.79 s (~6.7×).
- **How it flows.** `service_s` ↓ → Erlang-C `mu` ↑ → fewer replicas to meet the SLA
  (`genai_size_for_sla`) → fewer GPU-hours → higher goodput/$. It propagates through **genuine M/M/c
  queueing** (`aurelius/simulation/cluster/serving.erlang_c_wait_s`), **not** a scalar reward bonus.
- **Calibration.** Cold-start medians (basemodel 22 s, LoRA 4.4 s, ControlNet 3.9 s) are Alibaba GenAI
  pipeline-layer **measured medians**. Causal (current-tick `distinct_models/n`, no future truth). No
  oracle. The affinity lift is consistent (+33…+80 %) across all 5 sizing strategies → orthogonal,
  additive (Shapley interaction ≈ 0).

## The decisive insight for THIS PR

**The prior gain is a MODEL-residency effect on a MULTI-MODEL fleet (87 base models).** It is *avoiding
model-load cold-starts* by routing same-model-together — distinct from KV-prefix reuse (avoiding
*prefill recompute*). The gain scales with `distinct_models`: on a **single-model** stream
`distinct_models = 1` → `switch_rate = 1/n` (affinity) vs `0` (no-affinity) → **no affinity benefit**
(slightly negative). The canonical Azure conv world is effectively **single-model**, which is a
structural reason PR #104/#105 saw no prewarm/migration value: the proven channel was *absent from the
workload*, not just unmodelled.

## What to port (and how)

Two residency channels, same shape (residency on a replica → routing to a match avoids a cost → lower
service time/TTFT → fewer replicas / higher goodput/$), both **causal**:

| channel | residency | routing avoids | needs | source |
|--|--|--|--|--|
| **Model-affinity** (the prior winner) | `replica.model_id` (weights loaded) | model-load cold-start | multi-MODEL stream | genai cold-start medians (BENCHMARK_DERIVED) |
| **KV-prefix locality** (this PR's main ask) | per-replica resident prefix blocks | prefill recompute → TTFT | prefix REUSE stream (Mooncake) | Mooncake `hash_ids` (TRACE_DERIVED_REUSE_MODEL) |

**Port plan.** Re-implement both through persistent `ClusterState` per-replica residency + a per-request
routing sim inside the world simulator (`world_serving.py`), where the benefit flows ONLY through
service-time/TTFT/queue/cost (the genai pattern), never a bonus. The model-affinity formula is ported
faithfully from `genai_effective_service_s`; the KV-prefix channel reuses the existing causal paged-LRU
`StatefulKVCache`. The Azure+Mooncake dt=60 diagnostic exercises the **KV-prefix** channel (single
model); a **multi-model fixture** exercises the **model-affinity** channel (reproducing the prior
mechanism in the canonical sim). Do **not** copy the genai sizing monolith — extract the service-time
physics only.
