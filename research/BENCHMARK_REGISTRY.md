# Aurelius Benchmark Registry

> **Canonical registry of all public traces, frozen-synthetic benchmarks,
> and evaluation datasets used or considered by Aurelius.** Maintained by
> the autonomous research routine. Every committed benchmark appears here
> with source, link, schema, and assessment.
>
> **Binding rules:**
> - Every entry must have a verified source URL and license.
> - Simulator / trace results are directional only (NOT production savings).
> - No benchmark may be added without a schema-level audit.
> - `docs/RESULTS.md` §8 production-claim gate is not satisfied by any
>   trace or synthetic in this registry.

---

## 1. Committed Benchmarks (in rollup)

### 1.1 Azure LLM Inference Dataset 2024
- **ID:** `azure_llm_2024_week`
- **Source:** DynamoLLM HPCA 2025 (Microsoft Azure)
- **Link:** https://github.com/Azure/AzurePublicDataset (LLM Inference Dataset 2024)
- **Workload type:** llm_serving
- **Size:** 44,107,694 requests · 9 days · 12,960 ticks @ 60 s
- **Artifact:** `data/external/azure_llm_2024/processed/`
- **Strengths:** Largest committed trace; real production Azure LLM traffic;
  covers 9 days including diurnal + weekly patterns.
- **Weaknesses:** No per-request TTFT/TPOT/latency labels; no session/cache
  signal; no model identity; failure mode derived from `GeneratedTokens==0`.
- **Aurelius headline:** +25.75% SLA-safe goodput/$ vs `sla_aware`; -21.2%
  GPU-hours; SAFE (p99 / timeout ≤ 0.5× FIFO).
- **Notes:** Oracle analysis-only result: goodput/$ 2,422,789. Unsafe
  `utilization_aware` baseline excluded (timeout 12.10% > 10% gate).

### 1.2 Azure LLM Inference Trace 2023
- **ID:** `azure_llm_2023_conv`
- **Source:** Microsoft Azure (academic release)
- **Link:** https://github.com/Azure/AzurePublicDataset
- **Workload type:** llm_serving (conversational)
- **Size:** 19,366 requests · 0.003 days
- **Artifact:** `data/external/azure_llm/processed/`
- **Aurelius headline:** +19.86% vs `sla_aware`.

### 1.3 BurstGPT
- **ID:** `burstgpt_v1`
- **Source:** HPMLL / lzzmm (HuggingFace)
- **Link:** https://huggingface.co/datasets/lzzmm/BurstGPT
- **License:** CC-BY-4.0
- **Workload type:** llm_serving
- **Size:** 17,689 requests · 34 minutes
- **Artifact:** `data/external/burstgpt/processed/`
- **Aurelius headline:** +1.77% vs `cache_affinity_baseline`.
- **Notes:** Cache affinity proxy is model-level (not real KV hit rate).

### 1.4 Alibaba GenAI 2026 (GenTD26)
- **ID:** `alibaba_genai_2026`
- **Source:** Alibaba (academic release)
- **Link:** https://github.com/alibaba/AlibabaSystemTraces (GenTD26)
- **Workload type:** llm_serving (stable diffusion + LLM mixed)
- **Size:** 26,392 requests
- **Artifact:** `data/external/alibaba_genai/processed/`
- **Aurelius headline:** +89.46% vs `sla_aware` (model-affinity driven).
- **Notes:** Strongest trace for model-affinity / prewarm effect. Has
  `model_load_latency`, GPU duty-cycle, queue size — richest signal.

### 1.5 Alibaba Cluster Trace GPU v2023
- **ID:** `alibaba_gpu_v2023`
- **Source:** Alibaba (academic release)
- **Link:** https://github.com/alibaba/clusterdata
- **Workload type:** gpu_packing
- **Size:** 6,282 jobs · 149.3 days
- **Artifact:** `data/external/alibaba_gpu/processed/`
- **Aurelius headline:** SAFE TIE vs `best_fit` (CA already at frontier).

### 1.6 Microsoft Philly
- **ID:** `philly_training`
- **Source:** Microsoft (academic release)
- **Link:** https://github.com/msr-fiddle/philly-traces
- **Workload type:** training_gpu_scheduling (fixture scale)
- **Size:** 33 jobs (fixture) · 0.007 days
- **Artifact:** `data/external/philly/processed/`
- **Aurelius headline:** SAFE TIE vs `best_fit`.
- **Notes:** Fixture-scale (full ~1 GB LFS trace not committed).

### 1.7 MIT Supercloud Bounded Real Sample
- **ID:** `mit_supercloud_bounded`
- **Source:** MIT Supercloud datacenter challenge (S3)
- **Link:** https://supercloud.mit.edu
- **Workload type:** training_gpu_scheduling
- **Size:** 10,000 jobs · 55.9 days (~3 MB of ~1-2 TB full archive)
- **Artifact:** `data/external/mit_supercloud/processed/`
- **Aurelius headline:** +16% vs `best_fit`; SAFETY_WIN (FIFO starvation
  47%, queue p99 56h → UNSAFE baseline).

### 1.8 Canonical Energy Backtest
- **ID:** `canonical_energy_backtest`
- **Source:** Frozen synthetic (1,000 jobs on real CAISO/PJM/ERCOT prices)
- **Link:** `aurelius/benchmarks/golden/canonical_energy_backtest.json`
- **Workload type:** energy_flexible_workload
- **Size:** 1,000 jobs · 26 days
- **Aurelius headline:** +11% vs `current_price_only` at 0 deadline misses.
- **Notes:** `robust_energy_standalone` has lower energy cost but 143
  deadline misses → UNSAFE; always excluded from headline.

---

## 2. Committed Analysis / Frontier Audits (excluded from rollup aggregate)

| ID | type | summary |
|---|---|---|
| `azure_2024_safe_utilization_frontier` | frontier_static | Static SUF +13% over CA on Azure 2024 |
| `azure_2024_dynamic_frontier` | frontier_dynamic | 73.2% oracle-alpha capture |
| `azure_2024_dynamic_frontier_calibration` | calibration | 91.07% oracle-alpha (95% target NOT met) |
| `cross_trace_frontier_generalization` | generalization | 4 applicable, 2 skipped |
| `alibaba_genai_ablation` | ablation | Full-trace ablation for model-affinity signal |
| `alibaba_genai_residency_decision` | residency | n=60 per-request (diagnostic only) |

---

## 3. Ingested HF Datasets (not in rollup — research/training use only)

| dataset | license | rows | tier | signals |
|---|---|---:|---|---|
| `eth-easl/swissai-serving-trace` | research | 67,190 | Tier 4 | `reuse_percentage`, latency, model id |
| `semianalysisai/cc-traces-weka` | Apache-2.0 | 136,118 | Tier 5 | KV block hashes, TTFT, agentic structure |
| `asdwb/cara_latency_prediction` | — | 76,825 (train_flat) | Tier 3 | TTFT, E2E, queue state, GPU type |
| `sammshen/lmcache-agentic-traces` | MIT | 4,976 | Tier 5 | routing proxy, cache reuse |
| `lzzmm/BurstGPT` (HF) | CC-BY-4.0 | 59,999 | Tier 4 | arrivals, capacity proxy |
| `lsliwko/google-cluster-data-2019` | CC-BY-4.0 | 60,000 | Tier 4 | autoscaling proxy, model load/unload |
| `optimum-benchmark/llm-perf-leaderboard` | Apache-2.0 | 2,598 | Tier 4 | peak_vram_gb, throughput |

---

## 4. Candidate Datasets for Future Ingestion

### 4.1 Mooncake FAST25 Traces
- **Source:** https://github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces
- **License:** Apache-2.0
- **Files:** `conversation_trace.jsonl`, `synthetic_trace.jsonl`,
  `toolagent_trace.jsonl`, `mooncake_trace.jsonl`
- **Schema:** `{timestamp, input_length, output_length, hash_ids: [int,...]}`
- **Signals:** KV-block-hash list (prefix reuse), token counts, arrivals.
- **Verdict from scavenger audit:** `exact_dataset_found` (narrow).
  Closes: cross-dataset KV prefix reuse validation for cache forecaster.
  Does NOT add: latency, GPU, cold-start.
- **Bounded-ingestible:** Yes (small JSONL; commit normalized sample only).
- **Priority:** Medium. Next ingestion candidate.

### 4.2 Azure LMM 2025 (multimodal)
- **Source:** Azure (not yet public at audit time)
- **Signals:** images-per-request arrival dimension.
- **Verdict:** `partial_dataset_found` — additive arrival shape only.
- **Priority:** Low (limited signal beyond Azure 2024).

### 4.3 Vidur Profiling CSVs
- **Source:** Microsoft Research Vidur simulator
- **Signals:** Measured kernel latency A100/A40/H100.
- **Verdict:** `partial_dataset_found` — additive Tier-4 kernel-cost prior.
- **Priority:** Low-Medium (useful for heterogeneous placement scorer).

### 4.4 AcmeTrace (NSDI'24)
- **Source:** https://huggingface.co/datasets/Qinghao/AcmeTrace
- **License:** (check)
- **Signals:** GPU power (IPMI) + util (DCGM) + thermal-adjacent
  (training-class only).
- **Status:** Partially ingested.
- **Priority:** Medium if inference-class GPU power is needed.

---

## 5. Excluded / Unsafe Baselines (permanent exclusion list)

| trace | baseline | reason |
|---|---|---|
| Azure LLM 2024 | `utilization_aware` | timeout 12.10% > 10% gate → UNSAFE |
| Canonical energy | `robust_energy_standalone` | 143 deadline misses → UNSAFE |
| Canonical energy | `greedy_energy` | 119 deadline misses → UNSAFE |
| MIT Supercloud bounded | naive FIFO | queue p99 ~56h, starvation 47% → UNSAFE |
| Any trace | Oracle / clairvoyant | Analysis-only; NEVER headline comparator |

---

## 6. Benchmark Integrity Rules (from `docs/RESULTS.md`)

1. SLA-safe goodput/$ = `sla_compliant_goodput / (gpu_infra_cost + energy_cost + network_cost)`.
2. Timeout ≥ 50% hard-excludes queue's contribution.
3. FIFO is the sanity baseline; per-workload strongest realistic safe
   baseline is the headline comparator.
4. No oracle as headline. No future leakage. No benchmark-specific hacks.
5. Simulator results only — not production savings (§8 gate not met).
6. New datasets require: bounded ingest, schema audit, license check,
   `normalized_sample.jsonl` commit, no raw text committed.
