# MCS Public Dataset Inventory — traces for next-tick demand forecasting

> Scope: which **public** traces can support a *deployable* forecasted MCS, i.e.
> let us forecast next-tick demand from past data and evaluate SLA on a held-out
> replay. Repo searched first; external candidates listed where the repo lacks
> committed data. **No missing production telemetry is synthesized.**
>
> "Can forecast arrivals?" = has per-request arrival timestamps (or inter-arrival
> deltas) over enough duration to build per-tick arrival series with a
> train→test ordering. "Can forecast token demand?" = has per-request output
> (and ideally prompt) token counts. "Can evaluate SLA?" = arrival + token (→
> service time) is enough to replay a queue and measure response-time SLA.
>
> Directional simulator evidence only — NOT production savings.

---

## 0. Bottom line

Two committed traces fully support forecasted MCS today, and **both are used in
this PR's benchmark**:

| Trace | On disk | Events | Forecast arrivals | Forecast tokens | SLA eval | Realism |
|---|---|---:|:---:|:---:|:---:|:---:|
| **Azure LLM 2024** (fixture) | yes | 5,880 (of 44.1 M) | ✅ | ✅ | ✅ | 5 |
| **BurstGPT HF** | yes | 59,999 | ✅ | ✅ | ✅ | 4 |

A third committed trace (**SemiAnalysis cc-traces-weka**, agentic Claude-Code)
is *forecastable* (arrival deltas + tokens) but workload-narrow; a fourth
(**Google cluster 2019**) has timestamps but **no tokens** (generic autoscaling
only). Everything else committed is either non-serving, summary-only (raw not
committed), or has no arrival timestamps. The ideal-schema **Mooncake FAST25**
trace is external (not yet ingested).

This is **sufficient** for Phases 3–6: train/test forecasting + SLA replay on two
real LLM-serving traces with complementary character (Azure = smooth/diurnal,
BurstGPT = bursty). Result classification is therefore *FORECASTED MCS
IMPLEMENTED*, not *INCONCLUSIVE*.

---

## 1. Committed & forecast-ready (used in this PR)

### 1.1 Azure LLM Inference Dataset 2024
- **Source:** Microsoft Azure (DynamoLLM, HPCA 2025) — `github.com/Azure/AzurePublicDataset`
- **License:** Azure public dataset terms (research use, attribution).
- **Raw available?** Full 44.1 M-req / 9-day trace **not committed** (raw dir is
  `.gitkeep`). A **5,880-row sample** is committed at
  `tests/fixtures/azure_llm_2024_sample.csv` (~1 req/min downsample spanning
  2024-05-10 00:00 → 2024-05-11 02:00, ≈ 26 h).
- **Full or sample?** Sample (the leaderboard's "Azure 5,880" *is* this fixture).
- **Events / duration:** 5,880 req / ~26 h real → 72 ticks of 60 s after warp.
- **Fields available:** `TIMESTAMP`, `ContextTokens` (prompt), `GeneratedTokens`
  (output). Token dist: min 6, p50 90, p99 477, max 1,346.
- **Fields missing:** per-request TTFT/TPOT/latency, model identity, session/KV
  signal. Failure proxy = `GeneratedTokens==0`.
- **Can forecast arrivals?** ✅ real diurnal arrival shape (preserved up to the
  linear time-warp). **Token demand?** ✅ prompt+output. **SLA eval?** ✅
- **Realism: 5/5** — real production Azure LLM traffic.
- **Caveat:** the committed sample is a downsample; full-trace validation needs
  the uncommitted 44 M-req archive (bounded-ingest required).

### 1.2 BurstGPT (HuggingFace `lzzmm/BurstGPT`)
- **Source:** HPMLL / lzzmm — `huggingface.co/datasets/lzzmm/BurstGPT`
- **License:** CC-BY-4.0.
- **Raw available?** ✅ committed normalized sample
  `data/external/hf/lzzmm__BurstGPT/burstgpt_1_full/processed/normalized_sample.jsonl`.
- **Full or sample?** 59,999-row normalized sample (this PR uses the first 5,880
  for parity with the leaderboard; the full 60 K is available for longer splits).
- **Events / duration:** 59,999 req; arrivals `request_arrival_ts_s` span 5 →
  825,259 s (≈ 9.55 days). First 5,880 → 154 ticks of 60 s after warp.
- **Fields available:** `request_arrival_ts_s`, `input_tokens`, `output_tokens`,
  `total_tokens`, `model_id` (ChatGPT/GPT-4 class), `log_type`.
- **Fields missing:** latency labels, KV/session signal, GPU identity.
- **Can forecast arrivals?** ✅ (genuinely bursty — arrival rel-MAE ~34 % for a
  causal EWMA). **Token demand?** ✅ **SLA eval?** ✅
- **Realism: 4/5** — real GPT-class traffic; model field anonymized/coarse.

---

## 2. Committed & forecastable but workload-narrow

### 2.1 SemiAnalysis cc-traces-weka (`semianalysisai/cc-traces-weka-no-subagents`)
- **License:** Apache-2.0. **On disk:** ✅ `traces_3000mib` (and `traces_head`)
  normalized samples (~136 K rows per registry).
- **Fields:** `request_arrival_delta_s` (inter-arrival), `api_time_s`,
  `input_tokens`, `output_tokens`, `model_id` (claude-opus-class),
  `request_type`, `think_time_s`, KV `block_hashes`.
- **Can forecast arrivals?** ⚠️ yes via cumulative `request_arrival_delta_s`.
  **Token demand?** ✅ **SLA eval?** ✅ (has `api_time_s` ground truth too).
- **Realism: 4/5** for *agentic* serving — real Claude-Code production — but
  **narrow** (agentic tool-loop bursts, heavy prompt reuse). Not representative
  of general chat serving; a good *secondary* generalization trace, not a primary.
- **Status:** candidate for a follow-up forecasting run; not in this PR's two-trace
  headline to keep the comparison to broadly-representative serving workloads.

---

## 3. Committed but partial (cannot do full token-demand forecasting)

| Dataset | On disk | Why partial for forecasted MCS | arrivals | tokens | SLA | Realism |
|---|---|---|:---:|:---:|:---:|:---:|
| Google cluster 2019 (`lsliwko/...`) | ✅ 60 K | cluster instance events; **no tokens** → generic autoscaling demand only | ✅ (`event_time_us`) | ❌ | ❌ | 3 |
| eth-easl swissai-serving-trace | ✅ (bucket-reuse shards) | KV-reuse %, latency, model id; clean per-request **arrival series not exposed** in committed shards | ⚠️ | ✅ | ⚠️ | 3 |
| asdwb cara_latency_prediction | ✅ 76 K | TTFT/E2E/queue/GPU **labels** for latency regression; no arrival-rate series | ❌ | ⚠️ | ✅(labels) | 2 |
| optimum-benchmark llm-perf-leaderboard | ✅ | throughput/VRAM micro-bench; no arrivals | ❌ | ⚠️ | ❌ | 2 |
| ejhusom llm-inference-energy | ✅ | energy per inference; no arrival series | ❌ | ⚠️ | ❌ | 2 |
| AcmeTrace (`Qinghao/...`) | ✅ (heads) | GPU power/util for **training** jobs; not inference arrivals | ⚠️ | ❌ | ❌ | 3 |

---

## 4. Listed in registry but raw NOT committed (summary-only on disk)

These appear in `research/BENCHMARK_REGISTRY.md` but their raw/processed event
data is **not** in the repo (only `*_summary.json`), so they cannot drive a
forecasting run here without a bounded re-ingest:

| Dataset | Registry size | Forecast-relevant note |
|---|---|---|
| Azure LLM 2023 (conv) | 19,366 req / 0.003 d | tiny duration; weak for train/test split |
| Alibaba GenAI 2026 (GenTD26) | 26,392 req | rich (model-load latency, queue size) — **good candidate** if ingested |
| Alibaba GPU v2023 | 6,282 jobs / 149 d | GPU packing, not LLM serving |
| Microsoft Philly | 33 jobs (fixture) | training scheduler, fixture-scale |
| MIT Supercloud (bounded) | 10,000 jobs / 56 d | training scheduler, not serving |
| LMSYS chatbot arena | — | raw dir empty; arena votes, **no arrival timestamps** |
| ShareGPT (aiperf) | — | raw dir empty; ShareGPT has **no native timestamps** |

---

## 5. External candidates (not in repo) — for future ingestion

| Dataset | Schema | Forecast fit | License | Priority |
|---|---|---|---|---|
| **Mooncake FAST25** (`kvcache-ai/Mooncake`) | `{timestamp, input_length, output_length, hash_ids[]}` | **Ideal** — timestamps + both token counts + KV reuse | Apache-2.0 | **High** — cleanest 3rd forecasting trace |
| Azure LMM 2025 (multimodal) | images-per-req arrival dim | additive arrival shape | Azure terms | Low (not public at audit) |
| BurstGPT_2 / newer splits | as BurstGPT | longer duration for train/test | CC-BY-4.0 | Medium |

Mooncake is the strongest external add: it would give a third real LLM-serving
trace with explicit prompt+output lengths and arrival timestamps, enabling
cross-dataset forecast-MCS generalization. It is *bounded-ingestible* (small
JSONL) per the registry.

---

## 6. Verdict for Phase 3–6

- **Arrivals forecastable:** Azure 2024, BurstGPT HF, cc-traces-weka, Google-2019.
- **Token demand forecastable:** Azure 2024, BurstGPT HF, cc-traces-weka.
- **Full SLA replay (arrivals + tokens + service):** Azure 2024, BurstGPT HF,
  cc-traces-weka.
- **Chosen for the headline benchmark:** Azure LLM 2024 + BurstGPT HF — two real,
  broadly-representative LLM-serving traces with complementary burstiness, both
  already wired into the simulator and the published MCS leaderboard (so the
  oracle-vs-deployable comparison is exactly apples-to-apples with prior runs).

Public data is **sufficient** to implement and fairly evaluate a deployable
forecasted MCS. → not INCONCLUSIVE.
