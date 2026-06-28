# Open Simulator Capability Matrix (Phase 1)

Cells use: **FULL** · **PARTIAL** · **DERIVABLE** (computable from what it has, not built) · **ABSENT** ·
**CODE** (unclear in docs but code suggests support) · **PROP** (`REQUIRES_PROPRIETARY_DATA`). `UNKNOWN`
is not used. Classifications come from the Phase-0 source inspection.

Columns are grouped A–G. Rows: Aurelius (current) + the strongest open simulators. Mock/runtime layers
(llm-d-inference-sim, LMCache, KVServe) are summarized in notes, not full rows, because they are not
fleet simulators. "InferSim"=Alibaba InferSim; "LSSim"=LLMServingSim 2.0; "SplitwiseS"=SplitwiseSim;
"vLLM"=vLLM/PagedAttention reference.

---

## A. Workload / trace support

| capability | Aurelius | LSSim | BLIS | SplitwiseS | InferSim | Vidur | vLLM | Mooncake |
|--|--|--|--|--|--|--|--|--|
| arrival replay | FULL | FULL | FULL | FULL | PARTIAL | FULL | FULL | FULL(trace) |
| prompt-token replay | FULL | FULL | FULL | FULL | PARTIAL | FULL | FULL | FULL |
| output-token replay | FULL | FULL | FULL | FULL | PARTIAL | FULL | FULL | FULL |
| request priority / SLA | PARTIAL | PARTIAL | FULL | PARTIAL | ABSENT | PARTIAL | PARTIAL | FULL |
| multi-model replay | PARTIAL | FULL | FULL | FULL | PARTIAL | PARTIAL | FULL | ABSENT |
| tenant / org isolation | ABSENT | ABSENT | PARTIAL | ABSENT | ABSENT | ABSENT | PARTIAL | ABSENT |
| Azure-trace compat | FULL | FULL | DERIVABLE | FULL | DERIVABLE | FULL | DERIVABLE | DERIVABLE |
| Mooncake prefix/hash compat | FULL | FULL(tok_ids) | FULL | ABSENT | ABSENT | ABSENT | FULL | FULL |
| Alibaba topology compat | FULL | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT |

## B. Serving phases

| capability | Aurelius | LSSim | BLIS | SplitwiseS | InferSim | Vidur | vLLM |
|--|--|--|--|--|--|--|--|
| prefill phase | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| decode phase | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| TTFT | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| TPOT / ITL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| completion latency | FULL | FULL | FULL | FULL | DERIVABLE | FULL | FULL |
| prefill/decode disaggregation | PARTIAL | FULL | FULL | FULL | PARTIAL | PARTIAL | PARTIAL |
| KV handoff (transfer) | ABSENT | FULL | FULL | FULL | PARTIAL | PARTIAL | PARTIAL |
| chunked prefill | ABSENT | FULL | FULL | PARTIAL | PARTIAL | FULL | FULL |
| iteration-level scheduling | PARTIAL | FULL | FULL | FULL | ABSENT | FULL | FULL |
| active-decode-seq modeling | PARTIAL | FULL | FULL | FULL | PARTIAL | FULL | FULL |

## C. KV / cache

| capability | Aurelius | LSSim | BLIS | SplitwiseS | InferSim | Vidur | vLLM | LMCache |
|--|--|--|--|--|--|--|--|--|
| KV block model | FULL | FULL | FULL | PARTIAL | DERIVABLE | FULL | FULL | FULL |
| prefix hashes | FULL | FULL | FULL | ABSENT | ABSENT | ABSENT | FULL | FULL |
| partial prefix reuse | PARTIAL | FULL | FULL | ABSENT | ABSENT | ABSENT | FULL | FULL |
| exact prefix reuse | FULL | FULL | FULL | ABSENT | ABSENT | ABSENT | FULL | FULL |
| block-granular caching | FULL | FULL | FULL | ABSENT | DERIVABLE | FULL | FULL | FULL |
| GPU HBM cache | FULL | FULL | FULL | PARTIAL | FULL | FULL | FULL | FULL |
| CPU DRAM cache | ABSENT | FULL | FULL | ABSENT | ABSENT | ABSENT | PARTIAL | FULL |
| SSD/NVMe cache | ABSENT | PARTIAL(CXL) | FULL | ABSENT | ABSENT | ABSENT | ABSENT | FULL |
| remote/disaggregated KV | ABSENT | FULL | FULL | FULL | ABSENT | ABSENT | PARTIAL | FULL |
| eviction policy | FULL(LRU) | FULL(radix) | FULL | ABSENT | ABSENT | PARTIAL | FULL(LRU) | FULL(LRU) |
| cache lookup overhead | FULL | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT | PARTIAL | FULL |
| cache transfer overhead | ABSENT | FULL | FULL | FULL | PARTIAL | PARTIAL | PARTIAL | FULL |
| cache pollution / thrash | PARTIAL | FULL | FULL | ABSENT | ABSENT | ABSENT | PARTIAL | FULL |
| tenant-safe cache sharing | ABSENT | ABSENT | PARTIAL | ABSENT | ABSENT | ABSENT | PARTIAL | PARTIAL |
| real per-replica residency | PROP | PROP | PROP | PROP | PROP | PROP | PROP | PROP |
| real measured hit rate | PROP | PROP | PROP | PROP | PROP | PROP | PROP | PROP |

## D. Batching / scheduling

| capability | Aurelius | LSSim | BLIS | SplitwiseS | InferSim | Vidur | vLLM |
|--|--|--|--|--|--|--|--|
| continuous batching | PARTIAL | FULL | FULL | FULL | ABSENT | FULL | FULL |
| static batching | DERIVABLE | FULL | FULL | FULL | FULL | FULL | FULL |
| iteration-level scheduling | PARTIAL | FULL | FULL | FULL | ABSENT | FULL | FULL |
| admission control | FULL | FULL | FULL | PARTIAL | ABSENT | PARTIAL | FULL |
| routing | FULL | FULL | FULL | FULL | ABSENT | PARTIAL | PARTIAL |
| shortest-queue | FULL | FULL | FULL | FULL | ABSENT | ABSENT | ABSENT |
| locality routing | FULL | PARTIAL | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT |
| cache-aware routing | FULL | FULL | FULL | FULL | ABSENT | ABSENT | PARTIAL |
| model-affinity routing | FULL | PARTIAL | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT |
| prefill/decode allocation | PARTIAL | FULL | FULL | FULL | PARTIAL | PARTIAL | PARTIAL |
| priority scheduling | PARTIAL | FULL | FULL | PARTIAL | ABSENT | PARTIAL | FULL |
| queueing model | FULL(DES) | FULL(DES) | FULL(DES) | FULL(DES) | PARTIAL | FULL(DES) | FULL |

## E. Hardware / roofline

| capability | Aurelius (pre-PR) | Aurelius (this PR) | LSSim | BLIS | SplitwiseS | InferSim | Vidur | llm-analysis |
|--|--|--|--|--|--|--|--|--|
| GPU type modeling | PARTIAL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| FLOPs | ABSENT | **FULL** | DERIVABLE | FULL | DERIVABLE | FULL | DERIVABLE | FULL |
| memory bandwidth | ABSENT | **FULL** | DERIVABLE | FULL | DERIVABLE | FULL | DERIVABLE | FULL |
| HBM capacity | FULL | FULL | FULL | FULL | PARTIAL | FULL | FULL | FULL |
| arithmetic intensity | ABSENT | **FULL** | ABSENT | FULL | ABSENT | FULL | ABSENT | FULL |
| roofline ridge point | ABSENT | **FULL** | ABSENT | FULL | ABSENT | FULL | ABSENT | FULL |
| compute-bound class | ABSENT | **FULL** | ABSENT | FULL | ABSENT | FULL | DERIVABLE | FULL |
| memory-bound class | PARTIAL | **FULL** | ABSENT | FULL | ABSENT | FULL | FULL | FULL |
| kernel profiling | ABSENT | ABSENT | FULL | PARTIAL | FULL | PARTIAL(MFU) | FULL | ABSENT |
| heterogeneous accelerators | PARTIAL | FULL | FULL | FULL | FULL | FULL | FULL | FULL |
| tensor parallelism | ABSENT | DERIVABLE | FULL | FULL | FULL | FULL | FULL | FULL |
| pipeline parallelism | ABSENT | ABSENT | FULL | FULL | PARTIAL | FULL | FULL | FULL |
| MoE routing | ABSENT | DERIVABLE | FULL | FULL | ABSENT | FULL | PARTIAL | FULL |
| network topology | PARTIAL | PARTIAL | FULL(ASTRA) | FULL | PARTIAL | PARTIAL | PARTIAL | PARTIAL |

> "Aurelius (this PR)" reflects `aurelius/environment/roofline_external.py` (ported InferSim/llm-analysis/
> LLM-Viewer roofline), validated in `tests/test_roofline_external.py`. It is a **reference/validation**
> model today, not yet wired into the live service path (see Phase 5).

## F. Advanced controls

| capability | Aurelius | LSSim | BLIS | SplitwiseS | InferSim | Vidur | vLLM |
|--|--|--|--|--|--|--|--|
| speculative decoding | ABSENT | PARTIAL | PARTIAL | ABSENT | ABSENT | PARTIAL | FULL |
| precision / quantization | PARTIAL | FULL | FULL | ABSENT | FULL(fp8) | FULL | FULL |
| GPU clock / DVFS | ABSENT | ABSENT | ABSENT | PARTIAL | ABSENT | ABSENT | ABSENT |
| power cap | ABSENT | PARTIAL | PARTIAL | FULL | ABSENT | ABSENT | ABSENT |
| co-location | ABSENT | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT |
| prewarm | FULL | ABSENT | PARTIAL | PARTIAL | ABSENT | ABSENT | PARTIAL |
| migration | FULL | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT |
| placement | FULL | PARTIAL | PARTIAL | PARTIAL | ABSENT | ABSENT | ABSENT |
| scale-up/down | PARTIAL | PARTIAL | FULL | PARTIAL | ABSENT | ABSENT | ABSENT |
| warm pool | FULL | ABSENT | PARTIAL | PARTIAL | ABSENT | ABSENT | PARTIAL |
| autoscaling lifecycle | PARTIAL | PARTIAL | FULL | PARTIAL | ABSENT | ABSENT | ABSENT |

## G. Economics / objective

| capability | Aurelius | LSSim | BLIS | SplitwiseS | InferSim | Vidur | vLLM |
|--|--|--|--|--|--|--|--|
| GPU-seconds | FULL | DERIVABLE | DERIVABLE | DERIVABLE | DERIVABLE | DERIVABLE | ABSENT |
| GPU-hours | FULL | DERIVABLE | DERIVABLE | DERIVABLE | DERIVABLE | DERIVABLE | ABSENT |
| energy | FULL | FULL(kJ) | PARTIAL | PARTIAL | ABSENT | PARTIAL | ABSENT |
| power | FULL | FULL | PARTIAL | FULL | ABSENT | PARTIAL | ABSENT |
| cost ($) | FULL | ABSENT | PARTIAL | PARTIAL(isocost) | ABSENT | ABSENT | ABSENT |
| SLA-safe goodput | FULL | DERIVABLE | FULL | DERIVABLE | ABSENT | DERIVABLE | ABSENT |
| goodput / $ | FULL | ABSENT | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT |
| market price | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT |
| carbon | DERIVABLE | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT |
| multi-region economics | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT | ABSENT |
| MPC / counterfactual planning | FULL | ABSENT | PARTIAL | ABSENT | ABSENT | ABSENT | ABSENT |

> True internal operator $/GPU-hr, real energy draw, real carbon intensity per request → **PROP** for every
> system; Aurelius models these from public ISO + spec priors and labels the tier (see `cost_model.py`).

---

## Reading of the matrix

- **Group A/F/G (workload fidelity, fleet lifecycle, economics + MPC):** Aurelius is **ahead of every open
  simulator.** None of them models prewarm/migration/placement on a persistent state, operator $ economics,
  goodput/$, or counterfactual MPC. This is Aurelius' moat.
- **Group B/C/D (serving phases, KV transfer tiers, true continuous batching):** **LLMServingSim 2.0 and
  BLIS are ahead of Aurelius** (radix prefix cache + tier-aware eviction cost + iteration-level scheduling +
  KV transfer). These are the import/port targets.
- **Group E (roofline):** Aurelius **was ABSENT, is now FULL** as a *reference model* (this PR, ported from
  InferSim/llm-analysis/LLM-Viewer). Wiring it into the live service path is the recommended next build.
- **PROP cells are identical across all systems** — no public artifact closes them. They define the honest
  ceiling of any public-data world model (see Phase 6).
