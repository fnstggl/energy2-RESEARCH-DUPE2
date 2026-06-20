# Public Trace Realism Audit

> **Audit / reporting document. Simulator / public-trace results are
> directional only — NOT production savings** (`docs/RESULTS.md` §8). This audit
> assesses whether Aurelius has enough realistic public trace data to evaluate
> its optimizations, which optimizations can be fairly tested today, and which
> cannot because required production-telemetry fields are missing.
>
> Companion docs: `research/PUBLIC_TELEMETRY_REALISM_MATRIX.md` (field matrix),
> `research/RUN_G_VALIDITY_AUDIT.md` (run-g + stronger-baseline replay),
> `research/BENCHMARK_REGISTRY.md` (source-of-truth registry).
>
> Audited: 2026-06-20 · Branch: `claude/intelligent-goldberg-zkj5pz`

---

## Executive answers

1. **Do we have enough public trace data to evaluate Aurelius realistically?**
   **Partially.** We have excellent *real* coverage of two axes — (a) LLM
   arrival + token demand (Azure 2024/2023, BurstGPT, Alibaba GenAI) and (b)
   energy/carbon prices (real CAISO/PJM/ERCOT/WattTime). We are missing the
   *per-request serving telemetry* (latency labels, SLA classes, GPU type,
   queue wait, KV pressure) needed to fairly evaluate the per-request serving
   optimizations the project is now pivoting toward. The one trace that *does*
   carry those fields (CARA) is ingested for research only and is not in the
   rollup.

2. **Which optimizations can be fairly tested?** Energy/cost-aware regional
   scheduling, carbon-aware scheduling, autoscaling/replica provisioning, batch
   inference scheduling, and (with caveats) heterogeneous GPU placement.

3. **Which cannot be fairly tested?** Output-length-aware SRTF, per-request SLA-
   aware queue scheduling, admission/KV-pressure control, and migration-aware
   scheduling — all require per-request latency, SLA-class, or KV fields that the
   rollup traces lack, forcing synthetic assumptions that drive the result.

4. **What did run-g do differently?** See `RUN_G_VALIDITY_AUDIT.md`. In one line:
   it exposed a **per-request queue-ordering** decision surface (a new
   discrete-event M/G/c simulator) that the aggregate provisioning replay never
   had, on a **synthetically time-warped** load, against a **FIFO** baseline.

5. **Is run-g's +323% real / comparable / north-star-relevant?** Real and
   reproducible; **not comparable** to the +26% rollup; not yet a north-star win.
   Verdict: **VALID BUT NOT COMPARABLE**. Full reasoning in
   `RUN_G_VALIDITY_AUDIT.md`.

---

## PHASE 1 — Public dataset inventory

Scores are **fidelity-for-Aurelius-evaluation, 1–5** (5 = real production data
directly usable for a headline; 1 = synthetic/fixture). "Raw present locally"
means committed in-repo; most large traces are fetched on-demand and only a
fixture/sample is committed.

### 1.1 Azure LLM Inference Trace 2024 — `azure_llm_2024_week`
- **Source / license:** github.com/Azure/AzurePublicDataset (DynamoLLM, HPCA 2025, arXiv 2408.00741) · CC-BY
- **Raw local?** No — only a **5,880-row sample** (`tests/fixtures/azure_llm_2024_sample.csv`, 208 KB). Full 44.1M-row week is **SAS-gated (HTTP 401)** at run time.
- **Workload / range:** LLM serving (conv + code) · 9 days · 12,960 ticks @ 60 s · 44,107,694 requests (full) / 5,880 (committed)
- **Fields present:** `TIMESTAMP`, `ContextTokens` (prompt), `GeneratedTokens` (output) — **3 columns only**.
- **Fields missing:** model id (static `azure-llm`), session/conversation id, cache/prefix key, TTFT/TPOT/latency, SLA class/deadline, GPU type, region, queue wait, KV pressure, failures (derived from `GeneratedTokens==0`).
- **Can test:** autoscaling/provisioning (arrival+token replay), utilization-target cost efficiency, output-length *distribution* realism.
- **Cannot test:** per-request SLA/latency, SRTF ordering (no real contention), GPU placement, cache/KV, migration.
- **Realism: 4/5** (real production, massive) — but **headline +26% needs the SAS-gated full week; the committed sample reproduces +0.00%** (see Phase 3).
- **Caveats:** token-demand replay, not a measured-latency replay. The basis of run-g's SRTF and the rollup's largest trace, yet carries none of the per-request fields either needs.

### 1.2 Azure LLM Inference Trace 2023 — `azure_llm_2023_conv`
- **Source / license:** github.com/Azure/AzurePublicDataset · CC-BY
- **Raw local?** No — 44-row fixture (`tests/fixtures/azure_llm_sample.csv`). Full 19,366 requests fetched from GitHub raw.
- **Workload / range:** LLM serving (conversational) · ~0.003 days · 19,366 requests
- **Fields:** same 3-column schema as 2024. Same missing set.
- **Can / cannot test:** same as 2024 but tiny duration.
- **Realism: 3/5** (real, but very short; superseded by 2024).

### 1.3 BurstGPT — `burstgpt_v1`
- **Source / license:** github.com/HPMLL/BurstGPT · huggingface.co/datasets/lzzmm/BurstGPT · **CC-BY-4.0**
- **Raw local?** Fixture committed (51–54 rows); **full 1,429,738-request CSV fetched at CC-BY and replayed in full** (the project's most robust public evidence).
- **Workload / range:** LLM serving (real ChatGPT + GPT-4 traffic) · ~34 min canonical window (full HF variant 59,999 rows) · 17,689 requests in canonical window
- **Fields present:** `Timestamp`, `Model` (ChatGPT/GPT-4), `Request tokens`, `Response tokens`, `Total tokens`, `Log Type`.
- **Fields missing:** session id (absent in `BurstGPT_1.csv`), TTFT/TPOT/latency, SLA class, GPU type, region, queue wait, KV pressure (only a **model-level** cache-affinity proxy).
- **Can test:** autoscaling/provisioning, model-level cache-affinity proxy, admission (best-effort share derived from "API log" fraction).
- **Cannot test:** per-request SLA/latency, real KV hit rate, SRTF ordering, GPU placement.
- **Realism: 4/5** (real traffic, full trace, clean license).

### 1.4 Alibaba GenAI 2026 (GenTD26) — `alibaba_genai_2026`
- **Source / license:** github.com/alibaba/AlibabaSystemTraces (cluster-trace-v2026-GenAI) · academic release
- **Raw local?** Processed summaries + fixture only; raw fetched on demand.
- **Workload / range:** stable-diffusion + LLM mixed serving · ~9 h · 26,392 requests
- **Fields present (richest serving trace):** request id, model id, input/output tokens, arrival, **`model_load_latency`**, **e2e latency p95/p99**, **queue wait p95/p99**, **queue depth**, **GPU duty-cycle**, GPU memory used.
- **Fields missing:** explicit SLA class (derived), explicit per-request failure, energy/carbon, region pricing; cross-layer joins are imperfect (`no_join` between app and metric layers).
- **Can test:** model-affinity / prewarm, autoscaling, cold-start residency, queue-aware sizing.
- **Cannot test (cleanly):** per-request SRTF (aggregated latency), energy/carbon, migration.
- **Realism: 4/5** — strongest serving-signal trace; drives the +89% model-affinity headline.

### 1.5 Alibaba Cluster Trace GPU v2023 — `alibaba_gpu_v2023`
- **Source / license:** github.com/alibaba/clusterdata · academic release
- **Raw local?** Processed summary + fixture; raw on demand.
- **Workload / range:** GPU packing / training · 149.3 days · 6,282 jobs
- **Fields:** job id, submit/start/end, gpu_count, gpu_type, status (success/killed/failed); queue wait = start−submit (derived).
- **Missing:** tokens, TTFT, model id, energy, region, KV.
- **Can test:** GPU packing, heterogeneous placement (gpu_type), price-aware routing.
- **Cannot test:** any serving/latency/SLA/output-length optimization.
- **Realism: 3/5** (real, long; training not serving).

### 1.6 Microsoft Philly — `philly_training`
- **Source / license:** github.com/msr-fiddle/philly-traces · academic release
- **Raw local?** **Fixture-scale only (33 jobs)**; full ~1 GB LFS trace not committed.
- **Workload / range:** training GPU scheduling · 0.007 days (fixture)
- **Fields:** job/submit/start/end, gpu_count, gpu_type, status; queue wait derived.
- **Realism: 2/5** (fixture-scale; cannot reflect full-trace patterns).

### 1.7 MIT Supercloud (bounded) — `mit_supercloud_bounded`
- **Source / license:** supercloud.mit.edu (S3 datacenter challenge) · research use
- **Raw local?** Bounded **real** sample (10,000 jobs ≈ 3 MB of ~1–2 TB); manifest committed.
- **Workload / range:** training GPU scheduling · 55.9 days · 10,000 jobs
- **Fields:** SLURM job schema (submit/start/end, gpu_count, gpu_type, status incl. FAILED/TIMEOUT/PREEMPTED); **real queue wait + failures** (good signal).
- **Missing:** per-node capacity (not published), tokens, latency, energy.
- **Can test:** GPU scheduling, anti-starvation (FIFO p99 ~56 h is UNSAFE), failure-aware scheduling.
- **Realism: 3/5** (bounded real; capacity unpublished).

### 1.8 CAISO / PJM / ERCOT energy + WattTime carbon (real market data)
- **Source / license:** CAISO OASIS, PJM DataMart, ERCOT, WattTime MOER · public
- **Raw local? YES — fully committed.** `data/{caiso,pjm,ercot}_*_{dam,rt}.csv` (~1.7 k rows each, hourly), `data/q12026_3region_{dam,rt}.csv`, `data/watttime_carbon_q12026.csv` (1,571 rows), plus `data/{summer2025,fall2025,combined_2025_2026}/`.
- **Fields:** `timestamp, region, price_per_mwh, currency, source, source_granularity, fetched_at`; carbon: `timestamp, region, gco2_per_kwh, source`.
- **Can test:** energy/cost-aware regional scheduling, carbon-aware scheduling — **directly, with real prices.**
- **Realism: 5/5** (real public market data). This is the strongest real signal in the repo.

### 1.9 Canonical energy backtest (frozen synthetic workload over real prices)
- **Source:** `aurelius/benchmarks/golden/canonical_energy_backtest.json`
- **Raw local? YES** (golden JSON). **1,000 jobs · 26 days.**
- **Real:** CAISO/PJM/ERCOT prices, WattTime carbon. **Synthetic:** the 1,000-job workload, GPU counts, deadlines, SLA classes, region eligibility.
- **Realism: workload 2/5, prices 5/5.** The +11% headline is honest *directional* energy alpha; the workload is not customer-derived.

### 1.10 CARA — `asdwb/cara_latency_prediction` (ingested, NOT in rollup) — **the gold per-request serving trace**
- **Source / license:** huggingface.co/datasets/asdwb/cara_latency_prediction · (license unverified — see caveat)
- **Raw local?** Ingested sample (`train_flat` 76,825 rows, ~84 MB processed under `data/external/hf/`).
- **Fields present (the only trace with these):** **`num_predicted_output_tokens`** + **`actual_output_tokens`**, **`actual_ttft`**, **`actual_tpot`**, **`actual_e2e_latency`**, **`kv_cache_utilization`**, `kv_evictions_per_s`, `kv_free_blocks`, `num_waiting`/`num_running`/`num_preempted`, `instance_type` (GPU), `request_id`.
- **Can test (uniquely):** output-length-aware SRTF **with real predicted+actual tokens**, per-request SLA/latency, admission/KV-pressure control, heterogeneous GPU placement (real TTFT by GPU).
- **Cannot test:** energy/carbon/region (none), multi-region migration.
- **Realism: 4/5 for serving telemetry** (lab measurement, single environment, 76 k rows; license unverified → **must verify before any headline**).
- **This is the most important under-used asset for the project's current SRTF/serving pivot.**

### 1.11 Other ingested HF datasets (research/training only, not in rollup)
| dataset | license | rows | strongest signal | testable optimization |
|---|---|---:|---|---|
| `semianalysisai/cc-traces-weka` | Apache-2.0 | 136,118 | KV **block hashes**, TTFT, `request_type`, `session_id` | KV/cache reuse, admission, real SLA-class (`request_type`) |
| `eth-easl/swissai-serving-trace` | research | 67,190 | `reuse_percentage`, latency, model id | cache-affinity, output-length |
| `sammshen/lmcache-agentic-traces` | MIT | 4,976 | cache reuse, routing proxy | KV/routing (small) |
| `Qinghao/AcmeTrace` | check | — | GPU power (IPMI) + util (DCGM), training | carbon/thermal (training-class) |
| `lsliwko/google-cluster-data-2019` | CC-BY-4.0 | 60,000 | scheduling/resource events | autoscaling proxy |
| `optimum-benchmark/llm-perf-leaderboard` | Apache-2.0 | 2,598 | peak_vram, throughput by GPU | heterogeneous GPU cost prior |

### 1.12 Candidate (not yet ingested)
- **Mooncake FAST25** (Apache-2.0): `{timestamp, input_length, output_length, hash_ids[]}` — KV-prefix reuse validation. Next ingestion candidate.
- **Vidur profiling CSVs** (MSR): measured kernel latency A100/A40/H100 — heterogeneous-placement cost prior.
- **Azure LMM 2025** (multimodal): additive arrival shape only.

---

## PHASE 2 — Production telemetry requirements per optimization class

For each Aurelius optimization class: **required** fields (cannot evaluate
honestly without them), **nice-to-have** fields, supporting datasets, and
whether **current benchmark coverage is valid**.

### 1. Energy/cost-aware regional scheduling
- **Required:** arrival/submit time, job duration or token demand, deadline/flexibility window, region set, **real energy price by region+time**.
- **Nice-to-have:** carbon intensity, migration cost, GPU type/power.
- **Supports:** CAISO/PJM/ERCOT (real prices) + canonical workload. **Partial:** Alibaba GPU, MIT Supercloud (real jobs, no prices). **No:** Azure/BurstGPT.
- **Coverage valid?** **YES (directional)** — prices real; workload synthetic and labeled as such. Strongest real-signal class.

### 2. SLA-aware scheduling
- **Required:** per-request/job **SLA class or deadline/SLO**, arrival, service demand, a latency/completion signal to score violations.
- **Nice-to-have:** priority/tenant, per-class latency budget.
- **Supports:** CARA (`request_type` via cc-traces; deadlines derivable), cc-traces (`request_type`), MIT Supercloud (timelimit). **Partial:** Alibaba GenAI (aggregate e2e). **No:** Azure 2024/2023, BurstGPT (no SLA class → SLA-aware collapses to FIFO).
- **Coverage valid?** **NO for the rollup serving traces** — Azure/BurstGPT have no SLA-class field, so any "SLA-aware" baseline on them is either FIFO-equivalent or synthetic. Valid only on CARA/cc-traces (not in rollup).

### 3. Per-request LLM serving queue scheduling
- **Required:** per-request arrival, **per-request service/latency** (or token-based service model), server/replica count, queue-wait.
- **Nice-to-have:** TTFT, TPOT, batch state.
- **Supports:** CARA (real per-request latency + queue state). **Partial:** Azure 2024 (tokens real, latency + contention synthetic — run-g's case). **No:** aggregate-only traces.
- **Coverage valid?** **Synthetic-dependent.** Run-g's queue and contention are modelled (time-warp 21.95×); fair *within-simulator* but the magnitude is a function of synthetic knobs. Valid as a mechanism demo, not a production number.

### 4. Output-length-aware SRTF scheduling
- **Required:** **predicted output tokens** (at decision time) **and** actual output tokens (for physics), under genuine queue contention with a latency/SLA signal.
- **Nice-to-have:** forecast-error distribution, preemption support.
- **Supports:** **CARA only** (`num_predicted_output_tokens` + `actual_output_tokens` + latency + queue state — all real). **Partial:** Azure 2024 (actual tokens real; *predicted* synthesized as actual×noise; contention synthesized). **No:** others.
- **Coverage valid?** **NO on Azure (run-g)** as a production claim — predicted tokens, contention, SLA, and physics are all synthetic/derived. **A CARA-based test would be the first fair evaluation** and is not yet built.

### 5. Admission control / KV pressure control
- **Required:** **KV cache utilization / pressure**, queue depth, per-request arrival, a back-pressure/SLA signal.
- **Nice-to-have:** eviction rate, free blocks, best-effort vs latency-critical label.
- **Supports:** CARA (`kv_cache_utilization`, `kv_evictions_per_s`, `kv_free_blocks`, `num_waiting`), cc-traces (KV block hashes). **Partial:** BurstGPT/Azure (KV proxied by realized rho — synthetic). **No:** training traces.
- **Coverage valid?** **NO in the rollup** — `WorkloadAdmissionGate` was tested with a **realized-rho KV proxy** (synthetic) and came out NEUTRAL; a real KV-pressure test (CARA) is unbuilt.

### 6. Heterogeneous GPU placement
- **Required:** **GPU type per option**, a per-GPU latency/throughput signal, job/request demand.
- **Nice-to-have:** real region+GPU price, TTFT-by-GPU.
- **Supports:** CARA (`instance_type` + real TTFT), optimum-benchmark (throughput by GPU), Vidur (candidate). **Partial:** Alibaba GPU / MIT Supercloud (gpu_type, no LLM latency). **No:** Azure/BurstGPT.
- **Coverage valid?** **PARTIAL/synthetic.** `gpu_routing_backtest` uses **synthetic** region→GPU mapping + TTFT priors *calibrated to CARA medians* on synthetic rows. Directional only; regressed real KPI (−7.3%).

### 7. Migration-aware scheduling
- **Required:** per-job location over time, **migration cost**, a reason to migrate (price/carbon/failure).
- **Supports:** none directly (migration cost is **always synthetic**, `$0.5/migration` default = 0 in economics). **Partial:** canonical energy (synthetic migration accounting).
- **Coverage valid?** **NO** — migration cost has no public-trace anchor; it is a documented synthetic constant. Claims about migration savings are not credible from current data.

### 8. Carbon-aware scheduling
- **Required:** **real carbon intensity by region+time**, flexible workload with deadlines, region set.
- **Supports:** WattTime MOER (real) + canonical workload. **Partial:** AcmeTrace (GPU power, training). **No:** serving traces.
- **Coverage valid?** **YES (directional)** — carbon data real; workload synthetic and labeled.

### 9. Autoscaling / replica provisioning
- **Required:** real arrival timeline + token demand, a serving physics/latency model, replica cost.
- **Nice-to-have:** real per-tick latency, cache signal.
- **Supports:** Azure 2024/2023, BurstGPT, Alibaba GenAI (real arrivals+tokens). **Partial:** —. **No:** training traces.
- **Coverage valid?** **YES (directional)** — this is the rollup's main valid class (+26% Azure week vs sla_aware, −21% GPU-hours). Caveat: serving physics is an aggregate Erlang-C model; latency is modelled, not measured.

### 10. Batch inference scheduling
- **Required:** job arrival, duration/token demand, deadline, capacity cap.
- **Supports:** canonical energy, Alibaba GPU, MIT Supercloud, Philly. **Partial:** Azure tokens (→ runtime via proxy, as in `srtf_contention_backtest`). 
- **Coverage valid?** **YES (directional)** for packing/energy; the SRTF-contention probe correctly shows the merged batch scheduler has **no queue-wait semantics** (inert).

---

## PHASE 3 — Benchmark audit

Entry points, datasets, decision surface, baselines, KPIs, and what is real vs
synthetic. The KPI in every case is `economics.py:
compute_sla_safe_goodput_per_infra_dollar = SLA_compliant_goodput /
(gpu_infra + energy + network cost)` — except run-g's SRTF modules, which use a
**constant** denominator (see note).

| # | benchmark / entry | dataset(s) | decision surface | baselines | optimized | reports SLA-safe goodput/$? | baseline = FIFO/SLA-aware? | uses public trace directly? | synthetic elements |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `scripts/run_burstgpt_backtest.py` → `traces/backtest.py` | BurstGPT (real, full 1.43M) | per-tick replica provisioning | fifo, **sla_aware**, queue_aware, cache_affinity | `constraint_aware` | yes (varying $) | **vs sla_aware** | yes (real arrivals+tokens) | time-warp to ρ, model→GPU map, flat $0.10/kWh, aggregate Erlang-C latency |
| 2 | `scripts/run_azure_llm_2024_backtest.py` | Azure 2024 (**sample** unless SAS) | per-tick provisioning | fifo, **sla_aware**, queue_aware, cache(omitted) | `constraint_aware` | yes (varying $) | **vs sla_aware** | sample only (+0.00%); full week SAS-gated (+26%) | same as #1 + scale sweep |
| 3 | `scripts/run_canonical_backtests.py` → `canonical_backtests.py` | **synthetic** 1,000 jobs + **real** prices | region + start-hour routing | fifo, **current_price_only**, greedy/robust(unsafe), sla_aware | `constraint_aware_with_energy_adapter` | yes (varying $) | prices real, workload synthetic | vs current_price_only (+11%) | synthetic jobs/SLA/region/migration cost |
| 4 | `gpu_routing_backtest.py` | synthetic jobs + real prices | GPU type + region affinity | scorer-off | scorer-on (`GpuPlacementScorer`) | yes | baseline scheduler | prices real, jobs+GPU map synthetic | **synthetic GPU types, TTFT priors** (CARA-calibrated on synthetic rows); regressed −7.3% |
| 5 | `srtf_backtest.py` [run-f] | synthetic energy jobs | batch sort key | no-prior | predicted-token sort | yes | baseline | prices real | **0% delta** (no queue contention) |
| 6 | `srtf_contention_backtest.py` [run-g] | **real Azure tokens** + real CAISO | batch sort under power cap | fifo | srtf_perfect/forecast | yes | vs fifo | tokens real | **−0.03…−0.05%** (scheduler has no queue-wait semantics → inert) |
| 7 | **`srtf_serving_backtest.py` [run-g] — the +323%** | **real Azure tokens** | **per-request queue ordering** | **fifo** | srtf_perfect/forecast | yes but **constant $** | **vs FIFO** | tokens+arrivals real | **time-warp 21.95×**, TTFT/TPOT constants, SLA=10s, c=4, predicted=actual×noise |
| 8 | `module_backtest.py` | BurstGPT + Azure sample | provisioning + admission + outlen + GPU | sla_aware, constraint_aware | ca_admission / ca_outlen / ca_all | yes (varying $) | vs constraint_aware | real arrivals+tokens | KV proxy = realized rho; all modules NEUTRAL/HURT |
| 9 | `run_baseline_public_backtest.py` | all of the above | mixed | — | — | yes | snapshot | mixed | aggregates #1–4 |

### Key Phase-3 findings
- **The +26% rollup headline (Azure 2024 week) ≠ the +323% run-g number.** #2 is
  per-tick provisioning vs `sla_aware` on an aggregate Erlang-C model with a
  **varying** cost denominator (−21% GPU-hours). #7 is per-request ordering vs
  **FIFO** on a new discrete-event M/G/c model with a **constant** denominator.
  Different baseline, code path, decision surface, and metric meaning. (Detail:
  `RUN_G_VALIDITY_AUDIT.md`.)
- **The committed Azure 2024 sample reproduces +0.00%, not +26%** (see
  `research/results/baseline_public_backtest_2026-06-20.md`): CA vs sla_aware is
  +0.00% at 1× and 50× on the 5,880-row sample. The +25.75% requires the
  **SAS-gated full 44.1M-row week**, which is inaccessible from a clean checkout —
  a reproducibility gap for the largest headline.
- **Only one public path constructs a real `JobScheduler`** (#3 canonical
  energy). The serving paths use an aggregate latency model, so they cannot test
  per-request ordering at all — exactly the gap run-g surfaced.
- **`sla_aware` is the correct headline baseline** for serving (per
  `docs/RESULTS.md` §3); FIFO is a sanity baseline. Run-g's FIFO baseline is
  therefore weaker than the project's own headline rule prescribes.
- **No benchmark currently tests SRTF/SLA-aware ordering on a trace with real
  SLA-class labels.** That trace (CARA / cc-traces) exists but is unused for it.

---

## PHASE 7 — Next dataset strategy

1. **Do we need more public datasets?** **Yes — one specific kind:** a
   per-request serving trace with **latency + SLA-class + predicted/actual
   output tokens + KV/queue state**. Everything the serving/SRTF pivot needs is
   absent from the rollup traces and present in CARA / cc-traces.

2. **Which to add next (priority order):**
   - **(a) Promote CARA (`asdwb/cara_latency_prediction`) into a first-class
     serving benchmark** — it already carries `num_predicted_output_tokens`,
     `actual_output_tokens`, `actual_ttft/tpot/e2e`, `kv_cache_utilization`,
     queue state, `instance_type`. **First verify its license** before any
     headline. This single move converts SRTF, SLA-aware queue, admission/KV, and
     heterogeneous-placement from "synthetic-dependent" to "fairly testable."
   - **(b) `semianalysisai/cc-traces-weka` (Apache-2.0)** — real `request_type`
     (true SLA classes so SLA-aware ≠ FIFO), KV block hashes, TTFT, session id.
   - **(c) Mooncake FAST25 (Apache-2.0)** — KV-prefix reuse validation.
   - **(d) Vidur profiling CSVs** — real per-GPU kernel latency for heterogeneous
     placement cost priors.

3. **Most damaging missing fields:** (1) per-request **SLA class / request_type**
   (without it SLA-aware = FIFO and the entire "vs SLA-aware" axis is untestable
   on rollup traces); (2) per-request **latency labels** (TTFT/TPOT/e2e); (3)
   **KV cache pressure**; (4) **migration cost** (no public anchor at all).

4. **Which benchmark is the north-star benchmark?** Keep **Azure 2024 week vs
   `sla_aware` (provisioning, −21% GPU-hours)** as the north-star *today* — it is
   the only large real trace measuring a real cost-denominator move vs the right
   baseline — but flag its reproducibility gap (SAS-gated). The **aspirational**
   north-star is a **CARA-based per-request SLA-safe goodput/$ benchmark** with a
   real SLA-aware baseline and an anti-starvation guard; that does not exist yet.
   Run-g's serving backtest is the **prototype** for it, not the north-star
   itself.

5–6. **Should we build a unified "production-like public telemetry corpus"?**
   **Yes — by joining real fields, not synthesizing them.** It should contain:
   real arrivals+tokens (Azure/BurstGPT), real per-request latency + SLA class +
   KV/queue (CARA/cc-traces), real GPU latency priors (Vidur/optimum-benchmark),
   real energy+carbon (CAISO/PJM/ERCOT/WattTime). Each row's provenance and
   real/derived/synthetic status must be carried as metadata.

7. **What must NEVER be synthesized (into a headline):** SLA-class labels,
   per-request latency/TTFT, KV-cache pressure, migration cost, energy/carbon
   prices, and **contention/load** (the time-warp). Synthesizing any of these and
   feeding it into a savings claim contaminates the claim.

8. **What can be derived safely (and labeled "derived"):** predicted output
   tokens from a forecaster fit on a no-leakage warmup prefix; queue-wait from a
   documented physics model; failure from `output_tokens==0`; best-effort share
   from log-type. These are acceptable *if labeled* and *not the sole driver of a
   headline*.

9. **What must be labeled synthetic:** service physics constants (TTFT/TPOT),
   server count, GPU $/hr, SLA budget, the time-warp multiplier, synthetic
   GPU-type maps, and the canonical 1,000-job workload.

---

## Bottom line

- **Real and headline-ready:** energy/carbon-aware scheduling, autoscaling
  provisioning (Azure/BurstGPT), GPU packing — these have real-signal coverage
  and valid (directional) benchmarks.
- **Synthetic-dependent (not yet credible as savings):** SRTF / output-length
  ordering, per-request SLA-aware queueing, admission/KV control, migration-aware
  scheduling — they need CARA/cc-traces fields the rollup traces lack.
- **The one move that unlocks the most:** verify CARA's license and promote it to
  a first-class serving benchmark with a real SLA-aware baseline.
