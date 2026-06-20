# Public Telemetry Realism Matrix

> **Audit / reporting document.** Field-by-field provenance for every public /
> ingested dataset, marking each telemetry field as **Real**, **Derived**,
> **Synthetic**, or **Missing**. Companion to
> `research/PUBLIC_TRACE_REALISM_AUDIT.md`. Audited 2026-06-20.
>
> **Legend:** `R` = Real (in the raw data) · `D` = Derived (computed from real
> fields, no leakage, must be labeled) · `S` = Synthetic (invented by the
> benchmark; must never drive a savings headline) · `—` = Missing · `N/A` = not
> applicable (e.g. market price series is not a request trace).
>
> **Binding rule:** a savings claim is only as real as its *least-real driver*.
> If a field marked `S` is the dominant lever of a result, the result is a
> synthetic-mechanism demo, not a real-world saving.

---

## Table A — per-request serving fields

| dataset | arrival_time | request_type (SLA class) | prompt_tokens | predicted_output_tokens | actual_output_tokens | deadline / SLO | queue_wait | TTFT | TPOT | KV_cache_pressure |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Azure LLM 2024** | R | D¹ | R | S² | R | — | — | — | — | — |
| **Azure LLM 2023** | R | D¹ | R | S² | R | — | — | — | — | — |
| **BurstGPT** | R | — | R | S² | R | — | — | — | — | S³ |
| **Alibaba GenAI 2026** | R | D | R | — | R | D | R⁴ | D⁴ | — | D⁵ |
| **Alibaba GPU v2023** | R | — | — | — | — | — | D⁶ | — | — | — |
| **Philly (fixture)** | R | — | — | — | — | — | D⁶ | — | — | — |
| **MIT Supercloud** | R | — | — | — | — | R⁷ | R⁶ | — | — | — |
| **CARA** `asdwb` | R | — | R | **R** | **R** | — | **R** | **R** | **R** | **R** |
| **cc-traces** `semianalysisai` | R | **R** | R | — | R | — | — | R⁸ | D | R⁹ |
| **SwissAI** `eth-easl` | R | — | R | — | R | — | — | R⁸ | — | D |
| **Canonical energy** | S | S | — | — | — | S | — | — | — | — |
| **CAISO/PJM/ERCOT** | R | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| **WattTime carbon** | R | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## Table B — infra / placement / energy fields

| dataset | GPU_type | model_id | region | energy_price | carbon_intensity | migration_cost | capacity | failures / preemptions | replica_count |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| **Azure LLM 2024** | S | S¹⁰ | — | — | — | — | S | D¹¹ | S |
| **Azure LLM 2023** | S | S¹⁰ | — | — | — | — | S | D¹¹ | S |
| **BurstGPT** | S¹² | R | — | — | — | — | S | D¹¹ | S |
| **Alibaba GenAI 2026** | D | R | — | — | — | — | R | D | D |
| **Alibaba GPU v2023** | R | — | — | — | — | — | R | R | R |
| **Philly (fixture)** | R | — | — | — | — | — | R | R | R |
| **MIT Supercloud** | R | — | — | — | — | — | — | R | — |
| **CARA** `asdwb` | R | D | — | — | — | — | R | R | D |
| **cc-traces** `semianalysisai` | — | R | — | — | — | — | — | — | — |
| **SwissAI** `eth-easl` | — | R | — | — | — | — | — | — | — |
| **Canonical energy** | S | — | S | **R** | **R** | S | S | — | S |
| **CAISO/PJM/ERCOT** | N/A | N/A | R | **R** | — | N/A | N/A | N/A | N/A |
| **WattTime carbon** | N/A | N/A | R | N/A | **R** | N/A | N/A | N/A | N/A |

### Footnotes
1. Azure ships `conv` + `code` as separate files → a *file-level* workload label
   can be mapped to a class (Derived); there is **no per-request SLA class**. The
   committed run-g sample is a single file → effectively Missing.
2. `predicted_output_tokens` does **not** exist in these traces; benchmarks
   synthesize it as `actual × lognormal(noise)` (run-g) or a forecaster.
3. BurstGPT KV pressure is a **model-level cache-affinity proxy**, not a real KV
   hit rate.
4. Alibaba GenAI exposes **aggregate** e2e latency / queue-wait p95/p99, not
   per-request; TTFT is derived from the e2e aggregate.
5. KV pressure proxied by GPU memory used (Derived).
6. Queue wait = `start − submit` (Derived) for the training/packing traces.
7. MIT Supercloud `timelimit` is a real per-job deadline (Real).
8. cc-traces / SwissAI TTFT is **end-to-end (includes provider/network)**, not
   pure GPU TTFT — real but not directly engine-comparable.
9. cc-traces KV signal = **block hashes** (prefix-reuse evidence), not a
   utilization gauge — Real but a different KV signal than CARA's.
10. Azure model id is a **constant** (`azure-llm`) injected by the loader →
    Synthetic.
11. Failure = `output_tokens == 0` convention (Derived).
12. BurstGPT GPU type = `model → GPU` map (ChatGPT→A100, GPT-4→H100), Synthetic.

---

## What the matrix shows at a glance

- **`predicted_output_tokens` is Real in exactly one dataset: CARA.** Every other
  output-length-aware result must synthesize it.
- **`request_type` (true SLA class) is Real in exactly one dataset: cc-traces.**
  On the rollup serving traces it is Missing → **SLA-aware scheduling collapses
  to FIFO** (proven empirically in `RUN_G_VALIDITY_AUDIT.md`).
- **Per-request `TTFT/TPOT/queue_wait/KV` are Real only in CARA** (and partially
  cc-traces/SwissAI). The rollup traces (Azure, BurstGPT) have them all Missing.
- **`energy_price` + `carbon_intensity` are Real** (CAISO/PJM/ERCOT/WattTime) —
  the strongest real column block in the repo.
- **`migration_cost` is Synthetic everywhere** — no public anchor.
- **The two halves never overlap:** the trace with real serving telemetry (CARA)
  has no energy/region; the data with real energy has no serving telemetry. No
  single dataset supports both energy-aware *and* per-request serving claims.

---

## Testability verdict by optimization

### Fairly testable today (real-signal driven)
- **Energy/cost-aware regional scheduling** — real prices (CAISO/PJM/ERCOT).
- **Carbon-aware scheduling** — real carbon (WattTime).
- **Autoscaling / replica provisioning** — real arrivals+tokens (Azure/BurstGPT/
  Alibaba GenAI). *(latency is modelled, not measured — directional.)*
- **Batch inference scheduling / GPU packing** — real jobs (Alibaba GPU, MIT
  Supercloud, Philly-fixture).

### Only partially testable (real demand, synthetic mechanism)
- **Heterogeneous GPU placement** — GPU type Real in training traces, but LLM
  TTFT-by-GPU is Synthetic (CARA-calibrated priors); CARA could make it Real.
- **Per-request LLM serving queue scheduling** — real tokens, but contention,
  servers and SLA are Synthetic (run-g's time-warp). CARA could make it Real.

### Not fairly testable yet (the dominant lever is Synthetic/Missing)
- **Output-length-aware SRTF** — `predicted_output_tokens`, contention, SLA, and
  physics are all Synthetic/Derived on Azure. *(Real only via CARA — unbuilt.)*
- **SLA-aware queue scheduling** — `request_type` Missing on rollup traces →
  baseline = FIFO. *(Real only via cc-traces — unbuilt.)*
- **Admission control / KV pressure** — KV is a realized-ρ **proxy** (Synthetic)
  in the tested path. *(Real only via CARA/cc-traces — unbuilt.)*
- **Migration-aware scheduling** — `migration_cost` Synthetic everywhere; no
  public trace anchors it.

**One acquisition flips three rows from red to green: promote CARA (verify
license first) for SRTF + SLA-aware + admission/KV; add cc-traces for real
`request_type`.**
