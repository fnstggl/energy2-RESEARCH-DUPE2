# Model Residency Decision Engine — Full-Trace Validation (Alibaba GenAI 2026)

> **Measurement-only. Directional simulator / backtest result — not production savings** (`docs/RESULTS.md` §8). The decision engine is recommendation-only in real/customer mode; here it runs in simulator mode and **no constant was tuned to force a win**. The engine never substitutes the requested model/adapter.


**Question.** Does the Model Residency Decision Engine improve the full-trace Alibaba GenAI 2026 KPI vs existing constraint_aware, or did constraint_aware already capture most affinity/prewarm value?


- **Trace:** `data/external/alibaba_genai/raw` (FULL trace)
- **Requests:** 26,392 · **distinct models:** 79
- **Cold-start calibration (s):** basemodel 22.7, lora 4.4

## 1. Headline answer

- **KPI improved over existing constraint_aware?** **NO.**
- **Engine vs strongest residency-blind baseline:** TIE within ±1% (marginally below) at every pool size.
- **Did constraint_aware already capture the affinity value?** **YES.**


The standalone residency decision engine does NOT improve full-trace SLA-safe goodput/$ over the strongest residency-blind baseline (it ties within ±1%, marginally below, at every pool size). Its measurable benefit is a large cold-start / residency-hit-rate reduction (a latency/safety diagnostic) achieved WITHOUT the SLA blow-up that naive affinity (affinity_only) causes by concentrating load. The existing tick-based ablation shows current constraint_aware (affinity + anticipatory sizing) already captures the affinity/prewarm value (9.84 vs 7.05 goodput/$ with vs without affinity). The routing-only engine reproduces the affinity half on a fixed pool and adds no incremental KPI.

## 2. Per-request residency routing — full-trace results

> One fixed simulated GPU pool shared by all routing policies (same cost denominator); `sla_aware_naive_prewarm` additionally pays for replicas held warm beyond pool capacity. goodput_unit = completed_requests.


### 8 GPUs (primary)

| policy | goodput/$ | model hit | adapter hit | cold starts | cold p50/p95/p99 (s) | route→res | prewarm | evict | warm-pool GPU-h | SLA viol | e2e p99 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fifo_round_robin | 1.9674 | 0.6002 | 0.9339 | 10579 | 22.718/22.718/27.096 | 0 | 0 | 10515 | 0.0 | 299 | 127.718 |
| sla_aware_least_queue | 1.9819 | 0.6009 | 0.9328 | 10559 | 22.718/22.718/27.096 | 0 | 0 | 10500 | 0.0 | 107 | 119.718 |
| sla_aware_naive_prewarm | 0.3377 | 1.0 | 1.0 | 0 | —/—/— | 0 | 0 | 0 | 21551.35 | 78 | 107.0 |
| affinity_only | 1.7842 | 0.9808 | 0.9716 | 576 | 22.718/27.096/27.096 | 0 | 0 | 476 | 0.0 | 2729 | 2393.0 |
| residency_engine | 1.9674 | 0.9695 | 0.9743 | 853 | 22.718/27.096/27.096 | 25462 | 853 | 772 | 0.0 | 299 | 128.0 |

### 16 GPUs

| policy | goodput/$ | model hit | adapter hit | cold starts | cold p50/p95/p99 (s) | route→res | prewarm | evict | warm-pool GPU-h | SLA viol | e2e p99 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fifo_round_robin | 0.9946 | 0.5656 | 0.9032 | 11510 | 22.718/22.718/27.096 | 0 | 0 | 11391 | 0.0 | 11 | 119.718 |
| sla_aware_least_queue | 0.995 | 0.5688 | 0.9011 | 11423 | 22.718/22.718/27.096 | 0 | 0 | 11309 | 0.0 | 0 | 117.718 |
| sla_aware_naive_prewarm | 0.995 | 1.0 | 1.0 | 0 | —/—/— | 0 | 0 | 0 | 0.0 | 0 | 106.0 |
| affinity_only | 0.9239 | 0.9928 | 0.9798 | 247 | 22.718/27.096/27.096 | 0 | 0 | 137 | 0.0 | 1886 | 1471.0 |
| residency_engine | 0.9895 | 0.9905 | 0.9819 | 305 | 22.718/27.096/27.096 | 26087 | 305 | 188 | 0.0 | 146 | 115.0 |

### 32 GPUs

| policy | goodput/$ | model hit | adapter hit | cold starts | cold p50/p95/p99 (s) | route→res | prewarm | evict | warm-pool GPU-h | SLA viol | e2e p99 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fifo_round_robin | 0.4975 | 0.5448 | 0.8658 | 12086 | 22.718/22.718/27.096 | 0 | 0 | 11871 | 0.0 | 1 | 118.718 |
| sla_aware_least_queue | 0.4975 | 0.5446 | 0.8633 | 12096 | 22.718/22.718/27.096 | 0 | 0 | 11879 | 0.0 | 0 | 117.718 |
| sla_aware_naive_prewarm | 0.4975 | 1.0 | 1.0 | 0 | —/—/— | 0 | 0 | 0 | 0.0 | 0 | 106.0 |
| affinity_only | 0.463 | 0.9968 | 0.9862 | 129 | 22.718/27.096/27.096 | 0 | 0 | 10 | 0.0 | 1831 | 1471.0 |
| residency_engine | 0.4956 | 0.9959 | 0.9855 | 150 | 22.718/27.096/27.096 | 26242 | 150 | 1 | 0.0 | 103 | 113.0 |

### Per-pool verdict (engine vs strongest residency-blind baseline)

| pool | engine goodput/$ | best blind baseline | baseline goodput/$ | margin % | result | engine hit / fifo hit | engine cold / fifo cold | engine SLA viol / baseline | affinity_only goodput/$ (SLA viol) |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 1.9674 | sla_aware_least_queue | 1.9819 | -0.732 | TIE | 0.9695 / 0.6002 | 853 / 10579 | 299 / 107 | 1.7842 (2729) |
| 16 | 0.9895 | sla_aware_least_queue | 0.995 | -0.553 | TIE | 0.9905 / 0.5656 | 305 / 11510 | 146 / 0 | 0.9239 (1886) |
| 32 | 0.4956 | fifo_round_robin | 0.4975 | -0.382 | TIE | 0.9959 / 0.5448 | 150 / 12086 | 103 / 1 | 0.463 (1831) |

## 3. Existing tick-based ablation — full-trace (the constraint_aware numbers, preserved/unchanged)

> Different harness (variable replica **sizing**, not a fixed pool); goodput/$ magnitudes are **not** directly comparable to §2.

| config (tick-based) | goodput/$ | SLA-compliant | e2e p99 (s) | mean cold-start (s) | replica GPU-hrs |
|---|---|---|---|---|---|
| fifo | 1.7676 | 26392 | 53.46 | 23.55 | 4977.0 |
| sla_aware | 5.1938 | 17794 | 1219.35 | 23.55 | 1142.0 |
| sla_aware_plus_affinity | 8.1825 | 20399 | 846.39 | 2.87 | 831.0 |
| fifo_plus_affinity | 3.1816 | 26391 | 35.91 | 2.87 | 2765.0 |
| constraint_aware_no_affinity | 7.0548 | 26392 | 66.43 | 23.55 | 1247.0 |
| constraint_aware | 9.8404 | 26392 | 53.39 | 2.87 | 894.0 |

- Attribution: **affinity/prewarm ≈ 62.1%** of the +89.46% constraint_aware-vs-sla_aware gain; anticipatory sizing ≈ 37.9%. constraint_aware **with** affinity = 9.84 vs **without** = 7.05 goodput/$ — the value the engine reproduces.

## 4. Requested policy comparison (mapped across both harnesses)

| requested policy | harness | goodput/$ | note |
|---|---|---|---|
| fifo | tick-based | 1.7676 | static-peak sizing |
| fifo | per-request | 1.9674 | round-robin, residency-blind |
| sla_aware | tick-based | 5.1938 | reactive sizing (headline) |
| sla_aware + naive prewarm | per-request | 0.3377 | all models warm; warm-pool cost |
| sla_aware + naive prewarm | tick-based | 8.1825 | affinity≡prewarm in that harness |
| affinity_only | per-request | 1.7842 | route-to-resident, no SLA guard → SLA blow-up |
| affinity_only | tick-based | 3.1816 | closest analog (static sizing + affinity) |
| constraint_aware current | tick-based | 9.8404 | affinity + anticipatory sizing |
| constraint_aware without affinity | tick-based | 7.0548 | sizing only |
| constraint_aware + residency_decision_engine | per-request | 1.9674 | the engine (routing only; ties the blind baseline, no incremental KPI) |

## 5. Why the engine adds no incremental KPI (honest analysis)

- current constraint_aware already captures most affinity value — the ablation attributes ~62% of its +89.5% gain to affinity/prewarm, which the engine reproduces as per-request routing rather than unlocking new value;
- the engine is routing-ONLY — it performs no anticipatory replica SIZING (the other ~38% of constraint_aware's value), so on a fixed GPU pool it cannot match constraint_aware's sizing-driven gains;
- the harness uses a FIXED pool — limited routing degrees of freedom; the engine cannot add/remove replicas, only place requests;
- on this trace the SLA budget is loose relative to a single cold start (e2e p99 ≈ 106 s; SLA ≈ 30 + 2×service), so cold-start avoidance is a latency/SAFETY lever, not a goodput/$ ALPHA lever — visible as affinity_only's high hit-rate yet WORSE goodput/$ via SLA blow-up;
- methodology mismatch — the per-request fixed-pool harness and the tick-based variable-sizing ablation use different cost models, so their goodput/$ magnitudes are NOT directly comparable;
- the trace lacks a per-request request→GPU join (application↔infra is no_join), so the routing simulation is necessarily synthetic (placement is modelled, not replayed from real routing).

### Alpha vs safety

- **residency_engine_vs_residency_blind_baseline:** TIE on goodput/$ (no alpha); the cold-start / hit-rate gain is a latency/safety diagnostic, not economic alpha on this trace.
- **residency_engine_vs_affinity_only:** WIN — same residency hit-rate, far fewer SLA violations; the engine does affinity SAFELY (SLA/queue-aware), avoiding the naive concentration that collapses affinity_only's goodput/$.
- **naive_prewarm:** catastrophically expensive (warm-pool GPU-hours for every distinct model held warm) — lowest goodput/$.

## 6. What remains missing (before a real residency KPI claim)

- A per-request request→GPU join in the trace (it is `no_join`), so the routing replay is synthetic, not a real-router replay.
- Anticipatory replica **sizing** inside the engine (it is routing-only); the tick-based constraint_aware shows sizing carries ~38% of the value.
- A regime where cold-start avoidance is an **economic** lever (tighter SLA relative to load time), to separate alpha from the safety effect.
- Live telemetry + the `docs/RESULTS.md` §8 production-claim gate (unmet).


> No constant was tuned to force a win; the conclusion holds across all evaluated pool sizes.

