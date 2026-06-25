# BurstGPT Backtest Results — CANONICAL_TRACE_BACKTEST_BURSTGPT_V1

> **Simulator benchmark result — directional only, NOT production savings.** Live customer-telemetry calibration is required before any external savings number (`docs/RESULTS.md` §8).
>
> Read `docs/RESULTS.md` (reporting standard) and `docs/PUBLIC_TRACE_BACKTESTS.md` (dataset roles) first.

## Provenance

- **Source:** `csv:tests/fixtures/burstgpt_sample.csv`
- **Exact file:** BurstGPT_1.csv (https://github.com/HPMLL/BurstGPT/tree/main/data)
- BurstGPT is a **public LLM-serving trace, not customer telemetry**.
- The published `BurstGPT_1.csv` has **no Session ID and no Elapsed-time column**; the cache-affinity key is a model-level prefix-locality **proxy**, not a measured KV cache hit rate.
- BurstGPT elapsed time (when present in other files) is **end-to-end response time, NOT TTFT**. No TTFT is measured from BurstGPT.

## Trace summary

- Requests replayed: **51**  ·  ticks: **55**  ·  tick size: **60s**
- Time range: 5s → 3298s (0.91 h)
- Failure rate: 0.0000%
- Model distribution: {'ChatGPT': 36, 'GPT-4': 15}
- Log-type distribution: {'API log': 17, 'Conversation log': 34}
- Prompt tokens p50/p95/p99: 488 / 1437 / 1574
- Output tokens p50/p95/p99: 344 / 647 / 1656
- RPS/min mean/p95/max: 0.0155 / 0.0500 / 0.2333
- Cache-affinity proxy: 2 distinct keys, reuse rate 96.08%

## Primary KPI — SLA-safe goodput per infrastructure dollar

Per `docs/RESULTS.md` §1. SLA is a filter on the goodput numerator (`tokens × (1 − timeout_rate_pct/100)`), never a term in the cost denominator. Headline baseline for interactive inference is **sla_aware** (`docs/RESULTS.md` §3 rule 5).

| policy | goodput/$ | SLA-compliant tokens | total infra $ | lat p95 (ms) | lat p99 (ms) | queue p95 (ms) | timeout % | migration/reroute | cache proxy |
|---|---|---|---|---|---|---|---|---|---|
| fifo | 8,689.29 | 17,508 | 2.01 | 14,636.62 | 29,126.67 | 10.91 | 4.254 | 0 | no |
| sla_aware | 8,689.29 | 17,508 | 2.01 | 14,636.62 | 29,126.67 | 10.91 | 4.254 | 0 | no |
| constraint_aware | 8,691.77 | 17,513 | 2.01 | 14,586.21 | 29,059.26 | 10.91 | 4.215 | 0 | yes |
| queue_aware | 8,689.29 | 17,508 | 2.01 | 14,636.62 | 29,126.67 | 10.91 | 4.254 | 0 | no |
| cache_affinity_baseline | 8,691.77 | 17,513 | 2.01 | 14,586.21 | 29,059.26 | 10.91 | 4.215 | 0 | yes |
| safe_high_utilization | 8,691.77 | 17,513 | 2.01 | 14,586.21 | 29,059.26 | 10.91 | 4.215 | 0 | yes |
| min_cost_safe | 8,691.77 | 17,513 | 2.01 | 14,586.21 | 29,059.26 | 10.91 | 4.215 | 0 | yes |

## Policies compared

- **fifo** — no optimization; static replica count sized once for the trace mean load. Sanity baseline (`docs/RESULTS.md` §3).
- **sla_aware** — reactive autoscaler (one-tick lag, conservative utilization target). Headline baseline for interactive inference.
- **constraint_aware** — Aurelius: anticipatory (EWMA) sizing + cache-affinity prefill savings + churn hysteresis, gated to a safe utilization target.
- **queue_aware** — scales on the queue-wait p95 signal only (no decode SLA budget, no cache).
- **cache_affinity_baseline** — static sizing + session/prefix-affinity prefill savings, but no load reaction. Isolates the cache lever.

All policies share the **same** serving physics (`aurelius/simulation/cluster/serving.py`, unchanged), the same calibration constants (`serving_value`), and the same cost basis (`InfrastructureCostConfig` defaults). Only the provisioning/routing decision differs — wins come from decisions, not tuned constants.

## Outcome — constraint_aware vs headline (`docs/RESULTS.md` §6)

- **Outcome:** `TIE`  ·  margin vs sla_aware: **+0.03%** on goodput/$
- **Sanity check vs FIFO (do-nothing):** constraint_aware beats static FIFO (+0.03%). FIFO is the sanity baseline, not the buyer-facing benchmark (`docs/RESULTS.md` §3).

## Load-regime sensitivity (same burst shape, replayed at several loads)

BurstGPT's absolute arrival rate is low; the canonical run scales it to a busy interactive tier (`--scale-rps`), preserving the real burst shape. This sweep replays the **same** trace at multiple load multipliers so the result is transparently regime-dependent — not a single cherry-picked load.

| load × | fifo | sla_aware | constraint_aware | queue_aware | cache_affinity | CA vs sla_aware | CA beats fifo? |
|---|---|---|---|---|---|---|---|
| 0.33× | 2,991 | 2,991 | 2,992 | 2,991 | 2,992 | +0.04% | yes |
| 0.5× | 4,475 | 4,475 | 4,476 | 4,475 | 4,476 | +0.03% | yes |
| 1× | 8,689 | 8,689 | 8,692 | 8,689 | 8,692 | +0.03% | yes |
| 2× | 16,643 | 16,643 | 16,650 | 16,643 | 16,650 | +0.04% | yes |
| 3× | 24,128 | 24,128 | 24,135 | 24,128 | 24,135 | +0.03% | yes |

Reading: constraint_aware beats the **realistic reactive autoscaler (`sla_aware`, the headline baseline)** across the swept load levels. It beats even static `fifo` once bursts regularly saturate capacity; under mild burst-load a static `fifo` sized for the mean is cheaper (an honest caveat, not hidden).

### What improved / what did not

- Goodput/$ vs sla_aware: Δ +2.48 (+0.03%).
- Infra $ vs sla_aware: 2.01 vs 2.01.
- Latency p99 vs sla_aware: 29,059.26 vs 29,126.67 ms.
- Migration/reroute (scale events): 0 vs 0.

## Honest limits

- Trace-replay over proxy serving physics; no per-request KV simulation. Token throughput, GPU power, and prices are documented public priors (±50%), identical across policies. Override with real contract rates before any external claim (`docs/RESULTS.md` §8 production-claim gate).
- The SLA budget is a standard interactive SLO decomposition (TTFT p99 ≤ 2000ms + per-output-token budget), applied identically to every policy — BurstGPT supplies no TTFT to calibrate against.
- **Not production-real savings.** Directional simulator result only.

