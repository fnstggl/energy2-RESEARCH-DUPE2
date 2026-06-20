# SRTF Serving-Queue Backtest Results (Run 2026-06-20-g)

> **Simulator / public-trace directional result — NOT a production-savings
> claim.** Per `docs/RESULTS.md` §8 the production-claim gate is not met. Azure
> LLM 2024 is a public serving trace, not customer telemetry. Numbers below are
> from a discrete-event queue simulator with documented service-physics priors.

## Question

Run 2026-06-20-f wired a `predicted_output_tokens` SRTF sort key into the batch
`JobScheduler` and found it **neutral** on the 26-day canonical energy trace.
The diagnosis was "no queue contention." This run answers: **does SRTF produce
a real improvement once genuine queue contention is present, on a real LLM
serving trace?**

## Two findings

### 1. The batch `JobScheduler` cannot express the SRTF benefit (negative)

`aurelius/benchmarks/srtf_contention_backtest.py` builds a capacity-contended
batch workload (real Azure 2024 output-token sizes, binding power cap,
contention ratio up to 4.6×) and runs FIFO vs SRTF through the merged greedy
scheduler.

| horizon | contention ratio | FIFO goodput/$ | SRTF goodput/$ | Δ |
|---|---:|---:|---:|---:|
| 18h | 4.63 | 0.2998 | 0.2997 | −0.03% |
| 24h | 3.47 | 0.3041 | 0.3039 | −0.04% |
| 36h | 2.32 | 0.3035 | 0.3034 | −0.05% |

**Root cause:** the greedy batch scheduler has **no queue-wait semantics**. When
capacity is exhausted it places a job at `earliest_start` (a fallback that
ignores the cap) rather than making the job *wait*, so the processing *order*
never changes a completion time. Zero deadline misses occur even at 4.6×
contention. The analytical Erlang-C model in `simulation/cluster/serving.py` is
likewise an aggregate M/M/c formula with no per-request ordering. **Neither
merged code path can express request-level SRTF.**

### 2. Request-level SRTF on a real LLM serving queue (large positive)

`aurelius/benchmarks/srtf_serving_backtest.py` is a discrete-event,
non-preemptive **M/G/c** simulator that processes the real Azure LLM 2024 request
stream (5,880 requests, real heavy-tailed output lengths: p50≈90, p99≈479,
max≈1346 tokens) through `c=4` replicas, comparing FIFO vs shortest-predicted-
job-first. Arrivals are time-warped by a single scalar (applied identically to
every discipline) to reach a realistic cluster utilization.

**Headline @ ρ=0.85, c=4, E2E SLA=10s:**

| metric | FIFO | SRTF (perfect prior) | SRTF (30%-CV forecast) | improvement |
|---|---:|---:|---:|---:|
| short-request p90 response | 696.2 s | 3.03 s | 3.17 s | **+99.6% / +99.5%** |
| mean response | 343.9 s | 129.9 s | 141.2 s | **+62.2%** |
| SLA-safe goodput / $ | 13,336 | 56,481 | 56,855 | **+323.5%** |
| p50 response | 342.2 s | 2.71 s | 2.78 s | +99.2% |
| **long-tail p99 response** | **732.7 s** | **2188.7 s** | **2232.6 s** | **−199% (REGRESSES)** |

**Across utilization (perfect prior, c=4):**

| ρ | short-p90 improvement | mean improvement | SLA-goodput/$ Δ |
|---|---:|---:|---:|
| 0.80 | +99.5% | +63.9% | +252% |
| 0.85 | +99.6% | +62.2% | +324% |
| 0.92 | +99.6% | +60.1% | +314% |

## Interpretation (honest)

- **The SRTF principle works, and it works on the real trace.** Short requests
  stop waiting behind long ones; their p90 latency collapses from ~700s to ~3s
  (essentially pure service time). This exceeds the +32% p90 figure from
  arXiv:2604.06970 because this trace + load regime is burstier and more heavily
  loaded than the paper's setting — the gain is regime-dependent, not a constant.
- **A realistic forecast prior is enough.** With 30%-CV lognormal forecast noise
  (an `OutputLengthForecastBundle`-quality predictor) the short-request p90
  improvement is essentially unchanged (+99.5%). The ordering is robust to
  forecast error because it only needs to separate short from long, not predict
  exactly. **No actual-length leakage:** ordering uses the predicted value;
  service physics always uses the actual token count.
- **The honest cost: long-request starvation.** Non-preemptive SJF pushes the
  long-request tail from p99≈733s to p99≈2189s. The SLA-safe goodput/$ gain is a
  *net* of many short requests now meeting the SLA against a few long ones now
  missing it. This trade-off is asserted in the test suite so it cannot silently
  disappear; the mitigation (aging / SRPT preemption / hybrid bands) is the
  documented next step.

## Caveats

1. **Baseline is FIFO, not SLA-aware.** The +323% goodput/$ is vs first-come-
   first-served, a weaker baseline than the SLA-aware scheduler used for the
   mission's +300% aspirational target. This is **not** a claim of +300% vs
   SLA-aware.
2. **Magnitude is regime-dependent.** The numbers depend on load (ρ),
   burstiness, server count, and the SLA budget. At light load (ρ=0.10, 64
   servers) the effect is small — the win is fundamentally a contention
   phenomenon.
3. **Service-time model is a documented proxy** (`TTFT_BASE_S + tokens·TPOT_S`),
   applied identically to every discipline. Only the queue ordering differs, so
   every reported delta is attributable to ordering.
4. **Time-warp** is a single scalar to reach a realistic ρ on a downsampled
   public sample; it preserves the real token distribution and burst shape and
   is identical across disciplines.

## Where this leaves the system

- The merged batch-scheduler sort key (run -f) remains correct and backward-
  compatible but is **inert for serving workloads** — it lives in the wrong
  layer to capture request-level SRTF.
- The value lives in the **serving request queue**. The next implementation step
  (future run) is to expose an SRTF/SPRPT ordering option in the serving path
  driven by `OutputLengthForecastBundle.p50`, with an **aging/preemption guard**
  to bound long-request starvation, then re-run this backtest end-to-end.

## Reproduce

```python
from aurelius.benchmarks.srtf_serving_backtest import run_srtf_serving_backtest
print(run_srtf_serving_backtest(servers=4, target_rho=0.85).to_dict())
```

Tests: `tests/test_srtf_serving_backtest.py` (27), `tests/test_srtf_contention_backtest.py` (11).
