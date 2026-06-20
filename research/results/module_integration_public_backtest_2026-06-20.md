# Public Backtest — Module Integration Report

> **Directional simulator/backtest evidence only — NOT production savings** (`docs/RESULTS.md` §8). All variants share the same LOCKED serving physics, calibration constants and cost basis (`serving.py` / `economics.py`); only the provisioning/admission decision differs.

- Generated: 2026-06-20
- Load multipliers: {'burstgpt': [1.0, 100.0, 300.0, 600.0], 'azure_llm_2024': [1.0, 10.0, 50.0, 150.0]}

## Summary

Three shadow research modules wired into the public replay and measured against the locked `constraint_aware` baseline on real public traces:

- **B. WorkloadAdmissionGate** (`ca_admission`) — defers best-effort load under KV/queue pressure (KV proxy = realized rho).
- **C. OutputLengthForecastBundle** (`ca_outlen`) — forecast p50 (fit on a warmup prefix, no leakage) *replaces the autoscaler's clairvoyant read of the realized mean* for replica sizing. `ca_outlen_p90` is a tail-sizing over-provisioning sensitivity.
- **D. GpuPlacementScorer** — evaluated on the real-price GPU routing backtest (public LLM traces carry no GPU-type labels).
- **E. all serving modules** (`ca_all`).

### Commands run

```bash
python scripts/run_baseline_public_backtest.py \
    --sample-size 100000 --burstgpt-scales 1,300 --azure-scales 1,50
python scripts/run_module_integration_backtest.py \
    --sample-size 100000 --burstgpt-scales 1,100,300,600 \
    --azure-scales 1,10,50,150
```

Datasets: BurstGPT (real, 1.43M-request CC-BY-4.0 trace, 100k seeded sample) + Azure LLM 2024 (committed 5,880-request sample) + real CAISO/PJM/ERCOT price CSVs. Native (1×) load is sparse → policies tie; saturated multipliers expose the decision.

## burstgpt  (100,000 requests · `data/external/burstgpt/raw/BurstGPT_1.csv`)

### Load 1.0×  (ticks=87790, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 3,876.41 | 1,463.17 | 3,108.93 | 1.498 | 4.20 | 10,360.89 | 0 |
| constraint_aware (baseline / current main) | 3,879.37 | 1,463.17 | 3,108.93 | 1.441 | 4.20 | 10,283.25 | 0 |
| B. admission gate only | 3,879.40 | 1,463.17 | 3,108.93 | 1.441 | 4.19 | 10,283.23 | 0 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 3,879.37 | 1,463.17 | 3,108.93 | 1.441 | 4.20 | 10,283.25 | 0 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 3,879.37 | 1,463.17 | 3,108.93 | 1.441 | 4.20 | 10,283.25 | 0 |
| E. all serving modules | 3,879.40 | 1,463.17 | 3,108.93 | 1.441 | 4.19 | 10,283.23 | 0 |

### Load 100.0×  (ticks=878, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 312,584.32 | 15.37 | 35.89 | 4.178 | 13,151.45 | 33,916.58 | 34 |
| constraint_aware (baseline / current main) | 330,150.13 | 15.57 | 36.39 | 1.997 | 161.76 | 10,958.87 | 38 |
| B. admission gate only | 330,778.43 | 15.55 | 36.35 | 1.963 | 131.84 | 10,902.74 | 38 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 306,637.67 | 14.88 | 34.75 | 2.930 | 5,284.50 | 20,092.71 | 17 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 303,824.35 | 16.98 | 39.57 | 1.825 | 175.05 | 10,879.23 | 102 |
| E. all serving modules | 298,908.08 | 14.92 | 34.82 | 2.714 | 6,273.04 | 21,851.52 | 19 |

### Load 300.0×  (ticks=293, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 646,463.08 | 6.02 | 14.05 | 9.765 | 78,351.75 | 149,897.33 | 35 |
| constraint_aware (baseline / current main) | 789,331.10 | 6.47 | 15.22 | 2.355 | 134.48 | 11,151.51 | 35 |
| B. admission gate only | 786,613.42 | 6.48 | 15.28 | 2.344 | 128.76 | 11,145.62 | 36 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 700,177.80 | 5.93 | 13.80 | 3.666 | 15,356.03 | 38,277.04 | 39 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 572,665.11 | 8.75 | 20.42 | 2.021 | 1,825.97 | 13,942.98 | 99 |
| E. all serving modules | 687,161.32 | 5.93 | 13.80 | 3.739 | 16,991.60 | 41,225.06 | 38 |

### Load 600.0×  (ticks=147, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 946,447.90 | 3.78 | 8.78 | 13.059 | 108,315.69 | 203,661.85 | 34 |
| constraint_aware (baseline / current main) | 1,225,585.81 | 4.22 | 9.81 | 2.446 | 66.04 | 11,155.01 | 40 |
| B. admission gate only | 1,222,077.33 | 4.23 | 9.85 | 2.444 | 66.97 | 11,143.65 | 42 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 1,087,982.19 | 3.80 | 8.74 | 4.175 | 19,884.57 | 46,552.01 | 42 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 757,831.37 | 6.98 | 15.92 | 1.972 | 50.99 | 10,859.38 | 75 |
| E. all serving modules | 1,059,072.78 | 3.83 | 8.81 | 4.260 | 21,275.42 | 48,972.37 | 46 |

## azure_llm_2024  (5,880 requests · `tests/fixtures/azure_llm_2024_sample.csv`)

### Load 1.0×  (ticks=1560, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 9,535.18 | 0 |
| constraint_aware (baseline / current main) | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 9,535.18 | 0 |
| B. admission gate only | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 9,535.18 | 0 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 9,535.18 | 0 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 9,535.18 | 0 |
| E. all serving modules | 12,511.33 | 26.00 | 53.04 | 2.001 | 1.78 | 9,535.18 | 0 |

### Load 10.0×  (ticks=156, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 125,279.79 | 2.60 | 5.30 | 2.288 | 16.93 | 9,587.19 | 0 |
| constraint_aware (baseline / current main) | 125,279.79 | 2.60 | 5.30 | 2.288 | 16.93 | 9,587.19 | 0 |
| B. admission gate only | 125,279.79 | 2.60 | 5.30 | 2.288 | 16.93 | 9,587.19 | 0 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 125,279.79 | 2.60 | 5.30 | 2.288 | 16.93 | 9,587.19 | 0 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 125,279.79 | 2.60 | 5.30 | 2.288 | 16.93 | 9,587.19 | 0 |
| E. all serving modules | 125,279.79 | 2.60 | 5.30 | 2.288 | 16.93 | 9,587.19 | 0 |

### Load 50.0×  (ticks=32, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 604,601.10 | 0.53 | 1.09 | 3.317 | 240.78 | 10,366.29 | 0 |
| constraint_aware (baseline / current main) | 604,601.10 | 0.53 | 1.09 | 3.317 | 240.78 | 10,366.29 | 0 |
| B. admission gate only | 604,601.10 | 0.53 | 1.09 | 3.317 | 240.78 | 10,366.29 | 0 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 604,601.10 | 0.53 | 1.09 | 3.317 | 240.78 | 10,366.29 | 0 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 586,614.08 | 0.55 | 1.12 | 3.263 | 226.32 | 10,323.62 | 2 |
| E. all serving modules | 604,601.10 | 0.53 | 1.09 | 3.317 | 240.78 | 10,366.29 | 0 |

### Load 150.0×  (ticks=11, outlen_fitted=True)

| variant | SLA-safe goodput/$ | GPU-hours | total cost | timeout % | queue p99 (ms) | lat p99 (ms) | scale events |
|---|---:|---:|---:|---:|---:|---:|---:|
| sla_aware (headline baseline) | 1,203,049.63 | 0.27 | 0.54 | 3.810 | 206.18 | 10,750.28 | 2 |
| constraint_aware (baseline / current main) | 1,281,880.39 | 0.25 | 0.51 | 3.912 | 234.41 | 10,830.22 | 2 |
| B. admission gate only | 1,281,880.39 | 0.25 | 0.51 | 3.912 | 234.41 | 10,830.22 | 2 |
| C. output-length forecaster (p50, replaces clairvoyant mean) | 1,581,987.75 | 0.20 | 0.41 | 5.125 | 600.67 | 11,779.21 | 2 |
| C'. output-length forecaster (p90 tail-sizing sensitivity) | 880,625.67 | 0.37 | 0.75 | 3.192 | 87.63 | 10,265.35 | 7 |
| E. all serving modules | 1,581,987.75 | 0.20 | 0.41 | 5.125 | 600.67 | 11,779.21 | 2 |

## KPI Delta Table (module variant − constraint_aware baseline)

| variant | dataset | load | goodput/$ Δ% | GPU-hours Δ | cost Δ | timeout Δ | queue p99 Δ | scale-events Δ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ca_admission | burstgpt | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | -0.01 | +0 |
| ca_admission | burstgpt | 100.0× | +0.19 | -0.017 | -0.03 | -0.034 | -29.92 | +0 |
| ca_admission | burstgpt | 300.0× | -0.34 | +0.017 | +0.05 | -0.011 | -5.71 | +1 |
| ca_admission | burstgpt | 600.0× | -0.29 | +0.017 | +0.03 | -0.002 | +0.92 | +2 |
| ca_outlen | burstgpt | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen | burstgpt | 100.0× | -7.12 | -0.683 | -1.63 | +0.933 | +5122.74 | -21 |
| ca_outlen | burstgpt | 300.0× | -11.29 | -0.533 | -1.42 | +1.310 | +15221.55 | +4 |
| ca_outlen | burstgpt | 600.0× | -11.23 | -0.417 | -1.08 | +1.730 | +19818.53 | +2 |
| ca_outlen_p90 | burstgpt | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen_p90 | burstgpt | 100.0× | -7.97 | +1.417 | +3.18 | -0.172 | +13.29 | +64 |
| ca_outlen_p90 | burstgpt | 300.0× | -27.45 | +2.283 | +5.19 | -0.334 | +1691.50 | +64 |
| ca_outlen_p90 | burstgpt | 600.0× | -38.17 | +2.767 | +6.11 | -0.473 | -15.05 | +35 |
| ca_all | burstgpt | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | -0.01 | +0 |
| ca_all | burstgpt | 100.0× | -9.46 | -0.650 | -1.56 | +0.717 | +6111.29 | -19 |
| ca_all | burstgpt | 300.0× | -12.94 | -0.533 | -1.42 | +1.384 | +16857.12 | +3 |
| ca_all | burstgpt | 600.0× | -13.59 | -0.383 | -1.00 | +1.814 | +21209.38 | +6 |
| ca_admission | azure_llm_2024 | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_admission | azure_llm_2024 | 10.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_admission | azure_llm_2024 | 50.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_admission | azure_llm_2024 | 150.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen | azure_llm_2024 | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen | azure_llm_2024 | 10.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen | azure_llm_2024 | 50.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen | azure_llm_2024 | 150.0× | +23.41 | -0.050 | -0.10 | +1.213 | +366.26 | +0 |
| ca_outlen_p90 | azure_llm_2024 | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen_p90 | azure_llm_2024 | 10.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_outlen_p90 | azure_llm_2024 | 50.0× | -2.98 | +0.017 | +0.03 | -0.054 | -14.46 | +2 |
| ca_outlen_p90 | azure_llm_2024 | 150.0× | -31.30 | +0.117 | +0.24 | -0.720 | -146.78 | +5 |
| ca_all | azure_llm_2024 | 1.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_all | azure_llm_2024 | 10.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_all | azure_llm_2024 | 50.0× | +0.00 | +0.000 | +0.00 | +0.000 | +0.00 | +0 |
| ca_all | azure_llm_2024 | 150.0× | +23.41 | -0.050 | -0.10 | +1.213 | +366.26 | +0 |

## GPU Placement Routing (real prices, synthetic jobs) — proxy vs real KPI

| metric | value | kind |
|---|---:|---|
| routing improvement (pp more LC on best GPU) | 54.67 | **proxy** |
| mean GPU penalty reduction | 0.279 | **proxy** |
| realized energy cost Δ ($) | 111.06 | real |
| goodput/$ Δ (all jobs) | -0.000422 | **real KPI** |
| latency_critical goodput/$ Δ | -0.027211 | **real KPI** |

The scorer moves the routing **proxy** strongly (more latency_critical jobs on the fast GPU) but the **real economic KPI does not improve**: routing to the faster/pricier GPU raises cost without raising goodput in this model, so goodput/$ is flat-to-negative and the latency_critical subset regresses. Proxy movement is not success.

### Data caveats

- **BurstGPT (real, 100k sample) is the robust evidence**: 147–878 ticks at the saturated scales. Verdicts are read from it.
- **Azure-2024 is a small committed sample (5,880 reqs)**: at saturating multipliers it compresses to only 11–32 ticks, so its per-scale deltas are noisy. Any isolated Azure swing (e.g. a single-scale `ca_outlen` +23% at 150× / 11 ticks) is a small-sample artifact, contradicted by the well-sampled BurstGPT result for the same module — it is NOT evidence of improvement.
- Native (1×) load is sparse for both traces → all variants tie (already established by the locked runners).

## Interpretation — helped / hurt / neutral / inconclusive

| module | verdict | BurstGPT goodput/$ Δ (100/300/600×) | evidence |
|---|---|---|---|
| B. WorkloadAdmissionGate | **NEUTRAL** | +0.19%, -0.34%, -0.29% | baseline already provisions to a safe rho, so the gate rarely fires; deferral nets to ~0 |
| C. OutputLengthForecastBundle (p50) | **HURT** | -7.12%, -11.29%, -11.23% | forecast under-sizes vs the clairvoyant realized mean the baseline already uses → SLA violations up; SRTF ordering lever is absent from the aggregate replay |
| E. all serving modules | **HURT** | -9.46%, -12.94%, -13.59% | dominated by the output-length regression |
| D. GpuPlacementScorer | **HURT (proxy moved, real KPI regressed)** | n/a (no GPU labels in LLM traces) | real-price routing: goodput/$ flat-to-negative, latency_critical subset regressed |

## Recommendation

- **Do not enable any module in runtime.** No module improves SLA-safe goodput/$ on the robust public replay (BurstGPT).
- Keep **WorkloadAdmissionGate** shadow-only: neutral on the public replay because the autoscaling baseline is already SLA-safe (low rho), so admission back-pressure rarely fires.
- Keep **OutputLengthForecastBundle** shadow-only: it regresses the aggregate autoscaling benchmark (the baseline already reads the realized mean). Its designed SRTF-ordering benefit needs a per-request discrete-event queue the public benchmark does not model — revisit only with such a harness.
- Keep **GpuPlacementScorer** shadow-only: it improves the routing proxy but not the real economic KPI on the only available real-price evaluation; public LLM traces carry no GPU-type labels to validate it directly.
- Merge the **backtest infrastructure + this report** only (`module_backtest.py`, the two runner scripts, results artifacts). Runtime decision paths are unchanged.

> No benchmark definition, SLA budget, price trace, workload trace, or baseline policy was modified. The three modules remain shadow-only (`enabled=False` defaults); this run added evaluation infrastructure and this report only.

