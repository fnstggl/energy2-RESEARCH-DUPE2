# Telemetry Adapter Readiness Audit — Dynamic Serving Frontier Calibration

> **Audit only.** No new datasets ingested. No robust energy engine changes. No
> scheduler behavior changes. No real execution enabled. No production-readiness
> claim. The static frontier controller, dynamic estimator v1, `constraint_aware`
> default rho (0.65), and committed Azure / Philly / MIT / Alibaba / BurstGPT
> backtest artifacts are read-only.

This document audits whether the existing telemetry adapters in
`aurelius/connectors/`, `aurelius/state/`, `aurelius/ingestion/`,
`aurelius/sla/`, and `aurelius/residency/` already cover the fields the
Dynamic Serving Frontier Calibration + Shadow Evaluation harness
(`docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`) needs, so that
calibration can be run on **real pilot telemetry** (as required by
`docs/PILOT_TELEMETRY_CONTRACT.md`), not only on simulator output.

- **Read first:** `docs/PILOT_TELEMETRY_CONTRACT.md`,
  `docs/DYNAMIC_SERVING_FRONTIER_CALIBRATION.md`,
  `docs/DYNAMIC_SAFE_FRONTIER_ESTIMATOR.md`,
  `docs/CONSTRAINT_AWARE_FRONTIER_INTEGRATION.md`,
  `docs/RESULTS.md` §8.

## 1. What the dynamic frontier estimator needs

`aurelius/frontier/dynamic_models.py::ServingTelemetryTick` is the
canonical input. Every numeric field is `Optional` — missing
telemetry stays `None`, never zero-filled.

| ServingTelemetryTick field | type | unit | role in the estimator |
|---|---|---|---|
| `timestamp_s` | float | s | window ordering |
| `observed_rps` | float | req/s | required (`DEFAULT_REQUIRED_FIELDS`) — load + burstiness CV |
| `prompt_tokens_per_s`, `output_tokens_per_s`, `total_tokens_per_s` | float | tok/s | tokens-per-replica / per-$ proxies |
| `active_replicas` | int | — | **required** — replicas scaling under rho changes |
| `gpu_hours_delta` | float | h | denominator of goodput/$; can be derived from replicas × tick duration |
| `mean_utilization` | float | 0–1 | the observed rho EMA (calibration anchor for Erlang-C tail) |
| `queue_p50_ms`, `queue_p95_ms`, `queue_p99_ms` | float | ms | **queue_p99 required** — Erlang-C tail calibration + safety gate |
| `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms` | float | ms | optional safety gate |
| `timeout_pct` | float | % | safety gate (≤ 10 % default) + SLA-risk EMA |
| `sla_violation_pct` | float | % | diagnostic (not currently a safety gate) |
| `scale_events_delta` | int | — | churn-suppression signal |
| `churn_delta` | float | — | churn-suppression signal |
| `telemetry_confidence` | enum | unknown / low / medium / high | controls risk-estimator fallback |
| `source` | string | — | provenance |

The dynamic calibration harness additionally consumes the *realized
outcome* at the next decision window via
`DynamicFrontierObservedOutcome`: `observed_goodput_per_dollar`,
`observed_timeout_pct`, `observed_queue_p99_ms`,
`observed_latency_p99_ms`, `observed_sla_violation_pct`,
`observed_gpu_hours`, `observed_churn`, `was_safe`.

## 2. Adapter inventory (what is already in the repo)

| adapter | module | runtime | tested by |
|---|---|---|---|
| Prometheus HTTP / `/metrics` scrape (real + fake) | `aurelius/connectors/prometheus.py` | any Prometheus-compatible exporter | `tests/test_prometheus_connector.py`, `tests/test_fake_connectors.py` |
| DCGM (NVIDIA GPU) | `aurelius/connectors/dcgm.py` + `aurelius/ingestion/dcgm_provider.py` | DCGM-exporter | `tests/test_dcgm_adapter.py`, `tests/test_dcgm_provider.py` |
| vLLM | `aurelius/connectors/vllm.py` | vLLM ≥ V0 (V1 colon namespace) | `tests/test_vllm_triton_ray_adapters.py` |
| Triton | `aurelius/connectors/triton.py` | NVIDIA Triton 23.x+ | `tests/test_vllm_triton_ray_adapters.py` |
| Ray Serve | `aurelius/connectors/ray_serve.py` | Ray Serve 2.x | `tests/test_vllm_triton_ray_adapters.py` |
| OpenTelemetry (fixture-only OTLP JSON) | `aurelius/connectors/otel.py` | OTel SDK / Collector → OTLP | `tests/test_vllm_triton_ray_adapters.py` |
| Kubernetes read-only (pod / node / GPU capacity) | `aurelius/connectors/kubernetes.py` | K8s 1.24+ | `tests/test_kubernetes_connector.py` |
| GPU topology (`nvidia-smi topo`) | `aurelius/connectors/topology.py` | host with NVIDIA driver | `tests/test_topology_connector.py`, `tests/test_topology_realism.py` |
| Cluster-state assembler | `aurelius/state/assemble.py` | combines all of the above | `tests/test_state_assemble.py` |
| Queue / wait-time CSV provider | `aurelius/ingestion/queue_provider.py` | offline / CSV | `tests/test_queue_aware.py` |
| Model residency events / requests | `aurelius/residency/ingest.py` + `aurelius/residency/sim.py` | pilot CSV / JSONL | `tests/test_residency_ingestion.py`, `tests/test_residency_audit_report.py` |
| ArrivalTick (public-trace replay) → `ServingTelemetryTick` | `aurelius/frontier/dynamic_telemetry.py::telemetry_tick_from_arrival_tick` | offline replay only | `tests/test_dynamic_frontier_estimator.py` |
| `InferenceServiceState` → `ServingTelemetryTick` bridge | `aurelius/frontier/dynamic_telemetry.py::telemetry_tick_from_inference_service_state` | **wires real adapter output into the dynamic frontier** | `tests/test_telemetry_adapter_readiness.py` |

SGLang has no Prometheus adapter; it appears only as a runtime-tag in
`aurelius/simulation/cluster/calibration.py`. See §5.

## 3. Coverage matrix — required field → adapter source

Each row lists the **best available real-telemetry source** for the
ServingTelemetryTick field used by the dynamic frontier estimator.

| required field | best real source | status | tested? | confidence |
|---|---|---|---|---|
| `observed_rps` | vLLM `rate(vllm:request_success_total[1m])`, Ray Serve `rate(ray_serve_num_ongoing_requests_total[1m])`, Triton derived from `nv_inference_count` rate | **implemented** | yes | high (vLLM, Ray); medium (Triton — counter, not rate) |
| `active_replicas` | Kubernetes `K8sPlacementSnapshot.running_gpu_pods` count per workload **or** Ray Serve `ray_serve_deployment_replica_count` | **implemented** | yes (K8s + Ray) | high |
| `queue_p99_ms` | vLLM `vllm:request_first_token_seconds_bucket` p99 (TTFT) or Ray Serve `ray_serve_request_latency_ms_bucket` p99 | **partial — p99 is e2e latency, not queue wait** | yes | medium (vLLM derives ms from histogram; Triton has no histogram → average only) |
| `queue_p95_ms` | same family as p99 | **partial** | yes | medium |
| `queue_p50_ms` | vLLM `vllm:e2e_request_latency_seconds_bucket` p50 OR Triton `nv_inference_queue_duration_us / exec_count` (average, not percentile) | **implemented (vLLM); partial (Triton average)** | yes | medium |
| `latency_p99_ms` | vLLM `vllm:e2e_request_latency_seconds_bucket` p99; Ray Serve `ray_serve_request_latency_ms_bucket` p99 | **implemented** | yes | high (vLLM, Ray); not available (Triton default) |
| `latency_p95_ms` | same as p99 | **implemented** | yes | high (vLLM, Ray); not available (Triton default) |
| `latency_p50_ms` | same family | **implemented** | yes | high |
| `timeout_pct` | derived from `error_rate_pct` (Ray Serve / Triton failure ratio) OR computed upstream from latency_p99 vs SLA threshold | **partial — best-effort equivalence; not a true timeout share** | yes (bridge) | low (error_rate ≠ SLA-timeout in general) |
| `sla_violation_pct` | NOT directly emitted by vLLM / Triton / Ray Serve; must be computed by the pilot from `latency_p99_ms > latency_sla_p99_ms` with a request-level histogram | **missing** | n/a | low — requires pilot-side derivation |
| `mean_utilization` (rho) | DCGM `DCGM_FI_DEV_GPU_UTIL` (instantaneous), `DCGM_FI_PROF_GR_ENGINE_ACTIVE` (SM-active ratio), or controller-supplied rho | **implemented (DCGM); rho recovered by controller** | yes | high (DCGM) |
| GPU memory used / free / total | DCGM `DCGM_FI_DEV_FB_USED/_FREE/_RESERVED` | **implemented** (not on `ServingTelemetryTick` but on `GPUState` for safety gates) | yes | high |
| `scale_events_delta` | Kubernetes `Deployment.metadata.generation` delta over the window OR HPA `events` count | **partial — K8s pod count is available; explicit scale-event counter is not** | partial | medium |
| `churn_delta` | derived from migration history (`aurelius/state/models.py::MigrationHistory`) OR replica-count slope | **partial — derived signal, not a primary metric** | partial | medium |
| `gpu_hours_delta` | derived from `active_replicas × tick_duration / 3600` (the bridge helper does this) | **derived** | yes | high |
| `telemetry_confidence` | propagated from `Provenance.confidence` (coverage-based: high > 70 %, medium > 40 %, low otherwise) | **implemented** | yes | high |
| `source` | `TelemetrySnapshot.source` | **implemented** | yes | high |
| Prefix cache hit rate (diagnostic) | vLLM `vllm:gpu_prefix_cache_hit_rate` | **implemented** | yes | high (vLLM only) |
| Model residency / cold-start (diagnostic) | `aurelius/residency/ingest.py::adapt_vllm` ; pilot per-request CSV | **partial — vLLM has no per-request cold-start signal; pilot must emit per-request fields per `docs/PILOT_TELEMETRY_CONTRACT.md` §2** | yes | low (vLLM); high (pilot CSV w/ explicit fields) |

Legend:
- **implemented** — production-ready real-telemetry path with tests
- **partial** — real-telemetry path exists but with a documented caveat
  (averages instead of percentiles, derived rather than primary,
  missing on one of the three runtimes, etc.)
- **missing** — no real-telemetry source; pilot must derive upstream

## 4. Wiring real telemetry into the dynamic frontier (the bridge)

The audit identified a single missing piece: a typed bridge from
the connector layer's canonical `InferenceServiceState` (the output of
the vLLM / Triton / Ray Serve adapters) to the dynamic frontier's
`ServingTelemetryTick`. The previous v1 only knew how to build a tick
from an `ArrivalTick` (public-trace replay format).

That bridge is now a one-function export:

```python
from aurelius.frontier import telemetry_tick_from_inference_service_state

tick = telemetry_tick_from_inference_service_state(
    inference_service_state,
    tick_duration_s=60.0,
    mean_utilization=gpu_state.util_pct / 100.0,  # DCGM rho
    scale_events_delta=k8s_scale_events_in_window,
    churn_delta=migration_history.recent_churn,
)
```

Honesty rules built into the bridge:
- Missing fields stay `None`.
- `timeout_pct` is set to `error_rate_pct` only when the connector
  exposed it — never invented. Pilots that want a true timeout share
  must compute it upstream (latency_p99 vs SLA threshold) and surface
  it as `error_rate_pct` OR override on the resulting tick.
- `mean_utilization` is NOT in `InferenceServiceState` — it comes from
  DCGM (`gpu.util_pct`) or a controller-level signal. The caller passes
  it in.
- `gpu_hours_delta = active_replicas × tick_duration_s / 3600` is
  derived only when both inputs are known.
- `scale_events_delta` / `churn_delta` are passed in by the caller from
  K8s deployment generation deltas / migration history. No fabrication.
- `telemetry_confidence` is inherited from the connector's
  `Provenance.confidence`.

The bridge is **opt-in** — it is a pure helper. The static frontier
controller, the dynamic estimator v1, the `constraint_aware` engine,
the robust energy engine, and the simulator are all unchanged.

## 5. What is production-real vs simulated-only today

| signal | real source | simulator-only on which traces |
|---|---|---|
| `observed_rps` | vLLM / Ray Serve / Triton counters | Azure 2024 / Azure 2023 / BurstGPT (arrival-rate replay), MIT Supercloud (scheduler-log only) |
| `queue_p99_ms` | vLLM histogram, Ray Serve histogram | computed by the simulator from offered load × Erlang-C tail on every public trace (no public trace ships realized queue percentiles per tick) |
| `latency_p99_ms` | vLLM / Ray Serve histograms | computed by the simulator on every public trace |
| `timeout_pct` | derived from `error_rate_pct` (best-effort) | computed by the simulator from latency > SLA on every public trace; **no public trace ships a real timeout rate per tick** |
| `mean_utilization` | DCGM `DCGM_FI_DEV_GPU_UTIL` | computed by the simulator on every public trace |
| `active_replicas` | K8s pod count + Ray Serve replica count | scheduler/sizer output on every public trace |
| `gpu_hours_delta` | DCGM + K8s timing | derived on every public trace from replicas × tick |
| Model residency / cold-start | pilot per-request fields (`docs/PILOT_TELEMETRY_CONTRACT.md` §2) | the GenAI 2026 trace's request↔infra layers are `no_join` (cold-start unattributed); MIT Supercloud has node-level GPU residency but not per-model residency |
| Prefix cache hit rate | vLLM `vllm:gpu_prefix_cache_hit_rate` | not in any public trace |

The honest summary: **every safety + estimator signal the dynamic
frontier needs is available from real adapters that already live in
this repo for at least one of {vLLM, Triton, Ray Serve, DCGM, K8s}**.
The public-trace backtests use the simulator to *generate* these
signals from arrival-rate replay; that is the right approximation for
shadow evidence. A pilot that wires up vLLM + DCGM + K8s into the
bridge above can run the dynamic frontier calibration on real data
today, subject to the caveats in §3 (`timeout_pct` is best-effort;
`sla_violation_pct` requires pilot-side derivation; `scale_events_delta`
needs K8s deployment-generation deltas).

## 6. Can Dynamic Serving Frontier calibration run on real telemetry today?

**YES — for vLLM + DCGM + K8s pilots, with the caveats documented in §3
and §5.**

| capability | answer |
|---|---|
| Can vLLM real telemetry feed `ServingTelemetryTick`? | Yes — `VLLMAdapter` → `InferenceServiceState` → `telemetry_tick_from_inference_service_state` |
| Can Triton real telemetry feed `ServingTelemetryTick`? | Yes, but p95/p99 latency are unavailable from default Triton metrics (averages only). Use Triton's optional histogram exporter or a vLLM-fronted Triton ensemble for percentiles. |
| Can Ray Serve real telemetry feed `ServingTelemetryTick`? | Yes — `RayServeAdapter` provides p50/p95/p99 latency + replicas + queue + error rate. |
| Can DCGM real telemetry supply `mean_utilization`? | Yes — `gpu.util_pct` (0–100) divided by 100. |
| Can K8s real telemetry supply `active_replicas` + `scale_events_delta`? | Yes for replicas (`K8sPlacementSnapshot.running_gpu_pods`); scale-events requires a delta over a `Deployment.metadata.generation` snapshot — straightforward but not a primary export today. |
| Can pilot CSV/JSONL supply model residency for cold-start scoring? | Yes via `aurelius/residency/ingest.py`; vLLM cannot emit per-request load timestamps so the pilot MUST add them upstream. |
| Can the calibration **act** on the cluster in real mode? | **No.** Real-cluster execution stays disabled by default; the calibration harness is recommendation-only (`docs/SAFE_UTILIZATION_FRONTIER_CONTROLLER.md` §"Real-mode execution boundary"). |

## 7. Remaining real-telemetry gaps

These do NOT block shadow-mode calibration on a vLLM + DCGM + K8s pilot,
but they are open work before a production-savings claim is possible
(`docs/RESULTS.md` §8):

1. **True per-tick timeout share.** No mainstream serving runtime emits
   a SLA-aware timeout counter. The bridge falls back to
   `error_rate_pct`. Pilots that need a true timeout share must compute
   it upstream from `(latency_p99 > latency_sla_p99_ms) / total_requests`.
2. **`sla_violation_pct`.** Same provenance gap as `timeout_pct`. The
   `sla` package (`aurelius/sla/`) carries per-workload SLA policies;
   pilots can wire those into a request-level evaluator and emit the
   share, but no adapter does this today.
3. **`scale_events_delta`.** The K8s connector emits a placement
   snapshot, not a deployment-generation delta. A small follow-up to
   `KubernetesConnector` (`get_deployment_generations`) would close
   this, but is not in scope for the audit.
4. **SGLang.** No Prometheus adapter exists. Adding one would mirror
   `aurelius/connectors/vllm.py` (SGLang exposes `sglang_*` metrics).
   Out of scope for this audit.
5. **TensorRT-LLM.** Same as SGLang — no adapter today. Out of scope.
6. **Per-request cold-start timestamps.** Required by
   `docs/PILOT_TELEMETRY_CONTRACT.md` §2 for residency-actuation
   conformance. vLLM does not emit them; the pilot must instrument the
   model-loader hook. The audit doc surfaces this honestly.

## 8. Honesty / scope

- This audit is a **documentation + tiny bridge export** only. It adds
  one helper function (`telemetry_tick_from_inference_service_state`),
  the audit doc you are reading, a JSON summary, and a readiness test.
- **No new dataset is ingested.** No connector is changed beyond the
  single bridge export.
- **No safety gate is weakened.** The bridge preserves `None` for every
  missing field; the dynamic estimator's existing
  INSUFFICIENT_TELEMETRY fallback handles them.
- **No real execution is enabled.** Real-cluster execution stays
  disabled by default.
- **No production-savings claim.** Calibrating the dynamic frontier on
  real telemetry remains shadow-mode evidence until the `docs/RESULTS.md`
  §8 gate is satisfied.
