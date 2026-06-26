# Audit: Alibaba cluster-trace-gpu-v2026 for Aurelius — 2026-06-26

> Triggered by a correct challenge: my earlier Alibaba assessment looked at
> `cluster-trace-gpu-v2023` (packing) and `cluster-trace-v2026-GenAI` (diffusion)
> — **not** the new, much larger **`cluster-trace-gpu-v2026`** (155k GPUs). This
> audits the real v2026 trace from its published README/schema and revises the
> recommendation. Sources at the bottom.

## Do we have it implemented? — No.

The repo ingests `cluster-trace-gpu-v2023` (`alibaba_gpu.py`, packing) and
`cluster-trace-v2026-GenAI` (`alibaba_genai.py`, diffusion serving). It does
**not** ingest `cluster-trace-gpu-v2026`. The challenge was right: my class-mix
calibration used the *older, narrower* v2023 QoS column as a proxy.

## Is it for LLMs or GenAI? — Neither. It's a mixed production fleet.

`cluster-trace-gpu-v2026` is a 6-month anonymized trace of **Alibaba Serverless
Infrastructure (ASI): 155,410 GPUs across 37,707 servers** (OSDI '26). It is
**workload-heterogeneous**, not LLM- or GenAI-specific. The `pod_hourly` table
labels every pod two ways:

- `job_type_public` ∈ **{training, online_inference, offline_inference, dev, other}**
- `model_type_public` ∈ **{genai, rec, cv, embedding, dev, unknown}**

So GenAI is *one* model-type bucket alongside recommendation, CV and embedding;
LLM serving lives inside `online_inference`/`genai` but is not separated out. It
is a **fleet** trace, not a workload trace.

## What it actually contains (grounded in `docs/schema.md`)

| table | one row = | key fields |
|---|---|---|
| `asi_opensource_pod_hourly` | a pod's hour | `job_type_public`, `model_type_public`, `priority_class` (HP/LP), `avg_gpu_sm_util`, `avg_gpu_mem_gib`, `gpu_request`, `pod_id`/`workload_id`/`server_id` |
| `asi_opensource_server_hourly` | a server's hour | `server_id`, `cluster_id`, **`asw_id`** (rack/switch), `gpu_spec_public`, `gpu_count` |
| `asi_opensource_network_hourly` | a server's hour | `server_id`, `rx_gibps_avg`, `tx_gibps_avg` |
| `asi_opensource_job_execution_summary` | a pod's exec span | duration CDFs |

**Granularity = HOURLY pod aggregates.** There are **no per-request arrivals, no
token counts, no TTFT/TPOT, no per-request latency.** **Join = `server_id` + day
+ hour** across all three tables — a *real* cross-layer join (unlike the GenAI
trace's `no_join`), but at server-hour resolution, not per-request.

## How helpful for Aurelius? — Depends entirely on which decision.

### As the joint-SERVING optimizer's SPINE — No.
Aurelius's serving loop is a token-level discrete-event simulator (capacity /
ordering / admission / KV on per-request token traffic). v2026 is **hourly pod
aggregates** — three-plus orders of magnitude too coarse to drive that loop. It
cannot replace the Azure LLM token spine. This is a hard granularity limit, and
it is the same conclusion as before — but for a *better* reason now (it's coarse,
not just "training").

### As a CALIBRATION source — Yes, and materially better than v2023.
This is the real upgrade, and it fixes the exact thing that broke the +9%:

1. **Real serving best-effort ratio.** `job_type_public` gives the true
   **online_inference vs offline_inference** split — i.e. the real
   latency-critical-vs-batch *serving* ratio. My calibration currently proxies this
   with v2023 *training-pod* QoS (LS/BE ≈ 80/20), which is the wrong workload.
   v2026's online/offline-inference labels are the correct, on-domain ratio. Since
   the compounding magnitude is *bound* to this fraction (the whole point of the
   2026-06-26 correction), this is the single highest-value calibration upgrade.
2. **Real model mix** (`model_type_public`: genai/rec/cv/embedding) → grounds a
   future multi-model spine (unlocking the placement lever).
3. **Real per-class utilization** (`avg_gpu_sm_util`, `avg_gpu_mem_gib` by
   job_type) → sanity-checks/calibrates the simulator's implied utilization.
4. **NEW real macro network + topology.** `network_hourly` (per-server rx/tx) +
   `server_hourly` (`asw_id` rack topology) are the **first real public
   topology/traffic signals** in this program. In our signal matrix topology was
   PROXY (Chakra) and fabric-congestion was SIMULATOR_ONLY; v2026 upgrades
   **macro** network-utilization + rack-locality to a real PROXY (hourly,
   server-level). *Caveat:* it is hourly server aggregates — it still does **not**
   give per-link congestion / incast / PFC-ECN / straggler events, so
   micro-congestion stays simulator-only. Macro topology real; micro congestion not.

### As a FLEET-SCHEDULING substrate — Best public option that exists.
If Aurelius's scope extends to fleet-level placement / packing / priority-class
scheduling / topology-aware placement (beyond token serving), this 155k-GPU,
joinable, 6-month trace is the richest public production substrate available —
better than Borg (CPU-centric/older) or Philly (training-only). That is a real
strategic option, but it is a **different product surface** from the token-serving
joint optimizer this workstream has been building.

## Revised recommendation

1. **Serving spine stays Azure LLM (token-level).** v2026 cannot replace it
   (granularity). Unchanged.
2. **Upgrade the class-mix calibration from v2023 → v2026.** Replace the
   training-pod QoS proxy with v2026's real `online_inference`/`offline_inference`
   serving ratio. This is the correct on-domain number for the parameter that
   governs whether serving levers compound. (Shipped: a schema-grounded
   `alibaba_v2026_serving_class_mix()` hook + matrix entries; the precise ratio is
   computed from the pod_hourly table once the multi-GB trace is downloaded — no
   number is fabricated here.)
3. **Add v2026 as the real source for utilization + macro network/topology** in
   the signal matrix (PROXY tier, hourly-server caveat).
4. **Treat fleet-scheduling-on-v2026 as a separate, optional product surface** —
   genuinely promising, but not the token-serving optimizer.

## Honest bottom line

The challenge was right that I was on inferior older traces, and v2026 **does**
improve the dataset — but it improves the **calibration and the fleet-scheduling
substrate**, not the **serving spine**. It does not change the core finding
(token-level serving needs Azure/Mooncake; v2026 is hourly-aggregate). What it
*does* change: the best-effort serving ratio — the parameter whose miscalibration
collapsed the +9% — can now be grounded in **real online/offline inference labels**
instead of a training-pod proxy. That is the highest-value next data step, and the
hook for it is now in place.

## Sources

- [alibaba/clusterdata (program index)](https://github.com/alibaba/clusterdata)
- [cluster-trace-gpu-v2026 README](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/README.md)
- [cluster-trace-gpu-v2026 docs/schema.md](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2026/docs/schema.md)
- [cluster-trace-gpu-v2025 README (DLRM disaggregation, for contrast)](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2025/README.md)
