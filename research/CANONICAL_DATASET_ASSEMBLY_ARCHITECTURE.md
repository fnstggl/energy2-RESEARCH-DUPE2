# Canonical Production-Like Dataset — Assembly Architecture (with v2026) — 2026-06-26

> How to actually assemble the unified canonical dataset now that
> `cluster-trace-gpu-v2026` is on the table. The short version: **it is two
> planes, not one trace** — a per-request *serving* plane (Azure/Mooncake) and an
> hourly *fleet* plane (v2026), coupled by calibrated parameters, never merged
> row-by-row. Grounded in the real v2026 `schema.md`.

## 0. The one idea that resolves everything

Aurelius makes **two different decisions** at **two different timescales**, and
they need **two different datasets**:

| plane | decision | timescale | unit | spine |
|---|---|---|---|---|
| **Serving** | capacity / ordering / admission / KV routing | seconds | one request (tokens) | **Azure LLM** (+ Mooncake) |
| **Fleet** | placement / packing / topology / priority scheduling | hours | one pod / server | **cluster-trace-gpu-v2026** |

A "unified canonical dataset" that tries to be **one flat table** spanning both is
the monstrous-dataset trap. The honest unification is two planes that **share a
calibration/parameter layer and a manifest**, not rows.

## 1. Why hourly pod aggregates cannot be the serving spine (rigorously)

This is not "wrong workload" hand-waving — it is an **observability/resolution
mismatch**. The serving optimizer is a *discrete-event simulator over individual
requests at second resolution*. Each lever needs information that aggregation
destroys:

- **Ordering (SRPT)** picks which request to run next *among those queued right
  now*. Needs per-request arrival + size. An hourly row has **no individual
  requests** — there is literally nothing to order.
- **Admission** defers *this* request based on the instantaneous queue. Needs a
  second-scale queue. Hourly aggregates have **no queue dynamics** inside the hour.
- **Capacity (reactive/forecast)** re-sizes replicas every ~60 s reacting to the
  *burst in the last tick*. An hourly mean **smooths every burst away** — the exact
  signal capacity exists to handle is the thing aggregation removes.
- **SLA / goodput** is per-request ("did *this* finish in 10 s?"). Hourly rows have
  **no per-request latencies**.
- **Service physics** is `TTFT + tokens·TPOT`. Hourly rows have **no token counts**.

You cannot reconstruct 5,880 individual arrivals + token counts + per-request
latencies from "this pod averaged 62 % SM-util this hour." The information was
integrated out. (Analogy: you can't time a traffic light from "this road carried
4,200 cars this hour at 71 % capacity" — you need the car-by-car arrival process.)
v2026 **confirms** this: tables are "one row = one pod during one hour," partitioned
by day+hour only, **no sub-hourly timestamp, no per-request/per-inference data**.

So: v2026 is the *fleet* spine and a *serving* **calibrator** — never the serving
spine. Azure (token-level) stays the serving spine. This is a granularity fact,
not a preference.

## 2. Does v2026 give us topology + macro network we lacked? — Yes (new), with a granularity caveat

Before, fleet topology/network were absent (Chakra is *collective* traces, not
fleet topology). v2026 adds, as **real** signals:

- `server_hourly.asw_id` → **rack / access-switch topology** (ASW-local placement,
  rack locality) + `gpu_spec_public` heterogeneous inventory.
- `network_hourly.rx_gibps_avg / tx_gibps_avg` → **per-server macro network
  traffic**, joinable to pods by `server_id`+hour.

**Caveat (no overclaim):** these are *hourly server aggregates*. They give **macro**
network utilization and **rack topology** — not per-link congestion, incast,
PFC/ECN counters, or labeled stragglers (those remain proprietary / simulator-only).
So: topology + macro traffic = newly real; micro-congestion = still simulator-only.

## 3. Your 13-signal checklist, audited against the real v2026 schema

| signal | in v2026? | grounded note |
|---|---|---|
| Job arrivals | ✅ *job-level* | pod first-hour appearance — **job** arrivals at hourly res, **not request** arrivals (the serving spine still needs Azure for those) |
| Job durations | ✅ | `job_execution_summary` spans + `schedule_delay_sec`/`ready_delay_sec` |
| GPU utilization | ✅ | `avg_gpu_sm_util` (hourly) |
| CPU/memory utilization | ✅ | `avg_cpu_request_util`, `avg_memory_util`, `avg_gpu_mem_gib` |
| Job priority/QoS | ✅ | `priority_class` (HP/LP/Other) |
| Cluster topology | ✅ | `asw_id` (rack/switch) |
| Multi-GPU allocation | ✅ | `gpu_request`, `gpu_count` |
| Network traffic | ✅ *macro* | `network_hourly` rx/tx (server-hour; no per-link) |
| Machine inventory | ✅ | `server_hourly` (gpu_spec, gpu_count, cpu_capacity) |
| Heterogeneous GPU types | ✅ | `gpu_spec_public` |
| Scheduler outcomes | ⚠️ partial | placement (`server_id`) recorded; **why/preemption/rejection not** |
| Queue waiting times | ✅ | **`schedule_delay_sec` + `ready_delay_sec`** — actually *better* than "partial"; hourly/pod res |
| Kubernetes metadata | ⚠️ partial | `pod_id`/`workload_id`; ASI is k8s but limited k8s metadata exposed |

**Verdict:** for the **fleet plane** v2026 delivers ~11/13 fully (2 partial) — it is
the most complete public fleet-scheduling substrate that exists. The checklist's two
"Critical" serving signals are the catch: "job arrivals ✅" is *job*-level not
*request*-level, and none of it is sub-hourly — so the whole table is a **fleet**
asset, and a **serving calibrator**, not a serving spine.

## 4. The assembly — two planes + a calibration bridge

```
 SERVING PLANE  (per-request, ~1s)          FLEET PLANE  (per-pod/server, 1h)
 ┌───────────────────────────────┐          ┌───────────────────────────────┐
 │ SPINE: Azure LLM 2024         │          │ SPINE: cluster-trace-gpu-v2026 │
 │  real arrivals + tokens       │          │  pod_hourly + server_hourly +  │
 │ + Mooncake hash_ids (KV)      │          │  network_hourly (join: server_ │
 │ + energy price (hour-of-day)  │          │  id+hour) — topology, util,    │
 │ drives the joint serving loop │          │  priority, net, queue-delay    │
 └──────────────┬────────────────┘          └───────────────┬───────────────┘
                │   ▲                                        │
                │   │  CALIBRATION BRIDGE (parameters, not rows)
                │   └────────────────────────────────────────┘
                │   • best-effort ratio ← v2026 online/offline inference share
                │   • multi-model weights ← v2026 model_type mix
                │   • util sanity-check ← v2026 avg_gpu_sm_util by class
                ▼
        joint serving optimizer test bed (the thing that compounds)
```

- **ATTACH (real, per-record)** within a plane: tokens↔request (Azure), pod↔server
  ↔network by `server_id`+hour (v2026). Lossless.
- **CALIBRATE (parameters, cross-plane)**: v2026 *distributions* set the serving
  plane's scalar parameters (class ratio, model mix, util target). **Never a
  per-record join across planes** — different units (request vs pod) and timescales
  (second vs hour) make that meaningless.
- **ALIGN (exogenous)**: energy price by hour-of-day.
- **MANIFEST**: every signal carries its source + fidelity tier; nothing is laundered.

"Unified" here means *coupled by a shared parameter/fidelity layer*, the way a model
is unified — not concatenated into one table.

## 5. "Just Azure + v2026 + electricity, don't add more models" — mostly right, one correction

Your instinct is **80 % correct and the rule behind it is subtly wrong**, so it's
worth stating precisely:

- **What muddies quality is JOINS, not COUNT.** Quality degrades when you *merge
  mismatched sources row-by-row* (fake correlations, time-base collisions). It does
  **not** degrade when each extra dataset calibrates **one orthogonal parameter**
  with its own fidelity tag and no per-record join. A dataset used to set a single
  scalar (with a caveat) adds ~zero muddying risk.
- **So the right rule is:** *one dataset → one orthogonal signal → fidelity-tagged →
  never per-record-joined across mismatched sources.* Under that rule, more sources
  = strictly more coverage, not more mud.
- **The minimal strong core is exactly yours:** Azure (serving spine) + v2026 (fleet
  plane + calibration) + real electricity/ISO price (cost). That is ~90 % of the
  *operational* value, cleanly.
- **The one addition I'd make: Mooncake.** Not "another model" — it's the **only**
  real source for a *critical, request-level* signal the core lacks: KV prefix-reuse.
  It's used for exactly one orthogonal purpose (KV hit-rate calibration / routing),
  so it adds coverage without mud. Skipping it leaves the KV lever ungrounded.
- **Everything else (Zeus power, M100 thermal, Chakra collective): add only if you
  need that lever, only as single-purpose calibration with the HW caveat.** Don't
  add speculatively — that's where "too many datasets" *would* start to cost you in
  provenance-tracking overhead (not data quality per se).

Net: **Azure + v2026 + electricity + Mooncake** is the right unified core. Four
sources, four orthogonal roles, zero cross-plane row joins.

## 6. The 8 "still-missing" proprietary signals — honest map

Mapping ChatGPT's list against the assembled core (and where each actually lives):

| missing signal | status with our core | where it could come from |
|---|---|---|
| 1. **User intent** (tier, SLA, willingness-to-wait, deadline, retry budget) | ❌ absent | `priority_class` (HP/LP) is a *crude* proxy; the rest is **pilot-only** |
| 2. **Model metadata** (which model, KV size, TP degree) | ⚠️ partial | Azure `model`, v2026 `model_type_public` (coarse), Mooncake prefixes; exact model/TP/KV-size = pilot |
| 3. **Placement constraints** (region, NVLink/IB, anti-affinity) | ⚠️ partial | v2026 `asw_id`+`gpu_spec` let you *infer* topology constraints; the *policy* is pilot |
| 4. **Hardware health** (ECC, throttle, fan, degraded) | ❌ absent | the thermal blind spot — **pilot-only** (no public source) |
| 5. **KV-cache state** | ⚠️ partial | **Mooncake** real prefix-hit; live memory pressure = simulator |
| 6. **Operator actions** (why migrate/reject/override) | ❌ absent | traces record *what*, not *why* — **pilot-only** |
| 7. **Cost** (electricity, PUE, depreciation) | ⚠️ partial | **real ISO/EIA price** + a *modeled* PUE/depreciation overlay (the "separate cost model") |
| 8. **Forecasts at decision time** (demand, price, weather) | ✅ constructible | we already build *causal* forecasts from the trace; price/weather feeds are purchasable |

**The pattern:** the assembled core captures the **observable operational state**
(arrivals, utilization, topology, network, priority, cost). What it cannot capture
is the **decision-intent layer** — *user intent, hardware health, and operator "why"*
— which is structurally proprietary and is precisely the **read-only telemetry
pilot's** job. That is not ~6-10 % of bytes; it is a specific, high-importance
*decision-variable* tier that no public trace records. Build the public core now;
the pilot is what turns it from "production-like" into "production."

## 7. Bottom line

Assemble it as **two planes + a calibration bridge**, not one table:
**Azure LLM** (serving spine, per-request) + **cluster-trace-gpu-v2026** (fleet
spine + serving calibration, hourly) + **Mooncake** (KV, request-level) + **real
energy price** (cost). v2026 *does* finally give us real topology + macro network +
queue-delay — for the **fleet** plane. It does **not** become the serving spine,
because hourly pod aggregates physically lack the per-request/second-resolution
signal the serving levers consume. Keep the core tight, couple by calibrated
parameters (never row-joins), fidelity-tag everything, and reserve the
decision-intent tier (intent / health / "why") for the pilot.
