# Canonical Multi-Plane Environment — audit, composability verdict & plan — 2026-06-26

> Deliverable for "expose ONE canonical production-like training/eval environment
> over separate raw traces." This audits what EXISTS, **verifies (does not assume)
> that the existing modules compose**, maps gaps, and proposes the minimal next PR.

## Part A — Composability verdict (the 5 hard questions, with evidence)

Modules existing ≠ modules composing. Each answer is grounded in the code.

### Q1. Does `calibration.py` support Azure ↔ v2026 calibration? — **No.**
`aurelius/simulation/cluster/calibration.py` is a **static heuristic table**:
`CalibratedParam` values built by `_h(...)` ("engineering guess") and resolved by
name (`serving_value("x")`, `kv_value`, `thermal_value`, `topology_value`). There
is **no** `from_trace`, `fit`, `v2026`, or `azure` anywhere in it — it never
ingests a distribution. My `aurelius/datasets/calibration.py` (the v2026 class-mix
hook) is a **separate file, unwired** to it. **The Azure↔v2026 calibration bridge
does not exist** — it must be built as a new layer that *maps* trace distributions
onto these named params. (Good news: the target — a named, provenance-tagged param
registry — is already there; only the mapping is missing.)

### Q2. Is `kv_cache.py` generic KV behavior or tied to an old benchmark? — **Generic structure, but the key curve is an unfitted prior.**
`kv_cache.py` is deterministic and **model-parameterized** via
`MODEL_KV_PROFILES` (per-model `layers/kv_heads/head_dim/attention_type`, e.g.
`llama3-8b`) + `KV_CACHE_PARAMS` grounded in **vLLM/PagedAttention DOCUMENTED
defaults** (block_size=16, gpu_mem_util=0.9). So the *structure* is generic and
reusable — **not** tied to one benchmark. **But** its own docstring is explicit:
"the prefix hit-rate curve is a **configurable sigmoid prior, NOT a fitted
industry curve**." So KV is a good substrate whose **central parameter
(prefix-reuse) is a guess — which is exactly what Mooncake would fit, and that
wiring does not exist.**

### Q3. Does `topology_model.py` map onto v2026 `asw_id`? — **Partially.**
There IS a `RackLocalityState(rack_id, cross_rack_traffic_frac)` — so v2026's
`asw_id` (access-switch ≈ rack) maps naturally onto **the rack tier**. **But** the
model is a much deeper fabric hierarchy — `NVLinkDomainState`, `NVSwitchState`,
`PCIeFabricState`, `NUMAState`, `SocketLocalityState`, then rack, then
`CrossRegionFabricState`. v2026 supplies **only** the rack/switch level (`asw_id`)
+ macro `network_hourly` rx/tx (→ `InterconnectCongestionState`/`NICCongestionState`).
The **intra-node tiers (NVLink/NVSwitch/PCIe/NUMA) have no v2026 source** and stay
heuristic. So: **asw_id → rack tier (clean); rx/tx → congestion states (clean);
intra-node fabric (unmapped, stays heuristic).** A real mapping point exists, but
it is one tier of a richer model, not a 1:1.

### Q4. Two synchronized clocks (seconds + hours)? — **No. Fundamentally single-timescale each.**
The cluster engine is **hourly**: `SimulatorConfig.tick_duration_hours = 1.0`, and
`engine.tick()` updates aggregate fleet state once per hour (energy price, thermal,
KV, migration, topology, utilization, **queues via an M/M/1 proxy**, cost). The
serving sim (`unified_replay.py`) is **event-driven at seconds** over individual
token requests. They model serving at **different fidelities** (hourly M/M/1 proxy
vs per-request token-level) and **neither nests the other**. There is **no
two-clock orchestration.** *Silver lining:* the timescales align with the data
(v2026 hourly fleet ↔ cluster engine; Azure per-second ↔ serving sim), so the
two-clock *structure* is natural — the orchestrator must nest the seconds-serving
inside each hourly fleet tick and **replace the cluster engine's M/M/1 queue proxy
with the real serving sim**.

### Q5. Exposable behind one `CanonicalMultiPlaneEnvironment` without major debt? — **Yes, but not for free: three bounded seams.**
The planes genuinely exist (serving sim; hourly cluster engine that already does
energy/thermal/KV/topology/util/cost). But binding them has **three real seams**:

1. **Two calibration systems** (cluster's static `CalibratedParam` table ↔ my
   trace-derived `datasets/calibration`). A **CalibrationBridge** must map
   Azure/v2026/Mooncake distributions onto the named params. *New glue.*
2. **Two time models** (hours ↔ seconds) + **redundant serving** (cluster M/M/1
   proxy vs token-level sim). The orchestrator nests seconds-in-hour and uses the
   real sim for serving. *New glue.*
3. **Partial signal mapping** (asw_id→rack only; KV prefix curve unfitted). *Honest
   gaps, not blockers.*

**Verdict:** one API is achievable and the substrate is ~70% there, but it is
**orchestration + bridge work, plus a deliberately-deferred heavy-engine fusion**
— not "the modules already compose." Fusing the full cluster engine as the
FleetPlane is the **large** part and must be fenced out of the first PR.

## Part B — Component audit (exists / partial / missing)

| component | status | where / seam |
|---|---|---|
| **ServingPlane** | ✅ exists | `unified_replay.py` (+ `srtf_serving_backtest`) — token-level, per-second |
| **FleetPlane** | ✅ exists (hourly) | `simulation/cluster/engine.py` `tick()` + `model.py` (`SimCluster`/`SimNode`/`SimGPU`/`SimQueue`) — but static-calibrated + M/M/1 serving |
| **CalibrationBridge** | ❌ missing | two unwired systems (Q1); only the v2026 class-mix is trace-derived |
| **CostModel** | 🟡 partial | `economics.py` (per-GPU-type price, energy, network) — **no PUE, no depreciation, no region/ISO** |
| **FidelityManifest** | 🟡 partial | `simulation/cluster/calibration.py` provenance ladder + my `signal_matrix` tiers — **no single per-environment manifest object** |
| **ValidationSuite** | 🟡 partial | `realism_audit.py` (honesty-gated verdicts) + `scenario_lock` + `multi_seed_stochastic_audit` — **no distribution-matching vs held-out trace stats** |
| **CanonicalMultiPlaneEnvironment** | ❌ missing | the two-clock orchestrator (Q4/Q5) |
| **Mooncake KV loader** | ❌ missing | KV *state* exists; the *trace* to fit the prefix curve does not |

## Part C — The 10 calibration categories: which have a real source TODAY

| # | category | real source today? | note |
|---|---|---|---|
| 1 | GPU utilization | 🟡 v2026 `avg_gpu_sm_util` (download pending) | hook missing; cluster has static targets |
| 2 | GPU memory | 🟡 v2026 `avg_gpu_mem_gib` | feeds KV capacity; not wired |
| 3 | GPU type | ✅ v2026 `gpu_spec_public` + `economics` per-type price | partial map exists |
| 4 | Topology | 🟡 v2026 `asw_id` → rack tier only (Q3) | intra-node stays heuristic |
| 5 | Priority/QoS | ✅ v2026 `priority_class` + my class-mix hook | the one bridge that works |
| 6 | Queue-delay | 🟡 v2026 `schedule_delay_sec`/`ready_delay_sec` | hook missing |
| 7 | Arrival/class | 🟡 v2026 `job_type_public` (online/offline) | the corrected best-effort ratio |
| 8 | Capacity envelope | 🟡 v2026 `gpu_count`/`gpu_request` | hook missing |
| 9 | Placement/frag | 🟡 v2026 pod→server | hook missing |
| 10 | Cost | 🟡 `economics` + price model; **no PUE/depreciation/region** | extend cost model |

**Only #5 (priority/class mix) is actually trace-derived today.** The other nine
are either static heuristics or need the v2026 download + a hook. This is the real
state — not "calibration exists."

## Part D — Implementation plan (sequenced; each PR small + reversible)

1. **PR-1 (this one, minimal): the API + the seams made explicit.**
   `aurelius/environment/`: `FleetState` (the merged hourly state dataclass),
   `FleetPlane` v1 (calibrated FleetState from the v2026 class-mix hook + documented
   defaults — **not** the cluster engine), `CostModel` (extends `economics` with
   **PUE + depreciation + region**), `FidelityManifest` (unifies signal tiers + cost
   assumptions + names the seams), `ServingPlane` (adapter over `unified_replay`),
   `CanonicalMultiPlaneEnvironment` (two-clock `run()`: per-hour FleetState → nested
   per-second serving), plus **one** distribution-match validation seed. Composes
   only what genuinely composes; fences the cluster engine.
2. **PR-2: CalibrationBridge — real v2026 hooks.** Add the 9 missing category hooks
   (util, mem, queue-delay, capacity, placement…) reading the downloaded v2026
   tables into the named-param registry, each fidelity-tagged.
3. **PR-3: ValidationSuite — full distribution matching** (Part E).
4. **PR-4: Mooncake loader → fit the KV prefix-reuse curve** (replaces the sigmoid prior).
5. **PR-5 (large, fenced): fuse the cluster engine as the production FleetPlane**
   (replace its M/M/1 queue proxy with the nested serving sim; swap static calib
   for the bridge). This is the architectural-debt item — explicitly deferred.

## Part E — ValidationSuite spec (must prove sim ≈ held-out real)

For each distribution below, compare simulated output vs a **held-out** slice of
the real trace via a divergence metric (KS statistic / Wasserstein / histogram
L1) against a documented tolerance; emit PASS/WARN/FAIL + the number. Honesty
gate: the suite never returns "production-grade," only "matches-held-out-within-τ."

- request burstiness (inter-arrival CV, peak/mean) — Azure
- token distribution (p50/p95/p99) — Azure
- GPU utilization distribution — v2026 `avg_gpu_sm_util`
- GPU memory distribution — v2026 `avg_gpu_mem_gib`
- priority mix — v2026 `priority_class`
- queue-delay distribution — v2026 `schedule_delay_sec`/`ready_delay_sec`
- topology/placement distribution — v2026 `asw_id` co-location rate
- network-pressure distribution — v2026 `network_hourly` rx/tx
- cost distribution — assembled cost model vs an external sanity band

## Part F — Honest framing (preserved)

This is **not real full production telemetry**. It is a **production-like
multi-plane environment grounded by real public traces**, with every signal
fidelity-tagged. It becomes production-grade only when a **pilot** replaces the
calibrated assumptions with real operator telemetry: **intent, hardware health,
live KV state, migration/rejection reasons, internal cost model** (the ABSENT
decision-intent tier in `signal_matrix.py`). The environment inherits the existing
`is_sandbox=True` gate (`simulation/cluster/engine.py`) and the
`realism_audit.py` cap at `NOT_PRODUCTION_REALISTIC_YET` until a param is MEASURED.
