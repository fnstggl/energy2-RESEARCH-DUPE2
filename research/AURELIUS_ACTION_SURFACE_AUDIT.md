# Aurelius Action-Surface Audit (Phase 1)

**Purpose.** Before building a canonical action-surface architecture, classify every
infrastructure control action Aurelius could take by what the code *actually does today*,
with file-level evidence. The product goal is an MPC economic controller that *understands*
all major action surfaces but **only optimizes actions that exist and the simulator can
evaluate** — never fake knobs.

**Method.** Read the serving simulator (`aurelius/optimizer/unified_replay.py`,
`run_unified_replay`), the controller (`aurelius/environment/controller.py`), the optimizer
adapter (`aurelius/environment/optimizer_adapter.py`), the KV model/router
(`aurelius/environment/kv_cache.py`), the cost model (`aurelius/environment/cost_model.py`),
and the roadmap (`research/AURELIUS_FORECASTING_AND_MPC_CONTROLLER.md`). The decisive test
for **CONNECTED** is: *does the action parameter reach `run_unified_replay` (or the cost
model) and change the scored KPI / reward?*

## The hinge fact

`run_unified_replay(jobs, *, tick_seconds, sla_s, capacity, ordering, admission, mcs_gate,
warmup_c, best_effort_sla_s)` (`unified_replay.py:326`) takes **exactly three action
levers** — `capacity`, `ordering`, `admission`. Server selection inside the loop is pure
round-robin (`_free_sid()`, `unified_replay.py:378` — `for s in range(st.c): if
servers.get(s) is None: return s`). So **anything about routing, batching, clock, precision,
migration, placement, or speculation cannot change the scored outcome today** — there is no
parameter for it and no branch that reads it.

`enumerate_actions()` (`controller.py:42`) = `CAPACITY(3) × ORDERING(2) × ADMISSION(2)` =
**12** connected bundles.

### ⚠️ Fake-knob finding (the exact risk to fix)

`optimizer_adapter.ACTION_SPACE` (`optimizer_adapter.py:35`) advertises
`"kv_routing": [True, False]` — but **nothing consumes it**: a repo-wide grep finds
`kv_routing` only at its own declaration. Toggling it does **not** change any KPI. It is a
non-actuated knob. The registry built in this PR must **exclude such knobs from optimization**
(enumerate CONNECTED only). `"cost_scenario": ["owned","leased"]` is consumed (it selects the
cost basis → reward denominator) but is a **cost-accounting toggle, not an infrastructure
control action**.

## Status legend

- **CONNECTED** — implemented and changes the simulator KPI / reward.
- **SIMULATED_ONLY** — a faithful simulator/model exists, but it is **not wired into the
  reward path** (the controller's scored outcome does not depend on it).
- **PLANNED** — desired (often in the N1–N8 roadmap), not simulatable today.
- **REQUIRES_PILOT_TELEMETRY** — needs operator data that is structurally ABSENT.
- **REJECTED** — out of product scope.

---

## Per-action classification

### 1. Admission / defer — **CONNECTED**
- `AdmissionController.decide_deferred()` (`unified_replay.py:198`) splits deferred
  best-effort load at tick boundaries; admitted → `wait_queue`, deferred → `defer_buffer`
  (`unified_replay.py:433`). Param `admission ∈ {off, class_aware}` (`controller.py:38`).
- **Reward effect:** changes the live backlog the capacity controller sees → GPU-hours, SLA
  violations, `n_deferred` (`UnifiedKPI`). **State needed:** live LC backlog, best-effort
  fraction (have). **Fair baseline:** `admission=off`. **Safe for MPC: yes (already in).**

### 2. Ordering / scheduling — **CONNECTED**
- `_dispatch_index(st, ordering)` (`unified_replay.py:239`, called at `:397`) picks the next
  job: strict class priority then `fifo` vs `abs_conformal` (SRPT on predicted tokens aged by
  wait). Param `ordering ∈ {fifo, abs_conformal}` (`controller.py:37`).
- **Reward effect:** drains the queue faster/slower → SLA violations + queue tail.
  **Forecast needed:** predicted tokens (causal running-median prior exists). **Fair
  baseline:** `fifo`. **Safe for MPC: yes (already in).**

### 3. Routing (request → replica) — **SIMULATED_ONLY**
- A real router exists — `KVAwareRouter.route()` (`kv_cache.py`) scores servers by reusable
  prefix minus queue/mem/net penalties — but `run_unified_replay` never calls it; dispatch is
  round-robin `_free_sid()`. So routing **does not affect the reward** today.
- **Missing for CONNECTED:** pass a per-request server choice into the dispatch loop and let
  it change which cache/queue the job lands on. **Safe for MPC: no (not in reward path).**

### 4. KV-aware routing — **SIMULATED_ONLY**
- `StatefulKVCache` + `KVModel` (`kv_cache.py`) faithfully replay the Mooncake reuse trace and
  yield a per-request service discount; this discount is applied uniformly as
  `kv_service_factor` (`controller.py:100`, `training.py`), **not** as a per-request *routing*
  decision. The `kv_routing` knob (above) is inert.
- **Missing for CONNECTED:** one `StatefulKVCache` per server inside the serving loop + route
  to the best-prefix cache (roadmap **N4**). **Safe for MPC: behind an explicit flag only.**

### 5. Migration — **PLANNED**
- No migration state, cost, or simulator branch anywhere in `unified_replay.py` /
  `environment/`. Listed as a not-connected lever in the roadmap
  (`AURELIUS_FORECASTING_AND_MPC_CONTROLLER.md`). **Missing:** live-move cost model + replica
  assignment state. **Safe for MPC: no.**

### 6. Batching / batch composition — **PLANNED**
- The loop dispatches **one request per free server** (`_start`, `unified_replay.py:384`);
  there is no batch-size or batch-composition lever and no continuous-batching/roofline model.
  Roadmap **N1** ("batch-composition under a roofline-aware law") is unimplemented.
- **Missing:** an explicit batch model (throughput vs latency vs memory) + a composition
  action. **Safe for MPC: no.**

### 7. Replica count / capacity adjustment — **CONNECTED**
- `CapacityController.decide()` (`unified_replay.py:164`) sizes the window's server count:
  `reactive_lag1` (Erlang-C on last tick), `forecasted_mcs` (EWMA), `backlog_aware`
  (forecast + live LC-backlog correction). Param `capacity` (`controller.py:36`); drives
  `gpu_hours` → cost (`unified_replay.py:498`).
- **Forecast needed:** arrival rate + service time (have, and now beat naive on the week).
  **Fair baseline:** `reactive_lag1`. **Safe for MPC: yes (already in).**

### 8. Prewarming / pre-positioning — **PLANNED**
- No warm-replica state or anticipatory-scaling action; cold-start tax is not modeled.
  Roadmap **N7**. **Missing:** warm-pool state + cold-start cost term + the (existing)
  arrival forecast as the trigger. **Safe for MPC: no.**

### 9. KV placement / eviction — **PLANNED** (cache is SIMULATED state, not an action)
- `StatefulKVCache` evicts LRU under a memory budget shrunk by fleet memory pressure
  (`kv_cache.py`). This is **simulated state**, deterministic from the trace — there is **no
  action** to choose eviction policy or block placement. **Missing:** an eviction/placement
  policy lever + a counterfactual sim. **Safe for MPC: no.**

### 10. Placement / packing (job → node/rack) — **PLANNED → REQUIRES_PILOT_TELEMETRY for fidelity**
- v2026 topology (asw/rack locality, fragmentation) exists in the **fleet plane** as anchored
  marginals, but the serving loop has no node/rack placement decision; `run_unified_replay`
  servers are homogeneous and topology-free. The deeper validation (live residency, hardware
  health) is **ABSENT** (FidelityManifest). **Missing:** a topology-aware placement simulator;
  pilot telemetry to validate. **Safe for MPC: no.**

### 11. Clock / DVFS / power shaping — **PLANNED**
- Service time is fixed (`_service_time_s = TTFT_BASE_S + tokens·TPOT_S`,
  `srtf_serving_backtest.py`); no clock/frequency parameter, no power-vs-performance curve in
  the serving plane. Roadmap **N2**. (A separate `simulation/cluster/energy_model.py` exists
  for the *training/cluster* path — not this serving environment.) **Safe for MPC: no.**

### 12. Precision / model routing — **PLANNED**
- No quality/difficulty proxy, quality constraint, or per-precision service/quality model;
  service time is precision-agnostic. Roadmap **N5**. **Safe for MPC: no.**

### 13. Speculative decoding control — **PLANNED**
- No speculative-execution branch, draft-model overhead, or roofline (memory-bound vs
  compute-bound) indicator. Roadmap (N7 family). **Safe for MPC: no.**

### 14. Energy / price-aware scheduling — **PLANNED action; price is CONNECTED to the reward**
- Electricity price **is** wired into the objective: `CostModel.operator_cost(...,
  energy_price_per_kwh=...)` (`cost_model.py`) and the controller scores candidates with it
  (`controller.py`). But there is **no price-shifting *action*** (e.g. defer-to-cheap-hour,
  clock-down-when-expensive) — price only re-ranks the *existing* 12 connected bundles.
  **Missing:** a temporal-shift / power-shape action that the simulator honors. **Safe for
  MPC: no (as an action); the objective is already price-aware.**

### 15. Network / topology-aware routing — **SIMULATED_ONLY**
- `KVAwareRouter` scores a `net_penalty` per server (`kv_cache.py`), but the router is not in
  the serving loop and `run_unified_replay` has no network model. **Missing:** same wiring as
  #3/#4. **Safe for MPC: no.**

### 16. Other optimizer actions — **none beyond the above; tenant arbitrage REJECTED**
- `ACTION_SPACE` is exhaustive (capacity, ordering, admission, the inert `kv_routing`, the
  accounting `cost_scenario`). No other control lever exists in `optimizer/` or `environment/`.
- **REJECTED:** tenant-side spot / reserved / on-demand **arbitrage** — `cost_model.py:13`
  ("no spot/on-demand/reserved arbitrage or cloud instance-billing optimization here"); the
  serving cost denominator is pure on-demand `Σc[t]·tick_hr·GPU_HOUR_USD` with "no spot, no
  oracle" (`unified_replay.py:43`). This is an operator product, not a cloud-tenant arbitrage.

---

## Summary

| # | action | status | in reward path today? | first step to CONNECTED |
|---|---|---|---|---|
| 1 | admission / defer | **CONNECTED** | yes | — |
| 2 | ordering / scheduling | **CONNECTED** | yes | — |
| 7 | replica / capacity | **CONNECTED** | yes | — |
| 3 | routing (req→replica) | SIMULATED_ONLY | no | call a router in the dispatch loop |
| 4 | KV-aware routing | SIMULATED_ONLY | no | per-server StatefulKVCache + route (N4) |
| 15 | network/topology routing | SIMULATED_ONLY | no | net_penalty into dispatch |
| 6 | batching / composition | PLANNED | no | roofline batch model (N1) |
| 9 | KV placement / eviction | PLANNED | no (sim state only) | eviction-policy action + counterfactual |
| 8 | prewarming | PLANNED | no | warm-pool state + cold-start tax (N7) |
| 11 | clock / DVFS | PLANNED | no | power-vs-perf curve + clock action (N2) |
| 12 | precision / model routing | PLANNED | no | quality model per precision (N5) |
| 13 | speculative decoding | PLANNED | no | roofline + draft-overhead model |
| 14 | energy/price shifting | PLANNED (price→reward CONNECTED) | objective only | temporal-shift action the sim honors |
| 5 | migration | PLANNED | no | live-move cost + replica state |
| 10 | placement / packing | PLANNED → REQUIRES_PILOT_TELEMETRY | no | topology placement sim; pilot validation |
| — | tenant spot/arbitrage | **REJECTED** | n/a | out of scope (operator product) |

**3 CONNECTED, 3 SIMULATED_ONLY, 9 PLANNED, (placement needs pilot telemetry for fidelity),
1 REJECTED.** The MPC controller should optimize the 3 CONNECTED by default, allow the 3
SIMULATED_ONLY behind an explicit flag, never optimize PLANNED, and the registry must drop
the inert `kv_routing` knob from the optimized set. Phases 2–4 build that; Phase 5 gives each
PLANNED action a concrete path to CONNECTED.
