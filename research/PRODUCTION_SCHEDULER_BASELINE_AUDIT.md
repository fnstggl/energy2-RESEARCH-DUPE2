# Production Scheduler Baseline Audit (Phase A)

Before building a single canonical `production_scheduler` baseline — the realistic, modern GPU-fleet scheduler
that future headlines compare against — this audits what baselines exist, what each means, what is reusable,
and (critically) which knobs belong only to Aurelius MPC and must **not** leak into the baseline.
`production_scheduler` is a **benchmark-layer baseline only**: a deterministic heuristic `decide_fn` reacting
to current/recent observable state — never a planner mode, never sharing MPC-search / economic / oracle /
hierarchical code. Read-only audit; no behaviour change. Evidence is `file:line`.

## 1. What baselines exist today?

Two registries, plus dynamically-defined serving baselines:

| baseline | where | policy (capacity / ordering / admission + extras) |
|--|--|--|
| `fifo` (a.k.a. `fifo_weak`) | `training.py:28`, `diagnose_mpc_attribution.py:125` | `reactive_lag1 / fifo / off` — naive Erlang-C + FCFS |
| `sla_aware` = `SLA_AWARE_FALLBACK` | `controller.py:45` | `backlog_aware / abs_conformal / off` — backlog autoscale + SRPT-conformal |
| `greedy` | `diagnose_mpc_attribution.py:127` | `backlog_aware / fifo / off` + `routing=kv_aware, batching=aggressive` |
| `aurelius_canonical` + variants | `training.py:31-64` | sla_aware + `admission=class_aware`, ± kv-routing / batching / capacity / prewarm |
| `oracle` | `controller.py` (`planning_oracle_records`) | MPC planning against the **exact future** — diagnostic, non-deployable |
| legacy energy/scheduling baselines | `aurelius/backtesting/baselines.py:40-244` | fifo / peak_blind_asap / latency_first / round_robin … (NOT LLM serving) |

`WEAK_BASELINES = {"fifo_weak"}` (`training.py:27`); `claim_gate` (`training.py:244`) headlines only if Aurelius
beats the **strongest non-weak** baseline on SLA-safe gp/$ AND SLA is not worse.

**`vllm_only` and `topology_aware` do NOT exist** — they are referenced in `PRODUCTION_BASELINE_LADDER.md:15`
as "not separately represented." This PR adds them as simple ladder rungs alongside `production_scheduler`.

## 2. What does `fifo` mean?

`{capacity: reactive_lag1, ordering: fifo, admission: off}` — Erlang-C autoscaling on the **last tick's**
arrivals + first-come-first-served dispatch + no admission control. The naive reactive baseline; the only one
the gate treats as **weak**.

## 3. What does `sla_aware` mean?

`SLA_AWARE_FALLBACK = {capacity: backlog_aware, ordering: abs_conformal, admission: off}` (`controller.py:45`).
`backlog_aware` = Erlang-C + live-backlog autoscaling; `abs_conformal` = shortest-remaining-time scheduling
with a conformal SLA guard. A **research-grade latency scheduler** — strong, but it uses **none** of the
serving-stack levers (batching, KV-routing, placement, warm pool) a real deployment has. It is the current
claim-gate baseline (the *hardest* honest bar, not the production bar — `PRODUCTION_BASELINE_LADDER.md`).

## 4. What does `vllm_only` mean (if present)? — ABSENT → to be added

Not present. A faithful **vLLM default** = continuous batching + roughly-FIFO order + reactive autoscale,
**no** SLA scheduler, no KV-aware routing, no topology placement. This PR defines it as
`{capacity: backlog_aware, ordering: fifo, admission: off, batching: balanced, routing: round_robin}` — the
serving-stack default a stock vLLM/TGI deployment runs (rung 2 of the ladder).

## 5. What does `topology_aware` mean (if present)? — ABSENT → to be added

Not present. A placement-aware scheduler with rack locality but no SLA scheduler. This PR defines it as
`{capacity: backlog_aware, ordering: fifo, admission: off, placement: rack_local}` (rung 3) — isolating the
topology lever from the SLA lever.

## 6. What does `oracle` mean?

The MPC controller planning against `planning_oracle_records` (the **exact** realized future workload). It
beats every deployable arm and is **diagnostic only** — never a headline, never a deployable comparison.

## 7. Which existing code pieces can be reused for `production_scheduler`?

All through the **same** causal `decide_fn(frames[:p]) → action_dict` path `run_period_episode` already
merges (`controller.py:636-643`) — so `production_scheduler` reuses the unchanged reward path:

- **Continuous batching:** `batching_policy ∈ {conservative, balanced, aggressive}` → `BATCHING_MODELS`
  `(concurrency, service_factor)` (`actions.py:37`) → `run_unified_replay(batch_concurrency, batch_service_factor)`.
  `balanced=(2.0, 1.15)` ≈ a vLLM continuous-batch default; `aggressive=(4.0, 1.5)` for throughput.
- **SLA-aware ordering / admission:** `ordering=abs_conformal` (SRPT + conformal), `admission=class_aware`
  (defer best-effort under load — `unified_replay.py`).
- **Backlog autoscaling + warm pool:** `capacity=backlog_aware` + `capacity_multiplier` + `prewarm_policy`
  (`world_simulator.py`: `COLD_START_S=30`, `WARM_IDLE_TIMEOUT_S=300`, causal lag-1 warm sizing).
- **KV-aware routing:** `routing=kv_aware` → fleet KV prefix-hit service factor (`kv_service_factor_by_routing`,
  `controller.py:644-645`); Mooncake residency via `kv_state_pool`.
- **Topology placement:** `placement=rack_local` / `network_aware` → `world_simulator` macro topology discount
  (`TOPOLOGY_MAX_DISCOUNT=0.08`).
- **Recent-load signals:** the causal `frames` carry `arrival_rate`, `output_token_mean`, `interarrival_cv`,
  `n_requests` (`forecasting.py:40,94`) — production-plausible observables for reactive heuristics.

## 8. Which pieces would be unrealistic to include?

- **PagedAttention block tracking / chunked prefill** — not modelled in the repo (only a KV-prefix-hit
  reduction of prefill). Do **not** invent it; the continuous-batching concurrency model is the safe vLLM
  approximation.
- **Per-link / NVLink / NVSwitch / PFC-ECN network behaviour** — absent (only macro rack/network topology).
  `production_scheduler` uses `rack_local` and must **not** invent network behaviour.
- **Free capacity / instant scale / free shedding** — every replica costs warm-hold GPU-hours; admission
  deferral is accounted. No free lunch.

## 9. Which knobs belong only to Aurelius MPC and must NOT leak into the baseline?

`production_scheduler` runs the **deployed model as-is** and a **reactive** scheduler — it must NOT use the
economic-optimisation levers that are Aurelius's edge:

- **`precision_policy` (fp8 / int4)** — precision *arbitrage* for cost is an economic optimisation; a
  production scheduler runs the deployed precision → **bf16 only**.
- **`clock_policy` (DVFS low/high)** — energy/clock arbitrage (the N2 mechanism) → **base only**.
- **`migration_policy`** — MPC-planned live consolidation → **off**.
- **`spec_decode_policy`** — an optimisation knob → **off** (vanilla).
- **Future electricity prices / forecast-priced clock** — an economic signal → never used.
- **Oracle future workload, full action search, the global economic objective, hierarchical/beam search** —
  the clarification: production_scheduler shares **none** of this and is never a planner mode.

## 10. What would a real Lambda/Crusoe/CoreWeave-style scheduler likely have?

Continuous batching (vLLM/TGI), KV-cache-aware routing (prefix reuse), backlog-reactive autoscaling with a
warm pool + cooldown, rack/locality-aware placement, priority/SLA classes with admission control, and at most
a *simple* reactive arrival forecast — **but not** model-precision arbitrage, DVFS energy arbitrage,
MPC-planned migration, future-price optimisation, or a global economic search. So:

```
production_scheduler = { ordering: abs_conformal (SLA/deadline-aware),
                         capacity: backlog_aware + 1.25× headroom under pressure (autoscale),
                         admission: class_aware under pressure,
                         batching: balanced→aggressive (continuous batching ALWAYS on, load-shaped),
                         routing: kv_aware, placement: rack_local,
                         prewarm: off (warm pool via backlog_aware idle timeout, not an eager pool),
                         precision: bf16, clock: base, migration: off, spec: off }
```

This is **stronger than `sla_aware`** (it adds the serving-stack levers a real vLLM deployment has) and is the
**honest production bar** Aurelius must beat — without the economic arbitrage that is Aurelius's whole point.
It is deterministic, causal, and lives in the evaluation layer only.

**Two realism corrections found during integration (both make the bar *stronger*, the honest direction; both
documented in `PRODUCTION_SCHEDULER_BASELINE_RESULTS.md`):** (1) **continuous batching is always on** — an
earlier draft shrank the batch under burst, but turning off continuous batching is something no real vLLM/TGI
deployment does (it only raises cost/req); bursts are handled by admission + headroom instead. (2) **no eager
prewarm pool** — the warm pool is provided by `backlog_aware`'s idle timeout (`WARM_IDLE_TIMEOUT_S=300`); an
*eager* prewarm pool spun replicas up ahead of demand and, at backtest workload scale, held idle GPU-hours that
dwarfed the served work with **zero cold starts avoided** — pure cost a cost-conscious operator would not pay.
Aurelius MAY still prewarm (its optimisation to make when it pays); the production baseline does not gamble idle
capacity. Neither correction was tuned to a benchmark — each is a serving-realism fix.
