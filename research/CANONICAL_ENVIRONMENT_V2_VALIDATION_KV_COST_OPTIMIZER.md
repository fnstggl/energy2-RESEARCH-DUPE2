# Canonical Environment V2 — Validation, KV, Cost, Optimizer Integration

This documents the four-phase deepening of the `CanonicalMultiPlaneEnvironment`
(PR #93 built the first-principles scaffold; the v2026 ingestion that feeds it is
merged in #94). It is a **production-LIKE public-trace environment**, never
production telemetry. Every signal is fidelity-tagged; the honesty gate
(`is_production_grade`) is structurally `False` until a pilot supplies the ABSENT
proprietary signals.

The four planes stay **strictly separate** — only *state variables* and *calibrated
distributions* cross planes; raw rows are **never joined** across Azure / Mooncake /
v2026 / electricity.

---

## 1. Data tiers — what is what

| signal | tier | meaning |
|---|---|---|
| v2026 fleet marginals — GPU util/mem, priority/GPU-type mix, queue/ready delay, GPU request, network rx/tx | **FULL_TRACE_EXACT** | exact count/sum/mean/variance/min/max/category mixes over every one of the 6.57 B pod rows (and server/network/job tables); the env's fleet **distributions** are anchored to these |
| v2026 percentiles (p50/p95/p99 from fixed-bin histograms) | **FULL_TRACE_APPROX** | every row binned exactly; percentile read off documented bins |
| Azure serving spine (arrivals, tokens) | **FULL_TRACE** (held-out time split) | real per-request trace; train fits, holdout validates |
| Mooncake KV prefix reuse (`hash_ids`) | **TRACE_DERIVED / FULL_TRACE** | real block-prefix reuse; FULL_TRACE when the JSONL is present, else SAMPLE_FIXTURE |
| v2026 **topology/scale** — GPU counts, racks, capacity envelope | **SAMPLE_FIXTURE** | the committed per-hour sample slices; v2026 has no per-hour breakdown, so per-hour structure is a fixture |
| electricity price | **TRACE_DERIVED** (regional ISO) → **SAMPLE_FIXTURE** here | live PJM/ERCOT/CAISO pull is **BLOCKED** in this environment (auth flow unresolved — probed 401/302); the committed regional sample is used with an explicit status |
| per-model KV footprint (bytes/token) | **BENCHMARK_DERIVED** | computed from published model architecture (2·layers·kv_heads·head_dim·dtype) |
| KV cache capacity / eviction (LRU) | **INFERRED / HEURISTIC** | memory-budget split + LRU default; the trace doesn't expose the real policy |
| PUE, GPU CapEx/depreciation, average power draw | **INFERRED** | public-list priors; operator telemetry → MEASURED |
| leased GPU-hour rate | **EXTERNAL_OBSERVED** | public external list/contract rate, not operator-measured |
| live KV memory residency, internal operator cost model, operator intent, hardware health | **ABSENT** | structurally proprietary — pilot telemetry only |

No `MOCK` data is used by the environment. `BLOCKED` applies only to the live ISO
electricity pull (sample fallback, status-reported).

---

## 2. Phase 1 — ValidationSuite breadth

`validation_suite.py` + `validators.py` compare each plane's distribution to its
**held-out reference** with KS / Wasserstein-1 / histogram-L1 / percentile-error /
category total-variation, emitting `PASS / WARN / FAIL / SKIPPED`. A check whose
reference is unavailable is **SKIPPED with the exact artifact/path/command** — never
a silent pass. Run it: `python -m scripts.run_canonical_validation`.

Representative result (full v2026 artifacts present, Azure FULL_TRACE):

- **Azure (held-out time split):** token distribution + inter-arrival — PASS/WARN by load.
- **v2026 fleet (anchored → CONSISTENCY):** GPU utilization, GPU memory, queue/ready
  delay, priority mix, GPU-type mix, best-effort (job-type) fraction, network rx/tx,
  placement/fragmentation — **PASS** (the env reproduces the FULL_TRACE_EXACT marginals
  it's calibrated from; this is self-consistency, **not** an independent held-out test,
  and each check says so in its `detail`).
- **v2026 topology (SAMPLE_FIXTURE):** capacity GPU-count + rack/asw locality — **FAIL**,
  honestly labelled "env topology = SAMPLE_FIXTURE" (the committed sample fleet is small
  and not representative of the full-trace topology).
- **v2026 model-type mix + job duration:** **SKIPPED** (the env emits no model-type or
  job-duration signal yet — the required artifact/command is named).
- **Mooncake KV (train vs holdout):** exact-prefix reuse, partial overlap, cache hit
  rate, cold-vs-warm — **PASS**.
- **Electricity:** price sanity band — PASS (SAMPLE_FIXTURE); held-out ISO — **SKIPPED**
  with the auth/endpoint step.

**Overall verdict is capped at `NOT_PRODUCTION_REALISTIC_YET`** whenever any calibrated
param is below TRACE_DERIVED (the cost params are INFERRED) — by design.

---

## 3. Phase 2 — stateful Mooncake KV cache + routing

`kv_cache.py` replaces the fixed prefill discount with a **paged, LRU, stateful** KV
cache fitted by replaying the real Mooncake reuse trace, coupled to the v2026
GPU-memory calibration:

- **Reuse:** exact-prefix vs partial overlap, reuse depth (leading blocks), cold-start
  vs warm — TRACE_DERIVED from `hash_ids`.
- **Memory:** per-model footprint (BENCHMARK_DERIVED), capacity = `(GPU mem −
  weights)·kv_fraction` shrunk by live fleet memory pressure → eviction.
- **Eviction:** LRU (HEURISTIC); eviction rate; hit/miss after eviction.
- **Serving impact:** prefill tokens saved, TTFT/prefill factor, service-time discount.
- **Routing:** `KVAwareRouter` routes to the cache holding the most reusable prefix vs
  queue/memory/topology penalties, with fastest + shortest-queue baselines.

**Causality (proven by test):** a request's KV outcome depends only on blocks admitted
by EARLIER requests — the first-half outcomes are identical with or without later
requests. The env runs with KV **enabled or disabled**; per-hour metrics report hit
rate, prefill tokens saved, KV memory used, evictions, TTFT impact.

**Limitations:** the reuse RATE/depth are real, but *applying* the Mooncake reuse
dynamic to Azure serving is **SIMULATED** (no row-join). LRU + the capacity budget are
assumptions; **live KV residency is ABSENT** until pilot telemetry. KV savings are not
a headline claim.

---

## 4. Phase 3 — operator-side CostModel

`cost_model.py` estimates the economics of **operating** GPU infrastructure:

- **owned** (primary): depreciation of CapEx over a service life (utilization-adjusted)
  + PUE-scaled energy at the regional ISO price + optional network/queue/SLA-penalty.
- **leased / managed**: a contractual $/GPU-hour rate (EXTERNAL_OBSERVED) + the same
  energy/penalty terms.
- **sensitivity**: low/base/high bands per heuristic assumption (PUE, CapEx, electricity,
  service life, power draw, utilization) — the assumptions are made visible, not hidden.

Outputs: `energy_cost`, `depreciation_cost` | `lease_cost`, `network_cost`,
`queue_delay_cost`, `sla_penalty_cost`, `total_operator_cost`,
`cost_per_sla_safe_request`, `cost_per_sla_safe_token`, `goodput_per_dollar`, plus the
sensitivity bands.

**Hard prohibition (enforced by construction):** there is **no** tenant-side
spot/reserved/on-demand arbitrage — only `owned_depreciation` and `leased_contract`
bases exist; no `spot_cost`/`reserved_cost`/`arbitrage` surface exists.

**Limitations:** electricity is TRACE_DERIVED; everything else (PUE, CapEx,
depreciation, power) is INFERRED public-list prior; leased rates EXTERNAL_OBSERVED; the
true internal operator cost model is **ABSENT**.

---

## 5. Phase 4 — AureliusOptimizer integration

`optimizer_adapter.py` feeds the env's per-hour `EnvStep` (state, action, reward,
metrics) into AureliusOptimizer and scores every arm through the optimizer's own
**`ObjectiveLayer`** (SLA-safe goodput/$) — **no parallel optimizer path**; policies
drive the env's existing `policy(observation) → action` hook (the same levers
`unified_replay` executes).

- **State / Action / Reward** are explicit contracts (`EnvState`, `ACTION_SPACE`,
  `reward_from_step`). State is causal (start-of-hour only); reward is SLA-safe
  goodput per operator dollar.
- **Fair backtest** (`fair_backtest`, `python -m scripts.run_fair_backtest`): runs
  fifo_weak (weak reference) + sla_aware + greedy_packing + aurelius_canonical (current
  best config) + a candidate, picks the **strongest NON-weak** fair baseline (never
  silently FIFO), and reports per-arm gp/$, SLA-violation rate, GPU-hours, energy +
  total operator cost, queue-delay p50/p95/p99, KV hit rate, cost per useful unit, and
  the env's validation status.

**Headline gate (honest by construction):** a savings claim is allowed only when the
candidate **beats the fair (non-weak) baseline** AND the held-out Azure/Mooncake
validation passes AND no oracle is used. On a heavy bursty load the candidate does
**not** beat the SLA-aware baseline → `headline_claim_allowed = False`. No future
arrivals / oracle knowledge / tenant-side arbitrage are used anywhere.

---

## 6. Claims — safe vs unsafe

**Safe to say:**
- "We train and evaluate on a canonical multi-plane environment assembled from real
  public traces (Azure serving, Alibaba v2026 fleet, Mooncake KV, regional electricity)."
- "The environment does not fake row-level joins; it calibrates distributions and
  validates against held-out traces (Azure + Mooncake held-out pass)."
- "Fleet distributions are anchored to the FULL_TRACE_EXACT v2026 marginals (6.57 B rows)."
- "It includes a stateful KV cache/routing model, operator-side cost scenarios, and a
  tested adapter into AureliusOptimizer scored on SLA-safe goodput/$."
- "It is production-LIKE public-trace validation, not proprietary production telemetry."

**NOT safe to say (gated off):**
- "Production-grade" — the honesty gate is `False` (ABSENT proprietary tier unfilled).
- Any headline **savings %** — the fair-backtest gate currently blocks it (candidate
  does not beat the fair baseline; results are directional simulator evidence only,
  `docs/RESULTS.md` §8).
- That v2026 topology, model-type, job-duration, or live KV/cost state are validated —
  they are SAMPLE_FIXTURE / SKIPPED / ABSENT and reported as such.

---

## 7. Remaining fidelity gaps (next pilot/work)

- v2026 **topology** (GPU counts, racks) is SAMPLE_FIXTURE — needs a representative
  per-hour fleet slice or a topology artifact.
- v2026 **model-type mix** and **job-duration** distributions are not yet emitted by the
  env (SKIPPED, with the artifact/command named).
- **Live KV residency / eviction policy**, **internal operator cost model**, **operator
  intent**, **hardware health** remain ABSENT — pilot telemetry only.
- **Live ISO electricity** pull is BLOCKED (auth flow); the sample fallback is used.
- The candidate optimizer policy does not yet beat the strongest fair baseline — a real
  win requires a better state-conditioned policy, proven under the fair-backtest gate.
