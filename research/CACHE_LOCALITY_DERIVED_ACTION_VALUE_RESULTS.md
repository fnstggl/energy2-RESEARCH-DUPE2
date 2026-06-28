# Cache-Locality Derived Action-Value — Results (PR #106)

Does the per-replica KV/model residency channel (now causal in the serving path) make cache-affinity
routing / prewarm / migration / placement valuable? Two evidence tiers: **controlled fixtures** (does
the physics work?) and a **held-out dt=60 Azure+Mooncake diagnostic** (does it pay on the real load?).

> SIMULATED directional evidence; no calibration tuned to a result; Pareto gate unchanged.

## Tier 1 — controlled fixtures (the physics works) ✅

`tests/test_cache_locality_physics.py` (12 fixtures, all pass) + `world_validation.py` KV checks
(21 PASS / 0 FAIL / 3 SKIPPED). Each benefit flows through service time / TTFT, never a bonus:

- **prefix hit lowers service time** (cold 0.92 → full hit 0.094); **no hit = baseline**.
- **partial hit → partial benefit** (factor strictly between a full hit and 1.0).
- **cache-affinity beats round-robin under high reuse** (more exact hits, lower mean service factor).
- **shortest-queue can beat affinity when the cache is too small** (concentration thrashes → evictions).
- **memory pressure → no free hit** (an evicted prefix returns to a miss).
- **model-affinity avoids switch cold-starts** on a multi-model stream (the prior Alibaba-GenAI channel,
  ported); **inert on a single-model stream** (Azure).
- **no future prefix leakage** (request *i* sees only requests < *i*); **deterministic replay**;
  **clone isolation**; **scoring pass never pollutes the persistent cache**.

The mechanism is real and behaves correctly, including all the ways it can *fail* (no free win).

## Tier 2 — held-out dt=60 (Azure + Mooncake-derived prefixes), 360 periods, 6 h

`scripts/diagnose_cache_locality_dt60.py` → `data/external/mpc_controller/cache_locality_dt60.json`.
Static policies (isolating the physics; fair baseline = the strongest non-residency arm).

| config | gp/$ | SLA | q_p99 | GPU-h | KV hit | prefill saved | gate |
|--|--|--|--|--|--|--|--|
| fair round_robin (no residency) | 65 089 | 0.0190 | 1.49 | 10.28 | – | – | – |
| **fair kv_routing scalar (no residency)** | **69 211** | **0.0124** | 1.21 | 10.04 | – | – | baseline |
| kv_routing + per-replica residency | 65 021 | 0.0207 | 1.69 | 10.24 | **0.999** | 108 560 | −6.1 % · F/F/F |
| … + prewarm (conservative) | 10 275 | 0.0208 | 1.72 | 10.24 | 0.999 | 116 784 | −85 % · F/F/F |
| … + placement (network_aware) | 65 021 | 0.0207 | 1.69 | 10.24 | 0.999 | 108 560 | −6.1 % · F/F/F |
| … + migration (conservative) | 64 830 | 0.0207 | 1.69 | 10.24 | 0.999 | 108 560 | −6.3 % · F/F/F |

**No Pareto-safe win.** The residency channel **fires hard** (99.9 % exact-prefix hit rate, 108 k prefill
tokens saved) yet gp/$ is ~6 % *below* the fair KV-routing scalar with a *worse* SLA, and the gate is
`False` everywhere.

## Why it fires but does not pay — the physical explanation (the honest core)

Three compounding reasons, none of which is "the mechanism is broken":

1. **Cost is capacity-driven, not service-time-driven.** Per-period **GPU-hours are ~10.0–10.3 across
   every config** — they are set by the provisioned replica count for the period, not by how fast each
   request is served. So reuse that makes service *faster* does **not** reduce cost. The channel can
   only convert to gp/$ through **SLA-safe goodput**, i.e. only when service time is the *binding* SLA
   constraint — and at this load the fair baseline already meets the SLA, so faster service banks little
   extra safe goodput.

2. **The offline KV scalar was optimistically uniform.** The fair `kv_routing` baseline applies the
   *fleet-average* prefill saving (`service_factor ≈ 0.7`) to **every** request — **including
   cache-cold first occurrences that physically cannot reuse**. The per-replica residency is more
   realistic: a first-occurrence request pays **full** prefill (factor ≈ 0.92), only a returning prefix
   gets the deep saving (≈ 0.1). That realism produces a **heavier service tail** (q_p99 1.69 vs 1.21)
   → slightly **worse SLA** (0.0207 vs 0.0124). In other words, part of the old KV-routing "win" was an
   **artifact of the optimistic uniform scalar** crediting reuse to cold requests; the realistic
   per-replica model removes that artifact. This is a fidelity *gain* even though it lowers the headline.

3. **prewarm/migration have no cost channel to pay them back here.** Because cost is capacity-driven (1),
   migration's preserved-KV and prewarm's model-warmth cannot reduce GPU-hours; prewarm on this **light**
   load warms a forecast-sized pool that sits mostly idle → **−85 % gp/$ from wasted warm-hold** (the
   "no free prewarm" guard working). Placement is neutral (macro pressure already small).

## What this proves, and the next missing mechanism

- **Proven (safe to claim):** per-replica KV/model residency is now a **causal** serving-time channel
  (prefix hits skip prefill → lower TTFT/service time; model switches add cold-start), persistent across
  periods, moved by migration, with realistic eviction/memory-pressure limits — validated in 12 fixtures
  + 6 KV validation checks. The model-affinity channel that produced the prior Alibaba-GenAI gain is
  ported and proven on a multi-model fixture.
- **Not claimed:** any Pareto-safe gp/$ improvement on Azure+Mooncake at dt=60 (there is none); that the
  per-replica path beats the offline scalar (it does not, for reasons 1–2 above).
- **The next missing production-reality mechanism (named):** **cost / SLA that is sensitive to service
  time.** Today GPU-hours are per-period-capacity-fixed, so a faster request is free-but-worthless. The
  channel will convert to gp/$ only when (a) the period runs in an **SLA-binding regime** (load high
  enough that faster service rescues otherwise-violating requests — the Erlang-C regime where the prior
  GenAI model-affinity win lived), and/or (b) **cost scales with tokens-processed / occupancy** (a
  prefill-saving directly reduces GPU-seconds billed), and/or (c) **prefill/decode disaggregation**
  prices TTFT separately so a TTFT win is a goodput win. These are Phase 8B-A (prefill/decode) +
  a serving-time-sensitive cost model — the recommended next increment. The fair baseline should also
  drop the optimistic uniform scalar (charge cold requests full prefill) so comparisons are honest.

This is the anticipated outcome: **the mechanism is built and proven; the held-out non-win is explained
physically** (capacity-driven cost + an over-optimistic legacy baseline), and the precise next mechanism
is identified.
