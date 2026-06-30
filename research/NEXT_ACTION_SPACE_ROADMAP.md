# Next Action-Space Roadmap — highest-value missing optimizer knobs

Search regret over the *current* action space is ~0 (PR #123 tournament + PR #124 Phase F): `hierarchical_search`
finds the best bundle that exists. So the next marginal gain comes from **expanding the action space**, not
re-searching it. This PR is **research / audit / prioritization only — no implementation, no simulator/benchmark/
reward/planner/baseline change, no new headline numbers.** It ranks candidate knobs by **expected production
value**, not list length. Current space: `ACTION_SPACE_AUDIT.md`. Surveyed systems + papers in
`PRODUCTION_SCHEDULER_PUBLIC_BENCHMARK_RESEARCH.md` plus the sources cited inline.

## Phase B/C — candidate decision variables, evaluated

Each candidate against: real decision · who uses it · production-deployed · published evidence · fits MPC ·
needs new sim · needs telemetry · impl difficulty · runtime cost · interaction · impact (gp/$, latency, util,
energy, op-cost) · confidence. Compressed to the load-bearing columns; the narrative below covers Tier 1/2.

| candidate knob | decision | deployed in prod | fits MPC | new sim? | new telemetry? | impl | genuinely-new freedom vs today | confidence |
|--|--|--|--|--|--|--|--|--|
| **KV-cache precision** (fp8/int8 KV, separate from weights) | per-replica KV dtype | ✅ vLLM (shipped) | ✅ | extend roofline KV-bytes (low) | ❌ | **low** | **YES** — orthogonal to weight `precision_policy`; sets KV bytes → concurrency | high |
| **Heterogeneous GPU-type assignment** (model/phase → H100/A100/L4…) | which GPU class serves which work | ✅ Helix, ThunderServe, AIBrix, neoclouds | ✅ | per-GPU-type roofline (exists) + cost (mod) | ~ (per-type calib ideal) | **med-high** | **YES** — fleet has `gpu_type_mix` but no assignment ACTION | high |
| **Prefill/decode disaggregation (LIVE)** (promote `prefill_decode_policy`) | split capacity into P and D pools | ✅ NVIDIA Dynamo, DistServe, Splitwise | ✅ | **live disaggregated pools** (high) | ❌ | **high** | **YES** — today SIMULATED_ONLY (roofline-analytic only) | high |
| **Parallelism degree (TP/PP)** | tensor/pipeline shard count | ✅ Dynamo SLA Planner profiles it | ✅ | multi-GPU latency model (high) | ~ | high | **YES** — not in space | med |
| **GPU-mem oversubscription / KV-util target** (`gpu_memory_utilization`) | KV cache size → max concurrency | ✅ vLLM | ✅ | KV-capacity↔concurrency model (mod) | ❌ | med | **partial** — `capacity_multiplier` sizes replicas, not per-replica KV depth | med |
| **Energy/price temporal shift (ACTION)** (promote `energy_policy`) | defer deferrable work to cheap/clean hours | ~ research (Green-LLM); repo has deferrable state | ✅ | shift action the sim honours (mod) | ❌ (price in objective) | med | **partial** — `clock` does DVFS; no temporal shift action | med |
| **Explicit batch token budget / max_num_seqs / chunked-prefill budget** | finer continuous-batch control | ✅ vLLM | ✅ | refine batching model (low) | ❌ | low | **mostly captured** by `batching_policy` presets | med |
| **Request hedging / replication** | duplicate for tail | ✅ Ray Serve tail-tolerance | ✅ | replication cost model (mod) | ~ (tail dist) | med | YES (tail) but low gp/$ leverage | low-med |
| **Reservation vs spot/on-demand sourcing** | capacity price tier | ✅ all clouds | ✅ | spot price + eviction model (mod) | ✅ (spot telemetry) | med | YES — but spot was deliberately quarantined (`#9`) | low |
| **Region / cluster spillover** | overflow to another region | ✅ multi-region serving | ✅ | multi-cluster topology (high) | ✅ | high | YES — single-region benchmark today | low |
| **Draft-model selection for spec decode** | which draft model | ✅ vLLM/EAGLE | ✅ | acceptance-vs-draft model (mod) | ~ | med | refines `spec_decode_policy` | low-med |
| **Power cap (explicit)** | per-GPU power limit | ✅ DCGM | ✅ | ❌ (≈ clock) | ❌ | low | **captured** by `clock_policy` DVFS | med |
| **Congestion / network-path routing (LIVE)** (`topology_policy`) | per-collective network path | ~ research | ✅ | network/congestion sim (high) | ✅ | high | **mostly captured** by macro `placement_policy` | low |
| **Storage / checkpoint placement** | weight/ckpt tier | ✅ training infra | ~ | storage sim (high) | ✅ | high | NO — training-side, not serving gp/$ | low |
| **Maintenance scheduling** | when to drain/patch | ✅ ops | ~ | maintenance model (mod) | ✅ | med | NO — ops cadence ≠ per-period serving | low |
| **Carbon / renewable objective** | weight carbon in objective | ~ research | n/a | n/a | ✅ (grid carbon) | — | **objective change, not a knob** (contract: no reward change) | — |

Impact axes (gp/$, latency, util, energy, op-cost) summarized inline per tier below.

## Phase D — ranking (weighted: prod impact · realism · feasibility · benchmark value · telemetry · novelty)

**Tier 1 — highest priority** (large impact · genuinely new freedom · simulatable now · no new telemetry ·
production-shipped):
- **KV-cache precision** — gp/$ ↑ (more concurrency in decode-heavy/long-context), util ↑, op-cost ↓; latency
  ≈/↓; energy ↓. Headline-safe (fp8 KV ≈ lossless). Lowest effort of all Tier-1.
- **Heterogeneous GPU-type assignment** — the **biggest economic lever**: route cheap/decode work to L4/A100,
  hot/prefill to H100 → op-cost ↓↓ at equal SLA. gp/$ ↑↑. This is the lever that most distinguishes Aurelius
  from a single-pool serving scheduler.

**Tier 2 — worth adding**:
- **PD disaggregation (live)** — gp/$ ↑ (phase-matched hardware/parallelism), the modern Dynamo/DistServe lever;
  high sim cost (needs real disaggregated pools).
- **GPU-mem / KV-util target** — util ↑, concurrency ↑; moderate sim.
- **Parallelism degree (TP/PP)** — latency/throughput tradeoff; high sim (multi-GPU latency).
- **Energy/price temporal shift action** — op-cost ↓ on price-volatile windows; the economic-arbitrage
  companion to `clock`; the repo already has deferrable-work + electricity state to build on.

**Tier 3 — interesting, lower expected value**: finer batch token budget (mostly captured); request hedging
(tail, low gp/$ leverage); reservation-vs-spot (quarantined; telemetry-gated); region spillover (single-region
benchmark); draft-model selection (refines spec).

**Tier 4 — probably not worth implementing**: power cap (captured by `clock`); congestion/network-path routing
(no network model; captured by macro placement); storage/checkpoint placement (training-side); maintenance
scheduling (ops cadence); carbon-as-objective (objective change, out of scope for an action-space PR).

## Phase E — interaction analysis (new freedom vs captured)

- **KV-cache precision** — genuinely new: today `precision_policy` quantizes **weights**; KV bytes are the
  *separate, dominant* memory term in long-context decode. Incremental value is real and **orthogonal**.
- **Heterogeneous GPU assignment** — genuinely new: `placement_policy` chooses *racks*, not *GPU classes*; the
  fleet's `gpu_type_mix` is a constant, never a decision. Large incremental value.
- **PD disaggregation** — genuinely new structurally, but **partially overlaps** `batching_policy` +
  `capacity_policy` (both already shape prefill/decode service); incremental value is the *phase-specific
  hardware/parallelism* match, not the split per se.
- **GPU-mem/KV-util** — **partial overlap** with `capacity_multiplier` (both change effective concurrency); the
  new freedom is per-replica KV depth vs replica count. Moderate incremental.
- **Parallelism degree** — new, but **couples tightly** with batching + precision (all set the roofline
  operating point); risk of redundancy unless modelled jointly.
- **Energy shift** — **partial overlap** with `clock` (both economic/energy); the new freedom is *temporal*
  (move work in time) vs `clock`'s *intensity* (move work in power). Real but smaller than Tier 1.
- **Batch token budget / power cap / congestion** — **largely captured** by `batching_policy` / `clock_policy` /
  macro `placement_policy` respectively → low incremental value (the reason they are Tier 3/4).

## Phase F — search-space / tractability analysis

`hierarchical_search` decomposes by control timescale (slow capacity/placement/migration · medium precision/
batching/spec/clock · fast routing/admission/ordering) and stays tractable at ~75 evals/decision. Adding knobs:

| knob | added cardinality | tractability impact | recommended integration |
|--|--|--|--|
| KV-cache precision | ×2–3 | small — folds into the **medium** roofline group (couples with weight precision) | add to medium group; couple with `precision_policy` (joint precision sub-search) |
| heterogeneous GPU assignment | ×(#GPU types) | moderate — a **new slow-timescale** dimension (placement-like) | new **slow** sub-group; regime-gate to multi-type fleets only |
| PD disaggregation (live) | ×3–5 | moderate — interacts with capacity + parallelism | medium/slow; **only activate** when prefill and decode pressures diverge |
| parallelism degree | ×3–4 | **largest** — multiplies the medium group | gate to latency-tight regimes; coordinate-descend within medium |
| GPU-mem/KV-util | ×3 | small | fold into the **slow** capacity group |
| energy shift | ×2–3 | small — interacts with electricity forecast | fast/medium; activate only on price-volatile windows |

**Net:** Tier-1 knobs (KV precision, GPU-type) add **one** new slow dimension + a fold-in to the medium group →
hierarchical stays tractable (regime-gating keeps most off in any given window). Parallelism degree is the one
that most threatens combinatorial blow-up → needs explicit gating + coordinate polish, not a full grid. **No
new search algorithm is needed**; the hierarchical decomposition already supports adding groups and
regime-gating. Recommend: **regime-activate** every new knob (a knob that cannot help in the current workload
regime is frozen out of the candidate set, with a recorded reason — the existing `physics_guided_candidates`
pattern) so the *effective* search width stays ~constant.

## Phase G — final recommendation (the 10 questions)

1. **Which knob next?** **KV-cache precision** (fp8/int8 KV, separate from weight precision).
2. **Why?** Best value/effort ratio: genuinely-new, **orthogonal** decision freedom; lowest sim cost (extend the
   roofline KV-bytes term already used by `precision_policy`); **no new telemetry**; production-shipped (vLLM,
   ~15% throughput from 2× KV space); headline-safe (fp8 KV ≈ lossless); folds into the existing medium search
   group without widening it. It is the lowest-risk way to start expanding the space.
3. **Three knobs together for the largest production benefit?** **KV-cache precision + heterogeneous GPU-type
   assignment + prefill/decode disaggregation (live).** These are exactly the three modern serving levers
   (Dynamo / DistServe / AIBrix) Aurelius lacks; together they unlock joint **precision × hardware × phase**
   cost arbitrage — the largest remaining op-cost/$ headroom, and the levers a production scheduler does *not*
   jointly economically optimize.
4. **Negligible impact?** Power cap (captured by `clock`), congestion/network-path routing (captured by macro
   placement; no network model), maintenance scheduling, storage/checkpoint placement.
5. **Require production telemetry first?** Reservation-vs-spot (spot price/eviction), per-link congestion
   (network telemetry), request hedging (real tail distribution), and ideally heterogeneous-GPU magnitudes
   (per-type calibration — roofline can bootstrap, but pilot data tightens it).
6. **Materially strengthen the competitive advantage?** **Heterogeneous GPU-type assignment** and the
   **energy/price temporal-shift action** — these are *economic-arbitrage* levers (hardware-cost and
   price-in-time) that distinguish Aurelius from Dynamo/vLLM, which optimize SLO/throughput but not dollars.
7. **Improve benchmark but unlikely to matter in production?** Explicit batch token budget / finer batching
   granularity — the benchmark's batching presets already capture most of the effect; finer control mostly
   moves the simulated number, not real deployments.
8. **Improve production but may not move the benchmark?** Request hedging/replication (helps p99 tail, but the
   gp/$ benchmark scores SLA-violation *rate* + throughput, not tail) and region/cluster spillover
   (single-region benchmark) — both real in production, near-invisible to the current benchmark.
9. **Never implement (complexity without value)?** Storage/checkpoint placement (training-side, not serving
   gp/$), maintenance scheduling (ops cadence ≠ per-period serving), per-link NVLink/NVSwitch congestion (no
   model + low marginal over macro placement), and carbon-as-a-knob (it is an *objective* change — out of scope
   for an action-space PR, and would violate the no-reward-change contract).
10. **Roadmap for the next five optimizer PRs:**
    - **PR-1 — KV-cache precision** (Tier 1, low sim cost; folds into medium group). *Start here.*
    - **PR-2 — Heterogeneous GPU-type assignment** (Tier 1; the big economic lever; new slow sub-group,
      regime-gated to multi-type fleets).
    - **PR-3 — Prefill/decode disaggregation (live)** (Tier 2; promote `prefill_decode_policy` from
      SIMULATED_ONLY; needs real disaggregated pools — the largest sim investment).
    - **PR-4 — GPU-mem/KV-util target + parallelism degree** (Tier 2; capacity-side; parallelism gated to
      latency-tight regimes to protect tractability).
    - **PR-5 — Energy/price temporal-shift action** (Tier 2; promote `energy_policy` from PLANNED; build on the
      existing deferrable-work + electricity state; economic-arbitrage companion to `clock`).

**Sequencing rationale:** ascend the **value/effort** frontier — start with the orthogonal, low-sim-cost,
telemetry-free Tier-1 knob (KV precision), then the high-value economic lever (GPU-type), then the
higher-sim-cost structural levers. Every new knob must be **regime-gated** (frozen out where it cannot help,
with a recorded reason) so `hierarchical_search` stays tractable and search regret stays ~0. **Do not** add
knobs that duplicate existing freedom (power cap, congestion, finer batching) or that change the objective
(carbon) — they raise complexity and search width without proportional gp/$.

## Honest caveats

- This is a **prioritization**, not a measurement: expected impacts are reasoned from the cited systems/papers +
  the repo's roofline/world models, not benchmarked (no implementation in this PR).
- The Tier-1 picks are biased toward **simulatable-now, telemetry-free** knobs on purpose — the highest-value
  knob in the abstract (heterogeneous GPU, or PD disaggregation) may have a larger ceiling, but KV-cache
  precision is the lowest-risk first step that adds real freedom. The roadmap front-loads value while
  deferring the high-sim-cost levers.
- Magnitudes for any of these, once implemented, will be **SIMULATOR_INFERENCE** until pilot telemetry — the
  same fidelity caveat that governs the current roofline cluster (`WORLD_MODEL_ROBUSTNESS_AUDIT.md`).
