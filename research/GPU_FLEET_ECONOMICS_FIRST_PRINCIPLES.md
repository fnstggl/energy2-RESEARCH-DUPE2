# GPU Fleet Economics — First-Principles Decomposition & Step-Function Map

> Free first-principles thinking, not a description of the current system. The goal:
> reason from the physics and economics of what a GPU *is* up to where the
> non-obvious, multiplicative ("step-function") gains in economic yield actually
> live — and how each would be tested. Quantified levers are tagged with their
> realistic standalone ceiling and primary source.

---

## 0. The irreducible question

A GPU fleet is a machine that converts three inputs into one output:

```
  CAPITAL (silicon, depreciating ~30–40%/yr)
  ENERGY  (watts in → heat out)            ──►  SLA-compliant useful tokens
  TIME    (wall-clock the asset is alive)
```

"Economic yield" is **useful work per dollar of input**. Everything below is a way
of moving one ratio. There are only four of them, and they multiply:

```
                 tokens          tokens          1              1
  YIELD  =  ───────────── × ───────────── × ───────── × ─────────────
                joule         GPU-second      $/joule      $/GPU-second
              └─ physics ─┘  └─ utilization ┘ └── market / procurement ──┘
```

The central thesis of this whole document:

- **Each factor optimized alone is sublinear and saturating.** Energy arbitrage
  caps at ~20–30% of *energy* cost, which is itself ~15–25% of TCO → a few % of
  TCO. DVFS caps at ~20%. Each serving trick caps at 2–4×. They all hit a wall.
- **The factors are currently optimized by different parties who never talk:**
  firmware/NVML (factor 1), serving frameworks like vLLM (factor 2), schedulers/
  K8s (factor 2), and procurement/FinOps + grid (factors 3–4). The information
  to co-optimize them exists nowhere in one place.
- **Step functions live in the cross-terms** — the product, not the sum. A
  controller that holds all four factors in one objective and exploits their
  *correlations* (frequency × price, precision × difficulty, placement × thermal
  × carbon) captures gains that are structurally invisible to any single-layer
  optimizer. That is the whole opportunity.

---

## 1. First principles: what is actually happening inside the box

Strip an AI workload to physics and three facts dominate everything else.

### Fact 1 — Decode is memory-bound, not compute-bound. This is the master fact.

Autoregressive token generation re-reads the entire model's weights from HBM to
produce **one** token. Arithmetic intensity ≈ **1 FLOP/byte**. A modern GPU needs
~**156 FLOP/byte** (A100) / ~**295 FLOP/byte** (H100) to saturate its compute units.
So single-stream decode runs at **<1% of the GPU's compute capacity**
(roofline survey, arXiv 2402.16363). The expensive ALUs are idle; you are paying
for a Ferrari to sit in traffic reading numbers out of memory.

**Consequence:** "token generation speed *is* HBM bandwidth, full stop." Per-gen
peak FLOPs have grown faster than HBM bandwidth (the *memory wall*) — H100 3.35
TB/s → B200 8 TB/s → Rubin ~13–15 TB/s — so each new generation pushes *more*
work into the memory-bound regime, not less. **The economic frontier is moving
toward bandwidth, not FLOPs.** Most of the levers below are really "how do I stop
paying for idle FLOPs."

### Fact 2 — Real utilization is ~40%, and that's the *good* number.

- Best-in-class **training** lands at **38–55% MFU** (Llama 3 405B: 38–43%;
  MegaScale 175B: 55.2%; PaLM: 46.2%). Historically most systems were <30%.
- **Inference** MBU is ~**60% at batch 1** and *falls* as batch rises.

So **roughly half** of the FLOPs you bought are structurally wasted in the best
deployments, and far more in typical ones. This is the single largest pool of
waste in the entire stack — bigger than energy price, bigger than carbon, bigger
than procurement. **The denominator of factor 2 (GPU-seconds) is where the money
is**, and almost nobody optimizes it at the fleet level because it requires
seeing inside the serving engine *and* across the cluster at once.

### Fact 3 — Power and heat are a control surface, not a fixed cost.

A GPU's power draw is not a constant; it is a *dial* (NVML/DCGM power cap, clock).
The perf/watt curve is convex: the top ~20% of the clock range costs
disproportionate watts for marginal throughput. And aggregate fleet power is
*statistically multiplexed* — 5,000 Google servers never exceeded 72% of
aggregate peak over 6 months (Fan, ISCA 2007). Both facts mean the fleet is
**over-provisioned on power by design**, and that headroom is reclaimable money.

---

## 2. The optimization surfaces, by layer, with standalone ceilings

Organized by the four factors. Each entry: **mechanism → realistic standalone
ceiling → who's done it → where it saturates.** "Ceiling" = honest gain vs a
*good* baseline, not vs a strawman.

### LAYER 1 — Thermodynamic (tokens/joule, $/joule via heat)

| Lever | Mechanism | Standalone ceiling | Source / state |
|---|---|---|---|
| **DVFS / power capping** | Drop clock/power cap; convex perf/watt means big watt savings for small throughput loss | **~20% peak-power cut for <2% throughput** (low-pri); ~7% worst-case | POLCA (ASPLOS'24); Meta 100MW: 960W vs 1200W TDP → **+11% cluster throughput/W** |
| **Phase-aware frequency** | Prefill is compute-bound (wants high clock), decode is memory-bound (clock barely matters → downclock free) | Decode downclock is **near-free** throughput; open problem | POLCA names "phase-aware power mgmt" as *unsolved* |
| **Undervolting** | Lower V at same f; power ∝ V²f | Single-digit % extra perf/W | Vendor-locked; thin public data |
| **Power oversubscription** | Install more GPUs than peak power supports; multiplex diverse load | **+25–30% more servers** same power envelope = avoided capex | POLCA +30% (inference); Google MV-plane +25%, "hundreds of millions" saved |
| **Overclocking the headroom** | Inverse: spend reclaimed headroom to *overclock* hot requests | Risk-gated speedup | SmartOClock (ISCA'24) |
| **Cooling / PUE co-opt** | Hotspot avoidance, rack placement, inlet-temp-aware scheduling reduce *cooling* power (often 10–30% of facility draw) | PUE 1.5→1.1 ≈ **~25% facility-power cut** | Thermal-aware placement is researched; **price/carbon-coupled thermal scheduling is not** |

**Layer-1 saturation:** physics-bounded. The whole layer is maybe **15–25% of
energy cost + a one-time 25–30% capex-density win**. Real, bankable, but bounded.

### LAYER 2 — Silicon utilization (tokens/GPU-second) — *the big pool*

| Lever | Mechanism | Standalone ceiling | Source |
|---|---|---|---|
| **Continuous batching** | Iteration-level scheduling; finished requests exit, new join mid-flight | **up to 23×** vs naive; table stakes now | Orca (OSDI'22) 36.9× vs FasterTransformer; Anyscale 23× |
| **PagedAttention** | Virtual-memory KV blocks; KV waste 60–80% → <4% → bigger batches | **2–4×** throughput | vLLM (SOSP'23) |
| **Chunked prefill** | Piggyback prefill chunks on decode batches, no stalls | **2.6–5.6×** serving capacity | Sarathi-Serve (OSDI'24) |
| **Disaggregated prefill/decode** | Run the two phases on *different* GPUs sized for each → cheap GPUs decode, fast GPUs prefill | **7.4× more requests / 12.6× tighter SLO** (DistServe); **2.35× tput @ same cost+power** (Splitwise) | OSDI'24 / ISCA'24 |
| **Workload co-location** | Fill decode's idle SMs with compute-bound work (prefill, tuning, other models) | **1.38–1.46×** | MuxServe, Usher, Bullet, Harli |
| **KV offload / tiering** | Spill KV to CPU/NVMe; reuse 100× more cache than fits in HBM | **TTFT 2–22×, tput up to 9×** (CPU); FlexGen 40–100× (throughput regime) | LMCache, vLLM connector, CacheBlend, FlexGen |
| **Speculative decoding** | Trade idle FLOPs for bandwidth: draft proposes, target verifies in parallel | **2–6.5×** latency *when memory-bound*; **net loss** when compute-bound | EAGLE-3 5.6×; vLLM shows 1.4–1.8× *slowdown* at high QPS |

**Layer-2 saturation:** each lever 2–7×, but they **stack** (paged × disagg ×
spec-decode × co-location), and the *baseline is 40% MFU* — there's ~2.5× of pure
waste sitting there before any cleverness. This is the layer where step functions
are most accessible, because the gains are large *and* the levers are mostly
software *and* nobody orchestrates them jointly across a fleet.

### LAYER 3 — Workload structure (tokens/FLOP — change the work itself)

| Lever | Mechanism | Standalone ceiling | Source |
|---|---|---|---|
| **Quantization** | FP16→FP8→FP4 weights+activations+KV; less bandwidth/byte | ~**2× per precision halving**, quality-gated | vLLM kv_cache_dtype; FP4 on Blackwell |
| **Model cascades / routing** | Cheap model answers easy queries, big model only the hard ones | **RouteLLM: ~2× cost cut at same quality**, sometimes 3–5× | RouteLLM, FrugalGPT |
| **Early exit / layer skip** | Stop at layer E when confident | ~2× on easy tokens | LayerSkip (ACL'24) |
| **MoE serving** | Only activate K of N experts | Sparse FLOPs; routing-dependent | DeepSeek/Mixtral class |
| **Prompt/context compression** | Fewer tokens through the model per request | Linear in tokens removed | LLMLingua-class |

**Layer-3 saturation:** quality-gated, not physics-gated. The ceiling is "how
much can you degrade precision/model size before the answer gets worse" — and
that's *per-query*, which is exactly why adaptive routing (a fleet decision) beats
any static choice.

### LAYER 4 — Market / temporal ($/joule, $/GPU-second — decouple cost from physics)

| Lever | Mechanism | Standalone ceiling | Source |
|---|---|---|---|
| **Energy arbitrage (time-shift)** | Run flexible work in cheap hours | **~20–30% of energy cost**, flexibility-gated | wholesale 2× intraday spreads |
| **Region routing** | Place in cheapest interconnect | spread-dependent | CAISO/PJM/ERCOT |
| **Spot/preemptible arbitrage** | Buy interruptible capacity, checkpoint-migrate | **~2.5× cheaper $/GPU-hr**, churn-gated | cross-cloud spot |
| **Demand response / capacity mkts** | Get *paid* to curtail; flexible interconnection | **$/MW-yr payments**; 2025 studies show large grid headroom if DCs curtail a few % of hours | ISO programs |
| **Renewable soak / behind-the-meter** | Flexible load eats curtailed solar/wind + on-site battery co-opt | cheap/free marginal energy | co-location deals |
| **Procurement portfolio** | Reserved/on-demand/spot as an options portfolio | smooths $/GPU-hr | FinOps |
| **Carbon-aware** | Shift to low-marginal-emissions hours | real $ only where carbon is priced | CAISO MOER |

**Layer-4 saturation:** bounded by *flexibility* (how much work tolerates delay/
move) and by *price volatility*. Mature, real, but a few % of TCO each and
diminishing. The arbitrage shrinks as everyone does it.

---

## 3. Where the step functions actually are — the cross-terms

Single-layer optimization is the sum of saturating curves. Multiplicative gains
come from exploiting **correlations between layers** that no single-layer optimizer
can see. These are the ideas worth real effort.

### 3.1 Roofline-aware everything (the unifying principle)

The master fact (decode is memory-bound) implies a single control law that
*re-prices every lever by the GPU's instantaneous bottleneck*:

- When **memory-bound** (low batch, long context): FLOPs are free → **turn ON**
  speculative decoding, **downclock** (clock doesn't help bandwidth — free watts),
  **co-locate** compute-bound work into the idle SMs.
- When **compute-bound** (high batch, short context): FLOPs are scarce → **turn
  OFF** spec-decode (it's a 1.4–1.8× *slowdown* here), **upclock**, stop co-locating.

**Nobody ships a fleet controller that reads the live roofline position per
replica and flips all three knobs together.** vLLM has *planned* dynamic
spec-decode toggling; it's not done. This single control law is a step function
because it converts three separate "sometimes helps, sometimes hurts" levers into
three "always helps" levers — and it's the connective tissue for everything else.

### 3.2 Frequency × electricity price (thermo × market cross-term)

DVFS is treated as a *thermal/perf* decision. Electricity price is treated as a
*scheduling* decision. **They're the same decision.** The perf/watt curve says the
top 20% of clock costs disproportionate watts; the market says watts cost 2–5×
more at 6pm than 2am. So the optimal clock is a **function of the live price**:

> Run flexible work at **high clock when power is cheap**, **low clock when power
> is expensive** — and shift the *throughput*, not just the *timing*, into cheap
> hours.

This is strictly more than energy arbitrage (which only moves *when* work runs).
It modulates the *power intensity* of the work continuously against price. The
emergent property: a job that can't be time-shifted (latency-bound) can still be
*power-shaped* against price by trading a little tail latency for cheap watts.
**Unbuilt.** Step-function candidate because it unlocks the ~80% of the fleet that
is too latency-sensitive for classic time-shift arbitrage.

### 3.3 Precision × query difficulty × price (the adaptive-fidelity cross-term)

Quantization and model-cascade routing are usually *static* config. But the
*right* precision is a per-query, per-moment decision:

> Serve an easy query at FP4 on a cheap spot GPU in a cheap-power region; escalate
> a hard query to FP8/FP16 on a fast GPU only when a cheap confidence signal says
> it's needed — and bias the whole fleet's precision *down* during expensive-power
> hours, *up* when power is cheap.

The emergent property: **quality becomes a dial you spend money on only where it
changes the answer.** Combine difficulty-routing (2× from RouteLLM) × precision
(2× FP16→FP4) × power-aware biasing (Layer-4) and you're multiplying three
independent factors that today live in three teams. Step-function candidate.

### 3.4 Fleet-global KV cache as a memory-hierarchy (the bandwidth cross-term)

KV offload (LMCache) and cache-aware routing exist *per-node*. The step function
is treating the **entire fleet's HBM + CPU DRAM + NVMe + network store as one
addressable cache hierarchy** and routing requests to wherever the relevant
KV/prefix already lives — across machines:

> A request whose 30k-token system prompt is already cached on node 7 should be
> *routed to node 7*, not re-prefilled on node 3. At fleet scale, with shared
> prompts (RAG corpora, agent system prompts, few-shot templates), cross-node
> cache reuse turns the most expensive operation (prefill) into a near-free
> lookup. TTFT 3–10× and throughput 2–5× are demonstrated *per-node*; the
> *fleet-wide* version is largely unbuilt.

This is roofline thinking applied to the network: prefill is the compute-bound
phase; turning it into a cache hit removes the most expensive FLOPs entirely. The
emergent property is a **fleet-level prefill hit-rate** that rises with scale —
bigger fleet, more shared prefixes, higher hit rate — i.e., *increasing returns to
scale*, the opposite of arbitrage's diminishing returns. That asymmetry is what
makes it a step function and a moat.

### 3.5 Power oversubscription × workload phase diversity (the statistical cross-term)

Oversubscription works because aggregate power < sum of peaks. But the headroom
depends on **decorrelation**. Inference has 21% headroom, training ~3%, because
training's synchronous gradient-sync spikes are *correlated*. The cross-term:

> A controller that *actively shapes* the phase mix — interleaving memory-bound
> decode (low, steady power) with compute-bound prefill/training (high, spiky
> power) so their peaks *anti-correlate* — manufactures headroom that doesn't
> naturally exist. You're not just exploiting diversity; you're *engineering* it.

Push the safely-installed density past 30% by deliberately co-scheduling
anti-correlated power profiles, with priority-aware capping as the safety net.
Emergent property: density (a capex-amortization win, Layer-1) becomes a function
of scheduling intelligence (Layer-2). **Unbuilt at this framing.**

### 3.6 GPU fleet as a grid asset (the market step-function)

The biggest dollar lever isn't *reducing* energy cost — it's **getting paid for
flexibility**. A fleet that can curtail/shift on command is a virtual power plant.
2025 flexible-interconnection studies show grids can host vastly more compute load
*if* it curtails a few % of hours. The fleet's controllable power (DVFS + workload
shifting + battery) can sell: demand response, capacity, frequency regulation.

> The same flexibility machinery built for cost-arbitrage (Layer-4) can be
> *pointed at revenue*: bid the fleet's curtailable MW into capacity markets,
> earn $/MW-yr for being interruptible, and stack that on top of the energy saved.
> Revenue, not just cost reduction — which changes the unit economics entirely.

Step function because it adds a *new income line*, and because flexible
interconnection can be the difference between "wait 5 years for grid capacity" and
"energize now" — i.e., it gates whether the fleet exists at all.

---

## 4. The genuinely novel ideas — "nobody has truly done this"

Ranked by (upside × non-obviousness). These are the swing-for-the-fences bets.

### N1 — The unified roofline-economic controller (the meta-idea)

One objective — **SLA-safe goodput per dollar** — and one controller that holds
*all four layers' knobs simultaneously*: per-replica clock, precision, batch
composition, spec-decode depth, prefill/decode disaggregation ratio, KV placement,
region, power cap, and market position. Every existing system optimizes a slice.
The novel claim: the **joint optimum is far from the product of the local optima**,
because the knobs are coupled through the roofline and through price. This is the
thing that turns a sum of saturating curves into a product. Everything else in §4
is a special case of this.

### N2 — Power-shaping latency-bound work against price

Break the assumption that "latency-sensitive ⇒ not flexible." Latency-bound work
*can't* be time-shifted but *can* be **power-shaped**: trade a few ms of tail
latency for a lower clock during expensive-power minutes, within the SLA budget.
This unlocks the ~80% of the fleet that classic arbitrage can't touch. Nobody
treats the SLA slack as a *power-arbitrage* budget.

### N3 — Anti-correlation scheduling for manufactured power headroom

Actively co-schedule anti-correlated power profiles (steady decode + spiky
prefill/training) to *engineer* statistical-multiplexing headroom rather than
passively exploit it — pushing safe oversubscription past its natural ceiling.
Density is the single biggest capex-amortization lever; making it a software
output is a step function on $/GPU-hr.

### N4 — Fleet-global content-addressed KV hierarchy with increasing returns

Treat all fleet memory tiers as one cache; route by where KV already lives;
prefill hit-rate *rises with fleet size*. Increasing returns → moat. The hard
parts (consistency, eviction-by-reuse-prediction, prefetch pipelining across
GPU↔CPU↔NVMe) are explicitly noted as *unbuilt* in mainline vLLM.

### N5 — Adaptive fidelity as a priced dial

Per-query precision/model/early-exit chosen by a cheap difficulty predictor, with
the fleet's *global* fidelity biased by live power price. Quality becomes a
metered cost you pay only where it moves the answer. Three multiplied factors, all
currently static.

### N6 — Selling flexibility (VPP) as a revenue line

Bid curtailable MW into demand-response/capacity/regulation markets. Turns the
cost-optimizer into a revenue-generator and can gate grid interconnection itself.

### N7 — Speculative *fleet* scheduling (predictive, not reactive)

Forecast the request stream (arrival, prompt length, output length, difficulty)
and *pre-position*: pre-warm replicas, pre-fetch KV, pre-acquire spot capacity,
pre-shape power — before the work arrives. Reactive autoscaling always pays a
cold-start tax; a forecaster that's right even 60% of the time converts that tax
into a structural margin. The honest caveat: output-length prediction is *hard*
(Azure prompt↔output correlation r≈−0.02), so this must be gated on beating a
running-median baseline — but arrival-rate and prefix-reuse prediction are
tractable today.

### N8 — Hardware-software perf/watt co-design via closed-loop measurement

Most "perf/watt curves" are datasheet fiction; real silicon varies chip-to-chip
(the "silicon lottery"). A fleet that *continuously measures* each GPU's actual
perf/watt and tokens/joule and routes the most power-intensive work to the most
efficient *individual chips* extracts a free few % that nobody bothers to capture
because it requires per-chip telemetry + placement coupling.

---

## 5. How each would be tested (the part that makes it real, not a pitch)

The discipline: **simulate → shadow → controlled execution**, and never claim a
number you haven't measured against a *fair* baseline (not a strawman). The honest
failure mode of this entire field is inflated baselines — measure vs the strongest
*deployable* alternative, not vs FIFO or an oracle.

| Idea | Cheapest decisive test | Fair baseline | Kill criterion |
|---|---|---|---|
| **N1 unified controller** | Replay one real trace (e.g. Azure LLM) through a simulator that models roofline+price+thermal; compare joint vs best single-layer | Strongest per-class single-layer optimizer | Joint ≤ product of locals (no cross-term gain) |
| **N2 power-shaping** | Offline: per-request, compute watt-savings of clock-down within SLA slack vs realized price | `current_price_only` time-shift | Tail-latency SLA breach, or <2% energy gain |
| **N3 anti-correlation** | Trace-replay power sim: measure peak-power reduction from phase-interleaved vs random placement | Statistical-multiplex baseline (passive) | No headroom beyond passive multiplexing |
| **N4 fleet KV** | Replay shared-prefix workload (RAG/agents); measure fleet prefill hit-rate & TTFT vs per-node cache | Per-node LMCache + prefix routing | Hit-rate doesn't rise with fleet size |
| **N5 adaptive fidelity** | Offline router on labeled difficulty set; measure quality-matched cost vs static FP16 | RouteLLM / static best | Quality drop at matched cost, or router cost > savings |
| **N6 VPP revenue** | Paper trade: simulate curtailment bids against historical DR/capacity clearing prices | No-participation | Curtailment opportunity-cost > market payment |
| **N7 speculative scheduling** | Backtest forecaster vs running-median; measure cold-start tax removed | Reactive autoscale | Forecaster ≤ running-median ceiling |
| **N8 per-chip perf/watt** | Measure tokens/joule spread across nominally-identical GPUs; route & re-measure | Random placement | Spread < measurement noise |

**Two universal test rules that prevent fooling yourself:**

1. **SLA is a filter on the numerator, never a discount on the denominator.** A
   "win" that breaks p99 isn't a win; exclude timed-out tokens from goodput.
2. **Shadow before you actuate.** Record what the controller *would* have done
   against live data, compare predicted vs realized, and only promote a lever to
   live control after it beats the fair baseline in shadow. Advisory + reversible
   + degrade-to-status-quo is what makes any of this adoptable in production.

---

## 6. The one-paragraph synthesis

A GPU fleet's economic yield is a *product* of four ratios — tokens/joule,
tokens/GPU-second, $/joule, $/GPU-second — that the industry optimizes *as a sum*,
in four silos that never share state. The dominant physical fact is that decode is
memory-bandwidth-bound, so most fleets run their compute units at <1% utilization
and their overall silicon at ~40% MFU: **the largest pool of waste is idle FLOPs,
not expensive electrons.** Single-layer levers (DVFS ~20%, arbitrage ~25% of
energy, each serving trick 2–4×) are real but saturating. The step functions are
in the **cross-terms**: re-pricing every knob by the live roofline position (N1),
power-shaping latency-bound work against price (N2), engineering oversubscription
headroom by anti-correlating power phases (N3), a fleet-global content-addressed
KV hierarchy with *increasing* returns to scale (N4), per-query adaptive fidelity
biased by power price (N5), and pointing the same flexibility machinery at grid
*revenue* (N6). The unifying move is a single controller with one objective —
SLA-safe goodput per dollar — that holds all four layers' knobs at once, because
the joint optimum is provably far from the product of the local optima. The way
you keep it honest is simulate → shadow → controlled execution, always against the
strongest deployable baseline, with SLA as a filter on goodput and every lever
reversible.
