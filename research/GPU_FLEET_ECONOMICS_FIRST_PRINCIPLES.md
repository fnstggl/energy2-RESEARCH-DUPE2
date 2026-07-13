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
| **DVFS / power capping** | Drop clock/power cap; firmware slides V+f together down the curve | Curve is **flat to ~70–80% of TDP** (>95% throughput retained), then collapses below ~60% → **default TDP sits past the efficiency knee**. A100 inference: 200W cap (~80%) = **−12% energy for +8% time**; 150W (~60%) = knee passed (+22.7% time) | "From Words to Watts" (arXiv 2310.03003); MIT-LL SC'23: 200W cap = 10–20% energy at <5% perf; Meta 100MW: 960W vs 1200W → **+11% throughput/W** |
| **Iso-throughput frequency clipping** | Lower SM clock only into scheduling/pipeline slack — clip the top of the V/f curve where it's pure waste | **10–30% energy at <1–5% perf** | EnvPipe (ATC'23) −25–28% @ <1%; Perseus (SOSP'24) −30% @ ~0–2%; Zeus (NSDI'23) −3–31% iso-tput |
| **Phase-aware decode downclocking** ⭐ | Decode is memory-bound → SM clock barely affects token rate. Crucially, **decode only draws 137–300W on a 700W GPU, so a power *cap* never triggers — you must lock the *clock***, not the power | **−24% to −43% decode energy at <1–3% perf** (decode is 77–91% of wall-clock, so this dominates) | RTX PRO 6000: −42% @ ±0.4% (arXiv 2501.08219); H200: −24–32% (arXiv 2605.11999, "Illusion of Power Capping in Decode"); GreenLLM A100 (arXiv 2508.16449) |
| **Undervolting** (V²f) | Lower V at fixed f; reclaims the ~20% vendor voltage guardband | **~15–25% energy iso-frequency** (Leng MICRO'15: ~20% guardband, ~25% on one card) — but **datacenter GPUs expose no voltage knob** (only `-pl`/`-lgc`); risk is **silent data corruption / ECC / NaNs** below program-dependent Vmin | Leng MICRO'15; GreenMM/SAOU ~14–22%; HBM-undervolt DATE'21 |
| **Power oversubscription** | Install more GPUs than peak power supports; multiplex diverse load | **+25–30% more servers** same power envelope = avoided capex | POLCA +30% (inference); Google MV-plane +25%, "hundreds of millions" saved |
| **Overclocking the headroom** | Inverse: spend reclaimed headroom to *overclock* hot requests | Risk-gated speedup | SmartOClock (ISCA'24) |
| **Cooling / PUE co-opt** | Hotspot avoidance, rack placement, inlet-temp-aware scheduling reduce *cooling* power (often 10–30% of facility draw) | PUE 1.5→1.1 ≈ **~25% facility-power cut** | Thermal-aware placement is researched; **price/carbon-coupled thermal scheduling is not** |

**Layer-1 saturation:** physics-bounded. The whole layer is maybe **15–25% of
energy cost + a one-time 25–30% capex-density win**. Real, bankable, but bounded.
The sharpest under-exploited finding: **decode-phase clock-locking** (⭐) is a
near-free 24–43% energy cut that *most operators miss because they reach for power
caps, which don't even engage during decode* — it needs clock control, and it
feeds N2 directly.

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
| **Quantization — the precision ladder** | FP16→FP8→FP4/INT4; fewer bytes/param → less bandwidth (which *is* decode speed). Quality cost is sharply non-linear, and **weight-only is nearly free to 4-bit; quantizing activations or going sub-4-bit is where it costs** | FP16→**FP8 ~2× tput, free quality** (FP8-LM, SmoothQuant W8A8). FP8→**INT4/FP4 weight-only ~2–3.5× memory at ≤0.2 PPL / ≤1% acc** (GPTQ, AWQ — the sweet spot). **W4A4 highest theoretical (Atom 7.7×) but ~0.5 PPL / 1–2% acc and hardware-fragile**. **INT2 = collapse** (PPL >38,000 — hard cliff) | LLM.int8(), SmoothQuant (ICML'23), GPTQ, AWQ (MLSys'24), QServe W4A8KV4, Atom, NVFP4 |
| **KV-cache quantization** | FP8/FP4 KV → less HBM traffic per decode step; directly buys token rate since decode is bandwidth-bound | Folded into W4A8**KV4** stacks (QServe); KIVI/KVQuant | QServe, KVQuant, KIVI |
| **Model cascades / routing** | Cheap model answers easy queries, big model only the hard ones | **RouteLLM (peer-reviewed, reproducible): >85% cost cut on MT-Bench / 45% MMLU / 35% GSM8K at ~95% GPT-4 quality**; 95% quality using only 26% strong-model calls. (Vendor routers NotDiamond/Martian claim up to 98% but **self-reported, unverified**) | RouteLLM (ICLR'25); FrugalGPT |
| **Early exit / dynamic depth** | Stop at layer E when confident; per-token compute scaling | CALM **~3× at provably-maintained quality** (conformal); Mixture-of-Depths ~50% faster sampling (but needs training-time arch change) | CALM (NeurIPS'22), MoD (DeepMind'24), LayerSkip (ACL'24) |
| **Prompt/context compression** | Fewer tokens through the model per request | LLMLingua-class up to **~20× prompt compression**, quality-gated | LLMLingua / LongLLMLingua |

**Layer-3 saturation:** quality-gated, not physics-gated. The ceiling is "how
much can you degrade precision/model size before the answer gets worse" — and
that's *per-query*, which is exactly why adaptive routing (a fleet decision) beats
any static choice. **Critical honesty finding (feeds N5):** choosing model *size*
per-query is **productized** (GPT-5 router, OpenRouter, NotDiamond, Martian) — but
the only *reproducible* numbers are RouteLLM's. Choosing *precision* per-query by
difficulty is **research-only** (QAQ, MoQE, Any-Precision-LLM as substrate) — **no
mainstream stack ships it.** Production quantization is static and model-wide. That
gap is exactly where N5 lives.

### LAYER 4 — Market / temporal ($/joule, $/GPU-second — decouple cost from physics)

| Lever | Mechanism | Standalone ceiling | Source |
|---|---|---|---|
| **Energy arbitrage (time-shift)** | Run flexible work in cheap hours | **~20–30% of energy cost**, flexibility-gated | wholesale 2× intraday spreads |
| **Region routing** | Place in cheapest interconnect | spread-dependent | CAISO/PJM/ERCOT |
| **Spot/preemptible arbitrage** | Buy interruptible capacity, checkpoint-migrate | Headline "up to 90% off" is a ceiling; **GPU spot realistically ~50–70% off**. Fault-tolerant training: **2.4× cheaper vs on-demand** (Bamboo), **~5×** (Varuna); cross-region spot serving **−43% avg** (SkyServe). Checkpoint overhead bounded **<3.5%** (CheckFreq) | Bamboo (NSDI'23), Varuna (EuroSys'22), Oobleck, SkyServe, SkyPilot |
| **Demand response / capacity mkts** | Get *paid* to curtail | **PJM capacity ~$98k–$170k/MW-yr** (2026/27 BRA hit cap $329/MW-day) on *derated UCAP* w/ performance penalties; **ERCOT ~$85k/MW-yr** (AS + 4CP avoidance). **BUT ancillary revenue is *collapsing* (~80% ERCOT 2023→24 as batteries saturate)**, and *for AI the compute opportunity-cost often exceeds the curtailment payment* | PJM BRA reports; ERCOT IMM (Potomac); Carbon Direct |
| **Flexible interconnection** | Curtail a few % of hours → energize *now* instead of waiting years for grid capacity | **Duke: 76 / 98 / 126 GW** US headroom at **0.25 / 0.5 / 1.0%** curtailment (85 / 177 / 366 hrs-yr) — but authors disclaim it (no transmission modeling, flexibility "unclear"). This gates *whether the fleet exists*, not just its cost | Duke "Rethinking Load Growth" (2025); LBNL |
| **Renewable soak / behind-the-meter** | Flexible load eats curtailed solar/wind + on-site battery co-opt | **Soluna (filings-backed): ~$28–32/MWh** all-in (vs ~$50+ grid); CAISO curtailed 3.4M MWh in 2024. Crusoe "<1¢/kWh" / Lancium ">50%" are **marketing, unaudited**. Counter-signal: **standalone battery arbitrage collapsed ~$149→$17/kW** (ERCOT 2023→25) | Soluna 10-Q; EIA; Enverus (skeptic) |
| **Procurement portfolio** | Reserved/on-demand/spot as an options portfolio | Spot ~60–90% off, reserved ~30–50% off, neoclouds 2–3× under hyperscalers. H100 rents **collapsed ~64–70% ($8→~$2.5) then rebounded ~40%** — volatile. **No formal options-pricing model of GPU procurement exists** (a real gap) | SemiAnalysis, Silicon Data index, Princeton CITP |
| **Carbon-aware** | Shift to low-marginal-emissions hours | **Mostly reporting-only.** Real $ only where carbon is priced: **EU ETS ~€80/tonne**, or the **24/7-CFE premium $8–27/MWh** you avoid buying. US grid has no carbon price → SCC $190/t is a shadow price nobody pays. Meta Carbon Explorer: emissions win can cost **6–76% extra capex** | EU ETS; Princeton ZERO Lab; Meta Carbon Explorer |

**Layer-4 saturation:** bounded by *flexibility* (how much work tolerates delay/
move) and by *price volatility*. Mature, real, but a few % of TCO each and
diminishing — and the two biggest "revenue" levers are *shrinking*: ancillary-
service prices are collapsing under battery saturation, and curtailment payments
are usually below AI's compute opportunity-cost. **The honest version of N6 is
narrow:** grid-flexibility revenue wins specifically when compute is *already idle*
(so opportunity-cost ≈ 0) or when flexibility *gates interconnection itself* — not
as a general income line on busy fleets.

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

**The research now gives this a hard, near-free floor.** The decode phase — 77–91%
of inference wall-clock — is memory-bound, so dropping its SM clock saves **24–43%
of decode energy at <1–3% throughput loss** (RTX PRO 6000 −42% @ ±0.4%; H200
−24–32%). And the sharp, under-appreciated mechanism: **decode draws only 137–300W
on a 700W GPU, so a power *cap* never even engages — you must lock the *clock*.**
That means most operators' instinct (set a power limit) silently does nothing
during decode; the lever is clock control. So N2's price-shaping isn't trading
quality for cheap watts in the abstract — there's a measured ~30% decode-energy
band that is *already* nearly free to modulate, before touching any latency budget.

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

**Honest, research-disciplined version (this is where I'd most temper the original
pitch).** The revenue is real but bounded and frequently *negative-EV on a busy
fleet*: PJM capacity pays ~$98k–$170k/MW-yr but only on derated UCAP with
performance penalties; ancillary-service prices are *collapsing* (~80% in ERCOT
2023→24 as batteries flood in); and Carbon Direct's key point — **for AI, the
opportunity-cost of curtailed compute usually exceeds the curtailment payment.**
So the step-function is *narrow and specific*, not a general income line:
> (a) **Idle-capacity arbitrage** — sell flexibility only from compute that is
> *already* idle (opportunity-cost ≈ 0), which a fleet controller uniquely knows.
> (b) **Interconnection-gating** — the real prize isn't the $/MW-yr, it's that
> committing to curtail ~0.25–1% of hours can unlock **76–126 GW** of US grid
> headroom (Duke) and energize *years sooner*. That gates whether the fleet
> exists at all — worth far more than any capacity check.
The measured ceiling today is still small (Emerald AI's flagship: 25% off **256
GPUs for 3 hours**), and every headline "GW unlock" traces to one Duke study its
own authors disclaim. Treat the revenue as a tie-breaker; treat *interconnection
speed* as the actual step-function.

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
output is a step function on $/GPU-hr. **The same decorrelation principle has a
market twin the research surfaced:** spot-instance preemptions are **83–97%
correlated *within* a region but ≈0 correlated *across* regions** (SkyServe). So a
fleet that spreads interruptible work across regions converts spot from "risky" to
"reliable-in-aggregate" — geographic decorrelation, not raw price, is the real
capturable spot edge (cross-region spot serving −43% avg). Anti-correlation is the
unifying motif: in *power* (N3) you engineer it across workload phases; in *spot
risk* you harvest it across regions.

### N4 — Fleet-global content-addressed KV hierarchy with increasing returns

Treat all fleet memory tiers as one cache; route by where KV already lives;
prefill hit-rate *rises with fleet size*. Increasing returns → moat. The hard
parts (consistency, eviction-by-reuse-prediction, prefetch pipelining across
GPU↔CPU↔NVMe) are explicitly noted as *unbuilt* in mainline vLLM.

### N5 — Adaptive fidelity as a priced dial

Per-query precision/model/early-exit chosen by a cheap difficulty predictor, with
the fleet's *global* fidelity biased by live power price. Quality becomes a
metered cost you pay only where it moves the answer. Three multiplied factors, all
currently static. **Research sharpened the white space precisely:** choosing model
*size* per-query is productized (GPT-5 router, OpenRouter) with RouteLLM as the one
reproducible proof (>85% cost cut at ~95% quality). Choosing *precision* per-query
by difficulty is **research-only — nobody ships it** (production quant is static
and model-wide; the quality ladder is known: weight-only INT4 ≈ free, W4A4 costs
~1–2%, INT2 collapses). So N5's novelty is specifically the **price-biased,
per-query *precision* dial**, on top of the already-validated size router.

### N6 — Selling flexibility (VPP) as a revenue line — *narrowed by the data*

Bid curtailable MW into demand-response/capacity/regulation markets. **But the
research says treat the $/MW-yr as a tie-breaker, not the thesis:** PJM capacity
~$98–170k/MW-yr is gross-on-derated-UCAP with penalties; ancillary prices are
collapsing (~80% ERCOT YoY); and AI compute opportunity-cost usually exceeds the
curtailment payment. The real step-function is twofold: **(a)** sell flexibility
only from *already-idle* compute (a fleet controller uniquely knows this, making it
near-pure-margin), and **(b) interconnection-gating** — committing to curtail
~0.25–1% of hours can unlock 76–126 GW of grid headroom (Duke) and energize *years
sooner*, which dwarfs the capacity check. Revenue is the garnish; **grid-access
speed is the meal.**

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

---

## Appendix A — Research grounding (primary sources by layer)

Quantified claims above are drawn from these. Tagged where a number is *measured*
vs *vendor/marketing* vs *projection*. Numbers are directional, not guarantees.

**Layer 1 — thermodynamic / DVFS / power**
- Power-cap curve (A100 inference): "From Words to Watts" arXiv 2310.03003 — 200W cap ≈ −12% energy/+8% time; knee below ~60% TDP.
- Iso-throughput frequency clipping: Zeus (NSDI'23, arXiv 2208.06102); EnvPipe (ATC'23) −25–28% @ <1%; Perseus (SOSP'24, arXiv 2312.06902) −30% @ ~0–2%.
- Decode clock-locking ⭐: RTX PRO 6000 −42% @ ±0.4% (arXiv 2501.08219); H200 "Illusion of Power Capping in LLM Decode" (arXiv 2605.11999); GreenLLM (arXiv 2508.16449).
- Undervolting: Leng "Safe Limits on Voltage Reduction" MICRO'15 (~20% guardband, ~15–25% energy); datacenter GPUs expose only `-pl`/`-lgc`, risk = SDC/ECC/NaN.
- Oversubscription: POLCA (ASPLOS'24) +30% servers; Meta 100MW (arXiv 2605.24461) inference 21% vs training ~3% headroom, 960W cap +11% tput/W; Fan ISCA'07 (72% aggregate peak).

**Layer 2 — utilization (validated last round, unchanged)**
- Roofline/MBU: "LLM Inference Unveiled" arXiv 2402.16363; Databricks MBU. MFU: PaLM 46%, MegaScale 55% (NSDI'24), Llama 3 38–43%.
- Serving: Orca (OSDI'22), vLLM/PagedAttention (SOSP'23), Sarathi-Serve (OSDI'24), DistServe (OSDI'24), Splitwise (ISCA'24), MuxServe; KV offload: LMCache, CacheBlend (EuroSys'25), FlexGen (ICML'23); spec-decode: EAGLE-3, MagicDec, vLLM spec-decode blog.

**Layer 3 — workload structure**
- Quantization: FP8 Formats (arXiv 2209.05433); LLM.int8() (NeurIPS'22); SmoothQuant (ICML'23); GPTQ (ICLR'23); AWQ (MLSys'24 best paper); QServe W4A8KV4 (MLSys'25); Atom (MLSys'24, 7.7× *with* the W8A8-fragility caveat); NVFP4 (NVIDIA, conflates format w/ Blackwell generation); INT2-collapse (survey arXiv 2506.10205).
- Routing: RouteLLM (ICLR'25, the reproducible numbers); FrugalGPT; vendor routers GPT-5/OpenRouter/NotDiamond/Martian (self-reported). Per-query *precision* selection = research-only (QAQ, MoQE, Any-Precision-LLM). Dynamic depth: CALM (NeurIPS'22), Mixture-of-Depths (DeepMind'24).

**Layer 4 — market / grid (heavily skepticism-flagged)**
- Capacity/DR: PJM BRA reports ($269.92 → cap $329.17/MW-day); ERCOT IMM/Potomac (AS −80% YoY); Carbon Direct ("opportunity-cost > payment").
- Flexible interconnection: Duke "Rethinking Load Growth" (2025) 76/98/126 GW — authors disclaim (no transmission, flexibility "unclear").
- Spot: Bamboo (NSDI'23) 2.4× vs on-demand; Varuna (EuroSys'22) ~5×; SkyServe (intra-region 83–97% corr, inter-region ≈0, −43% avg); CheckFreq (<3.5% overhead).
- Behind-the-meter: Soluna 10-Q (~$28–32/MWh, filings-backed); EIA (CAISO 3.4M MWh curtailed 2024); Crusoe/Lancium figures = marketing; Enverus (ERCOT battery arbitrage ~$149→$17/kW — saturation counter-signal).
- VPP: Emerald AI Phoenix (measured: 25% off 256 GPUs, 3 hrs); Google Omaha DR (3 events); the "100 GW = $2T" = NVIDIA extrapolation on the Duke projection.
- Procurement: SemiAnalysis (all-in floor ~$1.525/hr), Silicon Data H100 index (−64–70% then +40% rebound), Princeton CITP (depreciation/economic-life). No formal options-pricing model of GPU procurement exists — a genuine gap.

**Confidence note.** Layer 2 (the utilization core, where N1/N3/N4 live) is the
best-sourced and load-bearing. Layer 1 decode-clock-locking and Layer 3
quantization ladder are well-grounded in primary papers. Layer 4 is real but the
softest — its big "revenue" numbers are either collapsing (ancillary), disclaimed
(Duke headroom), or marketing (Crusoe/Lancium); the doc has been written to reflect
that, not paper over it.
