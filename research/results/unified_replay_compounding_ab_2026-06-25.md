# Unified Replay Engine (Phase 1b-A) — the compounding A/B — 2026-06-25

> The closed joint **decision loop** (`aurelius/optimizer/unified_replay.py`,
> `AureliusOptimizer.optimize_joint_closed_loop`). All serving surfaces act in ONE
> discrete-event loop on ONE evolving cluster state — capacity REACTS to the live
> backlog that ordering+admission shaped — scored by ONE objective
> (`ObjectiveLayer`). Directional simulator only — not production savings
> (`docs/RESULTS.md` §8). Reproducible: deterministic + `jobs_hash` in
> `research/results/unified_replay_compounding_ab.json`.

## The question (restated)

Is the fact that combining serving levers does **not** compound (the open-loop
`joint.combination_search` result) a property of the **optimizer**, or of the
**data**? The hypothesis under test: our traces are too *stratified* —
single-class, all-latency-critical — so admission has nothing legal to defer and
the levers all fight for the same queue slack. If true, compounding should appear
the moment the data carries workload-class structure, with **no change to the
optimizer**.

## The instrument

Two things had to exist to answer this honestly:

1. **A closed loop** (not the open-loop fan-out). `joint.combination_search`
   pre-computes each lever's schedule from the *whole* trace offline, so capacity
   for tick *t* can never see the backlog this tick's ordering/admission produced
   — levers cannot interact. The unified engine runs a single event heap
   (arrivals + completions + tick-boundary control) over one `_State`; at each
   tick the capacity controller observes the **live** latency-critical backlog and
   sizes the next window from it. One surface's decision mutates the next's inputs.
2. **A multi-class dataset** (`aurelius/datasets/canonical.py`). The real Azure LLM
   2024 trace is the `latency_critical` spine; a documented, deterministic
   best-effort batch overlay (tokens resampled from the spine's own distribution,
   steady cadence) adds the one signal public serving traces strip out: a class
   that is *legal to defer*.

## Result — same optimizer, only the data changes

**Azure LLM 2024 · 5,880 reqs · on-demand · tick=60 s · SLA=10 s · GPU=$2/hr.**
Levers: **C** = backlog-aware capacity · **O** = abs-conformal SRPT ordering · **A** = class-aware admission.

### A) SINGLE-CLASS — raw Azure (every request latency-critical) — `jobs_hash=88d664b47d6dead4`

| levers | goodput/$ | c_mean | SLA-safe | vs base |
|---|---|---|---|---|
| base | 59,054.4 | 4.53 | 5,811 | +0.00% |
| A | 59,054.4 | 4.53 | 5,811 | +0.00% |
| O | 59,048.8 | 4.53 | 5,810 | −0.01% |
| C (and every combo with C) | 56,454.8 | 4.76 | 5,824 | −4.40% |

**best single = A (+0.00%) · best multi = O+A (−0.01%) · INTERACTION = NEUTRAL.**
Admission defers **nothing** (`defer=0` everywhere) — there is no legal best-effort
load. Capacity slightly *over*-provisions for a queue that is already SLA-met.

### B) MULTI-CLASS — Azure spine + best-effort overlay (5,880 LC + 2,352 BE) — `jobs_hash=82a6768ff5f35025`

| levers | goodput/$ | c_mean | SLA-safe | defer | vs base |
|---|---|---|---|---|---|
| **C+O+A** | **75,223.9** | 5.12 | 8,168 | 1,060 | **+9.00%** |
| C+A | 75,020.6 | 5.14 | 8,168 | 1,060 | +8.70% |
| C | 73,643.3 | 5.24 | 8,168 | 928 | +6.71% |
| C+O | 73,481.5 | 5.25 | 8,169 | 1,060 | +6.47% |
| O | 69,068.8 | 4.53 | 7,190 | 829 | +0.08% |
| base | 69,015.5 | 4.53 | 7,179 | 0 | +0.00% |
| O+A | 64,928.8 | 4.53 | 6,795 | 928 | −5.92% |
| A | 64,037.5 | 4.53 | 6,705 | 829 | −7.21% |

**best single = C (+6.71%) · best multi = C+O+A (+9.00%) · INTERACTION = COMPOUNDING.**
The best combination beats the best single lever by **+2.3 points** (9.00 vs 6.71) —
combining genuinely compounds.

## Verdict — it was a DATA issue, and now it is proven

The optimizer code is **byte-identical** across A and B. The only variable is the
data's workload-class structure. Combining the levers is **neutral/substitutive on
single-class data and compounding on multi-class data**. Therefore the
no-compounding result was a **property of the stratified data, not the optimizer**.
The user's hypothesis is confirmed by measurement.

## Why it compounds (the mechanism the closed loop revealed)

Compounding is not "more optimizers = more savings." It is three surfaces on
**different cost terms** finally sharing a state:

1. **Class-aware capacity (C)** sizes on-demand replicas for the *latency-critical*
   tier only and lets best-effort **backfill** spare capacity — so batch never
   triggers scale-up. On multi-class data this flips capacity from −4.40% (where it
   over-provisions) to **+6.71%** (it adds goodput at near-flat cost). The lever's
   sign is set entirely by the data.
2. **Class-aware admission (A)** defers best-effort *only while the latency-critical
   tier is genuinely backlogged* (a real burst) and drains it in the troughs — so
   batch is time-shifted into valleys instead of stealing the SLA tier's servers
   during peaks. Alone it is negative (it delays batch); *combined with C* it lets
   capacity stay even leaner (c_mean 5.24 → 5.12) at equal goodput → **+8.70%**.
3. **Ordering (O)** adds the last sliver by serving short latency-critical requests
   first within the protected tier → **+9.00%**.

An honest counter-finding the closed loop exposed: a **naive** backlog-chasing
capacity controller *fights* admission (it over-provisions to chase the
best-effort backlog admission just deferred — we measured −58% before fixing it).
The fix is the standard multi-tier model: latency-critical drives capacity,
best-effort backfills. Open-loop search cannot even *see* this interaction;
the closed loop makes it measurable.

## Honesty boundary

- The best-effort overlay is **SYNTHETIC** and labeled as such in the manifest. It
  is not real demand; it re-times + re-labels *real* token counts resampled from
  the spine. It is the minimal honest augmentation that supplies the missing class
  dimension — enough to prove the *mechanism* and the *data-vs-optimizer* question,
  **not** a production savings number.
- The real production version of this overlay is a fleet's actual batch/offline
  tier (evals, data-gen, batch inference), which a read-only telemetry pilot would
  capture directly. Until then this is a controlled test bed, not a claim.
- Numbers are simulator-directional under the shared on-demand denominator and
  Erlang-C/service physics; same caveats as `docs/RESULTS.md` §8.

## What this unblocks

The loop + a class-carrying dataset is the prerequisite for compounding. The next
cost terms (energy time-shift on the deferred batch, placement affinity, thermal
headroom) need the **further** signals audited in
`research/CANONICAL_PRODUCTION_DATASET_DESIGN.md` and
`aurelius/datasets/signal_matrix.py`. The engine is built to take them: controllers
are pluggable and the cost denominator is additive.
