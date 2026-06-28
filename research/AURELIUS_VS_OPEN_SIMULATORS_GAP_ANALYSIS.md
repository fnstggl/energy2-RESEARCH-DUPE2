# Aurelius vs Open Simulators — Gap Analysis (Phase 2)

Direct comparison of **current Aurelius** against each open simulator on the ten dimensions the brief
asks about. Each gap is classified:

- **A>ext** — already better in Aurelius
- **ext>A** — better in the external simulator
- **ext=val** — external simulator useful as *validation only*
- **ext-narrow** — external simulator too narrow to matter here
- **ext-incompat** — external simulator incompatible with Aurelius architecture
- **A-custom** — needs custom Aurelius implementation (no clean external source)

---

### 1. Prefill / decode realism — *who is more realistic than Aurelius?*

**Winner: BLIS and Alibaba InferSim (roofline) for the *physics*; Vidur for *calibrated accuracy*.**
Aurelius pre-PR used two scalar constants (`0.00015` prefill, `0.020` decode s/tok) with no GPU/model
resolution. InferSim/BLIS compute `max(compute_time, mem_time)` per stage from FLOPs + HBM bandwidth, so
they resolve the 4–40× spread across GPU×model that a constant cannot (verified: this PR's
`roofline_external.py` shows 8B-on-H100 decode ≈ 5.3 ms/tok vs 70B-on-A100 ≈ 84 ms/tok). Vidur is the most
*accurate* (RandomForest on profiled data, <9% error) but the wrong shape (needs a profiling corpus).

- vs InferSim: **ext>A** → **PORT-EQ** (done this PR as a reference model).
- vs BLIS: **ext>A** (trained-physics β-corrected) but Go → **PORT-EQ / ext=val**.
- vs Vidur: **ext>A** on accuracy → **ext=val** (calibration anchor) unless Aurelius adopts profiled tables.
- vs SplitwiseSim/LSSim: **ext>A** but profiled-table (needs CSVs) → **ext=val**.

### 2. Batching — *who is more realistic?*

**Winner: vLLM (reference), LLMServingSim 2.0, SplitwiseSim, Vidur, BLIS — all model true iteration-level
continuous batching.** Aurelius approximates with a decode `batch_factor` + Little's-law occupancy, not a
per-iteration hybrid batch. SplitwiseSim's `ORCAInstance`/`get_iteration_duration` and LSSim's chunked-
prefill stall-free batching (Sarathi Algorithm 3) are the design references.

- **ext>A** → **PORT-PATTERN** (iteration-level loop + chunked-prefill admission). Source: Orca pattern,
  Sarathi Algorithm 3, SplitwiseSim `instance.py`. This is the single biggest *serving-physics* gap.

### 3. KV cache — *who is more realistic?*

**Mixed. Aurelius is already strong here** (stateful paged LRU per-replica cache, Mooncake `hash_ids`,
mem-pressure eviction, lookup overhead) — **more realistic than SplitwiseSim** (bulk byte accounting, no
eviction/reuse), **InferSim/Vidur** (block budget, no prefix reuse / LRU). **LLMServingSim 2.0, BLIS, and
LMCache are ahead** on: radix-tree prefix matching, CPU/SSD/remote KV tiers, and KV *transfer* cost.

- vs SplitwiseSim/InferSim/Vidur: **A>ext** (keep Aurelius cache).
- vs LSSim/BLIS (radix + tiers): **ext>A** → **PORT-PATTERN** (tiered KV + transfer cost) — but only the
  *cost model*, not the runtime.
- vs LMCache (CacheBlend non-prefix reuse): **ext-narrow** today (Mooncake `hash_ids` are prefix-only).

### 4. Roofline / hardware — *who is more realistic?*

**Winner: InferSim, llm-analysis, LLM-Viewer, BLIS — all have a real FLOP/bandwidth roofline; Aurelius
pre-PR had none.** This was Aurelius' clearest deficit.

- **ext>A** → **PORT-EQ** (done this PR: `roofline_external.py`). LLMRoofline **ext-incompat** (no license,
  decode-only). See Phase 5 for the full decision.

### 5. Network / topology — *who is more realistic?*

**Winner: LLMServingSim 2.0 (ASTRA-sim cycle-level collectives/contention) and BLIS.** Aurelius models a
**macro** rack/ASW pressure penalty only (no per-link/NVLink claims) — honest but coarse.

- vs LSSim: **ext>A** but **ext-incompat as a dependency** (C++ ASTRA-sim, subprocess, owns its world, can't
  join Aurelius MPC rollouts) → **ext=val**. Per-link fidelity without production fabric telemetry is **PROP**;
  a finer-but-still-public topology model is **A-custom**.

### 6. Queueing / admission / routing — *who is more realistic?*

**Aurelius is competitive-to-ahead.** Its per-request DES queue (`unified_replay`, no M/M/1), class-aware
admission, and four routing policies (round-robin/shortest-queue/kv-aware/affinity) match or exceed
SplitwiseSim/BLIS schedulers and exceed InferSim/Vidur (which have weak/absent routing). Mooncake's Conductor
adds **prediction-based early rejection** (admit iff `TTFT≤SLO ∧ predicted TBT≤SLO`) Aurelius lacks.

- Mostly **A>ext**; Mooncake early-reject + Splitwise KV-aware schedulers: **ext>A** narrowly → **PORT-EQ**
  (early-reject admission test).

### 7. Power / energy — *who is more realistic?*

**Aurelius is ahead.** It has owned-hardware economics (CapEx depreciation + PUE-scaled ISO energy +
active/idle power per GPU type) and goodput/$. LLMServingSim 2.0 reports per-component energy (kJ) and
SplitwiseSim has an empirical power-cap model, but **neither converts to operator $ or goodput/$**.

- vs LSSim energy detail: **ext>A** narrowly on per-op energy granularity → **PORT-EQ** (the
  `(active−idle)·latency` per-op energy form) but **A>ext** overall (economics).

### 8. Counterfactual action evaluation — *who is more realistic?*

**Aurelius wins decisively.** It is the **only** system with a persistent state that is cloned per candidate
and rolled forward under a receding-horizon MPC. SplitwiseSim/LSSim/BLIS do "what-if" only by re-running from
a fresh config — they cannot evaluate "migrate *this* warm replica vs cold-start a new one" because they have
no persistent replica identity across periods.

- **A>ext** across the board. This is the irreplaceable core.

### 9. Who is *less* realistic than Aurelius because it lacks persistent ClusterState?

**All of them.** Every open simulator (LSSim, BLIS, SplitwiseSim, InferSim, Vidur, vLLM) rebuilds its world
from config at `t=0` and tears it down at run end. None models warm/cold replica lifecycle + migration +
placement on a state that persists across control periods. That is exactly Aurelius' PR #99–#107 contribution.

### 10. Which pieces should Aurelius adopt *immediately*?

| priority | adopt | from | mechanism | how |
|--|--|--|--|--|
| 1 | **FLOP/bandwidth roofline** | InferSim + llm-analysis + LLM-Viewer | per-stage `max(compute,mem)` + ridge label | PORT-EQ (**done this PR**, reference model) |
| 2 | **iteration-level continuous batching + chunked prefill** | Orca + Sarathi Alg. 3 + SplitwiseSim | hybrid batch, stall-free admission | PORT-PATTERN (next PR) |
| 3 | **KV transfer cost** | SplitwiseSim + Mooncake | `bytes/bandwidth`, layer-wise overlap | PORT-EQ (next PR) |
| 4 | **prediction-based early rejection** | Mooncake Conductor | admit iff TTFT,TBT ≤ SLO | PORT-EQ (cheap) |
| 5 | **tiered KV (CPU/SSD) cost** | LMCache + LSSim | tier residency + transfer surcharge | PORT-PATTERN (later) |

---

## Bottom line

Aurelius is **already ahead** on the dimensions that define a *fleet world model with counterfactual
control* (workload fidelity, KV residency state, lifecycle actions, economics, MPC). It is **behind** on
*single-replica serving micro-physics* (roofline, true continuous batching, KV transfer). The right move is
exactly the brief's hard rule: **port the equations that close the micro-physics gap (roofline first — done),
keep the persistent-state + MPC core that no open simulator has, and use the heavyweight simulators (LSSim,
BLIS, Vidur) as validation baselines** — never as in-loop dependencies, because none of them can be cloned
per MPC candidate and rolled forward deterministically.
