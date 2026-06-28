# Full World-Model Feasibility Assessment (Phase 6)

Can the open simulators combine into one production-like AI-inference fleet world model for Aurelius?
Answers are blunt and tied to the Phase-0/3 source evidence.

---

### 1. Is there any single open-source simulator that already provides the full AI-inference fleet simulation Aurelius needs?

**No.** The closest are **LLMServingSim 2.0** and **BLIS** (both model serving phases + KV tiers + batching +
network + some power), but **neither** has: a persistent fleet `ClusterState` that survives across control
periods; warm/cold replica lifecycle + migration + placement as first-class state transitions; operator-side
economics (CapEx depreciation + ISO energy + goodput/$); or counterfactual MPC over that state. Every open
simulator rebuilds its world from config at `t=0` and discards it at run end. They are **serving
micro-simulators**, not **fleet world models with control**.

### 2. If yes, should Aurelius use it?

N/A — none qualifies for wholesale adoption. As *validation baselines* and *equation sources*: yes (Phase 4).

### 3. If no, can the open simulators be combined into a full one?

**Partially, and only by porting — not by wiring engines together.** The serving micro-physics (roofline,
continuous batching, KV transfer, disaggregation) can be assembled from InferSim + Sarathi + SplitwiseSim +
Mooncake equations *inside* Aurelius. But the **fleet-lifecycle + economics + MPC layer has no open source to
combine** — it only exists in Aurelius. So the "combination" is: **Aurelius' persistent-state/MPC spine +
ported open serving-physics**, not a federation of external simulators. Federating the engines directly fails
because each owns its world, runs as a subprocess or CUDA runtime, and cannot be cloned per MPC candidate.

### 4. Which parts can be imported / ported?

- **Ported (this PR):** FLOP/bandwidth roofline + ridge classifier (InferSim/llm-analysis/LLM-Viewer).
- **Portable (next PRs):** iteration-level continuous batching + chunked prefill (Orca/Sarathi Alg. 3);
  KV-transfer cost (SplitwiseSim/Mooncake); disaggregation M/D/1 TTFT (DistServe); early-reject admission
  (Mooncake Conductor); radix-prefix-cache + tiered-KV *pattern* (LLMServingSim 2.0/LMCache); per-op energy
  form (LLMServingSim 2.0).
- **Vendored:** Mooncake/Azure traces (already in use).

### 5. Which parts must be built in Aurelius?

- Persistent `ClusterState` + clone-per-candidate (exists).
- Warm/cold/migration/placement/prewarm lifecycle physics (exists).
- Receding-horizon MPC + economics/objective (exists).
- The **glue** that maps the ported roofline onto continuous-batching occupancy and the per-request DES queue
  (new). The roofline gives a per-token floor; coupling it to batch occupancy + queueing is Aurelius-specific.

### 6. Which parts cannot be modelled faithfully without production telemetry? (PROP)

Identical across every public trace and repo — **structurally proprietary, pilot-only:**
- real **per-replica KV residency / eviction state** (no trace exposes live cache contents);
- real **measured per-request cache hit rate** (Mooncake gives a *process*, not labels; ~50% is an aggregate);
- real **cross-node KV-transfer bandwidth under production congestion** (specs give peak, not realized);
- true **per-request model identity** and **prompt content** (withheld by design in Azure/Alibaba);
- real **internal operator $/GPU-hr, energy draw, carbon intensity** (modelled from ISO + spec priors);
- real **per-link / NVLink fabric contention** (Alibaba v2026 is hourly macro rx/tx only).

These must be **modelled and tier-labelled** (BENCHMARK_DERIVED / INFERRED / ABSENT), never claimed as MEASURED.

### 7. Is current Aurelius already more complete in any dimension?

**Yes — in the dimensions that matter most for its purpose:** workload-trace fidelity (Azure+Mooncake+Alibaba
fusion without row-joins), persistent per-replica KV residency state, fleet lifecycle actions
(prewarm/migration/placement), operator economics + goodput/$, and counterfactual MPC. No open simulator has
any of these together; most have none.

### 8. Is an integrated Aurelius world model still justified?

**Yes, decisively.** The unique value — persistent state + counterfactual MPC + operator economics — exists
nowhere else and is the entire point. Porting open serving-physics *into* that spine is strictly additive and
closes the only real deficits (roofline, batching, KV transfer). Rebuilding on top of an external engine would
*lose* the persistent-state/MPC core and gain nothing the ports don't already provide.

### 9. Minimum implementation plan to reach "production-like enough"

Against the standard below, the minimum path is:
1. **Roofline service time (done this PR — reference).** Wire `roofline_external` into `prefill_decode` behind
   a flag, per (model, GPU); keep constants as fallback. *Closes Group E.*
2. **Iteration-level continuous batching + chunked prefill** (port Sarathi Algorithm 3 + Orca pattern). *Closes
   the biggest Group B/D gap.*
3. **KV-transfer cost** (port SplitwiseSim/Mooncake `bytes/bw` + layer-wise overlap) + a `disaggregated` flag
   (DistServe M/D/1 TTFT). *Closes Group B KV-handoff.*
4. **Early-reject admission** (Mooncake `TTFT,TBT ≤ SLO`). *Cheap Group D win.*
5. **Validation harness** comparing Aurelius vs Vidur/LLMServingSim-2.0/SplitwiseSim on shared fixtures.
6. **Carbon** (DERIVABLE from ISO + grid intensity) — small Group G add.

Each step is a port with a controlled-fixture test; none adds a runtime dependency or breaks PCS/MPC/DET.

### 10. What claims are safe / unsafe?

**Safe:**
- "Aurelius is a directional, public-data-grounded counterfactual fleet world model with a FLOP/bandwidth
  roofline (ported from InferSim/llm-analysis/LLM-Viewer), a stateful per-replica LRU KV cache, persistent
  replica lifecycle, and receding-horizon MPC."
- "Every parameter carries a fidelity tier (TRACE_DERIVED / BENCHMARK_DERIVED / INFERRED / ABSENT)."
- "More complete than any open simulator on fleet lifecycle, economics, and counterfactual control."

**Unsafe (do not claim):**
- "Production-accurate / exact" latency or cost — the roofline is a physical floor with constant MFU; real
  serving has per-kernel MFU, tile-quant, contention not modelled (PROP).
- "Measured" cache hit rates, KV residency, transfer bandwidth, or operator cost — all modelled, not observed.
- "Per-link / NVLink topology" — only macro hourly rx/tx is public.

---

## Standard applied — "production-like enough"

| criterion | status |
|--|--|
| every major serving transition has state | ✓ (persistent ClusterState; replica/migration identity) |
| every action mutates state causally | ✓ (prewarm/migrate/place/route mutate world, cloned for MPC) |
| every transition affects TTFT/latency/GPU-s/mem/net/power/SLA/cost | ✓ (service-time → DES queue → goodput/$) |
| every parameter has a trace/paper/spec/benchmark/sim-inferred source | ✓ (fidelity tiers; roofline now BENCHMARK_DERIVED) |
| every approximation has a validation test | ✓ for roofline (12 fixtures); batching/KV-transfer tests come with their PRs |
| every missing proprietary signal is named | ✓ (PROP list in §6) |

**Verdict: feasible to reach "production-like enough" with public data — as Aurelius' persistent-state/MPC
spine plus ported open serving-physics. Exact production fidelity is not feasible without pilot telemetry
(the §6 PROP signals), and that limit is named, not hidden.**
