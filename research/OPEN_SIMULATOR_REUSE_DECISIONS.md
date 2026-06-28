# Open Simulator Reuse Decisions (Phase 4)

For each external project: the decision (A import / B vendor module / C port equations / D port design
pattern / E validation baseline / F reject) with the required justification — why, license, dependency
burden, test burden, integration burden, expected realism gain, risk to Aurelius architecture, and the
four hard architecture gates:

- **PCS** — preserves persistent ClusterState? (does it leave Aurelius' world state authoritative?)
- **MPC** — can participate in MPC rollouts? (cheap, side-effect-free, callable per cloned candidate?)
- **DET** — deterministic, clone-safe? (no global singletons / RNG / subprocess / network at run time?)

A reuse is only safe for the in-loop world model if **PCS + MPC + DET all hold**. Anything failing them is
demoted to E (validation baseline) regardless of its realism.

---

## Alibaba InferSim → **C (port equations)** ✅ primary

- **Why:** the cleanest, license-clean, dependency-free FLOP/bandwidth roofline; closes Aurelius' single
  clearest deficit (no hardware roofline). KV-byte formula already matches Aurelius.
- **License:** Apache-2.0 (LOW). **Deps:** none added (pure-Python formulas re-implemented).
- **Test burden:** LOW — 12 controlled fixtures added (`tests/test_roofline_external.py`).
- **Integration burden:** LOW — stateless calculator, composes beside `prefill_decode.py`.
- **Realism gain:** HIGH — resolves the 4–40× GPU×model latency spread a scalar constant cannot.
- **Arch risk:** LOW. **PCS ✓ MPC ✓ DET ✓** (pure function, no state, deterministic).
- **Done this PR:** `aurelius/environment/roofline_external.py`.

## llm-analysis → **C (port equations)** — ridge point + GPU JSON with HBM capacity

- **Why:** explicit `get_pivot()` ridge classifier and a GPU spec schema that *includes capacity* (InferSim's
  dataclass omits it). Complements InferSim. **License:** Apache-2.0 (LOW–MED, keep NOTICE). **Deps:** none added.
- **Test/integration burden:** LOW. **Realism gain:** MED (compute/memory-bound labels). **Arch risk:** LOW.
- **PCS ✓ MPC ✓ DET ✓.** **Done this PR** (ridge label folded into `roofline_analyze`; GPU capacity in `GPU_SPECS`).

## LLM-Viewer → **C (port equations)** — `roofline_analyze` label

- **Why:** the cleanest standalone ridge-point classifier. **License:** MIT (LOW). **Deps:** none.
- **Burden:** LOW. **Gain:** MED. **PCS ✓ MPC ✓ DET ✓.** **Done this PR.**

## BLIS / inference-sim → **C (port equations) + E (validation baseline)**

- **Why:** the trained-physics β-corrected step-time is the most *production-tuned* public analytical model;
  worth porting the basis-function structure later and using the roofline back-end as a cross-check. **Go** →
  re-implement, don't FFI. **License:** Apache-2.0 (LOW). **Deps:** none added (math only).
- **Burden:** MED (Go transcription). **Gain:** MED-HIGH (β-correction over naive roofline). **Arch risk:** LOW.
- **PCS ✓ MPC ✓ DET ✓** for the ported formulas; the **Go engine itself is F (reject)** — fails DET/MPC
  (subprocess, owns its world). **Defer port to a follow-up**; use as a validation reference now.

## SplitwiseSim → **C (port equations) + D (port pattern) + E (baseline)**

- **Why:** layer-wise KV-transfer cost (`bytes/bw`, 10× overlap), mixed-batch ×1.1, and prompt/token-pool
  disaggregation are directly portable; its profiled `DatabasePerformanceModel` is a clean optional oracle.
- **License:** MIT (LOW; profiling CSVs CC-BY). **Deps:** scipy only if the table oracle is adopted (else none).
- **Burden:** MED. **Gain:** MED (KV transfer + disaggregation). **Arch risk:** MED — its **module-level
  singletons (`sim`, `performance_model`) fail DET/MPC**, so import the *engine* is **F**; port the equations.
- **PCS ✓ (eqs) · MPC ✓ (eqs) · DET ✓ (eqs)** / engine ✗. **Decision: port KV-transfer eqs next PR.**

## LLMServingSim 2.0 → **D (port pattern) + E (validation baseline)**

- **Why:** the best public reference for radix-tree prefix caching + tier-aware eviction cost + chunked-prefill
  scheduling + per-component energy. **License:** MIT (LOW). **Deps:** ASTRA-sim/Chakra C++ build (HEAVY) →
  **do not import**. **Burden:** HIGH (full) / MED (lift Python modules `scheduler.py`/`memory_model.py`/
  `radix_tree.py`). **Gain:** MED-HIGH (network + tiered KV). **Arch risk:** HIGH as a dependency.
- **PCS ✗ MPC ✗ DET ✗ as a whole** (subprocess, owns its world, C++). **Decision: validation baseline + port
  the radix/eviction *pattern* into Aurelius' own LRU cache; reject wholesale import.**

## Vidur → **A (import-optional, out-of-loop) / E (validation baseline)**

- **Why:** highest *accuracy* (RandomForest on profiled data, <9% error), CPU-only, MIT, deep-copyable Python.
  Could feed an out-of-loop calibration of Aurelius' roofline MFU/constants. **License:** MIT (LOW). **Deps:**
  numpy/pandas/sklearn (MEDIUM if imported). **Burden:** MED. **Gain:** MED (calibration anchor).
- **Arch risk:** MED — predictor is **not cheap enough for per-candidate MPC rollouts** (sklearn inference) and
  is coupled to its own `Batch`/`Request`; **MPC ✗** in-loop. **Decision: use its profiling CSVs + predictor as
  an offline calibration/validation baseline, not an in-loop dependency.** **PCS ✓ DET ✓ (offline) · MPC ✗ (in-loop).**

## Mooncake → **B (vendor trace, in use) + C (port Conductor equations); F (reject Transfer Engine)**

- **Why:** the `hash_ids` trace is already Aurelius' KV-reuse process (Apache-2.0 — clean to vendor); the
  Conductor `TTFT = T_transfer + T_queue + T_prefill` decomposition and early-reject SLO test are cheap, portable
  scheduler upgrades. The Transfer Engine is C++/RDMA hardware → reject. **License:** Apache-2.0 (LOW).
- **Burden:** LOW (eqs). **Gain:** MED (early-reject admission). **Arch risk:** LOW. **PCS ✓ MPC ✓ DET ✓** (eqs).
- **Granularity note:** document the 512-tok (Mooncake) ↔ 16-tok (Aurelius) block mapping.

## vLLM / PagedAttention → **C (port equations) + D (port pattern)**

- **Why:** the canonical LRU paged-cache reference Aurelius already mirrors; reuse its KV-bytes/token and
  memory-bound batch-ceiling formulas. **License:** Apache-2.0 (LOW). **Deps:** none (re-implement; engine is
  CUDA/torch → reject as dependency). **Burden:** LOW. **Gain:** LOW-MED (confirms/extends existing cache).
  **PCS ✓ MPC ✓ DET ✓** (eqs). **Decision: keep mirroring; no import.**

## DistServe → **C (port equation) + D (port pattern)**

- **Why:** the M/D/1 prefill-TTFT model and `simdistserve` simpy structure are the cleanest disaggregation
  references. **License:** Apache-2.0 (LOW). **Deps:** none (eqs). **Burden:** LOW. **Gain:** MED (disagg TTFT).
  **PCS ✓ MPC ✓ DET ✓** (eqs). **Decision: port M/D/1 TTFT when adding a `disaggregated` flag.**

## Sarathi-Serve → **D (port pattern: Algorithm 3) + E (baseline)**

- **Why:** stall-free chunked-prefill batching is the correct continuous-batching admission model. **License:**
  Apache-2.0 (LOW). **Burden:** MED. **Gain:** HIGH (the biggest serving-physics gap). **PCS ✓ MPC ✓ DET ✓.**
  **Decision: port Algorithm 3 into the iteration-level batching upgrade (next PR).**

## Orca → **D (port pattern only)**

- **Why:** the original continuous-batching + selective-batching idea. **No licensed artifact** — clone is
  unlicensed (**F reject code**). **Decision: implement the *pattern* from the paper; never copy the clone.**

## Reject outright (not simulators / unsafe to depend on)

| project | decision | reason |
|--|--|--|
| **LLMRoofline** | **F reject** | no license (HIGH risk), decode-only, no latency-in-seconds, no capacity; superseded by InferSim/LLM-Viewer |
| **LMCache** (runtime) | **F reject dep** (port CacheBlend pattern only *if* non-prefix reuse needed) | CUDA-bound runtime; fails DET/MPC |
| **KVServe** | **F reject** | a KV-compression vLLM plugin, not a simulator |
| **Mooncake Transfer Engine** | **F reject** | C++/RDMA hardware; fails DET |
| **llm-d-inference-sim** (engine) | **F reject** (port the trivial load-factor eq only) | wall-clock `time.Sleep` mock; fails DET (no simulated clock) |
| **Alibaba clusterdata** raw vendoring | reference at build time | no redistribution license (MEDIUM risk) |

---

## Decision summary

- **Adopt now (this PR):** roofline equations from **InferSim + llm-analysis + LLM-Viewer** → `roofline_external.py`.
- **Adopt next PR:** **Sarathi Algorithm 3** (continuous batching) + **SplitwiseSim/Mooncake** KV-transfer cost +
  **Mooncake** early-reject admission + **DistServe** M/D/1 for a `disaggregated` flag.
- **Validation baselines (never in-loop):** **LLMServingSim 2.0, BLIS, Vidur, SplitwiseSim** — compare Aurelius
  outputs against them on shared fixtures; do not import (all fail PCS/MPC/DET as dependencies).
- **Reject:** LLMRoofline (license), and all runtime engines (LMCache/KVServe/Mooncake-TE/vLLM/llm-d) as
  dependencies — port their equations, reference their designs.

**Architecture guardrail honored:** nothing is imported that does not integrate cleanly with the persistent
`ClusterState` and the clone-per-candidate MPC rollout. The one piece imported in-loop (the roofline) is a
pure deterministic function — it passes PCS/MPC/DET by construction.
