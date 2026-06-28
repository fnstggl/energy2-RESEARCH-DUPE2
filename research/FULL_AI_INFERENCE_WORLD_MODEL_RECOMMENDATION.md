# Full AI-Inference World-Model — Final Recommendation (Phase 8)

Plain-English, blunt answers. Evidence is in Phases 0–6 and the implemented roofline
(`aurelius/environment/roofline_external.py`, validated in `tests/test_roofline_external.py`).

---

### 1. Has anyone built this exact simulator already?

**No.** Nobody has published a persistent-state, counterfactual-MPC, operator-economics fleet world model
for LLM inference. The open projects are **serving micro-simulators** (single run from a config, torn down
at the end). The "world model + control + economics" layer that defines Aurelius exists nowhere else in open
source.

### 2. Which open project is closest?

**LLMServingSim 2.0** (most complete serving stack: phases + radix prefix cache + tiered KV + chunked-prefill
batching + ASTRA-sim network + per-component energy) and **BLIS / inference-sim** (clean DES + roofline +
trained-physics latency, CPU-only, deterministic). For *roofline only*, **Alibaba InferSim**. For *accuracy*,
**Vidur** (profiled RandomForest). None is close on fleet lifecycle, economics, or MPC.

### 3. What does each project model well?

| project | models well |
|--|--|
| LLMServingSim 2.0 | radix prefix cache, tiered KV (NPU/CPU/CXL), chunked prefill, ASTRA-sim network, energy (kJ) |
| BLIS | DES + dual latency back-ends (roofline + trained-physics), MoE/TP comm, KV thrash metrics, decision tracing |
| InferSim | per-stage FLOP/bandwidth roofline (prefill vs decode), fp8, calibrated vs real hw |
| SplitwiseSim | prompt/decode-pool disaggregation, KV-transfer cost, power caps, profiled-table timing |
| Vidur | most accurate latency (RandomForest on profiled data, <9% error), real schedulers |
| llm-analysis / LLM-Viewer | analytical roofline + explicit ridge point + GPU table with HBM capacity |
| Mooncake / LMCache | KVCache-centric disaggregation, prefix-chained hashing, tiered KV, CacheBlend non-prefix reuse |
| vLLM | the reference LRU paged KV cache + continuous-batching scheduler |

### 4. What does each project miss?

All miss the same things: **persistent cross-period ClusterState, warm/cold/migration/placement lifecycle,
operator $ economics + goodput/$, and counterfactual MPC.** Additionally: InferSim/llm-analysis/LLM-Viewer
miss batching/queueing (pure calculators); SplitwiseSim misses KV eviction/prefix reuse (bulk bytes) and is
non-re-entrant (global singletons); LLMServingSim 2.0/BLIS are heavyweight (C++/Go, own their world);
LLMRoofline misses a license, prefill, and latency-in-seconds.

### 5. Is any project more realistic than Aurelius today?

**Yes — on single-replica serving micro-physics:** InferSim/BLIS (FLOP/bandwidth roofline), vLLM/LSSim/
SplitwiseSim/Vidur (true iteration-level continuous batching), LSSim/SplitwiseSim/Mooncake (KV-transfer cost),
LSSim (ASTRA-sim network). **No project is more realistic than Aurelius** on workload-trace fusion, KV
residency *state*, fleet lifecycle, economics, or counterfactual control. The roofline gap is the most
material — and it is closed (as a reference model) in this PR.

### 6. Can they be combined into the full simulator?

**Only by porting equations into Aurelius' spine — not by federating engines.** Each engine owns its world,
runs as a subprocess or CUDA/Go runtime, and cannot be cloned per MPC candidate or rolled forward
deterministically (fails the PCS/MPC/DET gates in Phase 4). The viable "full simulator" is **Aurelius'
persistent-state + MPC + economics core, plus ported open serving-physics** (roofline ✓ now; batching,
KV-transfer, disaggregation next).

### 7. Should Aurelius import, vendor, port, or only reference them?

- **Port (equations):** InferSim + llm-analysis + LLM-Viewer (roofline — done); Sarathi Alg. 3 + Orca
  (batching); SplitwiseSim + Mooncake (KV-transfer); DistServe (M/D/1 disaggregation); Mooncake (early-reject).
- **Vendor (data):** Mooncake + Azure traces (already in use; Apache-2.0 / CC-BY).
- **Reference only (validation baselines):** LLMServingSim 2.0, BLIS, Vidur, SplitwiseSim — compare outputs,
  never import (all fail PCS/MPC/DET as dependencies).
- **Reject:** LLMRoofline (no license); runtime engines (LMCache/KVServe/Mooncake-TE/vLLM/llm-d) as deps.

### 8. Should Aurelius use LLMRoofline?

**No.** No license (all-rights-reserved, HIGH risk), decode-only, no latency-in-seconds, no HBM capacity,
stale, hardcoded local paths. Use **InferSim + llm-analysis + LLM-Viewer** instead — all permissive, active,
complete. (Full reasoning: `ROOFLINE_REUSE_DECISION.md`.)

### 9. What should the next implementation PR actually build?

In priority order (each a port + controlled-fixture test, no new runtime dependency):
1. **Wire the roofline into the live service path** — `roofline_external.{prefill,decode}_estimate` feeding
   `prefill_decode.compute_phase_serving` per (model, GPU), constants kept as the conservative fallback.
2. **Iteration-level continuous batching + chunked prefill** (Sarathi Algorithm 3 + Orca pattern) — the
   biggest remaining serving-physics gap.
3. **KV-transfer cost + a `disaggregated` flag** (SplitwiseSim/Mooncake `bytes/bw` + layer-wise overlap;
   DistServe M/D/1 TTFT).
4. **Early-reject admission** (Mooncake `TTFT,TBT ≤ SLO`) — cheap.
5. **External validation harness** comparing Aurelius vs Vidur / LLMServingSim 2.0 / SplitwiseSim on shared
   fixtures (the `external_sim_validation.py` seam added this PR generalizes to these).
6. **Carbon** (DERIVABLE: ISO price × grid-intensity series).

### 10. Is the "full production-like world model" feasible with public data?

**Yes — "production-like enough" is feasible** (Phase 6 standard met: stateful transitions, causal actions,
service-time → economics, tier-labelled parameters, validation tests, named proprietary gaps). **Exact
production fidelity is not feasible without pilot telemetry.**

### 11. What remains impossible without pilot telemetry? (PROP)

Real per-replica KV residency/eviction state; real measured per-request cache hit rates; real cross-node
KV-transfer bandwidth under production congestion; true per-request model identity + prompt content; real
internal operator $/GPU-hr, energy draw, and carbon intensity; real per-link/NVLink fabric contention. These
are absent from **every** public trace and repo and must stay modelled-and-labelled, never claimed as measured.

---

## Blunt bottom line

- **A prebuilt drop-in does not exist.** The closest (LLMServingSim 2.0, BLIS) are serving micro-simulators
  that own an ephemeral world and cannot host counterfactual MPC.
- **Aurelius remains justified** — its persistent-state + MPC + economics core is the unique, irreplaceable
  contribution; no open project has it.
- **The honest deficits were the roofline (now closed as a reference model), continuous batching, and
  KV-transfer.** Close them by *porting equations* from permissively-licensed sources (InferSim/Sarathi/
  SplitwiseSim/Mooncake), keep the open heavyweights as validation baselines, and reject LLMRoofline and all
  runtime engines as dependencies.
- **Do not overclaim.** The model is directional and public-data-grounded, not production-accurate; the
  proprietary signals that would make it exact are named and remain pilot-only.
