# Research Adaptation Memo — Phase 4 Run (2026-06-25)

## Status: Five-Failure Rule Active (5/5). All papers reviewed NOT APPLICABLE.

## Papers Reviewed

10 papers reviewed via arXiv search (LLM serving scheduling, GPU cluster optimization,
inference cost reduction, 2024–2025 venue). All NOT APPLICABLE under Five-Failure Rule.

### 1. DynamoLLM (arXiv:2408.00741)
**Venue:** ASPLOS 2025 / arXiv 2024  
**Claim:** Profile-based heterogeneous GPU tier selection for LLM inference cost reduction.  
**NOT APPLICABLE:** Requires heterogeneous GPU fleet profiles (not available in public benchmark physics). New module (profiled GPU tier selector). Oracle profiling data (per-model per-GPU throughput curves). Five-Failure Rule prohibits new modules.

### 2. Llumnix (OSDI 2024)
**Claim:** Live request migration across GPU instances using vLLM's `KVTransfer` API.  
**NOT APPLICABLE:** Requires live KV cache migration infrastructure — new module. Also requires per-instance session routing that the benchmark replay does not model.

### 3. Preble (NSDI 2024)
**Claim:** Global prefix-sharing scheduler; routes requests to GPUs sharing the KV prefix.  
**NOT APPLICABLE:** Requires per-request prefix identity data (not in BurstGPT/Azure LLM traces). New admission/routing module. Five-Failure Rule prohibits.

### 4. AlpaServe (OSDI 2022)
**Claim:** Model-parallel serving with statistical multiplexing across GPU instances.  
**NOT APPLICABLE:** Requires multi-GPU model-parallel layout decisions. New placement module. Orthogonal to per-tick replica sizing.

### 5. Orca (OSDI 2022)
**Claim:** Iteration-level scheduling for LLM inference with continuous batching.  
**NOT APPLICABLE:** Per-token scheduling requires step-level simulation — new module. Benchmark uses aggregated tick-level physics (Erlang-C), not iteration-level.

### 6. SpotLight / IterDet (ASPLOS 2024)
**Claim:** Proactive spot interruption detection and workload migration.  
**NOT APPLICABLE:** Requires hardware telemetry (power/thermal signals) for interruption prediction. New module. Five-Failure Rule prohibits.

### 7. SARATHI-Serve (OSDI 2024)
**Claim:** Chunked prefills to reduce decode stalls and improve utilization.  
**NOT APPLICABLE:** Chunk-level scheduling requires sub-tick prefill modeling — new simulation physics. Benchmark's Erlang-C/EWMA physics do not model chunked prefill at the resolution needed.

### 8. HexGen (arXiv:2311.11514)
**Claim:** Asymmetric partitioning for heterogeneous GPU inference serving.  
**NOT APPLICABLE:** Requires heterogeneous GPU allocation model. New module. Not applicable to homogeneous replica scaling.

### 9. Token-Budget Aware LLM Reasoning (arXiv:2024)
**Claim:** Output-length prediction to reduce reasoning token count.  
**NOT APPLICABLE:** Requires output-length prediction model — oracle information. Five-Failure Rule prohibits oracle claims.

### 10. FlexFlow Serve / SpecInfer (MLSys 2024)
**Claim:** Speculative decoding with tree-structured verification for LLM inference speedup.  
**NOT APPLICABLE:** Requires speculative decoding infrastructure (draft model + verification). New module. Orthogonal to capacity provisioning.

## Conclusion

No paper found applicable under Five-Failure Rule constraints:
- All require either new modules, oracle information, heterogeneous infrastructure, or hardware telemetry
- The Five-Failure Rule mandates: no new modules, focus on integration/architecture

**Next direction selected:** Phase 4 causal frontier rho adaptation (integrating existing `frontier/estimator.py` with causal rolling window — no new module, no oracle data).

**Phase 4 result:** NULL on fixtures. Implementation retained for production-scale evaluation.
