# Production GPU-Scheduler Public-Benchmark Research

What real, published production GPU schedulers do — so the Aurelius `production_scheduler` baseline can be
grounded in (and named after) recognizable systems rather than reading as an ad-hoc strawman. This is a
literature/产品 survey to answer: *is `production_scheduler` a fair, standard public baseline, and how should it
be worded?* Every source is recorded with its URL, type, and capability matrix. Searched June 2026; capability
claims are from the linked primary/official sources. (Diagnostic research only — no code/physics change.)

## How to read this

Two distinct layers appear in production, and conflating them is the main way a baseline looks cherry-picked:

- **Cluster/orchestration schedulers** (Slurm, KAI, Run:ai, Volcano, Kueue, SUNK, Slinky): place *jobs/pods* on
  GPUs with gang scheduling, quota, fairness, topology awareness. Mostly **training**-oriented. They do **not**
  do continuous batching, KV routing, or latency-SLA serving.
- **Inference serving schedulers** (vLLM, Dynamo, KServe, Ray Serve, llm-d, AIBrix): schedule *requests* within
  a serving fleet with continuous batching, KV-cache-aware routing, autoscaling, and (increasingly) SLO
  planning. This is what `production_scheduler` actually models.

Aurelius `production_scheduler` is a composite of the **serving** layer (continuous batching + KV routing +
backlog autoscaling + SLA-aware admission) plus the topology/placement of the cluster layer — i.e. the union of
what the systems below ship — **minus** the electricity/precision/clock economic arbitrage, which **none** of
them do (verified below). That is the honest boundary.

## Source records (capability matrix)

Columns: **T/I** = training / inference / both · **Topo** = topology-aware · **Quota** = quota/fairness ·
**CB/KV** = continuous batching / KV-cache routing · **SLA** = deadline/SLO-aware · **Econ** = electricity-price
/ precision / clock economic arbitrage · **Type** = impl / product-doc / paper / benchmark · **Use** = suitable
as a `production_scheduler` baseline component.

| # | System (org) | algorithm / architecture | T/I | Topo | Quota | CB/KV | SLA | Econ | Type | Use |
|--|--|--|--|--|--|--|--|--|--|--|
| 1 | **NVIDIA KAI Scheduler** | K8s-native batch sched; DRF + time-decay fairshare; gang (podgroups); GPU sharing; DRA; topology-aware (TAS) | both | ✅ | ✅ | ❌ | ❌ | ❌ | OSS impl + product | cluster layer: gang/quota/fairness/topology |
| 2 | **NVIDIA Run:ai** | fractional GPU; deserved-quota + fairshare; NVLink/rack/block topology; priority preemption | both | ✅ | ✅ | ❌ | ❌ | ❌ | product (KAI = OSS core) | quota/fairness/fractional layer |
| 3 | **vLLM** | serving engine; PagedAttention; continuous batching; chunked prefill; 3-queue (waiting/running/swapped) sched; priority | inf | ❌ | ❌ | ✅ | ~ (priority/decode-first) | ❌ | OSS impl + docs | **the continuous-batching/KV core** `production_scheduler` approximates |
| 4 | **NVIDIA Dynamo** (SLA Planner + KV Router) | disaggregated PD; **Planner forecasts traffic (time-series) + monitors GPU/queue metrics → adjusts PD worker counts to meet TTFT/ITL SLO within a GPU budget**; KV-aware router (overlap score) | inf | ✅ | ❌ | ✅ | ✅ | ❌ (verified: no price/precision/DVFS) | OSS framework + product | **closest production analog to Aurelius's forecast+capacity+routing** |
| 5 | **KServe** (CNCF) | LLMInferenceService CRD; KEDA autoscale on queue depth + KV util; disaggregated PD; prefix-cache-aware routing; vLLM underneath | inf | ~ | ❌ | ✅ | ~ (autoscale-to-latency) | ❌ | OSS platform | serving-platform layer: KV routing + autoscaling |
| 6 | **Ray Serve** (Anyscale) | request routing; `PrefixCacheAffinityRouter`; autoscale on queue depth + KV util; custom routing | inf | ❌ | ❌ | ✅ | ~ | ❌ | OSS + product | KV-aware routing + autoscaling |
| 7 | **Volcano** (CNCF) | K8s batch; gang; DRF; hierarchical queues; preemption; bin-packing; GPU sharing | train | ~ | ✅ | ❌ | ❌ | ❌ | OSS impl | batch/gang/fairness layer |
| 8 | **Kueue** (K8s SIG) | admission control + quota; decides admit/wait/preempt; topology-aware flavors | train/batch | ~ | ✅ | ❌ | ❌ | ❌ | OSS impl | **admission/quota layer** (maps to `admission=class_aware`) |
| 9 | **CoreWeave SUNK** | Slurm-on-K8s; tree/block topology plugins; NVLink-domain (GB200 NVL72) placement | train | ✅ | ✅ (Slurm) | ❌ | ❌ | ❌ | product-doc + blog | topology/placement layer (training) |
| 10 | **Crusoe** (Managed Slurm/CMK + KServe) | managed Slurm + K8s; block/topology scheduling; KServe inference (6k tok/s blog) | both | ✅ | ✅ (Slurm) | ✅ (via KServe) | ❌ | ~ infra-level cheap energy, **not** scheduler arbitrage | product-doc + blog | neocloud reference (infra energy ≠ scheduler $-opt) |
| 11 | **Lambda Labs** (1-Click Clusters) | managed Slurm + MK8s; InfiniBand/SHARP topology | both | ✅ | ✅ (Slurm) | ❌ | ❌ | ❌ | product-doc | neocloud managed-Slurm reference |
| 12 | **Slinky / SchedMD** (NVIDIA) | Slurm-on-K8s operator; topology-aware; DRA/ComputeDomains | train | ✅ | ✅ (Slurm) | ❌ | ❌ | ❌ | OSS impl | Slurm-on-K8s reference |
| 13 | **MLPerf Inference — Server** (MLCommons) | standardized benchmark; Poisson arrivals; **goodput = throughput meeting TTFT p99 ≤ 6 s & TPOT p99 ≤ 175 ms SLA**; reports QPS | inf | n/a | n/a | (SUT) | ✅ (defines the SLA) | ❌ ($ not in metric) | **benchmark (the standard)** | **the standardized SLA/goodput definition to anchor to** |
| 14 | **Academic SLO-serving** (DistServe OSDI'24; QLM/queue-mgmt SoCC'24; JITServe'25; "Optimal Scheduling for LLM Inference" 2025; Sarathi/chunked-prefill) | SLO-aware request scheduling; disaggregated PD; goodput-optimized; SRPT-style | inf | ~ | ❌ | ✅ | ✅ | ❌ | papers | grounding for the SLA-aware ordering/admission (== `sla_aware`) |
| 15 | **Energy/price/carbon-aware serving** (Green-LLM; FREESH; thermal-aware; carbon-aware routing) | energy/carbon/price-aware workload allocation & routing | inf | ~ | ❌ | ~ | ~ | ✅ (research) | papers | grounding for Aurelius's **economic** layer — research, **not** production |

### The decisive finding (verified at the source)

**No production serving scheduler does economic arbitrage.** NVIDIA Dynamo's Planner — the most advanced
production system, and the closest analog to Aurelius — "continuously monitors key GPU capacity metrics … and
combines them with application SLOs such as TTFT and ITL," deciding "whether to serve … with or without
disaggregation or if additional GPUs should be added," considering "KV cache transfer … queue wait times …
estimated processing times." It does **not** address "electricity pricing, model precision selection, or
DVFS/clock-speed arbitrage" (verified from the NVIDIA technical blog). Run:ai/KAI optimize **quota/fairness**,
not dollars. Crusoe's cost story is **infrastructure-level cheap/stranded energy**, not a scheduler that
arbitrages price/precision/clock per decision. **The electricity-price + precision + clock economic layer is
exactly what Aurelius adds on top of an otherwise production-standard serving stack** — and it only appears in
research papers (Green-LLM, FREESH), never in a shipped scheduler. This is the honest, defensible boundary for
the headline.

## Does `production_scheduler` faithfully represent these systems?

| `production_scheduler` lever | grounded in |
|--|--|
| continuous batching (always on) | vLLM (#3), the universal serving primitive |
| KV-aware routing | Dynamo KV Router (#4), Ray `PrefixCacheAffinityRouter` (#6), KServe (#5) |
| backlog autoscaling + headroom | Dynamo Planner (#4), KServe/Ray KEDA-on-queue-depth (#5,#6) |
| SLA-aware ordering (SRPT-conformal) | academic SLO-serving (#14), == `sla_aware` |
| class-aware admission under pressure | Kueue admission/quota (#8), vLLM priority (#3) |
| rack-local / topology placement | KAI TAS (#1), Run:ai (#2), SUNK/Slinky (#9,#12) |
| **NOT** precision/clock/migration/price arbitrage | **none** of the above — Aurelius's edge |

So `production_scheduler` is a **faithful union of the production serving stack**, not a strawman: every lever
it uses maps to a shipped system, and the levers it omits are omitted by all of them too.

## Recommendations (the four asks)

1. **Should `production_scheduler` remain the primary public baseline?** **Yes — but name it after what it is.**
   It faithfully composes the production serving stack (vLLM continuous batching + Dynamo/KServe/Ray KV-routing
   & autoscaling + Kueue-style admission + KAI/Run:ai/Slurm topology). Publicly, frame it as *"a production-class
   serving baseline (continuous batching + KV-aware routing + backlog autoscaling + SLA-aware admission +
   topology placement) — the vLLM / NVIDIA Dynamo / KServe stack."* Citing Dynamo's SLA Planner specifically is
   the strongest anti-cherry-pick move: it shows the baseline matches the *most advanced* production scheduler,
   not a weak one.

2. **Should `sla_aware` also be reported?** **Yes, as the secondary/conservative bar.** It is the
   SRPT-conformal latency scheduler — the academic SLO-serving analog (DistServe/JITServe/QLM family, #14).
   Reporting **both** brackets the claim honestly: `sla_aware` = "a research-grade latency scheduler,"
   `production_scheduler` = "the full production serving stack." Aurelius beating both (it does: +164% / +148%)
   reads as robust, not baseline-shopped.

3. **Is there a more standardized public baseline?** **Yes — anchor the METRIC to MLPerf Inference (Server).**
   MLPerf defines the standardized *goodput-under-SLA* (TTFT p99 ≤ 6 s, TPOT p99 ≤ 175 ms; Poisson arrivals),
   which is precisely Aurelius's "SLA-safe goodput" numerator. Recommendation: state that `production_scheduler`
   serves at MLPerf-style goodput-under-SLA and that Aurelius's contribution is the **per-dollar** denominator +
   economic optimization MLPerf does not score. This grounds both the baseline (Dynamo/vLLM systems) and the
   metric (MLPerf goodput) in recognizable standards. (Aurelius's own SLA is the repo's `sla_s` deadline, not
   MLPerf's exact thresholds — say so; don't claim MLPerf conformance.)

4. **Baseline wording least likely to look skeptical / cherry-picked.** Avoid "we beat production_scheduler by
   +148%." Prefer: *"Against a production-class serving baseline — continuous batching, KV-cache-aware routing,
   backlog autoscaling, and SLA-aware admission, the levers shipped by vLLM, NVIDIA Dynamo, KServe and Ray Serve
   — Aurelius improves SLA-safe goodput-per-dollar by adding economic optimization (price-responsive clock,
   lossless-safe precision, capacity consolidation) that no production scheduler performs. We report against
   both this stack and a research-grade SRPT latency scheduler (`sla_aware`); gains are SIMULATED directional
   evidence, bounded to the windows tested."* This (a) names real systems, (b) cites the strongest one (Dynamo),
   (c) localizes the edge to a clearly-additional capability, (d) reports two baselines, (e) labels fidelity.

## Honest caveats

- **Magnitudes are SIMULATOR_INFERENCE.** This survey grounds the baseline's *design realism*, not the
  simulator's number. The +148%/+164% is bounded directional evidence (`HIERARCHICAL_PLANNER_PRODUCTION_COMPARISON.md`).
- **Not a head-to-head against the real systems.** We did not run vLLM/Dynamo/KServe and measure them; we built
  a baseline whose *policy* matches their published behavior. A real head-to-head (e.g. an MLPerf Server harness
  with vLLM as the system-under-test) is the strongest future validation and is called out as such.
- **PagedAttention block-level KV and chunked-prefill are not modeled** in the repo (only a prefix-hit prefill
  reduction); `production_scheduler` does not claim them. The continuous-batching concurrency model is the safe
  approximation (noted in `PRODUCTION_SCHEDULER_BASELINE_AUDIT.md`).

## Sources

- NVIDIA KAI Scheduler — https://github.com/NVIDIA/KAI-Scheduler ; https://developer.nvidia.com/blog/nvidia-open-sources-runai-scheduler-to-foster-community-collaboration/
- NVIDIA Run:ai — https://run-ai-docs.nvidia.com/saas/platform-management/runai-scheduler/scheduling/concepts-and-principles ; https://run-ai-docs.nvidia.com/guides/platform-management/runai-scheduler/resource-optimization/fractions
- vLLM — https://github.com/vllm-project/vllm ; https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
- NVIDIA Dynamo — https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/ ; https://developer.nvidia.com/dynamo
- KServe — https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm ; https://medium.com/@simardeep.oberoi/building-production-llm-infrastructure-with-kserve-v0-15-a5eecb2311bc
- Ray Serve — https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing ; https://www.anyscale.com/blog/ray-serve-autoscaling-async-inference-custom-routing
- Volcano / Kueue — https://www.infracloud.io/blogs/batch-scheduling-on-kubernetes/ ; https://medium.com/@charleswan111/volcano-a-kubernetes-native-batch-scheduler-for-high-performance-workloads-a936014032ec
- CoreWeave SUNK — https://www.coreweave.com/products/sunk ; https://docs.coreweave.com/docs/products/sunk/optimize_workloads/topology-scheduling
- Crusoe — https://docs.crusoecloud.com/orchestration/slurm/overview/index.html ; https://www.crusoe.ai/resources/blog/serving-llms-on-crusoe-with-kserve
- Lambda Labs — https://lambda.ai/1-click-clusters ; https://lambda.ai/blog/lambda-managed-slurm
- Slinky / SchedMD — https://slurm.schedmd.com/slinky.html ; https://github.com/SlinkyProject/slurm-operator
- MLPerf Inference — https://mlcommons.org/2025/04/llm-inference-v5/ ; https://docs.mlcommons.org/inference/
- Academic SLO-serving — https://arxiv.org/html/2504.14966 (SLO-Aware Scheduling) ; https://users.ece.utexas.edu/~gustavo/papers/BHD25j.pdf (Optimal Scheduling for LLM Inference) ; https://arxiv.org/pdf/2504.20068 (JITServe)
- Energy/price/carbon-aware — https://arxiv.org/pdf/2507.09942 (Green-LLM) ; https://arxiv.org/pdf/2511.00807 (FREESH)
