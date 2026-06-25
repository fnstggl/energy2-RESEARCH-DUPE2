# Alibaba GenAI 2026 — Third-Trace Cross-Validation (2026-06-24)

## Classification

**Useful Research / Benchmark Realism**

Third-trace cross-validation on a new workload type (stable diffusion LoRA serving) using
existing infrastructure (genai_backtest.py, genai_ablation.py). Five-Failure Rule compliant:
no new modules, no new optimizer paths — benchmark realism only.

---

## Motivation

Aurelius's LLM-serving results (Azure LLM 2024, BurstGPT) show strong gains from
constraint_aware (AMCSG + model affinity). Both are text-generation traces. A structural
concern exists: are the gains trace-specific, or do they generalize to different serving
workloads?

The Alibaba GenAI 2026 lora_request_trace.csv provides an orthogonal workload:
stable-diffusion LoRA adapters (image generation), multi-model-class, fine-grained
model routing, GPU adapter prewarming. This is a materially different serving pattern
from token-streaming LLM inference.

---

## Dataset

| Field | Value |
|---|---|
| Source | Alibaba cluster-trace-v2026-GenAI (GitHub) |
| URL | https://github.com/alibaba/clusterdata/master/cluster-trace-v2026-GenAI |
| File | lora_request_trace.csv |
| Raw rows | 26,824 |
| Valid requests (after filter) | 26,392 |
| Ticks (60s bins) | 553 |
| Workload type | Stable diffusion LoRA serving (image generation) |
| GPU price | $3.00/hr (consistent with existing GenAI backtest) |
| SLA definition | `2.0 × exec_time_seconds + 30.0s` per request |
| Target utilization | ρ_SLA = 0.65 (Erlang-C M/M/c model) |
| Cold start | Default (~2.79s with affinity, ~22.85s without) |

---

## Ablation Results (Full Factorial)

Full ablation over 10 configs × 26,392 requests. Source: `aurelius/traces/genai_ablation.py`.

| Config | gp/$ | SLA-safe | Timeout% | GPU-hrs | p99-lat |
|---|---|---|---|---|---|
| **constraint_aware** | **9.8514** | **26,392** | **0.000%** | **893** | 53.7s |
| constraint_aware_no_affinity | 7.1291 | 26,392 | 0.000% | 1,234 | 65.9s |
| fifo_plus_affinity | 3.1817 | 26,392 | 0.000% | 2,765 | 35.8s |
| fifo | 1.7676 | 26,392 | 0.000% | 4,977 | 52.6s |
| sla_aware ❌ | 5.2720 | 17,888 | 6.214% | 1,131 | 1,213.7s |
| queue_aware ❌ | 5.3823 | 16,147 | 8.746% | 1,000 | 1,548.9s |
| utilization_aware ❌ | 6.9265 | 18,182 | 8.890% | 875 | 420.4s |

❌ = SLA violations > 0% → NOT a valid SLA-safe baseline per Aurelius integrity rules.

---

## Honest Headline

**+38.2% SLA-safe goodput/$ and −27.6% GPU-hours**
*constraint_aware vs constraint_aware_no_affinity — both SLA-safe (0.000% timeout)*

- constraint_aware: 9.8514 gp/$ — 893 GPU-hrs — 26,392/26,392 SLA-safe
- constraint_aware_no_affinity: 7.1291 gp/$ — 1,234 GPU-hrs — 26,392/26,392 SLA-safe
- Delta: +38.2% gp/$ | −341 GPU-hrs (−27.6%)

**Misleading comparison (excluded from headline):**
- "+86.9% vs sla_aware" — sla_aware has 6.214% SLA violations, making it UNSAFE
  and an invalid baseline. The large number comes from the sla_aware denominator being
  only slightly better on cost while losing 32.6% of completions.

---

## Gain Attribution (Shapley Decomposition)

| Component | gp/$ contribution | Share |
|---|---|---|
| Model affinity / prewarming | 2.826 | **61.7%** |
| Anticipatory sizing | 1.753 | **38.3%** |
| Interaction | 0.0 | 0.0% |
| **Total vs sla_aware baseline** | **4.579** | 100% |

Attribution from `genai_ablation.py` Shapley decomposition. Model affinity is the
dominant driver: the cold-start penalty (22.85s without affinity vs 2.79s with
affinity) dominates the cost denominator on this multi-model trace.

---

## Same-Conditions Checklist

- [x] Same trace (lora_request_trace.csv, 26,392 requests)
- [x] Same SLA (`2.0 × exec_time + 30.0s`)
- [x] Same cost denominator ($3.00/hr GPU, Erlang-C provisioning)
- [x] Same serving physics (queue-aware M/M/c)
- [x] Same evaluation KPI (SLA-safe goodput/$)
- [x] Baseline passes SLA gate (0.000% timeout, 26,392/26,392 SLA-safe)
- [x] No oracle: anticipatory sizing uses EWMA arrival forecasts, no future arrivals
- [x] No per-request token-length leakage: exec_time used, not predicted
- [x] p99 < SLA ceiling: 65.9s (baseline) and 53.7s (candidate) << 2×exec_time+30s

---

## Research Papers Surveyed

| Paper | Relevance |
|---|---|
| inference-fleet-sim (arXiv:2603.16054) | Erlang-C fleet sizing — directly applicable to AMCSG/OSOTSS provisioning |
| GreenLLM (arXiv:2508.16449) | SLO-aware DVFS (45% energy savings) — energy denominator reduction path |
| DynamoLLM (HPCA 2025) | EWMA autoscaling on Azure traces — validates OSOTSS EWMA approach |
| FREESH (arXiv:2511.00807) | Heterogeneous GPU + energy scheduling |
| Asymptotic Optimality (arXiv:2602.02987) | Prefill-decode contention control |
| Trail (arXiv:2410.01035) | Per-request token prediction (ICLR 2025) — prerequisite for queue discipline |

---

## Classification and Production Relevance

**This benchmark is RESEARCH ONLY — not yet through AureliusOptimizer.**

Current path:
- `aurelius/traces/genai_backtest.py` — standalone simulation, not routed through AureliusOptimizer
- `aurelius/traces/genai_ablation.py` — ablation harness, calls genai_backtest directly

Path to canonical integration (requires separate phase):
1. Implement `ReplicaScalingPolicy.optimize(mode="genai")` or a new `GenAIServingPolicy`
2. Route genai_backtest.py through `AureliusOptimizer(policy="replica_scaling")`
3. Confirm 0% KPI drift (parity gate)
4. Update OPTIMIZER_UNIFICATION_PLAN.md with new entry

**Production applicability:** Multi-model adapter serving (LoRA, ControlNet) is a real
infrastructure pattern for image-generation inference fleets. The affinity routing and
anticipatory sizing signals are infrastructure-level decisions (adapter preloading,
autoscaling), not tenant-side arbitrage.

---

## Five-Failure Rule Compliance

- Five-Failure Rule: ACTIVE (6/5)
- Work type: Benchmark realism (third-trace cross-validation)
- New modules added: 0
- New optimizer paths added: 0
- AureliusOptimizer changed: No
- Allowed under Five-Failure Rule: Yes (benchmark realism is explicitly permitted)

---

## Files

| File | Role |
|---|---|
| `aurelius/traces/genai_backtest.py` | Standalone GenAI serving simulation |
| `aurelius/traces/genai_ablation.py` | Factorial ablation + Shapley attribution |
| `aurelius/traces/alibaba_genai.py` | Trace ingestor |
| `scripts/run_alibaba_genai_ablation.py` | CLI runner |
| `data/external/alibaba_genai/raw/lora_request_trace.csv` | Raw trace (26,824 rows) |
| `data/external/alibaba_genai/processed/alibaba_genai_ablation_summary.json` | Machine-readable results |
| `docs/ALIBABA_GENAI_ABLATION_RESULTS.md` | Human-readable ablation summary |
