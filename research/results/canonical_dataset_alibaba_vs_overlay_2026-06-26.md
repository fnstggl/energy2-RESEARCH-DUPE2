# Which unified dataset is better: Alibaba cluster trace vs Azure+overlay — 2026-06-26

> Rigorous, grounded answer to "should the canonical dataset be built off Alibaba's
> Cluster Trace Program instead of the Azure-LLM-serving + best-effort overlay?"
> Verified against the **actual** Alibaba ingesters already in this repo
> (`aurelius/traces/alibaba_gpu.py`, `alibaba_genai.py`) and the committed
> fixtures — not dataset marketing. Includes a **correction** to the prior
> `unified_replay_compounding_ab` +9.00% headline.

## TL;DR

1. **The Alibaba traces are NOT a "unified production trace."** Both are internally
   stratified; one is the wrong *decision problem* (training/packing, not serving),
   the other has *real* serving + system state but **its layers cannot be joined**
   (verified in-repo). ChatGPT's "10/10 unified telemetry" framing does not survive
   contact with the actual files.
2. **Alibaba's real value is CALIBRATION, not a spine.** Its production QoS mix
   (LS/BE/Burstable) is the right way to *ground the class ratio* the overlay was
   guessing.
3. **Grounding the ratio CORRECTS the +9.00% to ~neutral.** That headline used a
   40% best-effort tier; the real Alibaba ratio is ~20% by count (and ~1% by
   GPU-work). At the grounded ratio the serving-lever compounding **disappears**.
4. **Verdict:** keep the **Azure-LLM-serving spine** (it is our ICP and carries the
   real serving physics), **calibrate its class mix from Alibaba** (done here), and
   accept that a *trustworthy positive* compounding magnitude needs real
   best-effort **serving** economics (pilot) or a different real lever (KV/energy)
   — not a rebuild on Alibaba.

## 1. What the Alibaba traces ACTUALLY are (grounded)

| | cluster-trace-gpu-v2023 | cluster-trace-v2026-GenAI |
|---|---|---|
| workload | training/batch **pods** | **stable-diffusion serving** (image gen) |
| real classes | ✅ `qos` = LS / BE / Burstable | ✅ `predict_type`, multi-model `checkpoint_id` |
| serving arrivals + work | ❌ job durations, no token serving | ✅ request arrivals + `exec_time_seconds` |
| GPU util / memory time-series | ❌ none (ingester returns empty) | ✅ `pod_gpu_duty_cycle`, `pod_gpu_memory_used` |
| fractional-GPU packing | ✅ `gpu_milli` | — |
| **per-request ↔ GPU-state JOIN** | n/a | ❌ **`no_join`** — app layer is 2024 wall-clock w/o `container_ip`; metric layers are 2022 anonymized epoch w/ `container_ip` |

The decisive line is the last row, and it is **verified in this repo's own
ingester** (`alibaba_genai.py` docstring): the request layer and the GPU/queue
metric layers are in **incompatible time bases with no shared key**, so they
"are treated as SEPARATE replay/calibration layers… no end-to-end request→GPU
causality is claimed." So even the GenAI trace — the serving one — does **not**
give you a request that carries its own GPU state. To make it "unified" you would
have to **fake the join**, which is precisely the *time-base-collision /
correlation-invention* failure mode catalogued in
`CANONICAL_PRODUCTION_DATASET_DESIGN.md` §5 (#5, #6), and which the repo authors
already refused to do.

**Critical reframe ChatGPT misses:** "production-like" is not one axis. The Alibaba
GPU trace is genuinely rich *cluster-scheduler* telemetry — but that is the
**training-job placement** decision, a different product from Aurelius's
**LLM-serving** decision (capacity/ordering/admission/KV/energy on token traffic).
Richness ≠ relevance. A 155k-GPU training trace, however large, is mostly telemetry
for a decision Aurelius is not making.

## 2. The honest correction (this is the important part)

The prior `unified_replay_compounding_ab_2026-06-25.md` reported **+9.00%
compounding** on the multi-class slice. That slice used an **arbitrary 40%
best-effort** overlay. Grounding the best-effort fraction in the real Alibaba QoS
mix flips the result:

**Azure LLM 2024 · 5,880 reqs · on-demand · closed loop:**

| dataset | best single | best multi | interaction |
|---|---|---|---|
| overlay 40% (the earlier headline) | +6.71% | **+9.00%** | compounding |
| **Alibaba-calibrated by count (0.20)** | −0.06% | **−2.95%** | substitutive |
| **Alibaba-calibrated by GPU-work (0.012)** | +0.20% | +0.19% | neutral |

Best-effort fraction sweep (magnitude is fraction-bound, and *non-monotonic*):

| fraction | best single | best multi | verdict |
|---|---|---|---|
| 0.05 | +0.16% | +0.15% | neutral |
| 0.10 | +0.31% | +0.31% | neutral |
| 0.175 | +0.27% | +0.20% | neutral |
| 0.20 | −0.06% | −2.95% | substitutive |
| 0.30 | −0.23% | −2.32% | substitutive |
| **0.40** | **+6.71%** | **+9.00%** | **compounding** |

The +9.00% is a **lone spike at exactly 0.40** — surrounded by neutral/negative
cells. That non-monotonic shape is itself the tell: it was an **artifact of an
unrealistically large best-effort tier**, not a robust property. **At every
production-grounded ratio the serving-lever compounding is neutral-to-negative.**

What survives the correction: the **mechanism** (a closed loop on multi-class data
*can* compound) and the **engine** (the loop measures it honestly). What does NOT
survive: the **magnitude**. Reproduce: `python -m scripts.run_canonical_calibration_ab`.

## 3. Why grounded compounding is small here (and what would make it real)

The overlay models best-effort as extra serving load that backfills spare
capacity. Its economic upside scales with the best-effort **share**, and real
shares are small *on this regime*. Three things the public data can't tell us —
each of which the magnitude is sensitive to:

1. **Best-effort SERVING economics.** Alibaba's BE is *training* pods (~1% of
   GPU-work in the sample). The serving best-effort tier (batch inference, evals,
   data-gen) has a different — and unknown — token/GPU-hour weight. This is a
   pilot-telemetry signal.
2. **Consolidation value.** The real operator win from admission is running batch
   on **otherwise-idle serving GPUs** instead of dedicated batch capacity. The
   current model understates this (it has one shared pool, no harvested/dedicated
   distinction).
3. **A different cost term.** The most robust compounding likely comes not from
   admission on a thin batch tier but from levers on **different** real cost terms:
   KV prefix-reuse (Mooncake, real `hash_ids`) and energy price (real ISO series).
   Those are the next dataset increments in `CANONICAL_PRODUCTION_DATASET_DESIGN.md` §6.

## 4. So which dataset, concretely

**Best available foundation = Azure-LLM-serving spine + Alibaba-calibrated class
mix** (shipped: `aurelius/datasets/calibration.py` + `canonical.assemble_calibrated`).
Rationale:

- The spine must be **LLM serving** (our ICP physics: TTFT+TPOT, token goodput,
  SLA). Azure provides it; neither Alibaba trace does (GPU=training, GenAI=diffusion).
- Alibaba supplies the one thing the overlay was inventing — a **real class ratio**
  — transferred as a *distribution*, never a per-record join (the only honest way
  to combine a training trace with a serving spine).
- Rebuilding *on* Alibaba would be a **regression**: GPU-trace = wrong decision;
  GenAI-trace = wrong serving workload + unjoinable layers.

**Where Alibaba GenAI does add future value:** its real **multi-model /
request-type** structure (`checkpoint_id`, `predict_type`) is the right calibration
for a **multi-model spine**, which would unlock the *placement/affinity* lever (a
numerator cost term) — a more promising compounding source than admission, and a
worthwhile next build. But that is multi-model *structure* calibrated onto the
serving spine, still not a per-record join.

## 5. Bottom line

The Azure+overlay was the right **structure** but its **number** was inflated.
Alibaba is not a better spine — it is the **calibration source that corrects the
number**. With the correction, the honest state is: the joint loop and the
mechanism are validated; **a trustworthy positive compounding magnitude on real
data is not yet established** and needs (a) real best-effort serving economics from
a pilot, or (b) a different real lever (KV/Mooncake, energy price). That is a more
valuable place to be than a confident-but-fragile +9%.
