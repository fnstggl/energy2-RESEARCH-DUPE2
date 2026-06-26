# The Canonical Production-Like Public Dataset — design, audit & risk analysis

**Status:** design + first realizable slice shipped. 2026-06-25.
**Code:** `aurelius/datasets/signal_matrix.py` (the machine-readable matrix),
`aurelius/datasets/canonical.py` (the assembler), consumed by
`aurelius/optimizer/unified_replay.py` (the closed joint loop).

---

## 0. Why this exists

The joint optimizer has surfaces on **different cost terms** — capacity,
ordering, admission, energy, placement, plus the thermal/topology/KV physics that
gate them. Compounding goodput/$ requires running them *together on one workload
that carries every surface's decision inputs*. No public trace does. Today's data
is **stratified**:

- **Inference traces** (Azure LLM, BurstGPT, Mooncake) = arrivals + tokens, **zero
  system state** (no batch sizes, replica counts, scale events, power, temp).
- **Training/cluster traces** (Alibaba PAI, Philly, Acme, MIT Supercloud) = GPU
  state, placement, utilization — but **no serving** and the wrong workload.
- **Energy/thermal datasets** (M100 ExaData, Zeus) = power/temp — but wrong
  hardware (V100 HPC) or no temperature at all.

So every optimizer test is **single-surface**: we can test capacity on Azure, KV
on Mooncake, packing on Alibaba — never the joint loop. We proved
(`research/results/unified_replay_compounding_ab_2026-06-25.md`) that this
stratification — *not* the optimizer — is why levers don't compound: the identical
optimizer goes from NEUTRAL on single-class data to **+9.00% COMPOUNDING** the
moment the data carries workload classes. The canonical dataset is how we supply
the *rest* of the missing structure, honestly.

This document is the **audit** the build must answer to. Its governing principle:
**make the gap between "production-like" and "production" explicit and measured**,
so we never ship a stitched-together dataset that *looks* comprehensive but
silently misleads the optimizer (a "monstrous" dataset — §5).

---

## 1. The signal matrix (what the canonical trace must carry)

Every signal a surface needs, the best public source, and its **fidelity tier**:
`MEASURED_REAL` (real telemetry, right HW+workload) ▸ `PROXY` (derived from a real
but indirect signal) ▸ `SYNTHETIC` (documented parameterized overlay) ▸
`SIMULATOR_ONLY` (only a model exists publicly) ▸ `ABSENT` (no public source).
This table is generated 1:1 from `CANONICAL_SIGNAL_MATRIX` — query it in code.

| signal | surface | best source | tier | in repo? | stitch risk |
|---|---|---|---|---|---|
| arrival_time | all | **Azure LLM 2024** | MEASURED | ✅ | none — the spine |
| output_tokens | ordering/capacity | **Azure LLM 2024** | MEASURED | ✅ | none — the spine |
| prompt_tokens | kv/capacity | **Azure LLM 2024** | MEASURED | ✅ | none — the spine |
| model_id | placement/routing | BurstGPT / Mooncake | MEASURED | ✅ | spine is single-model; 2nd trace has a different arrival base → align or tier |
| workload_class | admission | Philly / Alibaba PAI | PROXY | ✅ | label lives on *training* jobs; map, don't merge, onto serving |
| best_effort_overlay | admission/energy | *(synthetic)* | SYNTHETIC | ✅ | synthetic by construction — must stay labeled, never reported as real |
| kv_prefix_hit | kv/placement | **Mooncake** (hash_ids) | PROXY | ✅ | real reuse signal but on Kimi traffic, not Azure; hit-RATE computable, dynamics not |
| kv_block_reuse_dynamics | kv | *(simulator)* | SIMULATOR | ❌ | eviction/recompute/routing are policy-emergent — never in any trace |
| gpu_power_w | energy | **Zeus / ML.ENERGY** | PROXY | ❌ | real power but H100 micro-bench → curve transfers, absolute watts don't |
| energy_price | energy | public grid/ISO (EIA, CAISO) | MEASURED | ❌ | real + public; align by hour-of-day (Azure trace is time-anonymized) |
| gpu_temperature_c | thermal | **M100 ExaData** (CINECA) | PROXY | ❌ | V100/HPC not H100/inference → α/β shape transfers, absolute temps don't |
| throttle_events | thermal | *(simulator)* | SIMULATOR | ❌ | clock-throttle / thermal-violation events never published by anyone |
| collective_cost | placement/topology | **MLCommons Chakra** + nccl-tests | PROXY | ❌ | real traces+sweeps → alpha-beta model transfers; value is fabric-specific |
| fabric_congestion | placement/topology | *(ASTRA-sim)* | SIMULATOR | ❌ | congestion/incast/straggler is proprietary (HPN/C4, MegaScale, Meta) — sim-only |
| gpu_utilization | capacity/packing | **Acme** / Alibaba PAI | PROXY | ✅ | real 15s DCGM but *training*; serving util has no public ground truth |
| gpu_fragmentation | placement/packing | **Alibaba gpu (gpu_milli)** | MEASURED | ✅ | real fractional-GPU requests; joined at fleet level, not per-request |
| inference_autoscaling_truth | capacity | *(absent)* | ABSENT | ❌ | online-inference autoscale/batch/migration truth is paper-only (SageServe, Aegaeon, DynamoLLM) |

**Coverage (from `coverage_by_tier()`):** 22 signals → **8 MEASURED · 6 PROXY · 1
SYNTHETIC · 3 SIMULATOR-ONLY · 4 ABSENT**. So **15/22 (68%) are obtainable at
measured/proxy/synthetic fidelity; 7/22 (32%) are a hard ceiling** that no amount
of stitching fixes — they need a self-run telemetry pilot or stay simulator-only.
(The MEASURED workload-class + macro network/topology signals come from
`cluster-trace-gpu-v2026` — they *calibrate*, they do not become the token spine
since v2026 is hourly pod aggregates; see
`research/results/alibaba_gpu_v2026_audit_2026-06-26.md` and the two-plane
assembly in `research/CANONICAL_DATASET_ASSEMBLY_ARCHITECTURE.md`. The 4 ABSENT
include the **decision-intent tier** — user intent, hardware health, operator
rationale — which is structurally proprietary and is exactly the pilot's job.)

---

## 2. Which dataset for each, and why (the picks)

- **Spine = Azure LLM 2024.** The demand backbone everything attaches to: real
  arrivals + real output tokens, the one structure every serving trace agrees on
  and the one our physics (service time, goodput, SLA) already consume. Already
  in-repo (`aurelius/traces/azure_llm.py`).
- **KV reuse = Mooncake.** The single highest-value find: `hash_ids` are
  block-level prefix hashes, so prefix-hit rate is *computed, not guessed*. It
  upgrades our `cache_affinity_key` from a session-id proxy to a real prefix
  signal. (Its arrival base is Kimi, not Azure — so it informs the KV *model*, it
  does not replace the spine.)
- **Workload class = Philly/Alibaba PAI priority+type, realized as a synthetic
  overlay.** The class *taxonomy* is real (interactive vs batch/offline exists in
  every fleet); the *labels* can't be merged onto Azure (different workload). So
  the canonical slice ships a **documented best-effort overlay** (§3) — the minimal
  honest way to add the class dimension that unlocks admission + energy.
- **Power = Zeus/ML.ENERGY; Temperature = M100 ExaData.** Split because no single
  source has both on the right HW. Use Zeus to fit `power(utilization)` and ExaData
  to fit `temp(power, util, cooling)` α/β *shapes*; drive both from the spine's
  implied utilization. Calibrated models, not measured series.
- **Energy price = public ISO (EIA/CAISO).** Real, public, exogenous; joined to
  the trace by **hour-of-day** (the Azure trace is time-anonymized, so absolute
  wall-clock alignment is impossible — relative diurnal shape is the honest join).
- **Collective cost = Chakra + nccl-tests.** Real collective traces + real
  bandwidth sweeps → an alpha-beta cost model. The *congestion* half (fabric_
  congestion) has no public source and stays simulator-only.
- **Packing/fragmentation = Alibaba gpu_milli; Utilization sanity = Acme.** Real
  fractional-GPU requests bound the packing lever; Acme's 15s DCGM gives a real
  utilization *distribution* to sanity-check the simulator's implied numbers.

---

## 3. How we combine them (the assembly architecture)

The canonical dataset is **not** a single merged CSV — merging mismatched sources
into one flat table is exactly how you get a monstrous dataset (§5). It is a
**layered assembly** with four explicit join types, each carrying a fidelity tag:

```
                      ┌─────────────────────────────────────────────┐
   SPINE (real) ─────▶│ Azure LLM 2024: arrivals + tokens (per req)  │
                      └─────────────────────────────────────────────┘
                              │ 1. ATTACH (per-request, real)
                              ▼   model_id, prompt/output tokens
                      ┌─────────────────────────────────────────────┐
   OVERLAY (synth) ──▶│ best-effort batch tier (labeled SYNTHETIC)   │  ← unlocks admission/energy
                      └─────────────────────────────────────────────┘
                              │ 2. CALIBRATE (model params, proxy)
                              ▼   power(util)←Zeus, temp α/β←ExaData,
                              │   prefix-hit←Mooncake, collective αβ←Chakra
                      ┌─────────────────────────────────────────────┐
   EXOGENOUS (real) ─▶│ energy price by hour-of-day (EIA/CAISO)      │
                      └─────────────────────────────────────────────┘
                              │ 3. ALIGN (by relative hour-of-day, proxy)
                              ▼
                      ┌─────────────────────────────────────────────┐
   SIMULATOR ────────▶│ KV dynamics, throttle events, congestion     │  ← emergent, policy-dependent
                      └─────────────────────────────────────────────┘   4. SIMULATE (sim-only, lowest tier)
```

1. **ATTACH (real, per-request):** signals that live on the same request as the
   spine (tokens, model). Lossless real join.
2. **OVERLAY (synthetic, documented):** the best-effort tier. Deterministic,
   parameterized, manifest-stamped. *This is the only slice shipped today* —
   `augment_with_best_effort()` — because it is the one that unlocks compounding
   and needs no external download.
3. **CALIBRATE (proxy, model-level):** mismatched-but-real sources tune *model
   parameters* (a curve, an α/β), never per-request values. A V100 heat-up curve
   sets the *shape* of `temp(power)`; it never claims an H100's absolute temp.
4. **ALIGN (proxy, exogenous):** the energy price series joins by hour-of-day.
5. **SIMULATE (sim-only):** policy-emergent state (KV eviction, throttle,
   congestion) is produced by the replay engine, tagged lowest-fidelity, and
   **never** presented as measured.

Each assembled trace ships a **manifest** (`CanonicalManifest`) recording every
join's source + tier, so any downstream number can be traced to its fidelity.

---

## 4. How production-like will this be? (honest grading)

Graded **per surface**, because the fidelity is wildly uneven and a single
headline number would itself be misleading:

| surface | production-likeness | why |
|---|---|---|
| capacity / ordering | **HIGH** | real arrivals+tokens+SLA; only the service-time model is a parameter |
| admission (class) | **MEDIUM** | class taxonomy real; the *mix ratio* is a synthetic parameter, not measured |
| KV / prefix-cache | **MEDIUM** | prefix-hit rate real (Mooncake); memory-pressure dynamics simulated |
| energy | **MEDIUM** | price real, power-curve real-calibrated; absolute watts modeled |
| thermal | **LOW** | α/β shape from V100; absolute temps + throttle modeled/absent |
| topology | **LOW** | collective cost calibratable; congestion simulator-only |
| autoscaling truth | **NONE** | no public ground truth at any fidelity |

**Bottom line:** the canonical dataset can make capacity/ordering/admission/KV/
energy **jointly testable at MEDIUM-or-better fidelity** — which is exactly the
cluster of levers the compounding A/B needed, and enough to drive the joint loop
honestly. Thermal/topology can be **calibrated** (the optimizer can *reason* about
them) but not **validated** on public data. Autoscaling truth is a flat no.

---

## 5. The "monstrous dataset" failure modes (and the guardrails)

The user's central worry: a stitched dataset that *looks* production-like but
**misleads the optimizer**. These are the concrete ways that happens, and the
guardrail each one gets. A canonical dataset that violates any of these is worse
than no dataset, because it produces confident wrong answers.

| # | failure mode | what it looks like | guardrail |
|---|---|---|---|
| 1 | **Frankenstein join** | merge V100 temps with H100 power into one "GPU" that exists nowhere | never per-record-merge across HW; mismatched sources calibrate *model params* only (§3 join type 3), tagged PROXY |
| 2 | **Synthetic laundering** | the best-effort overlay's numbers get cited as real savings | overlay is `tier=SYNTHETIC` in the manifest + every result doc; it answers *mechanism* questions, never *magnitude* claims |
| 3 | **Fidelity flattening** | one headline "production-like" score hides that thermal is LOW | grade **per surface** (§4); `coverage_by_tier()` keeps the mix visible in code |
| 4 | **Distribution drift** | overlay arrivals/tokens have a shape the spine never had | overlay tokens are **resampled from the spine's own distribution**; only timing+label are new |
| 5 | **Correlation invention** | stitching imposes a power↔arrival correlation that isn't real | exogenous signals join by *coarse* keys (hour-of-day), never by faked per-request links; un-joinable pairs stay separate (the schema already refuses to join Alibaba-GenAI layers) |
| 6 | **Time-base collision** | two traces' anonymized clocks get treated as the same axis | only the spine defines the time axis; everything else aligns by relative/derived keys, never absolute |
| 7 | **Silent ceiling** | the 24% simulator-only/absent signals get presented as covered | `simulator_or_absent()` enumerates them explicitly; results that depend on them carry the SIMULATOR tag |
| 8 | **Calibration overreach** | a V100 α/β is reported as an H100 measurement | calibration upgrades a prior's provenance to `BENCHMARK_DERIVED` *with the HW caveat attached*, per the existing `calibration.py` ladder — never to MEASURED |

The discipline is already native to this repo: `traces/schema.py` documents
`cache_affinity_key` as "a PROXY, NOT a measured KV hit rate", returns empty
utilization for Alibaba "documented, not invented", and refuses to join GenAI
layers whose "anonymized time bases" are incompatible. The canonical dataset
**extends that same honesty**, it does not relax it.

---

## 6. Build sequence (realizable now → needs a pilot)

1. **DONE — multi-class slice.** `augment_with_best_effort()` + the closed loop.
   Proves the data-vs-optimizer *mechanism* question. No external download;
   deterministic; shipped.
1b. **DONE — class-ratio calibration from Alibaba (and the correction it forced).**
   `aurelius/datasets/calibration.py` derives the best-effort fraction from the real
   Alibaba cluster-trace-gpu-v2023 QoS mix (~20% by count) instead of the arbitrary
   40% the slice first used. This **corrected the +9.00% headline to neutral/−3%**
   at the production-grounded ratio — the magnitude was a fraction artifact (see
   `research/results/canonical_dataset_alibaba_vs_overlay_2026-06-26.md`). Lesson
   folded back into guardrail §5 #2: even a *labeled* synthetic overlay misleads on
   *magnitude* unless its parameters are calibrated from real data. Assessed and
   rejected the Alibaba traces as a replacement *spine* (GPU=training/wrong-decision;
   GenAI=diffusion + `no_join` layers) — they are a calibration source, not a spine.
2. **NEXT (download-only, no pilot) — calibration upgrades.** Pull the four real
   public sources and upgrade priors from HEURISTIC to BENCHMARK_DERIVED *with
   caveats*: KV prefix-hit ← Mooncake `hash_ids`; power(util) ← Zeus; thermal α/β
   ← M100 ExaData; collective αβ ← Chakra + nccl-tests. Each lands as a PROXY-tier
   calibration with its HW/workload caveat in the manifest.
3. **NEXT — energy price + KV replay.** Attach an ISO price series (hour-of-day)
   to exercise energy time-shift on the deferred batch tier; replay Mooncake
   through vLLM/SGLang to *measure* (our own) KV dynamics for the simulator's KV
   model.
4. **CEILING — read-only DCGM/Prometheus shadow pilot.** The only path to the four
   simulator-only/absent signals (KV dynamics under pressure, throttle events,
   fabric congestion, autoscaling truth) and the only way any of them becomes
   MEASURED. Until then they stay explicitly simulator-tagged.

---

## 7. One-line answer to "should we build it?"

Yes — but as a **layered, per-surface-graded, manifest-stamped assembly**, not a
merged table, **with every structural parameter calibrated from real data** (the
class ratio from Alibaba, §6 step 1b — not guessed). Built that way it makes the
capacity/ordering/admission/KV/energy cluster **jointly testable at MEDIUM+
fidelity**, while keeping thermal/topology honestly at "calibratable but not
validatable" and autoscaling truth at "needs a pilot." Built as a flat merge — or
with guessed parameters (the 40%→neutral correction) — it produces confident wrong
optimizer decisions, so the guardrails in §5 are not
optional, they are the design.
