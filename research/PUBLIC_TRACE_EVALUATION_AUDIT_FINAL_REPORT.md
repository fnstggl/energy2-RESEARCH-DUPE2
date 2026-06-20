# Public Trace Evaluation Stack — Audit Final Report

> **Scope:** realism, validity, and benchmark-meaning of the current public
> trace evaluation stack, and validity of parallel run-g's +323% claim. No new
> optimization implemented; no runtime path changed. Simulator / public-trace
> results are directional only — **NOT production savings** (`docs/RESULTS.md`
> §8). Audited 2026-06-20 on branch `claude/intelligent-goldberg-zkj5pz`.

## FINAL STATUS: **AUDIT COMPLETE** — public trace realism and run-g validity assessed.

No invalid claim was found in the repo as written: run-g's commit message and
results doc correctly say "vs FIFO" and explicitly disclaim "+300% vs SLA-aware."
The risk is **downstream mis-comparison** — treating the +323% as the same KPI
as, or additive to, the +26% rollup. This report draws that line precisely.

Supporting docs: `PUBLIC_TRACE_REALISM_AUDIT.md` (Phases 1–3, 7),
`PUBLIC_TELEMETRY_REALISM_MATRIX.md` (Phase 6), `RUN_G_VALIDITY_AUDIT.md`
(Phases 4–5), `results/srtf_stronger_baselines_audit_2026-06-20.{md,json}`
(Phase 5 data), `scripts/audit_srtf_stronger_baselines.py` (Phase 5 harness).

---

## 1. Dataset inventory (summary)

19 datasets committed/referenced. Real-signal coverage clusters on two axes;
the per-request serving axis is the gap. Fidelity score 1–5 (detail in
`PUBLIC_TRACE_REALISM_AUDIT.md` §Phase 1).

| dataset | workload | real local? | size | score | one-line caveat |
|---|---|---|---|:--:|---|
| Azure LLM 2024 | LLM serving | sample only (5,880) | 44.1M full | 4 | 3 columns; full week SAS-gated → sample reproduces +0.00% |
| Azure LLM 2023 | LLM serving | fixture | 19,366 | 3 | same 3 columns, ~0.003 days |
| BurstGPT | LLM serving | full fetchable (1.43M) | CC-BY-4.0 | 4 | model-level cache proxy; no latency |
| Alibaba GenAI 2026 | SD+LLM serving | summary | 26,392 | 4 | richest serving signal; aggregate latency; join gaps |
| Alibaba GPU v2023 | GPU packing | summary | 6,282 | 3 | training; no tokens/latency |
| Philly | training | **fixture 33 jobs** | 1 GB uncommitted | 2 | fixture-scale |
| MIT Supercloud | training | bounded real 10k | 3 MB of ~1–2 TB | 3 | capacity unpublished |
| CAISO/PJM/ERCOT | energy price | **full committed** | hourly Q1'26 | 5 | real market prices |
| WattTime carbon | carbon | **full committed** | 1,571 rows | 5 | real MOER (research access) |
| Canonical energy | energy (synth job) | golden JSON | 1,000 jobs | 2/5 | synthetic workload, real prices |
| **CARA** `asdwb` | **serving telemetry** | ingested 76,825 | not in rollup | 4 | **only trace with predicted+actual tokens, TTFT/TPOT, KV, queue** — license unverified |
| cc-traces `semianalysisai` | agentic serving | ingested 136,118 | Apache-2.0 | 4 | **only trace with real `request_type`**; KV block hashes |

## 2. Production telemetry requirement matrix (summary)

Per optimization class: is current benchmark coverage valid? (Full table in
`PUBLIC_TRACE_REALISM_AUDIT.md` §Phase 2.)

| optimization class | required fields present in a rollup trace? | coverage valid? |
|---|---|---|
| Energy/cost-aware regional | real prices ✓ | **YES (directional)** |
| Carbon-aware | real carbon ✓ | **YES (directional)** |
| Autoscaling / provisioning | real arrivals+tokens ✓ | **YES (directional)** |
| Batch inference / GPU packing | real jobs ✓ | **YES (directional)** |
| Heterogeneous GPU placement | GPU type ✓, LLM TTFT-by-GPU ✗ (synthetic) | **PARTIAL** |
| Per-request serving queue | tokens ✓, contention/SLA/servers ✗ | **SYNTHETIC-DEPENDENT** |
| Output-length-aware SRTF | predicted tokens ✗ (synthetic on Azure) | **NO (real only via CARA)** |
| SLA-aware queue | request_type ✗ → baseline=FIFO | **NO (real only via cc-traces)** |
| Admission / KV pressure | KV ✗ (realized-ρ proxy) | **NO (real only via CARA)** |
| Migration-aware | migration_cost ✗ (synthetic everywhere) | **NO** |

## 3. Benchmark audit (summary)

9 entry points. KPI = `economics.py` SLA-safe goodput/$ everywhere — **except**
the two run-g SRTF modules, which use a **constant** cost denominator. Full table
in `PUBLIC_TRACE_REALISM_AUDIT.md` §Phase 3. Headline findings:

- **The +26% rollup (Azure 2024 week)** = per-tick provisioning vs `sla_aware`,
  aggregate Erlang-C model, **−21% GPU-hours** (denominator moves).
- **The +323% run-g** = per-request ordering vs **FIFO**, new discrete-event
  M/G/c model, **GPU-hours constant** (denominator fixed).
- **Reproducibility gap:** the committed Azure 2024 sample yields **+0.00%** vs
  sla_aware (1× and 50×); the +25.75% needs the **SAS-gated full week**.
- Only the canonical energy path constructs a real `JobScheduler`; serving paths
  have no per-request ordering — the exact gap run-g surfaced.

## 4. Run-g audit

| question | answer |
|---|---|
| implemented | discrete-event M/G/c queue sim; FIFO vs shortest-predicted-first on real Azure tokens |
| new decision surface | per-request queue ordering (provisioning replay never had it) |
| dataset | Azure 2024 **sample** (5,880); full week not used |
| baseline | **FIFO** (not sla_aware) |
| metric | SLA-safe goodput/$ but **constant denominator** → really SLA-attainment at fixed cost |
| public trace direct? | tokens/arrivals real; **contention synthetic** (21.95× time-warp) |
| synthetic assumptions | TTFT/TPOT, c=4, $2/GPU-h, SLA=10s, time-warp |
| actual tokens at decision? | **No** — orders by predicted; physics uses actual (no leak) |
| predicted-only? | forecast variant orders by actual×30%CV noise; ≈ perfect (robust) |
| SLA safety preserved? | **partially** — long-tail p99 733s→2189s regression (disclosed, asserted in tests) |
| reproducible? | **Yes, exactly** (+323.51%, deterministic, 38 tests) |
| comparable to +26%? | **No** (different baseline, simulator, denominator, decision surface) |

### Verdict: **VALID BUT NOT COMPARABLE**
Real, reproducible, leak-free, honestly scoped — but measured against a weak
FIFO baseline on a separate simulator with a constant cost denominator, so it is
not comparable to the +26% sla_aware rollup and is not by itself a north-star
(vs-SLA-aware) improvement. It **is** the most promising research direction in
the repo.

## 5. Run-g stronger-baseline results (Phase 5)

Re-running run-g's own simulator against SLA-aware (EDF) and a preemptive
reference (`scripts/audit_srtf_stronger_baselines.py`):

| Policy | SLA-safe goodput/$ | Δ vs FIFO | Δ vs SLA-aware | Δ vs constraint-aware | SLA viol | queue p99 | cost | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo | 13,336 | 0% | 0% | N/A | 83.6% | 730s | $8.05 | 4.02 |
| sla_aware (EDF) | 13,336 | 0% | 0% | N/A | 83.6% | 730s | $8.05 | 4.02 |
| srtf_perfect | 56,481 | **+323.5%** | **+323.5%** | N/A | 14.2% | 2182s | $8.05 | 4.02 |
| srtf_forecast | 56,855 | +326.3% | +326.3% | N/A | 14.9% | 2225s | $8.05 | 4.02 |
| srpt_reference | 56,311 | +322.2% | +322.2% | N/A | 14.2% | — | $8.05 | 4.02 |

**`sla_aware` == `fifo` identically** (uniform SLA → EDF = arrival order): the
Azure trace has no `request_type`, so a differentiated SLA-aware baseline cannot
be built without synthesizing labels. **Δ vs constraint-aware = N/A** (orthogonal
decision surface). **Cost/GPU-hours identical across all rows** → the gain is
SLA-attainment at fixed cost, not cost reduction. (If +323% were *only* vs FIFO
and a real SLA-aware existed, the honest delta would be measured against it — but
here SLA-aware degenerates to FIFO, so +323% vs FIFO == +323% vs degenerate
SLA-aware, which is a statement about the trace's poverty, not SRTF's strength.)

## 6. Public telemetry realism matrix (Phase 6)

Full field-by-field provenance in `PUBLIC_TELEMETRY_REALISM_MATRIX.md`. The
decisive facts: `predicted_output_tokens` is Real in **one** dataset (CARA);
`request_type` is Real in **one** (cc-traces); per-request TTFT/TPOT/KV/queue are
Real only in CARA; `migration_cost` is Synthetic everywhere; energy+carbon are
Real. The serving-telemetry trace and the energy trace never overlap.

## 7. Current benchmark weaknesses

1. **FIFO-baseline inflation risk.** Run-g's headline uses FIFO; the project's own
   rule (`docs/RESULTS.md` §3) makes `sla_aware` the serving baseline.
2. **Constant-denominator "goodput/$".** Run-g's "/$" never moves; the metric is
   SLA-attainment, not cost-efficiency — easy to mis-read as comparable to the
   −21%-GPU-hour rollup.
3. **Largest headline not reproducible from a clean checkout.** Azure 2024 +26%
   needs the SAS-gated full week; the committed sample shows +0.00%.
4. **SLA-aware = FIFO on rollup serving traces** (no `request_type`).
5. **KV/admission tested with a synthetic proxy** (realized-ρ), not real KV.
6. **GPU-placement priors are synthetic** (CARA-calibrated on synthetic rows);
   regressed the real KPI.
7. **Migration savings have no public anchor** (synthetic cost constant).
8. **Synthetic contention drives the SRTF magnitude** (21.95× time-warp); the
   effect vanishes at the trace's native load.

## 8. Recommended dataset additions (priority order)

1. **Promote CARA to a first-class serving benchmark** (verify license first) —
   unlocks fair SRTF, SLA-aware queue, admission/KV, heterogeneous placement.
2. **Add cc-traces (Apache-2.0)** — real `request_type` so SLA-aware ≠ FIFO; KV
   block hashes; TTFT.
3. **Mooncake FAST25 (Apache-2.0)** — KV-prefix reuse validation.
4. **Vidur profiling CSVs** — real per-GPU kernel latency for placement priors.
5. **Build a provenance-tagged "production-like public telemetry corpus"** by
   *joining real fields* (never synthesizing them), each row carrying
   real/derived/synthetic metadata.

## 9. Optimizations currently VALID to test (directional)

- Energy/cost-aware regional scheduling (real prices)
- Carbon-aware scheduling (real carbon)
- Autoscaling / replica provisioning (real arrivals+tokens; modelled latency)
- Batch inference / GPU packing (real jobs)

## 10. Optimizations needing better traces before claims are credible

- **Output-length-aware SRTF** — needs CARA (real predicted+actual tokens, real
  contention/latency).
- **SLA-aware per-request queueing** — needs cc-traces (real `request_type`).
- **Admission / KV-pressure control** — needs CARA/cc-traces (real KV).
- **Heterogeneous GPU placement** — needs CARA/Vidur (real TTFT-by-GPU).
- **Migration-aware scheduling** — needs any trace with real migration cost
  (none exists; do not headline migration savings).

---

## One-paragraph bottom line

The Aurelius public-trace stack is **realistic and valid for the energy/carbon
and autoscaling-provisioning optimizations** — those rest on real market prices
and real arrival/token demand, and the +26% Azure rollup is a legitimate
(directional) cost-denominator win vs the right baseline, with the single caveat
that its largest number needs the SAS-gated full week. It is **not yet realistic
for the per-request serving optimizations the project is pivoting toward** —
SRTF, SLA-aware queueing, admission/KV — because the rollup traces lack SLA-class,
latency, and KV fields, forcing synthetic assumptions. **Run-g's +323% is real,
reproducible, and honest, but it is a synthetic-contention, constant-cost,
vs-FIFO queue-ordering demonstration — not comparable to the +26% rollup and not
yet a north-star win.** The highest-leverage next step is not a new optimizer but
a new dataset: verify CARA's license and promote it (plus cc-traces) so the
serving optimizations can be tested against real SLA classes and real latency
instead of synthesized ones.
