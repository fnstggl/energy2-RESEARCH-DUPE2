# Run-g Validity Audit — "+323% goodput/$ vs FIFO" (SRTF on Azure LLM 2024)

> **Audit / reporting document.** Assesses the validity, comparability, and
> north-star relevance of the run 2026-06-20-g claim. No runtime path changed.
> All numbers reproduced from a clean checkout on 2026-06-20.

## Verdict

### **VALID BUT NOT COMPARABLE** (with a *promising-but-needs-stronger-baseline* asterisk)

The +323% is **real, reproducible, and leak-free**, and run-g itself documents
all its caveats honestly (it never claims +323% vs SLA-aware). But it is **not
comparable to the +26% public benchmark rollup** and is **not, by itself, a
north-star (vs-SLA-aware / vs-constraint-aware) improvement**, for three
structural reasons proven below: (a) a different, weaker baseline (FIFO), (b) a
different, purpose-built simulator with a **constant cost denominator**, and
(c) a decision surface (queue ordering) that the rollup baseline does not act
on. **No invalid or leaky claim was found** — the headline in the commit
message is literally true and correctly scoped.

---

## What run-g is (commits + files)

| commit | what |
|---|---|
| `3501760` | CARA output-length forecaster v1 (semi-clairvoyant foundation) |
| `a6ef780` [run-f] | wired `predicted_output_tokens` SRTF sort key into the **batch** `JobScheduler._solve_greedy` |
| `3774085` [run-g] | **the audited commit** — two new benchmark modules + results + GAP/ROADMAP updates |

Run-g touched (no runtime/scheduler code):
- `aurelius/benchmarks/srtf_serving_backtest.py` (417 LOC) — **the +323% source**
- `aurelius/benchmarks/srtf_contention_backtest.py` (416 LOC) — companion negative finding
- `docs/SRTF_SERVING_BACKTEST_RESULTS.md`, `research/GAP_ANALYSIS.md`, `research/ROADMAP.md`
- `tests/test_srtf_serving_backtest.py` (+ contention tests) — 38 new tests

---

## The 12 audit questions

**1. What exactly did run-g implement?**
A standalone **discrete-event, non-preemptive M/G/c queue simulator** that
replays the real Azure LLM 2024 output-token stream through `c=4` replicas and
compares FIFO vs shortest-predicted-job-first request ordering. Plus a companion
batch-contention probe proving the *merged* batch scheduler (run-f) and the
analytical Erlang-C serving model **cannot express** request-level SRTF (no
queue-wait semantics). It is **research infrastructure** — no runtime decision
path changed; the run-f sort key remains inert for serving.

**2. What decision surface did it expose that the aggregate replay did not?**
**Per-request queue ordering.** The public rollup (BurstGPT / Azure 2024
backtests) exposes only *per-tick replica provisioning* (autoscaling) over an
aggregate Erlang-C M/M/c model with **no per-request ordering**. Run-g built the
missing surface: which waiting request to serve next. This is a genuine,
previously-unmeasured lever — and it is the single sharpest insight of run-g.

**3. What dataset did it run on?**
The **committed Azure LLM 2024 sample** (`tests/fixtures/azure_llm_2024_sample.csv`,
5,880 requests). Real fields used: arrival timestamps and per-request output
tokens (heavy-tailed p50≈90, p99≈479, max≈1346). The full 44.1M-row week is
SAS-gated (HTTP 401) and was **not** used.

**4. Was the +323% measured against FIFO or SLA-aware?**
**FIFO.** Explicitly. The commit title says "vs FIFO" and Caveat 1 of the
results doc says "This is **not** a claim of +300% vs SLA-aware."

**5. Was it measured using SLA-safe goodput/$?**
Yes — but with a **constant denominator**. `_sla_safe_goodput_per_dollar` divides
SLA-safe tokens by `Σ service_s × GPU_HOUR_USD`, which is **identical across all
disciplines** (same requests, same service times). So the metric moves only via
the numerator (SLA-safe token count). It is really **SLA-safe goodput** (an
SLA-attainment effect at fixed cost), with "/$" decorative. (Re-verified in the
stronger-baseline replay: GPU-hours = 4.02 and cost = $8.05 for *every* policy.)

**6. Did it use public traces directly?**
Yes for the token + arrival distribution. But the **contention is synthetic**:
the sample's native RPS leaves the pool ~85% idle, so arrivals are **time-warped
21.95×** to manufacture ρ=0.85. The entire effect is a contention phenomenon,
and the contention does not exist in the trace at native rate.

**7. Did it introduce synthetic assumptions?** Yes, all documented:
service physics `s = 0.150 + tokens·0.020` (TTFT/TPOT constants), `c=4` servers,
`GPU_HOUR_USD=2.0`, `SLA=10 s`, and the **21.95× time-warp**. Only the queue
ordering differs across disciplines, so deltas are attributable to ordering —
but the *magnitude* is a function of these synthetic knobs.

**8. Did it use actual output tokens at decision time?**
**No (good).** Ordering uses `predicted_tokens`. Service physics uses
`actual_tokens`. The two are genuinely decoupled.

**9. Did it use predicted output tokens only?**
The `srtf_forecast` variant orders by `actual × lognormal(30% CV)` noise — a
realistic forecast-quality prior. `srtf_perfect` uses actual-as-prior (clairvoyant
*ordering*, still no service-time leak). The forecast result (+326%) ≈ the
perfect result (+323%): robust to forecast error.

**10. Did it preserve SLA safety?**
**Partially — and it discloses the violation.** Short-request p90 collapses
696 s → 3 s, but the long-request tail **regresses** p99 733 s → 2,189 s
(non-preemptive SJF starvation). The net +323% is SLA-safe tokens *traded* from
a few starved long requests to many rescued short ones. The regression is
asserted in the test suite so it cannot silently disappear. **An anti-starvation
guard is a precondition for any runtime use.**

**11. Is the result reproducible from a clean checkout?**
**Yes, exactly.** `run_srtf_serving_backtest()` → `sla_goodput_delta_pct =
323.51`, deterministic (seeded). 38 tests pass.

**12. Is it comparable to the existing +26% public benchmark rollup?**
**No.** Five axes differ:

| axis | +26% rollup (Azure 2024 week) | +323% run-g |
|---|---|---|
| baseline | `sla_aware` (reactive autoscaler) | **FIFO** |
| decision surface | per-tick replica provisioning | per-request queue ordering |
| simulator | aggregate Erlang-C M/M/c (`serving.py`, locked) | **new** discrete-event M/G/c |
| cost denominator | **varies** (−21% GPU-hours) | **constant** (same GPU-hours) |
| metric meaning | cost-efficiency (fewer GPU-hours/token) | SLA-attainment at fixed cost |

They measure different things on different code with different baselines. They
must never be added, averaged, or presented as the same KPI.

---

## Phase 5 — stronger-baseline replay (data: `results/srtf_stronger_baselines_audit_2026-06-20.md`)

Re-running run-g's own simulator against EDF (SLA-aware) and a preemptive
reference:

- **`sla_aware` (EDF) == FIFO**, identically, at every ρ. With no SLA-class field
  in the trace, every request shares one deadline, so earliest-deadline-first =
  arrival order = FIFO. **Therefore the "real improvement vs SLA-aware" equals
  the improvement vs FIFO (+323%) — not because SRTF beats a strong SLA-aware
  policy, but because this trace cannot express one.** Computing a *differentiated*
  SLA-aware baseline requires synthetic class labels and is deliberately not done.
- **Δ vs constraint-aware = N/A** (orthogonal decision surface; cannot be placed
  in a single-queue simulator).
- **Preemptive SRPT reference (+322%)** carries the *same* long-tail p99 ≈ 2,373 s,
  showing the starvation is near-intrinsic at this load and needs aging, not just
  preemption.

---

## Bottom line for the north-star KPI

The Aurelius north-star is **SLA-safe goodput/$ vs the strongest realistic safe
baseline** (`sla_aware` for serving). Run-g:
- is **valid** (reproducible, leak-free, honestly scoped),
- is **not comparable** to the +26% rollup (different baseline, simulator, cost
  behavior, decision surface),
- is **not yet a demonstrated north-star win**, because on its own trace the
  SLA-aware baseline is indistinguishable from FIFO and the cost denominator
  never moves.

**It is the single most promising research direction in the repo** — request-
level output-length-aware ordering is a real, unexploited lever — but to become
a *credible north-star claim* it needs (1) a trace with real SLA-class labels so
SLA-aware ≠ FIFO, (2) an anti-starvation guard so the long tail stays SLA-safe,
and (3) a cost denominator that can actually move (or an honest re-label of the
metric as "SLA-attainment at fixed fleet," not "goodput/$").
