# Run-g SRTF — Stronger-Baseline Replay (Audit, 2026-06-20)

> **Audit / reporting artifact. Simulator / public-trace directional result —
> NOT production savings** (`docs/RESULTS.md` §8). Produced by
> `scripts/audit_srtf_stronger_baselines.py`, which REUSES the unmodified run-g
> physics, trace loader, time-warp and goodput/$ definition from
> `aurelius/benchmarks/srtf_serving_backtest.py`. It only adds extra queue
> *disciplines* (baselines) to the same M/G/c simulator. No runtime path changed.

## Why this exists

The run-g headline is **"+323% SLA-safe goodput/$ vs FIFO"**. FIFO is the
weakest possible baseline. The Aurelius north-star rollup (+26% on Azure 2024)
is measured vs `sla_aware`. This harness re-runs run-g's own simulator against
`sla_aware` (earliest-deadline-first) and a preemptive reference, to find the
real delta against a stronger baseline.

## Headline table (Azure LLM 2024 sample, c=4, ρ=0.85, SLA=10 s, warp=21.95×)

| Policy | Dataset | SLA-safe goodput/$ | Δ vs FIFO | Δ vs SLA-aware (EDF) | Δ vs constraint-aware | SLA violations | Queue p99 (s) | Cost ($) | GPU-hours |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fifo | Azure 2024 (5,880) | 13,336.3 | 0.0% | 0.0% | N/A | 83.62% | 730.3 | 8.05 | 4.02 |
| sla_aware (EDF) | Azure 2024 (5,880) | 13,336.3 | 0.0% | 0.0% | N/A | 83.62% | 730.3 | 8.05 | 4.02 |
| srtf_perfect | Azure 2024 (5,880) | 56,480.8 | **+323.5%** | **+323.5%** | N/A | 14.17% | 2,181.7 | 8.05 | 4.02 |
| srtf_forecast (30% CV) | Azure 2024 (5,880) | 56,854.8 | +326.3% | +326.3% | N/A | 14.90% | 2,224.9 | 8.05 | 4.02 |
| srpt_reference (preemptive) | Azure 2024 (5,880) | 56,311.4 | +322.2% | +322.2% | N/A | 14.17% | — | 8.05 | 4.02 |

Long-request p99 response (starvation cost): FIFO 733.6 s → SRTF 2,373.1 s →
SRPT 2,372.6 s.

## The three decisive findings

1. **SLA-aware == FIFO on this trace.** `sla_aware_edf` produces an *identical*
   goodput/$ (13,336.3) to FIFO at every load (ρ ∈ {0.80, 0.85, 0.92} all show
   `EDF == FIFO: True`). The Azure LLM 2024 trace has **no request_type /
   SLA-class field** (its only columns are `TIMESTAMP, ContextTokens,
   GeneratedTokens`), so every request gets the *same* SLA budget. Under a
   uniform deadline, earliest-deadline-first is mathematically arrival order =
   FIFO. **So "+323% vs FIFO" and "+323% vs SLA-aware" are the same number here —
   but only because the trace cannot express a differentiated SLA-aware policy.**
   A real differentiated SLA-aware baseline would require synthesizing class
   labels the trace does not contain; this audit refuses to do so for any
   headline.

2. **The cost denominator is constant.** GPU-hours (4.02) and cost ($8.05) are
   *identical* across every discipline by construction (same request set, same
   service times, same server pool). The "+323% goodput/$" is therefore a pure
   **SLA-attainment (latency) effect at fixed cost** — SLA violations fall 83.6%
   → 14.2% — **not a cost reduction.** This is categorically different from the
   +26% rollup, whose win is −21% GPU-hours (a genuine cost-efficiency / cost-
   denominator effect from the provisioning decision).

3. **`constraint_aware` has no analog here (N/A).** It is a provisioning /
   region / energy-timing policy; its decision surface is orthogonal to single-
   queue request ordering and has no expression in a fixed-c single-queue
   simulator. The two policies do not act on the same decision surface, so
   "Δ vs constraint-aware" is undefined in this harness — not zero, not a tie.

## ρ sweep (perfect prior — matches run-g's own table)

| ρ | srtf_perfect Δ vs FIFO=SLA-aware | SLA violations | long p99 (s) |
|---|---:|---:|---:|
| 0.80 | +252.2% | 12.31% | 2,237.8 |
| 0.85 | +323.5% | 14.17% | 2,373.1 |
| 0.92 | +314.0% | 16.58% | 2,469.7 |

The preemptive SRPT reference (+322% with the *same* long p99 ≈ 2,373 s) shows
the long-request starvation is near-intrinsic to shortest-first at this load —
preemption alone does not fix it; an aging / hybrid-band guard is required.

## Reproduce

```bash
PYTHONPATH=. python scripts/audit_srtf_stronger_baselines.py            # table
PYTHONPATH=. python scripts/audit_srtf_stronger_baselines.py --json     # data
```
