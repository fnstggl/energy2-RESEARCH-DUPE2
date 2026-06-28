# Wide Validation — Current Aurelius MPC (Track A)

**Validate before improving.** PR #114 reported +82.1% SLA-safe goodput/$ over the strongest SLA-aware
baseline — but on a single trace-tail window. Track A re-runs the *unchanged* current MPC (full action layer,
adaptive beam+local search, PR #112 planning/eval parity) against the strongest deployable baselines across
**8 regime-classified windows** to see whether the headline holds. Script:
`scripts/diagnose_wide_validation.py`; artifact: `data/external/mpc_controller/wide_validation.json`.

## Verdict: the direction is robust; the specific +82.1% is NOT a uniform headline

| | |
|--|--|
| MPC beats the strongest baseline on gp/$ in… | **8 / 8 windows** (+25.1% … +144.0%, **median +60.5%**) |
| MPC is Pareto-safe (SLA not worse) in… | **7 / 8 windows** |
| `headline_claim_allowed` across **all** windows | **False** (the `mixed` window sheds SLA) |

**The honest headline is "median +60% (range +25%…+144%), Pareto-safe in 7 of 8 regimes," not a flat +82.1%.**
+82.1% sits at the high end of the range (near the `long_prompt` window, +81.8%); it is not representative of
every regime, and in one regime the gp/$ win is *not* Pareto-safe.

## Per-window results

| regime | periods | fair baseline | Δ gp/$ | Pareto-safe | MPC SLA | baseline SLA |
|--|--|--|--|--|--|--|
| low_load | 269–276 | fifo | +62.3% | ✅ | 0.000 | 0.00 |
| bursty | 8837–8844 | greedy | +38.8% | ✅ | low | low |
| **long_prompt** | 656–663 | sla_aware | **+81.8%** | ✅ | **0.000** | 0.068 |
| long_output | 8902–8909 | greedy | +25.1% | ✅ | 0.018 | 0.026 |
| high_sla_pressure | 6235–6242 | greedy | +144.0% | ✅ | 0.050 | 0.077 |
| **mixed** | 1768–1775 | sla_aware | +80.3% | ❌ | **0.050** | 0.027 |
| tail | 10074–10079 | greedy | +58.6% | ✅ | low | low |
| long_24p (≈24 periods) | 10056–10079 | sla_aware | +53.8% | ✅ | low | low |

## Why one window fails Pareto — and why that's the gate working, not a bug

In `mixed` and `high_sla_pressure` the MPC selected **int4 precision in all 8 periods**. int4 carries a
conservative **0.05 quality-failure risk** (a wrong answer is not SLA-safe goodput — `PRECISION_QUALITY_RISK`
since PR #111), which surfaces as `sla_violation_rate = 0.050`. In `high_sla_pressure` the baseline (greedy)
is even worse (0.077), so the MPC is still Pareto-safe; in `mixed` the baseline (sla_aware) is better (0.027),
so the MPC's int4 choice makes SLA **worse** → the Pareto gate correctly returns `headline_claim_allowed =
False` for that window. The MPC bought +80% gp/$ by accepting answer-quality risk the baseline didn't — and
the gate caught it. Clean wins (`long_prompt`, `tail`) use **fp8** (zero quality risk).

## Representative full-metric block (long_prompt, +81.8%, Pareto-safe)

| metric | current MPC | sla_aware |
|--|--|--|
| SLA-safe gp/$ | **196 604** | 108 165 |
| SLA violation rate | **0.000** | 0.068 |
| GPU-hours | 0.181 | 0.256 |
| GPU-seconds | 307.2 | 386.9 |
| operator cost ($) | 0.08 | 0.11 |
| serving energy (J)¹ | 193 843 | 0 |
| TTFT p95 (s) | 1.127 | 1.042 |
| completion p95 / p99 (s) | 8.07 / 8.07 | 8.45 / 8.45 |
| precision / spec / clock mix | fp8×8 / off×8 / base×4 low×4 | (baseline: no roofline actions) |

¹ `energy_j` is a **roofline-action diagnostic** — populated only for arms that exercise the precision/clock
modulation (the MPC), so baselines read 0 here (their energy is not surfaced through that channel, not zero
physically).

## Method + honesty

- **Same harness that produced the headline.** Current main behaviour — no controller / forecaster / simulator
  change. Baselines (`fifo`, `greedy`, `sla_aware`) are deployable fixed policies (no oracle, no future info).
  The fair baseline per window is the strongest of the three by gp/$.
- **Regime selection** scans contiguous 8-period spans and picks the extreme window per dimension
  (arrival / output-median / prompt-median / interarrival-CV) plus a "mixed" central window, the tail, and a
  ≈24-period long window. Forecasters are trained on history strictly before the earliest evaluated period (no
  leakage).
- **Bounded windows → magnitudes are simulator-inferred.** The robust finding is the **direction and the
  spread** (MPC beats every baseline on gp/$; Pareto-safe in most but not all regimes), not any single
  percentage. The clock mix already shows the planner picking `low` in several windows even at the constant
  fleet price — Track D probes whether a *time-varying* price makes that choice price-driven.

**Bottom line for downstream tracks:** the MPC's gp/$ advantage is real and wide, but it is not uniformly
Pareto-safe, and the magnitude is window-dependent — so this PR reports the **range and median**, not +82.1%,
and treats the int4 SLA-shedding regime as a known failure mode the gate already blocks.
