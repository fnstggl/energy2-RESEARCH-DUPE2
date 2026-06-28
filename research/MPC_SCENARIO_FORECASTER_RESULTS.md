# MPC Scenario Forecaster + Oracle Diagnostic (small PR)

Is **forecast fidelity** — not search or serving physics — the next limiting factor for Aurelius' MPC? PR
#112 left the planner correctly cost-aware (it selects FP8) but net-regressing, because its single-median
planning workload under-represented the SLA pressure evaluation later sees. This PR replaces that median
with a small **trace-derived scenario ensemble** (`scenario_forecaster.build_scenarios`) and adds an
**oracle** planning mode (plan against the exact future), then decomposes the gap.

`scripts/diagnose_mpc_scenario_forecaster_dt60.py` →
`data/external/mpc_controller/mpc_scenario_forecaster_dt60.json` (dt=60, 120 eval periods, 12 MPC decisions,
median prompt 828). All four planners hold planning/eval **cost-parity ON** (PR #112), so only the
**workload model** varies: median → ensemble → exact future. No ML, no external simulator — scenarios are
statistical extrapolations of the forecast trajectory (mean/p90/p10/p99) with a fixed risk-averse weight
prior (not tuned). Reward = `E[gp/$] − risk·E[SLA] − tail·max(SLA)` (no reward shaping).

## Result

| arm | gp/$ | SLA | precision selected | spec selected |
|--|--|--|--|--|
| Static fp8+spec (best FIXED bundle) | 149 646 | 0.0000 | fp8 | medium |
| **Current** — median planning (PR #112) | 149 164 | 0.0112 | bf16/fp8 mix | aggressive/medium/off |
| **Scenario** — SLA-pressure ensemble (this PR) | **171 485** | **0.0000** | **fp8 ×12** | off ×11 |
| **Oracle** — exact future (diagnostic) | 174 062 | 0.0084 | bf16/fp8/int4 mix | mixed |

## Regret decomposition — `Current → Scenario → Oracle` (Static = best fixed bundle)

```
Current 149 164  →  Scenario 171 485  →  Oracle 174 062      (Static fixed: 149 646)
            +22 321 (ensemble)     +2 576 (perfect forecast)
```
- **Scenario − Current = +22 321** — the workload-model fix this PR adds. The SLA-pressure ensemble makes the
  planner pick a bundle that is both cheaper (fp8, no spec compute tax) **and** SLA-robust (SLA 0.0000 vs the
  median planner's 0.0112). **PR #112's regression is reversed.**
- **Oracle − Scenario = +2 576** — what a *perfect* forecast would add beyond the lightweight ensemble: only
  ~11% of the ensemble's own gain. **The ensemble captures ~90% of oracle planning.**
- **Scenario/Oracle − Static = +21 839 / +24 416** — both adaptive arms **exceed** the best fixed bundle, so
  "static optimum" is a **floor, not a ceiling**: per-period adaptation (mixing precision/spec by period)
  beats any single fixed stack.

## Success criteria — the questions answered

1. **Does better workload prediction improve MPC decisions?** **Yes, decisively** — +22 321 gp/$ (+15%) over
   median planning, with SLA *improving* to 0.0000. A faithful (SLA-pressure-aware) planning workload is the
   single biggest planner lever found so far.
2. **Does oracle planning recover the FP8+spec bundle?** It **exceeds** it — Oracle and Scenario both beat
   static fp8+spec by ~15%, because the controller adapts precision/spec **per period** instead of fixing one
   bundle. Recovering the static stack was the wrong target; the controller does better.
3. **Is forecasting now the dominant source of planner error?** **It was — and the ensemble nearly closes
   it.** The median→ensemble jump (+22 321) dwarfs the ensemble→oracle residual (+2 576, ~11%). So the
   limiting factor was the planning **workload model**, now largely fixed by a lightweight trace-derived
   ensemble. Not search (PR #112: beam ≈ exhaustive), not serving physics.
4. **If not, what is?** The small Scenario→Oracle residual (+2 576) is the remaining forecast imperfection (a
   perfect forecast would add a little more). Beyond that, the controller already exceeds the static optimum.

## Honesty + scope

Small by design: one forecaster module, one opt-in ensemble/oracle planning path (**default off** → the live
controller is unchanged), one comparison, four focused tests. **Caveats:** this is a **bounded 12-period
window**; the absolute magnitudes are simulator-inferred and window-sensitive (PR #112's 32-period median arm
scored lower) — the **robust** finding is the **direction**: the SLA-pressure ensemble fixes the planner's
workload model and forecasting was the dominant error source. The **oracle uses future information** and is a
labelled diagnostic, never a deployable policy. Given the strength + Pareto-safety of the result, the
scenario ensemble is a strong candidate to become the **default** planning workload — but on this bounded
evidence it ships **opt-in** (`planning_scenarios`), pending wider-window validation. The deliverable is the
**attribution**, not a gp/$ headline: forecast fidelity *was* the next limiting factor, and a cheap ensemble
largely closes it.
