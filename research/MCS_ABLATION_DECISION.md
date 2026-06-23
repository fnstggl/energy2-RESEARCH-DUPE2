# Forecasted-MCS in AureliusOptimizer — ablation & component decision

> Companion to `research/MCS_AUDIT.md`. Wires the deployable forecasted MCS into
> the canonical optimizer, ablates the optional Aurelius components against it,
> and keeps only the components that improve SLA-safe goodput/$.
>
> Directional simulator evidence only — NOT production savings (`docs/RESULTS.md` §8).
> Results: `research/results/mcs_ablation_backtest_2026-06-23.{json,md}`.

---

## 1. Canonical optimizer wrapper (already merged)

`AureliusOptimizer` (`aurelius/optimizer/`) already exists and is wired into the
benchmark path (Phase 1–3, plus the `ReplicaScalingPolicy` extraction in
`14d3fc9` / OSOTSS in `d35b94e`). No new wrapper was needed. Its decision-layer
registry already exposes `energy`, `serving_queue`, `replica_scaling`,
`placement` (stub), `admission` (stub).

## 2. Forecasted MCS added as the capacity-planning policy

`ReplicaScalingPolicy` gained a new **`mode="forecasted_mcs"`** delegating to
`aurelius/benchmarks/forecasted_mcs.py` (`forecast_method` ∈ ewma / quantile /
lag1). It is the **only** `replica_scaling` mode that uses no future information.

### Integrity finding — `online_sotss` is an *arrival-oracle*, not fully deployable

The pre-existing `online_sotss` mode is labelled "production-deployable … no
future token counts are accessed." That claim is **narrowly true but
incomplete**: OSOTSS makes the *token/service* prediction causal (EWMA) but its
capacity bounds come from `compute_mcs_c_schedule(raw, …)` — the oracle that
buckets **actual tick-`t` arrival counts** — and its violation simulator uses
**actual arrival times** (`replica_scaling.py:595-596, 620-623`). So it still
peeks at how many requests arrive in tick `t` to size tick `t`. It fixes
future-*tokens* but not future-*arrivals*.

| Mode | Future arrivals? | Future tokens? | Classification |
|---|:---:|:---:|---|
| `amcsg`, `sotss_min`, `sotss_gsf` | yes | yes | Oracle upper bound |
| `online_sotss` (OSOTSS) | **yes** | no (causal EWMA) | **Arrival-oracle** (partial) |
| **`forecasted_mcs`** (this PR) | no | no | **Deployable** |

This is quantified in the ablation: OSOTSS scores *above* the true oracle on
goodput/$ (it exploits the SOTSS min-cost loop with perfect arrival knowledge),
while the deployable forecast sits below both — exactly the gap the audit is
about. OSOTSS is left intact and **reclassified**, not removed; it is a useful
upper-middle reference, not a deployable headline.

## 3. Ablation (one physics model, one SLA, one cost denominator)

Components composed with the forecasted-MCS baseline (EWMA, FIFO, provisioned
GPU-hours over the fixed trace window). Full tables in the results file.

| Condition | Azure gp/$ | Δ | BurstGPT gp/$ | Δ |
|---|---:|---:|---:|---:|
| no MCS (best fixed c) | 38,403 | −35.0% | 34,534 | −15.8% |
| oracle MCS (upper bound) | 59,694 | +1.0% | 67,107 | +63.6% |
| OSOTSS (arrival-oracle ref) | 63,831 | +8.0% | 71,244 | +73.7% |
| **forecasted MCS [baseline]** | **59,097** | — | **41,006** | — |
| + queue policy (abs-conformal SRTF) | 57,139 | −3.3% | 56,861 | **+38.7%** |
| + energy routing (real prices) | 58,996 | −0.2% | 40,901 | −0.2% |
| + placement (real GPU menu) | 58,383 | −1.2% | 39,803 | −2.9% |

## 4. Component decision — keep only what improves goodput/$

Keep rule: improves goodput/$ vs the forecasted-MCS baseline by >0.5% **without**
material SLA regression, and (for cost levers) helps MCS more than it helps the
fixed baseline (no free denominator discount).

| Component | Verdict | Evidence |
|---|---|---|
| **Queue policy** (abs-conformal SRTF) | **KEEP (conditional)** | BurstGPT +38.7% (SLA violations 2,020→372): SRTF rescues bursts the forecast under-provisions. Azure −3.3% (capacity already adequate → reordering adds violations). It is a **substitute** for capacity foresight — most valuable exactly when the forecast is wrong (bursty load). Already a first-class `serving_queue` policy; apply it under load, not unconditionally. |
| **Energy routing** (CAISO/PJM/ERCOT) | **DROP** | Energy is **0.17–0.25% of GPU-hour cost** (A10 ~0.4 kW × ~$0.03/kWh ≈ $0.012/gpu-hr vs $2.00 rental). Cheapest-region routing moves goodput/$ by ~+0.4% — and by the **same** amount for the fixed baseline (zero MCS interaction). It is a procurement lever, not a capacity component (same class as the spot-pricing discount the audit excluded). Correctly stays in the **energy/batch** path. |
| **Placement** (heterogeneous GPU menu) | **DROP** | The benchmark's reference GPU (A10, $2.00/hr @ 50 tok/s) is already on the real $/throughput frontier (A10 $0.040, A100 $0.047, T4 $0.066 per tok/s-hr). Per-tick routing to cheaper GPUs (mix stays mostly A10) trades cost for **tail-SLA**: Azure violations 59→131, BurstGPT 2,020→2,259; goodput/$ −1.2%/−2.9%. Matches the prior module-integration regression (−7.3% real KPI). Correctly stays a **disabled stub** (`PlacementPolicy`). |

## 5. Outcome

- **Kept:** queue policy (abs-conformal SRTF), conditionally — it complements
  forecasted MCS on bursty traces and is already an implemented policy.
- **Dropped:** energy routing and placement for the serving/MCS workload — both
  are cost-denominator/hardware levers with no capacity-timing interaction; the
  ablation confirms the canonical architecture's existing gating (energy →
  batch path, placement → off).
- **No north-star inflation:** none of the kept/dropped numbers change the
  audit's verdict. The deployable forecasted MCS remains +54%/+71% over the
  strongest fixed SLA-aware baseline (not +300%); composing the one surviving
  component (queue policy) does not approach the north-star either.

The forecasted-MCS capacity policy is the deployable foundation; the only
component that compounds with it is the serving-queue discipline, and only when
demand is bursty enough that the forecast under-provisions.
