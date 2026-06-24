# MCS Audit — is Min-Cost-Safe an oracle or a deployable policy?

> **Status: AUDIT.** Scope: every MCS ("Min-Cost-Safe") capacity-provisioning
> implementation in the repo. Question: does MCS *forecast* next-tick demand and
> pick minimum safe capacity (deployable), or does it *peek* at the current
> tick's realised demand and pick the perfect capacity (oracle)?
>
> Directional simulator evidence only — NOT production savings
> (`docs/RESULTS.md` §8).

---

## 0. Executive summary

**Every MCS implementation in the repo is an oracle at the capacity-decision
level.** They all size the per-tick replica count `c[t]` from the **actual**
requests that arrive in tick `t` — using both the **actual arrival count** and
the **actual output-token counts** of those requests. Output tokens are not
known until a request *finishes*, so this is doubly clairvoyant.

The whole family reduces to one function:
`_joint_mcs_c_schedule()` (`aurelius/benchmarks/srtf_serving_backtest.py:7495`).
The spot-fleet / AMCSG / GSF / ZFHC / AFMS / DLAG / SOTSS "policies" are
**pricing and gate overlays on the same oracle schedule** — they change the
cost denominator (spot discount) or the Erlang-C gate %, not the information
set used to choose capacity. The original `min_cost_safe` policy in
`aurelius/traces/backtest.py` is the same idea in a different physics model and
is documented in its own source as a *"per-tick minimum-replica oracle."*

Three findings make the published headline numbers misleading:

1. **Oracle capacity.** `c[t]` is chosen with perfect knowledge of tick `t`'s
   arrivals and output tokens. Not deployable. (§3, §4)

2. **Inconsistent / under-sized baseline.** The leaderboard's "SLA-aware oracle"
   baseline (Azure 25,208 goodput/$) is `sla_aware` ordering at a **fixed c=4** —
   which is *under-provisioned* (2,475 SLA violations, p99 queue 844 s). A
   competently-sized fixed baseline is c=7 (Azure → 38,403 goodput/$) / c=8
   (BurstGPT → 34,534). Comparing MCS to the c=4 baseline inflates the gain
   roughly 2×. (§5)

3. **Cost-denominator trick.** The remaining distance from "+136.8%" (on-demand)
   to "+497%" is bought entirely by applying a 40–70 % **spot-instance discount**
   to the MCS fleet's cost denominator. The discount is orthogonal to MCS — it
   would help *any* policy (including the fixed baseline) by the same factor — so
   it is not an MCS/forecasting/capacity result. The spot-fleet doc itself says
   "the north-star gap … is closed entirely by the cost denominator." (§5)

When the audit re-runs the comparison **fairly** (one provisioned-GPU-hour cost
denominator over a fixed billing window, one SLA, one simulator, baseline c
swept to its best operating point), the real picture is:

| | Azure LLM 2024 | BurstGPT HF |
|---|---:|---:|
| Strongest fixed SLA-aware baseline (swept c) | 38,403 (c=7) | 34,534 (c=8) |
| **Oracle MCS** (upper bound) | 59,694 (+55%) | 67,107 (+95%) |
| **Best deployable forecast MCS** (this PR) | 59,097 (+54%) | 58,953 (+71%) |
| Forecast retains of oracle | **99.0 %** | **87.8 %** |
| Published leaderboard headline | +497% | +730% |

So MCS *does* deliver a genuine, deployable efficiency gain — **same SLA safety
at 35–46 % fewer GPU-hours** vs the best fixed fleet — and a simple causal
forecast captures ~88–99 % of the oracle's value. But it is **+54 %/+71 %**, not
+300 %/+497 %. **North-star (+300 % over the strongest SLA-aware MCS-enabled
baseline) is not achieved by any deployable MCS — nor by the oracle.** The
published figure was baseline under-sizing × spot pricing stacked on an oracle.

See the forecasted-MCS implementation and full numbers in
`research/results/forecasted_mcs_backtest_2026-06-23.md` (Phases 3–6).

---

## 1. What "MCS" stands for, and the family tree

**MCS = Min-Cost-Safe**: per tick, choose the *smallest* replica count `c` whose
modelled SLA-violation (queue-timeout) probability is below a gate (default
9.5 % < 10 % target). "Minimum cost" (fewest replicas) subject to "safe" (gate).

All variants are overlays on the same per-tick min-`c` Erlang-C search:

| Acronym | Expansion | What it adds on top of the MCS schedule |
|---|---|---|
| **MCS** | Min-Cost-Safe | the base per-tick min-`c` Erlang-C schedule |
| **AMCSG** | Adaptive MCS Gate (sweep) | sweeps the gate % (9.5→12.5) for max safe gate |
| **AMCSG-LFC** | AMCSG + Lower Fixed-C Calibration | lowers `fixed_c` 4→3 in the warp calibration |
| **DLAG** | Dynamic Load-Aware Gate | per-tick gate `= f(tick ρ)` (looser when idle) |
| **AFMS** | Absolute-Floor Max-Spot | spot pricing + 1 on-demand replica floor |
| **ZFHC** | Zero-Floor High-Capacity | spot pricing, drops on-demand floor when c≥thr |
| **GSF** | Graduated Spot Fleet | spot pricing with graduated spot fraction |
| **SOTSS** | **Simulation-Oracle** Tick-Selective Schedule | oracle loop that increments c on simulated violators |

The acronym **SOTSS literally contains "Oracle."** Its source comment reads:
*"Run SOTSS oracle loop (deterministic, uses actual tokens). The oracle can see
actual token counts — it is an offline capacity planner."*
(`srtf_serving_backtest.py:11383`).

---

## 2. Physics model (shared by the srtf-family MCS)

All srtf-family MCS share one physics model (`srtf_serving_backtest.py`):

- **Service time:** `s_i = TTFT_BASE_S(0.150) + output_tokens · TPOT_S(0.020)`
  i.e. sequential decode at ~50 tok/s/replica (`:156`, `:270`).
- **Gate:** Erlang-C M/M/c `P(wait > sla_s − mean_service) < gate%`
  (`_erlang_c_sla_timeout_pct`, `:7449`). M/M/c is a conservative approximation
  of the M/D/c queue actually simulated.
- **Capacity search:** smallest `c∈[1,1024)` meeting the gate (`:7551`).
- **Simulator:** non-preemptive FIFO M/G/c with per-tick variable `c`
  (`_simulate_fifo_variable_c`, `:7562`); the abs-conformal SRTF variant is
  `_simulate_abs_conformal_variable_c` (`:7642`).
- **Time-warp:** real arrivals are linearly rescaled so the trace runs at
  `target_rho=0.85` on `fixed_c=4` (`calibrate_time_warp`, `:376`). Azure's
  ~26 h / 5,880-req fixture compresses to 72 ticks of 60 s; BurstGPT to 154.
- **Cost denominator (MCS path):** provisioned GPU-hours
  `= Σ_t c[t] · tick_hr · $2.00/hr` (`GPU_HOUR_USD`, `:158`).
- **SLA:** E2E response ≤ `DEFAULT_SLA_S=10 s` (Azure) / `30 s` (BurstGPT)
  (`:162`, `:166`).

The original `min_cost_safe` (`aurelius/traces/backtest.py`) uses a **different**
physics model: continuous-batching throughput `FALLBACK_TOKENS_PER_S=2500`
(`:57`), Erlang-C via `serving.erlang_c_wait_s()`, gate `_MCS_TIMEOUT_GATE=9.5`
(`:106`), cost = GPU-hours by model-mix + energy + migration
(`compute_economic_kpi`). The two physics models are **not interchangeable** —
hence "Incompatible physics" appears in the classification.

---

## 3. The oracle, in code

`_joint_mcs_c_schedule` (`srtf_serving_backtest.py:7530-7557`), abridged:

```python
buckets = [[] for _ in range(n_ticks)]
for t, tok in warped:                      # ALL requests, including tick t's
    idx = min(n_ticks - 1, int(t / tick_seconds))
    buckets[idx].append(tok)               # bucket[t] = tick t's ACTUAL arrivals

for bucket in buckets:                      # size c[t] FROM tick t's own bucket
    n_req = len(bucket)                     # ACTUAL arrival count in tick t
    lam = n_req / tick_seconds              # ACTUAL arrival rate in tick t
    mean_service = statistics.mean(         # ACTUAL output tokens of tick t
        _service_time_s(tok) for tok in bucket)
    for c in range(1, 1024):
        if _erlang_c_sla_timeout_pct(lam, mean_service, c, sla_wait) < mcs_gate:
            chosen = c; break               # perfect minimum c for tick t
```

`bucket[t]` is exactly the set of requests that arrive **during** tick `t`. The
capacity *for* tick `t` is computed *from* tick `t`'s realised arrivals and
realised output tokens. This is the "peek at current tick demand → choose
perfect c" pattern the task targets. `_joint_mcs_dlag_c_schedule` (`:10650`) and
`_sotss_min_cost_schedule` (`:11137`) do the same with a per-tick gate / an
oracle simulation loop respectively.

**Verified empirically:** reproducing this on the Azure fixture yields exactly
the documented `c̄=4.50, min=1, max=8, n_ticks=72`, cost $10.80, 59,694
goodput/$, p99 9.95 s.

---

## 4. Per-implementation audit

All paths in `aurelius/benchmarks/srtf_serving_backtest.py` unless noted. "Future
arrivals?" / "Actual tokens?" describe the **capacity decision**.

| # | Symbol / runner | MCS root | Future arrivals? | Actual tokens? | Oracle? | Deployable? | Cost denom | Benchmark / baselines |
|---|---|---|:---:|:---:|:---:|:---:|---|---|
| 1 | `_joint_mcs_c_schedule` `:7495` | self | **yes** | **yes** | **yes** | no | provisioned GPU-h | core; used by all below |
| 2 | `_joint_mcs_dlag_c_schedule` `:10650` (DLAG) | #1-style | **yes** | **yes** | **yes** | no | provisioned GPU-h | `run_dlag_backtest.py`; vs AMCSG |
| 3 | `_sotss_min_cost_schedule` `:11137` (SOTSS) | #1 + sim loop | **yes** | **yes** | **yes** | no | spot provisioned | `run_sotss_backtest.py`; vs AMCSG |
| 4 | `run_joint_mcs_abs_conformal_*` `:7878` | #1 | **yes** | **yes** | **yes** | no | provisioned GPU-h | 2×2 {FIFO,absconf}×{fixed,MCS} |
| 5 | `run_spot_fleet_mcs_*` `:8249`,`:8393` | #1 + spot | **yes** | **yes** | **yes** | no | **spot** provisioned | vs SLA-oracle 25,208/20,280 |
| 6 | `run_abs_floor_spot_fleet_mcs_*` `:8820`,`:8881` (AFMS) | #1 + spot + floor | **yes** | **yes** | **yes** | no | **spot** provisioned | vs SLA-oracle |
| 7 | `run_zfhc_*` `:9288`,`:9354` (ZFHC) | #1 + spot | **yes** | **yes** | **yes** | no | **spot** provisioned | vs AFMS / SLA-oracle |
| 8 | `run_gsf_*` `:9778`,`:9850` (GSF) | #1 + spot | **yes** | **yes** | **yes** | no | **spot** provisioned | vs ZFHC / SLA-oracle |
| 9 | `run_amcsg_*` `:10207`,`:10274` (AMCSG) | #1 + spot + gate | **yes** | **yes** | **yes** | no | **spot** provisioned | gate sweep; vs GSF |
| 10 | `run_amcsg_lfc_*` `:10372`,`:10436` | #1 (fixed_c=3) | **yes** | **yes** | **yes** | no | **spot** provisioned | LFC null-result run |
| 11 | `run_amcsg_(lfc_)fine_grid_*` `:10497`,`:10560` | #1 | **yes** | **yes** | **yes** | no | **spot** provisioned | fine gate grid |
| 12 | `run_sotss_*` `:11452`,`:11527` | #3 (+#1 ref) | **yes** | **yes** | **yes** | no | **spot** provisioned | vs AMCSG; "+500% north-star" |
| 13 | `_min_cost_safe_replicas` `traces/backtest.py:467` | self (other physics) | **yes** | **yes** | **yes** | no | GPU-h + energy + migration | `run_min_cost_safe_backtest.py`; vs SHU/CA/sla_aware (deployable) |

Notes:
- The queue-ordering side (`predicted_tokens`, abs-conformal calibrator,
  `make_live_prior_predictions`) **is** causal/deployable. The audit is about
  **capacity provisioning** (`c[t]`), which is oracle in every row above.
- SOTSS builds its *queue* requests with a live causal prior (`:11351`) but its
  *capacity* schedule `c_sotss` comes from the oracle loop (`:11386`).
- Row 13's baselines (`sla_aware`, `constraint_aware`, `safe_high_utilization`)
  are **deployable** (lag-1 / EWMA) — so even there, an oracle is compared
  against deployable baselines.

---

## 5. Why the headline numbers are inflated (two independent effects)

**(a) Under-sized baseline.** The leaderboard's SLA-aware baseline is fixed
`c=4` (Azure 25,208 / BurstGPT 17,189 goodput/$). The fixed-`c` sweep (this PR,
same denominator) shows `c=4` is far from optimal:

| fixed c | Azure sla_aware gp/$ | Azure SLA viol | BurstGPT sla_aware gp/$ | BurstGPT SLA viol |
|---:|---:|---:|---:|---:|
| 4 (*documented baseline*) | 25,208 | 2,475 | 17,189 | 2,741 |
| 6 | 31,880 | 1,174 | 21,484 | 1,975 |
| **7 / 8 (*best*)** | **38,351** | 59 | **34,468** | 195 |
| 12 | 22,431 | 54 | 24,389 | 11 |

Against the **best** fixed baseline, oracle MCS is +55 %/+95 % — not +137 %.

**(b) Spot pricing.** On-demand FIFO+MCS is 59,694 (Azure). The "+304.7 %"/"+497 %"
figures multiply the *cost denominator* by a spot discount (`spot_fraction≈0.7–0.95`,
`spot_price≤$0.80` vs `$2.00`). That discount is **policy-independent**: apply it
to the fixed baseline and its goodput/$ rises by the same factor, leaving the
*relative* MCS gain unchanged. Crediting it to MCS conflates a procurement choice
with a provisioning algorithm.

**Decomposition of the published "+497 %" (Azure):**
`+497%  ≈  (oracle capacity vs under-sized c=4 baseline: ~+137%)  ×  (spot
discount on the denominator: ~×2.5)`. Neither factor is a deployable forecasting
gain, and neither is a GPU-hour saving — **GPU-hours go up** vs the under-sized
baseline (4.8→5.4 h).

---

## 6. Classification

| Implementation | Classification | Rationale |
|---|---|---|
| `_joint_mcs_c_schedule` (#1) and all spot/gate overlays (#4–#12) | **Oracle upper bound** | sizes `c[t]` from tick-`t` actual arrivals + actual tokens |
| `_joint_mcs_dlag_c_schedule` (#2, DLAG) | **Oracle upper bound** | same, with per-tick gate from tick-`t` actual ρ |
| `_sotss_min_cost_schedule` / `run_sotss_*` (#3, #12) | **Oracle upper bound** | explicit oracle loop over actual-token simulation |
| `min_cost_safe` (`traces/backtest.py`, #13) | **Oracle upper bound** *and* **Incompatible physics** | per-tick actual-demand oracle; continuous-batching model ≠ srtf sequential model |
| AMCSG-LFC fixed_c=3 runs (#10) | **Deprecated** | documented three-lever **null result** (run 2026-06-23) |
| Spot-pricing overlays as a *cost* model (#5–#12) | **Benchmark-only** | spot discount is a procurement assumption, valid only as a sensitivity, never a headline comparator |

No existing MCS implementation is **Deployable**. The deployable forecasted MCS
added in this PR (`aurelius/benchmarks/forecasted_mcs.py`) is the first.

---

## 7. What a deployable MCS must do instead (handed to Phases 3–6)

Replace the tick-`t` peek with a strictly causal forecast made at the `t-1→t`
boundary, keeping **everything else identical** (gate physics, service model,
simulator, cost denominator, SLA):

- forecast next-tick arrival count (EWMA / rolling quantile of past ticks);
- forecast next-tick mean service (EWMA of past per-tick mean service);
- size `c[t]` from the **forecast** via the same Erlang-C gate;
- evaluate by replaying the **actual** tick-`t` requests through the same FIFO
  simulator and the same provisioned-GPU-hour denominator.

Implemented and benchmarked in this PR — see
`research/results/forecasted_mcs_backtest_2026-06-23.md`. Result: deployable
forecast MCS retains 88–99 % of the oracle and beats the strongest fixed
SLA-aware baseline by +54 %/+71 %, but **does not** reach +300 %.
