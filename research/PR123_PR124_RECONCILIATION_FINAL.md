# PR #123 (+1273%) vs PR #124 (+164%) — Final Scientific Reconciliation

A controlled, variable-isolation diagnostic explaining the headline gap **exactly**. Same pjm·expensive window
(period 81), same `hierarchical_search`, same winning bundle, same `sla_aware` policy. **Conclusion up front:
`hierarchical_search` did not regress and the winning bundle did not change — the headline difference is almost
entirely the EVALUATION HARNESS** (a single forecast-rollout decision vs a multi-period real-trace episode),
with episode horizon, request cap, and electricity pricing each contributing only a few points. No simulator /
reward / gate / cost / action / baseline / planner change; no tuning; magnitudes SIMULATED. Evidence:
`scripts/diagnose_reconciliation_matrix.py` → `data/external/mpc_controller/reconciliation_matrix.json`
(supersedes the first-pass `PR123_VS_PR124_HEADLINE_RECONCILIATION.md` with a one-variable-at-a-time matrix).

## Method

Hold the **two action bundles fixed** — `sla_aware` (`SAFE_BASELINE_BUNDLE`) and the committed hierarchical
winner (`fp8`/`clock=high`/`kv_aware`/`network_aware`/`forecasted_mcs`/`aggressive`/`×0.75`/`spec=aggressive`) —
and score *those exact bundles* through each harness, changing **one** experimental variable at a time. Because
the action is fixed, every gp/$ change is attributable to the *measurement*, not to the planner choosing
differently. Variables: evaluation harness (A↔B), episode horizon (1↔3), request cap (80↔56), electricity
pricing (constant↔real).

- **Harness A** (PR #123 tournament): one decision via `_rollout_world` (H=1) — serves **synthetic jobs from the
  forecast distribution**; gp/$ = the single step's `gp_per_dollar`.
- **Harness B** (PR #124 ladder): `run_period_episode` over **real trace requests** through the persistent
  world simulator; gp/$ = episode goodput/$.

## The experiment matrix (fixed bundles)

| cell | harness | req_cap | periods | price | hier gp/$ (SLA) | sla_aware gp/$ (SLA) | hier vs sla_aware |
|--|--|--|--|--|--|--|--|
| **E0 = PR123** | A (forecast) | 80 | 1 | const | 1,454,636 (0.288) | 105,924 (0.374) | **+1273.3%** |
| E_reqcap (A) | A | **56** | 1 | const | 1,435,307 (0.289) | 105,831 (0.361) | +1256.2% |
| E_elec (A) | A | 80 | 1 | **real** | 1,397,004 (0.288) | 102,489 (0.374) | +1263.1% |
| E_harness | **B (real)** | 80 | 1 | const | 1,181,570 (0.000) | **344,919** (0.288) | **+242.6%** |
| E_horizon | B | 80 | **3** | const | 1,117,070 (0.000) | 352,581 (0.275) | +216.8% |
| E_bridge | B | **56** | 3 | const | 824,249 (0.000) | 309,534 (0.226) | +166.3% |
| **E5 = PR124** | B | 56 | 3 | **real** | 778,205 (0.000) | 295,338 (0.226) | **+163.5%** |

E5 reproduces the committed PR #124 (783,862 / 295,338 / +165.4%) to within **0.7%** (the residual is fixed
bundle vs re-searched bundle — i.e. the search re-finds essentially the same bundle). E0 reproduces the
committed PR #123 (+1273.28%) **exactly**.

## Cumulative bridge — one variable per step (PR #123 → PR #124)

| step (change one variable) | hier vs sla_aware | Δ from prior | share of total gap |
|--|--|--|--|
| PR #123 (E0) | +1273.3% | — | — |
| + evaluation harness A→B (forecast→real-trace) | +242.6% | **−1030.7 pts** | **~93%** |
| + episode horizon 1→3 (+ aggregation) | +216.8% | −25.7 pts | ~2.3% |
| + request cap 80→56 | +166.3% | −50.5 pts | ~4.6% |
| + electricity const→real = PR #124 (E5) | +163.5% | −2.8 pts | ~0.3% |

Single-variable-from-PR123 marginals confirm the same: request_cap alone (in Harness A) **−17 pts**, electricity
alone **−10 pts**, **harness alone −1031 pts**. The harness dominates by ~30–60×.

## Why the harness dominates

The **same** `sla_aware` bundle scores **105,924** serving forecast-synthetic jobs (Harness A) but **344,919**
serving the real trace requests (Harness B) — a **3.26× denominator jump**. Meanwhile hierarchical *drops*
1.23× (1,454,636 → 1,181,570). The forecast-rollout at req_cap 80 / median-prompt 1181 is a **heavy,
compute-bound synthetic workload** that the bf16/no-batching baseline serves badly (SLA 0.374, gp/$ 106k) while
the fp8/aggressive optimized bundle serves well — inflating the ratio to 13.7×. The real-trace episode is
lighter and far more favorable to the baseline (SLA 0.226), compressing the ratio to 2.6×. **The +1273% is the
optimized-vs-naive gap amplified by a harsh synthetic single-period workload; the harness, not the optimizer, is
the lever.**

## The 15 questions

**1. Did `hierarchical_search` regress?** **No.** Same method, same chosen bundle, search regret 0% in both. Its
*absolute* gp/$ is in fact **higher** in PR #123's harness (1,454,636 vs 778,205).

**2. Did the winning bundle change?** **No.** Identical surfaces in both harnesses (`forecasted_mcs` · `kv_aware`
· `network_aware` · `class_aware` · `×0.75` · `aggressive` · `fp8` · `clock=high` · `spec=aggressive`). The
fixed-bundle E5 reproduces the re-searched PR #124 within 0.7%.

**3. Did the baseline policy change?** **No.** `SAFE_BASELINE_BUNDLE` (PR #123) == `SLA_AWARE_FALLBACK` (PR #124)
by value: `backlog_aware` + `abs_conformal` + admission off, all else no-op. Same policy, two harnesses.

**4. Which variables explain the gap?** Evaluation harness (**~93%**), request cap (~4.6%), episode horizon /
aggregation (~2.3%), electricity pricing (~0.3%).

**5. Largest effect?** **The evaluation harness** — forecast-synthetic single rollout vs real-trace persistent
episode: **−1031 of the −1110 percentage-point gap.**

**6. Negligible effects?** Electricity pricing (−2.8 pts cumulative; −10 pts alone) and — in the forecast
harness — request cap (−17 pts alone) and electricity (−10 pts alone). Request cap matters more (−50 pts) once
inside the real-trace episode (it then caps actual served requests).

**7. Is the +1273% result still scientifically correct?** **Yes — for the question it answers.** It is a correct
measurement of the SLA-safe gp/$ improvement of the hierarchical bundle over the `sla_aware` bundle **on one
forecast-rollout decision at req_cap 80**. It passed its Pareto check there (SLA 0.288 < 0.374). It is *not* a
deployment estimate.

**8. Under what benchmark definition is +1273% correct?** The **PR #123 search-method-tournament** definition:
**one** planning decision, **forecast-distribution synthetic** workload, **single-period** gp/$, equal
*evaluation budget* across methods. Its scientific purpose was **ranking search methods** under a fixed budget
on a hard single decision — and hierarchical *did* win there (regret 0). The +1273% is the magnitude of that
win *in that harness*, valid as a method-ranking figure, not as a fleet deployment number.

**9. Is the +164% result still scientifically correct?** **Yes.**

**10. Under what benchmark definition is +164% correct?** The **PR #124 ladder/episode** definition: **real
trace requests**, **persistent world** (cold/warm/migration state evolves period→period), **3 decisions**
aggregated, **real diurnal electricity prices**, Pareto-gated vs `production_scheduler` and `sla_aware`.

**11. Which benchmark should Aurelius freeze for future reporting?** **The PR #124 episode benchmark.**

**12. Why?** It is the deployment-representative measurement: it serves the **actual** workload (not
forecast-synthetic), evolves **persistent state** (so warm pools / migration / cold starts are paid for
honestly), **aggregates over multiple decisions** (receding-horizon realism), and prices energy at **real**
diurnal prices. The forecast rollout is an **internal planning primitive** (how the controller *scores
candidates*), not an evaluation harness — using it to *report* results conflates "how the planner thinks" with
"what actually happened."

**13. Should public claims use the PR #123 or the PR #124 methodology?** **PR #124.** The PR #123 percent is
harness-inflated and is a method-ranking artifact; publishing it as a deployment gain would overstate by ~8×.

**14. What protocol should become the permanent Aurelius Benchmark v1?** The PR #124 episode protocol, pinned:
(a) **harness** = `run_period_episode` over real per-period trace requests through a persistent `world_state`;
(b) **workload** = a named window set (pjm/ercot/caiso · expensive [+ cheap/volatile/long24]) at a **declared
request cap**; (c) **horizon** = a declared number of decisions per window; (d) **pricing** = real diurnal
electricity prices applied to every arm's energy; (e) **metric** = episode SLA-safe goodput / operator-$
(`goodput_per_dollar`), with the SLA-violation rate reported alongside; (f) **gate** = Pareto vs
`production_scheduler` (primary) and `sla_aware` (secondary), SLA-not-worse required; (g) **reporting** =
absolute **and** percent deltas, every figure labeled SIMULATOR_INFERENCE; (h) **diagnostics** (oracle,
single-decision tournament, regret) are reported as *diagnostics*, never as the headline.

**15. What changes should require declaring Benchmark v2?** Any change to a **headline-moving** axis — proven
here to swing the number up to ~8× with **no optimizer change**: the **evaluation harness** (forecast vs
real-trace; single vs persistent state), the **request cap**, the **episode horizon / aggregation**, the
**electricity-pricing basis**, the **window/trace set**, the **baseline policies** (`production_scheduler` /
`sla_aware` definitions), or the **reward / cost model / action semantics**. Each must be version-pinned so
numbers stay comparable across PRs. (Changing only the *planner* or adding an *action knob* does **not** require
v2 — those are exactly what the frozen benchmark is meant to measure.)

## What PR #123 is still good for

It is a valid, useful **search-method tournament**: at equal evaluation budget on a single hard decision, it
ranks clock-only < grids < beam/CEM/annealing < **hierarchical** (regret 0). That ranking — *which search
architecture to use* — is its scientific contribution and does not depend on the harness magnitude. Keep it as
the planner-selection diagnostic; do not quote its percent as a deployment result.

## Bottom line

Same optimizer, same bundle, same baseline policy. **~93% of "+1273% → +164%" is the evaluation harness**
(forecast-synthetic single rollout → real-trace persistent episode); horizon, request cap, and electricity are
minor. **PR #124 is the more realistic benchmark and should be frozen as Aurelius Benchmark v1**; the +1273% is
a correct-but-harness-inflated single-decision method-ranking figure and should not be a public headline. The
objective was never a larger number — it was making future benchmark numbers **directly comparable**, which the
frozen v1 protocol now enables.
