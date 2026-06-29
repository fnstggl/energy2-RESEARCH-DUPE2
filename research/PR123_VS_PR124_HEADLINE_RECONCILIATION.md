# PR #123 (+1273%) vs PR #124 (+164%) — Headline Reconciliation

PR #123 reported `hierarchical_search` at **+1273% vs sla_aware**; PR #124 reported Aurelius hierarchical MPC at
**+165% vs sla_aware / +148% vs production_scheduler**. This diagnostic explains the gap **exactly**, on the
SAME pjm·expensive window (period 81), same hierarchical method, same sla_aware *policy*. **Bottom line:
`hierarchical_search` did NOT regress** — the two numbers come from two **different measurement harnesses**, and
the gap decomposes cleanly into a numerator effect × a denominator effect. No simulator / reward / gate / cost /
action / baseline change; magnitudes are SIMULATED. Evidence:
`scripts/diagnose_headline_reconciliation.py` → `data/external/mpc_controller/headline_reconciliation.json`.

## The one-paragraph answer

Both PRs measured the **same** pjm·expensive window (start period 81), the **same** `hierarchical_search`
method (which even chose the **same winning bundle**), and the **same** sla_aware *policy* (`backlog_aware` +
`abs_conformal` + admission off). They differ only in **how gp/$ is measured**: PR #123 scored **one** planning
decision through a single forecast-rollout (`_rollout_world`, horizon_steps=1, req_cap 80); PR #124 scored a
**3-period episode** over the **real trace** requests through the persistent world simulator (`run_period_episode`,
req_cap 56). The same sla_aware policy reads **2.79× higher** in PR #124's episode (295,338) than PR #123's
single rollout (105,924), and hierarchical reads **1.86× higher** in PR #123 (1,454,636) than PR #124 (783,862).
Those two harness effects multiply to the full gap. **It is a denominator + numerator measurement change, not a
method regression.**

## Side-by-side comparison table (same window, period 81)

| field | Setup A — PR #123 tournament | Setup B — PR #124 ladder |
|--|--|--|
| window | pjm · expensive | pjm · expensive |
| harness | single decision, forecast rollout (`_rollout_world`, H=1) | multi-period episode (`run_period_episode`, persistent world) |
| dt_seconds | 3600 | 3600 |
| request cap | **80** | **56** |
| periods / decisions | **1** (period 81) | **3** (periods 81–83) |
| gp/$ definition | one rollout period's `gp_per_dollar` (forecast jobs) | episode goodput/$ over real trace requests |
| **sla_aware gp/$** (baseline) | **105,924** (SLA 0.374) | **295,338** (SLA 0.226) |
| **production_scheduler gp/$** | **150,031** (SLA 0.314) | **330,711** (SLA 0.065) |
| **hierarchical_search gp/$** | **1,454,636** (SLA 0.288) | **783,862** (SLA 0.000) |
| hierarchical abs Δ vs sla_aware | +1,348,712 | +488,524 |
| hierarchical **% vs sla_aware** | **+1273.3%** | **+165.4%** |
| hierarchical % vs production_scheduler | **+869.6%** | **+148.2%** |
| production_scheduler % vs sla_aware | +41.6% | +12.0% |
| Pareto gate (hier vs sla_aware) | **PASS** (SLA 0.288 < 0.374) | **PASS** (SLA 0.000 < 0.226) |
| Pareto gate (hier vs production_scheduler) | **PASS** (SLA 0.288 < 0.314) | **PASS** (SLA 0.000 < 0.065) |
| hierarchical selected bundle | `forecasted_mcs` cap · `class_aware` adm · `kv_aware` rt · `network_aware` pl · cap×0.75 · `aggressive` batch · `fp8` · `clock=high` · `spec=aggressive` | **same surfaces** (`kv_aware`·`network_aware`·`fp8`·`clock=high`·`aggressive`·×0.75·`spec=aggressive`) |
| candidates evaluated | 80 | 75 |
| runtime | 37.2 s (1 decision) | 161.6 s (3 decisions) |
| search regret | **0.0%** (exact-ref) | 0.0% (Phase F synthetic; ladder is not exhaustive-able) |

**Note on the multiplier.** As gp/$ ratios: A = 1,454,636/105,924 = **13.73×**, B = 783,862/295,338 = **2.65×**.
13.73 / 2.65 = **5.17** = numerator_ratio (1.856) × denominator_ratio (2.788). (As *percentages*, +1273% vs
+165%; percent subtracts the 1.0 baseline, so the percent ratio looks larger — the clean multiplicative
decomposition is on the gp/$ ratios.)

## The 9 explicit questions

**1. Did `hierarchical_search` actually regress?** **No.** Same method, same window, same chosen bundle. Its
*absolute* gp/$ is in fact **higher** in PR #123 (1,454,636) than PR #124 (783,862). Search regret is 0% in
both. Nothing about the method got worse.

**2. Or did the denominator / window / baseline change?** **The harness changed** (not the window or the
baseline policy). The sla_aware **denominator** is 2.79× higher in PR #124's multi-period real-trace episode
than PR #123's single forecast rollout; the hierarchical **numerator** is 1.86× higher in PR #123. The window
(pjm·expensive, period 81) and the sla_aware *policy* are identical.

**3. What was the sla_aware baseline in PR #123?** `SAFE_BASELINE_BUNDLE = ActionBundle(capacity_policy=
"backlog_aware", ordering_policy="abs_conformal", admission_policy="off")` (`physics_guided_candidates.py:62`),
scored once via the controller's single-decision forecast rollout (`market_window_scorer` → `_rollout_world`).

**4. What was the sla_aware baseline in PR #124?** `SLA_AWARE_FALLBACK = {capacity: backlog_aware, ordering:
abs_conformal, admission: off}` (`controller.py`), scored via `run_period_episode` over 3 real periods. **Same
policy by value**; different harness.

**5. What exact factors explain +1273% vs +164%?** Four, all measurement-side (none is a method change):
(a) **harness** — a single forecast-rollout decision vs a 3-period real-trace episode (the dominant factor:
drives the 2.79× denominator change); (b) **request cap** 80 vs 56; (c) **decisions** 1 vs 3; (d) **gp/$
definition** — one rollout step's gp/$ vs episode-aggregated gp/$. The single heavy-prompt rollout (median
prompt 1181, req_cap 80) is compute-bound, where the optimized bundle's relative gain over the bf16/no-batching
baseline is largest; the multi-period real episode compresses both ends.

**6. Can the +1273% setup be compared against `production_scheduler`?** **Yes — done.** Scoring
`production_scheduler`'s causal action in the *exact* PR #123 single-decision harness gives **150,031 gp/$
(+41.6% vs sla_aware)**, and **hierarchical is +869.6% vs production_scheduler** there (vs +148.2% in the PR #124
episode). So in *both* harnesses hierarchical Pareto-beats production_scheduler; only the magnitude swings.

**7. Is the +1273% result Pareto-safe?** **Yes, within that harness** — hierarchical SLA 0.288 < sla_aware
0.374 (SLA *better*), `headline_safe=True`. But the **magnitude is inflated by the single-rollout harness**; the
Pareto *direction* is safe, the *number* is not a fair public figure (see Q9).

**8. Is +1273% driven by connected surfaces, simulator artifacts, or both?** **Both, but they are different
things.** The gp/$ **gain itself** is driven by connected surfaces **+** economic arbitrage (the winning bundle
uses `forecasted_mcs`/`network_aware`/`class_aware`/`kv_aware` **and** `fp8`/`clock=high`/`spec=aggressive`). The
**+1273%-vs-+164% difference** is a **measurement-harness artifact** (single forecast rollout @req_cap80 vs
multi-period real episode @req_cap56) — *not* extra connected-surface value. So: connected + economic levers
drive the win; the harness drives the headline size; both magnitudes are SIMULATOR_INFERENCE
(`CONNECTED_SURFACE_VALUE_ATTRIBUTION.md`, `WORLD_MODEL_ROBUSTNESS_AUDIT.md`).

**9. Which headline is fair to use publicly?** **The PR #124 episode figures: ~+165% vs sla_aware and ~+148%
vs production_scheduler.** They use the more realistic harness — real trace requests, persistent state
evolution (cold starts / warm transitions), multiple decisions, and a stricter denominator. The **+1273% is a
single-decision diagnostic** that is harness-inflated and should **not** be a public headline. Lead with
**vs production_scheduler** (the realistic production bar) and report **vs sla_aware** alongside it, both from
the multi-period episode, labeled SIMULATOR_INFERENCE.

## What changed vs what did not

| | PR #123 | PR #124 | same? |
|--|--|--|--|
| window (market, start period) | pjm·expensive, 81 | pjm·expensive, 81 | ✅ same |
| hierarchical_search method + budget | hierarchical, 100/≈80 evals | hierarchical, 100/≈75 evals | ✅ same |
| hierarchical chosen bundle | fp8/clock-high/kv/network/forecasted_mcs… | same surfaces | ✅ same |
| sla_aware *policy* | backlog_aware+abs_conformal+adm off | identical by value | ✅ same |
| dt_seconds | 3600 | 3600 | ✅ same |
| **evaluation harness** | single forecast rollout (H=1) | multi-period real-trace episode | ❌ **different** |
| **request cap** | 80 | 56 | ❌ **different** |
| **# decisions** | 1 | 3 | ❌ **different** |
| **gp/$ definition** | one rollout step | episode-aggregated | ❌ **different** |

The four "different" rows fully account for +1273% → +165%. The method is unchanged and did not regress.

## Recommendation

- **Public headline:** the PR #124 episode numbers — **Aurelius MPC vs `production_scheduler` (~+148%)** as the
  primary bar, **vs `sla_aware` (~+165%)** as the secondary bar — both labeled SIMULATED / SIMULATOR_INFERENCE
  and bounded to the windows tested.
- **Retire +1273% as a headline.** Keep it only as a *single-decision tournament diagnostic* (it correctly
  ranked the search methods at equal budgets — its purpose — but its absolute percent is harness-inflated).
- **The reconciliation strengthens, not weakens, the result:** hierarchical Pareto-beats sla_aware AND
  production_scheduler in **both** harnesses; only the magnitude is harness-dependent, which is exactly why the
  conservative episode harness is the honest one to publish. (For baseline naming/wording, see
  `PRODUCTION_SCHEDULER_PUBLIC_BENCHMARK_RESEARCH.md`.)
