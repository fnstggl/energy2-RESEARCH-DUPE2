# Benchmark V1 Request-Cap Sweep — can V1 be uncapped?

A diagnostic to determine whether **Aurelius Benchmark V1** (the PR #124 episode harness) can be run with the
request cap **removed**, and if not, the **highest stable request cap** all required arms complete within the V1
per-cell timeout. "Request cap" = the max replayed requests **per benchmark period** (not a runtime / candidate
/ GPU / capacity cap). Diagnostic ONLY: no simulator / reward / cost / gate / baseline / planner / action
change; no tuning. **The cap is chosen on stability/reproducibility, NOT on the Aurelius headline.** Evidence:
`scripts/run_request_cap_sweep.py` → `data/external/mpc_controller/request_cap_sweep.json`. Harness = PR #124
V1 (`run_period_episode`, persistent world, real diurnal prices, 3 decisions, pjm/ercot/caiso expensive
windows). Per-cell timeout = **300 s** (the PR #124 V1 cell timeout). Magnitudes SIMULATED.

## Headline finding

**Benchmark V1 cannot be run uncapped.** At the full trace (~440k–580k requests across the 3 periods) the
**`sla_aware` baseline times out** (> 300 s) on every market — **but the Aurelius planner and
`production_scheduler` both complete uncapped.** The binding arm is the *naive baseline*, not the optimizer:
`sla_aware` (conservative batching, no admission, SRPT over every request) is the most expensive policy to
*simulate* at scale, and its replay cost is **super-linear** in request count. The highest cap at which all
three arms complete is **≈ 100,000 requests/period** (gated by `sla_aware`'s simulation runtime, ~160–205 s at
300k requests; it times out by ~440k).

**The choice was NOT made for the headline.** The headline is in fact *highly* cap-sensitive and **higher caps
make Aurelius look better** (pjm: +137% vs production_scheduler at cap 56 → **+654%** at cap 100k) — but that is
**baseline degradation, not Aurelius improvement** (the inefficient baselines serve the heavy load worse, while
Aurelius's gp/$ saturates). So a conservative cap *understates* Aurelius; the cap is chosen purely for stable,
reproducible completion.

## Per-market results (V1 harness, 3 decisions, real prices)

Uncapped request counts and per-arm completion (✓ = completed ≤ 300 s; ✗ = TIMEOUT):

| market | uncapped req (3 periods) | `sla_aware` | `production_scheduler` | `aurelius_hierarchical` | highest stable cap |
|--|--|--|--|--|--|
| pjm·expensive | 576,912 | ✗ TIMEOUT (>300s) | ✓ 38.7s | ✓ 186.2s | **100,000** |
| ercot·expensive | 442,716 | ✗ TIMEOUT (>300s) | ✓ 29.9s | ✓ 116.8s | **100,000** |
| caiso·expensive | 529,621 | ✗ TIMEOUT (>300s) | ✓ 35.4s | ✓ 153.8s | **100,000** |

All three markets: **highest stable request cap = 100,000** → recommended benchmark cap = **100,000** (the min
across markets). 24 cells completed, 9 timeout (the uncapped/200k/150k `sla_aware` cells), 0 failed.

The planner is **never** the binding arm: its runtime is ~90–190 s and is **cap-independent** (search-dominated
— ~75 evaluations × 3 decisions over *forecast-synthetic* jobs; the final real-trace replay is cheap). Only the
baselines' replay cost grows with the cap, and `sla_aware`'s grows fastest.

### gp/$ and SLA at the tested caps (the cap-sensitivity)

| market·cap | actual req | binding? | `sla_aware` gp/$ (SLA) | `production` gp/$ (SLA) | `aurelius` gp/$ (SLA) | A vs prod (abs / %) | A vs sla (abs / %) | Pareto |
|--|--|--|--|--|--|--|--|--|
| pjm·56 | 168 | yes | 295,338 (0.226) | 330,711 (0.065) | 783,862 (0.000) | +453,151 / **+137%** | +488,524 / +165% | ✓✓ |
| pjm·100000 | 300,000 | yes | 110,271 (0.124) | 137,703 (0.040) | 1,038,887 (0.004) | +901,184 / **+654%** | +928,616 / +842% | ✓✓ |
| pjm·uncapped | 576,912 | no | — (TIMEOUT) | 130,539 (0.038) | 1,041,842 (0.002) | +911,303 / +698% | — | ✓ (vs prod) |
| ercot·56 | 168 | yes | 293,601 (0.143) | 303,982 (0.071) | 785,952 (0.000) | +481,969 / +159% | +492,351 / +168% | ✓✓ |
| ercot·100000 | 300,000 | yes | 104,497 (0.129) | 136,507 (0.044) | 1,081,275 (0.002) | +944,768 / **+692%** | +976,778 / +935% | ✓✓ |
| ercot·uncapped | 442,716 | no | — (TIMEOUT) | 130,178 (0.045) | 1,064,602 (0.001) | +934,424 / +718% | — | ✓ (vs prod) |
| caiso·56 | 168 | yes | 288,383 (0.149) | 301,228 (0.065) | 747,580 (0.000) | +446,352 / **+148%** | +459,197 / +159% | ✓✓ |
| caiso·100000 | 300,000 | yes | 107,999 (0.134) | 137,788 (0.047) | 1,113,472 (0.003) | +975,684 / **+708%** | +1,005,473 / +931% | ✓✓ |
| caiso·uncapped | 529,621 | no | — (TIMEOUT) | 130,000 (0.046) | 1,111,915 (0.002) | +981,915 / +755% | — | ✓ (vs prod) |

**Note the baselines *fall* as the cap rises** (pjm `sla_aware` 295k→110k; `production` 331k→138k) while
Aurelius *rises and saturates* (784k→1.04M). That asymmetry — not any change in Aurelius — is why the percent
explodes. Pareto holds at every cap (Aurelius gp/$ higher AND SLA lower than both baselines).

## The 9 questions

**1. Can Benchmark V1 be uncapped?** **No.** At the full trace the `sla_aware` baseline times out (> 300 s) on
every market. (The planner and `production_scheduler` *do* complete uncapped — the bottleneck is the naive
baseline's simulation cost, which is super-linear in request count.)

**2. If not, what is the highest cap that completes all required arms?** **100,000 requests/period** — pjm,
ercot, AND caiso all top out at exactly 100,000 (the min across markets is therefore 100,000), where
`sla_aware` runs ~160–205 s (300k served requests); at 150,000+ (≥440k served) it times out on every market.

**3. How sensitive is Aurelius gp/$ to the request cap?** **Mildly — it saturates.** pjm: 783,862 (cap 56) →
1,038,887 (cap 100k) → 1,041,842 (uncapped). Aurelius gains ~33% from 56→100k then plateaus; it is *robust* to
the cap above ~100k.

**4. How sensitive are `production_scheduler` and `sla_aware` to the request cap?** **Very — they degrade.**
pjm `production_scheduler`: 330,711 (56) → 137,703 (100k) → 130,539 (uncapped). `sla_aware`: 295,338 (56) →
110,271 (100k) → TIMEOUT. The naive/realistic baselines serve the heavier load *worse* (more SLA violations,
lower gp/$), because their fixed policies don't adapt to the load the way the optimizer does.

**5. Does the headline percent change materially with cap?** **Yes, dramatically.** pjm Aurelius vs
production_scheduler: **+137% (cap 56) → +654% (cap 100k)**; vs sla_aware: +165% → +842%. **This is driven by
baseline degradation, not Aurelius improvement** — so the percent must never be quoted without its cap, and the
cap must be frozen (confirming the prior reconciliation's Benchmark-versioning rule).

**6. Does the Pareto gate still pass?** **Yes, at every stable cap and market.** Aurelius has strictly higher
gp/$ AND strictly lower SLA-violation rate than both baselines at cap 56, cap 100k, and uncapped (where
measurable). E.g. pjm cap 100k: Aurelius SLA 0.004 < production 0.040 < sla_aware 0.124.

**7. What cap should Benchmark V1 freeze?** The **highest stable request cap is 100,000** (all three markets,
the literal answer). But "stable, *reproducible*" needs **margin**: at 100k, `sla_aware` runs ~160–205 s and
the planner ~155–189 s — **~52–68 % of the 300 s timeout**, i.e. the *edge* of stable (machine-load-sensitive,
not robustly reproducible). So the recommendation is split: (a) **keep the current V1 cap = 56** as the frozen,
comparable benchmark — it has huge margin, reproduces exactly, is what PR #124's published numbers already use,
and (crucially) does **not** flatter Aurelius (raising the cap *increases* the reported advantage, so keeping
56 is the un-headline-driven choice the instruction asks for); (b) **if a larger, more-realistic workload is
wanted, declare Benchmark V2** at the highest cap with *comfortable* margin — **~50,000–70,000/period** (every
arm ≤ ~50 % timeout from the runtime curve), **not** the 100k edge — and re-report with the cap stated. Either
way the cap is part of the benchmark definition and must be frozen; **100,000 is the documented stability
ceiling** under the V1 300 s timeout, and **uncapped is not an option** until the simulator's `sla_aware`
replay is made sub-linear (a separate, non-V1 change). The choice is made on stability/reproducibility, never
on the headline.

**8. What result should we report for that frozen cap?** At the frozen V1 cap (56): **Aurelius
`hierarchical_search` Pareto-dominates `production_scheduler` by +137%/+159%/+148% gp/$ (pjm/ercot/caiso) and
`sla_aware` by +159–168%, SLA strictly better, on the V1 episode harness — SIMULATED, bounded to these
windows.** Always state the cap and that the percent is cap-dependent.

**9. What would require declaring Benchmark V2?** Changing the **request cap** (proven here to move the headline
~5×), or any other frozen V1 axis (evaluation harness, episode horizon/aggregation, electricity-pricing basis,
window/trace set, baseline policies, reward/cost model, action semantics) — per `PR123_PR124_RECONCILIATION_FINAL.md`.
Additionally, **a simulator change that makes `sla_aware`'s replay sub-linear** (enabling a much higher or
uncapped V2) would itself require declaring V2, since it changes the achievable cap and thus the numbers.

## Why the binding arm is `sla_aware`, not the planner (the key insight)

| arm | what makes it slow at scale | runtime vs cap |
|--|--|--|
| `sla_aware` | conservative batching (1 req/replica) + **no admission** + SRPT over **every** request → super-linear replay | 0.1 s (56) → 160–205 s (300k) → TIMEOUT (≥440k) |
| `production_scheduler` | class admission sheds best-effort + continuous batching → far fewer serial steps | ~24 s at 300k; ~30–39 s uncapped |
| `aurelius_hierarchical` | ~75-evaluation search over **forecast-synthetic** jobs (cap-independent) + one cheap real replay | ~90–190 s at **every** cap |

So the cap exists to keep the **naive baseline** tractable to simulate — the optimizer was never the
bottleneck. This is a property of the simulator's replay cost, not of Aurelius.

## Honesty notes

- The cap was chosen for **stability/reproducibility**, explicitly **not** the headline; the higher (stable)
  cap would *increase* the reported advantage, so the conservative recommendation is the un-flattering one.
- Magnitudes are **SIMULATED / SIMULATOR_INFERENCE**; the cap-sensitivity is a property of the
  baseline-vs-optimizer simulation under heavier synthetic load, not validated production telemetry.
- The timeout boundary (~100k stable, ~150k not) is the V1 300 s cell budget on this machine; a faster machine
  or a vectorized replay would raise it. Reported as the boundary under the **frozen V1 timeout**, not a
  hardware-independent constant.
