# Benchmark v1 request-cap sensitivity & freeze decision (Batch-1 Phase 0)

**What this is.** Before adding new action knobs we measured how the Benchmark v1 headline (Aurelius vs the
production baselines) moves with the per-period request cap, so the freeze is grounded in data, not a guess.
The sweep runs the **same** ladder harness (`scripts/run_ladder_benchmark`) at caps **56 / 80 / 120 / 200 /
uncapped** on one market+window, varying ONLY `req_cap`. Runner: `scripts/run_request_cap_sensitivity.py`;
artifact: `research/results/request_cap_sensitivity.json`. Scope (honest): market `pjm`, the price-`expensive`
window, 3 decisions, all 8-arm-relevant arms (sla_aware, production_scheduler, the two Aurelius MPC arms,
oracle). The "uncapped" rung uses cap=100000 — far above the real per-period Azure volume, so no request is
dropped (the served-count column proves it).

## Results

| cap | requests served (window) | cells completed / timed-out | wall (s) | baseline gp/$ (production_scheduler) | Aurelius gp/$ (hierarchical_search) | abs Δ | % Δ | SLA viol (base → Aurelius) | Pareto |
|----:|----:|:--:|----:|----:|----:|----:|----:|:--:|:--:|
| 56 | 168 | 5 / 0 | 2.7 | 190,417.22 | 490,538.14 | +300,120.9 | **+157.6%** | 0.0 → 0.0 | ✅ |
| 80 | 180 | 5 / 0 | 2.8 | 180,936.94 | 551,149.66 | +370,212.7 | **+204.6%** | 0.00556 → **0.0** | ✅ |
| 120 | 180 | 5 / 0 | 2.6 | 180,936.94 | 551,149.66 | +370,212.7 | **+204.6%** | 0.00556 → **0.0** | ✅ |
| 200 | 180 | 5 / 0 | 2.6 | 180,936.94 | 551,149.66 | +370,212.7 | **+204.6%** | 0.00556 → **0.0** | ✅ |
| uncapped (100000) | 180 | 5 / 0 | 2.7 | 180,936.94 | 551,149.66 | +370,212.7 | **+204.6%** | 0.00556 → **0.0** | ✅ |

The key structural fact: **caps ≥ 80 are identical.** The real per-period Azure volume in this window
saturates at ≤ 60 requests/period (3 periods → 180 served), so cap=80, 120, 200 and uncapped all serve the
**full** real workload — they are *uncapped-equivalent*. cap=56 is the only rung that **truncates** real
requests (168 vs 180, ~7% dropped).

## Answers to the six questions

1. **What request cap is most production-like?** **Uncapped** (equivalently any cap ≥ 80), because it serves
   the complete real per-period Azure volume. cap=56 is *less* production-like: it discards ~7% of the real
   requests that actually arrived.

2. **Highest cap that all arms complete without timeout?** **Uncapped.** Every arm completed at every cap with
   zero timeouts; the slowest cell was ~1.2 s. There is no tractability reason to cap at all in this window —
   the workload is small and the planner is bounded (~80 candidate evaluations/decision).

3. **Does Aurelius's advantage shrink or grow with cap?** It **grows**: +157.6 % at cap=56 → **+204.6 %** at
   cap ≥ 80, then flat. Two reasons: (a) cap=56 truncates real load, understating the regime where Aurelius
   helps; (b) at the full volume the baselines begin **violating SLA** (rate 0.00556) while Aurelius holds
   SLA at **0.0** — so the higher cap reveals an SLA-pressure regime that *widens* Aurelius's gp/$ lead and
   makes it strictly Pareto-dominant (more goodput AND fewer violations).

4. **Is req_cap=56 too low?** **Yes.** It truncates ~7 % of real requests, understates the advantage by ~47
   percentage points (157.6 % vs 204.6 %), and hides the SLA-pressure regime that most distinguishes Aurelius.
   It is an artificially easy, sub-real workload.

5. **Freeze at 56 / 80 / 120 / uncapped?** **Freeze at cap = 100,000 (uncapped).** > **CORRECTION (Batch-1
   corrective PR):** an earlier draft of this doc recommended cap=120; that recommendation is **obsolete and
   withdrawn.** The cap must not be chosen to size the headline. The correct V1 decision is the **highest
   stable cap under the V1 timeout**: **100,000 requests/period (effectively uncapped)** — the
   `aurelius_mpc_hierarchical_search` and `production_scheduler` arms complete uncapped; the operative reason
   "uncapped" is bounded at 100,000 rather than ∞ is that `sla_aware` can time out under the full V1 harness
   (longer windows / more decisions than this fast smoke). Caps ≥ 80 serve the full real per-period volume, so
   100,000 = uncapped in volume terms here and never truncates a higher-volume window/market.

6. **What cap should future public claims use?** **cap = 100,000 (uncapped).** A public gp/$ claim is stated
   on the full real per-period volume. The number stands: **+204.6 % gp/$ vs production_scheduler at
   equal-or-better SLA (0.0 vs 0.00556)** on the uncapped-equivalent workload. **Do not** quote a cap=120
   number as the V1 headline.

## Honest caveats

- This is **one market × one window × 3 decisions**. The *saturation* conclusion (real volume ≤ cap=80 here)
  is window-specific; a higher-arrival window/market could bind a higher cap, which is exactly why the freeze
  is set at the **highest stable cap = 100,000 (uncapped)** rather than any tight per-window number — uncapped
  never truncates. The *direction* (advantage grows with cap; 56 truncates) is robust because it follows from
  the served-count and SLA-rate mechanics, not from tuning. (An earlier draft proposed cap=120 as a tractable
  proxy; that is withdrawn — the cap must not be chosen to size the headline.)
- Uncapped was **feasible** here (cheap), so we did not have to fall back to a lower completed cap. If a
  future heavier window makes uncapped expensive, report that and choose the highest completed cap — the
  runner records `cells_timed_out` per cap for exactly this.
- No reward, baseline, Pareto-gate, or planner change was made for this phase; only `req_cap` was swept.
