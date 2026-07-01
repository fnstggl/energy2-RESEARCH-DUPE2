# Headline audit — is +869.56% / ~+870% vs production_scheduler public-headline-safe?

**Audit only.** No change to simulator physics / reward / baselines / planner / action semantics / frontend
copy; nothing tuned; no invented numbers (every figure below is recomputed from committed raw values). Question:
can the **+869.56%** goodput-per-dollar (gp/$) advantage over `production_scheduler` be a public headline, or must
it stay a narrower diagnostic?

## Verdict (up front)

**Do NOT use +870% publicly as a performance claim.** It is a **single-market, single-decision,
forecast-synthetic planner-search diagnostic** whose *own run* pairs Aurelius with a **28.8% SLA-violation rate** —
it demonstrates that the planner sits at the top of its search space, **not** a deliverable production speedup. It
is **narrower and less realistic** than the real-trace episode results. The defensible public number is the
frozen-benchmark **real-trace episode** result: **~2.4–2.6× gp/$ (≈ +138–159%), Pareto-dominant, Aurelius SLA
0.000**, in simulation.

---

## 1. Where +869.56% comes from

| axis | value |
|--|--|
| harness | **single-decision planning tournament** (`market_window_scorer` → controller's `_rollout_world`) — the rollout the MPC uses to *plan*, not the episode-replay *measurement* harness |
| market / window / period | **pjm**, "expensive" window, **period 81** (first period of the window, `decision_index=0`) — one market, one window, one decision |
| request cap | **80** requests |
| decision structure | **single-decision** (`horizon_steps = 1`, `H = 1`) — **not** a multi-period episode |
| trace realism | **forecast-synthetic rollout**: bundles scored against the *forecaster's predicted trajectory* (`build_trajectory`), with **real electricity prices** and real trace-derived forecast inputs, but **no actual real-trace request replay** |
| origin | surfaced by the **action-space peak-search** diagnostic, whose purpose was "is hierarchical at the peak of its own search space" |

So +870% is the **least-realistic of the three harnesses** in play: real prices, but forecast-synthetic jobs, a
single planning step, on one market. In the reconciliation this is "Harness A," identifiable because `sla_aware`
reads ≈105,924 here (vs ≈295,338 in the real-trace episode harness).

## 2. Recompute from raw values (verified)

| arm | gp/$ | SLA-violation rate |
|--|--|--|
| `production_scheduler` | 150,030.7 | **0.3135** |
| Aurelius / hierarchical | 1,454,635.9 | **0.2883** |
| absolute delta | **+1,304,605.2** | — |
| percent delta | **+869.56%** | — |

Arithmetic checks out exactly. **But note the SLA row:** in this harness *both* arms violate SLA ~29–31%
(`sla_aware` is worse at 0.374). Aurelius is only marginally better on SLA here (0.288 vs 0.314) — it is **not**
the clean Pareto win (SLA 0.000) shown by the real-trace episode benchmark. A headline of "8.7× the throughput"
that internally admits "~29% of requests miss SLA" is self-undermining.

## 3. Compare to the PR-#126 uncapped high-load replay (+724% avg)

Uncapped real-trace episode, 3 markets, recomputed: pjm **+698.11%**, ercot **+717.80%**, caiso **+755.32%** →
**average +723.74% ("+724%")**.

| axis | +870% (tournament) | +724% (uncapped episode) | same? |
|--|--|--|--|
| metric | gp/$ | gp/$ | ✅ yes |
| baseline | production_scheduler | production_scheduler (sla_aware **times out** uncapped) | ⚠️ same baseline, but sla_aware not comparable uncapped |
| trace scope | 80 requests, 1 period | ~440k–580k requests, 3 periods | ❌ no |
| markets | pjm only | pjm + ercot + caiso | ❌ no |
| request-cap policy | cap 80 | uncapped | ❌ no |
| episode structure | single decision | 3-decision persistent episode | ❌ no |
| realism | forecast-synthetic single rollout | **real-trace multi-period replay** | ❌ no |

They agree on the *metric* and the *baseline name* and nothing else. **Different measurement in almost every
respect.**

## 4. Why +870% ≠ +724% (they are not the same quantity scaled)

The two are **close in magnitude by coincidence, via different mechanisms** — +870% is not a "stronger" +724%:

- **Harness (dominant).** Both the numerator and denominator of +870% are forecast-synthetic single-step
  numbers. Aurelius reads **1,454,635** here vs ~**1.04M** uncapped-episode; production reads **150,030** here vs
  **130,539** uncapped-episode vs **330,711** at frozen cap 56. The ratio is built from harness-specific
  magnitudes, so it is a **different quantity**, not a re-scaling.
- **Single market vs 3-market average.** +870% is **pjm only**; +724% averages pjm/ercot/caiso. No ercot/caiso
  tournament-harness comparison exists at cap 80, so +870% has **no cross-market support**.
- **Cap 80 vs uncapped is secondary here.** Within the *episode* harness the cap moves production a lot (330,711
  at 56 → 130,539 uncapped); but the tournament's 150,030 is a *forecast-synthetic-scoring* artifact, not the
  same axis.
- **Baseline degradation** explains most of the *uncapped* +724% (production degrades under sustained overload;
  sla_aware even times out) — it does **not** apply to +870% (only 80 requests; no overload).
- **Peak-search setting.** +870% is literally the peak-search's reported ceiling for the planner's own search
  space — a **search-quality** statistic.

Net: the +870% is inflated relative to real-replay results **from both ends** (higher synthetic Aurelius ceiling,
lower synthetic production denominator), on a single market, at a single decision.

## 5. Headline-safety under strict public-claim standards

A public performance headline should be (a) from the **most realistic** harness available, (b) **representative**
(not one market/window/decision), (c) **reproducible**, and (d) paired with an **acceptable SLA**. +870%:

- (a) ❌ forecast-synthetic single-step — the **least** realistic of the three harnesses.
- (b) ❌ one market, one window, one decision; no cross-market support.
- (c) ✅ reproducible (byte-identical on re-run) — but reproducibility ≠ representativeness.
- (d) ❌ its own run has **28.8% SLA violation**; it is not the clean Pareto result.

It is **strictly narrower and less realistic** than the +724% uncapped episode result — which is itself
**heavy-load-inflated** (baseline degradation) and needs its own caveats. The number that satisfies (a)–(d) is the
**frozen-benchmark real-trace episode**: **+147.92% average** (pjm +137.02%, ercot +158.55%, caiso +148.18%;
ratios **2.37–2.59×**), **Aurelius SLA 0.000 vs 0.065–0.071**, multi-period, real replay.

**Decision: use +870% only as an internal action-space / planner-search diagnostic (search-optimality), not as a
public performance figure — not even as an "up to."** If a high-water public number is ever wanted, the
**uncapped real-replay** figure (up to ~7–7.5× per market) is more defensible than +870% *because it is real
replay*, but it must be captioned as sustained-overload behavior driven partly by baseline degradation.

## 6. Recommended public wording (no internal names; simulated; conservative)

**Primary homepage headline (recommended):**

> **In simulation on a real week of production LLM-inference traffic, Aurelius delivered more than 2× the useful
> throughput per dollar of a strong production-grade GPU scheduler — while holding a stricter latency SLA.**

(Backed by the frozen-benchmark episode: 2.37–2.59× ⇒ "more than 2×", rounding down; Aurelius SLA 0.000 <
baseline.)

**Optional supporting technical-report sentence:**

> Across three electricity-price scenarios on a one-week production LLM-inference trace, a receding-horizon
> controller improved goodput-per-dollar by **+138–159%** over a continuous-batching production-scheduler baseline
> at the benchmark request cap, at a **strictly lower** SLA-violation rate (0.0 vs 0.065–0.071); under uncapped
> sustained overload the simulated margin widens to several-fold, partly reflecting baseline degradation.

**Required caveats:**
- **"In simulation"** — SIMULATOR_INFERENCE magnitudes, not production-validated telemetry.
- **Bounded** to the tested price scenarios and the one-week trace.
- Larger multiples (uncapped ~7×, or the +870% diagnostic) are **harness/load-specific** and partly reflect
  **baseline degradation** — not headline material.
- gp/$ = goodput (SLA-met throughput) per unit cost.

**Single safest one-liner (most conservative):**

> **In simulation, Aurelius more than doubled goodput per dollar versus a strong production GPU-scheduler
> baseline — at a stricter latency SLA.**

---

### Recommendation (explicit)

**→ Do not use +870% publicly.** Keep it as an internal planner-search / action-space diagnostic. Make the public
headline the frozen-benchmark **real-trace episode** result — conservatively **"more than 2× goodput per dollar,
in simulation, at a stricter SLA."**
