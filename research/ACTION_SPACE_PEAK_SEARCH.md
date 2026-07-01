# Genuine-peak search — is there a much-better bundle we never tried?

The question: hierarchical_search evaluates only ~75 bundles per decision, yet the deployable, headline-safe
action space has **209,952** bundles (12 connected knobs, int4 excluded). "Surely somewhere in that space there
is a search strategy/option much better than hierarchical's pick — if we searched all of it, at what magnitude
would the genuine peak beat production_scheduler?" This diagnostic searches **far harder than 75 evals** to find
out. Diagnostic ONLY: no simulator / reward / cost / gate / baseline / planner / action change; no tuning; no
new headline. Evidence: `scripts/diagnose_action_space_peak.py` → `data/external/mpc_controller/action_space_peak.json`.

Harness = the **same single-decision tournament rollout** PR #123 used to pick the winning planner
(`market_window_scorer`, pjm·expensive, req_cap 80, period 81). Magnitudes are SIMULATED and harness-dependent
(this tournament harness inflates the percent vs the realistic PR #124 episode harness — see caveats).

## Headline finding

**Nothing beat hierarchical_search's own ~75-eval pick, despite ~64× more evaluations.** The genuine deployable
peak *is* hierarchical's pick (gp/$ = **1,454,635.9**), and the reward surface near the top is a **broad
plateau**, not a needle hiding a much-better bundle.

| search | evals | best gp/$ | SLA | vs hierarchical | Pareto-better than hierarchical? |
|--|--|--|--|--|--|
| **hierarchical@100** (the ~75-eval pick) | 80 | **1,454,635.9** | 0.2883 | — | — |
| 5,000 uniform-random deployable bundles | 5,000 | 1,448,727.1 | 0.3016 | **−0.41%** | **No** (lower gp/$ AND worse SLA) |
| cross_entropy @budget 2000 | 29¹ | 1,114,798.5 | 0.2778 | −23.3% | No (23% lower gp/$) |
| **union peak (any method)** | **5,109 distinct** | **1,454,635.9** | 0.2883 | **+0.0%** | is hierarchical |

¹ cross_entropy converged and stopped early at 29 evals (its own stopping rule), not the 2000 budget.

- **Coverage:** 5,109 distinct bundles / 209,952 = **2.43%** of the deployable space. This is a large-sample
  **lower bound** on the true peak, not exhaustive proof of global optimality.
- **Peak vs production_scheduler:** **+869.56%** in this tournament harness — identical to hierarchical's own
  +869.56%, because the peak *is* hierarchical's pick.

## What this answers

**"Is there a much-better bundle we never tried?"** — In the deployable, headline-safe space: **no evidence of
one.** 5,000 random draws (64× hierarchical's eval count) topped out at **99.6%** of hierarchical's gp/$ — and
at a *worse* SLA (0.302 vs 0.288), so the best random bundle is not even Pareto-better; it would be **rejected**
by the claim-gate. A high-budget adaptive search (cross_entropy) did *worse* (1.11M), converging to a different,
lower plateau. Three independent search strategies (hierarchical tournament, uniform random, cross-entropy) all
land at or below the same ceiling. That is the signature of a **broad plateau**, not a hidden spike.

**"At what magnitude would the genuine peak beat production_scheduler if we searched all 315k / 210k options?"**
— The estimate the pre-run analysis gave (peak is at most *modestly* above hierarchical, **not a multiple**) is
confirmed at 2.43% coverage. The union peak is **+0.0%** over hierarchical and **+869.56%** over
production_scheduler *in this harness*. Extrapolating the plateau: even a genuinely exhaustive 209,952-bundle
sweep would be expected to land within a **low-single-digit percent** of hierarchical — i.e. the genuine peak vs
production_scheduler stays in the **same ~+870% band (tournament harness) / ~+140–165% band (episode harness)**,
not a step-change to a multiple of it. The "surely there's a much-better option" intuition does **not** hold
here: the ~75-eval planner already sits on the plateau's top.

## Why 75 evals is enough (mechanism, not luck)

hierarchical_search doesn't sample 75 *random* bundles — it searches **by control timescale** (slow capacity /
medium routing-batching / fast precision-clock knobs), fixing each tier's best before moving on, then polishes
couplings. The top of this surface is set by a few dominant knobs (aggressive batching, fp8, high clock,
forecasted/backlog capacity, kv-aware routing, class-aware admission); once those are pinned, the remaining
knobs move gp/$ by <1%. So a structured ~75-eval search reaches the plateau that 5,000 random draws also find —
random needs vastly more draws to *stumble* onto the same corner that structure reaches directly. This is
evidence the planner's tournament win was **not** an artifact of a lucky small sample.

## Reproducibility on post-Batch-1 `main`

The committed artifact was produced on the pre-Batch-1 tree; re-running the harness on current `main` (after the
Batch-1 action-knob work, PRs #125/#126) **reproduces the meaningful numbers exactly**: `sla_aware` = 105,924.1,
`production_scheduler` = 150,030.7, `hierarchical@100` peak = **1,454,635.9**, **+869.56%** vs
production_scheduler — byte-identical. This is expected: Batch-1 added the new serving-engine knobs
(`kv_cache_precision`, `prefill_decode`) as **default-OFF / frozen-at-no-op** for the headline, so the deployable
headline-safe space my search measured (the 12 core surfaces, int4 excluded, product 209,952) and the scoring of
every bundle in it are unchanged. The one cosmetic shift: hierarchical_search's evaluation count reads **88** on
post-Batch-1 `main` (vs 80 pre-Batch-1) because the new default-off knobs are *nominally* in the candidate space
but frozen at no-op — the winning bundle and its gp/$ are identical, so the peak conclusion is unaffected.

## Caveats (fidelity)

- **SIMULATED / SIMULATOR_INFERENCE.** Single decision, single market·window (pjm·expensive, period 81), req_cap
  80. Not a multi-period episode, not validated telemetry.
- **Harness-dependent percent.** The +869.56% is the *tournament* single-rollout harness (forecast-synthetic
  jobs), which inflates the headline; the realistic PR #124 episode harness compresses the *same* policy to
  ~+140–165% vs production_scheduler (see `PR123_PR124_RECONCILIATION_FINAL.md`). The *plateau* conclusion — no
  much-better bundle exists — is the harness-independent takeaway; the specific percent is not.
- **Lower bound, not proof.** 2.43% coverage. It is a strong large-sample argument, not a certificate of global
  optimality over all 209,952 bundles.
- **Deployable, headline-safe space only.** int4 is EXCLUDED (it can inflate gp/$ via quality risk the gate is
  meant to catch). Including int4 would change the raw ceiling but not the deployable, honest one.
