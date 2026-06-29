# Production Baseline Ladder (Diagnostic)

Is the **strongest internal SLA-aware baseline** the right thing to compare Aurelius against — or is it an
*unrealistically strong* internal heuristic that sets a harder bar than real production? This defines a realistic
ladder and places the measured numbers on it. **Conclusion: the strongest internal SLA-aware baseline is a
sophisticated heuristic (backlog-aware capacity + ABS-conformal SRPT ordering), likely stronger than a vanilla
vLLM/FIFO production deployment — so "the MPC must beat it" is a *hard* bar, not the production bar.**

## The ladder (gp/$ from the +82.1% Azure window, `mpc_attribution.json`)

| rung | policy | what it models | gp/$ (A-window) | represented in repo? |
|--|--|--|--|--|
| 0 | **FIFO / naive** | first-come-first-served, reactive capacity | **98,843** | yes (`fifo` arm; `unified_replay` ordering=`fifo`) |
| 1 | **greedy** | backlog + kv-aware + aggressive (no SLA scheduler) | **94,054** | yes (`greedy` arm) |
| 2 | **vLLM-like continuous batching + FIFO** | real serving-stack default: continuous batching, FIFO-ish order, reactive autoscale | **not separately represented** | partial — `batching_policy` + FIFO ordering exist, but no calibrated vLLM arm |
| 3 | **topology / utilization / SLA-aware heuristic** | SLA-aware ordering + backlog-aware capacity | ≈ rung 4 | folded into rung 4 |
| 4 | **strongest internal SLA-aware baseline** | `SLA_AWARE_FALLBACK = {capacity: backlog_aware, ordering: abs_conformal (SRPT w/ conformal SLA), admission: off}` | **100,555** | yes (`sla_aware`; `controller.py:45`) |
| 5 | **Aurelius economic MPC (full action layer)** | the controller, all knobs, adaptive search | **183,152** | yes |
| 6 | **oracle diagnostic** | plans against the exact future (non-deployable) | > rung 5 (diagnostic only) | yes (`planning_oracle_records`) |

## Is rung 4 the real production baseline? — No, not without justification

`SLA_AWARE_FALLBACK` is **not** a vanilla production default. Two of its three levers are *sophisticated*:

- **`ordering = abs_conformal`** — ABS-conformal SRPT: shortest-remaining-time scheduling with a conformal SLA
  guard. This is a research-grade latency-aware scheduler, **not** what a stock vLLM/TGI deployment runs (those
  are roughly FIFO / continuous-batch order). SRPT is known to be near-optimal for mean latency, so it is a
  *strong* baseline.
- **`capacity = backlog_aware`** — a backlog-reactive autoscaler (better than fixed or naive lag-1).

So rung 4 already embeds two of the wins a naive operator would *not* have. On the A-window, rung 4 (100,555)
sits **only ~1.7% above FIFO (98,843)** and **above greedy (94,054)** — i.e. on this window the SLA-aware
scheduler's edge over FIFO is small, but it is still the *highest* non-MPC arm, so the gate (correctly) uses it
as the fair baseline (`training.claim_gate` picks the strongest non-weak baseline).

**Implication for "did all-knobs get worse?":** failing to beat rung 4 does **not** mean the MPC is worse than
production. Rung 4 is a strong internal heuristic; a realistic production deployment is closer to rung 0–2
(FIFO / vLLM continuous-batching), which the full-search MPC beats by a wider margin (+82% vs rung 4 implies an
even larger margin vs rung 0–2). The PR #121 clock-only number (−2 to −5% vs rung 4 on a *different* window) is
**not** evidence the MPC is below production — see `BASELINE_DRIFT_AUDIT.md` + `ACTION_SUBSET_CONTAINMENT.md`.

## Recommendation

1. **Always report against the full ladder**, not a single baseline. A headline "vs the strongest internal
   SLA-aware baseline" is the *hardest* honest bar; also report "vs FIFO" and (when built) "vs a calibrated
   vLLM continuous-batching arm" so the production-relevant margin is visible.
2. **Keep the Pareto gate on rung 4** (the strongest non-weak baseline) for headline-safety — that is the
   conservative, honest gate, and this audit does **not** weaken it.
3. **Build rung 2 (a calibrated vLLM-like arm)** as a follow-up so the ladder has a true production anchor
   between FIFO and the SLA-aware heuristic. Until then, do **not** assert rung 4 *is* the production baseline —
   it is an upper-middle internal heuristic.

## Honesty

The ladder numbers are from the bounded A-window (`mpc_attribution.json`, simulator-inferred). No tuning, no
gate change. The claim is narrow: rung 4 is a strong heuristic, not a justified production baseline, so it
should not be the *only* comparison point — not that the gate should change.
