# Frontier-Driven Research Audit — 2026-06-24

> **Run type:** frontier-driven research (decisions, not pricing). **Outcome:**
> NO FRONTIER ADVANCE — FINDINGS RECORDED. No new optimizer logic, policy,
> benchmark, replay, eval, objective, dataset, or tuning change. Directional
> simulator only (`docs/RESULTS.md §8` gate unmet). Pushed to PR #54.
>
> Reads: `CANONICAL_FRONTIER.md`, `POLICY_INTERACTION_ANALYSIS.md`,
> `POLICY_ABLATION_REPORT.md`, `FRONTIER_RECOMMENDATIONS.md`, `ROADMAP.md`, and the
> current-main results `online_sotss_backtest_2026-06-23.md`,
> `joint_osotss_conformal_backtest_2026-06-24.md`, `gsf_spot_fleet_backtest_2026-06-26.md`.

## Start-of-run state
1. **Current frontier configuration:** `replica_scaling = OSOTSS` (causal EWMA
   per-tick capacity) + **FIFO** queue ordering. `serving_queue` is **off** at the
   frontier (it is negative under variable-c — see §3).
2. **Current best validated result:** OSOTSS, Azure 159,578 gp/$ (+533% vs
   SLA-oracle) / BurstGPT 178,109 (+778%) — **but these use the GSF spot-fleet
   cost model**; on honest provisioned cost the decision frontier is
   **FIFO+OSOTSS = 63,831 (Azure) / 71,244 (BurstGPT)**.
3. **Largest remaining bottleneck:** the decision surface expressible on the
   canonical public traces (arrival time + token counts) is **provisioning +
   ordering**, and both are at/near ceiling. The binding constraint is **data/
   signal**, not algorithms (see §4).
4. **Highest-EV direction:** a richer public trace (model identity / KV-cache /
   GPU heterogeneity) or pilot telemetry — to unlock placement/affinity/cache
   decisions the current traces cannot express (see §6).

## 1. Frontier governance — separating Aurelius value from pricing
Per this run's GPU-cluster product rule, gains that come from **pricing
assumptions** (spot/preemptible, on-demand switching) are **NOT AURELIUS VALUE**.
Decomposing the Azure headline (provisioned-cost basis, SLA-aware oracle = 25,208):

| Layer | gp/$ | vs SLA-oracle | Attribution |
|---|---:|---:|---|
| SLA-aware oracle (base) | 25,208 | 0% | external baseline |
| FIFO + fixed c=4 | 11,183 | −56% | weak ref |
| **FIFO + OSOTSS (provisioning DECISION)** | **63,831** | **+153%** | **AURELIUS VALUE (capacity right-sizing)** |
| + serving_queue (conformal SRPT) | 61,262 | +143% | **negative** (−4.0%) |
| Headline WITH GSF spot-fleet cost model | 159,578 | +533% | **+71% of the gain is pricing** → NOT AURELIUS VALUE |

**Decomposition:** of the +533% headline over SLA-oracle, the provisioning
*decision* contributes **29%** and **spot pricing contributes 71%** (a ×2.50
cost-denominator multiplier). The legitimate Aurelius decision frontier is
**+153% vs SLA-oracle (FIFO+OSOTSS)**, not +533%.

## 2. The real Aurelius decision value (what to keep claiming)
- **Provisioning (replica_scaling = OSOTSS)** is genuine, large, and decision-
  driven: FIFO+fixed → FIFO+OSOTSS = **+470.8% Azure / +1187.4% BurstGPT vs
  FIFO+fixed**, **+153% / +183% vs SLA-oracle**, at lower cost ($4.04 vs $4.28/hr
  Azure) and SLA-safe (p99 9.95s ≤ 10s). This is a real operator decision (how
  many replicas to run each minute from causal load) and is canonical in
  `AureliusOptimizer(policy="replica_scaling")`.
- This is the configuration future claims should headline — on **provisioned
  cost**, vs **SLA-aware oracle**, not FIFO and not spot-inflated.

## 3. Composition findings (serving_queue × replica_scaling)
Measured (current main, `joint_osotss_conformal_backtest_2026-06-24`): conformal
SRPT ordering is **negative** on top of variable-c provisioning —
conformal+OSOTSS < FIFO+OSOTSS by **−4.0% (Azure) / −6.4% (BurstGPT)** and it
**reduces SLA-safe request counts** (Azure −74, BurstGPT −120). Root cause is
structural: when capacity drops at a tick boundary, SRPT-preempted long jobs
starve and exceed the SLA budget (p99 13.5s > 10s). **Both directions of capacity
deviation (MCS over-provision, OSOTSS under-provision) produce the negative
interaction → not tunable.** Confirms the Phase-4 substitutive-interaction
finding empirically. **Recommendation:** `serving_queue` should be **off when
`replica_scaling` is active** (the current frontier already does this).

## 4. Bottleneck-first analysis (largest sources of loss)
| Source | Status on Azure/BurstGPT | Decision lever | Verdict |
|---|---|---|---|
| Capacity right-sizing (wasted GPU-h / SLA loss) | OSOTSS causal ≈ −0.3–0.4% SLA-safe vs oracle; cost ≤ AMCSG | replica_scaling | **near-oracle; causal improvements failed 5/5** |
| Queue delay (ordering) | FIFO optimal under variable-c | serving_queue | **negative; exhausted** |
| Admission control | all requests equal (no priority/class signal in trace) | admission | **no signal to act on** |
| Placement / routing | single homogeneous pool in trace | placement | **no heterogeneity to exploit** |
| Cache/prefix reuse | no KV-hash signal in trace | forecast/placement | **not expressible** |
The largest *legitimate* residual is the OSOTSS causal-vs-oracle provisioning gap
(~0.3–0.4% SLA-safe on BurstGPT), already attacked 5× and structurally limited by
trace burstiness. **Conclusion: the decision surface on these traces is
exhausted.**

## 5. Decision-lever exhaustion (evidence)
7 consecutive runs with no frontier advance (repo's Five-Failure Rule + 2 nulls):
C1PGS (fail), SOTSS-GSF (fail/null), Adaptive-EWMA OSOTSS (fail), Stochastic-margin
OSOTSS (fail), OSSC OSOTSS (fail — 5/5), Joint MCS+conformal (null), Joint
OSOTSS+conformal (null). Plus ML/stratified output-length priors (null; running-
statistics ceiling 70–82%, trained predictor blocked by missing pilot labels).

## 6. Optimizer-first + research (3 papers for the highest-EV legitimate direction)
No untried decision lever is feasible on the current traces (§4), so the highest-EV
direction is **richer signal**. Three systems, each a real operator decision, each
mapped to an `AureliusOptimizer` component — and each currently **blocked by trace
signal**, which *is* the recommendation:

1. **Mooncake (KVCache-centric serving, FAST'25).** Prefix-cache-aware routing/
   placement to avoid recompute. → `PlacementPolicy` / `ForecastLayer`.
   *Why help:* real GPU-hour savings from cache hits. *Why fail / blocker:* Azure/
   BurstGPT have **no KV-cache hashes** → cannot validate. *Action:* ingest
   Mooncake `FAST25` traces (Apache-2.0; has `hash_ids`).
2. **Heterogeneity-aware routing / deployment-co-optimization (MOBO routing,
   MLSys'26, arXiv:2602.10729).** Route by GPU type × model. → `PlacementPolicy`.
   *Why help:* exploits A100/H100/T4 TTFT spread. *Why fail / blocker:* the public
   traces are a **single homogeneous pool** (GpuPlacementScorer already regressed
   −7.3% lc precisely because there is no real heterogeneity to exploit here).
   *Action:* a trace with GPU-type/topology labels.
3. **Model-affinity prewarming / residency (Alibaba GenAI 2026 signal; cold-start
   literature).** Keep-warm / prewarm decisions by model. → `AdmissionPolicy` /
   residency `ShadowResearchLayer`. *Why help:* Alibaba GenAI shows +89% from
   model-affinity. *Why fail / blocker:* Azure/BurstGPT have **no model identity**.
   *Action:* validate on Alibaba GenAI / a model-labeled trace.

All three are real infrastructure decisions that plug into `AureliusOptimizer`,
but **none can be honestly validated on the required public traces** — confirming
the bottleneck is **data**, not algorithms.

## 7. Baselines (required reporting)
| Configuration | Azure gp/$ | BurstGPT gp/$ | Basis |
|---|---:|---:|---|
| FIFO (sanity) | 11,183 | 5,534 | fixed c=4, provisioned |
| Strongest external baseline (SLA-aware oracle) | 25,208 | ~20,280 | provisioned |
| **Current Main = Current Frontier (decision)** | **63,831** (FIFO+OSOTSS) | **71,244** | provisioned |
| Candidate (this run) | **none viable** | **none viable** | — |
| (headline w/ spot pricing — NOT Aurelius value) | 159,578 | 178,109 | GSF spot cost model |

No candidate beats the frontier → **not a frontier improvement** (none proposed,
by design — forcing one against a 7-run exhaustion streak would be
benchmark-chasing, which the rules forbid).

## 8. Classification & recommendation
**NO FRONTIER ADVANCE — FINDINGS RECORDED.**

Highest-value next actions (in order):
1. **Honesty correction (free, high value):** headline the **decision frontier**
   (FIFO+OSOTSS, +153%/+183% vs SLA-oracle, provisioned cost), and report the spot
   gain separately as a **pricing** effect (NOT Aurelius value). Update the
   leaderboard framing accordingly.
2. **Architecture cleanup:** make `serving_queue` **off by default when
   `replica_scaling` is active** (empirically negative; structural).
3. **Data unblock (the real bottleneck):** ingest a richer public trace
   (Mooncake KV-cache, Alibaba GenAI model-identity, or a GPU-heterogeneous trace)
   so placement / cache-affinity / admission become *expressible and validatable*
   decisions. This is the only path to the next legitimate decision frontier.
