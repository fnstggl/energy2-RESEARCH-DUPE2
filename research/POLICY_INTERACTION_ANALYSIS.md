# Policy Interaction Analysis (Phase 4 — Discovery Only)

> Discovery run. No new policy/optimizer/benchmark; no behavior changed. Measures
> whether existing `AureliusOptimizer` policies compose **positively, neutrally,
> or negatively** — using only combinations that can honestly run today.
> Directional simulator only.

## Step 2 — Policy matrix

| Policy | Implemented? | Workload class | Independently evaluable? | Jointly evaluable (through AO)? |
|---|---|---|---|---|
| `energy` | Yes | batch jobs / energy-price traces | **Yes** (canonical energy backtest) | only with another *batch-energy* policy — none exists |
| `serving_queue` | Yes | LLM request queue / serving traces | **Yes** (abs-conformal serving backtest) | only with another *serving* policy that runs in the same loop — none implemented in AO |
| `replica_scaling` | **No (stub)** | serving (provisioning) | No (raises) | No |
| `placement` | **No (stub)** | serving (routing) | No (raises) | No |
| `admission` | **No (stub)** | serving (flow control) | No (raises) | No |

**Conclusion of the matrix:** the only two implemented policies live in
**different workload classes** with no shared benchmark/replay, and the only
policies that *would* share `serving_queue`'s workload (`replica_scaling`,
`placement`, `admission`) are **unimplemented stubs**. Therefore **no honest
policy combination can be evaluated through `AureliusOptimizer` today.**

## Step 4 — Combination search

### What is NOT feasible (and why we do not fake it)
- **`energy` + `serving_queue`**: disjoint workloads (batch cost vs serving
  goodput/$); there is no benchmark where both make a decision on the same
  workload. A combined number would require a synthetic combined benchmark —
  **prohibited** by this run. **Not evaluated.**
- **Any combination with `replica_scaling` / `placement` / `admission`**: these
  raise `NotImplementedError`. **Not evaluable.**

So the formal AO matrix (A=`energy`, B=`serving_queue`) yields only the singletons
A and B (already in `CANONICAL_FRONTIER.md`); **A+B does not exist** as an
honest, runnable configuration.

### The one real interaction that *can* be measured (already benchmarked)
The serving-queue ordering decision and provisioning (replica count) **do**
interact, and the repo already measured it in a true 2×2 factorial
(`research/results/joint_mcs_abs_conformal_2026-06-23.md`, Azure 2024, vs
SLA-aware oracle, provisioned-hours cost). Provisioning here is the un-routed
MCS policy (the future `ReplicaScalingPolicy`), so this is an interaction the AO
*will* host once that policy is implemented — measured today outside AO:

| Condition | vs SLA-oracle |
|---|---:|
| FIFO + fixed c=4 | −56% |
| abs-conformal (`serving_queue`) + fixed c=4 | **+83%** |
| FIFO + MCS provisioning | **+137%** |
| abs-conformal + MCS (TRUE compound) | **+131%** |

**Interaction effect: NEGATIVE (substitutive).** Abs-conformal+MCS (+131%) is
**−6 pp below** FIFO+MCS (+137%). The two value levers attack the **same**
quantity (queue delay) by different means:
- `serving_queue` (SRPT ordering) only helps when capacity is **fixed and
  overloaded** — it reorders a deep queue.
- MCS provisioning removes the deep queue by **adding capacity**; once the queue
  is short, SRPT ordering has nothing to reorder, and its **preemption overhead
  makes it marginally worse than FIFO**.

Direct quote from the run: *"When MCS controls queue depth by scaling capacity,
the queue is short enough that SRPT ordering provides no benefit… The queue
discipline axis only pays off in the fixed-capacity overloaded regime."*

### Interaction summary
| Pair | Through AO today? | Interaction | Evidence |
|---|---|---|---|
| `energy` × `serving_queue` | No (disjoint workloads) | **undefined** (no shared benchmark) | workload-class split |
| `serving_queue` × provisioning (MCS→future `replica_scaling`) | not yet (MCS un-routed) | **NEGATIVE / substitutive** (−6 pp) | `joint_mcs_abs_conformal_2026-06-23.md` |
| `serving_queue` × spot pricing (cost denominator) | not yet (un-routed) | **independent / additive on cost** (ordering unaffected; cost ↓) | `spot_fleet_mcs_backtest_2026-06-23.md` |
| `placement` (shadow scorer) × `energy` | scorer is off; benched via gpu_routing | **NEGATIVE on real KPI** (−7.3% lc gpd/$) | gpu_routing ablation (see ablation report) |

**Headline finding:** the assumption that individually-useful levers compose
positively is **false here**. The strongest single lever for serving goodput/$
(provisioning + spot pricing) makes the `serving_queue` ordering lever
**redundant**, and one shadow lever (`placement`) is outright harmful. Composition
must be **measured**, not assumed — and today, through `AureliusOptimizer`, there
is **no positively-composing combination** of implemented policies.
