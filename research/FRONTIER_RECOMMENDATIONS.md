# Frontier Recommendations & Claims Audit (Phase 4 — Discovery Only)

> Discovery run. No optimization claim, no frontier-improvement claim, no new
> runtime behavior. Honest decomposition of existing claims + ranked bottlenecks.
> Directional simulator only (`docs/RESULTS.md §8` gate unmet).

## Step 6 — Claims audit (decomposition of major referenced claims)

For each headline claim currently in the repo, where did the gain come from?

| Claim (as stated) | Source of the gain | Optimizer intelligence? | Honest read |
|---|---|---|---|
| **`energy`/CA +11% vs `current_price_only`** (canonical energy) | optimizer intelligence: temporal shift + regional arbitrage + power throttle | **Yes** | Legitimate; beats the strongest *safe* baseline at 0 deadline misses. The cleanest real value. |
| **`serving_queue` abs-conformal +313%/+557% vs FIFO** (Azure/BurstGPT, fixed c=4) | **queue ordering** (SRPT) + **calibration** (abs-error conformal α) | Partly (ordering) | Real *ordering* effect, but **vs FIFO**, a "catastrophically overloaded" baseline (p99≈732 s). Honest comparator is **SLA-aware oracle: +83%**. No new prior (running-median); abs-vs-rel gain is calibration, not prediction. |
| **Spot Fleet MCS "north-star ACHIEVED" +304.7%/+381.2% vs SLA-oracle** (run -23B) | **cost denominator: spot/preemptible pricing** (−41.5% fleet $) + **MCS provisioning** | **No** (uses **FIFO** queue) | The run's own words: *"the north-star gap is closed entirely by the cost denominator."* This is **pricing + capacity**, not optimizer ordering intelligence, and it does **not** run through `AureliusOptimizer`. |
| **abs-conformal + MCS "TRUE compound" +131% vs SLA-oracle** (run -23) | provisioning (capacity) dominates; ordering adds nothing | Partly | TRUE 2×2 shows **FIFO+MCS (+137%) ≥ Abs+MCS (+131%)** → ordering is **redundant** once provisioning scales capacity. |
| **MCS vs SHU +24.5%/+2.6%** (run -22, extreme scales) | provisioning search (min safe replicas) | Yes (provisioning) | Mostly **TIE** at normal scale; wins only at 500×. MCS **raises** GPU-hours +12.5% on diurnal load. |
| **GpuPlacementScorer routing +54.7 pp on-best-GPU** | proxy metric only | No | **Proxy moved, real KPI regressed −7.3%.** Harmful; correctly shadow-only. |
| oracle / conformal-oracle ceilings (+322%/+644% vs FIFO) | **clairvoyant information** | No | Analysis ceilings only; never a headline (oracle uses actual token lengths). |

**Decomposition headline:** recent goodput/$ gains decompose into four distinct
sources, in descending measured leverage for serving: **(1) cost denominator
(spot pricing) ≫ (2) provisioning/capacity scaling > (3) queue ordering
[fixed-capacity only] ≫ (4) placement (negative)**. Only (3) is an
`AureliusOptimizer` policy today; (1) and (2) — the largest levers — are **not**.
For the energy world, the gain is genuine optimizer intelligence (arbitrage).

## Step 7 — Bottleneck discovery

**1. What contributes most to SLA-safe goodput/$?**
The **cost denominator** (spot/preemptible pricing, −41.5% fleet cost → +167 pp)
and **capacity provisioning** (FIFO+MCS +137% vs SLA-oracle). Both are *un-routed*
(not AO policies). Within `AureliusOptimizer`: **`energy` arbitrage** (+11% vs safe
baseline) and **`serving_queue` ordering** (+83% vs oracle, fixed-capacity only).

**2. What contributes least?**
`placement` (GpuPlacementScorer) is **harmful** (−7.3% lc); `admission` is
**neutral** (±0.34%); and `serving_queue` ordering contributes ≈0 once realistic
provisioning is present.

**3. Which policies should remain canonical?**
`energy` (HIGH, clean) and `serving_queue` (HIGH at fixed capacity; keep, but
scope its claim to the overloaded regime).

**4. Which policies should be deprecated / kept shadow-only?**
- `placement` shadow scorer → **keep OFF / research-only** (harmful on real KPI).
- `admission` shadow gate → **keep OFF / research-only** (neutral).
- The three stub policies (`replica_scaling`, `placement`, `admission`) remain
  importable seams; do not promote until validated.

**5. Which combinations are strongest?**
**None through `AureliusOptimizer` today** (the two implemented policies are
disjoint; the rest are stubs). The strongest *overall* serving stack is
**FIFO + MCS provisioning + spot pricing** — which is **not** an AO configuration
and whose dominant lever (pricing) is not optimizer intelligence. Measured
interactions are **negative/substitutive** (ordering × provisioning) or
**harmful** (placement), not positive.

**6. Largest remaining bottleneck.**
The biggest goodput/$ leverage (provisioning + cost/pricing) is **not represented
as an `AureliusOptimizer` policy**, and the optimizer **cannot compose** its
policies because (a) `energy` and `serving_queue` are disjoint workload classes
and (b) the replay layer is not unified (Plan Phase 1b). The optimizer's *coverage
of the real decision surface* — not its prediction accuracy — is the bottleneck.

**7. Highest-expected-value research direction (ranked).**
1. **Implement `ReplicaScalingPolicy` inside `AureliusOptimizer`** (wrap the
   validated SHU/MCS provisioning) — this is the largest measured serving lever
   and is currently un-routed.
2. **Unify the replay layer (Plan Phase 1b)** so provisioning, pricing, and
   ordering decisions co-exist on one workload — the only way an honest
   *combination* frontier (and the negative ordering×provisioning interaction)
   can be optimized rather than estimated.
3. **Represent the cost denominator (spot/preemptible pricing) as an explicit
   decision** (a cost-model input, not a benchmark-only overlay) so the dominant
   measured lever becomes a governable optimizer decision with a real production
   analogue.
4. **De-prioritize** further queue-ordering / length-prediction work: its benefit
   is already ~oracle at fixed capacity and **vanishes under realistic
   provisioning**. **Drop** placement-scorer promotion (harmful).

## Governance compliance (this run)
- **No optimization claim / no frontier-improvement claim** is made. Every number
  is an existing benchmark, reported with its honest comparator (SLA-oracle /
  strongest safe baseline), not FIFO-only.
- **Baseline governance** honored: comparisons are Current Main vs Best Aurelius
  vs strongest baseline (FIFO shown only as sanity).
- **Optimizer-first**: each recommendation maps to a real production decision
  (provisioning, pricing, routing); no benchmark-only optimization proposed.
- **No new policy/optimizer/benchmark/dataset/eval/replay/objective/SLA/pricing/
  trace changes**; no hyperparameter tuning for gains.
