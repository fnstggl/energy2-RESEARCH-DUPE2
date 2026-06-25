# The Canonical Aurelius Optimizer (Proposed Architecture)

> **Status:** PARTIALLY IMPLEMENTED (updated 2026-06-25, Phase B). The
> **Decision Layer is live**: `aurelius/optimizer/AureliusOptimizer` is a
> comprehensive fleet optimizer holding all five decision-layer surfaces
> (`energy`, `serving_queue`, `replica_scaling`, `placement`, `admission` — all
> in `IMPLEMENTED_POLICIES`) with a unified `optimize_fleet()` orchestration and a
> `serving_orchestration` accessor for the live `ConstraintAwareEngine`. Placement
> and admission are parity wirings of the existing `residency/` and
> `frontier/admission.py` surfaces. **Still target-only:** the single unified
> **Replay Layer** (4 loops not yet collapsed — Phase 1b), the **Forecast contract**
> and **Constraint Layer** promotion (Phase 4), and honest cross-surface
> combination (energy × serving are disjoint workloads today). The energy core
> remains "do not modify" (`docs/ENERGY_SYSTEM_MAP.md §8`, pinned by
> `tests/test_energy_core_preservation.py`); the comprehensive optimizer wraps it
> behavior-preservingly (0% KPI drift).

---

## Design principle

One optimizer, one objective, one replay engine, one forecast contract — with
workload-specialized **decision modules** plugged into a shared decision loop,
not five parallel re-implementations. The audit found four optimizers, four
replay loops, and a 26-file forecasting package that the optimizer never reads.
The canonical design collapses these into a single layered stack while
**preserving every benchmark definition and the public replay logic unchanged.**

```
                    ┌─────────────────────────────────────────────┐
   North Star  ───► │  SLA-safe goodput per infrastructure dollar  │
                    └─────────────────────────────────────────────┘
                                       ▲
 Workload  ─► Forecast Layer ─► Constraint Layer ─► Optimization Layer ─► Decision Layer
 (traces)     (advisory→        (SLA gate +          (objective +          (typed actions:
              decision-feeding   safe-utilization     unified policy:        provision /
              behind contract)   frontier as hard     SRPT+conformal |       order / route /
                                 constraints)         energy-arb | place)    throttle)
                                                                    │
                                              ┌─────────────────────┴───────────────────┐
                                              ▼                                          ▼
                                       Replay Layer (ONE engine)                  Shadow Research Layer
                                              │                                   (observational, no actuation)
                                              ▼
                                       Evaluation Layer ─► Benchmark Layer (frozen, unchanged)
```

---

## 1. North Star Objective
**Maximize SLA-safe goodput per infrastructure dollar** on the public-trace
rollup (`ROADMAP.md §1`):
`sla_safe_goodput / (gpu_infra_cost + energy_cost + network_cost)`.

- SLA is a **filter on the numerator**, never a subtraction term.
- FIFO is the sanity baseline; the **headline comparator is the strongest
  realistic safe baseline** (usually `sla_aware`).
- Energy cost is a **term in the denominator**, which is how the Era-1 energy
  optimizer becomes a *component* of the canonical objective rather than a
  competing objective. The current `JobScheduler` objective (min weighted cost)
  is reframed as "minimize the denominator for a fixed/duty-cycled numerator" —
  the energy work is preserved, not discarded.

## 2. Decision Inputs
Workload trace (arrivals + per-request tokens or per-job duration), live/replayed
`ClusterState` (`state/assemble.py`), price/carbon series, GPU topology, SLA
registry, and forecasts (below). One typed input schema feeding one loop.

## 3. Forecast Layer
Today: advisory-only (`forecasting/__init__.py:52`), 26 files, zero
decision-feeding. Target: a **single forecast contract** — each forecaster
exposes `(quantity, p50, p90, confidence, provenance)` and is consumed through
one adapter. Forecasts may be *promoted* from advisory to decision-feeding only
after passing a public-replay gate (the same gate that demoted the 3 shadow
modules). Output-length and TTFT forecasters stay advisory until they beat the
running-median ordering ceiling (Azure prompt↔output r=−0.022).

## 4. Constraint Layer
- **SLA gate** (`sla/`) — hard exclude HARD-violating decisions (already real on
  the constraint path).
- **Safe-utilization frontier** (`frontier/` BASE + DYNAMIC) promoted from a
  standalone recommender to a **hard ρ-ceiling constraint** the optimizer must
  respect. This is the natural home for the frontier work and removes the
  "recommendation that nothing consumes" problem.
- Feasibility (`optimization/constraints.py`) — deadlines, power caps, regions.

## 5. Optimization Layer
One optimizer object with **pluggable decision policies** selected by workload
class, all scored against the single north-star objective:
- **Serving policy** = the Era-2 Decoupled-Hybrid SRPT + conformal-α discipline
  (today inline in `srtf_serving_backtest.py`) extracted into a reusable
  `policy` module.
- **Energy/batch policy** = today's `JobScheduler` greedy/MILP (preserved
  behavior, wrapped behind the same interface).
- **Placement policy** = GPU/region routing (the `gpu_placement_scorer` hook),
  kept off by default until it stops regressing real KPI.

## 6. Decision Layer
A single typed `Decision` (provision N replicas / order request r / route to
region/GPU g / throttle to power p), carrying `executable_in_real_cluster`,
`shadow_only`, and provenance — unifying the five different `*FrontierDecision`
shapes and the scheduler's `ScheduleDecision`.

## 7. Replay Layer
**Collapse four loops into one** discrete-event engine that consumes a
`Decision` stream and the shared `simulation/cluster/serving.py` physics. The
SRTF serving sim's queue physics and the trace-replay tick loop must produce
identical KPIs for identical decisions (the unification's correctness gate).
The energy walk-forward (`backtesting/engine.py`) remains a thin mode of the
same engine.

## 8. Evaluation Layer
`benchmarks/economics.py` KPI math is already shared and **stays frozen**
(`compute_sla_safe_goodput_per_infra_dollar`). All policies report through it.

## 9. Benchmark Layer
**UNCHANGED.** Frozen scenario hashes (`benchmarks/v1/.scenario_hashes.json`),
the registry (`research/BENCHMARK_REGISTRY.md`), public commands
(`research/PUBLIC_BACKTEST_COMMANDS.md`), and `RESULTS.md §8` claim gate are not
touched by this design. Unification must reproduce current numbers before it may
change them.

## 10. Shadow Research Layer
Observational-only, no actuation (enforced as today). Home for: residency,
output-length/TTFT/cache forecasters, admission gate, economic overlay,
drift monitoring, and any not-yet-validated policy. Promotion out of this layer
requires public-replay evidence.

---

## Per-module classification (Keep / Integrate / Deprecate / Research-only)

| Module | Classification | Reasoning |
|---|---|---|
| `optimization/scheduler.py` (`JobScheduler`) | **KEEP** (wrap behind policy interface) | Canonical energy core; pinned by snapshot test; becomes the energy/batch policy |
| `optimization/objective.py`, `constraints.py` | **KEEP** | Reused as the cost-term + feasibility of the unified objective |
| `srtf_serving_backtest.py` disciplines | **INTEGRATE** (extract to `policy` + `calibration` libs) | Best goodput/$ lever in the repo; today trapped in a 6,628-LOC benchmark file |
| 3 inline conformal calibrators | **INTEGRATE → one shared calibration lib** | Duplicate; headline-bearing; belongs in the forecast/calibration layer |
| `traces/backtest.py` inline policies | **INTEGRATE → replay layer** | The public LLM leaderboard path; must be reconciled with the unified replay engine (behavior-preserving) |
| `constraints/engine.py` (`ConstraintAwareEngine`) | **INTEGRATE** (as constraint+recommendation front-end) | Real decision logic on the constraint path; overlaps "constraint_aware" naming |
| `frontier/` BASE, DYNAMIC | **INTEGRATE → constraint layer** | Safe-ρ ceiling is a real constraint; gives the frontier work a consumer |
| `frontier/` TRAINING | **RESEARCH-ONLY** (keep) | Training-trace studies; differentiated; no serving-runtime role yet |
| `frontier/` EVAL_WORKLOAD, BATCH_INFERENCE | **DEPRECATE** | Dead copy-paste of BASE; unexported; no benchmark/runtime consumer |
| `frontier/admission.py` | **RESEARCH-ONLY** (keep, off) | NEUTRAL on tested traces; optional shadow gate |
| `forecasting/` price/carbon | **KEEP** (advisory) | Feeds energy path advisorily; governed by learning loop |
| `forecasting/cara_*`, cache, output-length | **RESEARCH-ONLY** | Not validated to help; HURT when integrated; await predictor that beats the ceiling |
| `forecasting/gpu_placement_scorer.py` | **RESEARCH-ONLY** (keep hook, off) | Regressed real KPI; mechanism stays but stays disabled |
| `forecasting/ttft_shadow*`, `constraint_shadow_scorer`, `economic_*` | **RESEARCH-ONLY** | Shadow/diagnostic by construction |
| `residency/` | **RESEARCH-ONLY** (keep standalone) | Strong model-affinity signal but standalone, `MUTATION_ALLOWED=False`; later placement candidate |
| `simulation/replay.py` | **KEEP** (synthetic CLI/API mode) | Small, distinct purpose (synthetic demo/robustness) |
| `simulation/cluster/engine.py` + `serving.py` | **KEEP → become the one replay engine** | Largest sim; shared physics; the unification target host |
| `backtesting/engine.py` | **INTEGRATE** (mode of unified replay) | Canonical price/carbon walk-forward; fold in as a mode |
| `connectors/` DCGM/K8s/Prometheus/Topology | **KEEP** | Real telemetry I/O on the constraint path |
| `connectors/` vLLM/Triton/Ray/OTel | **RESEARCH-ONLY** (mark experimental) | Scaffolding, test-only, not exported |
| `connectors/dcgm.py` vs `ingestion/dcgm_provider.py` | **DEPRECATE one** | Duplicate; keep the one actually imported (`ingestion/dcgm_provider.py`) |
| `execution/` real executors | **RESEARCH-ONLY** | `allow_real_execution=False`; no live path exists |
| `sla/`, `state/`, `safety/`, `roi/`, `monitoring/`, `shadow/`, `api/` | **KEEP** | Distinct, working roles; no duplication |
| `OptimizationConfig.carbon_objective` family | **DEPRECATE** (dead config) | Never read by the optimizer; doc-vs-code drift |
| `JobScheduler` migration/MPC (`replan_remainder`, `*_migrate*`) | **RESEARCH-ONLY** | Test-only; no benchmark/runtime caller |

---

## What "done" looks like
- One optimizer entry point with a documented interface; four policies behind it.
- One replay engine; the four current loops reduced to modes of it, reproducing
  current benchmark numbers bit-for-bit before any change is claimed.
- The conformal-SRPT discipline reachable from a real serving-runtime path (even
  if shadow-gated), so the headline metric can actually be influenced at runtime.
- Forecast contract in place; promotion gate enforced.
- Dead duplicates (eval/batch frontier, duplicate DCGM, dead carbon config)
  removed; scaffolding clearly marked experimental.
- No benchmark definition, public replay logic, or evaluation infra changed.
