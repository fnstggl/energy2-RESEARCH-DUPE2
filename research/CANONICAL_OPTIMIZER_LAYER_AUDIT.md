# Canonical Optimizer — Layer / Forecast / Surface / Dataset Audit (2026-06-25)

> Honest record behind the Phase B+ "full-layer optimizer" + Phase C harness
> work. Read-only audit by four parallel agents + direct verification; the
> implementation that followed is summarized at the end. Directional simulator
> evidence only — not production savings (`docs/RESULTS.md` §8).

## 1. Is the canonical optimizer fully layered? (target architecture)

Before this work, of the 7 target concerns only the **Decision Layer** was a
first-class, optimizer-owned component. **Zero `*Layer` classes existed.**

| Layer | Before | After this PR | Wraps |
|---|---|---|---|
| Decision Inputs | scattered (3 schemas) | scattered (unchanged) | per-surface dicts / `ClusterState` / trace schemas |
| **Forecast** | scattered | **FIRST-CLASS** `.forecast` | causal `forecasted_mcs` + honest taxonomy |
| **Constraint** | scattered (4 subsystems) | **FIRST-CLASS** `.constraints` | `ConstraintAwareEngine` (+SLA, +frontier) |
| **Objective** | scattered (split-brain) | **FIRST-CLASS** `.objective` | `economics.py` goodput/$ |
| Decision | first-class | first-class | the 6 registered policies |
| **Replay** | scattered (4 loops) | **FIRST-CLASS result schema** `.replay` | `replay_result` adapters (engine still 4 loops — Phase 1b-A future) |
| **Evaluation** | scattered | **FIRST-CLASS** `.evaluation` | `economics` + `per_workload` |

Net: **1/7 → 6/7 first-class.** The one remaining gap is a *single unified
replay engine* (Phase 1b-A) — the 4 discrete-event loops still run separately;
only their result schema is unified. That requires a 0%-delta parity harness and
is deliberately out of scope here (changing replay can move every benchmark
number).

### What made the energy path "productized" that the others lacked
Energy alone had: a golden frozen benchmark (`benchmarks/golden/canonical_energy_backtest.json`),
a SHA-256 scenario lock (`benchmarks/v1/.scenario_hashes.json`), fair-baseline
selection (`per_workload.select_headline_baseline`), a claim gate (`RESULTS.md §3.1`),
a real objective, and parity tests. The serving/placement/admission surfaces were
reachable + wiring-tested but had **no frozen reference, no fair-baseline rule,
no claimable headline**. The `ObjectiveLayer` + `EvaluationLayer` + the Phase C
harness are the start of giving the serving surfaces the same productization.

## 2. ML forecasting reality — how real, how much helps

**No ML forecaster has a positive, measured goodput/$ delta wired into production.**

- ~7 genuinely-trained models exist: price/carbon (LightGBM quantile, real
  CAISO/PJM/ERCOT, **wired** but low-leverage) and the CARA family (HGB:
  latency, output-length, queue, cache-prefix, economic — all **shadow /
  un-persisted**).
- The KPI wins come from **utilization/target-ρ sizing + heuristic price
  corrections** (Regime/Spread), **not** ML. Benchmark-confirmed: output-length
  forecasting **HURTS −7..−11%**, gpu_placement **−7.3%**, admission neutral,
  demand forecasting **<0.3%** of the Azure +25.75%.
- The offline train→promote→drift pipeline (`ml/`, `learning/`, `monitoring/`)
  is real code but **operationally dormant** — zero model artifacts on disk, no
  scheduled job, CARA models fit at runtime and never persisted.

**Conclusion: do not chase forecasting alpha on the current public traces** — the
signal isn't there (Azure prompt↔output r≈−0.02; running-median ≈ oracle). The
`ForecastLayer` therefore surfaces only the one *causal-in-decision* forecaster
(`forecasted_mcs`) and labels price/carbon advisory and the rest research-only.
The `uncertainty.py`/`replay.py` `AttributeError` (legacy `UncertaintyEstimator`
API) is fixed (degrades to no-risk-penalty instead of crashing `/simulate`).

## 3. Is it optimizing every surface it can?

Honest count: **~7 controllable surfaces**, not "5/6". The six registered
policies + the **live-cluster orchestration** engine (`ConstraintAwareEngine` —
binding-constraint SPREAD / REROUTE / MIGRATE / SCALE; reachable via
`recommend_live` / `optimize_fleet(live=)`, now in `DECISION_SURFACES`).
Demand-forecast anticipation and SRPT preemption are **already folded** into
`genai_serving`/`replica_scaling` and `serving_queue`. Thermal-spread + migration
already live inside the constraint engine.

Physical-plane surfaces are **NOT productized** because they cannot be honestly
benchmarked on public data (the 12 pilot-only signals):

| Candidate surface | Verdict |
|---|---|
| Thermal-aware spread / reroute | already in the constraint engine (recommendation-only) |
| Thermal power-cap (serving) | SIM-ONLY — a throttle *consequence*, no operator knob; batch power-cap benchmark ≈0% |
| Topology-for-collectives | RESEARCH-ONLY — simulated cost exists, no decision API, no benchmark |
| Memory/KV (batch size, eviction) | SIM-ONLY — batch size is hard-coded; KV is preemption physics, no knob to lift |
| Carbon-aware serving/placement | RESEARCH-ONLY — action+gates exist, generator missing, no benchmark |

## 4. Dataset sufficiency + found public datasets for gaps

| Dimension | Have (committed) | Sufficient? | Gap-fill public dataset (found) |
|---|---|---|---|
| Demand/arrival | Azure LLM 2024, BurstGPT, Alibaba GenAI | ✅ | (optional: Mooncake FAST25) |
| Output-length/service | tokens present | ⚠️ signal-absent (ML hurts) | Mooncake agentic traces (stronger length signal) |
| GPU util / job | AcmeTrace, Alibaba GPU v2023, Philly, MIT Supercloud | ✅ | — |
| Energy price / carbon | CAISO/PJM/ERCOT + WattTime (wired) | ✅ | (ElectricityMaps/ENTSO-E live) |
| **Thermal / cooling** | **none** (DCGM connector fixture-only) | ❌ | **GWDG GPU telemetry (Zenodo)** — DCGM temp/power/util/clocks/mem; "When GPUs Fail Quietly" arXiv:2603.28781. Facility: NREL CoolerChips / RICO HVAC |
| **Topology / network** | none | ❌ | **NCCL-tests** (NVIDIA, self-generable bandwidth/latency × topology); **Vidur** kernel CSVs (MSR). Alibaba HPN is paper-only |
| **Memory / KV** | DGX-Spark bench (18 rows) | ❌ | **Mooncake FAST25** (Apache-2.0, `hash_ids[]` → prefix-reuse); LMCache agentic traces |

## 5. Phase C harness (implemented)

`aurelius/benchmarks/phase_c.py` — the three-way comparison that the governance
mandate (`OPTIMIZER_UNIFICATION_PLAN.md §1`) required but had no code:
- **Current-Main vs Best-Aurelius vs Candidate** on the same trace+seed+warp.
- **On-demand denominator** (reuses `forecasted_mcs.evaluate_c_schedule`,
  `sum(c)·tick_hr·GPU_HOUR_USD`, no spot) — re-measures forecasted_mcs/OSOTSS
  without the spot-fleet cost trick.
- **Reproducible** — every artifact serializes `seed` + a SHA-256 trace-content
  hash (public traces have no frozen-YAML hash).
- **Honest ranking** — ranked by the canonical `ObjectiveLayer`; a non-deployable
  oracle arm can never be the `deployable_winner`.
- `standard_replica_scaling_arms` builds the deployable, causal, on-demand
  default three-way (lag-1 reactive / forecasted_mcs EWMA / forecasted_mcs
  quantile).

## 6. Are all layers run at once? (combined multi-surface benchmark)

**No — and honestly, not yet possible.** `optimize_fleet` is a per-surface
fan-out; energy and serving operate on disjoint workloads. A combined run needs
(a) one trace carrying all surfaces' decision inputs simultaneously (no public
trace has this — the physical plane is pilot-only) and (b) the unified replay
engine (Phase 1b-A). Until then, "no honest combination exists, and none is
fabricated."

## 7. Next phase (recommendation)
**Phase 1b-A — unified replay engine** (0%-delta gated) is the last layer gap
and the prerequisite for any honest multi-surface combination. After that, the
highest-leverage move is **pilot telemetry** (the 12 missing signals) — not more
ML forecasting and not more public-trace backtests.
