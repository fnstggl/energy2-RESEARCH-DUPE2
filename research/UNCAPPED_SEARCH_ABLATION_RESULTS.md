# Uncapped search-strategy ablation - results

**Question.** Is search where the uncapped headline value lives? The published uncapped
benchmark (`request_cap_sweep.json`) compares one planner arm against the production
baseline. This ablation reruns the IDENTICAL uncapped harness and varies only the search
strategy available to the planner, so the headline gap can be attributed.

**Harness.** `scripts/run_uncapped_search_ablation.py`, July 2026. Same
`build_market(req_cap=None, mooncake_limit=12000)`, same `select_windows(win_len=6)`
expensive windows truncated to 3 decisions, same `run_period_episode` persistent-world
reward path, same real diurnal day-ahead prices, same 300 s isolated-subprocess cell
timeout as the PR #124 uncapped cells. Windows and load identical to the published run:
PJM [81,82,83] 576,912 requests; ERCOT and CAISO likewise reproduce the published
per-window totals (1,549,249 total). Forecaster, simulator, objective, cost model,
constraint gates, and baseline identical across arms by construction; every planner arm
shares one controller configuration and differs only in candidate set / search mode.

**Reproduction contract.** The `production_scheduler` anchor reproduced the published
gp/$ byte-identically in all three markets (130,538.91 / 130,178.26 / 130,000.36),
confirming identical inputs (trace, prices, windows, load, cost model). The
`aurelius_mpc_hierarchical_search` arm reproduced the published headline cells within
+0.58% / +1.22% / -0.003% (PJM / ERCOT / CAISO); the small planner-side deltas are
consistent with search nondeterminism / planner evolution since the published artifact,
and all arms in THIS run share one code version, so within-run comparisons are exact.

## Results (gp/$ = SLA-safe goodput per modeled infrastructure dollar)

| arm | PJM | ERCOT | CAISO | mean ratio | mean pct |
|---|---|---|---|---|---|
| production_scheduler (anchor) | 130,538.91 | 130,178.26 | 130,000.36 | 1.00x | 0 |
| clock_only | TIMEOUT (300 s) | TIMEOUT (300 s) | TIMEOUT (300 s) | see below | see below |
| fixed_24_grid | 463,646.35 | 465,788.37 | 484,669.90 | 3.62x | +261.9% |
| physics_guided_candidates (argmax, no beam) | 463,646.35 | 465,788.37 | 484,669.90 | 3.62x | +261.9% |
| exhaustive_default4 (full safe grid) | 618,044.64 | 625,712.59 | 645,507.33 | 4.84x | +383.6% |
| aurelius_mpc_current_default (bounded beam) | 620,784.74 | 627,443.32 | 648,064.41 | 4.85x | +385.4% |
| aurelius_mpc_hierarchical_search | 1,047,909.94 | 1,077,593.68 | 1,111,878.56 | 8.29x | +728.6% |

SLA violation rates (PJM / ERCOT / CAISO): production 0.0382 / 0.0447 / 0.0458;
fixed grid and candidates 0.0073 / 0.0102 / 0.0099; exhaustive grid and beam
0.0081 / 0.0105-0.0106 / 0.0105; hierarchical 0.0022 / 0.0026 / 0.0024. Every
completing planner arm lowered the SLA violation rate versus the baseline; the
hierarchical arm is lowest in all three markets.

`clock_only` did not complete the uncapped replay within the 300 s cell budget in any
market: the single-surface policy it selects leaves the serving posture unmanaged at
~190k requests per period, and the replay cost grows super-linearly (the same failure
mode as the naive `sla_aware` baseline in the published sweep). A retry pass at a 1,800 s
cell budget is recorded in the artifact (`uncapped_search_ablation.json`); see the
artifact for its final status.

## Reading

1. **Restoring a sane joint posture is worth ~3.6x.** Any arm that jointly sets
   batching / capacity / precision / clock (even a fixed 24-policy grid) massively beats
   the reactive baseline at this load, while cutting SLA violations ~4x.
2. **Searching a fixed grid harder is saturated.** Exhaustive enumeration of the safe
   fixed grid (4.84x) and the bounded beam (4.85x) land on the same result; the bounded
   search already extracts what the grid contains, at a fraction of the evaluations.
3. **The remaining near-doubling comes from representation.** The hierarchical
   coupled-policy search (8.29x) reaches joint policies outside any fixed grid. The gap
   between 4.85x and 8.29x is value in futures the grid-based searches never
   represented, which is the memo's central claim measured at headline load.
4. **The single-surface arm cannot even be scored at this load** within the standard
   harness budget, because the fleet state its policy produces is itself intractable to
   replay.

## Scope

Simulated, load-dependent, selected public-trace windows; a mechanism attribution inside
the published benchmark's own harness. Not production evidence, not a savings estimate,
and it does not validate the simulator. The V1 frozen audit (zero measured search regret
where enumeration was tractable) remains the only regret measurement; the uncapped
`exhaustive_default4` arm enumerates the fixed grid only, not the hierarchical space.
