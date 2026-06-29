# Production Scheduler Baseline — Results (Phase E/H)

`production_scheduler` is the single canonical, realistic, modern GPU-fleet scheduler baseline future headlines
compare against. This reports what it is, how it scores on the benchmark ladder, and — honestly — whether
Aurelius MPC beats it. Evidence: `data/external/mpc_controller/ladder_benchmark.json` (3 markets × expensive
window, the bounded Phase E run; 24/24 cells COMPLETED, 0 timeout, 0 failed). Magnitudes are **SIMULATED**
directional simulator evidence, not production telemetry. No tuning to the benchmark; the Pareto gate is
unchanged.

## 1. What is `production_scheduler`?

A deterministic, causal heuristic `decide_fn(history)→action` in the **evaluation layer**
(`aurelius/environment/production_baselines.py`) — a realistic vLLM/TGI-class scheduler that reacts to recent
observable load. It runs through the unchanged reward path (`run_period_episode`) exactly like `fifo`/`sla_aware`.
It is **not** a planner mode, shares **no** MPC-search / economic / oracle / hierarchical code (AST-enforced by
`tests/test_production_scheduler_baseline.py`), and is a **separate ladder arm** from the Aurelius MPC arms.

## 2. What levers does it use (and not use)?

**Uses** (the serving-stack levers a real deployment has): SLA-aware ordering (`abs_conformal`), backlog
autoscaling (`backlog_aware`) with a 1.25× safety headroom under pressure, class-aware admission under
pressure, **continuous batching always on** (balanced → aggressive when decode-heavy), KV-aware routing,
rack-local placement, and the warm pool via `backlog_aware`'s idle timeout. **Does NOT use** (Aurelius's edge):
model-precision arbitrage (bf16 only), DVFS/clock arbitrage (base only), MPC-planned migration (off),
speculative decoding (off), future electricity prices, oracle workload, the global economic objective, or any
search. Confirmed by the mixes: every period ran `precision=bf16, clock=base, migration=off, spec=off`.

## 3. How does it rank against the other baselines?

Per-market gp/$ (SLA-violation rate). production_scheduler has the **best SLA of every baseline** on all 3
markets, and Pareto-dominates `sla_aware`:

| arm | pjm gp/$ (SLA) | ercot gp/$ (SLA) | caiso gp/$ (SLA) |
|--|--|--|--|
| fifo (weak) | 292,966 (0.464) | 282,901 (0.405) | 295,935 (0.405) |
| topology_aware | 292,966 (0.464) | 282,901 (0.405) | 295,935 (0.405) |
| sla_aware | 295,338 (0.226) | 293,601 (0.143) | 288,383 (0.149) |
| vllm_only | 411,732 (0.095) | 375,779 (0.089) | 391,159 (0.065) |
| **production_scheduler** | **330,711 (0.065)** | **303,982 (0.071)** | **301,228 (0.065)** |

## 4. Is it stronger than `sla_aware` (the design intent)?

**Yes — it Pareto-dominates `sla_aware` on every market**: higher gp/$ AND lower SLA-violation rate.
pjm +35,373 gp/$ (**+12.0%**) at SLA 0.065 vs 0.226; ercot +10,381 (+3.5%) at 0.071 vs 0.143; caiso +12,845
(+4.5%) at 0.065 vs 0.149. The serving-stack levers (KV routing, rack placement, class admission, headroom)
buy both more goodput/$ and better deadline compliance than the bare SRPT-conformal scheduler.

## 5. Honest caveat: `vllm_only` has higher raw gp/$

`vllm_only` posts a higher gp/$ than production_scheduler (pjm 411,732 vs 330,711) but at a **worse** SLA
(0.095 vs 0.065). Neither Pareto-dominates the other — they sit at different points on the cost/SLA curve.
production_scheduler trades a little gp/$ for materially better deadline compliance via the 1.25× headroom +
class admission — a realistic ops operating point. The headline (next) beats **both**, so the choice of which
is "strongest" does not change the conclusion.

## 6. Does Aurelius MPC beat production_scheduler — the headline?

**Yes, Pareto-dominantly, on every market** (`aurelius_mpc_hierarchical_search`, the default planner, vs
`production_scheduler`), with **both** absolute and percent deltas, SLA never worse:

| market | hierarchical gp/$ | production gp/$ | abs Δ | pct Δ | SLA (hier vs prod) |
|--|--|--|--|--|--|
| pjm | 783,862 | 330,711 | **+453,151** | **+137.0%** | 0.000 vs 0.065 ✓ |
| ercot | 785,952 | 303,982 | **+481,969** | **+158.6%** | 0.000 vs 0.071 ✓ |
| caiso | 747,580 | 301,228 | **+446,352** | **+148.2%** | 0.000 vs 0.065 ✓ |

The prior default planner (`aurelius_mpc_current_default`, physics-guided beam) also beats it Pareto-safely
(pjm +354,403 / +107.2%; ercot +443,874 / +146.0%; caiso +342,001 / +113.5%). Both Aurelius arms also beat
`vllm_only` on gp/$ AND SLA, so Aurelius Pareto-dominates the **entire** ladder.

## 7. Does the win pass the Pareto gate?

**Yes.** Every headline cell has gp/$ strictly higher AND SLA-violation rate no worse (0.0 ≤ production's
~0.07). No deadline cheating, no free capacity (every replica/warm-hold GPU-hour is charged), no oracle data
(the oracle is a separate diagnostic arm), no quality-risked lever (`quality_sla_risk_mean=0.0` — fp8 is
lossless-safe, int4 excluded). The gate is honestly satisfied, not chased.

## 8. Where does Aurelius's edge over production_scheduler come from?

The **economic arbitrage production_scheduler is defined not to use**: the hierarchical winner ran
`precision=fp8` (lossless-safe), `clock=high`, `spec_decode=aggressive`, `batching=aggressive`, and **capacity
consolidation** (`capacity_multiplier=0.75` vs production's 1.25 headroom) — plus a marginally better connected
placement (`network_aware` vs `rack_local`). Routing was `kv_aware` for **both** (no edge there). Full
decomposition + fidelity labels: `research/CONNECTED_SURFACE_VALUE_ATTRIBUTION.md`. The edge is exactly
Aurelius's whole point — optimising the deployed stack's economics — not a connected-surface trick the baseline
lacks.

## 9. Two realism corrections found during integration (both make the bar *stronger*)

Surface-isolation during integration exposed two ways the first draft was **unrealistic**, each fixed toward
realism (the honest direction — a stronger bar), neither tuned to the benchmark:
1. **Continuous batching is always on.** The draft shrank the batch under burst; turning off continuous
   batching is something no real vLLM/TGI deployment does (it only raises cost/req). Bursts are handled by
   admission + headroom. Fix: balanced → aggressive, never conservative.
2. **No eager prewarm pool.** An eager warm pool spun replicas up ahead of demand; at backtest workload scale
   (req_cap 56) the warm-hold cost **dwarfed** the served work (warm-hold 1.0 GPU-h vs 0.033 GPU-h served)
   with **zero cold starts avoided** — a sub-scale artifact a cost-conscious operator would not pay. Fix: rely
   on `backlog_aware`'s idle-timeout warm pool. (Without the fix production_scheduler's gp/$ was ~10k — an
   obvious strawman; the user's "do not chase a positive result" is exactly why this was hunted down and fixed
   *against* an easy Aurelius win.)

## 10. What is the oracle, and is it a fair comparison?

`oracle_diagnostic` plans the strongest search against the **exact future** workload of each served period — a
**non-deployable upper bound**, never a headline. It is a separate arm. The Aurelius default sits within
**0.9–2.4%** of it (pjm 783,862 vs 803,030; ercot 785,952 vs 797,483; caiso 747,580 vs 754,189), i.e. the
forecast/search leaves very little on the table in these windows. (Its sub-second runtime is expected: planning
a single known-future scenario is far cheaper than the forecast uncertainty ensemble.)

## 11. What is the fidelity of this result?

**SIMULATED / SIMULATOR_INFERENCE.** The reward path is the repo's world simulator; the dominant edge levers
(fp8 / clock / spec roofline economics) are SIMULATOR_INFERENCE in magnitude (robust in direction —
`research/WORLD_MODEL_ROBUSTNESS_AUDIT.md`). The best-calibrated lever (kv-aware routing, TRACE_DERIVED from
Mooncake, prefix-hit 0.952) is **shared** with production_scheduler, so it is not the source of the edge. The
run is bounded: 3 markets, expensive window only, 3 decisions/window, req_cap 56. The **direction** (Aurelius's
economic optimisation beats a strong production scheduler) is robust; the exact +137–159% is not
production-validated.

## 12. What would strengthen or overturn this?

Strengthen: pilot telemetry on fp8/clock/spec throughput (converts the dominant levers from inferred to
measured); larger-scale workloads (re-prices the warm pool / capacity levers); the cheap/volatile windows and
the other markets (the `--full` mode of `scripts/run_ladder_benchmark.py`, resumable). Overturn: if real fp8/
clock/spec economics are materially smaller than the roofline model assumes, the magnitude shrinks — but
production_scheduler already runs those levers off, so the *sign* of the comparison would persist unless the
levers are net-negative in reality. We report the bounded result honestly and do not extrapolate the magnitude.
