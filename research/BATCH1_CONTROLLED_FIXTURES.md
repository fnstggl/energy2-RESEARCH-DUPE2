# Batch-1 controlled fixtures (Phase 6)

Twelve controlled fixtures exercise each new knob through its causal simulator path and show it **helping**,
**hurting**, and being **neutral**. Runner: `scripts/run_batch1_fixtures.py`; artifact:
`research/results/batch1_controlled_fixtures.json`. The reward proxy mirrors the production
`PeriodOutcome.goodput_per_dollar` exactly — `gp/$ = sla_safe_goodput·(1−quality_risk) / operator_cost`,
SLA-safe iff completion ≤ target — so a fixture win is the same *kind* of win the benchmark scores
(latency→SLA→goodput, GPU-seconds/energy→cost), never a bonus. KV precision + PD run through the roofline
serving point; GPU assignment runs through the heterogeneous-fleet model.

`gp% Δ = N/A` where the baseline goodput is 0 (the baseline violates the SLA): the win is then a goodput
*recovery* — reported as the absolute gp/$ and the SLA/latency delta. **All 12 pass.**

| # | fixture | knob | baseline → selected | gp/$ result | SLA Δ | latency Δ (s) | fidelity | pass |
|--|--|--|--|--|--|--|--|--|
| 1 | KV precision **helps** HBM-bound decode | KV precision | neutral → kv_fp8 | **+19.5 %** | 0 | faster | SIM_INFER (fp8 KV ≈ lossless, PUBLIC_BENCH) | ✅ |
| 2 | KV precision **neutral** when not memory-bound | KV precision | neutral → kv_fp8 | +0.28 % (≈0) | 0 | ≈0 | SIM_INFER (regime classifier) | ✅ |
| 3 | KV precision **unsafe** (int4) excluded from headline | KV precision | neutral → kv_int4 | quality_risk>0 → **not headline-safe** | — | — | SIM_INFER; quality UNMODELLED | ✅ |
| 4 | Heterogeneous assignment **helps** latency-sensitive (tight SLA) | GPU assign | homogeneous → fastest_for_latency | **+24.3 %** | −0.67 (fewer viol) | — | SIM_INFER; **NOT_APPLICABLE to prod** | ✅ |
| 5 | Heterogeneous assignment **helps** cost batch (slack SLA) | GPU assign | homogeneous(H100) → cheapest_for_batch | **+61.8 %** | 0 | — | SIM_INFER; NOT_APPLICABLE to prod | ✅ |
| 6 | **Wrong** GPU assignment hurts | GPU assign | correct(fastest) → wrong(cheap-on-tight-SLA) | 12.75M → 10.25M, **SLA viol 0.67** | worse | — | SIM_INFER; NOT_APPLICABLE to prod | ✅ |
| 7 | Homogeneous fleet → **no fake benefit** | GPU assign | any (all tie) | single gp/$ 10.21M (all deployable policies tie) | 0 | 0 | STRUCTURAL guarantee | ✅ |
| 8 | Prefill-heavy **benefits** prefill-heavy split | PD disagg | shared → prefill_heavy | shared violates → split 67,211 (goodput recovered) | −1.0 | **−0.069** | SIM_INFER (phase-pool M/M/c) | ✅ |
| 9 | Decode-heavy **benefits** decode-heavy split | PD disagg | shared → decode_heavy | shared violates → split 631,233 (goodput recovered) | −1.0 | **−2.79** | SIM_INFER (phase-pool M/M/c) | ✅ |
| 10 | Mixed workload **prefers shared** pool | PD disagg | split → shared | shared 231,558 ≥ split 0 | — | — | SIM_INFER (statistical multiplexing) | ✅ |
| 11 | Handoff overhead **erases** PD gains (light, huge ctx) | PD disagg | split → shared | shared 9,730 = split 9,730 (handoff 10.7 ms erases) | — | — | SIM_INFER (KV handoff bytes/latency) | ✅ |
| 12 | All knobs **interact** under memory+queue pressure | KV+weight precision (PD/assign regime-gated) | neutral → fp8 weights + kv_fp8 + balanced | **+84.0 %** | 0 | faster | SIM_INFER (roofline); assign NOT_APPLICABLE | ✅ |

## Reading the results

- **KV precision (1–3).** fp8 KV gives +19.5 % gp/$ on a long-context, memory-bandwidth-bound decode (KV
  bytes halve → decode tokens/s rise → less cost at equal SLA). On a tiny compute-bound workload the same knob
  is ~neutral (+0.28 %) — the roofline regime classifier correctly says KV bytes don't bind there. The int4-KV
  variant *appears* to win (+24 %) but carries an unmodelled quality risk → `headline_safe=False`; it is
  excluded from the headline planner (it only generates under `allow_quality_risk`).
- **Heterogeneous assignment (4–7).** Routing latency-sensitive work to a fast GPU under a tight SLA recovers
  the goodput a cheap-everywhere baseline loses (+24.3 %); routing slack batch work to a cheap GPU off an
  expensive-dominant fleet cuts cost (+61.8 %). The **wrong** assignment (latency work on a slow cheap GPU)
  hurts: gp/$ down and SLA-violation 0.67. On a **homogeneous** fleet every deployable policy returns the
  *identical* gp/$ — the structural no-fake-gain guarantee. These are fixture results only; the knob is
  **NOT_APPLICABLE** to the production benchmark.
- **PD disaggregation (8–11).** A prefill-heavy / decode-heavy workload under a tight SLA: the shared pool's
  prefill/decode interference makes it violate while the matched split (isolated phases) meets the SLA and
  recovers the goodput (−0.069 s and −2.79 s completion). A balanced workload **prefers shared** (statistical
  multiplexing beats two smaller pools + handoff). A light, huge-context workload shows the **KV handoff
  (10.7 ms) erasing** any split gain → shared wins. Disaggregation is never free.
- **Interaction (12).** Under combined memory + queue pressure, fp8 weights + fp8 KV + balanced batching
  compound to +84.0 % gp/$ at no SLA cost; GPU assignment stays NOT_APPLICABLE (homogeneous prod fleet) and is
  correctly absent.
