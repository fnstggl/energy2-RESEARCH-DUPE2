# dt=60 Prefill/Decode Economics Diagnostic (PR #107, Phase 9)

Does prefill/decode disaggregation + service-time-sensitive cost convert PR #106's KV prefill savings
into goodput/$? Bounded: dt=60, 6-hour Azure window, 360 decisions, Mooncake-derived prefixes, unchanged
Pareto gate. Static policies isolate the physics. `data/external/mpc_controller/prefill_decode_dt60.json`.

## Result

| config | cost mode | gp/$ | SLA | TTFT p95 | realized GPU-s | KV hit | gate (beats/pareto/headline) |
|--|--|--|--|--|--|--|--|
| fair_oldmodel_no_cache | (old constant prefill) | 65 089 | 0.0190 | – | – | – | reference only |
| **fair_phase_no_cache** | hybrid | 112 541 | 0.0333 | 0.893 | 14 789 | 0.000 | **the fair baseline** |
| legacy_kv_scalar_optimistic | (offline scalar) | 69 211 | 0.0124 | – | – | – | **unsafe** (credits cold reqs) |
| residency_provisioned | provisioned | 56 739 | 0.0333 | 0.891 | 14 773 | 0.999 | −49.6 % · F/T/F |
| **residency_hybrid** | hybrid | 112 578 | 0.0333 | 0.891 | 14 773 | 0.999 | **+0.03 % · T/T/T** |
| residency_realized | realized | 151 055 | 0.0333 | 0.891 | 14 773 | 0.999 | +34.2 % · T/T/F (upper bound) |

## Headline (honest)

**The bridge is built and, under the defensible hybrid cost mode against an apples-to-apples fair
baseline, the residency channel is Pareto-safe — the first gate `True` in this series — but the Azure
gain is marginal (+0.03 %).** The economics now respond to realized serving work (provisioned 56 739 →
hybrid 112 578 → realized 151 055), confirming the Cause-B fix. The Pareto gate passes because the fair
baseline now **also** pays realistic prefill (so SLA is not worse: 0.0333 = 0.0333) — fixing the
baseline-fairness flaw where a constant-prefill baseline made the realistic candidate look worse.

**Why marginal:** Azure is **decode-bound**. KV reuse saved **107 554 prefill tokens**, but prefill is a
tiny share of realized serving work, so realized GPU-seconds fell only **16 of 14 789 (0.1 %)** → a 0.03 %
cost win. The **realized-work upper bound** (+34.2 %) shows the full potential when cost follows serving
work, but it is a **counterfactual, not a production claim** (gate `headline=False` by the cost-mode
guard). A prefill-heavy workload monetizes far more (`test_prefill_heavy_benefits_more_from_reuse`:
>30 % realized-work reduction).

## Required interpretation

1. **Did disaggregation change TTFT?** Yes — TTFT is now explicit and prompt-driven (p95 ≈ 0.89 s,
   realistic) instead of a constant 0.15 s; a KV hit lowers it (fixtures).
2. **Did it change completion latency?** Marginally — completion stays decode-bound (long Azure outputs),
   so KV reuse barely moves it.
3. **Did it change batching/concurrency decisions?** The model makes batching decode-phase-specific; on
   this static run batching was fixed (balanced) — the MPC-search interaction is future work.
4. **Did it convert KV savings into lower realized GPU-seconds?** Yes but tiny on Azure (16/14 789, 0.1 %)
   — decode-bound; large on prefill-heavy fixtures.
5. **Into better provisioned-capacity gp/$?** No — provisioned cost is floor-bound (reproduces #106).
6. **Did hybrid cost produce a Pareto-safe result?** **Yes** (+0.03 %, gate `True/True/True`) — marginal
   but real and safe.
7. **Real work reduction or SLA-shedding?** Real work reduction — SLA is **not worse** (0.0333 = fair),
   the win is the 16 GPU-seconds of avoided prefill, not shed SLA.
8–10. **prewarm / migration / placement valuable?** Not exercised to a win here (static routing-only run);
   the channel they need (realized-work cost) now exists, but on decode-bound Azure their headroom is
   small. Prefill-heavy regimes are where they would pay.
11. **Which physical channel is still missing?** **Prefill/decode DISAGGREGATION as capacity** — separate
   prefill vs decode worker pools so a prefill saving *frees decode capacity* (DistServe/Splitwise),
   converting TTFT wins into completion/goodput wins on decode-bound loads. Today prefill and decode share
   one service number; the *work* is split but the *capacity* is not. That, plus prefill-heavy workloads
   (long-context/RAG/agent), is the next increment.

## Claim safety
- **Safe:** the cost bridge is built and validated; under hybrid + a fair realistic-prefill baseline the
  residency channel is Pareto-safe (+0.03 %); the realistic prompt-driven prefill is a fidelity gain;
  fixtures prove prefill-heavy loads monetize >30 %.
- **Upper-bound (not production):** the realized-work +34.2 % — labelled counterfactual, gate-blocked.
- **Unsafe (never claimed):** the legacy optimistic scalar's 69 211 (credits reuse to cold requests);
  any large Pareto-safe Azure win (the decode-bound magnitude is 0.03 %).
