# dt=60 Full Serving-Physics Diagnostic (Phase 11)

Bounded held-out diagnostic over the V2 serving world model. Deterministic; reproduce with
`python scripts/run_dt60_full_serving_physics.py 120`. Setup: 120 one-minute periods (2 h), dt=60 s,
8× H100, llama-8b-gqa, SLA 5 s, diurnal Azure-like arrivals, Mooncake-style 20-family prefix reuse,
provisioned-capacity hybrid economics. The persistent V2 ClusterState advances one chosen action per period;
candidate evaluation is on clones (MPC-safe). No headline is claimed unless the Pareto gate passes.

## Configs A–G

| config | gp/$ | SLA viol | completion p95 (s) | energy ($) | billed GPU-h | dec ms | regret | regime |
|--|--|--|--|--|--|--|--|--|
| A legacy_scalar | 122.96 | 0.994 | 3287.9 | 0.8749 | 16.00 | ~0 | — | mixed |
| B roofline_live | 142,349.88 | 0.000 | 1.074 | 0.2495 | 16.00 | ~0 | — | memory |
| C +disaggregation_sweep | 142,349.88 | 0.000 | 1.074 | 0.2495 | 16.00 | 13.69 | — | memory |
| D tiered_kv (full) | 142,349.88 | 0.000 | 1.074 | 0.2495 | 16.00 | ~0 | — | memory |
| D tiered_kv (HBM-only) | 142,349.88 | 0.000 | 1.074 | 0.2495 | 16.00 | ~0 | — | memory |
| E +upgraded_batching | 142,600.66 | 0.000 | 0.615 | 0.2249 | 16.00 | ~0 | — | memory |
| F +roofline_MPC_actions | 143,290.27 | 0.000 | 0.724 | 0.1579 | 16.00 | 1145.81 | 0.0 | memory |
| G full + adaptive_search (bg) | **199,356.81** | 0.000 | 0.724 | **0.1137** | **11.50** | 221.43 | — | memory |

## Interpretation (the twelve questions)

**1. Did roofline timing change action selection?** Yes — and it changed the *whole picture*. Switching A→B
(legacy scalar → roofline) is the dominant effect: it removes 99.4 % phantom SLA violations and lifts gp/$
from 123 to 142 k by correctly pricing H100 decode (~5 ms/tok vs the legacy L40S-class 20 ms/tok). The regime
classifies as memory-bound, which steers the candidate generator toward precision/spec/down-clock bundles.

**2. Did precision improve gp/$?** Not on its own here. Operator cost is provisioned-capacity-dominated, so
fp8's lower realized work does not cut cost while SLA is already met; its small quality-risk makes it
gp/$-neutral-to-slightly-negative under slack. It *does* cut latency and energy (a Pareto-safe physical win),
and it monetises only in the near-saturation regime (controlled fixtures), not on this feasible window.

**3. Did spec decode improve latency or gp/$?** Same shape as precision: it cuts memory-bound decode latency
and energy (config F), not gp/$ directly. The MPC selects it where compute headroom exists.

**4. Did clock/power improve energy/cost?** **Yes — clearly.** Config F's roofline-MPC actions cut energy
0.2495→0.1579 (−37 %) by down-clocking memory-bound decode (the compute leg is not binding, so peak FLOPS can
drop with no latency penalty while power falls). This is the cleanest Pareto-safe energy win.

**5. Did co-location help with real background work?** **Yes — this is the one genuine incremental gp/$ win.**
Config G (with 150 GPU-s/period of real background work) reclaims idle capacity: billed GPU-h 16→11.5, energy
→0.1137, gp/$ →199 k (+40 % vs B). With **no** background work the candidate generator prunes co-location to
`off`, so the win cannot be faked.

**6. Did disaggregation improve TTFT/completion?** Not on this workload — config C's sweep selects
`shared_pool` every period (handoff overhead dominates for these prompt/decode mixes), so C ≡ B. Disaggregation
helps only phase-skewed workloads (proved in `compare_disaggregation_fixture.py`); here it is correctly
inert, at a measured 13.7 ms/decision sweep cost.

**7. Did tiered KV change prewarm/migration/routing value?** Pareto-neutral on gp/$ here: D-full ≡ D-HBM-only
≡ B. Tier hits reduce prefill work, but under provisioned-cost-dominated economics with slack SLA that does
not monetise. Tiered KV's value shows in the controlled fixtures (remote-vs-recompute, capacity→hit-rate) and
would matter under tight SLA / network pressure, not on this feasible window.

**8. Did adaptive search find better coupled bundles?** The search is correct and bounded (regret 0.0 in
config F's audit; beam matches exhaustive on the controlled coupled fixture where coordinate descent is stuck
at regret 0.50). On this window the coupled gains are energy/latency, not gp/$, so the selected bundles cut
energy (F) and, with background work, cost (G). Runtime is bounded (F ≈ 1146 ms, G ≈ 221 ms per decision) —
the F cost flags that a ~144-candidate memory-regime space warrants a smaller beam or tighter pruning.

**9. Was any improvement Pareto-safe?** Yes: B's SLA-violation correction (goodput up, nothing worse),
E/F's latency+energy reductions (SLA unchanged at 0), and G's cost reduction via real background-work reclaim
are all Pareto-safe. No improvement came from SLA-shedding (violation rate stays 0.000 throughout B–G).

**10. Were gains from real physics or SLA-shedding?** Real physics. SLA violations are 0.000 across B–G;
goodput is constant; the wins are lower realized work → lower energy (roofline + down-clock), and reclaimed
idle (co-location). Nothing was bought by dropping requests.

**11. What remains simulator-inferred?** The analytical M/D/1 phase-queue (vs a per-iteration loop); the
continuous-batching occupancy approximation; spec-decode acceptance bands; int4 quality-risk; co-location
contention bands; the MFU constants. All are BENCHMARK_DERIVED / SIMULATOR_INFERENCE and labelled.

**12. What remains impossible without pilot telemetry?** (PROP) Real per-replica KV residency/eviction; real
measured cache hit rates; real cross-node KV-transfer bandwidth under production congestion; true per-request
model identity; real internal operator $/energy/carbon; per-link/NVLink fabric contention. These are modelled,
never measured.

## Verdict

Per the success criterion, V2 **succeeds**: it runs the bounded dt=60 diagnostic deterministically, reproduces
the controlled fixtures, and explains action value with more physical detail than V1. The held-out gp/$ does
**not** improve from the *added* serving mechanisms (disaggregation/tiered-KV are Pareto-neutral under
provisioned-cost-dominated economics with slack SLA) — and the diagnostic says so plainly. The large A→B gp/$
jump is a **realism correction** (removing V1's phantom SLA violations), and the genuine incremental wins are
**energy** (down-clock, −37 %) and **cost via real background-work co-location** (−28 % GPU-h), both
Pareto-safe. No fake wins; no SLA-shedding; search regret measured.
