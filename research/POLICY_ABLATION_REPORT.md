# Policy Ablation Report (Phase 4 — Discovery Only)

> Discovery run. No behavior changed. Each policy is measured **disabled vs
> enabled** against the strongest *safe* baseline (not FIFO-only), using existing
> benchmarks re-run on current main (`353efd9`). Directional simulator only.

## Ablation method
For each policy: **Current Main = Policy Enabled** (the policies are on by
construction where implemented). "Disabled" = the strongest safe non-oracle
baseline for that workload class. Oracle/clairvoyant configs are reported as a
ceiling only, never as the comparator (`docs/RESULTS.md §3`).

---

## A. `energy` (EnergySchedulingPolicy) — canonical 1000-job energy backtest

| Condition | gpd/$ | deadline misses | infra $ |
|---|---:|---:|---:|
| Disabled → `current_price_only` (strongest safe) | 0.303676 | 0 | 57,453 |
| Disabled → FIFO (sanity) | 0.165781 | 0 | 105,241 |
| **Enabled (`energy`/CA) = Current Main** | **0.337299** | **0** | 51,726 |

- SLA-safe goodput/$: **+11.1%** vs strongest safe baseline (+103% vs FIFO sanity).
- Cost: −10.0% vs `current_price_only`, −50.9% vs FIFO. GPU-hours/migrations folded into infra $.
- SLA violations / deadline misses: **0** (the unsafe baselines `greedy_energy`/`sla_aware`/`robust_energy` incur 119–143 misses and are excluded).
- **Value: HIGH.** Real optimizer intelligence (temporal shift + regional arbitrage + throttling) beats the strongest safe baseline at zero deadline misses. This is the cleanest validated value in the optimizer.

## B. `serving_queue` (abs-conformal SRPT) — Azure 2024 / BurstGPT HF, fixed c=4

| Condition (Azure) | gpd/$ | vs FIFO | vs SLA-oracle |
|---|---:|---:|---:|
| Disabled → FIFO (sanity) | 13,336 | 0% | −56% |
| Disabled → SLA-aware oracle (headline base) | 25,208¹ | — | 0% |
| rel-conformal | 45,933 | +244% | +53% |
| **Enabled (abs-conformal) = Current Main** | **55,097** | **+313%** | **+83%** |
| oracle-conformal (ceiling) | 56,311 | +322% | +87% |

BurstGPT HF: FIFO 6,529 → abs-conformal 42,902 (**+557% vs FIFO**; +88.3% oracle retention).
¹ SLA-oracle on the joint-run cost basis; ratios are basis-invariant.

- **Value: HIGH *but regime-dependent*.** At fixed/overloaded capacity the
  ordering lever is large (+83% vs the honest SLA-oracle comparator, ~97.8%/88.3%
  of the FIFO→oracle gap). **However, under realistic provisioning (MCS) the value
  collapses to ≈0 / slightly negative** (see `POLICY_INTERACTION_ANALYSIS.md`:
  abs+MCS +131% < FIFO+MCS +137%). Classification: **HIGH in fixed-capacity
  overloaded regime; NEUTRAL-to-slightly-HARMFUL once capacity scales.**
- No actual-output-token leakage at decision time (test-enforced); abs-conformal
  beats rel-conformal because absolute error ignores scheduling-irrelevant
  short-request mispredictions (calibration, not a new prior).

## C. `placement` (shadow `GpuPlacementScorer`) — gpu_routing ablation (1000 jobs)

| Metric | Disabled (baseline) | Enabled (scorer) | Δ |
|---|---:|---:|---:|
| latency_critical goodput/$ (real KPI) | 0.37187 | 0.344659 | **−7.3%** |
| overall goodput/$ | 0.300667 | 0.300246 | −0.14% |
| % lc on best GPU (H100) — **proxy** | 3.3% | 58.0% | **+54.7 pp** |
| mean GPU penalty — **proxy** | 0.4295 | 0.1505 | −0.279 |
| realized energy cost $ | 14,561 | 14,672 | +$111 |

- The scorer **moves its proxy strongly** (+54.7 pp on-best-GPU) but **regresses
  the real KPI** (−7.3% latency_critical goodput/$) and costs slightly more energy.
- **Value: HARMFUL.** `enabled=False` by default; correctly shadow-only. (This is
  the diagnostic behind the unimplemented `placement` policy.)

## D. `admission` (shadow `WorkloadAdmissionGate`)
- Prior module-integration result (`BENCHMARK_REGISTRY.md §5b`): **NEUTRAL**
  (BurstGPT ±0.34%). `enabled=False`. Not implemented as an AO policy.
- **Value: NEUTRAL.**

## E. `replica_scaling` (un-routed provisioning: SHU / MCS)
Not an AO policy (stub), but the provisioning decision it will host is **the
single largest serving goodput/$ lever measured**:
- FIFO+MCS = +137% vs SLA-oracle (vs +83% for ordering alone) — provisioning >
  ordering.
- MCS vs SHU (the "current Aurelius headline" provisioning policy): mostly **TIE**,
  with +24.5%/+2.6% only at extreme (500×) scales.
- Caveat (honesty): MCS **raises** GPU-hours +12.5% on the diurnal Azure trace —
  its goodput gain comes from SLA compliance under higher capacity, not cost savings.
- **Value: HIGH (provisioning), but lives outside `AureliusOptimizer`.**

---

## Value classification summary

| Policy | In AO? | SLA-safe goodput/$ impact | Class |
|---|---|---|---|
| `energy` | ✅ active | +11.1% vs strongest safe baseline @ 0 misses | **HIGH** |
| `serving_queue` | ✅ active | +83% vs SLA-oracle (fixed-c); ≈0/−6pp under provisioning | **HIGH (fixed-capacity) / NEUTRAL (provisioned)** |
| `replica_scaling` (SHU/MCS, un-routed) | ❌ stub | +137% vs SLA-oracle (provisioning) | **HIGH — but not an AO policy** |
| `admission` (shadow) | ❌ stub | ±0.34% | **NEUTRAL** |
| `placement` (shadow scorer) | ❌ stub | −7.3% lc goodput/$ | **HARMFUL** |
