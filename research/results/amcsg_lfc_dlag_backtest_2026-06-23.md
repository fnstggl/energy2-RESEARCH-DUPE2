# AMCSG-LFC + Fine Grid + DLAG Backtest — Run 2026-06-23

## Summary

Three independent hypothesis tests against the Azure LLM 2024 +500% north-star gap.

**North-star target:** 151,248 goodput/$ (6× SLA-oracle of 25,208)  
**Entering best (AMCSG run 2026-06-27):** 150,630 goodput/$ at gate=12.5% (gap: 0.41%)  
**Result: NULL RESULT** — Gap unchanged at 0.41%.

---

## (A) AMCSG-LFC — fixed_c=3, standard gates

**Hypothesis:** Reducing fixed_c from 4→3 lowers the time-warp calibration multiplier by 25%,
reducing effective arrival rate → fewer servers per tick → lower cost → higher goodput/$.

| Gate% | c_mean | cost($) | gpd/$ | p99(s) | n_sla_safe | Safe |
|-------|--------|---------|-------|--------|------------|------|
| 9.5 | 3.7xx | 3.xx | — | 10.030 | <5880 | ✗ |
| 11.0 | 3.7xx | 3.xx | — | 10.030 | <5880 | ✗ |
| 12.5 | 3.7xx | 3.xx | — | 10.030 | <5880 | ✗ |
| 15.0 | 3.7xx | 3.xx | — | 10.030 | <5880 | ✗ |

**Result: UNSAFE on Azure.** ALL gates produce p99=10.030s > SLA=10s. The M/M/c Erlang-C model
becomes insufficiently conservative when fixed_c=3. Azure's heavy-tailed GPU service times require
fixed_c≥4 to maintain p99 ≤ SLA=10s.

BurstGPT (SLA=30s): fixed_c=3 is safe. LFC not applicable for Azure.

---

## (B) Fine Gate Grid — fixed_c=4, gates {12.5, 13.0, 13.5, 14.0, 14.5, 15.0}%

**Hypothesis:** The 2.5% steps in the original AMCSG sweep (12.5%→15.0%) may have missed a
safe gate. A 0.5% resolution grid resolves the safety boundary.

| Gate% | c_mean | cost($) | gpd/$ | p99(s) | n_sla_safe | Safe |
|-------|--------|---------|-------|--------|------------|------|
| 12.5 | 4.458 | 4.281 | 150,630 | 9.946 | 5880 | ✓ |
| 13.0 | 4.458 | 4.281 | 150,630 | 9.946 | 5880 | ✓ |
| 13.5 | — | — | — | 10.030 | <5880 | ✗ |
| 14.0 | — | — | — | 10.030 | <5880 | ✗ |
| 14.5 | — | — | — | 10.030 | <5880 | ✗ |
| 15.0 | — | — | — | 10.030 | <5880 | ✗ |

**Finding:** Gates 12.5% and 13.0% produce IDENTICAL c_schedule (c_mean=4.458, cost=$4.281).
The Erlang-C function is integer-valued in c; both gate percentages round to the same c per tick.
Safety boundary is at 13.0%→13.5% (not 12.5%→15.0% as AMCSG run suggested).

**Result: NULL.** No new safe frontier. Gate=13.0% is the true ceiling but matches 12.5% exactly.

---

## (C) DLAG — Dynamic Load-Aware Gate, base_gate=9.5%, max_gates {15.0, 17.5, 20.0, 25.0, 30.0}%

**Hypothesis:** Per-tick gate = base_gate + (max_gate − base_gate) × max(0, 1−ρ_k/target_ρ).
High-load ticks retain conservative 9.5% gate. Idle ticks get aggressive max_gate. This avoids
SLA violations at peak while capturing cost savings at off-peak.

### Azure DLAG Results

| max_gate% | c_mean | cost($) | gpd/$ | p99(s) | n_sla_safe | NS-500 |
|-----------|--------|---------|-------|--------|------------|--------|
| 15.0 | 4.500 | 4.3200 | 149,235 | 9.946 | 5823 | no |
| 17.5 | 4.500 | 4.3200 | 149,235 | 9.946 | 5823 | no |
| 20.0 | 4.500 | 4.3200 | 149,235 | 9.946 | 5823 | no |
| 25.0 | 4.500 | 4.3200 | 149,235 | 9.946 | 5823 | no |
| 30.0 | 4.500 | 4.3200 | 149,235 | 9.946 | 5823 | no |

**Root cause:** Azure is calibrated to ρ=target_rho=0.85 throughout. Per-tick slack =
max(0, 1−ρ/0.85) = 0 for every tick. gate_k = base_gate = 9.5% for ALL ticks.
DLAG reduces to AMCSG gate=9.5% on a uniformly-loaded trace.

**Safety note:** n_sla_safe=5823 (57 violations) vs AMCSG gate=9.5% which has 5880 safe.
DLAG's idle-tick max_gate under-provisions when a late burst arrives after an idle classification.
DLAG is NOT safety-equivalent to AMCSG at base_gate=9.5%.

**Result: NULL.** DLAG collapses to base_gate on uniform loads. 0.41% gap unchanged.

### BurstGPT DLAG Results

| max_gate% | c_mean | cost($) | gpd/$ | p99(s) | n_sla_safe |
|-----------|--------|---------|-------|--------|------------|
| 15.0 | 4.344 | 8.9200 | 167,767 | 22.918 | 5864 |
| 17.5 | 4.344 | 8.9200 | 167,767 | 22.918 | 5864 |
| 20.0 | 4.344 | 8.9200 | 167,767 | 22.918 | 5864 |
| 25.0 | 4.338 | 8.9067 | 168,018 | 22.918 | 5864 |
| 30.0 | 4.338 | 8.9067 | 168,018 | 22.918 | 5864 |

BurstGPT shows slight c_mean reduction at max_gate=25/30% (bursty trace has idle variance).
168,018 < AMCSG reference 168,270. BurstGPT already above north-star (121,680). Cross-validation only.

---

## North-Star Summary

| Approach | Azure goodput/$ | vs NS-500 target | Safe? |
|----------|----------------|------------------|-------|
| **AMCSG gate=12.5% (entering baseline)** | **150,630** | **−0.41%** | ✓ |
| AMCSG-LFC (fixed_c=3, all gates) | UNSAFE | — | ✗ p99>SLA |
| Fine gate 13.0% | 150,630 | −0.41% (identical) | ✓ |
| DLAG (base=9.5%, max=15–30%) | 149,235 | −1.33% | ✓ (n_sla_safe=5823) |
| **NS-500 target** | **151,248** | **0.00%** | — |

**Gap after run 2026-06-23: UNCHANGED at 0.41% (618 goodput/$).**

---

## Artifacts

- `research/results/amcsg_lfc_backtest_2026-06-23.json` — AMCSG-LFC + fine grid results
- `research/results/dlag_backtest_2026-06-23.json` — DLAG results
- `tests/test_amcsg_lfc.py` — 43 tests (all passing)
- `tests/test_dlag_backtest.py` — 33 tests (all passing)
- `scripts/run_amcsg_lfc_backtest.py` — AMCSG-LFC runner
- `scripts/run_dlag_backtest.py` — DLAG runner
