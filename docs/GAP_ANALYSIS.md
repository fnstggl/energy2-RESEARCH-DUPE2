# Forecasting Gap Analysis

## Summary

The oracle ceiling analysis (2026-05-22) revealed a 28–50pp gap between perfect-foresight (oracle) scheduling and the ML forecaster on the primary benchmark (`caiso_pjm_ercot_da_rt`). This document tracks the root causes and remediation progress.

---

## Oracle Ceiling Findings (2026-05-22)

| Dataset | Workload | Baseline (seasonal_naive) | Oracle Ceiling | ML (v5.0) | ML Gap vs Oracle |
|---------|----------|--------------------------|----------------|-----------|-----------------|
| caiso_pjm_ercot Q1 2026 | training | 3.3% | 37.7% | 9.003% | 28.7pp |
| caiso_pjm_ercot Q1 2026 | fine_tuning | 11.2% | 61.4% | ~11% | 50pp |
| summer2025_3region | training | 16.2% | 25.8% | — | ~9.6pp |
| summer2025_3region | fine_tuning | 28.8% | 39.5% | — | ~10.7pp |

**Key insight:** Winter ERCOT data drives the largest gap. Summer data is more predictable; seasonal_naive performs well there. The gap is dominated by the multi-region routing decision: "which region will be cheapest during the eval window?"

---

## Root Cause Decomposition

### 1. Multi-Region Routing (Primary Driver, ~60% of gap)
The oracle routes 100% of jobs to whichever region is cheapest for each eval window. The ML forecaster can't reliably predict which region will be cheapest 24-168h in advance.

**v5.0 attempt:** Price rank/percentile features (`price_momentum_168h`, `price_vs_lag168_abs`) improved cheap-regime detection but don't directly encode cross-regional comparison.

**v6.0 fix:** Cross-regional spread features (`cross_price_vs_min`, `cross_price_vs_mean`, `cross_is_cheapest` at current + 24h lag + 168h lag) directly encode "is this region cheap vs peers?" The BacktestEngine now passes full multi-region `recent_context` to `predict()`, enabling these features at inference time. Implementation validated: +3.9pp delta over v5 in current environment.

### 2. Price Spike Uncertainty (Secondary Driver, ~25% of gap)
Winter ERCOT experiences 10-100x price spikes (scarcity events). The p90 forecast significantly underestimates spike magnitude. The oracle avoids ERCOT during spikes by definition.

**Current p90 coverage:** 0.6433 (measured on caiso_pjm_ercot_da_rt). Target: ≥0.88.

**Planned fix:** Conformal prediction (arXiv:2502.04935) — add calibration layer on top of LightGBM p90 using a held-out calibration set. Expected to close the coverage gap without degrading p50 accuracy.

### 3. Temporal Horizon Uncertainty (Minor Driver, ~15% of gap)
Jobs with long runtimes (training, fine_tuning) need price predictions 24-168h ahead. Uncertainty compounds with horizon. The oracle has perfect 7-day foresight.

**Current approach:** Lag features (1h, 6h, 24h, 168h, 336h) provide seasonal context. Limited by horizon uncertainty.

**Planned fix:** Ensemble of short-horizon (24h) and long-horizon (168h) models weighted by job runtime.

---

## Progress Tracker

| Gap Component | Status | Improvement |
|---------------|--------|-------------|
| Multi-region routing signal | ✅ v6.0 shipped (2026-06-20) | +3.9pp delta over v5 |
| p90 coverage calibration | 🔲 Planned (conformal prediction) | 0.6433 → target 0.88 |
| ERCOT spike feature engineering | 🔲 Planned | TBD |
| Long-horizon ensemble | 🔲 Research | TBD |

---

## Methodology Notes

- All gap measurements use strict walk-forward backtesting (no data leakage)
- Oracle uses actual prices for the FULL eval window — not a real forecaster, only a ceiling
- ML forecaster is re-fit per fold using only training data
- `current_price_only` baseline: picks cheapest region at `job.earliest_start` with no time-shifting
- Oracle diagnostic script: `benchmarks/run_benchmark.py --oracle --forecaster seasonal_naive`

---

## Reference Benchmarks

See [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md) for full result history.

Primary reference: `benchmarks/results/benchmark_20260523T045320Z.json` — ml_quantile_v5, caiso_pjm_ercot_da_rt, training: 9.003% vs current_price_only.
