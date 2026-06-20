# Aurelius Forecasting Roadmap

## North Star Objective

Maximize **SLA-safe goodput per dollar** on public benchmarks, targeting +300% improvement over SLA-aware schedulers.

Current primary metric: savings vs `current_price_only` on `caiso_pjm_ercot_da_rt` workloads.

---

## Shipped

### v2.0 — LightGBM Quantile Regression (baseline ML)
- LightGBM p50/p90 per-region quantile forecaster
- Temporal features: hour-of-day, day-of-week, month
- Lag features: 1h, 6h, 24h, 168h
- Rolling stats: 24h, 168h mean/std
- Volatility regime: vol_24h, spike_flag, recent_mean
- Reference result: varies by workload; primary reference = 9.003% vs current_price_only (training@caiso_pjm_ercot, 2026-05-23)

### v5.0 — Price Rank / Percentile Features
- `price_momentum_168h`: (price - lag_168h) / |lag_168h|, clipped [-1, 5]
- `price_vs_lag168_abs`: price / lag_168h, clipped [0, 10]
- `lag_336h`: bi-weekly lag for cold-snap recovery detection
- Rationale: rank features encode cheap-regime positioning without raw price scale dependence
- Config flag: `include_rank_features=True`

### v6.0 — Cross-Regional Price Spread Features (2026-06-20)
- 9 new features encoding relative positioning vs peer regions:
  - `cross_price_vs_min`: how expensive vs cheapest region now (≥0)
  - `cross_price_vs_mean`: signed deviation from mean regional price
  - `cross_is_cheapest`: binary cheapest-region routing signal
  - All three at current time, 24h lag, and 168h lag
- Built from multi-region recent_context passed to predict() (no leakage)
- Single-region graceful degradation: vs_min=0, vs_mean=0, is_cheapest=1
- BacktestEngine: predict() now receives full multi-region recent_context
- Config flag: `include_cross_region_features=True`
- Benchmark result (2026-06-20, caiso_pjm_ercot_da_rt, training, greedy_migrate):
  - v6 vs current_price_only: -4.0% (+3.9pp vs v5 -7.9% in same environment)
  - 15/15 acceptance tests passing
  - Reference (9.003% for v5) is the authoritative claim; delta validated

---

## Next Steps

### High Priority: Conformal Prediction for p90 Calibration
- Current p90 coverage: 0.6433 (target: 0.88)
- Method: split-conformal prediction (arXiv:2502.04935)
- Score: nonconformity score = p90 - actual for held-out calibration set
- Expected: p90 coverage → 88%+ without harming p50 accuracy
- Unblocked by: v6.0 merge

### Medium Priority: ERCOT Spike Feature Engineering
- Winter ERCOT volatility drives the largest forecasting gap (oracle 37.7% vs ML 9.0%)
- Cross-regional is-cheapest features already capture some of this
- Next: temperature × ERCOT interaction, ERCOT scarcity index (systemwide offers near 0)
- Blocked on: conformal prediction (p90 coverage fix first)

### Medium Priority: DynamoLLM Energy Management (arXiv:2408.00741)
- 53% energy savings on LLM inference via SLO-aware power capping
- Integration target: realtime_inference workload type
- Source: HPCA 2025

### Lower Priority: Carbon-Aware Scheduling Improvements
- Current: carbon is secondary objective (β=0.5)
- Research: WattTime MOER forecasting integration for real-time carbon signal
- Unblocked by: v6.0 merge + conformal calibration

---

## Oracle Ceiling Gap

| Workload | Oracle | ML (v5) | Gap | Primary Driver |
|----------|--------|---------|-----|----------------|
| training@caiso_pjm_ercot | 37.7% | 9.003% | 28pp | Multi-region routing (winter ERCOT) |
| fine_tuning@caiso_pjm_ercot | 61.4% | 11.2% | 50pp | Extended eval horizon, spike periods |

Cross-regional features (v6.0) directly attack this gap by teaching the model "which region is cheapest right now."

---

## Research Artifacts

- [BENCHMARK_REGISTRY.md](BENCHMARK_REGISTRY.md) — all benchmark results with methodology
- [GAP_ANALYSIS.md](GAP_ANALYSIS.md) — oracle ceiling vs ML gap analysis
- [FORECAST_LEVERAGE_AUDIT.md](FORECAST_LEVERAGE_AUDIT.md) — oracle diagnostics (2026-05-22)
