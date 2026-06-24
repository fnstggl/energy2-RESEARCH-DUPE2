# min_cost_safe Policy Backtest

> **Directional simulator evidence only — NOT production savings** (`docs/RESULTS.md` §8).

- Generated: 2026-06-24
- MCS timeout gate: 9.5% per-tick (aggregate guaranteed < 10%)
- SHU reference target rho: 0.75 (anticipatory EWMA)
- **Overall vs SHU: `MIXED_ALPHA_WIN_TIE`**
- **Overall vs CA: `MIXED_ALPHA_WIN_TIE`**

## Results

| dataset | scale | MCS gpd/$ | SHU gpd/$ | MCS vs SHU % | MCS vs CA % | MCS timeout % | MCS GPU-h | SHU GPU-h | verdict vs SHU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| burstgpt | 1× | 8,691.77 | 8,691.77 | 0.00% | 0.00% | 4.21% | 0.92 | 0.92 | `TIE` |
| burstgpt | 300× | 448,129.05 | 448,129.05 | 0.00% | 0.00% | 4.73% | 0.02 | 0.02 | `TIE` |
| azure_2024 | 1× | 12,511.33 | 12,511.33 | 0.00% | 0.00% | 2.00% | 26.00 | 26.00 | `TIE` |
| azure_2024 | 50× | 604,601.10 | 604,601.10 | 0.00% | 0.00% | 3.32% | 0.53 | 0.53 | `TIE` |
| azure_2024 | 500× | 2,657,445.38 | 2,133,669.93 | 24.55% | 52.06% | 7.05% | 0.12 | 0.15 | `ALPHA_WIN` |
| burstgpt_hf | 100× | 450,118.60 | 450,118.60 | 0.00% | 0.00% | 2.14% | 2.30 | 2.30 | `TIE` |
| burstgpt_hf | 500× | 1,715,476.85 | 1,672,445.39 | 2.57% | 16.35% | 2.59% | 0.60 | 0.62 | `ALPHA_WIN` |

## Interpretation

- `min_cost_safe` searches from MIN_REPLICAS upward for the smallest fleet where per-tick timeout_rate_pct < 9.5% gate (with cache prefill savings).
- Because each per-tick value is strictly below 9.5%, aggregate timeout < 9.5% < 10% is guaranteed by construction — stronger than the 10% aggregate gate alone.
- No EWMA anticipation: purely reactive. Advantage over SHU during gradual-load ramp-down (never over-provisions). Disadvantage during sudden burst ramp-up.
- `safe_high_utilization` (SHU, rho=0.75, EWMA-anticipatory) is the primary comparison baseline and the current Aurelius headline policy.
- Simulator results only — not production savings.
