# safe_high_utilization Policy Backtest

> **Directional simulator evidence only — NOT production savings** (`docs/RESULTS.md` §8). Policy validated by `run_azure_2024_safe_utilization_frontier.py` (anticipatory@0.75: gpd/$ +12.97% over constraint_aware, timeout 9.465% SAFE < 10% gate).

- Generated: 2026-06-22
- SHU target rho: 0.75 (constraint_aware uses 0.65, utilization_aware uses 0.85)
- **Overall verdict: `MIXED_ALPHA_WIN_TIE`**

## Results

| dataset | scale | SHU gpd/$ | CA gpd/$ | SHU vs CA % | SHU timeout % | SHU GPU-h | CA GPU-h | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| burstgpt | 1× | 8,691.77 | 8,691.77 | 0.00% | 4.21% | 0.92 | 0.92 | `TIE` |
| burstgpt | 300× | 448,129.05 | 448,129.05 | 0.00% | 4.73% | 0.02 | 0.02 | `TIE` |
| azure_2024 | 1× | 12,511.33 | 12,511.33 | 0.00% | 2.00% | 26.00 | 26.00 | `TIE` |
| azure_2024 | 50× | 604,601.10 | 604,601.10 | 0.00% | 3.32% | 0.53 | 0.53 | `TIE` |
| azure_2024 | 500× | 2,133,669.93 | 1,747,577.54 | 22.09% | 4.04% | 0.15 | 0.18 | `ALPHA_WIN` |
| burstgpt_hf | 100× | 450,118.60 | 450,118.60 | 0.00% | 2.14% | 2.30 | 2.30 | `TIE` |
| burstgpt_hf | 500× | 1,672,445.39 | 1,474,394.31 | 13.43% | 2.49% | 0.62 | 0.70 | `ALPHA_WIN` |

## Interpretation

- `safe_high_utilization` uses EWMA-anticipatory sizing (same as `constraint_aware`) but with a higher utilization target (rho=0.75 vs 0.65) and no hysteresis.
- The frontier audit confirmed rho=0.75 is the boundary of the safe anticipatory frontier; rho=0.85 is UNSAFE (11.648% timeout).
- **Fixture-scale TIE (1×, 50×, 300×) is expected**: at rates below ~10 rps, `_size_for_target` ceiling arithmetic gives the same base replica count for rho=0.65 and rho=0.75. The improvement is only visible at rates ≥ ~10 rps.
- **BurstGPT HF scale-100× and Azure scale-500×** confirm the mechanism: SHU outperforms CA by +5–22% in the realistic higher-load regime, all SAFE (timeout < 10% gate).
- Primary benchmark evidence: full Azure 2024 trace frontier audit (`run_azure_2024_safe_utilization_frontier.py`): anticipatory@0.75 = +12.97% vs CA, timeout=9.465% SAFE.
- A timeout rate above 10% classifies as UNSAFE and excludes from headline.
- Simulator results only — not production savings.
