# Research Result: BurstGPT HF Preemption Overhead Cross-Validation [run 2026-06-21-s]

**Date:** 2026-06-21
**Run label:** 2026-06-21-s
**Category:** Infrastructure Improvement (validation completeness)

## Benchmark Command

```python
from aurelius.benchmarks.srtf_serving_backtest import run_burstgpt_hf_preemption_overhead_backtest
r = run_burstgpt_hf_preemption_overhead_backtest(
    overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.00),
    servers=4, target_rho=0.85, job_limit=5880, sla_s=30.0,
)
```

## Dataset

- **Dataset:** BurstGPT HF (lzzmm/BurstGPT, CC-BY-4.0)
- **Path:** `data/external/hf/lzzmm__BurstGPT/burstgpt_1_full/processed/normalized_sample.jsonl`
- **Records used:** 5,880 (Azure comparability scale; full file: 59,999)
- **Simulation:** M/G/c discrete-event queue, servers=4, ρ=0.85, SLA=30s

## Results

| overhead_s | SRPT gp/$ | Decoupled gp/$ | FIFO gp/$ | SRPT vs FIFO | Dec vs FIFO |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 48,598.82 | 38,695.42 | 6,528.76 | +644.38% | +492.69% |
| 0.15 | 47,575.20 | 38,315.67 | 6,528.76 | +628.70% | +486.88% |
| 0.30 | 46,319.85 | 37,169.09 | 6,528.76 | +609.47% | +469.31% |
| 0.50 | 44,894.71 | 36,229.99 | 6,528.76 | +587.65% | +454.93% |
| 1.00 | 43,456.53 | 34,942.30 | 6,528.76 | +565.62% | +435.21% |

**Retention at 0.30s/event:**
- SRPT: 94.58%
- Decoupled: 95.25%
- Breakeven: None (never reached within 1.0s sweep)

## Comparison vs Azure LLM 2024 [run -o]

| metric | Azure | BurstGPT HF | result |
|---|---:|---:|---|
| SRPT retention @0.30s | 92.9% | **94.58%** | BurstGPT +1.7pp |
| Decoupled retention @0.30s | 92.65% | **95.25%** | BurstGPT +2.6pp |
| Breakeven SRPT | >1.0s | >1.0s | both safe |
| Breakeven Decoupled | >1.0s | >1.0s | both safe |

## All Cross-Trace Gates Status

| gate | Azure LLM 2024 | BurstGPT HF | status |
|---|---|---|---|
| Core SRTF gain | +322.2% [run -g] | +644.4% [run -p/-r] | ✅ CLOSED |
| Alpha sweep | α=0.001 optimal [run -m] | +492.7% at α=0.001 [run -p/-r] | ✅ CLOSED |
| Noisy prior 30%-CV | 100% retention [run -n] | 100% retention [run -r] | ✅ CLOSED |
| SLA-aware baseline | +125.4% vs FIFO [run -n] | +210.6% vs FIFO [run -r] | ✅ CLOSED |
| Conformal adaptive α | +322.24% vs FIFO [run -q] | +644.4% vs FIFO [run -r] | ✅ CLOSED |
| Preemption overhead | 92.65% ret [run -o] | **95.25% ret [run -s]** | ✅ **CLOSED** |

## Tests

- 15 new tests added in `tests/test_preemption_overhead_backtest.py` (Class 11)
- All 85 tests passing
- No regressions

## Conclusion

BurstGPT HF is confirmed MORE robust to preemption overhead than Azure LLM 2024.
All six cross-trace validation gates now closed on both public LLM traces.
