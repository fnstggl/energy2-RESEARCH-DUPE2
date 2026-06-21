# BurstGPT HF Preemption Overhead Sensitivity Results [run 2026-06-21-s]

> **Directional simulator result — not production savings.**
> Benchmark harness: `aurelius/benchmarks/srtf_serving_backtest.py`
> Function: `run_burstgpt_hf_preemption_overhead_backtest()`
> Dataset: BurstGPT HF (lzzmm/BurstGPT, CC-BY-4.0, normalized sample)
> n=5,880 requests (Azure comparability scale), servers=4, ρ=0.85, SLA=30s

---

## Summary

This run closes the last documented overhead cross-validation gap: preemption
overhead sensitivity was validated on Azure LLM 2024 (run 2026-06-21-o) but
**not** on BurstGPT HF. This run fills that gap.

**Key finding: BurstGPT is MORE robust to preemption overhead than Azure LLM 2024.**

| metric | Azure LLM 2024 [run -o] | BurstGPT HF [run -s] | delta |
|---|---:|---:|---|
| SRPT @0 overhead vs FIFO | +322.2% | **+644.4%** | +322pp (2× baseline gain) |
| Decoupled @0 overhead vs FIFO | +274.0% | **+492.7%** | +219pp |
| SRPT retention @0.30s overhead | 92.9% | **94.58%** | **+1.7pp more robust** |
| Decoupled retention @0.30s overhead | 92.65% | **95.25%** | **+2.6pp more robust** |
| SRPT breakeven overhead | >1.0s | **None (>1.0s)** | Same: never reached |
| Decoupled breakeven overhead | >1.0s | **None (>1.0s)** | Same: never reached |

**Both preemptive disciplines remain well above FIFO at every tested overhead level,
including the conservative 1.0s upper bound.**

---

## Physical Explanation

BurstGPT's heavier output-token distribution makes each preemption overhead
a *smaller fraction* of total service time:

| | BurstGPT HF | Azure LLM 2024 |
|---|---|---|
| p50 output tokens | ~236 | ~90 |
| p50 service time (s) | 0.15 + 236×0.02 ≈ **4.87s** | 0.15 + 90×0.02 ≈ **1.95s** |
| 0.30s overhead / p50 service | 0.30/4.87 = **6.2%** | 0.30/1.95 = **15.4%** |

The same 0.30s overhead per preemption event is a 2.5× smaller relative penalty
on BurstGPT, explaining the higher retention. This exactly matches SRPT multiserver
theory (arXiv:1805.07686): the gain from SRPT grows with service-time variance,
and so does overhead robustness (overhead/service → 0 as service → ∞).

---

## Overhead Sweep Table

Command used:
```python
from aurelius.benchmarks.srtf_serving_backtest import run_burstgpt_hf_preemption_overhead_backtest
r = run_burstgpt_hf_preemption_overhead_backtest(
    overhead_values_s=(0.0, 0.15, 0.30, 0.50, 1.00),
    servers=4, target_rho=0.85, job_limit=5880, sla_s=30.0,
)
```

| overhead_s | SRPT gp/$ | Decoupled gp/$ | FIFO gp/$ | SRPT vs FIFO | Dec vs FIFO | SRPT preempts | Dec preempts |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 48,598.82 | 38,695.42 | 6,528.76 | **+644.38%** | **+492.69%** | 3,088 | 3,206 |
| 0.15 | 47,575.20 | 38,315.67 | 6,528.76 | +628.70% | +486.88% | 3,095 | 3,267 |
| 0.30 | 46,319.85 | 37,169.09 | 6,528.76 | **+609.47%** | **+469.31%** | 3,129 | 3,316 |
| 0.50 | 44,894.71 | 36,229.99 | 6,528.76 | +587.65% | +454.93% | 3,134 | 3,359 |
| 1.00 | 43,456.53 | 34,942.30 | 6,528.76 | +565.62% | +435.21% | 3,298 | 3,524 |

**Retention at 0.30s/event (canonical measurement point):**
- SRPT: **94.58%** (vs 92.9% on Azure — +1.7pp more robust)
- Decoupled: **95.25%** (vs 92.65% on Azure — +2.6pp more robust)

Breakeven overhead (point where gain drops to zero vs FIFO):
- SRPT: **None reached** (never drops below FIFO within 1.0s sweep)
- Decoupled: **None reached**

---

## Comparison with Azure LLM 2024 [run 2026-06-21-o]

The same overhead model (preemption_overhead_s swept over identical values) was
applied to Azure LLM 2024 in run -o. Direct comparison:

| overhead_s | Azure SRPT vs FIFO | BurstGPT SRPT vs FIFO | Azure Dec vs FIFO | BurstGPT Dec vs FIFO |
|---:|---:|---:|---:|---:|
| 0.00 | +322.2% | **+644.4%** | +274.0% | **+492.7%** |
| 0.30 | +299.4% | **+609.5%** | +253.9% | **+469.3%** |
| 1.00 | ~+255% | **+565.6%** | ~+232% | **+435.2%** |

At every overhead level, BurstGPT shows larger absolute gains (heavier distribution)
AND higher retention (overhead is smaller relative to longer service times).

---

## Cross-Trace Validation Summary (All Gates)

All cross-trace validation gates now passed on both Azure LLM 2024 and BurstGPT HF:

| gate | Azure LLM 2024 | BurstGPT HF | status |
|---|---|---|---|
| Core SRTF gain | +322.2% SRPT vs FIFO [run -g] | +644.4% SRPT vs FIFO [run -p/-r] | ✅ PASSED |
| Alpha sweep Pareto | α=0.001 optimal [run -m] | +492.7% at α=0.001 [run -p/-r] | ✅ PASSED |
| Noisy prior robustness (30%-CV) | 100% retention [run -n] | 100% retention [run -r] | ✅ PASSED |
| SLA-aware baseline measured | +125.4% vs FIFO [run -n] | +210.6% vs FIFO [run -r] | ✅ PASSED |
| Conformal adaptive α | +322.24% vs FIFO [run -q] | +644.4% vs FIFO [run -r] | ✅ PASSED |
| Preemption overhead @0.30s | 92.65% retention [run -o] | **95.25% retention [run -s]** | ✅ **PASSED** |

All six cross-trace validation gates now closed on both public LLM traces.

---

## KPI Summary Table (required by benchmark governance)

| KPI | Main (FIFO) | Candidate (SRPT @0.30s) | Delta |
|---|---:|---:|---|
| SLA-safe goodput/$ | 6,528.76 | 46,319.85 | **+609.47%** |
| Retention vs 0-overhead | — | 94.58% | — |
| GPU-hours | identical | identical | 0 |
| SLA violations | — | — | same model |
| Preemption overhead | 0s/event (FIFO n/a) | 0.30s/event | — |
| Migration count | 0 | 0 | 0 |
| Optimizer runtime | 2.5s | 2.5s | 0 |

| KPI | Main (FIFO) | Candidate (Decoupled @0.30s) | Delta |
|---|---:|---:|---|
| SLA-safe goodput/$ | 6,528.76 | 37,169.09 | **+469.31%** |
| Retention vs 0-overhead | — | 95.25% | — |

---

## Research Basis

- **FastSwitch** (arXiv:2411.18424, NeurIPS 2024): 1.4–11.2× TTFT regression
  from context switching; 0.30s overhead is within the realistic range.
- **arXiv:2411.07447**: recomputation < swapping for sequences < 4,000 tokens;
  BurstGPT p99=934 tokens well within this threshold.
- **SRPT multiserver** (arXiv:1805.07686): SRPT throughput optimality holds for
  M/G/c with heavy-tailed service times; gain and overhead robustness scale together.
- **BurstGPT** (arXiv:2401.17644): real production LLM inference trace; heavier
  output distribution than Azure LLM 2024 amplifies SRPT benefits.

---

## Safety

- Shadow-only simulator result (not production savings).
- No benchmark parameters (physics, SLA definition, price trace) were modified.
- Oracle prior used throughout (same as all prior runs -g through -r).
- 15 new tests, all passing.
- `shadow_tag = "shadow_only_simulator_result_not_production_savings"` present.
