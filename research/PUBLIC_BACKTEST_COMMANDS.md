# Public Economic Backtest Commands

> Canonical, reproducible commands for the Aurelius public-trace economic
> backtests, plus the module-integration harness used to evaluate the three
> research modules (WorkloadAdmissionGate, OutputLengthForecastBundle,
> GpuPlacementScorer). **Directional simulator/backtest evidence only — NOT
> production savings** (`docs/RESULTS.md` §8).

## 0. Environment

```bash
pip install numpy pandas scipy scikit-learn pulp   # core deps (lightgbm optional)
```

All commands run from the repo root.

## 1. Datasets

| dataset | path | how obtained | size |
|---|---|---|---|
| BurstGPT (real, full) | `data/external/burstgpt/raw/BurstGPT_1.csv` | `https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv` (CC-BY-4.0) | 1,429,738 requests |
| BurstGPT (fixture) | `tests/fixtures/burstgpt_sample.csv` | committed sample | 51 requests |
| Azure LLM 2024 (sample) | `tests/fixtures/azure_llm_2024_sample.csv` | committed sample of the week-long trace | 5,880 requests |
| Azure LLM 2024 (full week) | Azure blob `azurellminfererencetrace/*_1week.csv` | **inaccessible** (HTTP 401 — SAS-gated at run time) | ~44M requests |
| Canonical price data | `data/caiso_us_west_dam.csv`, `data/ercot_us_south_dam.csv`, … | committed real CAISO/PJM/ERCOT prices | present |

Notes:
- The full Azure-2024 week file is gated behind an authenticated Azure blob SAS
  token at run time (anonymous HEAD → 409, ranged GET with `x-ms-version` →
  401). The committed 5,880-row **sample** is the reproducible Azure-2024 trace
  in this environment; the canonical Azure runner itself falls back to it
  (`raw absent — SAMPLE`).
- BurstGPT's full 1.43M-request CSV is fetched at CC-BY-4.0 and replayed in full.

## 2. Existing public backtests (locked evaluation infrastructure)

### 2.1 BurstGPT trace-replay backtest

```bash
# Full real trace (downloads to data/external/burstgpt/raw if absent):
python scripts/run_burstgpt_backtest.py --csv data/external/burstgpt/raw/BurstGPT_1.csv

# Committed fixture (fast, tiny):
python scripts/run_burstgpt_backtest.py
```

- Entry point: `scripts/run_burstgpt_backtest.py:main` →
  `aurelius/traces/backtest.py:run_backtest`.
- Replays real arrivals + per-request tokens through the **unchanged** serving
  physics (`aurelius/simulation/cluster/serving.py`); scores SLA-safe goodput/$
  via `aurelius/benchmarks/economics.py`.
- Policies: `fifo`, `sla_aware` (headline baseline), `constraint_aware`
  (Aurelius), `queue_aware`, `cache_affinity_baseline`. The decision being made
  is **replica provisioning per 60 s tick** (autoscaling); physics & cost basis
  are identical across policies.
- Writes `data/external/burstgpt/processed/burstgpt_backtest_summary.json`,
  `docs/BURSTGPT_BACKTEST_RESULTS.md`.

### 2.2 Azure LLM Inference 2024 week backtest

```bash
python scripts/run_azure_llm_2024_backtest.py            # uses sample if raw absent
```

- Entry point: `scripts/run_azure_llm_2024_backtest.py:main` →
  `aurelius/traces/azure_llm.py:stream_week_aggregate` →
  `aurelius/traces/backtest.py` policy runner (same serving physics as BurstGPT).
- Replays the real arrival SHAPE + token distribution at a documented busy-tier
  load multiplier (`--primary-scale`, default 10×) + a scale sweep.
- Writes `data/external/azure_llm_2024/processed/azure_llm_2024_backtest_summary.json`,
  `docs/AZURE_LLM_2024_BACKTEST_RESULTS.md`.

### 2.3 Canonical energy backtest (per-job scheduler path)

```bash
python scripts/run_canonical_backtests.py               # console summary
python scripts/run_canonical_backtests.py --json        # golden JSON
```

- Entry point: `scripts/run_canonical_backtests.py:main` →
  `aurelius/benchmarks/canonical_backtests.py:run_canonical_backtest`.
- This is the ONLY public path that constructs a `JobScheduler`
  (`aurelius/optimization/scheduler.py`) — 1,000 energy-flexible jobs on real
  CAISO/PJM/ERCOT prices. The `GpuPlacementScorer` is wired here (default off).
- KPIs: `sla_safe_goodput_per_infra_dollar`, energy/infra cost, deadline misses,
  migrations, migration cost, optimizer candidates.

### 2.4 Public benchmark rollup

The rollup `docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md` is a **manually
aggregated** synthesis of the per-trace summary JSONs above (there is no single
rollup CLI). The component summaries are produced by 2.1–2.3.

## 3. GPU placement routing backtest (real price data)

```bash
python -c "from aurelius.benchmarks.gpu_routing_backtest import run_gpu_routing_backtest, CANONICAL_REGION_GPU_TYPES; \
import json; print(json.dumps(run_gpu_routing_backtest().to_dict(), indent=2, default=str))"
```

- `aurelius/benchmarks/gpu_routing_backtest.py:run_gpu_routing_backtest` —
  compares baseline `JobScheduler` vs one with `GpuPlacementScorer` enabled +
  `region_gpu_types` mapping, on the canonical 1,000-job trace + **real** price
  CSVs. Jobs are synthetic (energy-flex with synthetic SLA classes); price data
  is real. KPI: routing quality + goodput/$ delta for `latency_critical` subset.

## 4. Module-integration harness (this work)

```bash
# Full before/after KPI table across datasets + load regimes -> research/results/
python scripts/run_module_integration_backtest.py
```

- `scripts/run_module_integration_backtest.py` →
  `aurelius/traces/module_backtest.py:run_module_comparison`.
- Reuses the LOCKED `backtest.py` / `serving.py` / `economics.py` functions
  verbatim. Adds three additive provisioning variants compared against the
  locked `constraint_aware` baseline:
  - `ca_admission` — WorkloadAdmissionGate sheds/defers best-effort load under
    KV/queue pressure (KV proxy = realized rho; best-effort share = BurstGPT
    "API log" / Azure code-batch fraction).
  - `ca_outlen` — OutputLengthForecastBundle p90 forecast (fit on a warmup
    prefix, no leakage) sizes the decode tail instead of the realized mean.
  - `ca_all` — both.
- Runs each on BurstGPT (real) + Azure-2024 (sample) at native + saturated load
  multipliers. Emits per-variant KPI rows:
  SLA-safe goodput/$, GPU-hours, total cost, timeout% (SLA violations), queue
  p99, latency p99, scale events (migrations).

## 5. KPI definitions (from locked infra)

| KPI | source |
|---|---|
| SLA-safe goodput/$ | `economics.py:compute_sla_safe_goodput_per_infra_dollar` = `sla_compliant_goodput / total_infrastructure_cost` |
| SLA-compliant goodput | `economics.py` = `Σ tokens_tick × (1 − timeout_pct_tick/100)` |
| GPU-hours | `economics.py` active GPU-hours by type |
| total cost | gpu_infra + energy + network cost |
| SLA violations / deadline misses | per-tick `timeout_rate_pct` (serving) / deadline_misses (canonical) |
| queue delay | request-weighted `queue_p99_ms` across ticks |
| migration count | `scale_events` (serving) / region migrations (canonical) |
| migration cost | `$0.5 × migrations` (canonical economics) |
| optimizer runtime | wall-clock of the solve loop (canonical) |
