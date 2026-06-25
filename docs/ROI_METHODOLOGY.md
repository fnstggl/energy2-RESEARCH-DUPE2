# Aurelius ROI Methodology

**Version:** 1.1  
**Date:** 2026-05-23 (claims-truth correction 2026-06-25)  
**Status:** DIRECTIONAL SIMULATOR EVIDENCE — leakage-free historical-replay backtest.
**NOT a production-savings claim** (`docs/RESULTS.md` §8 gate is unmet).

> **Claims-truth note (Phase A, audit 2026-06-25).** This document describes the
> **energy-arbitrage** lever (one workload class) and reports an *energy-cost
> reduction* number. The **canonical Aurelius KPI is SLA-safe goodput per
> infrastructure dollar** (`docs/RESULTS.md` §1), not energy-cost %. Treat the
> figures below as **directional, simulator/historical-replay observations,
> requiring live customer-telemetry calibration before any savings is claimed** —
> they are not "proven" or "validated" production results.

---

## Executive Summary

Aurelius reduces AI compute costs by optimizing when and where GPU workloads run
across energy markets. In a leakage-free walk-forward backtest on real CAISO, PJM,
and ERCOT day-ahead electricity prices, the system showed a **directional mean
energy-cost reduction of ~25% versus a strong current-price-only baseline**
(historical-replay observation, not a production guarantee). On the canonical
goodput/$ KPI the comparable energy result is **+11.07% vs `current_price_only`
at zero deadline misses** (`docs/AURELIUS_PUBLIC_TRACE_BENCHMARK_ROLLUP.md`).

| Workload Type           | Monthly Savings Range | Expected (p50) |
|-------------------------|-----------------------|----------------|
| Background Maintenance  | 25% – 50%             | 40%            |
| Data Processing         | 28% – 46%             | 38%            |
| LLM Batch Inference     | 22% – 42%             | 34%            |
| Scheduled Batch         | 18% – 33%             | 25%            |
| Training                | 8% – 22%              | 15%            |
| Fine-Tuning             | 8% – 29%              | 15%            |
| Realtime Inference      | 1% – 12%              | 7%             |
| **Mean (all workloads)**| **~13% – ~35%**       | **25%**        |

---

## What Aurelius Optimizes

Aurelius acts as an infrastructure intelligence layer that operates at up to three control levels:

### Tier 1: Region/Time Optimization (Proven, Tier 1 Pilot-Ready)
- Chooses **which region** and **which hour** to run each workload
- Inputs: energy price forecasts, carbon intensity signals, SLA constraints
- Control required: ability to submit jobs to different cloud regions or data centers
- No GPU-level control required

### Tier 2: Cluster/Queue Optimization (Infrastructure Ready)
- Chooses **which cluster**, queue, or node pool within a region
- Inputs: queue depth, GPU availability, wait-time estimates
- Control required: multi-cluster or multi-queue visibility and placement control

### Tier 3: GPU/Node Placement (Infrastructure Ready, Requires Customer DCGM)
- Chooses **specific nodes or GPUs** based on health, temperature, and utilization
- Inputs: DCGM metrics via Prometheus/dcgm-exporter
- Control required: scheduler integration (Kubernetes labels, Slurm GRES, etc.)

**This document covers proven Tier 1 savings only.**
Tier 2 and Tier 3 provide additional savings when customer infrastructure exposes
the necessary scheduler controls.

---

## ROI Calculation Methodology

### Step 1: Identify Customer GPU Spend

Determine the monthly GPU infrastructure cost that Aurelius can optimize:
- Cloud GPU rental (on-demand or reserved instances)
- Owned GPU hardware (amortized CapEx)
- Include only workloads where placement or timing is controllable

**Exclude:**
- Storage costs
- Networking/data-transfer costs (unless modeled separately)
- Software licensing
- Workloads with zero scheduling flexibility (true latency-hard realtime)

### Step 2: Characterize Workload Mix

For each workload category, estimate its **fraction of total GPU spend**:

| Workload Type          | Scheduling Flexibility | Typical Fraction (Neocloud) |
|------------------------|------------------------|------------------------------|
| Training               | High (±48h delay OK)   | 30–40%                       |
| Fine-Tuning            | Medium (±24h delay OK) | 10–20%                       |
| LLM Batch Inference    | Medium (±24h delay OK) | 15–25%                       |
| Data Processing        | High (±48h delay OK)   | 8–15%                        |
| Scheduled Batch        | Medium (±24h delay OK) | 8–15%                        |
| Realtime Inference     | None (cannot delay)    | 5–15%                        |
| Background Maintenance | Very High (±48h OK)    | 2–5%                         |

If the customer has ≥80% scheduling-flexible workloads, expected savings are
at the higher end of the range. If ≥30% is true realtime-critical, savings are
at the lower end.

### Step 3: Apply Proven Benchmark Savings Rates

**Savings rates are derived from the Aurelius ml_quantile v2.0 backtest:**
- **Data:** CAISO OASIS (us-west), PJM Data Miner API (us-east), ERCOT CDAT API (us-south)
- **Period:** Q1 2026 (Jan–Mar) and Summer 2025 (Jun–Aug) historical day-ahead prices
- **Methodology:** Leakage-free walk-forward backtest, 5–7 folds, 30-day training windows
- **Baseline:** `current_price_only` — the strongest realistic baseline
  (always schedules to the region with the current lowest price, using live information)
- **Missing price hours:** 0%

```
Projected Monthly Savings = Σ (workload_fraction_i × savings_rate_p50_i × monthly_gpu_cost)
```

The p10 and p90 bounds reflect:
- Seasonal variation: Q1 2026 (winter, higher volatility) vs Summer 2025 (stable)
- Fold variance: observed range across walk-forward evaluation windows
- Capped at oracle ceiling (upper bound on theoretically achievable savings)

### Step 4: Compute Contract-Period Projection

```
Total Savings (N months) = Monthly Savings × N
Annual Savings           = Monthly Savings × 12
```

### Step 5: Validate with Shadow Mode (Recommended)

Before committing, run Aurelius in shadow mode on 2–4 weeks of the customer's
**actual workload trace and real market data** to validate predicted savings
against realized RT settlement prices.

```bash
# Step 1: Run shadow mode (no workloads are executed)
python -m aurelius.cli shadow run \
  --price-file customer_da_prices.csv \
  --jobs-file customer_workload_trace.csv \
  --regions us-west,us-east,us-south \
  --forecaster ml_quantile \
  --output-dir reports/shadow/

# Step 2 (7-14 days later): Compare predicted vs realized savings
python -m aurelius.cli shadow realize \
  --decisions-file reports/shadow/decisions_<ts>.jsonl \
  --rt-price-file customer_rt_prices.csv \
  --output-file reports/shadow/realized_<ts>.jsonl

# Step 3: Generate report
python -m aurelius.cli shadow report \
  --decisions-file reports/shadow/realized_<ts>.jsonl \
  --output-dir reports/shadow/
```

The shadow report shows **predicted vs realized savings** per job, per workload type,
with forecast accuracy metrics. This is the most credible economic evidence for a
pilot because it uses the customer's actual compute footprint and their actual
energy market prices.

---

## Using the ROI Calculator CLI

```bash
# Basic calculation (default neocloud workload mix, 12-month projection)
python -m aurelius.cli roi --monthly-cost 500000

# Custom workload mix
python -m aurelius.cli roi \
  --monthly-cost 1000000 \
  --workload-mix '{"training":0.50,"llm_batch_inference":0.30,"realtime_inference":0.20}' \
  --contract-months 24 \
  --output reports/roi_projection.json

# With customer context
python -m aurelius.cli roi \
  --monthly-cost 750000 \
  --num-gpus 512 \
  --gpu-type H100 \
  --region us-west \
  --contract-months 12
```

**Example output:**
```
========================================================================
AURELIUS ROI PROJECTION
========================================================================
Monthly GPU Infrastructure Cost:  $    500,000
Projection Period:                12 months

PROJECTED MONTHLY SAVINGS
----------------------------------------
  Conservative (p10):  $    85,000  (17.0% of spend)
  Expected     (p50):  $   132,500  (26.5% of spend)
  Optimistic   (p90):  $   177,500  (35.5% of spend)

PROJECTED 12-MONTH SAVINGS
----------------------------------------
  Conservative (p10):  $ 1,020,000
  Expected     (p50):  $ 1,590,000
  Optimistic   (p90):  $ 2,130,000

ANNUALIZED (12-month) SAVINGS (p50): $ 1,590,000
```

---

## Honesty Constraints

The following constraints are enforced by the ROI calculator and must NOT be
overridden in customer-facing materials:

1. **60% savings is NOT a current claim.** The aspirational stretch target
   is 60% only if every oracle diagnostic gap closes (forecasting, migration,
   multi-region correlation). Current proven mean savings: **25.0%**.

2. **Savings rates use real-data leakage-free backtesting.** No synthetic
   simulations are used for economic claims. All results are reproducible
   using public CAISO, PJM, and ERCOT historical data.

3. **Realtime inference savings are limited (~7%).** If a customer's workload
   is predominantly realtime-critical (>50%), Aurelius cannot significantly
   delay or relocate those jobs. The ROI calculator will warn about this.

4. **Season and region matter.** Q1 2026 winter results show lower savings
   for training/fine_tuning (driven by ERCOT cold-snap volatility) than
   Summer 2025. The p10 bound captures this seasonal risk.

5. **EU and Asia-Pacific regions are not yet validated.** Current benchmark
   data covers US West, East, and South only. EU savings projections require
   ENTSO-E historical data (pending API key) and separate validation.

6. **GPU-level savings are not included in these rates.** Tier 2 (queue) and
   Tier 3 (GPU/node) savings are on top of the Tier 1 region/time savings
   shown here, and require customer cluster integration.

---

## Pilot Validation Checklist

For a first enterprise pilot, the customer should provide:

| Data Required                | Format                   | Purpose                        |
|------------------------------|--------------------------|--------------------------------|
| Workload job trace           | CSV (see sample schema)  | Real workload mix for ROI calc |
| Primary compute region(s)   | String list              | Market data source selection   |
| Current monthly GPU spend    | USD amount               | ROI magnitude calculation      |
| SLA class per workload type  | Enum (strict/flex/bg)    | Safety gate configuration      |
| Scheduling flexibility       | hours, per workload type | Optimizer headroom             |

**Optional (improves accuracy):**
- Historical energy costs from invoices (validates price data alignment)
- Queue state logs from Kubernetes/Slurm/Ray (enables Tier 2 optimization)
- PROMETHEUS_URL/DCGM_EXPORTER_URL (enables Tier 3 GPU health optimization)

---

## Baseline Comparison: Why current_price_only Is the Right Benchmark

The `current_price_only` baseline always schedules each job to the region with
the **lowest current market price at the time of submission**, using the same
live information available to the operator. This is the strongest realistic
baseline because it:

- Uses real market data (not synthetic prices)
- Has perfect information about the current price (no forecasting needed)
- Represents what a sophisticated manual operator could achieve

Aurelius beats this baseline by:
1. **Forecasting** future price movements to schedule jobs when prices will be lower
2. **Multi-region routing** to exploit price spreads between regions
3. **Migration/checkpointing** to move long-running jobs when better prices emerge

Weak baselines (e.g., "fixed to one region") would produce inflated savings figures.
Aurelius only reports savings against `current_price_only`.

---

## Oracle Diagnostic: The Upper Bound

The oracle optimizer has perfect future knowledge of all prices and demonstrates
the theoretical maximum savings from region/time optimization:

| Workload           | ml_quantile v2.0 (proven) | Oracle Ceiling | Remaining Gap |
|--------------------|---------------------------|----------------|---------------|
| Training           | 15.0%                     | 29.9%          | 14.9pp        |
| Fine-Tuning        | 13.4%                     | 46.8%          | 33.4pp        |
| LLM Batch          | 33.6%                     | 42.7%          | 9.1pp         |
| Scheduled Batch    | 25.3%                     | ~35%           | ~10pp         |

For LLM batch inference, the optimizer is near-optimal (9pp gap from oracle).
For training and fine-tuning, the oracle gap is larger — driven by winter ERCOT
cold-snap volatility that the forecaster partially misses. Summer data shows
much smaller oracle gaps (4–11pp), confirming the optimizer is nearly optimal
in stable market conditions.

---

## Reproduction Commands

All benchmark results can be independently reproduced:

```bash
# Clone the repo, install dependencies
git clone https://github.com/fnstggl/energy2.git
cd energy2
pip install -e "aurelius[dev]"

# Run the validated benchmark (requires CAISO, PJM, ERCOT data in data/)
python benchmarks/run_benchmark.py \
  --region-combo caiso_pjm_ercot_da_rt \
  --forecaster ml_quantile \
  --train-days 30 \
  --num-jobs 100

# Run oracle diagnostics
python benchmarks/run_benchmark.py \
  --region-combo caiso_pjm_ercot_da_rt \
  --forecaster ml_quantile \
  --oracle

# Compute ROI for a specific customer profile
python -m aurelius.cli roi \
  --monthly-cost 500000 \
  --contract-months 12
```

Benchmark artifacts are archived under:
```
benchmarks/results/benchmark_ml_quantile_v2_3region_q12026_20260522.json
benchmarks/results/benchmark_ml_quantile_v3_weather_q12026_20260523.json
```

---

## Frequently Asked Questions

**Q: Can Aurelius guarantee 25% savings?**
A: No. The 25% is the mean observed in a leakage-free backtest on real historical data.
Actual savings depend on workload mix, region, season, and scheduling flexibility.
We recommend running shadow mode for 2–4 weeks to validate projected savings against
the customer's actual environment before committing to a contract.

**Q: Why is training savings lower than background_maintenance?**
A: Training jobs run for 8–48+ hours continuously and are often not checkpointable.
Once started, they cannot be interrupted or migrated. The optimizer can only choose
the start time and region, limiting its impact. Background maintenance jobs are
short-duration and can be freely rescheduled, giving the optimizer maximum flexibility.

**Q: Does Aurelius work with AWS, GCP, Azure, and neoclouds?**
A: Yes, as long as job submission can be redirected to different regions or
data centers. Aurelius integrates via dry-run Kubernetes, Slurm, AWS Batch,
Ray, or a simple CSV replay interface. Cloud energy prices are modeled from
public ISO grid data in the same market area as each cloud region.

**Q: How does carbon pricing affect the ROI?**
A: Carbon cost is weighted in the optimizer (default beta=0.3). When WattTime
carbon data is available (currently CAISO only on the free plan), the optimizer
trades off price savings against carbon reduction. Carbon pricing is a feature
flag and does not affect the energy-cost savings calculations shown above.

**Q: What happens if a price spike is missed by the forecaster?**
A: The safety gate blocks decisions where the forecaster's p90 confidence interval
projects costs more than a workload-specific threshold above the baseline. For
realtime inference (2% threshold), fine-tuning (5–8%), and training (10%), the
optimizer falls back to the `current_price_only` decision rather than risk an
SLA violation or cost overrun. This limits upside savings slightly but prevents
downside exposure.
