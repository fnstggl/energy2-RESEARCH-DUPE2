# Price-Aware Clock / Power Shaping Diagnostic (Track D)

Electricity price showed **0.0%** planner attribution in PR #114 — but that does not mean the opportunity is
zero. It means that, on the Azure window, price is fed to the planner as a **constant** (~$0.042/kWh) and at
that level energy is a small share of operator cost, so the controller never trades latency for trivial energy
savings. The causal pathway already exists end-to-end and needs **no new knob**:

```
clock_policy → power_factor (power_w = TDP·(0.4 + 0.6·clock^2.4))
            → energy_j  → operator_cost via  energy = gpu_hours · power_kw · power_scale · pue · electricity_price_per_kwh
clock_policy → decode/prefill service time → completion latency → SLA
```

This diagnostic drives that path with **real day-ahead prices** (`price_series.py`: PJM / ERCOT / CAISO,
$/MWh→$/kWh) across the required regimes, sweeping `clock ∈ {base, low, high}` through `run_period_episode`
(constant-clock decider — only the clock varies). Script: `scripts/diagnose_price_aware_clock.py`; artifact:
`data/external/mpc_controller/price_aware_clock.json`. **No reward bonus; a saving is reported only when
operator cost falls through actual electricity_price × energy AND SLA is not worse.**

## Causality checks (both pass)

- **Cost rises with electricity price** (base clock, decode workload): operator cost $0.0775 (p10) → $0.0908
  (p90) → $0.0958 (p95). The energy term scales with price, exactly as `energy = … · electricity_price`.
- **Downclock cuts power and energy**: low **564 W / 181 kJ** vs high **867 W / 278 kJ** (the `power^clock^2.4`
  DVFS band). At neutral `base` the energy_j/power_w *diagnostic* reads 0 (it only populates under non-neutral
  roofline actions) — but cost still books energy at `power_scale = 1.0`, which is why cost still rises with
  price at base clock.

## The key physics: memory-bound decode is clock-independent

Measured roofline factors (Llama-8B, H100, decode-heavy workload):

| clock | decode_factor | prefill_factor | completion_factor | power_factor |
|--|--|--|--|--|
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| **low** | **1.000** | 1.173 | **1.001** | **0.806** |
| high | 1.000 | 0.865 | 0.999 | 1.239 |

**Decode is memory-bandwidth-bound (AI ≈ 7.8 ≪ H100 ridge ≈ 295), so the SM clock has *zero* effect on decode
time (`decode_factor = 1.000`).** Only prefill carries a compute component (+17% at low clock), but for a
decode-heavy workload completion is decode-dominated → `completion_factor = 1.001`. So **downclocking cuts ~19%
power with ~0% latency cost** — precisely the hypothesis ("memory-bound decode → lower clock saves energy with
minimal throughput loss").

## The six required scenarios (PJM real prices: cheap p10 = $0.026, expensive p90 = $0.281, p95 = $0.377)

`low` vs `base`; Δ are downclock − base. SLA slack is set by `sla_s` (slack ≈ 0 violations, tight ≈ material).

| scenario | price | SLA | roofline regime | Δ energy/cost ($) | Δ SLA | Δ gp/$ | downclock Pareto-safe? |
|--|--|--|--|--|--|--|--|
| expensive_slack_decode | p90 | slack | memory-bound | **−0.00285** | 0.000 | +10 966 | ✅ |
| expensive_tight_decode | p90 | tight (0.167) | memory-bound | −0.00330 | 0.000 | +7 904 | ✅ |
| cheap_slack_decode | p10 | slack | memory-bound | −0.00026 | 0.000 | +1 348 | ✅ |
| cheap_tight_decode | p10 | tight | memory-bound | −0.00030 | 0.000 | +971 | ✅ |
| expensive_slack_prefill | p90 | slack | memory-bound | −0.00080 | 0.000 | +1 218 | ✅ |
| expensive_tight_prefill | p90 | tight | memory-bound | −0.00080 | 0.000 | +1 218 | ✅ |

**Downclocking is Pareto-safe in every scenario** (SLA never worsens because memory-bound service time is
clock-independent), and the **dollar saving scales with electricity price: −$0.00285 at p90 vs −$0.00026 at p10
≈ 11× larger when power is expensive.** This is the price-arbitrage signal — real, causal, and Pareto-safe.

## Honest limits

- **The "compute-bound → downclock hurts" contrast is not reachable** for Llama-8B on H100/A100: both decode
  (AI 7.8) and prefill (AI 6.3) sit far below the ridge (~295), so nothing is compute-bound. Even a
  4096-token-prompt workload stays memory-bound. We therefore **cannot demonstrate a compute-bound downclock
  penalty with this model** — documented, not fabricated. (Only `high` clock ever *speeds* prefill, at higher
  power.)
- **Magnitude is small in absolute terms.** Even at p90, downclock saves ~$0.0029 on ~$0.091 total ≈ **3% of
  operator cost** (depreciation dominates the owned-GPU cost). At p10 it is ~0.3%. So price-aware clocking is a
  **minor lever** that grows with energy's cost share (higher prices, higher PUE, cheaper/older hardware).
- **Model-fidelity caveat:** the roofline model treats memory-bound service time as *exactly* clock-independent
  (`decode_factor = 1.000`). Real GPUs show a small penalty even in memory-bound regimes (memory-controller
  clock, scheduler overheads), so "Pareto-safe everywhere" is an **upper bound** on downclock attractiveness;
  confirming it needs real serving telemetry.
- **Markets:** PJM/ERCOT/CAISO day-ahead are wired and real (used here). **EIA, ENTSO-E, and 5-minute real-time
  prices are NOT wired** (`price_series.ABSENT_MARKETS`) and are not fabricated.

## Did electricity price become decision-relevant once clock/power shaping was modeled?

**Partially — yes in mechanism, still minor in magnitude.** The gp/$-optimal clock IS price-sensitive: the
downclock advantage is ~11× larger at p90 than p10, so under a *real time-varying* price the controller would
prefer `low` more aggressively during expensive hours. But two things keep it at 0.0% in the live MPC today:
(1) the Azure window feeds a **constant** price, so there is no temporal price signal to exploit; and (2) even
at p90 the saving is ~3% of cost — below what reorders the action ranking against precision/spec. **Wiring the
real diurnal price series into the planner's period frames is the concrete next step** to convert this
Pareto-safe-but-latent saving into a selected action (and is left as a documented follow-up, since this PR is
diagnostic-first per scope).
