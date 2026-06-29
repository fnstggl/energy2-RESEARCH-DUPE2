# All-Knobs Backtest Results (Phase 6)

The checkpointed N2 backtest (`scripts/run_checkpointed_n2_backtest.py`, artifact
`data/external/mpc_controller/checkpointed_n2_backtest.json`). **48/48 cells COMPLETED, 0 TIMEOUT, 0 FAILED**
— 3 markets × 2 windows (expensive, volatile — where electricity matters most) × 8 arms, hourly cadence,
`max_decisions=3`, ≤80 req/period, seed-0 world-state. The clock-focused search isolates the N2 knob cleanly
and tractably; the **full adaptive all-knobs search is heavy at hourly cadence** (see "All-knobs tractability").

## Arms (gp/$ at the expensive windows; N2 = electricity_price_aware clock)

| market·window | baseline | no_n2 | **n2_dvfs** | oracle | scenario |
|--|--|--|--|--|--|
| pjm·expensive | 336410 | 328114 | **332974** (`low×3`) | 333076 | 324022 (`high×3`) |
| ercot·expensive | 394098 | 374565 | **375745** (`base×3`) | 383266 | 381092 |
| caiso·expensive | 390060 | 396136 | **397676** (`high×2`) | 399348 | 397020 |

The N2 arm makes **price-appropriate clock choices in both directions**: it **downclocks** at high price (pjm,
p90 ≈ $0.18 → `low×3`), **drops the upclock** at medium price (ercot, $0.05 → `base×3` vs no_n2's
`base×2,high×1`), and **upclocks** at low price (caiso, $0.03 → `high×2`, spending cheap energy for
throughput). All three raise gp/$ vs. the price-unaware `no_n2` arm.

## The seven comparisons (median gp/$ lift across the 6 cells)

| question | result | tier |
|--|--|--|
| **A. strongest SLA-aware baseline** | 336–394k gp/$ (the fair baseline) | — |
| **B. all-knobs flat price** vs no_n2 | flat uses the constant fleet price; not an electricity result | — |
| **C. all-knobs real price, no N2** | the price-unaware reference | — |
| **D. + N2** (n2_value = D−C) | **+4859 / +5784 gp/$ in high-price PJM** (≈+1.5%); +1180 ERCOT, +1539 CAISO; **0** in low-price volatile windows; mean **+2227** over 6 cells | DIRECTIONAL_ONLY |
| **E. + deferrable** (serving Δ) | **0** to serving gp/$ in all 6 cells (separate ledger) | FIXTURE_ONLY |
| **F. + N2 + deferrable** | = N2 serving lift (interaction **0** — orthogonal) | DIRECTIONAL + FIXTURE |
| **G. oracle all-knobs** (gap = G−D) | oracle beats causal N2 by **+102 … +7520 gp/$** (forecast regret); see regret doc | diagnostic |

- **best-window N2 lift: +5784 gp/$ (pjm·volatile)** — but NOT headline-safe (see below).
- **median N2 lift ≈ +1360 gp/$** across the 6 cells; **worst window 0** (low-price volatile).
- **Pareto-safe fraction: 0/6** — every cell fails `pareto_sla_not_worse`.

## Headline-safe vs directional-only

**No cell is HEADLINE_SAFE.** In all 6, the best N2 all-knobs arm carries a higher SLA-violation rate than the
SLA-aware baseline (`pareto_sla_not_worse = False`); 2 cells beat the baseline on gp/$ (`ercot·volatile`,
`caiso·expensive`) but fail the SLA clause. So the N2 gp/$ lift is reported as **DIRECTIONAL_ONLY** (a real,
causal within-MPC isolation), never a headline. This is the same SLA-bought edge the base MPC has always shown
(`training.claim_gate`).

## All-knobs tractability (honest)

The full **adaptive** all-knobs search (every connected knob live, beam-pruned) is **heavy at hourly cadence**:
a single `pjm·expensive` adaptive cell ran **> 2.5 min without completing** before the bounded run was stopped
— the same intractability #117 found for the exhaustive 314928-bundle space. So the **all-knobs total at
hourly cadence is not tractable within a bounded per-cell cap**; the runner marks such cells TIMEOUT and
continues (never jams). The clock-focused arms above are the tractable measurement of the **N2-relevant** knob
(clock), and they answer the electricity questions cleanly. A complete adaptive all-knobs sweep needs a larger
per-cell cap (or coarser cadence) and is left to a follow-up — the runner supports it via `--search adaptive`.

## Bottom line

- **Current best all-knobs gp/$ (clock-focused, real price + N2): 333–398k**, but **below the SLA-aware
  baseline on the Pareto frontier in every window** → **no headline all-knobs gp/$ saving is claimed.**
- N2's contribution is a **real, causal, price-directional clock lever** (+4859/+5784 in high-price PJM,
  ~0 at low price), reported DIRECTIONAL_ONLY.
- Nothing was tuned; the Pareto gate is unchanged; 48/48 cells completed (the runner cannot jam).
