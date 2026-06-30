# Combined Results — Token-Shape Forecaster + Price-Aware Clock (Track E)

One bounded re-run on the same Azure eval window, full action layer, same Pareto gate
(`scripts/diagnose_combined.py`, artifact `data/external/mpc_controller/combined.json`). Four arms:

| arm | SLA-safe gp/$ | SLA | GPU-h | energy (J) | precision | clock mix |
|--|--|--|--|--|--|--|
| **A** sla_aware (strongest baseline) | 111 091 | 0.022 | 0.203 | — | (no roofline) | base×6 |
| **B** current main MPC (single-median) | **177 916** | **0.000** | 0.136 | 112 254 | fp8 | **low×6** |
| **C** token-shape MPC (this PR) | 174 933 | **0.050** | 0.133 | 100 608 | **int4** | base×6 |
| **D** token-shape MPC @ PJM p90 price | 153 214 | 0.050 | 0.133 | 96 417 | int4 | **base×3 / low×3** |

## Finding 1 — the token-shape forecaster is NOT a robust win (honest negative)

- **C vs A (baseline): +57.5%** — still clears the strongest baseline by a wide margin (consistent with Track
  A's median +60%).
- **C vs B (current main): −1.68%, and NOT Pareto-safe.** Feeding the token-shape ensemble nudged the planner
  to **int4** (carrying the 0.05 quality-risk → SLA 0.050) instead of B's **fp8** (SLA 0.000). So on this
  window the forecaster *slightly regressed* and shed SLA via int4 — the same int4 failure mode Track A flagged
  in the `mixed` window. The Pareto gate correctly returns `False` for arm C.
- This corroborates Track C (`TOKEN_SHAPE_FORECAST_GAP_RESULTS.md`): the token-shape forecaster closes only
  58.5% of the oracle gap vs the PR #113 scenario ensemble's 96.7%. **It is marginal, config-sensitive, and
  here Pareto-unsafe — not a result to promote.** (Note the "current MPC" gp/$ is itself planning-config
  sensitive: 177 916 here with the 6-period median prompt vs 163 336 in Track C with the 120-period median —
  the token-shape effect lives inside that noise band.)

## Finding 2 — price-aware clocking: expensive power DOES move the live MPC's clock decision (clean positive)

Holding the workload and forecaster fixed and changing ONLY the electricity price (arm C → arm D):

```
electricity price  0.042 → 0.281 $/kWh (PJM p90)   ⇒   clock 'low' fraction  0.0 → 0.5
```

At the default constant price the token-shape MPC never downclocks (base×6); at the **real PJM p90 price it
downclocks half the periods** (base×3 / low×3), and its energy falls (100 608 → 96 417 J). This is the
full-MPC confirmation of Track D: **once electricity is genuinely expensive, the planner selects downclocking on
its own**, through the real `energy = … · power_factor · electricity_price` path — no new knob, no reward bonus.
gp/$ is lower at p90 (153 214) only because expensive power costs more *per se*; the **behavioural** response
(more downclocking) is the result.

## Forecast attribution after the improvement

From the Track C artifact (`token_shape_gap.json`), the leave-one-out attribution AFTER the token-shape
forecaster: output_length 69.0% / prompt_length 17.3% / interarrival_cv 13.5% / arrival_rate 0.3% — prompt's
*raw* planner-value fell 36% while output's was unchanged (the rise in output's normalized share is a
renormalisation artifact; see `TOKEN_SHAPE_FORECAST_GAP_RESULTS.md`).

## Honest bottom line

- **Best deployable gp/$ vs strongest baseline:** the **current main MPC** at **+57–60%** (Track A median),
  Pareto-safe in 7/8 regimes — *not* the token-shape MPC, which here regressed and shed SLA.
- **The token-shape forecaster does not beat what PR #113 already ships;** keep it **opt-in and off by
  default** (the `scenario_builder` hook). Its one real, reproducible effect is on prompt-length; output-length
  remains the hard, ~irreducible lever.
- **Price-aware clocking is a real, causal, Pareto-safe lever that the live MPC already exploits when power is
  expensive** — but it is minor (~3% of cost at p90) and dormant on the Azure window because that window feeds
  a constant cheap price. Wiring a real diurnal price into the planner's frames is the highest-value follow-up.

Bounded window → magnitudes simulator-inferred; the robust findings are the **directions** (token-shape ≈ no
net win and not Pareto-safe here; price → more downclocking).
