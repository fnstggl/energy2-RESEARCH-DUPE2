# Real-Price DVFS Validation

> **Bounded validation only — the longer multi-window DVFS sweep is deferred with the full backtest
> ([`ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md`](ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md)).**

Clock/DVFS is already a live CONNECTED MPC action; this PR wires it to choose against **real diurnal prices**.
When `controller.electricity_price_aware=True`, the horizon rollout prices each step at the forecast electricity
price path (`traj.point("electricity_price", k)`), so the planner trades clock for energy cost when power is
expensive — through the real `energy = … · power_factor · electricity_price` term, no reward bonus.

The causal path is proven by the fixture-level smoke (`scripts/smoke_electricity_validation.py`): the real
diurnal price varies (P1), high prices cost more (P2), and **downclocking memory-bound decode is a Pareto-safe
gp/$ win whose advantage scales with price** — PJM gp/$ gain **+3281 at p90 vs +403 at p10**, SLA unchanged
(P3). So a gp/$-maximising price-aware planner downclocks more when power is expensive. See the smoke-results
block in `ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md`. PR #115 Track D/E independently confirmed this at the
full-MPC level (downclock fraction 0.0→0.5 at PJM p90). The longer 24h × multi-market DVFS sweep with a headline
gp/$ delta is part of the deferred checkpointed backtest.
