# Electricity Attribution With Actions

> **Deferred with the full backtest** — see
> [`ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md`](ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md).

PR #114 measured electricity_price at **0.0%** of planner forecast value (constant price in the frames). This
PR makes the price a real, varying, per-period signal and adds price-aware clock + deferrable shifting — so
re-running the leave-one-out attribution to see whether electricity becomes attribution-relevant is now
*possible*. It is **not run here**: the attribution harness re-plans the full MPC many times, which is the same
hourly-replay cost that made the backtest TOO_HEAVY (deferred). What this PR establishes instead, via the
bounded smoke + decision-diagnostics electricity fields (`Decision.forecast["electricity"]`): the controller
now **consumes** a varying price and records the price + selected clock per decision, so the attribution will
be meaningful once the checkpointed backtest runs it. **No claim is made yet that attribution rose above 0%.**
