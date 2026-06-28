# Electricity Action-Value Results

> **Headline value deferred with the full backtest** — see
> [`ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md`](ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md).

The five-arm value comparison (flat / real-no-actions / real+DVFS / real+deferrable / real+both) requires the
full-week hourly all-arm sweep, which is **SKIPPED (TOO_HEAVY)** and deferred to a checkpointed follow-up. This
PR lands the infrastructure + a bounded causal validation only: it proves the mechanisms (price varies in
frames; high price costs more; price-aware clock downclocks more; deferrable shifts to cheap hours; flat price
yields no fake shifting value; serving SLA is protected), **without** a headline gp/$ number. The deferrable
*shifting* value (price_aware vs asap electricity cost on the synthetic pool) is shown in the smoke's P4, but
that pool is `SIMULATOR_INFERENCE` and is not a production headline.
