# N2 SLA-Slack Power-Arbitrage Audit (Phase 0)

**N2** = the MPC treats remaining **SLA slack** (milliseconds of headroom before the latency target) as a
**power-arbitrage budget**: it spends slack to run latency-bound *online* serving work at a lower clock,
saving watts → joules → electricity dollars, **only while the SLA stays within budget**. This audit maps where
the pieces already are, and — critically — establishes that **N2 must be an explanatory decomposition of the
existing reward, not a new reward term** (the clock→power→cost channel is already in the objective). All
citations are file:line on `main` (with #116 + #117 merged).

## The 10 questions

**1. Where is SLA slack computed today?** It is **not**. Only **post-hoc violations** exist:
`unified_replay.py:543-548` marks a job SLA-safe iff `response ≤ _sla_for(job)` (`sla_s` for latency class,
`best_effort_sla_s` otherwise) and counts `sla_violations = n_total − n_sla_safe`
(`unified_replay.py:564`); `world_simulator.py:96` exposes `sla_violation_rate`. The **queue-wait** tail
(`queue_delay_p95/p99`) is computed (`world_simulator.py:405-408`) but the **completion-latency** tail and any
notion of `sla_target − tail_latency` (slack) are **absent**.

**2. Does the controller value slack, or only observe violations?** Today it **observes violations after the
fact** — the rollout reward (`controller.py:251`) is `exp_gpd − risk_weight·risk_viol·exp_gpd`, where
`risk_viol` is the realised p90 SLA-violation rate. Slack (the *distance* to the target, when not violated) is
never surfaced; the planner only sees the binary safe/violated outcome through `risk_viol` and through
goodput (only SLA-safe tokens count, `unified_replay.py:546`).

**3. Where does clock affect latency?** `roofline.py:104-120` `_tokens_per_s`: `compute = peak_flops ·
clock_factor / flops_per_token` (compute throughput scales with clock; bandwidth term is **clock-independent**),
feeding `serving_point` TTFT/completion (`roofline.py:128-167`). Clock factors:
`roofline_actions.py:42` `CLOCK_TO_ROOFLINE = {base:1.0, low:0.85, high:1.15}`.

**4. Where does clock affect energy/cost?** `roofline.py:122-125` `_power_w = TDP·(0.4 + 0.6·clock^2.4)` →
`power_factor` (`roofline_actions.py:124`) → **applied to operator cost** at
`world_simulator.py:439-443` (`power_scale=power_factor`) → `cost_model.py:217-218`
(`energy = gpu_hours · power_kw · power_scale · pue · price`). **This is the load-bearing fact for N2.**

**5. Does the planner see the marginal latency cost of clock?** Yes, **implicitly**: the rollout calls
`simulate_period` per candidate (`controller.py:233-241`), whose KPI reflects clock→throughput→GPU-seconds→
SLA-safe goodput. The marginal latency is *folded into* `goodput_per_dollar`; it is **not isolated** as a
visible "ms of slack consumed" term.

**6. Does the planner see the marginal electricity cost of clock?** Yes, **when `electricity_price_aware=True`**
(`controller.py:149`): the rollout prices each horizon step at the forecast price
(`controller.py:223-241`, `pr_k`) and `simulate_period` applies `power_scale` to the energy cost (Q4). So
**the slack-for-dollars tradeoff is already in the reward** — PR #117 proved the planner downclocks in
high-price PJM windows because of exactly this channel. With `electricity_price_aware=False` the price is the
constant fleet scalar (flat) → no price signal.

**7. Does the planner distinguish memory-bound decode vs compute-bound prefill?** Yes.
`roofline.py:95-102` `roofline_regime` classifies compute-bound vs memory-bandwidth-bound by arithmetic
intensity vs the ridge point; `search_planner.py:71-79` prunes clock options by regime
(`memory_bandwidth_bound → {base, low}`, `compute_bound → {base, high}`). Memory-bandwidth-bound decode is
treated as **clock-independent in latency** (bandwidth term in Q3) — an **upper-bound SIMULATOR_INFERENCE**
assumption (real decode has some clock sensitivity).

**8. Where can N2 be implemented without a fake reward?** **As a diagnostic decomposition layered on the
existing reward — not a new term.** Because the clock→power→cost channel already exists (Q4/Q6), the planner
*already* performs the slack-for-dollars arbitrage through gp/$. N2 adds: (a) an **explicit SLA-slack
computation** (`sla_target − completion-latency tail`) on the KPI/PeriodOutcome; (b) a **decomposition** of the
selected decision vs. its base-clock counterfactual (slack consumed ms / energy saved / dollars saved / gp/$
delta / Pareto-safe), computed **offline** in the backtest (where an extra base-clock solve is allowed) and
**online** only from values the search already produced. **No `power_scale` re-wiring, no objective rewrite, no
bonus** — those would double-count the cost channel that already exists.

**9. Which diagnostics already expose enough data?** PeriodOutcome already carries `power_w`, `energy_j`,
`electricity_price_per_kwh`, `queue_delay_p95/p99`, `quality_sla_risk`, `sla_violation_rate`
(`world_simulator.py:66-100`); the Decision carries `selected_clock`, `forecast_price_per_kwh`, `price_aware`
(`controller.py:451-460`); the Decision Diagnostics Engine records `expected_sla_violation`, the candidate
field, confidence (`decision_diagnostics.py:50-75`), with `electricity_price` already in
`CONSUMED_FORECASTS`. **Missing:** explicit `sla_target_s`, `completion_p95/p99_s`, `sla_slack_s/pct`, and the
N2 marginal decomposition.

**10. What remains simulator-inferred?** The DVFS power curve (`TDP·(0.4+0.6·clock^2.4)`), the
memory-bandwidth-bound-decode = clock-independent-latency assumption, the completion-latency tail model, the
synthetic deferrable workload, spec-decode acceptance, int4 quality risk — all **SIMULATOR_INFERENCE**. The
PJM/ERCOT/CAISO price path is **TRACE_DERIVED**. Real GPU power telemetry, real per-request output-length, real
cache-hit rates, and true demand charges would be **NEEDS_PRODUCTION_TELEMETRY** (Phase 7 robustness audit).

## Design consequence

N2 is built as **explicit slack diagnostics + an offline marginal decomposition**, leaving the reward, the
cost model, and the Pareto gate **byte-identical**. The "value" N2 reports is the *electricity dollars the
existing reward already saved by spending slack* — measured rigorously by the N2 arm vs. the base-clock arm in
the checkpointed backtest (Phase 5), never by a synthetic bonus. Deferrable time-shifting is **excluded** from
N2 (it is not online serving work) and stays in its separate energy ledger.
