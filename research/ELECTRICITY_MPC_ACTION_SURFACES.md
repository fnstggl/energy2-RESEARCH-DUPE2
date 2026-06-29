# Electricity MPC Action Surfaces (Phase 1)

Which electricity controls are **true MPC actions** (chosen inside the `ActionBundle` search), which are
**scheduler** actions, which are **diagnostic-only**, and which are **skipped** — with evidence and honest
limits. No knob is faked: a control is only called an MPC action if the controller actually selects it.

## A. Price-aware DVFS / clock — TRUE MPC ACTION ✅

`clock_policy ∈ {low, base, high}` is a CONNECTED `ActionBundle` surface (`actions.py`,
`reward_channel="roofline_serving"`, since PR #111). The MPC search scores each clock against the full
economics:

- **price path** — when `controller.electricity_price_aware=True`, the horizon rollout prices each step at the
  forecast electricity price (`traj.point("electricity_price", k)`);
- **roofline regime** — `decode_factor`/`prefill_factor` from `roofline_actions` (memory-bound decode is
  clock-independent → free to downclock; compute-bound prefill is not);
- **SLA slack + queue pressure** — through the serving replay's violation rate in `simulate_period`;
- **power draw + energy cost** — `power_w = TDP·(0.4+0.6·clock^2.4)` → `energy_J → kWh → × price → $`.

So clock is genuinely selected by the MPC as a function of price. Validated causally in the bounded smoke
(`ELECTRICITY_ECONOMIC_CONTROLLER_RESULTS.md` P3) and PR #115 Track D/E (downclock fraction 0.0→0.5 at PJM p90).

## B. Deferrable work / energy shifting — PLANNER-VISIBLE SCHEDULER (not an ActionBundle knob) ⚠️

Deferrable shifting is implemented as a **persistent pool + price-aware look-ahead scheduler**
(`deferrable.py`), **not** as an `ActionBundle` surface the MPC search enumerates. The scheduler exposes the
required options — run-now / delay-within-deadline / run-at-cheapest-period-before-deadline / force-near-deadline
/ protect-serving-capacity / miss-only-with-penalty — and is driven by the same price path the MPC sees.

**Honest statement for this PR:** deferrable shifting is **optimised by a deterministic price-aware policy, not
by the MPC's candidate search.** It is "jointly optimised enough" for this backtest in the sense that serving
(the MPC) runs first and **dominates** — deferrable work only consumes the spare GPU-seconds serving leaves, so
the two never fight for capacity; the deferrable policy then minimises energy cost over that spare under the
SAME real prices. Its value is reported as a **separate energy-cost ledger**, never folded into serving gp/$.
Promoting deferrable shifting into a true `ActionBundle`/MPC-selected knob (so the planner trades serving
capacity vs deferrable value jointly) is **deferred** — it needs the controller to co-schedule serving and
deferrable in one objective, which is out of scope here. We do **not** fake a controller knob for it.

## C. Region shifting — SKIPPED (with reason)

Region eligibility exists on `DeferrableJob` (`region_eligibility`, `region_shiftable`) and a region→market
registry exists (`region_registry.py`), but there is **no multi-region fleet model** (the world state is a
single sampled cluster). Region shifting would therefore be a free cross-region move with no capacity/SLA
consequence on the other region — i.e. a fake saving. **Skipped**; `region_shiftable` is forced `False` and
tested.

## Summary

| control | classification | why |
|--|--|--|
| clock / DVFS | **TRUE MPC action** | selected in `ActionBundle`, priced vs the real price path + roofline + SLA |
| precision / spec-decode / batching / routing / capacity / prewarm / placement / migration | TRUE MPC actions (pre-existing) | live in `ActionBundle` (arm 7) |
| deferrable energy shifting | **scheduler** (planner-visible, serving-dominated) | price-aware policy over serving's spare capacity; separate ledger; NOT MPC-search-selected (documented, not faked) |
| ElectricityState / PowerState / price percentile / spike | **diagnostic** | observed signals + per-decision electricity fields |
| region shifting | **skipped** | no multi-region fleet → would be a fake saving |
