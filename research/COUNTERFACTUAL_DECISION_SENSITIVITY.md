# Counterfactual Decision Sensitivity

How fragile is each MPC decision? If the winning action bundle barely edged out the runner-up, a tiny forecast
or simulator perturbation would flip it — that decision is *fragile* and worth flagging. This is the **online**
robustness signal in the Decision Diagnostics Engine: `counterfactual_sensitivity` + the `decision_margin`
fields on every `DecisionExplanation` (`aurelius/environment/decision_diagnostics.py`).

## Online methodology — negligible overhead, no re-solves

The online tier exposes **only values the search already computed**. No counterfactual re-planning, no
leave-one-out, no oracle reruns online — those are strictly offline. The robustness signal is read straight off
the already-scored candidate list the planner produced:

- **decision margin** = winner reward − runner-up reward; **`decision_margin_pct`** = margin / |winner|.
- **robustness_score** = the normalised margin (winner→runner gap relative to the winner), the cheap proxy for
  "how far would inputs have to move to flip this decision."
- **stable** = margin_pct above a small threshold. A near-tie ⇒ `stable=False` ⇒ the decision is fragile and
  several bundles are effectively interchangeable.

Cost is a list-append per scored candidate plus one sort — measured at **0.48 s end-to-end** on a real Azure
decision, i.e. negligible vs the MPC solve itself.

## Validated example — a fragile decision, surfaced honestly

On the validated Azure decision the winner came out essentially tied with the field:

```
decision_margin ≈ 0   decision_margin_pct ≈ 0   stable = False   (≈207 candidates evaluated)
```

Several bundles scored within rounding of the winner → the engine reports `stable=False` rather than
pretending the choice was decisive. This is the *honest* output: when the gp/$ surface is flat, the diagnostics
say so instead of manufacturing confidence. Contrast a clearly-separated decision (winner 180 vs runner 150 →
`margin_pct ≈ 0.17`, `stable=True`): the same machinery reports it as robust.

## Local switching thresholds (also online)

For the surfaces where the winner differs from the runner-up (the `why_won` set), the engine reports the
decision margin as the **cheap robustness proxy** — "this is how much head-room the winning precision/spec/clock
choice had." Precise per-variable switching thresholds (the exact forecast value at which the decision flips)
require re-solving under perturbed inputs and are therefore an **offline** analysis, not paid online.

## What this is and isn't

- **Is:** a permanent, per-decision fragility flag the controller emits alongside every action, at no
  meaningful runtime cost, derived purely from already-computed scores.
- **Isn't:** a counterfactual *re-plan*. We deliberately do not run perturbed solves online; when a precise
  flip-point is needed it is computed offline (`scripts/diagnose_mpc_attribution.py`). Tests:
  `tests/test_decision_diagnostics.py::test_decision_margin_and_robustness`.

Diagnostic / observability only — this changes **no** controller decision; it explains the decision the
controller already made.
