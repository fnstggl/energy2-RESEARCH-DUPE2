"""ForecastState — the canonical, persistent record of *planner belief* vs *realized outcome*.

The planner already forecasts arrivals, token lengths, electricity price, etc. (`forecast_trajectory`,
`scenario_forecaster`). What it has NOT had is a first-class persistent state that preserves **what the
controller believed at each decision** and **what actually happened**, so that forecast error, the oracle gap,
and per-variable regret can be attributed causally.

ForecastState does NOT generate forecasts (no new model, no duplicated forecast code). It **references** the
forecaster outputs the controller already computed and stores them as evolving state. It is a *belief record,
not a reward term* — nothing here changes the MPC objective or the Pareto gate.

Causality / honesty rules (validated in `state_validation.py`):
  * a belief is recorded BEFORE the period runs (`record_belief`); the realized outcome and the error are
    recorded only AFTER realization (`record_realized`) — forecast error can never be computed from the future;
  * a record's `forecast_error` stays None until its realized value is known (no leakage);
  * provenance + confidence + horizon are preserved per record;
  * clone-safe (plain dataclasses → `copy.deepcopy`), so it rides on `CanonicalWorldState.clone()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# the belief variables we persist (a superset of decision_diagnostics.CONSUMED_FORECASTS + emergent pressures
# that are realized-only today — kept so the path to forecasting them is auditable)
BELIEF_VARS = ("arrival_rate", "interarrival_cv", "prompt_length", "output_token_mean", "output_token_p95",
               "electricity_price", "queue_pressure", "sla_pressure")


@dataclass
class ForecastRecord:
    """One forecast belief for one target period, plus its realized outcome + error once known."""
    decision_index: int
    target_period: int                       # the period this belief is about
    made_at_period: int                      # the period the controller was standing in (≤ target_period)
    horizon_index: int                       # k in the rollout (0 = next step)
    belief: dict = field(default_factory=dict)     # var -> forecast scalar (what the planner believed)
    provenance: str = "FORECAST_DERIVED"
    confidence: float = 0.0
    uncertainty: dict = field(default_factory=dict)   # var -> {p10,p90,fidelity} when available
    realized: dict | None = None             # var -> realized scalar (filled AFTER the period runs)
    forecast_error: dict | None = None       # var -> (forecast - realized); None until realized
    oracle: dict | None = None               # var -> exact-future value (oracle arm only)
    realized_at_period: int | None = None

    @property
    def is_realized(self) -> bool:
        return self.realized is not None

    def to_dict(self) -> dict:
        return {"decision_index": self.decision_index, "target_period": self.target_period,
                "made_at_period": self.made_at_period, "horizon_index": self.horizon_index,
                "belief": {k: round(v, 5) for k, v in self.belief.items()},
                "provenance": self.provenance, "confidence": round(self.confidence, 4),
                "realized": ({k: round(v, 5) for k, v in self.realized.items()} if self.realized else None),
                "forecast_error": ({k: round(v, 5) for k, v in self.forecast_error.items()}
                                   if self.forecast_error else None),
                "oracle": ({k: round(v, 5) for k, v in self.oracle.items()} if self.oracle else None)}


@dataclass
class ForecastState:
    """Persistent, clone-safe record of every planner belief and its realized error. Canonical owner of
    'what the planner believed' across all MPC rollouts."""
    records: list = field(default_factory=list)      # ForecastRecord, oldest first
    horizon_steps: int = 0
    n_decisions: int = 0

    def record_belief(self, *, decision_index: int, target_period: int, made_at_period: int,
                      horizon_index: int, belief: dict, provenance: str = "FORECAST_DERIVED",
                      confidence: float = 0.0, uncertainty: dict | None = None) -> ForecastRecord:
        """Record what the planner believes about `target_period`, BEFORE it runs. No future truth here."""
        rec = ForecastRecord(decision_index=decision_index, target_period=int(target_period),
                             made_at_period=int(made_at_period), horizon_index=int(horizon_index),
                             belief={k: float(v) for k, v in belief.items() if v is not None},
                             provenance=provenance, confidence=float(confidence),
                             uncertainty=dict(uncertainty or {}))
        self.records.append(rec)
        return rec

    def record_realized(self, target_period: int, realized: dict, *, at_period: int | None = None) -> int:
        """Fill the realized outcome + forecast error for every belief about `target_period`. Called AFTER the
        period runs — this is the only place error is computed (causal: realized is known). Returns #updated."""
        realized = {k: float(v) for k, v in realized.items() if v is not None}
        n = 0
        for rec in self.records:
            if rec.target_period == int(target_period) and rec.realized is None:
                rec.realized = realized
                rec.forecast_error = {k: round(rec.belief[k] - realized[k], 6)
                                      for k in rec.belief if k in realized}
                rec.realized_at_period = int(at_period if at_period is not None else target_period)
                n += 1
        return n

    def record_oracle(self, target_period: int, oracle: dict) -> int:
        """Attach the exact-future (oracle) value for a target period (oracle arm only)."""
        oracle = {k: float(v) for k, v in oracle.items() if v is not None}
        n = 0
        for rec in self.records:
            if rec.target_period == int(target_period):
                rec.oracle = oracle
                n += 1
        return n

    # -- diagnostics ---------------------------------------------------------
    def realized_records(self) -> list:
        return [r for r in self.records if r.is_realized]

    def forecast_error_summary(self) -> dict:
        """Per-variable mean absolute error + mean absolute percentage error over realized records."""
        out: dict = {}
        for var in BELIEF_VARS:
            errs, pcts = [], []
            for r in self.realized_records():
                if r.forecast_error and var in r.forecast_error and r.realized and var in r.realized:
                    errs.append(abs(r.forecast_error[var]))
                    denom = abs(r.realized[var])
                    if denom > 1e-9:
                        pcts.append(abs(r.forecast_error[var]) / denom)
            if errs:
                out[var] = {"mae": round(sum(errs) / len(errs), 5), "n": len(errs),
                            "mape_pct": round(100.0 * sum(pcts) / len(pcts), 3) if pcts else None}
        return out

    def to_dict(self, *, max_records: int = 200) -> dict:
        return {"n_records": len(self.records), "n_decisions": self.n_decisions,
                "horizon_steps": self.horizon_steps, "n_realized": len(self.realized_records()),
                "forecast_error_summary": self.forecast_error_summary(),
                "records": [r.to_dict() for r in self.records[:max_records]]}


__all__ = ["ForecastState", "ForecastRecord", "BELIEF_VARS"]
