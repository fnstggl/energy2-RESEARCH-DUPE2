"""A clock-aligned forecast TRAJECTORY over the MPC planning horizon.

The receding-horizon MPC consumes a forecast for every future step it rolls out, not just the next
one. This wraps the existing :class:`ForecastingModel` (which already predicts ``H`` steps ahead,
feeding its own predictions forward — causal, no future truth) into a per-step trajectory aligned to
the :class:`SimulationClock`, and exposes uncertainty honestly:

- **deterministic** mode → the point (mean/p50) path; the only mode when quantiles are absent;
- **quantile** modes (pessimistic / optimistic) → the p90 / p10 paths for risk-aware rollouts,
  available only for targets whose forecaster emits real quantiles (else marked ABSENT, never faked);
- **ensemble** mode is ABSENT here (no probabilistic sampler in the repo) — wrapped, not invented.

No labels from the evaluation trace ever enter the trajectory: it is built from ``history`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

# targets the rollout consumes; the rest of the bundle is carried for diagnostics/provenance.
ROLLOUT_TARGETS = ("arrival_rate", "output_token_mean", "output_token_p95", "interarrival_cv",
                   "electricity_price")
_QUANTILE = {"deterministic": "mean", "median": "p50", "pessimistic": "p90", "optimistic": "p10"}


@dataclass
class ForecastTrajectory:
    horizon_steps: int
    dt_seconds: float
    bundle: object                      # the ForecastBundle (H points per target)
    uncertainty_mode: str = "deterministic"
    t0_period_index: int = 0

    # -- per-step access -----------------------------------------------------
    def _q(self, mode: str | None) -> str:
        return _QUANTILE.get(mode or self.uncertainty_mode, "mean")

    def at(self, target: str, step: int, *, mode: str | None = None):
        """Forecast scalar for ``target`` at future ``step`` (0-based) under the quantile mode.
        Falls back to the mean if the requested quantile is absent for this target."""
        pt = self.bundle.at(target, step)
        if pt is None:
            return 0.0
        return getattr(pt, self._q(mode), pt.mean)

    def step_forecast(self, step: int, *, mode: str | None = None) -> dict:
        """All rollout targets at ``step`` as a plain dict (what one rollout timestep consumes)."""
        return {t: self.at(t, step, mode=mode) for t in ROLLOUT_TARGETS}

    def point(self, target: str, step: int):
        return self.bundle.at(target, step)

    def has_uncertainty(self, target: str) -> bool:
        """True iff this target's forecaster emits a real (non-degenerate) quantile band."""
        pt = self.bundle.at(target, 0)
        if pt is None:
            return False
        return not (pt.p10 == pt.p90 == pt.mean) and pt.fidelity not in ("ABSENT", "RUNNING_STATISTIC")

    def uncertainty_manifest(self) -> dict:
        """Per-target uncertainty availability + provenance — honest about what is ABSENT."""
        out = {}
        for t in ROLLOUT_TARGETS:
            pt = self.bundle.at(t, 0)
            out[t] = {"has_quantiles": self.has_uncertainty(t),
                      "fidelity": (pt.fidelity if pt else "ABSENT")}
        return out

    def to_dict(self) -> dict:
        return {"horizon_steps": self.horizon_steps, "dt_seconds": self.dt_seconds,
                "uncertainty_mode": self.uncertainty_mode, "ensemble_scenarios": "ABSENT",
                "uncertainty": self.uncertainty_manifest(),
                "path": [self.step_forecast(k) for k in range(self.horizon_steps)]}


def build_trajectory(forecasters, history, clock, horizon_steps, *, mode="deterministic"):
    """Build a causal ``ForecastTrajectory`` from the fitted model + history (no future truth).
    The forecaster predicts ``horizon_steps`` ahead recursively from ``history`` alone."""
    bundle = forecasters.predict(history, horizon=max(1, horizon_steps))
    return ForecastTrajectory(horizon_steps=max(1, horizon_steps), dt_seconds=clock.dt_seconds,
                              bundle=bundle, uncertainty_mode=mode,
                              t0_period_index=clock.period_index)


__all__ = ["ForecastTrajectory", "build_trajectory", "ROLLOUT_TARGETS"]
