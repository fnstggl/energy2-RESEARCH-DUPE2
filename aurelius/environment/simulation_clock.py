"""Explicit simulation clock for the receding-horizon MPC.

The MPC horizon is expressed in **simulation steps**, never hours. ``H=4`` means four future
simulation steps; the real lookahead duration is ``H × dt_seconds`` — 20 minutes at ``dt=300s``,
4 hours at ``dt=3600s``. Nothing in the controller may assume ``H`` is hours (see the horizon
analysis doc). This clock carries ``dt_seconds`` + the current step index and converts between a
step horizon and a real-time lookahead, plus per-plane native-resolution metadata so a coarse plane
(electricity, hourly) and a fine plane (serving, sub-minute) can be reconciled honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Native resolution of each canonical plane (seconds) — documented provenance, not assumptions
# pulled from thin air: the serving spine is Azure per-request (sub-second, binned), the fleet /
# electricity planes are hourly marginals.
PLANE_NATIVE_DT = {
    "serving": 1.0,          # Azure LLM trace: per-request arrivals (binned to dt for control)
    "fleet": 3600.0,         # v2026 pod/server/network: hourly marginals
    "electricity": 3600.0,   # hourly price/carbon
    "kv": 1.0,               # Mooncake per-request prefix reuse
}

# Control intervals the serving/control loop must support (seconds).
SUPPORTED_CONTROL_DT = (60.0, 300.0, 900.0, 3600.0)


@dataclass
class SimulationClock:
    """A discrete simulation clock. ``period_index`` counts control steps from ``t0``."""
    dt_seconds: float = 3600.0
    period_index: int = 0
    t0_unix: float | None = None             # wall-clock anchor if available (else None)
    plane_native_dt: dict = field(default_factory=lambda: dict(PLANE_NATIVE_DT))

    def __post_init__(self):
        if self.dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive")

    # -- current time --------------------------------------------------------
    @property
    def current_time_s(self) -> float:
        """Seconds since t0 at the current step."""
        return self.period_index * self.dt_seconds

    @property
    def wall_clock_unix(self) -> float | None:
        return None if self.t0_unix is None else self.t0_unix + self.current_time_s

    def advance(self, steps: int = 1) -> "SimulationClock":
        """Step the clock forward ``steps`` control intervals (receding-horizon execution)."""
        self.period_index += int(steps)
        return self

    def clone(self) -> "SimulationClock":
        return SimulationClock(dt_seconds=self.dt_seconds, period_index=self.period_index,
                               t0_unix=self.t0_unix, plane_native_dt=dict(self.plane_native_dt))

    # -- horizon <-> real-time conversion (THE point of this class) ----------
    def lookahead_seconds(self, horizon_steps: int) -> float:
        return horizon_steps * self.dt_seconds

    def lookahead_minutes(self, horizon_steps: int) -> float:
        return self.lookahead_seconds(horizon_steps) / 60.0

    def lookahead_hours(self, horizon_steps: int) -> float:
        return self.lookahead_seconds(horizon_steps) / 3600.0

    def step_times_s(self, horizon_steps: int) -> list:
        """Absolute (since-t0) seconds at each of the H future step boundaries."""
        return [(self.period_index + 1 + k) * self.dt_seconds for k in range(horizon_steps)]

    def plane_steps_per_control_step(self, plane: str) -> float:
        """How many native plane intervals fit in one control step (≥1 = control is coarser)."""
        native = self.plane_native_dt.get(plane, self.dt_seconds)
        return self.dt_seconds / native if native > 0 else 1.0

    def horizon_meta(self, horizon_steps: int) -> dict:
        """The clock facts the controller must report for a decision."""
        return {"dt_seconds": self.dt_seconds, "horizon_steps": horizon_steps,
                "lookahead_seconds": self.lookahead_seconds(horizon_steps),
                "lookahead_minutes": round(self.lookahead_minutes(horizon_steps), 4),
                "lookahead_hours": round(self.lookahead_hours(horizon_steps), 5),
                "period_index": self.period_index}


__all__ = ["SimulationClock", "PLANE_NATIVE_DT", "SUPPORTED_CONTROL_DT"]
