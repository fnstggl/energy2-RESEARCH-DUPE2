"""FidelityManifest — make it impossible to launder an assumption as data.

Aggregates a :class:`SignalProvenance` record for every signal the environment
emits (name / source / table-column / tier / method / limitations /
safe-for-headline), lists the structurally-proprietary signals as explicitly
ABSENT, and exposes the honesty gate: the environment is never "production grade"
while any emitted signal is below TRACE_DERIVED.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schemas import (
    ABSENT,
    HEADLINE_SAFE_TIERS,
    SignalProvenance,
)

# Structurally-proprietary signals — no public source at any fidelity (pilot only).
ABSENT_PROPRIETARY = (
    SignalProvenance("user_operator_intent", "—", "—", ABSENT,
                     "pilot: customer tier/SLA/deadline/willingness-to-wait",
                     "v2026 priority_class is only a crude proxy", False),
    SignalProvenance("hardware_health", "—", "—", ABSENT,
                     "pilot: DCGM ECC/throttle/fan/degraded",
                     "the thermal blind spot; not in any public trace", False),
    SignalProvenance("live_kv_memory_state", "—", "—", ABSENT,
                     "pilot: per-instance KV residency/eviction",
                     "Mooncake gives hit RATE, not live memory pressure", False),
    SignalProvenance("migration_rejection_reasons", "—", "—", ABSENT,
                     "pilot: scheduler decision log",
                     "traces record WHAT happened, not WHY", False),
    SignalProvenance("internal_cost_model", "—", "—", ABSENT,
                     "pilot: operator contract rates",
                     "depreciation/PUE/power are public-list priors, not contracts", False),
)

FRAMING = (
    "Production-LIKE multi-plane environment grounded by real public traces "
    "(Azure serving spine + Mooncake KV + Alibaba v2026 fleet + ISO electricity), "
    "every signal fidelity-tagged. NOT real production telemetry. Production-grade "
    "only when a pilot replaces the ABSENT tier with operator telemetry."
)


@dataclass
class FidelityManifest:
    signals: list                      # list[SignalProvenance] (present signals)
    absent: tuple = ABSENT_PROPRIETARY
    framing: str = FRAMING

    @classmethod
    def from_params(cls, params: list) -> "FidelityManifest":
        return cls(signals=[SignalProvenance.from_param(p) for p in params])

    def is_production_grade(self) -> bool:
        """Honesty gate: never True while any present signal is below TRACE_DERIVED,
        and never True while the ABSENT proprietary tier is unfilled."""
        return bool(self.signals) and all(
            s.tier in HEADLINE_SAFE_TIERS for s in self.signals) and not self.absent

    def headline_safe_signals(self) -> list:
        return [s.name for s in self.signals if s.safe_for_headline]

    def to_dict(self) -> dict:
        return {
            "signals": [s.to_dict() for s in self.signals],
            "absent_proprietary": [s.to_dict() for s in self.absent],
            "framing": self.framing,
            "is_production_grade": self.is_production_grade(),
            "headline_safe_signals": self.headline_safe_signals(),
        }


__all__ = ["FidelityManifest", "ABSENT_PROPRIETARY", "FRAMING"]
