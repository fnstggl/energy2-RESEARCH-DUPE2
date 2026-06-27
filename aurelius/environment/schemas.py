"""Canonical environment — shared data contracts (built from first principles).

The clean schemas the multi-plane environment is built on. Every value the
environment emits is provenance-tagged via :class:`CalibratedParam` /
:class:`SignalProvenance`, so an assumption can never be laundered as data.

Fidelity ladder (best → worst), per the build spec:
  MEASURED          — measured on real hardware / operator telemetry
  TRACE_DERIVED     — fitted from a real public-trace distribution (this repo)
  BENCHMARK_DERIVED — a public benchmark/paper number
  INFERRED          — reasoned from a documented mechanism (no number)
  HEURISTIC         — engineering guess (must be calibrated before any claim)
  ABSENT            — no public source; structurally proprietary (pilot only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MEASURED = "MEASURED"
TRACE_DERIVED = "TRACE_DERIVED"
BENCHMARK_DERIVED = "BENCHMARK_DERIVED"
EXTERNAL_OBSERVED = "EXTERNAL_OBSERVED"   # a public external list/contract rate (not operator-measured)
INFERRED = "INFERRED"
HEURISTIC = "HEURISTIC"
ABSENT = "ABSENT"

TIER_ORDER = {
    MEASURED: 0, TRACE_DERIVED: 1, BENCHMARK_DERIVED: 2,
    EXTERNAL_OBSERVED: 3, INFERRED: 4, HEURISTIC: 5, ABSENT: 6,
}
# A value is trustworthy for a headline only at these tiers.
HEADLINE_SAFE_TIERS = frozenset({MEASURED, TRACE_DERIVED})


@dataclass(frozen=True)
class CalibratedParam:
    """One environment parameter, with the full provenance the build spec requires."""

    name: str
    value: Any                      # float | int | dict | list
    source_dataset: str             # e.g. "azure_llm_2024", "alibaba_gpu_v2026", "mooncake", "iso_caiso"
    table_column: str               # e.g. "pod_hourly.avg_gpu_sm_util"
    fitting_method: str             # e.g. "mean", "p95", "online/offline ratio", "prefix-overlap"
    train_holdout_split: str        # e.g. "first 70% by time / last 30% holdout"
    trace_version: str              # e.g. "v2026", "2024-11"
    tier: str                       # one of the fidelity tiers
    limitations: str = ""
    safe_for_headline: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name, "value": self.value,
            "source_dataset": self.source_dataset, "table_column": self.table_column,
            "fitting_method": self.fitting_method,
            "train_holdout_split": self.train_holdout_split,
            "trace_version": self.trace_version, "tier": self.tier,
            "limitations": self.limitations, "safe_for_headline": self.safe_for_headline,
        }


@dataclass(frozen=True)
class SignalProvenance:
    """Provenance record for one signal the environment emits (manifest entry)."""

    name: str
    source: str
    table_column: str
    tier: str
    method: str
    limitations: str = ""
    safe_for_headline: bool = False

    @classmethod
    def from_param(cls, p: CalibratedParam) -> "SignalProvenance":
        return cls(name=p.name, source=p.source_dataset, table_column=p.table_column,
                   tier=p.tier, method=p.fitting_method, limitations=p.limitations,
                   safe_for_headline=p.safe_for_headline)

    def to_dict(self) -> dict:
        return {
            "name": self.name, "source": self.source, "table_column": self.table_column,
            "tier": self.tier, "method": self.method, "limitations": self.limitations,
            "safe_for_headline": self.safe_for_headline,
        }


@dataclass
class FleetState:
    """Hourly fleet-plane state, derived from the v2026 distributions."""

    hour: int
    total_gpus: int
    gpu_type_inventory: dict          # {"H100": 16, "A100": 16}
    gpu_type_mix: dict                # {"H100": 0.5, "A100": 0.5}
    util_target: float                # mean avg_gpu_sm_util (0..1)
    util_by_class: dict               # {"HP": .., "LP": ..}
    mem_pressure: float               # mean avg_memory_util (0..1)
    priority_mix: dict                # {"HP": .., "LP": .., "Other": ..}
    best_effort_fraction: float       # offline/(online+offline) inference
    queue_delay_s: float              # mean schedule_delay_sec
    ready_delay_s: float              # mean ready_delay_sec
    rack_locality: dict               # asw_id -> gpu_count
    net_pressure: float               # normalized rx+tx (0..1)
    capacity_envelope: int            # max GPUs provisionable for serving this hour
    fragmentation: float              # stranded/packed proxy (0..1)
    energy_price_per_kwh: float
    region: str
    fidelity: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "hour": self.hour, "total_gpus": self.total_gpus,
            "gpu_type_mix": self.gpu_type_mix, "util_target": round(self.util_target, 4),
            "mem_pressure": round(self.mem_pressure, 4), "priority_mix": self.priority_mix,
            "best_effort_fraction": round(self.best_effort_fraction, 4),
            "queue_delay_s": round(self.queue_delay_s, 4),
            "net_pressure": round(self.net_pressure, 4),
            "capacity_envelope": self.capacity_envelope,
            "fragmentation": round(self.fragmentation, 4),
            "energy_price_per_kwh": round(self.energy_price_per_kwh, 5),
            "region": self.region,
        }

    def to_dict(self) -> dict:
        d = self.summary()
        d.update({"gpu_type_inventory": self.gpu_type_inventory,
                  "util_by_class": self.util_by_class,
                  "ready_delay_s": round(self.ready_delay_s, 4),
                  "rack_locality": self.rack_locality, "fidelity": self.fidelity})
        return d


@dataclass
class ServingRequest:
    """A token-level serving request on the per-second serving plane (Azure spine)."""

    idx: int
    arrival_s: float
    tokens: int
    predicted_tokens: float
    cls: str                          # "latency_critical" | "best_effort"
    kv_prefix_id: str = ""            # block-prefix id (Mooncake) for KV routing
    kv_reuse_prob: float = 0.0        # calibrated prefix-hit probability
    kv_service_factor: float = 1.0    # stateful-KV service-time multiplier (≤1 on a hit)
    kv_tokens_saved: int = 0          # prefill tokens skipped by a KV hit


@dataclass
class EnvObservation:
    """What a policy/optimizer sees at an hourly decision boundary."""

    hour: int
    fleet: dict
    n_requests: int
    arrival_rate_per_s: float
    best_effort_fraction: float

    def to_dict(self) -> dict:
        return {"hour": self.hour, "fleet": self.fleet, "n_requests": self.n_requests,
                "arrival_rate_per_s": round(self.arrival_rate_per_s, 4),
                "best_effort_fraction": round(self.best_effort_fraction, 4)}


@dataclass
class EnvStep:
    """One (observation, action, reward, metrics) tuple — the training/eval unit."""

    hour: int
    observation: dict
    action: dict
    reward: float                     # SLA-safe goodput per dollar
    metrics: dict

    def to_dict(self) -> dict:
        return {"hour": self.hour, "observation": self.observation,
                "action": self.action, "reward": round(self.reward, 4),
                "metrics": self.metrics}


__all__ = [
    "MEASURED", "TRACE_DERIVED", "BENCHMARK_DERIVED", "EXTERNAL_OBSERVED",
    "INFERRED", "HEURISTIC", "ABSENT", "TIER_ORDER", "HEADLINE_SAFE_TIERS",
    "CalibratedParam", "SignalProvenance", "FleetState", "ServingRequest",
    "EnvObservation", "EnvStep",
]
