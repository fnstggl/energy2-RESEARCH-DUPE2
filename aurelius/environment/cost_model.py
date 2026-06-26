"""CostModel — GPU-hours / energy / network → dollars (built from first principles).

No generic "GPU-hour cost." Cost is always resolved by **GPU type**, **region**,
**PUE**, and explicit **depreciation/amortization**, per the build spec. For an
operator ICP (owned hardware) the GPU cost basis is **depreciation**, not cloud
rental — so the default accounting is ``depreciation + energy(+network)``; the
cloud-rental basis (public list priors from ``economics``) is exposed as a
cross-check, not the default.

Every cost knob carries a fidelity tier. Depreciation, power draw and PUE are
modeled HEURISTIC assumptions (operator contracts would make them MEASURED); the
electricity price is TRACE_DERIVED from the regional ISO series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmarks.economics import InfrastructureCostConfig
from .schemas import HEURISTIC, TRACE_DERIVED, CalibratedParam

# Public-list priors (NOT operator contract rates) — per provisioned GPU-hour.
_DEFAULT_DEPRECIATION_PER_GPU_HOUR = {"H100": 1.10, "A100": 0.55, "L40S": 0.35}
_DEFAULT_POWER_KW = {"H100": 0.70, "A100": 0.40, "L40S": 0.35}
_FALLBACK_DEPRECIATION = 0.70
_FALLBACK_POWER_KW = 0.45


@dataclass
class CostBreakdown:
    gpu_depreciation_cost: float
    energy_cost: float
    network_cost: float
    rental_cross_check: float          # cloud-rental basis (not in total)

    @property
    def total(self) -> float:
        return (max(0.0, self.gpu_depreciation_cost) + max(0.0, self.energy_cost)
                + max(0.0, self.network_cost))

    def to_dict(self) -> dict:
        return {
            "gpu_depreciation_cost": round(self.gpu_depreciation_cost, 4),
            "energy_cost": round(self.energy_cost, 4),
            "network_cost": round(self.network_cost, 4),
            "total": round(self.total, 4),
            "rental_cross_check": round(self.rental_cross_check, 4),
        }


@dataclass
class CostModel:
    """Operator (owned-hardware) cost model: depreciation + energy(PUE) + network."""

    pue: float = 1.3
    depreciation_per_gpu_hour: dict = field(
        default_factory=lambda: dict(_DEFAULT_DEPRECIATION_PER_GPU_HOUR))
    power_kw: dict = field(default_factory=lambda: dict(_DEFAULT_POWER_KW))
    rental_cfg: InfrastructureCostConfig = field(default_factory=InfrastructureCostConfig)

    def _dep(self, gpu_type: str) -> float:
        return self.depreciation_per_gpu_hour.get(gpu_type, _FALLBACK_DEPRECIATION)

    def _pwr(self, gpu_type: str) -> float:
        return self.power_kw.get(gpu_type, _FALLBACK_POWER_KW)

    def cost(
        self, *, gpu_hours: float, gpu_type: str, energy_price_per_kwh: float,
        migrations: int = 0, egress_gb: float = 0.0,
    ) -> CostBreakdown:
        dep = gpu_hours * self._dep(gpu_type)
        energy_kwh = gpu_hours * self._pwr(gpu_type) * self.pue
        energy = energy_kwh * energy_price_per_kwh
        network = (migrations * self.rental_cfg.network_cost_per_migration
                   + egress_gb * self.rental_cfg.network_cost_per_gb_egress)
        rental = gpu_hours * self.rental_cfg.gpu_price(gpu_type)
        return CostBreakdown(dep, energy, network, rental)

    def params(self, gpu_type: str = "H100") -> list:
        return [
            CalibratedParam(
                "pue", self.pue, "engineering/EIA", "modeled", "public-list prior",
                "n/a", "n/a", HEURISTIC, "cooling/overhead; operator-measured in pilot", False),
            CalibratedParam(
                "gpu_depreciation_per_gpu_hour", self._dep(gpu_type), "public-list",
                f"capex_amort[{gpu_type}]", "list-price amortization", "n/a", "n/a",
                HEURISTIC, "owned-hardware capex; operator contract makes it MEASURED", False),
            CalibratedParam(
                "power_kw", self._pwr(gpu_type), "vendor TDP", f"power[{gpu_type}]",
                "rated TDP × avg load", "n/a", "n/a", HEURISTIC,
                "avg draw; Zeus/DCGM would make it BENCHMARK_DERIVED/MEASURED", False),
            CalibratedParam(
                "energy_price_per_kwh", "from FleetPlane (ISO)", "iso", "electricity.price",
                "regional hour-of-day", "n/a", "n/a", TRACE_DERIVED,
                "regional marginal price series", True),
        ]


__all__ = ["CostModel", "CostBreakdown"]
