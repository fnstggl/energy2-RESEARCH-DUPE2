"""CostModel — operator-side GPU-fleet economics (owned + leased + sensitivity).

This estimates the cost of OPERATING GPU infrastructure, not of buying cloud
capacity. Two scenarios, never tenant-side:

  * **owned** (primary) — the GPU cost basis is **depreciation** of CapEx over a
    service life (utilization-adjusted), plus PUE-scaled energy at the regional ISO
    price, plus optional network / queue-delay / SLA-penalty costs.
  * **leased / managed** (optional) — a contractual $/GPU-hour rate (an external
    list/contract observation), plus the same energy/network/penalty terms.

Hard prohibition (enforced by construction): there is **no** tenant-side
spot/on-demand/reserved arbitrage or cloud instance-billing optimization here —
those are cloud-customer levers, not operator economics.

Fidelity (per the build spec): electricity price is TRACE_DERIVED (regional ISO);
PUE, GPU acquisition/depreciation, and average power draw are INFERRED public-list
priors (operator telemetry would make them MEASURED); leased rates are
EXTERNAL_OBSERVED; the true internal operator cost model is ABSENT until a pilot.
Every component carries its source + tier, and ``sensitivity()`` exposes how much
each heuristic assumption moves the total (low/base/high bands).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmarks.economics import InfrastructureCostConfig
from .schemas import (
    EXTERNAL_OBSERVED,
    INFERRED,
    TRACE_DERIVED,
    CalibratedParam,
)

_HOURS_PER_YEAR = 8760

# Legacy per-GPU-hour priors (kept for the back-compat ``cost()`` cross-check).
_DEFAULT_DEPRECIATION_PER_GPU_HOUR = {"H100": 1.10, "A100": 0.55, "L40S": 0.35}
_DEFAULT_POWER_KW = {"H100": 0.70, "A100": 0.40, "L40S": 0.35}
_FALLBACK_DEPRECIATION = 0.70
_FALLBACK_POWER_KW = 0.45


# ---------------------------------------------------------------------------
# Owned-hardware economics (public-list priors → INFERRED)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GPUEconomics:
    """Per-GPU-type owned economics: CapEx, service life, active/idle power draw."""

    gpu_type: str
    acquisition_usd: float
    active_power_kw: float
    idle_power_kw: float
    service_life_years: float = 4.0

    def depreciation_per_gpu_hour(self, service_life_years: float | None = None) -> float:
        yrs = service_life_years or self.service_life_years
        return self.acquisition_usd / (yrs * _HOURS_PER_YEAR)

    def power_kw(self, utilization: float = 1.0) -> float:
        u = max(0.0, min(1.0, utilization))
        return self.idle_power_kw + (self.active_power_kw - self.idle_power_kw) * u


OWNED_ECONOMICS = {
    "H100": GPUEconomics("H100", 30000.0, 0.70, 0.10),
    "H800": GPUEconomics("H800", 28000.0, 0.70, 0.10),
    "H20": GPUEconomics("H20", 18000.0, 0.50, 0.08),
    "A100": GPUEconomics("A100", 15000.0, 0.40, 0.06),
    "A800": GPUEconomics("A800", 14000.0, 0.40, 0.06),
    "A10": GPUEconomics("A10", 3000.0, 0.15, 0.03),
    "L40S": GPUEconomics("L40S", 9000.0, 0.35, 0.05),
    "L40": GPUEconomics("L40", 8500.0, 0.30, 0.05),
    "L20": GPUEconomics("L20", 6000.0, 0.28, 0.04),
    "XPU-B": GPUEconomics("XPU-B", 12000.0, 0.40, 0.06),
}
_FALLBACK_ECON = GPUEconomics("generic", 12000.0, 0.45, 0.06)

# Leased / managed contract rates ($/GPU-hour, public list → EXTERNAL_OBSERVED).
LEASE_USD_PER_GPU_HOUR = {
    "H100": 2.50, "H800": 2.40, "H20": 1.80, "A100": 1.50, "A800": 1.40,
    "A10": 0.60, "L40S": 1.00, "L40": 0.95, "L20": 0.80, "XPU-B": 1.20,
}
_FALLBACK_LEASE = 1.50


# ---------------------------------------------------------------------------
# Legacy breakdown (unchanged — back-compat cross-check)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Operator-side breakdown (owned or leased) + per-useful-unit costs
# ---------------------------------------------------------------------------

@dataclass
class OperatorCostBreakdown:
    basis: str                         # "owned_depreciation" | "leased_contract"
    gpu_type: str
    energy_cost: float
    depreciation_cost: float           # owned (0 for leased)
    lease_cost: float                  # leased (0 for owned)
    network_cost: float
    queue_delay_cost: float
    sla_penalty_cost: float

    @property
    def total_operator_cost(self) -> float:
        return sum(max(0.0, x) for x in (
            self.energy_cost, self.depreciation_cost, self.lease_cost,
            self.network_cost, self.queue_delay_cost, self.sla_penalty_cost))

    def cost_per_sla_safe_request(self, n_sla_safe: int) -> float:
        return self.total_operator_cost / n_sla_safe if n_sla_safe else 0.0

    def cost_per_sla_safe_token(self, sla_safe_tokens: float) -> float:
        return self.total_operator_cost / sla_safe_tokens if sla_safe_tokens else 0.0

    def goodput_per_dollar(self, sla_safe_goodput: float) -> float:
        return sla_safe_goodput / self.total_operator_cost if self.total_operator_cost else 0.0

    def to_dict(self, *, n_sla_safe: int = 0, sla_safe_tokens: float = 0.0,
                sla_safe_goodput: float = 0.0) -> dict:
        return {
            "basis": self.basis, "gpu_type": self.gpu_type,
            "energy_cost": round(self.energy_cost, 4),
            "depreciation_cost": round(self.depreciation_cost, 4),
            "lease_cost": round(self.lease_cost, 4),
            "network_cost": round(self.network_cost, 4),
            "queue_delay_cost": round(self.queue_delay_cost, 4),
            "sla_penalty_cost": round(self.sla_penalty_cost, 4),
            "total_operator_cost": round(self.total_operator_cost, 4),
            "cost_per_sla_safe_request": round(self.cost_per_sla_safe_request(n_sla_safe), 6),
            "cost_per_sla_safe_token": round(self.cost_per_sla_safe_token(sla_safe_tokens), 8),
            "goodput_per_dollar": round(self.goodput_per_dollar(sla_safe_goodput), 4),
        }


@dataclass
class CostModel:
    """Operator (owned-hardware default) cost model + leased scenario + sensitivity."""

    pue: float = 1.3
    depreciation_per_gpu_hour: dict = field(
        default_factory=lambda: dict(_DEFAULT_DEPRECIATION_PER_GPU_HOUR))
    power_kw: dict = field(default_factory=lambda: dict(_DEFAULT_POWER_KW))
    rental_cfg: InfrastructureCostConfig = field(default_factory=InfrastructureCostConfig)
    owned: dict = field(default_factory=lambda: dict(OWNED_ECONOMICS))
    lease_usd_per_gpu_hour: dict = field(default_factory=lambda: dict(LEASE_USD_PER_GPU_HOUR))
    service_life_years: float = 4.0
    sla_penalty_per_violation_usd: float = 0.0
    queue_delay_cost_per_s_usd: float = 0.0

    # -- legacy (back-compat cross-check) --------------------------------
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

    # -- operator-side cost (owned or leased) ----------------------------
    def _econ(self, gpu_type: str) -> GPUEconomics:
        return self.owned.get(gpu_type, _FALLBACK_ECON)

    def _lease(self, gpu_type: str) -> float:
        return self.lease_usd_per_gpu_hour.get(gpu_type, _FALLBACK_LEASE)

    def operator_cost(
        self, *, gpu_hours: float, gpu_type: str, energy_price_per_kwh: float,
        utilization: float = 1.0, scenario: str = "owned",
        migrations: int = 0, egress_gb: float = 0.0, sla_violations: int = 0,
        queue_delay_s: float = 0.0,
        pue: float | None = None, service_life_years: float | None = None,
        acquisition_scale: float = 1.0, power_scale: float = 1.0,
        electricity_scale: float = 1.0,
    ) -> OperatorCostBreakdown:
        """Operator cost for ``gpu_hours`` of one GPU type. ``*_scale`` knobs feed
        the sensitivity sweep; defaults reproduce the base case."""
        pue = (pue if pue is not None else self.pue)
        econ = self._econ(gpu_type)
        power = econ.power_kw(utilization) * power_scale
        energy = gpu_hours * power * pue * (energy_price_per_kwh * electricity_scale)
        network = (migrations * self.rental_cfg.network_cost_per_migration
                   + egress_gb * self.rental_cfg.network_cost_per_gb_egress)
        queue = queue_delay_s * self.queue_delay_cost_per_s_usd
        sla = sla_violations * self.sla_penalty_per_violation_usd
        if scenario == "leased":
            lease = gpu_hours * self._lease(gpu_type)
            return OperatorCostBreakdown("leased_contract", gpu_type, energy, 0.0, lease,
                                         network, queue, sla)
        dep = (gpu_hours * econ.depreciation_per_gpu_hour(service_life_years)
               * acquisition_scale)
        return OperatorCostBreakdown("owned_depreciation", gpu_type, energy, dep, 0.0,
                                     network, queue, sla)

    # -- sensitivity bands (make the heuristic assumptions visible) ------
    def sensitivity(
        self, *, gpu_hours: float, gpu_type: str, energy_price_per_kwh: float,
        utilization: float = 1.0, scenario: str = "owned", factors: dict | None = None,
    ) -> dict:
        """Low/base/high band for total operator cost, per heuristic assumption and
        combined. Each factor scales one input; the band shows how much the total
        moves when that assumption is wrong."""
        f = factors or {
            "pue": (0.92, 1.0, 1.15),
            "acquisition": (0.70, 1.0, 1.50),      # CapEx
            "electricity": (0.50, 1.0, 2.00),
            "service_life": (1.50, 1.0, 0.75),     # longer life → LOWER dep (inverted band)
            "power_draw": (0.85, 1.0, 1.20),
            "utilization": (0.50, 1.0, 1.0),
        }

        def _total(**kw):
            kw.setdefault("utilization", utilization)
            return self.operator_cost(
                gpu_hours=gpu_hours, gpu_type=gpu_type,
                energy_price_per_kwh=energy_price_per_kwh,
                scenario=scenario, **kw).total_operator_cost

        base = _total()
        per_factor = {}
        for name, (lo, _b, hi) in f.items():
            per_factor[name] = {
                "low": round(_total(**self._scale_kwargs(name, lo, utilization)), 4),
                "high": round(_total(**self._scale_kwargs(name, hi, utilization)), 4),
            }
        # combined extremes (all factors simultaneously low / high)
        low_kw, high_kw = {}, {}
        for name, (lo, _b, hi) in f.items():
            low_kw.update(self._scale_kwargs(name, lo, utilization))
            high_kw.update(self._scale_kwargs(name, hi, utilization))
        return {
            "scenario": scenario, "gpu_type": gpu_type,
            "base_total": round(base, 4),
            "low_total": round(_total(**low_kw), 4),
            "high_total": round(_total(**high_kw), 4),
            "per_factor": per_factor,
        }

    def _scale_kwargs(self, factor: str, mult: float, utilization: float) -> dict:
        """Map one sensitivity factor + multiplier → an ``operator_cost`` kwarg."""
        if factor == "pue":
            return {"pue": self.pue * mult}
        if factor == "acquisition":
            return {"acquisition_scale": mult}
        if factor == "electricity":
            return {"electricity_scale": mult}
        if factor == "service_life":
            return {"service_life_years": self.service_life_years * mult}
        if factor == "power_draw":
            return {"power_scale": mult}
        if factor == "utilization":
            return {"utilization": max(0.0, min(1.0, utilization * mult))}
        return {}

    # -- provenance ------------------------------------------------------
    def params(self, gpu_type: str = "H100", scenario: str = "owned") -> list:
        econ = self._econ(gpu_type)
        base = [
            CalibratedParam(
                "energy_price_per_kwh", "from FleetPlane (ISO)", "iso", "electricity.price",
                "regional hour-of-day", "n/a", "n/a", TRACE_DERIVED,
                "regional marginal price series", True),
            CalibratedParam(
                "pue", self.pue, "engineering/EIA", "modeled", "public-list prior", "n/a", "n/a",
                INFERRED, "cooling/overhead; operator telemetry would make it MEASURED", False),
            CalibratedParam(
                "gpu_acquisition_usd", econ.acquisition_usd, "public-list", f"capex[{gpu_type}]",
                "list price", "n/a", "n/a", INFERRED,
                "owned-hardware CapEx; operator contract → MEASURED", False),
            CalibratedParam(
                "gpu_depreciation_per_gpu_hour",
                round(econ.depreciation_per_gpu_hour(self.service_life_years), 4),
                "derived", f"capex[{gpu_type}]/(service_life·8760)", "straight-line amortization",
                "n/a", "n/a", INFERRED, f"{self.service_life_years}y life; operator schedule → MEASURED", False),
            CalibratedParam(
                "gpu_power_kw_active", econ.active_power_kw, "vendor TDP", f"power[{gpu_type}]",
                "rated TDP", "n/a", "n/a", INFERRED,
                "avg draw inferred from TDP; Zeus/DCGM → MEASURED", False),
        ]
        if scenario == "leased":
            base.append(CalibratedParam(
                "leased_usd_per_gpu_hour", self._lease(gpu_type), "public-list lease",
                f"lease[{gpu_type}]", "external contract list rate", "n/a", "n/a",
                EXTERNAL_OBSERVED, "external observed lease rate, not operator-measured", False))
        return base


__all__ = ["CostModel", "CostBreakdown", "OperatorCostBreakdown", "GPUEconomics",
           "OWNED_ECONOMICS", "LEASE_USD_PER_GPU_HOUR"]
