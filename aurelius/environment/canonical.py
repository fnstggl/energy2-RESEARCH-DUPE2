"""CanonicalMultiPlaneEnvironment — the two-clock orchestrator (PR-1 scaffold).

Exposes ONE production-like environment over SEPARATE raw traces, per
``research/CANONICAL_ENVIRONMENT_PLAN.md``. It does **not** row-join Azure to
v2026; it merges *state variables* and *calibrates distributions*.

Two synchronized clocks:
  * **hourly** — :class:`FleetPlane` produces a :class:`FleetState` (capacity,
    GPU-type mix, utilization target, priority/best-effort mix, queue delay,
    network pressure, energy price), calibrated from v2026 distributions.
  * **per-second** — :class:`ServingPlane` runs the token-level serving loop
    (``unified_replay``) for that hour's requests, under the hour's fleet state.

Scope discipline (deliberately minimal — see plan Part A/D): this composes only
what genuinely composes today — the serving sim, ``economics``, the v2026
class-mix hook, and the signal matrix. It does **not** fuse the heavy hourly
cluster engine (``simulation/cluster/engine.py``); that is the fenced PR-5. The
three integration seams (two calibration systems, two time models, partial
signal mapping) are recorded honestly in the :class:`FidelityManifest`, not
papered over.

Honest framing: this is a production-LIKE environment grounded by real public
traces, every field fidelity-tagged — NOT real production telemetry. It is
production-grade only when a pilot replaces calibrated assumptions with operator
telemetry (intent / hardware health / live KV / migration reasons / internal cost).
Directional simulator only (``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..benchmarks.economics import InfrastructureCostConfig
from ..datasets.calibration import ClassMix, default_alibaba_class_mix
from ..optimizer.unified_replay import run_unified_replay

# Fidelity tiers (mirror datasets.signal_matrix / simulation.cluster.calibration).
TIER_MEASURED = "MEASURED_REAL"
TIER_PROXY = "PROXY"
TIER_SYNTHETIC = "SYNTHETIC"
TIER_HEURISTIC = "HEURISTIC"


# ---------------------------------------------------------------------------
# Fleet plane (hourly state) — calibrated distributions, NOT raw rows
# ---------------------------------------------------------------------------

@dataclass
class FleetState:
    """Merged hourly fleet-plane state. Each field carries its fidelity tier."""

    hour: int
    available_gpus: int
    gpu_type_mix: dict                 # {"H100": 0.6, "A100": 0.4}
    util_target: float                 # v2026 avg_gpu_sm_util (0..1)
    mem_pressure: float                # v2026 avg_gpu_mem_gib / capacity (0..1)
    priority_mix: dict                 # {"HP": 0.8, "LP": 0.2}
    best_effort_fraction: float        # v2026 online/offline inference share
    queue_delay_s: float               # v2026 schedule_delay_sec
    net_pressure: float                # v2026 network rx/tx (0..1)
    energy_price_per_kwh: float        # electricity layer (region, by hour)
    fidelity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hour": self.hour, "available_gpus": self.available_gpus,
            "gpu_type_mix": self.gpu_type_mix, "util_target": round(self.util_target, 4),
            "mem_pressure": round(self.mem_pressure, 4), "priority_mix": self.priority_mix,
            "best_effort_fraction": round(self.best_effort_fraction, 4),
            "queue_delay_s": round(self.queue_delay_s, 4),
            "net_pressure": round(self.net_pressure, 4),
            "energy_price_per_kwh": round(self.energy_price_per_kwh, 5),
            "fidelity": self.fidelity,
        }


class FleetPlane:
    """Produces a :class:`FleetState` per hour from calibrated distributions.

    v1 (PR-1): only the **best-effort / class mix** is genuinely trace-derived
    (from the v2026/Alibaba hook — the one bridge that exists today). Every other
    field is a documented default tagged ``HEURISTIC`` until its v2026 hook lands
    (plan PR-2). The honesty is in the per-field ``fidelity`` tags, not in
    pretending the defaults are real.
    """

    def __init__(
        self,
        *,
        class_mix: ClassMix | None = None,
        available_gpus: int = 64,
        gpu_type_mix: dict | None = None,
        util_target: float = 0.6,
        mem_pressure: float = 0.65,
        queue_delay_s: float = 0.0,
        net_pressure: float = 0.3,
        energy_price_series: list | None = None,
        default_energy_price_per_kwh: float = 0.08,
    ) -> None:
        self.class_mix = class_mix or default_alibaba_class_mix()
        self.available_gpus = available_gpus
        self.gpu_type_mix = gpu_type_mix or {"H100": 0.5, "A100": 0.5}
        self.util_target = util_target
        self.mem_pressure = mem_pressure
        self.queue_delay_s = queue_delay_s
        self.net_pressure = net_pressure
        self.energy_price_series = energy_price_series
        self.default_energy_price_per_kwh = default_energy_price_per_kwh

    def state_at(self, hour: int) -> FleetState:
        be = self.class_mix.best_effort_fraction_by_count
        price = self.default_energy_price_per_kwh
        price_tier = TIER_HEURISTIC
        if self.energy_price_series:
            price = self.energy_price_series[hour % len(self.energy_price_series)]
            price_tier = TIER_MEASURED  # real ISO series when supplied
        return FleetState(
            hour=hour,
            available_gpus=self.available_gpus,
            gpu_type_mix=dict(self.gpu_type_mix),
            util_target=self.util_target,
            mem_pressure=self.mem_pressure,
            priority_mix={"HP": round(1.0 - be, 4), "LP": round(be, 4)},
            best_effort_fraction=be,
            queue_delay_s=self.queue_delay_s,
            net_pressure=self.net_pressure,
            energy_price_per_kwh=price,
            fidelity={
                # the ONE genuinely trace-derived field today (plan Part C #5/#7):
                "best_effort_fraction": self.class_mix.tier,
                "priority_mix": self.class_mix.tier,
                # everything else is a documented default until PR-2 hooks land:
                "available_gpus": TIER_HEURISTIC, "gpu_type_mix": TIER_HEURISTIC,
                "util_target": TIER_HEURISTIC, "mem_pressure": TIER_HEURISTIC,
                "queue_delay_s": TIER_HEURISTIC, "net_pressure": TIER_HEURISTIC,
                "energy_price_per_kwh": price_tier,
            },
        )


# ---------------------------------------------------------------------------
# Cost model — economics + PUE + depreciation + region energy price
# ---------------------------------------------------------------------------

@dataclass
class CostBreakdown:
    gpu_infra_cost: float
    energy_cost: float
    network_cost: float
    depreciation_cost: float

    @property
    def total(self) -> float:
        return (max(0.0, self.gpu_infra_cost) + max(0.0, self.energy_cost)
                + max(0.0, self.network_cost) + max(0.0, self.depreciation_cost))

    def to_dict(self) -> dict:
        return {
            "gpu_infra_cost": round(self.gpu_infra_cost, 4),
            "energy_cost": round(self.energy_cost, 4),
            "network_cost": round(self.network_cost, 4),
            "depreciation_cost": round(self.depreciation_cost, 4),
            "total": round(self.total, 4),
        }


@dataclass
class CostModel:
    """Wraps ``economics.InfrastructureCostConfig`` and adds the explicit
    operator-cost assumptions it lacks: **PUE** and **GPU depreciation**.

    Every added knob is a MODELED ASSUMPTION (not measured) — flagged as such in
    the manifest. ``pue`` multiplies energy draw (cooling/overhead);
    ``gpu_depreciation_per_gpu_hour`` amortizes capex per provisioned GPU-hour.
    """

    cfg: InfrastructureCostConfig = field(default_factory=InfrastructureCostConfig)
    pue: float = 1.3                                  # modeled assumption
    gpu_depreciation_per_gpu_hour: float = 0.0        # modeled assumption (capex amort)
    gpu_kw: float = 0.7                               # avg draw per provisioned GPU (kW)

    def cost(
        self, *, gpu_hours: float, gpu_type: str, energy_price_per_kwh: float,
        migrations: int = 0, egress_gb: float = 0.0,
    ) -> CostBreakdown:
        gpu_infra = gpu_hours * self.cfg.gpu_price(gpu_type)
        energy_kwh = gpu_hours * self.gpu_kw * self.pue
        energy = energy_kwh * energy_price_per_kwh
        network = (migrations * self.cfg.network_cost_per_migration
                   + egress_gb * self.cfg.network_cost_per_gb_egress)
        depreciation = gpu_hours * self.gpu_depreciation_per_gpu_hour
        return CostBreakdown(gpu_infra, energy, network, depreciation)

    def assumptions(self) -> dict:
        return {
            "pue": {"value": self.pue, "tier": TIER_HEURISTIC,
                    "note": "modeled cooling/overhead multiplier, not measured"},
            "gpu_depreciation_per_gpu_hour": {
                "value": self.gpu_depreciation_per_gpu_hour, "tier": TIER_HEURISTIC,
                "note": "modeled capex amortization, not operator contract"},
            "gpu_kw": {"value": self.gpu_kw, "tier": TIER_HEURISTIC,
                       "note": "modeled per-GPU draw"},
            "gpu_hour_prices": {"value": "InfrastructureCostConfig (public-list priors)",
                                "tier": TIER_HEURISTIC},
        }


# ---------------------------------------------------------------------------
# Fidelity manifest — unifies signal tiers + cost assumptions + the SEAMS
# ---------------------------------------------------------------------------

# The three integration seams (plan Part A) — recorded, not papered over.
SEAMS = (
    "calibration: cluster static CalibratedParam table is NOT wired to the "
    "trace-derived datasets/calibration; only the v2026 class-mix is bridged today",
    "time-model: serving sim (seconds, event-driven) and cluster engine (hourly, "
    "M/M/1 proxy) are separate; this env nests seconds-in-hour but does NOT yet "
    "use the cluster engine (fenced PR-5)",
    "signal-mapping: v2026 asw_id maps to the rack tier only; intra-node fabric "
    "(NVLink/NVSwitch) and the KV prefix-reuse curve stay heuristic (Mooncake pending)",
)

FRAMING = (
    "Production-LIKE environment grounded by real public traces, every field "
    "fidelity-tagged — NOT real production telemetry. Production-grade only when a "
    "pilot replaces calibrated assumptions with operator telemetry (intent / "
    "hardware health / live KV / migration reasons / internal cost model)."
)


@dataclass
class FidelityManifest:
    """Per-environment provenance: every signal's source + tier, plus the seams."""

    fleet_fidelity: dict
    cost_assumptions: dict
    seams: tuple = SEAMS
    framing: str = FRAMING

    def is_production_grade(self) -> bool:
        """Honesty gate: never production-grade while any field is below MEASURED."""
        tiers = list(self.fleet_fidelity.values()) + [
            a.get("tier") for a in self.cost_assumptions.values()]
        return all(t == TIER_MEASURED for t in tiers)

    def to_dict(self) -> dict:
        return {
            "fleet_fidelity": self.fleet_fidelity,
            "cost_assumptions": self.cost_assumptions,
            "seams": list(self.seams),
            "framing": self.framing,
            "is_production_grade": self.is_production_grade(),
        }


# ---------------------------------------------------------------------------
# Serving plane — adapter over the token-level serving loop
# ---------------------------------------------------------------------------

class ServingPlane:
    """Runs one hour's requests through ``unified_replay`` under a fleet state.

    v1 coupling (light, honest): the fleet's ``available_gpus`` caps the cold-start
    capacity (``warmup_c``); the fleet's ``energy_price`` flows to the cost model.
    Richer fleet→serving coupling (GPU-type TPOT, mem→KV capacity, topology
    penalties) lands with the calibration bridge (plan PR-2).
    """

    def run_hour(
        self, jobs: list, fleet: FleetState, *, tick_seconds: float, sla_s: float,
        capacity: str = "backlog_aware", ordering: str = "abs_conformal",
        admission: str = "class_aware",
    ):
        warmup_c = max(1, min(fleet.available_gpus, 4))
        return run_unified_replay(
            jobs, tick_seconds=tick_seconds, sla_s=sla_s, capacity=capacity,
            ordering=ordering, admission=admission, warmup_c=warmup_c)


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentResult:
    hours: list                    # per-hour {fleet, kpi, cost}
    total_goodput: float
    total_cost: float
    goodput_per_dollar: float
    manifest: dict

    def to_dict(self) -> dict:
        return {
            "hours": self.hours, "total_goodput": round(self.total_goodput, 2),
            "total_cost": round(self.total_cost, 4),
            "goodput_per_dollar": round(self.goodput_per_dollar, 4),
            "manifest": self.manifest,
        }


class CanonicalMultiPlaneEnvironment:
    """The two-clock environment: per-hour fleet state × per-second serving."""

    def __init__(
        self, *, fleet_plane: FleetPlane | None = None,
        serving_plane: ServingPlane | None = None, cost_model: CostModel | None = None,
    ) -> None:
        self.fleet_plane = fleet_plane or FleetPlane()
        self.serving_plane = serving_plane or ServingPlane()
        self.cost_model = cost_model or CostModel()

    def manifest(self, sample_hour: int = 0) -> FidelityManifest:
        fleet = self.fleet_plane.state_at(sample_hour)
        return FidelityManifest(
            fleet_fidelity=fleet.fidelity,
            cost_assumptions=self.cost_model.assumptions(),
        )

    def run(
        self, hourly_jobs: dict, *, tick_seconds: float = 60.0, sla_s: float = 10.0,
    ) -> EnvironmentResult:
        """Run the environment over ``{hour: [Job, ...]}``.

        For each hour: produce the fleet state, run the serving loop under it,
        price the result on the assembled cost model (GPU + PUE energy +
        depreciation). Returns per-hour detail + the rolled-up goodput/$ + manifest.
        """
        hours: list = []
        total_goodput = 0.0
        total_cost = 0.0
        for hour in sorted(hourly_jobs):
            jobs = hourly_jobs[hour]
            fleet = self.fleet_plane.state_at(hour)
            kpi = self.serving_plane.run_hour(
                jobs, fleet, tick_seconds=tick_seconds, sla_s=sla_s)
            # price on the assembled cost model (dominant GPU type for v1)
            gpu_type = max(fleet.gpu_type_mix, key=fleet.gpu_type_mix.get)
            cost = self.cost_model.cost(
                gpu_hours=kpi.gpu_hours, gpu_type=gpu_type,
                energy_price_per_kwh=fleet.energy_price_per_kwh)
            total_goodput += kpi.sla_safe_goodput
            total_cost += cost.total
            hours.append({
                "hour": hour, "fleet": fleet.to_dict(),
                "kpi": kpi.to_dict(), "cost": cost.to_dict(),
            })
        gpd = total_goodput / max(total_cost, 1e-9)
        return EnvironmentResult(
            hours=hours, total_goodput=total_goodput, total_cost=total_cost,
            goodput_per_dollar=gpd, manifest=self.manifest().to_dict())


__all__ = [
    "FleetState", "FleetPlane", "CostModel", "CostBreakdown", "FidelityManifest",
    "ServingPlane", "CanonicalMultiPlaneEnvironment", "EnvironmentResult",
    "SEAMS", "FRAMING",
]
