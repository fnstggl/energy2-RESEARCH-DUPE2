"""CanonicalMultiPlaneEnvironment — the two-clock environment (first principles).

ONE production-like training/evaluation environment over SEPARATE raw traces:
  * **Azure** request-level serving spine (per-second :class:`ServingPlane`)
  * **Mooncake** KV prefix-reuse calibration
  * **Alibaba v2026** hourly fleet spine (:class:`V2026FleetPlane`)
  * **ISO electricity** regional hourly cost (:class:`CostModel`)

It owns the two-clock loop — **serving seconds synchronized inside fleet hours** —
keeps the raw traces strictly separate (no row-joins; only *state variables* and
*calibrated distributions* cross planes), and emits ``(observation, action,
reward, metrics)`` per hour for AureliusOptimizer training / fair backtesting.
Every emitted signal is provenance-tagged (:class:`FidelityManifest`); a held-out
distribution check runs through the :mod:`validation_suite`.

This is bespoke and trace-grounded — it does NOT wrap the heuristic hourly cluster
engine. Directional simulator only (``docs/RESULTS.md`` §8); never production
telemetry (see the manifest framing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .calibration_bridge import CalibrationBridge, build_bridge
from .cost_model import CostModel
from .fidelity_manifest import FidelityManifest
from .fleet_plane_v2026 import (
    V2026FleetPlane,
)
from .ingestion.mooncake import ingest_mooncake
from .kv_cache import KVModel, gpu_mem_for
from .schemas import EnvObservation, EnvStep
from .serving_plane import ServingPlane
from .validation_suite import run_validation
from .validators import build_all_checks

# Default policy: the canonical best closed-loop config (the env is policy-pluggable).
DEFAULT_ACTION = {"capacity": "backlog_aware", "ordering": "abs_conformal",
                  "admission": "class_aware"}


@dataclass
class EnvironmentResult:
    steps: list                        # list[EnvStep]
    total_goodput: float
    total_cost: float
    goodput_per_dollar: float
    manifest: dict
    validation: dict
    cost_sensitivity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_goodput": round(self.total_goodput, 2),
            "total_cost": round(self.total_cost, 4),
            "goodput_per_dollar": round(self.goodput_per_dollar, 4),
            "cost_sensitivity": self.cost_sensitivity,
            "manifest": self.manifest, "validation": self.validation,
        }


class CanonicalMultiPlaneEnvironment:
    """The canonical two-clock environment. Pass a ``policy(observation)->action``
    to plug AureliusOptimizer in; the default is the canonical best config."""

    def __init__(
        self,
        *,
        mooncake_path: str,
        fleet_plane: V2026FleetPlane | None = None,
        serving_plane: ServingPlane | None = None,
        cost_model: CostModel | None = None,
        warp: float = 1.0,
        tick_seconds: float = 60.0,
        sla_s: float = 10.0,
        processed_dir: str | None = None,
        kv_enabled: bool = True,
        cost_scenario: str = "owned",
    ) -> None:
        self.fleet_plane = fleet_plane or V2026FleetPlane(processed_dir=processed_dir)
        self.serving_plane = serving_plane or ServingPlane()
        self.cost_model = cost_model or CostModel()
        self.mooncake_path = mooncake_path
        self.warp = warp
        self.tick_seconds = tick_seconds
        self.sla_s = sla_s
        self.processed_dir = processed_dir
        self.kv_enabled = kv_enabled
        self.cost_scenario = cost_scenario
        self._bridge: CalibrationBridge | None = None
        self._kv: KVModel | None = None
        self._kv_source_tier = "n/a"

    # -- calibration -----------------------------------------------------
    def calibrate(self, azure_raw: list) -> CalibrationBridge:
        """Fit all distribution-derived params (Azure train split + Mooncake + v2026)."""
        self._bridge = build_bridge(
            azure_raw, mooncake_path=self.mooncake_path, fleet_plane=self.fleet_plane)
        return self._bridge

    def _build_kv(self) -> KVModel:
        """Fit the stateful KV cache on the Mooncake train split, sized to this
        fleet's representative GPU memory budget + live memory pressure (couples KV
        capacity/eviction to the v2026 GPU-memory calibration). Causal; holdout is
        validated separately. ``kv_enabled=False`` yields a no-op (KV-disabled) model."""
        reqs, status = ingest_mooncake()
        n = len(reqs)
        train = reqs[: int(n * 0.7)] if n > 3 else reqs
        hours = self.fleet_plane.hours()
        fleet0 = self.fleet_plane.state_at(hours[0]) if hours else None
        gpu_type = (max(fleet0.gpu_type_mix, key=fleet0.gpu_type_mix.get)
                    if (fleet0 and fleet0.gpu_type_mix) else "H100")
        gpu_mem = gpu_mem_for(gpu_type)
        mem_pressure = fleet0.mem_pressure if fleet0 else 0.0
        self._kv = KVModel.fit(train, gpu_mem_gib=gpu_mem, mem_pressure=mem_pressure,
                               enabled=self.kv_enabled)
        self._kv_source_tier = status.tier
        return self._kv

    def manifest(self) -> FidelityManifest:
        params = list(self._bridge.params) if self._bridge else []
        params += self.fleet_plane.full_trace_params()   # FULL_TRACE_EXACT fleet marginals
        if self._kv is None and self._bridge is not None:
            self._build_kv()
        if self._kv is not None:
            params += self._kv.params()                  # KV reuse/footprint/eviction provenance
        params += self.cost_model.params(scenario=self.cost_scenario)
        return FidelityManifest.from_params(params)

    # -- the two-clock run ----------------------------------------------
    def run(self, azure_hourly: dict, *, policy=None) -> EnvironmentResult:
        """Run the environment over ``{hour: [(arrival_s, tokens), ...]}``.

        For each fleet hour: read the v2026 fleet state, build that hour's serving
        requests from the Azure slice (classes from the fleet's best-effort mix; KV
        hits from Mooncake), let the policy choose the action, run the **token-level
        per-second** serving loop under the fleet state, and price the result on the
        owned-hardware cost model (depreciation + PUE energy at the hour's ISO price).
        Returns per-hour ``EnvStep``s + the rolled-up reward + manifest + validation.
        """
        if self._bridge is None:
            # calibrate from the union of hourly slices (train split inside)
            allraw = [r for h in sorted(azure_hourly) for r in azure_hourly[h]]
            self.calibrate(allraw)
        if self._kv is None:
            self._build_kv()
        policy = policy or (lambda obs: dict(DEFAULT_ACTION))

        steps: list = []
        total_goodput = 0.0
        total_cost = 0.0
        total_gpu_hours = 0.0
        gpu_type_hours: dict = {}
        util_sum = 0.0
        price_sum = 0.0
        for hour in sorted(azure_hourly):
            raw_slice = azure_hourly[hour]
            fleet = self.fleet_plane.state_at(hour)
            obs = EnvObservation(
                hour=hour, fleet=fleet.summary(), n_requests=len(raw_slice),
                arrival_rate_per_s=(len(raw_slice) / max(1e-9, self.tick_seconds)),
                best_effort_fraction=fleet.best_effort_fraction)
            action = policy(obs)
            requests = self.serving_plane.build_requests(
                raw_slice, warp=self.warp,
                best_effort_fraction=fleet.best_effort_fraction, kv_model=self._kv)
            kpi, run_action = self.serving_plane.run_hour(
                requests, fleet, tick_seconds=self.tick_seconds, sla_s=self.sla_s,
                kv_model=self._kv,
                capacity=action["capacity"], ordering=action["ordering"],
                admission=action["admission"])
            gpu_type = max(fleet.gpu_type_mix, key=fleet.gpu_type_mix.get) if fleet.gpu_type_mix else "H100"
            cost = self.cost_model.operator_cost(
                gpu_hours=kpi.gpu_hours, gpu_type=gpu_type,
                energy_price_per_kwh=fleet.energy_price_per_kwh,
                utilization=fleet.util_target, scenario=self.cost_scenario,
                sla_violations=kpi.sla_violations)
            reward = kpi.sla_safe_goodput / max(cost.total_operator_cost, 1e-9)
            total_goodput += kpi.sla_safe_goodput
            total_cost += cost.total_operator_cost
            total_gpu_hours += kpi.gpu_hours
            gpu_type_hours[gpu_type] = gpu_type_hours.get(gpu_type, 0.0) + kpi.gpu_hours
            util_sum += fleet.util_target
            price_sum += fleet.energy_price_per_kwh
            steps.append(EnvStep(
                hour=hour, observation=obs.to_dict(), action={**action, **run_action},
                reward=reward,
                metrics={"kpi": kpi.to_dict(),
                         "cost": cost.to_dict(n_sla_safe=kpi.n_sla_safe,
                                              sla_safe_tokens=kpi.sla_safe_goodput,
                                              sla_safe_goodput=kpi.sla_safe_goodput),
                         "gpu_type": gpu_type,
                         "kv": self._kv.stats(len(requests)) if self._kv else {}}))

        n = max(1, len(steps))
        rep_gpu = max(gpu_type_hours, key=gpu_type_hours.get) if gpu_type_hours else "H100"
        sensitivity = self.cost_model.sensitivity(
            gpu_hours=total_gpu_hours or 1.0, gpu_type=rep_gpu,
            energy_price_per_kwh=(price_sum / n) or 0.06, utilization=(util_sum / n),
            scenario=self.cost_scenario)
        validation = self.validate(processed_dir=self.processed_dir).to_dict()
        return EnvironmentResult(
            steps=steps, total_goodput=total_goodput, total_cost=total_cost,
            goodput_per_dollar=total_goodput / max(total_cost, 1e-9),
            manifest=self.manifest().to_dict(), validation=validation,
            cost_sensitivity=sensitivity)

    # -- validation ------------------------------------------------------
    def validate(self, *, processed_dir: str | None = None):
        """Full held-out distribution validation across all four planes.

        Azure (token + inter-arrival, held-out time split), v2026 fleet (the env's
        committed sample vs the FULL_TRACE_EXACT artifacts), Mooncake KV (train vs
        holdout prefix reuse), and electricity (price sanity / held-out ISO skipped).
        Each check carries its real reference's data tier; SKIPPED checks name the
        exact artifact/command required. The overall verdict is capped by the
        honesty gate (NOT_PRODUCTION_REALISTIC_YET unless every calibrated param is
        ≥ TRACE_DERIVED)."""
        checks = build_all_checks(
            bridge=self._bridge, fleet_plane=self.fleet_plane, cost_model=self.cost_model,
            processed_dir=processed_dir or self.processed_dir)
        params = list(self._bridge.params) if self._bridge else []
        params += self.fleet_plane.full_trace_params()
        if self._kv is not None:
            params += self._kv.params()
        params += self.cost_model.params(scenario=self.cost_scenario)
        return run_validation(checks, params)


__all__ = ["CanonicalMultiPlaneEnvironment", "EnvironmentResult", "DEFAULT_ACTION"]
