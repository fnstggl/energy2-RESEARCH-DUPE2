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

from dataclasses import dataclass

from .calibration_bridge import CalibrationBridge, build_bridge
from .cost_model import CostModel
from .fidelity_manifest import FidelityManifest
from .fleet_plane_v2026 import (
    V2026FleetPlane,
)
from .schemas import EnvObservation, EnvStep
from .serving_plane import KVReuseModel, ServingPlane
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

    def to_dict(self) -> dict:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_goodput": round(self.total_goodput, 2),
            "total_cost": round(self.total_cost, 4),
            "goodput_per_dollar": round(self.goodput_per_dollar, 4),
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
    ) -> None:
        self.fleet_plane = fleet_plane or V2026FleetPlane(processed_dir=processed_dir)
        self.serving_plane = serving_plane or ServingPlane()
        self.cost_model = cost_model or CostModel()
        self.mooncake_path = mooncake_path
        self.warp = warp
        self.tick_seconds = tick_seconds
        self.sla_s = sla_s
        self.processed_dir = processed_dir
        self._bridge: CalibrationBridge | None = None

    # -- calibration -----------------------------------------------------
    def calibrate(self, azure_raw: list) -> CalibrationBridge:
        """Fit all distribution-derived params (Azure train split + Mooncake + v2026)."""
        self._bridge = build_bridge(
            azure_raw, mooncake_path=self.mooncake_path, fleet_plane=self.fleet_plane)
        return self._bridge

    def _kv_model(self) -> KVReuseModel:
        p = self._bridge.by_name("kv_prefix_hit_rate") if self._bridge else None
        rate = p.value["prefix_hit_rate"] if p else 0.0
        return KVReuseModel(hit_rate=rate)

    def manifest(self) -> FidelityManifest:
        params = list(self._bridge.params) if self._bridge else []
        params += self.fleet_plane.full_trace_params()   # FULL_TRACE_EXACT fleet marginals
        params += self.cost_model.params()
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
        kv = self._kv_model()
        policy = policy or (lambda obs: dict(DEFAULT_ACTION))

        steps: list = []
        total_goodput = 0.0
        total_cost = 0.0
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
                best_effort_fraction=fleet.best_effort_fraction, kv=kv)
            kpi, run_action = self.serving_plane.run_hour(
                requests, fleet, tick_seconds=self.tick_seconds, sla_s=self.sla_s, kv=kv,
                capacity=action["capacity"], ordering=action["ordering"],
                admission=action["admission"])
            gpu_type = max(fleet.gpu_type_mix, key=fleet.gpu_type_mix.get) if fleet.gpu_type_mix else "H100"
            cost = self.cost_model.cost(
                gpu_hours=kpi.gpu_hours, gpu_type=gpu_type,
                energy_price_per_kwh=fleet.energy_price_per_kwh)
            reward = kpi.sla_safe_goodput / max(cost.total, 1e-9)
            total_goodput += kpi.sla_safe_goodput
            total_cost += cost.total
            steps.append(EnvStep(
                hour=hour, observation=obs.to_dict(), action={**action, **run_action},
                reward=reward,
                metrics={"kpi": kpi.to_dict(), "cost": cost.to_dict(),
                         "gpu_type": gpu_type}))

        validation = self.validate(processed_dir=self.processed_dir).to_dict()
        return EnvironmentResult(
            steps=steps, total_goodput=total_goodput, total_cost=total_cost,
            goodput_per_dollar=total_goodput / max(total_cost, 1e-9),
            manifest=self.manifest().to_dict(), validation=validation)

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
            bridge=self._bridge, fleet_plane=self.fleet_plane,
            processed_dir=processed_dir or self.processed_dir)
        params = list(self._bridge.params) if self._bridge else []
        params += self.fleet_plane.full_trace_params()
        params += self.cost_model.params()
        return run_validation(checks, params)


__all__ = ["CanonicalMultiPlaneEnvironment", "EnvironmentResult", "DEFAULT_ACTION"]
