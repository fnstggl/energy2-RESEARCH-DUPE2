"""ModelPredictiveEconomicController — forecast → simulate → choose (no deep RL).

A debuggable model-predictive controller over the **connected** environment actions
(``capacity`` / ``ordering`` / ``admission`` — see the Phase-0 audit; KV routing /
DVFS / placement are NOT connected and are NOT offered here). Each decision period:

    observe history  →  ForecastBundle over horizon H (causal, from train data)
      →  enumerate candidate action plans (connected actions only)
        →  simulate each plan on the FORECASTED load (point + p90 risk scenario)
          →  score by expected SLA-safe goodput / operator-$
            →  choose the best safe action  (fallback to SLA-aware if low confidence)

Strictly causal: the decision uses only the forecast (built from periods ≤ now) — never
the real next period's arrivals. The harness applies the chosen action to the real
period AFTER the decision. Savings are SIMULATED; the claim gate lives in
``training.py``.
"""

from __future__ import annotations

import bisect
import statistics
from dataclasses import dataclass, field

from ..benchmarks.srtf_serving_backtest import _service_time_s
from ..optimizer.unified_replay import (
    CLASS_BEST_EFFORT,
    CLASS_LATENCY,
    Job,
    run_unified_replay,
)
from .cost_model import CostModel
from .forecasting import ForecastingModel

# Connected action levers (the only ones the environment actually executes).
CAPACITY = ("reactive_lag1", "backlog_aware", "forecasted_mcs")
ORDERING = ("fifo", "abs_conformal")
ADMISSION = ("off", "class_aware")
SLA_AWARE_FALLBACK = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off"}


def enumerate_actions() -> list:
    return [{"capacity": c, "ordering": o, "admission": a}
            for c in CAPACITY for o in ORDERING for a in ADMISSION]


@dataclass
class Decision:
    action: dict
    expected_gpd: float
    risk_gpd: float                    # gp/$ in the p90 high-load scenario
    score: float
    used_fallback: bool
    confidence: float
    forecast: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"action": self.action, "expected_gpd": round(self.expected_gpd, 2),
                "risk_gpd": round(self.risk_gpd, 2), "score": round(self.score, 2),
                "used_fallback": self.used_fallback, "confidence": round(self.confidence, 3)}


def _synth_jobs(arrival_rate: float, tok_mean: float, tok_p95: float, cv: float, *,
                period_seconds: float, best_effort_fraction: float, kv_service_factor: float) -> list:
    """Deterministic synthetic job set matching a forecasted period profile (no RNG)."""
    n = max(0, int(round(arrival_rate * period_seconds)))
    if n == 0:
        return []
    be_stride = max(1, round(1.0 / best_effort_fraction)) if best_effort_fraction > 0 else 0
    # bursty arrivals: clump fraction grows with CV (deterministic interleave)
    burst = min(0.9, max(0.0, cv / 4.0))
    jobs = []
    for i in range(n):
        frac = i / n
        # tokens: most at mean, a deterministic 1-in-20 tail at p95
        tok = tok_p95 if (i % 20 == 0 and tok_p95 > 0) else tok_mean
        tok = max(1, int(tok))
        # arrival time: compress `burst` of the mass into the first 30% of the period
        arr = (frac * 0.3 if frac < burst else 0.3 + (frac - burst) / max(1e-9, 1 - burst) * 0.7)
        arr *= period_seconds
        cls = CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY
        jobs.append(Job(idx=i, arrival_s=arr, actual_tokens=tok, predicted_tokens=float(tok_mean),
                        service_s=_service_time_s(tok) * kv_service_factor, cls=cls))
    return jobs


@dataclass
class ModelPredictiveEconomicController:
    """MPC over connected actions, scored by expected SLA-safe goodput / operator-$."""

    forecasters: ForecastingModel
    fleet_state: object                # FleetState (constant anchored marginals)
    cost_model: CostModel
    horizon: int = 4
    sla_s: float = 10.0
    period_seconds: float = 60.0
    tick_seconds: float = 60.0
    risk_weight: float = 0.5           # penalty on the p90 high-load SLA-violation rate
    confidence_min: float = 0.15       # below this, fall back to the SLA-aware action
    kv_service_factor: float = 1.0     # avg KV service discount (≤1), applied to all candidates
    cost_scenario: str = "owned"
    candidates: list = field(default_factory=enumerate_actions)

    def _gpd(self, jobs: list, action: dict, price: float) -> tuple:
        if not jobs:
            return 0.0, 0.0
        kpi = run_unified_replay(jobs, tick_seconds=self.tick_seconds, sla_s=self.sla_s,
                                 capacity=action["capacity"], ordering=action["ordering"],
                                 admission=action["admission"],
                                 warmup_c=max(1, min(self.fleet_state.capacity_envelope, 4)))
        gpu_type = (max(self.fleet_state.gpu_type_mix, key=self.fleet_state.gpu_type_mix.get)
                    if self.fleet_state.gpu_type_mix else "H100")
        cost = self.cost_model.operator_cost(
            gpu_hours=kpi.gpu_hours, gpu_type=gpu_type, energy_price_per_kwh=price,
            utilization=self.fleet_state.util_target, scenario=self.cost_scenario,
            sla_violations=kpi.sla_violations)
        gpd = kpi.sla_safe_goodput / max(cost.total_operator_cost, 1e-9)
        viol_rate = kpi.sla_violations / max(1, kpi.n_total)
        return gpd, viol_rate

    def decide(self, history: list) -> Decision:
        """Choose the action for the next period from the causal forecast only."""
        if not self.forecasters.fitted or len(history) < 3:
            return Decision(dict(SLA_AWARE_FALLBACK), 0.0, 0.0, 0.0, True, 0.0)
        bundle = self.forecasters.predict(history, horizon=self.horizon)
        ar = bundle.at("arrival_rate", 0)
        tm = bundle.at("output_token_mean", 0)
        tp = bundle.at("output_token_p95", 0)
        cv = bundle.at("interarrival_cv", 0)
        pr = bundle.at("electricity_price", 0)
        # confidence: tighter band (relative) → higher confidence
        spread = (ar.p90 - ar.p10) / max(1e-9, ar.mean) if ar else 1.0
        confidence = max(0.0, 1.0 - spread)
        if confidence < self.confidence_min:
            return Decision(dict(SLA_AWARE_FALLBACK), 0.0, 0.0, 0.0, True, confidence,
                            forecast={"arrival_rate": ar.to_dict() if ar else {}})

        be = self.fleet_state.best_effort_fraction
        point = _synth_jobs(ar.mean, tm.mean, tp.value, cv.mean, period_seconds=self.period_seconds,
                            best_effort_fraction=be, kv_service_factor=self.kv_service_factor)
        risk = _synth_jobs(ar.p90, tm.p90, tp.p99, cv.p90, period_seconds=self.period_seconds,
                           best_effort_fraction=be, kv_service_factor=self.kv_service_factor)
        best = None
        for action in self.candidates:
            exp_gpd, _ = self._gpd(point, action, pr.value)
            risk_gpd, risk_viol = self._gpd(risk, action, pr.value)
            score = exp_gpd - self.risk_weight * risk_viol * exp_gpd     # risk-adjusted
            if best is None or score > best.score:
                best = Decision(action, exp_gpd, risk_gpd, score, False, confidence,
                                forecast={"arrival_rate": ar.to_dict(), "price": pr.value})
        return best


def _causal_pred(slice_sorted: list) -> list:
    """Running-median causal token prior (deployable, no oracle)."""
    n = len(slice_sorted)
    if n == 0:
        return []
    gmed = sorted(t for _, t, *_ in slice_sorted)[n // 2]
    pred, seen = [0.0] * n, []
    for i, rec in enumerate(slice_sorted):
        pred[i] = float(seen[(len(seen) - 1) // 2]) if seen else float(gmed)
        bisect.insort(seen, rec[1])
    return pred


@dataclass
class EpisodeReport:
    name: str
    n_periods: int
    sla_safe_goodput: float
    total_operator_cost: float
    goodput_per_dollar: float
    sla_violation_rate: float
    gpu_hours: float
    energy_cost: float
    n_sla_safe: int
    queue_delay_p95: float
    used_fallback_frac: float = 0.0

    def to_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def run_period_episode(name, decide_fn, real_per_period, frames, eval_indices, *,
                       fleet_state, cost_model, sla_s=10.0, tick_seconds=60.0,
                       period_seconds=60.0, kv_service_factor=1.0, cost_scenario="owned"):
    """Run ``decide_fn(history_frames)`` over the REAL eval periods (causal: the action
    for period p is chosen from frames[:p], then applied to the real requests of p)."""
    gpu_type = (max(fleet_state.gpu_type_mix, key=fleet_state.gpu_type_mix.get)
                if fleet_state.gpu_type_mix else "H100")
    be = fleet_state.best_effort_fraction
    be_stride = max(1, round(1.0 / be)) if be > 0 else 0
    tot_g = tot_cost = tot_energy = tot_gpu_h = 0.0
    tot_viol = tot_n = tot_safe = 0
    waits_p95 = []
    fb = 0
    for p in eval_indices:
        out = decide_fn(frames[:p])
        action = out["action"] if isinstance(out, dict) and "action" in out else out
        fb += int(bool(isinstance(out, dict) and out.get("used_fallback")))
        recs = sorted(real_per_period.get(p, []), key=lambda r: r[0])
        if not recs:
            continue
        t0 = recs[0][0]
        pred = _causal_pred(recs)
        jobs = [Job(idx=i, arrival_s=(r[0] - t0), actual_tokens=int(r[1]),
                    predicted_tokens=float(pred[i]),
                    service_s=_service_time_s(int(r[1])) * kv_service_factor,
                    cls=(CLASS_BEST_EFFORT if (be_stride and i % be_stride == 0) else CLASS_LATENCY))
                for i, r in enumerate(recs)]
        kpi = run_unified_replay(jobs, tick_seconds=tick_seconds, sla_s=sla_s,
                                 capacity=action["capacity"], ordering=action["ordering"],
                                 admission=action["admission"],
                                 warmup_c=max(1, min(fleet_state.capacity_envelope, 4)))
        cost = cost_model.operator_cost(
            gpu_hours=kpi.gpu_hours, gpu_type=gpu_type,
            energy_price_per_kwh=fleet_state.energy_price_per_kwh,
            utilization=fleet_state.util_target, scenario=cost_scenario,
            sla_violations=kpi.sla_violations)
        tot_g += kpi.sla_safe_goodput
        tot_cost += cost.total_operator_cost
        tot_energy += cost.energy_cost
        tot_gpu_h += kpi.gpu_hours
        tot_viol += kpi.sla_violations
        tot_n += kpi.n_total
        tot_safe += kpi.n_sla_safe
        waits = sorted(max(0.0, j.start_s - j.arrival_s) for j in jobs if j.start_s >= 0)
        if waits:
            waits_p95.append(waits[min(len(waits) - 1, int(len(waits) * 0.95))])
    ne = len(eval_indices)
    return EpisodeReport(
        name=name, n_periods=ne, sla_safe_goodput=tot_g, total_operator_cost=tot_cost,
        goodput_per_dollar=tot_g / max(tot_cost, 1e-9),
        sla_violation_rate=(tot_viol / tot_n if tot_n else 0.0), gpu_hours=tot_gpu_h,
        energy_cost=tot_energy, n_sla_safe=tot_safe,
        queue_delay_p95=(statistics.mean(waits_p95) if waits_p95 else 0.0),
        used_fallback_frac=(fb / ne if ne else 0.0))


__all__ = [
    "CAPACITY", "ORDERING", "ADMISSION", "SLA_AWARE_FALLBACK", "enumerate_actions",
    "Decision", "ModelPredictiveEconomicController", "EpisodeReport", "run_period_episode",
]
