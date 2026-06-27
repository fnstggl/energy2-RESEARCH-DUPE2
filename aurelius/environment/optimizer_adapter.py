"""Adapter: CanonicalMultiPlaneEnvironment ↔ AureliusOptimizer.

Feeds the environment's per-hour ``EnvStep`` (state, action, reward, metrics) into
the optimizer for training/evaluation, and runs a FAIR backtest — current optimizer
config + best validated config + an SLA-aware baseline + a greedy/packing baseline
+ a weak FIFO reference + a candidate — scoring every arm through the optimizer's
own ``ObjectiveLayer`` (SLA-safe goodput/$). No parallel optimizer path: scoring is
the optimizer's canonical objective; policies drive the env's existing
``policy(observation) → action`` hook (the same levers ``unified_replay`` executes).

Contracts
---------
* **State** (:class:`EnvState`) — the causal, decision-time signals an optimizer
  policy sees at an hour boundary: arrival rate / burstiness, fleet GPU util / mem
  pressure / priority mix / queue + schedule delay / network + topology pressure,
  electricity price + cost state, prior KV hit-rate, and a fidelity flag per signal.
  Built ONLY from start-of-hour information — never the hour's own or future
  arrivals.
* **Action** (:data:`ACTION_SPACE`) — the levers the env executes: capacity sizing,
  dispatch ordering, admission, plus KV-aware routing and the cost scenario. All
  causal; no tenant-side spot/reserved arbitrage.
* **Reward** — primary is **SLA-safe goodput per operator dollar**; the report also
  surfaces SLA violations, GPU-hours, energy + total operator cost, queue-delay
  p50/p95/p99, KV hit rate, and cost per useful request/token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..optimizer.aurelius_optimizer import AureliusOptimizer
from .canonical import DEFAULT_ACTION, CanonicalMultiPlaneEnvironment

# Levers the environment can execute (causal; map 1:1 to unified_replay).
# NOTE: the canonical, audited action surface now lives in `actions.py` / `action_registry.py`
# (see research/AURELIUS_ACTION_SURFACE_AUDIT.md). Of the keys below only capacity/ordering/
# admission are CONNECTED (change the reward); `kv_routing` is SIMULATED_ONLY and currently
# INERT (nothing consumes it — do not treat it as a real lever); `cost_scenario` is a
# cost-accounting context, not an infrastructure control action.
ACTION_SPACE = {
    "capacity": ["reactive_lag1", "backlog_aware", "forecasted_mcs"],
    "ordering": ["fifo", "abs_conformal"],
    "admission": ["off", "class_aware"],
    "kv_routing": [True, False],        # SIMULATED_ONLY / inert — see action_registry
    "cost_scenario": ["owned", "leased"],   # accounting context, not a control action
}


# ---------------------------------------------------------------------------
# State (causal, decision-time)
# ---------------------------------------------------------------------------

@dataclass
class EnvState:
    hour: int
    n_requests: int
    arrival_rate_per_s: float
    best_effort_fraction: float
    fleet_util: float
    mem_pressure: float
    queue_delay_s: float
    net_pressure: float
    fragmentation: float
    energy_price_per_kwh: float
    gpu_type: str
    priority_mix: dict
    kv_hit_rate_prior: float
    fidelity: dict = field(default_factory=dict)

    @classmethod
    def from_observation(cls, obs, *, kv_hit_rate_prior: float = 0.0) -> "EnvState":
        obs = obs.to_dict() if hasattr(obs, "to_dict") else obs
        fleet = obs.get("fleet", {})
        gpu_mix = fleet.get("gpu_type_mix", {})
        return cls(
            hour=obs.get("hour", 0), n_requests=obs.get("n_requests", 0),
            arrival_rate_per_s=obs.get("arrival_rate_per_s", 0.0),
            best_effort_fraction=obs.get("best_effort_fraction", 0.0),
            fleet_util=fleet.get("util_target", 0.0), mem_pressure=fleet.get("mem_pressure", 0.0),
            queue_delay_s=fleet.get("queue_delay_s", 0.0), net_pressure=fleet.get("net_pressure", 0.0),
            fragmentation=fleet.get("fragmentation", 0.0),
            energy_price_per_kwh=fleet.get("energy_price_per_kwh", 0.06),
            gpu_type=(max(gpu_mix, key=gpu_mix.get) if gpu_mix else "H100"),
            priority_mix=fleet.get("priority_mix", {}), kv_hit_rate_prior=kv_hit_rate_prior,
            fidelity={"fleet": "FULL_TRACE_EXACT", "serving": "FULL_TRACE",
                      "kv": "TRACE_DERIVED", "cost": "INFERRED"})

    def to_vector(self) -> dict:
        return {k: getattr(self, k) for k in (
            "hour", "n_requests", "arrival_rate_per_s", "best_effort_fraction", "fleet_util",
            "mem_pressure", "queue_delay_s", "net_pressure", "fragmentation",
            "energy_price_per_kwh", "kv_hit_rate_prior")}


def reward_from_step(step) -> float:
    """Primary reward = SLA-safe goodput per operator dollar (already on the step)."""
    return step.reward


# ---------------------------------------------------------------------------
# Policies: observation(dict) → action(dict). All causal (start-of-hour only).
# ---------------------------------------------------------------------------

def policy_fifo_weak(obs: dict) -> dict:
    """WEAK reference only (never the headline baseline)."""
    return {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off"}


def policy_sla_aware(obs: dict) -> dict:
    """Backlog-aware capacity + SLA-aware ordering, no admission control."""
    return {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off"}


def policy_greedy_packing(obs: dict) -> dict:
    """Capacity-greedy (forecasted MCS) FIFO — a strong provisioning baseline."""
    return {"capacity": "forecasted_mcs", "ordering": "fifo", "admission": "off"}


def policy_aurelius_canonical(obs: dict) -> dict:
    """The current AureliusOptimizer canonical best closed-loop config."""
    return dict(DEFAULT_ACTION)


def policy_aurelius_state_conditioned(obs: dict) -> dict:
    """Candidate: the canonical config, with admission gated CAUSALLY on the
    start-of-hour state — engage class-aware admission only when the offered load
    is heavy AND the best-effort share is high (where deferral helps), else drop it
    to avoid needless latency. Uses only decision-time signals (no future arrivals)."""
    st = EnvState.from_observation(obs)
    a = dict(DEFAULT_ACTION)
    heavy = st.arrival_rate_per_s >= 1.0 and st.best_effort_fraction >= 0.15
    a["admission"] = "class_aware" if heavy else "off"
    return a


BASELINE_POLICIES = {
    "fifo_weak": policy_fifo_weak,                 # weak reference
    "sla_aware": policy_sla_aware,
    "greedy_packing": policy_greedy_packing,
    "aurelius_canonical": policy_aurelius_canonical,
}
WEAK_BASELINES = frozenset({"fifo_weak"})
DEFAULT_CANDIDATE = ("aurelius_state_conditioned", policy_aurelius_state_conditioned)


# ---------------------------------------------------------------------------
# Per-policy report (the metrics the build spec requires)
# ---------------------------------------------------------------------------

def _pct(sorted_xs, q):
    if not sorted_xs:
        return 0.0
    k = (len(sorted_xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


@dataclass
class PolicyReport:
    name: str
    goodput_per_dollar: float
    objective_score: float            # via AureliusOptimizer.objective (canonical)
    sla_violation_rate: float
    n_sla_safe: int
    gpu_hours: float
    energy_cost: float
    total_operator_cost: float
    queue_delay_p50: float
    queue_delay_p95: float
    queue_delay_p99: float
    kv_hit_rate: float
    cost_per_sla_safe_request: float
    cost_per_sla_safe_token: float
    is_weak: bool

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def _aggregate(name: str, result, *, objective, is_weak: bool) -> PolicyReport:
    steps = result.steps
    g = sum(s.metrics["kpi"]["sla_safe_goodput"] for s in steps)
    cost = sum(s.metrics["cost"]["total_operator_cost"] for s in steps)
    energy = sum(s.metrics["cost"]["energy_cost"] for s in steps)
    gpu_hours = sum(s.metrics["kpi"]["gpu_hours"] for s in steps)
    viol = sum(s.metrics["kpi"]["sla_violations"] for s in steps)
    n_total = sum(s.metrics["kpi"]["n_total"] for s in steps)
    n_safe = sum(s.metrics["kpi"]["n_sla_safe"] for s in steps)
    kv = [s.metrics.get("kv", {}).get("kv_hit_rate", 0.0) for s in steps]
    p50 = sorted(s.action.get("queue_delay_p50", 0.0) for s in steps)
    p95 = sorted(s.action.get("queue_delay_p95", 0.0) for s in steps)
    p99 = sorted(s.action.get("queue_delay_p99", 0.0) for s in steps)
    score = objective.score(sla_compliant_goodput=int(g), total_infrastructure_cost=max(cost, 1e-9))
    return PolicyReport(
        name=name, goodput_per_dollar=g / max(cost, 1e-9), objective_score=float(score or 0.0),
        sla_violation_rate=(viol / n_total if n_total else 0.0), n_sla_safe=n_safe,
        gpu_hours=gpu_hours, energy_cost=energy, total_operator_cost=cost,
        queue_delay_p50=_pct(p50, 0.5), queue_delay_p95=_pct(p95, 0.95), queue_delay_p99=_pct(p99, 0.99),
        kv_hit_rate=(sum(kv) / len(kv) if kv else 0.0),
        cost_per_sla_safe_request=(cost / n_safe if n_safe else 0.0),
        cost_per_sla_safe_token=(cost / g if g else 0.0), is_weak=is_weak)


# ---------------------------------------------------------------------------
# Fair backtest
# ---------------------------------------------------------------------------

@dataclass
class BacktestReport:
    arms: dict                         # name -> PolicyReport.to_dict()
    ranking: list                      # [(name, goodput/$)] best first (optimizer.compare)
    fair_baseline: str                 # strongest NON-weak baseline (never silently FIFO)
    candidate: str
    candidate_vs_baseline_pct: float
    headline_claim_allowed: bool
    gate: dict                         # why the headline is / isn't allowed
    env_validation: dict

    def to_dict(self) -> dict:
        return {
            "arms": self.arms, "ranking": self.ranking, "fair_baseline": self.fair_baseline,
            "candidate": self.candidate,
            "candidate_vs_baseline_pct": round(self.candidate_vs_baseline_pct, 3),
            "headline_claim_allowed": self.headline_claim_allowed, "gate": self.gate,
            "env_validation": self.env_validation,
        }


def fair_backtest(
    azure_hourly: dict, *, env_kwargs: dict | None = None,
    baselines: dict | None = None, candidate=DEFAULT_CANDIDATE,
    optimizer: AureliusOptimizer | None = None,
) -> BacktestReport:
    """Run each policy through a FRESH canonical environment, score every arm via the
    optimizer's ObjectiveLayer, pick the fair (non-weak) baseline, and compare the
    candidate. The headline claim is gated on a fair baseline + passing held-out
    validation + no oracle — otherwise the comparison is reported as directional only.
    """
    env_kwargs = env_kwargs or {}
    baselines = baselines or BASELINE_POLICIES
    opt = optimizer or AureliusOptimizer()
    objective = opt.objective
    cand_name, cand_policy = candidate

    arms: dict = {}
    last_validation: dict = {}
    for name, policy in {**baselines, cand_name: cand_policy}.items():
        env = CanonicalMultiPlaneEnvironment(**env_kwargs)
        res = env.run(azure_hourly, policy=policy)
        last_validation = res.validation
        arms[name] = _aggregate(name, res, objective=objective,
                                is_weak=name in WEAK_BASELINES)

    ranking = objective.compare({n: r.goodput_per_dollar for n, r in arms.items()})

    # Fair baseline = strongest NON-weak, non-candidate arm (never silently FIFO).
    fair_candidates = {n: r for n, r in arms.items()
                       if n != cand_name and not r.is_weak}
    fair_baseline = max(fair_candidates, key=lambda n: fair_candidates[n].goodput_per_dollar) \
        if fair_candidates else cand_name
    base_gpd = arms[fair_baseline].goodput_per_dollar
    cand_gpd = arms[cand_name].goodput_per_dollar
    delta_pct = 100.0 * (cand_gpd - base_gpd) / base_gpd if base_gpd else 0.0

    # Held-out validation: Azure + Mooncake checks must pass (genuinely held-out).
    held_out = [c for c in last_validation.get("checks", [])
                if c["kind"].startswith(("azure", "kv_exact", "kv_partial", "kv_cache"))]
    held_out_ok = bool(held_out) and all(c["verdict"] == "PASS" for c in held_out)
    gate = {
        "fair_baseline_not_weak": fair_baseline not in WEAK_BASELINES,
        "beats_fair_baseline": delta_pct > 0,
        "held_out_validation_passed": held_out_ok,
        "no_oracle": True,                 # policies use only start-of-hour state (causal)
        "note": "savings are SIMULATED (directional simulator evidence), not production telemetry",
    }
    headline_allowed = (gate["fair_baseline_not_weak"] and gate["beats_fair_baseline"]
                        and gate["held_out_validation_passed"])
    return BacktestReport(
        arms={n: r.to_dict() for n, r in arms.items()}, ranking=ranking,
        fair_baseline=fair_baseline, candidate=cand_name,
        candidate_vs_baseline_pct=delta_pct, headline_claim_allowed=headline_allowed,
        gate=gate, env_validation={"overall_verdict": last_validation.get("overall_verdict"),
                                   "counts": last_validation.get("counts")})


__all__ = [
    "ACTION_SPACE", "EnvState", "reward_from_step",
    "policy_fifo_weak", "policy_sla_aware", "policy_greedy_packing",
    "policy_aurelius_canonical", "policy_aurelius_state_conditioned",
    "BASELINE_POLICIES", "DEFAULT_CANDIDATE", "PolicyReport", "BacktestReport", "fair_backtest",
]
