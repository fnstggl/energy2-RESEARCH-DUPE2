"""Production scheduler baseline (benchmark/evaluation layer ONLY).

`production_scheduler` is the single canonical, realistic modern GPU-fleet scheduler baseline future headlines
compare against — a deterministic **heuristic** that reacts to current/recent observable load. It lives in the
evaluation layer as a `decide_fn(history) -> action_dict`, exactly like `fifo` / `sla_aware`, and runs through
the unchanged reward path (`run_period_episode` merges the connected surfaces). It also defines the two missing
ladder rungs `vllm_only` and `topology_aware`.

**Hard separation (by design + enforced by test):** this module imports NO planner / MPC-search / economic /
oracle / hierarchical code. `production_scheduler` does not choose Aurelius actions, is not a planner mode, and
never uses future electricity prices, oracle future workload, the global economic objective, model-precision /
DVFS arbitrage, or MPC-planned migration. It uses only realistic operator signals available causally — recent
arrival pressure, burstiness, output-length shape, SLA class — to set the serving-stack levers a real vLLM/TGI
deployment has (continuous batching, SLA-aware ordering, KV-aware routing, rack-local placement, backlog
autoscaling + warm pool, class admission). It is **stronger than `sla_aware`** (the honest production bar) but
runs the deployed model as-is (bf16 / base clock / no migration / no spec) — the economic arbitrage is
Aurelius's edge, not the baseline's. Deterministic (no RNG).
"""

from __future__ import annotations

import statistics

# Static ladder baselines (decide_fn returns the same dict every period). Keep in sync by VALUE with the
# canonical definitions (sla_aware == controller.SLA_AWARE_FALLBACK); do not import the controller to avoid
# any coupling to the MPC path.
FIFO = {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off"}
SLA_AWARE = {"capacity": "backlog_aware", "ordering": "abs_conformal", "admission": "off"}
# vLLM default: continuous batching + roughly-FIFO order + reactive autoscale; NO SLA scheduler / KV routing.
VLLM_ONLY = {"capacity": "backlog_aware", "ordering": "fifo", "admission": "off",
             "batching_policy": "balanced", "routing_policy": "round_robin"}
# topology-aware: rack locality but no SLA scheduler (isolates the placement lever from the ordering lever).
TOPOLOGY_AWARE = {"capacity": "backlog_aware", "ordering": "fifo", "admission": "off",
                  "placement_policy": "rack_local"}

# the static ladder registry (excludes production_scheduler, which is stateful — see below).
STATIC_BASELINES = {"fifo": FIFO, "sla_aware": SLA_AWARE, "vllm_only": VLLM_ONLY,
                    "topology_aware": TOPOLOGY_AWARE}

# --- reactive-operator thresholds (ops-motivated defaults; NOT tuned to any benchmark) ---------------
_RECENT_K = 4                 # frames of recent history the scheduler reacts to
_RISING = 1.15                # arrival rising if recent EWMA > _RISING × the prior window
_BURST_HI = 1.0               # interarrival CV considered bursty (queue/tail risk)
_DECODE_HEAVY = 200.0         # mean output tokens above which decode dominates (batch-friendly)
_HEADROOM = 1.25              # capacity safety headroom under pressure (a standard ~80%-util autoscaling
                              # target) ON TOP of backlog_aware sizing — NOT tuned to any benchmark


def _ewma(vals, alpha=0.5):
    if not vals:
        return 0.0
    e = vals[0]
    for v in vals[1:]:
        e = alpha * v + (1 - alpha) * e
    return e


def _attr(frame, name, default=0.0):
    """Read a feature off a Frame dataclass / dict / namespace (causal observables only)."""
    if isinstance(frame, dict):
        return float(frame.get(name, default))
    return float(getattr(frame, name, default))


class ProductionScheduler:
    """The canonical production heuristic. `decide(history)` reads the recent causal frames and returns a
    fixed action dict (deterministic). No future/oracle/economic information; no planner/MPC search."""

    name = "production_scheduler"

    def __init__(self, *, recent_k: int = _RECENT_K):
        self.recent_k = recent_k

    def decide(self, history) -> dict:
        # static safe policy until there is enough recent history to react to.
        if not history:
            return {**SLA_AWARE, "batching_policy": "balanced", "routing_policy": "kv_aware",
                    "placement_policy": "rack_local", "prewarm_policy": "off"}
        recent = history[-self.recent_k:]
        prior = history[-2 * self.recent_k:-self.recent_k] or recent
        ar_recent = _ewma([_attr(f, "arrival_rate") for f in recent])
        ar_prior = _ewma([_attr(f, "arrival_rate") for f in prior])
        tok = _ewma([_attr(f, "output_token_mean") for f in recent])
        cv = statistics.mean([_attr(f, "interarrival_cv", 1.0) for f in recent]) if recent else 1.0

        rising = ar_prior > 1e-9 and ar_recent > _RISING * ar_prior
        bursty = cv >= _BURST_HI
        decode_heavy = tok >= _DECODE_HEAVY

        # SLA-aware ordering + backlog autoscaling are always on (a real scheduler protects deadlines).
        action = {"capacity": "backlog_aware", "ordering": "abs_conformal"}
        # admission: defer best-effort only under genuine pressure (no free shedding otherwise).
        action["admission"] = "class_aware" if (rising or bursty) else "off"
        # autoscale / warm pool: backlog_aware already sizes replicas to load; under pressure add a modest
        # safety headroom (a standard ~80%-util autoscaling target). Never under-provision (≥ 1.0). A real
        # autoscaler does NOT blanket-double the fleet — that would forfeit cost-efficiency for no SLA gain.
        action["capacity_multiplier"] = _HEADROOM if (rising or bursty) else 1.0
        # CONTINUOUS BATCHING IS ALWAYS ON — it is the defining vLLM/TGI feature; production never disables it.
        # Load-shaped between the two continuous-batch operating points: pack throughput when decode-heavy,
        # otherwise the balanced continuous-batch default. (Burstiness is handled by admission + headroom
        # above, NOT by shrinking the batch — a smaller batch would only raise cost per request.)
        action["batching_policy"] = "aggressive" if decode_heavy else "balanced"
        # KV-aware routing + rack-local placement (the serving-stack levers a real deployment has).
        action["routing_policy"] = "kv_aware"
        action["placement_policy"] = "rack_local"
        # Warm pool: handled by `backlog_aware` autoscaling itself — it keeps recently-used replicas warm
        # through the idle timeout (world_simulator WARM_IDLE_TIMEOUT_S=300), the standard cost-efficient
        # production warm pool. We do NOT run an EAGER prewarm pool on top: spinning replicas up ahead of
        # demand holds idle GPU-hours that, absent an actual cold-start to prevent, are pure cost — a
        # cost-conscious operator does not do this. (Confirmed empirically: eager prewarm added warm-hold
        # cost dwarfing the served work with zero cold starts avoided. Aurelius MAY still prewarm — that is
        # its optimisation to make when it pays; the production baseline does not gamble idle capacity.)
        action["prewarm_policy"] = "off"
        # the deployed model as-is — NO precision / clock / migration / spec arbitrage (Aurelius's edge).
        action["precision_policy"] = "bf16"
        action["clock_policy"] = "base"
        action["migration_policy"] = "off"
        action["spec_decode_policy"] = "off"
        return action


def baseline_decider(name: str):
    """Return a deterministic `decide_fn(history) -> action_dict` for a ladder baseline. `production_scheduler`
    is the stateful heuristic; the rest are static. Raises for unknown names (oracle / aurelius_mpc arms are
    NOT here — they are the MPC controller path, kept strictly separate)."""
    if name == "production_scheduler":
        sched = ProductionScheduler()
        return sched.decide
    if name in STATIC_BASELINES:
        policy = dict(STATIC_BASELINES[name])
        return lambda history: dict(policy)
    raise ValueError(f"unknown production baseline {name!r} (oracle / aurelius arms are the MPC path)")


# the canonical baseline registry for the ladder (heuristic arms only — NOT the Aurelius MPC arms).
BASELINE_REGISTRY = {n: (lambda nm=n: baseline_decider(nm)) for n in
                     ("fifo", "vllm_only", "topology_aware", "sla_aware", "production_scheduler")}

# the headline production-comparable baseline (future gp/$ claims default to this, not fifo or oracle).
HEADLINE_BASELINE = "production_scheduler"


def is_economic_or_oracle_free(action: dict) -> bool:
    """True iff an action dict uses NO economic-arbitrage / quality-risked lever (the production-baseline
    contract): bf16 precision, base clock, migration off, spec off. Used by the tests."""
    return (action.get("precision_policy", "bf16") == "bf16"
            and action.get("clock_policy", "base") == "base"
            and action.get("migration_policy", "off") == "off"
            and action.get("spec_decode_policy", "off") == "off")


__all__ = ["ProductionScheduler", "baseline_decider", "BASELINE_REGISTRY", "HEADLINE_BASELINE",
           "STATIC_BASELINES", "FIFO", "SLA_AWARE", "VLLM_ONLY", "TOPOLOGY_AWARE",
           "is_economic_or_oracle_free"]
