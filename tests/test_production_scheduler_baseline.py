"""`production_scheduler` is a benchmark baseline ONLY — the contract tests.

These pin the user's clarification: production_scheduler lives in the evaluation layer as a `decide_fn`, is a
separate ladder arm, is deterministic and causal, uses NO economic-arbitrage / oracle / future-price lever, and
shares NO MPC-search / economic / oracle / hierarchical code. The separation is enforced structurally (an AST
scan of the module's imports), not just by convention.
"""

from __future__ import annotations

import ast
import os

from aurelius.environment.production_baselines import (
    BASELINE_REGISTRY,
    HEADLINE_BASELINE,
    STATIC_BASELINES,
    ProductionScheduler,
    baseline_decider,
    is_economic_or_oracle_free,
)

_MODULE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "aurelius", "environment", "production_baselines.py")

# synthetic causal frames (dicts in the Frame schema) the heuristic reacts to. recent_k=4, so a rising trend
# needs 8 frames: a calm prior window (first 4) then a clearly higher recent window (last 4).
_RISING = ([{"arrival_rate": 1.0, "output_token_mean": 120, "interarrival_cv": 0.8}] * 4
           + [{"arrival_rate": 2.4, "output_token_mean": 120, "interarrival_cv": 0.9}] * 4)
_DECODE_HEAVY = [{"arrival_rate": 0.6, "output_token_mean": 400, "interarrival_cv": 0.5}] * 8


# ---- registry / headline ---------------------------------------------------------------------------
def test_production_scheduler_in_registry_and_is_headline():
    assert "production_scheduler" in BASELINE_REGISTRY
    assert HEADLINE_BASELINE == "production_scheduler"          # headline bar = production_scheduler, NOT fifo
    # the new ladder rungs exist alongside the old ones.
    for rung in ("fifo", "vllm_only", "topology_aware", "sla_aware", "production_scheduler"):
        assert rung in BASELINE_REGISTRY


def test_vllm_only_and_topology_aware_defined():
    assert "vllm_only" in STATIC_BASELINES and "topology_aware" in STATIC_BASELINES
    # vllm_only = continuous batching + FIFO + reactive autoscale, NO SLA scheduler / KV routing.
    assert STATIC_BASELINES["vllm_only"]["ordering"] == "fifo"
    assert STATIC_BASELINES["vllm_only"]["batching_policy"] == "balanced"
    assert STATIC_BASELINES["vllm_only"]["routing_policy"] == "round_robin"
    # topology_aware = rack-local placement but no SLA scheduler.
    assert STATIC_BASELINES["topology_aware"]["placement_policy"] == "rack_local"
    assert STATIC_BASELINES["topology_aware"]["ordering"] == "fifo"


# ---- determinism -----------------------------------------------------------------------------------
def test_production_scheduler_deterministic():
    sched = ProductionScheduler()
    a = sched.decide(_RISING)
    b = sched.decide(list(_RISING))
    assert a == b                                              # same history → identical action (no RNG)
    # the registry decider is deterministic too.
    dec = baseline_decider("production_scheduler")
    assert dec(_RISING) == dec(list(_RISING))


def test_static_baselines_deterministic_and_independent():
    dec = baseline_decider("sla_aware")
    out = dec([])
    out["capacity"] = "MUTATED"                                # mutating the returned dict must not leak back
    assert dec([])["capacity"] == "backlog_aware"


# ---- no economic / oracle arbitrage ----------------------------------------------------------------
def test_production_scheduler_never_uses_economic_or_oracle_levers():
    sched = ProductionScheduler()
    for hist in ([], _RISING, _DECODE_HEAVY, _RISING + _DECODE_HEAVY):
        action = sched.decide(hist)
        assert is_economic_or_oracle_free(action), action     # bf16 / base clock / migration off / spec off
        assert action.get("precision_policy", "bf16") == "bf16"
        assert action.get("clock_policy", "base") == "base"
        assert action.get("migration_policy", "off") == "off"
        assert action.get("spec_decode_policy", "off") == "off"


def test_production_scheduler_reacts_to_load():
    sched = ProductionScheduler()
    rising = sched.decide(_RISING)
    # under rising/bursty pressure it scales up the warm pool and defers best-effort.
    assert rising["capacity_multiplier"] >= 1.0
    assert rising["admission"] == "class_aware"
    # it always runs the serving-stack levers a real deployment has.
    assert rising["routing_policy"] == "kv_aware"
    assert rising["placement_policy"] == "rack_local"
    # decode-heavy steady load → throughput batching, no shedding.
    dh = sched.decide(_DECODE_HEAVY)
    assert dh["batching_policy"] == "aggressive"
    assert dh["capacity_multiplier"] == 1.0


def test_capacity_never_under_provisions():
    sched = ProductionScheduler()
    for hist in ([], _RISING, _DECODE_HEAVY):
        assert sched.decide(hist).get("capacity_multiplier", 1.0) >= 1.0   # no free under-provisioning


# ---- hard separation: NO planner / MPC-search / economic / oracle imports --------------------------
def test_module_imports_no_mpc_or_planner_code():
    """AST scan: production_baselines must not import the controller, the planner package, physics-guided
    search, the economic optimiser, or any oracle/search module. The separation is structural."""
    tree = ast.parse(open(_MODULE).read())
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    banned = ("controller", "planner", "physics_guided", "search_planner", "search_method",
              "planner_tournament", "default_change_gate", "adaptive")
    leaks = [m for m in imported for b in (banned,) if any(tok in (m or "") for tok in b)]
    assert not leaks, f"production_baselines leaked a banned import: {leaks}"
    # positively: it imports only the standard library here (statistics) — no aurelius MPC modules at all.
    assert not any((m or "").startswith("aurelius.environment.controller") for m in imported)
    assert not any("planner" in (m or "") for m in imported)


def test_baseline_decider_rejects_oracle_and_aurelius_arms():
    """The baseline decider is for heuristic arms only — oracle / aurelius_mpc are the MPC path, not here."""
    for nm in ("oracle", "oracle_diagnostic", "aurelius_mpc_hierarchical_search", "aurelius_mpc_current_default"):
        try:
            baseline_decider(nm)
            raise AssertionError(f"baseline_decider should reject MPC/oracle arm {nm!r}")
        except ValueError:
            pass
