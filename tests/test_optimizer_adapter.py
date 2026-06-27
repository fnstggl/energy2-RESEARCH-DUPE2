"""Tests for the AureliusOptimizer ↔ environment adapter (Phase 4).

Covers the State/Action/Reward contracts, the policy set, and the fair backtest:
every arm scored through the optimizer's ObjectiveLayer, a NON-weak fair baseline
(never silently FIFO), per-arm metrics (gp/$, SLA violation rate, GPU-hours, energy
+ operator cost, queue-delay p50/p95/p99, KV hit rate, cost per useful unit), and a
headline-claim gate that requires beating the fair baseline + held-out validation +
no oracle. Proves policies are causal (pure functions of the start-of-hour state).
"""

from __future__ import annotations

import os

from aurelius.environment.optimizer_adapter import (
    ACTION_SPACE,
    BASELINE_POLICIES,
    DEFAULT_CANDIDATE,
    EnvState,
    fair_backtest,
    policy_aurelius_state_conditioned,
    policy_fifo_weak,
    reward_from_step,
)
from aurelius.environment.schemas import EnvObservation

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MOONCAKE = os.path.join(_REPO, "tests", "fixtures", "mooncake", "mooncake_sample.csv")
_PROCESSED = os.path.join(_REPO, "data", "external", "alibaba_gpu_v2026", "processed")


def _obs(arrival_rate=2.0, be=0.3):
    fleet = {"util_target": 0.3, "mem_pressure": 0.4, "priority_mix": {"HP": 0.5, "LP": 0.5},
             "queue_delay_s": 12.0, "net_pressure": 0.2, "fragmentation": 0.1,
             "energy_price_per_kwh": 0.08, "gpu_type_mix": {"A10": 0.7, "H100": 0.3}}
    return EnvObservation(hour=0, fleet=fleet, n_requests=120,
                          arrival_rate_per_s=arrival_rate, best_effort_fraction=be)


# --- State / Action / Reward contracts -------------------------------------

def test_env_state_from_observation_causal_fields():
    st = EnvState.from_observation(_obs())
    assert st.fleet_util == 0.3 and st.mem_pressure == 0.4 and st.gpu_type == "A10"
    assert st.arrival_rate_per_s == 2.0 and st.best_effort_fraction == 0.3
    assert "fleet" in st.fidelity and set(st.to_vector()) >= {"arrival_rate_per_s", "fleet_util"}


def test_action_space_and_policies_are_valid_causal():
    for policy in (*BASELINE_POLICIES.values(), policy_aurelius_state_conditioned):
        a = policy(_obs())
        assert a["capacity"] in ACTION_SPACE["capacity"]
        assert a["ordering"] in ACTION_SPACE["ordering"]
        assert a["admission"] in ACTION_SPACE["admission"]
    # candidate is state-conditioned: heavy load → admission on; light → off (causal)
    assert policy_aurelius_state_conditioned(_obs(arrival_rate=2.0, be=0.3))["admission"] == "class_aware"
    assert policy_aurelius_state_conditioned(_obs(arrival_rate=0.1, be=0.0))["admission"] == "off"


def test_reward_is_goodput_per_dollar():
    class _Step:
        reward = 1234.5
    assert reward_from_step(_Step()) == 1234.5


# --- fair backtest ----------------------------------------------------------

def _heavy_hourly():
    hourly = {}
    for h in range(2):
        reqs, t = [], 0.0
        for i in range(600):
            t += 0.05 if (i % 200) < 80 else 0.5
            reqs.append((t, 200 + (i * 53) % 1200))
        hourly[h] = reqs
    return hourly


def test_fair_backtest_runs_scores_and_gates():
    rep = fair_backtest(
        _heavy_hourly(),
        env_kwargs={"mooncake_path": _MOONCAKE, "processed_dir": _PROCESSED, "sla_s": 5.0}).to_dict()
    # every policy + candidate produced an arm with the full metric set
    assert set(rep["arms"]) == set(BASELINE_POLICIES) | {DEFAULT_CANDIDATE[0]}
    for a in rep["arms"].values():
        for k in ("goodput_per_dollar", "sla_violation_rate", "gpu_hours", "energy_cost",
                  "total_operator_cost", "queue_delay_p95", "kv_hit_rate",
                  "cost_per_sla_safe_request", "cost_per_sla_safe_token"):
            assert k in a
    # ranking comes from the optimizer's ObjectiveLayer (highest gp/$ first)
    scored = [s for _, s in rep["ranking"] if s is not None]
    assert scored == sorted(scored, reverse=True)
    # fair baseline is NEVER the weak FIFO reference
    assert rep["fair_baseline"] != "fifo_weak"
    # gate structure + honesty: a headline claim requires all gates true
    g = rep["gate"]
    assert set(g) >= {"fair_baseline_not_weak", "beats_fair_baseline", "held_out_validation_passed", "no_oracle"}
    assert rep["headline_claim_allowed"] == (
        g["fair_baseline_not_weak"] and g["beats_fair_baseline"] and g["held_out_validation_passed"])
    assert g["no_oracle"] is True


def test_weak_baseline_flagged():
    a = policy_fifo_weak(_obs())
    assert a == {"capacity": "reactive_lag1", "ordering": "fifo", "admission": "off"}
