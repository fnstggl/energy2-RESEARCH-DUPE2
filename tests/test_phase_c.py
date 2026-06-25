"""Tests for the Phase C three-way, on-demand-priced benchmark harness.

Verifies: the fair three-way (Current-Main vs Best-Aurelius vs Candidate) on a
pure on-demand denominator (no spot discount), reproducible seed + trace-content
hash, ranking by SLA-safe goodput/$, and that a non-deployable (oracle) arm can
never be presented as the deployable winner.
"""

from __future__ import annotations

from aurelius.benchmarks.phase_c import (
    ROLE_BEST_AURELIUS,
    ROLE_CANDIDATE,
    ROLE_CURRENT_MAIN,
    ArmSpec,
    run_three_way,
    standard_replica_scaling_arms,
    trace_content_hash,
)


def _trace(n=1500):
    out = []
    for i in range(n):
        tok = 100 + (i % 6) * 50 + (250 if 400 < i < 650 else 0)
        out.append((float(i) * 1.5, tok))
    return out


def test_trace_content_hash_deterministic_and_sensitive():
    raw = _trace()
    h1 = trace_content_hash(raw, tick_seconds=60.0, warp=1.0, sla_s=10.0)
    h2 = trace_content_hash(raw, tick_seconds=60.0, warp=1.0, sla_s=10.0)
    assert h1 == h2 and len(h1) == 16
    # a different trace -> different hash
    raw2 = _trace()
    raw2[0] = (raw2[0][0], raw2[0][1] + 1)
    assert trace_content_hash(raw2, tick_seconds=60.0, warp=1.0, sla_s=10.0) != h1
    # different SLA provenance -> different hash
    assert trace_content_hash(raw, tick_seconds=60.0, warp=1.0, sla_s=20.0) != h1


def test_standard_three_way_roles_and_deployability():
    raw = _trace()
    arms = standard_replica_scaling_arms(raw, 60.0, 1.0, 10.0)
    roles = [a.role for a in arms]
    assert roles == [ROLE_CURRENT_MAIN, ROLE_BEST_AURELIUS, ROLE_CANDIDATE]
    # the standard arms are ALL deployable + causal (no oracle, no spot)
    assert all(a.deployable and not a.uses_future_info for a in arms)


def test_run_three_way_on_demand_ranking_and_provenance():
    raw = _trace()
    arms = standard_replica_scaling_arms(raw, 60.0, 1.0, 10.0)
    res = run_three_way(
        raw, arms, tick_seconds=60.0, warp=1.0, sla_s=10.0,
        seed=42, trace_id="unit_trace",
    )
    # on-demand denominator, provenance serialized
    assert res.denominator == "on_demand"
    assert res.seed == 42
    assert res.trace_hash == trace_content_hash(
        raw, tick_seconds=60.0, warp=1.0, sla_s=10.0
    )
    # three arms, ranked descending by goodput/$
    assert len(res.arms) == 3
    gpds = [g for _, g in res.ranking]
    assert gpds == sorted(gpds, reverse=True)
    # winner is the top of the ranking
    assert res.winner == res.ranking[0][0]
    # round-trip serialization
    d = res.to_dict()
    assert d["denominator"] == "on_demand" and len(d["arms"]) == 3


def test_oracle_arm_never_wins_deployable_slot():
    raw = _trace()
    arms = standard_replica_scaling_arms(raw, 60.0, 1.0, 10.0)
    # add a cheap ORACLE arm (c=1 everywhere): high goodput/$ but NOT deployable.
    n_ticks = len(arms[0].c_schedule)
    arms.append(ArmSpec(
        "oracle_cheap", ROLE_CANDIDATE, [1] * n_ticks,
        uses_future_info=True, deployable=False,
        classification="oracle_reference",
    ))
    res = run_three_way(
        raw, arms, tick_seconds=60.0, warp=1.0, sla_s=10.0,
        seed=7, trace_id="unit_trace",
    )
    # the deployable winner is one of the deployable arms, never the oracle
    deployable_names = {a.name for a in res.arms if a.deployable}
    assert res.deployable_winner in deployable_names
    assert res.deployable_winner != "oracle_cheap"
