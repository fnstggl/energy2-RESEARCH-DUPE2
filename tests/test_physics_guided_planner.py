"""Regression guarantees for the physics-guided planner (Phase 7).

These prove the honesty + correctness contract WITHOUT the world simulator (synthetic score functions):
anchors are always present, the known-good bundles are contained, the beam captures coupling a coordinate
descent misses, progressive widening expands/stops by margin, the regret auditor detects a missed optimum,
the candidate count stays bounded, co-location is never generated without background work, the reward/gate
are untouched, and the planner is deterministic. Fast (no rollouts)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aurelius.environment.actions import ActionBundle
from aurelius.environment.physics_guided_candidates import (
    KNOWN_STRONG_BUNDLES,
    NEUTRAL_BUNDLE,
    PhysicsGuidedCandidateGenerator,
    PlannerRegimeState,
)
from aurelius.environment.physics_guided_planner import (
    BoundedBeamPlanner,
    beam_search,
    coordinate_descent,
)
from aurelius.environment.search_regret_auditor import (
    audit_search_regret,
    clock_only_candidates,
)

WINNER = ActionBundle(precision_policy="fp8", batching_policy="aggressive", clock_policy="high")


def _key(b):
    return tuple(sorted(b.to_dict().items()))


def _states():
    return [
        PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.9),
        PlannerRegimeState(decode_regime="compute_bound", queue_pressure=0.2, confidence=0.8),
        PlannerRegimeState(decode_regime=None, capacity_pressure=0.7, confidence=0.5),
        PlannerRegimeState(decode_regime="memory_bandwidth_bound", price_percentile=0.95, sla_slack=0.4),
    ]


# --- anchors + containment ----------------------------------------------------------------------------
def test_neutral_always_included():
    gen = PhysicsGuidedCandidateGenerator()
    for st in _states():
        cs = gen.generate(st)
        assert cs.contains(NEUTRAL_BUNDLE), f"neutral missing in regime {st.regime_label}"
        assert cs.anchor_flags["neutral"] is True


def test_previous_best_always_included_when_provided():
    gen = PhysicsGuidedCandidateGenerator()
    prev = ActionBundle(routing_policy="kv_aware", capacity_multiplier=1.5, clock_policy="low")
    for st in _states():
        st.prev_bundle = prev
        cs = gen.generate(st)
        assert cs.contains(prev), "previous best not contained"
        assert cs.anchor_flags["previous_best"] is True


def test_known_strong_fp8_aggressive_always_included():
    gen = PhysicsGuidedCandidateGenerator()
    for st in _states():
        cs = gen.generate(st)
        for b, _label in KNOWN_STRONG_BUNDLES:
            assert cs.contains(b), f"known-strong {b.non_default_surfaces()} missing in {st.regime_label}"
        assert cs.contains(WINNER), "the PR #121 24-grid winner is not contained"
        assert cs.anchor_flags["known_strong"] is True


def test_physics_grid_contains_24grid_winner():
    """The fp8 + aggressive + high winner (PR #121 24-grid argmax) is reachable in every regime."""
    gen = PhysicsGuidedCandidateGenerator()
    for st in _states():
        assert gen.generate(st).contains(WINNER)


# --- no clock-only fallback ---------------------------------------------------------------------------
def test_never_falls_back_to_clock_only():
    """The generated set is never the 3-bundle clock-only set, and always carries non-clock surfaces."""
    gen = PhysicsGuidedCandidateGenerator()
    clock_keys = {_key(b) for b in clock_only_candidates()}
    for st in _states():
        cs = gen.generate(st)
        got = {_key(b) for b in cs.bundles}
        assert got != clock_keys
        assert len(cs.bundles) > len(clock_keys)
        # at least one candidate moves precision or batching (a non-clock surface)
        assert any(b.precision_policy != "bf16" or b.batching_policy != "conservative" for b in cs.bundles)


def test_clock_only_must_be_explicitly_requested():
    """A physics-guided controller never sets a clock-only candidate list on its own."""
    from aurelius.environment.controller import ModelPredictiveEconomicController as C
    c = C(forecasters=SimpleNamespace(fitted=False), fleet_state=None, cost_model=None)
    assert c.physics_guided is False          # opt-in
    assert c.candidates is None               # no clock-only list unless the caller sets it


# --- bounded count ------------------------------------------------------------------------------------
def test_generated_count_bounded():
    gen = PhysicsGuidedCandidateGenerator(hard_cap=120)
    for st in _states():
        cs = gen.generate(st)
        assert cs.generated_count <= 120
        assert cs.generated_count >= 8        # always at least the anchor spine


def test_hard_cap_drops_only_grid_never_anchors():
    gen = PhysicsGuidedCandidateGenerator(hard_cap=14)   # force a cap below the grid size
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3)
    cs = gen.generate(st)
    assert cs.generated_count <= 14
    assert cs.capped is True
    # every anchor category survives the cap
    assert cs.contains(NEUTRAL_BUNDLE)
    assert cs.contains(WINNER)
    assert cs.anchor_flags["neutral"] and cs.anchor_flags["known_strong"]


# --- co-location guard --------------------------------------------------------------------------------
def test_colocation_never_generated_without_background_work():
    gen = PhysicsGuidedCandidateGenerator(background_work=False)
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound")
    cs = gen.generate(st, optional_surfaces=("colocation_policy",))
    assert all(b.colocation_policy == "off" for b in cs.bundles)
    assert "colocation_excluded" in cs.pruned_reasons


# --- beam beats coordinate descent on a coupled optimum ----------------------------------------------
def test_beam_finds_coupled_optimum_coordinate_misses():
    def coupled(b):
        fp8, ag = b.precision_policy == "fp8", b.batching_policy == "aggressive"
        if fp8 and ag:
            return 200.0
        if fp8 or ag:
            return 99.0          # each lever alone is slightly worse than neutral
        return 100.0
    surfaces = {"precision_policy": ("bf16", "fp8"), "batching_policy": ("conservative", "aggressive")}
    cd_b, cd_s, _ = coordinate_descent(surfaces, coupled)
    beam, _ = beam_search(surfaces, coupled, beam_width=4, anchors=())   # no anchor for the combo
    assert cd_b.non_default_surfaces() == {}                  # coordinate descent stuck at neutral
    assert beam[0][1] == 200.0 and beam[0][1] > cd_s          # beam found the coupled optimum


# --- progressive widening -----------------------------------------------------------------------------
def test_widening_expands_when_margin_small():
    pl = BoundedBeamPlanner()
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.9)
    _, rep = pl.plan(st, lambda b: 100.0 + 1e-4 * len(b.non_default_surfaces()))   # near-tie
    assert rep.widening_rounds >= 1
    assert any(t["action"] == "widened" for t in rep.widening_trace)


def test_widening_stops_when_margin_large():
    pl = BoundedBeamPlanner()
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.9)

    def clear(b):
        # pin the Batch-1 no-op knobs too, so the winner is UNIQUE (else the kv_fp8 anchor variant of the
        # winning bundle ties it under a kv-agnostic scorer and the planner widens to break the tie).
        return 1000.0 if (b.precision_policy == "fp8" and b.batching_policy == "aggressive"
                          and b.clock_policy == "high"
                          and b.kv_cache_precision_policy == "inherit_weight_precision"
                          and b.prefill_decode_policy == "shared") else 50.0
    _, rep = pl.plan(st, clear)
    assert rep.widening_rounds == 0
    assert rep.decision_margin >= pl.margin_threshold


# --- search-regret auditor detects a missed optimum --------------------------------------------------
def test_auditor_detects_missed_contained_optimum():
    """A broken planner that evaluates the known-strong winner but returns neutral must FAIL the audit."""
    def score(b):
        return 500.0 if _key(b) == _key(WINNER) else 100.0

    class BrokenPlanner:
        widen = False
        def plan(self, state, score_fn, **kw):                # noqa: D401
            for b, _ in KNOWN_STRONG_BUNDLES:
                score_fn(b)                                   # evaluate them → "contained"
            score_fn(NEUTRAL_BUNDLE)
            return NEUTRAL_BUNDLE, SimpleNamespace(to_dict=lambda: {})

    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3)
    rep = audit_search_regret(score, st, planner=BrokenPlanner())
    assert rep.lost_to_contained_old_best is True
    assert rep.passed is False


def test_auditor_passes_for_real_planner():
    def score(b):
        s = 100.0
        s += 20 if b.precision_policy == "fp8" else (-40 if b.precision_policy == "int4" else 0)
        s += 15 if b.batching_policy == "aggressive" else 0
        s += 80 if (b.precision_policy == "fp8" and b.batching_policy == "aggressive") else 0
        s += 5 if b.clock_policy == "high" else 0
        s += 3 if b.capacity_multiplier == 1.5 else 0
        return s
    st = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.9)
    rep = audit_search_regret(score, st)
    assert rep.passed is True
    assert rep.strategies["physics_guided"]["regret_abs"] == 0.0          # matches exhaustive
    assert rep.strategies["clock_only"]["regret_abs"] > 0.0               # clock-only leaves value


# --- reward / gate untouched --------------------------------------------------------------------------
def test_generated_bundles_are_all_valid_actions():
    """No generated bundle ever actuates a PLANNED/REJECTED surface (reward path stays honest)."""
    from aurelius.environment.action_registry import validate_action_bundle
    gen = PhysicsGuidedCandidateGenerator()
    for st in _states():
        for b in gen.generate(st, optional_surfaces=("routing_policy", "prewarm_policy",
                                                     "spec_decode_policy", "migration_policy")).bundles:
            assert validate_action_bundle(b)["ok"], f"invalid bundle {b.non_default_surfaces()}"


def test_pareto_gate_unchanged():
    """The claim gate still blocks an SLA-worse 'win' (we did not weaken it)."""
    from aurelius.environment.training import claim_gate
    worse_sla = claim_gate({
        "mpc_controller": SimpleNamespace(goodput_per_dollar=200.0, sla_violation_rate=0.5),
        "sla_aware": SimpleNamespace(goodput_per_dollar=100.0, sla_violation_rate=0.2)})
    assert worse_sla["beats_fair_baseline"] is True
    assert worse_sla["pareto_sla_not_worse"] is False          # higher gp/$ but worse SLA → not headline-safe
    pareto = claim_gate({
        "mpc_controller": SimpleNamespace(goodput_per_dollar=200.0, sla_violation_rate=0.1),
        "sla_aware": SimpleNamespace(goodput_per_dollar=100.0, sla_violation_rate=0.2)})
    assert pareto["beats_fair_baseline"] and pareto["pareto_sla_not_worse"]


# --- determinism --------------------------------------------------------------------------------------
def test_deterministic_replay():
    def score(b):
        return 100.0 + (10 if b.precision_policy == "fp8" else 0) + (8 if b.batching_policy == "aggressive" else 0)
    pl = BoundedBeamPlanner()
    st1 = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.9)
    st2 = PlannerRegimeState(decode_regime="memory_bandwidth_bound", sla_slack=0.3, confidence=0.9)
    b1, r1 = pl.plan(st1, score)
    b2, r2 = pl.plan(st2, score)
    assert _key(b1) == _key(b2)
    assert r1.evaluated_candidates == r2.evaluated_candidates
    assert r1.selected_bundle == r2.selected_bundle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
