"""Candidate generators + regime classification for the search-method tournament.

The tournament compares search methods on the SAME action space. This module supplies the candidate
sources every method draws from: a **regime classifier** (11 bottleneck labels), **regime-aware
generators** (physics-guided / random / fixed-grid / expanded-grid), and the **required-anchor contract**.

Hard invariant (the honesty core): the candidate set a method searches must ALWAYS contain the required
anchors — `ActionBundle()` (neutral), the production SLA-aware baseline bundle, the previous-best MPC bundle
(when known), the `ACTION_SUBSET_CONTAINMENT` diagnostic winner (`fp8 + aggressive + high`), and the entire
bounded core grid (`precision × batching × capacity × clock`). Physics-guided generation may ADD candidates;
it may not REMOVE an anchor. `enforce_anchors` returns the missing anchors so the caller can FAIL the
diagnostic. The regime labels are candidate-generation PRIORS only — the simulator still chooses by causal
rollout. Reuses the merged `physics_guided_candidates.py` (no duplication of the prior logic).
"""

from __future__ import annotations

import random as _random
from itertools import product

from ..actions import ACTION_SPECS, CONNECTED_SURFACES, ActionBundle
from ..physics_guided_candidates import (
    NEUTRAL_BUNDLE,
    SAFE_BASELINE_BUNDLE,
    PhysicsGuidedCandidateGenerator,
    PlannerRegimeState,
)

# the 11 bottleneck regimes the classifier may report (a state can be in several at once).
REGIMES = ("memory_bound", "compute_bound", "queue_bound", "SLA_tight", "SLA_slack",
           "power_expensive", "power_cheap", "HBM_pressure", "KV_reuse_possible",
           "migration_pressure", "capacity_pressure")

# the ACTION_SUBSET_CONTAINMENT diagnostic winner — a REQUIRED anchor in every candidate set.
DIAGNOSTIC_WINNER = ActionBundle(precision_policy="fp8", batching_policy="aggressive", clock_policy="high")

# the bounded core grid surfaces (the fixed multi-knob grid: precision × batching × capacity × clock).
# SAFE options only (no int4 — quality-risked, opt-in elsewhere). 2·3·3·3 = 54 bundles.
CORE_GRID_SURFACES = {
    "precision_policy": ("bf16", "fp8"),
    "batching_policy": ("conservative", "balanced", "aggressive"),
    "capacity_multiplier": (1.0, 0.75, 1.5),
    "clock_policy": ("base", "low", "high"),
}
# the expanded grid adds speculative decoding (precision × batching × capacity × clock × spec). 54·3 = 162.
EXPANDED_EXTRA_SURFACES = {"spec_decode_policy": ("off", "shallow", "medium")}

_PRESSURE_HI = 0.60
_PRICE_HI = 0.70
_PRICE_LO = 0.30
_SLACK_POS = 0.10


def _grid(surfaces: dict) -> list:
    names = list(surfaces)
    return [ActionBundle(**dict(zip(names, combo))) for combo in product(*(surfaces[n] for n in names))]


def core_grid() -> list:
    """The fixed bounded multi-knob grid (precision × batching × capacity × clock) — 54 bundles."""
    return _grid(CORE_GRID_SURFACES)


def expanded_grid() -> list:
    """Core grid × speculative decoding (precision × batching × capacity × clock × spec) — 162 bundles."""
    return _grid({**CORE_GRID_SURFACES, **EXPANDED_EXTRA_SURFACES})


def classify_regimes(state: PlannerRegimeState, *, kv_reuse: float | None = None,
                     migration_pressure: float | None = None) -> set:
    """Classify the planning state into the active bottleneck regimes (soft, multi-label). PRIORS only —
    the reward is unaffected; the simulator chooses by rollout. `kv_reuse` / `migration_pressure` are extra
    optional signals beyond `PlannerRegimeState` (prefix-reuse opportunity, pending consolidation)."""
    labels: set = set()
    if state.decode_regime == "memory_bandwidth_bound":
        labels.add("memory_bound")
    elif state.decode_regime == "compute_bound":
        labels.add("compute_bound")
    if state.queue_pressure >= _PRESSURE_HI:
        labels.add("queue_bound")
    if state.capacity_pressure >= _PRESSURE_HI:
        labels.add("capacity_pressure")
    if state.sla_tight:
        labels.add("SLA_tight")
    elif state.sla_slack is not None and state.sla_slack >= _SLACK_POS:
        labels.add("SLA_slack")
    if state.price_percentile is not None:
        if state.price_percentile >= _PRICE_HI:
            labels.add("power_expensive")
        elif state.price_percentile <= _PRICE_LO:
            labels.add("power_cheap")
    if state.hbm_pressure is not None and state.hbm_pressure >= _PRESSURE_HI:
        labels.add("HBM_pressure")
    if kv_reuse is not None and kv_reuse >= 0.3:
        labels.add("KV_reuse_possible")
    if migration_pressure is not None and migration_pressure >= _PRESSURE_HI:
        labels.add("migration_pressure")
    return labels


def required_anchors(state: PlannerRegimeState | None = None, *, prev_best=None,
                     include_core_grid: bool = True) -> list:
    """The anchors that MUST be in every candidate set (deduped, named-anchors first, then the core grid).

    Named anchors: neutral, the production SLA-aware baseline, the previous-best MPC bundle (if given), and
    the ACTION_SUBSET_CONTAINMENT diagnostic winner (`fp8 + aggressive + high`). With `include_core_grid`
    the entire bounded core grid follows. Named anchors come first so an evaluation budget ≥ ~5 always scores
    the critical ones."""
    named = [NEUTRAL_BUNDLE, SAFE_BASELINE_BUNDLE, DIAGNOSTIC_WINNER,
             ActionBundle(precision_policy="fp8", batching_policy="aggressive")]
    if prev_best is not None and hasattr(prev_best, "to_dict"):
        named.insert(2, prev_best)
    out, seen = [], set()
    for b in named + (core_grid() if include_core_grid else []):
        k = _key(b)
        if k not in seen:
            seen.add(k)
            out.append(b)
    return out


def named_anchor_keys(prev_best=None) -> set:
    """The keys of the must-evaluate named anchors (used by the anchor-containment check / the hard rule)."""
    named = [NEUTRAL_BUNDLE, SAFE_BASELINE_BUNDLE, DIAGNOSTIC_WINNER,
             ActionBundle(precision_policy="fp8", batching_policy="aggressive")]
    if prev_best is not None and hasattr(prev_best, "to_dict"):
        named.append(prev_best)
    return {_key(b) for b in named}


def enforce_anchors(candidates: list, *, prev_best=None, include_core_grid: bool = True) -> tuple:
    """Prepend any missing anchors to `candidates` and report what was missing.

    Returns `(candidates_with_anchors, missing_named_anchor_keys)`. The named anchors must never be absent —
    if `missing` is non-empty the caller FAILS the diagnostic (a generator dropped a required anchor)."""
    have = {_key(b) for b in candidates}
    anchors = required_anchors(prev_best=prev_best, include_core_grid=include_core_grid)
    merged, seen = [], set()
    for b in anchors + list(candidates):
        k = _key(b)
        if k not in seen:
            seen.add(k)
            merged.append(b)
    # missing = named anchors that were absent from the ORIGINAL candidate list (now restored by anchors)
    missing = named_anchor_keys(prev_best) - have
    return merged, missing


def _ordered_unique(*groups) -> list:
    """Concatenate candidate groups, deduped, PRESERVING ORDER. The first group is highest priority under a
    budget cap — so a method's own candidates are evaluated before the core-grid fill that the invariant
    requires (the core grid is still in the set; it just sits last)."""
    out, seen = [], set()
    for g in groups:
        for b in g:
            k = _key(b)
            if k not in seen:
                seen.add(k)
                out.append(b)
    return out


def physics_guided_candidates(state: PlannerRegimeState, *, prev_best=None,
                              optional_surfaces: tuple = (), allow_quality_risk: bool = False) -> list:
    """Regime-aware physics-guided candidate set (reuses `PhysicsGuidedCandidateGenerator`). Order:
    named anchors → the REGIME-specific generated candidates → the core-grid fill. So the regime prior
    controls which candidates a budget evaluates first, while the invariant (core grid present) still holds
    and the named anchors (incl. the diagnostic winner) are always evaluated."""
    if prev_best is not None:
        state.prev_bundle = prev_best
    gen = PhysicsGuidedCandidateGenerator(allow_quality_risk=allow_quality_risk)
    cs = gen.generate(state, optional_surfaces=optional_surfaces)
    named = required_anchors(prev_best=prev_best, include_core_grid=False)
    return _ordered_unique(named, cs.bundles, core_grid())


def random_candidates(n: int, *, seed: int = 0, prev_best=None, include_simulated: bool = False) -> list:
    """Deterministic RANDOM candidate set over the connected action space (the physics-prior ablation
    control). Order: named anchors → random candidates → core-grid fill (same anchor contract, no regime
    prior — so any physics-guided advantage over this control is attributable to the prior, not the anchors)."""
    rng = _random.Random(seed)
    surfaces = list(CONNECTED_SURFACES) + ([] if not include_simulated else [])
    out = []
    for _ in range(max(0, n)):
        choice = {s: rng.choice(ACTION_SPECS[s].options) for s in surfaces}
        # never propose int4 (quality-risked) or co-location (no background work) in the random control
        if choice.get("precision_policy") == "int4":
            choice["precision_policy"] = "fp8"
        out.append(ActionBundle(**choice))
    named = required_anchors(prev_best=prev_best, include_core_grid=False)
    return _ordered_unique(named, out, core_grid())


def _key(b) -> tuple:
    return tuple(sorted(b.to_dict().items()))


__all__ = [
    "REGIMES", "DIAGNOSTIC_WINNER", "CORE_GRID_SURFACES", "EXPANDED_EXTRA_SURFACES",
    "core_grid", "expanded_grid", "classify_regimes", "required_anchors", "named_anchor_keys",
    "enforce_anchors", "physics_guided_candidates", "random_candidates",
]
