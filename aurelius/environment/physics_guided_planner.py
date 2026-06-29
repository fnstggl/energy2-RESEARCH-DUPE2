"""Bounded beam MPC + progressive widening over the physics-guided candidate set.

The planner takes the physics-guided candidate generator (``physics_guided_candidates.py``) and searches it
with a **bounded beam** that (a) always evaluates the guaranteed anchors and (b) builds cross-surface
combinations the way a single-dimension coordinate descent cannot — so a *coupled* optimum like
``{precision=fp8, batching=aggressive}`` (neither lever a win alone, both a win together) is found. It then
applies **progressive widening**: it starts on the 4 high-value surfaces and only expands to
routing/prewarm/spec/migration when the decision is *close* (small margin / low confidence / tight SLA /
a recent regret failure), stopping early when the margin is large. Every decision reports the raw /
generated / evaluated counts, the selected bundle, the top-K, the decision margin, runtime, whether the
anchors / previous-best / known-strong bundle were contained, and the widening trace.

It never enumerates the full Cartesian product (314,928 — intractable per-eval, see the audit) and **never
falls back to clock-only**. Deterministic: a fixed state + score function yields a fixed plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .actions import ActionBundle
from .physics_guided_candidates import (
    KNOWN_STRONG_BUNDLES,
    OPTIONAL_SURFACES,
    PhysicsGuidedCandidateGenerator,
    PlannerRegimeState,
)

_EPS = 1e-9


def _key(b) -> tuple:
    return tuple(sorted(b.to_dict().items()))


def _topk(scored: list, k: int) -> list:
    """Top-k by score, deterministic tie-break by the bundle's surface tuple; deduped by bundle key."""
    seen: set = set()
    uniq = []
    for b, s in sorted(scored, key=lambda t: (-t[1], _key(t[0]))):
        kb = _key(b)
        if kb in seen:
            continue
        seen.add(kb)
        uniq.append((b, s))
    return uniq[:k]


def _cached_score(b, score_fn, cache: dict) -> float:
    kb = _key(b)
    if kb not in cache:
        cache[kb] = float(score_fn(b))
    return cache[kb]


def beam_search(surfaces: dict, score_fn, *, beam_width: int = 8, anchors: tuple = (),
                cache: dict | None = None) -> tuple:
    """Pure bounded beam over ``surfaces`` (a ``{surface: options}`` map), seeded with the no-op +
    ``anchors`` (force-evaluated, always retained in contention). Returns ``(beam, cache)`` where ``beam``
    is the final top-``beam_width`` ``[(bundle, score), …]``. Adds one surface at a time, keeping the
    top-``beam_width`` partial bundles — capturing cross-surface coupling a coordinate descent misses.
    Deterministic. ``cache`` (bundle-key → score) is reused across widening rounds so nothing re-evaluates."""
    cache = cache if cache is not None else {}
    names = list(surfaces)
    seeds = [ActionBundle()] + [a for a in anchors]
    beam = _topk([(b, _cached_score(b, score_fn, cache)) for b in seeds], beam_width)
    for nm in names:
        expanded = list(beam)
        for b, _s in beam:
            cur = b.to_dict()
            for o in surfaces[nm]:
                if o == cur.get(nm):
                    continue
                nb = ActionBundle(**{**cur, nm: o})
                expanded.append((nb, _cached_score(nb, score_fn, cache)))
        beam = _topk(expanded, beam_width)
    return beam, cache


# the full SAFE ranges of the two cheap headline levers — the polish always considers these around the beam
# winner so a known-good point's coupling (e.g. fp8+aggressive WITH capacity=1.5 / clock=high) is never missed
# even when the regime prior gated them out of the grid. Precision/batching stay regime-pruned (no int4 here).
_POLISH_FULL = {"capacity_multiplier": (1.0, 0.75, 1.5), "clock_policy": ("base", "low", "high")}


def coordinate_descent(surfaces: dict, score_fn, *, start=None, passes: int = 3,
                       cache: dict | None = None) -> tuple:
    """Greedy single-dimension local search from ``start`` (default no-op) — the reference the beam must
    beat on coupled optima. Returns ``(best_bundle, best_score, cache)``. Used by the regret auditor and
    the beam-vs-coordinate test; NOT a fallback in the live planner."""
    cache = cache if cache is not None else {}
    cur = start if start is not None else ActionBundle()
    cur_s = _cached_score(cur, score_fn, cache)
    for _ in range(max(1, passes)):
        improved = False
        for nm in surfaces:
            for o in surfaces[nm]:
                if o == getattr(cur, nm):
                    continue
                cand = cur.with_overrides(**{nm: o})
                s = _cached_score(cand, score_fn, cache)
                if s > cur_s + _EPS:
                    cur, cur_s, improved = cand, s, True
        if not improved:
            break
    return cur, cur_s, cache


@dataclass
class PlanReport:
    """The auditable record of ONE physics-guided MPC decision (the user's required per-decision report)."""
    strategy: str = "physics_guided_beam"
    raw_candidates: int = 0                 # Cartesian size of the regime prior grid (before cap)
    generated_candidates: int = 0           # candidates the generator emitted (anchors + grid, deduped)
    evaluated_candidates: int = 0           # distinct bundles actually scored (the rollout count)
    selected_bundle: dict = field(default_factory=dict)     # non-default surfaces of the winner
    top_k: list = field(default_factory=list)               # [{surfaces, score}], best first
    decision_margin: float = 0.0            # (best − 2nd) / |best|  among evaluated (relative)
    decision_margin_abs: float = 0.0        # best − 2nd (absolute score units)
    runtime_s: float = 0.0
    anchors_included: dict = field(default_factory=dict)    # which anchor categories were present
    prev_best_contained: bool = False
    known_strong_contained: bool = False
    beam_width: int = 8
    widening_rounds: int = 0
    widening_trace: list = field(default_factory=list)      # per-round: surfaces added, reason, +cands, runtime
    pruned_reasons: dict = field(default_factory=dict)
    regime: str = "mixed_or_uncertain"

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "raw_candidates": self.raw_candidates,
                "generated_candidates": self.generated_candidates,
                "evaluated_candidates": self.evaluated_candidates, "selected_bundle": self.selected_bundle,
                "top_k": self.top_k, "decision_margin": round(self.decision_margin, 6),
                "decision_margin_abs": round(self.decision_margin_abs, 6),
                "runtime_s": round(self.runtime_s, 4), "anchors_included": self.anchors_included,
                "prev_best_contained": self.prev_best_contained,
                "known_strong_contained": self.known_strong_contained, "beam_width": self.beam_width,
                "widening_rounds": self.widening_rounds, "widening_trace": self.widening_trace,
                "pruned_reasons": self.pruned_reasons, "regime": self.regime}


@dataclass
class BoundedBeamPlanner:
    """Beam over the physics-guided set + progressive widening. Bounded, deterministic, no clock-only fallback."""
    beam_width: int = 8
    max_evaluated: int = 120                 # hard cap on distinct rollouts per decision
    margin_threshold: float = 0.05           # relative margin at/above which the decision is "clear" → stop
    confidence_floor: float = 0.35           # below this, widen (low forecast confidence)
    max_widen_rounds: int = 2
    generator: object = None                 # PhysicsGuidedCandidateGenerator (else a default one)
    widen: bool = True
    beam: bool = True                        # False → evaluate the generated set directly (containment only)

    def _margin(self, beam: list) -> tuple:
        if len(beam) < 2:
            return 1.0, (beam[0][1] if beam else 0.0)
        best, second = beam[0][1], beam[1][1]
        rel = (best - second) / max(abs(best), _EPS)
        return rel, best - second

    def _polish(self, beam: list, surfaces: dict, score_fn, cache: dict) -> list:
        """Coordinate-polish the beam winner over the full SAFE capacity×clock range (+ the regime grid's
        precision/batching options), so a known-good point's cheap coupling is never missed when the regime
        prior gated capacity=1.5 / clock=high out of the grid. Bounded (~a dozen evals); re-ranks the beam."""
        if not beam:
            return beam
        polish_surfaces = {**surfaces, **_POLISH_FULL}
        pb, ps, _ = coordinate_descent(polish_surfaces, score_fn, start=beam[0][0], cache=cache)
        merged = beam + [(pb, ps)]
        return _topk(merged, self.beam_width)

    def plan(self, state: PlannerRegimeState, score_fn, *, prev_best=None,
             recent_regret_failed: bool = False) -> tuple:
        """Search the physics-guided candidate set for ``state`` with ``score_fn(bundle) → float``.
        Returns ``(best_bundle, PlanReport)``. ``prev_best`` (the previous decision's winner) is injected
        as a continuity anchor. ``recent_regret_failed`` forces a widening round (the auditor's feedback)."""
        t0 = time.monotonic()
        gen = self.generator or PhysicsGuidedCandidateGenerator()
        if prev_best is not None:
            state.prev_bundle = prev_best
        cache: dict = {}
        optional: tuple = ()
        trace: list = []
        cs = gen.generate(state, optional_surfaces=optional)
        if self.beam:
            beam, cache = beam_search(cs.surfaces_options, score_fn, beam_width=self.beam_width,
                                      anchors=tuple(cs.anchors), cache=cache)
            beam = self._polish(beam, cs.surfaces_options, score_fn, cache)
        else:
            # "physics-guided candidates" — evaluate the generated set directly, no surface beam / polish.
            beam = _topk([(b, _cached_score(b, score_fn, cache)) for b in cs.bundles], self.beam_width)
        rel, ab_margin = self._margin(beam)
        rounds = 0
        raw_total, gen_total = cs.raw_grid_count, cs.generated_count
        pruned = dict(cs.pruned_reasons)

        # ---- progressive widening: expand only when the decision is close / uncertain -----------------
        while self.widen and self.beam and rounds < self.max_widen_rounds and len(cache) < self.max_evaluated:
            triggers = []
            if rel < self.margin_threshold:
                triggers.append(f"margin {rel:.4f} < {self.margin_threshold}")
            if state.confidence < self.confidence_floor:
                triggers.append(f"confidence {state.confidence:.2f} < {self.confidence_floor}")
            if state.sla_tight:
                triggers.append("sla_tight")
            if recent_regret_failed:
                triggers.append("recent_regret_failed")
            if not triggers:
                trace.append({"round": rounds + 1, "action": "stopped_early",
                              "reason": f"margin {rel:.4f} ≥ {self.margin_threshold} (clear winner)",
                              "candidates_added": 0, "runtime_s": 0.0})
                break
            rt0 = time.monotonic()
            optional = tuple(OPTIONAL_SURFACES[: rounds + 1])     # add one more optional surface per round
            before = len(cache)
            cs = gen.generate(state, optional_surfaces=optional)
            beam, cache = beam_search(cs.surfaces_options, score_fn, beam_width=self.beam_width,
                                      anchors=tuple(cs.anchors), cache=cache)
            beam = self._polish(beam, cs.surfaces_options, score_fn, cache)
            rel, ab_margin = self._margin(beam)
            raw_total, gen_total = cs.raw_grid_count, cs.generated_count
            pruned = dict(cs.pruned_reasons)
            rounds += 1
            trace.append({"round": rounds, "action": "widened", "added_surfaces": list(optional),
                          "reason": "; ".join(triggers), "candidates_added": len(cache) - before,
                          "runtime_s": round(time.monotonic() - rt0, 4),
                          "margin_after": round(rel, 6)})
            recent_regret_failed = False                          # consumed

        best = beam[0][0] if beam else ActionBundle()
        top_k = [{"surfaces": b.non_default_surfaces(), "score": round(s, 4)} for b, s in beam[:5]]
        known = any(_key(b) in cache for b, _ in KNOWN_STRONG_BUNDLES)
        prev_in = prev_best is not None and _key(prev_best) in cache
        strategy = ("physics_guided_candidates" if not self.beam else
                    ("physics_guided_beam_widening" if rounds > 0 else "physics_guided_beam"))
        report = PlanReport(
            strategy=strategy,
            raw_candidates=raw_total, generated_candidates=gen_total, evaluated_candidates=len(cache),
            selected_bundle=best.non_default_surfaces(), top_k=top_k, decision_margin=rel,
            decision_margin_abs=ab_margin, runtime_s=time.monotonic() - t0,
            anchors_included=cs.anchor_flags, prev_best_contained=prev_in, known_strong_contained=known,
            beam_width=self.beam_width, widening_rounds=rounds, widening_trace=trace,
            pruned_reasons=pruned, regime=state.regime_label)
        return best, report


__all__ = ["BoundedBeamPlanner", "PlanReport", "beam_search", "coordinate_descent"]
