"""Offline search-regret auditor — measure what each search strategy leaves on the table.

On a small, controlled surface space where an **exhaustive** enumeration is tractable (the default-4 space,
``precision × batching × capacity × clock`` = 81 bundles), this compares the bounded strategies against the
exhaustive ground truth and reports, per strategy, the **search regret** (best-exhaustive − best-found, in
absolute reward and %), the **missed bundle** (the exhaustive argmax), and — for the physics-guided planner —
whether that missed bundle was **generated / pruned / evaluated**.

Strategies compared: ``physics_guided`` (the new bounded beam), ``clock_only`` (the PR #121 artifact, 3
bundles), ``fixed_24_grid`` (the diagnostic grid), ``exhaustive`` (the 81-bundle ground truth), and
``adaptive`` (the existing beam/CE planner over the same 81-surface space).

**Hard rule:** if the bounded physics planner *loses to a CONTAINED old-best bundle* — a known-good bundle
that was in its candidate set and scores strictly higher than the planner's pick — the audit **FAILS**
(``lost_to_contained_old_best=True``). A genuinely *pruned* bundle (e.g. int4 outside the memory-bound
regime, excluded with a recorded reason) is **not** a failure — it is reported as a prune, not a search bug.

Pure measurement: no simulator/reward/gate change. Deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .actions import ActionBundle
from .physics_guided_candidates import (
    KNOWN_STRONG_BUNDLES,
    PhysicsGuidedCandidateGenerator,
    PlannerRegimeState,
)
from .physics_guided_planner import BoundedBeamPlanner

_EPS = 1e-9


def _key(b) -> tuple:
    return tuple(sorted(b.to_dict().items()))


def clock_only_candidates() -> list:
    """The PR #121 clock-only set (the artifact this PR deletes)."""
    return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]


def grid24_candidates() -> list:
    """The PR #121 diagnostic 24-grid: clock × {bf16,fp8} × {1.0,1.5} × {conservative,aggressive}."""
    out = []
    for cl, pr, ca, ba in product(("base", "low", "high"), ("bf16", "fp8"), (1.0, 1.5),
                                  ("conservative", "aggressive")):
        out.append(ActionBundle(clock_policy=cl, precision_policy=pr, capacity_multiplier=ca,
                                batching_policy=ba))
    return out


def exhaustive_default4_candidates(*, allow_quality_risk: bool = False) -> list:
    """Exhaustive ground truth over the default-4 surfaces. Default = the 54-bundle SAFE space (bf16/fp8);
    ``allow_quality_risk=True`` adds int4 → the 81-bundle diagnostic ceiling (int4 is quality-risked)."""
    precisions = ("bf16", "fp8", "int4") if allow_quality_risk else ("bf16", "fp8")
    out = []
    for pr, ba, ca, cl in product(precisions, ("conservative", "balanced", "aggressive"),
                                  (1.0, 0.75, 1.5), ("base", "low", "high")):
        out.append(ActionBundle(precision_policy=pr, batching_policy=ba, capacity_multiplier=ca,
                                clock_policy=cl))
    return out


def _best_over(bundles, score_fn, cache: dict) -> tuple:
    """Argmax over an explicit candidate list. Returns ``(best, best_score, n_distinct_evaluated)``."""
    best, best_s = None, -1e18
    keys = set()
    for b in bundles:
        kb = _key(b)
        keys.add(kb)
        if kb not in cache:
            cache[kb] = float(score_fn(b))
        s = cache[kb]
        if s > best_s or best is None:
            best, best_s = b, s
    return best, best_s, len(keys)


@dataclass
class AuditReport:
    strategies: dict = field(default_factory=dict)      # name -> {best_bundle, best_reward, evaluated, regret_abs, regret_pct}
    exhaustive_best_reward: float = 0.0
    missed_bundle: dict = field(default_factory=dict)    # the exhaustive argmax (non-default surfaces)
    physics_missed_generated: bool = False               # was the exhaustive winner in the physics set?
    physics_missed_pruned: bool = False                  # …or excluded by a regime prune (honest)?
    physics_missed_evaluated: bool = False               # …or actually scored by the beam?
    lost_to_contained_old_best: bool = False             # the HARD-FAIL flag
    failure_detail: str = ""
    regime: str = "mixed_or_uncertain"

    @property
    def passed(self) -> bool:
        return not self.lost_to_contained_old_best

    def to_dict(self) -> dict:
        return {"passed": self.passed, "regime": self.regime,
                "exhaustive_best_reward": round(self.exhaustive_best_reward, 4),
                "missed_bundle": self.missed_bundle,
                "physics_missed": {"generated": self.physics_missed_generated,
                                   "pruned": self.physics_missed_pruned,
                                   "evaluated": self.physics_missed_evaluated},
                "lost_to_contained_old_best": self.lost_to_contained_old_best,
                "failure_detail": self.failure_detail, "strategies": self.strategies}


def audit_search_regret(score_fn, state: PlannerRegimeState, *, generator=None, planner=None,
                        old_best_bundles: tuple = (), allow_quality_risk: bool = False) -> AuditReport:
    """Compare the bounded strategies against the exhaustive default-4 ground truth for ``state``.

    ``score_fn(bundle) → float`` is the reward (the controller's ``_score`` offline, or a fixture). The
    physics planner runs with widening OFF so the comparison is apples-to-apples on the default-4 space.
    ``old_best_bundles`` are the known-good bundles the planner must never lose to when they are contained
    (defaults to the known-strong family + any provided previous best). ``allow_quality_risk`` includes int4
    in BOTH the generator and the exhaustive ground truth (the diagnostic ceiling); default off = safe space."""
    gen = generator or PhysicsGuidedCandidateGenerator(allow_quality_risk=allow_quality_risk)
    pl = planner or BoundedBeamPlanner(widen=False, generator=gen)
    cache: dict = {}

    # exhaustive ground truth over the default-4 space (safe by default; +int4 only when allowed)
    exh = exhaustive_default4_candidates(allow_quality_risk=allow_quality_risk)
    exh_best, exh_s, exh_n = _best_over(exh, score_fn, cache)

    # physics-guided: run the beam, recording exactly which bundles it evaluates
    cs = gen.generate(state)
    physics_eval: set = set()

    def _rec(b):
        physics_eval.add(_key(b))
        kb = _key(b)
        if kb not in cache:
            cache[kb] = float(score_fn(b))
        return cache[kb]

    pg_best, pg_report = pl.plan(state, _rec)
    pg_s = cache[_key(pg_best)]

    strategies: dict = {}

    def _record(name, best, best_s, evaluated):
        regret = max(0.0, exh_s - best_s)
        strategies[name] = {"best_bundle": best.non_default_surfaces() if best else {},
                            "best_reward": round(best_s, 4), "evaluated": evaluated,
                            "regret_abs": round(regret, 4),
                            "regret_pct": round(100.0 * regret / abs(exh_s), 4) if abs(exh_s) > _EPS else None}

    _record("physics_guided", pg_best, pg_s, len(physics_eval))
    co_b, co_s, co_n = _best_over(clock_only_candidates(), score_fn, cache)
    _record("clock_only", co_b, co_s, co_n)
    g24_b, g24_s, g24_n = _best_over(grid24_candidates(), score_fn, cache)
    _record("fixed_24_grid", g24_b, g24_s, g24_n)
    _record("exhaustive", exh_best, exh_s, exh_n)
    # adaptive (existing beam/CE planner) over the same 81-surface space
    try:
        from .search_planner import AdaptiveSearchPlanner
        surfaces = {"precision_policy": ("bf16", "fp8", "int4"),
                    "batching_policy": ("conservative", "balanced", "aggressive"),
                    "capacity_multiplier": (1.0, 0.75, 1.5), "clock_policy": ("base", "low", "high")}
        ad_best, ad_plan = AdaptiveSearchPlanner(exhaustive_max=0).plan(
            score_fn, surfaces=surfaces, regret_audit=False)
        _record("adaptive", ad_best, float(score_fn(ad_best)), ad_plan.candidates_evaluated)
    except Exception as e:                              # adaptive is a comparator, never the gate
        strategies["adaptive"] = {"error": repr(e)}

    # missed-bundle provenance (for the physics planner)
    missed_key = _key(exh_best)
    generated = cs.contains(exh_best)
    evaluated = missed_key in physics_eval
    pruned = (not generated) and _excluded_by_prune(exh_best, cs)

    # HARD RULE: lose to a CONTAINED old-best?  (a known-good bundle that was evaluated and scores higher)
    olds = list(old_best_bundles) + [b for b, _ in KNOWN_STRONG_BUNDLES]
    if state.prev_bundle is not None and hasattr(state.prev_bundle, "to_dict"):
        olds.append(state.prev_bundle)
    lost, detail = False, ""
    for ob in olds:
        kb = _key(ob)
        if kb in physics_eval and cache.get(kb, -1e18) > pg_s + _EPS:
            lost = True
            detail = (f"physics planner chose reward {pg_s:.4f} but CONTAINED old-best "
                      f"{ob.non_default_surfaces()} scores {cache[kb]:.4f}")
            break

    return AuditReport(strategies=strategies, exhaustive_best_reward=exh_s,
                       missed_bundle=exh_best.non_default_surfaces(),
                       physics_missed_generated=generated, physics_missed_pruned=pruned,
                       physics_missed_evaluated=evaluated, lost_to_contained_old_best=lost,
                       failure_detail=detail, regime=state.regime_label)


def _excluded_by_prune(bundle, candidate_set) -> bool:
    """True if ``bundle`` uses an option the regime prior excluded (an honest prune, not a search miss).
    Checks each default surface's generated option set; a value outside it was pruned with a recorded reason."""
    surfaces = candidate_set.surfaces_options
    for s, opts in surfaces.items():
        if getattr(bundle, s, None) not in opts and getattr(bundle, s, None) is not None:
            return True
    return False


__all__ = ["audit_search_regret", "AuditReport", "clock_only_candidates", "grid24_candidates",
           "exhaustive_default4_candidates"]
