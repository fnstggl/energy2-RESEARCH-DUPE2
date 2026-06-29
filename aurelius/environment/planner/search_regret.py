"""Search regret for the tournament — how much SLA-safe gp/$ each method left on the table.

For a tractable fixture an EXHAUSTIVE enumeration gives the true optimum, so regret is exact
(`TRUE_EXHAUSTIVE`). For a larger window where exhaustive is impossible we use the **best-known** reward
across all methods (their shared, memoized evaluation cache) as the reference and mark it
`NOT_TRUE_EXHAUSTIVE` — honestly labelled, never presented as a true optimum. For every method we report the
regret (absolute + percent), the selected bundle, whether the reference (true/best-known) optimum was
**contained** in the method's reachable space and whether it was actually **evaluated**, and the
candidates-generated / evaluated counts. Pure measurement — no simulator / reward / gate change.
"""

from __future__ import annotations

from itertools import product

from ..actions import ActionBundle
from .candidate_generators import CORE_GRID_SURFACES, named_anchor_keys

_EPS = 1e-9


def _key(b) -> tuple:
    return tuple(sorted(b.to_dict().items()))


def _core_grid_keys() -> set:
    names = list(CORE_GRID_SURFACES)
    return {_key(ActionBundle(**dict(zip(names, c))))
            for c in product(*(CORE_GRID_SURFACES[n] for n in names))}


def compute_window_regret(results: dict, scorer_cache: dict, *, true_optimum: tuple | None = None,
                          prev_best=None) -> dict:
    """Compute per-method search regret for one window.

    `results`: {method -> MethodResult}. `scorer_cache`: shared {bundle_key -> reward} (the union of every
    method's evaluations — its argmax is the best-known reward). `true_optimum`: `(reward, bundle_key)` when
    an exhaustive enumeration was run (→ exact regret), else None (→ best-known reference). Returns a dict
    with the reference, its kind, and the per-method regret table."""
    best_known_key = max(scorer_cache, key=scorer_cache.get) if scorer_cache else None
    best_known = scorer_cache.get(best_known_key, 0.0) if best_known_key else 0.0
    if true_optimum is not None:
        ref_reward, ref_key, kind = true_optimum[0], true_optimum[1], "TRUE_EXHAUSTIVE"
    else:
        ref_reward, ref_key, kind = best_known, best_known_key, "NOT_TRUE_EXHAUSTIVE"

    reachable = _core_grid_keys() | named_anchor_keys(prev_best)   # every method searches ⊇ this
    table = {}
    for m, r in results.items():
        regret_abs = max(0.0, ref_reward - r.best_reward)
        ev = getattr(r, "evaluated_keys", set())
        table[m] = {
            "best_reward": round(r.best_reward, 4),
            "regret_abs": round(regret_abs, 4),
            "regret_pct": round(100.0 * regret_abs / abs(ref_reward), 4) if abs(ref_reward) > _EPS else None,
            "selected_bundle": r.best_bundle.non_default_surfaces() if r.best_bundle else {},
            "true_opt_contained": (ref_key in reachable) if m != "clock_only" else (ref_key in ev),
            "true_opt_evaluated": ref_key in ev,
            "candidates_generated": r.candidates_generated,
            "candidates_evaluated": r.candidates_evaluated,
            "anchors_evaluated": r.anchors_evaluated,
        }
    return {"reference_kind": kind, "reference_reward": round(ref_reward, 4),
            "reference_bundle": (dict(ref_key) if ref_key else {}),
            "best_known_reward": round(best_known, 4), "per_method": table}


def anchor_contract_violations(results: dict, *, prev_best=None,
                               exempt: tuple = ("clock_only",)) -> dict:
    """The HARD rule: every non-exempt method must have evaluated the named anchors (incl. the diagnostic
    winner). Returns {method -> missing} for any violator; empty dict = the contract held. `clock_only` is
    exempt — it is the deliberate degenerate artifact whose missing anchors are the finding, not a bug."""
    named = named_anchor_keys(prev_best)
    bad = {}
    for m, r in results.items():
        if m in exempt:
            continue
        missing = named - getattr(r, "evaluated_keys", set())
        if missing:
            bad[m] = [dict(k) for k in missing]
    return bad


__all__ = ["compute_window_regret", "anchor_contract_violations"]
