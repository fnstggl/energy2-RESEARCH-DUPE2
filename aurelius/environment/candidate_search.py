"""CandidateBundleGenerator — search the CONNECTED action space for the MPC planner.

The planner optimizes whole :class:`ActionBundle`s, not isolated knobs. This generator
enumerates candidates over the connected (and optionally SIMULATED_ONLY) surfaces, choosing
the search method by the size of the space:

- **exhaustive** when the connected space is small (the default today: 36 bundles);
- **latin_hypercube** deterministic sampling when it is large;
- **coordinate** local search around an incumbent (single-dimension moves), used for the
  ablation that reports each surface's contribution.

Hard rules (the honesty contract): only CONNECTED surfaces vary by default (SIMULATED_ONLY
opt-in); PLANNED surfaces are never generated; **no connected surface is silently excluded —
a surface only stops varying if it is explicitly frozen with a recorded reason** (``frozen`` /
``frozen_reasons``). The planner reports the total connected dimensions, the theoretical
combination count, how many candidates it evaluated, the method used, the best bundle, and a
per-dimension ablation — so the search is auditable, never a hand-picked preset list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .action_registry import optimizable_surfaces
from .actions import ACTION_SPECS, ActionBundle

EXHAUSTIVE_BUDGET = 256          # enumerate fully at or below this many combinations


@dataclass
class CandidateBundleGenerator:
    include_simulated: bool = False
    frozen: dict = field(default_factory=dict)          # surface -> pinned value
    frozen_reasons: dict = field(default_factory=dict)  # surface -> why it is frozen

    def surfaces(self) -> list:
        """Connected (+simulated if opted-in) surfaces the planner may move, minus frozen."""
        return [s for s in optimizable_surfaces(include_simulated=self.include_simulated)
                if s not in self.frozen]

    def theoretical_combinations(self) -> int:
        c = 1
        for s in self.surfaces():
            c *= len(ACTION_SPECS[s].options)
        return c

    def _base(self) -> dict:
        return dict(self.frozen)                         # frozen surfaces pinned to their value

    def exhaustive(self) -> list:
        surfaces = self.surfaces()
        opts = [ACTION_SPECS[s].options for s in surfaces]
        return [ActionBundle(**{**self._base(), **dict(zip(surfaces, combo))})
                for combo in product(*opts)]

    def latin_hypercube(self, n: int) -> list:
        """Deterministic space-filling sample (no RNG): stride each dimension's options with
        a per-dimension coprime offset so combinations spread across the space."""
        surfaces = self.surfaces()
        out = []
        for k in range(n):
            choice = {s: ACTION_SPECS[s].options[(k + 7 * j) % len(ACTION_SPECS[s].options)]
                      for j, s in enumerate(surfaces)}
            out.append(ActionBundle(**{**self._base(), **choice}))
        return out

    def coordinate_neighbors(self, center: ActionBundle) -> list:
        """``center`` plus every single-dimension move (for local search / ablation)."""
        out = [center]
        for s in self.surfaces():
            for o in ACTION_SPECS[s].options:
                if o != getattr(center, s):
                    out.append(center.with_overrides(**{s: o}))
        return out

    def generate(self, *, method: str = "auto", budget: int = EXHAUSTIVE_BUDGET) -> tuple:
        """Return ``(bundles, method_used)``. ``auto`` → exhaustive if the space ≤ budget,
        else a latin-hypercube sample of ``budget`` bundles."""
        comb = self.theoretical_combinations()
        if method == "exhaustive" or (method == "auto" and comb <= budget):
            return self.exhaustive(), "exhaustive"
        if method == "coordinate":
            base = ActionBundle(**self._base())
            return self.coordinate_neighbors(base), "coordinate"
        return self.latin_hypercube(budget), "latin_hypercube"

    def search(self, score_fn, *, budget: int = EXHAUSTIVE_BUDGET, passes: int = 3,
               start=None) -> tuple:
        """Find the best-scoring bundle. Returns ``(best, candidates_evaluated, method)``.
        Exhaustive when the space ≤ ``budget``; otherwise **coordinate descent** — from a start
        bundle (default the no-op), repeatedly move each free surface to its best option until no
        improvement. Covers every connected dimension at a cost ≈ surfaces·options·passes, far
        cheaper than enumerating a large space, and no connected knob is skipped."""
        comb = self.theoretical_combinations()
        if comb <= budget:
            bundles = self.exhaustive()
            return max(bundles, key=score_fn), len(bundles), "exhaustive"
        cur = start if start is not None else ActionBundle(**self._base())
        cur_s = score_fn(cur)
        evaluated = 1
        for _ in range(max(1, passes)):
            improved = False
            for s in self.surfaces():
                for o in ACTION_SPECS[s].options:
                    if o == getattr(cur, s):
                        continue
                    cand = cur.with_overrides(**{s: o})
                    sc = score_fn(cand)
                    evaluated += 1
                    if sc > cur_s:
                        cur, cur_s, improved = cand, sc, True
            if not improved:
                break
        return cur, evaluated, "coordinate_descent"


@dataclass
class SearchReport:
    method: str
    connected_dimensions: int
    theoretical_combinations: int
    candidates_evaluated: int
    frozen: dict
    best: dict                          # best bundle's non-default surfaces
    best_score: float
    ablation: list                      # per-surface contribution at the incumbent
    pareto_safe_vs: dict                # {baseline_name: bool} SLA-not-worse vs baselines
    top_bundles: list = field(default_factory=list)   # top-10 scored bundles AMONG THOSE EVALUATED

    def to_dict(self) -> dict:
        return {"method": self.method, "connected_dimensions": self.connected_dimensions,
                "theoretical_combinations": self.theoretical_combinations,
                "candidates_evaluated": self.candidates_evaluated, "frozen": self.frozen,
                "best": self.best, "best_score": round(self.best_score, 4),
                "ablation": self.ablation, "top_bundles": self.top_bundles,
                "pareto_safe_vs": self.pareto_safe_vs}


def plan_bundle(gen: CandidateBundleGenerator, score_fn, *,
                budget: int = EXHAUSTIVE_BUDGET) -> tuple:
    """Search with ``gen`` using ``score_fn(bundle) -> (score, sla_violation_rate)``; return
    ``(best_bundle, SearchReport)``. Uses ``gen.search`` (exhaustive when small, coordinate
    descent when large — the SAME path the controller takes), then a per-surface ablation: from
    the best bundle, the score range achievable by moving ONLY that surface — i.e. how much that
    dimension contributed. No connected surface is omitted. We also surface the top-10 bundles —
    honestly labelled the best AMONG THOSE EVALUATED (the full enumeration when exhaustive; the
    coordinate-descent trajectory otherwise), not the global top-10 of the theoretical space."""
    seen: dict = {}                     # bundle key → (score, non_default_surfaces)

    def _scored(b):
        sc = score_fn(b)[0]
        seen[tuple(sorted(b.to_dict().items()))] = (sc, b.non_default_surfaces())
        return sc

    best, evaluated, method = gen.search(_scored, budget=budget)
    best_score = score_fn(best)[0]
    ablation = []
    for s in gen.surfaces():
        moves = [(o, score_fn(best.with_overrides(**{s: o}))[0]) for o in ACTION_SPECS[s].options]
        scores = [sc for _o, sc in moves]
        ablation.append({"surface": s, "incumbent": getattr(best, s),
                         "score_range": round(max(scores) - min(scores), 4),
                         "best_value": max(moves, key=lambda m: m[1])[0]})
    ablation.sort(key=lambda a: -a["score_range"])
    top = sorted(seen.values(), key=lambda v: -v[0])[:10]
    top_bundles = [{"surfaces": surf, "score": round(sc, 4)} for sc, surf in top]
    report = SearchReport(
        method=method, connected_dimensions=len(gen.surfaces()),
        theoretical_combinations=gen.theoretical_combinations(),
        candidates_evaluated=evaluated, frozen=dict(gen.frozen_reasons),
        best=best.non_default_surfaces(), best_score=best_score, ablation=ablation,
        pareto_safe_vs={}, top_bundles=top_bundles)
    return best, report


__all__ = ["CandidateBundleGenerator", "SearchReport", "plan_bundle", "EXHAUSTIVE_BUDGET"]
