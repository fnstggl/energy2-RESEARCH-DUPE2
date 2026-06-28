"""AdaptiveMPCSearchV2 — adaptive MPC candidate search with regret audit (V2).

Replaces a fixed hard candidate cap (e.g. 256) with an adaptive planner over a discrete action space:

  * ``exhaustive_cartesian`` — full product, used when the raw candidate count is tractable.
  * ``beam_search``          — keep the top-K partial bundles while adding action surfaces (finds COUPLED
    optima that one-knob-at-a-time search misses).
  * ``coordinate_descent``   — cheap one-knob-at-a-time fallback (can get stuck on coupled wins).
  * ``auto``                 — exhaustive if raw ≤ threshold, else beam.
  * ``search_regret_audit``  — compare an approximate strategy against exhaustive on the same space and
    report the regret = (best_exhaustive − selected) / best_exhaustive.

No silent cap: the raw candidate count, evaluated count, strategy, runtime, selected bundle, best-exhaustive
reward (when computed), and search regret are all reported. The optimiser must not miss obvious coupled
action wins because of primitive search — if regret is high, the result warns and recommends a stronger
search. Deterministic (fixed dim/value ordering; no RNG).
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

STRATEGIES = ("exhaustive_cartesian", "beam_search", "coordinate_descent", "auto")


@dataclass
class SearchResult:
    strategy: str
    raw_candidate_count: int
    evaluated_candidate_count: int
    selected: dict
    selected_reward: float
    runtime_s: float
    best_exhaustive_reward: float | None = None
    search_regret: float | None = None
    warning: str = ""
    diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "raw_candidate_count": self.raw_candidate_count,
                "evaluated_candidate_count": self.evaluated_candidate_count, "selected": self.selected,
                "selected_reward": round(self.selected_reward, 4),
                "best_exhaustive_reward": (round(self.best_exhaustive_reward, 4)
                                           if self.best_exhaustive_reward is not None else None),
                "search_regret": (round(self.search_regret, 6) if self.search_regret is not None else None),
                "runtime_s": round(self.runtime_s, 5), "warning": self.warning,
                **({"diagnostics": self.diagnostics} if self.diagnostics else {})}


@dataclass
class AdaptiveMPCSearchV2:
    action_space: dict                    # {dim: [values...]}, fixed order
    exhaustive_threshold: int = 4096
    beam_k: int = 6
    regret_warn: float = 0.05             # warn if approximate search regret exceeds this

    def _raw_count(self) -> int:
        n = 1
        for vs in self.action_space.values():
            n *= max(1, len(vs))
        return n

    def _default(self) -> dict:
        return {d: vs[0] for d, vs in self.action_space.items()}

    def _exhaustive(self, eval_fn):
        dims = list(self.action_space)
        best, best_r, evaluated = None, float("-inf"), 0
        for combo in itertools.product(*[self.action_space[d] for d in dims]):
            cand = dict(zip(dims, combo))
            r = eval_fn(cand)
            evaluated += 1
            if r > best_r:
                best, best_r = cand, r
        return best, best_r, evaluated

    def _beam(self, eval_fn):
        dims = list(self.action_space)
        base = self._default()
        beam = [dict(base)]                          # partial bundles (default-filled)
        evaluated = 0
        cache = {}

        def score(cand):
            nonlocal evaluated
            key = tuple(cand[d] for d in dims)
            if key not in cache:
                cache[key] = eval_fn(cand)
                evaluated += 1
            return cache[key]

        for d in dims:
            expanded = []
            for partial in beam:
                for v in self.action_space[d]:
                    c = dict(partial)
                    c[d] = v
                    expanded.append((score(c), c))
            expanded.sort(key=lambda x: x[0], reverse=True)
            # dedupe by full key, keep top-K
            seen, kept = set(), []
            for r, c in expanded:
                k = tuple(c[x] for x in dims)
                if k in seen:
                    continue
                seen.add(k)
                kept.append(c)
                if len(kept) >= self.beam_k:
                    break
            beam = kept
        best = max(beam, key=score)
        return best, score(best), evaluated

    def _coordinate(self, eval_fn, passes: int = 1):
        dims = list(self.action_space)
        cur = self._default()
        evaluated = 0
        cache = {}

        def score(cand):
            nonlocal evaluated
            key = tuple(cand[d] for d in dims)
            if key not in cache:
                cache[key] = eval_fn(cand)
                evaluated += 1
            return cache[key]

        cur_r = score(cur)
        for _ in range(passes):
            for d in dims:
                best_v, best_r = cur[d], cur_r
                for v in self.action_space[d]:
                    c = dict(cur)
                    c[d] = v
                    r = score(c)
                    if r > best_r:
                        best_v, best_r = v, r
                cur[d], cur_r = best_v, best_r
        return cur, cur_r, evaluated

    def search(self, eval_fn, *, strategy: str = "auto", audit: bool = False) -> SearchResult:
        raw = self._raw_count()
        t0 = time.perf_counter()
        chosen = strategy
        if strategy == "auto":
            chosen = "exhaustive_cartesian" if raw <= self.exhaustive_threshold else "beam_search"
        if chosen == "exhaustive_cartesian" and raw > self.exhaustive_threshold:
            chosen = "beam_search"  # never silently exhaust a huge space
        if chosen == "exhaustive_cartesian":
            sel, sel_r, ev = self._exhaustive(eval_fn)
        elif chosen == "coordinate_descent":
            sel, sel_r, ev = self._coordinate(eval_fn)
        else:
            chosen = "beam_search"
            sel, sel_r, ev = self._beam(eval_fn)
        runtime = time.perf_counter() - t0

        best_ex, regret, warn = None, None, ""
        if audit and chosen != "exhaustive_cartesian" and raw <= self.exhaustive_threshold:
            _, best_ex, _ = self._exhaustive(eval_fn)
            regret = (best_ex - sel_r) / best_ex if best_ex not in (0, None) else 0.0
            if regret > self.regret_warn:
                warn = (f"search regret {regret:.3f} > {self.regret_warn}: approximate '{chosen}' missed a "
                        f"coupled optimum — recommend exhaustive_cartesian or larger beam_k")
        elif chosen == "exhaustive_cartesian":
            best_ex, regret = sel_r, 0.0
        return SearchResult(chosen, raw, ev, sel, sel_r, runtime, best_ex, regret, warn)


__all__ = ["AdaptiveMPCSearchV2", "SearchResult", "STRATEGIES"]
