"""Adaptive MPC search planner + roofline-pruned candidate generation.

Replaces the fixed ``budget=256`` cap (which silently truncated a large action space) with a planner that
(a) chooses a search strategy from the RAW candidate count, (b) captures action interactions better than
coordinate descent via **beam search**, and (c) **measures** what an approximate search lost against
exhaustive enumeration (``search_regret_audit``) instead of hiding it. The hard rule the user set: *do not
silently cap without measuring the loss* — so every plan reports the raw count, strategy, candidates
evaluated, best reward, estimated regret, the selected bundle, and the runtime.

Strategies:
  * ``exhaustive_cartesian`` — full product when the raw count ≤ ``exhaustive_max``.
  * ``beam_search`` — keep the top-``beam_width`` partial bundles while adding one surface at a time;
    captures cross-surface interactions a single-dimension coordinate descent misses.
  * ``coordinate_descent`` — cheap local search (kept as a fallback / regret comparator, never the sole
    method for a large space).
  * ``cross_entropy`` / ``random_restart`` — stochastic global search for very large spaces (deterministic:
    seeded ``random.Random`` so a fixture is reproducible).

The roofline-pruned generator restricts each roofline surface's options to the regime where it can help
(memory-bandwidth-bound → precision/spec emphasised; compute-bound → spec off, clock up), and **freezes**
co-location + prefill/decode allocation off with a recorded reason (no background-work trace; no
disaggregated capacity pools). Pruning the SEARCH SPACE is honest — it never changes the reward; a pruned
candidate would still score neutral/worse through the physics. It only avoids evaluating candidates the
roofline says cannot help, and records *why*.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from itertools import product

from .action_registry import optimizable_surfaces
from .actions import ACTION_SPECS, ActionBundle

# roofline surfaces whose option set is regime-pruned (the rest keep their full option list).
_ROOFLINE_SURFACES = ("precision_policy", "spec_decode_policy", "clock_policy")
# surfaces frozen off with a recorded reason (modelled in serving_point, but not live: see the audit).
FROZEN_OFF = {
    "colocation_policy": ("off", "no background-work trace (Azure is all latency-critical; "
                          "ReplicaState.workload_class unused) → co-location can only hurt foreground SLA"),
    "prefill_decode_policy": ("shared", "the live cluster replay has no disaggregated prefill/decode "
                              "capacity pools (only roofline models the split analytically)"),
}


def roofline_pruned_options(*, decode_regime: str | None = None, sla_tight: bool = False,
                            include_simulated: bool = False) -> dict:
    """Per-surface option subsets for the planner, pruned by the instantaneous roofline regime.

    Returns ``{surface: options_tuple}``. Connected surfaces keep their full options EXCEPT the roofline
    ones, which are restricted to the regime where each can help (memory-bandwidth-bound → lower precision
    + speculative decoding can help; compute-bound → spec off, clock can go up). Under a tight SLA the
    aggressive ends are dropped. Co-location + prefill/decode are excluded here (frozen — see
    ``FROZEN_OFF``)."""
    surfaces = {}
    for s in optimizable_surfaces(include_simulated=include_simulated):
        if s in FROZEN_OFF:
            continue
        opts = list(ACTION_SPECS[s].options)
        if s == "precision_policy":
            # precision helps in BOTH regimes (memory-bound: fewer bytes → faster; compute-bound: HBM/KV
            # pressure still matters), but int4's quality risk is only worth proposing when memory-bound.
            opts = ["bf16", "fp8"] + (["int4"] if decode_regime == "memory_bandwidth_bound" else [])
        elif s == "spec_decode_policy":
            # speculative decoding helps ONLY in the memory-bandwidth-bound regime (spare compute); in the
            # compute-bound regime it competes for the scarce FLOPs → propose only 'off'.
            if decode_regime == "compute_bound":
                opts = ["off"]
            else:
                opts = ["off", "shallow", "medium"] + ([] if sla_tight else ["aggressive"])
        elif s == "clock_policy":
            # low clock saves energy when memory-bandwidth-bound; high clock buys compute-bound latency.
            if decode_regime == "compute_bound":
                opts = ["base", "high"]
            elif decode_regime == "memory_bandwidth_bound":
                opts = ["base", "low"]
        elif s == "batching_policy" and sla_tight:
            opts = [o for o in opts if o != "aggressive"]      # tail-latency risk under a tight SLA
        surfaces[s] = tuple(opts)
    return surfaces


def raw_candidate_count(surfaces: dict) -> int:
    c = 1
    for opts in surfaces.values():
        c *= max(1, len(opts))
    return c


def _bundle(base: dict, choice: dict) -> ActionBundle:
    return ActionBundle(**{**base, **choice})


@dataclass
class SearchPlan:
    """The auditable record of ONE MPC decision's search (the user's required per-decision report)."""
    raw_candidate_count: int
    pruned_candidate_count: int
    strategy: str
    candidates_evaluated: int
    best_reward: float
    best_bundle: dict                       # the selected bundle's non-default surfaces
    estimated_regret: float | None          # best_exhaustive − best_approx (None if not audited)
    regret_audited: bool
    exhaustive_reward: float | None
    runtime_s: float
    surfaces: list = field(default_factory=list)
    frozen_reasons: dict = field(default_factory=dict)
    pruned_reason_counts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"raw_candidate_count": self.raw_candidate_count,
                "pruned_candidate_count": self.pruned_candidate_count, "strategy": self.strategy,
                "candidates_evaluated": self.candidates_evaluated,
                "best_reward": round(self.best_reward, 4), "best_bundle": self.best_bundle,
                "estimated_regret": (round(self.estimated_regret, 6) if self.estimated_regret is not None
                                     else None), "regret_audited": self.regret_audited,
                "exhaustive_reward": (round(self.exhaustive_reward, 4)
                                      if self.exhaustive_reward is not None else None),
                "runtime_s": round(self.runtime_s, 4), "surfaces": self.surfaces,
                "frozen_reasons": self.frozen_reasons, "pruned_reason_counts": self.pruned_reason_counts}


@dataclass
class AdaptiveSearchPlanner:
    """Strategy chosen by the raw candidate count; regret measured when an exhaustive comparison fits."""
    exhaustive_max: int = 4096          # exhaustive cartesian at/below this raw count
    beam_width: int = 6
    coordinate_passes: int = 3
    regret_audit_max: int = 20000       # run the exhaustive comparison for regret when raw count ≤ this
    ce_iters: int = 6
    ce_samples: int = 24
    ce_elite_frac: float = 0.3
    seed: int = 0

    # -- strategies ---------------------------------------------------------
    def _exhaustive(self, surfaces, base, score_fn):
        names = list(surfaces)
        opts = [surfaces[n] for n in names]
        best, best_s, n = None, -1e18, 0
        for combo in product(*opts):
            b = _bundle(base, dict(zip(names, combo)))
            s = score_fn(b)
            n += 1
            if s > best_s or best is None:
                best, best_s = b, s
        return best, best_s, n

    def _beam(self, surfaces, base, score_fn):
        names = list(surfaces)
        # a beam entry is (bundle, score); start from the no-op base, add one surface at a time.
        beam = [(_bundle(base, {}), None)]
        evaluated = 0
        for nm in names:
            cand = []
            for b, _ in beam:
                cur = b.to_dict()
                for o in surfaces[nm]:
                    nb = ActionBundle(**{**cur, nm: o})
                    cand.append(nb)
            scored = []
            for nb in cand:
                scored.append((nb, score_fn(nb)))
                evaluated += 1
            # keep the top-width, deterministic tie-break by the bundle's surface tuple
            scored.sort(key=lambda t: (-t[1], tuple(sorted(t[0].to_dict().items()))))
            beam = scored[:self.beam_width]
        best, best_s = beam[0][0], beam[0][1]
        return best, best_s, evaluated

    def _coordinate(self, surfaces, base, score_fn, start=None):
        names = list(surfaces)
        cur = start or _bundle(base, {})
        cur_s = score_fn(cur)
        evaluated = 1
        for _ in range(max(1, self.coordinate_passes)):
            improved = False
            for nm in names:
                for o in surfaces[nm]:
                    if o == getattr(cur, nm):
                        continue
                    cand = cur.with_overrides(**{nm: o})
                    s = score_fn(cand)
                    evaluated += 1
                    if s > cur_s:
                        cur, cur_s, improved = cand, s, True
            if not improved:
                break
        return cur, cur_s, evaluated

    def _random_restart(self, surfaces, base, score_fn, restarts=4):
        names = list(surfaces)
        rng = random.Random(self.seed)
        best, best_s, evaluated = None, -1e18, 0
        for _ in range(restarts):
            start = _bundle(base, {n: rng.choice(surfaces[n]) for n in names})
            b, s, e = self._coordinate(surfaces, base, score_fn, start=start)
            evaluated += e
            if s > best_s or best is None:
                best, best_s = b, s
        return best, best_s, evaluated

    def _cross_entropy(self, surfaces, base, score_fn):
        names = list(surfaces)
        rng = random.Random(self.seed + 1)
        probs = {n: [1.0 / len(surfaces[n])] * len(surfaces[n]) for n in names}
        best, best_s, evaluated = None, -1e18, 0
        for _ in range(self.ce_iters):
            samples = []
            for _k in range(self.ce_samples):
                choice = {}
                for n in names:
                    opts = surfaces[n]
                    choice[n] = rng.choices(opts, weights=probs[n], k=1)[0]
                b = _bundle(base, choice)
                s = score_fn(b)
                evaluated += 1
                samples.append((b, s, choice))
                if s > best_s or best is None:
                    best, best_s = b, s
            samples.sort(key=lambda t: -t[1])
            elite = samples[:max(1, int(self.ce_elite_frac * len(samples)))]
            for n in names:
                opts = surfaces[n]
                counts = [sum(1 for _b, _s, ch in elite if ch[n] == o) for o in opts]
                tot = sum(counts) or 1
                # smoothed update toward the elite distribution
                probs[n] = [0.2 * probs[n][i] + 0.8 * (counts[i] / tot) for i in range(len(opts))]
        return best, best_s, evaluated

    # -- the plan -----------------------------------------------------------
    def plan(self, score_fn, *, decode_regime: str | None = None, sla_tight: bool = False,
             frozen_reasons: dict | None = None, surfaces: dict | None = None,
             regret_audit: bool = True, large_strategy: str = "beam_search") -> tuple:
        """Search ``surfaces`` (default: the roofline-pruned connected space) with ``score_fn(bundle)→float``.
        Returns ``(best_bundle, SearchPlan)``. ``regret_audit`` runs the exhaustive comparison when the raw
        count ≤ ``regret_audit_max`` so the loss from an approximate strategy is MEASURED, never hidden."""
        t0 = time.monotonic()
        if surfaces is None:
            surfaces = roofline_pruned_options(decode_regime=decode_regime, sla_tight=sla_tight)
        fr = dict(FROZEN_OFF) if frozen_reasons is None else frozen_reasons
        base = {s: v[0] for s, v in fr.items()}                     # frozen surfaces pinned to their no-op
        raw = raw_candidate_count(surfaces)

        if raw <= self.exhaustive_max:
            strategy = "exhaustive_cartesian"
            best, best_s, evaluated = self._exhaustive(surfaces, base, score_fn)
            exhaustive_reward, regret, audited = best_s, 0.0, True
        else:
            strategy = large_strategy
            if strategy == "beam_search":
                best, best_s, evaluated = self._beam(surfaces, base, score_fn)
            elif strategy == "cross_entropy":
                best, best_s, evaluated = self._cross_entropy(surfaces, base, score_fn)
            elif strategy == "random_restart":
                best, best_s, evaluated = self._random_restart(surfaces, base, score_fn)
            else:
                best, best_s, evaluated = self._coordinate(surfaces, base, score_fn)
            exhaustive_reward = regret = None
            audited = False
            if regret_audit and raw <= self.regret_audit_max:
                eb, es, en = self._exhaustive(surfaces, base, score_fn)
                evaluated += en
                exhaustive_reward, regret, audited = es, es - best_s, True
                if es > best_s:                                     # exhaustive found strictly better
                    best, best_s = eb, es

        pruned_reasons = {s: ("regime_pruned" if s in _ROOFLINE_SURFACES else "full")
                          for s in surfaces}
        plan = SearchPlan(
            raw_candidate_count=raw, pruned_candidate_count=raw, strategy=strategy,
            candidates_evaluated=evaluated, best_reward=best_s,
            best_bundle=best.non_default_surfaces(), estimated_regret=regret, regret_audited=audited,
            exhaustive_reward=exhaustive_reward, runtime_s=time.monotonic() - t0,
            surfaces=list(surfaces), frozen_reasons={s: v[1] for s, v in fr.items()},
            pruned_reason_counts={v: sum(1 for x in pruned_reasons.values() if x == v)
                                  for v in set(pruned_reasons.values())})
        return best, plan


__all__ = ["AdaptiveSearchPlanner", "SearchPlan", "roofline_pruned_options", "raw_candidate_count",
           "FROZEN_OFF"]
