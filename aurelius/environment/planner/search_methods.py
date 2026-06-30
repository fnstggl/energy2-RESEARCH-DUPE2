"""Search methods for the MPC tournament — compared at equal EVALUATION budgets, not just runtime.

Every method searches the SAME action space with the SAME `score_fn(bundle) -> reward` (a causal world
rollout, memoized upstream) and the SAME required anchors, under a cap on the number of DISTINCT bundle
evaluations (`budget`). The cap is the portable, hardware-independent currency — `BudgetedScorer` enforces
it and records the distinct evaluations, total calls, and best-so-far. Each method returns a `MethodResult`
with candidates generated / evaluated, node expansions, and the chosen bundle, so the tournament can compute
gp/$ per evaluation and gp/$ per rollout, not only gp/$ per second.

Methods (the listed minimum + literature-justified additions):
  generator methods (ordered candidate lists, anchors first):
    fixed_grid, expanded_grid, physics_guided_grid, random_grid (the physics-prior ablation control)
  adaptive methods (drive their own evaluation under the budget):
    beam_search, progressive_widening, hierarchical_search, coordinate_descent, cross_entropy,
    random_restart, simulated_annealing, hybrid (physics-guided generation → beam → polish → widening),
    exhaustive_small (tiny fixtures only)

Additions justified: discrete combinatorial bundles, no gradient, expensive black-box evaluations, small
budgets — the regime where **cross-entropy method** (CEM) and **simulated annealing** are the standard strong
global searches, and a **hybrid** (good prior + beam coupling + local polish + widening) is the natural
ensemble. MCTS/Bayesian-optimisation were considered but rejected for this budget: their per-node bookkeeping
/ surrogate-fitting overhead is not worth it at ≤1000 evaluations over a ≤6-surface space (documented in
`research/MPC_SEARCH_METHOD_TOURNAMENT.md`). Deterministic given the seed. Reuses `physics_guided_planner`
and `physics_guided_candidates`; mirrors the strategies in `search_planner.py`.
"""

from __future__ import annotations

import math
import random as _random
from dataclasses import dataclass, field

from ..actions import ACTION_SPECS, ActionBundle
from ..physics_guided_candidates import PlannerRegimeState
from ..physics_guided_planner import _topk  # deterministic top-k (dedup + stable tie-break)
from .candidate_generators import (
    CORE_GRID_SURFACES,
    EXPANDED_EXTRA_SURFACES,
    core_grid,
    enforce_anchors,
    expanded_grid,
    physics_guided_candidates,
    random_candidates,
    required_anchors,
)

_EPS = 1e-9
# action-group decomposition for hierarchical search (by control timescale).
SLOW_SURFACES = ("capacity_policy", "capacity_multiplier", "placement_policy", "migration_policy", "prewarm_policy")
# kv_cache_precision couples with weight precision (same roofline byte/bandwidth group); prefill_decode is a
# medium-timescale capacity-shape knob. Both are regime-gated in `_hierarchical` so width stays ~constant.
MEDIUM_SURFACES = ("precision_policy", "kv_cache_precision_policy", "batching_policy",
                   "spec_decode_policy", "clock_policy", "prefill_decode_policy")
FAST_SURFACES = ("routing_policy", "admission_policy", "ordering_policy")


def _key(b) -> tuple:
    return tuple(sorted(b.to_dict().items()))


@dataclass
class MethodResult:
    method: str
    best_bundle: object = None
    best_reward: float = -1e18
    candidates_generated: int = 0       # bundles the method PRODUCED (before the budget cap)
    candidates_evaluated: int = 0       # DISTINCT bundles actually scored (== world rollouts here)
    total_score_calls: int = 0          # incl. cache hits (a coupling step may re-request a bundle)
    node_expansions: int = 0            # method-specific (beam/CE iterations, anneal steps, …)
    anchors_evaluated: bool = False     # were the named anchors among the evaluated set?
    top_k: list = field(default_factory=list)
    evaluated_keys: set = field(default_factory=set)   # distinct bundle keys scored (for regret containment)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"method": self.method,
                "best_bundle": self.best_bundle.non_default_surfaces() if self.best_bundle else {},
                "best_reward": round(self.best_reward, 4),
                "candidates_generated": self.candidates_generated,
                "candidates_evaluated": self.candidates_evaluated,
                "total_score_calls": self.total_score_calls, "node_expansions": self.node_expansions,
                "anchors_evaluated": self.anchors_evaluated, "top_k": self.top_k, "extra": self.extra}


class BudgetExhaustedError(Exception):
    """Raised inside a budgeted scorer once the distinct-evaluation budget is spent."""


@dataclass
class BudgetedScorer:
    """Wrap a memoized `score_fn(bundle)->float` with a cap on DISTINCT evaluations (the portable budget).

    A cache hit is free (no budget cost) — methods may legitimately re-request a bundle while coupling. When
    a NEW bundle would exceed `budget`, `score` raises `BudgetExhaustedError` (the method stops and keeps its
    best). Records the evaluated set, the best, and the total call count."""
    score_fn: object
    budget: int
    cache: dict = field(default_factory=dict)     # bundle key -> reward (shared upstream if passed in)
    evaluated: set = field(default_factory=set)   # distinct keys THIS method paid for
    calls: int = 0
    best_bundle: object = None
    best_reward: float = -1e18

    def score(self, bundle) -> float:
        self.calls += 1
        k = _key(bundle)
        if k in self.cache:
            r = self.cache[k]
        else:
            if len(self.evaluated) >= self.budget:
                raise BudgetExhaustedError()
            r = float(self.score_fn(bundle))
            self.cache[k] = r
        self.evaluated.add(k)
        if r > self.best_reward:
            self.best_reward, self.best_bundle = r, bundle
        return r

    def try_score(self, bundle):
        """Score, returning None instead of raising when the budget is exhausted."""
        try:
            return self.score(bundle)
        except BudgetExhaustedError:
            return None


# --- generator methods (ordered candidate lists; anchors first) --------------------------------------
def _gen_list(name, candidates, state):
    merged, _missing = enforce_anchors(candidates, prev_best=getattr(state, "prev_bundle", None))
    return merged


def candidates_for(method: str, state: PlannerRegimeState, *, seed: int = 0,
                   allow_quality_risk: bool = False) -> list:
    """The ordered candidate list a GENERATOR method proposes (anchors first), before the budget cap."""
    prev = getattr(state, "prev_bundle", None)
    if method == "fixed_grid":
        return _gen_list(method, core_grid(), state)
    if method == "expanded_grid":
        return _gen_list(method, expanded_grid(), state)
    if method == "physics_guided_grid":
        return physics_guided_candidates(state, prev_best=prev, allow_quality_risk=allow_quality_risk)
    if method == "random_grid":
        return random_candidates(60, seed=seed, prev_best=prev)
    if method == "clock_only":
        # the PR #121 artifact, kept ONLY as a comparator: anchors are NOT added (that is the whole point —
        # clock-only is the degenerate set that excludes the multi-knob winner).
        return [ActionBundle(clock_policy=c) for c in ("base", "low", "high")]
    raise ValueError(method)


def _run_generator(method, score_fn, *, budget, state, seed, allow_quality_risk, named_keys):
    cands = candidates_for(method, state, seed=seed, allow_quality_risk=allow_quality_risk)
    bs = BudgetedScorer(score_fn, budget)
    for b in cands:
        if bs.try_score(b) is None:
            break
    top = _topk([(ActionBundle(**dict(k)), r) for k, r in bs.cache.items() if k in bs.evaluated], 5)
    return MethodResult(method=method, best_bundle=bs.best_bundle, best_reward=bs.best_reward,
                        candidates_generated=len(cands), candidates_evaluated=len(bs.evaluated),
                        total_score_calls=bs.calls, node_expansions=0,
                        anchors_evaluated=named_keys.issubset(bs.evaluated), evaluated_keys=set(bs.evaluated),
                        top_k=[{"surfaces": b.non_default_surfaces(), "reward": round(r, 4)} for b, r in top])


# --- adaptive methods --------------------------------------------------------------------------------
def _regime_surfaces(state: PlannerRegimeState, *, expanded: bool, allow_quality_risk: bool) -> dict:
    """Surface→options the adaptive methods build bundles over (the core grid, optionally + spec)."""
    s = {k: tuple(v) for k, v in CORE_GRID_SURFACES.items()}
    if allow_quality_risk:
        s["precision_policy"] = ("bf16", "fp8", "int4")
    if expanded:
        s.update({k: tuple(v) for k, v in EXPANDED_EXTRA_SURFACES.items()})
    return s


def _seed_anchors(state):
    return required_anchors(prev_best=getattr(state, "prev_bundle", None), include_core_grid=False)


def _beam(score, *, budget, state, beam_width, expanded, allow_quality_risk, named_keys, label="beam_search",
          seed_anchors=True, seed_physics=False):
    """Bounded beam over the core (optionally + spec) surfaces. Ablation toggles: `seed_anchors` (the anchor
    floor incl. the diagnostic winner) and `seed_physics` (seed from the physics-guided candidate set)."""
    surfaces = _regime_surfaces(state, expanded=expanded, allow_quality_risk=allow_quality_risk)
    bs = BudgetedScorer(score, budget)
    names = list(surfaces)
    expansions = 0
    seeds = [ActionBundle()]
    if seed_anchors:
        seeds += _seed_anchors(state)
    if seed_physics:
        seeds += physics_guided_candidates(state, prev_best=getattr(state, "prev_bundle", None),
                                           allow_quality_risk=allow_quality_risk)
    beam = []
    for b in seeds:
        if bs.try_score(b) is None:
            break
        beam.append((b, bs.cache[_key(b)]))
    beam = _topk(beam, beam_width)
    try:
        for nm in names:
            nxt = list(beam)
            for b, _s in beam:
                cur = b.to_dict()
                for o in surfaces[nm]:
                    if o == cur.get(nm):
                        continue
                    nb = ActionBundle(**{**cur, nm: o})
                    r = bs.score(nb)
                    expansions += 1
                    nxt.append((nb, r))
            beam = _topk(nxt, beam_width)
    except BudgetExhaustedError:
        pass
    return _finish(label, bs, named_keys, node_expansions=expansions)


def _coordinate(score, *, budget, state, start=None, passes=3, expanded=False, allow_quality_risk=False,
                named_keys=frozenset(), label="coordinate_descent"):
    surfaces = _regime_surfaces(state, expanded=expanded, allow_quality_risk=allow_quality_risk)
    bs = BudgetedScorer(score, budget)
    cur = start if start is not None else ActionBundle()
    expansions = 0
    try:
        for b in _seed_anchors(state):           # anchor floor (the invariant) — descent still runs from `cur`
            bs.score(b)
        cur_s = bs.score(cur)
        for _ in range(max(1, passes)):
            improved = False
            for nm in surfaces:
                for o in surfaces[nm]:
                    if o == getattr(cur, nm):
                        continue
                    cand = cur.with_overrides(**{nm: o})
                    r = bs.score(cand)
                    expansions += 1
                    if r > cur_s + _EPS:
                        cur, cur_s, improved = cand, r, True
            if not improved:
                break
    except BudgetExhaustedError:
        pass
    return _finish(label, bs, named_keys, node_expansions=expansions)


def _progressive_widening(score, *, budget, state, beam_width, allow_quality_risk, named_keys):
    """Start narrow (core grid + anchors); widen to spec / optional surfaces only when the decision is close
    (small top-2 margin or low confidence). Reuses the beam at each width. Stops early on a clear winner."""
    bs = BudgetedScorer(score, budget)
    rounds, trace = 0, []
    expansions = [0]

    def _beam_round(expanded):
        surfaces = _regime_surfaces(state, expanded=expanded, allow_quality_risk=allow_quality_risk)
        beam = []
        for b in [ActionBundle()] + _seed_anchors(state):
            v = bs.try_score(b)
            if v is None:
                return None
            beam.append((b, v))
        beam = _topk(beam, beam_width)
        for nm in list(surfaces):
            nxt = list(beam)
            for b, _s in beam:
                cur = b.to_dict()
                for o in surfaces[nm]:
                    if o == cur.get(nm):
                        continue
                    v = bs.try_score(ActionBundle(**{**cur, nm: o}))
                    if v is None:
                        return _topk(nxt, beam_width)
                    expansions[0] += 1
                    nxt.append((ActionBundle(**{**cur, nm: o}), v))
            beam = _topk(nxt, beam_width)
        return beam

    beam = _beam_round(expanded=False) or []
    while beam and rounds < 2 and len(bs.evaluated) < budget:
        margin = ((beam[0][1] - beam[1][1]) / max(abs(beam[0][1]), _EPS)) if len(beam) > 1 else 1.0
        widen = margin < 0.05 or state.confidence < 0.35 or state.sla_tight
        if not widen:
            trace.append({"round": rounds + 1, "action": "stopped_early", "margin": round(margin, 5)})
            break
        nb = _beam_round(expanded=True)
        rounds += 1
        trace.append({"round": rounds, "action": "widened", "margin_before": round(margin, 5)})
        if nb:
            beam = nb
    res = _finish("progressive_widening", bs, named_keys, node_expansions=expansions[0])
    res.extra = {"widening_rounds": rounds, "widening_trace": trace}
    return res


def _hierarchical(score, *, budget, state, allow_quality_risk, named_keys):
    """Search by control timescale (slow / medium / fast groups) WITH cross-group coupling, never a greedy
    one-knob commit. Each group runs a small beam seeded from the running-best bundle; a final coordinate
    polish over ALL surfaces recovers cross-group coupling so no irreversible per-knob decision is made."""
    bs = BudgetedScorer(score, budget)
    expansions = 0
    incumbent = ActionBundle()
    try:
        bs.score(incumbent)
        for b in _seed_anchors(state):                 # anchors always in contention
            bs.score(b)
        if bs.best_bundle is not None:
            incumbent = bs.best_bundle
        for group in (MEDIUM_SURFACES, SLOW_SURFACES, FAST_SURFACES):   # medium first (highest value density)
            opts = {s: ACTION_SPECS[s].options for s in group}
            if not allow_quality_risk and "precision_policy" in opts:
                opts["precision_policy"] = ("bf16", "fp8")
            # Batch-1 regime gating (keep hierarchical width ~constant; new knobs only where they can help):
            if "kv_cache_precision_policy" in opts:
                kv_opts = ["inherit_weight_precision", "kv_fp8", "kv_int8"]
                if allow_quality_risk:
                    kv_opts.append("kv_int4_diagnostic_only")     # diagnostic only; never the headline
                # freeze KV precision to the no-op outside the memory-bound / HBM-pressed regime
                mem = state.decode_regime == "memory_bandwidth_bound"
                hbm_hi = state.hbm_pressure is not None and state.hbm_pressure >= 0.60
                opts["kv_cache_precision_policy"] = tuple(kv_opts) if (mem or hbm_hi) \
                    else ("inherit_weight_precision",)
            if "prefill_decode_policy" in opts:
                opts["prefill_decode_policy"] = ("shared", "p40_d60", "p60_d40") \
                    if getattr(state, "pd_divergence", False) else ("shared",)
            # ablation mask: hold any disabled Batch-1 knob at its no-op.
            allowed = getattr(state, "allowed_new_knobs", None)
            if allowed is not None:
                for knob, noop in (("kv_cache_precision_policy", "inherit_weight_precision"),
                                   ("prefill_decode_policy", "shared")):
                    if knob in opts and knob not in allowed:
                        opts[knob] = (noop,)
            beam = [(incumbent, bs.cache[_key(incumbent)])]
            for nm in group:                            # build coupled combos WITHIN the group (small beam)
                nxt = list(beam)
                for b, _s in beam:
                    cur = b.to_dict()
                    for o in opts[nm]:
                        if o == cur.get(nm):
                            continue
                        r = bs.score(ActionBundle(**{**cur, nm: o}))
                        expansions += 1
                        nxt.append((ActionBundle(**{**cur, nm: o}), r))
                beam = _topk(nxt, 4)
            incumbent = beam[0][0]                       # carry the group winner forward (still revisable)
        # final coupled polish over ALL core surfaces from the incumbent (no group is frozen greedily)
        surfaces = _regime_surfaces(state, expanded=True, allow_quality_risk=allow_quality_risk)
        cur, cur_s = incumbent, bs.cache[_key(incumbent)]
        for _ in range(2):
            improved = False
            for nm in surfaces:
                for o in surfaces[nm]:
                    if o == getattr(cur, nm):
                        continue
                    r = bs.score(cur.with_overrides(**{nm: o}))
                    expansions += 1
                    if r > cur_s + _EPS:
                        cur, cur_s, improved = cur.with_overrides(**{nm: o}), r, True
            if not improved:
                break
    except BudgetExhaustedError:
        pass
    return _finish("hierarchical_search", bs, named_keys, node_expansions=expansions)


def _cross_entropy(score, *, budget, state, allow_quality_risk, named_keys, seed=0,
                   iters=8, samples=12, elite_frac=0.3):
    """Cross-entropy method over the core surfaces: sample bundles from per-surface categorical
    distributions, refit toward the elite. Standard strong global search for discrete black-box budgets."""
    surfaces = _regime_surfaces(state, expanded=True, allow_quality_risk=allow_quality_risk)
    names = list(surfaces)
    rng = _random.Random(seed + 1)
    probs = {n: [1.0 / len(surfaces[n])] * len(surfaces[n]) for n in names}
    bs = BudgetedScorer(score, budget)
    expansions = 0
    try:
        for b in [ActionBundle()] + _seed_anchors(state):    # anchors seed the search
            bs.score(b)
        for _ in range(iters):
            batch = []
            for _k in range(samples):
                choice = {n: rng.choices(surfaces[n], weights=probs[n], k=1)[0] for n in names}
                b = ActionBundle(**choice)
                batch.append((b, choice, bs.score(b)))
                expansions += 1
            batch.sort(key=lambda t: -t[2])
            elite = batch[:max(1, int(elite_frac * len(batch)))]
            for n in names:
                counts = [sum(1 for _b, ch, _r in elite if ch[n] == o) for o in surfaces[n]]
                tot = sum(counts) or 1
                probs[n] = [0.2 * probs[n][i] + 0.8 * (counts[i] / tot) for i in range(len(surfaces[n]))]
    except BudgetExhaustedError:
        pass
    return _finish("cross_entropy", bs, named_keys, node_expansions=expansions)


def _random_restart(score, *, budget, state, allow_quality_risk, named_keys, seed=0, restarts=4):
    surfaces = _regime_surfaces(state, expanded=True, allow_quality_risk=allow_quality_risk)
    names = list(surfaces)
    rng = _random.Random(seed)
    bs = BudgetedScorer(score, budget)
    expansions = 0
    try:
        for b in [ActionBundle()] + _seed_anchors(state):
            bs.score(b)
        for _ in range(restarts):
            cur = ActionBundle(**{n: rng.choice(surfaces[n]) for n in names})
            cur_s = bs.score(cur)
            improved = True
            while improved:
                improved = False
                for nm in names:
                    for o in surfaces[nm]:
                        if o == getattr(cur, nm):
                            continue
                        r = bs.score(cur.with_overrides(**{nm: o}))
                        expansions += 1
                        if r > cur_s + _EPS:
                            cur, cur_s, improved = cur.with_overrides(**{nm: o}), r, True
    except BudgetExhaustedError:
        pass
    return _finish("random_restart", bs, named_keys, node_expansions=expansions)


def _simulated_annealing(score, *, budget, state, allow_quality_risk, named_keys, seed=0,
                         t0=1.0, cooling=0.9):
    """Simulated annealing over single-surface moves with a cooling schedule — accepts worse moves early to
    escape local optima, then exploits. A strong, cheap global search for discrete black-box budgets."""
    surfaces = _regime_surfaces(state, expanded=True, allow_quality_risk=allow_quality_risk)
    names = list(surfaces)
    rng = _random.Random(seed + 2)
    bs = BudgetedScorer(score, budget)
    expansions = 0
    try:
        for b in [ActionBundle()] + _seed_anchors(state):
            bs.score(b)
        cur = bs.best_bundle or ActionBundle()
        cur_s = bs.cache[_key(cur)]
        scale = max(abs(cur_s), 1.0)
        t = t0
        while True:
            nm = rng.choice(names)
            o = rng.choice(surfaces[nm])
            cand = cur.with_overrides(**{nm: o})
            r = bs.score(cand)
            expansions += 1
            d = (r - cur_s) / scale
            if d > 0 or rng.random() < math.exp(d / max(t, 1e-3)):
                cur, cur_s = cand, r
            t *= cooling
            if t < 1e-3:
                t = t0 * 0.5          # gentle reheat to keep exploring until the budget runs out
    except BudgetExhaustedError:
        pass
    return _finish("simulated_annealing", bs, named_keys, node_expansions=expansions)


def _hybrid(score, *, budget, state, beam_width, allow_quality_risk, named_keys):
    """The natural ensemble: physics-guided generation seeds a beam, a coordinate polish recouples cheap
    headline levers, then progressive widening if the decision is close. Shares one budget across stages."""
    surfaces = _regime_surfaces(state, expanded=False, allow_quality_risk=allow_quality_risk)
    bs = BudgetedScorer(score, budget)
    expansions = 0
    try:
        # stage 1: physics-guided candidates (anchored)
        for b in physics_guided_candidates(state, prev_best=getattr(state, "prev_bundle", None),
                                           allow_quality_risk=allow_quality_risk):
            bs.score(b)
        # stage 2: beam coupling over the core surfaces seeded by the best so far
        beam = _topk([(ActionBundle(**dict(k)), r) for k, r in bs.cache.items() if k in bs.evaluated], beam_width)
        for nm in list(surfaces):
            nxt = list(beam)
            for b, _s in beam:
                cur = b.to_dict()
                for o in surfaces[nm]:
                    if o == cur.get(nm):
                        continue
                    r = bs.score(ActionBundle(**{**cur, nm: o}))
                    expansions += 1
                    nxt.append((ActionBundle(**{**cur, nm: o}), r))
            beam = _topk(nxt, beam_width)
        # stage 3: coordinate polish over the full safe range from the beam winner
        cur, cur_s = beam[0]
        full = {**surfaces, "capacity_multiplier": (1.0, 0.75, 1.5), "clock_policy": ("base", "low", "high")}
        for _ in range(2):
            improved = False
            for nm in full:
                for o in full[nm]:
                    if o == getattr(cur, nm):
                        continue
                    r = bs.score(cur.with_overrides(**{nm: o}))
                    expansions += 1
                    if r > cur_s + _EPS:
                        cur, cur_s, improved = cur.with_overrides(**{nm: o}), r, True
            if not improved:
                break
    except BudgetExhaustedError:
        pass
    return _finish("hybrid", bs, named_keys, node_expansions=expansions)


def _exhaustive_small(score, *, budget, state, surfaces=None, named_keys=frozenset()):
    """Enumerate the full product over `surfaces` (default: the core grid). True optimum on a tractable
    fixture. Capped by `budget` — if the product exceeds it, this is NOT a true exhaustive (the caller marks
    it). Anchors are folded in so the winner is never missed by an off-grid anchor."""
    surfaces = surfaces or {k: tuple(v) for k, v in CORE_GRID_SURFACES.items()}
    names = list(surfaces)
    from itertools import product
    bundles = [ActionBundle(**dict(zip(names, c))) for c in product(*(surfaces[n] for n in names))]
    bundles = required_anchors(prev_best=getattr(state, "prev_bundle", None), include_core_grid=False) + bundles
    bs = BudgetedScorer(score, budget)
    complete = True
    for b in bundles:
        if bs.try_score(b) is None:
            complete = False
            break
    res = _finish("exhaustive_small", bs, named_keys, node_expansions=0)
    res.candidates_generated = len(bundles)
    res.extra = {"true_exhaustive": complete and len(bs.evaluated) >= len({_key(b) for b in bundles})}
    return res


def _finish(name, bs: BudgetedScorer, named_keys, *, node_expansions=0) -> MethodResult:
    evaluated = [(ActionBundle(**dict(k)), r) for k, r in bs.cache.items() if k in bs.evaluated]
    top = _topk(evaluated, 5)
    return MethodResult(method=name, best_bundle=bs.best_bundle, best_reward=bs.best_reward,
                        candidates_generated=len(bs.evaluated), candidates_evaluated=len(bs.evaluated),
                        total_score_calls=bs.calls, node_expansions=node_expansions,
                        anchors_evaluated=named_keys.issubset(bs.evaluated), evaluated_keys=set(bs.evaluated),
                        top_k=[{"surfaces": b.non_default_surfaces(), "reward": round(r, 4)} for b, r in top])


# --- dispatcher --------------------------------------------------------------------------------------
GENERATOR_METHODS = ("clock_only", "fixed_grid", "expanded_grid", "physics_guided_grid", "random_grid")
ADAPTIVE_METHODS = ("beam_search", "progressive_widening", "hierarchical_search", "coordinate_descent",
                    "cross_entropy", "random_restart", "simulated_annealing", "hybrid", "exhaustive_small")
# ablation variants (run only in the ablation study, not the main tournament roster).
ABLATION_METHODS = ("random_grid", "physics_guided_grid", "beam_no_anchors", "beam_search",
                    "beam_physics_seed", "progressive_widening")
ALL_METHODS = GENERATOR_METHODS + ADAPTIVE_METHODS


def run_method(method: str, score_fn, *, budget: int, state: PlannerRegimeState, named_keys=frozenset(),
               seed: int = 0, beam_width: int = 8, allow_quality_risk: bool = False,
               exhaustive_surfaces=None) -> MethodResult:
    """Run one search method under an evaluation `budget` and return its `MethodResult`. Deterministic."""
    if method in GENERATOR_METHODS:
        return _run_generator(method, score_fn, budget=budget, state=state, seed=seed,
                              allow_quality_risk=allow_quality_risk, named_keys=named_keys)
    if method == "beam_search":
        return _beam(score_fn, budget=budget, state=state, beam_width=beam_width, expanded=False,
                     allow_quality_risk=allow_quality_risk, named_keys=named_keys)
    if method == "beam_no_anchors":          # ablation: beam without the anchor floor
        return _beam(score_fn, budget=budget, state=state, beam_width=beam_width, expanded=False,
                     allow_quality_risk=allow_quality_risk, named_keys=named_keys,
                     label="beam_no_anchors", seed_anchors=False)
    if method == "beam_physics_seed":        # ablation: beam seeded by physics-guided generation
        return _beam(score_fn, budget=budget, state=state, beam_width=beam_width, expanded=False,
                     allow_quality_risk=allow_quality_risk, named_keys=named_keys,
                     label="beam_physics_seed", seed_physics=True)
    if method == "progressive_widening":
        return _progressive_widening(score_fn, budget=budget, state=state, beam_width=beam_width,
                                     allow_quality_risk=allow_quality_risk, named_keys=named_keys)
    if method == "hierarchical_search":
        return _hierarchical(score_fn, budget=budget, state=state, allow_quality_risk=allow_quality_risk,
                             named_keys=named_keys)
    if method == "coordinate_descent":
        return _coordinate(score_fn, budget=budget, state=state, allow_quality_risk=allow_quality_risk,
                           named_keys=named_keys)
    if method == "cross_entropy":
        return _cross_entropy(score_fn, budget=budget, state=state, allow_quality_risk=allow_quality_risk,
                              named_keys=named_keys, seed=seed)
    if method == "random_restart":
        return _random_restart(score_fn, budget=budget, state=state, allow_quality_risk=allow_quality_risk,
                               named_keys=named_keys, seed=seed)
    if method == "simulated_annealing":
        return _simulated_annealing(score_fn, budget=budget, state=state, allow_quality_risk=allow_quality_risk,
                                    named_keys=named_keys, seed=seed)
    if method == "hybrid":
        return _hybrid(score_fn, budget=budget, state=state, beam_width=beam_width,
                       allow_quality_risk=allow_quality_risk, named_keys=named_keys)
    if method == "exhaustive_small":
        return _exhaustive_small(score_fn, budget=budget, state=state, surfaces=exhaustive_surfaces,
                                 named_keys=named_keys)
    raise ValueError(f"unknown method {method!r}")


__all__ = ["MethodResult", "BudgetedScorer", "BudgetExhaustedError", "run_method", "candidates_for",
           "GENERATOR_METHODS", "ADAPTIVE_METHODS", "ABLATION_METHODS", "ALL_METHODS",
           "SLOW_SURFACES", "MEDIUM_SURFACES", "FAST_SURFACES"]
