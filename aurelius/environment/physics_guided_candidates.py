"""Physics-guided candidate generation — the first-class planner layer that PR #121 was missing.

The PR #121 regression ("all-knobs got worse") was a *candidate-containment* artifact: the bounded runner
replaced the search with 3 clock-only bundles, so the `{precision=fp8, batching=aggressive, clock=high}`
winner (which doubled gp/$ while improving SLA) was **absent from the enumerated set** — not pruned by a
rule, just never generated (`research/PHYSICS_GUIDED_PLANNER_AUDIT.md`, Q2/Q3). The fix is not in the MPC,
the reward, or the gate: it is to **generate a small but expressive candidate set that always contains the
known-good bundles**, then let the simulator choose by causal evaluation.

This module generates `ActionBundle` candidates from the current planner state using **physics priors**
(roofline regime, SLA slack, queue / price / token / capacity / HBM pressure, the previous selection) over
the high-value CONNECTED surfaces — `precision × batching × capacity × clock` by default, optional surfaces
only when the planner widens. The priors are a **soft prior only**: they decide which candidates are
*generated*, never the reward (a generated candidate still scores through the real physics). The honesty
guarantees are structural, not advisory:

  * **Anchors are always included and never silently dropped** — `ActionBundle()` (neutral), the
    production-safe SLA-aware bundle, the previous best, the `fp8 + aggressive` known-strong family, the old
    +82% family, capacity-adjusted bundles, and clock low/base/high. The hard cap drops *prior-grid*
    candidates first and records why; an anchor is never removed.
  * **Out-of-scope surfaces stay out** — co-location is generated only with `background_work=True` (no
    background-work trace exists); SIMULATED_ONLY / PLANNED surfaces are never default-generated; `int4` is
    proposed only in the memory-bandwidth-bound regime and is **never** a known-strong anchor (it carries a
    quality/SLA risk). Every exclusion is recorded in `pruned_reasons`.

Deterministic (no RNG): a fixed state yields a fixed candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from .actions import ActionBundle

# --- pressure thresholds (soft priors; physics-motivated, not benchmark-tuned) -----------------------
# A signal at/above its threshold flips the corresponding option set ON. They only ADD plausible
# candidates to the SEARCH; the reward is unaffected and the anchors cover the essentials regardless.
_PRESSURE_HI = 0.60          # queue / capacity / hbm pressure considered "high" at/above this
_PRICE_HI = 0.70             # electricity price percentile considered "high" at/above this
_PRICE_LO = 0.30             # …and "low" at/below this
_SLACK_POS = 0.10            # SLA slack considered comfortably positive at/above this (fraction of budget)

# --- the high-value default search surfaces (Q4 of the audit: the 4 that produced +100.48%) ----------
DEFAULT_SURFACES = ("precision_policy", "batching_policy", "capacity_multiplier", "clock_policy")
# optional CONNECTED surfaces, added ONLY by progressive widening (each can hurt + adds cost / runtime).
OPTIONAL_SURFACES = ("routing_policy", "prewarm_policy", "spec_decode_policy", "migration_policy")

# --- known-strong anchors (the bundles PR #121 proved recover the win; always searched) --------------
# The fp8 + aggressive-batching family + the +82% full-search family, expressed as ActionBundles. int4 is
# deliberately absent (quality-risked → never a headline anchor). These are GUARANTEED into every set.
KNOWN_STRONG_BUNDLES = (
    (ActionBundle(precision_policy="fp8", batching_policy="aggressive", clock_policy="high"),
     "known_strong:fp8_aggressive_high"),                          # the PR #121 24-grid winner
    (ActionBundle(precision_policy="fp8", batching_policy="aggressive"),
     "known_strong:fp8_aggressive"),                               # fp8 + aggressive at base clock
    (ActionBundle(capacity_policy="backlog_aware", ordering_policy="abs_conformal",
                  routing_policy="kv_aware", batching_policy="aggressive", precision_policy="fp8"),
     "known_strong:full_family"),                                  # the +82% combined-lever family
)
# the production-safe / SLA-aware bundle == controller.SLA_AWARE_FALLBACK (kept in sync by value).
SAFE_BASELINE_BUNDLE = ActionBundle(capacity_policy="backlog_aware", ordering_policy="abs_conformal",
                                    admission_policy="off")
NEUTRAL_BUNDLE = ActionBundle()


@dataclass
class PlannerRegimeState:
    """A cheap, pre-search snapshot of the world the priors read. Every field is OPTIONAL — a missing
    signal (None) simply contributes no prior (the anchors still cover the space). The controller fills
    these from the forecast + world state before the search; they are a soft prior, never a reward term."""
    decode_regime: str | None = None        # "memory_bandwidth_bound" | "compute_bound" | None(mixed)
    sla_slack: float | None = None          # headroom as a fraction of the SLA budget (>0 = comfortable)
    queue_pressure: float = 0.0             # 0..1  (backlog / capacity)
    capacity_pressure: float = 0.0          # 0..1  (forecast arrivals / capacity envelope)
    price_percentile: float | None = None   # 0..1  electricity price percentile (None = unknown → neutral)
    output_token_mean: float | None = None  # token pressure (longer outputs → more decode → memory-bound)
    hbm_pressure: float | None = None       # 0..1  HBM / KV residency pressure (None = unavailable)
    confidence: float = 1.0                 # forecast confidence (low → widen / uncertain regime)
    prev_bundle: object = None              # the previously selected ActionBundle (continuity anchor)
    # Batch-1 new-knob gates (soft priors; a knob frozen off here is recorded in pruned_reasons):
    pd_divergence: bool = False             # prefill/decode pressures diverge → disaggregation candidate
    prefill_heavy: bool | None = None       # True → prefill-heavy split; False → decode-heavy; None → neutral
    heterogeneous_fleet: bool = False       # fleet exposes per-workload GPU-type assignment (else NOT_APPLICABLE)
    allowed_new_knobs: frozenset | None = None   # ablation mask: which Batch-1 knobs may vary (None = all)

    @property
    def sla_tight(self) -> bool:
        """SLA is tight when slack is known-low OR queue/capacity pressure is high OR confidence is low."""
        slack_low = self.sla_slack is not None and self.sla_slack < _SLACK_POS
        return slack_low or self.queue_pressure >= _PRESSURE_HI or self.capacity_pressure >= _PRESSURE_HI \
            or self.confidence < 0.3

    @property
    def regime_label(self) -> str:
        return self.decode_regime or "mixed_or_uncertain"


@dataclass
class CandidateSet:
    """The generated candidate set + a full provenance record (the user's reporting requirement)."""
    bundles: list = field(default_factory=list)         # list[ActionBundle], anchors first, deduped
    origins: list = field(default_factory=list)         # parallel origin label per bundle
    anchor_flags: dict = field(default_factory=dict)    # which anchor categories are present
    raw_grid_count: int = 0                             # Cartesian size of the regime prior BEFORE cap/dedup
    generated_count: int = 0                            # len(bundles)
    target: tuple = (30, 100)
    capped: bool = False
    pruned_reasons: dict = field(default_factory=dict)  # reason -> count / detail
    surfaces_used: list = field(default_factory=list)
    surfaces_options: dict = field(default_factory=dict)  # surface -> options tuple (for the beam)
    anchors: list = field(default_factory=list)         # the anchor ActionBundles (never dropped)
    regime: str = "mixed_or_uncertain"

    def keys(self) -> set:
        return {tuple(sorted(b.to_dict().items())) for b in self.bundles}

    def contains(self, bundle) -> bool:
        return tuple(sorted(bundle.to_dict().items())) in self.keys()

    def to_dict(self) -> dict:
        return {"generated_count": self.generated_count, "raw_grid_count": self.raw_grid_count,
                "target": list(self.target), "capped": self.capped, "regime": self.regime,
                "surfaces_used": self.surfaces_used, "anchor_flags": self.anchor_flags,
                "pruned_reasons": self.pruned_reasons,
                "origins": {o: self.origins.count(o) for o in set(self.origins)}}


def regime_surface_options(state: PlannerRegimeState, *, optional_surfaces: tuple = (),
                           background_work: bool = False, allow_quality_risk: bool = False) -> tuple:
    """Per-surface option subsets for the candidate grid, shaped by the physics regime (Phase 2).

    Returns ``(surfaces, pruned_reasons)`` where ``surfaces`` maps each search surface → a tuple of options.
    The policy is a UNION of the regime rules (more matching conditions → a broader search), because the
    priors only widen the SEARCH and the simulator picks the winner. ``optional_surfaces`` (progressive
    widening) adds routing/prewarm/spec/migration. Co-location is never added unless ``background_work``.
    ``int4`` is generated only when ``allow_quality_risk`` is set (default off → headline-safe): it carries
    an UNMODELLED quality/SLA risk, so an int4 win is diagnostic, never a headline (like co-location).
    """
    reasons: dict = {}
    mem = state.decode_regime == "memory_bandwidth_bound"
    comp = state.decode_regime == "compute_bound"
    mixed = not mem and not comp
    qhi = state.queue_pressure >= _PRESSURE_HI or state.capacity_pressure >= _PRESSURE_HI
    price_hi = state.price_percentile is not None and state.price_percentile >= _PRICE_HI
    price_lo = state.price_percentile is not None and state.price_percentile <= _PRICE_LO
    hbm_hi = state.hbm_pressure is not None and state.hbm_pressure >= _PRESSURE_HI
    slack_pos = state.sla_slack is not None and state.sla_slack >= _SLACK_POS
    tight = state.sla_tight

    # ---- precision: bf16 + fp8 always (both lossless-safe); int4 OPT-IN only (quality risk, no model) --
    precision = ["bf16", "fp8"]
    if allow_quality_risk and (mem or mixed):
        precision.append("int4")                  # diagnostic only; never a headline
    elif not allow_quality_risk:
        reasons["int4_excluded"] = "quality/SLA risk with no quality model → headline-unsafe (opt-in via allow_quality_risk)"
    else:
        reasons["int4_excluded"] = "not memory-bandwidth-bound (int4 only worth its quality risk there)"

    # ---- clock: base always; low when memory-bound/price-high & slack; high when compute/queue pressure
    clock = ["base"]
    if (mem or price_hi) and not (qhi and not slack_pos):
        clock.append("low")                       # save energy when memory-bound or price-high (if SLA ok)
    if comp or qhi or mixed:
        clock.append("high")                      # buy compute-bound / queue-pressure latency
    if price_hi and not qhi:
        # power-price high and no SLA pressure → do NOT propose high clock (energy waste)
        clock = [c for c in clock if c != "high"]
        reasons["high_clock_excluded"] = "price high and no queue/SLA pressure (avoid energy waste)"

    # ---- batching: conservative always; balanced unless HBM-pressed; aggressive when memory-bound & SLA ok
    batching = ["conservative"]
    if not hbm_hi:
        batching.append("balanced")
    if (mem or mixed) and not tight and not hbm_hi:
        batching.append("aggressive")
    elif tight:
        reasons["aggressive_batching_excluded"] = "SLA tight (tail-latency risk)"
    elif hbm_hi:
        reasons["aggressive_batching_excluded"] = "HBM/KV pressure high (avoid larger active set)"

    # ---- capacity_multiplier: 1.0 always; 1.5 under pressure; 0.75 when slack + price-high (cost save) --
    capacity = [1.0]
    if qhi or tight:
        capacity.append(1.5)
    if slack_pos and price_lo or (slack_pos and not qhi and price_hi):
        capacity.append(0.75)                     # under-provision only when there is real slack

    # ---- kv_cache_precision (Batch-1): separate KV dtype. inherit + kv_fp8 always (both lossless-safe);
    # kv_int8 when memory-bound or HBM-pressed (its win regime); kv_int4 OPT-IN only (no quality model).
    # Frozen to inherit (no-op) when neither memory-bound nor HBM-pressed (KV bytes don't bind there).
    kv_precision = ["inherit_weight_precision", "kv_fp8"]
    if mem or hbm_hi:
        kv_precision.append("kv_int8")
        if allow_quality_risk:
            kv_precision.append("kv_int4_diagnostic_only")   # diagnostic only; never a headline
        else:
            reasons["kv_int4_excluded"] = "int4 KV quality/SLA risk with no quality model → headline-unsafe (opt-in)"
    elif comp:
        kv_precision = ["inherit_weight_precision"]
        reasons["kv_precision_frozen"] = "compute-bound: KV bytes don't bind decode bandwidth (no-op)"

    # ---- prefill_decode (Batch-1): disaggregation only when prefill/decode pressures DIVERGE (else the
    # shared pool's statistical multiplexing wins and the KV handoff is pure overhead — no free disaggregation).
    pd = ["shared"]
    if state.pd_divergence:
        if state.prefill_heavy is True:
            pd.append("p60_d40")                             # prefill-heavy → more prefill capacity
        elif state.prefill_heavy is False:
            pd.append("p40_d60")                             # decode-heavy → more decode capacity
        else:
            pd.extend(["p40_d60", "p60_d40"])
    else:
        reasons["prefill_decode_frozen"] = "prefill/decode pressures not diverging → shared pool optimal (handoff is overhead)"

    surfaces = {"precision_policy": tuple(dict.fromkeys(precision)),
                "kv_cache_precision_policy": tuple(dict.fromkeys(kv_precision)),
                "clock_policy": tuple(dict.fromkeys(clock)),
                "batching_policy": tuple(dict.fromkeys(batching)),
                "capacity_multiplier": tuple(dict.fromkeys(capacity)),
                "prefill_decode_policy": tuple(dict.fromkeys(pd))}

    # ---- gpu_assignment (Batch-1): heterogeneous GPU-type assignment is NOT_APPLICABLE to the production
    # benchmark (one dominant GPU type in the cost path; gpu_type constant per server). Frozen off unless the
    # fleet exposes per-workload assignment — it can only fake a benefit otherwise. Fixture-only today.
    if state.heterogeneous_fleet:
        surfaces["gpu_assignment_policy"] = ("homogeneous_default", "fastest_for_latency_sensitive",
                                             "cheapest_for_batch", "memory_heavy_to_high_hbm",
                                             "balanced_heterogeneous")
    else:
        reasons["gpu_assignment_frozen"] = ("homogeneous-cost production fleet (single dominant GPU type; "
                                            "gpu_type constant per server) → NOT_APPLICABLE, fixture-only")

    # ---- optional surfaces (progressive widening only) -----------------------------------------------
    for s in optional_surfaces:
        if s == "routing_policy":
            surfaces["routing_policy"] = ("round_robin", "kv_aware")     # KV reuse can cut service time
        elif s == "prewarm_policy":
            surfaces["prewarm_policy"] = ("off", "conservative") if (qhi or state.capacity_pressure >= 0.4) \
                else ("off",)
        elif s == "spec_decode_policy":
            if comp:
                surfaces["spec_decode_policy"] = ("off",)               # compute-bound → spec competes for FLOPs
                reasons["spec_decode_off"] = "compute-bound (draft+verify FLOPs not worth it)"
            else:
                surfaces["spec_decode_policy"] = ("off", "shallow") if tight else ("off", "shallow", "medium")
        elif s == "migration_policy":
            surfaces["migration_policy"] = ("off", "conservative")
        elif s == "colocation_policy":
            if background_work:
                surfaces["colocation_policy"] = ("off", "conservative")
            else:
                reasons["colocation_excluded"] = "no background-work trace (co-location can only hurt SLA)"

    if not background_work:
        reasons.setdefault("colocation_excluded", "no background-work trace (co-location can only hurt SLA)")

    # ablation mask: freeze any Batch-1 knob not in `allowed_new_knobs` to its no-op (records the reason).
    allowed = state.allowed_new_knobs
    if allowed is not None:
        for knob, noop in (("kv_cache_precision_policy", "inherit_weight_precision"),
                           ("prefill_decode_policy", "shared"),
                           ("gpu_assignment_policy", "homogeneous_default")):
            if knob not in allowed and knob in surfaces and len(surfaces[knob]) > 1:
                surfaces[knob] = (noop,)
                reasons[f"{knob}_ablation_disabled"] = "ablation arm: knob held at no-op"
    return surfaces, reasons


def _anchor_bundles(state: PlannerRegimeState) -> list:
    """The always-included bundles (never dropped by the cap). Returns ``[(ActionBundle, origin), …]``."""
    anchors = [(NEUTRAL_BUNDLE, "neutral"), (SAFE_BASELINE_BUNDLE, "baseline_safe")]
    if state.prev_bundle is not None and hasattr(state.prev_bundle, "to_dict"):
        anchors.append((state.prev_bundle, "previous_best"))
    anchors.extend(KNOWN_STRONG_BUNDLES)
    # capacity-adjusted anchors (so the capacity lever is always represented both ways)
    anchors.append((ActionBundle(capacity_multiplier=1.5), "capacity_adjusted:up"))
    anchors.append((ActionBundle(capacity_multiplier=0.75), "capacity_adjusted:down"))
    # clock anchors (low / base / high always searchable — base == neutral, deduped later)
    anchors.append((ActionBundle(clock_policy="low"), "clock_anchor:low"))
    anchors.append((ActionBundle(clock_policy="high"), "clock_anchor:high"))
    # Batch-1 new-knob anchors (only in their active regime, so they are never strided out of the grid).
    # The ablation mask gates them too — an anchor must not reintroduce a knob the arm disabled.
    allowed = state.allowed_new_knobs
    def _on(knob):
        return allowed is None or knob in allowed
    mem = state.decode_regime == "memory_bandwidth_bound"
    hbm_hi = state.hbm_pressure is not None and state.hbm_pressure >= _PRESSURE_HI
    if (mem or hbm_hi) and _on("kv_cache_precision_policy"):
        anchors.append((ActionBundle(kv_cache_precision_policy="kv_fp8"), "kv_precision_anchor:fp8"))
        anchors.append((ActionBundle(precision_policy="fp8", kv_cache_precision_policy="kv_fp8"),
                        "kv_precision_anchor:weight_fp8_kv_fp8"))
    if state.pd_divergence and _on("prefill_decode_policy"):
        split = "p60_d40" if state.prefill_heavy else "p40_d60"
        anchors.append((ActionBundle(prefill_decode_policy=split), f"pd_anchor:{split}"))
    if state.heterogeneous_fleet and _on("gpu_assignment_policy"):
        anchors.append((ActionBundle(gpu_assignment_policy="balanced_heterogeneous"), "gpu_assign_anchor:balanced"))
    return anchors


@dataclass
class PhysicsGuidedCandidateGenerator:
    """Generate a bounded, expressive, anchor-guaranteed candidate set from the planner state."""
    target_min: int = 30
    target_max: int = 100
    hard_cap: int = 120                 # absolute ceiling on generated candidates (configurable)
    grid_budget: int = 90               # max prior-grid candidates BEFORE anchors (strided if exceeded)
    background_work: bool = False       # gate co-location (no background-work trace by default)
    allow_quality_risk: bool = False    # gate int4 (quality-risked); default off → headline-safe

    def generate(self, state: PlannerRegimeState, *, optional_surfaces: tuple = ()) -> CandidateSet:
        surfaces, reasons = regime_surface_options(state, optional_surfaces=optional_surfaces,
                                                    background_work=self.background_work,
                                                    allow_quality_risk=self.allow_quality_risk)
        names = list(surfaces)
        opt_lists = [surfaces[n] for n in names]
        raw = 1
        for o in opt_lists:
            raw *= max(1, len(o))

        # build the prior grid (deterministic stride if the Cartesian exceeds the grid budget — recorded)
        grid: list = []
        if raw <= self.grid_budget:
            for combo in product(*opt_lists):
                grid.append(ActionBundle(**dict(zip(names, combo))))
        else:
            allc = list(product(*opt_lists))
            stride = (len(allc) + self.grid_budget - 1) // self.grid_budget
            grid = [ActionBundle(**dict(zip(names, allc[i]))) for i in range(0, len(allc), stride)]
            reasons["grid_strided"] = f"prior grid {raw} > budget {self.grid_budget}; strided to {len(grid)}"

        # anchors first → they win dedup (and are never dropped by the cap)
        ordered = _anchor_bundles(state) + [(b, "prior_grid") for b in grid]
        seen: set = set()
        bundles: list = []
        origins: list = []
        n_anchor = 0
        for b, origin in ordered:
            key = tuple(sorted(b.to_dict().items()))
            if key in seen:
                continue
            seen.add(key)
            bundles.append(b)
            origins.append(origin)
            if origin != "prior_grid":
                n_anchor += 1

        capped = False
        if len(bundles) > self.hard_cap:
            # drop ONLY prior-grid candidates (from the end), never anchors
            keep_b, keep_o = [], []
            for b, o in zip(bundles, origins):
                if o != "prior_grid" or len([x for x in keep_o if x == "prior_grid"]) < (self.hard_cap - n_anchor):
                    keep_b.append(b)
                    keep_o.append(o)
            dropped = len(bundles) - len(keep_b)
            bundles, origins = keep_b, keep_o
            capped = True
            reasons["hard_cap"] = f"dropped {dropped} prior-grid candidates to respect hard_cap={self.hard_cap} (anchors kept)"

        anchor_flags = {
            "neutral": "neutral" in origins,
            "baseline_safe": "baseline_safe" in origins,
            "previous_best": "previous_best" in origins,
            "known_strong": any(o.startswith("known_strong") for o in origins),
            "capacity_adjusted": any(o.startswith("capacity_adjusted") for o in origins),
            "clock_low_base_high": all(any(_clock_of(b) == c for b in bundles) for c in ("low", "base", "high")),
        }
        anchors = [b for b, o in zip(bundles, origins) if o != "prior_grid"]
        return CandidateSet(bundles=bundles, origins=origins, anchor_flags=anchor_flags,
                            raw_grid_count=raw, generated_count=len(bundles),
                            target=(self.target_min, self.target_max), capped=capped,
                            pruned_reasons=reasons, surfaces_used=names, surfaces_options=dict(surfaces),
                            anchors=anchors, regime=state.regime_label)


def _clock_of(bundle) -> str:
    return getattr(bundle, "clock_policy", "base")


__all__ = ["PlannerRegimeState", "CandidateSet", "PhysicsGuidedCandidateGenerator",
           "regime_surface_options", "KNOWN_STRONG_BUNDLES", "SAFE_BASELINE_BUNDLE", "NEUTRAL_BUNDLE",
           "DEFAULT_SURFACES", "OPTIONAL_SURFACES"]
