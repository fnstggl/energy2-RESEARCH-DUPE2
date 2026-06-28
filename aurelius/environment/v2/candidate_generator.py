"""Roofline-aware candidate generator (V2) — a SOFT PRIOR on which bundles to generate/prioritise.

The roofline regime (compute / memory / HBM-bound) and the slack state condition WHICH action bundles are
generated and prioritised — it does NOT force selection. The simulator still chooses by the causal outcome
under the Pareto gate, and may override the prior if a non-prioritised bundle is Pareto-better (the override
is reported with a reason). There is no reward effect here at all — this only prunes/orders the candidate
list (hard rule: no roofline law changes reward directly).

Priors (from research/ROOFLINE_REUSE_DECISION.md / FULL_SERVING_PHYSICS_INTEGRATION_PLAN.md):
  * memory-bound  → lower precision; higher batching if SLA/HBM allow; spec-decode if compute headroom +
    acceptance; low/base clock; conservative co-location iff real background work and SLA-safe.
  * compute-bound → spec off; base/high clock; conservative/balanced batching; co-location off; precision
    only if HBM still matters.
  * HBM-bound     → lower precision; lower active-sequence target; no co-location; no aggressive batching.
  * mixed         → evaluate both memory- and compute-oriented bundles.

Co-location is pruned to ``off`` whenever there is no real/trace-derived background work (no imaginary goodput).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

FULL_SPACE = {
    "precision": ["bf16", "fp8", "int4"],
    "spec_decode": ["off", "shallow", "medium", "aggressive"],
    "clock": ["base", "low", "high"],
    "colocation_mode": ["off", "conservative", "aggressive"],
    "max_active_sequences": [32, 64, 128],
    "serving_mode": ["shared_pool", "disaggregated_static"],
    "prefill_frac": [None, 0.5],
}


@dataclass
class CandidateSet:
    candidates: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def _allowed_values(dim, regime, *, hbm_pressure, sla_slack, has_background, pruned):
    """Per-dim allowed values under the regime prior. Records pruned reasons."""
    vals = list(FULL_SPACE[dim])
    if dim == "colocation_mode" and not has_background:
        for v in ("conservative", "aggressive"):
            pruned["coloc_no_background"] = pruned.get("coloc_no_background", 0) + 1
        return ["off"]
    if dim == "precision":
        if regime == "compute" and hbm_pressure < 0.5:
            pruned["precision_compute_bound"] = pruned.get("precision_compute_bound", 0) + 1
            return ["bf16", "fp8"]            # int4 quality risk not worth it when compute-bound, HBM slack
        return vals
    if dim == "spec_decode":
        if regime == "compute":
            pruned["spec_compute_bound"] = pruned.get("spec_compute_bound", 0) + 1
            return ["off"]                   # spec adds draft compute when already compute-bound
        return vals
    if dim == "clock":
        if regime == "memory":
            pruned["clock_memory_bound"] = pruned.get("clock_memory_bound", 0) + 1
            return ["base", "low"]           # high clock wastes power when memory-bound
        return ["base", "high"]
    if dim == "max_active_sequences":
        if regime == "HBM" or hbm_pressure > 1.0:
            pruned["batching_hbm_bound"] = pruned.get("batching_hbm_bound", 0) + 1
            return [32, 64]                  # don't oversubscribe KV memory
        return vals
    if dim in ("serving_mode", "prefill_frac"):
        return vals
    return vals


def generate_candidates(regime: str, *, hbm_pressure: float = 0.0, sla_slack_s: float = 1.0,
                        has_background_work: bool = False, cap: int = 4096) -> CandidateSet:
    """Generate the regime-prioritised candidate bundles + diagnostics. ``regime`` ∈ {compute, memory, HBM,
    mixed}. The returned list is the soft-prior subset the search runs over (the simulator can still evaluate
    the full space and override)."""
    pruned = {}
    dims = list(FULL_SPACE)
    per_dim = {d: _allowed_values(d, regime if regime != "mixed" else "memory", hbm_pressure=hbm_pressure,
                                  sla_slack=sla_slack_s, has_background=has_background_work, pruned=pruned)
               for d in dims}
    if regime == "mixed":
        # union memory- and compute-oriented values so both are evaluated
        comp = {d: _allowed_values(d, "compute", hbm_pressure=hbm_pressure, sla_slack=sla_slack_s,
                                   has_background=has_background_work, pruned=pruned) for d in dims}
        per_dim = {d: sorted(set(map(repr, per_dim[d])) | set(map(repr, comp[d])), key=str) and
                   list(dict.fromkeys(per_dim[d] + comp[d])) for d in dims}
    raw = 1
    for vs in FULL_SPACE.values():
        raw *= len(vs)
    cands = []
    for combo in itertools.product(*[per_dim[d] for d in dims]):
        c = dict(zip(dims, combo))
        # keep prefill_frac coherent with serving_mode
        if c["serving_mode"] == "shared_pool":
            c["prefill_frac"] = None
        elif c["prefill_frac"] is None:
            c["prefill_frac"] = 0.5
        key = tuple(sorted(c.items(), key=lambda kv: kv[0]))
        cands.append(key)
        if len(cands) >= cap:
            break
    # dedupe
    uniq = [dict(k) for k in dict.fromkeys(cands)]
    diag = {"roofline_regime": regime, "hbm_pressure": round(hbm_pressure, 4),
            "candidate_count_raw": raw, "candidate_count_after_pruning": len(uniq),
            "pruned_reason_counts": pruned, "has_background_work": has_background_work}
    return CandidateSet(candidates=uniq, diagnostics=diag)


__all__ = ["generate_candidates", "CandidateSet", "FULL_SPACE"]
