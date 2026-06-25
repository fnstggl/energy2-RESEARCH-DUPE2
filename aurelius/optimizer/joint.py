"""Joint serving optimization — combination search over composable levers.

This is the first real increment of the unified joint loop — the path to
COMPOUNDING goodput/$ that the per-surface ``optimize_fleet`` fan-out cannot
produce. It composes the **deployable** serving levers on ONE trace through one
on-demand evaluation and MEASURES the interaction (compound vs substitutive),
scored by the canonical :class:`ObjectiveLayer`.

Levers (all deployable, on-demand — no spot, no oracle):
  * **capacity**  — ``forecasted_mcs`` (causal) vs ``reactive_lag1`` (baseline)
  * **ordering**  — abs-conformal SRPT (causal, running-median token prior) vs FIFO
  * **admission** — peak-shave flow control (defer load under transient overload) vs off

Honest framing (audit 2026-06-25): the evidence says OVERLAPPING levers
(ordering × capacity) are substitutive / NEGATIVE — once capacity meets the SLA
there is little queue left for ordering to optimise (BENCHMARK_REGISTRY §2A:
``conformal+OSOTSS < FIFO+OSOTSS``). Compounding requires levers that hit
*different* cost terms. This harness does not ASSUME a combination compounds — it
runs the full lattice and reports which combinations actually do.

Everything here is deterministic and priced on the pure on-demand denominator
(``sum(c)·tick_hr·GPU_HOUR_USD``). Directional simulator only — not production
savings (``docs/RESULTS.md`` §8).
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass

from ..benchmarks.forecasted_mcs import (
    GPU_HOUR_USD,
    evaluate_c_schedule,
    forecast_mcs_c_schedule,
    reactive_lag1_c_schedule,
)
from ..benchmarks.phase_c import trace_content_hash
from .layers import ObjectiveLayer

# Lever option labels.
CAP_BASELINE = "reactive_lag1"
CAP_AURELIUS = "forecasted_mcs"
ORDER_FIFO = "fifo"
ORDER_SRPT = "abs_conformal"


# ---------------------------------------------------------------------------
# Admission lever — deterministic peak-shave flow control
# ---------------------------------------------------------------------------

def peak_shave_admission(
    raw: list, tick_seconds: float, warp: float, *, threshold: float = 1.5
) -> list:
    """Defer load from over-threshold ticks (token-bucket flow control).

    Caps each tick's admitted arrivals at ``round(threshold × mean_per_tick)`` and
    cascades the excess (latest-arriving requests) to later ticks. This SMOOTHS
    arrival spikes so the capacity lever can provision fewer replicas; deferred
    requests are served later (their wait — and thus SLA outcome — is affected,
    which the downstream evaluation prices honestly). No request is dropped, so
    the goodput numerator's potential is unchanged — only timing + cost move.

    Deterministic; operates in warped tick space and returns reshaped ``raw``
    (unwarped ``(arrival_s, tokens)``).
    """
    if len(raw) < 2:
        return list(raw)
    warped = sorted((arr / warp, tok) for arr, tok in raw)
    t_max = warped[-1][0]
    n_ticks = max(1, int(t_max / tick_seconds) + 1)
    mean = len(warped) / n_ticks
    cap = max(1, round(threshold * mean))

    buckets: list[list] = [[] for _ in range(n_ticks)]
    for t, tok in warped:
        k = min(n_ticks - 1, int(t / tick_seconds))
        buckets[k].append((t, tok))

    reshaped: list = []
    carry: list = []
    max_k = n_ticks + (len(warped) // cap) + 2
    for k in range(max_k):
        avail = carry + (buckets[k] if k < n_ticks else [])
        admit = avail[:cap]
        defer = avail[cap:]
        tick_start = k * tick_seconds
        for t, tok in admit:
            reshaped.append((max(t, tick_start), tok))
        # deferred requests wait at least until the next tick
        carry = [(max(t, (k + 1) * tick_seconds), tok) for t, tok in defer]
        if k >= n_ticks - 1 and not carry:
            break
    base = max_k * tick_seconds
    for i, (t, tok) in enumerate(carry):  # flush any residue (rare)
        reshaped.append((base + i, tok))

    return sorted((t * warp, tok) for t, tok in reshaped)


def _causal_predicted_tokens(trace: list) -> list:
    """Running-median causal token prediction (deployable — no token oracle).

    For each request (in arrival order) predict the median of tokens seen *before*
    it; warm-start the first with the global median. Used as the ordering prior so
    the SRPT lever is deployable (it never reads a request's own actual tokens to
    order it).
    """
    n = len(trace)
    order = sorted(range(n), key=lambda i: trace[i][0])
    if n == 0:
        return []
    global_median = sorted(t[1] for t in trace)[n // 2]
    pred = [0.0] * n
    seen: list = []
    for i in order:
        if seen:
            pred[i] = float(seen[(len(seen) - 1) // 2])
        else:
            pred[i] = float(global_median)
        bisect.insort(seen, trace[i][1])
    return pred


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class JointCell:
    """One combination of levers, evaluated on the on-demand denominator."""

    capacity: str
    ordering: str
    admission: bool
    levers_on: tuple          # which non-baseline levers are active
    goodput_per_dollar: float
    cost_usd: float
    gpu_hours: float
    n_sla_safe: int
    sla_violations: int

    @property
    def label(self) -> str:
        on = "+".join(self.levers_on) if self.levers_on else "base"
        return on

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "ordering": self.ordering,
            "admission": self.admission,
            "levers_on": list(self.levers_on),
            "label": self.label,
            "goodput_per_dollar": round(self.goodput_per_dollar, 4),
            "cost_usd": round(self.cost_usd, 4),
            "gpu_hours": round(self.gpu_hours, 4),
            "n_sla_safe": self.n_sla_safe,
            "sla_violations": self.sla_violations,
        }


@dataclass
class JointResult:
    """Combination-search result: the full lattice + the interaction verdict."""

    trace_id: str
    seed: int
    trace_hash: str
    sla_s: float
    tick_seconds: float
    warp: float
    denominator: str
    gpu_hour_usd: float
    cells: list                 # list[JointCell]
    base_gpd: float
    best_single_label: str
    best_single_gpd: float
    best_overall_label: str
    best_overall_gpd: float
    compounding: bool           # does the best multi-lever combo beat the best single?
    interaction: str            # "compounding" | "substitutive" | "neutral"
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "seed": self.seed,
            "trace_hash": self.trace_hash,
            "sla_s": self.sla_s,
            "tick_seconds": self.tick_seconds,
            "warp": round(self.warp, 9),
            "denominator": self.denominator,
            "gpu_hour_usd": self.gpu_hour_usd,
            "cells": [c.to_dict() for c in self.cells],
            "base_gpd": round(self.base_gpd, 4),
            "best_single": {"label": self.best_single_label,
                            "goodput_per_dollar": round(self.best_single_gpd, 4)},
            "best_overall": {"label": self.best_overall_label,
                             "goodput_per_dollar": round(self.best_overall_gpd, 4)},
            "compounding": self.compounding,
            "interaction": self.interaction,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Combination search
# ---------------------------------------------------------------------------

def combination_search(
    raw: list,
    *,
    tick_seconds: float,
    warp: float,
    sla_s: float,
    seed: int,
    trace_id: str,
    mcs_gate: float = 9.5,
    admission_threshold: float = 1.5,
    notes=(),
) -> JointResult:
    """Run the 2×2×2 lever lattice on ``raw`` and measure interaction effects.

    Levers (off→on vs the reactive/FIFO/no-admission base):
        capacity:  reactive_lag1 → forecasted_mcs   (lever "C")
        ordering:  fifo          → abs_conformal SRPT (lever "O")
        admission: off           → peak-shave         (lever "A")

    Returns a :class:`JointResult` with every cell's on-demand goodput/$, the best
    single-lever cell, the best overall cell, and whether the best multi-lever
    combination COMPOUNDS (beats the best single lever) or is SUBSTITUTIVE.
    """
    obj = ObjectiveLayer()

    # Pre-shape the admitted trace once (admission lever on).
    admitted = peak_shave_admission(raw, tick_seconds, warp, threshold=admission_threshold)
    traces = {False: raw, True: admitted}

    # Cache capacity schedules per (admission, capacity-mode).
    def _schedule(trace, cap_mode):
        if cap_mode == CAP_BASELINE:
            return reactive_lag1_c_schedule(trace, tick_seconds, warp,
                                            mcs_gate=mcs_gate, sla_s=sla_s)
        sched, _diag = forecast_mcs_c_schedule(trace, tick_seconds, warp,
                                               method="ewma", mcs_gate=mcs_gate,
                                               sla_s=sla_s)
        return sched

    cells: list[JointCell] = []
    for cap_mode, order_disc, admit in itertools.product(
        (CAP_BASELINE, CAP_AURELIUS), (ORDER_FIFO, ORDER_SRPT), (False, True)
    ):
        trace = traces[admit]
        c_sched = _schedule(trace, cap_mode)
        predicted = _causal_predicted_tokens(trace) if order_disc == ORDER_SRPT else None
        kpi = evaluate_c_schedule(
            trace, c_sched, tick_seconds, warp, sla_s,
            policy=f"{cap_mode}|{order_disc}|adm={int(admit)}",
            uses_future_info=False, deployable=True,
            classification="joint_deployable", discipline=order_disc,
            predicted_tokens=predicted,
        )
        levers = tuple(
            x for x, on in (
                ("C", cap_mode == CAP_AURELIUS),
                ("O", order_disc == ORDER_SRPT),
                ("A", admit),
            ) if on
        )
        cells.append(JointCell(
            capacity=cap_mode, ordering=order_disc, admission=admit,
            levers_on=levers, goodput_per_dollar=kpi.goodput_per_dollar,
            cost_usd=kpi.cost_usd, gpu_hours=kpi.gpu_hours,
            n_sla_safe=kpi.n_sla_safe, sla_violations=kpi.sla_violations,
        ))

    # Rank every cell by SLA-safe goodput/$ through the canonical ObjectiveLayer.
    by_label = {c.label: c for c in cells}
    ranked = obj.compare({c.label: c.goodput_per_dollar for c in cells})
    best_overall = by_label[ranked[0][0]]

    base = next(c for c in cells if not c.levers_on)
    singles = [c for c in cells if len(c.levers_on) == 1]
    multis = [c for c in cells if len(c.levers_on) >= 2]
    best_single = max(singles, key=lambda c: c.goodput_per_dollar)
    best_multi = max(multis, key=lambda c: c.goodput_per_dollar)

    # Compounding = the best multi-lever combo beats the best single lever by a
    # meaningful margin (>0.5%). Otherwise the levers are substitutive/neutral.
    margin = (best_multi.goodput_per_dollar - best_single.goodput_per_dollar)
    rel = margin / best_single.goodput_per_dollar if best_single.goodput_per_dollar else 0.0
    if rel > 0.005:
        interaction, compounding = "compounding", True
    elif rel < -0.005:
        interaction, compounding = "substitutive", False
    else:
        interaction, compounding = "neutral", False

    return JointResult(
        trace_id=trace_id, seed=seed,
        trace_hash=trace_content_hash(raw, tick_seconds=tick_seconds, warp=warp, sla_s=sla_s),
        sla_s=sla_s, tick_seconds=tick_seconds, warp=warp,
        denominator="on_demand", gpu_hour_usd=GPU_HOUR_USD,
        cells=cells, base_gpd=base.goodput_per_dollar,
        best_single_label=best_single.label, best_single_gpd=best_single.goodput_per_dollar,
        best_overall_label=best_overall.label, best_overall_gpd=best_overall.goodput_per_dollar,
        compounding=compounding, interaction=interaction, notes=tuple(notes),
    )


__all__ = [
    "CAP_BASELINE", "CAP_AURELIUS", "ORDER_FIFO", "ORDER_SRPT",
    "peak_shave_admission", "JointCell", "JointResult", "combination_search",
]
