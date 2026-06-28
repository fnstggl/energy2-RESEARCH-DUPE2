"""Lightweight trace-derived scenario forecaster for MPC planning.

PR #112 found the live MPC's remaining error is NOT the search algorithm or the serving physics, but the
planner's predictive workload model: ``_rollout_world`` planned against a single synthetic median workload
that under-represents the SLA pressure evaluation later experiences, so the planner under-valued speculative
decoding's SLA protection. This module gives the planner a small **ensemble of plausible future workloads**
instead of one — so the planning score is an expectation over realistic demand conditions, INCLUDING the
SLA-pressure ones.

No ML, no external simulator: each scenario is a simple statistical extrapolation from the forecast
trajectory the controller already produces (mean / p90 / p10 / p99 of arrival rate, output tokens,
inter-arrival CV — all fit on the public traces). The weights are a fixed, risk-averse prior (base highest;
stress scenarios down-weighted) — NOT tuned to a benchmark.
"""

from __future__ import annotations

# (label, arrival key, output-token-mean key, output-p95 key, cv key, prompt multiplier, weight).
# The keys index the forecast point objects; prompt_mult stresses prefill/regime; weight is a risk-averse
# prior over which futures to optimize for (base dominant, stress futures down-weighted but present).
SCENARIOS = (
    ("base", "mean", "mean", "value", "mean", 1.0, 1.0),          # the forecast central path
    ("burst", "p90", "mean", "value", "p90", 1.0, 0.8),           # high arrival / bursty inter-arrivals
    ("long_output", "mean", "p90", "p99", "mean", 1.0, 0.7),      # decode-heavy (longer generations)
    ("long_prompt", "mean", "mean", "value", "mean", 1.6, 0.7),   # prefill-heavy (longer prompts)
    ("tight_sla", "p90", "p90", "p99", "p90", 1.0, 0.6),          # combined stress (the SLA-pressure future)
    ("calm", "p10", "mean", "value", "mean", 1.0, 0.5),           # low load
)


def _val(pt, *keys) -> float:
    """First present attribute among ``keys`` on a forecast point object (mean/p90/p10/value/p99…)."""
    for k in keys:
        v = getattr(pt, k, None)
        if v is not None:
            return float(v)
    return 0.0


def build_scenarios(ar, tm, tp, cv, *, prompt_tokens=None) -> list:
    """A small ensemble of planning-workload descriptors from the forecast points ``ar`` (arrival rate),
    ``tm`` (output-token mean), ``tp`` (output p95), ``cv`` (inter-arrival CV). Each descriptor is a dict
    ``{label, arrival_rate, tm, tp, cv, prompt_mult, weight}``; the caller synthesises jobs from it. Robust
    to missing percentiles (falls back to the mean / value)."""
    out = []
    for label, ak, mk, pk, ck, pmult, w in SCENARIOS:
        out.append({
            "label": label,
            "arrival_rate": _val(ar, ak, "mean"),
            "tm": _val(tm, mk, "mean"),
            "tp": _val(tp, pk, "value", "p95", "mean"),
            "cv": _val(cv, ck, "mean"),
            "prompt_mult": pmult,
            "weight": w,
        })
    return out


__all__ = ["SCENARIOS", "build_scenarios"]
