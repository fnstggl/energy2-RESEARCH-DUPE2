"""Canonical production-like dataset — assembler (the first realizable slice).

The signal matrix (:mod:`aurelius.datasets.signal_matrix`) audits *every* signal
the joint optimizer needs and what fidelity each is obtainable at. This module
assembles the part we can build **today**, end to end, and hands it to the
unified replay engine (:mod:`aurelius.optimizer.unified_replay`).

What it builds
--------------
A multi-class serving dataset = a **real interactive spine** (the Azure LLM 2024
trace — real arrivals + real output tokens, the ``latency_critical`` class) plus
a **documented, deterministic best-effort batch overlay** (the ``best_effort``
class). The overlay is the minimal honest augmentation that supplies the one
dimension every public serving trace strips out — a *workload class that is legal
to defer* — without which admission and energy time-shift have nothing to act on.

Why this is the whole experiment
--------------------------------
``joint.combination_search`` measured admission as catastrophic (−56%) on raw
Azure because every request is latency-critical, so deferring anything blows the
SLA. The open question was: is the no-compounding result a property of the
optimizer, or of the single-class data? This assembler lets us answer it by A/B:
run the unified loop on the spine alone (single-class) vs. spine + overlay
(multi-class) and see whether compounding appears *only* when the data carries
the class structure. If it does, the blocker was data, not the loop.

Honesty discipline
------------------
The overlay is SYNTHETIC and labeled as such in the manifest. It is NOT real
demand: it re-times and re-labels token counts **resampled from the spine's own
empirical distribution** (so token physics stay real) into a steady background
batch stream. It is fully deterministic (no RNG — index-strided), reproducible,
and parameterized. Nothing here is a production claim; it is a controlled test
bed whose job is to isolate the data variable.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..benchmarks.srtf_serving_backtest import _service_time_s
from ..optimizer.unified_replay import CLASS_BEST_EFFORT, CLASS_LATENCY, Job


def to_jobs(
    raw: list,
    *,
    warp: float,
    cls: str = CLASS_LATENCY,
    predicted_tokens: list | None = None,
    idx_offset: int = 0,
) -> list:
    """Convert a ``(arrival_s, output_tokens)`` trace to sim-time :class:`Job`s.

    ``arrival_s`` is divided by ``warp`` into simulator time (same convention as
    ``evaluate_c_schedule``). ``predicted_tokens`` (causal ordering prior) defaults
    to the actual tokens (clairvoyant) — pass a causal estimate for deployable
    ordering. ``idx_offset`` keeps indices unique when concatenating streams.
    """
    out = []
    for i, (arr, tok) in enumerate(raw):
        pt = float(tok) if predicted_tokens is None else float(predicted_tokens[i])
        out.append(Job(
            idx=idx_offset + i, arrival_s=arr / warp, actual_tokens=int(tok),
            predicted_tokens=pt, service_s=_service_time_s(int(tok)), cls=cls,
        ))
    return out


@dataclass
class CanonicalManifest:
    """Provenance + fidelity record for one assembled canonical slice."""

    spine_dataset: str
    n_latency_critical: int
    n_best_effort: int
    best_effort_fraction: float
    overlay_tier: str            # "SYNTHETIC" — the overlay is not real demand
    overlay_method: str
    warp: float
    notes: tuple = ()

    def to_dict(self) -> dict:
        return {
            "spine_dataset": self.spine_dataset,
            "n_latency_critical": self.n_latency_critical,
            "n_best_effort": self.n_best_effort,
            "best_effort_fraction": round(self.best_effort_fraction, 4),
            "overlay_tier": self.overlay_tier,
            "overlay_method": self.overlay_method,
            "warp": round(self.warp, 9),
            "n_total": self.n_latency_critical + self.n_best_effort,
            "notes": list(self.notes),
        }


def augment_with_best_effort(
    spine_raw: list,
    *,
    warp: float,
    fraction: float = 0.4,
    token_multiplier: float = 1.0,
    causal_predicted: bool = True,
) -> tuple:
    """Build the multi-class slice: real interactive spine + best-effort overlay.

    The overlay places ``round(fraction · len(spine))`` best-effort batch jobs at a
    **steady cadence** across the spine's time window. Each overlay job's token
    count is **resampled deterministically** from the spine's own sorted token
    distribution (index-strided — no RNG), optionally scaled by ``token_multiplier``
    (batch jobs tend to be larger). Overlay jobs are class ``best_effort`` (relaxed
    SLA, deferrable); spine jobs are ``latency_critical`` (tight SLA, never deferred).

    Returns ``(jobs, manifest)``. Deterministic and reproducible.
    """
    if not spine_raw:
        return [], CanonicalManifest("azure_llm_2024", 0, 0, 0.0,
                                     "SYNTHETIC", "index-strided resample", warp)

    spine_sorted = sorted(spine_raw, key=lambda r: r[0])
    t0 = spine_sorted[0][0]
    t1 = spine_sorted[-1][0]
    span = max(1e-9, t1 - t0)

    # Causal running-median token prior for the spine (deployable ordering).
    spine_pred = _causal_predicted(spine_sorted) if causal_predicted else None
    jobs = to_jobs(spine_sorted, warp=warp, cls=CLASS_LATENCY,
                   predicted_tokens=spine_pred, idx_offset=0)

    n_be = round(fraction * len(spine_sorted))
    tokens_sorted = sorted(tok for _, tok in spine_sorted)
    overlay_raw = []
    if n_be > 0:
        stride = max(1, len(tokens_sorted) // n_be)
        for j in range(n_be):
            # steady background cadence across the window
            arr = t0 + span * (j + 0.5) / n_be
            tok = int(max(1, round(tokens_sorted[(j * stride) % len(tokens_sorted)] * token_multiplier)))
            overlay_raw.append((arr, tok))
    overlay_pred = _causal_predicted(overlay_raw) if (causal_predicted and overlay_raw) else None
    overlay_jobs = to_jobs(overlay_raw, warp=warp, cls=CLASS_BEST_EFFORT,
                           predicted_tokens=overlay_pred, idx_offset=len(spine_sorted))

    jobs = jobs + overlay_jobs
    manifest = CanonicalManifest(
        spine_dataset="azure_llm_2024",
        n_latency_critical=len(spine_sorted),
        n_best_effort=len(overlay_jobs),
        best_effort_fraction=fraction,
        overlay_tier="SYNTHETIC",
        overlay_method=(
            f"steady-cadence batch overlay; tokens index-strided-resampled from "
            f"spine distribution ×{token_multiplier}"),
        warp=warp,
        notes=(
            "Overlay is a documented synthetic batch tier, NOT real demand.",
            "Spine (latency_critical) is the real Azure LLM 2024 trace.",
            "Purpose: isolate the data variable for the compounding A/B.",
        ),
    )
    return jobs, manifest


def assemble_calibrated(
    spine_raw: list,
    *,
    warp: float,
    class_mix=None,
    weight: str = "count",
    token_multiplier: float = 1.0,
) -> tuple:
    """Assemble the multi-class slice with the best-effort fraction CALIBRATED from
    a real production class mix (Alibaba GPU qos), not an arbitrary guess.

    ``class_mix`` defaults to :func:`...calibration.default_alibaba_class_mix`.
    ``weight`` selects which calibrated fraction to use: ``"count"`` (by job count,
    the stable structural anchor) or ``"gpu_work"`` (by GPU-hours — sample-
    sensitive). The chosen ratio + its provenance are recorded in the manifest, so
    the overlay is no longer a free parameter but a measured one (PROXY tier).
    """
    from .calibration import default_alibaba_class_mix

    mix = class_mix if class_mix is not None else default_alibaba_class_mix()
    frac = (mix.best_effort_fraction_by_gpu_work if weight == "gpu_work"
            else mix.best_effort_fraction_by_count)
    jobs, manifest = augment_with_best_effort(
        spine_raw, warp=warp, fraction=frac, token_multiplier=token_multiplier)
    manifest.notes = manifest.notes + (
        f"best_effort_fraction CALIBRATED from {mix.source} "
        f"(by {weight}={frac:.4f}, tier={mix.tier})",
        "ratio transferred as a distribution, NOT a per-record join (Alibaba is "
        "training/packing telemetry, a different workload from LLM serving)",
    )
    return jobs, manifest, mix


def _causal_predicted(trace: list) -> list:
    """Running-median causal token prediction (no token oracle)."""
    import bisect
    n = len(trace)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: trace[i][0])
    global_median = sorted(t[1] for t in trace)[n // 2]
    pred = [0.0] * n
    seen: list = []
    for i in order:
        pred[i] = float(seen[(len(seen) - 1) // 2]) if seen else float(global_median)
        bisect.insort(seen, trace[i][1])
    return pred


__all__ = [
    "to_jobs", "augment_with_best_effort", "assemble_calibrated",
    "CanonicalManifest", "CLASS_LATENCY", "CLASS_BEST_EFFORT",
]
